#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "./entry_info.wgsl"

// Per-frame plant selection + LOD bucketing. Evaluates the shared scatter over
// a bounded camera-centered cell region (already clamped to the stand's cell
// range on the CPU), frustum-culls each plant's bounding sphere, picks the LOD
// from its projected pixel height, and appends it to that LOD's bucket. Cost is
// O(region area), never O(total plants in the stand).
//
// The dispatch is 2D: x walks this ENTRY'S OWN slot count (info.carpet_info.z
// — carpet_div² = 484 for the bog moss, SCATTER_MAX_PER_CELL = 128 for a
// scattered entry), y is one cell per workgroup row. So every workgroup lives
// inside exactly one cell: the cell-level rejects below are workgroup-uniform
// and let whole workgroups exit before a single terrain texel is fetched, and
// no integer div/mod is needed to recover the cell.

// 32 B. Per-plant work that the draw would otherwise redo in every one of the
// plant's ~8192x4 vertex invocations is finished HERE, once: yaw sin/cos, the
// wind displacement (4 transcendentals) and the coverage fade.
struct PlantInst {
  pos_scale: vec4f,
  packed: vec4<u32>,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;
// drawIndexedIndirect args are 5 u32; main.ts pads the stride to 8 for a
// 32-byte alignment that WebGPU is happy to fetch.
const ARG_STRIDE: u32 = 8u;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u32(info.carpet_info.z)) {
    return;
  }
  let side_x = u32(info.side_xz.x);
  let cell_lin = gid.y;
  let cx = i32(info.origin_cell.x) + i32(cell_lin % side_x);
  let cz = i32(info.origin_cell.y) + i32(cell_lin / side_x);

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);

  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > info.region_r) {
    return;
  }

  let s_max = stand_table[u32(info.entry_index)].scale_max;
  let rad_max = info.cull_radius * s_max + 0.35;
  let mid_y = (info.splat_y0 + info.splat_y1) * 0.5 * s_max;
  let h_bound = frame.terrain_height_scale * TERRAIN_BOUND_SLACK;
  let box_c = vec3f(cell_mid.x, mid_y * 0.5, cell_mid.y);
  let box_e = vec3f(
    SCATTER_CELL_SIZE * 0.5,
    h_bound + abs(mid_y) * 0.5 + rad_max,
    SCATTER_CELL_SIZE * 0.5,
  );
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, box_c) + pl.w + dot(abs(pl.xyz), box_e) < 0.0) {
      return;
    }
  }

  // --- exact per-plant tests -------------------------------------------------
  let sp = scatter_candidate(u32(info.seed), u32(info.entry_index), vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }

  let d_xz = distance(sp.pos.xz, frame.camera_pos.xz);
  if (d_xz > info.region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, (info.splat_y0 + info.splat_y1) * 0.5 * sp.scale, 0.0);
  let rad = info.cull_radius * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  // --- LOD from projected pixel height --------------------------------------
  // The ladder is geometric (each level halves the splat count and moves the
  // switch distance by sqrt(2)), so the level index is ONE log2 — no loop, no
  // threshold table walk.
  let dist = max(distance(center, frame.camera_pos), 0.05);
  // Projected size of the plant, measured along the axis that actually carries
  // its extent: HEIGHT for an upright plant, FOOTPRINT for a carpet tile
  // (0.18m wide vs 0.07m tall — measuring a mat by its height picks a level two
  // steps too coarse and turns the tile into a handful of fat ellipses).
  let px_h = info.carpet_info.y * sp.scale * info.px_scale / dist;
  // Per-plant jitter decorrelates LOD transitions spatially, so a switch reads
  // as scattered plants changing rather than a ring sweeping over the field.
  let jitter = 0.82 + 0.36 * hash_f32(hash3(u32(info.seed) ^ 0x9e37u, bitcast<u32>(sp.pos.x), bitcast<u32>(sp.pos.z)));
  let lod_f = log(max(info.lod_t0 * jitter / px_h, 1.0)) * info.inv_log_ratio;
  let lod = min(u32(max(lod_f, 0.0)), u32(info.lod_count) - 1u);

  // --- coverage falloff (no dither) -----------------------------------------
  // Region-rim plants shrink their splats toward nothing, so the canopy thins
  // out honestly and every remaining edge stays hard and early-z friendly. A
  // plant that has faded out is dropped here outright.
  var fade = 1.0 - smoothstep(info.region_r * 0.86, info.region_r, d_xz);
  // The camera-inside fade applies to plants the camera can stand INSIDE. A
  // carpet is a mat you stand ON: fading it would open a hole under your feet.
  if (info.carpet_info.x < 0.5) {
    let seg_y = clamp(frame.camera_pos.y, sp.pos.y + info.splat_y0 * sp.scale, sp.pos.y + info.splat_y1 * sp.scale);
    let d_core = length(vec3f(sp.pos.x, seg_y, sp.pos.z) - frame.camera_pos);
    let r_p = max(info.splat_rxz * sp.scale, 1e-3);
    fade = fade * smoothstep(r_p * 0.18, r_p * 0.8, d_core);
  }
  if (fade < 0.02) {
    return;
  }

  let cap = u32(info.lod_cap[lod >> 2u][lod & 3u]);
  let w = atomicAdd(&draw_args[lod * ARG_STRIDE + 1u], 1u);
  if (w >= cap) {
    atomicSub(&draw_args[lod * ARG_STRIDE + 1u], 1u); // bucket full — stay clamped
    return;
  }
  let sway = wind_sway(sp.pos, frame.time, stand_table[u32(info.entry_index)].sway, sp.phase);
  let base_slot = u32(info.lod_inst_base[lod >> 2u][lod & 3u]);
  out_insts[base_slot + w] = PlantInst(
    vec4f(sp.pos, sp.scale),
    vec4<u32>(
      pack2x16float(sway.xy),
      pack2x16float(vec2f(sway.z, fade)),
      pack2x16float(vec2f(cos(sp.yaw), sin(sp.yaw))),
      0u,
    ),
  );
}
