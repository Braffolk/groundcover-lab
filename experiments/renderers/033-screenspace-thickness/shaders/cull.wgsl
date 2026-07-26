#include "src/wgsl/scatter.wgsl"
#include "./dome_lib.wgsl"

// Per-frame plant selection: evaluates the shared scatter over a bounded
// camera-centered cell region (already clamped to the stand's cell range on
// the CPU), frustum-culls, picks the baked view and the shell LOD, and
// compacts survivors into three per-LOD instance lists + their indirect draw
// args. Cost is O(region area), never O(total plants in the stand).
//
// A workgroup is 64 invocations and a cell holds SCATTER_MAX_PER_CELL = 128
// candidate slots, so every workgroup covers exactly half of ONE cell: the
// cell-level region/frustum rejects are workgroup-uniform and let whole
// workgroups exit before a single terrain texel is fetched.

@group(1) @binding(1) var<storage, read_write> out_lod0: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> out_lod1: array<PlantInst>;
@group(1) @binding(3) var<storage, read_write> out_lod2: array<PlantInst>;
// Three drawIndexedIndirect arg sets (5 u32 each); [.,1], [.,6], [.,11] are
// the instance counts this pass fills.
@group(1) @binding(4) var<storage, read_write> draw_args: array<atomic<u32>, 15>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.side_x);
  let total = side_x * u32(info.side_z) * SCATTER_MAX_PER_CELL;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % SCATTER_MAX_PER_CELL;
  let cell_lin = idx / SCATTER_MAX_PER_CELL;
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
  let mid_y = (info.y0 + info.y1) * 0.5 * s_max;
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
  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }

  let center = card_center(sp.pos, sp.yaw, sp.scale);
  let rad = info.cull_radius * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  // Fully eroded cards (camera standing inside one, region rim) would discard
  // every fragment — and they are exactly the ones covering the most pixels.
  if (card_erode(sp.pos, center, sp.scale) > 1.0) {
    return;
  }

  let to_cam = frame.camera_pos - center;
  let dist = length(to_cam);
  var lod = 2u;
  if (dist < info.lod0_r) {
    lod = 0u;
  } else if (dist < info.lod1_r) {
    lod = 1u;
  }
  var cap = u32(info.cap2);
  if (lod == 0u) {
    cap = u32(info.cap0);
  } else if (lod == 1u) {
    cap = u32(info.cap1);
  }

  let w = atomicAdd(&draw_args[lod * 5u + 1u], 1u);
  if (w >= cap) {
    atomicSub(&draw_args[lod * 5u + 1u], 1u); // full — final count stays clamped
    return;
  }
  let inst = PlantInst(sp.pos, pack_inst_bits(sp.yaw, sp.scale, sp.phase, pick_view(to_cam, sp.yaw, sp.phase / TAU - 0.5)));
  if (lod == 0u) {
    out_lod0[w] = inst;
  } else if (lod == 1u) {
    out_lod1[w] = inst;
  } else {
    out_lod2[w] = inst;
  }
}
