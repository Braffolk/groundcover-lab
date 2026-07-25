#include "src/wgsl/scatter.wgsl"
#include "./entry_info.wgsl"

// Per-frame plant selection AND coarse depth sort in one pass. The shared
// scatter twin is evaluated over a bounded camera-centered cell region (already
// clamped to the stand's cell range on the CPU); survivors are frustum-culled
// and compacted into ONE instance buffer, but into six disjoint sub-ranges by
// camera distance. Each sub-range gets its own indirect draw, so the draw order
// is near-to-far: the biggest occluders write depth first and early-z rejects
// most of the fragments behind them. The two nearest buckets are drawn with the
// 3-tap relief shader, the middle two with the 2-tap parallax shader, the outer
// two with the plain billboard shader — that is the whole "detail only where the
// camera is" mechanism, and it costs one compare per plant.
//
// Cost is O(region area), never O(total plants in the stand). A workgroup is 64
// invocations and a cell holds SCATTER_MAX_PER_CELL = 128 candidate slots, so
// every workgroup covers exactly half of ONE cell: the cell-level region/frustum
// rejects below are workgroup-uniform and let whole workgroups exit before a
// single terrain texel is fetched.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
// NB x [vertexCount, instanceCount, firstVertex, firstInstance]
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>>;

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

  // Circular region edge: closest point of the cell rect to the camera.
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > info.region_r) {
    return;
  }

  // Frustum vs a conservative box over every plant this cell can hold:
  // cell footprint in xz, terrain bound + tallest card + wind margin in y.
  let s_max = stand_table[u32(info.entry_index)].scale_max;
  let rad_max = info.cull_radius * s_max + 0.35;
  let mid_y = (info.card_y0 + info.card_y1) * 0.5 * s_max;
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

  let d_cam = distance(sp.pos.xz, frame.camera_pos.xz);
  if (d_cam > info.region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, (info.card_y0 + info.card_y1) * 0.5 * sp.scale, 0.0);
  let rad = info.cull_radius * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  // --- append into this plant's distance bucket ------------------------------
  // Bucketing (and therefore the shader tier) goes by the 3D distance, not the
  // horizontal one: what a tier is really selecting is apparent size, so a
  // camera 40m up must get the cheap far shader even directly below itself.
  let b = bucket_of(distance(center, frame.camera_pos), info.bucket_edge[0], info.bucket_edge[1]);
  let cap = u32(pick8(info.bucket_cap[0], info.bucket_cap[1], b));
  let base = u32(pick8(info.bucket_base[0], info.bucket_base[1], b));
  let w = atomicAdd(&draw_args[b * 4u + 1u], 1u);
  if (w >= cap) {
    atomicSub(&draw_args[b * 4u + 1u], 1u); // bucket full — count stays clamped
    return;
  }
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  out_insts[base + w] = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));
}
