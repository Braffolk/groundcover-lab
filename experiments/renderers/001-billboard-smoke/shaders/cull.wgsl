#include "src/wgsl/scatter.wgsl"
#include "./entry_info.wgsl"

// Per-frame plant selection: evaluates the shared scatter over a bounded
// camera-centered cell region (already clamped to the stand's cell range on
// the CPU), frustum-culls each plant's bounding sphere, and compacts survivors
// into an instance buffer + indirect draw args. Cost is O(region area), never
// O(total plants in the stand).
//
// A workgroup is 64 invocations. For a scattered entry a cell holds
// SCATTER_MAX_PER_CELL = 128 candidate slots, so a workgroup covers half of ONE
// cell; a CARPET entry has carpet_div^2 slots instead (484 at life size), so it
// simply takes more workgroups per cell. Either way the count comes from
// info.slots_per_cell — never from the global budget. Originally: the
// cell-level region/frustum rejects below are workgroup-uniform and let whole
// workgroups exit before a single terrain texel is fetched. Only cells that
// can contribute run the shared scatter (4 heightmap loads + 7 hash rounds per
// candidate) — the per-plant tests afterwards stay exact.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>, 4>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.side_x);
  let slots = u32(info.slots_per_cell);
  let total = side_x * u32(info.side_z) * slots;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % slots;
  let cell_lin = idx / slots;
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

  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, (info.card_y0 + info.card_y1) * 0.5 * sp.scale, 0.0);
  let rad = info.cull_radius * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  let w = atomicAdd(&draw_args[1], 1u);
  if (w >= u32(info.capacity)) {
    atomicSub(&draw_args[1], 1u); // full — final count stays clamped
    return;
  }
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  out_insts[w] = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));
}
