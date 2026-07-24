#include "src/wgsl/scatter.wgsl"
#include "./entry_info.wgsl"

// Per-frame plant selection: evaluates the shared scatter over a bounded
// camera-centered cell region (clamped to the stand's cell range, exactly as
// the CPU twin clamps it), frustum-culls each plant's bounding sphere, and
// compacts survivors into an instance buffer + indirect draw args. Cost is
// O(region area), never O(total plants in the stand).

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>, 4>;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side = u32(info.side);
  let total = side * side * SCATTER_MAX_PER_CELL;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % SCATTER_MAX_PER_CELL;
  let cell_lin = idx / SCATTER_MAX_PER_CELL;
  let cx = i32(info.origin_cell.x) + i32(cell_lin % side);
  let cz = i32(info.origin_cell.y) + i32(cell_lin / side);

  // Stand region: same cell-index clamp as Scatter.region() on the CPU.
  let cmin = i32(info.stand_cell_min);
  let cmax = i32(info.stand_cell_max);
  if (cx < cmin || cx > cmax || cz < cmin || cz > cmax) {
    return;
  }

  let sp = scatter_candidate(u32(info.seed), u32(info.entry_index), vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }

  // Circular region edge — the draw erodes coverage to zero at region_r.
  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }

  // Frustum test on the plant's bounding sphere (+ wind margin).
  let mid_y = (info.card_y0 + info.card_y1) * 0.5 * sp.scale;
  let center = sp.pos + vec3f(0.0, mid_y, 0.0);
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
