#include "src/wgsl/scatter.wgsl"

// Camera-region instance builder. Walks ONLY the scatter cells within
// max_dist of the camera (a fixed-size grid, independent of stand radius /
// total plant count), evaluates the shared scatter twin for one stand entry
// and appends surviving plants to that entry's slice of the instance buffer,
// bumping its indirect draw count. 100 plants or 1 billion: the dispatch is
// the same size.

struct CullParams {
  base_cell: vec2i,
  entry_index: u32,
  seed: u32,
  max_dist: f32,
  stand_r: f32,
  capacity: u32,
  _pad: u32,
}
@group(1) @binding(0) var<uniform> cp: CullParams;
// Indirect draw args, 4 u32 per stand entry: [6, instanceCount, 0, 0].
@group(1) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>>;
// 2 vec4 per plant: [x, y, z, yaw], [scale, entry, phase, 0].
@group(1) @binding(2) var<storage, read_write> plants: array<vec4f>;

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) slot: u32) {
  let cell = cp.base_cell + vec2i(i32(wg.x), i32(wg.y));
  // Mirror Scatter.region(): every cell whose span overlaps the stand aabb
  // contributes all its candidates (no per-plant position clamp).
  let lo = i32(floor(-cp.stand_r / SCATTER_CELL_SIZE));
  let hi = i32(floor(cp.stand_r / SCATTER_CELL_SIZE));
  if (cell.x < lo || cell.x > hi || cell.y < lo || cell.y > hi) {
    return;
  }
  let c = scatter_candidate(cp.seed, cp.entry_index, cell, slot);
  if (!c.exists) {
    return;
  }
  let dxz = c.pos.xz - frame.camera_pos.xz;
  if (dot(dxz, dxz) > cp.max_dist * cp.max_dist) {
    return;
  }
  let counter = cp.entry_index * 4u + 1u;
  let idx = atomicAdd(&draw_args[counter], 1u);
  if (idx >= cp.capacity) {
    // Every over-capacity thread takes back exactly its own +1, so the final
    // count converges to `capacity` instead of running past the buffer.
    atomicSub(&draw_args[counter], 1u);
    return;
  }
  let base = idx * 2u;
  plants[base] = vec4f(c.pos, c.yaw);
  plants[base + 1u] = vec4f(c.scale, f32(cp.entry_index), c.phase, 0.0);
}
