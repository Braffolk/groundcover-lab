#include "src/wgsl/scatter.wgsl"

// Camera-region instance builder — per-frame work is bounded by max_dist, not
// by the stand's total plant count. Walks only the scatter cells within
// max_dist of the camera (a fixed-size dispatch), evaluates the shared
// scatter twin for one stand entry and appends surviving plants to that
// entry's instance buffer, bumping its indirect draw count. 100 plants or
// 1 billion: the dispatch is identical.

struct CullParams {
  base_cell: vec2i,
  entry_index: u32,
  seed: u32,
  max_dist: f32,
  stand_r: f32,
  capacity: u32,
  sphere_y: f32,   // plant bound-sphere center height, mesh units (× scale)
  sphere_r: f32,   // plant bound-sphere radius incl. center xz offset (× scale)
  _p0: f32,
  _p1: f32,
  _p2: f32,
}
@group(1) @binding(0) var<uniform> cp: CullParams;
// Indirect draw args, 4 u32 per stand entry: [6, instanceCount, 0, 0].
@group(1) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>>;
// 2 vec4 per plant: [x, y, z, yaw], [scale, entry, phase, 0].
@group(1) @binding(2) var<storage, read_write> plants: array<vec4f>;

// Sphere-vs-frustum test with planes extracted from view_proj (WebGPU clip
// space, z in [0,1]). A fixed 6-plane loop — culling, not marching.
fn frustum_visible(c: vec3f, r: f32) -> bool {
  let m = frame.view_proj;
  let r0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
  let r1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
  let r2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
  let r3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);
  var planes = array<vec4f, 5>(r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2);
  for (var i = 0u; i < 5u; i++) {
    let p = planes[i];
    if (dot(p.xyz, c) + p.w < -r * length(p.xyz)) { return false; }
  }
  return true;
}

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) slot: u32) {
  let cell = cp.base_cell + vec2i(i32(wg.x), i32(wg.y));
  // Mirror Scatter.region(): every cell whose span overlaps the stand aabb
  // contributes all of its candidates (no per-plant position clamp).
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
  let sphere_c = vec3f(c.pos.x, c.pos.y + cp.sphere_y * c.scale, c.pos.z);
  if (!frustum_visible(sphere_c, cp.sphere_r * c.scale + 0.5)) {
    return;
  }
  let counter = cp.entry_index * 4u + 1u;
  let idx = atomicAdd(&draw_args[counter], 1u);
  if (idx >= cp.capacity) {
    // Over-capacity threads take back their +1 so the count converges to
    // `capacity` instead of running past the buffer.
    atomicSub(&draw_args[counter], 1u);
    return;
  }
  let base = idx * 2u;
  plants[base] = vec4f(c.pos, c.yaw);
  plants[base + 1u] = vec4f(c.scale, f32(cp.entry_index), c.phase, 0.0);
}
