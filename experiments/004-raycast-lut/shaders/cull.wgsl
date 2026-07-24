#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"

// Camera-region instance builder. Walks ONLY the scatter cells within
// max_dist of the camera (a fixed-size grid, independent of stand radius /
// total plant count), evaluates the shared scatter twin for one stand entry
// and appends surviving plants to that entry's slice of the instance buffer,
// bumping its indirect draw count. 100 plants or 1 billion: the dispatch is
// the same size.
//
// Survivors are then filtered by the EXACT quad the vertex shader would build
// (same sphere centre, same wind shear, same perspective growth, same fade):
// anything whose quad is fully outside the side frustum planes, or whose fade
// factor is zero, would have produced no pixels at all — so rejecting it here
// is image-identical and skips its vertex+setup work entirely.

struct CullParams {
  // Side frustum planes (left, right, bottom, top), xyz normalized.
  planes: array<vec4f, 4>,
  center: vec3f, // baked bounding-sphere centre, anchored mesh space
  radius: f32,
  base_cell: vec2i,
  entry_index: u32,
  seed: u32,
  max_dist: f32,
  stand_r: f32,
  capacity: u32,
  top_h: f32,
  fade_band: f32,
  sway_mul: f32,
  _pad0: f32,
  _pad1: f32,
}
@group(1) @binding(0) var<uniform> cp: CullParams;
// Indirect draw args, 4 u32 per stand entry: [6, instanceCount, 0, 0].
@group(1) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>>;
// 2 vec4 per plant: [x, y, z, yaw], [scale, entry, phase, 0].
@group(1) @binding(2) var<storage, read_write> plants: array<vec4f>;

// Sphere fully behind a plane -> the whole quad is off screen.
fn outside(pl: vec4f, c: vec3f, r: f32) -> bool {
  return dot(pl.xyz, c) + pl.w < -r;
}

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

  // --- rebuild this plant's quad exactly as vs_main will ---
  let st = stand_table[cp.entry_index];
  let k = wind_sway(c.pos, frame.time, st.sway, c.phase) * cp.sway_mul;
  let cy = cos(c.yaw);
  let sy = sin(c.yaw);
  let c_rot = vec2f(
    cp.center.x * cy + cp.center.z * sy,
    -cp.center.x * sy + cp.center.z * cy,
  );
  let wc = c.pos + vec3f(c_rot.x, cp.center.y, c_rot.y) * c.scale + k * (cp.center.y / cp.top_h);
  let rw = cp.radius * c.scale * 1.06 + length(k) * 0.6;
  let d = distance(wc, frame.camera_pos);

  // fade == 0 means every fragment of this quad discards (camera inside the
  // plant, or past the fade band) — drop it before it costs a draw.
  let near_fade = clamp((d / rw - 0.55) / 0.65, 0.0, 1.0);
  let dist_fade = clamp((cp.max_dist - d) / max(cp.fade_band, 0.01), 0.0, 1.0);
  if (near_fade * dist_fade <= 0.0) {
    return;
  }

  // Bounding sphere of the screen-facing quad: half-size to a corner is
  // size * sqrt(2). Same `grow` as the vertex shader, so the bound is tight.
  let grow = clamp(d / sqrt(max(d * d - rw * rw, rw * rw * 0.04)), 1.0, 5.0);
  let bound = rw * grow * 1.4143;
  if (outside(cp.planes[0], wc, bound) || outside(cp.planes[1], wc, bound)
    || outside(cp.planes[2], wc, bound) || outside(cp.planes[3], wc, bound)) {
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
