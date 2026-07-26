#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"

// Camera-region instance builder. Walks ONLY the scatter cells within
// max_dist of the camera (a fixed-size grid, independent of stand radius /
// total plant count), evaluates the shared scatter twin for one stand entry
// and appends surviving plants to that entry's slice of the instance buffer,
// bumping its indirect draw count. 100 plants or 1 billion: the dispatch is
// the same size.
//
// EVERY slot of a cell must be visited: a carpet entry has carpet_div^2 slots
// (484 for the bog moss), NOT SCATTER_MAX_PER_CELL. The dispatch therefore
// carries ceil(slots / 128) workgroups in z and the slot index spans them;
// with 128 slots hardcoded only the first 6 of 22 tile rows per cell existed
// and the mat rendered as 1.1m stripes with 3m gaps.
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
  half_len: f32, // circumradius of the baked box — bounds the quad's support sizing
  _pad1: f32,
}
@group(1) @binding(0) var<uniform> cp: CullParams;
// Indirect draw args, 4 u32 per stand entry: [6, instanceCount, 0, 0].
@group(1) @binding(1) var<storage, read_write> draw_args: array<atomic<u32>>;
// Packed instance records, u32 words. Stride depends on the entry:
//   scattered plant (8 words): pos.xyz, yaw, scale, entry, phase, 0 (as f32 bits)
//   carpet tile     (2 words): cell.xz (+32768, 16b each) | slot | yaw index
// A carpet tile carries no float at all — its position is the grid node, which
// the vertex shader rebuilds with the same arithmetic the scatter used, and its
// scale is the entry's constant. 8B instead of 32B is what keeps 745k
// life-size moss tiles inside the VRAM budget.
@group(1) @binding(2) var<storage, read_write> plants: array<u32>;

const CULL_WG: u32 = 128u;
// Slack on the quad bound for entries that shear into the terrain, covering the
// largest vertical span the shear + per-fragment conforming can add. Only the
// cull bound is loosened (a conservative bound may only over-accept); the
// vertex shader computes the tight sheared radius.
const SLOPE_BOUND_SLACK: f32 = 1.6;
// Must match CONFORM_SLACK in impostor.wgsl (the per-fragment conforming slack
// the vertex shader pads the quad with).
const SLOPE_QUAD_PAD: f32 = 0.035;

// Sphere fully behind a plane -> the whole quad is off screen.
fn outside(pl: vec4f, c: vec3f, r: f32) -> bool {
  return dot(pl.xyz, c) + pl.w < -r;
}

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let cell = cp.base_cell + vec2i(i32(wg.x), i32(wg.y));
  // Mirror Scatter.region(): every cell whose span overlaps the stand aabb
  // contributes all its candidates (no per-plant position clamp).
  let lo = i32(floor(-cp.stand_r / SCATTER_CELL_SIZE));
  let hi = i32(floor(cp.stand_r / SCATTER_CELL_SIZE));
  if (cell.x < lo || cell.x > hi || cell.y < lo || cell.y > hi) {
    return;
  }
  let slot = wg.z * CULL_WG + lane;
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
  let is_carpet = st.carpet_div > 0.0;
  let k = wind_sway(c.pos, frame.time, st.sway, c.phase) * cp.sway_mul;
  let cy = cos(c.yaw);
  let sy = sin(c.yaw);
  let c_rot = vec2f(
    cp.center.x * cy + cp.center.z * sy,
    -cp.center.x * sy + cp.center.z * cy,
  );
  let wc = c.pos + vec3f(c_rot.x, cp.center.y, c_rot.y) * c.scale + k * (cp.center.y / cp.top_h);
  var rw = cp.radius * c.scale * 1.06 + length(k) * 0.6;
  if (st.slope_align > 0.0) {
    rw = rw * SLOPE_BOUND_SLACK;
  }
  let d = distance(wc, frame.camera_pos);

  // fade == 0 means every fragment of this quad discards (camera inside the
  // plant, or past the fade band) — drop it before it costs a draw. A carpet
  // has NO near fade: a mat you stand on must not open a hole under you.
  var near_fade = clamp((d / rw - 0.55) / 0.65, 0.0, 1.0);
  if (is_carpet) {
    near_fade = 1.0;
  }
  let dist_fade = clamp((cp.max_dist - d) / max(cp.fade_band, 0.01), 0.0, 1.0);
  if (near_fade * dist_fade <= 0.0) {
    return;
  }

  // Bounding sphere of the screen-facing quad. The vertex shader sizes that
  // quad by the baked box's support along the view axes, so the conservative
  // bound here is the box's circumradius (which can slightly exceed the
  // bounding-sphere radius, since a box corner may hold no vertex) with the
  // same perspective growth, and a corner factor of sqrt(2).
  let hlen = (cp.half_len * c.scale + length(k)) * 1.1 + SLOPE_QUAD_PAD;
  let grow = clamp(d / max(d - hlen, hlen * 0.2), 1.0, 5.0);
  let bound = hlen * grow * 1.4143;
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
  if (is_carpet) {
    let base = idx * 2u;
    plants[base] = u32(cell.x + 32768) | (u32(cell.y + 32768) << 16u);
    plants[base + 1u] = slot | (u32(round(c.yaw / QUARTER_TURN)) << 16u);
    return;
  }
  let base = idx * 8u;
  plants[base] = bitcast<u32>(c.pos.x);
  plants[base + 1u] = bitcast<u32>(c.pos.y);
  plants[base + 2u] = bitcast<u32>(c.pos.z);
  plants[base + 3u] = bitcast<u32>(c.yaw);
  plants[base + 4u] = bitcast<u32>(c.scale);
  plants[base + 5u] = bitcast<u32>(f32(cp.entry_index));
  plants[base + 6u] = bitcast<u32>(c.phase);
  plants[base + 7u] = 0u;
}
