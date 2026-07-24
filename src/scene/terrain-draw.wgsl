#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"

// Harness base-pass terrain: bufferless grid, vertices derived from
// vertex_index and displaced by the shared heightmap. Draw 6*QUADS*QUADS.

const QUADS: u32 = 256u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  // Corner offsets for the two triangles of a quad.
  // CCW seen from above (+y) — front faces up, back-face culling keeps tops.
  var corners = array<vec2u, 6>(
    vec2u(0u, 0u), vec2u(0u, 1u), vec2u(1u, 0u),
    vec2u(1u, 0u), vec2u(0u, 1u), vec2u(1u, 1u),
  );
  let quad = vi / 6u;
  let corner = corners[vi % 6u];
  let q = vec2u(quad % QUADS, quad / QUADS);
  let grid = vec2f(q + corner) / f32(QUADS);
  let xz = (grid - 0.5) * frame.terrain_size;
  let world = vec3f(xz.x, terrain_height(xz), xz.y);
  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let n = terrain_normal(in.world.xz);
  let slope = 1.0 - n.y;
  // Soil showing through on slopes; subtle deterministic mottling so the
  // ground is not a flat billiard table under the groundcover.
  let cellIdx = vec2u(vec2i(floor(in.world.xz * 2.0)) + vec2i(16384));
  let mottle = hash_f32(hash3(78123u, cellIdx.x, cellIdx.y)) * 0.12;
  let grass = vec3f(0.16, 0.22, 0.10) * (0.92 + mottle);
  let soil = vec3f(0.26, 0.20, 0.14);
  let albedo = mix(grass, soil, clamp(slope * 6.0, 0.0, 1.0));
  var color = light_surface(albedo, n, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
