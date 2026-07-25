#include "src/wgsl/frame.wgsl"

// Terrain sampling — manual bilinear over textureLoad (no sampler) so the
// arithmetic is deterministic and mirrored step-for-step (with f32 rounding)
// by Terrain.height()/normal() in src/scene/terrain.ts. Heightmap texels are
// f16-quantized on CPU; both sides therefore interpolate identical values.

fn terrain_texel_uv(xz: vec2f) -> vec2f {
  let n = frame.terrain_resolution;
  return (xz / frame.terrain_size + 0.5) * n - 0.5;
}

fn terrain_load(t: vec2i) -> vec4f {
  let n = i32(frame.terrain_resolution);
  let c = clamp(t, vec2i(0), vec2i(n - 1));
  return textureLoad(terrain_heightmap, c, 0);
}

// Bilinear (h, nx, nz) at world xz.
fn terrain_sample(xz: vec2f) -> vec3f {
  let uv = terrain_texel_uv(xz);
  let base = floor(uv);
  let f = uv - base;
  let i0 = vec2i(base);
  let s00 = terrain_load(i0).rgb;
  let s10 = terrain_load(i0 + vec2i(1, 0)).rgb;
  let s01 = terrain_load(i0 + vec2i(0, 1)).rgb;
  let s11 = terrain_load(i0 + vec2i(1, 1)).rgb;
  let a = s00 + (s10 - s00) * f.x;
  let b = s01 + (s11 - s01) * f.x;
  return a + (b - a) * f.y;
}

fn terrain_height(xz: vec2f) -> f32 {
  return terrain_sample(xz).x;
}

fn terrain_normal(xz: vec2f) -> vec3f {
  let s = terrain_sample(xz);
  let ny = sqrt(max(1.0 - s.y * s.y - s.z * s.z, 0.0));
  return vec3f(s.y, ny, s.z);
}

/**
 * Orthonormal basis for placing a plant on sloped ground. Columns are
 * (tangent, up, bitangent): multiply a plant-local offset by it, so local +Y
 * is the plant's own up axis and local XZ is its footprint.
 *
 * `align` blends up from world-vertical toward the terrain normal — pass the
 * species' stand_table[].slope_align. A flat mat MUST use 1.0 or it buries one
 * edge and floats the other on any slope; tall plants want only a little,
 * because grass grows upright no matter what it grows on.
 */
fn plant_basis(xz: vec2f, yaw: f32, align: f32) -> mat3x3f {
  let up = normalize(mix(vec3f(0.0, 1.0, 0.0), terrain_normal(xz), clamp(align, 0.0, 1.0)));
  // Yawed tangent, projected into the plane so the basis stays orthonormal.
  var t = vec3f(cos(yaw), 0.0, -sin(yaw));
  let proj = t - up * dot(up, t);
  // Degenerate only where the ground is a vertical wall; fall back to any
  // in-plane direction rather than emitting NaNs.
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  return mat3x3f(t, up, cross(t, up));
}
