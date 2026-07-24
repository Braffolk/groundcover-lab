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
