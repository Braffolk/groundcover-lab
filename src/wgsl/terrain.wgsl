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

// ---------------------------------------------------------------------------
// Fitting a plant to the ground.
//
// These are PRIMITIVES, not a prescribed method. How precisely a renderer
// conforms its representation to the terrain is the experiment's decision, and
// different representations want genuinely different answers — a rigid tilt, a
// plane fit over the footprint, per-vertex conforming, or warping the query
// inside a ray/volume lookup. The stand only says HOW MUCH a species should
// conform (`stand_table[i].slope_align`, a botanical property: moss follows
// the ground, grass grows upright); the mechanism and its fidelity are yours.
// See the ladder of options in CLAUDE.md.
// ---------------------------------------------------------------------------

/// A locally fitted ground plane. `up` is its normal; `h` is the fitted height
/// at the query point, which is NOT the same as terrain_height(xz) — over a
/// bumpy footprint the fitted plane sits at the local mean instead of chasing
/// the exact centre sample.
struct TerrainPlane {
  up: vec3f,
  h: f32,
}

/**
 * Least-squares ground plane over a footprint of the given radius, from 5
 * taps. For anything wider than a few centimetres this is a better rigid fit
 * than the point normal, which only knows about one spot. Use more taps of
 * your own if your footprint is large or your terrain regime is rougher —
 * nothing here stops you.
 */
fn terrain_plane_fit(xz: vec2f, radius: f32) -> TerrainPlane {
  let c = terrain_height(xz);
  let hx0 = terrain_height(xz - vec2f(radius, 0.0));
  let hx1 = terrain_height(xz + vec2f(radius, 0.0));
  let hz0 = terrain_height(xz - vec2f(0.0, radius));
  let hz1 = terrain_height(xz + vec2f(0.0, radius));
  let dhdx = (hx1 - hx0) / (2.0 * radius);
  let dhdz = (hz1 - hz0) / (2.0 * radius);
  var p: TerrainPlane;
  p.up = normalize(vec3f(-dhdx, 1.0, -dhdz));
  // For symmetric taps the LS plane passes through their mean at the centre.
  p.h = (c + hx0 + hx1 + hz0 + hz1) * 0.2;
  return p;
}

/**
 * Orthonormal basis (tangent, up, bitangent) around an arbitrary up vector,
 * yawed about it. Multiply a plant-local offset by it: local +Y becomes the
 * plant's own up axis, local XZ its footprint. Decoupled from where `up` came
 * from, so it works with a point normal, a plane fit, or your own construction.
 */
fn plant_basis_from_up(up_in: vec3f, yaw: f32) -> mat3x3f {
  let up = normalize(up_in);
  var t = vec3f(cos(yaw), 0.0, -sin(yaw));
  let proj = t - up * dot(up, t);
  // Degenerate only where the ground is a vertical wall; fall back to any
  // in-plane direction rather than emitting NaNs.
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  return mat3x3f(t, up, cross(t, up));
}

/// Convenience for the simplest option: rigid tilt from the point normal,
/// blended toward vertical by `align`. One of several valid choices, not the
/// recommended one.
fn plant_basis(xz: vec2f, yaw: f32, align: f32) -> mat3x3f {
  return plant_basis_from_up(mix(vec3f(0.0, 1.0, 0.0), terrain_normal(xz), clamp(align, 0.0, 1.0)), yaw);
}
