// Offscreen bake: renders the GCMESH1 source mesh orthographically from one
// hemi-octahedral node direction into one atlas tile (placed via viewport).
// MRT:
//   target0 rgba8unorm    = albedo rgb + coverage a
//   target1 rgba16float   = signed height h = dot(unit_pos, fwd)  (the relief
//                           heightfield, +1 = nearest to the capture camera),
//                           oct-encoded local-frame normal (flipped toward the
//                           capture direction, foliage is two-sided), spare.
// The runtime relief shader intersects eye rays against target1's heightfield.

#include "src/wgsl/gcmesh.wgsl"

struct Tile {
  center: vec3f, inv_r: f32,
  right: vec3f, _p0: f32,
  up: vec3f, _p1: f32,
  fwd: vec3f, _p2: f32,
  bmin: vec3f, _p3: f32,
  bmax: vec3f, _p4: f32,
}
@group(0) @binding(0) var<uniform> tile: Tile;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) h: f32,
}

// This experiment's OWN encoding for the geom target (y-derived), decoded by
// oct_decode() in relief.wgsl and octDecode() in bake.ts — a matched triple,
// unrelated to the GCMESH1 source convention.
fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  var e = vec2f(n.x, n.z) / max(s, 1e-6);
  if (n.y < 0.0) {
    e = (vec2f(1.0) - abs(vec2f(e.y, e.x))) * vec2f(sign(e.x), sign(e.y));
  }
  return e;
}

@vertex
fn vs(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = tile.bmin + q * (tile.bmax - tile.bmin);
  let rel = (p - tile.center) * tile.inv_r; // unit-sphere space

  // Orthographic projection onto the tile basis. clip.y is NEGATED so that
  // up-positive points land at texture v = 1 — the runtime samples with
  // v = dot(p, up)*0.5 + 0.5 and no flip.
  let cx = dot(rel, tile.right);
  let cy = -dot(rel, tile.up);
  let h = dot(rel, tile.fwd);          // +1 = closest to capture camera
  let cz = 0.5 - h * 0.5;              // [0,1], nearest wins with 'less'

  var o: VOut;
  o.pos = vec4f(cx, cy, cz, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  o.normal = gcmesh_normal_decode_u16(a1.z, a1.w);
  o.h = h;
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) geom: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  var n = normalize(i.normal);
  // Two-sided foliage: keep the normal on the capture-camera side.
  if (dot(n, tile.fwd) < 0.0) { n = -n; }
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.geom = vec4f(i.h, oct_encode(n), 1.0);
  return o;
}
