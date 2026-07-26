// Offscreen bake: renders the GCMESH1 source mesh orthographically into one
// atlas tile per view (viewport + dynamic-offset uniform selects the tile).
//
// 8 views per species:
//   tiles 0..5 — azimuthal ortho captures every 60° (the 3 static fins show
//                tile k from their front side and tile k+3, U-mirrored, from
//                their back side → real content change as you orbit),
//   tile  6    — top-down capture of the LOWER vertical slab of the plant,
//   tile  7    — top-down capture of the UPPER vertical slab.
// The slab split gives the two horizontal cards genuinely different content,
// so looking down produces parallax between them instead of a flat poster.
//
// MRT: target0 = albedo(rgb) + coverage(a), target1 = mesh-local normal.

#include "src/wgsl/gcmesh.wgsl"

struct Tile {
  center: vec3f, inv_hu: f32,   // framing center; 1/half-extent along `right`
  right: vec3f, inv_hv: f32,    //                 1/half-extent along `up`
  fwd: vec3f, inv_hd: f32,      // toward capture camera; 1/depth half-extent
  up: vec3f, _p0: f32,
  bmin: vec3f, slab_y0: f32,    // dequantization bounds + slab filter (mesh y)
  bmax: vec3f, slab_y1: f32,
}
@group(0) @binding(0) var<uniform> tile: Tile;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) mesh_y: f32,
}

@vertex
fn vs(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = tile.bmin + q * (tile.bmax - tile.bmin);
  let rel = p - tile.center;
  // Orthographic projection onto the tile basis. Points nearer the capture
  // camera (larger rel·fwd) get smaller depth (depthCompare 'less').
  let cx = dot(rel, tile.right) * tile.inv_hu;
  let cy = dot(rel, tile.up) * tile.inv_hv;
  let cz = 0.5 - dot(rel, tile.fwd) * tile.inv_hd * 0.5;

  var o: VOut;
  o.pos = vec4f(cx, cy, cz, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  o.normal = gcmesh_normal_decode_u16(a1.z, a1.w);
  o.mesh_y = p.y;
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) normal: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  // Slab filter for the top-down captures (full range for azimuth tiles).
  if (i.mesh_y < tile.slab_y0 || i.mesh_y > tile.slab_y1) { discard; }
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.normal = vec4f(normalize(i.normal) * 0.5 + 0.5, 1.0);
  return o;
}
