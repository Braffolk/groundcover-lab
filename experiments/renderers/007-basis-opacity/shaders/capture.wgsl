// Offscreen bake capture: renders the GCMESH1 source mesh orthographically
// from one (azimuth, elevation) view into one TILE-wide column of the ring
// atlas (set via viewport). One draw per azimuth; V draws per elevation ring.
// MRT: target0 = albedo(rgb) + coverage(a),
//      target1 = mesh-frame normal(xyz * 0.5 + 0.5) + a depth channel.
// For the upright ring capture the depth channel is the signed distance from
// the card plane toward the capture camera in bounding-radius units, remapped
// to [0,1]; the CARPET capture reuses the same pass with a top-down basis and
// maps it to absolute height above the tile base instead (depth_scale/bias).
//
// `period` > 0 replicates the mesh over a 3x3 block of the periodic community
// tile (one instance per lattice cell), which is what makes a top-down capture
// of a carpet tile exactly periodic: geometry that overhangs one edge is the
// geometry that must reappear on the opposite edge. period = 0 with one
// instance is the plain single-specimen capture.

#include "src/wgsl/gcmesh.wgsl"

struct Tile {
  center: vec3f, inv_r: f32,
  right: vec3f, _p0: f32,
  up: vec3f, _p1: f32,
  fwd: vec3f, _p2: f32,
  bmin: vec3f, _p3: f32,
  bmax: vec3f, _p4: f32,
  period: f32, depth_scale: f32, depth_bias: f32, _p5: f32,
}
@group(0) @binding(0) var<uniform> tile: Tile;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) store_depth: f32, // depth channel, already mapped to [0,1]
}

@vertex
fn vs(@builtin(instance_index) inst: u32, @location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  // Periodic lattice offset (instance 0 of a 1-instance draw with period 0 is
  // the identity, so the ring capture is bit-identical to before).
  let lat = vec2f(f32(i32(inst % 3u) - 1), f32(i32(inst / 3u) - 1)) * tile.period;
  let p = tile.bmin + q * (tile.bmax - tile.bmin) + vec3f(lat.x, 0.0, lat.y);
  let rel = p - tile.center;
  // Orthographic projection onto the card basis; nearer the capture camera
  // means smaller clip z (depthCompare 'less' keeps the front surface).
  let cx = dot(rel, tile.right) * tile.inv_r;
  let cy = dot(rel, tile.up) * tile.inv_r;
  let d = dot(rel, tile.fwd) * tile.inv_r; // [-1,1], + = toward camera
  let cz = 0.5 - d * 0.5;

  var o: VOut;
  o.pos = vec4f(cx, cy, cz, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  o.normal = gcmesh_normal_decode_u16(a1.z, a1.w);
  o.store_depth = dot(rel, tile.fwd) * tile.depth_scale + tile.depth_bias;
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) normal: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.normal = vec4f(normalize(i.normal) * 0.5 + 0.5, clamp(i.store_depth, 0.0, 1.0));
  return o;
}
