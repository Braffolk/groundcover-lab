// Offscreen bake pass: renders the source mesh orthographically from one
// hemi-octahedral view direction into a single atlas tile (set via viewport).
// One draw per tile; the per-tile basis + framing come from the Tile uniform.
// MRT: target0 = albedo(rgb) + coverage(a), target1 = local-frame normal.
// (Capture path adapted from 005-octa-impostors — the runtime is entirely
// different: see bin.wgsl / resolve.wgsl.)

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
}

// Standard octahedral normal decode (y up) — mirrors GcMesh.normalAt().
fn oct_decode(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

@vertex
fn vs(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = tile.bmin + q * (tile.bmax - tile.bmin);
  let rel = p - tile.center;
  // Orthographic projection onto the tile basis; z mapped to [0,1] depth
  // (points nearer the capture camera get smaller z, so depthCompare 'less').
  let cx = dot(rel, tile.right) * tile.inv_r;
  let cy = dot(rel, tile.up) * tile.inv_r;
  let cz = 0.5 - dot(rel, tile.fwd) * tile.inv_r * 0.5;

  var o: VOut;
  o.pos = vec4f(cx, cy, cz, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  let e = vec2f(f32(a1.z), f32(a1.w)) / 65535.0 * 2.0 - 1.0;
  o.normal = oct_decode(e);
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
  o.normal = vec4f(normalize(i.normal) * 0.5 + 0.5, 1.0);
  return o;
}
