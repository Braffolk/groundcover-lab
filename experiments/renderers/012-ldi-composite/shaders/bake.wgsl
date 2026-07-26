// Offscreen LDI bake pass. Renders the GCMESH1 source mesh orthographically
// from one key direction into one tile of a layer strip (viewport-selected).
// Depth peeling with a MINIMUM SEPARATION: a fragment of layer L survives only
// if it lies at least `sep` metres BEHIND layer L-1's surface at that pixel.
// Plain peeling wastes layers on near-coincident surfaces of the same tuft;
// the separation forces each successive layer deeper into the crown, which is
// what makes a handful of layers useful for thin foliage.
//
// MRT: target0 = albedo.rgb + coverage, target1 ("aux") = oct normal (rg) +
// 16-bit linear depth split across (b, a). Layer L-1's aux strip is bound as
// the peel input; hardware depth test then keeps the NEAREST surviving
// fragment, i.e. the first surface beyond the separation threshold.

#include "src/wgsl/gcmesh.wgsl"

struct TileU {
  right: vec4f,      // xyz = ortho right axis, w = half width (m)
  up: vec4f,         // xyz = ortho up axis,    w = half height (m)
  fwd: vec4f,        // xyz = axis toward capture camera, w = sMax (m)
  center: vec4f,     // xyz = mesh bbox center, w = 1 for the first layer
  bounds_a: vec4f,   // xyz = mesh bounds min,  w = sep normalized to [0,1] depth
  bounds_b: vec4f,   // xyz = mesh bounds max
}
@group(0) @binding(0) var<uniform> tile: TileU;
@group(1) @binding(0) var prev_aux: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) dlin: f32,   // linear depth in [0,1]: 0 = nearest to capture cam
}

fn sign_not_zero2(v: vec2f) -> vec2f {
  return select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

// Octahedral encode (y up) — mirrored by oct_decode_yup() in ldi.wgsl.
fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  var p = vec2f(n.x, n.z) / max(s, 1e-6);
  if (n.y < 0.0) {
    p = (vec2f(1.0) - abs(vec2f(p.y, p.x))) * sign_not_zero2(p);
  }
  return p * 0.5 + 0.5;
}

@vertex
fn vs(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = tile.bounds_a.xyz + q * (tile.bounds_b.xyz - tile.bounds_a.xyz);
  let rel = p - tile.center.xyz;
  let cx = dot(rel, tile.right.xyz) / tile.right.w;
  let cy = dot(rel, tile.up.xyz) / tile.up.w;
  let s = dot(rel, tile.fwd.xyz);
  let dlin = (tile.fwd.w - s) / (2.0 * tile.fwd.w);

  var o: VOut;
  o.pos = vec4f(cx, cy, dlin, 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  o.normal = gcmesh_normal_decode_u16(a1.z, a1.w);
  o.dlin = dlin;
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) aux: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  if (tile.center.w < 0.5) {
    // Peel: only surfaces at least `sep` behind the previous layer survive.
    let prev = textureLoad(prev_aux, vec2i(i.pos.xy), 0);
    let prev_d = (prev.b * 255.0 * 256.0 + prev.a * 255.0) / 65535.0;
    if (i.dlin <= prev_d + tile.bounds_a.w) { discard; }
  }

  let d16 = clamp(i.dlin, 0.0, 1.0) * 65535.0;
  let hi = floor(d16 / 256.0);
  let lo = floor(d16 - hi * 256.0);

  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.aux = vec4f(oct_encode(normalize(i.normal)), hi / 255.0, lo / 255.0);
  return o;
}
