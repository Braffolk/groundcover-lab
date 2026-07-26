// One-time card-proxy bake: orthographic capture of the GCMESH1 source mesh
// into one 256x256 tile per view (side, top), selected via viewport. MRT:
// target0 = albedo.rgb + coverage, target1 = plant-local unit normal packed as
// rgb = n * 0.5 + 0.5, FLIPPED into the hemisphere around the capture axis.
// Hardware depth keeps the nearest surface — the card shows the plant's
// visible shell from that direction.
//
// Two reasons the normal is stored as a flipped plain vector rather than a raw
// octahedral pair (which is what v3 did, and which is why 017 lit flat):
//  1. Two-sidedness. A blade's two faces carry opposite mesh normals, but a
//     card must light both like the front — the same rule 000-ground-truth
//     applies per fragment. Flipping toward the capture axis bakes it in.
//  2. Mip safety. The runtime samples this atlas at mip 1.5-3.5 for every
//     cached cluster. Box-averaging *octahedral* codes of opposing normals
//     cancels to (0,0), which decodes to exactly (0,1,0) — so the whole far
//     field lit with a straight-up normal: maximum half-lambert AND maximum
//     hemisphere term, i.e. a flat, blown-out field. Once every normal in a
//     tile sits in one hemisphere, an alpha-weighted box average is a genuine
//     aggregate normal and it cannot cancel; plain vectors also average
//     linearly, which octahedral codes do not do across the fold.

struct ViewU {
  right: vec4f,   // xyz = ortho right axis, w = half width (m)
  up: vec4f,      // xyz = ortho up axis,    w = half height (m)
  fwd: vec4f,     // xyz = axis toward capture camera, w = sMax (m)
  center: vec4f,  // xyz = mesh bbox center
  bounds_a: vec4f, // xyz = mesh bounds min
  bounds_b: vec4f, // xyz = mesh bounds max
}
@group(0) @binding(0) var<uniform> view_u: ViewU;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
}

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
  let p = view_u.bounds_a.xyz + q * (view_u.bounds_b.xyz - view_u.bounds_a.xyz);
  let rel = p - view_u.center.xyz;
  let cx = dot(rel, view_u.right.xyz) / view_u.right.w;
  let cy = dot(rel, view_u.up.xyz) / view_u.up.w;
  let s = dot(rel, view_u.fwd.xyz);
  let dlin = (view_u.fwd.w - s) / (2.0 * view_u.fwd.w);

  var o: VOut;
  o.pos = vec4f(cx, cy, clamp(dlin, 0.0, 1.0), 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  let e = vec2f(f32(a1.z), f32(a1.w)) / 65535.0 * 2.0 - 1.0;
  // Thin foliage is lit from both sides: flip the mesh normal into the
  // hemisphere around the capture axis so the card's front face is what gets
  // lit, and so the mip chain averages a coherent set (see header).
  var n = oct_decode(e);
  if (dot(n, view_u.fwd.xyz) < 0.0) {
    n = -n;
  }
  o.normal = n;
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.nrm = vec4f(normalize(i.normal) * 0.5 + 0.5, 1.0);
  return o;
}
