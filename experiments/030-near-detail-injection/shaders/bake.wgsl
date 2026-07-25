// Offline (in-browser, once) atlas bake: renders the raw GCMESH1 mesh
// orthographically into one atlas tile per view — 15 side views at 24deg azimuth
// steps plus 1 straight-down top view (a 4x4 grid). Two targets:
//
//   target 0  rgb = authored albedo, a = 1 (coverage comes from the downsample)
//   target 1  rg  = oct-encoded mesh-frame normal (flipped toward the capture
//                   axis so both faces of a blade light like the front)
//             b   = SIGNED DEPTH along the capture axis, (f*0.5+0.5) — this is
//                   what the runtime relief step walks the eye ray onto
//             a   = volumetric sky visibility sampled from a precomputed
//                   extinction/AO grid (self-occlusion + ground contact)
//
// The three view axes are handed in with 1/extent folded in, so the vertex
// stage is three dot products and the runtime uses the identical convention.
// No @group(0) frame include here — the bake is self-contained.

struct BakeView {
  au: vec4f,      // xyz: u axis (mesh frame, 1/extent folded in)
  av: vec4f,      // xyz: v axis
  af: vec4f,      // xyz: depth axis, points toward the bake camera
  centre: vec4f,  // xyz: capture centre (cx, ymid, cz); w: 1 = top view
  ao_org: vec4f,  // xyz: AO grid origin
  ao_inv: vec4f,  // xyz: 1 / AO grid extent
  b_min: vec4f,   // mesh bounds min (dequantization)
  b_range: vec4f, // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;
@group(0) @binding(1) var ao_grid: texture_3d<f32>;
@group(0) @binding(2) var ao_sampler: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) depth_n: f32,
  @location(3) ao_uvw: vec3f,
}

// Octahedral decode, y-primary — mirrors GcMesh.normalAt() in src/mesh/gcmesh.ts.
fn oct_decode_mesh(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

// Octahedral encode, y-primary — exact inverse of the decode above and of
// oct_decode_card() in cards.wgsl.
fn oct_encode_mesh(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1e-6) {
    return vec2f(0.5, 0.5);
  }
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * select(-1.0, 1.0, n.x >= 0.0);
    let fv = (1.0 - abs(u)) * select(-1.0, 1.0, n.z >= 0.0);
    u = fu;
    v = fv;
  }
  return vec2f(u, v) * 0.5 + 0.5;
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = bake_view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * bake_view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let oct = (vec2f(f32(q_attr.z), f32(q_attr.w)) / 65535.0) * 2.0 - 1.0;

  let off = p - bake_view.centre.xyz;
  let u = dot(off, bake_view.au.xyz);
  let v = dot(off, bake_view.av.xyz);
  let f = dot(off, bake_view.af.xyz);

  var n = oct_decode_mesh(oct);
  if (dot(n, bake_view.af.xyz) < 0.0) {
    n = -n;
  }

  var out: VOut;
  // Clip y is flipped relative to the texture v axis (clip +1 = texture row 0);
  // depth 0 = nearest the bake camera, so the depth test keeps the first
  // surface along the capture axis.
  out.pos = vec4f(u, -v, clamp(0.5 - 0.5 * f, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.depth_n = f;
  out.ao_uvw = (p - bake_view.ao_org.xyz) * bake_view.ao_inv.xyz;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) geo: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  let ao = textureSample(ao_grid, ao_sampler, in.ao_uvw).r;
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.geo = vec4f(oct_encode_mesh(normalize(in.nrm)), clamp(in.depth_n * 0.5 + 0.5, 0.0, 1.0), ao);
  return out;
}
