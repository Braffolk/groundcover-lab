// Offline (in-browser, once) view-set bake for the true-depth impostor.
//
// Renders the raw GCMESH1 mesh orthographically into one array layer per
// baked view. Every view has its OWN orthonormal basis (R, U, F) and its own
// half-extents (eu, ev, ew) taken from the plant's local AABB projected onto
// that basis, so each layer fills its tile and the runtime knows the exact
// metric meaning of every texel.
//
// Outputs per fragment:
//   target 0  rgb = authored albedo, a = coverage
//   target 1  rg = octahedral mesh-frame normal (flipped toward the bake
//             camera so both faces of a blade light like the front),
//             b  = signed depth along F remapped to [0,1] over [-ew, +ew],
//             a  = baked ambient occlusion from the canopy density grid
//
// The depth channel is the whole point: it is what lets the runtime rebuild a
// real 3D position for every texel, warp the imagery to the exact current view
// and write true per-pixel depth.

#include "src/wgsl/gcmesh.wgsl"

struct BakeView {
  r_axis: vec4f,  // xyz = R (mesh frame), w = eu
  u_axis: vec4f,  // xyz = U (mesh frame), w = ev
  f_axis: vec4f,  // xyz = F (toward the bake camera), w = ew
  center: vec4f,  // xyz = local capture centre in the mesh frame
  b_min: vec4f,   // mesh bounds min (dequantization)
  b_range: vec4f, // mesh bounds max - min
  ao_min: vec4f,  // xyz = AO grid min corner (mesh frame)
  ao_inv: vec4f,  // xyz = 1 / AO grid extent
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;
@group(0) @binding(1) var ao_grid: texture_3d<f32>;
@group(0) @binding(2) var ao_sampler: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) depth01: f32,
  @location(3) ao: f32,
}

/// Octahedral encode, y-primary — this experiment's OWN atlas convention; the
/// matching decode is oct_decode() in impostor.wgsl / carpet.wgsl and octDecode()
/// in bake.ts. Unrelated to the GCMESH1 source convention (gcmesh.wgsl).
fn oct_encode_mesh(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1.0e-6) {
    return vec2f(0.5, 0.5); // decodes to +Y
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

  let off = p - bake_view.center.xyz;
  let u = dot(off, bake_view.r_axis.xyz) / bake_view.r_axis.w;
  let v = dot(off, bake_view.u_axis.xyz) / bake_view.u_axis.w;
  let d = dot(off, bake_view.f_axis.xyz) / bake_view.f_axis.w;

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  if (dot(n, bake_view.f_axis.xyz) < 0.0) {
    n = -n;
  }

  var out: VOut;
  // Larger d == nearer the bake camera, so clip z = 0.5 - 0.5*d and a 'less'
  // depth test keeps the surface actually visible from this view.
  out.pos = vec4f(u, v, clamp(0.5 - 0.5 * d, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.depth01 = clamp(d * 0.5 + 0.5, 0.0, 1.0);
  // The AO grid is coarse (24x40x24) and the mesh is dense, so a vertex-rate
  // lookup is already far finer than the grid — no per-fragment work needed.
  out.ao = textureSampleLevel(ao_grid, ao_sampler, (p - bake_view.ao_min.xyz) * bake_view.ao_inv.xyz, 0.0).r;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) geo: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.geo = vec4f(oct_encode_mesh(normalize(in.nrm)), in.depth01, clamp(in.ao, 0.0, 1.0));
  return out;
}
