// Offline (in-browser, once) atlas bake. The raw GCMESH1 mesh is rendered
// orthographically into one atlas tile per view: 8 side views at 45deg azimuth
// steps plus 1 straight-down canopy view. Every view is described by the SAME
// generic frame — an orthonormal (R, V, F) basis with half-extents
// (ru, rv, rw) around an origin Ov in the mesh frame — so side and top tiles
// decode identically at runtime:
//
//   u = 0.5 + 0.5 * dot(p - Ov, R) / ru
//   v = 0.5 - 0.5 * dot(p - Ov, V) / rv     (v grows downward in the texture)
//   s = 0.5 + 0.5 * dot(p - Ov, F) / rw     <- the stored HEIGHTFIELD
//
// Outputs:
//   target 0: rgb = authored albedo, a = coverage (1 where the mesh drew)
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5 (flipped toward F so both
//             faces of a blade light like the front), a = s
// Depth = 1 - s with 'less' compare, so the surviving fragment per texel is
// the one CLOSEST to the capture camera — i.e. exactly the front surface whose
// height the runtime warp needs.

#include "src/wgsl/gcmesh.wgsl"

struct BakeView {
  r_axis: vec4f, // xyz = R, w = ru
  v_axis: vec4f, // xyz = V, w = rv
  f_axis: vec4f, // xyz = F, w = rw
  origin: vec4f, // xyz = Ov (mesh frame)
  b_min: vec4f,  // mesh bounds min (dequantization)
  b_range: vec4f,// mesh bounds max - min
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) height: f32,
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = bake_view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * bake_view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let off = p - bake_view.origin.xyz;
  let u = dot(off, bake_view.r_axis.xyz) / bake_view.r_axis.w;
  let v = -dot(off, bake_view.v_axis.xyz) / bake_view.v_axis.w;
  let s = clamp(0.5 + 0.5 * dot(off, bake_view.f_axis.xyz) / bake_view.f_axis.w, 0.0, 1.0);

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  if (dot(n, bake_view.f_axis.xyz) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(u, v, 1.0 - s, 1.0);
  out.color = color;
  out.nrm = n;
  out.height = s;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) geom: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.geom = vec4f(normalize(in.nrm) * 0.5 + 0.5, clamp(in.height, 0.0, 1.0));
  return out;
}
