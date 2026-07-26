// Offline (in-browser, once) view capture: renders the raw GCMESH1 mesh
// orthographically into one atlas tile per baked view direction, using that
// view's TIGHT silhouette box so the tile is fully used.
//
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = mesh normal * 0.5 + 0.5 (flipped toward the bake camera —
//             foliage is two-sided), a = normalized view depth q
//             (0 at the near plane of the tight box, 1 at the far plane)
//
// No @group(0) frame include — the bake is self-contained.

#include "src/wgsl/gcmesh.wgsl"

struct ViewUni {
  r_axis: vec4f,   // xyz = tile right axis (plant-local), w = box centre x
  u_axis: vec4f,   // xyz = tile up axis, w = box centre y
  b_axis: vec4f,   // xyz = toward the bake camera, w = d_near
  box: vec4f,      // x = half width, y = half height, z = depth range, w unused
  anchor: vec4f,   // xyz = mesh-space origin of the plant-local frame
  b_min: vec4f,    // mesh bounds min (dequantization)
  b_range: vec4f,  // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> view_uni: ViewUni;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) q_depth: f32,
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = view_uni.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * view_uni.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let pl = p - view_uni.anchor.xyz;
  let xr = dot(pl, view_uni.r_axis.xyz);
  let yu = dot(pl, view_uni.u_axis.xyz);
  let dd = dot(pl, view_uni.b_axis.xyz);
  let q = clamp((view_uni.b_axis.w - dd) / view_uni.box.z, 0.0, 1.0);

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  if (dot(n, view_uni.b_axis.xyz) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(
    (xr - view_uni.r_axis.w) / view_uni.box.x,
    (yu - view_uni.u_axis.w) / view_uni.box.y,
    q,
    1.0,
  );
  out.color = color;
  out.nrm = n;
  out.q_depth = q;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm_depth: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm_depth = vec4f(normalize(in.nrm) * 0.5 + 0.5, clamp(in.q_depth, 0.0, 1.0));
  return out;
}
