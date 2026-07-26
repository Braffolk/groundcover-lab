// Offline (in-browser, once) view-set bake: the raw GCMESH1 mesh rendered
// orthographically, once per hemi-octahedral view node, into one atlas tile.
// The projection basis is built by the SAME rule the runtime uses
// (view_right() in impostor_info.wgsl), and each view gets its own extents —
// the plant AABB projected onto that basis — so no tile wastes texels on
// empty margin and the runtime can reproject into any view exactly.
//
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = local-frame normal * 0.5 + 0.5, flipped toward the bake
//             camera so the supersample average of a two-sided blade does not
//             cancel to zero. a = 1.
//
// Self-contained: no @group(0) frame include.

#include "src/wgsl/gcmesh.wgsl"

struct BakeView {
  right: vec4f,   // xyz view right axis, w = half extent u (m)
  up: vec4f,      // xyz view up axis,    w = half extent v (m)
  fwd: vec4f,     // xyz toward the camera, w = half extent along fwd (depth)
  centre: vec4f,  // mesh-frame capture centre xyz
  b_min: vec4f,   // mesh bounds min (dequantization)
  b_range: vec4f, // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = bake_view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * bake_view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let off = p - bake_view.centre.xyz;
  // Orthographic: +u right, +v up. NDC y points up, so framebuffer row 0 is
  // the TOP of the tile — the runtime undoes that with tex_v = 0.5 - 0.5*v.
  let u = dot(off, bake_view.right.xyz) / bake_view.right.w;
  let v = dot(off, bake_view.up.xyz) / bake_view.up.w;
  let d = 0.5 - 0.5 * dot(off, bake_view.fwd.xyz) / bake_view.fwd.w;

  var out: VOut;
  out.pos = vec4f(u, v, clamp(d, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  // Grass blades are single-sided sheets; store the face that looks at this
  // view so the 2x2 supersample resolve averages agreeing normals instead of
  // cancelling front against back.
  var n = normalize(in.nrm);
  if (dot(n, bake_view.fwd.xyz) < 0.0) {
    n = -n;
  }
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm = vec4f(n * 0.5 + 0.5, 1.0);
  return out;
}
