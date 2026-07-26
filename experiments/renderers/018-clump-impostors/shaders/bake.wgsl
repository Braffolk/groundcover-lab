// Offline (in-browser, once) clump capture: renders one UNIT of the raw
// GCMESH1 mesh — either one of the K sub-clumps (a cluster of whole blades) or
// the merged whole plant — orthographically into one atlas tile per view.
// 8 side views at 45deg azimuth steps plus 1 straight-down top view. Outputs:
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5 (flipped toward the bake
//             camera so both faces of a blade light like the front), a = 1
// The unit selection is the index range passed to drawIndexed, so this shader
// is identical for every unit; only the capture box in the uniform changes.
// No @group(0) frame include here — the bake is self-contained.

#include "src/wgsl/gcmesh.wgsl"

struct BakeView {
  right_axis: vec4f, // xyz: card U axis (mesh frame)
  up_axis: vec4f,    // xyz: card V axis for the top view
  fwd_axis: vec4f,   // xyz: toward the bake camera; w: 1 = top view
  capture: vec4f,    // cx, cz, y0, y1 of THIS unit
  extent: vec4f,     // x: this unit's horizontal support radius r
  b_min: vec4f,      // mesh bounds min (dequantization)
  b_range: vec4f,    // mesh bounds max - min
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

  let off = p - vec3f(bake_view.capture.x, 0.0, bake_view.capture.y);
  let r = bake_view.extent.x;
  let y0 = bake_view.capture.z;
  let y1 = bake_view.capture.w;
  let is_top = bake_view.fwd_axis.w > 0.5;

  let u = dot(off, bake_view.right_axis.xyz) / r;
  var v: f32;
  var d: f32;
  if (is_top) {
    v = dot(off, bake_view.up_axis.xyz) / r;
    d = (y1 - p.y) / max(y1 - y0, 1e-5);
  } else {
    v = ((p.y - y0) / max(y1 - y0, 1e-5)) * 2.0 - 1.0;
    d = 0.5 - 0.5 * (dot(off, bake_view.fwd_axis.xyz) / r);
  }

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  let toward_cam = select(bake_view.fwd_axis.xyz, vec3f(0.0, 1.0, 0.0), is_top);
  if (dot(n, toward_cam) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(u, v, clamp(d, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm = vec4f(normalize(in.nrm) * 0.5 + 0.5, 1.0);
  return out;
}
