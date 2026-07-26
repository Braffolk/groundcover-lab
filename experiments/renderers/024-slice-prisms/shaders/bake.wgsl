// Offline (in-browser, once) prism bake. The raw GCMESH1 mesh is rendered
// orthographically into one atlas tile per PRISM: a prism is the part of the
// plant inside one slice of the depth (or height) axis, so every tile holds a
// disjoint third of the geometry and the three of them reconstruct the plant
// exactly.
//
//   24 tiles — 8 azimuths x 3 vertical depth slabs (slab 0 = nearest the
//              bake camera). Cut planes are per-azimuth equal-mass quantiles.
//    3 tiles — 3 horizontal height slabs seen straight down (slab 0 = top).
//
// The merged (single-card LOD) tiles are composited from these on the CPU —
// front-most covered slab wins, which is exactly what a z-buffer would do.
//
// Outputs:
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5 (flipped toward the bake
//             camera so both faces of a blade light like the front),
//             a = the winning surface's normalized depth along the bake axis.
//             The prism resolve ignores that alpha; the CARPET resolve uses it
//             as a free full-resolution HEIGHT FIELD of the visible surface —
//             free because it falls out of the same depth test that already
//             decided which surface the texel shows.
// No @group(0) frame include here — the bake is self-contained.

#include "src/wgsl/gcmesh.wgsl"

struct BakeView {
  right_axis: vec4f, // xyz: tile U axis (mesh frame)
  up_axis: vec4f,    // xyz: tile V axis for the top view
  fwd_axis: vec4f,   // xyz: toward the bake camera; w: 1 = top view
  capture: vec4f,    // cx, cz, y0, y1
  extent: vec4f,     // x: horizontal support radius r; y,z: slab [d_lo, d_hi);
                     // w: 1 = write the normalized depth as colour instead of
                     // albedo (the carpet bake histograms the VISIBLE surface
                     // height before it decides where to cut its slices)
  b_min: vec4f,      // mesh bounds min (dequantization)
  b_range: vec4f,    // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) slab_d: f32, // normalized depth along the bake axis
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
    d = (y1 - p.y) / (y1 - y0);
  } else {
    v = ((p.y - y0) / (y1 - y0)) * 2.0 - 1.0;
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
  out.slab_d = d;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  // Prism membership. Triangles straddling a cut plane are trimmed per
  // fragment, so the three slabs partition the geometry exactly.
  if (in.slab_d < bake_view.extent.y || in.slab_d >= bake_view.extent.z) {
    discard;
  }
  var out: FOut;
  // Depth probe: the colour target carries the normalized height of whatever
  // surface won the depth test, so the CPU can quantile the VISIBLE surface.
  if (bake_view.extent.w > 0.5) {
    out.albedo = vec4f(in.slab_d, in.slab_d, in.slab_d, 1.0);
    out.nrm = vec4f(0.5, 0.5, 0.5, 1.0);
    return out;
  }
  out.albedo = vec4f(in.color, 1.0);
  // alpha carries the depth of whatever won here (see the header): the carpet
  // resolve turns it into a height field, the prism resolve ignores it.
  out.nrm = vec4f(normalize(in.nrm) * 0.5 + 0.5, clamp(in.slab_d, 0.0, 1.0));
  return out;
}
