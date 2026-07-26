// Offline (in-browser, once) slab bake. The raw GCMESH1 mesh is rendered
// orthographically once per (view group, slab) into its own atlas tile:
//   - 8 azimuth view groups, each split into 3 DEPTH SLABS along that view's
//     axis (front / middle / back) plus one merged tile,
//   - 1 top view group, split into 3 HEIGHT slabs plus one merged tile.
// The slab test is per FRAGMENT (interpolated depth along the view axis), so
// no triangle sorting is needed and slab boundaries are exact.
//
// Each instance of the draw is one tile: the record supplies the view axes,
// the slab range and the NDC sub-rectangle of the atlas to render into. Tiles
// never overlap, so one shared depth buffer serves the whole atlas and one
// pass produces every tile of a view group.
//
// Outputs:
//   target 0: rgb = authored albedo, a = 1 (coverage after downsampling)
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5, flipped toward the bake
//             camera so both faces of a blade light like the front

#include "src/wgsl/gcmesh.wgsl"

struct TileView {
  right_axis: vec4f, // xyz: tile U axis in the mesh frame
  up_axis: vec4f,    // xyz: tile V axis (top view only); w: 1 = top view
  fwd_axis: vec4f,   // xyz: toward the bake camera (side views)
  slab: vec4f,       // x: slab min (exclusive), y: slab max (inclusive)
  ndc: vec4f,        // xy: NDC offset, zw: NDC scale of this tile's rect
}

struct BakeInfo {
  capture: vec4f, // cx, cz, y0, y1
  extent: vec4f,  // x: horizontal support radius R
  b_min: vec4f,   // mesh bounds min (dequantization)
  b_range: vec4f, // mesh bounds max - min
}

@group(0) @binding(0) var<storage, read> tile_views: array<TileView>;
@group(0) @binding(1) var<uniform> bake_info: BakeInfo;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  // Slab coordinate: depth along the view axis (side) or height (top).
  // Orthographic, so screen-space interpolation of it is exact.
  @location(2) slab_t: f32,
  @location(3) @interpolate(flat) slab_range: vec2f,
}

@vertex
fn vs(
  @builtin(instance_index) ii: u32,
  @location(0) q_pos: vec4<u32>,
  @location(1) q_attr: vec4<u32>,
) -> VOut {
  let view = tile_views[ii];

  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = bake_info.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * bake_info.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let cx = bake_info.capture.x;
  let cz = bake_info.capture.y;
  let y0 = bake_info.capture.z;
  let y1 = bake_info.capture.w;
  let r = bake_info.extent.x;
  let off = p - vec3f(cx, 0.0, cz);
  let is_top = view.up_axis.w > 0.5;

  // Tile-space coords, v = 0 at the TOP row (texture convention).
  var tu: f32;
  var tv: f32;
  var slab_t: f32;
  var depth01: f32;
  if (is_top) {
    tu = dot(off, view.right_axis.xyz) / r * 0.5 + 0.5;
    tv = dot(off, view.up_axis.xyz) / r * 0.5 + 0.5;
    slab_t = p.y;
    depth01 = (y1 - p.y) / max(y1 - y0, 1e-5);
  } else {
    tu = dot(off, view.right_axis.xyz) / r * 0.5 + 0.5;
    tv = 1.0 - (p.y - y0) / max(y1 - y0, 1e-5);
    slab_t = dot(off, view.fwd_axis.xyz);
    depth01 = 0.5 - 0.5 * (slab_t / r);
  }

  let base_ndc = vec2f(tu * 2.0 - 1.0, 1.0 - tv * 2.0);
  let ndc = base_ndc * view.ndc.zw + view.ndc.xy;

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  let toward_cam = select(view.fwd_axis.xyz, vec3f(0.0, 1.0, 0.0), is_top);
  if (dot(n, toward_cam) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(ndc, clamp(depth01, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.slab_t = slab_t;
  out.slab_range = view.slab.xy;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  if (in.slab_t <= in.slab_range.x || in.slab_t > in.slab_range.y) {
    discard;
  }
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm = vec4f(normalize(in.nrm) * 0.5 + 0.5, 1.0);
  return out;
}
