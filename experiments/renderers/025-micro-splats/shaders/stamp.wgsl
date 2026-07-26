// Stamp source render (in-browser, once per species): one square-pixel
// orthographic side view of the raw GCMESH1 mesh. bake.ts then cuts 16 64px
// windows out of it, picked so their coverage spans [0,1] — those become the
// micro-splat stamp atlas, i.e. real blade imagery rather than a procedural
// blob. No @group(0) frame include: the bake is self-contained.

struct StampView {
  center: vec4f,   // xyz: capture centre (mesh frame), w: half-extent (m)
  b_min: vec4f,    // mesh bounds min (dequantization)
  b_range: vec4f,  // mesh bounds max - min
  _pad: vec4f,
}
@group(0) @binding(0) var<uniform> stamp_view: StampView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
}

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = stamp_view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * stamp_view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let s = stamp_view.center.w;
  let off = (p - stamp_view.center.xyz) / s;
  var out: VOut;
  out.pos = vec4f(off.x, off.y, clamp(0.5 - 0.5 * off.z, 0.0, 1.0), 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
