// Offline (in-browser, once) bake of the raw GCMESH1 mesh into
//   (a) one atlas tile per (node, azimuth): node 0 is the whole plant, nodes
//       1..4 are its four xz quadrants — a spatial partition of the TRIANGLES,
//       so the four tiles reassemble the plant exactly, at four different
//       depths;
//   (b) one straight-down tile used later to composite the tileable canopy
//       texture of the continuous shell.
//
// Targets:
//   0: rgb = authored albedo, a = coverage
//   1: rgb = mesh-frame normal * 0.5 + 0.5 (flipped toward the bake camera so
//      both faces of a blade light like the front), a = height fraction of the
//      plant box (top view only; 1 for side views)
// No @group(0) frame include — the bake is self-contained.

struct BakeView {
  right_axis: vec4f, // xyz: tile U axis (world); w: 1 = top view
  up_axis: vec4f,    // xyz: tile V axis for the top view
  fwd_axis: vec4f,   // xyz: toward the bake camera
  box_c: vec4f,      // node center xyz; w: horizontal support radius
  box_y: vec4f,      // y0, y1 of the tile's vertical range
  b_min: vec4f,      // mesh bounds min (dequantization)
  b_range: vec4f,    // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) height: f32,
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

@vertex
fn vs(@location(0) q_pos: vec4<u32>, @location(1) q_attr: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = bake_view.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * bake_view.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let oct = (vec2f(f32(q_attr.z), f32(q_attr.w)) / 65535.0) * 2.0 - 1.0;

  let off = p - bake_view.box_c.xyz;
  let hw = bake_view.box_c.w;
  let y0 = bake_view.box_y.x;
  let y1 = bake_view.box_y.y;
  let is_top = bake_view.right_axis.w > 0.5;

  let u = dot(off, bake_view.right_axis.xyz) / hw;
  var v: f32;
  var d: f32;
  var height = 1.0;
  if (is_top) {
    v = dot(off, bake_view.up_axis.xyz) / hw;
    d = (y1 - p.y) / (y1 - y0);
    height = (p.y - y0) / (y1 - y0);
  } else {
    v = ((p.y - y0) / (y1 - y0)) * 2.0 - 1.0;
    d = 0.5 - 0.5 * (dot(off, bake_view.fwd_axis.xyz) / hw);
  }

  var n = oct_decode_mesh(oct);
  let toward_cam = select(bake_view.fwd_axis.xyz, vec3f(0.0, 1.0, 0.0), is_top);
  if (dot(n, toward_cam) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(u, v, clamp(d, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.height = clamp(height, 0.0, 1.0);
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) aux: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.aux = vec4f(normalize(in.nrm) * 0.5 + 0.5, in.height);
  return out;
}
