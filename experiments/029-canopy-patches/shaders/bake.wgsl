// Offline (in-browser, once) canopy-patch bake. One atlas tile per SLICE; a
// slice is an orthographic capture of the raw GCMESH1 mesh restricted to one
// depth slab (side slices) or one height slab (crown slices) of the plant.
// Each slice is framed on the TIGHT bounding rect of the foliage it contains,
// so no texel is spent on empty margin.
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5 (flipped toward the capture
//             camera so both faces of a blade light like the front), a = 1
// No @group(0) frame include — the bake is self-contained.

struct SliceView {
  right_axis: vec4f, // xyz: capture U axis (mesh frame)
  v_axis: vec4f,     // xyz: capture V axis (crown slices; side slices use height)
  fwd_axis: vec4f,   // xyz: toward the capture camera; w: 1 = crown (top-down)
  rect: vec4f,       // a0, a1, b0, b1 — tight framing in capture (u, v) metres
  slab: vec4f,       // x: slab lo, y: slab hi, z: depth normalisation radius
  capture: vec4f,    // cx, cz (clump centre), y0, y1
  b_min: vec4f,      // mesh bounds min (dequantisation)
  b_range: vec4f,    // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> slice: SliceView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  // Slab coordinate: depth along the capture axis (side) or height (crown).
  @location(2) slab_coord: f32,
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
  let p = slice.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * slice.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let oct = (vec2f(f32(q_attr.z), f32(q_attr.w)) / 65535.0) * 2.0 - 1.0;

  let off = p - vec3f(slice.capture.x, 0.0, slice.capture.y);
  let is_top = slice.fwd_axis.w > 0.5;
  let y0 = slice.capture.z;
  let y1 = slice.capture.w;

  let u = dot(off, slice.right_axis.xyz);
  var v: f32;
  var coord: f32;
  var d: f32;
  if (is_top) {
    v = dot(off, slice.v_axis.xyz);
    coord = p.y;
    d = (y1 - p.y) / max(y1 - y0, 1e-4); // higher = nearer the top-down camera
  } else {
    v = p.y;
    coord = dot(off, slice.fwd_axis.xyz);
    d = 0.5 - 0.5 * (coord / slice.slab.z); // larger depth toward camera = nearer
  }

  // Tight framing: map the capture rect onto clip [-1, 1].
  let uc = (slice.rect.x + slice.rect.y) * 0.5;
  let ur = max((slice.rect.y - slice.rect.x) * 0.5, 1e-5);
  let vc = (slice.rect.z + slice.rect.w) * 0.5;
  let vr = max((slice.rect.w - slice.rect.z) * 0.5, 1e-5);

  var n = oct_decode_mesh(oct);
  let toward_cam = select(slice.fwd_axis.xyz, vec3f(0.0, 1.0, 0.0), is_top);
  if (dot(n, toward_cam) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f((u - uc) / ur, (v - vc) / vr, clamp(d, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.slab_coord = coord;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  // Slab clipping BEFORE the depth test (discard suppresses the depth write),
  // so each slice records the frontmost surface *inside its own slab* — that is
  // what makes the slabs a partition of the plant rather than depth peels.
  if (in.slab_coord < slice.slab.x || in.slab_coord > slice.slab.y) {
    discard;
  }
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm = vec4f(normalize(in.nrm) * 0.5 + 0.5, 1.0);
  return out;
}
