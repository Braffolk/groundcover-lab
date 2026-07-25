// Offline (in-browser, once) part bake. Renders a SUBSET of the raw GCMESH1
// mesh — one part's triangle range — orthographically into one atlas tile,
// using that part's own tight box. Fully general: every tile is described by an
// orthonormal basis (right, up, fwd) + box center + half extents, so side
// views, part views and straight-down views all go through the same path.
//
// Outputs:
//   target 0: rgb = authored albedo, a = coverage
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5, flipped toward the bake
//             camera (two-sided foliage lights the same from either face)
// No @group(0) frame include — the bake is self-contained.

struct TileView {
  right_axis: vec4f, // xyz: tile U axis (mesh frame), unit
  up_axis: vec4f,    // xyz: tile V axis (mesh frame), unit
  fwd_axis: vec4f,   // xyz: toward the bake camera, unit
  box_center: vec4f, // xyz: ortho box center in the mesh frame
  box_ext: vec4f,    // x,y,z = half extents along right/up/fwd
  b_min: vec4f,      // mesh bounds min (dequantization)
  b_range: vec4f,    // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> tile: TileView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
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
  let p = tile.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * tile.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;
  let oct = (vec2f(f32(q_attr.z), f32(q_attr.w)) / 65535.0) * 2.0 - 1.0;

  let off = p - tile.box_center.xyz;
  let u = dot(off, tile.right_axis.xyz) / tile.box_ext.x;
  let v = dot(off, tile.up_axis.xyz) / tile.box_ext.y;
  // Depth 0 at the box face nearest the bake camera.
  let d = 0.5 - 0.5 * (dot(off, tile.fwd_axis.xyz) / tile.box_ext.z);

  var n = oct_decode_mesh(oct);
  if (dot(n, tile.fwd_axis.xyz) < 0.0) {
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
