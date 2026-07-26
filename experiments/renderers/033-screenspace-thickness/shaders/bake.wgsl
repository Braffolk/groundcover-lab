// Offline (in-browser, once) depth-shell bake. The raw GCMESH1 mesh is
// rendered orthographically once per view into a supersampled tile:
//
//   front pass (depthCompare less):
//     target0  rgb = authored albedo, a = coverage
//     target1  rgb = mesh-frame normal * 0.5 + 0.5 (flipped toward the bake
//              camera so both faces of a blade light like the front),
//              a   = normalized depth of the FRONT surface (0 = nearest)
//   back pass (depthCompare greater, depth cleared to 0):
//     target0  r   = normalized depth of the BACK surface
//
// front/back together give per-texel thickness; the front depth is what the
// runtime depth shell (and the baked burial AO) is built from.
// No @group(0) frame include — the bake is self-contained.

struct BakeView {
  right_axis: vec4f, // xyz: card U axis (mesh frame)
  up_axis: vec4f,    // xyz: card V axis (mesh frame)
  fwd_axis: vec4f,   // xyz: toward the bake camera
  center: vec4f,     // xyz: capture box center (mesh frame)
  extent: vec4f,     // su, sv, sd  (half-extents along right/up/fwd)
  b_min: vec4f,      // mesh bounds min (dequantization)
  b_range: vec4f,    // mesh bounds max - min
}
@group(0) @binding(0) var<uniform> bake_view: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) depth01: f32,
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

  let off = p - bake_view.center.xyz;
  let u = dot(off, bake_view.right_axis.xyz) / bake_view.extent.x;
  let v = dot(off, bake_view.up_axis.xyz) / bake_view.extent.y;
  let d = dot(off, bake_view.fwd_axis.xyz) / bake_view.extent.z;
  let depth01 = clamp(0.5 - 0.5 * d, 0.0, 1.0);

  var n = oct_decode_mesh(oct);
  if (dot(n, bake_view.fwd_axis.xyz) < 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(u, v, depth01, 1.0);
  out.color = color;
  out.nrm = n;
  out.depth01 = depth01;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm_depth: vec4f,
}

@fragment
fn fs_front(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm_depth = vec4f(normalize(in.nrm) * 0.5 + 0.5, in.depth01);
  return out;
}

@fragment
fn fs_back(in: VOut) -> @location(0) vec4f {
  return vec4f(in.depth01, 0.0, 0.0, 1.0);
}
