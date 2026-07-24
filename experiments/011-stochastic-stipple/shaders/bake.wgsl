// One-time bake: orthographic capture of the quantized GCMESH1 source mesh
// into one atlas tile per view. MRT: premultiplied albedo+coverage and
// premultiplied plant-local normal (faceforwarded toward the view).

struct BakeView {
  center: vec3f,
  r_card: f32,
  right: vec3f,
  r_depth: f32,
  up_axis: vec3f,
  _p0: f32,
  fwd: vec3f,
  _p1: f32,
  bmin: vec3f,
  _p2: f32,
  bmax: vec3f,
  _p3: f32,
}
@group(0) @binding(0) var<uniform> view_uni: BakeView;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
}

fn oct_decode(e: vec2f) -> vec3f {
  let u = e.x * 2.0 - 1.0;
  let v = e.y * 2.0 - 1.0;
  var x = u;
  var z = v;
  let y = 1.0 - abs(u) - abs(v);
  if (y < 0.0) {
    x = (1.0 - abs(v)) * sign(u);
    z = (1.0 - abs(u)) * sign(v);
  }
  return normalize(vec3f(x, y, z));
}

@vertex
fn vs(@location(0) q0: vec4<u32>, @location(1) q1: vec4<u32>) -> VOut {
  let qp = vec3f(vec3<u32>(q0.xyz)) / 65535.0;
  let pos = view_uni.bmin + qp * (view_uni.bmax - view_uni.bmin);
  let color = vec3f(f32(q0.w), f32(q1.x), f32(q1.y)) / 65535.0;
  let n = oct_decode(vec2f(f32(q1.z), f32(q1.w)) / 65535.0);

  let p = pos - view_uni.center;
  var out: VOut;
  out.pos = vec4f(
    dot(p, view_uni.right) / view_uni.r_card,
    dot(p, view_uni.up_axis) / view_uni.r_card,
    0.5 - 0.5 * dot(p, view_uni.fwd) / view_uni.r_depth,
    1.0,
  );
  out.color = color;
  // Faceforward toward the capture direction so cards light like the side
  // of the plant the camera sees.
  out.nrm = select(n, -n, dot(n, view_uni.fwd) < 0.0);
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) normal: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.normal = vec4f(normalize(in.nrm) * 0.5 + 0.5, 1.0);
  return out;
}
