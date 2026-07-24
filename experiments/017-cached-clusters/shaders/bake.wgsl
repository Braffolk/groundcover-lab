// One-time card-proxy bake: orthographic capture of the GCMESH1 source mesh
// into one 256x256 tile per view (side, top), selected via viewport. MRT:
// target0 = albedo.rgb + coverage, target1 = plant-local octahedral normal.
// Hardware depth keeps the nearest surface — the card shows the plant's
// visible shell from that direction.

struct ViewU {
  right: vec4f,   // xyz = ortho right axis, w = half width (m)
  up: vec4f,      // xyz = ortho up axis,    w = half height (m)
  fwd: vec4f,     // xyz = axis toward capture camera, w = sMax (m)
  center: vec4f,  // xyz = mesh bbox center
  bounds_a: vec4f, // xyz = mesh bounds min
  bounds_b: vec4f, // xyz = mesh bounds max
}
@group(0) @binding(0) var<uniform> view_u: ViewU;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
}

fn oct_decode(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

fn sign_not_zero2(v: vec2f) -> vec2f {
  return select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  var p = vec2f(n.x, n.z) / max(s, 1e-6);
  if (n.y < 0.0) {
    p = (vec2f(1.0) - abs(vec2f(p.y, p.x))) * sign_not_zero2(p);
  }
  return p * 0.5 + 0.5;
}

@vertex
fn vs(@location(0) a0: vec4<u32>, @location(1) a1: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = view_u.bounds_a.xyz + q * (view_u.bounds_b.xyz - view_u.bounds_a.xyz);
  let rel = p - view_u.center.xyz;
  let cx = dot(rel, view_u.right.xyz) / view_u.right.w;
  let cy = dot(rel, view_u.up.xyz) / view_u.up.w;
  let s = dot(rel, view_u.fwd.xyz);
  let dlin = (view_u.fwd.w - s) / (2.0 * view_u.fwd.w);

  var o: VOut;
  o.pos = vec4f(cx, cy, clamp(dlin, 0.0, 1.0), 1.0);
  o.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  let e = vec2f(f32(a1.z), f32(a1.w)) / 65535.0 * 2.0 - 1.0;
  o.normal = oct_decode(e);
  return o;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(i: VOut) -> FOut {
  var o: FOut;
  o.albedo = vec4f(i.color, 1.0);
  o.nrm = vec4f(oct_encode(normalize(i.normal)), 0.0, 1.0);
  return o;
}
