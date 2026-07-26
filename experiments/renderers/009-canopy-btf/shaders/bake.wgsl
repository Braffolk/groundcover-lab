// Canopy BTF capture: orthographic render of the (replicated) periodic
// community tile along one hemi-oct bin direction, parameterised by the ray's
// ground-plane intersection. Texel (u,v) = ground point (u*T, 0, v*T);
// depth = 1 - y/topH so the fragment nearest the (infinitely far) viewer wins.
// MRT: target0 = albedo (a=1 marks a hit), target1 = (y/topH, oct normal, 0).

struct BakeU {
  dir: vec4f,   // xyz: bin direction (surface -> viewer), y > 0
  bmin: vec4f,  // xyz: mesh bounds min, w: tile size T (m)
  bmax: vec4f,  // xyz: mesh bounds max, w: capture topH (m)
}

@group(0) @binding(0) var<uniform> bu: BakeU;
// Per tile-copy instance: world offset x, z, yaw, scale.
@group(0) @binding(1) var<storage, read> inst: array<vec4f>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) hy: f32,
}

fn oct_decode16(e: vec2f) -> vec3f {
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

fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  var x = n.x / s;
  var z = n.z / s;
  if (n.y < 0.0) {
    let ox = (1.0 - abs(z)) * sign(x);
    let oz = (1.0 - abs(x)) * sign(z);
    x = ox;
    z = oz;
  }
  return vec2f(x, z) * 0.5 + vec2f(0.5);
}

@vertex
fn vs(
  @builtin(instance_index) ii: u32,
  @location(0) a: vec4u,
  @location(1) b: vec4u,
) -> VOut {
  let p = bu.bmin.xyz + (vec3f(vec3u(a.xyz)) / 65535.0) * (bu.bmax.xyz - bu.bmin.xyz);
  let color = vec3f(f32(a.w), f32(b.x), f32(b.y)) / 65535.0;
  let oct = vec2f(f32(b.z), f32(b.w)) / 65535.0;

  let tr = inst[ii];
  let c = cos(tr.z);
  let s = sin(tr.z);
  let ps = p * tr.w;
  let w = vec3f(c * ps.x + s * ps.z, ps.y, -s * ps.x + c * ps.z) + vec3f(tr.x, 0.0, tr.y);

  let n0 = oct_decode16(oct);
  let nr = vec3f(c * n0.x + s * n0.z, n0.y, -s * n0.x + c * n0.z);

  let T = bu.bmin.w;
  let topH = bu.bmax.w;
  let d = bu.dir.xyz;
  // Project along -d down to the ground plane; that intersection is the texel.
  let q = w.xz - (w.y / d.y) * d.xz;
  let uv = q / T;
  let depth = clamp(1.0 - w.y / topH, 0.0, 1.0);

  var out: VOut;
  out.pos = vec4f(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
  out.color = color;
  out.normal = nr;
  out.hy = clamp(w.y / topH, 0.0, 1.0);
  return out;
}

struct FOut {
  @location(0) alb: vec4f,
  @location(1) attr: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  var out: FOut;
  out.alb = vec4f(in.color, 1.0);
  out.attr = vec4f(in.hy, oct_encode(normalize(in.normal)), 0.0);
  return out;
}
