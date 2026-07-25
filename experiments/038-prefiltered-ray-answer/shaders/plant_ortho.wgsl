// BAKE STAGE 1 — per-species directional "first hit" images of ONE source
// plant, rasterized straight from the GCMESH1 mesh.
//
// For a baked direction d, this is a plain orthographic render along d with a
// depth test on dot(p, d): the winning fragment of every pixel is exactly the
// first surface a ray of direction d meets. That makes the tile a table of ray
// answers for one plant, which stage 2 composites into the patch answer.
//
//   target 0 (surf): rgb = authored albedo, a = coverage (binary, 1 sample)
//   target 1 (geom): r = height fraction (y - y0) / (y1 - y0),
//                    gb = octahedral normal * 0.5 + 0.5 (mesh frame, flipped
//                         toward the incoming ray so both blade faces light),
//                    a = 1 (coverage marker)
// Both are effectively premultiplied (coverage is 0 or 1), so stage 2 can
// bilinear-filter them and divide by the filtered coverage.
// Self-contained: no @group(0) frame include.

struct Slice {
  axis_a: vec4f,  // xyz: ortho right (unit), w: ortho half-extent R (mesh units)
  axis_b: vec4f,  // xyz: ortho up (unit)
  dir: vec4f,     // xyz: view direction (unit, y < 0)
  centre: vec4f,  // xyz: capture centre (mesh frame), w: 1 / (y1 - y0)
  bmin: vec4f,    // xyz: mesh bounds min (dequantization), w: y0
  bspan: vec4f,   // xyz: mesh bounds span
}
@group(0) @binding(0) var<uniform> sl: Slice;

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

fn oct_encode_mesh(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1e-6) {
    return vec2f(0.0);
  }
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * select(-1.0, 1.0, n.x >= 0.0);
    let fv = (1.0 - abs(u)) * select(-1.0, 1.0, n.z >= 0.0);
    u = fu;
    v = fv;
  }
  return vec2f(u, v);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) h01: f32,
}

@vertex
fn vs(@location(0) qp: vec4<u32>, @location(1) qa: vec4<u32>) -> VOut {
  // Vertex record: [x y z r] [g b octU octV], u16 UNORM against the bounds.
  let p = sl.bmin.xyz + (vec3f(vec3<u32>(qp.xyz)) / 65535.0) * sl.bspan.xyz;
  let color = vec3f(f32(qp.w), f32(qa.x), f32(qa.y)) / 65535.0;
  let oct = (vec2f(f32(qa.z), f32(qa.w)) / 65535.0) * 2.0 - 1.0;

  let off = p - sl.centre.xyz;
  let r = sl.axis_a.w;
  let u = dot(off, sl.axis_a.xyz) / r;
  let v = dot(off, sl.axis_b.xyz) / r;
  // depth grows along d, so 'less' keeps the FIRST hit along the ray.
  let w = dot(off, sl.dir.xyz) / (2.0 * r) + 0.5;

  var n = oct_decode_mesh(oct);
  if (dot(n, sl.dir.xyz) > 0.0) {
    n = -n;
  }

  var out: VOut;
  out.pos = vec4f(u, v, clamp(w, 0.0, 1.0), 1.0);
  out.color = color;
  out.nrm = n;
  out.h01 = clamp((p.y - sl.bmin.w) * sl.centre.w, 0.0, 1.0);
  return out;
}

struct FOut {
  @location(0) surf: vec4f,
  @location(1) geom: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  let oct = oct_encode_mesh(normalize(in.nrm));
  var out: FOut;
  out.surf = vec4f(in.color, 1.0);
  out.geom = vec4f(in.h01, oct * 0.5 + 0.5, 1.0);
  return out;
}
