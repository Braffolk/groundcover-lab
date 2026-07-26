// Shared cone-frustum proxy math: chord (entry/exit) intersection in closed
// form, the (phi, meridian) surface chart, and octahedral normal codecs.
// Mirrored constants live in ../chords.ts — keep in sync.

const CF_A_E: f32 = 48.0;  // entry azimuth bins
const CF_M_E: f32 = 32.0;  // entry meridian bins
const CF_A_X: f32 = 32.0;  // exit relative-azimuth samples (tile width +1 gutter)
const CF_M_X: f32 = 24.0;  // exit meridian samples
const CF_W: f32 = 1584.0;  // atlas width  = A_E * (A_X + 1)
const CF_H: f32 = 768.0;   // atlas height = M_E * M_X
const CF_TWO_PI: f32 = 6.28318530718;

// Frustum proxy dimensions (unit plant scale, proxy frame: base at y=0).
struct FrustumDims {
  r0: f32,       // radius at y=0
  r1: f32,       // radius at y=h
  h: f32,
  side_len: f32,
  cap_b: f32,    // bottom-cap meridian extent (chart [-cap_b, 0])
  cap_t: f32,    // top-cap meridian extent (chart [1, 1+cap_t])
}

// Faces: 0 = lateral, 1 = bottom cap, 2 = top cap.
struct ChordHit {
  ok: bool,
  t0: f32,
  t1: f32,
  face0: u32,
  face1: u32,
}

// Closed-form ray vs (cone lateral surface ∩ y-slab). No marching: one
// quadratic + a slab. The frustum keeps its apex outside [0, h] (r0, r1 > 0),
// so the inside region within the slab is convex and the ray meets it in at
// most one interval.
fn frustum_chord(o: vec3f, d: vec3f, fd: FrustumDims) -> ChordHit {
  var out: ChordHit;
  out.ok = false;
  let big = 1e8;

  var s0 = -big; var s1 = big;
  var sf0 = 0u; var sf1 = 0u;
  if (abs(d.y) < 1e-6) {
    if (o.y < 0.0 || o.y > fd.h) { return out; }
  } else {
    let ta = (0.0 - o.y) / d.y;
    let tb = (fd.h - o.y) / d.y;
    if (d.y > 0.0) { s0 = ta; s1 = tb; sf0 = 1u; sf1 = 2u; }
    else { s0 = tb; s1 = ta; sf0 = 2u; sf1 = 1u; }
  }

  let k = (fd.r1 - fd.r0) / fd.h;
  let ry = fd.r0 + k * o.y;
  let qa = d.x * d.x + d.z * d.z - k * k * d.y * d.y;
  let qb = 2.0 * (o.x * d.x + o.z * d.z - k * ry * d.y);
  let qc = o.x * o.x + o.z * o.z - ry * ry;
  var c0 = -big; var c1 = big;
  if (abs(qa) > 1e-7) {
    let disc = qb * qb - 4.0 * qa * qc;
    if (disc < 0.0) {
      if (qa > 0.0) { return out; } // entirely outside the cone
      // qa < 0, no roots: ray is inside the infinite cone for all t.
    } else {
      let sq = sqrt(disc);
      let ra = (-qb - sq) / (2.0 * qa);
      let rb = (-qb + sq) / (2.0 * qa);
      let lo = min(ra, rb);
      let hi = max(ra, rb);
      if (qa > 0.0) {
        c0 = lo; c1 = hi;
      } else {
        // Inside on (-inf, lo] u [hi, +inf); only one branch can genuinely
        // overlap the slab (convexity) — pick the longer overlap.
        let upper_len = s1 - max(hi, s0);
        let lower_len = min(lo, s1) - s0;
        if (upper_len <= 0.0 && lower_len <= 0.0) { return out; }
        if (upper_len >= lower_len) { c0 = hi; } else { c1 = lo; }
      }
    }
  } else {
    // Ray parallel to the cone surface: linear equation.
    if (abs(qb) > 1e-7) {
      let t = -qc / qb;
      if (qb > 0.0) { c1 = t; } else { c0 = t; }
    } else if (qc > 0.0) { return out; }
  }

  var t0 = s0; var f0 = sf0;
  if (c0 > t0) { t0 = c0; f0 = 0u; }
  var t1 = s1; var f1 = sf1;
  if (c1 < t1) { t1 = c1; f1 = 0u; }
  if (t1 <= t0 + 1e-5) { return out; }
  out.ok = true;
  out.t0 = t0; out.t1 = t1;
  out.face0 = f0; out.face1 = f1;
  return out;
}

// Surface chart (phi, m) of a point known to lie on face `face`.
fn chart_of(p: vec3f, face: u32, fd: FrustumDims) -> vec2f {
  let phi = atan2(p.z, p.x);
  var m: f32;
  if (face == 0u) {
    m = clamp(p.y / fd.h, 0.0, 1.0);
  } else if (face == 2u) {
    let rho = length(p.xz);
    m = 1.0 + (1.0 - clamp(rho / fd.r1, 0.0, 1.0)) * fd.cap_t;
  } else {
    let rho = length(p.xz);
    m = -(1.0 - clamp(rho / fd.r0, 0.0, 1.0)) * fd.cap_b;
  }
  return vec2f(phi, m);
}

// Chart -> surface point (used by the bake gather to place bin centers).
fn point_of(phi: f32, m: f32, fd: FrustumDims) -> vec3f {
  var rho: f32;
  var y: f32;
  if (m < 0.0) {
    rho = fd.r0 * (1.0 + m / max(fd.cap_b, 1e-5));
    y = 0.0;
  } else if (m <= 1.0) {
    rho = mix(fd.r0, fd.r1, m);
    y = m * fd.h;
  } else {
    rho = fd.r1 * (1.0 - (m - 1.0) / max(fd.cap_t, 1e-5));
    y = fd.h;
  }
  return vec3f(rho * cos(phi), y, rho * sin(phi));
}

fn chart_m_lo(fd: FrustumDims) -> f32 { return -fd.cap_b; }
fn chart_m_range(fd: FrustumDims) -> f32 { return 1.0 + fd.cap_b + fd.cap_t; }

// Inner-tile UV for entry bin (bi, bj) + continuous relative chord coords.
// dphi01 in [0,1) (= dphi/2pi), mx01 in [0,1] over the meridian chart range.
fn chord_uv(bi: f32, bj: f32, dphi01: f32, mx01: f32) -> vec2f {
  let u = (bi * (CF_A_X + 1.0) + 0.5 + dphi01 * CF_A_X) / CF_W;
  let v = (bj * CF_M_X + 0.5 + mx01 * (CF_M_X - 1.0)) / CF_H;
  return vec2f(u, v);
}

// This experiment's OWN octahedral codec (y primary axis), used for the chord
// field's aggregate normals and for the view-direction grid — a matched pair
// with oct_nrm_encode() below, NOT the GCMESH1 source convention (that lives in
// src/wgsl/gcmesh.wgsl). e in [-1,1]^2.
fn oct_nrm_decode(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

// sign() that never returns 0 — sign(0.0) = 0 collapses straight-up/down
// directions onto the octahedron center instead of its corners.
fn nz_sign(v: f32) -> f32 { return select(-1.0, 1.0, v >= 0.0); }

fn oct_nrm_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  let px = n.x / max(s, 1e-6);
  let pz = n.z / max(s, 1e-6);
  if (n.y < 0.0) {
    return vec2f((1.0 - abs(pz)) * nz_sign(px), (1.0 - abs(px)) * nz_sign(pz));
  }
  return vec2f(px, pz);
}

// Yaw rotation with the sin/cos supplied by the caller. The fragment stage
// only ever rotates by a PER-INSTANCE yaw, so the transcendentals are
// evaluated once in the vertex stage and passed down flat.
fn rot_y_cs(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn rot_y(v: vec3f, a: f32) -> vec3f {
  return rot_y_cs(v, cos(a), sin(a));
}
