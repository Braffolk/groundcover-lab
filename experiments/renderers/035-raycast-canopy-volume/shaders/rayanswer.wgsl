// BAKE STAGE 2 — resolve the ray answers.
//
// One thread per (table texel, azimuth) of one elevation band. Each thread
// casts 4 sub-rays that enter the periodic canopy tile at the top plane and
// descend with the band's exact shear, and records the FIRST hit:
//
//   q   = vertical drop / canopy height        (the whole ray answer, closed
//   cov = fraction of sub-rays that hit         form at runtime)
//   n   = averaged surface normal at the hits
//
// This is the only place a ray is ever marched: at runtime the answer is a
// single texture fetch. Empty space is skipped through the coarse 4cm mask,
// so a typical ray costs a few dozen fetches here.

struct RayInfo {
  dims: vec4u,   // occupancy VX, VY, VZ, wordsX
  adims: vec4u,  // attribute AX, AY, AZ, table U
  cdims: vec4u,  // coarse CX, CY, CZ, wordsCX
  geom: vec4f,   // tile period, canopy height, fine voxel (m), coarse voxel (m)
  band: vec4f,   // tan(zenith), azimuth count, layer base, unused
  info: vec4f,   // max march steps, unused...
}

@group(0) @binding(0) var<uniform> rinfo: RayInfo;
@group(0) @binding(1) var<storage, read> occ: array<u32>;
@group(0) @binding(2) var<storage, read> coarse: array<u32>;
@group(0) @binding(3) var<storage, read> acc_n: array<i32>;
@group(0) @binding(4) var<storage, read_write> out_table: array<u32>;

fn wrap01(x: f32) -> f32 {
  return x - floor(x);
}

fn fine_occupied(pxz: vec2f, t: f32) -> bool {
  let period = rinfo.geom.x;
  let fy = 1.0 - t / rinfo.geom.y;
  if (fy < 0.0 || fy >= 1.0) { return false; }
  let vx = min(u32(wrap01(pxz.x / period) * f32(rinfo.dims.x)), rinfo.dims.x - 1u);
  let vy = min(u32(fy * f32(rinfo.dims.y)), rinfo.dims.y - 1u);
  let vz = min(u32(wrap01(pxz.y / period) * f32(rinfo.dims.z)), rinfo.dims.z - 1u);
  let word = (vy * rinfo.dims.z + vz) * rinfo.dims.w + (vx >> 5u);
  return (occ[word] & (1u << (vx & 31u))) != 0u;
}

fn coarse_occupied(pxz: vec2f, t: f32) -> bool {
  let period = rinfo.geom.x;
  let fy = 1.0 - t / rinfo.geom.y;
  if (fy < 0.0 || fy >= 1.0) { return false; }
  let cx = min(u32(wrap01(pxz.x / period) * f32(rinfo.cdims.x)), rinfo.cdims.x - 1u);
  let cy = min(u32(fy * f32(rinfo.cdims.y)), rinfo.cdims.y - 1u);
  let cz = min(u32(wrap01(pxz.y / period) * f32(rinfo.cdims.z)), rinfo.cdims.z - 1u);
  let word = (cy * rinfo.cdims.z + cz) * rinfo.cdims.w + (cx >> 5u);
  return (coarse[word] & (1u << (cx & 31u))) != 0u;
}

fn attr_normal(pxz: vec2f, t: f32) -> vec3f {
  let period = rinfo.geom.x;
  let fy = clamp(1.0 - t / rinfo.geom.y, 0.0, 0.999999);
  let ax = min(u32(wrap01(pxz.x / period) * f32(rinfo.adims.x)), rinfo.adims.x - 1u);
  let ay = min(u32(fy * f32(rinfo.adims.y)), rinfo.adims.y - 1u);
  let az = min(u32(wrap01(pxz.y / period) * f32(rinfo.adims.z)), rinfo.adims.z - 1u);
  let ai = ((ay * rinfo.adims.z + az) * rinfo.adims.x + ax) * 3u;
  let v = vec3f(f32(acc_n[ai]), f32(acc_n[ai + 1u]), f32(acc_n[ai + 2u]));
  let len = length(v);
  if (len < 1.0) { return vec3f(0.0, 1.0, 0.0); }
  return v / len;
}

struct MarchOut {
  hit: bool,
  q: f32,
  n: vec3f,
}

fn march(e: vec2f, sdir: vec2f) -> MarchOut {
  let height = rinfo.geom.y;
  let arc = sqrt(1.0 + dot(sdir, sdir));
  let dt_fine = rinfo.geom.z * 0.7 / arc;
  let dt_coarse = rinfo.geom.w * 0.6 / arc;
  let max_steps = u32(rinfo.info.x);

  var out: MarchOut;
  out.hit = false;
  out.q = 1.0;
  out.n = vec3f(0.0, 1.0, 0.0);

  var t = dt_fine * 0.5;
  var guard = 0u;
  loop {
    if (t >= height || guard >= max_steps) { break; }
    guard += 1u;
    if (!coarse_occupied(e + sdir * t, t)) {
      t += dt_coarse;
      continue;
    }
    let t_end = min(t + dt_coarse, height);
    var found = false;
    loop {
      if (t >= t_end || guard >= max_steps) { break; }
      guard += 1u;
      if (fine_occupied(e + sdir * t, t)) {
        found = true;
        break;
      }
      t += dt_fine;
    }
    if (found) {
      out.hit = true;
      out.q = clamp(t / height, 0.0, 1.0);
      out.n = attr_normal(e + sdir * t, t);
      break;
    }
  }
  return out;
}

/**
 * Octahedral encode, y-primary — the ray table's OWN convention, matched by
 * `oct_decode` in mipgen.wgsl and canopy.wgsl.
 */
fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1.0e-6) { return vec2f(0.0); }
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * select(-1.0, 1.0, n.x >= 0.0);
    let fv = (1.0 - abs(u)) * select(-1.0, 1.0, n.z >= 0.0);
    u = fu;
    v = fv;
  }
  return vec2f(u, v) * 0.5 + vec2f(0.5);
}

@compute @workgroup_size(8, 8, 1)
fn answer(@builtin(global_invocation_id) gid: vec3u) {
  let u_side = rinfo.adims.w;
  let n_az = u32(rinfo.band.y);
  if (gid.x >= u_side || gid.y >= u_side || gid.z >= n_az) { return; }

  let theta = 6.2831853071 * f32(gid.z) / f32(n_az);
  let sdir = vec2f(cos(theta), sin(theta)) * rinfo.band.x;

  var offs = array<vec2f, 4>(
    vec2f(0.125, 0.375), vec2f(0.625, 0.125), vec2f(0.375, 0.875), vec2f(0.875, 0.625),
  );
  let texel = rinfo.geom.x / f32(u_side);

  var hits = 0.0;
  var qsum = 0.0;
  var nsum = vec3f(0.0);
  for (var s = 0u; s < 4u; s++) {
    let e = (vec2f(f32(gid.x), f32(gid.y)) + offs[s]) * texel;
    let m = march(e, sdir);
    if (m.hit) {
      hits += 1.0;
      qsum += m.q;
      nsum += m.n;
    }
  }

  var q = 1.0;
  var nrm = vec3f(0.0, 1.0, 0.0);
  if (hits > 0.0) {
    q = qsum / hits;
    let l = length(nsum);
    if (l > 1.0e-4) { nrm = nsum / l; }
  }
  let cov = hits * 0.25;
  let oct = oct_encode(nrm);

  let r = u32(clamp(q, 0.0, 1.0) * 255.0 + 0.5);
  let g = u32(clamp(cov, 0.0, 1.0) * 255.0 + 0.5);
  let b = u32(clamp(oct.x, 0.0, 1.0) * 255.0 + 0.5);
  let a = u32(clamp(oct.y, 0.0, 1.0) * 255.0 + 0.5);

  let layer = u32(rinfo.band.z) + gid.z;
  out_table[(layer * u_side + gid.y) * u_side + gid.x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
