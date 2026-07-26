#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/hash.wgsl"

// ============================================================================
// RAYCAST CANOPY VOLUME — the runtime.
//
// NOTHING about a plant is rasterized. The only geometry is a coarse polar
// "carrier" shell conformal to the terrain (fixed 144 x NR quads, identical for
// 20 plants and 134M). Every pixel of it asks the baked table one question and
// the answer IS the grass: hit depth, normal, coverage, height fraction.
//
// Per pixel, per stand entry, all closed form:
//   1. eye ray  ->  local canopy frame (the terrain plane is an affine change
//      of variables, so the ray stays a straight line: d_y -= grad . d_xz)
//   2. inverse wind shear (linear-in-height shear; lines stay lines)
//   3. entry-plane crossing, then hit = entry + shear * drop and
//      along-ray = drop / |d_y|  (the |OB| = |OA| / cos alpha closed form)
//   4. ONE textureSampleGrad per elevation band tap -> q, coverage, normal
//   5. hit position, world depth, AO, albedo curve — arithmetic only
// ============================================================================

const TAU: f32 = 6.2831853071;
const NS: u32 = 144u;         // carrier spokes (must match main.ts)
const TABLE_U: f32 = 192.0;   // table texels per tile side
const MAX_TAN: f32 = 26.0;    // clamp for near-horizontal rays

// Band layout is a compile-time constant of the bake (see layout.ts): zenith
// centres 0/26/48/66/81 deg with 1/12/20/30/48 azimuth bins. Keeping it in
// code (not in a uniform array) keeps the hot path free of indexed loads.
fn band_tan(b: u32) -> f32 {
  if (b == 0u) { return 0.0; }
  if (b == 1u) { return 0.4877326; }
  if (b == 2u) { return 1.1106125; }
  if (b == 3u) { return 2.2460368; }
  return 6.3137515;
}
fn band_zenith(b: u32) -> f32 {
  if (b == 0u) { return 0.0; }
  if (b == 1u) { return 0.4537856; }
  if (b == 2u) { return 0.8377580; }
  if (b == 3u) { return 1.1519173; }
  return 1.4137167;
}
fn band_azim(b: u32) -> f32 {
  if (b == 0u) { return 1.0; }
  if (b == 1u) { return 12.0; }
  if (b == 2u) { return 20.0; }
  if (b == 3u) { return 30.0; }
  return 48.0;
}
fn band_base(b: u32) -> f32 {
  if (b == 0u) { return 0.0; }
  if (b == 1u) { return 1.0; }
  if (b == 2u) { return 13.0; }
  if (b == 3u) { return 33.0; }
  return 63.0;
}

struct Entry {
  tile: vec4f,   // period, canopy height, lattice psi, active
  shade: vec4f,  // sway, aoMin, aoPow, unused
  meanq: vec4f,  // per-band mean q (parallax reference), bands 0..3
  meanq4: vec4f, // x = band 4
}

struct CanopyParams {
  entries: array<Entry, 3>,
  glob: vec4f,     // entryCount, standRadius, alphaRef, lodBias
  opts: vec4f,     // pixelAngle, nearFade, detail, tintVar
  carrier: vec4f,  // carrierA local h, carrierB local h, start margin, windScale
  flags: vec4f,    // bandBlend, shearFix, camera height over its own terrain, -
}

@group(1) @binding(0) var<uniform> params: CanopyParams;
@group(1) @binding(1) var table_samp: sampler;
@group(1) @binding(2) var table0: texture_2d_array<f32>;
@group(1) @binding(3) var table1: texture_2d_array<f32>;
@group(1) @binding(4) var table2: texture_2d_array<f32>;
@group(1) @binding(5) var<storage, read> ring_r: array<f32>;
// 3 x (32 height bins x 2 colour clusters) baked albedo curve.
struct Palette {
  c: array<vec4f, 192>,
}
@group(1) @binding(6) var<uniform> palette: Palette;

// ---------------------------------------------------------------------------
// Carrier shell: a camera-centred polar grid, conformal to terrain + a local
// height. Two instances: A above the whole canopy (catches every ray entering
// from above, plus the tips that stick into the sky when the camera is inside
// the canopy), B just under the eye (catches descending rays when the camera is
// below the canopy top — the grazing case).
// ---------------------------------------------------------------------------

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) carrier_h: f32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let cell = vi / 6u;
  let corner = vi % 6u;
  let ring = cell / NS;
  let spoke = cell % NS;
  var dr = array<u32, 6>(0u, 1u, 0u, 1u, 1u, 0u);
  var ds = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
  let ir = ring + dr[corner];
  let is = (spoke + ds[corner]) % NS;

  let r = ring_r[ir];
  let a = TAU * f32(is) / f32(NS);
  let xz = frame.camera_pos.xz + vec2f(cos(a), sin(a)) * r;
  let local_h = select(params.carrier.y, params.carrier.x, ii == 0u);
  let world = vec3f(xz.x, terrain_height(xz) + local_h, xz.y);

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.carrier_h = local_h;
  return out;
}

// ---------------------------------------------------------------------------
// The ray answer
// ---------------------------------------------------------------------------

struct RaySetup {
  o: vec3f,
  d: vec3f,
  dl_y: f32,   // local vertical slope of the ray (terrain plane removed)
  h0: f32,     // ground plane height at `base`
  g: vec2f,    // ground plane gradient
  base: vec2f, // ground plane anchor
  t_frag: f32,     // along-ray distance to the carrier fragment
  carrier_h: f32,  // fragment height above the local ground (the carrier shell)
  ddx: vec3f,  // d(ray direction)/d(pixel x), for the lookup Jacobian
  ddy: vec3f,
}

struct Hit {
  ok: bool,
  t: f32,
  pos: vec3f,
  n: vec3f,
  cov: f32,
  hf: f32,
  ei: u32,
  lod: f32,
}

struct BandTap {
  q: f32,
  cov: f32,
  n: vec3f,
  lat: vec2f,
}

fn oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - vec2f(1.0);
  var x = f.x;
  var z = f.y;
  let y = 1.0 - abs(f.x) - abs(f.y);
  if (y < 0.0) {
    x = (1.0 - abs(f.y)) * select(-1.0, 1.0, f.x >= 0.0);
    z = (1.0 - abs(f.x)) * select(-1.0, 1.0, f.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

/** Local ground plane: (height, d(height)/dx, d(height)/dz). One fetch. */
fn ground_at(xz: vec2f) -> vec3f {
  let hn = textureSampleLevel(
    terrain_heightmap, linear_sampler, xz / frame.terrain_size + vec2f(0.5), 0.0);
  let ny = sqrt(max(1.0 - hn.g * hn.g - hn.b * hn.b, 1.0e-4));
  return vec3f(hn.r, -hn.g / ny, -hn.b / ny);
}

fn rot_neg(v: vec2f, c: f32, s: f32) -> vec2f {
  return vec2f(v.x * c + v.y * s, -v.x * s + v.y * c);
}

fn rot_pos(v: vec2f, c: f32, s: f32) -> vec2f {
  return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

/** Smooth tuft-scale value noise — mixes the two baked colour clusters. */
fn tuft(pxz: vec2f, salt: u32) -> f32 {
  let s = pxz * 2.4;
  let i0 = floor(s);
  let f = s - i0;
  let sm = f * f * (3.0 - 2.0 * f);
  let cx = bitcast<u32>(i32(i0.x));
  let cy = bitcast<u32>(i32(i0.y));
  let h00 = hash_f32(hash3(cx, cy, salt));
  let h10 = hash_f32(hash3(cx + 1u, cy, salt));
  let h01 = hash_f32(hash3(cx, cy + 1u, salt));
  let h11 = hash_f32(hash3(cx + 1u, cy + 1u, salt));
  return mix(mix(h00, h10, sm.x), mix(h01, h11, sm.x), sm.y);
}

fn band_mean(meanq: vec4f, meanq4: vec4f, b: u32) -> f32 {
  if (b == 0u) { return meanq.x; }
  if (b == 1u) { return meanq.y; }
  if (b == 2u) { return meanq.z; }
  if (b == 3u) { return meanq.w; }
  return meanq4.x;
}

/** ONE texture fetch: the baked first-hit answer for band `b`. */
fn band_tap(
  tex: texture_2d_array<f32>,
  b: u32,
  s_t: vec2f,
  ebr: vec2f,
  period: f32,
  height: f32,
  qref: f32,
  grad_x: vec2f,
  grad_y: vec2f,
) -> BandTap {
  let n_az = band_azim(b);
  var jf = 0.0;
  if (n_az > 1.5) {
    jf = floor(atan2(s_t.y, s_t.x) / TAU * n_az + 0.5);
    jf = jf - floor(jf / n_az) * n_az;
  }
  let theta_b = TAU * jf / n_az;
  let dir_b = vec2f(cos(theta_b), sin(theta_b));
  let tan_b = band_tan(b);

  // The baked answer is read as a HORIZONTAL free path, not a fixed drop: for a
  // near-vertical-blade canopy the horizontal distance to the first blade is
  // (almost) independent of how fast the ray descends, while the drop is not.
  // `g_ref` is the band's mean horizontal hit distance, so the only residual
  // mismatch is the azimuth bin, and correcting THAT is a centimetre-scale
  // shift of the entry instead of the metre-scale one a drop anchor needs.
  let s_len = max(length(s_t), 1.0e-4);
  let dir_t = s_t / s_len;
  let g_ref = qref * height * tan_b;
  let delta = (dir_t - dir_b) * g_ref;
  let uv = (ebr + delta) / period;
  // Explicit (analytic) gradients: the entry point sweeps along the ray far
  // faster than across it at grazing angles, so the lookup footprint is very
  // anisotropic. Feeding the real Jacobian is what makes distant grass an
  // honestly averaged canopy instead of a shimmering one.
  let t4 = textureSampleGrad(tex, table_samp, uv, i32(band_base(b) + jf), grad_x, grad_y);

  // Re-express the baked drop for the ray's true shear. beta ramps from 0
  // (steep: the answer IS the canopy height field, drop-invariant) to ~1
  // (grazing: horizontal-path-invariant).
  var scale = 1.0;
  var cov = t4.g;
  if (tan_b > 1.0e-3) {
    let ratio = tan_b / s_len;
    let beta = clamp(s_len / (s_len + 1.2), 0.0, 1.0) * params.flags.y;
    scale = pow(ratio, beta);
    // A ray that travels further per unit drop meets more canopy: Beer-Lambert
    // on the baked transmittance keeps grazing coverage honest.
    if (ratio < 1.0) { cov = 1.0 - pow(max(1.0 - cov, 1.0e-4), clamp(1.0 / ratio, 1.0, 4.0)); }
  }

  var out: BandTap;
  out.q = clamp(t4.r * scale, 0.0, 1.0);
  out.cov = cov;
  out.n = oct_decode(t4.ba);
  // Reconstructed along the TRUE ray, so the hit sits on the eye ray.
  out.lat = ebr + delta + s_t * (height * out.q);
  return out;
}

fn resolve(
  tex: texture_2d_array<f32>,
  ei: u32,
  rs: RaySetup,
  tile: vec4f,
  shade: vec4f,
  meanq: vec4f,
  meanq4: vec4f,
) -> Hit {
  var hit: Hit;
  hit.ok = false;
  hit.t = 1.0e30;
  if (tile.w < 0.5) { return hit; }
  let period = tile.x;
  let height = tile.y;

  // --- entry plane, referenced to the CARRIER FRAGMENT ---------------------
  // The fragment lies exactly on the carrier shell, i.e. `carrier_h` above the
  // local ground, and the ground plane is sampled right there — so the entry
  // crossing is a short step from a point we know precisely. Referencing the
  // camera instead means extrapolating the plane tens of metres, which put the
  // canopy under the terrain on slopes (and produced missing bands).
  let h0 = rs.h0;
  let g = rs.g;
  let base = rs.base;
  var dl_y = rs.dl_y;

  // Entry height and the vertical compression are per-FRAME constants derived
  // from the camera's own terrain height: deriving them per pixel made
  // neighbouring pixels flip between the compressed and exact regimes (speckle).
  let ys_global = min(height, params.flags.z - params.carrier.z);
  if (ys_global < 0.04 && dl_y < 0.0) { return hit; }
  let k = clamp(ys_global / height, 0.03, 1.0);
  var y_start = height;
  if (dl_y < 0.0) {
    // Camera inside the canopy: the entry plane sits just under the eye and the
    // canopy is read as vertically compressed into it (k == 1, i.e. exact,
    // whenever the camera is above the canopy top).
    y_start = ys_global;
  } else {
    // Ascending ray: only tips above the eye can be hit. Query the reversed
    // (descending) ray from the top plane.
    if (params.flags.z >= height) { return hit; }
  }
  let t_entry = rs.t_frag + (y_start - rs.carrier_h) / dl_y;
  if (t_entry <= 0.02) { return hit; }
  if (dl_y < 0.0) {
    if (t_entry < rs.t_frag - 0.02) { return hit; }
  } else if (t_entry > rs.t_frag + 0.02) {
    return hit;
  }
  let entry_xz = rs.o.xz + rs.d.xz * t_entry;

  // --- wind: inverse shear of the query (exact for a linear shear) ---------
  let entry_y = h0 + dot(g, entry_xz - base) + y_start;
  let w = wind_sway(vec3f(entry_xz.x, entry_y, entry_xz.y), frame.time, shade.x, 0.0).xz
    * params.carrier.w;

  let s_world = -rs.d.xz / dl_y;                 // line shear (horizontal per drop)
  let s_table = k * s_world + w / height;

  let c = cos(tile.z);
  let s = sin(tile.z);
  let s_t = rot_neg(s_table, c, s);
  let ebr = rot_neg(entry_xz - w, c, s);

  // --- direction bands ----------------------------------------------------
  let alpha = atan(length(s_t));
  var bi = 0.0;
  bi += select(0.0, 1.0, alpha >= band_zenith(1u));
  bi += select(0.0, 1.0, alpha >= band_zenith(2u));
  bi += select(0.0, 1.0, alpha >= band_zenith(3u));
  bi += select(0.0, 1.0, alpha >= band_zenith(4u));
  let b0 = u32(bi);
  let b1 = select(b0, min(b0 + 1u, 4u), params.flags.x > 0.5);
  var tb = 0.0;
  if (b1 != b0) {
    let za = band_zenith(b0);
    tb = clamp((alpha - za) / max(band_zenith(b1) - za, 1.0e-4), 0.0, 1.0);
  }
  var q0 = band_mean(meanq, meanq4, b0);
  if (b1 != b0) { q0 = mix(q0, band_mean(meanq, meanq4, b1), tb); }

  // --- prefiltering: analytic Jacobian of the lookup ----------------------
  // The lookup coordinate is the ray's own position at the anchor depth (that
  // is what the parallax correction achieves), so its screen derivative is
  //   d(across the ray) = d(dir) * t     and     d(along) = d * dt,
  // and dt blows up as 1/dl_y at grazing angles. Feeding the real (very
  // anisotropic) Jacobian to textureSampleGrad is what makes distant grass an
  // honestly averaged canopy instead of a shimmering one.
  let dly_x = rs.ddx.y - dot(g, rs.ddx.xz);
  let dly_y = rs.ddy.y - dot(g, rs.ddy.xz);
  let inv_p = exp2(params.glob.w) / period;
  let gx0 = rs.ddx.xz - rs.d.xz * (dly_x / dl_y);
  let gy0 = rs.ddy.xz - rs.d.xz * (dly_y / dl_y);
  // Anchor the footprint at the expected hit (entry + the band's mean
  // horizontal free path) — a per-band constant, so no feedback, no shimmer.
  let t_anchor = t_entry + q0 * y_start * band_tan(b0) / max(length(rs.d.xz), 1.0e-3);
  var grad_x = rot_neg(gx0 * t_anchor, c, s) * inv_p;
  var grad_y = rot_neg(gy0 * t_anchor, c, s) * inv_p;
  // Clamp the anisotropy: past ~4:1 the hardware would keep full detail across
  // the ray while averaging along it, which reads as long smears. Blurring both
  // axes instead collapses very grazing views to an honest averaged canopy.
  let lx = length(grad_x);
  let ly = length(grad_y);
  let lmax = max(lx, ly);
  let lmin = max(min(lx, ly), 1.0e-9);
  if (lmax > lmin * 4.0) {
    let boost = lmax / (lmin * 4.0);
    if (lx < ly) { grad_x *= boost; } else { grad_y *= boost; }
  }
  let lod = max(log2(max(lmax * TABLE_U, 1.0)), 0.0);

  var tap = band_tap(tex, b0, s_t, ebr, period, height, q0, grad_x, grad_y);
  if (b1 != b0) {
    let tb2 = band_tap(tex, b1, s_t, ebr, period, height, q0, grad_x, grad_y);
    tap.q = mix(tap.q, tb2.q, tb);
    tap.cov = mix(tap.cov, tb2.cov, tb);
    tap.lat = mix(tap.lat, tb2.lat, tb);
    tap.n = normalize(mix(tap.n, tb2.n, tb));
  }

  let q = clamp(tap.q, 0.0, 1.0);
  if (q >= 0.9995) { return hit; }   // the ray reached the ground: no canopy

  // --- reconstruct the hit (undo lattice rotation, wind shear, squash) -----
  let y_baked = height * (1.0 - q);
  let y_local = k * y_baked;
  let hit_xz = rot_pos(tap.lat, c, s) + w * (1.0 - q);
  let hit_pos = vec3f(hit_xz.x, h0 + dot(g, hit_xz - base) + y_local, hit_xz.y);
  let t_hit = dot(hit_pos - rs.o, rs.d);
  if (t_hit <= 0.03) { return hit; }

  // --- coverage: hard alpha test, no dither -------------------------------
  let region = params.glob.y;
  let sd = max(abs(hit_pos.x), abs(hit_pos.z));
  var fade = 1.0 - smoothstep(region - 5.0, region, sd);
  if (params.flags.z < height) {
    let nf = params.opts.y;
    fade *= smoothstep(nf * 0.35, nf, t_hit);
  }
  var thresh = params.glob.z * mix(1.0, 0.5, clamp((lod - 1.0) * 0.4, 0.0, 1.0));
  if (params.opts.z > 0.001) {
    let dfade = params.opts.z / (1.0 + max(lod - 0.5, 0.0) * 2.0);
    thresh += (tuft(hit_pos.xz * 7.3, 7717u + ei) - 0.5) * dfade;
  }
  let cov = tap.cov * fade;
  if (cov < thresh) { return hit; }

  // --- normal: lattice -> local (unshear, unsquash) -> world (terrain tilt)
  let n_rot = rot_pos(vec2f(tap.n.x, tap.n.z), c, s);
  let n_l = vec3f(n_rot.x, tap.n.y, n_rot.y);
  let sh = w / height;
  var n = vec3f(n_l.x, (n_l.y - dot(sh, n_l.xz)) / k, n_l.z);
  n = vec3f(n.x - g.x * n.y, n.y, n.z - g.y * n.y);
  n = normalize(n);
  if (dot(n, rs.d) > 0.0) { n = -n; }

  hit.ok = true;
  hit.t = t_hit;
  hit.pos = hit_pos;
  hit.n = n;
  hit.cov = cov;
  hit.hf = clamp(1.0 - q, 0.0, 1.0);
  hit.ei = ei;
  hit.lod = lod;
  return hit;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VOut) -> FOut {
  let o = frame.camera_pos;
  let d = normalize(in.world - o);

  // One heightmap fetch at the carrier crossing gives a first ground plane;
  // resolve() re-samples it at the guessed hit, which is what actually matters.
  let gr = ground_at(in.world.xz);
  var rs: RaySetup;
  rs.o = o;
  rs.d = d;
  rs.h0 = gr.x;
  rs.g = gr.yz;
  rs.base = in.world.xz;
  rs.t_frag = dot(in.world - o, d);
  rs.carrier_h = in.carrier_h;
  // Camera basis from the view matrix rows -> exact per-pixel ray differentials.
  rs.ddx = vec3f(frame.view[0].x, frame.view[1].x, frame.view[2].x) * params.opts.x;
  rs.ddy = vec3f(frame.view[0].y, frame.view[1].y, frame.view[2].y) * params.opts.x;

  // Local vertical slope, clamped so near-horizontal rays stay answerable.
  var dl_y = d.y - dot(rs.g, d.xz);
  let min_slope = length(d.xz) / MAX_TAN;
  if (abs(dl_y) < min_slope) { dl_y = select(-min_slope, min_slope, dl_y >= 0.0); }
  rs.dl_y = dl_y;

  var best = resolve(table0, 0u, rs, params.entries[0].tile, params.entries[0].shade,
    params.entries[0].meanq, params.entries[0].meanq4);
  if (params.glob.x > 1.5) {
    let h1 = resolve(table1, 1u, rs, params.entries[1].tile, params.entries[1].shade,
      params.entries[1].meanq, params.entries[1].meanq4);
    if (h1.ok && h1.t < best.t) { best = h1; }
  }
  if (params.glob.x > 2.5) {
    let h2 = resolve(table2, 2u, rs, params.entries[2].tile, params.entries[2].shade,
      params.entries[2].meanq, params.entries[2].meanq4);
    if (h2.ok && h2.t < best.t) { best = h2; }
  }
  if (!best.ok) { discard; }
  // The local ground plane is an approximation; on steep terrain it can place a
  // hit in mid-air. One fetch of the real heightmap at the WINNING hit rejects
  // anything that would float (this is what kept grass out of the sky).
  if (best.pos.y > ground_at(best.pos.xz).x + params.flags.w) { discard; }

  // --- albedo from the baked height curve + tuft cluster mix ---------------
  let bin = best.hf * 31.0;
  let i0 = u32(bin);
  let i1 = min(i0 + 1u, 31u);
  let ft = bin - f32(i0);
  let pb = best.ei * 64u;
  let ca = mix(palette.c[pb + i0 * 2u].rgb, palette.c[pb + i1 * 2u].rgb, ft);
  let cb = mix(palette.c[pb + i0 * 2u + 1u].rgb, palette.c[pb + i1 * 2u + 1u].rgb, ft);
  let amp = clamp(params.opts.w / (1.0 + max(best.lod - 1.0, 0.0) * 0.8), 0.0, 1.0);
  let mixv = clamp(0.5 + (tuft(best.pos.xz, 101u + best.ei * 31u) - 0.5) * 2.0 * amp, 0.0, 1.0);
  let albedo = mix(ca, cb, mixv);

  // Baked-depth AO (hits deep under the canopy top see little sky) is part of
  // the LIGHT term, not the albedo, so the debug views stay honest.
  var ao_min = params.entries[0].shade.y;
  var ao_pow = params.entries[0].shade.z;
  if (best.ei == 1u) { ao_min = params.entries[1].shade.y; ao_pow = params.entries[1].shade.z; }
  else if (best.ei == 2u) { ao_min = params.entries[2].shade.y; ao_pow = params.entries[2].shade.z; }
  let ao = mix(ao_min, 1.0, pow(best.hf, ao_pow));

  // Thin-leaf transmission: a blade lit from behind still carries sun through
  // it. Without it every surface facing away from the sun — i.e. everything you
  // see looking UP through the canopy — resolves to near-black.
  let back = max(-dot(best.n, frame.sun_dir), 0.0);
  var color = light_surface(albedo, best.n, best.pos) * ao
    + albedo * frame.sun_color * (back * back * 0.5 * ao);
  if (debug_mode() == DEBUG_OFF) { color = apply_fog(color, best.pos); }

  let clip = frame.view_proj * vec4f(best.pos, 1.0);
  var out: FOut;
  out.color = vec4f(debug_shade(color, albedo, best.n, best.cov, best.pos), 1.0);
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return out;
}
