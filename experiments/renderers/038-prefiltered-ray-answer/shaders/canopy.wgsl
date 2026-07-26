#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/hash.wgsl"

// RUNTIME — the whole groundcover, resolved as a per-pixel ray query against a
// baked, mip-prefiltered answer table. NOT ONE PLANT PRIMITIVE IS RASTERIZED:
// the only geometry is a coarse camera-centred grid conforming to the terrain
// at the canopy-top height, whose tessellation is a fixed constant and has
// nothing to do with how many plants the stand contains.
//
// Per pixel, all closed form, no marching, no loops over the ray:
//   1. one heightmap fetch  -> local ground height + gradient at the crossing
//   2. world -> patch space -> the ground is flat, wind unsheared, collapse undone
//   3. one division          -> E, where the eye LINE crosses y = H
//   4. one geom fetch        -> probe hit height (parallax anchor)
//   5. one geom + one surf   -> the answer at the corrected entry point
//   6. one heightmap fetch   -> snap the hit onto the true terrain, and the
//                               snap residual is the confidence (it is what
//                               detects a ray that left the canopy over a dip)
// = 5 texture fetches, constant, for every pixel at every distance and every
// plant count. Hardware trilinear + anisotropic filtering of the table IS the
// distance prefilter: mip n is an honestly averaged canopy over a 2^n footprint
// of entry points.

const N_AZ: u32 = 24u;
const N_EL: u32 = 6u;
const AZ_PER_BAND: u32 = 8u;
const RES_F: f32 = 192.0;
// Elevation bins are uniform in ln(cot theta): fine where the shear changes
// fast (grazing), coarse where it barely changes (steep).
const LC0: f32 = 2.8;
const LC_STEP: f32 = 1.02;
const TWO_PI: f32 = 6.2831853;
/// cot(theta) of the flattest baked elevation bin.
const COT_MAX: f32 = 16.4446;

struct Cfg {
  canopy_h: f32,
  tile_l: f32,
  /// Height of the carrier surface above the terrain. Normally the canopy top;
  /// dropped below the eye when the eye is inside the canopy, which only moves
  /// the surface the rasterizer intersects — never the canopy itself.
  carrier_h: f32,
  region_half: f32,

  warp_amp: f32,
  ao_strength: f32,
  sharpen: f32,
  lod_bias: f32,

  carrier_cell: f32,
  carrier_side: f32,
  snap_tol: f32,
  correct: f32,

  mean_sway: f32,
  near_fade: f32,
  az_blend: f32,
  detail: f32,
}
@group(1) @binding(0) var<uniform> cfg: Cfg;
@group(1) @binding(1) var tab_samp: sampler;
// Three (surf, geom) pairs, one per species budget row: the table is shared by
// the whole canopy, so it is split by azimuth band into three equal slabs.
@group(1) @binding(2) var surf0: texture_2d_array<f32>;
@group(1) @binding(3) var geom0: texture_2d_array<f32>;
@group(1) @binding(4) var surf1: texture_2d_array<f32>;
@group(1) @binding(5) var geom1: texture_2d_array<f32>;
@group(1) @binding(6) var surf2: texture_2d_array<f32>;
@group(1) @binding(7) var geom2: texture_2d_array<f32>;

fn tab_geom(band: u32, layer: u32, uv: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  if (band == 0u) {
    return textureSampleGrad(geom0, tab_samp, uv, layer, gx, gy);
  }
  if (band == 1u) {
    return textureSampleGrad(geom1, tab_samp, uv, layer, gx, gy);
  }
  return textureSampleGrad(geom2, tab_samp, uv, layer, gx, gy);
}

fn tab_surf(band: u32, layer: u32, uv: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  if (band == 0u) {
    return textureSampleGrad(surf0, tab_samp, uv, layer, gx, gy);
  }
  if (band == 1u) {
    return textureSampleGrad(surf1, tab_samp, uv, layer, gx, gy);
  }
  return textureSampleGrad(surf2, tab_samp, uv, layer, gx, gy);
}

fn oct_decode(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

/// Hardware-bilinear terrain sample: (height, normal.x, normal.z). Same texels
/// and same bilinear weights as the shared terrain_sample(), one fetch instead
/// of four loads — this is a lookup for the local ground PLANE, never placement.
fn terr(xz: vec2f) -> vec3f {
  let uv = xz / frame.terrain_size + 0.5;
  return textureSampleLevel(terrain_heightmap, linear_sampler, uv, 0.0).xyz;
}

fn vhash(i: vec2i) -> f32 {
  return hash_f32(hash3(0x51ed37u, bitcast<u32>(i.x), bitcast<u32>(i.y)));
}

fn vnoise(p: vec2f) -> f32 {
  let base = floor(p);
  let f = p - base;
  let s = f * f * (3.0 - 2.0 * f);
  let b = vec2i(base);
  let x0 = mix(vhash(b), vhash(b + vec2i(1, 0)), s.x);
  let x1 = mix(vhash(b + vec2i(0, 1)), vhash(b + vec2i(1, 1)), s.x);
  return mix(x0, x1, s.y);
}

/// Smooth lookup translation. A constant offset per pixel is exactly a
/// translated copy of the periodic patch, so it is geometrically free: it
/// bends the 6 m answer lattice out of alignment with itself without any seam
/// and without touching the reconstructed hit.
fn warp2(xz: vec2f) -> vec2f {
  let p = xz * 0.029;
  return vec2f(vnoise(p) - 0.5, vnoise(p + vec2f(19.3, 7.7)) - 0.5) * 2.0 * cfg.warp_amp;
}

fn wrap_az(i: i32) -> u32 {
  return u32(((i % i32(N_AZ)) + i32(N_AZ)) % i32(N_AZ));
}

/// Unit direction of a baked bin, in patch space.
fn bin_dir(az: u32, el: u32) -> vec3f {
  let th = atan2(1.0, exp(LC0 - LC_STEP * f32(el)));
  let ph = TWO_PI * f32(az) / f32(N_AZ);
  return vec3f(cos(ph) * cos(th), -sin(th), sin(ph) * cos(th));
}

struct Ans {
  surf: vec4f,
  geom: vec4f,
}

/// One bin's answer, re-anchored so its hit sits on the true eye ray (the
/// closed-form parallax correction that makes direction quantisation cheap).
fn fetch_bin(az: u32, el: u32, xz_a: vec2f, y_probe: f32, e0: vec2f, woff: vec2f, gx: vec2f, gy: vec2f) -> Ans {
  let dq = bin_dir(az, el);
  let e_corr = xz_a - dq.xz * ((y_probe - cfg.canopy_h) / dq.y);
  let e1 = select(e0, e_corr, cfg.correct > 0.5);
  let uv = (e1 + woff) / cfg.tile_l;
  let band = az / AZ_PER_BAND;
  let layer = (az % AZ_PER_BAND) * N_EL + el;
  var a: Ans;
  a.geom = tab_geom(band, layer, uv, gx, gy);
  a.surf = tab_surf(band, layer, uv, gx, gy);
  return a;
}

/// Blade-scale detail for the regime where the table is MAGNIFIED (a texel wider
/// than a pixel). It invents no coverage — only a normal tilt and a small albedo
/// shift, anchored to the hit's world position so it never swims — and its
/// amplitude goes to zero the moment the footprint reaches one texel, i.e. as
/// soon as the baked answer resolves what the pixel is asking for.
fn detail_noise(hit: vec3f) -> vec3f {
  // Cells are wide-in-y / narrow-in-xz, so the detail reads as upright blades.
  let q = vec3f(hit.x * 42.0, hit.y * 7.0, hit.z * 42.0);
  let i = floor(q);
  let f = q - i;
  let s = f * f * (3.0 - 2.0 * f);
  let b = vec3i(i);
  let h000 = hash4(0x2f19u, bitcast<u32>(b.x), bitcast<u32>(b.y), bitcast<u32>(b.z));
  let h100 = hash4(0x2f19u, bitcast<u32>(b.x + 1), bitcast<u32>(b.y), bitcast<u32>(b.z));
  let h001 = hash4(0x2f19u, bitcast<u32>(b.x), bitcast<u32>(b.y), bitcast<u32>(b.z + 1));
  let h101 = hash4(0x2f19u, bitcast<u32>(b.x + 1), bitcast<u32>(b.y), bitcast<u32>(b.z + 1));
  let x0 = mix(hash_f32(h000), hash_f32(h100), s.x);
  let x1 = mix(hash_f32(h001), hash_f32(h101), s.x);
  let a = mix(x0, x1, s.z);
  let x0b = mix(hash_f32(hash2(h000, 7u)), hash_f32(hash2(h100, 7u)), s.x);
  let x1b = mix(hash_f32(hash2(h001, 7u)), hash_f32(hash2(h101, 7u)), s.x);
  let bb = mix(x0b, x1b, s.z);
  return vec3f(a * 2.0 - 1.0, bb * 2.0 - 1.0, mix(a, bb, s.y) * 2.0 - 1.0);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0),
    vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0),
  );
  let side = u32(cfg.carrier_side);
  let cell = vi / 6u;
  let g = vec2f(f32(cell % side), f32(cell / side)) + corners[vi % 6u];
  let c = cfg.carrier_cell;
  // Snapped to its own cell size, so the facets are stationary in the world and
  // the answer never crawls as the camera moves.
  let org = floor(frame.camera_pos.xz / c) * c - vec2f(f32(side) * 0.5 * c);
  let xz = org + g * c;
  let y = terr(xz).x + cfg.carrier_h;
  var out: VOut;
  out.world = vec3f(xz.x, y, xz.y);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  return out;
}

struct FOut {
  @builtin(frag_depth) depth: f32,
  @location(0) color: vec4f,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let l = cfg.tile_l;
  let h = cfg.canopy_h;
  let ro = frame.camera_pos;
  // The carrier fragment lies on the canopy-top surface AND on this pixel's eye
  // ray, so it is both the ray/canopy entry point and the anchor of the local
  // patch frame. Everything below is expressed FROM it — never extrapolated
  // back to the camera, which is what a ground plane 100 m away cannot support.
  let p = in.world;
  let rd = normalize(p - ro);
  let dist_p = length(p - ro);

  // --- 1. local ground plane at the crossing (fetch 1) ----------------------
  // The GROUND HEIGHT comes from the carrier itself, not from this fetch: the
  // carrier is coarse, and at grazing incidence any mismatch between the two
  // gets divided by |ray . plane normal| (~1/16 here), which turns a 10 cm
  // tessellation error into metres of entry-point jitter — the faceted swirl
  // this used to show. Anchoring the plane on the carrier makes the entry point
  // exactly the rasterized crossing, and the terrain snap in step 6 puts the
  // hit back on the true ground anyway. Only the gradient is read here.
  let ts = terr(p.xz);
  let base_y = p.y - cfg.carrier_h;
  let ny = sqrt(max(1.0 - ts.y * ts.y - ts.z * ts.z, 1.0e-4));
  let grad = clamp(vec2f(-ts.y, -ts.z) / max(ny, 0.35), vec2f(-0.7), vec2f(0.7));

  // --- 2. wind as an exact shear of space ----------------------------------
  // A horizontal displacement proportional to height is linear, so unshearing
  // the RAY is exact (lines stay lines). The phase comes from a smooth field,
  // which reads as gusts travelling across the meadow.
  let phase = TWO_PI * vnoise(p.xz * 0.045);
  let sway = wind_sway(vec3f(p.x, base_y, p.z), frame.time, cfg.mean_sway, phase).xz;
  let wsh = sway / h;

  // --- 3. world -> patch space, then the closed-form entry point ------------
  // The carrier is coarse, so its interpolated height is NOT exactly the local
  // canopy top; using it directly would warp the entry point per facet (a
  // visible faceted swirl). Slide along the ray onto the exact tangent-plane
  // canopy top instead — one division, and the entry is then exactly y_p = H.
  let d_hgt_raw = rd.y - dot(grad, rd.xz);
  // A ray flatter than the flattest baked bin makes the entry point run off to
  // infinity (the plane and the ray are parallel). Clamp the descent rate to
  // that bin's cotangent: the answer is the one that bin already holds, and the
  // terrain snap below puts the hit back at the right distance.
  let d_hgt = min(d_hgt_raw, -length(rd.xz) / COT_MAX - 1.0e-5);
  let d_py = d_hgt;
  // Slide from the carrier crossing onto the canopy-top plane. When the carrier
  // sits below the canopy top (eye inside the canopy) this slide goes BACKWARDS
  // past the eye — which is fine: E only indexes the table, it is not a sample.
  let t_fix = clamp((h - (p.y - base_y)) / d_hgt, -24.0, 24.0);
  let q = p + rd * t_fix;
  let d_pxz = rd.xz - wsh * d_py;
  let e0 = q.xz - wsh * h;

  // Screen footprint of the entry point == the prefilter footprint. Computed
  // before any branch so the derivatives stay uniform.
  let raw_x = dpdx(e0) / l;
  let raw_y = dpdy(e0) / l;
  // At grazing the footprint is a hugely elongated ellipse along the view
  // azimuth. Averaging over its long axis is nearly free of information (two
  // entry points offset along the azimuth are two parallel rays offset only
  // sin(theta) as much, so the table is smooth there), while letting it drive
  // the mip level destroys the short axis. Cap the anisotropy to what the
  // hardware filter can actually take, then the LOD follows the short axis.
  let lx = max(length(raw_x), 1.0e-9);
  let ly = max(length(raw_y), 1.0e-9);
  let cap = min(lx, ly) * 16.0;
  let bias = exp2(cfg.lod_bias);
  let gx = raw_x * min(1.0, cap / lx) * bias;
  let gy = raw_y * min(1.0, cap / ly) * bias;
  // The SHORT axis is what says whether a silhouette edge is resolvable.
  let foot = min(lx, ly) * RES_F;

  // --- 4. direction quantisation ------------------------------------------
  let dpn = normalize(vec3f(d_pxz.x, d_py, d_pxz.y));
  let cot = length(dpn.xz) / max(-dpn.y, 1.0e-4);
  let el_f = (LC0 - log(max(cot, 1.0e-5))) / LC_STEP;
  let el = u32(clamp(floor(el_f + 0.5), 0.0, f32(N_EL - 1u)));
  let az_f = atan2(dpn.z, dpn.x) / TWO_PI * f32(N_AZ);
  let az_lo = wrap_az(i32(floor(az_f)));
  let az_hi = wrap_az(i32(floor(az_f)) + 1);
  let aw = az_f - floor(az_f);
  let az_near = select(az_hi, az_lo, aw < 0.5);

  let woff = warp2(p.xz);

  // --- 5. probe + one closed-form parallax correction ----------------------
  // The probe is shared; the correction is pure ALU per azimuth bin, so blending
  // the two neighbouring bins costs fetches only, and both of them land their
  // hit on the SAME true ray — which is what keeps the blend from ghosting.
  let g0 = tab_geom(az_near / AZ_PER_BAND, (az_near % AZ_PER_BAND) * N_EL + el, (e0 + woff) / l, gx, gy);
  let y_probe = (g0.r / max(g0.a, 1.0e-4)) * h;
  let t_a = (y_probe - h) / d_py;
  let xz_a = e0 + d_pxz * t_a;

  var g1: vec4f;
  var s1: vec4f;
  if (cfg.az_blend > 0.5) {
    let a = fetch_bin(az_lo, el, xz_a, y_probe, e0, woff, gx, gy);
    let b = fetch_bin(az_hi, el, xz_a, y_probe, e0, woff, gx, gy);
    // Premultiplied channels, so a straight lerp IS the correct blend.
    g1 = mix(a.geom, b.geom, aw);
    s1 = mix(a.surf, b.surf, aw);
  } else {
    let a = fetch_bin(az_near, el, xz_a, y_probe, e0, woff, gx, gy);
    g1 = a.geom;
    s1 = a.surf;
  }
  let cov = g1.a;
  let inv = 1.0 / max(cov, 1.0e-4);
  let y_p = (g1.r * inv) * h;
  let albedo = s1.rgb * inv;
  let ao_raw = s1.a * inv;
  var n = oct_decode((g1.gb * inv) * 2.0 - 1.0);

  // --- 6. reconstruct, then snap onto the true terrain (fetch 5) ------------
  // t is measured from the entry point q, so no long extrapolation exists.
  let t0 = max((y_p - h) / d_py, 0.0);
  let hit0 = q + rd * t0;
  let th = terr(hit0.xz).x;
  let sgn = select(1.0, -1.0, rd.y < 0.0);
  let rdy = sgn * max(abs(rd.y), 1.0e-3);
  let dt = (th + y_p - hit0.y) / rdy;
  // Hits resolved behind the eye (the eye is inside the canopy's top sliver, so
  // the first hit from OUTSIDE is already behind it) are pinned to the near fade
  // distance instead of dropped: the answer is still a real canopy sample, only
  // its distance is unknowable from a first-hit table. See NOTES.
  // The snap stays LOCAL to its own carrier crossing. A grazing ray crosses the
  // canopy-top surface several times over rolling terrain; letting one crossing
  // snap its hit tens of metres forward let a far crossing win the depth test
  // over a near one, which showed up as a patchwork. Bounded snap + the
  // confidence fade below means each crossing answers its own neighbourhood and
  // the nearest one always wins.
  let t_raw = t0 + clamp(dt, -t0 - 0.5, 2.5 * t0 + 1.5);
  let t_min = cfg.near_fade - dist_p - t_fix;
  let t1 = max(t_raw, t_min);
  let hit = q + rd * t1;
  let cam_dist = dist_p + t_fix + t1;
  // The snap residual is the honest slab test. dt > 0 means "the true canopy is
  // lower here than the tangent plane predicted, so the ray must travel further"
  // — a few t0 of that is ordinary terrain curvature and the snapped hit is
  // right; a huge value means the ray actually flew over a dip and left the
  // canopy, and the pixel must show whatever is really behind it (sky, or the
  // next carrier crossing further along the same ray).
  let resid = max(dt, 0.0) / max(t0, 1.0);
  let conf = 1.0 - smoothstep(cfg.snap_tol, cfg.snap_tol * 2.5, resid);

  // --- coverage: hard alpha-test edge near, honest averaged coverage far ----
  let width = clamp(foot, cfg.sharpen, 1.0);
  var alpha = clamp((cov - 0.5) / width + 0.5, 0.0, 1.0) * conf;
  let rmax = max(abs(hit.x), abs(hit.z));
  alpha *= 1.0 - smoothstep(cfg.region_half - 3.0, cfg.region_half, rmax);
  alpha *= smoothstep(cfg.near_fade * 0.3, cfg.near_fade, cam_dist);

  // --- shade ---------------------------------------------------------------
  if (dot(n, rd) > 0.0) {
    n = -n;
  }
  var alb = albedo;
  let mag = cfg.detail * clamp(1.0 - foot, 0.0, 1.0);
  if (mag > 0.01) {
    let d3 = detail_noise(hit);
    n = normalize(n + d3 * 0.55 * mag);
    alb *= 1.0 + d3.z * 0.16 * mag;
  }
  let ao = mix(1.0 - cfg.ao_strength, 1.0, ao_raw);
  var color = light_surface(alb, n, hit) * ao;
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, hit);
  }
  let clip = frame.view_proj * vec4f(hit, 1.0);

  // Everything above is unconditional; all rejection happens here.
  // d_py_raw < 0 IS the "entering from above" test: the ray must descend
  // relative to the LOCAL canopy-top plane, which also culls the carrier's
  // underside (rays leaving the canopy) for free.
  if (d_hgt_raw > -1.0e-4 || cam_dist <= 0.05 || clip.w <= 0.0 || alpha < 0.01) {
    discard;
  }

  var out: FOut;
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  out.color = vec4f(debug_shade(color, alb, n, alpha, hit) * alpha, alpha);
  return out;
}
