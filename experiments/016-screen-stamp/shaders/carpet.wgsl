#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/debug.wgsl"

// DEFERRED CARPET STAMP — the mat layer, resolved straight from the depth
// buffer. Runs BEFORE the near-card pass, when every non-sky depth texel is
// still terrain, so no list, no binning and no ordering question arises.
//
// Why not stamps: a carpet entry has carpet_div^2 = 484 tiles per 4m cell
// (life size), so one 16x8 screen tile overlaps hundreds of them at any real
// distance and a 64-entry list is hopeless — the mat would be confetti plus
// tint. But a mat is exactly the case where the tile list is unnecessary: the
// map from a ground point to its carpet tile is a pure function of xz, and the
// terrain base pass already handed us the visible ground point per pixel. So
// per pixel: unproject -> grid node -> wetness picks the owning state ->
// 90-degree yaw -> sample the tile's own top-view bake. O(1), no overflow, no
// slot exhaustion, exact at any plant count, and terrain-conforming by
// construction (ladder rung 3+: the sample point IS the visible ground point,
// so the mat cannot crack, float or bury an edge).
//
// Thickness comes from the baked height field, in two places: the lookup slides
// along the view ray by the layer's own height (parallax), and the surface normal
// is taken from the height GRADIENT at capitulum scale plus a cavity term for the
// gaps — which is what a flat ground-parallel quad (001) cannot express, however
// good its texture is.

struct StampParams {
  dims: vec4u,                  // x = card slots, y = carpet slots, z = seed, w unused
  tuning: vec4f,                // x = max_dist, y = tint_strength, z = tile overlay, w = tint coverage
  cards: vec4u,                 // stand entry index per card slot
  carpets: vec4u,               // stand entry index per carpet slot
  grid: vec4f,                  // x = carpet_div, y = tile step (m), z = 1/texel (m), w = lod max
  entry_meta: array<vec4f, 4>,  // per card slot: impostor local center.xyz, radius
  entry_info: array<vec4f, 4>,  // per card slot: species average albedo rgb
  carpet_zone: array<vec4f, 4>, // per carpet slot: wet_lo, wet_hi, h_mean (m), y_range (m)
}
@group(1) @binding(0) var<uniform> sp: StampParams;
@group(1) @binding(1) var scene_depth: texture_depth_2d;
@group(1) @binding(2) var alb0: texture_2d<f32>;
@group(1) @binding(3) var alb1: texture_2d<f32>;
@group(1) @binding(4) var alb2: texture_2d<f32>;
@group(1) @binding(5) var nh0: texture_2d<f32>;
@group(1) @binding(6) var nh1: texture_2d<f32>;
@group(1) @binding(7) var nh2: texture_2d<f32>;
@group(1) @binding(8) var carpet_samp: sampler;

// Atlas albedo/normal are COVERAGE-WEIGHTED at every mip level (see
// downsampleWeighted in bake.ts), i.e. already normalised — dividing by alpha
// here would inflate distant mips (the 017 trap). Do not add a divide.
fn sample_nh(slot: u32, uv: vec2f, lod: f32) -> vec4f {
  if (slot == 0u) { return textureSampleLevel(nh0, carpet_samp, uv, lod); }
  if (slot == 1u) { return textureSampleLevel(nh1, carpet_samp, uv, lod); }
  return textureSampleLevel(nh2, carpet_samp, uv, lod);
}

// Anisotropic variants: the ground's pixel footprint is an ellipse, and the two
// axes are known analytically (no screen derivatives — uv jumps at every tile
// border, so implicit derivatives would blow up along the whole lattice).
fn sample_alb_grad(slot: u32, uv: vec2f, ddx: vec2f, ddy: vec2f) -> vec4f {
  if (slot == 0u) { return textureSampleGrad(alb0, carpet_samp, uv, ddx, ddy); }
  if (slot == 1u) { return textureSampleGrad(alb1, carpet_samp, uv, ddx, ddy); }
  return textureSampleGrad(alb2, carpet_samp, uv, ddx, ddy);
}

fn sample_nh_grad(slot: u32, uv: vec2f, ddx: vec2f, ddy: vec2f) -> vec4f {
  if (slot == 0u) { return textureSampleGrad(nh0, carpet_samp, uv, ddx, ddy); }
  if (slot == 1u) { return textureSampleGrad(nh1, carpet_samp, uv, ddx, ddy); }
  return textureSampleGrad(nh2, carpet_samp, uv, ddx, ddy);
}

struct CarpetNode {
  ok: bool,
  slot: u32,
  center: vec2f,
  cos_yaw: f32,
  sin_yaw: f32,
}

/**
 * Which carpet tile owns this ground point, and how it is turned. A bit-exact
 * mirror of the carpet branch of scatter_candidate(): same grid (cell*4 +
 * (g+0.5)*step), same node-only jitter hash, same half-open wetness interval,
 * same `h & 3` quarter turn. Inverted from "slot index -> position" to
 * "position -> slot index", which is what a deferred pass needs.
 */
fn carpet_node(xz: vec2f) -> CarpetNode {
  var out: CarpetNode;
  out.ok = false;
  let seed = sp.dims.z;
  let n = u32(sp.grid.x);
  let step = sp.grid.y;
  let ni = vec2i(floor(xz / step));
  let cell = vec2i(floor(vec2f(ni) / sp.grid.x));
  let g = vec2u(ni - cell * i32(n));
  let i = g.y * n + g.x;
  let center = (vec2f(ni) + 0.5) * step;
  out.center = center;

  // Node-only jitter (never hashed with the entry index — that is what used to
  // make the three states disagree about the same node).
  let node_h = hash4(seed, bitcast<u32>(cell.x), bitcast<u32>(cell.y), i ^ CARPET_JITTER_SALT);
  let jitter = (hash_f32(node_h) - 0.5) * CARPET_JITTER;
  let w = clamp(scatter_wetness(seed, center) + jitter, 0.0, 0.9999);

  let cnt = sp.dims.y;
  for (var s = 0u; s < cnt; s++) {
    let z = sp.carpet_zone[s];
    if (w >= z.x && w < z.y) {
      out.ok = true;
      out.slot = s;
      let entry_index = sp.carpets[s];
      let h = hash4(seed, bitcast<u32>(cell.x), bitcast<u32>(cell.y), (entry_index << 16u) ^ i);
      // 90-degree steps only: the one rotation set a square periodic tile keeps
      // its neighbours' agreement under.
      let k = f32(h & 3u);
      out.cos_yaw = cos(k * QUARTER_TURN);
      out.sin_yaw = sin(k * QUARTER_TURN);
      break;
    }
  }
  return out;
}

/// Tile-local uv of a ground point, un-rotating the tile's quarter turn.
fn carpet_uv(nd: CarpetNode, xz: vec2f) -> vec2f {
  let rel = xz - nd.center;
  // rot_y_v(v, -yaw).xz for the yaw the tile was placed with.
  let lx = nd.cos_yaw * rel.x - nd.sin_yaw * rel.y;
  let lz = nd.sin_yaw * rel.x + nd.cos_yaw * rel.y;
  return vec2f(lx, lz) / sp.grid.y + 0.5;
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  if (sp.dims.y == 0u) { discard; } // no carpet species in this stand
  let px = vec2u(in.pos.xy);
  let d = textureLoad(scene_depth, vec2i(px), 0);
  if (d >= 0.99999) { discard; } // sky: no ground, no mat

  let cam = frame.camera_pos;
  let ndc = vec2f(
    (f32(px.x) + 0.5) / frame.viewport.x * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / frame.viewport.y * 2.0,
  );
  let wp = frame.inv_view_proj * vec4f(ndc, d, 1.0);
  let ground = wp.xyz / wp.w;
  let ray = normalize(ground - cam);
  let dist = length(ground - cam);

  var nd = carpet_node(ground.xz);
  if (!nd.ok) { discard; } // no state claims this node (partition hole)
  var uv = carpet_uv(nd, ground.xz);

  // --- footprint. A ground plane seen at an angle has an ANISOTROPIC pixel
  // footprint: `minor` across the view direction, minor/cos(incidence) along it.
  // Choosing the mip from the long axis (the isotropic-safe choice) is what turns
  // a grazing mat into flat paint — at 5m and 10 degrees that is a 4cm footprint,
  // the flattest mip in the chain, for ground you are standing next to. Both axes
  // go to textureSampleGrad; see the note at `maj` below for what the driver
  // actually does with them.
  let pixel_scale = 2.0 / (frame.viewport.y * frame.proj[1][1]);
  let g0 = terrain_sample(ground.xz);
  let up = vec3f(g0.y, sqrt(max(1.0 - g0.y * g0.y - g0.z * g0.z, 0.0)), g0.z);
  let cos_i = max(abs(dot(ray, up)), 0.12);
  let foot_minor = dist * pixel_scale;
  let lod_min = clamp(log2(max(foot_minor * sp.grid.z, 1.0)), 0.0, sp.grid.w);
  // Relief baseline: a capitulum (1.2cm), and never finer than ~2.5 pixels —
  // a normal field whose features are sub-pixel is aliasing, not detail, and it
  // is what makes a distant mat read as noise. As the baseline grows the slopes
  // it reports get gentler (measured on the bake: mean |dh| 0.61cm over 1.2cm
  // vs 0.88cm over 4.5cm), so the mat naturally goes from capitula up close to
  // gentle hummocks with distance.
  let meso_m = max(0.012, 2.5 * foot_minor);
  let lod_meso = log2(meso_m * sp.grid.z);
  // Two levels off the capitulum scale: the relief GRADIENT wants sharp taps
  // over a 1.2cm baseline, the cavity term wants a genuinely coarser local mean
  // to compare the fine height against. Taking both at one level collapses the
  // cavity to zero wherever the pixel footprint already sits near the capitulum
  // scale — which is most of a grazing view.
  let lod_c = clamp(lod_meso - 2.0, lod_min, sp.grid.w);
  let lod_mean = clamp(lod_meso, lod_min, sp.grid.w);

  // --- one parallax step: the mat's surface is ~6.7cm above the terrain the
  // depth buffer gave us, so the lookup slides toward the camera by
  // h/(-ray.y) — exact for a layer of constant height, and the layer's mean
  // height is baked (zone.z), so it costs no tap. Deliberately NOT the
  // per-texel height: that ripples the warp at the relief's own scale and combs
  // the mat into radial streaks (visible at 0.3m, measured with and without).
  let z0 = sp.carpet_zone[nd.slot];
  var t = z0.z / max(-ray.y, 0.05);
  let horiz = length(ray.xz) * t;
  let max_off = 0.45 * sp.grid.y;
  if (horiz > max_off) { t *= max_off / max(horiz, 1e-6); }
  let xz1 = ground.xz - ray.xz * t;
  let nd1 = carpet_node(xz1);
  if (nd1.ok) {
    nd = nd1;
    uv = carpet_uv(nd1, xz1);
  }

  // Footprint axes in the tile's own (quarter-turned) uv frame.
  let fwd = select(vec2f(1.0, 0.0), normalize(ray.xz), length(ray.xz) > 1e-5);
  let side = vec2f(-fwd.y, fwd.x);
  // Measured: Dawn/Metal does not apply maxAnisotropy to an explicit-gradient
  // sample, so the level comes from the LONGER axis and a grazing mat lands on
  // the flattest mip (verified by forcing both axes to the minor one: the same
  // ground went from mud to crisp). Shortening the major axis to the geometric
  // mean lands halfway — sqrt(major*minor) is the standard compromise between
  // over-blur and shimmer — and still degrades gracefully to true anisotropy on
  // a driver that honours it.
  let maj = fwd * (foot_minor / sqrt(cos_i));
  let rot_maj = vec2f(nd.cos_yaw * maj.x - nd.sin_yaw * maj.y, nd.sin_yaw * maj.x + nd.cos_yaw * maj.y);
  let mn = side * foot_minor;
  let rot_mn = vec2f(nd.cos_yaw * mn.x - nd.sin_yaw * mn.y, nd.sin_yaw * mn.x + nd.cos_yaw * mn.y);
  let ddx_uv = rot_maj / sp.grid.y;
  let ddy_uv = rot_mn / sp.grid.y;

  let zone = sp.carpet_zone[nd.slot];
  let alb4 = sample_alb_grad(nd.slot, uv, ddx_uv, ddy_uv);
  let nh = sample_nh_grad(nd.slot, uv, ddx_uv, ddy_uv);
  // A mat is a closed surface: its coverage must not thin out with distance
  // (mips pull alpha toward the tile mean). Solid above ~1/4 coverage, and the
  // genuinely empty texels — the gaps down to the peat — still open. Hard
  // ramp, no dither.
  let alpha = clamp((alb4.a - 0.02) * 4.0, 0.0, 1.0);
  if (alpha < 0.004) { discard; }

  let albedo = alb4.rgb;
  var n_local = vec3f(nh.x * 2.0 - 1.0, 0.0, nh.y * 2.0 - 1.0);
  // Normals were flipped toward the (straight-down) capture camera, so the
  // missing component is the positive root — mip-averageable by construction.
  n_local.y = sqrt(max(1.0 - n_local.x * n_local.x - n_local.z * n_local.z, 0.01));
  // Work in slopes (dh/dx, dh/dz), which add.
  var slope = vec2f(-n_local.x, -n_local.z) / max(n_local.y, 0.25);

  // MESO RELIEF — the cushion shape, from the gradient of the baked HEIGHT over
  // the capitulum baseline above. This is the whole ballgame for a moss mat:
  // averaging leaf normals over a texel footprint collapses them toward straight
  // up (mip 3 of this atlas is already an almost flat green sheet), so a mipped
  // normal map alone makes the mat read as printed texture, while the mean height
  // keeps its shape at every level. The two terms are complementary, not
  // double-counted: the baked normal is the leaf's own tilt (and self-fades to
  // flat as the chain deepens), the height gradient is the cushion the leaves
  // sit on.
  // Step INWARD near the tile border. The atlas wraps (it is one period of a
  // periodic field), but the neighbouring tile is turned by its own quarter
  // turn, so a tap that crosses the edge reports a height step that is not
  // there — and that draws the lattice as a grid of shading lines. Both taps
  // inside the tile keeps the relief honest; the real content change at the
  // seam then reads as moss meeting moss, which is what the overlapping source
  // geometry does anyway.
  var off_uv = vec2f(meso_m / sp.grid.y);
  if (uv.x + off_uv.x > 1.0) { off_uv.x = -off_uv.x; }
  if (uv.y + off_uv.y > 1.0) { off_uv.y = -off_uv.y; }
  let hc = sample_nh(nd.slot, uv, lod_c).z;
  let hx = sample_nh(nd.slot, uv + vec2f(off_uv.x, 0.0), lod_c).z;
  let hz = sample_nh(nd.slot, uv + vec2f(0.0, off_uv.y), lod_c).z;
  let d_h = vec2f(hx - hc, hz - hc) * zone.w / meso_m;
  slope += clamp(d_h * sign(off_uv), vec2f(-3.0), vec2f(3.0));
  n_local = normalize(vec3f(-slope.x, 1.0, -slope.y));

  // Ground frame at the FINAL sample point: per-pixel conforming, so the mat
  // lights as the slope it lies on and neighbouring tiles cannot disagree.
  let g1 = terrain_sample(xz1);
  let up1 = vec3f(g1.y, sqrt(max(1.0 - g1.y * g1.y - g1.z * g1.z, 0.0)), g1.z);
  let tang = normalize(vec3f(nd.cos_yaw, 0.0, -nd.sin_yaw) - up1 * dot(up1, vec3f(nd.cos_yaw, 0.0, -nd.sin_yaw)));
  let bitan = cross(tang, up1);
  let n_ws = normalize(tang * n_local.x + up1 * n_local.y + bitan * n_local.z);

  let h1 = nh.z * zone.w;
  let world = vec3f(xz1.x, g1.x + h1, xz1.y);
  // Cavity term: how far this texel sits below the local canopy top (the
  // capitulum-scale mean is already sampled for the relief). The gaps between
  // cushions see almost no sky and a normal alone cannot express that — this is
  // what stops a lit heightfield from looking like embossed paint. Free: both
  // heights are already in hand.
  // Anchored at the local canopy mean (never brightens): a texel level with the
  // cushion tops keeps full light, one ~1.5cm below it loses most of its sky.
  let h_mean = sample_nh(nd.slot, uv, lod_mean).z;
  let cavity = clamp(1.0 + (nh.z - h_mean) * zone.w * 42.0, 0.38, 1.0);
  var col = light_surface(albedo, n_ws, world) * cavity;
  if (debug_mode() == DEBUG_OFF) {
    col = apply_fog(col, world);
  }
  let shaded = debug_shade(col, albedo, n_ws, alpha, world);
  if (debug_mode() == DEBUG_COVERAGE) {
    // Coverage is written opaque (the terrain wrote 1.0 everywhere, so a
    // blended coverage map washes out); the stamp pass leaves carpet pixels
    // alone rather than overwriting them with its own ~0.
    return vec4f(shaded, 1.0);
  }
  return vec4f(shaded * alpha, alpha);
}
