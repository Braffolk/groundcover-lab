#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/hash.wgsl"

// ---------------------------------------------------------------------------
// DEFERRED GRASS FIELD — the grass is grown in screen space.
//
// ONE fullscreen triangle. No plant primitive of any kind. For each pixel:
//
//  1. Unproject the scene depth -> G, the point where THIS eye ray meets the
//     ground (exact, from the terrain the base pass already rasterized).
//  2. Slide back up the ray to where it ENTERED the canopy, in closed form:
//     A = G - phi * s * H (s = cot elevation). That entry point, the azimuth
//     and s index the baked table, which answers with albedo, coverage,
//     surface normal and the HEIGHT h of the first thing the ray meets.
//  3. Put the hit back on the eye ray in closed form — the article's
//     |OB| = |OA| / cos(alpha), here t = (g + h - eye.y) / d.y — and shade it.
//
// No march, no loop, no per-plant work: 4 slab taps (2 elevation cells x
// surface/geometry, azimuth interpolated by the hardware along the 3rd texture
// axis) + 1 canopy-top tap for burial AO + 1 depth tap.
//
// Wind is exact for the model it uses: sway is a linear-in-height shear of the
// canopy, and a shear maps lines to lines, so instead of bending 500k plants
// the eye RAY is inverse-sheared into the baked frame. It costs 4 ALU.
// ---------------------------------------------------------------------------

const ELEV_MAX: i32 = 5;

struct Params {
  geom: vec4f,   // canopy H, tile metres, tile px, stand radius
  elevp: vec4f,  // q0, dq, elev max index, azimuth slices
  shade: vec4f,  // coverage threshold, ao rate, ao floor, detail amp
  ctrl: vec4f,   // max dist, mean sway, lod bias, edge sharpness
  extra: vec4f,  // detail freq, wind scale, macro amp, elev lerp (0/1)
  shade2: vec4f, // hit-plane align depth (m), relief amplitude (m), unused x2
}

@group(1) @binding(0) var<uniform> pp: Params;
@group(1) @binding(1) var field_samp: sampler;
@group(1) @binding(2) var surf0: texture_3d<f32>;
@group(1) @binding(3) var surf1: texture_3d<f32>;
@group(1) @binding(4) var surf2: texture_3d<f32>;
@group(1) @binding(5) var surf3: texture_3d<f32>;
@group(1) @binding(6) var surf4: texture_3d<f32>;
@group(1) @binding(7) var surf5: texture_3d<f32>;
@group(1) @binding(8) var geom0: texture_3d<f32>;
@group(1) @binding(9) var geom1: texture_3d<f32>;
@group(1) @binding(10) var geom2: texture_3d<f32>;
@group(1) @binding(11) var geom3: texture_3d<f32>;
@group(1) @binding(12) var geom4: texture_3d<f32>;
@group(1) @binding(13) var geom5: texture_3d<f32>;
@group(1) @binding(14) var scene_depth: texture_depth_2d;

struct Answer {
  surf: vec4f,
  geom: vec4f,
}

/// Shear of elevation cell e — the inverse of the runtime's q warp.
fn cell_shear(e: i32) -> f32 {
  let q = pp.elevp.x + pp.elevp.y * f32(e);
  return q / (1.0 - q);
}

/// One elevation cell of the table. The azimuth axis is the 3rd texture axis,
/// so azimuth (and, in the mips, direction-averaging with distance) comes free
/// from the hardware's trilinear filter with wrap addressing.
fn slab(e: i32, uvw: vec3f, lod: f32) -> Answer {
  var a: Answer;
  switch (e) {
    case 0: {
      a.surf = textureSampleLevel(surf0, field_samp, uvw, lod);
      a.geom = textureSampleLevel(geom0, field_samp, uvw, lod);
    }
    case 1: {
      a.surf = textureSampleLevel(surf1, field_samp, uvw, lod);
      a.geom = textureSampleLevel(geom1, field_samp, uvw, lod);
    }
    case 2: {
      a.surf = textureSampleLevel(surf2, field_samp, uvw, lod);
      a.geom = textureSampleLevel(geom2, field_samp, uvw, lod);
    }
    case 3: {
      a.surf = textureSampleLevel(surf3, field_samp, uvw, lod);
      a.geom = textureSampleLevel(geom3, field_samp, uvw, lod);
    }
    case 4: {
      a.surf = textureSampleLevel(surf4, field_samp, uvw, lod);
      a.geom = textureSampleLevel(geom4, field_samp, uvw, lod);
    }
    default: {
      a.surf = textureSampleLevel(surf5, field_samp, uvw, lod);
      a.geom = textureSampleLevel(geom5, field_samp, uvw, lod);
    }
  }
  return a;
}

fn ground_fast(xz: vec2f) -> f32 {
  let uv = xz / frame.terrain_size + 0.5;
  return textureSampleLevel(terrain_heightmap, linear_sampler, uv, 0.0).r;
}

fn vcorner(c: vec3i) -> f32 {
  return hash_f32(hash3(bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(c.z)));
}

/// Value noise plus its analytic gradient (the smoothstep-weighted trilinear
/// derivative) — the gradient is what tilts the normal, and it costs no extra
/// hashes.
fn vnoise3g(p: vec3f) -> vec4f {
  let b = floor(p);
  let f = p - b;
  let sm = f * f * (3.0 - 2.0 * f);
  let ds = 6.0 * f * (1.0 - f);
  let i = vec3i(b);
  let c000 = vcorner(i);
  let c100 = vcorner(i + vec3i(1, 0, 0));
  let c010 = vcorner(i + vec3i(0, 1, 0));
  let c110 = vcorner(i + vec3i(1, 1, 0));
  let c001 = vcorner(i + vec3i(0, 0, 1));
  let c101 = vcorner(i + vec3i(1, 0, 1));
  let c011 = vcorner(i + vec3i(0, 1, 1));
  let c111 = vcorner(i + vec3i(1, 1, 1));
  let x00 = mix(c000, c100, sm.x);
  let x10 = mix(c010, c110, sm.x);
  let x01 = mix(c001, c101, sm.x);
  let x11 = mix(c011, c111, sm.x);
  let y0 = mix(x00, x10, sm.y);
  let y1 = mix(x01, x11, sm.y);
  let v = mix(y0, y1, sm.z);
  let dx = mix(mix(c100 - c000, c110 - c010, sm.y), mix(c101 - c001, c111 - c011, sm.y), sm.z) * ds.x;
  let dy = mix(x10 - x00, x11 - x01, sm.z) * ds.y;
  let dz = (y1 - y0) * ds.z;
  return vec4f(v, dx, dy, dz);
}

fn vnoise2(p: vec2f) -> f32 {
  let b = floor(p);
  let f = p - b;
  let s = f * f * (3.0 - 2.0 * f);
  let i = vec2i(b);
  let c00 = vcorner(vec3i(i, 0));
  let c10 = vcorner(vec3i(i + vec2i(1, 0), 0));
  let c01 = vcorner(vec3i(i + vec2i(0, 1), 0));
  let c11 = vcorner(vec3i(i + vec2i(1, 1), 0));
  return mix(mix(c00, c10, s.x), mix(c01, c11, s.x), s.y);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  let H = pp.geom.x;
  let tile = pp.geom.y;
  let cam = frame.camera_pos;
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
  let far4 = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  let d = normalize(far4.xyz / far4.w - cam);

  // --- 1. where does this eye ray meet the ground? -------------------------
  let zbuf = textureLoad(scene_depth, vec2i(in.pos.xy), 0);
  var G = vec3f(0.0);
  var terrain_t = 1.0e9;
  var alive = abs(d.y) > 2.0e-3;
  if (zbuf < 1.0) {
    // Exact: the terrain the base pass already rasterized for this pixel.
    let h4 = frame.inv_view_proj * vec4f(ndc, zbuf, 1.0);
    G = h4.xyz / h4.w;
    terrain_t = dot(G - cam, d);
  } else {
    // Sky pixel (above the terrain silhouette, or looking up from inside the
    // canopy): fall back to the ground plane under the camera. The crossing
    // may be BEHIND the eye — legal, the table answers a line, and only hits
    // in front of the eye survive the t > 0 test below.
    let g = ground_fast(cam.xz);
    G = cam + d * ((g - cam.y) / d.y);
  }
  let g = G.y;

  // --- 2. inverse-shear the ray into the baked (rest) frame ---------------
  // Sway is a linear-in-height shear; shears map lines to lines, so bending
  // the ray is exact for the model instead of an approximation of it.
  let sway = wind_sway(vec3f(G.x, g, G.z), frame.time, pp.ctrl.y, 0.0);
  let k = sway.xz * (pp.extra.y / max(H, 0.05));
  let dq = normalize(vec3f(d.x - k.x * d.y, d.y, d.z - k.y * d.y));
  // The table is baked for rays descending along +phi; a ray climbing the same
  // line is the same query traversed backwards.
  let dd = select(-dq, dq, dq.y < 0.0);
  let s = length(dd.xz) / max(-dd.y, 1.0e-3);
  let q = s / (1.0 + s);
  let phi = normalize(dd.xz + vec2f(1.0e-6, 0.0));

  // --- 3. the ray answer -------------------------------------------------
  // Closed-form slide from the ground crossing back up the ray to where the
  // answer lives. The table is indexed by the canopy entry (y = H), but the
  // lookup point is placed at the ray's crossing of the EXPECTED FIRST-HIT
  // plane, y = H - align, and each cell is then offset by its own shear: that
  // is where the cell's answer physically sits, so a cell baked at shear s_e
  // and a pixel whose true shear is s_true agree about WHERE they are talking
  // about. Without it the two bracketing cells are misregistered by
  // (s1-s0)*align — metres at the grazing end — and their lerp is mush.
  let align = pp.shade2.x;
  let hit_plane = G.xz - phi * (s * (H - align));
  let fe = clamp((q - pp.elevp.x) / pp.elevp.y, 0.0, pp.elevp.z);
  let e0 = i32(floor(fe));
  let e1 = min(e0 + 1, ELEV_MAX);
  let az = atan2(dd.z, dd.x) * 0.15915494;
  let azw = az + 0.5 / pp.elevp.w;
  let uv0 = (hit_plane - phi * (cell_shear(e0) * align)) / tile;

  // Footprint of this pixel on the answer plane, in table texels. The answer's
  // structure is smeared ALONG the ray at grazing angles, so the geometric mean
  // of the two axes is the honest bandwidth, not the major axis.
  let du = length(dpdx(uv0));
  let dv = length(dpdy(uv0));
  let texels = sqrt(max(du * dv, 1.0e-12)) * pp.geom.z;
  let lod = max(0.0, log2(max(texels, 1.0e-6)) + pp.ctrl.z);

  let a0 = slab(e0, vec3f(uv0, azw), lod);
  var surf = a0.surf;
  var geo = a0.geom;
  // Elevation is the one axis the hardware cannot filter for us (it selects a
  // texture, not a slice), so it is an explicit 2-tap lerp — a uniform branch, so
  // `elevLerp=false` really is 2 taps per pixel instead of 4.
  if (pp.extra.w > 0.5 && e1 != e0) {
    let uv1 = (hit_plane - phi * (cell_shear(e1) * align)) / tile;
    let a1 = slab(e1, vec3f(uv1, azw), lod);
    let we = fe - floor(fe);
    surf = mix(surf, a1.surf, we);
    geo = mix(geo, a1.geom, we);
  }

  var cov = surf.a;
  var albedo = surf.rgb / max(cov, 1.0e-3);
  let icg = 1.0 / max(geo.a, 1.0e-3);
  var nrm = vec3f(geo.x * icg, 0.0, geo.y * icg);
  var h = clamp(geo.z * icg, 0.0, 1.0) * H;

  // The steepest elevation slab IS the canopy-top height map — its answer for a
  // near-vertical ray. One tap at the region this ray is heading into gives the
  // local top height, which serves twice below: as the burial reference for AO,
  // and as the honest ceiling when the eye sits inside the canopy.
  let topv = textureSampleLevel(geom0, field_samp, vec3f(hit_plane / tile, azw), lod);
  let top_local = clamp(topv.z / max(topv.a, 1.0e-3), 0.0, 1.0) * H;

  // --- 4. blade-scale relief ---------------------------------------------
  // One table texel is 2.3 cm; a near pixel is a millimetre. The sub-texel
  // structure is restored as a deterministic world-space relief field (never a
  // screen hash, so it does not crawl) that displaces the HIT HEIGHT and tilts
  // the normal with its analytic gradient. It fades out with the pixel
  // footprint, so it is gone before it could ever alias.
  var relief_cov = 0.0;
  var relief_fade = 0.0;
  if (pp.shade2.y > 0.0 && lod < 3.0) {
    let fade = 1.0 - lod / 3.0;
    let freq = pp.extra.x;
    // Anchored to the (smooth) hit plane, NOT to the hit height: h comes out of
    // an 8-bit channel, and feeding its 1 cm staircase into a 2 cm-wavelength
    // noise turns the whole near field into contour marbling. Found the hard
    // way at cam=far-horizon.
    let pos = vec3f(hit_plane.x, 0.37, hit_plane.y);
    // Two octaves: the coarse one is clump-scale (it is what reads as separate
    // tufts leaning different ways), the fine one is blade-scale.
    let ng = vnoise3g(pos * (freq * 0.22));
    let nf = vnoise3g(pos * freq);
    let amp = pp.shade2.y * fade;
    h += ((ng.x - 0.5) * 1.4 + (nf.x - 0.5) * 0.4) * amp;
    // Soft-normalized so the relief tilts the normal by at most ~45 degrees: the
    // raw gradient of a 1 cm-wavelength field is enormous and saturates the
    // normal into rainbow noise (visible immediately in debug=normals).
    var grad = (vec2f(ng.y, ng.w) * (freq * 0.22 * 1.4) + vec2f(nf.y, nf.w) * (freq * 0.4)) * amp;
    grad = grad / (1.0 + 0.6 * length(grad));
    nrm = vec3f(nrm.x - grad.x, 0.0, nrm.z - grad.y);
    albedo *= 1.0 + (ng.x - 0.5) * 0.35 * fade;
    relief_cov = (ng.x - 0.5) * 0.7 + (nf.x - 0.5) * 0.3;
    relief_fade = fade;
  }

  // --- 5. put the hit back on the true eye ray ---------------------------
  // The table answers for a ray entering the canopy from above. When the EYE is
  // inside the canopy (grazing sits 9 cm under the tallest tips, inside-plant is
  // half way down) part of that answer lies behind the eye, and the table cannot
  // report the next hit further down. Both branches below are approximations,
  // and both are the "camera inside a plant" degradation the rules allow.
  let camH = cam.y - g;
  var t_hit = (g + h - cam.y) / d.y;
  if (camH < H) {
    // The eye is inside the canopy slab of this ground reference. (Everywhere
    // else — including hillsides above the eye, whose rays still enter the
    // canopy through its top — the answer above is exact and used as is.)
    if (d.y > 0.0) {
      // Climbing out through the tips: only canopy above the eye can be in
      // front of it, and how much of the sky those tips cover is the extinction
      // of the remaining segment (path (H-camH)*s through the same canopy),
      // which falls off with elevation exactly as a fringe of tips should.
      if (h <= camH + 0.01) { alive = false; }
      t_hit = min(t_hit, 0.9);
      h = camH + t_hit * d.y;
      // Extinction of the segment above the eye, faded out once the eye is
      // properly buried: at eye level the fringe of tips over the sky is real,
      // but half way down the canopy the table has nothing honest to say about
      // what is overhead, and inventing blobs is worse than fading out.
      cov *= (1.0 - exp(-1.4 * (H - camH) * s)) * smoothstep(0.55, 0.95, camH / H);
    } else if (h > camH) {
      // Descending: hits above the eye lie behind it, and the table cannot
      // report the next hit down. What the eye really meets is the canopy top
      // where the ray dips under it, so fall back to the LOCAL top height —
      // which keeps the canopy's real height variation instead of ironing it
      // flat — and only cap at eye level when even that is above the eye.
      h = min(top_local, camH - 0.02);
      t_hit = (g + h - cam.y) / d.y;
    }
  }
  if (t_hit <= 0.03 || t_hit > min(terrain_t + 0.05, pp.ctrl.x)) { alive = false; }
  let X = cam + d * t_hit;
  // The stand's region, tested on the HIT — the ground crossing of a
  // near-horizontal ray can be 150 m behind the camera and says nothing.
  if (max(abs(X.x), abs(X.z)) > pp.geom.w) { alive = false; }

  // --- 6. burial AO: how deep under the canopy top does the hit sit? ------
  // Relief dips darken automatically, which is what gives the canopy an
  // interior instead of a shell. At distance both terms are averages, so a
  // floor of interior shading is faded in with the footprint — the averaged
  // canopy of a far pixel is genuinely part shadowed interior, not all top.
  let burial = max(top_local - h, 0.0);
  var ao = mix(pp.shade.z, 1.0, exp(-burial * pp.shade.y));
  ao *= mix(1.0, pp.shade2.z, clamp(lod * 0.25, 0.0, 1.0));

  if (pp.extra.z > 0.0) {
    // Low-frequency colour drift breaks up the table's tiling period.
    let m = vnoise2(G.xz * 0.077) - 0.5;
    albedo *= vec3f(1.0 + m * pp.extra.z, 1.0 + m * 0.75 * pp.extra.z, 1.0 - m * 0.5 * pp.extra.z);
  }

  // Camera-inside-plant rule: dissolve the last few centimetres smoothly.
  cov *= smoothstep(0.10, 0.40, t_hit);

  // Blade-scale relief breaks up whatever coverage survived — but only the
  // PARTIAL band (4*c*(1-c)), so a solid canopy stays solid and only genuine
  // edges (the fringe of tips against the sky, the averaged far canopy) get
  // their raggedness back. This is a world-space detail field, not a dither:
  // hard-edged after the sharpen below, stable under motion, and faded out with
  // the pixel footprint before it could alias.
  cov = clamp(cov + relief_cov * pp.shade.w * relief_fade * 4.0 * cov * (1.0 - cov), 0.0, 1.0);

  // Sharpen the coverage ramp up close (hard-edged, no dither), keep the
  // honest prefiltered coverage at distance so far grass is an averaged canopy.
  let sharp = clamp(pp.ctrl.w * (1.0 - clamp(lod * 0.4, 0.0, 1.0)), 0.0, 1.0);
  let thr = pp.shade.x;
  let hard = smoothstep(thr - 0.07, thr + 0.07, cov);
  var out_a = mix(cov, hard, sharp);
  if (!alive || out_a < 0.004) { discard; }

  let ny = sqrt(max(1.0 - nrm.x * nrm.x - nrm.z * nrm.z, 0.0));
  let n = normalize(vec3f(nrm.x, ny + 0.05, nrm.z));
  let lit_albedo = albedo * ao;
  var color = light_surface(lit_albedo, n, X);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, X);
  } else {
    out_a = 1.0;
  }
  return vec4f(debug_shade(color, lit_albedo, n, out_a, X), out_a);
}
