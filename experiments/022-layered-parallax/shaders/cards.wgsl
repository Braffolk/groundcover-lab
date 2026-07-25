#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// LAYERED PARALLAX CARDS.
//
// A plant is baked, per azimuth, into three DEPTH SLABS (front / middle /
// back). Each slab is a flat image plus ONE number: the mean depth of the
// geometry in it, i.e. the plane it lives on inside the plant. The card that
// gets rasterised is only a proxy: the fragment shader takes the real eye ray,
// intersects it with those three planes in the plant's own frame, and probes
// the slabs front-to-back until one is opaque. The first opaque hit wins —
// that is a hard alpha test AND the correct occlusion order in one step.
//
// Because the planes are real 3D planes at real depths, the three images slide
// against each other exactly as geometry would: the interior parallaxes, the
// union silhouette changes shape with the view (it does not merely rotate),
// and each layer magnifies differently as you walk up to a plant. Cost stays
// at billboard scale: ONE quad per plant (+ a horizontal top card), 1-3 albedo
// taps + 1 normal tap, no marching, no frag_depth, no blending.
//
// Two entry-point pairs, one per distance ring, so neither pays for the other:
//   vs_near/fs_near — the layered reprojection, near atlas.
//   vs_far/fs_far   — one merged tile, one tap, implicit-derivative sampling
//                     off a small fully-mipped atlas. This IS the billboard
//                     baseline, and it is what most of the screen runs.
// Each instance draws 12 vertices: 0..5 the side card, 6..11 the top card.

const N_AZIM: u32 = 8u;
const N_LAYER: u32 = 3u;
const TILE_TOP0: u32 = 24u;
const TILE_MERGED_SIDE0: u32 = 27u;
const TILE_MERGED_TOP: u32 = 35u;
const TWO_PI: f32 = 6.2831853;
const PI: f32 = 3.14159265;
/** Floor on |cos| between the eye ray and a slab normal: below it the
    reprojection is ill conditioned (card seen almost edge-on). */
const MIN_SLANT: f32 = 0.45;

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> near_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read> far_insts: array<PlantInst>;
// Two vec4 per tile: [u0, v0, du, dv] and [plane depth, texW, texH, isFar].
@group(1) @binding(3) var<storage, read> tiles: array<vec4f>;
@group(1) @binding(4) var near_albedo: texture_2d<f32>;
@group(1) @binding(5) var near_normal: texture_2d<f32>;
@group(1) @binding(6) var far_albedo: texture_2d<f32>;
@group(1) @binding(7) var far_normal: texture_2d<f32>;
@group(1) @binding(8) var slab_sampler: sampler;

/// Rotate about +Y by the angle whose (cos, sin) is `cs`.
fn rot_y(v: vec3f, cs: vec2f) -> vec3f {
  return vec3f(cs.x * v.x + cs.y * v.z, v.y, -cs.y * v.x + cs.x * v.z);
}

struct Plant {
  base: vec3f,
  axis: vec3f,     // baked capture centre at ground level, world space
  yaw_cs: vec2f,
  inv_yaw_cs: vec2f,
  scale: f32,
  shear: vec3f,    // horizontal wind displacement at the plant tip
  to_cam: vec3f,
  d_xz: f32,
  fwd_xz: vec2f,
  fade: f32,       // near-fade x region-rim fade
}

fn decode_plant(inst: PlantInst) -> Plant {
  let bits = inst.packed_bits;
  var p: Plant;
  let yaw = f32(bits & 1023u) * (TWO_PI / 1024.0);
  p.scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (TWO_PI / 1024.0);
  p.base = inst.pos;
  p.yaw_cs = vec2f(cos(yaw), sin(yaw));
  p.inv_yaw_cs = vec2f(p.yaw_cs.x, -p.yaw_cs.y);
  // The baked capture is centred on the clump, which sits off the mesh origin.
  p.axis = p.base + rot_y(vec3f(info.card_cx, 0.0, info.card_cz), p.yaw_cs) * p.scale;

  // Wind: horizontal component only, so the sway stays an exact affine shear
  // of space, shear(X) = X + S * vfrac(X.y), which can be inverted on the eye
  // ray. wind_sway's small vertical dip is dropped on purpose.
  let entry = stand_table[u32(info.entry_index)];
  let sway3 = wind_sway(p.base, frame.time, entry.sway, phase);
  p.shear = vec3f(sway3.x, 0.0, sway3.z);

  p.to_cam = frame.camera_pos - p.axis;
  p.d_xz = length(p.to_cam.xz);
  p.fwd_xz = p.to_cam.xz / max(p.d_xz, 1e-4);

  let r = info.card_r;
  let seg_y = clamp(frame.camera_pos.y, p.base.y + info.card_y0 * p.scale, p.base.y + info.card_y1 * p.scale);
  let d_core = length(vec3f(p.to_cam.x, frame.camera_pos.y - seg_y, p.to_cam.z));
  let near_fade = smoothstep(r * p.scale * 0.35, r * p.scale * 1.05, d_core);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, p.d_xz);
  p.fade = near_fade * edge_fade;
  return p;
}

/// sin(elevation) of the eye above a point `h` up the plant (unit scale).
fn view_elev(p: Plant, h: f32) -> f32 {
  let dy = frame.camera_pos.y - (p.base.y + h * p.scale);
  return abs(dy) / max(length(vec3f(p.to_cam.x, dy, p.to_cam.z)), 1e-4);
}

/// The top card only makes sense from well above; below that it reads as a
/// pale floating cutout among the side cards, so its coverage erodes away.
fn top_fade(p: Plant, h: f32) -> f32 {
  return smoothstep(0.35, 0.6, view_elev(p, h));
}

fn card_corner(vi: u32) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  return corners[vi % 6u];
}

// ---------------------------------------------------------------------------
// FAR RING — one merged tile, one tap. Structurally the billboard baseline.
// ---------------------------------------------------------------------------

struct FarOut {
  @builtin(position) pos: vec4f,
  @location(0) uv_card: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) rect: vec4f,   // atlas rect of the tile
  @location(3) @interpolate(flat) yaw_cs: vec2f,
  @location(4) @interpolate(flat) erode: f32,
  @location(5) shade: f32,
}

@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FarOut {
  let p = decode_plant(far_insts[ii]);
  let c = card_corner(vi);
  let is_top = vi >= 6u;
  let r = info.card_r;
  let y0 = info.card_y0;
  let hspan = max(info.card_y1 - y0, 1e-4);
  var fade = p.fade;

  var tile: u32;
  var world: vec3f;
  var uv_card: vec2f;
  var shade: f32;
  if (is_top) {
    tile = TILE_MERGED_TOP;
    let h = tiles[tile * 2u + 1u].x;
    let c2 = vec2f(c.x, c.y * 2.0 - 1.0);
    let local = vec3f(info.card_cx + c2.x * r, h, info.card_cz + c2.y * r);
    world = p.base + rot_y(local, p.yaw_cs) * p.scale + p.shear * ((h - y0) / hspan);
    uv_card = c2 * 0.5 + vec2f(0.5);
    shade = 1.0 - info.bottom_shade * 0.3;
    fade *= top_fade(p, h);
  } else {
    // Nearest baked azimuth in the plant's local frame.
    let local_dir = rot_y(vec3f(p.fwd_xz.x, 0.0, p.fwd_xz.y), p.inv_yaw_cs);
    let ang = atan2(local_dir.x, local_dir.z);
    let k = u32((i32(round(ang * (f32(N_AZIM) / TWO_PI))) + i32(N_AZIM) * 2) % i32(N_AZIM));
    tile = TILE_MERGED_SIDE0 + k;
    let right = vec3f(p.fwd_xz.y, 0.0, -p.fwd_xz.x);
    world = p.axis + right * (c.x * r * p.scale);
    world.y = p.base.y + mix(y0, info.card_y1, c.y) * p.scale;
    world += p.shear * c.y;
    uv_card = vec2f(c.x * 0.5 + 0.5, 1.0 - c.y);
    shade = mix(1.0 - info.bottom_shade, 1.0, c.y);
  }

  var out: FarOut;
  // A card whose fade dropped below the alpha reference discards EVERY
  // fragment — emit it behind the near plane instead of rasterizing it.
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv_card = uv_card;
  out.world = world;
  out.rect = tiles[tile * 2u];
  out.yaw_cs = p.yaw_cs;
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.shade = shade;
  return out;
}

@fragment
fn fs_far(in: FarOut) -> @location(0) vec4f {
  let uv = in.rect.xy + in.uv_card * in.rect.zw;
  // Sampled before any non-uniform discard (implicit derivatives are legal).
  let alb = textureSample(far_albedo, slab_sampler, uv);
  let enc = textureSample(far_normal, slab_sampler, uv).xy;
  if (alb.a < in.erode) {
    discard;
  }
  let n = world_normal(enc, in.yaw_cs);
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}

// ---------------------------------------------------------------------------
// NEAR RING — the layered reprojection.
// ---------------------------------------------------------------------------


// Sampling a slab is a central projection from the eye between two planes —
// the proxy quad and the slab plane — i.e. a HOMOGRAPHY. Its numerator and
// denominator are both affine in the proxy's own surface coordinates, so the
// rasterizer's perspective-correct interpolator can evaluate them for free:
// the fragment shader gets `proj` = (Bu, Bv, beta) interpolated, does ONE
// reciprocal, and every slab is then two multiply-adds. All three slabs share
// the denominator, because they are parallel planes.
//
//   tu(L) = cu + k(L) * Bu / beta,  tv(L) = cv + k(L) * Bv / beta
//   k(L)  = plane_depth(L) - dot(eye, plane_normal)
//
// That is exact perspective for every slab, at a handful of instructions.
struct NearOut {
  @builtin(position) pos: vec4f,
  @location(0) proj: vec3f,                       // Bu, Bv, beta (see above)
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) rect0: vec4f,   // atlas rects of the 3 slabs
  @location(3) @interpolate(flat) rect1: vec4f,
  @location(4) @interpolate(flat) rect2: vec4f,
  @location(5) @interpolate(flat) kk: vec3f,      // k(L) per slab, front first
  @location(6) @interpolate(flat) cc: vec4f,      // cu, cv, erode, |beta| floor
  @location(7) @interpolate(flat) yaw_cs: vec2f,
  @location(8) @interpolate(flat) tex_flags: vec4f, // texW0, texH0, is_top, 0
}

@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> NearOut {
  let p = decode_plant(near_insts[ii]);
  let c = card_corner(vi);
  let c01 = vec2f(c.x * 0.5 + 0.5, c.y);
  let is_top = vi >= 6u;
  let r = info.card_r;
  let y0 = info.card_y0;
  let y1 = info.card_y1;
  let hspan = max(y1 - y0, 1e-4);
  let c3 = vec3f(info.card_cx, 0.0, info.card_cz);
  var fade = p.fade;

  // Ray from the eye toward the plant — drives the proxy's margins. The
  // elevation is measured to the plant's MID height, not its base: the base
  // would size the proxy for the most extreme fragment it has and make every
  // close-up card ~70% taller, and what then falls off the bottom edge is the
  // deepest slab down at the soil line.
  let ray = normalize(-p.to_cam);
  let mid = mix(y0, y1, 0.5) * p.scale;
  let tan_e = (frame.camera_pos.y - (p.base.y + mid)) / max(p.d_xz, 1e-3);

  // Eye in the plant's own unsheared frame (the wind shear is an affine map of
  // space, so inverting it on the eye keeps the reprojected ray a straight line).
  let vfrac_eye = ((frame.camera_pos.y - p.base.y) / max(p.scale, 1e-4) - y0) / hspan;
  let q_eye = rot_y(frame.camera_pos - p.shear * vfrac_eye - p.base, p.inv_yaw_cs) / max(p.scale, 1e-4);

  var tile_base: u32;
  var world: vec3f;
  var q_frag: vec3f;
  var nrm_axis: vec3f;
  var tan_u: vec3f;
  var tan_v: vec3f;
  var scale_u: f32;
  var scale_v: f32;
  var depth_bias = 0.0; // plane equation is dot(X, n) = depth + depth_bias

  if (is_top) {
    // Height-band planes. The proxy sits on the FRONT (highest) one, so the
    // deeper bands show up displaced toward the camera and it only has to
    // stretch in that one direction.
    tile_base = TILE_TOP0;
    let h_top = tiles[tile_base * 2u + 1u].x;
    let ray_l = rot_y(ray, p.inv_yaw_cs);
    let horiz = ray_l.xz;
    let ext = clamp(info.top_span * length(horiz) / max(-ray_l.y, 1e-3), 0.0, r * 0.8);
    let e = -normalize(horiz + vec2f(1e-6, 0.0)) * ext;
    let lo = vec2f(-r) + min(e, vec2f(0.0));
    let hi = vec2f(r) + max(e, vec2f(0.0));
    let p_xz = mix(lo, hi, c01);
    q_frag = vec3f(c3.x + p_xz.x, h_top, c3.z + p_xz.y);
    world = p.base + rot_y(q_frag, p.yaw_cs) * p.scale + p.shear * ((h_top - y0) / hspan);
    nrm_axis = vec3f(0.0, 1.0, 0.0);
    tan_u = vec3f(1.0, 0.0, 0.0);
    tan_v = vec3f(0.0, 0.0, 1.0);
    scale_u = 0.5 / r;
    scale_v = 0.5 / r;
    fade *= top_fade(p, h_top);
  } else {
    // Depth planes. The proxy stays on the plant's own axis — pushing it out to
    // the front slab would put it closer to the eye and inflate every close-up
    // card by its perspective factor, which is exactly where the fragments are.
    let local_dir = rot_y(vec3f(p.fwd_xz.x, 0.0, p.fwd_xz.y), p.inv_yaw_cs);
    let ang = atan2(local_dir.x, local_dir.z);
    let k = u32((i32(round(ang * (f32(N_AZIM) / TWO_PI))) + i32(N_AZIM) * 2) % i32(N_AZIM));
    let a_k = f32(k) * (TWO_PI / f32(N_AZIM));
    var theta = ang - a_k;
    if (theta > PI) { theta -= TWO_PI; }
    if (theta < -PI) { theta += TWO_PI; }
    tile_base = k * N_LAYER;
    nrm_axis = vec3f(sin(a_k), 0.0, cos(a_k));
    tan_u = vec3f(cos(a_k), 0.0, -sin(a_k));
    tan_v = vec3f(0.0, 1.0, 0.0);
    scale_u = 0.5 / r;
    scale_v = -1.0 / hspan;
    // Slab depths were baked as dot(X - capture centre, n).
    depth_bias = dot(c3, nrm_axis);

    let d_front = tiles[tile_base * 2u + 1u].x;
    let d_back = tiles[(tile_base + N_LAYER - 1u) * 2u + 1u].x;
    let right = vec3f(p.fwd_xz.y, 0.0, -p.fwd_xz.x);

    // Margins. Reprojecting a slab `m` in front of the proxy plane maps its
    // content to proxy coordinate u/cos(a) + m*tan(a), so the range the proxy
    // needs is the plant's own extent shifted by -m*sin(a). Front slab and back
    // slab shift opposite ways, which is one edge each.
    let sin_e = tan_e * inverseSqrt(1.0 + tan_e * tan_e);
    let su = clamp(vec2f(-d_front, -d_back) * sin(theta), vec2f(-r * 0.5), vec2f(r * 0.5));
    let sv = clamp(vec2f(-d_front, -d_back) * sin_e, vec2f(-hspan * 0.5), vec2f(hspan * 0.5));
    let y_lo = y0 + min(min(sv.x, sv.y), 0.0);
    let y_hi = y1 + max(max(sv.x, sv.y), 0.0);
    let u_lo = -r + min(min(su.x, su.y), 0.0);
    let u_hi = r + max(max(su.x, su.y), 0.0);

    let off = mix(u_lo, u_hi, c01.x) * p.scale;
    let local_y = mix(y_lo, y_hi, c.y);
    let xu = vec3f(p.axis.x + right.x * off, p.base.y + local_y * p.scale, p.axis.z + right.z * off);
    q_frag = rot_y(xu - p.base, p.inv_yaw_cs) / max(p.scale, 1e-4);
    world = xu + p.shear * ((local_y - y0) / hspan);

    // Steeply-viewed vertical slabs are seen edge-on: one pixel then spans a
    // wide swath of the slab plane and the reprojection turns to mush. That is
    // exactly the regime the top card owns, and top_fade has already brought it
    // in by 0.6, so erode the side card away over the same range instead.
    fade *= 1.0 - smoothstep(0.62, 0.88, view_elev(p, 0.5));
  }

  // Homography terms — affine in the proxy's surface coordinates, so the
  // interpolator reproduces them exactly at every fragment.
  let rel = q_frag - q_eye;
  let eye_n = dot(q_eye, nrm_axis);
  let aux0 = tiles[tile_base * 2u + 1u];
  var cv: f32;
  if (is_top) {
    cv = 0.5 + (q_eye.z - c3.z) * scale_v;
  } else {
    cv = 1.0 - (q_eye.y - y0) / hspan;
  }

  var out: NearOut;
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.proj = vec3f(dot(rel, tan_u) * scale_u, dot(rel, tan_v) * scale_v, dot(rel, nrm_axis));
  out.world = world;
  out.rect0 = tiles[tile_base * 2u];
  out.rect1 = tiles[(tile_base + 1u) * 2u];
  out.rect2 = tiles[(tile_base + 2u) * 2u];
  out.kk =
    vec3f(aux0.x, tiles[(tile_base + 1u) * 2u + 1u].x, tiles[(tile_base + 2u) * 2u + 1u].x) +
    vec3f(depth_bias - eye_n);
  out.cc = vec4f(
    0.5 + dot(q_eye - c3, tan_u) * scale_u,
    cv,
    info.alpha_ref / max(fade, 1e-3),
    MIN_SLANT * length(rel),
  );
  out.yaw_cs = p.yaw_cs;
  out.tex_flags = vec4f(aux0.y, aux0.z, select(0.0, 1.0, is_top), 0.0);
  return out;
}

fn tile_lod(dx: vec2f, dy: vec2f, texels: vec2f) -> f32 {
  let a = dx * texels;
  let b = dy * texels;
  return 0.5 * log2(max(max(dot(a, a), dot(b, b)), 1e-12));
}

fn in_tile(uv: vec2f) -> bool {
  return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

@fragment
fn fs_near(in: NearOut) -> @location(0) vec4f {
  let erode = in.cc.z;
  // Floor |beta| so an almost edge-on proxy degrades into a mild warp instead
  // of exploding; at those angles the top card carries the image anyway.
  let beta = select(max(in.proj.z, in.cc.w), min(in.proj.z, -in.cc.w), in.proj.z < 0.0);
  let ab = in.proj.xy / beta;

  let puv0 = in.cc.xy + in.kk.x * ab;
  let lod0 = max(tile_lod(dpdx(puv0), dpdy(puv0), in.tex_flags.xy), 0.0);
  // Slabs 1 and 2 live in quarter-area tiles: exactly one mip level coarser.
  let lod1 = max(lod0 - 1.0, 0.0);

  var hit_col = vec4f(0.0);
  var hit_uv = vec2f(0.0);
  var hit_v = 0.0;
  var hit_lod = lod0;
  var hit_layer = 3u; // 3 = miss

  // Front to back, stopping at the first opaque slab: that is simultaneously
  // the hard alpha test and the occlusion decision. A slab whose reprojected
  // uv left the tile holds nothing for this pixel, and skipping it skips its
  // tap — most fragments in the proxy's margin cost no fetch at all.
  if (in_tile(puv0)) {
    let uv = in.rect0.xy + puv0 * in.rect0.zw;
    let s = textureSampleLevel(near_albedo, slab_sampler, uv, lod0);
    if (s.a >= erode) {
      hit_col = s;
      hit_uv = uv;
      hit_v = puv0.y;
      hit_layer = 0u;
    }
  }
  if (hit_layer == 3u) {
    let puv1 = in.cc.xy + in.kk.y * ab;
    if (in_tile(puv1)) {
      let uv = in.rect1.xy + puv1 * in.rect1.zw;
      let s = textureSampleLevel(near_albedo, slab_sampler, uv, lod1);
      if (s.a >= erode) {
        hit_col = s;
        hit_uv = uv;
        hit_v = puv1.y;
        hit_lod = lod1;
        hit_layer = 1u;
      }
    }
  }
  if (hit_layer == 3u) {
    let puv2 = in.cc.xy + in.kk.z * ab;
    if (in_tile(puv2)) {
      let uv = in.rect2.xy + puv2 * in.rect2.zw;
      let s = textureSampleLevel(near_albedo, slab_sampler, uv, lod1);
      if (s.a >= erode) {
        hit_col = s;
        hit_uv = uv;
        hit_v = puv2.y;
        hit_lod = lod1;
        hit_layer = 2u;
      }
    }
  }
  if (hit_layer == 3u) {
    discard;
  }

  // The normal atlas is half the albedo resolution — one mip level coarser.
  let enc = textureSampleLevel(near_normal, slab_sampler, hit_uv, max(hit_lod - 1.0, 0.0)).xy;
  let n = world_normal(enc, in.yaw_cs);

  // Grounding gradient + per-slab occlusion: slabs behind the front one sit
  // deeper in the canopy and really do receive less light. Both belong to the
  // light term, so the albedo debug view stays the raw baked colour.
  let height_f = select(clamp(1.0 - hit_v, 0.0, 1.0), 0.85, in.tex_flags.z > 0.5);
  var shade = mix(1.0 - info.bottom_shade, 1.0, height_f);
  shade *= max(1.0 - info.layer_shade * f32(hit_layer), 0.05);

  var color = light_surface(hit_col.rgb * shade, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, hit_col.rgb, n, hit_col.a, in.world), 1.0);
}

// ---------------------------------------------------------------------------
// shared shading helpers
// ---------------------------------------------------------------------------

/// Octahedral decode (y-primary, the convention the bake stored), then rotate
/// the mesh-frame normal by the plant's yaw.
fn world_normal(e2: vec2f, yaw_cs: vec2f) -> vec3f {
  let e = e2 * 2.0 - 1.0;
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  let n = normalize(vec3f(x, y, z));
  return vec3f(yaw_cs.x * n.x + yaw_cs.y * n.z, n.y, -yaw_cs.y * n.x + yaw_cs.x * n.z);
}
