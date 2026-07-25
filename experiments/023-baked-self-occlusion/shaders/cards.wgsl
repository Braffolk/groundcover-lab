#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./info.wgsl"

// ONE view-aligned quad per plant, textured from the baked view whose
// direction is nearest the current one. Two things turn that flat quad into
// something that reads as solid geometry, and neither of them is geometry:
//
//  1. DEPTH WARP (near tier). The geometry atlas stores, per texel, how deep
//     the first surface sits behind the tile's near plane. The fragment
//     intersects its own eye ray with that near plane (closed form), takes ONE
//     depth tap, and slides along the ray to the stored depth. Near parts of
//     the plant therefore shift against far parts as the camera moves, and the
//     alpha-tested silhouette is the warped silhouette — it changes shape with
//     view direction instead of merely rotating. One extra tap, no marching.
//
//  2. BAKED SELF-OCCLUSION (both tiers). Each texel also carries the fraction
//     of the sky it can see and a normal leaned toward its openness direction.
//     Sun visibility comes out of a cone test against those two numbers, so
//     interiors stay dark, tips catch light, and the shading changes when the
//     sun moves.
//
// LOD: beyond `nearSplit` the plant is drawn by vs_far/fs_far, which resolves
// the same tile UV at the SIX quad vertices instead of per fragment (the eye
// ray is near-parallel at that distance) and drops the warp — 2 implicit taps
// and no ray maths per fragment. The warp fades out before the split so the
// two tiers agree there.
//
// Hard alpha test with depth write, no dither, no blending; near/region fades
// erode coverage through the alpha reference.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var<uniform> tiles: TileTable;
@group(1) @binding(3) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(4) var atlas_geom: texture_2d<f32>;
@group(1) @binding(5) var albedo_sampler: sampler;
@group(1) @binding(6) var geom_sampler: sampler;

const GRID: u32 = 5u;
const INV_GRID: f32 = 0.2;
const UV_INSET: f32 = 0.0026; // 1 texel of a 384px tile
// Baked rows: 8 azimuths at 6deg and 30deg, 4 azimuths at 52deg and 74deg,
// one straight-down tile. Switch elevations sit midway between rows and are
// pre-sined, so the row test compares the view direction's y directly.
const ROW_SW0: f32 = 0.3090; // sin(18deg)
const ROW_SW1: f32 = 0.6561; // sin(41deg)
const ROW_SW2: f32 = 0.8910; // sin(63deg)
const ROW_SW3: f32 = 0.9903; // sin(82deg)
// Largest parallax offset, in tile widths, the single warp step is allowed to
// produce (0.16 tile ~= 12 cm on a calamagrostis clump).
const INV_WARP_LIMIT2: f32 = 39.0625; // 1 / 0.16^2

fn rot_y(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn atlas_uv(tile: u32, uv: vec2f) -> vec2f {
  let cell = vec2f(f32(tile % GRID), f32(tile / GRID));
  return (cell + clamp(uv, vec2f(UV_INSET), vec2f(1.0 - UV_INSET))) * INV_GRID;
}

/// Where the eye ray through `p_local` crosses the tile's near plane, in that
/// tile's [0,1]^2 box coordinates. `w` need not be normalized: every term the
/// result depends on is invariant to its length.
fn tile_uv_of(t0: vec4f, t1: vec4f, t2: vec4f, t3: vec4f, eye: vec3f, w: vec3f) -> vec2f {
  let wb = min(dot(w, t2.xyz), -1e-6);
  let p0 = eye + w * ((t2.w - dot(eye, t2.xyz)) / wb);
  let inv_box = 0.5 / vec2f(t3.x, t3.y);
  return vec2f(0.5, 0.5) + vec2f(dot(p0, t0.xyz) - t0.w, -(dot(p0, t1.xyz) - t1.w)) * inv_box;
}

// ---------------------------------------------------------------------------
// Shared per-plant setup: quad, baked tile choice, wind, fades.
// ---------------------------------------------------------------------------

struct Card {
  world: vec3f,      // rasterized (wind-displaced) position
  world_rest: vec3f, // at-rest position — the plant-local mapping uses this
  cam_local: vec3f,
  tile: u32,
  fade: f32,
  dist: f32,
  cs: f32,
  sn: f32,
  inv_scale: f32,
  base: vec3f,
}

fn build_card(vi: u32, ii: u32) -> Card {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let base = inst.pos;
  let entry = stand_table[u32(info.ids.y)];
  let cs = cos(yaw);
  let sn = sin(yaw);

  let ctr_world = base + rot_y(info.aabb_c.xyz, cs, sn) * scale;
  let to_cam = frame.camera_pos - ctr_world;
  let dist = max(length(to_cam), 1e-4);
  let fwd = to_cam / dist;

  // View-aligned quad frame (degenerate only when looking straight down).
  var right = vec3f(1.0, 0.0, 0.0);
  if (abs(fwd.y) < 0.999) {
    right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  }
  let up = cross(fwd, right);

  // Baked view nearest the current one, in the plant's yawed frame.
  let v_local = rot_y(fwd, cs, -sn);
  var tile = 24u;
  if (v_local.y < ROW_SW3) {
    var tile_base = 0u;
    var n_az = 8;
    if (v_local.y >= ROW_SW0) { tile_base = 8u; }
    if (v_local.y >= ROW_SW1) { tile_base = 16u; n_az = 4; }
    if (v_local.y >= ROW_SW2) { tile_base = 20u; n_az = 4; }
    let ang = atan2(v_local.x, v_local.z);
    let k = (i32(round(ang * (f32(n_az) / 6.2831853))) + n_az) % n_az;
    tile = tile_base + u32(k);
  }

  // Conservative but tight screen extent: the plant-local AABB projected onto
  // the quad axes — exact, continuous across tile switches, and much smaller
  // than a bounding-sphere quad at grazing angles.
  let right_l = rot_y(right, cs, -sn);
  let up_l = rot_y(up, cs, -sn);
  let hx = dot(abs(right_l), info.aabb_h.xyz) * scale;
  let hy = dot(abs(up_l), info.aabb_h.xyz) * scale;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];
  let world_rest = ctr_world + right * (c.x * hx) + up * (c.y * hy);

  // Wind: tips lean, roots do not. The plant-local mapping uses the AT-REST
  // position, so the imagery shears with the quad instead of sliding inside it.
  let plant_h = max(info.aabb_c.w * scale, 1e-3);
  let sway = wind_sway(base, frame.time, entry.sway, phase);
  let hf = clamp((world_rest.y - base.y) / plant_h, 0.0, 1.0);
  let hf_mid = clamp((ctr_world.y - base.y) / plant_h, 0.0, 1.0);

  // Camera-inside fade + region rim fade, both applied by eroding coverage.
  // Cards magnified far past one screen pixel per atlas texel are this
  // technique's breakdown zone (a 384 px tile is ~2 mm/texel, the geometry
  // atlas a third of that), so the sanctioned camera-inside-plant fade is
  // pitched wide enough to erode them instead of showing melted blades.
  let near_fade = smoothstep(info.caps.w * scale * 0.45, info.caps.w * scale * 1.7, dist);
  let d_xz = length(frame.camera_pos.xz - base.xz);
  let edge_fade = 1.0 - smoothstep(info.ids.z * 0.86, info.ids.z, d_xz);
  let inv_scale = 1.0 / max(scale, 1e-4);

  var card: Card;
  card.world = world_rest + sway * hf;
  card.world_rest = world_rest;
  card.cam_local = rot_y(frame.camera_pos - sway * hf_mid - base, cs, -sn) * inv_scale;
  card.tile = tile;
  card.fade = near_fade * edge_fade;
  card.dist = dist;
  card.cs = cs;
  card.sn = sn;
  card.inv_scale = inv_scale;
  card.base = base;
  return card;
}

// A card whose fade fell below the alpha reference discards EVERY fragment —
// emit it behind the near plane instead of rasterizing a guaranteed-empty
// quad (that set is exactly the screen-filling one at inside-plant).
fn clip_pos(card: Card) -> vec4f {
  if (card.fade < info.caps.z) {
    return vec4f(0.0, 0.0, -1.0, 1.0);
  }
  return frame.view_proj * vec4f(card.world, 1.0);
}

// ---------------------------------------------------------------------------
// Near tier — per-fragment ray, depth warp.
// ---------------------------------------------------------------------------

struct NearOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec3f,                        // fragment in plant-local space
  @location(1) @interpolate(flat) cam_local: vec3f, // eye in plant-local space
  @location(2) world: vec3f,
  @location(3) @interpolate(flat) tile: u32,
  @location(4) @interpolate(flat) erode: f32,
  @location(5) @interpolate(flat) yaw_cs: vec2f,
  @location(6) @interpolate(flat) warp_k: f32,
}

@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> NearOut {
  let card = build_card(vi, ii);
  var out: NearOut;
  out.pos = clip_pos(card);
  out.local = rot_y(card.world_rest - card.base, card.cs, -card.sn) * card.inv_scale;
  out.cam_local = card.cam_local;
  out.world = card.world;
  out.tile = card.tile;
  out.erode = info.caps.z / max(card.fade, 1e-3);
  out.yaw_cs = vec2f(card.cs, card.sn);
  // Fade the warp out before the tier switch so near and far agree there.
  out.warp_k = info.shade.w * (1.0 - smoothstep(info.ids.w * 0.6, info.ids.w, card.dist));
  return out;
}

@fragment
fn fs_near(in: NearOut) -> @location(0) vec4f {
  let t0 = tiles.v[in.tile * 4u];
  let t1 = tiles.v[in.tile * 4u + 1u];
  let t2 = tiles.v[in.tile * 4u + 2u];
  let t3 = tiles.v[in.tile * 4u + 3u];

  let w = in.local - in.cam_local; // unnormalized eye ray, plant-local
  let uv0 = tile_uv_of(t0, t1, t2, t3, in.cam_local, w);

  // Gradients come from the UNWARPED coordinate: it is smooth across the quad,
  // so mip selection never explodes at a parallax discontinuity. The depth
  // probe itself rides on that smooth coordinate, so it can use the cheap
  // implicit-derivative path — which also has to happen before any discard.
  let gx = dpdx(uv0) * INV_GRID;
  let gy = dpdy(uv0) * INV_GRID;
  let q = textureSample(atlas_geom, geom_sampler, atlas_uv(in.tile, uv0)).a;

  // The quad is the conservative AABB projection; trim it to the tile's tight
  // silhouette box before spending another tap.
  if (any(uv0 < vec2f(-0.02)) || any(uv0 > vec2f(1.02))) {
    discard;
  }

  // Slide along the eye ray to the surface the depth probe reported.
  let wb = min(dot(w, t2.xyz), -1e-6);
  let shift = q * t3.z / (-wb) * in.warp_k;
  let inv_box = 0.5 / vec2f(t3.x, t3.y);
  let duv_raw = vec2f(dot(w, t0.xyz), -dot(w, t1.xyz)) * (shift * inv_box);
  // A single offset step is a one-iteration fixed point: it is accurate while
  // the shift is small and diverges into liquid smears once it is large (deep
  // texels seen well off the baked azimuth, or a fragment at the edge of a
  // quad the camera is nearly touching). Saturate the offset MAGNITUDE
  // smoothly at WARP_LIMIT of the tile instead of letting it run: the front of
  // the clump then parallaxes exactly and the back merely lags, which the eye
  // reads as depth rather than as melting.
  let duv = duv_raw * inverseSqrt(1.0 + dot(duv_raw, duv_raw) * INV_WARP_LIMIT2);
  let uv1 = uv0 + duv;

  let alb = textureSampleGrad(atlas_albedo, albedo_sampler, atlas_uv(in.tile, uv1), gx, gy);
  if (alb.a < in.erode) {
    discard;
  }
  let g = textureSampleGrad(atlas_geom, geom_sampler, atlas_uv(in.tile, uv1), gx, gy);
  return shade(alb, g, in.yaw_cs, in.world);
}

// ---------------------------------------------------------------------------
// Far tier — tile UV resolved at the quad vertices, no warp.
// ---------------------------------------------------------------------------

struct FarOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f,
  @location(3) @interpolate(flat) erode: f32,
  @location(4) @interpolate(flat) tile: u32,
}

@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FarOut {
  let card = build_card(vi, ii);
  let local = rot_y(card.world_rest - card.base, card.cs, -card.sn) * card.inv_scale;
  let t = card.tile * 4u;
  var out: FarOut;
  out.pos = clip_pos(card);
  out.uv = tile_uv_of(tiles.v[t], tiles.v[t + 1u], tiles.v[t + 2u], tiles.v[t + 3u], card.cam_local, local - card.cam_local);
  out.world = card.world;
  out.yaw_cs = vec2f(card.cs, card.sn);
  out.erode = info.caps.z / max(card.fade, 1e-3);
  out.tile = card.tile;
  return out;
}

@fragment
fn fs_far(in: FarOut) -> @location(0) vec4f {
  if (any(in.uv < vec2f(-0.02)) || any(in.uv > vec2f(1.02))) {
    discard;
  }
  let uv = atlas_uv(in.tile, in.uv);
  let alb = textureSample(atlas_albedo, albedo_sampler, uv);
  if (alb.a < in.erode) {
    discard;
  }
  let g = textureSample(atlas_geom, geom_sampler, uv);
  return shade(alb, g, in.yaw_cs, in.world);
}

// ---------------------------------------------------------------------------
// Shared shading — the shared lighting model with baked visibility folded in.
// ---------------------------------------------------------------------------

// Octahedral decode, y-primary — matches oct_encode in bake_resolve.wgsl.
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

fn shade(alb: vec4f, g: vec4f, yaw_cs: vec2f, world: vec3f) -> vec4f {
  let n_local = oct_decode(g.rg * 2.0 - 1.0);
  let n = rot_y(n_local, yaw_cs.x, yaw_cs.y);

  // The bake occludes each plant with ONE ring of same-species neighbours at
  // the mesh's own tile spacing (and already folds in a height ramp for the
  // canopy layers overhead). A real stand is several times denser and mixes
  // species, and occlusion compounds multiplicatively with canopy layers — so
  // the baked openness is raised to a canopy-depth exponent here.
  let ao = pow(g.b, info.shade.z);

  let occl = info.shade.x;
  // Floors stand in for the light that reaches a shaded blade after bouncing
  // around inside the canopy. Without them fully-occluded texels go black,
  // which reads as holes rather than depth.
  let sky_vis = mix(1.0, 0.10 + 0.90 * ao, occl);
  // Ambient-aperture visibility: the open set around a texel is a cone about
  // the stored (bent) normal whose half-angle follows from the openness,
  // cos(aperture) = 1 - ao. The sun is visible when it lies inside that cone,
  // with a soft rim — so self-shadowing MOVES when the sun moves.
  let cone = smoothstep(-0.25, 0.15, dot(n, frame.sun_dir) + ao - 1.0);
  let sun_vis = mix(1.0, 0.07 + 0.93 * cone, occl);

  // The shared lighting model with per-texel visibility folded into its two
  // terms — identical to light_surface() when both visibilities are 1.
  let ndl = dot(n, frame.sun_dir) * 0.5 + 0.5;
  let sun = frame.sun_color * (ndl * ndl) * sun_vis;
  let hemi = mix(0.6, 1.0, n.y * 0.5 + 0.5) * sky_vis;
  var color = alb.rgb * (sun + frame.ambient * hemi);
  // Thin-blade transmission: open texels lit from behind glow warmly.
  let back = clamp(-dot(n, frame.sun_dir), 0.0, 1.0);
  color += alb.rgb * frame.sun_color * (info.shade.y * back * ao * ao);

  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, world), 1.0);
}
