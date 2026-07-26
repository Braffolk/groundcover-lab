#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/hash.wgsl"
#include "./tables.wgsl"

// The card cloud. A near plant is FOUR spatially separated sub-clump cards
// (the mesh's four quadrant parts, each with its own 8-azimuth + crown
// impostor) plus their crown cards. Because the four cards sit at four
// different world positions, they parallax against each other, occlude each
// other through the depth buffer, and their union silhouette changes with the
// view direction instead of merely rotating. Beyond `near_r` the cloud
// collapses to the single whole-plant card (part 4) — the far field costs
// exactly one billboard.
//
// Per fragment: ONE albedo tap decides the hard alpha test; only survivors pay
// the normal tap and the single canopy-cache tap. No loops, no blending, no
// frag_depth — depth is written by real geometry so early-z stays alive.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<uniform> atlas_info: AtlasInfo;
@group(1) @binding(2) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(3) var card_albedo: texture_2d<f32>;
@group(1) @binding(4) var card_normal: texture_2d<f32>;
@group(1) @binding(5) var card_sampler: sampler;
@group(1) @binding(6) var canopy_tex: texture_3d<f32>;
@group(1) @binding(7) var canopy_sampler: sampler;
@group(1) @binding(8) var<uniform> canopy: CanopyInfo;

const UV_INSET: f32 = 0.0078125; // 2 texels of a 256 tile

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn tile_uv(tile: u32, local: vec2f) -> vec2f {
  let col = f32(tile % 7u);
  let row = f32(tile / 7u);
  let clamped = clamp(local, vec2f(UV_INSET), vec2f(1.0 - UV_INSET));
  return (vec2f(col, row) + clamped) / ATLAS_GRID;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) @interpolate(flat) tint: vec3f,   // per-plant albedo variation
  @location(5) shade: f32,                       // grounding gradient
  @location(6) @interpolate(flat) base_y: f32,   // plant base (canopy cache ref)
  @location(7) occ_y: f32,                       // height the canopy cache is read at
}

/** Unit-square corners of a quad as a triangle list. */
fn quad_corner(vi: u32) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  return corners[vi];
}

struct Plant {
  base: vec3f,
  yaw: f32,
  scale: f32,
  phase: f32,
  sway: vec3f,
  top_y: f32,   // whole-plant top, world metres above base
  fade: f32,    // region-edge x camera-inside coverage fade
  tint: vec3f,
}

fn read_plant(ii: u32) -> Plant {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  var p: Plant;
  p.base = inst.pos;
  p.yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  p.scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  p.phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let entry = stand_table[u32(info.entry_index)];
  p.sway = wind_sway(p.base, frame.time, entry.sway, p.phase);
  p.top_y = info.whole_y1 * p.scale;

  // Camera-inside fade uses the WHOLE plant so all four cards erode together.
  let whole_r = atlas_info.part_r[4].x * p.scale;
  let to_cam = frame.camera_pos - p.base;
  let seg_y = clamp(frame.camera_pos.y, p.base.y, p.base.y + p.top_y);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(whole_r * 0.35, whole_r * 1.05, d_core);
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  p.fade = near_fade * edge_fade;

  // Per-plant albedo variation: brightness plus a small green/straw shift, so
  // a meadow reads as many individuals instead of one stamp repeated.
  let h0 = hash_f32(hash2(bits, 17u)) * 2.0 - 1.0;
  let h1 = hash_f32(hash2(bits, 29u)) * 2.0 - 1.0;
  let v = info.tint_var;
  p.tint = vec3f(1.0 + v * (h0 + 0.45 * h1), 1.0 + v * h0, 1.0 + v * (h0 - 0.55 * h1));
  return p;
}

struct Card {
  world: vec3f,
  uv: vec2f,
  hfrac: f32,       // height above the plant base / plant top (wind + grounding)
  occ_y: f32,       // world height the canopy cache is sampled at
  ok: bool,
}

/**
 * A crown card only makes sense seen from well above: at flatter views it reads
 * as a pale floating cutout among the side cards, so it is dropped below ~25deg
 * elevation (~38deg when seen from underneath, where it shows the wrong face).
 * (`topCards` off sets top_frac to 0 and drops them entirely.)
 */
fn crown_visible(p: Plant, part: u32) -> bool {
  if (info.top_frac <= 0.0) {
    return false;
  }
  let box = atlas_info.part_box[part];
  let center = p.base + rot_y(vec3f(box.x, 0.0, box.y) * p.scale, p.yaw);
  let dy = frame.camera_pos.y - (p.base.y + mix(box.z, box.w, info.top_frac) * p.scale);
  let to_cam = frame.camera_pos - center;
  let elev = abs(dy) / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
  // Seen from BELOW a crown card shows the wrong face (the straight-down
  // capture), so it has to be steeper still before it earns its place.
  return elev > select(0.62, 0.42, dy > 0.0);
}

/**
 * One corner of one card. `part` selects the baked sub-clump (4 = whole plant);
 * side cards pick the baked azimuth nearest their own view direction, so the
 * four cards of a cloud can legitimately disagree about which view they show.
 */
fn build_card(part: u32, is_top: bool, c: vec2f, p: Plant) -> Card {
  let box = atlas_info.part_box[part];
  let r = atlas_info.part_r[part].x * p.scale;
  let y0 = box.z * p.scale;
  let y1 = box.w * p.scale;
  let center_ws = p.base + rot_y(vec3f(box.x, 0.0, box.y) * p.scale, p.yaw);

  var out: Card;
  if (is_top) {
    let tile = part * N_VIEWS + N_SIDE;
    let tb = atlas_info.tiles[tile];
    out.ok = tb.y > tb.x && tb.w > tb.z;
    let a = mix(tb.x, tb.y, c.x);
    let b = mix(tb.z, tb.w, c.y);
    // Crown capture axes: U = +X, V = -Z (see bake.wgsl).
    let local = vec3f(box.x + a * atlas_info.part_r[part].x, 0.0, box.y - b * atlas_info.part_r[part].x);
    out.world = p.base + rot_y(local * p.scale, p.yaw);
    out.world.y = p.base.y + mix(y0, y1, info.top_frac);
    out.uv = tile_uv(tile, vec2f(a * 0.5 + 0.5, 0.5 - b * 0.5));
  } else {
    let to_cam = frame.camera_pos - center_ws;
    let fwd_xz = to_cam.xz / max(length(to_cam.xz), 1e-4);
    // Nearest baked azimuth in the plant's local frame.
    let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -p.yaw);
    let ang = atan2(local_dir.x, local_dir.z);
    let k = u32((i32(round(ang * (8.0 / 6.2831853))) + 8) % 8);
    let tile = part * N_VIEWS + k;
    let tb = atlas_info.tiles[tile];
    out.ok = tb.y > tb.x && tb.w > tb.z;
    let u = mix(tb.x, tb.y, c.x);
    let h = mix(tb.z, tb.w, c.y);
    let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
    out.world = center_ws + right * (u * r);
    out.world.y = p.base.y + mix(y0, y1, h);
    out.uv = tile_uv(tile, vec2f(u * 0.5 + 0.5, 1.0 - h));
  }
  out.hfrac = clamp((out.world.y - p.base.y) / max(p.top_y, 1e-3), 0.0, 1.4);
  out.world += p.sway * out.hfrac;
  // A crown card stands in for the TOP surface of its part: what you see from
  // above is the topmost foliage, so it must be lit like the canopy top, not
  // like the mid-canopy height the quad happens to sit at.
  out.occ_y = select(out.world.y, p.base.y + 0.92 * p.top_y, is_top);
  return out;
}

fn finish(p: Plant, card: Card, is_top: bool, visible: bool) -> VOut {
  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards — emit it behind the near plane rather
  // than rasterizing a guaranteed-empty quad (that set is exactly the
  // screen-filling one: crown cards at eye level, cards the camera stands
  // inside, the region rim).
  if (!visible || !card.ok || p.fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(card.world, 1.0);
  }
  out.uv = card.uv;
  out.world = card.world;
  out.yaw_cs = vec2f(cos(p.yaw), sin(p.yaw));
  out.erode = info.alpha_ref / max(p.fade, 1e-3);
  out.tint = p.tint;
  out.base_y = p.base.y;
  out.occ_y = card.occ_y;
  // Fake grounding: card bottoms darken toward the soil. Crown cards sit high.
  let g = select(card.hfrac, 1.0, is_top);
  out.shade = mix(1.0 - info.bottom_shade, 1.0, clamp(g * 1.6, 0.0, 1.0));
  return out;
}

/**
 * NEAR LOD — 8 quads: four sub-clump side cards, then four crown cards.
 * The side cards are emitted NEAREST FIRST (quadrant depth order derived from
 * the view direction in plant space) so early-z can reject the hidden parts of
 * the cards behind them.
 */
@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = read_plant(ii);
  let quad = vi / 6u;
  let c = quad_corner(vi % 6u);

  let to_cam = frame.camera_pos - p.base;
  let local_cam = rot_y(to_cam, -p.yaw);
  // Part index bit 0 = +x half, bit 1 = +z half (see bake.ts partitionMesh).
  let near_p = select(0u, 1u, local_cam.x > 0.0) | select(0u, 2u, local_cam.z > 0.0);
  let x_major = abs(local_cam.x) > abs(local_cam.z);
  var part = quad;
  if (quad < 4u) {
    // depth order: nearest, the cheaper of the two neighbours, the other, opposite
    var order = array<u32, 4>(near_p, near_p ^ 1u, near_p ^ 2u, near_p ^ 3u);
    if (x_major) {
      order[1] = near_p ^ 2u;
      order[2] = near_p ^ 1u;
    }
    part = order[quad];
  } else {
    part = quad - 4u;
  }
  let is_top = quad >= 4u;
  let card = build_card(part, is_top, c, p);
  var visible = true;
  if (is_top) {
    visible = crown_visible(p, part);
  }
  return finish(p, card, is_top, visible);
}

/** FAR LOD — the whole-plant card plus its crown card (2 quads, 12 verts). */
@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = read_plant(ii + u32(info.cap_near));
  let quad = vi / 6u;
  let c = quad_corner(vi % 6u);
  let is_top = quad == 1u;
  let card = build_card(4u, is_top, c, p);
  var visible = true;
  if (is_top) {
    visible = crown_visible(p, 4u);
  }
  return finish(p, card, is_top, visible);
}

// Octahedral decode, y-primary — the convention the bake stored.
fn oct_decode_card(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

/**
 * One trilinear tap of the canopy cache: (sun transmittance, sky visibility).
 * The cache is world-anchored with toroidal wrap, so the lookup is just the
 * world position scaled by the window span with a repeating sampler. Influence
 * fades to a neutral constant at the window rim, which is also where a freshly
 * scrolled slab may still be a frame or two stale.
 */
fn canopy_read(world: vec3f, occ_y: f32, base_y: f32) -> vec2f {
  let span = canopy.cell * CANOPY_NX;
  let u = world.x / span;
  let w = world.z / span;
  let v = clamp((occ_y - base_y) / canopy.y_top, 0.0, 1.0);
  let s = textureSampleLevel(canopy_tex, canopy_sampler, vec3f(u, v, w), 0.0);
  let d = max(abs(world.x - frame.camera_pos.x), abs(world.z - frame.camera_pos.z));
  let fade = 1.0 - smoothstep(span * 0.34, span * 0.47, d);
  return mix(vec2f(1.0, 0.9), s.yz, fade);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Uniform control flow: the coverage tap and the texel-space derivatives
  // happen before the alpha test, so rejected fragments cost ONE tap and the
  // survivors' later taps can use an explicit LOD.
  let alb = textureSample(card_albedo, card_sampler, in.uv);
  let px = in.uv * ATLAS_PX;
  let ddx = dpdx(px);
  let ddy = dpdy(px);
  let lod = max(0.0, 0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy))));
  if (alb.a < in.erode) {
    discard;
  }

  let enc = textureSampleLevel(card_normal, card_sampler, in.uv, max(lod - 1.0, 0.0)).xy;
  let n_mesh = oct_decode_card(enc * 2.0 - 1.0);
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );

  let albedo = alb.rgb * in.tint;
  // Canopy cache: sky visibility occludes everything (it IS how much canopy is
  // above this point), the sun term is floored so shadowed foliage keeps its
  // ambient. Both are lighting, not albedo, so `debug=albedo` stays the baked
  // capture and `debug=lighting` shows sun x ambient x canopy occlusion.
  let occ = canopy_read(in.world, in.occ_y, in.base_y);
  let sun_mul = mix(1.0, mix(0.45, 1.0, occ.x), info.sun_shadow);
  let sky_mul = mix(1.0, occ.y, info.ao_strength);
  var color = light_surface(albedo, n, in.world) * (in.shade * sun_mul * sky_mul);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}
