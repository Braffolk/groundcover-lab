#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/hash.wgsl"
#include "./entry_info.wgsl"

// Two LODs over one capture box.
//
// vs_cloud / fs_cloud (near): N_CARDS vertical planes FIXED IN THE PLANT'S
//   OWN FRAME, each textured with the wedge of source geometry that lies in
//   it, plus a horizontal top card. The planes do not face the camera, so the
//   silhouette is the union of three differently-foreshortened images: it
//   genuinely changes shape with view direction, near blades slide across far
//   ones as the camera moves, and the planes depth-test against each other so
//   the crossing points read as real self-occlusion.
//
// vs_far / fs_far (far): one camera-facing card from the 8-azimuth impostor
//   sheet plus the top card — i.e. exactly a billboard, at billboard cost.
//
// Both: hard alpha test with depth write (no blending, no dithering, no
// frag_depth), 2 texture taps per fragment, card bottoms sunk into the ground
// so the terrain depth buffer clips the cut edge away. The interpolant set is
// kept to 11 components in 4 slots — a card is mostly alpha-test rejects, so
// per-fragment attribute traffic is a first-order cost here.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var near_albedo: texture_2d_array<f32>;
@group(1) @binding(2) var near_normal: texture_2d_array<f32>;
@group(1) @binding(3) var far_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var far_normal: texture_2d_array<f32>;
@group(1) @binding(5) var card_sampler: sampler;
@group(2) @binding(0) var<storage, read> insts: array<PlantInst>;

const TOP_LAYER_NEAR: u32 = 3u;              // = N_CARDS
const TOP_LAYER_FAR: u32 = 12u;              // = NEAR_LAYERS + N_AZI

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv_shade: vec3f,                 // tile uv, grounding gradient
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) card: vec4f,  // cos yaw, sin yaw, erode, tint
  @location(3) @interpolate(flat) layer: u32,
}

struct Plant {
  base: vec3f,
  axis: vec3f,
  yaw: f32,
  scale: f32,
  h0: f32,
  h1: f32,
  r: f32,
  sway: vec3f,
  to_cam: vec3f,
  fade: f32,
  tint: f32,
}

fn load_plant(ii: u32) -> Plant {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  var p: Plant;
  p.yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  p.scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  p.base = inst.pos;
  p.r = info.card_r * p.scale;
  p.h0 = info.card_y0 * p.scale;
  p.h1 = info.card_y1 * p.scale;
  // The baked capture is centered on the clump, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  p.axis = p.base + rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * p.scale, p.yaw);
  p.to_cam = frame.camera_pos - p.axis;

  let entry = stand_table[u32(info.entry_index)];
  p.sway = wind_sway(p.base, frame.time, entry.sway, phase);

  // Camera-inside fade: 3D distance to the plant's core segment.
  let seg_y = clamp(frame.camera_pos.y, p.base.y + p.h0, p.base.y + p.h1);
  let d_core = length(vec3f(p.to_cam.x, frame.camera_pos.y - seg_y, p.to_cam.z));
  let near_fade = smoothstep(p.r * 0.35, p.r * 1.05, d_core);
  // Region edge: erode to nothing exactly at region_r (cull matches).
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, length(p.to_cam.xz));
  p.fade = near_fade * edge_fade;
  p.tint = hash_f32(pcg(bits)) * 2.0 - 1.0;
  return p;
}

/// The top card only makes sense from well above — at flatter views it reads
/// as a pale floating cutout, so erode it away unless it is seen steeply.
fn top_fade(p: Plant) -> f32 {
  let dy = frame.camera_pos.y - (p.base.y + mix(p.h0, p.h1, info.top_frac));
  let elev = abs(dy) / max(length(vec3f(p.to_cam.x, dy, p.to_cam.z)), 1e-4);
  return smoothstep(0.35, 0.6, elev);
}

fn quad_corner(ci: u32, b: vec4f) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let c = corners[ci];
  return vec2f(mix(b.x, b.z, c.x), mix(b.y, b.w, c.y));
}

/// World position of a vertical card corner. `dir_ws` is the card's in-plane
/// horizontal axis; the bottom edge is pushed `sink` below the capture box so
/// the terrain (already in the depth buffer) clips the cut line away.
fn side_vertex(p: Plant, dir_ws: vec3f, t: vec2f) -> vec3f {
  let hf = 1.0 - t.y;
  var w = p.axis + dir_ws * ((t.x * 2.0 - 1.0) * p.r);
  w.y = p.base.y + mix(p.h0, p.h1, hf) - info.sink * p.scale * (1.0 - hf);
  return w + p.sway * hf;
}

fn top_vertex(p: Plant, t: vec2f) -> vec3f {
  let off = vec3f(
    info.card_cx + (t.x * 2.0 - 1.0) * info.card_r,
    0.0,
    info.card_cz + (t.y * 2.0 - 1.0) * info.card_r,
  ) * p.scale;
  var w = p.base + rot_y(off, p.yaw);
  w.y += mix(p.h0, p.h1, info.top_frac);
  return w + p.sway * info.top_frac;
}

/// A card whose fade dropped below the alpha reference has erode > 1, so EVERY
/// one of its fragments discards — emit it behind the near plane instead of
/// rasterizing a guaranteed-empty quad. Those are precisely the cards that
/// would cover the most screen area (fully faded top cards at low elevations,
/// cards the camera stands inside, the region rim).
fn finish(p: Plant, world: vec3f, uv: vec2f, layer: u32, fade: f32, shade: f32) -> VOut {
  var out: VOut;
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv_shade = vec3f(uv, shade);
  out.world = world;
  out.card = vec4f(cos(p.yaw), sin(p.yaw), info.alpha_ref / max(fade, 1e-3), p.tint);
  out.layer = layer;
  return out;
}

// --- shared shading ---------------------------------------------------------

// Octahedral decode, y-primary — same convention the bake stored.
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

fn shade_card(in: VOut, alb: vec4f, enc: vec2f) -> vec4f {
  let n_mesh = oct_decode_card(enc * 2.0 - 1.0);
  // Rotate the mesh-frame normal by the plant's yaw.
  var n = vec3f(
    in.card.x * n_mesh.x + in.card.y * n_mesh.z,
    n_mesh.y,
    -in.card.y * n_mesh.x + in.card.x * n_mesh.z,
  );
  // Blades are thin two-sided sheets and the bake stored the normal facing the
  // bake camera, so on a vertical card seen from its far side, flip it toward
  // the viewer. (Top cards are only ever seen from above — leave them alone.)
  let two_sided = in.layer != TOP_LAYER_NEAR && in.layer != TOP_LAYER_FAR;
  if (two_sided && dot(n, frame.camera_pos - in.world) < 0.0) {
    n = -n;
  }
  // The grounding gradient and the per-plant tint belong to the albedo so the
  // lighting debug view stays exact.
  let j = info.tint_jitter * in.card.w;
  let tinted = alb.rgb * vec3f(1.0 + j, 1.0 + j * 0.45, 1.0 - j * 0.25);
  let albedo = clamp(tinted * in.uv_shade.z, vec3f(0.0), vec3f(1.0));
  // Shared model + a little sun transmission (grass is translucent). Still
  // multiplicative in albedo, so DEBUG_LIGHTING divides out exactly.
  let back = max(0.0, -dot(n, frame.sun_dir));
  var color = light_surface(albedo, n, in.world) + albedo * frame.sun_color * (info.translucency * back * back);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the baked alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin (everything below `erode` is gone).
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}

// --- near LOD: the card cloud ----------------------------------------------

@vertex
fn vs_cloud(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = load_plant(ii);
  let ci = vi % 6u;

  if (vi >= CLOUD_VERTS) {
    let t = quad_corner(ci, info.tile_bounds[TOP_LAYER_NEAR]);
    return finish(p, top_vertex(p, t), t, TOP_LAYER_NEAR, p.fade * top_fade(p), 1.0 - info.bottom_shade * 0.3);
  }

  // One quad per (card, height band): same mapping as one tall quad — both the
  // height lerp and the sway are linear in height — but only as wide as the
  // ink actually reaches at that height, which drops ~35% of the empty area
  // a card of 4-10% coverage would otherwise rasterize and reject.
  let card = vi / (BANDS * 6u);
  let band = (vi / 6u) % BANDS;
  let i = card * BANDS + band;
  let e = info.card_bands[i >> 1u];
  let u = select(e.xy, e.zw, (i & 1u) == 1u);
  let v0 = f32(band) / f32(BANDS);
  let t = quad_corner(ci, vec4f(u.x, v0, u.y, v0 + 1.0 / f32(BANDS)));

  // Card k's plane contains the horizontal direction a_k = k*pi/N_CARDS in the
  // plant's own frame — the same axis the wedge was baked onto.
  let a = f32(card) * (3.14159265 / f32(N_CARDS));
  let dir_ws = rot_y(vec3f(cos(a), 0.0, -sin(a)), p.yaw);
  let shade = mix(1.0 - info.bottom_shade, 1.0, 1.0 - t.y);
  return finish(p, side_vertex(p, dir_ws, t), t, card, p.fade, shade);
}

@fragment
fn fs_cloud(in: VOut) -> @location(0) vec4f {
  let alb = textureSample(near_albedo, card_sampler, in.uv_shade.xy, in.layer);
  let enc = textureSample(near_normal, card_sampler, in.uv_shade.xy, in.layer).xy;
  if (alb.a < in.card.z) {
    discard;
  }
  return shade_card(in, alb, enc);
}

// --- far LOD: the impostor --------------------------------------------------

@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = load_plant(ii);
  let ci = vi % 6u;

  if (vi >= 6u) {
    let t = quad_corner(ci, info.tile_bounds[TOP_LAYER_FAR]);
    let fade = p.fade * top_fade(p);
    return finish(p, top_vertex(p, t), t, TOP_LAYER_FAR, fade, 1.0 - info.bottom_shade * 0.3);
  }

  // Nearest baked azimuth in the plant's local frame.
  let fwd_xz = p.to_cam.xz / max(length(p.to_cam.xz), 1e-4);
  let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -p.yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  let k = u32((i32(round(ang * (f32(N_AZI) / 6.2831853))) + i32(N_AZI)) % i32(N_AZI));
  let layer = NEAR_LAYERS + k;
  let t = quad_corner(ci, info.tile_bounds[layer]);
  let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
  let shade = mix(1.0 - info.bottom_shade, 1.0, 1.0 - t.y);
  return finish(p, side_vertex(p, right, t), t, layer, p.fade, shade);
}

@fragment
fn fs_far(in: VOut) -> @location(0) vec4f {
  let l = in.layer - NEAR_LAYERS;
  let alb = textureSample(far_albedo, card_sampler, in.uv_shade.xy, l);
  let enc = textureSample(far_normal, card_sampler, in.uv_shade.xy, l).xy;
  if (alb.a < in.card.z) {
    discard;
  }
  return shade_card(in, alb, enc);
}
