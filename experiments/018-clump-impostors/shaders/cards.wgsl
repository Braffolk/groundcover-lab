#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./info.wgsl"

// Clump-impostor cards.
//
// NEAR list (one instance = one plant, 12 * K_SUB vertices):
//   verts 0 .. 6*K_SUB-1   — one camera-facing card per SUB-CLUMP, standing at
//                            that sub-clump's real offset inside the plant, so
//                            the cards sit at genuinely different depths: they
//                            parallax against each other, occlude each other
//                            through the depth buffer, and interleave with the
//                            neighbouring plants' sub-clumps.
//   verts 6*K_SUB .. 12K-1 — one horizontal top card per sub-clump.
// FAR list (12 vertices): the merged whole-plant card + its top card, i.e.
// exactly a billboard — the distance collapse.
//
// Every quad spans the baked TIGHT rect (the alpha bounding box measured per
// tile at bake time), so no empty margin is ever rasterized. Hard alpha test
// with depth write; near/region fades erode coverage through the alpha
// reference instead of dithering or blending.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: ClumpDyn;
@group(1) @binding(1) var<uniform> atlas_info: ClumpAtlas;
@group(1) @binding(2) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(3) var card_albedo: texture_2d<f32>;
@group(1) @binding(4) var card_normal: texture_2d<f32>;
@group(1) @binding(5) var card_sampler: sampler;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) shade: f32,                       // grounding gradient
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 2047u) * (4.0 / 2047.0);
  let phase = f32((bits >> 21u) & 1023u) * (6.2831853 / 1024.0);
  let is_near = (bits >> 31u) == 1u;
  let base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];

  // Vertex layout: `n_side` side cards, then the same count of top cards.
  let n_side = select(1u, K_SUB, is_near);
  let card = vi / 6u;
  let corner_i = vi % 6u;
  let is_top = card >= n_side;
  let unit = select(MERGED, card % K_SUB, is_near);

  let box = atlas_info.unit_box[unit];      // cx, cz, rXZ, y0
  let y1_unit = atlas_info.unit_ext[unit].x;
  let r = box.z * scale;
  let h0 = box.w * scale;
  let h1 = y1_unit * scale;
  // The capture was centred on this unit's own bbox, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  let axis = base + rot_y(vec3f(box.x, 0.0, box.y) * scale, yaw);
  let to_cam = frame.camera_pos - axis;

  // --- pick the baked view ---------------------------------------------------
  let fwd_xz = to_cam.xz / max(length(to_cam.xz), 1e-4);
  let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  let k_side = u32((i32(round(ang * (N_SIDE / 6.2831853))) + i32(N_SIDE)) % i32(N_SIDE));
  let view = select(k_side, u32(N_SIDE), is_top);
  let tile = (unit * N_VIEWS + view) * 2u;
  let uv_rect = atlas_info.tiles[tile];     // u0, v0, du, dv (already tight)
  let loc = atlas_info.tiles[tile + 1u];    // lx0, lt0, lw, lh in card space

  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let q = corners[corner_i];
  // Card-space coordinates: s across (0 = -r), tt down (0 = the card's top).
  let s = loc.x + q.x * loc.z;
  let tt = loc.y + q.y * loc.w;

  // --- fades -----------------------------------------------------------------
  // Region edge is per plant; the camera-inside fade is per CARD, measured to
  // that card's own core. A sub-clump card stands up to ~0.25m off the plant
  // axis, so a per-plant test would happily leave a card 0.1m from the eye:
  // one enormous smeared quad across the screen. The erosion band is still
  // scaled by the whole PLANT's radius, so it only ever triggers for cards the
  // camera is practically touching.
  let plant_h1 = info.merged_y1 * scale;
  let h_ref = max(plant_h1, 1e-3);
  let to_plant = frame.camera_pos - base;
  let fade_r = info.merged_r * scale;
  let d_xz = length(to_plant.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);

  // Sub-clumps sway slightly out of step with each other — the strongest cue
  // that they are separate masses rather than one flat card.
  let spread = select(0.0, (f32(unit) - 0.5 * (f32(K_SUB) - 1.0)) * info.sway_spread, is_near);
  let sway = wind_sway(base, frame.time, entry.sway, phase + spread);

  var world: vec3f;
  var ly: f32;
  var d_card: f32;
  var fade = edge_fade;
  if (is_top) {
    ly = mix(h0, h1, info.top_frac);
    let off = vec3f(box.x + (2.0 * s - 1.0) * box.z, 0.0, box.y + (2.0 * tt - 1.0) * box.z) * scale;
    world = base + rot_y(off, yaw);
    world.y += ly;
    // The top card only makes sense from well above: at flatter views it reads
    // as a pale floating cutout among the side cards, and from BELOW it is the
    // classic "pale pancake overhead" artifact (the straight-down capture seen
    // from underneath). The elevation is therefore signed — negative dy erodes
    // the card away completely.
    let dy = frame.camera_pos.y - (base.y + ly);
    d_card = length(vec3f(to_cam.x, dy, to_cam.z));
    fade *= smoothstep(0.35, 0.6, dy / max(d_card, 1e-4));
  } else {
    ly = mix(h0, h1, 1.0 - tt);
    let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
    world = axis + right * (r * (2.0 * s - 1.0));
    world.y = base.y + ly;
    let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
    d_card = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  }
  // Side cards may come close — grass in your face is the point of the inside
  // view — but a horizontal top card that close is always a smeared disc, so
  // it erodes from twice the distance.
  let fade_lo = select(fade_r * 0.30, fade_r * 0.50, is_top);
  let fade_hi = select(fade_r * 0.80, fade_r * 1.30, is_top);
  fade *= smoothstep(fade_lo, fade_hi, d_card);
  let hf = clamp(ly / h_ref, 0.0, 1.0);
  world += sway * hf;

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards; an empty baked tile has a zero-area
  // tight rect. Emit both behind the near plane instead of rasterizing quads
  // that are guaranteed to produce nothing.
  let a_ref = info.alpha_ref * select(1.0, info.near_alpha_bias, is_near);
  if (fade < a_ref || loc.z <= 0.0) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = vec2f(uv_rect.x + q.x * uv_rect.z, uv_rect.y + q.y * uv_rect.w);
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = a_ref / max(fade, 1e-3);
  out.shade = mix(1.0 - info.bottom_shade, 1.0, hf);
  return out;
}

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

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let alb = textureSample(card_albedo, card_sampler, in.uv);
  let enc = textureSample(card_normal, card_sampler, in.uv).xy;
  if (alb.a < in.erode) {
    discard;
  }
  let n_mesh = oct_decode_card(enc * 2.0 - 1.0);
  // Rotate the mesh-frame normal by the plant's yaw.
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  // in.shade is a fake grounding occlusion, so it belongs to the light term:
  // the albedo view then shows the baked atlas colour exactly as captured and
  // the lighting view shows sun+ambient x grounding.
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
