#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// Relief-lit cards. Geometry per plant is exactly the billboard baseline's: one
// camera-facing side quad + one horizontal canopy quad (12 vertices). What the
// fragments do with it is not:
//
//   Every atlas tile is a capture in a generic orthonormal frame (R, V, F) with
//   half-extents (ru, rv, rw), and its blue channel is the HEIGHTFIELD s — the
//   front surface's coordinate along F. A fragment therefore knows (a) where its
//   card point sits in that frame, from an interpolated affine varying, and
//   (b) its own eye ray. Intersecting ray with heightfield analytically in ONE
//   step (t = (z_card - z_surface)/e_F) moves the sample sideways in the tile by
//   exactly the parallax the real surface would show: with x_card = x*cos(b) the
//   solve reduces to x/cos(b) + z*tan(b), which is the EXACT reprojection for the
//   height it sampled — not a fudge factor. Result: interior parallax, a
//   silhouette that deforms with view direction instead of merely rotating, and
//   far less popping across the 45deg view bins, because both neighbours
//   reconstruct the SAME surface.
//
//   Lighting then carries the rest of the 3D read: per-texel baked normals,
//   baked canopy occlusion, one sun-direction tap of the canopy heightfield for
//   self-shadowing, and forward scatter for backlit blades.
//
// No marching, no frag_depth (early-z stays intact), hard alpha test, no dither.
//
// TWO structural rules, both learned from A/B measurements against the baseline
// on this exact scene:
//
//  1. LOD_TIER is an override constant, not a varying. The cull sorts plants
//     into distance rings and each ring is drawn by its own specialized
//     pipeline, so a far card's shader does not merely SKIP the relief and
//     shadow code — it does not contain it. Carrying all four tiers in one
//     shader cost ~1.5x on the whole pass even where every branch was skipped
//     (peak register allocation sets occupancy, not the path taken).
//  2. Varyings are four slots. The stages rebuild the tile frame from two
//     cos/sin pairs instead of passing nine axis floats; interpolant traffic on
//     ~2M vertices/frame measured as expensive as every texture tap combined.

/** 0 = relief + self-shadow, 1 = relief, 2 = flat lit, 3 = albedo only. */
override LOD_TIER: i32 = 0;
/** Newton refinement of the relief solve (tiers 0-1). */
override RELIEF_STEPS: i32 = 1;
/** Paint the tier instead of shading it. */
override SHOW_LOD: i32 = 0;

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b | phase 10b (bit 22 doubles as mirror)
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(3) var atlas_geom: texture_2d<f32>;
@group(1) @binding(4) var atlas_sampler: sampler;

const GRID: f32 = 3.0;
/** Atlas edge in texels — keep in sync with ATLAS in bake.ts (TILE * GRID). */
const ATLAS_PX: f32 = 1344.0;
const N_SIDE: f32 = 8.0;
const TAU: f32 = 6.2831853;

fn rot_y_cs(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn atlas_uv(tile_rc: vec2f, local: vec2f) -> vec2f {
  let clamped = clamp(local, vec2f(info.tile_inset), vec2f(1.0 - info.tile_inset));
  return (tile_rc + clamped) / GRID;
}

// Octahedral decode, y-primary — same convention the bake stored.
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

/**
 * The capture frame of a tile, rebuilt from the plant's mirrored view azimuth
 * phi = yaw + mirror*ak. Side tiles look along the horizon with world-down as
 * their v axis; the canopy tile looks straight down. Half-extents come from the
 * uniform, so nothing about the frame needs to travel between stages.
 */
struct TileFrame {
  r_ax: vec3f,
  v_ax: vec3f,
  f_ax: vec3f,
  radii: vec3f, // ru, rv, rw
}

fn tile_frame(cphi: f32, sphi: f32, mirror: f32, is_top: bool) -> TileFrame {
  var tf: TileFrame;
  tf.r_ax = vec3f(mirror * cphi, 0.0, -mirror * sphi);
  let half_h = (info.card_y1 - info.card_y0) * 0.5;
  if (is_top) {
    tf.v_ax = vec3f(sphi, 0.0, cphi);
    tf.f_ax = vec3f(0.0, 1.0, 0.0);
    tf.radii = vec3f(info.card_r, info.card_r, half_h);
  } else {
    tf.v_ax = vec3f(0.0, -1.0, 0.0);
    tf.f_ax = vec3f(sphi, 0.0, cphi);
    tf.radii = vec3f(info.card_r, half_h, info.card_r);
  }
  return tf;
}

struct VOut {
  @builtin(position) pos: vec4f,
  /** (tile uv of the card point, card depth along F) — affine, so exact. */
  @location(0) card: vec3f,
  @location(1) world: vec3f,
  /** (cos yaw, sin yaw, cos phi, sin phi) */
  @location(2) @interpolate(flat) cs: vec4f,
  /** (alpha reference, view index + 16*is_top + 32*mirrored, jitter, mip level) */
  @location(3) @interpolate(flat) pk: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi_all: u32, @builtin(instance_index) ii: u32) -> VOut {
  // firstVertex carries the distance ring (see cull.wgsl) — 12 verts per plant.
  // LOD_TIER *is* that ring, so the base is a compile-time selection.
  let vi = vi_all % 12u;
  var ring_base = u32(info.bucket_base.x);
  if (LOD_TIER == 1) {
    ring_base = u32(info.bucket_base.y);
  } else if (LOD_TIER == 2) {
    ring_base = u32(info.bucket_base.z);
  } else if (LOD_TIER == 3) {
    ring_base = u32(info.bucket_base.w);
  }

  let inst = insts[ring_base + ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (TAU / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (TAU / 1024.0);
  // One hash bit reused as a chirality flip: mirroring the baked plant about its
  // own x=0 plane doubles the apparent variety for free and never moves,
  // rescales or respecies the instance.
  let mirrored = info.mirror > 0.5 && ((bits >> 22u) & 1u) == 1u;
  let m = select(1.0, -1.0, mirrored);
  let cy = cos(yaw);
  let sy = sin(yaw);

  let base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];
  let h0 = info.card_y0 * scale;
  let h1 = info.card_y1 * scale;
  let hgt = h1 - h0;
  let r = info.card_r * scale;
  let y_mid_u = (info.card_y0 + info.card_y1) * 0.5;

  let clump = rot_y_cs(vec3f(m * info.card_cx, 0.0, info.card_cz) * scale, cy, sy);
  let axis = base + clump;
  let to_cam = frame.camera_pos - axis;
  let sway = wind_sway(base, frame.time, entry.sway, phase);

  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];
  let is_top = vi >= 6u;

  // --- coverage fades -------------------------------------------------------
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(r * 0.35, r * 1.05, d_core);
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  var fade = near_fade * edge_fade;
  if (is_top) {
    // The canopy card is only honest seen from ABOVE (that is the view it was
    // baked from), so it fades in with SIGNED elevation, never from below.
    let dy = frame.camera_pos.y - (base.y + mix(h0, h1, info.top_plane));
    let elev = dy / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
    fade *= smoothstep(0.28, 0.55, elev) * info.top_card;
  }

  // --- pick the baked view; phi folds yaw, azimuth and mirroring together ----
  let fwd_xz = to_cam.xz / max(d_xz, 1e-4);
  var view_index = 8u;
  var phi = yaw;
  if (!is_top) {
    // Viewing azimuth in the plant's own (mirrored) frame.
    let dm_x = m * (cy * fwd_xz.x - sy * fwd_xz.y);
    let dm_z = sy * fwd_xz.x + cy * fwd_xz.y;
    let ang = atan2(dm_x, dm_z);
    view_index = u32((i32(round(ang * (N_SIDE / TAU))) + i32(N_SIDE)) % i32(N_SIDE));
    phi = yaw + m * f32(view_index) * (TAU / N_SIDE);
  }
  let cphi = cos(phi);
  let sphi = sin(phi);
  let tf = tile_frame(cphi, sphi, m, is_top);

  // --- card geometry --------------------------------------------------------
  var world_geo: vec3f;
  var q_pos: vec3f;
  if (is_top) {
    let c2 = vec2f(c.x, c.y * 2.0 - 1.0);
    let local = vec3f(m * (info.card_cx + c2.x * info.card_r), 0.0, info.card_cz + c2.y * info.card_r);
    q_pos = rot_y_cs(local, cy, sy) + vec3f(0.0, mix(info.card_y0, info.card_y1, info.top_plane), 0.0);
    world_geo = base + q_pos * scale + sway * info.top_plane;
  } else {
    let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
    var p = axis + right * (c.x * r);
    p.y = base.y + mix(h0, h1, c.y);
    q_pos = (p - base) / scale;
    if (c.y < 0.5) {
      // Ground contact: bury the bottom edge below the terrain so a card on a
      // slope never floats or hovers over a dip. The atlas coords keep the
      // un-buried value, so the skirt just repeats the base row. The exact
      // per-corner terrain height needs four heightmap loads, which measured as
      // ~13% of the whole pass when every card paid it — so only the two near
      // tiers do, and the rest use the flat skirt (a plant 30 m out cannot show
      // a base that hovers by a centimetre).
      var y_ground = p.y;
      if (LOD_TIER <= 1) {
        y_ground = min(p.y, terrain_height(p.xz));
      }
      p.y = y_ground - info.contact_skirt * hgt;
    }
    world_geo = p + sway * c.y;
  }
  // Offset from the capture origin, at unit scale, in world orientation:
  // dot(qq, *_ax) == dot(p_mesh - Ov, *_mesh) by construction.
  let qq = q_pos - rot_y_cs(vec3f(m * info.card_cx, y_mid_u, info.card_cz), cy, sy);
  let kco = vec3f(dot(qq, tf.r_ax), dot(qq, tf.v_ax), dot(qq, tf.f_ax));

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so every
  // one of its fragments would discard — emit it behind the near plane instead
  // of rasterizing a guaranteed-empty quad (faded canopy cards at low elevation,
  // cards the camera stands inside, the region rim).
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world_geo, 1.0);
  }
  out.card = vec3f(vec2f(0.5) + kco.xy * (0.5 / tf.radii.xy), kco.z);
  out.world = world_geo;
  out.cs = vec4f(cy, sy, cphi, sphi);
  // Mip level, chosen ONCE per card instead of per fragment. A card's uv-to-pixel
  // ratio is a per-plant constant (its width in metres, its distance, and the
  // projection), so the usual dpdx/dpdy + log2 in the fragment shader is pure
  // waste here — and worse, taking derivatives of the WARPED uv makes the sampler
  // drop mip levels wherever the foliage heightfield jumps. The only view-
  // dependent term is the canopy card's foreshortening, folded in below.
  let px_scale = frame.proj[1].y * frame.viewport.y * 0.5 / max(d_core, 0.05);
  var px_u = 2.0 * tf.radii.x * scale * px_scale;
  var px_v = 2.0 * tf.radii.y * scale * px_scale;
  if (is_top) {
    let dy2 = frame.camera_pos.y - (base.y + mix(h0, h1, info.top_plane));
    px_v *= max(0.25, abs(dy2) / max(length(vec3f(to_cam.x, dy2, to_cam.z)), 1e-4));
  }
  out.pk = vec4f(
    info.alpha_ref / max(fade, 1e-3),
    f32(view_index) + select(0.0, 16.0, is_top) + select(0.0, 32.0, mirrored),
    fract(f32(bits & 1023u) * 0.61803399),
    max(0.0, log2(max(ATLAS_PX / GRID / max(min(px_u, px_v), 1.0), 1.0))),
  );
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let flags = u32(in.pk.y + 0.5);
  let vidx = flags & 15u;
  let is_top = (flags & 16u) != 0u;
  let m = select(1.0, -1.0, (flags & 32u) != 0u);
  let tile_rc = vec2f(f32(vidx % 3u), f32(vidx / 3u));
  let uv0 = in.card.xy;

  var alb: vec4f;
  var n: vec3f;
  var occ = 1.0;
  var glow = 0.0;

  if (LOD_TIER <= 1) {
    let e = normalize(frame.camera_pos - in.world);
    let tf = tile_frame(in.cs.z, in.cs.w, m, is_top);
    let lod = in.pk.w;
    // THE tap that buys the 3D: the geometry plane at the card point. It carries
    // the height that drives the warp AND the normal/occlusion used to shade the
    // warped colour. Reusing one fetch for both keeps the relief path at
    // billboard tap count; the normal is then sampled a few millimetres from the
    // colour it shades, which is invisible on a field of blade normals but would
    // matter on a smooth surface.
    var geo = textureSampleLevel(atlas_geom, atlas_sampler, atlas_uv(tile_rc, uv0), lod);

    // --- ray vs heightfield, one analytic step -----------------------------
    let inv = 0.5 / tf.radii.xy;
    let ez = dot(e, tf.f_ax);
    let slide = vec2f(dot(e, tf.r_ax), dot(e, tf.v_ax)) * inv; // uv per unit t
    let t_max = info.relief_clamp / max(length(slide), 1e-5);
    let ez_g = select(ez, sign(ez + 1e-6) * 0.08, abs(ez) < 0.08);
    // s == 0 is the bake's "no data" sentinel: no surface there, no parallax.
    let live0 = smoothstep(0.0, 0.03, geo.b);
    var t = clamp((in.card.z - (2.0 * geo.b - 1.0) * tf.radii.z) * (info.relief * live0) / ez_g, -t_max, t_max);
    if (RELIEF_STEPS > 1) {
      // Optional Newton refinement: re-probe at the offset and re-solve. Buys
      // sharper stems on the closest plants for one more tap.
      let g1 = textureSampleLevel(atlas_geom, atlas_sampler, atlas_uv(tile_rc, uv0 - slide * t), lod);
      let live1 = smoothstep(0.0, 0.03, g1.b);
      t = clamp(t + (in.card.z - t * ez - (2.0 * g1.b - 1.0) * tf.radii.z) * live1 / ez_g, -t_max, t_max);
      geo = select(geo, g1, live1 > 0.5);
    }
    let uvf = uv0 - slide * t;

    alb = textureSampleLevel(atlas_albedo, atlas_sampler, atlas_uv(tile_rc, uvf), lod);
    if (alb.a < in.pk.x) {
      discard;
    }
    var n_mesh = oct_decode(geo.rg * 2.0 - 1.0);
    n_mesh.x *= m;
    n = rot_y_cs(n_mesh, in.cs.x, in.cs.y);
    let ao = 1.0 - info.ao_strength * (1.0 - geo.a);
    occ = ao;
    // Forward scatter: thin backlit blades glow. Weighted by openness so it
    // never lights the shaded interior of the clump. Near-field only.
    let back = max(0.0, -dot(e, frame.sun_dir));
    glow = info.translucency * back * back * back * ao;

    // Deep inside the clump the sun term is already ~gone, so the shadow tap
    // cannot be seen: gate it on openness and ramp the strength through the
    // same window so the gate itself is never visible.
    if (LOD_TIER == 0 && info.self_shadow > 0.001 && geo.a > 0.30) {
      // --- canopy self-shadow: ONE tap of the top-view heightfield ---------
      // The hit's full 3D position is known (tile uv + sampled height), so walk
      // along the sun to the height the canopy lid sits at and ask the lid
      // whether it is still above us there.
      let hh = info.card_y1 - info.card_y0;
      let w_hit = (2.0 * geo.b - 1.0) * tf.radii.z;
      let x_hit = (uvf.x - 0.5) * 2.0 * tf.radii.x;
      let y_hit = (uvf.y - 0.5) * 2.0 * tf.radii.y;
      // Mesh-frame azimuth of this tile: ak = mirror * (phi - yaw).
      let ck = in.cs.z * in.cs.x + in.cs.w * in.cs.y;
      let sk = m * (in.cs.w * in.cs.x - in.cs.z * in.cs.y);
      var mxz = vec2f(x_hit * ck + w_hit * sk, -x_hit * sk + w_hit * ck);
      var my = (info.card_y0 + info.card_y1) * 0.5 - y_hit;
      if (is_top) {
        mxz = vec2f(x_hit, y_hit);
        my = (info.card_y0 + info.card_y1) * 0.5 + w_hit;
      }
      let sd = frame.sun_dir;
      let sun = vec3f(m * (in.cs.x * sd.x - in.cs.y * sd.z), sd.y, in.cs.y * sd.x + in.cs.x * sd.z);
      let lh = max(length(sun.xz), 1e-3);
      let step_m = info.shadow_step * info.card_r;
      let uv_top = vec2f(0.5) + (mxz + (sun.xz / lh) * step_m) * (0.5 / info.card_r);
      let s_top = textureSampleLevel(atlas_geom, atlas_sampler, atlas_uv(vec2f(2.0, 2.0), uv_top), 0.0).b;
      let y_can = select(info.card_y0, info.card_y0 + s_top * hh, s_top > 0.004);
      let strength = info.self_shadow * smoothstep(0.30, 0.42, geo.a);
      occ *= 1.0 - strength * smoothstep(0.0, 0.06 * hh, y_can - (my + (sun.y / lh) * step_m));
    }
  } else if (LOD_TIER == 2) {
    // Flat tier: no warp, no eye ray, no glow — but still per-texel normals and
    // baked occlusion, which is what keeps the mid field from going poster-flat.
    alb = textureSampleLevel(atlas_albedo, atlas_sampler, atlas_uv(tile_rc, uv0), in.pk.w);
    if (alb.a < in.pk.x) {
      discard;
    }
    let geo = textureSampleLevel(atlas_geom, atlas_sampler, atlas_uv(tile_rc, uv0), in.pk.w);
    var n_mesh = oct_decode(geo.rg * 2.0 - 1.0);
    n_mesh.x *= m;
    n = rot_y_cs(n_mesh, in.cs.x, in.cs.y);
    occ = 1.0 - info.ao_strength * (1.0 - geo.a);
  } else {
    // Farthest tier: ONE tap, no geometry plane at all. A canopy-ish normal
    // (baked normals average toward the capture axis, tilted skyward — and its
    // length is constant, so no normalize) plus the mean occlusion is
    // indistinguishable at this range.
    alb = textureSampleLevel(atlas_albedo, atlas_sampler, atlas_uv(tile_rc, uv0), in.pk.w);
    if (alb.a < in.pk.x) {
      discard;
    }
    n = vec3f(in.cs.w * 0.5, 1.0, in.cs.z * 0.5) * 0.894427;
    occ = 1.0 - info.ao_strength * 0.28;
  }

  // --- shading -------------------------------------------------------------
  // The shared light model with occlusion split the way it physically acts:
  // canopy occlusion is a SKY-visibility term, so it attenuates the hemisphere
  // ambient fully and the sun only partially (the sun's own visibility is the
  // self-shadow tap). Multiplying the whole light term by AO — the obvious
  // thing — turned the meadow into mud; this keeps sunlit blade tips crisp while
  // the clump interior still goes deep.
  // With occ = 1 this is exactly light_surface(), so groundcover and terrain sit
  // in the same light, and the whole expression stays multiplicative in albedo,
  // which is what makes debug=lighting exact.
  let j = in.pk.z - 0.5;
  let albedo = alb.rgb * vec3f(1.0 + j * 0.20, 1.0 + j * 0.06, 1.0 - j * 0.16);
  let ndl = dot(n, frame.sun_dir) * 0.5 + 0.5;
  let hemi = mix(0.6, 1.0, n.y * 0.5 + 0.5);
  let light = frame.sun_color * (ndl * ndl * (0.4 + 0.6 * occ)) + frame.ambient * (hemi * occ) + frame.sun_color * glow;
  var color = albedo * light;

  if (SHOW_LOD == 1) {
    var lod_col = vec3f(0.95, 0.30, 0.22);
    if (LOD_TIER == 1) {
      lod_col = vec3f(0.95, 0.78, 0.20);
    } else if (LOD_TIER == 2) {
      lod_col = vec3f(0.28, 0.72, 0.34);
    } else if (LOD_TIER == 3) {
      lod_col = vec3f(0.25, 0.42, 0.92);
    }
    color = lod_col * (0.35 + 0.65 * occ);
  }
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}
