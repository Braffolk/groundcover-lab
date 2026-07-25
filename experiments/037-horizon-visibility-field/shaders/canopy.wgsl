#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// THE RUNTIME. No plant geometry of any kind is rasterized. The only surfaces
// that exist are ground-conformal ISO-CLEARANCE SHELLS — one instanced draw of a
// polar grid whose tessellation is a shader constant, with nothing in it derived
// from the plant count — and every pixel's grass is the ANSWER to its own eye
// ray, read from a baked visibility table in a constant number of fetches.
// Nothing is marched, stepped or iterated along the eye ray.
//
// Per pixel:
//   1  the shell fragment is the point where the ray's clearance above the terrain
//      equals that shell's height (the rasterizer solving ray/offset-surface in
//      hardware) and carries the exact local ground height + normal;
//   2  the canopy-hull field (stamped from the stand's real scatter) says how tall
//      the canopy is there, which species entry owns it and its wind phase, and a
//      two-stage bracket hands the pixel to the one shell at the canopy's height;
//   3  one clamped closed-form walk lands the entry on the local canopy top;
//   4  the query is un-sheared by the exact inverse of the harness wind shear
//      (which is linear in height, hence exactly invertible);
//   5  1-4 table taps return where the canopy first blocks that ray, with its
//      normal, coverage, albedo and baked sky occlusion.

struct TableInfo {
  hmax: f32,       // world metres for normalized height 1 (height_scale * scale_max)
  sway: f32,       // stand entry wind response
  tile_world: f32, // tile period in metres (tile_norm * hmax)
  max_lod: f32,
  mean_rgb: vec3f, // the table's own coarsest colour — the far-field canopy
  mean_ao: f32,
}

struct CanopyParams {
  field_origin: vec2f,
  field_texel: f32,
  field_res: f32,
  lid_y: f32,
  region_radius: f32,
  alpha_ref: f32,
  lod_bias: f32,
  rings: f32,
  sectors: f32,
  r_min: f32,
  r_max: f32,
  tile_norm: f32,
  res_table: f32,
  sin_min: f32,
  n_phi: f32,
  n_theta: f32,
  taps: u32,
  flags: u32,
  n_levels: f32,
  hull_window: f32,
  shell_lo: f32,
  shell_hi: f32,
  warp_amp: f32,
  tab: array<TableInfo, 4>,
}

const FLAG_GROUND: u32 = 1u;
const FLAG_WIND: u32 = 2u;
const FLAG_REFINE: u32 = 4u;
/// Metres the entry solve may walk from a shell crossing before it is clamped.
const ENTRY_CLAMP: f32 = 9.0;

@group(1) @binding(0) var<uniform> cp: CanopyParams;
@group(1) @binding(1) var field_tex: texture_2d<f32>;
@group(1) @binding(2) var tab_samp: sampler;
@group(1) @binding(3) var ray_a0: texture_2d_array<f32>;
@group(1) @binding(4) var ray_a1: texture_2d_array<f32>;
@group(1) @binding(5) var ray_a2: texture_2d_array<f32>;
@group(1) @binding(6) var ray_a3: texture_2d_array<f32>;
@group(1) @binding(7) var ray_b0: texture_2d_array<f32>;
@group(1) @binding(8) var ray_b1: texture_2d_array<f32>;
@group(1) @binding(9) var ray_b2: texture_2d_array<f32>;
@group(1) @binding(10) var ray_b3: texture_2d_array<f32>;
// The field is clamped (a world region); the ray table repeats (a periodic tile).
@group(1) @binding(11) var field_samp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) ground_y: f32,
  @location(2) ground_n: vec3f,
  @location(3) @interpolate(flat) hull_h: f32,
  @location(4) @interpolate(flat) is_top: f32,
}

// ---------------------------------------------------------------------------
// THE CARRIER: a camera-centred polar grid with geometric ring spacing (screen-
// uniform triangles), instanced into a few ground-conformal ISO-CLEARANCE shells
// at fixed heights above the terrain. RINGS x SECTORS x LEVELS is a constant of
// the shader — the stand's plant count never enters it, and there is no per-plant
// primitive anywhere.
//
// Why shells: rasterizing them hands the fragment the exact point where the eye
// ray's clearance above the terrain equals that shell's height, i.e. a hardware
// solve of the ray/offset-surface crossing. Each fragment only answers when its
// own height brackets the LOCAL canopy top, so whichever shell sits at the
// canopy's height owns the pixel and its entry point is exact — including when
// the eye is level with the canopy (the grazing camera), where a single shell
// above the canopy would never be crossed at all.
// ---------------------------------------------------------------------------
@vertex
fn vs_lid(@builtin(vertex_index) vi: u32, @builtin(instance_index) level: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0),
    vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0),
  );
  let quad = vi / 6u;
  let c = corners[vi % 6u];
  let sectors = u32(cp.sectors);
  let ring = f32(quad / sectors) + c.y;
  let sect = f32(quad % sectors) + c.x;
  let r = cp.r_min * pow(cp.r_max / cp.r_min, ring / cp.rings);
  let ang = sect * 6.28318531 / cp.sectors;
  let xz = frame.camera_pos.xz + r * vec2f(cos(ang), sin(ang));
  let g = terrain_sample(xz);
  let ny = sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0));
  // Shells span [shell_lo, shell_hi], a range the CPU picks from the camera's
  // clearance: a shell above the eye can never be crossed, so when the camera is
  // clear of the canopy ONE shell does the whole job.
  let span = max(cp.n_levels - 1.0, 1.0);
  let hull_h = select(mix(cp.shell_lo, cp.shell_hi, f32(level) / span), cp.shell_hi, cp.n_levels < 1.5);
  var out: VOut;
  out.world = vec3f(xz.x, g.x + hull_h, xz.y);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  out.ground_y = g.x;
  out.ground_n = vec3f(g.y, ny, g.z);
  out.hull_h = hull_h;
  out.is_top = select(0.0, 1.0, f32(level) + 1.5 > cp.n_levels);
  return out;
}

fn oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var x = f.x;
  var z = f.y;
  let y = 1.0 - abs(f.x) - abs(f.y);
  if (y < 0.0) {
    x = (1.0 - abs(f.y)) * select(-1.0, 1.0, f.x >= 0.0);
    z = (1.0 - abs(f.x)) * select(-1.0, 1.0, f.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

fn tap_a(which: u32, uv: vec2f, layer: u32, lod: f32) -> vec4f {
  switch (which) {
    case 0u: { return textureSampleLevel(ray_a0, tab_samp, uv, layer, lod); }
    case 1u: { return textureSampleLevel(ray_a1, tab_samp, uv, layer, lod); }
    case 2u: { return textureSampleLevel(ray_a2, tab_samp, uv, layer, lod); }
    default: { return textureSampleLevel(ray_a3, tab_samp, uv, layer, lod); }
  }
}

fn tap_b(which: u32, uv: vec2f, layer: u32, lod: f32) -> vec4f {
  switch (which) {
    case 0u: { return textureSampleLevel(ray_b0, tab_samp, uv, layer, lod); }
    case 1u: { return textureSampleLevel(ray_b1, tab_samp, uv, layer, lod); }
    case 2u: { return textureSampleLevel(ray_b2, tab_samp, uv, layer, lod); }
    default: { return textureSampleLevel(ray_b3, tab_samp, uv, layer, lod); }
  }
}

/// Smooth two-channel value noise, used to warp the tile lookup. Tiling is the
/// known weakness of any baked ray table (the source article hides it with
/// texture bombing); warping the query with a continuous field breaks the
/// lattice without introducing a single seam, and costs no memory.
fn warp_corner(c: vec2i) -> vec2f {
  let h = hash3(bitcast<u32>(c.x), bitcast<u32>(c.y), 0x51ed37u);
  return vec2f(hash_f32(h), hash_f32(pcg(h))) * 2.0 - 1.0;
}

fn warp_octave(u: vec2f) -> vec2f {
  let b = floor(u);
  let f = u - b;
  let s = f * f * (3.0 - 2.0 * f);
  let c = vec2i(b);
  let lo = mix(warp_corner(c), warp_corner(c + vec2i(1, 0)), s.x);
  let hi = mix(warp_corner(c + vec2i(0, 1)), warp_corner(c + vec2i(1, 1)), s.x);
  return mix(lo, hi, s.y);
}

/// Two long octaves (16 m and 5 m). Long on purpose: at grazing angles the entry
/// point races along the ray, so a SHORT-scale warp would have a gradient
/// comparable to the tile lookup itself and create stationary points — smooth
/// stretched patches. Long octaves break the long-range lattice, which is exactly
/// what reads as tiling, while staying nearly rigid per pixel.
fn tile_warp(xz: vec2f) -> vec2f {
  return warp_octave(xz * 0.0625) + warp_octave(xz * 0.2 + vec2f(3.7, 1.3)) * 0.25;
}

fn field_uv(xz: vec2f) -> vec2f {
  return (xz - cp.field_origin) / (cp.field_texel * cp.field_res);
}

fn field_at(xz: vec2f) -> vec4f {
  return textureSampleLevel(field_tex, field_samp, field_uv(xz), 0.0);
}

struct FragOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

/// The whole technique, per pixel. `gy0`/`gn`/`anchor` describe the carrier's
/// local ground plane; `hull_h` is the shell's clearance above the terrain.
fn resolve(d: vec3f, gy0: f32, gn_in: vec3f, anchor: vec2f, hull_h: f32, is_top: f32) -> FragOut {
  var out: FragOut;
  out.color = vec4f(0.0);
  out.depth = 1.0;
  let ro = frame.camera_pos;
  // Only descending rays can enter the canopy through its top plane. Ascending
  // ones are the camera-inside-a-plant case, which fades out by contract.
  if (d.y > -0.004) {
    discard;
  }

  // --- the local ground plane, from the carrier's own interpolated attributes --
  let gn0 = normalize(gn_in);
  let k_slope = -gn0.xz / max(gn0.y, 1e-3);
  // Rate at which the ray's clearance above that plane falls, per metre travelled.
  let fall = -(d.y - dot(k_slope, d.xz));

  var uv_f = field_uv(anchor);
  if (uv_f.x < 0.0 || uv_f.x > 1.0 || uv_f.y < 0.0 || uv_f.y > 1.0) {
    discard;
  }
  var fld = field_at(anchor);
  // Stage 1 of the shell bracket — a wide, cheap reject on the canopy height at
  // the shell's own footprint. One field fetch, before any table work.
  let dh = fld.x - hull_h;
  if (dh < -cp.hull_window || (dh > cp.hull_window && is_top < 0.5)) {
    discard;
  }

  // Walk from this shell's crossing to the canopy-top crossing — CLAMPED. At
  // grazing angles the eye ray runs nearly parallel to the canopy top, so solving
  // that intersection exactly is ill-conditioned: it diverges by tens of metres
  // and paints the carrier's own quads across the screen as marbled blocks.
  // Bounding the walk and then snapping the entry vertically onto the top plane
  // keeps the query continuous everywhere, at the price of a sub-metre entry
  // offset in exactly the views where a metre along the ray cannot be seen.
  let shell_pt = vec3f(anchor.x, gy0 + hull_h, anchor.y);
  let t_shell = dot(shell_pt - ro, d);
  // The walk is also kept in FRONT of the eye. When the eye sits below the canopy
  // top (the inside-plant camera) the canopy-top crossing is behind it, and
  // discarding there erodes not just the near field but the whole view. Clamping
  // instead keeps the query answerable: the entry becomes a virtual point on the
  // canopy top just ahead, which reads as "grass a couple of metres in front" —
  // what being inside a canopy actually looks like.
  var dt = clamp((hull_h - fld.x) / max(fall, 1e-3), 0.4 - t_shell, ENTRY_CLAMP);
  var e_pt = shell_pt + d * dt;

  var ground_y = gy0 + dot(k_slope, e_pt.xz - anchor);
  var gn = gn0;
  if ((cp.flags & FLAG_REFINE) != 0u) {
    // Re-anchor on the true heightmap at the entry — one tap, closed form, no
    // search. Without it the answer depends on WHICH shell delivered the
    // fragment, and neighbouring shells band visibly.
    let g = terrain_sample(e_pt.xz);
    let ny = sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0));
    ground_y = g.x;
    gn = vec3f(g.y, ny, g.z);
  }
  let uv1 = field_uv(e_pt.xz);
  if (uv1.x < 0.0 || uv1.x > 1.0 || uv1.y < 0.0 || uv1.y > 1.0) {
    discard;
  }
  uv_f = uv1;
  fld = field_at(e_pt.xz);

  // Stage 2 — the narrow bracket, on the canopy height AT THE CONVERGED ENTRY.
  // Every shell that survived stage 1 has re-anchored to (nearly) the same entry
  // point, so they now agree on the canopy height there and exactly one of them
  // owns the pixel. Bracketing on the shell's own footprint alone leaves holes
  // wherever the canopy height changes fast along the ray, which at grazing
  // angles is everywhere (those were visible as bald patches).
  let post = cp.hull_window * 0.36;
  let dh2 = fld.x - hull_h;
  if (dh2 < -post || (dh2 > post && is_top < 0.5)) {
    discard;
  }

  let r_cam = length(e_pt.xz - ro.xz);
  if (r_cam > cp.region_radius) {
    discard;
  }
  let which = min(u32(textureLoad(field_tex, vec2u(clamp(uv_f, vec2f(0.0), vec2f(0.99999)) * cp.field_res), 0).w + 0.5), 3u);
  let info = cp.tab[which];
  let canopy_h = fld.x;
  // Bare ground straight out of the real scatter: no plant covers this column.
  if (canopy_h < 0.1 * info.hmax) {
    discard;
  }

  // Wind: the harness displaces a point by D * (height fraction) — a LINEAR
  // shear, so the ray query can be un-sheared exactly instead of re-baked.
  var d_world = vec3f(0.0);
  if ((cp.flags & FLAG_WIND) != 0u && info.sway > 0.0) {
    let phase = atan2(fld.y, fld.z);
    d_world = wind_sway(vec3f(e_pt.x, 0.0, e_pt.z), frame.time, info.sway, phase);
  }
  let ch = max(canopy_h, 0.05 * info.hmax);
  let h_eff = max(ch + d_world.y, 0.15 * ch);
  // Snap the entry exactly onto the (wind-sheared) canopy top plane at its own xz.
  e_pt.y = ground_y + h_eff;
  let ground_at_e = ground_y;

  // --- into table space ------------------------------------------------------
  // Anisotropic: horizontal by the species' nominal size (so the tile lattice is
  // world-stable), vertical by the LOCAL canopy height (a shorter plant is a
  // shorter canopy). Both are exact linear maps, so the ray transforms exactly.
  let inv_s = vec3f(1.0 / info.hmax, 1.0 / ch, 1.0 / info.hmax);
  let fwd_s = vec3f(info.hmax, ch, info.hmax);
  let d_tab = d * inv_s;
  let w_tab = d_world * inv_s;
  let inv_w = 1.0 / (1.0 + w_tab.y);
  let p_tab = vec3f(e_pt.x * inv_s.x, 1.0 + w_tab.y, e_pt.z * inv_s.z);
  // M^-1 (un-shear): p - w * p.y / (1 + w.y)
  let up = p_tab - w_tab * (p_tab.y * inv_w);
  let ud = normalize(d_tab - w_tab * (d_tab.y * inv_w));

  // --- direction bins --------------------------------------------------------
  var phi = atan2(ud.z, ud.x);
  if (phi < 0.0) {
    phi += 6.28318531;
  }
  let sin_t = clamp(-ud.y, cp.sin_min, 1.0);
  let fu = phi / 6.28318531 * cp.n_phi;
  let ft = (sin_t - cp.sin_min) / (1.0 - cp.sin_min) * (cp.n_theta - 1.0);
  let iu0 = u32(floor(fu)) % u32(cp.n_phi);
  let iu1 = (iu0 + 1u) % u32(cp.n_phi);
  let wu = fract(fu);
  let it0 = min(u32(floor(ft)), u32(cp.n_theta) - 1u);
  let it1 = min(it0 + 1u, u32(cp.n_theta) - 1u);
  let wt = clamp(ft - f32(it0), 0.0, 1.0);
  let nth = u32(cp.n_theta);

  // --- prefiltering: the ray answer IS the LOD ------------------------------
  // A pixel's footprint on the canopy-top plane is ANISOTROPIC: px wide across
  // the ray, px/sin(elevation) long along it (20x at grazing). Filtering to the
  // long axis turns the distance into flat mush; filtering to the short one
  // shimmers. So: mip to the short axis with the anisotropy capped, and take two
  // taps spread along the ray to cover the rest of the footprint honestly.
  let px_world = max(length(e_pt - ro), 0.2) * 2.0 / max(frame.proj[1][1] * frame.viewport.y, 1e-3);
  let texel_world = info.tile_world / cp.res_table;
  let long_world = px_world / max(sin_t, 0.02);
  // Near-horizontal rays: a pixel's footprint on the canopy top plane is metres
  // long and the honest answer is a low-frequency average, so converge the whole
  // query onto the coarsest baked level instead of letting a half-resolved one
  // read as marbling. Detail returns as soon as the ray has some elevation.
  let mush = 1.0 - smoothstep(0.07, 0.24, sin_t);
  let lod = clamp(
    mix(log2(max(max(px_world, long_world * 0.125) / texel_world, 1e-4)) + cp.lod_bias, info.max_lod, mush),
    0.0,
    info.max_lod,
  );
  let uv_t = fract(up.xz / cp.tile_norm + tile_warp(e_pt.xz) * cp.warp_amp);
  // Footprint spread along the ray, in tile units.
  let step_uv = normalize(ud.xz + vec2f(1e-6, 0.0)) * (0.3 * long_world / (info.hmax * cp.tile_norm));
  var a = tap_a(which, uv_t, iu0 * nth + it0, lod);
  if (cp.taps >= 2u) {
    a = 0.5 * (tap_a(which, uv_t - step_uv, iu0 * nth + it0, lod) + tap_a(which, uv_t + step_uv, iu0 * nth + it0, lod));
    if (cp.taps >= 4u) {
      let a1 = 0.5 * (tap_a(which, uv_t - step_uv, iu0 * nth + it1, lod) + tap_a(which, uv_t + step_uv, iu0 * nth + it1, lod));
      a = mix(a, a1, wt);
    }
  }

  // --- the answer -----------------------------------------------------------
  let cov = a.a;
  let dist_tab = 1.0 / max(a.r, 1.0 / 255.0) - 1.0;
  var erode = 1.0 - smoothstep(cp.region_radius * 0.86, cp.region_radius, r_cam);

  var hit_tab = up + ud * dist_tab;
  var is_ground = false;
  if (cov * erode < cp.alpha_ref) {
    // Not blocked by foliage: the ray reaches the ground, and its distance is
    // closed form in table space (entry at y = 1, ground at y = 0).
    let t_g = up.y / max(-ud.y, 1e-4);
    if ((cp.flags & FLAG_GROUND) == 0u || t_g > 5.0) {
      discard;
    }
    hit_tab = up + ud * t_g;
    is_ground = true;
  }

  // table space -> world: shear back, then undo the anisotropic scale.
  let sheared = hit_tab + w_tab * hit_tab.y;
  var world_hit = vec3f(sheared.x * fwd_s.x, ground_at_e + sheared.y * fwd_s.y, sheared.z * fwd_s.z);
  let along = dot(world_hit - ro, d);
  if (along < 0.05) {
    discard;
  }
  // Camera inside the canopy: the first blocker is behind the eye. Erode out.
  erode *= smoothstep(0.10, 0.55, along);
  if (!is_ground && cov * erode < cp.alpha_ref) {
    discard;
  }
  if (is_ground && erode < 0.35) {
    discard;
  }

  // --- shading --------------------------------------------------------------
  let shade = tap_b(which, uv_t, iu0 * nth + it0, lod);
  var n_tab = oct_decode(a.gb);
  // Normals map by the transpose of the world->table matrix.
  let n_shear = vec3f(n_tab.x, n_tab.y - dot(w_tab, n_tab) * inv_w, n_tab.z);
  var n_world = normalize(vec3f(n_shear.x * inv_s.x, n_shear.y * inv_s.y, n_shear.z * inv_s.z));
  var albedo = shade.rgb;
  var ao = shade.a;
  if (is_ground) {
    n_world = gn;
    // Ground under the canopy: the terrain's own colour, occluded by how much
    // canopy the real scatter put above this spot.
    albedo = vec3f(0.16, 0.22, 0.10);
    ao = mix(1.0, 0.3, clamp(canopy_h / info.hmax, 0.0, 1.0));
    // Bias toward the eye so this beats the base pass's un-occluded terrain.
    world_hit -= d * (0.02 + 0.004 * along);
  }
  // ...and shade that converged far field as a wall of blades facing the eye.
  if (mush > 0.0 && !is_ground) {
    albedo = mix(albedo, info.mean_rgb, mush * 0.85);
    ao = mix(ao, info.mean_ao, mush * 0.85);
    let facing = normalize(vec3f(-d.x, 0.45, -d.z));
    n_world = normalize(mix(n_world, facing, mush * 0.7));
  }
  albedo *= ao;
  var color = light_surface(albedo, n_world, world_hit);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, world_hit);
  }
  let clip = frame.view_proj * vec4f(world_hit, 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);
  // The coverage view shows the table's real answer for this ray, so it doubles
  // as the alpha-test margin (a ground hit is a legitimately uncovered ray).
  out.color = vec4f(debug_shade(color, albedo, n_world, cov * erode, world_hit), 1.0);
  return out;
}

@fragment
fn fs_hull(in: VOut) -> FragOut {
  let d = normalize(in.world - frame.camera_pos);
  return resolve(d, in.ground_y, in.ground_n, in.world.xz, in.hull_h, in.is_top);
}
