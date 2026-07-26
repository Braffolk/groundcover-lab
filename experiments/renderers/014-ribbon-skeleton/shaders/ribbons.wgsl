#include "src/wgsl/frame.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/debug.wgsl"

// Ribbon unroll. Each instance is one plant; each ribbon is a triangle strip
// whose entire shape (centreline, half-width, colour, occlusion, fluffiness)
// is four texels per sample of a 12 x 360 atlas. Nothing here is camera-facing:
// a ribbon's width axis is the AZIMUTHAL direction of its own centreline, so a
// tuft looks like a tuft from the side and like a rosette from above, and it
// never spins, shimmers or collapses edge-on with the camera.
//
// Continuous LOD: a ribbon is morphed toward its parent ribbon at the next
// coarser level; the morph reaches 1 exactly where the next bucket takes over
// with half as many (now coincident) ribbons. Width, not alpha, is the fade.
//
// CARPET entries (stand_table[i].carpet_div > 0) take a second path through the
// same unroll: their atlas rows are a radial fan of the tile square carrying the
// cushion's surface relief, so the ribbon frame is a real surface frame (radial
// slope x azimuthal slope from the neighbouring wedges) instead of a blade frame
// with a curl, and every vertex is conformed to the terrain by a vertical shear.
// A shear is the only rung of the fitting ladder that keeps a TILED species
// seamless: the footprint stays exactly on its grid step, and neighbouring tiles
// evaluate the same ground height at a shared edge, so nothing cracks.

struct LodInfo {
  // x: ribbons drawn, y: samples drawn, z: atlas row base of this level,
  // w: atlas row base of the parent level
  counts: vec4f,
  // x: rows per variant, y: atlas sample count, z: instance capacity,
  // w: global width scale
  cfg: vec4f,
  // x: blade curl (rad), y: flutter amount, z: show mode, w: stand entry index
  look: vec4f,
  // x: LOD bucket index (debug), y: canopy normal lift, z,w reserved
  extra: vec4f,
}

@group(1) @binding(0) var<uniform> lod_info: LodInfo;
@group(1) @binding(1) var<storage, read> instances: array<vec4f>;
@group(1) @binding(2) var atlas_geom: texture_2d<f32>;
@group(1) @binding(3) var atlas_shade: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) albedo: vec3f,
  @location(1) world: vec3f,
  @location(2) nrm: vec3f,
  // x: ambient occlusion, y: coverage, z: fluffiness
  @location(3) surf: vec3f,
}

fn rot2(v: vec2f, c: f32, s: f32) -> vec2f {
  return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

/// Which of the atlas' samples this draw's sample j maps to (round to nearest).
fn atlas_sample(j: u32, drawn: u32, stored: u32) -> u32 {
  if (drawn <= 1u) {
    return 0u;
  }
  return (j * (stored - 1u) + (drawn - 1u) / 2u) / (drawn - 1u);
}

fn tint_of(h: u32) -> vec3f {
  return vec3f(hash_f32(hash2(h, 11u)), hash_f32(hash2(h, 23u)), hash_f32(hash2(h, 37u)));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, -1.0, 1.0); // clipped: overflow instances draw nothing
  out.albedo = vec3f(0.0);
  out.world = vec3f(0.0);
  out.nrm = vec3f(0.0, 1.0, 0.0);
  out.surf = vec3f(1.0, 0.0, 0.0);
  if (ii >= u32(lod_info.cfg.z)) {
    return out;
  }

  let ribbons = u32(lod_info.counts.x);
  let drawn = u32(lod_info.counts.y);
  let stored = u32(lod_info.cfg.y);
  let rows_per_variant = u32(lod_info.cfg.x);

  let d0 = instances[ii * 2u];
  let d1 = instances[ii * 2u + 1u];
  let base = d0.xyz;
  let yaw = d0.w;
  let variant = floor(d1.z);
  let morph = d1.z - variant;
  let coverage = d1.w;

  let entry = stand_table[u32(lod_info.look.w)];
  let plant_h = entry.height_scale * d1.x;
  // A tiled mat, not a tuft: surface frame, terrain shear, no fade, no jitter.
  let is_carpet = entry.carpet_div > 0.0;

  let ribbon = vi / (2u * drawn);
  let rest = vi % (2u * drawn);
  let j = rest >> 1u;
  let side = f32(rest & 1u) * 2.0 - 1.0;

  let jp = select(j - 1u, 0u, j == 0u);
  let jn = select(j + 1u, drawn - 1u, j + 1u >= drawn);
  let s_at = atlas_sample(j, drawn, stored);
  let s_prev = atlas_sample(jp, drawn, stored);
  let s_next = atlas_sample(jn, drawn, stored);

  let row_a = u32(variant) * rows_per_variant + u32(lod_info.counts.z) + ribbon;
  var g = textureLoad(atlas_geom, vec2u(s_at, row_a), 0);
  var g_prev = textureLoad(atlas_geom, vec2u(s_prev, row_a), 0);
  var g_next = textureLoad(atlas_geom, vec2u(s_next, row_a), 0);
  var sh = textureLoad(atlas_shade, vec2u(s_at, row_a), 0);
  var ao = clamp(sh.w - 2.0 * floor(sh.w * 0.5), 0.0, 1.0);
  var fluff = floor(sh.w * 0.5) / 7.0;
  // Pre-morph height: the carpet's azimuthal slope is taken from this level's
  // rows on both sides of an edge, so it stays continuous mid-morph.
  let y_level = g.y;
  var albedo = sh.rgb;

  if (morph > 0.0) {
    let row_b = u32(variant) * rows_per_variant + u32(lod_info.counts.w) + (ribbon >> 1u);
    let gb = textureLoad(atlas_geom, vec2u(s_at, row_b), 0);
    let gb_prev = textureLoad(atlas_geom, vec2u(s_prev, row_b), 0);
    let gb_next = textureLoad(atlas_geom, vec2u(s_next, row_b), 0);
    let shb = textureLoad(atlas_shade, vec2u(s_at, row_b), 0);
    g = mix(g, gb, morph);
    g_prev = mix(g_prev, gb_prev, morph);
    g_next = mix(g_next, gb_next, morph);
    albedo = mix(albedo, shb.rgb, morph);
    ao = mix(ao, clamp(shb.w - 2.0 * floor(shb.w * 0.5), 0.0, 1.0), morph);
    fluff = mix(fluff, floor(shb.w * 0.5) / 7.0, morph);
  }

  // --- plant frame ----------------------------------------------------------
  let cy = cos(yaw);
  let sy = sin(yaw);
  let local = g.xyz;
  let hf = clamp(local.y, 0.0, 1.0);

  // Width axis: azimuthal about this ribbon's own centreline (never view-based).
  let r2 = dot(local.xz, local.xz);
  let nominal_a = (f32(ribbon) + 0.5) * 6.2831853 / max(f32(ribbons), 1.0);
  let radial = select(vec2f(cos(nominal_a), sin(nominal_a)), local.xz * inverseSqrt(max(r2, 1.0e-12)), r2 > 1.0e-6);
  let lat_local = vec2f(-radial.y, radial.x);
  let lat = rot2(lat_local, cy, sy);
  let lat3 = vec3f(lat.x, 0.0, lat.y);

  let cxz = rot2(local.xz, cy, sy) * plant_h;
  var world = base + vec3f(cxz.x, local.y * plant_h, cxz.y);

  // --- wind -----------------------------------------------------------------
  let sway = wind_sway(base, frame.time, entry.sway, d1.y);
  world += sway * hf * hf;
  // Per-ribbon flutter so blades of one plant do not move in lockstep.
  let flut = lod_info.look.y * entry.sway * plant_h * 0.03 * hf * hf
    * sin(frame.time * 4.7 + d1.y * 2.3 + f32(ribbon) * 1.87);
  world += lat3 * flut;

  let half_w = g.w * lod_info.cfg.w * coverage * plant_h;
  world += lat3 * (side * half_w);

  // --- frame: tangent along the ribbon, normal across it --------------------
  let d_local = g_next.xyz - g_prev.xyz;
  let d_xz = rot2(d_local.xz, cy, sy) * plant_h;
  // Wind is a function of height, so its contribution to the tangent is exact.
  var tangent = vec3f(d_xz.x, d_local.y * plant_h, d_xz.y) + sway * (2.0 * hf * d_local.y);
  var nrm: vec3f;
  if (is_carpet) {
    // Real surface frame: the radial tangent above, crossed with the azimuthal
    // tangent taken from the NEIGHBOUR THIS EDGE FACES. A centred difference
    // would be constant across a wedge and shade the tile as a 32-spoke star;
    // the one-sided difference is shared by the two wedges meeting at an edge,
    // so the normal is continuous around the fan and interpolates across it.
    // Clamping the azimuthal slope to 45 degrees keeps the fan apex (where the
    // arc length goes to zero) from producing a horizontal normal.
    let neighbour = select((ribbon + ribbons - 1u) % ribbons, (ribbon + 1u) % ribbons, side > 0.0);
    let row_base = u32(variant) * rows_per_variant + u32(lod_info.counts.z);
    let y_side = textureLoad(atlas_geom, vec2u(s_at, row_base + neighbour), 0).y;
    let d_theta = 6.2831853 / max(f32(ribbons), 1.0);
    let arc_len = max(d_theta * sqrt(max(r2, 0.0)) * plant_h, 1.0e-5);
    let dy_theta = clamp((y_side - y_level) * side * plant_h, -arc_len, arc_len);
    nrm = normalize(cross(lat3 * arc_len + vec3f(0.0, dy_theta, 0.0), tangent));
  } else {
    let t_len = length(tangent);
    let tan_n = select(vec3f(0.0, 1.0, 0.0), tangent / t_len, t_len > 1.0e-7);
    let flat_c = cross(tan_n, lat3);
    let flat_len = length(flat_c);
    let flat_n = select(vec3f(0.0, 1.0, 0.0), flat_c / flat_len, flat_len > 1.0e-5);
    // Blades are channelled, not flat: rotate the normal across the width.
    let curl = lod_info.look.x * (1.0 - 0.65 * fluff);
    nrm = flat_n * cos(curl * side) + lat3 * sin(curl * side);
    // Fluff (panicles) shade as a soft volume, not as a plastic strip.
    nrm = normalize(mix(nrm, normalize(flat_n * 0.5 + vec3f(0.0, 1.0, 0.0)), fluff * 0.6));
  }

  // --- terrain conforming (carpets) ----------------------------------------
  // The scatter only gives a position and a yaw, so a mat would otherwise sit
  // bolt upright on its centre sample and bury one edge on any slope. One
  // terrain_sample per vertex gives the height AND the (nx, nz) needed to shear
  // the normal with it, for the price of a single bilinear fetch.
  if (is_carpet) {
    let ground = terrain_sample(world.xz);
    world.y = ground.x + (world.y - base.y);
    let ny = sqrt(max(1.0 - ground.y * ground.y - ground.z * ground.z, 1.0e-4));
    nrm = normalize(vec3f(nrm.x + nrm.y * ground.y / ny, nrm.y, nrm.z + nrm.y * ground.z / ny));
  }

  // --- per-plant colour variation ------------------------------------------
  // A carpet gets only a whisper of it: at 0.18 m spacing, per-tile brightness
  // jitter reads as a checkerboard rather than as natural mottling.
  let ph = hash3(bitcast<u32>(base.x * 91.7), bitcast<u32>(base.z * 91.7), u32(lod_info.look.w));
  let jitter_amp = select(0.28, 0.10, is_carpet);
  albedo *= 1.0 - jitter_amp * 0.5 + jitter_amp * hash_f32(ph);

  let show = u32(lod_info.look.z + 0.5);
  if (show == 1u) {
    albedo = mix(vec3f(0.95, 0.25, 0.1), vec3f(0.15, 0.35, 0.95), lod_info.extra.x * 0.25);
  } else if (show == 2u) {
    albedo = tint_of(hash2(ribbon, 7u)) * 0.8 + 0.1;
  } else if (show == 3u) {
    albedo = tint_of(hash2(u32(variant), 3u)) * 0.8 + 0.1;
  } else if (show == 4u) {
    albedo = mix(vec3f(0.15, 0.45, 0.15), vec3f(0.95, 0.25, 0.75), fluff);
  }

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.albedo = albedo;
  out.world = world;
  out.nrm = nrm;
  out.surf = vec3f(ao, coverage, fluff);
  return out;
}

@fragment
fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  // Ribbons are two-sided surfaces: shade whichever face the eye sees.
  //
  // For a CARPET the flip is against the VIEW VECTOR, not against
  // @builtin(front_facing). The strip winding is fixed in parameter space, and
  // for a fan whose centreline runs outward rather than upward the winding-based
  // flip hands back the normal of the face you CANNOT see: every Sphagnum tile
  // was lit from underneath and the whole mat rendered black. The view flip is
  // winding-independent and cannot get this wrong.
  //
  // The tuft path deliberately keeps the winding flip: switching it there is a
  // visible (and, against 000-ground-truth, better-looking) change to the
  // default stand's grass, which is out of scope here — see NOTES.md.
  let is_carpet = stand_table[u32(lod_info.look.w)].carpet_div > 0.0;
  var n = normalize(in.nrm);
  let away = select(!front, dot(n, frame.camera_pos - in.world) < 0.0, is_carpet);
  if (away) {
    n = -n;
  }
  // Canopy lift: bias the visible face toward the canopy normal — the standard
  // stand-in for the multiple scattering a single opaque blade cannot show.
  // Per-fragment variation is fully preserved; blades just stop reading as
  // black shards against sunlit terrain.
  n = normalize(n + vec3f(0.0, lod_info.extra.y, 0.0));
  let albedo = in.albedo;
  // AO is a light term, not albedo — keeps the lighting debug view honest.
  var color = light_surface(albedo, n, in.world) * in.surf.x;
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, in.surf.y, in.world), 1.0);
}
