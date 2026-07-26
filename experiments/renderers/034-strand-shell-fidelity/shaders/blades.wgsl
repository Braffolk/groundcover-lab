#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// 034-strand-shell-fidelity — a meadow of REAL blades.
//
// cs_cull  per frame, walks the camera-region scatter cells (bit-identical
//          WGSL twin of the harness scatter), frustum-tests every plant and
//          appends it to one of five distance rings. Work is bounded by the
//          region, never by the stand's plant count.
// vs_main  every vertex is (blade row, station, lateral index). The baked
//          blade library says where the blade is, how wide it is, which way
//          its PLANE faces, what colour it is and how occluded it is. The
//          ribbon is built in the blade's OWN frame, not facing the camera,
//          so the silhouette genuinely changes with view direction and blades
//          occlude each other in 3D. Only past `orient_far`, where a blade is
//          sub-pixel and orientation buys nothing but shimmer, does the width
//          axis rotate to face the viewer.
//
// Zero texture taps per fragment; four per vertex. No marching, no discard,
// no dither: hard opaque geometry that writes depth, drawn front-to-back so
// early-z can reject the layers behind it.

@group(1) @binding(0) var<uniform> globals: Globals;

// -- cull bindings ----------------------------------------------------------
@group(1) @binding(1) var<storage, read_write> out_instances: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>>;

// -- draw bindings ----------------------------------------------------------
@group(1) @binding(3) var<storage, read> instances: array<vec4f>;
@group(1) @binding(4) var<uniform> ring_info: RingInfo;
@group(1) @binding(5) var blade_pos_tex: texture_2d_array<f32>;
@group(1) @binding(6) var blade_nor_tex: texture_2d_array<f32>;
@group(1) @binding(7) var blade_col_tex: texture_2d_array<f32>;

const TWO_PI: f32 = 6.2831853;

/// Conservative "whole box is behind this plane". Planes come straight from
/// the view-proj rows and are NOT normalized — fine, the AABB form scales with
/// the plane so the test is exact either way.
fn aabb_outside(p: vec4f, c: vec3f, e: vec3f) -> bool {
  return dot(p.xyz, c) + p.w + dot(abs(p.xyz), e) < 0.0;
}

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= SCATTER_MAX_PER_CELL) { return; }
  let dims = globals.region_dims;
  if (i32(gid.y) >= dims.x * dims.y) { return; }
  let entry_index = gid.z;
  let cell = globals.region_min + vec2i(i32(gid.y) % dims.x, i32(gid.y) / dims.x);
  let entry = stand_table[entry_index];

  let m = frame.view_proj;
  let row0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
  let row1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
  let row2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
  let row3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);
  let cull_on = globals.ring_debug < 0.5;

  // CELL-level reject first: a whole cell that cannot be on screen dies for
  // five dot products instead of hashing + terrain-sampling all 128 slots.
  // The box bounds EVERY plant sphere the per-plant test could accept, so the
  // surviving set is provably identical to the per-plant test alone.
  let h_max = entry.height_scale * entry.scale_max;
  let pad = h_max * 0.9 + 0.6;
  let cell_c = (vec2f(cell) + 0.5) * SCATTER_CELL_SIZE;
  let box_c = vec3f(cell_c.x, 0.0, cell_c.y);
  let box_e = vec3f(
    SCATTER_CELL_SIZE * 0.5 + pad,
    frame.terrain_height_scale + h_max + pad + 2.0,
    SCATTER_CELL_SIZE * 0.5 + pad,
  );
  if (cull_on) {
    if (aabb_outside(row3 + row0, box_c, box_e)) { return; }
    if (aabb_outside(row3 - row0, box_c, box_e)) { return; }
    if (aabb_outside(row3 + row1, box_c, box_e)) { return; }
    if (aabb_outside(row3 - row1, box_c, box_e)) { return; }
    if (aabb_outside(row2, box_c, box_e)) { return; }
  }

  let cand = scatter_candidate(globals.seed, entry_index, cell, slot);
  if (!cand.exists) { return; }

  // REGION MEMBERSHIP is horizontal, so an overhead view keeps every plant.
  let d = distance(cand.pos.xz, frame.camera_pos.xz);
  if (d >= globals.r_outer) { return; }

  let h = entry.height_scale * cand.scale;
  let center = cand.pos + vec3f(0.0, h * 0.5, 0.0);
  let rad = h * 0.9 + 0.6;
  let c4 = vec4f(center, 1.0);
  if (cull_on) {
    if (dot(row3 + row0, c4) < -rad) { return; }
    if (dot(row3 - row0, c4) < -rad) { return; }
    if (dot(row3 + row1, c4) < -rad) { return; }
    if (dot(row3 - row1, c4) < -rad) { return; }
    if (dot(row2, c4) < -rad) { return; }
  }

  // ...but RING SELECTION is by true 3D distance, which is what decides how
  // many pixels the plant covers. 013 used horizontal distance here too, and
  // that is why `topdown` cost it dearly: a plant 42 m straight below the
  // camera has d_xz ~ 0, so it landed in the richest ring and drew the full
  // near-field topology for a 20-pixel smudge. Ring 4 is the catch-all, so
  // nothing is lost from overhead.
  let d3 = distance(cand.pos, frame.camera_pos);
  var ring = 4u;
  for (var k = 0u; k < 4u; k++) {
    if (d3 < globals.rings[k].x) {
      ring = k;
      break;
    }
  }
  let rr = globals.rings[ring];
  let cap = u32(rr.z);

  let idx = atomicAdd(&draw_args[ring * 5u + 1u], 1u);
  if (idx >= cap) {
    atomicSub(&draw_args[ring * 5u + 1u], 1u);
    return;
  }

  let yaw_q = u32(clamp(cand.yaw / TWO_PI, 0.0, 1.0) * 255.0 + 0.5) & 255u;
  let scale_t = (cand.scale - entry.scale_min) / max(entry.scale_max - entry.scale_min, 1e-5);
  let scale_q = u32(clamp(scale_t, 0.0, 1.0) * 255.0 + 0.5) & 255u;
  let phase_q = u32(clamp(cand.phase / TWO_PI, 0.0, 1.0) * 255.0 + 0.5) & 255u;
  let packed = yaw_q | (scale_q << 8u) | (phase_q << 16u) | (entry_index << 24u);
  out_instances[u32(rr.y) + idx] = vec4f(cand.pos, bitcast<f32>(packed));
}

// ---------------------------------------------------------------------------

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) world: vec3f,
  @location(3) ao: f32,
}

fn station_x(u: f32) -> f32 {
  return (u * (globals.stations_baked - 1.0) + 0.5) / globals.stations_baked;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let ri = ring_info.ring_index;
  let rr = globals.rings[ri];
  let rec = instances[u32(rr.y) + ii];
  let packed = bitcast<u32>(rec.w);
  let entry_index = (packed >> 24u) & 255u;
  let entry = stand_table[entry_index];
  let yaw = f32(packed & 255u) * (TWO_PI / 255.0);
  let scale = mix(entry.scale_min, entry.scale_max, f32((packed >> 8u) & 255u) / 255.0);
  let phase = f32((packed >> 16u) & 255u) * (TWO_PI / 255.0);
  let species = u32(entry.species_index);
  let top_h = max(globals.canopy[species].w, 0.05);

  let plant_h = hash3(bitcast<u32>(rec.x), bitcast<u32>(rec.z), packed);
  // Per-plant mirror + baked-row window rotation. Both preserve the LOD
  // contract: rows are emitted round-robin over 8 spatial buckets at bake
  // time, so a window offset by a multiple of 8 is just as balanced, and a
  // mirrored blade is still a real blade.
  let mir = select(-1.0, 1.0, (plant_h & 1u) == 0u);
  let row_rot = ((plant_h >> 1u) & 15u) * 8u;

  let ts = ring_info.stations;
  let lat_n = ring_info.lat_count;
  let per_blade = ts * lat_n;
  let row = vi / per_blade;
  let rem = vi % per_blade;
  let station = rem / lat_n;
  let lat_i = rem % lat_n;
  let lat_t = f32(lat_i) / f32(lat_n - 1u) * 2.0 - 1.0;

  let bgs = rec.xyz;
  let d3 = distance(bgs, frame.camera_pos);
  let d_xz = distance(bgs.xz, frame.camera_pos.xz);

  // Blade-count LOD: rows = c_rows / d, capped by what this ring draws. The
  // marginal row fades fractionally, so the count is continuous in distance.
  // `row_gain` sheds rows faster in the far rings: coverage is conserved by
  // the width boost below, so a 27-pixel-tall plant becomes 8 wide strands
  // instead of 16 thin ones — visually identical, half the vertices, and it
  // is what pays for ring 0's 320-blade near field.
  let rows_ring = rr.w;
  let f_live = clamp(globals.c_rows * ring_info.row_gain / max(d3, 1e-3), 1.0, rows_ring);
  let live = clamp(f_live - f32(row), 0.0, 1.0);
  // Coverage conservation: the baked widths are silhouette-calibrated for the
  // FULL library, so drawing n of them widens by (blades_baked / n)^cover_pow.
  let boost = pow(globals.blades_baked / max(f_live, 1.0), globals.cover_pow);

  let blade_id = (row + row_rot) % u32(globals.blades_baked);
  let v = (f32(blade_id) + 0.5) / globals.blades_baked;

  let du = 1.0 / f32(ts - 1u);
  let u = f32(station) * du;
  let ua = min(u, 1.0 - du);
  let uv_a = vec2f(station_x(ua), v);
  let uv_b = vec2f(station_x(ua + du), v);
  let pw_a = textureSampleLevel(blade_pos_tex, linear_sampler, uv_a, species, 0.0);
  let pw_b = textureSampleLevel(blade_pos_tex, linear_sampler, uv_b, species, 0.0);
  let at_end = station == ts - 1u;
  let pw = select(pw_a, pw_b, at_end);
  let uv = select(uv_a, uv_b, at_end);
  let nr = textureSampleLevel(blade_nor_tex, linear_sampler, uv, species, 0.0);
  let cl = textureSampleLevel(blade_col_tex, linear_sampler, uv, species, 0.0);

  let cy = cos(yaw);
  let sy = sin(yaw);
  let local = vec3f(pw.x * mir, pw.y, pw.z);
  var world = bgs + yaw_rotate(local, cy, sy) * scale;

  let hn = clamp(pw.y / top_h, 0.0, 1.0);
  // Wind: shared field, sheared up the blade, per-blade phase jitter.
  let bh = hash2(blade_id, plant_h);
  let jit = (hash_f32(bh) - 0.5) * 1.7;
  world += wind_sway(bgs, frame.time, entry.sway, phase + jit) * pow(hn, 1.3);
  // Static per-blade bend + per-plant lean: the cheapest possible cure for
  // "every plant is the same mesh". Both grow with height, like wind, so the
  // blade stays attached at its base.
  let ba = hash_f32(hash2(bh, 5u)) * TWO_PI;
  let la = hash_f32(hash2(plant_h, 9u)) * TWO_PI;
  let bend = vec3f(cos(ba), 0.0, sin(ba)) * (globals.bend_amp * hash_f32(hash2(bh, 17u)));
  let lean = vec3f(cos(la), 0.0, sin(la)) * (globals.bend_amp * 0.9);
  world += (bend * pow(hn, 1.7) + lean * pow(hn, 1.2)) * scale;

  // ---- the blade's own frame ----------------------------------------------
  let tan_l = vec3f((pw_b.x - pw_a.x) * mir, pw_b.y - pw_a.y, pw_b.z - pw_a.z);
  var t_w = yaw_rotate(tan_l, cy, sy);
  let tl = length(t_w);
  t_w = select(vec3f(0.0, 1.0, 0.0), t_w / max(tl, 1e-9), tl > 1e-7);

  var n_w = yaw_rotate(vec3f(nr.x * mir, nr.y, nr.z), cy, sy);
  n_w = n_w - t_w * dot(n_w, t_w);
  let nl = length(n_w);
  n_w = select(normalize(cross(t_w, vec3f(0.0, 0.0, 1.0)) + vec3f(1e-4, 0.0, 0.0)), n_w / max(nl, 1e-9), nl > 1e-4);
  var b_base = cross(t_w, n_w);
  b_base = normalize(b_base + vec3f(1e-9, 0.0, 0.0));

  let view_v = world - frame.camera_pos;
  let view_len = max(length(view_v), 1e-5);
  let view_dir = view_v / view_len;
  var b_cam = cross(t_w, view_dir);
  let bcl = length(b_cam);
  b_cam = select(b_base, b_cam / max(bcl, 1e-9), bcl > 1e-4);
  b_cam *= sign(dot(b_cam, b_base) + 1e-6);

  // A fluffy panicle station has no meaningful plane (isotropic neighbourhood
  // at bake time), so it camera-faces regardless of distance; a real blade
  // keeps its baked plane until it is sub-pixel.
  let blend = clamp(max(smoothstep(globals.orient_near, globals.orient_far, d3), cl.a * 0.85), 0.0, 1.0);
  let b_eff = normalize(mix(b_base, b_cam, blend) + vec3f(1e-9, 0.0, 0.0));
  let n_eff = normalize(cross(b_eff, t_w) + vec3f(0.0, 1e-9, 0.0));

  // ---- width ---------------------------------------------------------------
  var w = pw.w * globals.width_scale * boost * scale;
  // Minimum PROJECTED width: an oriented blade seen edge-on is genuinely
  // sub-pixel, which is correct but shimmers. Widening it to ~1px keeps the
  // silhouette honest while killing the sparkle. Capped so a blade can never
  // balloon past 3x its true width.
  //
  // The projected extent of the width axis is sin(angle between b_eff and the
  // view ray) — NOT |n_eff . view|. They differ exactly for a blade pointing
  // at the camera, which is foreshortened along its tangent but still shows
  // its FULL width; using the normal there widened near blades into fat
  // sheets, the one real artifact this method had.
  let bv = dot(b_eff, view_dir);
  let width_proj = sqrt(max(1.0 - bv * bv, 0.0));
  let world_per_px = 2.0 * view_len / max(frame.proj[1][1] * frame.viewport.y, 1e-3);
  let min_half = 0.5 * globals.min_px * world_per_px;
  w = max(w, min(min_half / max(width_proj, 0.10), w * 3.0 + min_half));
  // Camera-inside-plant + region-edge fades collapse WIDTH, never alpha.
  let near_fade = clamp((d_xz - 0.28) / 0.45, 0.0, 1.0);
  let edge_t = smoothstep(globals.r_outer - 5.0, globals.r_outer, d_xz);
  w = w * live * near_fade * (1.0 - edge_t * 0.35);
  if (globals.ring_debug > 0.5) { w = 0.006 * boost; }

  // ---- cross-section -------------------------------------------------------
  // 3-wide rings get a real keel: the mid vertex lifts along the plane normal,
  // so the blade has geometric thickness, self-shadows across its own crease
  // and never fully vanishes edge-on.
  let bulge = select(0.0, globals.keel * w * (1.0 - lat_t * lat_t), lat_n == 3u);
  var vpos = world + b_eff * (w * lat_t) + n_eff * bulge;

  // Region edge: compress plants onto the far-shell canopy height.
  let sink = mix(1.0, clamp(globals.shell_h / max(top_h * scale, 0.01), 0.0, 1.0) * 0.9 + 0.1, edge_t);
  vpos.y = bgs.y + (vpos.y - bgs.y) * sink;

  // Shading normal rotates across the width -> a channelled blade catches a
  // highlight along its crease instead of reading as a flat card.
  let vn = normalize(n_eff + b_eff * (lat_t * globals.curl));

  let tint = 0.86 + 0.28 * hash_f32(hash2(plant_h, 3u));

  var out: VOut;
  out.pos = frame.view_proj * vec4f(vpos, 1.0);
  out.color = cl.rgb * tint;
  if (globals.ring_debug > 0.5) {
    var dbg = vec3f(0.95, 0.25, 0.25);
    if (ri == 1u) { dbg = vec3f(0.95, 0.7, 0.2); }
    if (ri == 2u) { dbg = vec3f(0.25, 0.9, 0.3); }
    if (ri == 3u) { dbg = vec3f(0.25, 0.5, 0.95); }
    if (ri == 4u) { dbg = vec3f(0.75, 0.35, 0.95); }
    out.color = dbg;
  }
  out.normal = vn;
  out.world = vpos;
  out.ao = nr.w;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  var n = in.normal;
  let nl = length(n);
  n = select(vec3f(0.0, 1.0, 0.0), n / max(nl, 1e-9), nl > 1e-4);
  // Thin foliage is two-sided: flip the normal toward the viewer, and darken
  // the underside slightly — a real cue that you are looking at the back of a
  // blade, and one more thing that makes the canopy read as depth.
  let to_cam = frame.camera_pos - in.world;
  let facing = dot(n, to_cam);
  n *= sign(facing + 1e-6);
  // Small +Y nudge only. 013 used 0.9, which flattens every blade normal onto
  // one direction and is a large part of why it read as a stand-in.
  n = normalize(n + vec3f(0.0, globals.up_bias, 0.0));

  let back = select(1.0, 0.84, facing < 0.0);
  // Baked occlusion is self-shadowing of the plant's own colour, so it darkens
  // ALBEDO — debug_shade's lighting/albedo split then stays exact.
  let albedo = in.color * back * mix(globals.ao_min, 1.0, in.ao);
  var color = light_surface(albedo, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Hard-edged opaque geometry: every rasterized fragment is full coverage,
  // writes depth, and occludes what is behind it. No alpha test, no dither.
  return vec4f(debug_shade(color, albedo, n, 1.0, in.world), 1.0);
}
