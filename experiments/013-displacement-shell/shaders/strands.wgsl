#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// 013-displacement-shell: the meadow as one crumpled sheet.
//
// cs_cull  — per frame, walks the camera-region scatter cells (bit-identical
//            WGSL scatter twin), frustum-tests every existing plant and
//            appends it to one of three distance rings. Work is bounded by
//            the region, never by the stand's plant count.
// vs_main  — each plant is a flat patch of ribbon strips; the baked strand
//            vector-displacement field (posW/norAo/color textures) tells every
//            vertex WHERE the plant surface is. Strand-count LOD is
//            continuous: rows beyond s0*r0/d collapse to zero width and the
//            survivors widen by sqrt to conserve coverage.

@group(1) @binding(0) var<uniform> globals: Globals;

// -- cull bindings ----------------------------------------------------------
@group(1) @binding(1) var<storage, read_write> out_ring0: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> out_ring1: array<vec4f>;
@group(1) @binding(3) var<storage, read_write> out_ring2: array<vec4f>;
@group(1) @binding(4) var<storage, read_write> draw_args: array<atomic<u32>>;

/// Carpet (mat) instances: ONE buffer, six LOD-bucket slices at the bases in
/// globals.carpet_base. Six separate storage bindings would blow the 8-per-
/// stage limit; the draw side binds each slice as its own offset view.
@group(1) @binding(10) var<storage, read_write> out_carpet: array<vec4f>;

// -- draw bindings ----------------------------------------------------------
@group(1) @binding(5) var<storage, read> instances: array<vec4f>;
@group(1) @binding(6) var<uniform> ring_info: RingInfo;
@group(1) @binding(7) var strand_pos_tex: texture_2d_array<f32>;
@group(1) @binding(8) var strand_nor_tex: texture_2d_array<f32>;
@group(1) @binding(9) var strand_col_tex: texture_2d_array<f32>;

const TWO_PI: f32 = 6.2831853;
/// Carpet grid: quads per axis at full detail, and the sheet's extent as a
/// multiple of the periodic tile (mirror of GRID_N / GRID_EXT_FACTOR in bake.ts).
const GRID_N: u32 = 64u;
const GRID_EXT_FACTOR: f32 = 1.15;
const CARPET_LEVELS: u32 = 6u;
/// Thin-foliage hack: the baked per-station normal is pulled toward +Y so
/// leaves turned away from the sun do not read as black holes. It biases, it
/// does not replace — `debug=normals` shows the full baked normal variation
/// still coming through (including downward-facing undersides).
const UP_BIAS: f32 = 0.9;

/// Conservative "whole box is behind this plane" test. Planes come straight
/// from the view-proj rows and are NOT normalized — which is fine here: the
/// AABB form scales with the plane, so the test is exact either way.
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
  // Carpet entries are mats, not bundles of upward strands: they have their own
  // cull (cs_cull_carpet, all carpet_div^2 slots) and their own draw path.
  if (entry.carpet_div > 0.0) { return; }

  // Frustum planes (side + near) from the view-proj rows.
  let m = frame.view_proj;
  let row0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
  let row1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
  let row2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
  let row3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);
  let cull_on = globals.ring_debug < 0.5;

  // CELL-level reject first: a whole cell that cannot be on screen is killed
  // by five dot products instead of hashing + terrain-sampling all 128 of its
  // slots. The box below bounds EVERY plant sphere the per-plant test can
  // accept in this cell (cell footprint + max plant radius horizontally, the
  // full terrain amplitude + max plant height vertically), so the surviving
  // plant set is identical to the per-plant test alone.
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

  let h = entry.height_scale * cand.scale;
  let center = cand.pos + vec3f(0.0, h * 0.5, 0.0);
  // Region membership and rings use HORIZONTAL distance so overhead views
  // keep their plants (altitude only shrinks the strand-count LOD).
  let d = distance(cand.pos.xz, frame.camera_pos.xz);
  if (d >= globals.r_outer) { return; }

  // Frustum sphere test per plant.
  let rad = h * 0.9 + 0.6;
  let c4 = vec4f(center, 1.0);
  if (cull_on) {
    if (dot(row3 + row0, c4) < -rad) { return; }
    if (dot(row3 - row0, c4) < -rad) { return; }
    if (dot(row3 + row1, c4) < -rad) { return; }
    if (dot(row3 - row1, c4) < -rad) { return; }
    if (dot(row2, c4) < -rad) { return; }
  }

  var ring = 2u;
  if (d < globals.r0) { ring = 0u; } else if (d < globals.r1) { ring = 1u; }
  let cap = globals.caps[0][ring];

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
  let rec = vec4f(cand.pos, bitcast<f32>(packed));
  if (ring == 0u) { out_ring0[idx] = rec; }
  else if (ring == 1u) { out_ring1[idx] = rec; }
  else { out_ring2[idx] = rec; }
}

/**
 * Carpet cull. Same shape as cs_cull, three differences that matter:
 *  - it evaluates ALL carpet_div^2 slots of a cell (484 for the bog moss), not
 *    the 128-slot scatter budget — that budget is for scattered species and
 *    clamping to it renders a quarter of the mat in 4 m bands;
 *  - its region is much smaller (carpet_outer, ~12 m) because a mat aggregates
 *    into the terrain-conformal far shell almost immediately;
 *  - the bucket it appends to IS the LOD level: an integer texel stride over
 *    the baked grid, chosen to keep one grid quad at ~carpet_px pixels.
 */
@compute @workgroup_size(64)
fn cs_cull_carpet(@builtin(global_invocation_id) gid: vec3u) {
  let entry_index = gid.z;
  let entry = stand_table[entry_index];
  if (entry.carpet_div <= 0.0) { return; }
  let n = u32(entry.carpet_div);
  let slot = gid.x;
  if (slot >= n * n) { return; }
  let dims = globals.carpet_dims;
  if (i32(gid.y) >= dims.x * dims.y) { return; }
  let cell = globals.carpet_min + vec2i(i32(gid.y) % dims.x, i32(gid.y) / dims.x);

  let m = frame.view_proj;
  let row0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
  let row1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
  let row2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
  let row3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);
  let cull_on = globals.ring_debug < 0.5;

  // A carpet tile is wide and low: bound it from footprint_m (NEVER from
  // height_scale — 0.07 m tall, 0.23 m wide).
  let scale = entry.scale_min;
  let sheet_r = entry.footprint_m * GRID_EXT_FACTOR * 0.71 * scale + 0.02;
  let h_max = entry.height_scale * scale + 0.02;
  let cell_c = (vec2f(cell) + 0.5) * SCATTER_CELL_SIZE;
  let box_c = vec3f(cell_c.x, 0.0, cell_c.y);
  let box_e = vec3f(
    SCATTER_CELL_SIZE * 0.5 + sheet_r,
    frame.terrain_height_scale + h_max + 2.0,
    SCATTER_CELL_SIZE * 0.5 + sheet_r,
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
  // A mat is judged by 3D distance, not horizontal: from overhead its 7 cm of
  // relief is worth a pixel, so an overhead camera should get the far shell
  // rather than a disc of full-detail sheets under it. (The shell's inner radius
  // is derived from the same criterion on the CPU, so the two always meet.)
  let d = distance(cand.pos, frame.camera_pos);
  if (d >= globals.carpet_outer) { return; }

  let center = cand.pos + vec3f(0.0, h_max * 0.5, 0.0);
  let rad = sheet_r + h_max;
  let c4 = vec4f(center, 1.0);
  if (cull_on) {
    if (dot(row3 + row0, c4) < -rad) { return; }
    if (dot(row3 - row0, c4) < -rad) { return; }
    if (dot(row3 + row1, c4) < -rad) { return; }
    if (dot(row3 - row1, c4) < -rad) { return; }
    if (dot(row2, c4) < -rad) { return; }
  }

  // LOD level = smallest power-of-two texel stride whose quad is >= carpet_px
  // wide on screen. The stride is uniform for the whole sheet, so both edges of
  // every strip land on real baked grid lines and the mat stays watertight.
  let step_m = (entry.footprint_m * GRID_EXT_FACTOR / f32(GRID_N)) * cand.scale;
  let focal_px = frame.proj[1][1] * frame.viewport.y * 0.5;
  let quad_px = step_m * focal_px / max(d, 0.05);
  let need = globals.carpet_px / max(quad_px, 1e-5);
  var lvl = 0u;
  if (need > 1.0) { lvl = u32(ceil(log2(need))); }
  lvl = min(lvl, CARPET_LEVELS - 1u);

  // A full bucket PROMOTES to the next coarser one instead of dropping the
  // tile: a missing carpet tile is a hole in the mat, which is exactly the
  // artifact that reads as a placement bug.
  var idx = 0u;
  loop {
    idx = atomicAdd(&draw_args[(3u + lvl) * 5u + 1u], 1u);
    if (idx < globals.caps[(3u + lvl) / 4u][(3u + lvl) % 4u]) { break; }
    atomicSub(&draw_args[(3u + lvl) * 5u + 1u], 1u);
    if (lvl + 1u >= CARPET_LEVELS) { return; }
    lvl += 1u;
  }
  let yaw_q = u32(clamp(cand.yaw / TWO_PI, 0.0, 1.0) * 255.0 + 0.5) & 255u;
  let scale_t = (cand.scale - entry.scale_min) / max(entry.scale_max - entry.scale_min, 1e-5);
  let scale_q = u32(clamp(scale_t, 0.0, 1.0) * 255.0 + 0.5) & 255u;
  let packed = yaw_q | (scale_q << 8u) | (entry_index << 24u);
  out_carpet[globals.carpet_base[lvl / 4u][lvl % 4u] + idx] = vec4f(cand.pos, bitcast<f32>(packed));
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
  return (u * (globals.t_baked - 1.0) + 0.5) / globals.t_baked;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let rec = instances[ii];
  let packed = bitcast<u32>(rec.w);
  let entry_index = (packed >> 24u) & 255u;
  let entry = stand_table[entry_index];
  let yaw = f32(packed & 255u) * (TWO_PI / 255.0);
  let scale = mix(entry.scale_min, entry.scale_max, f32((packed >> 8u) & 255u) / 255.0);
  let phase = f32((packed >> 16u) & 255u) * (TWO_PI / 255.0);
  let species = u32(entry.species_index);
  let top_h = max(globals.canopy[species].w, 0.05);

  let t_ring = ring_info.t_ring;
  let strand = vi / (t_ring * 2u);
  let rem = vi % (t_ring * 2u);
  let station = rem >> 1u;
  let side = f32(rem & 1u) * 2.0 - 1.0;

  let base = rec.xyz;
  let d3 = distance(base, frame.camera_pos);

  // Continuous strand-count LOD + sqrt coverage conservation.
  let f_cont = min(globals.s0, globals.s0 * globals.r0 / max(d3, globals.r0));
  let strand_live = clamp(f_cont - f32(strand), 0.0, 1.0);
  let boost = sqrt(globals.s_baked / max(f_cont, 1.0));

  // Sample the strand displacement field at this station (+ the next one for
  // the tangent). u lands exactly on baked columns; linear filter handles
  // ring station counts below the baked count.
  let du = 1.0 / f32(t_ring - 1u);
  let u = f32(station) * du;
  let ua = min(u, 1.0 - du);
  let v = (f32(strand) + 0.5) / globals.s_baked;
  let uv_a = vec2f(station_x(ua), v);
  let uv_b = vec2f(station_x(ua + du), v);
  let pw_a = textureSampleLevel(strand_pos_tex, linear_sampler, uv_a, species, 0.0);
  let pw_b = textureSampleLevel(strand_pos_tex, linear_sampler, uv_b, species, 0.0);
  let at_end = u > 1.0 - du + 1e-4;
  let pw = select(pw_a, pw_b, at_end);
  let uv = select(uv_a, uv_b, at_end);

  let cy = cos(yaw);
  let sy = sin(yaw);
  let local = pw.xyz;
  var world = base + yaw_rotate(local, cy, sy) * scale;

  // Wind: shared field, sheared up the strand, small per-strand phase jitter.
  let hn = clamp(local.y / top_h, 0.0, 1.0);
  let s_jit = (hash_f32(hash2(strand, 977u)) - 0.5) * 1.6;
  world += wind_sway(base, frame.time, entry.sway, phase + s_jit) * pow(hn, 1.3);

  // Camera-facing ribbon: side = tangent x view.
  let tan_w = yaw_rotate(pw_b.xyz - pw_a.xyz, cy, sy);
  let view = world - frame.camera_pos;
  var side_v = cross(tan_w, view);
  let sl = length(side_v);
  if (sl > 1e-6) { side_v = side_v / sl; } else { side_v = vec3f(0.0); }

  // Width: baked half-width, LOD boost, camera-inside fade, region-edge trim.
  // Inside-plant fade uses HORIZONTAL distance (the camera hovers ~1.5 m above
  // the plant bases, so 3D distance would never reach zero).
  let d_xz = distance(base.xz, frame.camera_pos.xz);
  let near_fade = clamp((d_xz - 0.3) / 0.5, 0.0, 1.0);
  let edge_t = smoothstep(globals.r_outer - 5.0, globals.r_outer, d_xz);
  var w = pw.w * globals.width_scale * min(boost, 2.4) * scale * strand_live * near_fade * (1.0 - edge_t * 0.35);
  w = min(w, 0.085);
  if (globals.ring_debug > 0.5) { w = 0.02; }

  // Region edge: compress plants down onto the far-shell canopy height.
  let sink = mix(1.0, clamp(globals.shell_h / max(top_h * scale, 0.01), 0.0, 1.0) * 0.9 + 0.1, edge_t);
  world.y = base.y + (world.y - base.y) * sink;

  let nr = textureSampleLevel(strand_nor_tex, linear_sampler, uv, species, 0.0);
  let cl = textureSampleLevel(strand_col_tex, linear_sampler, uv, species, 0.0);
  let tint = 0.82 + 0.36 * fract(f32(packed & 255u) * 0.161803);

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world + side_v * (w * side), 1.0);
  out.color = cl.rgb * tint;
  if (globals.ring_debug > 0.5) {
    var dbg = vec3f(0.9, 0.2, 0.2);
    if (ring_info.ring_index == 1u) { dbg = vec3f(0.2, 0.9, 0.2); }
    if (ring_info.ring_index == 2u) { dbg = vec3f(0.2, 0.3, 0.9); }
    out.color = dbg;
  }
  out.normal = yaw_rotate(nr.xyz, cy, sy);
  out.world = world;
  out.ao = nr.w;
  return out;
}

/**
 * Carpet mat: the tile IS a displaced grid sheet.
 *
 * A vertex is (strip row, column, side); the baked grid gives its offset from
 * the tile centre, its surface normal, colour and relief occlusion. Both edges
 * of a strip land exactly on baked grid LINES (side selects row or row+stride),
 * so the sheet is watertight at every LOD level — no seams, no stochastic
 * coverage, hard opaque depth-writing geometry.
 *
 * Terrain fitting is per-vertex (ladder rung 3): the ground height is sampled
 * under each vertex and the cushion's own height rides on top. Because that is
 * a pure function of world xz, neighbouring tiles agree exactly at their shared
 * edge; a per-tile plane fit would crack the mat apart over every bump.
 */
@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let rec = instances[ii];
  let packed = bitcast<u32>(rec.w);
  let entry_index = (packed >> 24u) & 255u;
  let entry = stand_table[entry_index];
  let yaw = f32(packed & 255u) * (TWO_PI / 255.0);
  let scale = mix(entry.scale_min, entry.scale_max, f32((packed >> 8u) & 255u) / 255.0);
  let species = u32(entry.species_index);

  let stride = ring_info.grid_stride;
  let cols = ring_info.grid_bands + 1u;
  let strip = vi / (cols * 2u);
  let rem = vi % (cols * 2u);
  let col = rem >> 1u;
  let side = rem & 1u;
  let texel = vec2i(i32(col * stride), i32((strip + side) * stride));

  let pw = textureLoad(strand_pos_tex, texel, species, 0);
  let nr = textureLoad(strand_nor_tex, texel, species, 0);
  let cl = textureLoad(strand_col_tex, texel, species, 0);

  let cy = cos(yaw);
  let sy = sin(yaw);
  let local = yaw_rotate(pw.xyz, cy, sy) * scale;
  let xz = rec.xz + local.xz;
  // One bilinear fetch for height AND slope (terrain_height + terrain_normal
  // would pay for the same taps twice).
  let g = terrain_sample(xz);
  let up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  // Sub-millimetre per-tile depth phase: sheets of identical species and yaw
  // one period apart would otherwise be exactly coplanar where they overlap.
  let phase = hash_f32(hash2(bitcast<u32>(rec.x), bitcast<u32>(rec.z))) * 0.0004;
  var world = vec3f(xz.x, g.x + local.y + phase, xz.y);
  if (entry.sway > 0.0) {
    world += wind_sway(rec.xyz, frame.time, entry.sway, 0.0) * clamp(local.y * 8.0, 0.0, 1.0);
  }

  let align = clamp(entry.slope_align, 0.0, 1.0);
  let up_a = normalize(mix(vec3f(0.0, 1.0, 0.0), up, align));
  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.color = cl.rgb;
  out.normal = tilt_from_up(yaw_rotate(nr.xyz, cy, sy), up_a);
  out.world = world;
  out.ao = nr.w;
  if (globals.ring_debug > 0.5) {
    var dbg = vec3f(0.9, 0.2, 0.2);
    if (ring_info.ring_index == 4u) { dbg = vec3f(0.9, 0.6, 0.1); }
    if (ring_info.ring_index == 5u) { dbg = vec3f(0.2, 0.9, 0.2); }
    if (ring_info.ring_index == 6u) { dbg = vec3f(0.1, 0.7, 0.7); }
    if (ring_info.ring_index == 7u) { dbg = vec3f(0.2, 0.3, 0.9); }
    if (ring_info.ring_index == 8u) { dbg = vec3f(0.7, 0.2, 0.9); }
    out.color = dbg;
  }
  return out;
}

/**
 * Carpet fragment: the mat's baked relief normals are the whole point, so
 * (unlike the thin-foliage strand path) they are used as they are — no up-bias
 * and no flip toward the viewer.
 */
@fragment
fn fs_carpet(in: VOut) -> @location(0) vec4f {
  var n = in.normal;
  let nl = length(n);
  if (nl < 1e-4) { n = vec3f(0.0, 1.0, 0.0); } else { n = n / nl; }
  // Relief occlusion darkens the ALBEDO (self-shadowing of the mat's own
  // colour, so debug_shade's albedo/lighting split stays exact). It carries the
  // crevice depth the 3 mm grid cannot resolve geometrically, which is why the
  // floor is this low.
  let albedo = in.color * mix(0.42, 1.0, in.ao);
  var color = light_surface(albedo, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, 1.0, in.world), 1.0);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  var n = in.normal;
  let nl = length(n);
  if (nl < 1e-4) { n = vec3f(0.0, 1.0, 0.0); } else { n = n / nl; }
  // Thin foliage: shade both sides alike (flip toward the viewer) and bias
  // up so baked normals facing away from the sun don't go black.
  let to_cam = frame.camera_pos - in.world;
  n *= sign(dot(n, to_cam) + 1e-5);
  n = normalize(n + vec3f(0.0, UP_BIAS, 0.0));
  // Baked vertical occlusion darkens the ALBEDO (it is self-shadowing of the
  // plant's own colour), so debug_shade's lighting/albedo split stays exact.
  let albedo = in.color * mix(0.55, 1.0, in.ao);
  var color = light_surface(albedo, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Hard-edged opaque geometry: every rasterized fragment is full coverage.
  return vec4f(debug_shade(color, albedo, n, 1.0, in.world), 1.0);
}
