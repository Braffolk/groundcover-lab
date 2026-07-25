#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// Stochastic thinning cascade. Placement is evaluated procedurally with the
// shared scatter twin over concentric cell annuli around the camera. A plant
// at distance d survives iff its slot rank r < keep(d) = (R0/d)^2; survivors
// widen by 1/sqrt(keep) so the ENSEMBLE frontal area is statistically
// conserved. Ring k doubles the covered radius and quarters the slot cap, so
// per-ring instance count is constant and total per-frame work is
// O(log(view radius)) — independent of stand plant count.
//
// Coverage is realized per pixel: the atlas alpha is mip-averaged fractional
// coverage (premultiplied bake), tested against a world/texel-anchored hash
// threshold. Near the camera the threshold is a hard 0.5 (crisp silhouettes,
// solid early-z); with distance it morphs into the hashed test that IS the
// distance-collapse mechanism (see NOTES.md for the taste-rule justification).

struct CardUni {
  rings: array<vec4<u32>, 6>, // start_instance, n (outer half-cells), h (hole half-cells), cap
  ring_count: u32,
  seed: u32,
  entry_index: u32,
  debug_rings: u32,
  cam_cell: vec2i,
  stand_radius: f32,
  detail_r: f32,
  card_center: vec3f,
  r_card: f32,
  width_cap: f32,
  densify: f32,
  stoch_start: f32,
  stoch_full: f32,
  // --- carpet (mat) path; carpet_flag = 0 leaves every card path untouched ---
  carpet_flag: u32,
  carpet_cells: u32,   // block half-width in scatter cells around the camera
  carpet_slots: u32,   // carpet_div^2 — NEVER SCATTER_MAX_PER_CELL
  carpet_shells: u32,  // relief shells per tile (index count = 6 * shells)
  tile_scale: f32,     // mesh -> world factor for the tile (entry scale)
  moss_h_min: f32,     // height range stored in the tile's height channel
  moss_h_range: f32,
  moss_radius: f32,    // tile draw radius; the mean field carries past it
  cover_ref: f32,      // carpet alpha reference (a mat must stay closed)
  shell_r: f32,        // distance at which the first relief shell drops out
  cam_above: f32,      // camera height over the ground (shell LOD in 3D)
  relief: f32,         // weight of the height-field normal perturbation
  lift: f32,           // concavity lift over the coarse drawn terrain mesh
  _cpad0: f32,
  _cpad1: f32,
  _cpad2: f32,
  // Per-shell (height threshold, placement height), both in the baked height
  // field's normalized units. Equal-AREA bands of the cushion top surface, so
  // no shell sits in empty air; x = 0 for shell 0, which therefore covers the
  // whole tile and keeps the mat closed whatever the shell LOD drops.
  shells_tab: array<vec4f, 6>,
}

@group(1) @binding(0) var<uniform> card_uni: CardUni;
@group(1) @binding(1) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(2) var atlas_normal: texture_2d<f32>;
@group(1) @binding(3) var atlas_samp: sampler;

// Everything the fragment stage needs that is CONSTANT over the card is
// resolved here in the vertex stage and carried flat: yaw's sin/cos, the
// densify exponent (from the width amplification) and the stochastic-blend
// factor (from distance). The FS never re-derives them per pixel.
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,                          // card uv, v up
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) pid: u32,        // stable per-plant hash
  @location(3) @interpolate(flat) yaw_gs: vec4f,   // cos(yaw), sin(yaw), gamma, sigma
  @location(4) @interpolate(flat) axes: vec4f,     // right.xz, up2.xz (top-view mapping)
  @location(5) @interpolate(flat) top_vf: vec2f,   // top-view weight, azimuth view coord
  @location(6) @interpolate(flat) ring_id: u32,
}

fn kill_vertex() -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped
  return out;
}

/**
 * CARPET (mat) vertex path — used when stand_table[i].carpet_div > 0.
 *
 * None of the card machinery applies to a 0.18m periodic cushion tile: a
 * camera-facing quad breaks the lattice, stochastic thinning would punch holes
 * in a surface that must stay closed, and width amplification would change the
 * tile scale the stand fixed. So a carpet tile is drawn as what it is: a
 * grid-snapped, ground-parallel, per-vertex terrain-conforming square at the
 * exact grid step, yawed only by the scatter's quarter turns (applied to the
 * TEXTURE, since a quarter turn maps the square onto itself and neighbours must
 * keep sharing corners).
 *
 * Thickness comes from a stack of nested shells: shell k sits in the k-th band
 * of the baked cushion height field and keeps only the texels whose cushion top
 * reaches that band, so the stack is a quantized height field with real
 * silhouette, real parallax and solid depth. Shell 0 covers every texel, so no
 * shell LOD can ever open the mat.
 */
fn vs_carpet(vi: u32, ii: u32) -> VOut {
  let entry = stand_table[card_uni.entry_index];
  let slots = card_uni.carpet_slots;
  let cells = card_uni.carpet_cells;
  let w_cells = 2u * cells;
  let cell_idx = ii / slots;
  let slot = ii % slots;
  let cell = card_uni.cam_cell + vec2i(i32(cell_idx % w_cells) - i32(cells), i32(cell_idx / w_cells) - i32(cells));

  // The grid node is pure arithmetic — no hashes, no terrain — so distance and
  // shell rejection happen before any placement work is paid for.
  let div = u32(entry.carpet_div);
  let step = SCATTER_CELL_SIZE / entry.carpet_div;
  let g = vec2f(f32(slot % div), f32(slot / div));
  let node = vec2f(cell) * SCATTER_CELL_SIZE + (g + 0.5) * step;
  let d = distance(frame.camera_pos.xz, node);
  if (d > card_uni.moss_radius) { return kill_vertex(); }
  if (abs(node.x) > card_uni.stand_radius || abs(node.y) > card_uni.stand_radius) { return kill_vertex(); }
  // Relief shells collapse with distance (each higher shell dies ~40% nearer
  // than the one below); the mat itself never thins. The metric is the 3D
  // distance — from 40m up a tile is 2px and its relief is not worth a draw,
  // even though it is directly below the camera.
  let layer = vi / 4u;
  let d3 = length(vec2f(d, card_uni.cam_above));
  if (layer > 0u && d3 > card_uni.shell_r * pow(0.6, f32(layer - 1u))) { return kill_vertex(); }

  // Existence is the stand's business: the shared twin resolves which of the
  // competing carpet entries owns this node from the wetness partition.
  let sp = scatter_candidate(card_uni.seed, card_uni.entry_index, cell, slot);
  if (!sp.exists) { return kill_vertex(); }

  // Footprint from the species' periodic tile size, never from height_scale.
  let fp = select(step, entry.footprint_m * entry.scale_min, entry.footprint_m > 0.0);
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let c = corners[vi % 4u];
  let corner_xz = node + c * (0.5 * fp);
  // Ladder rung 3: every corner gets its own height from ONE bilinear fetch.
  // Neighbouring tiles share corner positions exactly, so the mat stays C0
  // continuous — a per-tile plane fit would crack at every shared edge.
  let ts = terrain_sample(corner_xz);
  let cyaw = cos(sp.yaw);
  let syaw = sin(sp.yaw);
  let local = vec2f(cyaw * c.x + syaw * c.y, cyaw * c.y - syaw * c.x);
  let shell = card_uni.shells_tab[layer];
  let shell_h = card_uni.moss_h_min + shell.y * card_uni.moss_h_range;
  let align = clamp(entry.slope_align, 0.0, 1.0);
  var base_h = ts.x;
  var up = normalize(mix(
    vec3f(0.0, 1.0, 0.0),
    vec3f(ts.y, sqrt(max(1.0 - ts.y * ts.y - ts.z * ts.z, 0.0)), ts.z),
    align,
  ));
  if (card_uni.lift > 0.0) {
    // The base pass draws the terrain as a COARSE triangle mesh (1m quads),
    // whose chords bridge every concavity of the finer bilinear heightmap that
    // terrain_sample() returns — measured up to 7.2cm above it, 3cm+ over 1.6%
    // of the area. A mat conformed to the true surface therefore sinks INTO the
    // drawn ground in hollows, and where the gap exceeds the cushion height it
    // is swallowed whole — a hazard every per-vertex-conforming renderer
    // inherits. Lift by the local concavity: the plane fit's mean height (which
    // is what a chord across the hollow approximates) minus the point height.
    // Per CORNER, so neighbours still agree and the mat stays continuous, and
    // it assumes nothing about the base pass's tessellation. Mean lift is 4mm.
    let pf = terrain_plane_fit(corner_xz, 0.5);
    base_h += card_uni.lift * max(0.0, pf.h - ts.x);
    up = normalize(mix(vec3f(0.0, 1.0, 0.0), pf.up, align));
  }
  // Offset along the GROUND normal, not straight up, so a cushion on a slope
  // stands out of the slope. slope_align says how much the species conforms.
  let world = vec3f(corner_xz.x, base_h, corner_xz.y) + up * (shell_h * card_uni.tile_scale);

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.uv = local * 0.5 + 0.5;
  out.world = world;
  out.pid = 0u;
  // z = the shell's lower height bound in the baked height field's units.
  out.yaw_gs = vec4f(cyaw, syaw, shell.x, 0.0);
  out.axes = vec4f(0.0);
  out.top_vf = vec2f(0.0);
  out.ring_id = layer;
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  if (card_uni.carpet_flag == 1u) { return vs_carpet(vi, ii); }
  // --- ring lookup (<= 6 iterations) ---
  var r = 0u;
  loop {
    if (r + 1u >= card_uni.ring_count || ii < card_uni.rings[r + 1u].x) { break; }
    r = r + 1u;
  }
  let ring = card_uni.rings[r];
  let n = i32(ring.y);
  let h = i32(ring.z);
  let cap = ring.w;
  let local = ii - ring.x;
  let cell_idx = local / cap;
  let slot = local % cap;

  // --- annulus cell decomposition: 2 full-width strips + 2 side blocks ---
  let w_cells = u32(2 * n);
  var cx: i32;
  var cz: i32;
  if (h == 0) {
    cx = i32(cell_idx % w_cells) - n;
    cz = i32(cell_idx / w_cells) - n;
  } else {
    let strip = w_cells * u32(n - h);
    let side_w = u32(n - h);
    let side_c = side_w * u32(2 * h);
    if (cell_idx < strip) {
      cx = i32(cell_idx % w_cells) - n;
      cz = h + i32(cell_idx / w_cells);
    } else if (cell_idx < 2u * strip) {
      let j = cell_idx - strip;
      cx = i32(j % w_cells) - n;
      cz = -n + i32(j / w_cells);
    } else {
      let j = cell_idx - 2u * strip;
      let side = j / side_c;
      let l = j % side_c;
      cx = select(-n, h, side == 1u) + i32(l % side_w);
      cz = i32(l / side_w) - h;
    }
  }
  let cell = card_uni.cam_cell + vec2i(cx, cz);

  // --- cheap existence + rank rejection before touching the terrain ---
  let eh = hash4(card_uni.seed, bitcast<u32>(cell.x), bitcast<u32>(cell.y), (card_uni.entry_index << 16u) ^ slot);
  let entry = stand_table[card_uni.entry_index];
  if (hash_f32(hash2(eh, 0u)) >= min(entry.density, SCATTER_MAX_DENSITY) / SCATTER_MAX_DENSITY) {
    return kill_vertex();
  }
  let ox = hash_f32(hash2(eh, 1u));
  let oz = hash_f32(hash2(eh, 2u));
  let xz = (vec2f(cell) + vec2f(ox, oz)) * SCATTER_CELL_SIZE;
  if (abs(xz.x) > card_uni.stand_radius || abs(xz.y) > card_uni.stand_radius) {
    return kill_vertex();
  }
  let d = distance(frame.camera_pos.xz, xz);

  // --- stochastic thinning: survival by slot rank vs keep(d) ---
  let rd = card_uni.detail_r / max(d, card_uni.detail_r);
  let keep = rd * rd; // min(1, (R0/d)^2)
  let rank = (f32(slot) + 0.5) / 128.0;
  if (rank >= keep) {
    return kill_vertex();
  }
  let fade_w = clamp((keep - rank) / (0.35 * keep), 0.0, 1.0);
  let amp = min(card_uni.width_cap, inverseSqrt(keep));

  // Full candidate (shared twin) — exact stand position/yaw/scale/phase.
  let sp = scatter_candidate(card_uni.seed, card_uni.entry_index, cell, slot);

  let scale = sp.scale;
  let cyaw = cos(sp.yaw);
  let syaw = sin(sp.yaw);
  let off_l = card_uni.card_center;
  let off_w = vec2f(cyaw * off_l.x - syaw * off_l.z, syaw * off_l.x + cyaw * off_l.z) * scale;
  let center_w = vec3f(sp.pos.x + off_w.x, sp.pos.y + off_l.y * scale, sp.pos.z + off_w.y);

  var fwd = frame.camera_pos - center_w;
  let fl = length(fwd);
  let near_w = clamp((fl - 0.35) / 0.4, 0.0, 1.0); // camera-inside-plant fade
  fwd = fwd / max(fl, 1e-4);
  var right = cross(vec3f(0.0, 1.0, 0.0), fwd);
  let rl = length(right);
  right = select(right / max(rl, 1e-5), vec3f(1.0, 0.0, 0.0), rl < 1e-3);
  let up2 = cross(fwd, right);

  // 4 unique corners, indexed 0,1,2, 1,3,2 by the card index buffer.
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let c = corners[vi];
  let half_size = card_uni.r_card * scale;
  let sz = fade_w * near_w;
  var world = center_w
    + right * (c.x * half_size * amp * sz)
    + up2 * (c.y * half_size * sz);

  // Wind: shear by normalized height so roots stay put and tips travel.
  let hf = clamp((world.y - sp.pos.y) / max(2.0 * half_size, 1e-3), 0.0, 1.0);
  world += wind_sway(sp.pos, frame.time, entry.sway, sp.phase) * hf;

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.uv = c * 0.5 + 0.5;
  out.world = world;
  out.pid = eh;
  var vf = (atan2(fwd.z, fwd.x) - sp.yaw) * 1.2732395; // / (pi/4)
  vf = vf - 8.0 * floor(vf / 8.0);
  // Per-card constants: densify exponent (a card standing in for 1/keep
  // plants) and the hard->hashed alpha-test blend factor.
  let gamma = 1.0 + (amp - 1.0) * card_uni.densify;
  let sigma = smoothstep(card_uni.stoch_start, card_uni.stoch_full, d);
  out.yaw_gs = vec4f(cyaw, syaw, gamma, sigma);
  out.axes = vec4f(right.x, right.z, up2.x, up2.z);
  out.top_vf = vec2f(smoothstep(0.30, 0.95, fwd.y), vf);
  out.ring_id = r;
  return out;
}

fn tile_uv(idx: u32, uvl: vec2f) -> vec2f {
  let cuv = clamp(uvl, vec2f(0.004), vec2f(0.996));
  return (cuv + vec2f(f32(idx % 3u), f32(idx / 3u))) / 3.0;
}

fn tap_albedo(idx: u32, uvl: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  return textureSampleGrad(atlas_albedo, atlas_samp, tile_uv(idx, uvl), gx, gy);
}

fn tap_normal(idx: u32, uvl: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  return textureSampleGrad(atlas_normal, atlas_samp, tile_uv(idx, uvl), gx, gy);
}

/// Map the card plane into the plant-local top frame (foreshortened by the
/// projected card axes, de-rotated by yaw). Only evaluated inside the flat
/// `top_vf.x > 0` branch, so it costs nothing at grazing angles.
fn top_uv(uv_t: vec2f, axes: vec4f, cy: f32, sy: f32) -> vec2f {
  let q = uv_t - 0.5;
  let wxz = axes.xy * q.x - axes.zw * q.y;
  return vec2f(cy * wxz.x + sy * wxz.y, -sy * wxz.x + cy * wxz.y) + 0.5;
}

/**
 * CARPET fragment path. The tile texture is a top-down capture of the tile's
 * own period: rgb = albedo*coverage / a = coverage, plus (normal, height) in
 * the second texture, both premultiplied so the mip chain averages statistics.
 *
 * Two hard tests, no dither anywhere: coverage against a carpet-specific
 * reference (a mat must stay a solid depth-writing occluder — the technique's
 * stochastic realization is for thinning ensembles of separate plants, not for
 * a closed surface), and the cushion height against this shell's band.
 */
fn fs_carpet(in: VOut) -> vec4f {
  let uv = in.uv;
  let gx = dpdx(uv);
  let gy = dpdy(uv);
  let alb = textureSampleGrad(atlas_albedo, atlas_samp, uv, gx, gy);
  let cov = alb.a;
  if (cov < card_uni.cover_ref) { discard; }
  let nh = textureSampleGrad(atlas_normal, atlas_samp, uv, gx, gy);
  // Cushion top height at this texel, coverage-weighted (mip-safe).
  let h0 = nh.a / cov;
  if (h0 < in.yaw_gs.z) { discard; }

  let entry = stand_table[card_uni.entry_index];
  let albedo = alb.rgb / cov;
  var n_l = nh.rgb / cov * 2.0 - 1.0;
  n_l = select(n_l, vec3f(0.0, 1.0, 0.0), dot(n_l, n_l) < 1e-4);
  n_l = normalize(n_l);

  // Cushion-scale relief from the baked height field. The leaf-scale normals
  // average toward straight up as the mip level rises (that is what an average
  // of a two-sided leaf canopy IS), so a mat 1m away lights as a flat sheet
  // with fine noise. The height field mips into the CUSHION shape, and its
  // gradient — taken one screen pixel apart, so the scale always follows what
  // the screen can resolve — puts the dome shading back.
  if (card_uni.relief > 0.0) {
    let step_uv = max(max(max(abs(gx.x), abs(gx.y)), max(abs(gy.x), abs(gy.y))), 1.0 / 512.0);
    let hu = textureSampleGrad(atlas_normal, atlas_samp, uv + vec2f(step_uv, 0.0), gx, gy);
    let hv = textureSampleGrad(atlas_normal, atlas_samp, uv + vec2f(0.0, step_uv), gx, gy);
    let cu = textureSampleGrad(atlas_albedo, atlas_samp, uv + vec2f(step_uv, 0.0), gx, gy).a;
    let cv = textureSampleGrad(atlas_albedo, atlas_samp, uv + vec2f(0.0, step_uv), gx, gy).a;
    // Normalized height per uv unit -> world slope. The tile scale cancels:
    // (h_range*scale) / (footprint*scale).
    let k = card_uni.moss_h_range / max(entry.footprint_m, 1e-3) / step_uv;
    let du = (hu.a / max(cu, 1e-3) - h0) * k;
    let dv = (hv.a / max(cv, 1e-3) - h0) * k;
    let g = clamp(vec2f(du, dv), vec2f(-3.0), vec2f(3.0)) * card_uni.relief;
    n_l = normalize(n_l + vec3f(-g.x, 0.0, -g.y));
  }

  // Ground frame per fragment: the mat lights as the slope it lies on.
  // slope_align says HOW MUCH the species conforms (1 for a carpet).
  let ts = terrain_sample(in.world.xz);
  let tn = vec3f(ts.y, sqrt(max(1.0 - ts.y * ts.y - ts.z * ts.z, 0.0)), ts.z);
  let align = clamp(entry.slope_align, 0.0, 1.0);
  let up = normalize(mix(vec3f(0.0, 1.0, 0.0), tn, align));
  var tang = vec3f(in.yaw_gs.x, 0.0, -in.yaw_gs.y);
  let proj = tang - up * dot(up, tang);
  tang = select(vec3f(1.0, 0.0, 0.0), normalize(proj), dot(proj, proj) > 1e-6);
  let n_w = normalize(tang * n_l.x + up * n_l.y + cross(tang, up) * n_l.z);

  var color = light_surface(albedo, n_w, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
    if (card_uni.debug_rings == 1u) {
      var shell_tints = array<vec3f, 6>(
        vec3f(0.2, 0.3, 1.0), vec3f(0.2, 1.0, 0.6), vec3f(1.0, 1.0, 0.2),
        vec3f(1.0, 0.5, 0.1), vec3f(1.0, 0.2, 0.2), vec3f(1.0, 1.0, 1.0),
      );
      color = mix(color, shell_tints[min(in.ring_id, 5u)], 0.55);
    }
  }
  return vec4f(debug_shade(color, albedo, n_w, cov, in.world), 1.0);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  if (card_uni.carpet_flag == 1u) { return fs_carpet(in); }
  let cy = in.yaw_gs.x;
  let sy = in.yaw_gs.y;
  let gamma = in.yaw_gs.z;
  let sigma = in.yaw_gs.w;
  let ktop = in.top_vf.x;

  let uv_t = vec2f(in.uv.x, 1.0 - in.uv.y);
  let gx = dpdx(uv_t) / 3.0;
  let gy = dpdy(uv_t) / 3.0;

  // Two nearest azimuth views, then the top view by elevation. Coverage is
  // resolved from the ALBEDO atlas alone — the normal atlas is only fetched
  // for fragments that survive both alpha tests.
  let i0 = u32(floor(in.top_vf.y)) % 8u;
  let i1 = (i0 + 1u) % 8u;
  let t = fract(in.top_vf.y);
  var alb = mix(tap_albedo(i0, uv_t, gx, gy), tap_albedo(i1, uv_t, gx, gy), t);
  if (ktop > 0.004) {
    alb = mix(alb, tap_albedo(8u, top_uv(uv_t, in.axes, cy, sy), gx, gy), ktop);
  }

  let a = alb.a;
  if (a < 0.004) { discard; }

  // Opacity as statistics: densify amplified clumps (a card standing in for
  // 1/keep plants raises its interior coverage like independent overlap
  // would), then realize the fractional coverage with a stable hashed test.
  let a_eff = 1.0 - pow(1.0 - min(a, 0.995), gamma);
  let px = vec2u(clamp(uv_t, vec2f(0.0), vec2f(0.999)) * 128.0);
  let xi = hash_f32(hash3(in.pid, px.x, px.y));
  let tau = mix(0.5, mix(0.03, 0.97, xi), sigma);
  if (a_eff < tau) { discard; }

  var nrm = mix(tap_normal(i0, uv_t, gx, gy), tap_normal(i1, uv_t, gx, gy), t);
  if (ktop > 0.004) {
    nrm = mix(nrm, tap_normal(8u, top_uv(uv_t, in.axes, cy, sy), gx, gy), ktop);
  }

  let mean_rgb = alb.rgb / a; // un-premultiply: mean radiance of covered area
  var n_l = nrm.rgb / max(nrm.a, 0.01) * 2.0 - 1.0;
  n_l = select(n_l, vec3f(0.0, 1.0, 0.0), dot(n_l, n_l) < 1e-4);
  n_l = normalize(n_l);
  let n_w0 = vec3f(cy * n_l.x - sy * n_l.z, n_l.y, sy * n_l.x + cy * n_l.z);
  // Far cards drift toward the statistical canopy normal (up-dominated).
  let n_w = normalize(mix(n_w0, vec3f(0.0, 1.0, 0.0), 0.25 * sigma));

  let albedo = mean_rgb * (0.62 + 0.38 * in.uv.y); // vertical card AO
  var color = light_surface(albedo, n_w, in.world);

  // Fog and the ring tint only in the normal view — debug views stay honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
    if (card_uni.debug_rings == 1u) {
      var ring_tints = array<vec3f, 6>(
        vec3f(1.0, 0.2, 0.2), vec3f(1.0, 1.0, 0.2), vec3f(0.2, 1.0, 0.2),
        vec3f(0.2, 0.6, 1.0), vec3f(0.8, 0.2, 1.0), vec3f(1.0, 1.0, 1.0),
      );
      color = mix(color, ring_tints[in.ring_id], 0.55);
    }
  }
  // Coverage view shows the densified statistic the hashed test realizes.
  return vec4f(debug_shade(color, albedo, n_w, a_eff, in.world), 1.0);
}
