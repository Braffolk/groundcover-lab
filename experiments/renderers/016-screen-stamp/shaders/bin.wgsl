#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"

// SCREEN-TILE BINNING — the inverted loop. One workgroup per 16x8 screen
// tile. Instead of iterating plants and projecting them to the screen, each
// tile works out which plants project INTO IT:
//
//   1. reduce the tile's 128 scene-depth texels -> visible-ground cell bbox,
//      max ground distance, sky presence (the terrain base pass has already
//      written depth, so the depth buffer IS the tile->world footprint map);
//   2. pick a strategy per tile:
//        enumerate — footprint is a few cells (close-up / top-down): test
//                    every scatter cell in the (padded) bbox;
//        columns   — footprint is a long grazing wedge (or sky above the
//                    horizon): walk cell columns front-to-back along the
//                    center ray's ground track, a 1-cell-padded band wide,
//                    early-stopping when the list fills or the walk passes
//                    min(max_dist, visible ground);
//        tint-only — footprint too large for individual plants (very high
//                    camera): leave the list empty, the resolve pass paints
//                    the aggregate meadow field;
//   3. every candidate SCATTERED plant (procedural scatter twin, no arrays
//      anywhere; carpet species are resolved per pixel in carpet.wgsl)
//      is sphere-tested against the tile's 4 frustum planes, wind-swayed,
//      faded (near/far), assigned its hemi-octa atlas view, packed to 32 B
//      and appended to the tile list;
//   4. thread 0 sorts the list front-to-back; threads write it out.
//
// Per-frame cost is tiles x bounded work: independent of both total plant
// count AND stand area. The per-pixel resolve pass then composites at most
// MAX_LIST precomputed lookups per pixel with early termination.

const TILE_W: u32 = 16u;
const TILE_H: u32 = 8u;
const MAX_LIST: u32 = 64u;
const HEADER_U32: u32 = 8u;
const ENTRY_U32: u32 = 8u;
const STRIDE_U32: u32 = HEADER_U32 + MAX_LIST * ENTRY_U32;
const WG_SIZE: u32 = 128u;
const MAX_COLUMNS: u32 = 30u;
const GRID_F: f32 = 10.0;          // hemi-octa atlas grid (bake.ts GRID)
const MAX_PLANT_H: f32 = 2.2;      // conservative world-space plant height
const ENUM_MAX_DIM: i32 = 4;       // bbox cell span allowed for enumerate mode
const TWO_PI_S: f32 = 6.2831853;

struct StampParams {
  dims: vec4u,                  // x = card slots, y = carpet slots, z = seed, w = stand entries
  tuning: vec4f,                // x = max_dist, y = tint_strength, z = tile overlay, w = tint coverage
  cards: vec4u,                 // stand entry index per card slot
  carpets: vec4u,               // stand entry index per carpet slot
  grid: vec4f,                  // carpet grid (see carpet.wgsl) — unused here
  entry_meta: array<vec4f, 4>,  // per card slot: impostor local center.xyz, radius
  entry_info: array<vec4f, 4>,  // per card slot: species average albedo rgb
  carpet_zone: array<vec4f, 4>, // per carpet slot: wet_lo, wet_hi, h_mean, y_range
}
@group(1) @binding(0) var<uniform> sp: StampParams;
@group(1) @binding(1) var<storage, read_write> tile_lists: array<u32>;
@group(1) @binding(2) var scene_depth: texture_depth_2d;

// Everything that is workgroup-uniform for the tile: the depth reduction AND
// the tile frustum. Both are computed ONCE by thread 0 and broadcast through
// a single workgroupUniformLoad (which barriers before and after the load),
// instead of 128 threads each redoing four inv_view_proj unprojections and
// four plane normals for the same tile.
struct Reduced {
  sky: u32,
  ground_far: f32,
  cmin: vec2i,
  cmax: vec2i,
  center_ray: vec3f,
  plane0: vec3f,
  plane1: vec3f,
  plane2: vec3f,
  plane3: vec3f,
}

var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_cells: atomic<u32>;
var<workgroup> wg_sky: atomic<u32>;
var<workgroup> wg_far_bits: atomic<u32>;
var<workgroup> wg_cmin_x: atomic<i32>;
var<workgroup> wg_cmax_x: atomic<i32>;
var<workgroup> wg_cmin_z: atomic<i32>;
var<workgroup> wg_cmax_z: atomic<i32>;
var<workgroup> wg_red: Reduced;
var<workgroup> wg_snap: u32;
var<workgroup> wg_e: array<array<u32, ENTRY_U32>, MAX_LIST>;
var<workgroup> wg_dist: array<f32, MAX_LIST>;
var<workgroup> wg_order: array<u32, MAX_LIST>;

// Per-invocation tile state (identical across the workgroup by construction).
var<private> g_planes: array<vec3f, 4>;
var<private> g_center_ray: vec3f;
var<private> g_bin_limit: f32;
var<private> g_lid: u32;

fn rot_y_v(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn hemioct_encode(v: vec3f) -> vec2f {
  let s = abs(v.x) + abs(v.y) + abs(v.z);
  let p = vec2f(v.x, v.z) / max(s, 1e-6);
  return vec2f(p.x + p.y, p.x - p.y);
}

fn pixel_ray(px: vec2f) -> vec3f {
  let ndc = vec2f(px.x / frame.viewport.x * 2.0 - 1.0, 1.0 - px.y / frame.viewport.y * 2.0);
  let wp = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  return normalize(wp.xyz / wp.w - frame.camera_pos);
}

fn plane_n(a: vec3f, b: vec3f, inward_ref: vec3f) -> vec3f {
  var n = cross(a, b);
  if (dot(n, inward_ref) < 0.0) { n = -n; }
  return normalize(n);
}

// Evaluate one procedural scatter slot of one CARD slot (carpet species never
// enter the lists — see carpet.wgsl); if the plant exists and its sphere
// intersects this tile's frustum (within the bin limit), append it.
// Hash math is a bit-identical mirror of scatter_candidate() with early outs.
fn try_plant(cell: vec2i, card_slot: u32, slot: u32) {
  let entry_index = sp.cards[card_slot];
  let entry = stand_table[entry_index];
  let h = hash4(sp.dims.z, bitcast<u32>(cell.x), bitcast<u32>(cell.y), (entry_index << 16u) ^ slot);
  let density = min(entry.density, SCATTER_MAX_DENSITY);
  if (hash_f32(hash2(h, 0u)) >= density / SCATTER_MAX_DENSITY) { return; }

  let ox = hash_f32(hash2(h, 1u));
  let oz = hash_f32(hash2(h, 2u));
  let xz = (vec2f(cell) + vec2f(ox, oz)) * SCATTER_CELL_SIZE;
  // Habitat band, exactly as scatter_candidate() resolves it (hash slot 6):
  // without this the bog's fen grasses grow on the dry hummock crests too.
  if (entry.wet_width > 0.0) {
    let wdist = abs(scatter_wetness(sp.dims.z, xz) - entry.wet_center) / entry.wet_width;
    if (hash_f32(hash2(h, 6u)) >= clamp(1.0 - wdist, 0.0, 1.0)) { return; }
  }
  let yaw = hash_f32(hash2(h, 3u)) * TWO_PI_S;
  let scale = mix(entry.scale_min, entry.scale_max, hash_f32(hash2(h, 4u)));
  let phase = hash_f32(hash2(h, 5u)) * TWO_PI_S;

  let pos = vec3f(xz.x, terrain_height(xz), xz.y);
  let im = sp.entry_meta[card_slot];
  let r_world = im.w * scale;
  let swayv = wind_sway(pos, frame.time, entry.sway, phase);
  let center = pos + rot_y_v(im.xyz * scale, yaw) + swayv * 0.6;

  let cam = frame.camera_pos;
  let rel = center - cam;
  let dcam = length(rel);
  // View-angle card shrink: from overhead, full-radius cards overlap ~40x
  // and no bounded list could hold them; clipping every card to 0.45R when
  // seen top-down bounds overlap below MAX_LIST. Per-plant (not per-tile),
  // and mirrored exactly in resolve.wgsl, so tiles always agree -> no seams.
  let updot = -rel.y / max(dcam, 1e-4);
  let k_clip = mix(1.0, 0.45, smoothstep(0.55, 0.9, updot));
  let rext = r_world * k_clip + 0.2;
  let dc = dot(rel, g_center_ray);
  if (dc < -rext || dc - rext > g_bin_limit) { return; }
  for (var i = 0u; i < 4u; i++) {
    if (dot(g_planes[i], rel) < -rext) { return; }
  }

  // The near field (< ~8 m) is owned by the near-quad pass (near.wgsl):
  // stamping starts where the quads crossfade out.
  if (dcam < 6.5) { return; }
  let max_dist = sp.tuning.x;
  let far_fade = 1.0 - smoothstep(max_dist * 0.72, max_dist, dcam);
  let near_fade = smoothstep(6.5, 8.0, dcam);
  let fade = far_fade * near_fade;
  if (fade < 0.01) { return; }

  // Snap the plant-local viewing direction to its nearest hemi-octa node.
  var cam_local = rot_y_v(-rel / max(dcam, 1e-4), -yaw);
  cam_local.y = max(cam_local.y, 0.03);
  cam_local = normalize(cam_local);
  let e2 = hemioct_encode(cam_local);
  let g = clamp((e2 * 0.5 + 0.5) * (GRID_F - 1.0), vec2f(0.0), vec2f(GRID_F - 1.0));
  let ni = u32(round(g.x));
  let nj = u32(round(g.y));

  let idx = atomicAdd(&wg_count, 1u);
  if (idx >= MAX_LIST) { return; }

  let yaw8 = u32(round(yaw * (255.0 / TWO_PI_S))) & 255u;
  let fade8 = u32(round(fade * 255.0));
  // Low 3 bits are the CARD SLOT (which atlas pair), not the stand entry index.
  let packed = card_slot | (ni << 3u) | (nj << 8u) | (yaw8 << 13u) | (fade8 << 21u);
  wg_e[idx][0] = bitcast<u32>(center.x);
  wg_e[idx][1] = bitcast<u32>(center.y);
  wg_e[idx][2] = bitcast<u32>(center.z);
  wg_e[idx][3] = bitcast<u32>(r_world);
  wg_e[idx][4] = packed;
  wg_e[idx][5] = pack2x16float(vec2f(swayv.x, swayv.z));
  wg_e[idx][6] = bitcast<u32>(dcam);
  wg_e[idx][7] = pack2x16float(vec2f(pos.y, entry.height_scale * scale));
  wg_dist[idx] = dcam;
}

// Cooperatively test every scatter slot of one cell (cell-level sphere cull
// first — evaluated redundantly per thread, SIMD-coherent, no barriers).
fn process_cell(cell: vec2i) {
  // The list is full: try_plant() can only atomicAdd past MAX_LIST and throw
  // the result away, so stop evaluating scatter slots entirely. wg_count only
  // grows, so once any thread sees it full no thread can append again — the
  // accepted set is bit-identical to running on. Without this, an enumerate
  // tile (close-up / top-down) keeps grinding all ~36 footprint cells x 384
  // slots even though a single 4m cell already overfills the 64-entry list.
  if (atomicLoad(&wg_count) >= MAX_LIST) { return; }
  let cc = (vec2f(cell) + 0.5) * SCATTER_CELL_SIZE;
  let ty = terrain_height(cc);
  let csph = vec3f(cc.x, ty + MAX_PLANT_H * 0.5, cc.y) - frame.camera_pos;
  let crad = 5.6; // cell half-diagonal + half plant height + plant radius + terrain slope slack
  let dc = dot(csph, g_center_ray);
  if (dc < -crad || dc - crad > g_bin_limit) { return; }
  for (var i = 0u; i < 4u; i++) {
    if (dot(g_planes[i], csph) < -crad) { return; }
  }
  if (g_lid == 0u) { atomicAdd(&wg_cells, 1u); }
  // Card slots only, and every scattered entry has exactly
  // SCATTER_MAX_PER_CELL slots (a carpet's carpet_div^2 slots are not binned).
  let n_items = sp.dims.x * SCATTER_MAX_PER_CELL;
  for (var idx = g_lid; idx < n_items; idx += WG_SIZE) {
    try_plant(cell, idx / SCATTER_MAX_PER_CELL, idx % SCATTER_MAX_PER_CELL);
  }
}

@compute @workgroup_size(WG_SIZE)
fn cs_bin(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_index) lid: u32) {
  g_lid = lid;
  let tiles_x = (u32(frame.viewport.x) + TILE_W - 1u) / TILE_W;
  let tile_xy = wg_id.xy;
  let base = (tile_xy.y * tiles_x + tile_xy.x) * STRIDE_U32;
  let cam = frame.camera_pos;
  let max_dist = sp.tuning.x;

  if (lid == 0u) {
    atomicStore(&wg_count, 0u);
    atomicStore(&wg_cells, 0u);
    atomicStore(&wg_sky, 0u);
    atomicStore(&wg_far_bits, 0u);
    atomicStore(&wg_cmin_x, 0x3fffffff);
    atomicStore(&wg_cmax_x, -0x3fffffff);
    atomicStore(&wg_cmin_z, 0x3fffffff);
    atomicStore(&wg_cmax_z, -0x3fffffff);
  }
  workgroupBarrier();

  // --- 1. depth reduce: tile footprint from the already-rendered scene ---
  let vp_max = vec2u(u32(frame.viewport.x) - 1u, u32(frame.viewport.y) - 1u);
  {
    let i = lid;
    let px = min(tile_xy * vec2u(TILE_W, TILE_H) + vec2u(i % TILE_W, i / TILE_W), vp_max);
    let d = textureLoad(scene_depth, vec2i(px), 0);
    if (d >= 0.99999) {
      atomicAdd(&wg_sky, 1u);
    } else {
      let ndc = vec2f(
        (f32(px.x) + 0.5) / frame.viewport.x * 2.0 - 1.0,
        1.0 - (f32(px.y) + 0.5) / frame.viewport.y * 2.0,
      );
      let wp = frame.inv_view_proj * vec4f(ndc, d, 1.0);
      let pos = wp.xyz / wp.w;
      atomicMax(&wg_far_bits, bitcast<u32>(distance(pos, cam)));
      atomicMin(&wg_cmin_x, i32(floor(pos.x / SCATTER_CELL_SIZE)));
      atomicMax(&wg_cmax_x, i32(floor(pos.x / SCATTER_CELL_SIZE)));
      atomicMin(&wg_cmin_z, i32(floor(pos.z / SCATTER_CELL_SIZE)));
      atomicMax(&wg_cmax_z, i32(floor(pos.z / SCATTER_CELL_SIZE)));
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    wg_red.sky = atomicLoad(&wg_sky);
    wg_red.ground_far = bitcast<f32>(atomicLoad(&wg_far_bits));
    wg_red.cmin = vec2i(atomicLoad(&wg_cmin_x), atomicLoad(&wg_cmin_z));
    wg_red.cmax = vec2i(atomicLoad(&wg_cmax_x), atomicLoad(&wg_cmax_z));
    // Tile frustum — four corner rays + four inward plane normals. Depends
    // only on workgroup_id and the frame uniform, so one thread does it.
    let x0 = f32(tile_xy.x * TILE_W);
    let y0 = f32(tile_xy.y * TILE_H);
    let x1 = x0 + f32(TILE_W);
    let y1 = y0 + f32(TILE_H);
    let r_tl = pixel_ray(vec2f(x0, y0));
    let r_tr = pixel_ray(vec2f(x1, y0));
    let r_br = pixel_ray(vec2f(x1, y1));
    let r_bl = pixel_ray(vec2f(x0, y1));
    let cray = normalize(r_tl + r_tr + r_br + r_bl);
    wg_red.center_ray = cray;
    wg_red.plane0 = plane_n(r_tl, r_bl, cray);
    wg_red.plane1 = plane_n(r_tr, r_tl, cray);
    wg_red.plane2 = plane_n(r_br, r_tr, cray);
    wg_red.plane3 = plane_n(r_bl, r_br, cray);
  }
  // NOTE: uniformity-critical values (mode, bin_limit, center_ray) are
  // function-scope lets derived only from workgroup_id, uniforms and
  // workgroupUniformLoad results, so the barriers below validate. The
  // g_* privates only feed the barrier-free helper functions.
  let red = workgroupUniformLoad(&wg_red);
  let center_ray = red.center_ray;
  g_center_ray = center_ray;
  g_planes[0] = red.plane0;
  g_planes[1] = red.plane1;
  g_planes[2] = red.plane2;
  g_planes[3] = red.plane3;

  // --- 2. strategy ---
  let has_ground = red.cmax.x >= red.cmin.x;
  let span = red.cmax - red.cmin;
  let xz_len = length(center_ray.xz);
  var mode = 0u; // 0 = tint-only, 1 = enumerate, 2 = columns
  var bin_limit = 0.0;
  if (has_ground && red.sky == 0u && max(span.x, span.y) <= ENUM_MAX_DIM) {
    mode = 1u;
    bin_limit = max_dist;
  } else if (xz_len > 0.2) {
    mode = 2u;
    bin_limit = max_dist;
    if (red.sky == 0u) { bin_limit = min(max_dist, red.ground_far + 8.0); }
    // All-sky tiles can only catch plant tops poking above a ridge; beyond
    // ~48 m those are subpixel — do not walk the whole wedge for them.
    if (red.sky >= WG_SIZE) { bin_limit = min(bin_limit, 48.0); }
  }
  g_bin_limit = bin_limit;

  // --- 3. bin ---
  if (mode == 1u) {
    let lo = red.cmin - vec2i(1);
    let hi = red.cmax + vec2i(1);
    for (var cz = lo.y; cz <= hi.y; cz++) {
      for (var cx = lo.x; cx <= hi.x; cx++) {
        process_cell(vec2i(cx, cz));
      }
    }
  } else if (mode == 2u) {
    let dir_xz = normalize(center_ray.xz);
    let major_is_z = abs(dir_xz.y) > abs(dir_xz.x);
    var dmaj = dir_xz.x;
    var dmin = dir_xz.y;
    var cam2 = vec2f(cam.x, cam.z); // (major, minor)
    if (major_is_z) {
      dmaj = dir_xz.y;
      dmin = dir_xz.x;
      cam2 = vec2f(cam.z, cam.x);
    }
    let sgn = i32(sign(dmaj));
    let limit_xz = bin_limit * max(xz_len, 0.2) + 8.0;
    let col0 = i32(floor(cam2.x / SCATTER_CELL_SIZE)) - sgn;
    for (var c = 0u; c < MAX_COLUMNS; c++) {
      let col = col0 + sgn * i32(c);
      let e0 = f32(col) * SCATTER_CELL_SIZE;
      let e1 = e0 + SCATTER_CELL_SIZE;
      let t0 = max((e0 - cam2.x) / dmaj, 0.0);
      let t1 = max((e1 - cam2.x) / dmaj, 0.0);
      if (min(t0, t1) > limit_xz) { break; }
      let mn0 = cam2.y + dmin * t0;
      let mn1 = cam2.y + dmin * t1;
      let mlo = i32(floor(min(mn0, mn1) / SCATTER_CELL_SIZE)) - 1;
      let mhi = min(i32(floor(max(mn0, mn1) / SCATTER_CELL_SIZE)) + 1, mlo + 5);
      for (var m = mlo; m <= mhi; m++) {
        var cell = vec2i(col, m);
        if (major_is_z) { cell = vec2i(m, col); }
        process_cell(cell);
      }
      workgroupBarrier();
      if (lid == 0u) { wg_snap = atomicLoad(&wg_count) | (atomicLoad(&wg_cells) << 16u); }
      let snap = workgroupUniformLoad(&wg_snap);
      // Stop when the list is full or the per-tile cell budget is spent —
      // this is the hard per-tile work bound (<= 32 cells x 3 entries x 128
      // slot evals) that keeps grazing tiles from walking the whole wedge.
      if ((snap & 0xffffu) >= MAX_LIST || (snap >> 16u) >= 32u) { break; }
    }
  }

  // --- 4. sort front-to-back + write out ---
  workgroupBarrier();
  if (lid == 0u) {
    let n = min(atomicLoad(&wg_count), MAX_LIST);
    for (var i = 0u; i < n; i++) { wg_order[i] = i; }
    for (var i = 1u; i < n; i++) {
      let oi = wg_order[i];
      let di = wg_dist[oi];
      var j = i;
      while (j > 0u && wg_dist[wg_order[j - 1u]] > di) {
        wg_order[j] = wg_order[j - 1u];
        j--;
      }
      wg_order[j] = oi;
    }
    // Tint horizon: where stamps actually stop — the global max_dist unless
    // the list overflowed (then the farthest kept plant). The depth-clamped
    // walk limit must NOT leak in here (it varies tile-to-tile at ridges and
    // would make the tint gate blocky).
    var limit_out = sp.tuning.x;
    if (mode == 0u) { limit_out = 0.0; }
    if (n >= MAX_LIST) { limit_out = min(limit_out, wg_dist[wg_order[n - 1u]]); }
    tile_lists[base + 0u] = n;
    tile_lists[base + 1u] = bitcast<u32>(limit_out);
    tile_lists[base + 2u] = mode;
    tile_lists[base + 3u] = red.sky;
  }
  workgroupBarrier();
  let n_out = min(atomicLoad(&wg_count), MAX_LIST);
  if (lid < n_out) {
    let src = wg_order[lid];
    let dst = base + HEADER_U32 + lid * ENTRY_U32;
    for (var k = 0u; k < ENTRY_U32; k++) {
      tile_lists[dst + k] = wg_e[src][k];
    }
  }
}
