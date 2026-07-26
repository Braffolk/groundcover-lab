#include "src/wgsl/scatter.wgsl"
#include "./impostor_info.wgsl"

// Per-frame tile selection for a CARPET entry (stand_table[i].carpet_div > 0).
//
// Why this is a separate pass from cs_cull and not a branch inside it:
//
//  * a carpet has carpet_div^2 slots per scatter cell (484 for the bog moss),
//    NOT SCATTER_MAX_PER_CELL. One workgroup of 128 lanes can only look at 128
//    slots, so the dispatch gets a THIRD dimension — one workgroup per
//    (cell, slot chunk) — and the caller sizes it from standEntrySlots().
//    Driving this from 128 renders a quarter of the mat.
//  * a surviving tile needs FOUR BYTES, not sixteen. Its position is the grid
//    node (cell + slot), its scale is the entry's constant, and its yaw is one
//    of four quarter turns, so the whole record is a packed index. At ~890k
//    tiles inside a 128 m region that is the difference between 3.6 MB and
//    14 MB per species.
//
// Everything else matches cs_cull: workgroup-uniform cell rejects first (so a
// cell that cannot contribute exits before a hash round or a heightmap fetch),
// then the exact per-node test, then compaction into four front-to-back
// distance buckets.

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_recs: array<u32>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(128)
fn cs_cull_carpet(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let entry_index = u32(info.setup.y);
  let entry = stand_table[entry_index];
  let div = u32(entry.carpet_div);
  let slot = wg.z * 128u + lane;
  if (slot >= div * div) {
    return;
  }
  let cx = i32(info.region.x) + i32(wg.x);
  let cz = i32(info.region.y) + i32(wg.y);
  let region_r = info.setup.z;
  let scale = entry.scale_min;
  // The tile is a square of exactly one grid step (footprint_m * scale), lying
  // on the ground: half-diagonal plus the canopy height bounds it.
  let step = entry.footprint_m * scale;
  let canopy = entry.footprint_m * CARPET_LIFT_FRAC;   // matches carpet.wgsl's proxy plane
  let rad = step * 0.7072 + canopy + 0.02;

  // --- cell-level rejects (workgroup-uniform, no memory traffic) -------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > region_r) {
    return;
  }
  let h_bound = frame.terrain_height_scale * TERRAIN_BOUND_SLACK;
  let box_c = vec3f(cell_mid.x, 0.0, cell_mid.y);
  let box_e = vec3f(SCATTER_CELL_SIZE * 0.5 + rad, h_bound + rad, SCATTER_CELL_SIZE * 0.5 + rad);
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, box_c) + pl.w + dot(abs(pl.xyz), box_e) < 0.0) {
      return;
    }
  }

  // --- exact per-node test ---------------------------------------------------
  // The shared scatter decides which of the three Sphagnum states owns this
  // grid node (the wetness partition) and which quarter turn it takes.
  let sp = scatter_candidate(u32(info.setup.x), entry_index, vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }
  let d_xz = distance(sp.pos.xz, frame.camera_pos.xz);
  if (d_xz > region_r) {
    return;
  }
  let centre = sp.pos + vec3f(0.0, canopy, 0.0);
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, centre) + pl.w < -rad) {
      return;
    }
  }

  let dist = distance(centre, frame.camera_pos);
  let br = info.bucket_r;
  var bucket = 0u;
  if (dist > br.x) { bucket = 1u; }
  if (dist > br.y) { bucket = 2u; }
  if (dist > br.z) { bucket = 3u; }

  let caps = info.bucket_cap;
  let bases = info.bucket_base;
  let w = atomicAdd(&draw_args[bucket * 4u + 1u], 1u);
  if (w >= u32(caps[bucket])) {
    atomicSub(&draw_args[bucket * 4u + 1u], 1u); // full — the count stays clamped
    return;
  }
  // slot 9b | cell x within the region 7b | cell z 7b | quarter turn 2b.
  let quarter = u32(sp.yaw * (2.0 / 3.14159265) + 0.5) & 3u;
  let rec = slot | (wg.x << 9u) | (wg.y << 16u) | (quarter << 23u);
  out_recs[u32(bases[bucket]) + w] = rec;
}
