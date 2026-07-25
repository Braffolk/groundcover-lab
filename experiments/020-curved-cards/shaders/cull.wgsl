#include "src/wgsl/scatter.wgsl"
#include "./card_info.wgsl"

// Per-frame plant selection + LOD binning. Evaluates the shared scatter over a
// bounded camera-centered cell region (already clamped to the stand's cell
// range on the CPU), frustum-culls each plant's bounding sphere, and compacts
// survivors into ONE of three distance rings inside a single instance buffer,
// each ring feeding its own drawIndexedIndirect. Cost is O(region area), never
// O(total plants in the stand).
//
// A workgroup is 64 invocations. A scattered entry holds SCATTER_MAX_PER_CELL
// = 128 candidate slots per cell, but a CARPET entry holds carpet_div^2 (484
// at life size) — deliberately over the scatter budget — so the slot count
// comes from info.ring.z, never from the global constant. Hardcoding 128 would
// enumerate the first ~6 of 22 tile rows in every cell and render the mat as
// bands. The cell-level region/frustum rejects below stay workgroup-uniform
// and let whole workgroups exit before a single terrain texel is fetched.
//
// The ring index is the LOD: ring 0 draws a 5x7 curved patch + 4x4 canopy
// patch, ring 1 a 3x5 + 3x3, ring 2 a single flat quad + flat top card. The
// binning metric is distance / plant scale, i.e. inverse projected size, so a
// big plant keeps its relief further out than a small one.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: CardInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>, 16>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.side_x);
  let slots = u32(info.ring.z);
  let total = side_x * u32(info.side_z) * slots;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % slots;
  let cell_lin = idx / slots;
  let cx = i32(info.origin_cell.x) + i32(cell_lin % side_x);
  let cz = i32(info.origin_cell.y) + i32(cell_lin / side_x);

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);

  // Circular region edge: closest point of the cell rect to the camera.
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > info.region_r) {
    return;
  }

  // Frustum vs a conservative box over every plant this cell can hold:
  // cell footprint in xz, terrain bound + tallest card + wind margin in y.
  let s_max = stand_table[u32(info.entry_index)].scale_max;
  let rad_max = info.cull_radius * s_max + 0.35;
  let mid_y = (info.card_y0 + info.card_y1) * 0.5 * s_max;
  let h_bound = frame.terrain_height_scale * TERRAIN_BOUND_SLACK;
  let box_c = vec3f(cell_mid.x, mid_y * 0.5, cell_mid.y);
  let box_e = vec3f(
    SCATTER_CELL_SIZE * 0.5,
    h_bound + abs(mid_y) * 0.5 + rad_max,
    SCATTER_CELL_SIZE * 0.5,
  );
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, box_c) + pl.w + dot(abs(pl.xyz), box_e) < 0.0) {
      return;
    }
  }

  // --- exact per-plant tests -------------------------------------------------
  let sp = scatter_candidate(u32(info.seed), u32(info.entry_index), vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }

  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, (info.card_y0 + info.card_y1) * 0.5 * sp.scale, 0.0);
  let rad = info.cull_radius * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  // --- LOD ring binning ------------------------------------------------------
  let metric = distance(center, frame.camera_pos) / max(sp.scale, 0.05);
  var ring = 2u;
  var base = info.lod_base.z;
  var cap = info.lod_cap.z;
  if (metric < info.lod_dist.x) {
    ring = 0u;
    base = info.lod_base.x;
    cap = info.lod_cap.x;
  } else if (metric < info.lod_dist.y) {
    ring = 1u;
    base = info.lod_base.y;
    cap = info.lod_cap.y;
  }

  let w = atomicAdd(&draw_args[ring * 5u + 1u], 1u);
  if (w >= u32(cap)) {
    atomicSub(&draw_args[ring * 5u + 1u], 1u); // full — final count stays clamped
    return;
  }
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  out_insts[u32(base) + w] = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));
}
