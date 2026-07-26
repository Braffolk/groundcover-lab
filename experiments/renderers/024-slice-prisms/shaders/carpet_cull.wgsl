#include "src/wgsl/scatter.wgsl"
#include "./carpet_info.wgsl"

// Per-frame tile selection for a CARPET entry (stand_table.carpet_div > 0).
//
// Same shape as cull.wgsl — bounded camera-centred cell region, cell-level
// rejects first, exact per-tile tests after, two compacted lists (near = the
// slice stack, far = one merged card) — with three carpet-specific changes:
//
//  * SLOT COUNT. A carpet entry has carpet_div^2 slots per cell (484 at life
//    size), NOT the 128-slot scatter budget. info.slots_per_cell carries that,
//    rounded up to a multiple of 64 so a workgroup still covers one cell and
//    the cell-level rejects stay workgroup-uniform; the padding slots fall out
//    of scatter_candidate() as "does not exist".
//  * 4-BYTE INSTANCES. A tile is fully described by (cell, slot, quarter turn):
//    its position is the grid node, its scale is the entry's constant, and it
//    has no wind phase. Packing that into ONE u32 instead of a 16B PlantInst
//    is what keeps ~600k tiles per species inside a couple of MB, which is
//    where the VRAM belongs for a mat — in the atlas, not in the instance list.
//  * The bounding sphere is the TILE, sized from footprint_m; nothing here may
//    be derived from height_scale (a Sphagnum tile is 0.07m tall, 0.24m wide).

@group(1) @binding(0) var<uniform> info: CarpetInfo;
@group(1) @binding(1) var<storage, read_write> near_nodes: array<u32>;
@group(1) @binding(2) var<storage, read_write> far_nodes: array<u32>;
// [0..3] near draw args, [4..7] far draw args (vertexCount is CPU-written).
@group(1) @binding(3) var<storage, read_write> draw_args: array<atomic<u32>, 8>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_carpet_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slots = u32(info.slots_per_cell);
  let side_x = u32(info.side_x);
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

  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > info.region_r) {
    return;
  }

  let mat_y = (info.y0 + info.y1) * 0.5 * info.tile_scale;
  let h_bound = frame.terrain_height_scale * TERRAIN_BOUND_SLACK;
  let box_c = vec3f(cell_mid.x, mat_y * 0.5, cell_mid.y);
  let box_e = vec3f(
    SCATTER_CELL_SIZE * 0.5,
    h_bound + abs(mat_y) * 0.5 + info.cull_radius,
    SCATTER_CELL_SIZE * 0.5,
  );
  for (var i = 0; i < 6; i++) {
    let pl = info.planes[i];
    if (dot(pl.xyz, box_c) + pl.w + dot(abs(pl.xyz), box_e) < 0.0) {
      return;
    }
  }

  // --- exact per-tile tests --------------------------------------------------
  let sp = scatter_candidate(u32(info.seed), u32(info.entry_index), vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }

  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, mat_y, 0.0);
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -info.cull_radius) {
      return;
    }
  }

  // LOD distance: identical formula to carpet_sep() in carpet.wgsl (camera
  // height above its own ground), so the bucket and the stack separation can
  // never disagree at the boundary.
  let d_lod = length(vec2f(distance(sp.pos.xz, frame.camera_pos.xz), info.cam_height - mat_y));

  // cell (10b + 10b, +-512 cells = +-2048m) | slot (10b) | quarter turn (2b).
  // The draw rebuilds the grid node from this with the same arithmetic the
  // scatter uses, so nothing about placement is re-decided here.
  let quarter = u32(round(sp.yaw / QUARTER_TURN)) & 3u;
  let packed = (u32(cx + 512) << 22u) | (u32(cz + 512) << 12u) | (slot << 2u) | quarter;

  if (d_lod < info.slice_dist) {
    let w = atomicAdd(&draw_args[1], 1u);
    if (w >= u32(info.near_capacity)) {
      atomicSub(&draw_args[1], 1u); // full — final count stays clamped
      return;
    }
    near_nodes[w] = packed;
  } else {
    let w = atomicAdd(&draw_args[5], 1u);
    if (w >= u32(info.far_capacity)) {
      atomicSub(&draw_args[5], 1u);
      return;
    }
    far_nodes[w] = packed;
  }
}
