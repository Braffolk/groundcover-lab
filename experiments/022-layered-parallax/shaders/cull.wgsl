#include "src/wgsl/scatter.wgsl"
#include "./entry_info.wgsl"

// Per-frame plant selection: evaluates the shared scatter over a bounded
// camera-centered cell region (already clamped to the stand's cell range on
// the CPU), frustum-culls each plant's bounding sphere, and compacts survivors
// into an instance buffer + indirect draw args. Cost is O(region area), never
// O(total plants in the stand).
//
// Slots per cell come from the ENTRY, not from SCATTER_MAX_PER_CELL: a carpet
// entry has carpet_div^2 slots (484 for the bog moss, deliberately over the
// 128 scatter budget, because div 22 is what puts a 0.18m tile at life size).
// Driving this loop from the constant would render about a quarter of the mat
// and leave holes that look exactly like a placement bug.
//
// A workgroup is 64 invocations, so with the 128-slot default every workgroup
// covers exactly half of ONE cell and the cell-level region/frustum rejects
// below are workgroup-uniform — whole workgroups exit before a single terrain
// texel is fetched. With a carpet's 484 slots a workgroup can straddle two
// cells; the rejects stay correct, they just stop being free.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> near_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> far_insts: array<PlantInst>;
// Two sets of indirect draw args: [0..3] the near ring, [4..7] the far ring.
@group(1) @binding(3) var<storage, read_write> draw_args: array<atomic<u32>, 8>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.side_x);
  let slots = u32(info.slots_per_cell);
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

  // Distance ring: near plants get the layered reprojection, everything past
  // lod_dist (scaled with the plant, so big and small plants switch at the
  // same apparent size) collapses to the merged single-tap tile. A carpet has
  // no rings — one pipeline dissolves its height bands into the merged tile
  // per texel by mip level — so all its tiles go into the one list.
  let near = info.carpet < 0.5 && distance(center, frame.camera_pos) < info.lod_dist * sp.scale;
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  let plant = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));

  if (near) {
    let w = atomicAdd(&draw_args[1], 1u);
    if (w >= u32(info.near_capacity)) {
      atomicSub(&draw_args[1], 1u); // full — final count stays clamped
      return;
    }
    near_insts[w] = plant;
  } else {
    let w = atomicAdd(&draw_args[5], 1u);
    if (w >= u32(info.capacity)) {
      atomicSub(&draw_args[5], 1u);
      return;
    }
    far_insts[w] = plant;
  }
}
