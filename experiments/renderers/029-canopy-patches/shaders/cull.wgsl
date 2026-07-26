#include "src/wgsl/scatter.wgsl"
#include "./patch_info.wgsl"

// Per-frame plant selection + LOD bucketing. Evaluates the shared scatter over
// a bounded camera-centered cell region (already clamped to the stand's cell
// range on the CPU), frustum-culls each plant, and compacts survivors into TWO
// contiguous ranges of one instance array: [0, cap_near) for plants close
// enough to be drawn as a depth-layered patch stack, and [far_base, ...) for
// plants that collapsed to a single composite card. Cost is O(region area),
// never O(total plants in the stand).
//
// A workgroup is 64 invocations and a cell holds SCATTER_MAX_PER_CELL = 128
// candidate slots, so every workgroup covers exactly half of ONE cell: the
// cell-level region/frustum rejects below are workgroup-uniform and let whole
// workgroups exit before a single terrain texel is fetched.
//
// Record (12 B): xz quantised to 2x i16 around info.pos_origin (4.9 mm at
// pos_range 160 m), y as f32, and yaw 10b | scale 12b | phase 10b.

struct PlantInst {
  xz: u32,
  y: f32,
  bits: u32,
}

@group(1) @binding(0) var<uniform> info: PatchInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.side_x);
  let total = side_x * u32(info.side_z) * SCATTER_MAX_PER_CELL;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % SCATTER_MAX_PER_CELL;
  let cell_lin = idx / SCATTER_MAX_PER_CELL;
  let cx = i32(info.origin_cell_x) + i32(cell_lin % side_x);
  let cz = i32(info.origin_cell_z) + i32(cell_lin / side_x);

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);

  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > info.region_r) {
    return;
  }

  let s_max = stand_table[u32(info.entry_index)].scale_max;
  let rad_max = info.cull_radius * s_max + 0.35;
  let mid_y = (info.y0 + info.y1) * 0.5 * s_max;
  let h_bound = frame.terrain_height_scale * TERRAIN_BOUND_SLACK;
  let box_c = vec3f(cell_mid.x, mid_y * 0.5, cell_mid.y);
  let box_e = vec3f(SCATTER_CELL_SIZE * 0.5, h_bound + abs(mid_y) * 0.5 + rad_max, SCATTER_CELL_SIZE * 0.5);
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

  let dist = distance(sp.pos.xz, frame.camera_pos.xz);
  if (dist > info.region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, (info.y0 + info.y1) * 0.5 * sp.scale, 0.0);
  let rad = info.cull_radius * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  // --- LOD bucket ------------------------------------------------------------
  // The collapse is geometric (see patches.wgsl): at patch_dist the stack's
  // slabs are already coincident and camera-facing, so switching to the
  // composite card here is continuous rather than a pop.
  // TRUE 3D distance, matching the vertex shader: with the xz distance a
  // top-down camera 30 m up would treat everything under it as near.
  let is_near = distance(center, frame.camera_pos) < info.patch_dist;
  let lod = select(1u, 0u, is_near);
  let cap = select(u32(info.cap_far), u32(info.cap_near), is_near);
  let w = atomicAdd(&counters[u32(info.entry_index) * 2u + lod], 1u);
  if (w >= cap) {
    atomicSub(&counters[u32(info.entry_index) * 2u + lod], 1u); // full — count stays clamped
    return;
  }

  let q_xz = pack2x16snorm((sp.pos.xz - vec2f(info.pos_origin_x, info.pos_origin_z)) / info.pos_range);
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  let base_i = select(u32(info.far_base), 0u, is_near);
  out_insts[base_i + w] = PlantInst(q_xz, sp.pos.y, yaw_q | (scale_q << 10u) | (phase_q << 22u));
}
