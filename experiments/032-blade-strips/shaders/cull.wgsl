#include "src/wgsl/scatter.wgsl"
#include "./strip_info.wgsl"

// Per-frame plant selection + LOD bucketing. Evaluates the shared scatter over
// a bounded camera-centered cell region (already clamped to the stand's cell
// range on the CPU), frustum-culls each plant's bounding sphere, classifies it
// by PROJECTED DIAMETER IN PIXELS into one of five strip-count buckets
// (16/8/4/2/1 ribbons) and compacts it into that bucket's instance range +
// indexed-indirect draw args. Cost is O(region area), never O(plants in the
// stand); a plant's cost collapses with its screen size, which is what keeps
// the total flat as the meadow fills the screen.
//
// A workgroup is 64 invocations and a cell holds SCATTER_MAX_PER_CELL = 128
// candidate slots, so every workgroup covers exactly half of ONE cell: the
// cell-level region/frustum rejects are workgroup-uniform and let whole
// workgroups exit before a single terrain texel is fetched.

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
// 5 buckets x 8 u32: [indexCount, instanceCount, firstIndex, baseVertex,
// firstInstance, pad, pad, pad] — only instanceCount is touched here.
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.region.z);
  let total = side_x * u32(info.region.w) * SCATTER_MAX_PER_CELL;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % SCATTER_MAX_PER_CELL;
  let cell_lin = idx / SCATTER_MAX_PER_CELL;
  let cx = i32(info.region.x) + i32(cell_lin % side_x);
  let cz = i32(info.region.y) + i32(cell_lin / side_x);
  let entry_index = u32(info.ids.y);
  let region_r = info.ids.z;

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);

  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > region_r) {
    return;
  }

  let s_max = stand_table[entry_index].scale_max;
  let rad_max = info.ids.w * s_max + 0.35;
  let mid_y = (info.box.x + info.box.y) * 0.5 * s_max;
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
  let sp = scatter_candidate(u32(info.ids.x), entry_index, vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }
  if (distance(sp.pos.xz, frame.camera_pos.xz) > region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, (info.box.x + info.box.y) * 0.5 * sp.scale, 0.0);
  let rad = info.ids.w * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  // --- LOD bucket from projected diameter -----------------------------------
  let dist = max(distance(center, frame.camera_pos), 0.05);
  let px = (info.ids.w * sp.scale / dist) * info.box.w;
  var bucket = N_BUCKETS - 1u;
  for (var b = 0u; b < N_BUCKETS - 1u; b++) {
    if (px >= info.thresholds[b]) {
      bucket = b;
      break;
    }
  }

  let cap = u32(info.caps[bucket >> 2u][bucket & 3u]);
  let base = u32(info.bases[bucket >> 2u][bucket & 3u]);
  let w = atomicAdd(&draw_args[bucket * 8u + 1u], 1u);
  if (w >= cap) {
    atomicSub(&draw_args[bucket * 8u + 1u], 1u); // full — final count stays clamped
    return;
  }
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  out_insts[base + w] = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));
}
