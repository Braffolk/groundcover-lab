#include "src/wgsl/scatter.wgsl"
#include "./info.wgsl"

// Per-frame plant selection: evaluates the shared scatter over a bounded
// camera-centered cell region (already clamped to the stand's cell range on
// the CPU), frustum-culls each plant's bounding sphere, and compacts survivors
// into TWO instance lists — near (parallax-warped cards) and far (flat cards).
// Cost is O(region area), never O(total plants in the stand).
//
// A workgroup is 64 invocations. A scattered entry holds SCATTER_MAX_PER_CELL
// = 128 candidate slots per cell, so a workgroup covers half of ONE cell; a
// CARPET entry holds carpet_div^2 instead (484 at Sphagnum life size, well over
// the scatter budget) and simply takes more workgroups per cell. Either way the
// slot count comes from info.carpet.x — hardcoding 128 renders about a quarter
// of a mat, in bands, which looks exactly like a placement bug. The cell-level
// region/frustum rejects below stay workgroup-uniform, so whole workgroups exit
// before a single terrain texel is fetched.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b (0..4 m) | phase 10b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_near: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> out_far: array<PlantInst>;
@group(1) @binding(3) var<storage, read_write> args_near: array<atomic<u32>, 4>;
@group(1) @binding(4) var<storage, read_write> args_far: array<atomic<u32>, 4>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015.
const TERRAIN_BOUND_SLACK: f32 = 1.05;
const WIND_MARGIN: f32 = 0.35;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(info.region.z);
  let slots = u32(info.carpet.x);
  let total = side_x * u32(info.region.w) * slots;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let slot = idx % slots;
  let cell_lin = idx / slots;
  let cx = i32(info.region.x) + i32(cell_lin % side_x);
  let cz = i32(info.region.y) + i32(cell_lin / side_x);
  let region_r = info.ids.z;

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > region_r) {
    return;
  }

  // A mat is rigid (sway 0), so it needs no wind margin on its bounds.
  let margin = select(WIND_MARGIN, 0.02, info.carpet.z > 0.5);
  let s_max = stand_table[u32(info.ids.y)].scale_max;
  let rad_max = info.caps.w * s_max + margin;
  let mid_y = info.aabb_c.y * s_max;
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
  let sp = scatter_candidate(u32(info.ids.x), u32(info.ids.y), vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }
  let d_xz = distance(sp.pos.xz, frame.camera_pos.xz);
  if (d_xz > region_r) {
    return;
  }

  let center = sp.pos + vec3f(0.0, info.aabb_c.y * sp.scale, 0.0);
  let rad = info.caps.w * sp.scale + margin;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return;
    }
  }

  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  let packed = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u));

  // Near = full parallax warp; far = flat card, half the taps. The split is a
  // 3D distance to the plant's centre so a plant directly overhead is "near".
  if (distance(center, frame.camera_pos) < info.ids.w) {
    let w = atomicAdd(&args_near[1], 1u);
    if (w >= u32(info.caps.x)) {
      atomicSub(&args_near[1], 1u);
      return;
    }
    out_near[w] = packed;
  } else {
    let w = atomicAdd(&args_far[1], 1u);
    if (w >= u32(info.caps.y)) {
      atomicSub(&args_far[1], 1u);
      return;
    }
    out_far[w] = packed;
  }
}
