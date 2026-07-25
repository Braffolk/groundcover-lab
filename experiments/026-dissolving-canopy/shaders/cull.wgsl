#include "src/wgsl/scatter.wgsl"
#include "./common.wgsl"

// Per-frame plant selection for the splat passes. Evaluates the shared scatter
// over a bounded camera-centered cell region (already clamped to the stand's
// cell range on the CPU), rejects plants the dissolve field has already handed
// over to the continuous shell, frustum-culls the rest and compacts them into
// TWO segments of one instance buffer:
//   band 0 (near): drawn as 4 sub-tuft splats  -> 24 vertices
//   band 1 (far):  drawn as 1 whole-plant splat ->  6 vertices
// The segments are radial, so drawing band 0 before band 1 is also a coarse
// front-to-back order (early-z keeps the far band cheap).
//
// Cost is O(region area), never O(total plants in the stand): a workgroup is 64
// invocations and a cell holds SCATTER_MAX_PER_CELL = 128 candidate slots, so
// every workgroup covers exactly half of ONE cell and the two cell-level
// rejects below are workgroup-uniform — whole workgroups exit before a single
// heightmap texel is fetched.

struct PlantInst {
  pos: vec3f,
  bits: u32, // yaw 10b | scale 11b (0..4 m) | phase 10b | band 1b
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out_insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: array<atomic<u32>, 8>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015.
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
  let region_r = info.cfg0.z;

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > region_r) {
    return;
  }

  let entry = stand_table[u32(info.cfg0.y)];
  let s_max = entry.scale_max;
  let rad_max = info.cfg3.x * s_max + 0.35;
  let mid_y = info.cfg2.z * 0.5 * s_max;
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
  let sp = scatter_candidate(u32(info.cfg0.x), u32(info.cfg0.y), vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }
  if (distance(sp.pos.xz, frame.camera_pos.xz) > region_r) {
    return;
  }

  // Already dissolved into the shell? Then every fragment of every splat of
  // this plant would discard — do not emit it at all. This is the LOD collapse:
  // looking down, the radius shrinks and the sprite set empties out.
  let mid = sp.pos + vec3f(0.0, info.cfg2.z * sp.scale * 0.5, 0.0);
  let d = dissolve_amount(mid, info.cfg2.x, info.cfg2.y);
  if (tuft_fade(d) < info.cfg1.z) {
    return;
  }

  let rad = info.cfg3.x * sp.scale + 0.35;
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, mid) + info.planes[i].w < -rad) {
      return;
    }
  }

  let near = distance(sp.pos.xz, frame.camera_pos.xz) < info.cfg0.w;
  let band = select(1u, 0u, near);
  let cap = select(u32(info.cfg1.y), u32(info.cfg1.x), near);
  let base = select(u32(info.cfg1.x), 0u, near);
  let w = atomicAdd(&draw_args[band * 4u + 1u], 1u);
  if (w >= cap) {
    atomicSub(&draw_args[band * 4u + 1u], 1u); // full — final count stays clamped
    return;
  }
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 2047.0) & 2047u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  out_insts[base + w] = PlantInst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 21u) | (band << 31u));
}
