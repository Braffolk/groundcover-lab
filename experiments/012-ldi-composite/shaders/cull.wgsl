#include "src/wgsl/scatter.wgsl"
#include "./ldi_common.wgsl"

// Per-frame visibility pass. One workgroup = one scatter cell (128 candidate
// slots); the region is a bounded window of cells around the camera, so the
// dispatch size depends ONLY on regionRadius — never on the stand's total
// plant count. Survivors (existing + inside stand + inside region circle +
// inside frustum) are compacted into ONE instance array shared by two
// indirect draws: NEAR plants (full 12-card LDI, frag_depth) fill from index
// 0 up, FAR plants (2 cheap cards, early-z friendly) fill from the top end
// down — the far draw's firstInstance is maintained with the same atomics.

@group(1) @binding(0) var<uniform> uni: SpeciesU;
@group(1) @binding(1) var<storage, read_write> out_instances: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> near_args: array<atomic<u32>, 4>;
@group(1) @binding(3) var<storage, read_write> far_args: array<atomic<u32>, 4>;

@compute @workgroup_size(128)
fn cs_cull(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let cell = vec2i(i32(uni.region.x) + i32(wg.x), i32(uni.region.y) + i32(wg.y));
  let entry = u32(uni.fade.w);
  let sp = scatter_candidate(u32(uni.region.w), entry, cell, li);
  if (!sp.exists) { return; }

  // The stand's square region is the world; nothing grows beyond it.
  let stand_r = uni.misc.z;
  if (abs(sp.pos.x) > stand_r || abs(sp.pos.z) > stand_r) { return; }

  let center_w = sp.pos + rot_yaw(uni.center.xyz, sp.yaw) * sp.scale;
  let r = uni.center.w * sp.scale + 0.45; // + wind sway margin

  // Bounded-region circle around the camera.
  let d_xz = distance(center_w.xz, frame.camera_pos.xz);
  if (d_xz > uni.fade.x) { return; }

  // Conservative sphere-vs-frustum in clip space.
  let c = frame.view_proj * vec4f(center_w, 1.0);
  let rx = r * frame.proj[0][0];
  let ry = r * frame.proj[1][1];
  if (c.w < -r) { return; }
  if (c.x > c.w + rx || c.x < -c.w - rx) { return; }
  if (c.y > c.w + ry || c.y < -c.w - ry) { return; }

  var slot: u32;
  if (distance(center_w, frame.camera_pos) < uni.fade.y) {
    slot = atomicAdd(&near_args[1], 1u);
    if (slot >= u32(uni.misc.y)) {
      atomicSub(&near_args[1], 1u);
      return;
    }
  } else {
    let first = atomicSub(&far_args[3], 1u);
    if (first == 0u) {
      atomicAdd(&far_args[3], 1u);
      return;
    }
    slot = first - 1u;
    atomicAdd(&far_args[1], 1u);
  }
  out_instances[slot * 2u] = vec4f(sp.pos, sp.yaw);
  out_instances[slot * 2u + 1u] = vec4f(sp.scale, sp.phase, f32(entry), 0.0);
}
