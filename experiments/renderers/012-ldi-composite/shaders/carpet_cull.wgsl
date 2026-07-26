#include "src/wgsl/scatter.wgsl"
#include "./carpet_common.wgsl"

// Visibility pass for ONE carpet stand entry (a mat: stand_table.carpet_div > 0).
//
// Separate from cull.wgsl for one reason that matters: a carpet has exactly
// carpet_div^2 slots per 4 m cell — 484 for the bog's life-size Sphagnum,
// deliberately ABOVE the 128-slot scatter budget. cull.wgsl is one workgroup of
// 128 threads per cell, so on a carpet it evaluated slots 0..127 of 484 and the
// mat rendered as 1.1 m stripes with 2.9 m of bare peat between them. Here the
// z dimension of the dispatch walks the remaining slots (ceil(slots/128)
// workgroups per cell), so EVERY node of the grid is evaluated.
//
// Survivors are compacted into a 4-byte packed record (see carpet_common.wgsl).
// Near tiles (inside tune.x) fill upward from 0 and draw with per-texel depth;
// far tiles fill downward from the top of the same array and draw one flat
// layer with early-z intact. Both indirect draws keep firstInstance = 0 (a
// non-zero firstInstance silently no-ops the whole draw without the optional
// `indirect-first-instance` feature) — vs_carpet_far mirrors the descending
// fill itself.

@group(1) @binding(0) var<uniform> uni: CarpetU;
@group(1) @binding(1) var<storage, read_write> out_instances: array<u32>;
@group(1) @binding(2) var<storage, read_write> near_args: array<atomic<u32>, 4>;
@group(1) @binding(3) var<storage, read_write> far_args: array<atomic<u32>, 4>;

@compute @workgroup_size(128)
fn cs_carpet(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let slot = wg.z * 128u + li;
  if (slot >= u32(uni.grid.y)) { return; }

  let cell = vec2i(i32(uni.region.x) + i32(wg.x), i32(uni.region.y) + i32(wg.y));
  let entry = u32(uni.ids.y);
  let sp = scatter_candidate(u32(uni.ids.x), entry, cell, slot);
  if (!sp.exists) { return; }  // this node belongs to another wetness zone

  // The stand's square region is the world; nothing grows beyond it.
  let stand_r = uni.ids.w;
  if (abs(sp.pos.x) > stand_r || abs(sp.pos.z) > stand_r) { return; }

  // Bounded-region circle around the camera (matches the vertex-side fade).
  let d_xz = distance(sp.pos.xz, frame.camera_pos.xz);
  if (d_xz > uni.region.w) { return; }

  // Conservative sphere-vs-frustum in clip space. The sphere covers the tile
  // footprint (half width, corner to corner) plus the cushion's own relief.
  let mid = vec3f(sp.pos.x, sp.pos.y + uni.geom.z * 0.5, sp.pos.z);
  let r = uni.geom.y * 1.45 + uni.geom.z;
  let c = frame.view_proj * vec4f(mid, 1.0);
  let rx = r * frame.proj[0][0];
  let ry = r * frame.proj[1][1];
  if (c.w < -r) { return; }
  if (c.x > c.w + rx || c.x < -c.w - rx) { return; }
  if (c.y > c.w + ry || c.y < -c.w - ry) { return; }

  let quad = u32(sp.yaw / CARPET_QUARTER + 0.5) & 3u;
  let packed = carpet_pack(wg.x, wg.y, slot, quad);
  let cap = u32(uni.ids.z);
  var out_slot: u32;
  if (distance(mid, frame.camera_pos) < uni.tune.x) {
    out_slot = atomicAdd(&near_args[1], 1u);
    if (out_slot >= cap) {
      atomicSub(&near_args[1], 1u);
      return;
    }
  } else {
    let k = atomicAdd(&far_args[1], 1u);
    if (k >= cap) {
      atomicSub(&far_args[1], 1u);
      return;
    }
    out_slot = cap - 1u - k;
  }
  out_instances[out_slot] = packed;
}
