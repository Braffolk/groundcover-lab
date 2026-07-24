#include "src/wgsl/scatter.wgsl"

// Instance materialization — the ONLY place plants come into existence, and
// always over a bounded cell window (never the whole stand). Two callers:
//  - near mode (mode=1): every frame, cells of the DIRECT cluster bitmask
//    around the camera (per-frame crossed-card rendering);
//  - refresh mode (mode=0): when a cached cluster slot is rebuilt, exactly
//    that cluster's cells. This is the amortized path: it runs only for the
//    few clusters refreshed this frame, not for the visible world.
// One workgroup = one scatter cell (128 candidate slots); dispatch.z = stand
// entry. Survivors are appended to a compact instance list + indirect args.

struct FillU {
  cell0: vec2i,        // window origin in scatter-cell coords
  side: u32,           // window side in cells
  seed: u32,
  mask_origin: vec2i,  // direct bitmask origin in 8m-cluster coords
  mask_side: u32,      // bitmask side (16)
  mode: u32,           // 1 = near (bitmask filter), 0 = refresh
  stand_r: f32,
  capacity: u32,
  base_instance: u32, // range start in the shared instance scratch
  pad1: u32,
  mask0: vec4u,
  mask1: vec4u,
}

@group(1) @binding(0) var<uniform> fill: FillU;
@group(1) @binding(1) var<storage, read_write> out_instances: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> args: array<atomic<u32>, 4>;

fn direct_bit(cell: vec2i) -> bool {
  let cl = vec2i(floor(vec2f(cell) * 0.5)); // 8m cluster = 2x2 scatter cells
  let local = cl - fill.mask_origin;
  let side = i32(fill.mask_side);
  if (local.x < 0 || local.y < 0 || local.x >= side || local.y >= side) { return false; }
  let bit = u32(local.y * side + local.x);
  let word = bit >> 5u;
  var v: u32;
  if (word < 4u) { v = fill.mask0[word]; } else { v = fill.mask1[word - 4u]; }
  return ((v >> (bit & 31u)) & 1u) != 0u;
}

@compute @workgroup_size(128)
fn cs_fill(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let cell = fill.cell0 + vec2i(i32(wg.x), i32(wg.y));
  if (fill.mode == 1u && !direct_bit(cell)) { return; }
  let entry = wg.z;
  let sp = scatter_candidate(fill.seed, entry, cell, li);
  if (!sp.exists) { return; }
  if (abs(sp.pos.x) > fill.stand_r || abs(sp.pos.z) > fill.stand_r) { return; }

  let slot = atomicAdd(&args[1], 1u);
  if (slot >= fill.capacity) {
    atomicSub(&args[1], 1u);
    return;
  }
  let at = fill.base_instance + slot;
  out_instances[at * 2u] = vec4f(sp.pos, sp.yaw);
  out_instances[at * 2u + 1u] = vec4f(sp.scale, sp.phase, f32(entry), 0.0);
}
