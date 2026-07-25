#include "src/wgsl/scatter.wgsl"
#include "./entry_info.wgsl"

// Per-frame plant selection over a bounded camera-centred cell region (already
// clamped to the stand's cell range on the CPU). Cost is O(region area), never
// O(total plants in the stand).
//
// Upright entries compact into ONE output list (cards). A CARPET entry compacts
// into THREE, by distance band, because a mat's cost has to fall with distance:
//   list 0  displaced patch  (near: real cushion geometry, 6*div^2 verts)
//   list 1  flat tile quad   (mid: parallax does the relief, 6 verts)
//   list 2  2x2 merged quad  (far: one quad per four tiles, 6 verts)
//
// The band is decided from the 2x2 BLOCK centre, never from the node, so all
// four nodes of a block always agree about whether they merged — otherwise a
// node on the boundary would be drawn twice (as itself and inside a merged
// neighbour). In the far band only the block's base node runs, and it evaluates
// all four candidates itself: the total number of scatter evaluations per cell
// is unchanged, but four tiles collapse into one quad.
//
// Every slot must be VISITED (info.slots_per_cell = carpet_div^2 = 484 at life
// size); only the expected survivors need capacity.

struct Inst {
  pos: vec3f,
  // carpet: rot0 | rot1<<2 | rot2<<4 | rot3<<6 | merged<<8
  // upright: yaw 10b | scale 12b (0..4 m) | phase 10b
  bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read_write> out0: array<Inst>;
@group(1) @binding(2) var<storage, read_write> out1: array<Inst>;
@group(1) @binding(3) var<storage, read_write> out2: array<Inst>;
// Three draw-arg sets in one buffer: [0..3] LOD0, [4..7] LOD1, [8..11] LOD2.
@group(1) @binding(4) var<storage, read_write> args: array<atomic<u32>, 12>;

// |terrain_height| bound: the shared FBM's octave amplitudes sum to 1.015, so
// the heightfield never leaves +/- 1.05 * frame.terrain_height_scale.
const TERRAIN_BOUND_SLACK: f32 = 1.05;

fn sphere_visible(center: vec3f, rad: f32) -> bool {
  for (var i = 0; i < 6; i++) {
    if (dot(info.planes[i].xyz, center) + info.planes[i].w < -rad) {
      return false;
    }
  }
  return true;
}

fn push(lod: u32, cap: f32, inst: Inst) {
  let w = atomicAdd(&args[lod * 4u + 1u], 1u);
  if (w >= u32(cap)) {
    atomicSub(&args[lod * 4u + 1u], 1u); // full — the count stays clamped
    return;
  }
  if (lod == 0u) {
    out0[w] = inst;
  } else if (lod == 1u) {
    out1[w] = inst;
  } else {
    out2[w] = inst;
  }
}

fn cap_of(lod: u32) -> f32 {
  if (lod == 0u) { return info.cap0; }
  if (lod == 1u) { return info.cap1; }
  return info.cap2;
}

/// Quarter-turn index the scatter assigned this node (yaw is k * 90 degrees).
fn rot_of(yaw: f32) -> u32 {
  return u32(yaw * 0.63661977 + 0.5) & 3u;
}

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
  let cell = vec2i(cx, cz);
  let entry_index = u32(info.entry_index);
  let entry = stand_table[entry_index];

  // --- cell-level rejects (workgroup-uniform, no memory traffic) ------------
  let cell_lo = vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE;
  let cell_mid = cell_lo + vec2f(SCATTER_CELL_SIZE * 0.5);
  let d_rect = max(abs(frame.camera_pos.xz - cell_mid) - vec2f(SCATTER_CELL_SIZE * 0.5), vec2f(0.0));
  if (length(d_rect) > info.region_r) {
    return;
  }
  let s_max = entry.scale_max;
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

  if (entry.carpet_div > 0.0) {
    cull_carpet(cell, cell_lo, slot, entry_index, entry.carpet_div);
    return;
  }

  // --- upright plant: one card list -----------------------------------------
  let sp = scatter_candidate(u32(info.seed), entry_index, cell, slot);
  if (!sp.exists) {
    return;
  }
  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }
  let center = sp.pos + vec3f(0.0, (info.card_y0 + info.card_y1) * 0.5 * sp.scale, 0.0);
  if (!sphere_visible(center, info.cull_radius * sp.scale + 0.35)) {
    return;
  }
  let yaw_q = u32(sp.yaw * (1024.0 / 6.2831853)) & 1023u;
  let scale_q = u32(clamp(sp.scale * 0.25, 0.0, 1.0) * 4095.0) & 4095u;
  let phase_q = u32(sp.phase * (1024.0 / 6.2831853)) & 1023u;
  push(0u, info.cap0, Inst(sp.pos, yaw_q | (scale_q << 10u) | (phase_q << 22u)));
}

fn cull_carpet(cell: vec2i, cell_lo: vec2f, slot: u32, entry_index: u32, div: f32) {
  let n = u32(div);
  if (slot >= n * n) {
    return;
  }
  let step = SCATTER_CELL_SIZE / div;
  let gx = slot % n;
  let gz = slot / n;
  // 2x2 block this node belongs to (carpet_div is even at life size, so the
  // blocks tile a cell exactly).
  let bx = gx & ~1u;
  let bz = gz & ~1u;
  let block_c = cell_lo + (vec2f(f32(bx), f32(bz)) + vec2f(1.0)) * step;
  let d_block = distance(frame.camera_pos.xz, block_c);
  let seed = u32(info.seed);
  let rad = info.cull_radius + info.relief_h;

  if (d_block > info.merge_dist) {
    // --- far band: one quad per 2x2 block -----------------------------------
    if (gx != bx || gz != bz) {
      return; // handled by the block's base node
    }
    if (d_block > info.region_r) {
      return;
    }
    var bits = 0u;
    var count = 0u;
    var base_pos = vec3f(0.0);
    var pos_sum = vec3f(0.0);
    for (var k = 0u; k < 4u; k++) {
      let s = (bz + k / 2u) * n + (bx + k % 2u);
      let sp = scatter_candidate(seed, entry_index, cell, s);
      if (!sp.exists) {
        continue;
      }
      bits |= rot_of(sp.yaw) << (2u * k);
      count += 1u;
      base_pos = sp.pos;
      pos_sum += sp.pos;
    }
    if (count == 0u) {
      return;
    }
    if (count == 4u) {
      let c = vec3f(block_c.x, pos_sum.y * 0.25, block_c.y);
      if (sphere_visible(c + vec3f(0.0, info.plane_h, 0.0), rad * 2.0)) {
        push(2u, info.cap2, Inst(c, bits | (1u << 8u)));
      }
      return;
    }
    // Mixed block (a wetness-zone boundary): fall back to per-tile quads. The
    // base node emits them all, so no node is ever drawn twice.
    for (var k = 0u; k < 4u; k++) {
      let s = (bz + k / 2u) * n + (bx + k % 2u);
      let sp = scatter_candidate(seed, entry_index, cell, s);
      if (!sp.exists) {
        continue;
      }
      if (sphere_visible(sp.pos + vec3f(0.0, info.plane_h, 0.0), rad)) {
        push(1u, info.cap1, Inst(sp.pos, rot_of(sp.yaw)));
      }
    }
    return;
  }

  // --- near / mid bands: one instance per tile ------------------------------
  let sp = scatter_candidate(seed, entry_index, cell, slot);
  if (!sp.exists) {
    return;
  }
  if (distance(sp.pos.xz, frame.camera_pos.xz) > info.region_r) {
    return;
  }
  if (!sphere_visible(sp.pos + vec3f(0.0, info.plane_h, 0.0), rad)) {
    return;
  }
  let lod = select(1u, 0u, d_block < info.patch_dist);
  push(lod, cap_of(lod), Inst(sp.pos, rot_of(sp.yaw)));
}
