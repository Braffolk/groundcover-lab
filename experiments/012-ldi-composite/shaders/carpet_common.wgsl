// Shared between carpet_cull.wgsl (compute) and carpet.wgsl (render): the
// per-entry uniform block and the packed instance record for a CARPET species
// (stand_table[i].carpet_div > 0 — the bog's Sphagnum mat).
//
// Field packing must match the Float32Array layout in main.ts. As in
// ldi_common.wgsl the block is split into a static half (bake + stand
// constants, written once) and a contiguous dynamic pair (`region`, `tune`,
// floats 16..23) rewritten per frame.
//
// Include AFTER src/wgsl/scatter.wgsl — SCATTER_CELL_SIZE comes from there,
// and the node position below must stay bit-identical to the carpet branch of
// scatter_candidate().

const CARPET_QUARTER: f32 = 1.5707963;

struct CarpetU {
  // --- static ---------------------------------------------------------------
  geom: vec4f,    // grid step (m), card half width (m), y_top (m), y_span (m)
  planes: vec4f,  // card plane height above the ground, per drawn layer (m)
  ids: vec4f,     // seed, stand entry index, instance capacity, stand radius (m)
  grid: vec4f,    // carpet_div, slots (div*div), uniform overscale, unused
  // --- dynamic (contiguous: floats 16..27) ----------------------------------
  region: vec4f,  // origin cell x, origin cell z, side (cells), region radius (m)
  tune: vec4f,    // near/far split (m), parallax 0/1, alpha reference, inspect
  tune2: vec4f,   // mip bias (negative = sharper), unused, unused, unused
}

/**
 * Packed instance: 9 bits of window cell dx, 9 of dz, 9 of grid slot, 2 of yaw
 * quadrant = 4 bytes per tile instead of 32. A life-size mat has 484 slots per
 * 4 m cell (22x22), so for a carpet the instance array IS the VRAM story: at
 * 32 B/tile a 128 m window costs 25 MB before a single texel is stored, and
 * the position is fully implied by (cell, slot) anyway.
 */
fn carpet_pack(dx: u32, dz: u32, slot: u32, quad: u32) -> u32 {
  return (dx & 511u) | ((dz & 511u) << 9u) | ((slot & 511u) << 18u) | ((quad & 3u) << 27u);
}

struct CarpetNode {
  cell: vec2i,
  node: vec2f,  // world xz of the tile centre (the scatter's grid node)
  yaw: f32,     // 90-degree steps only — the lattice invariant
  quad: u32,
}

fn carpet_unpack(packed: u32, origin: vec2i, div: f32) -> CarpetNode {
  let n = u32(div);
  let slot = (packed >> 18u) & 511u;
  var out: CarpetNode;
  out.cell = origin + vec2i(i32(packed & 511u), i32((packed >> 9u) & 511u));
  let step = SCATTER_CELL_SIZE / div;
  let g = vec2f(f32(slot % n), f32(slot / n));
  out.node = vec2f(out.cell) * SCATTER_CELL_SIZE + (g + 0.5) * step;
  out.quad = (packed >> 27u) & 3u;
  out.yaw = f32(out.quad) * CARPET_QUARTER;
  return out;
}

/// Yaw rotation on the horizontal plane — the (x, z) half of rot_yaw().
fn carpet_rot(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(c * v.x + s * v.y, -s * v.x + c * v.y);
}

/// 16-bit linear capture depth stored across (b, a): 0 = top of the cushion.
fn carpet_depth(aux: vec4f) -> f32 {
  return (aux.b * 255.0 * 256.0 + aux.a * 255.0) / 65535.0;
}
