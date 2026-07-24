/**
 * Integer PCG hash — bit-identical twin of src/wgsl/hash.wgsl. All 32-bit
 * multiplies go through Math.imul and every intermediate is forced back to
 * u32 with >>> 0, so results match WGSL exactly for all inputs.
 */

export function pcg(v: number): number {
  const state = (Math.imul(v, 747796405) + 2891336453) >>> 0
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0
  return ((word >>> 22) ^ word) >>> 0
}

export function hash2(a: number, b: number): number {
  return pcg((a ^ pcg(b)) >>> 0)
}

export function hash3(a: number, b: number, c: number): number {
  return pcg((a ^ hash2(b, c)) >>> 0)
}

export function hash4(a: number, b: number, c: number, d: number): number {
  return pcg((a ^ hash3(b, c, d)) >>> 0)
}

/** Uniform [0,1) with a 24-bit mantissa — exact in f32, identical to WGSL. */
export function hashF32(h: number): number {
  return (h >>> 8) / 16777216
}

/** i32 → u32 reinterpret, matching WGSL bitcast<u32>. */
export function asU32(v: number): number {
  return v >>> 0
}
