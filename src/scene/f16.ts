/**
 * f32 <-> f16 conversion (round-to-nearest-even), used to quantize terrain
 * texels on CPU so the CPU sampling mirror reads exactly the values the GPU
 * reads from the rgba16float heightmap.
 */

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

export function toF16Bits(value: number): number {
  f32[0] = value
  const x = u32[0]!
  const sign = (x >>> 16) & 0x8000
  let exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff

  if (exp === 0xff) return sign | 0x7c00 | (mant !== 0 ? 0x200 : 0) // inf / nan
  // Rebias 127 -> 15.
  let e = exp - 127 + 15
  if (e >= 0x1f) return sign | 0x7c00 // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign // underflow -> 0
    // Subnormal: shift mantissa (with implicit leading 1) right.
    mant |= 0x800000
    const shift = 14 - e
    const half = mant >>> shift
    const rem = mant & ((1 << shift) - 1)
    const halfway = 1 << (shift - 1)
    if (rem > halfway || (rem === halfway && (half & 1) !== 0)) return sign | (half + 1)
    return sign | half
  }
  // Normal: round 23-bit mantissa to 10 bits, ties to even.
  const half = (e << 10) | (mant >>> 13)
  const rem = mant & 0x1fff
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1) !== 0)) return sign | (half + 1) // may carry into exp — correct
  return sign | half
}

export function fromF16Bits(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1
  const exp = (bits >>> 10) & 0x1f
  const mant = bits & 0x3ff
  if (exp === 0) return sign * mant * 2 ** -24
  if (exp === 0x1f) return mant !== 0 ? NaN : sign * Infinity
  return sign * (1 + mant / 1024) * 2 ** (exp - 15)
}

/** Round-trip through f16 — the exact value stored in an rgba16float texel. */
export function quantizeF16(value: number): number {
  return fromF16Bits(toF16Bits(value))
}
