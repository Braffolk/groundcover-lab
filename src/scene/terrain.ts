import { fround } from '../util/math.ts'
import type { VramScope } from '../gpu/resources.ts'
import { asU32, hash4, hashF32 } from './hash.ts'
import { quantizeF16, toF16Bits } from './f16.ts'

export interface TerrainDesc {
  seed: number
  /** World extent in meters (square, centered on origin). */
  size: number
  /** Heightmap texels per side. */
  resolution: number
  /** Approximate total height amplitude in meters. */
  heightScale: number
}

export const TERRAIN_DEFAULTS: TerrainDesc = { seed: 1337, size: 256, resolution: 512, heightScale: 14 }

/**
 * Seeded FBM heightfield. The rgba16float heightmap texture (r=height,
 * g=normal.x, b=normal.z) is the single source of truth for plant placement:
 * texels are f16-quantized on CPU, and `height()`/`normal()` mirror the WGSL
 * `terrain_height()`/`terrain_normal()` manual-bilinear sampling with f32
 * rounding forced at every step, so CPU-placed and shader-placed plants agree.
 */
export class Terrain {
  readonly texture: GPUTexture

  private constructor(
    readonly desc: TerrainDesc,
    texture: GPUTexture,
    private heights: Float32Array,
    private nxs: Float32Array,
    private nzs: Float32Array,
  ) {
    this.texture = texture
  }

  static generate(scope: VramScope, queue: GPUQueue, desc: TerrainDesc = TERRAIN_DEFAULTS): Terrain {
    const n = desc.resolution
    const raw = new Float32Array(n * n)
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const wx = ((ix + 0.5) / n - 0.5) * desc.size
        const wz = ((iz + 0.5) / n - 0.5) * desc.size
        raw[iz * n + ix] = fbm(desc.seed, wx, wz) * desc.heightScale
      }
    }

    const texel = desc.size / n
    const heights = new Float32Array(n * n)
    const nxs = new Float32Array(n * n)
    const nzs = new Float32Array(n * n)
    const texData = new Uint16Array(n * n * 4)
    const at = (ix: number, iz: number): number =>
      raw[Math.min(n - 1, Math.max(0, iz)) * n + Math.min(n - 1, Math.max(0, ix))]!
    for (let iz = 0; iz < n; iz++) {
      for (let ix = 0; ix < n; ix++) {
        const i = iz * n + ix
        const dhdx = (at(ix + 1, iz) - at(ix - 1, iz)) / (2 * texel)
        const dhdz = (at(ix, iz + 1) - at(ix, iz - 1)) / (2 * texel)
        const inv = 1 / Math.hypot(dhdx, 1, dhdz)
        // Store the exact values the GPU will read back out of the texture.
        heights[i] = quantizeF16(raw[i]!)
        nxs[i] = quantizeF16(-dhdx * inv)
        nzs[i] = quantizeF16(-dhdz * inv)
        texData[i * 4] = toF16Bits(raw[i]!)
        texData[i * 4 + 1] = toF16Bits(-dhdx * inv)
        texData[i * 4 + 2] = toF16Bits(-dhdz * inv)
        texData[i * 4 + 3] = 0
      }
    }

    const texture = scope.createTexture(
      {
        label: 'scene/terrain-heightmap',
        size: [n, n],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { tag: 'terrain' },
    )
    queue.writeTexture({ texture }, texData, { bytesPerRow: n * 8 }, [n, n])
    return new Terrain(desc, texture, heights, nxs, nzs)
  }

  /** Bilinear (h, nx, nz) at world xz — f32-exact mirror of WGSL terrain_sample(). */
  private sample(x: number, z: number): [number, number, number] {
    const n = this.desc.resolution
    const ux = fround(fround(fround(fround(x / this.desc.size) + 0.5) * n) - 0.5)
    const uz = fround(fround(fround(fround(z / this.desc.size) + 0.5) * n) - 0.5)
    const bx = Math.floor(ux)
    const bz = Math.floor(uz)
    const fx = fround(ux - bx)
    const fz = fround(uz - bz)
    const clamp = (v: number): number => Math.min(n - 1, Math.max(0, v))
    const load = (ix: number, iz: number): [number, number, number] => {
      const i = clamp(iz) * n + clamp(ix)
      return [this.heights[i]!, this.nxs[i]!, this.nzs[i]!]
    }
    const s00 = load(bx, bz)
    const s10 = load(bx + 1, bz)
    const s01 = load(bx, bz + 1)
    const s11 = load(bx + 1, bz + 1)
    const out: [number, number, number] = [0, 0, 0]
    for (let c = 0; c < 3; c++) {
      const a = fround(s00[c]! + fround(fround(s10[c]! - s00[c]!) * fx))
      const b = fround(s01[c]! + fround(fround(s11[c]! - s01[c]!) * fx))
      out[c] = fround(a + fround(fround(b - a) * fz))
    }
    return out
  }

  height(x: number, z: number): number {
    return this.sample(x, z)[0]
  }

  normal(x: number, z: number): [number, number, number] {
    const [, nx, nz] = this.sample(x, z)
    const ny = Math.sqrt(Math.max(1 - nx * nx - nz * nz, 0))
    return [nx, ny, nz]
  }

  /**
   * Squared horizontal slope (nx² + nz²) — f32-exact mirror of
   * `g.y*g.y + g.z*g.z` on terrain_sample() in WGSL. Used by the scatter
   * wetness field, which needs bit-identical twins and so avoids the sqrt in
   * normal().
   */
  slopeSq(x: number, z: number): number {
    const [, nx, nz] = this.sample(x, z)
    return fround(fround(nx * nx) + fround(nz * nz))
  }

  /**
   * Least-squares ground plane over a footprint, from 5 taps — the CPU twin of
   * terrain_plane_fit() in src/wgsl/terrain.wgsl. `h` is the fitted height at
   * (x, z), which over a bumpy footprint sits at the local mean rather than the
   * exact centre sample. See CLAUDE.md for the ladder of terrain-fitting
   * options; this is one primitive, not a prescribed method.
   */
  planeFit(x: number, z: number, radius: number): { up: [number, number, number]; h: number } {
    const c = this.height(x, z)
    const hx0 = this.height(x - radius, z)
    const hx1 = this.height(x + radius, z)
    const hz0 = this.height(x, z - radius)
    const hz1 = this.height(x, z + radius)
    const dhdx = (hx1 - hx0) / (2 * radius)
    const dhdz = (hz1 - hz0) / (2 * radius)
    const inv = 1 / Math.hypot(dhdx, 1, dhdz)
    return { up: [-dhdx * inv, inv, -dhdz * inv], h: (c + hx0 + hx1 + hz0 + hz1) * 0.2 }
  }
}

/**
 * CPU-only FBM value noise (generation is not mirrored in WGSL — the texture
 * is the contract). Octave 1 is ridged, giving proper steep hillsides so
 * techniques get debugged against strong grazing angles, not just flats.
 */
function fbm(seed: number, x: number, z: number): number {
  const octaves: [wavelength: number, amp: number, ridged: boolean][] = [
    [150, 0.5, false],
    [60, 0.28, true],
    [24, 0.14, false],
    [10, 0.06, false],
    [4.5, 0.025, false],
    [2, 0.01, false],
  ]
  let sum = 0
  octaves.forEach(([wavelength, amp, ridged], o) => {
    let n = valueNoise(seed + o * 101, x / wavelength, z / wavelength)
    if (ridged) n = 1 - 2 * Math.abs(n)
    sum += n * amp
  })
  return sum
}

function valueNoise(seed: number, x: number, z: number): number {
  const ix = Math.floor(x)
  const iz = Math.floor(z)
  const fx = x - ix
  const fz = z - iz
  const sx = fx * fx * (3 - 2 * fx)
  const sz = fz * fz * (3 - 2 * fz)
  const corner = (dx: number, dz: number): number =>
    hashF32(hash4(asU32(seed), asU32(ix + dx), asU32(iz + dz), 7)) * 2 - 1
  const a = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * sx
  const b = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * sx
  return a + (b - a) * sz
}
