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

export const TERRAIN_DEFAULTS: TerrainDesc = { seed: 1337, size: 256, resolution: 512, heightScale: 4 }

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
}

/** CPU-only FBM value noise (generation is not mirrored in WGSL — the texture is the contract). */
function fbm(seed: number, x: number, z: number): number {
  let sum = 0
  let amp = 0.52
  let wavelength = 96
  for (let o = 0; o < 5; o++) {
    sum += valueNoise(seed + o * 101, x / wavelength, z / wavelength) * amp
    amp *= 0.5
    wavelength *= 0.5
  }
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
