import { fround } from '../util/math.ts'
import type { VramScope } from '../gpu/resources.ts'
import { asU32, hash2, hash4, hashF32 } from './hash.ts'
import type { Stand, StandSpecies } from './stands.ts'
import type { Terrain } from './terrain.ts'

/**
 * Deterministic plant placement — bit-identical twin of
 * src/wgsl/scatter.wgsl. A Scatter is bound to one stand + seed; placement is
 * then a pure function of (stand entry index, cell, slot index): there is no
 * global plant array at ANY plant count, which is what makes 100 plants and
 * 1 billion cost the same to set up. Buffer-based renderers materialize
 * regions with `region()`/`instanceBuffer()`; procedural renderers evaluate
 * scatter_candidate() in shader and land every plant in exactly the same
 * spot.
 */

export const SCATTER_CELL_SIZE = 4
export const SCATTER_MAX_PER_CELL = 128
export const SCATTER_MAX_DENSITY = 8

const TWO_PI = fround(6.2831853)

// Habitat-modulation constants — must match src/wgsl/scatter.wgsl exactly.
const WET_CELL = 12
const WET_SALT = 0x9e37
const WET_SLOPE_SQ = fround(0.3276)
const CARPET_JITTER = fround(0.06)
const QUARTER_TURN = fround(1.5707963)

export interface ScatterPoint {
  x: number
  y: number
  z: number
  yaw: number
  scale: number
  phase: number
  /** Index into the stand's species list (and the GPU stand table). */
  entry: number
}

/** Instance packing: 8 floats — [x, y, z, yaw, scale, entryIndex, phase, 0]. */
export const SCATTER_INSTANCE_FLOATS = 8

export interface Aabb2 {
  minX: number
  minZ: number
  maxX: number
  maxZ: number
}

export class Scatter {
  constructor(
    private terrain: Terrain,
    readonly seed: number,
    readonly stand: Stand,
  ) {}

  /** The stand's full region as an aabb — the default extent everywhere. */
  standRegion(): Aabb2 {
    const r = this.stand.radius
    return { minX: -r, minZ: -r, maxX: r, maxZ: r }
  }

  private entry(entryIndex: number): StandSpecies {
    const e = this.stand.species[entryIndex]
    if (!e) throw new Error(`stand "${this.stand.id}" has no species entry ${entryIndex}`)
    return e
  }

  /**
   * Shared wetness field in [0,1] that stands zone species against — smooth
   * value noise on a 12m grid, damped on slopes. f32-exact mirror of
   * scatter_wetness() in src/wgsl/scatter.wgsl (every step is fround'ed
   * because a threshold comparison would otherwise flip plants near a band
   * edge between the two twins).
   */
  wetness(x: number, z: number): number {
    const ux = fround(x / WET_CELL)
    const uz = fround(z / WET_CELL)
    const bx = Math.floor(ux)
    const bz = Math.floor(uz)
    const fx = fround(ux - bx)
    const fz = fround(uz - bz)
    const sx = fround(fround(fx * fx) * fround(3 - fround(2 * fx)))
    const sz = fround(fround(fz * fz) * fround(3 - fround(2 * fz)))
    const corner = (dx: number, dz: number): number =>
      hashF32(hash4(this.seed, asU32(bx + dx), asU32(bz + dz), WET_SALT))
    const a = corner(0, 0)
    const c = corner(1, 0)
    const d = corner(0, 1)
    const e = corner(1, 1)
    const lo = fround(a + fround(fround(c - a) * sx))
    const hi = fround(d + fround(fround(e - d) * sx))
    const n = fround(lo + fround(fround(hi - lo) * sz))
    const flat = Math.min(
      Math.max(fround(fround(WET_SLOPE_SQ - this.terrain.slopeSq(x, z)) / WET_SLOPE_SQ), 0),
      1,
    )
    return Math.min(Math.max(fround(n * fround(0.3 + fround(0.7 * flat))), 0), 1)
  }

  /** All plants of stand entry `entryIndex` in cell (cx, cz). Mirror of scatter_candidate(). */
  cell(entryIndex: number, cx: number, cz: number): ScatterPoint[] {
    const e = this.entry(entryIndex)
    const threshold = Math.min(e.density, SCATTER_MAX_DENSITY) / SCATTER_MAX_DENSITY
    const wetWidth = e.wetWidth ?? 0
    const wetCenter = e.wetCenter ?? 0
    const out: ScatterPoint[] = []

    // Carpet layout — mirror of the carpet branch in scatter.wgsl.
    const div = e.carpetDiv ?? 0
    if (div > 0) {
      const step = fround(SCATTER_CELL_SIZE / div)
      const lo = fround(wetCenter - fround(wetWidth * 0.5))
      const hi = fround(wetCenter + fround(wetWidth * 0.5))
      for (let i = 0; i < div * div; i++) {
        const h = hash4(this.seed, asU32(cx), asU32(cz), ((entryIndex << 16) ^ i) >>> 0)
        const x = fround(fround(cx * SCATTER_CELL_SIZE) + fround(fround((i % div) + 0.5) * step))
        const z = fround(fround(cz * SCATTER_CELL_SIZE) + fround(fround(Math.floor(i / div) + 0.5) * step))
        const jitter = fround(fround(hashF32(hash2(h, 7)) - 0.5) * CARPET_JITTER)
        const w = Math.min(Math.max(fround(this.wetness(x, z) + jitter), 0), 0.9999)
        if (!(w >= lo && w < hi)) continue
        out.push({
          x,
          y: this.terrain.height(x, z),
          z,
          yaw: fround((h & 3) * QUARTER_TURN),
          scale: e.scaleMin,
          phase: 0,
          entry: entryIndex,
        })
      }
      return out
    }

    for (let i = 0; i < SCATTER_MAX_PER_CELL; i++) {
      const h = hash4(this.seed, asU32(cx), asU32(cz), ((entryIndex << 16) ^ i) >>> 0)
      if (hashF32(hash2(h, 0)) >= threshold) continue
      const ox = hashF32(hash2(h, 1))
      const oz = hashF32(hash2(h, 2))
      const x = fround(fround(cx + ox) * SCATTER_CELL_SIZE)
      const z = fround(fround(cz + oz) * SCATTER_CELL_SIZE)
      if (wetWidth > 0) {
        const dist = fround(Math.abs(fround(this.wetness(x, z) - wetCenter)) / wetWidth)
        if (hashF32(hash2(h, 6)) >= Math.min(Math.max(fround(1 - dist), 0), 1)) continue
      }
      const t = hashF32(hash2(h, 4))
      out.push({
        x,
        y: this.terrain.height(x, z),
        z,
        yaw: fround(hashF32(hash2(h, 3)) * TWO_PI),
        // mix() as specified: a*(1-t)+b*t. May differ from a driver's fma by
        // 1 ulp of plant scale — visually irrelevant, noted for honesty.
        scale: fround(fround(e.scaleMin * fround(1 - t)) + fround(e.scaleMax * t)),
        phase: fround(hashF32(hash2(h, 5)) * TWO_PI),
        entry: entryIndex,
      })
    }
    return out
  }

  /** All plants of stand entry `entryIndex` in cells overlapping the region. */
  region(entryIndex: number, aabb: Aabb2 = this.standRegion()): ScatterPoint[] {
    const c0x = Math.floor(aabb.minX / SCATTER_CELL_SIZE)
    const c0z = Math.floor(aabb.minZ / SCATTER_CELL_SIZE)
    const c1x = Math.floor(aabb.maxX / SCATTER_CELL_SIZE)
    const c1z = Math.floor(aabb.maxZ / SCATTER_CELL_SIZE)
    const out: ScatterPoint[] = []
    for (let cz = c0z; cz <= c1z; cz++) {
      for (let cx = c0x; cx <= c1x; cx++) {
        out.push(...this.cell(entryIndex, cx, cz))
      }
    }
    return out
  }

  /** Materialize one stand entry as a packed STORAGE|VERTEX instance buffer. */
  instanceBuffer(
    scope: VramScope,
    queue: GPUQueue,
    entryIndex: number,
    aabb: Aabb2 = this.standRegion(),
  ): { buffer: GPUBuffer; count: number } {
    const speciesId = this.entry(entryIndex).species
    const points = this.region(entryIndex, aabb)
    const data = new Float32Array(Math.max(points.length, 1) * SCATTER_INSTANCE_FLOATS)
    points.forEach((pt, i) => {
      data.set([pt.x, pt.y, pt.z, pt.yaw, pt.scale, pt.entry, pt.phase, 0], i * SCATTER_INSTANCE_FLOATS)
    })
    const buffer = scope.createBuffer(
      {
        label: `scatter/${this.stand.id}/${entryIndex}-${speciesId}`,
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      },
      { species: speciesId, tag: 'scatter-instances' },
    )
    queue.writeBuffer(buffer, 0, data)
    return { buffer, count: points.length }
  }
}
