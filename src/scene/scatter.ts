import { fround } from '../util/math.ts'
import type { VramScope } from '../gpu/resources.ts'
import { asU32, hash2, hash4, hashF32 } from './hash.ts'
import type { SpeciesDesc } from './species.ts'
import type { Terrain } from './terrain.ts'

/**
 * Deterministic plant placement — bit-identical twin of
 * src/wgsl/scatter.wgsl. Placement is a pure function of
 * (seed, cell, species, slot index): there is no global plant array at ANY
 * plant count, which is what makes 100 plants and 1 billion cost the same to
 * set up. Buffer-based experiments materialize regions with `region()` /
 * `instanceBuffer()`; procedural experiments evaluate scatter_candidate() in
 * shader and land every plant in exactly the same spot.
 */

export const SCATTER_CELL_SIZE = 4
export const SCATTER_MAX_PER_CELL = 128
export const SCATTER_MAX_DENSITY = 8

const TWO_PI = fround(6.2831853)

export interface ScatterPoint {
  x: number
  y: number
  z: number
  yaw: number
  scale: number
  phase: number
  species: number
}

/** Instance packing: 8 floats — [x, y, z, yaw, scale, speciesIndex, phase, 0]. */
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
  ) {}

  /** All existing plants of `species` in cell (cx, cz). Mirror of scatter_candidate(). */
  cell(species: SpeciesDesc, cx: number, cz: number, densityScale: number): ScatterPoint[] {
    const density = Math.min(Math.max(species.density * densityScale, 0), SCATTER_MAX_DENSITY)
    const threshold = density / SCATTER_MAX_DENSITY
    const out: ScatterPoint[] = []
    for (let i = 0; i < SCATTER_MAX_PER_CELL; i++) {
      const h = hash4(this.seed, asU32(cx), asU32(cz), ((species.index << 16) ^ i) >>> 0)
      if (hashF32(hash2(h, 0)) >= threshold) continue
      const ox = hashF32(hash2(h, 1))
      const oz = hashF32(hash2(h, 2))
      const x = fround(fround(cx + ox) * SCATTER_CELL_SIZE)
      const z = fround(fround(cz + oz) * SCATTER_CELL_SIZE)
      out.push({
        x,
        y: this.terrain.height(x, z),
        z,
        yaw: fround(hashF32(hash2(h, 3)) * TWO_PI),
        // mix() as specified: a*(1-t)+b*t. May differ from a driver's fma by
        // 1 ulp of plant scale — visually irrelevant, noted for honesty.
        scale: fround(
          fround(species.scaleMin * fround(1 - hashF32(hash2(h, 4)))) +
            fround(species.scaleMax * hashF32(hash2(h, 4))),
        ),
        phase: fround(hashF32(hash2(h, 5)) * TWO_PI),
        species: species.index,
      })
    }
    return out
  }

  /** All existing plants of `species` whose cells overlap the region. */
  region(species: SpeciesDesc, aabb: Aabb2, densityScale: number): ScatterPoint[] {
    const c0x = Math.floor(aabb.minX / SCATTER_CELL_SIZE)
    const c0z = Math.floor(aabb.minZ / SCATTER_CELL_SIZE)
    const c1x = Math.floor(aabb.maxX / SCATTER_CELL_SIZE)
    const c1z = Math.floor(aabb.maxZ / SCATTER_CELL_SIZE)
    const out: ScatterPoint[] = []
    for (let cz = c0z; cz <= c1z; cz++) {
      for (let cx = c0x; cx <= c1x; cx++) {
        out.push(...this.cell(species, cx, cz, densityScale))
      }
    }
    return out
  }

  /** Materialize a region as a packed STORAGE|VERTEX instance buffer. */
  instanceBuffer(
    scope: VramScope,
    queue: GPUQueue,
    species: SpeciesDesc,
    aabb: Aabb2,
    densityScale: number,
  ): { buffer: GPUBuffer; count: number } {
    const points = this.region(species, aabb, densityScale)
    const data = new Float32Array(Math.max(points.length, 1) * SCATTER_INSTANCE_FLOATS)
    points.forEach((pt, i) => {
      data.set([pt.x, pt.y, pt.z, pt.yaw, pt.scale, pt.species, pt.phase, 0], i * SCATTER_INSTANCE_FLOATS)
    })
    const buffer = scope.createBuffer(
      {
        label: `scatter/${species.id}`,
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      },
      { species: species.id, tag: 'scatter-instances' },
    )
    queue.writeBuffer(buffer, 0, data)
    return { buffer, count: points.length }
  }
}
