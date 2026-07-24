import type { VramScope } from '../gpu/resources.ts'
import { SCATTER_MAX_DENSITY } from './scatter.ts'
import { speciesById } from './species.ts'

/**
 * A stand is THE definition of what grows in the scene: which species, at
 * what density, size range and wind response, over what region. Stand + seed
 * fully determines every plant instance (position, species, scale, yaw,
 * phase) via the deterministic scatter — identical for every experiment.
 *
 * Experiments are pure RENDERERS of a stand. They never define placement;
 * their params may only affect how the stand is drawn. Stands are
 * harness-owned (this file) so they stay standardized — bench results and
 * A/B comparisons are only meaningful within one stand + seed.
 */

export interface StandSpecies {
  /** Species id in the SPECIES catalog (identity + source mesh). */
  species: string
  /** Plants per m² (scatter hard cap: 8). */
  density: number
  scaleMin: number
  scaleMax: number
  /** Wind response, 0 = rigid. */
  sway: number
}

export interface Stand {
  id: string
  title: string
  description: string
  /** Half-size (m) of the square region around the origin. */
  radius: number
  /** Order matters — the entry index keys placement and the GPU stand table. */
  species: StandSpecies[]
}

const CALAMAGROSTIS: Omit<StandSpecies, 'density'> = {
  species: 'calamagrostis-canescens',
  scaleMin: 0.8,
  scaleMax: 1.25,
  sway: 0.6,
}
const GRASS: Omit<StandSpecies, 'density'> = { species: 'grass-blade', scaleMin: 0.7, scaleMax: 1.3, sway: 1 }
const MOSS: Omit<StandSpecies, 'density'> = { species: 'moss-patch', scaleMin: 0.8, scaleMax: 1.6, sway: 0 }

export const STANDS: readonly Stand[] = [
  {
    id: 'default',
    title: 'default mixed meadow',
    description: 'The standard scene: three species, ~560k plants over ±128m. Used everywhere unless overridden.',
    radius: 128,
    species: [
      { ...CALAMAGROSTIS, density: 3 },
      { ...GRASS, density: 4 },
      { ...MOSS, density: 1.5 },
    ],
  },
  {
    id: 'calamagrostis-pure',
    title: 'pure calamagrostis',
    description: 'Single-species field of the reference plant, ~200k over ±128m.',
    radius: 128,
    species: [{ ...CALAMAGROSTIS, density: 3 }],
  },
  {
    id: 'close-quality',
    title: 'close-up quality',
    description: 'Small dense plot (±24m, ~20k) for judging per-plant fidelity against ground truth.',
    radius: 24,
    species: [
      { ...CALAMAGROSTIS, density: 3 },
      { ...GRASS, density: 4 },
      { ...MOSS, density: 1.5 },
    ],
  },
  {
    id: 'dense-mixed',
    title: 'dense mixed',
    description: 'Heavy overlap stress: ~7.7M plants over ±384m.',
    radius: 384,
    species: [
      { ...CALAMAGROSTIS, density: 5 },
      { ...GRASS, density: 6 },
      { ...MOSS, density: 2 },
    ],
  },
  {
    id: 'scaling-100m',
    title: 'scaling 100M+',
    description: 'The plant-count-must-be-free test: ~134M plants over ±2048m.',
    radius: 2048,
    species: [
      { ...CALAMAGROSTIS, density: 3 },
      { ...GRASS, density: 4 },
      { ...MOSS, density: 1 },
    ],
  },
]

export function standById(id: string): Stand {
  const stand = STANDS.find((s) => s.id === id)
  if (!stand) throw new Error(`unknown stand "${id}" — known: ${STANDS.map((s) => s.id).join(', ')}`)
  return stand
}

/** Expected plant counts (density × area; the scatter threshold matches). */
export function standPlantCounts(stand: Stand): { total: number; bySpecies: { species: string; count: number }[] } {
  const area = (stand.radius * 2) ** 2
  const bySpecies = stand.species.map((e) => ({
    species: e.species,
    count: Math.min(e.density, SCATTER_MAX_DENSITY) * area,
  }))
  return { total: bySpecies.reduce((a, e) => a + e.count, 0), bySpecies }
}

/**
 * GPU stand table — matches `struct StandEntry` in src/wgsl/frame.wgsl
 * (32B stride): density, scaleMin, scaleMax, sway, heightScale, speciesIndex.
 */
export function createStandBuffer(scope: VramScope, queue: GPUQueue, stand: Stand): GPUBuffer {
  const data = new Float32Array(stand.species.length * 8)
  stand.species.forEach((entry, i) => {
    const species = speciesById(entry.species)
    data.set(
      [entry.density, entry.scaleMin, entry.scaleMax, entry.sway, species.heightScale, species.index, 0, 0],
      i * 8,
    )
  })
  const buffer = scope.createBuffer(
    { label: `stand/${stand.id}`, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'stand-table' },
  )
  queue.writeBuffer(buffer, 0, data)
  return buffer
}
