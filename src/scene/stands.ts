import type { VramScope } from '../gpu/resources.ts'
import { SCATTER_CELL_SIZE, SCATTER_MAX_DENSITY, SCATTER_MAX_PER_CELL } from './scatter.ts'
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
  /**
   * Optional habitat band. The scatter exposes a shared wetness field in
   * [0,1] (smooth 12m value noise, damped on slopes); an entry with a band
   * only grows where wetness is near `wetCenter`, thinning out linearly over
   * `wetWidth`. This is what turns a species list into a community with
   * coherent zones instead of a random shuffle. Omit both for uniform cover —
   * stands without them place exactly as they always did.
   */
  wetCenter?: number
  wetWidth?: number
  /**
   * @deprecated Carpets are a deprecated concept. A tiled ground mat turned out
   * to be a MATERIAL, not a kind of plant, and forcing both through one renderer
   * generalised badly — nearly every renderer in the repo independently mis-sized
   * its enumeration loop and drew a quarter of the mat. Ground cover of this kind
   * now belongs in a `material` experiment. The machinery below still works and
   * no stand currently uses it; do not build anything new on it.
   *
   * Lay this species out as a SEAMLESS CARPET rather than scattering it:
   * `carpetDiv` × `carpetDiv` periodic tiles per 4m scatter cell, grid-snapped
   * at constant scale with 90°-only rotations, which is what a periodic square
   * tile needs to abut its neighbours invisibly. Requires the species to have
   * `tileM`; `scaleMin`/`scaleMax` are then computed for you so a tile exactly
   * fills its grid step, and are ignored if you set them.
   *
   * In carpet mode wetCenter/wetWidth become a half-open interval
   * [c - w/2, c + w/2) instead of a soft falloff, and carpet entries sharing a
   * grid must PARTITION [0,1) so exactly one claims each node — otherwise the
   * mat gets holes (gaps) or double-stacked tiles (overlaps).
   *
   * A carpet entry has EXACTLY carpetDiv² slots per scatter cell, and that is
   * deliberately NOT bounded by SCATTER_MAX_PER_CELL — a life-size 0.18m tile
   * needs div 22 (484 slots) to fill a 4m cell, where the 128-slot budget would
   * force 2x life size. Size buffers and enumeration loops from
   * standEntrySlots(), never from SCATTER_MAX_PER_CELL, or you will render only
   * a fraction of the mat.
   */
  carpetDiv?: number
  /**
   * How much this species lies into the terrain plane rather than standing
   * upright (0..1). Defaults to 1 for carpet entries — a flat mat must follow
   * the ground — and 0 otherwise. Tall plants look right at ~0.2-0.4: real
   * grass grows upright regardless of the slope it grows on.
   */
  slopeAlign?: number
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
const ELYMUS: Omit<StandSpecies, 'density'> = { species: 'elymus-repens', scaleMin: 0.8, scaleMax: 1.2, sway: 0.65 }
const POA: Omit<StandSpecies, 'density'> = { species: 'poa-pratensis', scaleMin: 0.75, scaleMax: 1.3, sway: 0.85 }

export const STANDS: readonly Stand[] = [
  {
    id: 'default',
    title: 'default mixed meadow',
    description: 'The standard scene: three species, ~560k plants over ±128m. Used everywhere unless overridden.',
    radius: 128,
    species: [
      { ...CALAMAGROSTIS, density: 3 },
      { ...ELYMUS, density: 2.5 },
      { ...POA, density: 3 },
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
      { ...ELYMUS, density: 2.5 },
      { ...POA, density: 3 },
    ],
  },
  {
    id: 'dense-mixed',
    title: 'dense mixed',
    description: 'Heavy overlap stress: ~7.7M plants over ±384m.',
    radius: 384,
    species: [
      { ...CALAMAGROSTIS, density: 5 },
      { ...ELYMUS, density: 4 },
      { ...POA, density: 4 },
    ],
  },
  // The `bog` stand lived here: a life-size Sphagnum carpet in three
  // micro-habitat states zoned across the wetness field. It was removed with the
  // moss meshes when the lab split materials out from plant renderers — a ground
  // carpet is a MATERIAL, not a species of plant. The wetness field, the habitat
  // bands and the carpet machinery all survive (see the deprecation notes on
  // `carpetDiv`), so an equivalent stand can be reconstructed if one is ever
  // wanted; nothing about its removal is one-way except the meshes themselves.
  {
    id: 'scaling-100m',
    title: 'scaling 100M+',
    description: 'The plant-count-must-be-free test: ~134M plants over ±2048m.',
    radius: 2048,
    species: [
      { ...CALAMAGROSTIS, density: 3 },
      { ...ELYMUS, density: 2.5 },
      { ...POA, density: 2.5 },
    ],
  },
]

/**
 * Constant tile scale for a carpet entry: the periodic tile must exactly fill
 * its grid step, or the mat shows seams (too small) or overlaps (too large).
 * Returns null for ordinary scattered entries.
 *
 * @deprecated Carpets are deprecated — see `StandSpecies.carpetDiv`. Known
 * defect, left unfixed deliberately: `Scatter.cell()`'s carpet branch returns
 * the stand's placeholder `scaleMin` rather than this computed value, so the TS
 * and WGSL placement twins disagree for carpet entries. Nothing uses either path
 * now; do not repair a deprecated path.
 */
export function carpetScale(entry: StandSpecies): { min: number; max: number } | null {
  if (!entry.carpetDiv || entry.carpetDiv <= 0) return null
  const species = speciesById(entry.species)
  if (species.tileM === undefined) {
    throw new Error(`stand entry "${entry.species}" uses carpetDiv but the species has no periodic tileM`)
  }
  const s = SCATTER_CELL_SIZE / entry.carpetDiv / species.tileM
  return { min: s, max: s }
}

/**
 * Slots per scatter cell for a stand entry — carpetDiv² for a carpet (which may
 * exceed SCATTER_MAX_PER_CELL), otherwise the ordinary scatter budget. Use this
 * for every enumeration loop and buffer capacity.
 *
 * Still the correct call for ANY entry: with carpets gone it simply returns the
 * scatter budget, and a renderer that asks the entry rather than hardcoding 128
 * stays right if the budget ever changes.
 */
export function standEntrySlots(entry: StandSpecies): number {
  return entry.carpetDiv && entry.carpetDiv > 0 ? entry.carpetDiv ** 2 : SCATTER_MAX_PER_CELL
}

export function standById(id: string): Stand {
  const stand = STANDS.find((s) => s.id === id)
  if (!stand) throw new Error(`unknown stand "${id}" — known: ${STANDS.map((s) => s.id).join(', ')}`)
  return stand
}

/**
 * Expected plant counts (density × area). For entries with a habitat band the
 * count is scaled by the band's expected acceptance — the linear falloff
 * integrates to roughly `wetWidth` over the wetness range — so the figure
 * stays an estimate rather than a hard upper bound.
 */
export function standPlantCounts(stand: Stand): { total: number; bySpecies: { species: string; count: number }[] } {
  const area = (stand.radius * 2) ** 2
  const bySpecies = stand.species.map((e) => {
    const band = e.wetWidth === undefined || e.wetWidth <= 0 ? 1 : Math.min(1, e.wetWidth)
    // Carpet coverage comes from the grid, not from `density`: div² tiles per
    // scatter cell, of which the entry's wetness interval claims `band`.
    const perM2 = e.carpetDiv
      ? e.carpetDiv ** 2 / SCATTER_CELL_SIZE ** 2
      : Math.min(e.density, SCATTER_MAX_DENSITY)
    return { species: e.species, count: perM2 * area * band }
  })
  return { total: bySpecies.reduce((a, e) => a + e.count, 0), bySpecies }
}

/**
 * GPU stand table — matches `struct StandEntry` in src/wgsl/frame.wgsl
 * (32B stride): density, scaleMin, scaleMax, sway, heightScale, speciesIndex,
 * wetCenter, wetWidth.
 */
export function createStandBuffer(scope: VramScope, queue: GPUQueue, stand: Stand): GPUBuffer {
  const data = new Float32Array(stand.species.length * 12)
  stand.species.forEach((entry, i) => {
    const species = speciesById(entry.species)
    const scale = carpetScale(entry) ?? { min: entry.scaleMin, max: entry.scaleMax }
    data.set(
      [
        entry.density,
        scale.min,
        scale.max,
        entry.sway,
        species.heightScale,
        species.index,
        entry.wetCenter ?? 0,
        entry.wetWidth ?? 0,
        entry.carpetDiv ?? 0,
        species.tileM ?? 0,
        entry.slopeAlign ?? (entry.carpetDiv ? 1 : 0),
        0,
      ],
      i * 12,
    )
  })
  const buffer = scope.createBuffer(
    { label: `stand/${stand.id}`, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'stand-table' },
  )
  queue.writeBuffer(buffer, 0, data)
  return buffer
}
