import type { VramScope } from '../gpu/resources.ts'

export interface SpeciesDesc {
  id: string
  /** Index into the WGSL species_table (stable — append only). */
  index: number
  /** GCMESH1 source mesh in mesh/raw/<meshId>/, if this species has one. */
  meshId?: string
  /** Plants per m^2 at density scale 1. Hard cap 8 (scatter slot budget). */
  density: number
  scaleMin: number
  scaleMax: number
  /** Wind response, 0 = rigid (moss). */
  sway: number
  /** Nominal plant height in meters (for budgeting/LOD heuristics). */
  heightScale: number
}

export const SPECIES: readonly SpeciesDesc[] = [
  {
    id: 'calamagrostis-canescens',
    index: 0,
    meshId: 'calamagrostis-canescens',
    density: 3,
    scaleMin: 0.8,
    scaleMax: 1.25,
    sway: 0.6,
    heightScale: 1.18,
  },
  { id: 'grass-blade', index: 1, meshId: 'grass-blade', density: 6, scaleMin: 0.7, scaleMax: 1.3, sway: 1, heightScale: 0.6 },
  { id: 'moss-patch', index: 2, meshId: 'moss-patch', density: 2, scaleMin: 0.8, scaleMax: 1.6, sway: 0, heightScale: 0.15 },
]

export function speciesById(id: string): SpeciesDesc {
  const s = SPECIES.find((s) => s.id === id)
  if (!s) throw new Error(`unknown species "${id}" — known: ${SPECIES.map((s) => s.id).join(', ')}`)
  return s
}

/** Storage buffer matching `struct Species` in src/wgsl/frame.wgsl (32B stride). */
export function createSpeciesBuffer(scope: VramScope, queue: GPUQueue): GPUBuffer {
  const data = new Float32Array(SPECIES.length * 8)
  for (const s of SPECIES) {
    data.set([s.density, s.scaleMin, s.scaleMax, s.sway, s.heightScale, 0, 0, 0], s.index * 8)
  }
  const buffer = scope.createBuffer(
    { label: 'scene/species-table', size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'species-table' },
  )
  queue.writeBuffer(buffer, 0, data)
  return buffer
}
