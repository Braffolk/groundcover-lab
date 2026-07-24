/**
 * Species catalog — IDENTITY only: which plants exist and which source mesh
 * they come from. Everything about placement (density, scale range, sway,
 * region) lives in stands (src/scene/stands.ts); experiments never define
 * either.
 */

export interface SpeciesDesc {
  id: string
  /** Stable catalog index (append only) — carried into the GPU stand table. */
  index: number
  /** GCMESH1 source mesh in mesh/raw/<meshId>/. */
  meshId: string
  /** Nominal plant height in meters (physical property of the species). */
  heightScale: number
}

export const SPECIES: readonly SpeciesDesc[] = [
  { id: 'calamagrostis-canescens', index: 0, meshId: 'calamagrostis-canescens', heightScale: 1.18 },
  { id: 'elymus-repens', index: 1, meshId: 'elymus-repens', heightScale: 1.21 },
  { id: 'poa-pratensis', index: 2, meshId: 'poa-pratensis', heightScale: 0.75 },
]

export function speciesById(id: string): SpeciesDesc {
  const s = SPECIES.find((s) => s.id === id)
  if (!s) throw new Error(`unknown species "${id}" — known: ${SPECIES.map((s) => s.id).join(', ')}`)
  return s
}
