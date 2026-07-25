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
  /**
   * Periodic community-tile footprint in meters, from the mesh manifest, for
   * meshes that tile (omit for single specimens). This is the species' true
   * horizontal footprint and is the ONLY correct way to size a plant's width —
   * never derive width from `heightScale`. It reaches shaders as
   * `stand_table[i].footprint_m`.
   */
  tileM?: number
}

export const SPECIES: readonly SpeciesDesc[] = [
  { id: 'calamagrostis-canescens', index: 0, meshId: 'calamagrostis-canescens', heightScale: 1.18, tileM: 0.52 },
  { id: 'elymus-repens', index: 1, meshId: 'elymus-repens', heightScale: 1.21, tileM: 0.62 },
  { id: 'poa-pratensis', index: 2, meshId: 'poa-pratensis', heightScale: 0.75 },
  // Indices 3-5 were three Sphagnum palustre micro-habitat states, removed when
  // the lab stopped treating a ground carpet as a kind of plant. Moss is now a
  // MATERIAL (experiments/materials/...), and the maps measured off those meshes
  // survive as PNGs in assets/materials/sphagnum-*. Do not reuse 3-5: the index
  // is carried into the GPU stand table and this list is append-only.
]

export function speciesById(id: string): SpeciesDesc {
  const s = SPECIES.find((s) => s.id === id)
  if (!s) throw new Error(`unknown species "${id}" — known: ${SPECIES.map((s) => s.id).join(', ')}`)
  return s
}
