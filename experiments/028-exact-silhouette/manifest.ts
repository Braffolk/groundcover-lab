import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /**
   * LOD radius (m): inside it a plant is assembled from its parts, outside it
   * it is one billboard pair. 0 = billboards everywhere (A/B reference).
   */
  partRadius: p.num(16, { min: 0, max: 32, step: 1 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Per-part sun-side modelling: sunward parts brighten, shaded parts darken. */
  selfShade: p.num(0.3, { min: 0, max: 0.7, step: 0.02 }),
  /** Canopy-depth gradient: darken toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
}

export default defineExperiment({
  id: '028-exact-silhouette',
  title: 'part-assembled plants',
  description:
    'A near plant is not a card: the bake PARTITIONS the source mesh into 3 equal-height bands x 4 azimuth sectors and captures each part from 4 staggered azimuths with its own tight box, so every part card stands at its real 3-D position inside the plant. The union of the alpha-tested cutouts is the real silhouette, parts parallax against each other, and per-part sun-side occlusion makes the assembly read as one volume; beyond the LOD radius a plant collapses to a whole-plant billboard pair.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
