import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centred region the cull pass enumerates. */
  regionRadius: p.num(128, { min: 24, max: 128, step: 4 }),
  /** Alpha-test cutoff on blended coverage (hard edges, never dithered). */
  alphaRef: p.num(0.38, { min: 0.05, max: 0.9, step: 0.01 }),
  /**
   * How the view set resolves per plant: 'blend3' = barycentric blend of the
   * 3 hemi-octahedral views around the viewing direction, 'nearest' = hard
   * switch to the closest view (shows what the blend is buying).
   */
  viewBlend: p.enum('blend3', ['blend3', 'nearest'] as const),
  /** Debug: paint each plant by the id of its dominant baked view. */
  viewTint: p.bool(false),
}

export default defineExperiment({
  id: '005-octa-impostors',
  title: 'Octahedral view-set impostors',
  description:
    '121 hemi-octahedral views per species in a mipped texture array; each plant picks its 3 surrounding views and reprojects the card into each one, so the same quad reads correctly from grazing to straight down.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
