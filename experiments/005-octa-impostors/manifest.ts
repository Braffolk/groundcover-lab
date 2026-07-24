import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  // Radius (m) of the region around the camera that is populated each frame.
  // Bounds the per-frame work; does NOT change placement, only visibility/LOD.
  regionRadius: p.num(96, { min: 24, max: 200, step: 4 }),
  // Alpha-test coverage cutoff for the impostor cards.
  coverage: p.num(0.34, { min: 0.05, max: 0.9, step: 0.01 }),
}

export default defineExperiment({
  id: '005-octa-impostors',
  title: 'Octahedral impostors',
  description: 'One plant baked into ~144 hemi-octahedral views; the 4 nearest are projected and blended per fragment so it holds from grazing to top-down.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
