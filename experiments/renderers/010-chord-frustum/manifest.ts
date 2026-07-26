import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  regionRadius: p.num(56, { min: 16, max: 80, step: 4 }),
  coverage: p.num(0.45, { min: 0.05, max: 0.9, step: 0.05 }),
  /** Alpha reference for CARPET entries (a mat must close, see NOTES.md). */
  carpetCoverage: p.num(0.2, { min: 0.02, max: 0.9, step: 0.02 }),
  /** Carpet lift in proxy heights — clears the 1m-quad base terrain (NOTES.md). */
  carpetLift: p.num(0.4, { min: 0, max: 1.5, step: 0.05 }),
  /** Carpet: blend the baked leaf normal towards the mat's macro normal. */
  carpetNormalMix: p.num(0.85, { min: 0, max: 1, step: 0.05 }),
  entrySmooth: p.bool(true),
  debugChart: p.bool(false),
}

export default defineExperiment({
  id: '010-chord-frustum',
  title: 'chord-field frustum proxies',
  description:
    'Each plant is a convex cone-frustum proxy whose interior appearance is precomputed as a 4D chord light field: the surface shader intersects the eye ray with the frustum in closed form and looks up what that entry->exit chord sees.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: [
    'calamagrostis-canescens',
    'elymus-repens',
    'poa-pratensis',
    'spaghnum-palustre-wet-vigorous',
    'spaghnum-palustre-late-season',
    'spaghnum-palustre-sun-exposed',
  ],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
