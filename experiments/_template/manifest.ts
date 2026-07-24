import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  widthRatio: p.num(0.22, { min: 0.05, max: 0.6, step: 0.01 }),
}

export default defineExperiment({
  id: '__SLUG__',
  title: '__NAME__',
  description: 'TODO: one-line description of the technique idea.',
  status: 'idea',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  load: () => import('./main.ts'),
})
