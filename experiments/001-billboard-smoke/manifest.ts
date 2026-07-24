import { defineExperiment, HARNESS_API, p } from '@harness'

export const PARAMS = {
  density: p.num(1, { min: 0, max: 4, step: 0.05 }),
  height: p.num(0.6, { min: 0.1, max: 2, step: 0.01 }),
  sway: p.num(1, { min: 0, max: 3, step: 0.05 }),
}

export default defineExperiment({
  id: '001-billboard-smoke',
  title: 'billboard smoke',
  description: 'TODO: one-line description of the technique idea.',
  status: 'idea',
  harnessApi: HARNESS_API,
  species: ['grass-blade'],
  params: PARAMS,
  load: () => import('./main.ts'),
})
