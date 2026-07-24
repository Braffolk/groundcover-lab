import { defineExperiment, HARNESS_API, p } from '@harness'

export const PARAMS = {
  tiles: p.num(3, { min: 1, max: 16, step: 1 }),
  sway: p.num(1, { min: 0, max: 3, step: 0.05 }),
}

export default defineExperiment({
  id: '000-ground-truth',
  title: 'ground truth (brute force)',
  description:
    'The real 2.17M-tri Calamagrostis source mesh, periodically tiled and brute-force rendered with wind. NOT a technique — the visual reference every experiment A/Bs against. Exempt from the perf/VRAM rules.',
  status: 'reference',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens'],
  params: PARAMS,
  load: () => import('./main.ts'),
})
