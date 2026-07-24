import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of full-detail ribbon plants around the camera. */
  detailRadius: p.num(14, { min: 4, max: 40, step: 1 }),
  /** Cross-fade band (m) beyond detailRadius where detail collapses / aggregate grows. */
  blendBand: p.num(8, { min: 2, max: 24, step: 1 }),
  /** Radius (m) of the aggregate-card ring — the far horizon of drawn plants. */
  regionRadius: p.num(60, { min: 12, max: 160, step: 2 }),
  /** Global ribbon width multiplier (fill vs. gaps). */
  widthScale: p.num(1.25, { min: 0.5, max: 2.5, step: 0.05 }),
}

export default defineExperiment({
  id: '014-ribbon-skeleton',
  title: 'ribbon skeleton',
  description:
    'Plants distilled into a handful of curved ribbons stored in tiny textures and unrolled as triangle strips in the vertex shader.',
  status: 'wip',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
