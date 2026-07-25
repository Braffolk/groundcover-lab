import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region evaluated per frame (matches the billboard baseline). */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** LOD ladder shift — higher keeps the fine splat sets further out (more detail, more fill). */
  lodBias: p.num(1, { min: 0.4, max: 2.5, step: 0.05 }),
  /** Global multiplier on baked splat extents (coverage vs. silhouette tightness). */
  splatScale: p.num(1, { min: 0.6, max: 1.6, step: 0.02 }),
  /** Hard alpha-test reference against the baked stamp coverage. */
  alphaRef: p.num(0.45, { min: 0.15, max: 0.75, step: 0.05 }),
  /** Minimum projected half-extent (px) — keeps far splats from going sub-pixel. */
  minPx: p.num(0.5, { min: 0, max: 2, step: 0.05 }),
  /** Strength of the baked canopy occlusion term. */
  aoStrength: p.num(0.65, { min: 0, max: 1, step: 0.05 }),
  /** Cross-splat normal bend — makes each micro-card read as a curved blade. */
  bulge: p.num(0.55, { min: 0, max: 1.2, step: 0.05 }),
}

export default defineExperiment({
  id: '025-micro-splats',
  title: 'micro splats',
  description:
    'Each plant is a few thousand TINY oriented surfels welded to the real mesh surface: a balanced kd-tree over the GCMESH1 triangles turns every cluster into one hard-edged elliptical micro-card (PCA frame, mesh colour, baked canopy occlusion, real blade-imagery stamp). All 14 tree levels ship, so a plant collapses 8192 -> 1 splat as it recedes and the far field ends up cheaper than a billboard.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
