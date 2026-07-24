import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Global multiplier on baked splat radii. */
  splatScale: p.num(0.85, { min: 0.4, max: 2.5, step: 0.05 }),
  /** Scales the LOD rings + far fade distance (perf/quality dial). */
  detail: p.num(1.0, { min: 0.4, max: 1.5, step: 0.05 }),
  /** How aggressively the reconstruction pass closes gaps between splats. */
  gapFill: p.num(1.0, { min: 0.0, max: 2.0, step: 0.05 }),
  /** Depth tolerance (m) for treating neighboring splats as one surface. */
  surfaceTol: p.num(0.55, { min: 0.1, max: 2.0, step: 0.05 }),
  /** Per-pixel albedo variance re-injection from the baked variance byte. */
  variance: p.num(1.0, { min: 0.0, max: 2.5, step: 0.05 }),
  /** Bypass the reconstruction filter (shows the raw dithered splats). */
  reconstruct: p.bool(true),
  // No local debug enum: the renderer answers the ONE global `view` selector
  // (frame.debug_mode / URL debug=) in the reconstruction pass instead.
}

export default defineExperiment({
  id: '002-splat-reconstruct',
  title: 'splat cloud + screen-space reconstruction',
  description:
    'Plants baked to ~2.5k anisotropic covariance splats (16B each, 5 LODs); half-res dithered splat pass + depth-affinity screen-space reconstruction turns the sparse point cloud back into continuous lit foliage.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
