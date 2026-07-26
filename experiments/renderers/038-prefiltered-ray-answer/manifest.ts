import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** One closed-form parallax-correction probe (kills direction-quantisation seams). */
  correct: p.bool(true),
  /** Blend the two neighbouring azimuth bins (removes 15deg banding; +2 fetches). */
  azBlend: p.bool(true),
  /** Baked self-occlusion strength (depth below the local canopy top). */
  aoStrength: p.num(0.55, { min: 0, max: 1, step: 0.05 }),
  /** Smooth lookup warp (m) that breaks the 6 m answer lattice without seams. */
  warpAmp: p.num(1.4, { min: 0, max: 3, step: 0.1 }),
  /** Alpha-test width floor in texels: small = hard silhouette edges near. */
  sharpen: p.num(0.18, { min: 0.05, max: 1, step: 0.01 }),
  /** Blade-scale normal/albedo detail where the table is magnified (0 = off). */
  detail: p.num(0.85, { min: 0, max: 1, step: 0.05 }),
  /** Mip bias on the prefiltered ray answer (0 = footprint-exact). */
  lodBias: p.num(0, { min: -1, max: 2, step: 0.1 }),
  /** Carrier grid cell size (m) — tessellation is fixed, this only sets its scale. */
  carrierCell: p.num(2.4, { min: 1, max: 4, step: 0.2 }),
  /** Snap residual (in units of the un-snapped hit distance) above which coverage fades. */
  snapTol: p.num(1.2, { min: 0.2, max: 8, step: 0.1 }),
  /** Keep the carrier surface this far below the eye when the eye is inside the canopy. */
  eyeClearance: p.num(0.25, { min: 0.05, max: 1.2, step: 0.05 }),
}

export default defineExperiment({
  id: '038-prefiltered-ray-answer',
  title: 'prefiltered ray answer',
  description:
    'Zero plant primitives: the whole 3-species canopy is a baked, mip-prefiltered table of ray ANSWERS (hit height, albedo, oct normal, coverage, AO) indexed by where the eye ray crosses the canopy-top plane and by its quantised direction. One coarse terrain-conformal carrier grid is the only thing rasterized; each pixel resolves in 5 texture fetches with a closed-form entry plus one parallax correction — no marching.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
