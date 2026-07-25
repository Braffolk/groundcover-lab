import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Coverage the sharpened edge is centred on (alpha-blended, never dithered). */
  covThresh: p.num(0.45, { min: 0.1, max: 0.9, step: 0.01 }),
  /** How hard the near-field coverage ramp is; 0 = raw prefiltered coverage. */
  edgeSharp: p.num(0.85, { min: 0, max: 1, step: 0.01 }),
  /**
   * Depth below the canopy top where the answer's first hit is expected to sit.
   * The per-pixel lookup point is placed on that plane, which is what registers
   * the elevation cells against each other (and against the true ray).
   */
  alignDepth: p.num(0.14, { min: 0, max: 0.8, step: 0.01 }),
  /** Sub-texel relief displacing the hit height (m), with its normal tilt. */
  reliefAmp: p.num(0.10, { min: 0, max: 0.15, step: 0.005 }),
  /** Interior-shading floor faded in with distance (1 = none). */
  farShade: p.num(0.78, { min: 0.4, max: 1, step: 0.01 }),
  /** Blade-scale detail on the partial-coverage edge band. */
  detailAmp: p.num(1.0, { min: 0, max: 1.5, step: 0.01 }),
  detailFreq: p.num(70, { min: 4, max: 160, step: 1 }),
  /** Burial AO: darkening per metre below the canopy top, and its floor. */
  aoRate: p.num(3.2, { min: 0, max: 6, step: 0.05 }),
  aoFloor: p.num(0.18, { min: 0, max: 1, step: 0.01 }),
  /** Low-frequency colour drift that breaks up the table's tiling period. */
  macroAmp: p.num(0.16, { min: 0, max: 0.6, step: 0.01 }),
  /** Wind shear amplitude (1 = the shared wind model as-is). */
  windScale: p.num(1, { min: 0, max: 3, step: 0.05 }),
  lodBias: p.num(0, { min: -2, max: 3, step: 0.05 }),
  maxDist: p.num(600, { min: 40, max: 900, step: 10 }),
  /** Interpolate the two bracketing elevation cells (4 taps) or take the nearer (2). */
  elevLerp: p.bool(true),
}

export default defineExperiment({
  id: '036-deferred-grass-field',
  title: 'deferred grass field',
  description:
    'One fullscreen triangle, zero plant primitives: each pixel unprojects the ground it sees, slides back up its eye ray to the canopy entry in closed form, and asks a baked ray-answer table what it meets — hit height, albedo, normal, coverage. Cost is bounded by pixels, so 557k and 134M plants render identically.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
