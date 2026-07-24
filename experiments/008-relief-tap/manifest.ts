import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Groundcover draw distance (m); plants dissolve over fadeBand before it. */
  maxDist: p.num(64, { min: 24, max: 88, step: 4 }),
  fadeBand: p.num(14, { min: 4, max: 32, step: 2 }),
  /** Alpha-test threshold on bilinear coverage. */
  covThresh: p.num(0.38, { min: 0.05, max: 0.9, step: 0.01 }),
  /** Depth-in-clump darkening (fake AO), 0 = off. */
  ao: p.num(0.35, { min: 0, max: 0.8, step: 0.05 }),
  swayMul: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /**
   * The experiment's core question — what each extra FIXED tap buys:
   * flat-1tap   = plain single-view impostor (no interior parallax),
   * linear-2tap = one analytic reprojection against the tapped height,
   * secant-3tap = + secant (false-position) solve between the two taps.
   */
  reliefMode: p.enum('secant-3tap', ['flat-1tap', 'linear-2tap', 'secant-3tap'] as const),
  /** View-cell selection: per-pixel stochastic (dithered bilinear) or nearest. */
  viewSelect: p.enum('stochastic', ['stochastic', 'nearest'] as const),
  debugView: p.enum('lit', ['lit', 'albedo', 'normal', 'height'] as const),
}

export default defineExperiment({
  id: '008-relief-tap',
  title: 'Secant relief cards',
  description:
    'Depth-augmented impostor cards: 25 baked ortho views with heightfields; fragments solve the eye-ray/heightfield intersection with a fixed 3-tap secant scheme (no marching) and write true frag_depth.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  load: () => import('./main.ts'),
})
