import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Full-detail radius R0 (m): inside it every stand plant is drawn; beyond it
   *  survival probability keep(d) = (R0/d)^2 thins plants stochastically. */
  detailRadius: p.num(18, { min: 8, max: 40, step: 1 }),
  /** Number of cascade rings (each doubles the covered radius: 32m * 2^k). */
  rings: p.num(4, { min: 3, max: 5, step: 1 }),
  /** Max width amplification of surviving "clump" cards (1/sqrt(keep), capped). */
  widthCap: p.num(5, { min: 1, max: 10, step: 0.5 }),
  /** How strongly amplified clumps densify their interior coverage statistics. */
  densify: p.num(0.6, { min: 0, max: 1, step: 0.05 }),
  /** Distance where the alpha test starts morphing from hard 0.5 to hashed. */
  stochStart: p.num(25, { min: 5, max: 80, step: 1 }),
  /** Distance where the alpha test is fully stochastic (coverage-realizing). */
  stochFull: p.num(70, { min: 20, max: 200, step: 5 }),
  /** Statistical mean-field carpet beyond the cascade (scaling stands). */
  carpet: p.bool(true),
  /** Tint cards by cascade ring for debugging. */
  debugRings: p.bool(false),
}

export default defineExperiment({
  id: '011-stochastic-stipple',
  title: 'Stochastic thinning cascade',
  description:
    'Opacity as statistics: plants thin stochastically with distance (survivors widen to conserve ensemble coverage), mip coverage is realized per pixel by world-anchored hashed alpha, and a mean-field carpet absorbs the horizon.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
