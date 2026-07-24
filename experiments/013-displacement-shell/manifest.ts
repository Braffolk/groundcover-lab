import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Strand (ribbon) rows per plant at zero distance; falls off as r0/d. */
  strands: p.num(64, { min: 16, max: 96, step: 4 }),
  /** Full-detail radius (m); strand count decays hyperbolically past it. */
  r0: p.num(6.5, { min: 3, max: 12, step: 0.5 }),
  /** Ring1->ring2 boundary (m): station count drops 12 -> 8 -> 5. */
  r1: p.num(15, { min: 8, max: 24, step: 1 }),
  /** Plant region edge (m); beyond it the single canopy shell is the meadow. */
  rOuter: p.num(40, { min: 20, max: 56, step: 2 }),
  widthScale: p.num(0.75, { min: 0.3, max: 2.5, step: 0.05 }),
  /** Far-shell canopy height as a fraction of the mixed stand height. */
  shellHeight: p.num(0.6, { min: 0.2, max: 1.0, step: 0.05 }),
  debugRings: p.bool(false),
}

export default defineExperiment({
  id: '013-displacement-shell',
  title: 'strand displacement shell',
  description:
    'One flat sheet of ribbon patches crumpled into the meadow by baked strand vector-displacement textures; GPU-procedural scatter cull into three distance rings, collapsing to a single terrain-conformal canopy shell far away.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
