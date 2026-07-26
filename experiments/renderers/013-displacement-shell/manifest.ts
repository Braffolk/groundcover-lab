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
  /** Carpet (mat) sheet geometry radius (m); the shell takes over past it. */
  carpetOuter: p.num(12, { min: 4, max: 20, step: 1 }),
  /** Target screen pixels per baked carpet grid quad — the mat's LOD knob. */
  carpetPx: p.num(3.5, { min: 2.5, max: 14, step: 0.5 }),
  debugRings: p.bool(false),
}

export default defineExperiment({
  id: '013-displacement-shell',
  title: 'strand displacement shell',
  description:
    'One flat sheet crumpled into the stand by baked vector-displacement textures: strand ribbons for scattered plants, a watertight top-shell displacement grid for carpet species; GPU-procedural cull into distance/LOD buckets, collapsing to a single terrain-conformal canopy shell far away.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: [
    'calamagrostis-canescens',
    'elymus-repens',
    'poa-pratensis',
    'spaghnum-palustre-wet-vigorous',
    'spaghnum-palustre-late-season',
    'spaghnum-palustre-sun-exposed',
  ],
  // Terrain-conformance views over the steepest ridge inside the bog stand
  // (~28-29 deg around x=28,z=-44). Absolute poses, so the y values below are
  // ground height + eye height at that spot, not "metres above zero".
  cams: {
    'slope-view': '22,0.4,-38,0.785,-0.1,60',
    'slope-close': '28,2.0,-44,0.785,-0.35,60',
  },
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
