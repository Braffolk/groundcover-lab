import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** 3D distance under which 8m clusters render directly every frame. */
  directRadius: p.num(34, { min: 16, max: 64, step: 1 }),
  /** Parallax error (camera motion / cluster distance) that invalidates a slot. */
  refreshTau: p.num(0.05, { min: 0.01, max: 0.3, step: 0.005 }),
  /** Max cluster reconstructions per frame (steady state). */
  refreshBudget: p.num(3, { min: 1, max: 16, step: 1 }),
  /** Freeze all cache refreshes to inspect staleness/invalidation visually. */
  freezeCache: p.bool(false),
}

export default defineExperiment({
  id: '017-cached-clusters',
  title: 'Amortized cluster cache',
  description:
    'Temporal amortization: a distance-adaptive quadtree of world-anchored plant clusters is reconstructed into a color+depth slot atlas from baked card proxies, re-projected per-pixel every frame, and refreshed only when parallax error exceeds a threshold — a fixed per-frame refresh budget makes cost view-bounded, never plant-bounded.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
