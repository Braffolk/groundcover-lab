import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  kNear: p.num(12, { min: 2, max: 16, step: 1 }),
  kMid: p.num(4, { min: 1, max: 8, step: 1 }),
  rNear: p.num(14, { min: 4, max: 30, step: 1 }),
  rMid: p.num(38, { min: 10, max: 60, step: 1 }),
  rRegion: p.num(64, { min: 24, max: 96, step: 4 }),
  alphaThresh: p.num(0.35, { min: 0.05, max: 0.9, step: 0.01 }),
  // Carpet species (stand_table.carpet_div > 0, i.e. the bog's Sphagnum) use a
  // HARD alpha reference instead of the dithered one, and their own
  // sky-visibility floor — see shaders/slices.wgsl.
  carpetAlpha: p.num(0.3, { min: 0.02, max: 0.9, step: 0.01 }),
  carpetOcc: p.num(0.55, { min: 0.2, max: 1, step: 0.01 }),
  sigma: p.num(1.4, { min: 0.2, max: 4, step: 0.05 }),
  lodBias: p.num(0, { min: -2, max: 2, step: 0.1 }),
  axisBiasY: p.num(1.15, { min: 0.5, max: 2, step: 0.05 }),
  farShell: p.bool(true),
  debugAxis: p.bool(false),
}

export default defineExperiment({
  id: '003-shell-slices',
  title: 'axis-locked slice-stack shells',
  description:
    'Plants as baked voxel volumes drawn as K translucent slices whose stack axis snaps to the view (horizontal shells from above, vertical curtains at grazing); K collapses 12→4→1 with distance and the meadow degenerates into a single terrain-draped canopy shell toward the horizon.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
