import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked strip coverage. */
  alphaRef: p.num(0.42, { min: 0.1, max: 0.8, step: 0.02 }),
  /** LOD threshold scale: >1 keeps fine strip counts further out (costlier). */
  detail: p.num(1, { min: 0.35, max: 2.5, step: 0.05 }),
  /** Finest strip count a plant may use (fill-cost ceiling). */
  finest: p.enum('16', ['16', '8', '4']),
  /** How far each ribbon twists about its own arc toward the camera (0 = fully world-anchored). */
  faceCam: p.num(0.55, { min: 0, max: 1, step: 0.05 }),
  /** Horizontal canopy card from the baked top view (fills steep/top-down views). */
  topCard: p.bool(true),
  /** Fake grounding: darken strip bottoms toward the soil. */
  bottomShade: p.num(0.28, { min: 0, max: 0.6, step: 0.02 }),
}

export default defineExperiment({
  id: '032-blade-strips',
  title: 'blade strips',
  description:
    'The source mesh is split into 16 blade bundles; each bundle gets a curved ribbon (its real mean blade arc + a hugging width profile) and its real geometry is captured UNROLLED into that ribbon frame (albedo+coverage, tangent-frame normals). A plant draws its bundle ribbons as world-anchored curved strips — real parallax, view-dependent silhouette, real self-occlusion — collapsing 16/8/4/2/1 strips by projected size, with a baked top card for steep views.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
