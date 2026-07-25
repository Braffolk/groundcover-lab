import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Radius (m, x plant scale) inside which a plant is drawn as a 3-plane card cloud. */
  cloudRadius: p.num(16, { min: 0, max: 36, step: 1 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Horizontal canopy card (fills the top-down view). */
  topCard: p.bool(true),
  /** Fake grounding: darken card bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
  /** Sun transmission through the blades (back-lit glow). */
  translucency: p.num(0.14, { min: 0, max: 0.5, step: 0.02 }),
  /** Per-plant albedo variation (breaks the clone-army look). */
  tintJitter: p.num(0.09, { min: 0, max: 0.3, step: 0.01 }),
}

export default defineExperiment({
  id: '021-geometry-near-cards-far',
  title: 'card cloud near, impostor far',
  description:
    'Near plants are a three-plane billboard cloud: the source mesh is split by azimuth into wedges baked onto three vertical planes fixed in the plant frame, so the silhouette, interior parallax and self-occlusion are real 3D. Past a per-plant jittered apparent-size radius each plant collapses to one camera-facing 8-azimuth impostor cut from the same capture box, so distant plants cost exactly what a billboard costs and the handover has nothing left to change.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
