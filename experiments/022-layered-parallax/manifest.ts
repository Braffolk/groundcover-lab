import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Distance (m, x plant scale) past which a plant collapses to one merged tile. */
  lodDistance: p.num(16, { min: 0, max: 48, step: 1 }),
  /** Extra darkening per slab behind the front one (canopy self-occlusion). */
  layerShade: p.num(0.17, { min: 0, max: 0.4, step: 0.01 }),
  /** Fake grounding: darken card bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
  /** Horizontal canopy card (fills the top-down view). */
  topCard: p.bool(true),
}

export default defineExperiment({
  id: '022-layered-parallax',
  title: 'layered parallax',
  description:
    'One quad per plant, but its pixels are not flat: each species is baked per azimuth into three DEPTH SLABS, each storing the mean depth of the geometry it holds. The fragment shader intersects the real eye ray with those three planes in the plant frame and takes the first opaque hit — interior parallax, a silhouette that reshapes with the view, and correct front-to-back self-occlusion. The reprojection is a homography, so the interpolator evaluates it: one reciprocal, then two multiply-adds per slab, 1-3 taps, no marching and no frag_depth. Past lodDistance a plant collapses to one merged tile off a small fully-mipped atlas and costs exactly a billboard.',
  status: 'working',
  bakeVersion: 3,
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  load: () => import('./main.ts'),
})
