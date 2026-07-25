import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 112, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Distance (m) below which cards get the depth warp (LOD split). */
  nearSplit: p.num(14, { min: 0, max: 44, step: 2 }),
  /** Depth-warp strength — 0 turns the card back into a plain billboard. */
  parallax: p.num(1, { min: 0, max: 1.4, step: 0.05 }),
  /** How much of the baked sky/sun visibility is applied. */
  occlusion: p.num(1, { min: 0, max: 1, step: 0.05 }),
  /** Thin-blade back-light glow. */
  transmission: p.num(0.2, { min: 0, max: 1, step: 0.05 }),
  /** Canopy-depth exponent on the baked openness (1 = the bake's own lattice). */
  canopyDepth: p.num(2.6, { min: 1, max: 4.5, step: 0.1 }),
}

export default defineExperiment({
  id: '023-baked-self-occlusion',
  title: 'self-occlusion cards',
  description:
    'One view-aligned card per plant over a 25-view baked atlas that also stores per-texel surface depth, sky visibility and a bent shading normal. A single depth tap warps the card along the eye ray (parallax inside the silhouette, view-dependent silhouette shape); the baked visibility does the rest — dark interiors, lit tips, sun-direction-dependent self-shadowing, ground contact. 3 taps near, 2 far, no marching, no dither.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 5,
  load: () => import('./main.ts'),
})
