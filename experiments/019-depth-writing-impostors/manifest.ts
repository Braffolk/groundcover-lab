import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Distance (m) out to which impostors write true per-pixel depth. */
  depthDist: p.num(6, { min: 0, max: 24, step: 1 }),
  /** Distance (m) out to which the depth warp runs (beyond it: 1 tap, flat). */
  warpDist: p.num(30, { min: 4, max: 64, step: 1 }),
  /** Depth-warp strength — 0 turns the method back into a plain billboard. */
  warp: p.num(1, { min: 0, max: 1, step: 0.05 }),
  /** Mip floor the warp reads its depth estimate from (higher = smoother). */
  warpBlur: p.num(0.5, { min: 0, max: 6, step: 0.5 }),
  /** Baked canopy occlusion mixed into the light term. */
  aoStrength: p.num(0.75, { min: 0, max: 1, step: 0.05 }),
}

export default defineExperiment({
  id: '019-depth-writing-impostors',
  title: 'true-depth impostors',
  description:
    'Impostors whose texels are real 3D points: a baked view set stores signed depth (plus normals and canopy AO) per texel, the fragment shader inverts the image->screen map with one depth tap to warp the imagery to the exact current view, and the near band writes that reconstructed point as frag_depth so plants interpenetrate each other and the ground per pixel.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
