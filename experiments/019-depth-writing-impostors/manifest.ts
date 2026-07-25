import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /**
   * Alpha-test reference for CARPET species, instead of `alphaRef`. A mat is a
   * closed surface and must not dissolve with distance: a tile's own alpha is
   * ~80% solid up close, but the mip chain pulls it toward the whole capture's
   * mean, so at the 0.4 grass reference entire distant tiles fail the test and
   * punch tile-shaped holes in the carpet. Low keeps the mat a solid
   * depth-writing occluder while the genuinely empty texels still open.
   */
  carpetAlphaRef: p.num(0.06, { min: 0.02, max: 0.5, step: 0.02 }),
  /** Distance (m) out to which impostors write true per-pixel depth. */
  depthDist: p.num(6, { min: 0, max: 24, step: 1 }),
  /** Distance (m) out to which the depth warp runs (beyond it: 1 tap, flat). */
  warpDist: p.num(30, { min: 4, max: 64, step: 1 }),
  /**
   * Depth-warp strength — 0 turns the method back into a plain billboard. For
   * carpet species it scales the relief step instead (the per-pixel depth
   * write stays; `depthDist` owns that).
   */
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
    'Impostors whose texels are real 3D points: a baked view set stores signed depth (plus normals and canopy AO) per texel, the fragment shader inverts the image->screen map with one depth tap to warp the imagery to the exact current view, and the near band writes that reconstructed point as frag_depth so plants interpenetrate each other and the ground per pixel. Carpet species keep the same idea on a ground-parallel, terrain-conformed tile: the top view\'s depth channel becomes real cushion relief — parallax plus true per-pixel depth — instead of a flat plane.',
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
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
