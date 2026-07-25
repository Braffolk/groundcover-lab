import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Strength of the parallax (depth) reprojection: 0 = plain billboards. */
  reliefScale: p.num(1, { min: 0, max: 1.5, step: 0.05 }),
  /** Distance (m) where the 2-step near tier hands over to the 1-step tier. */
  nearDist: p.num(10, { min: 4, max: 32, step: 1 }),
  /** Distance (m) where parallax fades out and the flat far tier takes over. */
  midDist: p.num(34, { min: 16, max: 110, step: 2 }),
  /** Baked volumetric self-occlusion (sky visibility) strength. */
  aoStrength: p.num(0.55, { min: 0, max: 1.5, step: 0.05 }),
  /** Sun transmission through blades (near/mid tiers only). */
  translucency: p.num(0.35, { min: 0, max: 1, step: 0.05 }),
  /** Horizontal canopy card (fills the top-down view). */
  topCard: p.bool(true),
}

export default defineExperiment({
  id: '030-near-detail-injection',
  title: 'near detail injection',
  description:
    'Billboard-cost cards that reconstruct real 3D: the bake stores per-view depth + volumetric sky-visibility beside the albedo, and the fragment shader steps the true eye ray onto that depth field (1-2 analytic steps, no marching) so the card parallaxes inside its silhouette and the silhouette morphs with the view. Detail is spent by distance — 3 taps near, 2 taps with parallax mid, plain 2-tap billboard far — and the compute cull sorts plants into 6 distance buckets so near occluders draw first and early-z eats the rest.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
