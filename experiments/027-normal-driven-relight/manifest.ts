import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Horizontal canopy card (carries the top-down view). */
  topCard: p.bool(true),
  /** Height of the canopy card as a fraction of the plant box. */
  topPlane: p.num(0.8, { min: 0.3, max: 1, step: 0.05 }),
  /** Relief strength: 0 = plain billboard, 1 = full ray-vs-heightfield warp. */
  relief: p.num(1, { min: 0, max: 1, step: 0.05 }),
  /**
   * Relief probes in the NEAREST ring only (2 = one Newton refinement). Default
   * 1: the refinement measurably sharpens stems inside ~7 m but costs 1.28x ->
   * 1.42x of the baseline pass, and the budget is the harder constraint.
   */
  reliefSteps: p.num(1, { min: 1, max: 2, step: 1 }),
  /** Max warp offset, in units of the card half-width (clamps disocclusion). */
  reliefClamp: p.num(0.14, { min: 0.02, max: 0.4, step: 0.02 }),
  /**
   * Canopy self-shadow strength (1 sun-direction tap of the canopy heightfield).
   * OFF by default: measured contribution at the harness sun elevation (49deg)
   * is RMSE 27/65535 — see NOTES. Raise it for a low sun.
   */
  selfShadow: p.num(0, { min: 0, max: 1, step: 0.05 }),
  /** Sun-ray step for that tap, in units of the plant radius. */
  shadowStep: p.num(0.45, { min: 0.1, max: 1.2, step: 0.05 }),
  /** Baked canopy-occlusion strength. */
  ao: p.num(1, { min: 0, max: 1, step: 0.05 }),
  /** Forward-scatter glow through backlit foliage. */
  translucency: p.num(0.4, { min: 0, max: 1.5, step: 0.05 }),
  /** Card skirt driven below the terrain so bases never float (m/plant-height). */
  contactSkirt: p.num(0.05, { min: 0, max: 0.2, step: 0.01 }),
  /** Per-plant mirrored variants (free silhouette variety). */
  mirrorVariants: p.bool(true),
  /** Paint the LOD tiers instead of shading (verification view). */
  showLod: p.bool(false),
}

export default defineExperiment({
  id: '027-normal-driven-relight',
  title: 'relief-lit cards',
  description:
    'Billboard geometry (12 verts/plant) that behaves like geometry: the baked atlas carries albedo+coverage AND a per-texel normal / heightfield / canopy-occlusion field, and each fragment intersects its own eye ray with that heightfield in ONE analytic step — parallax and silhouette move inside the card, with no march and no frag_depth. Occlusion is applied as sky visibility, so clump interiors go deep without muddying sunlit tips. The cull sorts plants into four distance rings drawn near-to-far by four pipelines specialized on an override LOD_TIER, collapsing 3 taps to 1.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
