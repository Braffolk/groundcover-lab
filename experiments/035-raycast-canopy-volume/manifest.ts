import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Hard alpha-test threshold on the baked coverage answer. */
  alphaRef: p.num(0.42, { min: 0.05, max: 0.95, step: 0.01 }),
  /** Mip bias for the ray-answer table (prefiltering with distance). */
  lodBias: p.num(-0.55, { min: -1.5, max: 3, step: 0.05 }),
  /** Blend the two nearest elevation bands (2 fetches) instead of snapping. */
  bandBlend: p.bool(true),
  /** Strength of the shear re-interpretation of the baked drop (see NOTES). */
  shearFix: p.num(1.0, { min: 0, max: 1, step: 0.05 }),
  /** Baked-AO strength derived from canopy depth. */
  aoStrength: p.num(0.55, { min: 0, max: 1, step: 0.02 }),
  /** Tuft-scale mixing between the two baked colour clusters. */
  tintVar: p.num(0.9, { min: 0, max: 1.5, step: 0.05 }),
  /** Sub-texel silhouette detail carved into the alpha threshold. */
  detail: p.num(0.0, { min: 0, max: 0.5, step: 0.01 }),
  /** Camera-inside erosion radius (m). */
  nearFade: p.num(0.5, { min: 0, max: 2, step: 0.05 }),
  /** Wind shear scale (1 = the shared wind model). */
  windScale: p.num(1.0, { min: 0, max: 2, step: 0.05 }),
}

export default defineExperiment({
  id: '035-raycast-canopy-volume',
  title: 'raycast canopy volume',
  description:
    'No plant primitives at all: the canopy is a baked 4D ray-answer table over a periodic tile, and one coarse ground-conformal shell resolves every eye ray to its first hit with a single texture fetch per elevation band.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
