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
  /**
   * CARPET species only (Sphagnum): ground-parallel relief shells per tile in
   * the near bucket. Shell k keeps only the texels whose baked surface height
   * reaches its own plane, so the stack is a terraced reconstruction of the
   * cushion. 1 would be a flat card; 5 puts the terraces ~1cm apart over the
   * mesh's 3.3cm of capitulum relief.
   */
  shellCount: p.num(5, { min: 2, max: 8, step: 1 }),
  /** Radius (m) inside which a carpet tile draws the shell stack instead of one quad. */
  shellRadius: p.num(8, { min: 0, max: 24, step: 1 }),
  /** Cushion occlusion: how much darker the deepest shell is than the capitula. */
  cushionShade: p.num(0.35, { min: 0, max: 0.7, step: 0.05 }),
}

export default defineExperiment({
  id: '021-geometry-near-cards-far',
  title: 'card cloud near, impostor far',
  description:
    'Near plants are a three-plane billboard cloud: the source mesh is split by azimuth into wedges baked onto three vertical planes fixed in the plant frame, so the silhouette, interior parallax and self-occlusion are real 3D. Past a per-plant jittered apparent-size radius each plant collapses to one camera-facing 8-azimuth impostor cut from the same capture box, so distant plants cost exactly what a billboard costs and the handover has nothing left to change. A carpet species (Sphagnum) takes the same near/far idea in the other axis: near, its tile is a stack of ground-parallel relief shells cut from one top-down capture with a height channel, so the cushion has real thickness; far, the stack collapses to the single terrain-conformed quad a flat card would draw.',
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
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
