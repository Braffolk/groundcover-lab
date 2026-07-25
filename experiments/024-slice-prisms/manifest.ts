import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /**
   * Distance (m, at scale 1) inside which a plant is drawn as three depth
   * prisms. Beyond it a single merged card — exactly a billboard — is used.
   * This is the LOD dial: raising it buys parallax, costs fill.
   */
  slabDist: p.num(16, { min: 0, max: 20, step: 0.5 }),
  /** Canopy self-occlusion spread between the front and rear prism. */
  canopyShade: p.num(0.4, { min: 0, max: 0.8, step: 0.05 }),
  /** Fake grounding: darken prism bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
  /** Horizontal prisms (height slices) that carry the top-down view. */
  topCards: p.bool(true),
}

export default defineExperiment({
  id: '024-slice-prisms',
  title: 'slice prisms',
  description:
    'Each plant is baked into vertical prisms: per azimuth the mesh is cut into three equal-mass depth slabs and each slab becomes its own camera-facing card, placed at that slab\'s centroid ALONG THE BAKED AZIMUTH — so the layers slide against each other as the camera orbits, occlude each other with real depth writes, and meet the ground over a footprint instead of a line. The three prisms are an exact per-texel partition of the billboard image, so nothing is lost or double-drawn. Three horizontal height slabs do the same for top-down views; four baked azimuths mirrored to eight buy 384px prism tiles in billboard VRAM; beyond ~16m the prisms collapse continuously into one merged card (a plain billboard).',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 3,
  load: () => import('./main.ts'),
})
