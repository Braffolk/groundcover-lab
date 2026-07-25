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
  /**
   * CARPET species only (the bog's Sphagnum): distance (m) inside which a tile
   * is drawn as its five-slice height stack. Beyond it the stack collapses
   * continuously into one merged ground-parallel card.
   */
  carpetSliceDist: p.num(14, { min: 0, max: 24, step: 1 }),
  /** Interstitial occlusion between the top and bottom slice of a carpet tile. */
  carpetAO: p.num(0.35, { min: 0, max: 0.8, step: 0.05 }),
  /**
   * Alpha reference for a carpet RELIEF SLICE (the mat's base card keeps its
   * own low one). At 0.5 a slice only paints where it owns the majority of the
   * footprint it is minified into, so relief survives while it is coherent and
   * dissolves into the flat card as it stops being resolvable.
   */
  carpetReliefRef: p.num(0.5, { min: 0.1, max: 0.9, step: 0.05 }),
  /**
   * CARPET species only: strength of the baked cushion normal. The atlas
   * stores the surface tangent, so this simply scales it — 1.0 is the measured
   * capitulum relief, 0 is a flat-lit mat (what every version before the
   * height-field bake drew). Useful as a live A/B of the normals alone.
   */
  carpetNormalGain: p.num(1, { min: 0, max: 2, step: 0.05 }),
}

export default defineExperiment({
  id: '024-slice-prisms',
  title: 'slice prisms',
  description:
    'Each plant is baked into vertical prisms: per azimuth the mesh is cut into three equal-mass depth slabs and each slab becomes its own camera-facing card, placed at that slab\'s centroid ALONG THE BAKED AZIMUTH — so the layers slide against each other as the camera orbits, occlude each other with real depth writes, and meet the ground over a footprint instead of a line. The three prisms are an exact per-texel partition of the billboard image, so nothing is lost or double-drawn. Three horizontal height slabs do the same for top-down views; four baked azimuths mirrored to eight buy 384px prism tiles in billboard VRAM; beyond ~16m the prisms collapse continuously into one merged card (a plain billboard). A CARPET species (Sphagnum) rotates the idea 90°: its whole atlas holds five horizontal slices of one 0.18m periodic tile at 0.35mm/texel, drawn as a terrain-conforming stack of ground-parallel quads, so the mat has cushion relief and parallax instead of being a flat picture.',
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
