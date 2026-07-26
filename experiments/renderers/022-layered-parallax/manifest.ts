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

  // --- carpet species (a periodic mat, e.g. the bog's Sphagnum) ------------
  /**
   * Alpha reference for a carpet. A mat is a closed surface and must not
   * dissolve with distance: its tile is ~80% solid up close, but the mip chain
   * pulls the coverage toward the tile mean, so the grass reference (0.4)
   * punches tile-shaped holes in the mat at range.
   */
  carpetAlphaRef: p.num(0.06, { min: 0.01, max: 0.5, step: 0.01 }),
  /**
   * Cap on how far the parallax ray may travel horizontally per metre of
   * descent through the cushion. Uncapped, a few degrees above the mat sends a
   * 5cm drop two tiles sideways and the height bands stop being correlated.
   */
  carpetMaxSlope: p.num(3, { min: 0.5, max: 12, step: 0.5 }),
  /** Mip level at which the height bands start dissolving into the merged tile. */
  carpetMergeLod: p.num(5, { min: 0, max: 9, step: 0.5 }),
  /**
   * Coverage a height band needs to CLAIM a fragment. Much higher than
   * carpetAlphaRef on purpose: "does this band have geometry here" is a
   * different question from "is there moss here at all", and answering the
   * first at the mat's own reference lets the top band win on fringe texels
   * and flattens the cushion to nothing.
   */
  carpetBandRef: p.num(0.35, { min: 0.05, max: 0.9, step: 0.05 }),
  /** Canopy occlusion at the bottom of the cushion (exp falloff with depth). */
  carpetDepthShade: p.num(0.5, { min: 0, max: 0.9, step: 0.05 }),
}

export default defineExperiment({
  id: '022-layered-parallax',
  title: 'layered parallax',
  description:
    'One quad per plant, but its pixels are not flat: each species is baked per azimuth into three DEPTH SLABS, each storing the mean depth of the geometry it holds. The fragment shader intersects the real eye ray with those three planes in the plant frame and takes the first opaque hit — interior parallax, a silhouette that reshapes with the view, and correct front-to-back self-occlusion. The reprojection is a homography, so the interpolator evaluates it: one reciprocal, then two multiply-adds per slab, 1-3 taps, no marching and no frag_depth. Past lodDistance a plant collapses to one merged tile off a small fully-mipped atlas and costs exactly a billboard. A CARPET species turns the same idea on its side: the periodic tile is baked straight down into four HEIGHT BANDS over a wrapped, repeat-addressable square, and one terrain-conforming ground quad walks the eye ray down through them — cushion relief instead of canopy depth.',
  status: 'working',
  bakeVersion: 3,
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
  load: () => import('./main.ts'),
})
