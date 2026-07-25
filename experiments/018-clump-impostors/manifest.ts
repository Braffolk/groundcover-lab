import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),
  /** Alpha-test reference for the baked coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /**
   * Distance (m) at which a plant collapses from 4 sub-clump cards to the one
   * merged clump card. This IS the LOD knob: 0 = always merged (plain
   * billboard behaviour), large = sub-clump cards everywhere.
   */
  lodDistance: p.num(14, { min: 0, max: 40, step: 1 }),
  /** Horizontal canopy cards (fill the top-down view). */
  topCard: p.bool(true),
  /** Per-sub-clump wind phase spread — clumps sway out of step with each other. */
  swaySpread: p.num(0.55, { min: 0, max: 1.5, step: 0.05 }),
  /**
   * Alpha-reference multiplier for sub-clump cards only — the near/far density
   * balance knob. A sub-clump tile resolves ~2.3x more world detail per texel
   * than the merged tile, so the two LODs can in principle drift apart in
   * apparent density; the bake's coverage calibration already handles that, so
   * the default is a no-op. Lower it to thicken the near band.
   */
  nearAlphaBias: p.num(1, { min: 0.5, max: 1, step: 0.05 }),
  /** Fake grounding: darken card bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
  /**
   * CARPET species only (Sphagnum): distance (m) at which a tile collapses from
   * its SHELLS ground-parallel relief shells to one flat quad. Separate from
   * lodDistance because a 0.18m tile stops resolving its 3.3cm of relief at
   * ~7m, far sooner than a 1m grass clump stops showing parallax. 0 = flat
   * quads everywhere, i.e. the 001-billboard-smoke carpet, which is the honest
   * self-A/B control for what the shells buy.
   */
  carpetShellDist: p.num(8, { min: 0, max: 24, step: 1 }),
  /**
   * How much a carpet shell darkens where it is drawn below the cushion apex
   * above it — the depth cue that turns four stacked planes into a cushion.
   */
  carpetShade: p.num(0.45, { min: 0, max: 0.8, step: 0.05 }),
}

export default defineExperiment({
  id: '018-clump-impostors',
  title: 'clump impostors',
  description:
    'Each plant is a 3D arrangement of 4 sub-clump cards instead of one flat quad: the source mesh is split along whole-blade boundaries into 4 spatial clumps, each baked as its own 8-azimuth + top atlas with a per-view tight crop. The sub-cards stand at their real offsets inside the plant, so they parallax against each other, self-occlude and interleave with the neighbours; past lodDistance they collapse to one merged clump card. A carpet species (Sphagnum) turns the same idea on its side: its own bake keeps only the periodic tile seen from above plus the cushion height field, and the near LOD draws four ground-parallel shells through that relief, terrain-conformed per vertex.',
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
