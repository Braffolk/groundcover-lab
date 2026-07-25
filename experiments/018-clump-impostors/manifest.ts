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
}

export default defineExperiment({
  id: '018-clump-impostors',
  title: 'clump impostors',
  description:
    'Each plant is a 3D arrangement of 4 sub-clump cards instead of one flat quad: the source mesh is split along whole-blade boundaries into 4 spatial clumps, each baked as its own 8-azimuth + top atlas with a per-view tight crop. The sub-cards stand at their real offsets inside the plant, so they parallax against each other, self-occlude and interleave with the neighbours; past lodDistance they collapse to one merged clump card.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
