import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Half-size (m) of the camera-centred region that gets real draws. */
  regionRadius: p.num(96, { min: 16, max: 128, step: 4 }),
  /** Alpha-test threshold on the sampled coverage. */
  coverage: p.num(0.45, { min: 0.05, max: 0.95, step: 0.05 }),
  /** One-tap depth reprojection that closes inter-layer gaps close up. */
  parallax: p.bool(true),
  /** Beyond this camera distance only the front peel layer of each stack draws. */
  layerCullDist: p.num(70, { min: 10, max: 200, step: 5 }),
  /**
   * Method-specific inspection on top of the global `view` debug modes (it
   * only applies when that is off): which baked capture direction a fragment
   * came from, which peel layer, or which draw path (near 12-card vs far
   * 2-card). Colours are flat, not lit.
   */
  inspect: p.enum('off', ['off', 'dir', 'layer', 'path'] as const),
}

export default defineExperiment({
  id: '012-ldi-composite',
  title: 'LDI composite',
  description:
    'Layered depth images: 5 key directions x 4 min-separation depth-peeled layers per species; runtime composites the 2 azimuth-nearest side stacks + the top stack as depth-writing cards with per-texel baked depth.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['calamagrostis-canescens', 'elymus-repens', 'poa-pratensis'],
  params: PARAMS,
  bakeVersion: 2,
  load: () => import('./main.ts'),
})
