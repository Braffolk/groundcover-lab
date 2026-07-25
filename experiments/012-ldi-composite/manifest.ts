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
   * Carpet species only (a mat, `carpet_div > 0`): distance at which a tile
   * drops from 2 depth-writing layers to 1 flat early-z layer. The cushion's
   * relief is 3.3cm, so a few metres out it is worth well under a pixel.
   */
  carpetNear: p.num(10, { min: 0, max: 60, step: 1 }),
  /**
   * Alpha reference for mat tiles, instead of `coverage`. A mat is a closed
   * surface: its mip chain pulls coverage toward the tile mean, so the grass
   * reference would fail entire distant tiles and punch holes in the carpet.
   */
  carpetAlpha: p.num(0.06, { min: 0.02, max: 0.6, step: 0.02 }),
  /**
   * Uniform overscale of every mat tile (grid spacing and yaw untouched, so
   * the lattice invariant holds). 1.0 = exact abutment; see NOTES.md for the
   * measurement.
   */
  carpetOverscale: p.num(1, { min: 1, max: 1.4, step: 0.05 }),
  /**
   * Mip bias on the mat texture (negative = sharper). A ground-parallel card of
   * a relief surface asks for a mip level as if the surface were flat, which
   * over-blurs at oblique angles because the bumps facing the camera are not
   * actually foreshortened. See NOTES.md for the measured trade-off.
   */
  carpetSharpen: p.num(-1, { min: -3, max: 0, step: 0.25 }),
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
    'Layered depth images: 5 key directions x 4 min-separation depth-peeled layers per species; runtime composites the 2 azimuth-nearest side stacks + the top stack as depth-writing cards with per-texel baked depth. Carpet species (Sphagnum) instead use the top capture alone as a displacement map: one ground-parallel tile-sized quad per grid node, periodic + mipped, terrain-conformed, with the baked depth written per texel so the mat has real cushion relief.',
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
