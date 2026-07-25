import { defineExperiment, HARNESS_API, p } from '@harness'

// Render params ONLY — placement comes from the stand (see CLAUDE.md).
export const PARAMS = {
  /** Radius (m) of the camera-centered region that is evaluated per frame. */
  regionRadius: p.num(110, { min: 24, max: 128, step: 4 }),

  // --- moss LOD bands --------------------------------------------------------
  /**
   * Within this distance (m) a carpet tile is a displaced patch, not a quad.
   * The max is bounded because the near instance list is sized for it.
   */
  patchDist: p.num(7, { min: 0, max: 12, step: 0.5 }),
  /** Tessellation of a near patch (NxN quads). */
  patchDiv: p.num(8, { min: 2, max: 12, step: 2 }),
  /**
   * Beyond this distance (m) 2x2 tiles merge into one phase-locked quad, and
   * the fragment path drops parallax and the self-shadow tap. Bounded like
   * patchDist because the mid instance list is sized for it.
   */
  mergeDist: p.num(26, { min: 8, max: 32, step: 2 }),

  // --- relief ----------------------------------------------------------------
  /** Strength of the (single-step, offset-limited) parallax into the cushion. */
  parallax: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** One secant refinement of the parallax offset (1 extra height tap). */
  parallaxRefine: p.bool(true),
  /** Geometric displacement of a near patch, x the baked relief. */
  displace: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** Height below which a texel is a real gap down to the peat (alpha test). */
  gapRef: p.num(0.05, { min: 0, max: 0.4, step: 0.01 }),

  // --- the Uncharted 4 fuzzy-moss material ----------------------------------
  /**
   * Baked cavity AO, applied as pow(ao, aoStrength) — the paper's moss AO
   * "goes very dark", and a physically measured one is darker still (mean 0.60
   * over the tile). 0.65 keeps the inter-capitulum crevices black while letting
   * the tops light properly; at 1.0 the whole mat sits at ~55% of the light of
   * an AO-free renderer, which reads as a hole rather than as a cushion.
   */
  aoStrength: p.num(0.65, { min: 0, max: 2, step: 0.05 }),
  /** Sun micro shadowing from the AO aperture (paper, slide 37). */
  microShadow: p.bool(true),
  /**
   * How much a glancing view angle fades the AO out ("AO Fresnel", slide 40).
   * 1.0 is the paper verbatim; a carpet is seen at a glancing angle nearly
   * everywhere, and fading its AO out entirely there erases the only cue that
   * makes the cushion read as 3D.
   */
  aoFresnel: p.num(0.25, { min: 0, max: 1, step: 0.05 }),
  /** Wrapped diffuse past the terminator, tinted by the baked base colour. */
  lightWrap: p.num(0.55, { min: 0, max: 1, step: 0.05 }),
  /** Microfiber sheen ("Fuzz BRDF"), tinted by the baked capitulum-tip colour. */
  fuzz: p.num(0.25, { min: 0, max: 1.5, step: 0.05 }),
  /** Bounce/SSS colour added back where AO is dark ("AO saturation"). */
  aoSaturation: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** One-tap parallax self-shadow toward the sun (paper, slide 53). */
  parallaxShadow: p.bool(true),
  /** The paper's unified wetness function, driven by each state's wetness zone. */
  wetness: p.num(0.7, { min: 0, max: 1, step: 0.05 }),

  // --- grass path (upright plants; unchanged card baseline) -----------------
  /** Alpha-test reference for the baked grass coverage. */
  alphaRef: p.num(0.4, { min: 0.1, max: 0.8, step: 0.05 }),
  /** Horizontal canopy card on upright plants (fills the top-down view). */
  topCard: p.bool(true),
  /** Fake grounding: darken card bottoms toward the soil. */
  bottomShade: p.num(0.25, { min: 0, max: 0.6, step: 0.05 }),
}

export default defineExperiment({
  id: '039-nd-moss',
  title: 'ND moss (Uncharted 4 fuzzy-moss material)',
  description:
    "Naughty Dog's Uncharted 4 moss recipe, rebuilt for a Sphagnum carpet. Its moss is a MATERIAL, not a geometry method, so the four maps it eats (colour / normal / HEIGHT / very dark AO) are baked off the 19.8M-tri source at 0.088mm per texel, with the mesh drawn 3x3 so the tile is exactly periodic; the shader then runs the talk's fuzzy-surface stack — one-step offset-limited parallax, one-tap parallax self-shadow, AO Fresnel, sun micro shadowing, tinted wrapped diffuse, microfibre fuzz, additive AO saturation, and the unified wetness function driven by each state's own habitat zone. The thickness the material alone cannot give comes from geometry: within 7m a tile is a displaced patch whose boundary follows the ROTATION-SYMMETRISED heightfield, which is invariant under the lattice's quarter turns and therefore crack-free without pinning the edge flat; mid tiles are flat quads with parallax; beyond 26m 2x2 tiles merge into one phase-locked quad carrying four packed rotations. Upright plants keep the baked-card baseline verbatim.",
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
  bakeVersion: 1,
  load: () => import('./main.ts'),
})
