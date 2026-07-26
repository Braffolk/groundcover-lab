import { defineMaterial, HARNESS_API, p } from '@harness'

/**
 * The Uncharted 4 fuzzy-moss stack, as a graph material.
 *
 * The shading params are 039-nd-moss' — same names, same ranges, same
 * DEFAULTS — because this is a port, and a port that quietly retunes is not a
 * port. The ones 039 does not have (`tileScale`, `normalStrength`, `parallax`,
 * `pomSteps` / `parallaxShadowSteps`) exist because a material has no stand to
 * inherit a tile size from, and because the view-uv stage is chosen
 * declaratively here rather than compiled in by hand.
 */
export const PARAMS = {
  /** Which measured micro-habitat state of the Sphagnum tile to render. */
  habitat: p.enum('wet-vigorous', ['wet-vigorous', 'late-season', 'sun-exposed'] as const),
  /** 1.0 = the measured 0.18 m tile at life size on the preview object. */
  tileScale: p.num(1, { min: 0.25, max: 4, step: 0.05 }),
  /** 0 = geometry normal only, 1 = the measured map, >1 exaggerated. */
  normalStrength: p.num(1, { min: 0, max: 2, step: 0.05 }),

  // --- the paper's fuzzy-moss stack (039's params, verbatim) ----------------

  /**
   * Measured cavity AO, applied as `pow(ao, aoStrength)` — "for moss, the AO
   * goes very dark" (slide 29), and a *measured* cavity AO is darker still.
   * 0 gives exactly 1.0 (no occlusion at all), which is what makes the
   * `light_surface()` identity test falsifiable; 039 clamped the exponent to
   * 0.01 and could therefore never quite reach it.
   */
  aoStrength: p.num(0.65, { min: 0, max: 2, step: 0.05 }),
  /**
   * AO Fresnel (slides 38-40): how much a glancing view angle fades the AO
   * out. 1.0 is the paper verbatim. 039 shipped 0.25 and this port keeps that
   * — see NOTES.md; the short version is that the paper's moss coats geometry
   * that owns the silhouette, so fading its cracks out at a glancing angle
   * costs nothing, while a mat (or a sphere seen at its limb) is glancing
   * everywhere and loses the only cue that makes it read as a cushion.
   */
  aoFresnel: p.num(0.25, { min: 0, max: 1, step: 0.05 }),
  /** Sun micro shadowing from the AO aperture (slide 37, verbatim). */
  microShadow: p.bool(true),
  /** Wrapped diffuse past the terminator, tinted by the measured base colour. */
  lightWrap: p.num(0.55, { min: 0, max: 1, step: 0.05 }),
  /** Microfibre sheen ("Fuzz BRDF"), tinted by the measured tip colour. */
  fuzz: p.num(0.25, { min: 0, max: 1.5, step: 0.05 }),
  /** Bounce/SSS colour added back where the AO gets dark ("AO saturation"). */
  aoSaturation: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /**
   * The unified wetness function (slides 63-65). In 039 this param was
   * multiplied by the stand entry's own habitat-zone centre; a material has no
   * stand, so here it IS the wetness. 0 leaves albedo and normal untouched.
   */
  wetness: p.num(0.7, { min: 0, max: 1, step: 0.05 }),

  // --- structural: these change the SHAPE of the program, not a uniform -----

  /**
   * Which view-uv stage to compile. `offset` is 039's: ONE offset-limited
   * parallax step plus one secant refinement — two dependent taps, not a
   * march. `pom` is the only way to get the paper's parallax SHADOWS, because
   * `parallaxOffset` never fills `ViewUv.shadow` (see NOTES.md).
   */
  parallax: p.enum('offset', ['off', 'offset', 'pom'] as const),
  /** Gain on the MEASURED relief (y1 - y0) that sets the parallax depth. */
  parallaxDepth: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** March step count for `pom`. A compile-time loop bound, hence structural. */
  pomSteps: p.num(24, { min: 4, max: 64, step: 1 }),
  /** `pom` only: steps of the second march along the sun. 0 = no self-shadow. */
  parallaxShadowSteps: p.num(8, { min: 0, max: 32, step: 1 }),
}

export default defineMaterial({
  id: '003-nd-fuzz',
  title: 'Sphagnum — the Uncharted 4 fuzzy-moss BRDF',
  description:
    "Naughty Dog's fuzzy-moss material (Maximov/Ganev, SIGGRAPH 2016) ported onto the material graph. The measured " +
    'maps AND the measured scalars are graph data — three habitats as a discrete variantBlend, the capitulum-tip and ' +
    'deep-cushion tints and the cushion relief as per-habitat constant nodes — while the paper\'s genuinely ' +
    'non-standard BRDF stays authored WGSL: a light wrap keyed off the GEOMETRY normal, a fuzz/sheen layer as the ' +
    'explicit subpixel answer, sun micro shadowing from the AO aperture, AO Fresnel, an additive AO-saturation ' +
    'bounce, and a subtraction that splits the shared lighting model so cavity AO occludes the indirect half while ' +
    'the micro shadow occludes the sun.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['003-nd-fuzz'],
  previewObject: 'sphere',
  params: PARAMS,
  load: () => import('./main.ts'),
})
