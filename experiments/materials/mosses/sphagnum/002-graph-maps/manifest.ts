import { defineMaterial, HARNESS_API, p } from '@harness'

/**
 * 001-flat-maps' four shading params, verbatim and with the same defaults —
 * this material must be visually equivalent to it at defaults — plus four that
 * exist because a stage with no consumer cannot be verified at all:
 * `aoSoftenPx` gives the `filter` node kind a real user, and `parallax` /
 * `parallaxDepth` / `pomSteps` / `silhouette` give the view-uv stage one.
 * All of those are OFF or neutral by default, so the default render is 001.
 */
export const PARAMS = {
  /** Which measured micro-habitat state of the Sphagnum tile to render. */
  habitat: p.enum('wet-vigorous', ['wet-vigorous', 'late-season', 'sun-exposed'] as const),
  /** 1.0 = the measured 0.18 m tile at life size on the preview object. */
  tileScale: p.num(1, { min: 0.25, max: 4, step: 0.05 }),
  /** 0 = geometric normal only, 1 = the measured map, >1 exaggerated. */
  normalStrength: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** Weight of the measured cavity AO. */
  aoStrength: p.num(1, { min: 0, max: 1, step: 0.05 }),

  // --- structural: these change the SHAPE of the graph, not a uniform ------

  /**
   * Box-blur radius, in texels, applied to the cavity AO by a `filter` node.
   * 0 removes the node from the graph entirely (so nothing is materialized and
   * nothing extra is resident) — that is the default, and it is what makes the
   * default configuration byte-for-byte 001's.
   */
  aoSoftenPx: p.num(0, { min: 0, max: 6, step: 1 }),
  /**
   * How the `occlusion` procedural node is EXECUTED — not what it computes.
   * 'live' inlines it into the channels stage; 'materialized' bakes the exact
   * same generated `n_occlusion_eval` into a texture through a compute entry
   * point and reads it back. The two must agree, and this param is what makes
   * that falsifiable rather than asserted: flip it and diff the frames.
   */
  aoNode: p.enum('live', ['live', 'materialized'] as const),
  /** Which view-uv stage to compile. 'off' = identity, i.e. 001. */
  parallax: p.enum('off', ['off', 'offset', 'pom'] as const),
  /** March step count for `pom`. A compile-time loop bound, hence structural. */
  pomSteps: p.num(24, { min: 4, max: 64, step: 1 }),
  /** POM only: clip where the march left the mesh's uv rectangle. */
  silhouette: p.bool(false),
  /** Depth of the height field, in metres. Measured relief is ~0.09 m. */
  parallaxDepth: p.num(0.09, { min: 0, max: 0.25, step: 0.005 }),
}

export default defineMaterial({
  id: '002-graph-maps',
  title: 'Sphagnum — the same maps, as a graph',
  description:
    '001-flat-maps rebuilt as a declarative material graph: three habitats as a discrete variantBlend over image ' +
    'nodes, a filter node for the cavity AO, and the parallax/POM view-uv stages — with the BRDF still authored ' +
    'WGSL. Visually equivalent to 001 at defaults; the point is that the shader is generated, not written.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['002-graph-maps'],
  previewObject: 'sphere',
  params: PARAMS,
  load: () => import('./main.ts'),
})
