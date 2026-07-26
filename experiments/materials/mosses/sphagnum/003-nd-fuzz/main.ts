import shadeSrc from './shaders/shade.wgsl'
import surfaceSrc from './shaders/surface.wgsl'
import {
  assetUrl,
  createMaterialExperiment,
  tilesPerMetre,
  type ChannelSpec,
  type Experiment,
  type ExperimentContext,
  type MaterialContextExt,
  type MaterialDef,
  type MaterialGraph,
  type MaterialNode,
  type ViewUvSpec,
} from '@harness'
import { PARAMS } from './manifest.ts'

/**
 * 039-nd-moss' shading stack, on the material graph.
 *
 * The split the port had to respect:
 *
 * - **Everything measured is graph DATA.** Not only the four maps — the
 *   *scalars* too. `tipColor`, `baseColor` and the cushion relief differ per
 *   micro-habitat exactly the way the maps do, so they are `procedural`
 *   constant nodes behind the SAME discrete `variantBlend` the maps use. One
 *   selector moves the maps and the scalars together, and there is no second
 *   mechanism for "per-habitat number".
 * - **The BRDF is authored WGSL** (`shaders/shade.wgsl`), because the paper's
 *   stack interleaves a geometry-normal light wrap, a fuzz layer, an AO
 *   aperture folded into the sun term only and a subtraction that splits the
 *   shared lighting model in half. That is code.
 *
 * There is no pipeline, no bind group, no mip plan and no `debug_shade` call
 * in this file, and no bake: the maps under assets/materials/sphagnum-* are
 * committed source data, measured off the 19.8M-tri cushion that has since
 * been deleted.
 */

const HABITATS = ['wet-vigorous', 'late-season', 'sun-exposed'] as const
type Habitat = (typeof HABITATS)[number]

/** What assets/materials/<dir>/measured.json carries. */
interface Measured {
  texPx: number
  /** Period of the periodic tile, in metres. */
  tileM: number
  /** Height range the grey height map spans, in metres at mesh scale 1. */
  y0: number
  y1: number
  /** Mean cushion plane, metres — the surface water pools BELOW. */
  planeH: number
  /** Mean capitulum apex, metres. */
  apexH: number
  /** Fraction of the tile that is moss rather than a hole down to peat. */
  cover: number
  /** Mean albedo of the top height decile — the capitulum tips. */
  tipColor: [number, number, number]
  /** Mean albedo of the bottom 45% — the deep cushion. */
  baseColor: [number, number, number]
  meanColor: [number, number, number]
}

// ---------------------------------------------------------------------------
// Channel specs — ONE declaration each, driving BOTH the mip filter and the
// generated decoder (src/gpu/texture.ts), so the two cannot disagree.
// ---------------------------------------------------------------------------

/**
 * `srgb: false` is MEASURED, not assumed: the byte mean of albedo.png matches
 * measured.json's meanColor to three decimals, so these bytes already are the
 * linear albedo. An sRGB decode here would darken the moss by a full gamma.
 *
 * The albedo PNG has no alpha, so every texel reads a = 1 and
 * `color-with-coverage` degenerates to a plain box over straight colour with
 * an identity decoder — which is the point of declaring it rather than
 * `premultiplied-color`, whose decoder would divide by that alpha.
 */
const ALBEDO: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'color-with-coverage', weight: 'a' },
  srgb: false,
  doc: 'measured linear albedo',
}

/**
 * The normal map is MESH-FRAME with +Y up and axis order (T, N, B), so the
 * map's GREEN channel is the surface normal, not the conventional blue. That
 * is why the hemisphere is +Y: measured.json says the map is plain
 * `xyz*0.5+0.5` (never octahedral, because it is mipped and octahedral pairs
 * are not mip-averageable), and the green channel never drops below 128.
 * Reading it with the usual tangent-space assumption tilts every fragment ~90
 * degrees and looks merely "a bit off".
 */
const NORMAL: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'unit-vector', hemisphere: [0, 1, 0] },
  doc: 'mesh-frame normal, axis order (T, N, B) — green IS the surface normal',
}

/** Measured cavity AO. A plain scalar with no holes, so a plain box average. */
const AO: ChannelSpec = {
  type: 'f32',
  semantics: { kind: 'scalar-field', empty: 'zero' },
  doc: 'measured cavity AO — 8 azimuths x 5 radii, 1.05-16.9 mm',
}

/** Cushion height, 0 = y0, 1 = y1 (the top of the capture range). */
const HEIGHT: ChannelSpec = {
  type: 'f32',
  semantics: { kind: 'scalar-field', empty: 'zero' },
  doc: 'cushion height, 0 = y0, 1 = y1',
}

/**
 * A measured colour that happens to be constant over the tile. The semantics
 * are inert for a live node (nothing is filtered and nothing is decoded), but
 * they still have to be honest, because the inspector will evaluate this node
 * into a texture and read it back through exactly this declaration.
 */
const TINT: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'color-with-coverage', weight: 'a' },
  srgb: false,
  doc: 'measured constant, per habitat',
}

/** A measured scalar that happens to be constant over the tile. */
const CONST_SCALAR: ChannelSpec = {
  type: 'f32',
  semantics: { kind: 'scalar-field', empty: 'zero' },
  doc: 'measured constant, per habitat',
}

function mapUrl(habitat: Habitat, map: string): string {
  return `/assets/materials/sphagnum-${habitat}/${map}.png`
}

async function fetchMeasured(habitat: Habitat): Promise<Measured> {
  const url = assetUrl(`/assets/materials/sphagnum-${habitat}/measured.json`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  const text = await res.text()
  try {
    return JSON.parse(text) as Measured
  } catch {
    // The dev server answers a MISSING file with index.html at HTTP 200, so a
    // 200 is not proof the file exists.
    throw new Error(`${url}: not JSON (starts with ${JSON.stringify(text.slice(0, 40))}) — is the file there?`)
  }
}

/** A WGSL float literal that never comes out as an int. */
function f32(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

export async function create(ctx: ExperimentContext<typeof PARAMS, MaterialContextExt>): Promise<Experiment> {
  ctx.progress(0.02, 'sphagnum: measured.json')
  const pairs = await Promise.all(HABITATS.map(async (h) => [h, await fetchMeasured(h)] as const))
  const measured = Object.fromEntries(pairs) as Record<Habitat, Measured>

  /**
   * One `image` node per habitat plus the discrete `variantBlend` that selects
   * between them. Discrete is why three ~50 MiB habitats cost one: the runtime
   * binds a single slot and loads only the active variant, and the generated
   * WGSL is selection-independent, so switching rebinds rather than recompiles.
   */
  const mapChannel = (map: string, spec: ChannelSpec, id: string): MaterialNode[] => {
    const variants: Record<string, string> = {}
    const nodes: MaterialNode[] = []
    for (const habitat of HABITATS) {
      const nodeId = `${id}-${habitat}`
      variants[habitat] = nodeId
      nodes.push({ kind: 'image', id: nodeId, url: mapUrl(habitat, map), spec })
    }
    nodes.push({ kind: 'variantBlend', id, select: 'discrete', selector: 'habitat', variants, spec })
    return nodes
  }

  /**
   * THE MEASURED SCALARS, as graph data.
   *
   * `tipColor` / `baseColor` / the cushion relief are per-habitat measured
   * numbers, exactly like the maps, and they change with the SAME selector. So
   * they are constant `procedural` nodes behind the same discrete
   * `variantBlend` — which costs nothing (every variant is live, so the blend
   * stays live and no texture exists), keeps `habitat` a pure selector rather
   * than a structural param, and puts the numbers where the inspector can show
   * them beside the maps they were measured with.
   *
   * The alternative — baking them into the shade WGSL, or adding nine
   * `p.num()` sliders for numbers nobody should drag — is what this makes
   * unnecessary.
   */
  const measuredNode = (
    id: string,
    spec: ChannelSpec,
    body: (m: Measured) => string,
    doc: string,
  ): MaterialNode[] => {
    const variants: Record<string, string> = {}
    const nodes: MaterialNode[] = []
    for (const habitat of HABITATS) {
      const nodeId = `${id}-${habitat}`
      variants[habitat] = nodeId
      nodes.push({
        kind: 'procedural',
        id: nodeId,
        body: `return ${body(measured[habitat])};`,
        spec,
        doc: `${doc} (${habitat})`,
      })
    }
    nodes.push({ kind: 'variantBlend', id, select: 'discrete', selector: 'habitat', variants, spec, doc })
    return nodes
  }

  const vec3 = (v: readonly [number, number, number] | number[]): string =>
    `vec3f(${v.map((c) => f32(c)).join(', ')})`

  const def: MaterialDef<typeof PARAMS> = {
    id: ctx.id,
    params: PARAMS,
    surface: surfaceSrc,
    shade: shadeSrc,
    // Metres of surface -> tiles. ONE isotropic number, never a per-axis fit.
    uvTransform: (mesh, params) => ({
      scale: tilesPerMetre(mesh, measured[params.habitat].tileM, params.tileScale),
    }),
    structuralParams: ['parallax', 'pomSteps', 'parallaxShadowSteps'],

    graph: (params): MaterialGraph => {
      const nodes: MaterialNode[] = [
        ...mapChannel('albedo', ALBEDO, 'albedo'),
        ...mapChannel('normal', NORMAL, 'normal'),
        ...mapChannel('ao', AO, 'ao-map'),
        // Unlike 002, `height` is a CHANNEL here as well as the view-uv
        // stage's input: the paper's wetness pools water in the crevices, and
        // that is a per-fragment function of the height. See NOTES.md — a node
        // with two consumers is evaluated twice, and this is the first
        // material in the repo where that is not free.
        ...mapChannel('height', HEIGHT, 'height'),
        ...measuredNode('tip-color', TINT, (m) => vec3(m.tipColor), 'measured capitulum-tip albedo — the fuzz tint'),
        ...measuredNode('base-color', TINT, (m) => vec3(m.baseColor), 'measured deep-cushion albedo — the wrap tint'),
        // The mean cushion plane, in height-map units. Water pools BELOW it;
        // 039 used a fixed midpoint here because its bake header did not carry
        // the plane into the shader.
        ...measuredNode(
          'plane-norm',
          CONST_SCALAR,
          (m) => f32((m.planeH - m.y0) / (m.y1 - m.y0)),
          'mean cushion plane in height-map units',
        ),
      ]

      // The AO curve. A curve over measured data is DATA, so it is a node —
      // and unlike 039's `pow(ao, max(k, 0.01))` this reaches exactly 1.0 at
      // k = 0, which is what makes the light_surface() identity checkable.
      nodes.push({
        kind: 'procedural',
        id: 'occlusion',
        inputs: { ao: 'ao-map' },
        body: `let k = u.p.ao_strength;
if (k <= 0.0) {
  return 1.0;
}
return pow(max(in_ao, 1.0e-4), k);`,
        spec: AO,
        doc: 'cavity AO, darkened by pow(ao, aoStrength)',
      })

      let viewUv: ViewUvSpec = { kind: 'identity' }
      if (params.parallax !== 'off') {
        // Parallax depth in METRES = the measured relief x a gain. `Scalar` is
        // a literal OR a param OR a node — it has no arithmetic — so combining
        // a per-habitat measured constant with a user gain is exactly what
        // `combine: multiply` is for. (002's NOTES flagged `combine` as the
        // least-tested node kind, shipped without a consumer; this is one.)
        nodes.push(
          ...measuredNode(
            'relief-measured',
            CONST_SCALAR,
            (m) => f32(m.y1 - m.y0),
            'measured cushion relief, metres (y1 - y0)',
          ),
        )
        nodes.push({
          kind: 'procedural',
          id: 'parallax-gain',
          body: 'return u.p.parallax_depth;',
          spec: CONST_SCALAR,
          doc: 'user gain on the measured relief',
        })
        nodes.push({
          kind: 'combine',
          id: 'relief-m',
          a: 'relief-measured',
          b: 'parallax-gain',
          op: { kind: 'multiply' },
          spec: CONST_SCALAR,
          doc: 'parallax depth in metres',
        })

        viewUv =
          params.parallax === 'offset'
            ? // 039's stage: one offset-limited step (Kaneko/Welsh) plus one
              // secant refinement. Two dependent taps, fixed cost, no march.
              //
              // WHY limit = 2.0, and not 039's PARALLAX_LIMIT of 0.30. 039 also
              // BOUNDS THE TRAVEL at half the relief, and says why: on a
              // periodic tile a large offset lands in the next copy of the same
              // moss and reads as rings of repeats rather than as parallax.
              // This stage has no travel bound — only `limit`, the clamp on the
              // denominator — so the bound has to be expressed through it.
              // The generated code is
              //   step = -vt.xy / max(|vt.z|, limit) * scale
              // and `vt.xy` is already in TILES per metre, so the worst-case
              // offset is tilesPerMetre * scale / limit tiles. Setting that
              // equal to 039's bound (0.5 * relief * tilesPerMetre tiles) gives
              //   limit = 2 * parallaxDepth
              // with tilesPerMetre CANCELLING — so a fixed 2.0 reproduces the
              // bound on every preview object and at every tileScale, and it
              // tracks the depth gain automatically.
              //
              // Consequence, stated plainly: at limit >= 1 the max() can never
              // select |vt.z| (it is <= 1), so this degenerates to
              // `step = -vt.xy * scale/2` — Welsh's offset-limited parallax (the
              // variant that drops the divide by z entirely) at half strength,
              // and the half IS the travel bound. It matters here because the
              // relief is HALF the tile period (0.093 m in 0.18 m). Measured at
              // `macro`: 0.45 -> 1.15 tiles of travel and a ~90 px radial smear,
              // 1.0 -> 0.52 tiles and visible streaking, 2.0 -> 0.26 tiles and
              // individual capitula resolve. See NOTES.md.
              { kind: 'parallaxOffset', height: 'height', scale: { node: 'relief-m' }, limit: 2.0, refine: 1 }
            : {
                kind: 'pom',
                height: 'height',
                steps: Math.round(params.pomSteps),
                scale: { node: 'relief-m' },
                // The ONLY way to get the paper's parallax shadows (slide 53)
                // out of this system: `parallaxOffset` never fills
                // `ViewUv.shadow`. Silhouette clipping stays OFF — deferred.
                shadowSteps: Math.round(params.parallaxShadowSteps),
              }
      }

      return {
        nodes,
        channels: {
          albedo: 'albedo',
          normal: 'normal',
          ao: 'occlusion',
          height: 'height',
          tip_color: 'tip-color',
          base_color: 'base-color',
          plane_norm: 'plane-norm',
        },
        viewUv,
      }
    },
  }

  return createMaterialExperiment(ctx, def)
}
