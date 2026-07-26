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
 * 001-flat-maps, rebuilt as a material GRAPH.
 *
 * The whole experiment is a declaration: three habitats' measured maps as
 * `image` nodes, one discrete `variantBlend` per channel so exactly one
 * habitat is resident, an optional `filter` on the cavity AO, and the view-uv
 * stage picked by a param. There is no pipeline, no bind group, no mip
 * plumbing and no `debug_shade` call anywhere in this file — the generator
 * owns all of it, which is precisely the point.
 *
 * The maps in assets/materials/sphagnum-* were measured off the 19.8M-tri
 * source mesh, which has since been deleted; they are source data now and
 * cannot be regenerated.
 */

const HABITATS = ['wet-vigorous', 'late-season', 'sun-exposed'] as const
type Habitat = (typeof HABITATS)[number]

/** What assets/materials/<dir>/measured.json carries that this material uses. */
interface Measured {
  texPx: number
  /** Period of the periodic tile, in metres. */
  tileM: number
  /** Height range the grey height map spans, in metres at mesh scale 1. */
  y0: number
  y1: number
}

// ---------------------------------------------------------------------------
// Channel specs — ONE declaration each, driving BOTH the mip filter and the
// generated decoder (src/gpu/texture.ts). The two cannot disagree, so neither
// of the documented mip bugs is expressible here.
// ---------------------------------------------------------------------------

/**
 * The albedo PNG has no alpha, so every texel reads a = 1 and this filter
 * degenerates to exactly a box over straight colour while the decoder is the
 * identity. Declaring `premultiplied-color` instead would make the decoder
 * divide by an alpha that is 1 everywhere: harmless here, and precisely the
 * reasoning that went wrong in 017-cached-clusters when alpha was not 1.
 *
 * `srgb: false` is measured, not assumed — the byte mean of albedo.png matches
 * measured.json's meanColor to three decimals, and the lab writes fragment
 * colour to a non-sRGB target throughout, so an sRGB decode here would darken
 * the moss by a full gamma for no reason.
 */
const ALBEDO: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'color-with-coverage', weight: 'a' },
  srgb: false,
  doc: 'measured linear albedo',
}

/**
 * measured.json is explicit that this map is plain xyz*0.5+0.5 and NOT
 * octahedral, "because this map is mipped and octahedral pairs are not
 * mip-averageable". `unit-vector` is the filter that matches. The hemisphere
 * is +Y because the map is MESH-FRAME with Y up (measured: the green channel
 * never drops below 128) — the axis order is (T, N, B), not (T, B, N).
 */
const NORMAL: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'unit-vector', hemisphere: [0, 1, 0] },
  doc: 'mesh-frame normal, axis order (T, N, B) — green IS the surface normal',
}

/** Cavity AO: a plain scalar with no holes, so a plain box average. */
const AO: ChannelSpec = {
  type: 'f32',
  semantics: { kind: 'scalar-field', empty: 'zero' },
  doc: 'measured cavity AO',
}

/** Cushion height, 0 = y0, 1 = y1. Only in the graph when parallax is on. */
const HEIGHT: ChannelSpec = {
  type: 'f32',
  semantics: { kind: 'scalar-field', empty: 'zero' },
  doc: 'cushion height, 0 = y0, 1 = y1',
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

/**
 * One `image` node per habitat plus the discrete `variantBlend` that selects
 * between them. Discrete is the whole reason three 48 MiB habitats cost one:
 * the runtime binds a single slot and loads only the active variant, and the
 * generated WGSL is selection-independent, so switching rebinds rather than
 * recompiles.
 */
function mapChannel(map: string, spec: ChannelSpec, id: string): MaterialNode[] {
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

export async function create(ctx: ExperimentContext<typeof PARAMS, MaterialContextExt>): Promise<Experiment> {
  // The three measured.json files are ~1.2 KB each; the tile period is what
  // the uv transform is built from, so it is read rather than guessed.
  ctx.progress(0.02, 'sphagnum: measured.json')
  const pairs = await Promise.all(HABITATS.map(async (h) => [h, await fetchMeasured(h)] as const))
  const measured = Object.fromEntries(pairs) as Record<Habitat, Measured>

  const def: MaterialDef<typeof PARAMS> = {
    id: ctx.id,
    params: PARAMS,
    surface: surfaceSrc,
    shade: shadeSrc,
    // Metres of surface -> tiles. ONE isotropic number, never a per-axis fit;
    // the only correction is snapping to a whole number of tiles on an axis
    // that closes on itself, applied to BOTH axes so tiles stay square.
    uvTransform: (mesh, params) => ({
      scale: tilesPerMetre(mesh, measured[params.habitat].tileM, params.tileScale),
    }),
    structuralParams: ['aoSoftenPx', 'aoNode', 'parallax', 'pomSteps', 'silhouette'],

    graph: (params): MaterialGraph => {
      const nodes: MaterialNode[] = [
        ...mapChannel('albedo', ALBEDO, 'albedo'),
        ...mapChannel('normal', NORMAL, 'normal'),
        ...mapChannel('ao', AO, 'ao-map'),
      ]

      // The cavity AO, optionally softened. At radius 0 the filter node is not
      // in the graph at all: an identity gather would still cost a
      // materialized 2048^2 texture and a compute pass for nothing, and the
      // default configuration has to be 001's exactly.
      let ao = 'ao-map'
      if (params.aoSoftenPx > 0) {
        ao = 'ao-soft'
        nodes.push({
          kind: 'filter',
          id: ao,
          input: 'ao-map',
          op: { kind: 'box-blur', radiusPx: Math.round(params.aoSoftenPx) },
          spec: AO,
          doc: 'cavity AO, box-blurred — the one node here that is baked to a PNG',
        })
      }

      // The occlusion curve, as a node rather than a line of the surface
      // stage. It is here for one reason beyond tidiness: it is the material's
      // only node that can honestly run BOTH ways, so `aoNode` flips its
      // execution mode and the two frames can be diffed. mix() is AFFINE in
      // the AO, and a box mip filter is affine too, so materializing commutes
      // with mip generation EXACTLY — any residual is 8-bit quantisation of
      // the intermediate and nothing else, which is what makes the comparison
      // a clean test of the duality rather than of a filter.
      nodes.push({
        kind: 'procedural',
        id: 'occlusion',
        inputs: { ao },
        body: 'return mix(1.0, in_ao, u.p.ao_strength);',
        spec: AO,
        materialize: params.aoNode === 'materialized' ? 'always' : 'auto',
        doc: 'cavity AO weighted by aoStrength',
      })

      let viewUv: ViewUvSpec = { kind: 'identity' }
      if (params.parallax !== 'off') {
        nodes.push(...mapChannel('height', HEIGHT, 'height'))
        viewUv =
          params.parallax === 'offset'
            ? { kind: 'parallaxOffset', height: 'height', scale: { param: 'parallaxDepth' }, limit: 0.3, refine: 1 }
            : {
                kind: 'pom',
                height: 'height',
                steps: Math.round(params.pomSteps),
                scale: { param: 'parallaxDepth' },
                ...(params.silhouette && { silhouette: { bounds: 'mesh' as const, clip: 'discard' as const } }),
              }
      }

      return {
        nodes,
        // Exactly the channels this material declares. `height` is NOT one of
        // them: the view-uv stage reads that node directly, so at parallax=off
        // height.png is not reachable and never gets loaded — the same
        // decision 001 made by hand, now enforced by reachability instead.
        channels: { albedo: 'albedo', normal: 'normal', ao: 'occlusion' },
        viewUv,
      }
    },
  }

  return createMaterialExperiment(ctx, def)
}
