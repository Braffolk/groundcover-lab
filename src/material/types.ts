/**
 * The material graph: a declarative, typed description of a surface's CHANNEL
 * DATA, plus the two places where authored WGSL takes over.
 *
 * ---------------------------------------------------------------------------
 * THE ONE ARCHITECTURAL RULE: graph for data, WGSL for behaviour.
 * ---------------------------------------------------------------------------
 *
 * A material's channels — albedo, normal, height, AO, whatever it declares —
 * are a graph the UI can render, diff and bake. Its BEHAVIOUR — the view-uv
 * stage (parallax/POM) and the BRDF — is authored WGSL with a fixed signature
 * that the generator pastes in.
 *
 * The BRDF is deliberately NOT nodes. `experiments/renderers/039-nd-moss/shaders/moss.wgsl`
 * is the reason: it interleaves a light wrap keyed off the GEOMETRY normal, a
 * fuzz layer keyed off dot(up, v), AO-Fresnel off the geometry normal again, an
 * `aperture = 2*ao*ao` micro-shadow folded into the sun term ONLY, an additive
 * bounce tinted by a measured base colour, and a subtraction that splits
 * `light_surface()` into sun and ambient halves so each gets a different
 * occluder. Expressing that as node soup would be a shader-graph-editor project
 * and would end in a worse version of code that already exists.
 *
 * ---------------------------------------------------------------------------
 * ONE DECLARATION, TWO EXECUTION MODES
 * ---------------------------------------------------------------------------
 *
 * Every node compiles to exactly one WGSL function
 *
 *     fn n_<id>_eval(c: EvalCtx) -> N_<id>_Out
 *
 * and codegen chooses the call site:
 *
 *   live         the channels stage emits `let v = n_<id>_eval(ctx);`
 *   materialized a compute entry point builds `EvalCtx` from the DESTINATION
 *                texel and calls the SAME `n_<id>_eval`, writing a storage
 *                texture; the channels stage emits a generated reader instead.
 *
 * `materialize: 'auto'` resolves as:
 *   - `image` and `filter` ALWAYS materialize. An image *is* a texture, and a
 *     filter is a neighbourhood gather — expressible per-pixel, but only by
 *     re-evaluating its whole kernel at every fragment.
 *   - `procedural` and `combine` are live.
 *   - `variantBlend` is live if all its variants are live, else the variants
 *     materialize and the blend stays live.
 *
 * THE READER'S DECODE COMES FROM THE SAME `ChannelSpec` THAT DROVE MIP
 * GENERATION — `TexelSemantics` + `wgslDecoder` from src/gpu/texture.ts, never
 * a second convention. That is what makes the premultiply/coverage trap and
 * the octahedral-mip trap unrepresentable, and it must stay that way.
 */

import type { ParamSchema, ParamValues } from '../harness/params.ts'
import type { PreviewMesh } from '../scene/preview.ts'
import type { WgslSource } from '../gpu/shaders.ts'
import type { TexelSemantics } from '../gpu/texture.ts'

// ---------------------------------------------------------------------------
// Channel specs
// ---------------------------------------------------------------------------

/** WGSL type a node produces. */
export type ChannelType = 'f32' | 'vec2f' | 'vec3f' | 'vec4f'

export const CHANNEL_COMPONENTS: Record<ChannelType, 1 | 2 | 3 | 4> = {
  f32: 1,
  vec2f: 2,
  vec3f: 3,
  vec4f: 4,
}

/**
 * What a node's value IS. The `semantics` member is the same
 * `TexelSemantics` declaration that `generateMips` filters with and
 * `wgslDecoder` generates the reader from — one declaration, so the filter and
 * the decoder cannot disagree.
 */
export interface ChannelSpec {
  type: ChannelType
  semantics: TexelSemantics
  /**
   * `true` only for a map whose bytes are sRGB-ENCODED colour. Every measured
   * or data map is linear and must leave this false. (The Sphagnum albedo is
   * LINEAR despite being a colour: its byte mean matches the measured
   * meanColor to three decimals. See the material's NOTES.md.)
   */
  srgb?: boolean
  /** One line, for NOTES/UI. */
  doc?: string
}

// ---------------------------------------------------------------------------
// Scalars a node can be driven by
// ---------------------------------------------------------------------------

/** A number in a node declaration: a literal, a param, or another node's value. */
export type Scalar = number | { param: string } | { node: string }

// ---------------------------------------------------------------------------
// Nodes — a FROZEN set of five kinds
// ---------------------------------------------------------------------------

export interface NodeBase {
  /** Unique within the graph. Becomes a WGSL identifier (`-` -> `_`). */
  id: string
  spec: ChannelSpec
  /** See the module header. Default 'auto'. */
  materialize?: 'auto' | 'always' | 'never'
  /**
   * Size of the texture this node materializes into. Defaults to the size of
   * its first texture-backed dependency (which is what a filter wants); a node
   * with no texture dependency must state it.
   */
  size?: readonly [number, number]
  doc?: string
}

/**
 * A PNG on disk. Always materialized, because an image already IS a texture.
 * The load path branches on the PNG header: gray8 goes in as `r8unorm`, 8-bit
 * colour through the ImageBitmap path, 16-bit through the hand-written codec
 * (createImageBitmap silently truncates 16-bit to 8).
 */
export interface ImageNode extends NodeBase {
  kind: 'image'
  /** Repo-absolute asset path; resolved through `assetUrl()`. */
  url: string
}

/**
 * A WGSL body evaluated per texel. Sees `c: EvalCtx`, the uniform `u`, and one
 * `in_<name>` local per declared input. Must `return` a value of `spec.type`.
 *
 * A procedural node that reads `c.world`, `c.geo_n` or `c.view_ws` cannot be
 * materialized — the validator says so rather than baking a texture off a
 * zeroed view vector.
 */
export interface ProceduralNode extends NodeBase {
  kind: 'procedural'
  /** local name -> node id. Available in `body` as `in_<name>`. */
  inputs?: Record<string, string>
  body: string
}

/** A neighbourhood gather. Always materialized. */
export type FilterOp =
  /**
   * Square box average of (2r+1)^2 taps at the INPUT texture's texel spacing,
   * reduced according to the node's `semantics` — so blurring a unit-vector
   * map flips into the hemisphere, averages and renormalises rather than
   * lerping toward zero.
   */
  { kind: 'box-blur'; radiusPx: number }

export interface FilterNode extends NodeBase {
  kind: 'filter'
  /** Must resolve to a TEXTURE-backed node: a gather needs a texel spacing. */
  input: string
  op: FilterOp
}

/** Two values combined. Live by default. */
export type CombineOp =
  | { kind: 'lerp'; t: Scalar }
  | { kind: 'multiply' }
  /**
   * Height-aware blend: whichever input stands higher wins, softened by
   * `contrast`. The plan's guard rail against linear-lerping two materials
   * into mush — each keeps its own structure.
   */
  | { kind: 'heightBlend'; heightA: string; heightB: string; t: Scalar; contrast?: number }

export interface CombineNode extends NodeBase {
  kind: 'combine'
  a: string
  b: string
  op: CombineOp
}

/**
 * N variants of the same channel, selected or blended.
 *
 * `discrete` is the only mode shipped: an enum param names exactly one active
 * variant, and the runtime binds only that variant's texture — so N habitats
 * cost ONE habitat of VRAM, which is what the Sphagnum port needs (three
 * 48 MiB map sets resident at once would be 144 MiB for a param nobody moves
 * per frame). Codegen is selection-independent: switching rebinds, it does not
 * recompile.
 *
 * DOCUMENTED HOLE: a continuous blend (spatially-varying life stage, the
 * landscape-layer case) is not built. It needs all variants resident and a
 * weight field, and neither target material needs it on day one.
 */
export interface VariantBlendNode extends NodeBase {
  kind: 'variantBlend'
  select: 'discrete'
  /** Name of an enum param in the material's schema. */
  selector: string
  /** enum option -> node id. Must cover every option of the selector. */
  variants: Record<string, string>
}

export type MaterialNode = ImageNode | ProceduralNode | FilterNode | CombineNode | VariantBlendNode

export type MaterialNodeKind = MaterialNode['kind']

/**
 * DEFERRED, deliberately: `meshCapture` (an ortho render of a source mesh into
 * albedo/height/normal, the way 039-nd-moss baked its maps) is NOT a node kind.
 * The source meshes it would capture were deleted; images are the data source
 * now. This is the documented hole, not an oversight — do not build it
 * speculatively.
 */
export const DEFERRED_NODE_KINDS = ['meshCapture'] as const

// ---------------------------------------------------------------------------
// The view-uv stage
// ---------------------------------------------------------------------------

export type ViewUvSpec =
  /** The uv the vertex stage produced, unmodified. */
  | { kind: 'identity' }
  /**
   * One offset-limited parallax step (Kaneko/Welsh), optionally refined once —
   * 039's two-tap path. NOT a march: fixed cost, two dependent taps.
   */
  | {
      kind: 'parallaxOffset'
      /** Node id of the height channel, f32 in [0,1] (1 = the surface). */
      height: string
      /**
       * Depth of the height field IN METRES. Metres, not uv units, because the
       * generated code converts the view vector into uv units first — so the
       * one number an author has to supply is a physical thickness.
       */
      scale: Scalar
      /** Smallest |dot(v, n)| the divide may see. Classic offset limiting. */
      limit?: number
      /** 0 or 1 extra solve at the new height. */
      refine?: 0 | 1
    }
  /**
   * Crimson-Desert-style silhouette POM: ray-march the height field, then
   * inverse-rotate and inverse-scale the offset, test it against the MESH's uv
   * bounds and clip where it overshot — so the silhouette follows the height
   * field rather than the triangle edge.
   */
  | {
      kind: 'pom'
      height: string
      /** Compile-time loop bound — list the param that drives it as structural. */
      steps: number
      /** Depth of the height field IN METRES (see parallaxOffset). */
      scale: Scalar
      /**
       * Present = clip against the mesh's uv rectangle. `bounds: 'mesh'` uses
       * `Geo.uv_min`/`uv_max`, which the harness fills from
       * `PreviewMesh.uvBounds`.
       */
      silhouette?: { bounds: 'mesh'; clip: 'discard' | 'alpha' }
      /**
       * Pixel depth offset. DEFAULT OFF, and that default is load-bearing:
       * WGSL has no conservative-depth qualifier, so merely DECLARING
       * `@builtin(frag_depth)` costs early-z across the whole surface. The
       * generator emits the declaration only when this is true.
       */
      pdo?: boolean
      /** Steps of a second march along the sun. 0 = no self-shadow. */
      shadowSteps?: number
    }

// ---------------------------------------------------------------------------
// The graph and the material
// ---------------------------------------------------------------------------

export interface MaterialGraph {
  nodes: readonly MaterialNode[]
  /**
   * Channel name -> node id. This is what the material DECLARES it produces;
   * the generated `Channels` struct has exactly these fields, in this order,
   * and the surface stage receives it.
   */
  channels: Record<string, string>
  viewUv: ViewUvSpec
}

export interface UvTransform {
  /** Tiles per metre. ONE isotropic number is the sane answer — see below. */
  scale: number
  /** Radians. */
  rotation?: number
}

/**
 * A material declaration. `graph` is a FUNCTION of the params, which is what
 * makes "this param changes the SHAPE of the program" expressible: listing it
 * in `structuralParams` makes the runtime rebuild the graph, the generated
 * WGSL and the pipeline when it changes, instead of only rewriting a uniform.
 */
export interface MaterialDef<S extends ParamSchema = ParamSchema> {
  /** The experiment id — namespaces baked artifacts and shader module ids. */
  id: string
  params: S
  graph: (params: ParamValues<S>) => MaterialGraph
  /** Params whose change rebuilds the graph. Everything else is a uniform. */
  structuralParams?: readonly string[]
  /**
   * Stage 4. Must define
   *   `fn material_surface(g: Geo, ch: Channels, s: Surface) -> Surface`
   * where `s` arrives as `mat_surface_default(g)`.
   */
  surface: WgslSource
  /**
   * Stage 5. Must define
   *   `fn material_shade(g: Geo, s: Surface, vu: ViewUv) -> Shaded`.
   * This is the BRDF, and it is authored WGSL on purpose (see types.ts header).
   */
  shade: WgslSource
  /**
   * Mesh uv (metres of surface) -> material uv (tile units).
   *
   * ONE isotropic number, never a per-axis fit — a per-axis factor is the
   * "material stretched to fit the polygons" bug. `tilesPerMetre()` implements
   * the only sanctioned correction (snapping to a whole number of tiles on an
   * axis that closes on itself, applied to BOTH axes so tiles stay square).
   */
  uvTransform?: (mesh: PreviewMesh, params: ParamValues<S>) => UvTransform
  /**
   * Apply the shared distance fog. Default FALSE: a studio backdrop has no
   * atmosphere, and at ~1 m the fog term is 1.6e-5. When true the generator
   * emits it guarded by `debug_mode() == DEBUG_OFF`.
   */
  fog?: boolean
  /** VRAM `species` tag. Defaults to `id` (see ExperimentManifest.species). */
  species?: string
}

// ---------------------------------------------------------------------------
// Validation output — DATA, not UI
// ---------------------------------------------------------------------------

export type IssueLevel = 'error' | 'warning'

export interface MaterialIssue {
  level: IssueLevel
  /** Stable machine-readable code, e.g. 'graph.cycle'. */
  code: string
  message: string
  nodeId?: string
  channel?: string
}

export interface ValidationReport {
  ok: boolean
  issues: MaterialIssue[]
}

export function formatIssues(issues: readonly MaterialIssue[]): string {
  return issues
    .map((i) => `  [${i.level}] ${i.code}${i.nodeId ? ` (${i.nodeId})` : ''}: ${i.message}`)
    .join('\n')
}
