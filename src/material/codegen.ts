/**
 * WGSL codegen for a material graph.
 *
 * Two module shapes come out of here, and the ONE thing worth understanding is
 * that they share the node functions verbatim:
 *
 *   main         vertex + fragment. Live nodes are emitted as their bodies;
 *                materialized ones as a generated READER over their texture.
 *   materialize  one compute entry point per materialized node. It builds an
 *                `EvalCtx` from the destination texel and calls THE SAME
 *                `n_<id>_eval` the live path would have called.
 *
 * `emitNode()` is the single source of both, which is what makes
 * live-vs-materialized a caching decision rather than two implementations.
 *
 * Two traps this file is written around, both already documented in the repo:
 *
 *  1. `ShaderRegistry.module()` keys its cache on `src.id` and returns the
 *     FIRST code ever registered for that id. Every generated module id here
 *     carries a hash of its own text (`hashText`), so a regenerated module is
 *     never served stale code.
 *  2. Included WGSL is pulled through `shaders.latestCode(src)`, never
 *     `src.code`, or a hot edit to material.wgsl / a material's own shade.wgsl
 *     is silently ignored.
 *
 * The generator — not the material — routes the fragment through
 * `debug_shade()`. A graph material therefore cannot violate the
 * mandatory-debug-view rule; there is no code path that skips it.
 */

import type { ParamDef, ParamSchema, ParamValues } from '../harness/params.ts'
import type { ShaderRegistry, WgslSource } from '../gpu/shaders.ts'
import { wgslDecoder, wgslEncoder, type TexelSemantics } from '../gpu/texture.ts'
import materialAbi from '../wgsl/material.wgsl'
import {
  MATERIAL_FIRST_TEXTURE_BINDING,
  MATERIAL_SAMPLER_BINDING,
  MATERIAL_UNIFORM_BINDING,
  dependenciesOf,
  scalarNodes,
  type ResolvedGraph,
  type ResolvedNode,
} from './resolve.ts'
import {
  type ChannelSpec,
  type ChannelType,
  type MaterialGraph,
  type MaterialNode,
  type Scalar,
  type ViewUvSpec,
} from './types.ts'

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** FNV-1a over text — the same hash `paramsHash` uses, on a string. */
export function hashText(text: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16).padStart(8, '0')
}

/** A node id as a WGSL identifier: `001-albedo` -> `_001_albedo`. */
export function wgslIdent(id: string): string {
  const s = id.replace(/[^A-Za-z0-9_]/g, '_')
  return /^[A-Za-z_]/.test(s) ? s : `_${s}`
}

/** camelCase param key -> snake_case WGSL member. */
export function wgslParamName(key: string): string {
  return key.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase()
}

export function f32(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

function zeroOf(type: ChannelType): string {
  return type === 'f32' ? '0.0' : `${type}(0.0)`
}

/** `.r` / `.rg` / `.rgb` / the whole vec4, to narrow a decoder's vec4f. */
function swizzleFor(type: ChannelType): string {
  switch (type) {
    case 'f32':
      return '.r'
    case 'vec2f':
      return '.rg'
    case 'vec3f':
      return '.rgb'
    case 'vec4f':
      return ''
  }
}

/** Widen a node value back to the vec4f the encoder takes. */
function widen(expr: string, type: ChannelType): string {
  switch (type) {
    case 'f32':
      return `vec4f(${expr}, 0.0, 0.0, 1.0)`
    case 'vec2f':
      return `vec4f(${expr}, 0.0, 1.0)`
    case 'vec3f':
      return `vec4f(${expr}, 1.0)`
    case 'vec4f':
      return expr
  }
}

// ---------------------------------------------------------------------------
// The params uniform — generated from the SAME ParamSchema the Tweakpane UI,
// the URL round trip and `paramsHash` provenance are generated from.
// ---------------------------------------------------------------------------

/**
 * Every param is stored as an f32: a number verbatim, a boolean as 0/1, an
 * enum as its index in `options`. All-f32 keeps the uniform layout trivially
 * correct (align 4 throughout, one pad to a 16-byte multiple) — a mixed
 * f32/u32/vec2 struct is where hand-written uniform layouts go wrong.
 */
export function paramFloat(def: ParamDef, value: unknown): number {
  switch (def.kind) {
    case 'number':
      return typeof value === 'number' ? value : def.def
    case 'boolean':
      return value ? 1 : 0
    case 'enum': {
      const i = def.options.indexOf(String(value))
      return i >= 0 ? i : Math.max(0, def.options.indexOf(def.def))
    }
  }
}

export interface UniformLayout {
  /** Total bytes, 16-aligned. */
  size: number
  /** Float index of the first param. */
  paramOffset: number
  /** Param keys in uniform order. */
  keys: string[]
}

export const MATERIAL_UNIFORM_HEADER_FLOATS = 8 // uv_scale, uv_rot, pad, uv_min, uv_max

export function uniformLayout(schema: ParamSchema): UniformLayout {
  const keys = Object.keys(schema)
  const padded = Math.max(4, Math.ceil(Math.max(keys.length, 1) / 4) * 4)
  return {
    size: (MATERIAL_UNIFORM_HEADER_FLOATS + padded) * 4,
    paramOffset: MATERIAL_UNIFORM_HEADER_FLOATS,
    keys,
  }
}

export function writeUniforms<S extends ParamSchema>(
  schema: S,
  values: ParamValues<S>,
  uv: { scale: readonly [number, number]; rotation: number; min: readonly [number, number]; max: readonly [number, number] },
  out: Float32Array,
): void {
  out[0] = uv.scale[0]
  out[1] = uv.scale[1]
  out[2] = uv.rotation
  out[3] = 0
  out[4] = uv.min[0]
  out[5] = uv.min[1]
  out[6] = uv.max[0]
  out[7] = uv.max[1]
  const layout = uniformLayout(schema)
  layout.keys.forEach((key, i) => {
    out[layout.paramOffset + i] = paramFloat(schema[key]!, (values as Record<string, unknown>)[key])
  })
}

function paramsStructWgsl(schema: ParamSchema): string {
  const keys = Object.keys(schema)
  const members = keys.map((k) => {
    const def = schema[k]!
    const note =
      def.kind === 'enum' ? ` // enum index into [${def.options.join(', ')}]` : def.kind === 'boolean' ? ' // 0 or 1' : ''
    return `  ${wgslParamName(k)}: f32,${note}`
  })
  const padded = Math.max(4, Math.ceil(Math.max(keys.length, 1) / 4) * 4)
  for (let i = keys.length; i < padded; i++) members.push(`  _param_pad${i}: f32,`)
  return `struct MatParams {
${members.join('\n')}
}`
}

function uniformsStructWgsl(schema: ParamSchema): string {
  return `${paramsStructWgsl(schema)}

/// Harness-owned half of the material uniform. uv_scale/uv_rot are the uv
/// transform a silhouette POM has to invert; uv_min/uv_max are the mesh's
/// uv rectangle, in metres, that it tests the inverted offset against.
struct MatUniforms {
  uv_scale: vec2f,
  uv_rot: f32,
  _uni_pad0: f32,
  uv_min: vec2f,
  uv_max: vec2f,
  p: MatParams,
}`
}

// ---------------------------------------------------------------------------
// Node emission
// ---------------------------------------------------------------------------

export interface EmitContext {
  schema: ParamSchema
  resolved: ResolvedGraph
}

function nodeFn(id: string): string {
  return `n_${wgslIdent(id)}_eval`
}

function texVar(id: string): string {
  return `t_${wgslIdent(id)}`
}

function decoderName(id: string): string {
  return `dec_${wgslIdent(id)}`
}

function outAlias(id: string): string {
  return `N_${wgslIdent(id)}_Out`
}

function scalarExpr(s: Scalar, ctxVar = 'c'): string {
  if (typeof s === 'number') return f32(s)
  if ('param' in s) return `u.p.${wgslParamName(s.param)}`
  return `${nodeFn(s.node)}(${ctxVar})`
}

/** The mip/decode plan a node's texture is built and read with. */
export function planOf(spec: ChannelSpec): { semantics: TexelSemantics } {
  return { semantics: spec.semantics }
}

/**
 * The 2x2-equivalent reduction for a filter kernel, chosen by the node's OWN
 * semantics. Averaging unit vectors linearly is what the shared mip filter
 * does and is the only mip-safe way to do it; a box average of an octahedral
 * pair lands at (0,0) and decodes to exactly straight up. Same declaration,
 * same reduction, everywhere.
 */
function filterReduction(spec: ChannelSpec): { init: string; accumulate: string; resolve: string } {
  const t = spec.type
  if (spec.semantics.kind === 'unit-vector' && t === 'vec3f') {
    const h = `vec3f(${spec.semantics.hemisphere.map(f32).join(', ')})`
    return {
      init: 'var acc = vec3f(0.0);',
      accumulate: `let s = ${'${TAP}'};
      acc += select(-s, s, dot(s, ${h}) >= 0.0);`,
      resolve: `let len = length(acc);
  return select(normalize(${h}), acc / len, len > 1.0e-6);`,
    }
  }
  return {
    init: `var acc = ${zeroOf(t)};
  var count = 0.0;`,
    accumulate: `acc += ${'${TAP}'};
      count += 1.0;`,
    resolve: '  return acc / max(count, 1.0);',
  }
}

/**
 * ONE node -> ONE WGSL function. `mode` picks the call site, not the maths:
 *
 *   'compute'  the node's own body (a live evaluation)
 *   'read'     a generated reader over the texture it was materialized into
 */
export function emitNode(node: MaterialNode, mode: 'compute' | 'read', ctx: EmitContext): string {
  const fn = nodeFn(node.id)
  const alias = outAlias(node.id)
  const swz = swizzleFor(node.spec.type)
  const head = `alias ${alias} = ${node.spec.type};\n`

  if (mode === 'read') {
    return `${head}${node.spec.doc ? `// ${node.spec.doc}\n` : ''}// ${node.kind} "${node.id}" is MATERIALIZED — this is the generated reader. Its
// decode comes from the same ChannelSpec that drove mip generation.
fn ${fn}(c: EvalCtx) -> ${alias} {
  return ${decoderName(node.id)}_read(${texVar(node.id)}, mat_sampler, c.uv, c.ddx, c.ddy)${swz};
}
`
  }

  switch (node.kind) {
    case 'image':
      // Unreachable: an image is always texture-backed (see resolve.ts).
      throw new Error(`emitNode: image node "${node.id}" cannot be evaluated live — an image IS a texture`)

    case 'procedural': {
      const lets = Object.entries(node.inputs ?? {})
        .map(([name, dep]) => `  let in_${name} = ${nodeFn(dep)}(c);`)
        .join('\n')
      return `${head}${node.doc ? `// ${node.doc}\n` : ''}fn ${fn}(c: EvalCtx) -> ${alias} {
${lets}${lets ? '\n' : ''}${node.body
        .split('\n')
        .map((l) => (l.trim().length > 0 ? `  ${l}` : l))
        .join('\n')}
}
`
    }

    case 'filter': {
      const r = node.op.radiusPx
      const red = filterReduction(node.spec)
      const tap = `${nodeFn(node.input)}(cc)${node.spec.type === 'f32' ? '' : ''}`
      return `${head}${node.doc ? `// ${node.doc}\n` : ''}// filter box-blur r=${r} over "${node.input}", reduced with ${node.spec.semantics.kind}.
// ALWAYS materialized: this is a (2r+1)^2 gather, expressible per-fragment but
// absurd to pay for per-fragment.
fn ${fn}(c: EvalCtx) -> ${alias} {
  let texel = 1.0 / vec2f(textureDimensions(${texVar(node.input)}, 0));
  ${red.init}
  for (var j = -${r}; j <= ${r}; j = j + 1) {
    for (var i = -${r}; i <= ${r}; i = i + 1) {
      var cc = c;
      cc.uv = c.uv + vec2f(f32(i), f32(j)) * texel;
      ${red.accumulate.replace('${TAP}', tap)}
    }
  }
${red.resolve}
}
`
    }

    case 'combine': {
      const a = `${nodeFn(node.a)}(c)`
      const b = `${nodeFn(node.b)}(c)`
      switch (node.op.kind) {
        case 'multiply':
          return `${head}fn ${fn}(c: EvalCtx) -> ${alias} {
  return ${a} * ${b};
}
`
        case 'lerp':
          return `${head}fn ${fn}(c: EvalCtx) -> ${alias} {
  return mix(${a}, ${b}, ${scalarExpr(node.op.t)});
}
`
        case 'heightBlend': {
          const contrast = f32(node.op.contrast ?? 0.1)
          return `${head}// Height-aware blend: whichever input stands higher wins. A linear lerp of
// two materials reads as mush; this keeps each one's structure.
fn ${fn}(c: EvalCtx) -> ${alias} {
  let t = ${scalarExpr(node.op.t)};
  let ha = ${nodeFn(node.op.heightA)}(c) + (1.0 - t);
  let hb = ${nodeFn(node.op.heightB)}(c) + t;
  let m = max(ha, hb) - ${contrast};
  let wa = max(ha - m, 0.0);
  let wb = max(hb - m, 0.0);
  return (${a} * wa + ${b} * wb) / max(wa + wb, 1.0e-5);
}
`
        }
      }
      break
    }

    case 'variantBlend': {
      const def = ctx.schema[node.selector]
      const options = def && def.kind === 'enum' ? def.options : Object.keys(node.variants)
      const branches = options
        .map((opt, i) => {
          const dep = node.variants[opt]
          if (!dep) return ''
          return i === options.length - 1
            ? `  return ${nodeFn(dep)}(c);`
            : `  if (sel == ${i}u) { return ${nodeFn(dep)}(c); }`
        })
        .filter((s) => s.length > 0)
      return `${head}// variantBlend over "${node.selector}", all variants live.
fn ${fn}(c: EvalCtx) -> ${alias} {
  let sel = u32(u.p.${wgslParamName(node.selector)} + 0.5);
${branches.join('\n')}
}
`
    }
  }
  throw new Error(`emitNode: unhandled node kind`)
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

function textureDeclarations(textures: readonly ResolvedNode[], group: number, firstBinding: number): string {
  return textures
    .map((r, i) => {
      const decoder = wgslDecoder(decoderName(r.node.id), planOf(r.node.spec))
      return `@group(${group}) @binding(${firstBinding + i}) var ${texVar(r.node.id)}: texture_2d<f32>;
${decoder}`
    })
    .join('\n')
}

// ---------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------

function channelsWgsl(graph: MaterialGraph, resolved: ResolvedGraph): string {
  const entries = Object.entries(graph.channels)
  const members = entries
    .map(([name, nodeId]) => {
      const node = resolved.byId.get(nodeId)?.node
      const doc = node?.spec.doc ? ` // ${node.spec.doc}` : ''
      return `  ${name}: ${node?.spec.type ?? 'f32'},${doc}`
    })
    .join('\n')
  const assigns = entries.map(([name, nodeId]) => `  ch.${name} = ${nodeFn(nodeId)}(c);`).join('\n')
  return `/// Exactly the channels this material DECLARES it produces.
struct Channels {
${members || '  _empty: f32,'}
}

fn mat_channels(c: EvalCtx) -> Channels {
  var ch: Channels;
${assigns}
  return ch;
}
`
}

// ---------------------------------------------------------------------------
// The view-uv stage
// ---------------------------------------------------------------------------

function viewUvWgsl(spec: ViewUvSpec): string {
  const head = `fn mat_view_uv(g: Geo) -> ViewUv {
  var r: ViewUv;
  r.uv = g.uv;
  r.clip = false;
  r.shadow = 1.0;
  r.depth = 0.0;
  r.has_depth = false;`

  if (spec.kind === 'identity') {
    return `// view-uv: identity.
${head}
  return r;
}
`
  }

  if (spec.kind === 'parallaxOffset') {
    const limit = f32(spec.limit ?? 0.3)
    const refine = spec.refine ?? 0
    return `// view-uv: one offset-limited parallax step (Kaneko/Welsh)${refine ? ', refined once' : ''}.
// NOT a march — ${1 + refine + 1} taps, fixed cost.
${head}
  var c = mat_eval_ctx(g);
  let vt = mat_view_tangent(g);
  let scale = ${scalarExpr(spec.scale, 'c')};
  // Offset limiting: cap the divide so a grazing ray cannot slide the lookup
  // most of a tile away (on a periodic tile that lands in the NEXT copy of the
  // same material and reads as rings of repeats, not as parallax).
  let denom = max(abs(vt.z), ${limit});
  let step = -vt.xy / denom * scale;
  let h0 = ${nodeFn(spec.height)}(c);
  r.uv = g.uv + step * (1.0 - h0);
${
  refine
    ? `  c.uv = r.uv;
  let h1 = ${nodeFn(spec.height)}(c);
  r.uv = g.uv + step * (1.0 - h1);
`
    : ''
}  return r;
}
`
  }

  // --- pom -----------------------------------------------------------------
  const steps = Math.max(1, Math.round(spec.steps))
  const shadowSteps = Math.max(0, Math.round(spec.shadowSteps ?? 0))
  const sil = spec.silhouette
  return `// view-uv: silhouette POM. March the height field, then INVERSE-rotate and
// INVERSE-scale the offset and test it against the MESH's uv rectangle —
// leaving that rectangle means the ray left the patch, not merely one tile.
${head}
  var c = mat_eval_ctx(g);
  let vt = mat_view_tangent(g);
  let scale = ${scalarExpr(spec.scale, 'c')};
  let denom = max(abs(vt.z), 0.05);
  // uv travel per unit of NORMALISED depth (0 at the surface, 1 at the floor).
  let sweep = -vt.xy / denom * scale;
  let dt = 1.0 / ${f32(steps)};

  var t = 0.0;
  var prev_t = 0.0;
  var prev_gap = 1.0 - ${nodeFn(spec.height)}(c); // surface depth minus ray depth, at t=0
  var gap = prev_gap;
  var hit = false;
  for (var i = 0; i < ${steps}; i = i + 1) {
    prev_t = t;
    prev_gap = gap;
    t = t + dt;
    c.uv = g.uv + sweep * t;
    gap = (1.0 - ${nodeFn(spec.height)}(c)) - t;
    if (gap <= 0.0) { hit = true; break; }
  }
  // Linear solve between the last two samples (relief mapping's refinement).
  let denom_t = max(prev_gap - gap, 1.0e-6);
  let t_hit = select(t, mix(prev_t, t, prev_gap / denom_t), hit);
  r.uv = g.uv + sweep * t_hit;
${
  sil
    ? `
  // The silhouette test, in the MESH's own uv metres.
  let uv0_hit = g.uv0 + mat_uv_to_surface(r.uv - g.uv, g.uv_scale, g.uv_rot);
  r.clip = !hit || mat_uv_outside(uv0_hit, g.uv_min, g.uv_max);`
    : ''
}${
    spec.pdo
      ? `
  // Pixel depth offset. The declaration of @builtin(frag_depth) costs early-z
  // across the whole surface (WGSL has no conservative-depth qualifier), so it
  // is emitted ONLY because pdo was explicitly turned on.
  let depth_m = t_hit * scale;
  let world_hit = g.world - g.geo_n * depth_m;
  let clip_pos = frame.view_proj * vec4f(world_hit, 1.0);
  r.depth = clamp(clip_pos.z / max(clip_pos.w, 1.0e-6), 0.0, 1.0);
  r.has_depth = true;`
      : ''
  }${
    shadowSteps > 0
      ? `
  // Second march, along the sun, from the hit point upward.
  let sun_t = vec3f(dot(frame.sun_dir, g.tangent), dot(frame.sun_dir, g.bitangent), dot(frame.sun_dir, g.geo_n));
  let sun_sweep = mat_uv_forward(sun_t.xy, g.uv_scale, g.uv_rot) / max(sun_t.z, 0.15) * scale;
  var occluded = 0.0;
  let sdt = t_hit / ${f32(shadowSteps)};
  for (var i = 1; i <= ${shadowSteps}; i = i + 1) {
    let st = t_hit - sdt * f32(i);
    c.uv = r.uv - sun_sweep * (t_hit - st);
    let sh = 1.0 - ${nodeFn(spec.height)}(c);
    occluded += select(0.0, 1.0, sh < st);
  }
  r.shadow = 1.0 - saturate(occluded / ${f32(shadowSteps)}) * 0.85;`
      : ''
  }
  return r;
}
`
}

// ---------------------------------------------------------------------------
// The generated entry points
// ---------------------------------------------------------------------------

function entryPointsWgsl(def: { fog: boolean; pdo: boolean; clipDiscard: boolean }): string {
  const fragOut = def.pdo
    ? `struct MatFragOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}
`
    : ''
  const ret = def.pdo ? 'MatFragOut' : '@location(0) vec4f'
  const finish = def.pdo
    ? `  var fo: MatFragOut;
  fo.color = vec4f(debug_shade(color, sh.albedo, sh.normal, sh.coverage, g.world), 1.0);
  fo.depth = select(in.pos.z, vu.depth, vu.has_depth);
  return fo;`
    : `  return vec4f(debug_shade(color, sh.albedo, sh.normal, sh.coverage, g.world), 1.0);`

  return `struct MatVOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) tangent: vec3f,
  @location(3) bitangent: vec3f,
  @location(4) uv0: vec2f,
}
${fragOut}
// Vertex layout is PREVIEW_VERTEX_LAYOUT (src/scene/preview.ts). The preview
// objects live in world space already, so there is no model matrix and the
// tangent frame needs no transform.
@vertex
fn vs_main(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) tangent: vec4f,
  @location(3) uv: vec2f,
) -> MatVOut {
  var v: MatVOut;
  v.pos = frame.view_proj * vec4f(position, 1.0);
  v.world = position;
  v.normal = normal;
  v.tangent = tangent.xyz;
  // tangent.w is the handedness the geometry builder measured from its own
  // analytic dp/dv — never a hand-typed constant.
  v.bitangent = cross(normal, tangent.xyz) * tangent.w;
  v.uv0 = uv;
  return v;
}

fn mat_geo(in: MatVOut) -> Geo {
  var g: Geo;
  let n = normalize(in.normal);
  g.geo_n = n;
  // Re-orthogonalise after interpolation; across a curved patch the
  // interpolated T drifts out of the tangent plane.
  g.tangent = normalize(in.tangent - n * dot(n, in.tangent));
  g.bitangent = normalize(in.bitangent - n * dot(n, in.bitangent));
  g.world = in.world;
  g.view_ws = normalize(frame.camera_pos - in.world);
  g.uv0 = in.uv0;
  g.uv_scale = u.uv_scale;
  g.uv_rot = u.uv_rot;
  g.uv_min = u.uv_min;
  g.uv_max = u.uv_max;
  g.uv = mat_uv_forward(in.uv0, u.uv_scale, u.uv_rot);
  // UNPERTURBED gradients, taken before any view-uv offset. A parallax offset
  // is a per-fragment function of the sampled height, so its screen-space
  // derivative is the HEIGHTMAP's gradient — enormous — and a sampler left to
  // pick its own mip from it collapses the far field into flat paint.
  g.ddx = dpdx(g.uv);
  g.ddy = dpdy(g.uv);
  return g;
}

@fragment
fn fs_main(in: MatVOut) -> ${ret} {
  var g = mat_geo(in);
  let vu = mat_view_uv(g);
${def.clipDiscard ? '  if (vu.clip) { discard; }\n' : ''}  g.uv = vu.uv;
  let ch = mat_channels(mat_eval_ctx(g));
  let s = material_surface(g, ch, mat_surface_default(g));
  let sh = material_shade(g, s, vu);
  var color = sh.color;
${
  def.fog
    ? `  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, g.world);
  }
`
    : ''
}${finish}
}
`
}

// ---------------------------------------------------------------------------
// Module assembly
// ---------------------------------------------------------------------------

export interface GeneratedModule {
  /** Content-hashed id — see the trap note in the module header. */
  id: string
  code: string
}

export interface MainModuleInput {
  materialId: string
  schema: ParamSchema
  resolved: ResolvedGraph
  surface: WgslSource
  shade: WgslSource
  fog: boolean
  shaders: ShaderRegistry
}

export function generateMainModule(input: MainModuleInput): GeneratedModule {
  const { resolved, schema } = input
  const graph = resolved.graph
  const emitCtx: EmitContext = { schema, resolved }
  const pom = graph.viewUv.kind === 'pom' ? graph.viewUv : null

  const parts: string[] = []
  parts.push(input.shaders.latestCode(materialAbi))
  parts.push(uniformsStructWgsl(schema))
  parts.push(`@group(1) @binding(${MATERIAL_UNIFORM_BINDING}) var<uniform> u: MatUniforms;`)
  parts.push(`@group(1) @binding(${MATERIAL_SAMPLER_BINDING}) var mat_sampler: sampler;`)
  parts.push(textureDeclarations(resolved.textures, 1, MATERIAL_FIRST_TEXTURE_BINDING))
  for (const id of resolved.order) {
    const r = resolved.byId.get(id)!
    if (r.ownedBy !== undefined) continue // only the owner is bound
    parts.push(emitNode(r.node, r.mode === 'texture' ? 'read' : 'compute', emitCtx))
  }
  parts.push(channelsWgsl(graph, resolved))
  parts.push(viewUvWgsl(graph.viewUv))
  parts.push(`// ---- material-authored stage 4 (surface): ${input.surface.id} ----`)
  parts.push(input.shaders.latestCode(input.surface))
  parts.push(`// ---- material-authored stage 5 (shade / BRDF): ${input.shade.id} ----`)
  parts.push(input.shaders.latestCode(input.shade))
  parts.push(
    entryPointsWgsl({
      fog: input.fog,
      pdo: pom?.pdo === true,
      clipDiscard: pom?.silhouette?.clip === 'discard',
    }),
  )
  const code = parts.join('\n')
  return { id: `${input.materialId}/material#${hashText(code)}`, code }
}

export interface MaterializeModuleInput {
  materialId: string
  schema: ParamSchema
  resolved: ResolvedGraph
  nodeId: string
  format: GPUTextureFormat
  shaders: ShaderRegistry
}

export interface MaterializeModule extends GeneratedModule {
  nodeId: string
  /** Texture-mode dependencies, in @group(1) binding order after the sampler. */
  dependencies: ResolvedNode[]
}

/** Every node in `nodeId`'s transitive subtree, in topological order. */
export function subtreeOf(resolved: ResolvedGraph, nodeId: string): string[] {
  const seen = new Set<string>()
  const walk = (id: string): void => {
    if (seen.has(id)) return
    const r = resolved.byId.get(id)
    if (!r) return
    seen.add(id)
    // A dependency that is ITSELF materialized is a texture here — stop.
    if (id !== nodeId && r.mode === 'texture') return
    for (const dep of dependenciesOf(r.node)) walk(dep)
  }
  walk(nodeId)
  return resolved.order.filter((id) => seen.has(id))
}

export function generateMaterializeModule(input: MaterializeModuleInput): MaterializeModule {
  const { resolved, schema, nodeId } = input
  const emitCtx: EmitContext = { schema, resolved }
  const order = subtreeOf(resolved, nodeId)
  const dependencies = order
    .filter((id) => id !== nodeId && resolved.byId.get(id)!.mode === 'texture')
    .map((id) => resolved.byId.get(id)!)
  const target = resolved.byId.get(nodeId)!

  const parts: string[] = []
  parts.push(input.shaders.latestCode(materialAbi))
  parts.push(uniformsStructWgsl(schema))
  parts.push(`@group(1) @binding(0) var mat_out: texture_storage_2d<${input.format}, write>;`)
  parts.push(`@group(1) @binding(1) var<uniform> u: MatUniforms;`)
  parts.push(`@group(1) @binding(2) var mat_sampler: sampler;`)
  parts.push(textureDeclarations(dependencies, 1, 3))
  for (const id of order) {
    const r = resolved.byId.get(id)!
    if (id === nodeId) continue
    parts.push(emitNode(r.node, r.mode === 'texture' ? 'read' : 'compute', emitCtx))
  }
  // THE SAME function the live path would have called.
  parts.push(emitNode(target.node, 'compute', emitCtx))
  parts.push(wgslEncoder(`enc_${wgslIdent(nodeId)}`, planOf(target.node.spec)))
  parts.push(`// The materialize entry point. It builds EvalCtx from the DESTINATION texel and
// calls ${nodeFn(nodeId)} — the very function the channels stage would have
// inlined had this node stayed live. That is the whole duality.
@compute @workgroup_size(8, 8, 1)
fn cs_materialize(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(mat_out);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let c = mat_eval_ctx_texel(gid.xy, dims);
  let v = ${nodeFn(nodeId)}(c);
  textureStore(mat_out, gid.xy, enc_${wgslIdent(nodeId)}_encode(${widen('v', target.node.spec.type)}));
}
`)
  const code = parts.join('\n')
  return {
    id: `${input.materialId}/materialize/${nodeId}#${hashText(code)}`,
    code,
    nodeId,
    dependencies,
  }
}

/**
 * The params a node's subtree actually reads, found by scanning the generated
 * WGSL for `u.p.<name>`. This is what keeps a bake cache key SMALL: hashing
 * every param would invalidate an expensive node whenever an unrelated slider
 * moved, and nobody iterates after that.
 */
export function paramsUsedBy(schema: ParamSchema, code: string): string[] {
  return Object.keys(schema).filter((key) => code.includes(`u.p.${wgslParamName(key)}`))
}

/** Node ids whose declarations belong in a subtree's cache key. */
export function subtreeSignature(resolved: ResolvedGraph, nodeId: string): unknown[] {
  return subtreeOf(resolved, nodeId).map((id) => {
    const node = resolved.byId.get(id)!.node
    return { ...node, deps: dependenciesOf(node), scalars: scalarDeps(node) }
  })
}

function scalarDeps(node: MaterialNode): string[] {
  if (node.kind !== 'combine') return []
  if (node.op.kind === 'lerp' || node.op.kind === 'heightBlend') return scalarNodes(node.op.t)
  return []
}
