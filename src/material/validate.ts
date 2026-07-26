/**
 * The material validator. It produces DATA — a list of typed issues — and
 * never touches the DOM: a later material page renders it, a headless script
 * reads it, and `createMaterialExperiment` throws on the errors so a broken
 * graph fails loudly at create() instead of drawing something plausible.
 *
 * What it checks, and why each one is here rather than in a comment:
 *
 *  - no cycles (a cycle makes codegen emit infinitely recursive WGSL);
 *  - every declared channel and every input names a node that exists;
 *  - a materialized node's subtree does not read `world`/`geo_n`/`view_ws`,
 *    which are ZERO in a materialize pass — the one way the live and
 *    materialized paths could honestly disagree;
 *  - a filter's input is texture-backed (a gather needs a texel spacing);
 *  - every variant of a blend shares one ChannelSpec, so ONE generated decoder
 *    is correct for all of them;
 *  - the material's own stage WGSL does not redefine a shared symbol (the
 *    generator already emitted frame/lighting/debug/material, and the include
 *    mechanism has no idea);
 *  - the generated fragment routes through `debug_shade` exactly once.
 *
 * The final check — that the generated WGSL actually COMPILES — needs a
 * device, so it lives in `validateCompilation()` and is awaited by the runtime.
 */

import type { ParamSchema } from '../harness/params.ts'
import type { WgslSource } from '../gpu/shaders.ts'
import { wgslIdent, wgslParamName } from './codegen.ts'
import { dependenciesOf, viewUvDependencies, type ResolvedGraph } from './resolve.ts'
import type { MaterialGraph, MaterialIssue, ValidationReport } from './types.ts'

/**
 * WGSL reserves a pile of innocent-looking identifiers. A node id or param
 * name that lands on one produces a `'X' is a reserved keyword` failure from
 * `createShaderModule` — which does not throw, it arrives as a toast.
 */
const WGSL_RESERVED = new Set([
  'active', 'alias', 'as', 'asm', 'auto', 'binding_array', 'break', 'case', 'cast', 'catch', 'class', 'common',
  'compile', 'const', 'const_assert', 'continue', 'continuing', 'crate', 'default', 'delete', 'demote', 'discard',
  'do', 'else', 'enum', 'explicit', 'export', 'extends', 'extern', 'external', 'false', 'filter', 'final', 'finally',
  'fn', 'for', 'friend', 'from', 'get', 'goto', 'groupshared', 'handle', 'has', 'if', 'impl', 'import', 'in',
  'inline', 'instanceof', 'interface', 'layout', 'let', 'line', 'linear', 'loop', 'macro', 'match', 'mediump',
  'meta', 'module', 'move', 'mut', 'mutable', 'namespace', 'new', 'nil', 'noexcept', 'noinline', 'noperspective',
  'null', 'nullptr', 'of', 'operator', 'out', 'override', 'package', 'packoffset', 'partition', 'pass', 'patch',
  'pixelfragment', 'precise', 'precision', 'premerge', 'priv', 'protected', 'pub', 'public', 'quat', 'readonly',
  'ref', 'regardless', 'register', 'reinterpret_cast', 'require', 'resource', 'restrict', 'return', 'self',
  'set', 'shared', 'sizeof', 'smooth', 'snorm', 'source', 'stage', 'standard', 'static', 'std', 'struct', 'sub',
  'subroutine', 'super', 'switch', 'target', 'template', 'this', 'thread', 'true', 'try', 'type', 'typedef',
  'typeof', 'union', 'unless', 'unorm', 'unsafe', 'unsized', 'use', 'using', 'var', 'varying', 'virtual', 'where',
  'while', 'writeonly', 'yield',
])

/** Storage-texture formats WebGPU core allows a write-only binding for. */
export const STORAGE_FORMATS = new Set<GPUTextureFormat>([
  'rgba8unorm', 'rgba8snorm', 'rgba8uint', 'rgba8sint',
  'rgba16uint', 'rgba16sint', 'rgba16float',
  'r32uint', 'r32sint', 'r32float',
  'rg32uint', 'rg32sint', 'rg32float',
  'rgba32uint', 'rgba32sint', 'rgba32float',
])

/** Shared symbols a material's own stage WGSL must not redefine. */
const SHARED_SYMBOLS = [
  'struct Frame',
  'struct Geo',
  'struct Surface',
  'struct Shaded',
  'struct EvalCtx',
  'fn light_surface',
  'fn debug_shade',
  'fn apply_fog',
]

export interface ValidateInput {
  schema: ParamSchema
  graph: MaterialGraph
  resolved: ResolvedGraph
  surface: WgslSource
  shade: WgslSource
  /** Text of `surface`/`shade` as the generator will paste it (latestCode). */
  surfaceCode: string
  shadeCode: string
}

export function validateMaterial(input: ValidateInput): ValidationReport {
  const { graph, resolved, schema } = input
  const issues: MaterialIssue[] = []
  const err = (code: string, message: string, extra: Partial<MaterialIssue> = {}): void => {
    issues.push({ level: 'error', code, message, ...extra })
  }
  const warn = (code: string, message: string, extra: Partial<MaterialIssue> = {}): void => {
    issues.push({ level: 'warning', code, message, ...extra })
  }

  // --- ids -----------------------------------------------------------------
  const ids = new Set<string>()
  for (const node of graph.nodes) {
    if (ids.has(node.id)) {
      err('graph.duplicateId', `two nodes share the id "${node.id}"`, { nodeId: node.id })
    }
    ids.add(node.id)
    const ident = wgslIdent(node.id)
    if (WGSL_RESERVED.has(ident)) {
      err(
        'graph.reservedId',
        `node id "${node.id}" becomes the WGSL identifier "${ident}", which is a reserved keyword — ` +
          'createShaderModule would fail with "is a reserved keyword". Rename it.',
        { nodeId: node.id },
      )
    }
  }
  for (const key of Object.keys(schema)) {
    const name = wgslParamName(key)
    if (WGSL_RESERVED.has(name)) {
      err('params.reservedName', `param "${key}" becomes the WGSL member "${name}", a reserved keyword — rename it.`)
    }
  }

  // --- references ----------------------------------------------------------
  for (const [channel, nodeId] of Object.entries(graph.channels)) {
    if (!ids.has(nodeId)) {
      err('channels.missingNode', `channel "${channel}" names node "${nodeId}", which does not exist`, { channel })
    }
  }
  for (const node of graph.nodes) {
    for (const dep of dependenciesOf(node)) {
      if (!ids.has(dep)) {
        err('graph.missingInput', `node "${node.id}" references "${dep}", which does not exist`, { nodeId: node.id })
      }
    }
  }
  for (const dep of viewUvDependencies(graph)) {
    if (!ids.has(dep)) err('viewUv.missingNode', `the view-uv stage references node "${dep}", which does not exist`)
  }

  // --- cycles --------------------------------------------------------------
  if (resolved.hasCycle) {
    err(
      'graph.cycle',
      `the graph has a cycle through ${resolved.cycleNodes.map((n) => `"${n}"`).join(', ')} — ` +
        'codegen would emit infinitely recursive WGSL',
    )
  }

  // --- unreachable ---------------------------------------------------------
  for (const node of graph.nodes) {
    if (!resolved.reachable.has(node.id)) {
      warn('graph.unreachable', `node "${node.id}" is not reachable from any declared channel or the view-uv stage`, {
        nodeId: node.id,
      })
    }
  }

  // --- per-kind ------------------------------------------------------------
  for (const node of graph.nodes) {
    if (!resolved.reachable.has(node.id)) continue
    const r = resolved.byId.get(node.id)
    if (!r) continue

    if ((node.kind === 'image' || node.kind === 'filter') && node.materialize === 'never') {
      err(
        'node.cannotStayLive',
        `${node.kind} node "${node.id}" declares materialize:'never', but ${
          node.kind === 'image' ? 'an image IS a texture' : 'a filter is a neighbourhood gather'
        } and cannot be a per-pixel function of EvalCtx alone`,
        { nodeId: node.id },
      )
    }

    if (node.kind === 'filter') {
      const inputRes = resolved.byId.get(node.input)
      if (inputRes && inputRes.mode !== 'texture') {
        err(
          'filter.inputNotTexture',
          `filter "${node.id}" gathers from "${node.input}", which resolves LIVE — a gather needs a texel spacing, ` +
            "so give the input materialize:'always' or make it an image",
          { nodeId: node.id },
        )
      }
      if (node.op.radiusPx < 0 || !Number.isInteger(node.op.radiusPx)) {
        err('filter.badRadius', `filter "${node.id}" has radiusPx=${node.op.radiusPx}; it must be a non-negative integer`, {
          nodeId: node.id,
        })
      }
    }

    if (node.kind === 'variantBlend') {
      const def = schema[node.selector]
      if (!def || def.kind !== 'enum') {
        err('variant.badSelector', `variantBlend "${node.id}" selects on "${node.selector}", which is not an enum param`, {
          nodeId: node.id,
        })
      } else {
        for (const option of def.options) {
          if (!node.variants[option]) {
            err('variant.missingOption', `variantBlend "${node.id}" has no variant for "${option}"`, { nodeId: node.id })
          }
        }
      }
      const specs = new Set<string>()
      for (const variantId of Object.values(node.variants)) {
        const v = resolved.byId.get(variantId)
        if (!v) continue
        specs.add(JSON.stringify(v.node.spec))
        if (v.ownedBy !== undefined && v.ownedBy !== node.id) {
          err(
            'variant.shared',
            `"${variantId}" is a variant of both "${node.id}" and "${v.ownedBy}". A discrete blend binds ONE slot and ` +
              'loads only the active variant, so a variant cannot be shared or used outside its blend',
            { nodeId: node.id },
          )
        }
      }
      if (specs.size > 1) {
        err(
          'variant.specMismatch',
          `the variants of "${node.id}" do not share one ChannelSpec. They are read through ONE generated decoder, so ` +
            'differing semantics or types would decode all but one of them wrongly',
          { nodeId: node.id },
        )
      }
    }
  }

  // --- materialization is only honest without view/world -------------------
  for (const [id, r] of resolved.byId) {
    if (r.source !== 'materialize') continue
    for (const sub of subtreeIds(resolved, id)) {
      const node = resolved.byId.get(sub)!.node
      if (node.kind !== 'procedural') continue
      const used = ['world', 'geo_n', 'view_ws'].filter((f) => new RegExp(`\\bc\\.${f}\\b`).test(node.body))
      if (used.length > 0) {
        err(
          'materialize.viewDependent',
          `"${id}" is materialized, but "${sub}" in its subtree reads c.${used.join('/c.')}. Those are ZERO in a ` +
            'materialize pass (there is no camera and no surface point), so the baked texture would not match the ' +
            "live evaluation. Give it materialize:'never', or move the view-dependent part into the shade stage",
          { nodeId: sub },
        )
      }
    }
  }

  // --- view-uv -------------------------------------------------------------
  const v = graph.viewUv
  if (v.kind === 'parallaxOffset' || v.kind === 'pom') {
    const h = resolved.byId.get(v.height)
    if (h && h.node.spec.type !== 'f32') {
      err(
        'viewUv.heightType',
        `the view-uv stage's height node "${v.height}" has type ${h.node.spec.type}; it must be f32 (0 = floor, 1 = surface)`,
        { nodeId: v.height },
      )
    }
  }
  if (v.kind === 'pom' && v.silhouette?.clip === 'alpha') {
    err(
      'viewUv.alphaClipUnimplemented',
      "silhouette clip:'alpha' is declared but not implemented: the material pipeline has no blend state, and the " +
        "house taste rule prefers a hard edge anyway. Use clip:'discard'",
    )
  }

  // --- the material's own stage WGSL ---------------------------------------
  for (const [label, src, code] of [
    ['surface', input.surface, input.surfaceCode],
    ['shade', input.shade, input.shadeCode],
  ] as const) {
    for (const symbol of SHARED_SYMBOLS) {
      if (code.includes(symbol)) {
        err(
          'stage.redefinesShared',
          `${label} stage (${src.id}) contains "${symbol}". The generator concatenates material.wgsl (which already ` +
            'brings in frame/lighting/debug) with your stage sources into ONE module, so a #include of a shared file ' +
            'or a redefinition is a duplicate-symbol error. Delete the include — everything is already in scope',
        )
      }
    }
  }
  if (!/\bfn\s+material_surface\s*\(/.test(input.surfaceCode)) {
    err(
      'stage.missingSurface',
      `the surface stage (${input.surface.id}) must define ` +
        'fn material_surface(g: Geo, ch: Channels, s: Surface) -> Surface',
    )
  }
  if (!/\bfn\s+material_shade\s*\(/.test(input.shadeCode)) {
    err(
      'stage.missingShade',
      `the shade stage (${input.shade.id}) must define fn material_shade(g: Geo, s: Surface, vu: ViewUv) -> Shaded`,
    )
  }

  return { ok: !issues.some((i) => i.level === 'error'), issues }
}

function subtreeIds(resolved: ResolvedGraph, nodeId: string): string[] {
  const seen = new Set<string>()
  const walk = (id: string): void => {
    if (seen.has(id)) return
    const r = resolved.byId.get(id)
    if (!r) return
    seen.add(id)
    if (id !== nodeId && r.mode === 'texture') return
    for (const dep of dependenciesOf(r.node)) walk(dep)
  }
  walk(nodeId)
  return [...seen]
}

/**
 * The generator, not the material, is what routes the fragment through
 * `debug_shade`. This is the regression guard on that promise: if a future
 * edit to `entryPointsWgsl` drops the call, every material starts silently
 * failing the mandatory-debug-view rule and only this fires.
 */
export function validateGeneratedMain(code: string): MaterialIssue[] {
  const issues: MaterialIssue[] = []
  const calls = code.match(/debug_shade\s*\(/g) ?? []
  // One in debug.wgsl's own definition site is not a call; count call sites in
  // the generated fragment only.
  const fragment = code.slice(code.lastIndexOf('@fragment'))
  const inFragment = (fragment.match(/debug_shade\s*\(/g) ?? []).length
  if (inFragment !== 1) {
    issues.push({
      level: 'error',
      code: 'generated.debugShade',
      message:
        `the generated fragment entry calls debug_shade ${inFragment} times (expected exactly 1; ${calls.length} in the ` +
        'whole module). Debug views are mandatory — a method whose normals and lighting cannot be inspected is a ' +
        'method nobody can trust.',
    })
  }
  return issues
}

/**
 * Compile-check a generated module. WebGPU validation errors do NOT throw —
 * they arrive on `device.onuncapturederror` as a toast — so a generator has to
 * ASK for its compilation info rather than assume silence means success.
 */
export async function validateCompilation(module: GPUShaderModule, label: string): Promise<MaterialIssue[]> {
  const info = await module.getCompilationInfo()
  const issues: MaterialIssue[] = []
  for (const m of info.messages) {
    if (m.type === 'info') continue
    issues.push({
      level: m.type === 'error' ? 'error' : 'warning',
      code: 'wgsl.compile',
      message: `${label}:${m.lineNum}:${m.linePos} ${m.message}`,
    })
  }
  return issues
}
