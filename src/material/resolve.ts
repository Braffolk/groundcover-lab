/**
 * Graph resolution: topological order, reachability, and the ONE decision that
 * codegen turns on — is this node evaluated live in the channels stage, or
 * materialized into a texture and read back?
 *
 * The resolution rules are in types.ts; this file is where they are applied,
 * once, so codegen and the runtime cannot disagree about them.
 */

import type { MaterialGraph, MaterialNode, Scalar } from './types.ts'

export type NodeMode = 'live' | 'texture'

/** How a texture-mode node's texture is produced. */
export type TextureSource =
  /** Loaded from a PNG. */
  | 'image'
  /** Written by a generated compute entry point. */
  | 'materialize'
  /** A slot the runtime points at the ACTIVE variant of a discrete blend. */
  | 'variant'

export interface ResolvedNode {
  node: MaterialNode
  mode: NodeMode
  source?: TextureSource
  /** @group(1) binding index, for texture-mode nodes. */
  binding?: number
  /**
   * Nodes this one OWNS: a discrete variantBlend owns its variants, which get
   * no binding of their own because only one is ever resident.
   */
  owned: string[]
  /** True when some other node owns this one. */
  ownedBy?: string
}

export interface ResolvedGraph {
  graph: MaterialGraph
  byId: Map<string, ResolvedNode>
  /** Topological order over the REACHABLE nodes (dependencies first). */
  order: string[]
  /** Texture-mode, non-owned nodes in binding order. */
  textures: ResolvedNode[]
  /** Reachable from a declared channel or from the view-uv stage. */
  reachable: Set<string>
  /** True when the graph has a cycle (order is then partial). */
  hasCycle: boolean
  cycleNodes: string[]
}

/** Node ids a node depends on, in declaration order. */
export function dependenciesOf(node: MaterialNode): string[] {
  switch (node.kind) {
    case 'image':
      return []
    case 'procedural':
      return Object.values(node.inputs ?? {})
    case 'filter':
      return [node.input]
    case 'combine': {
      const extra: string[] = []
      if (node.op.kind === 'lerp') extra.push(...scalarNodes(node.op.t))
      if (node.op.kind === 'heightBlend') {
        extra.push(node.op.heightA, node.op.heightB, ...scalarNodes(node.op.t))
      }
      return [node.a, node.b, ...extra]
    }
    case 'variantBlend':
      return Object.values(node.variants)
  }
}

export function scalarNodes(s: Scalar): string[] {
  return typeof s === 'object' && 'node' in s ? [s.node] : []
}

/** Node ids the view-uv stage needs. */
export function viewUvDependencies(graph: MaterialGraph): string[] {
  const v = graph.viewUv
  switch (v.kind) {
    case 'identity':
      return []
    case 'parallaxOffset':
      return [v.height, ...scalarNodes(v.scale)]
    case 'pom':
      return [v.height, ...scalarNodes(v.scale)]
  }
}

/** The default execution mode for a node kind, before `materialize` is read. */
function autoMode(node: MaterialNode, byId: Map<string, ResolvedNode>): { mode: NodeMode; source?: TextureSource } {
  switch (node.kind) {
    // An image IS a texture.
    case 'image':
      return { mode: 'texture', source: 'image' }
    // A gather cannot honestly be a per-pixel function: expressible, but only
    // by re-running the whole kernel at every fragment.
    case 'filter':
      return { mode: 'texture', source: 'materialize' }
    case 'procedural':
    case 'combine':
      return { mode: 'live' }
    case 'variantBlend': {
      const allLive = Object.values(node.variants).every((id) => byId.get(id)?.mode === 'live')
      return allLive ? { mode: 'live' } : { mode: 'texture', source: 'variant' }
    }
  }
}

export function resolveGraph(graph: MaterialGraph): ResolvedGraph {
  const nodes = new Map<string, MaterialNode>()
  for (const n of graph.nodes) nodes.set(n.id, n)

  // --- reachability from the declared channels + the view-uv stage ---------
  const roots = [...Object.values(graph.channels), ...viewUvDependencies(graph)]
  const reachable = new Set<string>()
  const stack = [...roots]
  while (stack.length > 0) {
    const id = stack.pop()!
    if (reachable.has(id)) continue
    const node = nodes.get(id)
    if (!node) continue // reported by the validator
    reachable.add(id)
    stack.push(...dependenciesOf(node))
  }

  // --- topological order (DFS, dependencies first) -------------------------
  const order: string[] = []
  const state = new Map<string, 'open' | 'done'>()
  const cycleNodes: string[] = []
  const visit = (id: string): void => {
    const s = state.get(id)
    if (s === 'done') return
    if (s === 'open') {
      if (!cycleNodes.includes(id)) cycleNodes.push(id)
      return
    }
    const node = nodes.get(id)
    if (!node) return
    state.set(id, 'open')
    for (const dep of dependenciesOf(node)) visit(dep)
    state.set(id, 'done')
    order.push(id)
  }
  for (const id of [...reachable].sort()) visit(id)

  // --- modes, in dependency order so variantBlend can see its variants -----
  const byId = new Map<string, ResolvedNode>()
  for (const id of order) {
    const node = nodes.get(id)!
    const auto = autoMode(node, byId)
    let mode = auto.mode
    let source = auto.source
    const want = node.materialize ?? 'auto'
    if (want === 'always' && mode === 'live') {
      mode = 'texture'
      source = 'materialize'
    }
    // 'never' on an image/filter is a validator error; honour the kind here so
    // codegen never sees an impossible combination.
    if (want === 'never' && source === 'materialize' && node.kind !== 'filter') {
      mode = 'live'
      source = undefined
    }
    byId.set(id, { node, mode, ...(source && { source }), owned: [] })
  }

  // --- ownership: a discrete variantBlend owns its variants ----------------
  for (const id of order) {
    const r = byId.get(id)!
    if (r.node.kind !== 'variantBlend' || r.source !== 'variant') continue
    for (const variantId of Object.values(r.node.variants)) {
      const v = byId.get(variantId)
      if (!v) continue
      r.owned.push(variantId)
      v.ownedBy = id
    }
  }

  // --- binding assignment, deterministic (sorted by id) --------------------
  const textures = [...byId.values()]
    .filter((r) => r.mode === 'texture' && r.ownedBy === undefined)
    .sort((a, b) => a.node.id.localeCompare(b.node.id))
  textures.forEach((r, i) => {
    r.binding = MATERIAL_FIRST_TEXTURE_BINDING + i
  })

  return {
    graph,
    byId,
    order,
    textures,
    reachable,
    hasCycle: cycleNodes.length > 0,
    cycleNodes,
  }
}

/** @group(1): 0 = uniforms, 1 = sampler, 2.. = textures. */
export const MATERIAL_UNIFORM_BINDING = 0
export const MATERIAL_SAMPLER_BINDING = 1
export const MATERIAL_FIRST_TEXTURE_BINDING = 2
