/**
 * What a built material knows about itself, as DATA.
 *
 * This is the validator's surface AND the material page's: the resolved node
 * modes and EDGES, each node's ChannelSpec, the generated WGSL, the validation
 * issues, and which side of the bake cache every materialized node landed on.
 * A headless script reads it off `globalThis.__materialReports`; `#/material/
 * <id>` renders it, and renders NOTHING that is not in here or in the live
 * instance beside it — which is what makes any material inspectable the moment
 * it exists, with no per-material UI code.
 *
 * Two rules keep it that way:
 *
 *  - everything here is plain JSON-able data (no GPU objects, no DOM). Live
 *    GPU handles live in `instance.ts`, separately, precisely so this half
 *    stays readable from a headless script;
 *  - a material that FAILS validation publishes a report too, before it
 *    throws. A broken material has to be able to explain itself somewhere
 *    other than a stack trace.
 */

import type { ChannelSpec, ChannelType, MaterialIssue, MaterialNodeKind } from './types.ts'
import { dependenciesOf, type NodeMode, type ResolvedGraph, type TextureSource } from './resolve.ts'

export interface MaterializationRecord {
  nodeId: string
  key: string
  /** 'cache' = the PNG already existed; 'baked' = the compute pass ran. */
  source: 'baked' | 'cache'
  ms: number
  size: [number, number]
  /** Node ids that went into the cache key — the node's TRANSITIVE subtree. */
  subtree: string[]
  /** ONLY the params that subtree reads. */
  params: Record<string, number>
}

/**
 * One node of the resolved graph. `deps` is what makes this a GRAPH rather
 * than a list: it is the same `dependenciesOf()` codegen walks, so a picture
 * drawn from it is the topology the shader was actually generated from.
 */
export interface MaterialNodeReport {
  id: string
  kind: MaterialNodeKind
  mode: NodeMode
  source?: TextureSource
  /** Set when a discrete variantBlend owns this node (only one is resident). */
  ownedBy?: string
  /** Node ids this one reads, in declaration order — the graph's edges. */
  deps: string[]
  /** Type, TexelSemantics and srgb flag, exactly as declared on the node. */
  spec: ChannelSpec
  /** Declared materialize size, when the node states one. */
  size?: readonly [number, number]
  /** The node's own doc (NodeBase.doc), distinct from spec.doc. */
  doc?: string
}

export interface MaterialReport {
  id: string
  /** False when validation found an error — the material did not start. */
  ok: boolean
  moduleId: string
  code: string
  channels: { channel: string; nodeId: string; mode: NodeMode; type: ChannelType }[]
  nodes: MaterialNodeReport[]
  viewUv: string
  /** Node ids the view-uv stage reads (its height field, any scalar node). */
  viewUvNodes: string[]
  issues: MaterialIssue[]
  materializations: MaterializationRecord[]
  /** Bytes of every texture the material has resident, mips included. */
  textureBytes: number
}

/** Every reachable node of a resolved graph, as report data. */
export function nodeReportsOf(resolved: ResolvedGraph): MaterialNodeReport[] {
  return [...resolved.byId.values()].map((r) => ({
    id: r.node.id,
    kind: r.node.kind,
    mode: r.mode,
    ...(r.source && { source: r.source }),
    ...(r.ownedBy && { ownedBy: r.ownedBy }),
    deps: dependenciesOf(r.node),
    spec: r.node.spec,
    ...(r.node.size && { size: r.node.size }),
    ...(r.node.doc !== undefined && { doc: r.node.doc }),
  }))
}

const reports = new Map<string, MaterialReport>()
const listeners = new Set<(report: MaterialReport) => void>()

export function publishMaterialReport(report: MaterialReport): void {
  reports.set(report.id, report)
  const host = globalThis as { __materialReports?: Record<string, MaterialReport> }
  host.__materialReports = Object.fromEntries(reports)
  for (const fn of listeners) fn(report)
}

export function materialReport(id: string): MaterialReport | undefined {
  return reports.get(id)
}

export function allMaterialReports(): MaterialReport[] {
  return [...reports.values()]
}

/** Called on every publish — a rebuild replaces the graph, not just a uniform. */
export function onMaterialReport(fn: (report: MaterialReport) => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}
