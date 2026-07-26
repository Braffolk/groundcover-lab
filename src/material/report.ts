/**
 * What a built material knows about itself, as DATA.
 *
 * This is the validator's surface (and the future material page's) without any
 * UI: the resolved node modes, the generated WGSL, the validation issues, and
 * which side of the bake cache every materialized node landed on. A headless
 * script reads it off `globalThis.__materialReports`; nothing renders it yet,
 * on purpose.
 */

import type { MaterialIssue, MaterialNodeKind } from './types.ts'
import type { NodeMode, TextureSource } from './resolve.ts'

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

export interface MaterialReport {
  id: string
  moduleId: string
  code: string
  channels: { channel: string; nodeId: string; mode: NodeMode; type: string }[]
  nodes: { id: string; kind: MaterialNodeKind; mode: NodeMode; source?: TextureSource; ownedBy?: string }[]
  viewUv: string
  issues: MaterialIssue[]
  materializations: MaterializationRecord[]
  /** Bytes of every texture the material has resident, mips included. */
  textureBytes: number
}

const reports = new Map<string, MaterialReport>()

export function publishMaterialReport(report: MaterialReport): void {
  reports.set(report.id, report)
  const host = globalThis as { __materialReports?: Record<string, MaterialReport> }
  host.__materialReports = Object.fromEntries(reports)
}

export function materialReport(id: string): MaterialReport | undefined {
  return reports.get(id)
}

export function allMaterialReports(): MaterialReport[] {
  return [...reports.values()]
}
