/**
 * The LIVE half of what a built material knows about itself: the GPU handles.
 *
 * `report.ts` is deliberately JSON-able so a headless script can read it; this
 * file is deliberately NOT, because an inspector that wants to show the actual
 * texture a node resolved to needs the actual texture. Keeping the two apart
 * is what lets `globalThis.__materialReports` stay a data structure while
 * `#/material/<id>` still shows real texels.
 *
 * A material publishes an instance whenever it (re)builds, and clears it on
 * dispose. `generation` changes on every publish, which is the signal an open
 * inspector re-reads on: a structural param rebuilds the graph AND replaces
 * every texture, so nothing a viewer is holding survives it.
 */

import type { ParamSchema, ParamValues } from '../harness/params.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import type { VramScope } from '../gpu/resources.ts'
import type { ResolvedGraph } from './resolve.ts'

/** One @group(1) texture binding of the built material. */
export interface MaterialTextureSlot {
  /** The node the binding belongs to (an image, a filter, a variantBlend…). */
  nodeId: string
  /**
   * The node whose data the texture actually HOLDS. Differs from `nodeId`
   * only for a discrete variantBlend, which binds one slot and points it at
   * whichever variant is active.
   */
  sourceNodeId: string
  /** @group(1) binding index. */
  binding: number
  texture: GPUTexture
  /** For a discrete variantBlend: which option is currently resident. */
  variant?: string
}

export interface MaterialInstance {
  id: string
  /** Bumped on every publish. A viewer holding stale handles compares this. */
  generation: number
  device: GPUDevice
  shaders: ShaderRegistry
  /** The material's own VRAM scope — an inspector allocates in its OWN. */
  res: VramScope
  /** The frozen @group(0), needed to run a node's materialize entry point. */
  frame: { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup }
  sampler: GPUSampler
  uniformBuffer: GPUBuffer
  schema: ParamSchema
  /** Live values — the same object the params panel mutates. */
  values: ParamValues<ParamSchema>
  resolved: ResolvedGraph
  slots: MaterialTextureSlot[]
}

const instances = new Map<string, MaterialInstance>()
const listeners = new Set<(instance: MaterialInstance | null, id: string) => void>()

export function publishMaterialInstance(instance: MaterialInstance): void {
  instances.set(instance.id, instance)
  for (const fn of listeners) fn(instance, instance.id)
}

export function clearMaterialInstance(id: string): void {
  instances.delete(id)
  for (const fn of listeners) fn(null, id)
}

export function materialInstance(id: string): MaterialInstance | undefined {
  return instances.get(id)
}

export function onMaterialInstance(fn: (instance: MaterialInstance | null, id: string) => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}
