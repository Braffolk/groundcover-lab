/**
 * Materializing a node: run its generated compute entry point into a texture,
 * and cache the result as a PNG through the ordinary bake flow
 * (`BakeContext.ext = 'png'` -> OPFS -> `mesh/baked/<exp>/<key>.png`).
 *
 * ---------------------------------------------------------------------------
 * THE CACHE KEY IS THE WHOLE POINT
 * ---------------------------------------------------------------------------
 *
 * It hashes the node's TRANSITIVE SUBTREE — its own declaration and its
 * dependencies' — plus the generated WGSL for that subtree (which carries any
 * inline leaf bodies verbatim) and ONLY the params that subtree actually
 * reads, found by scanning the generated text for `u.p.<name>`.
 *
 * It deliberately does NOT hash the whole graph or the whole params object. A
 * whole-graph key means touching any node — or nudging any slider — throws
 * away every expensive bake, and after that nobody iterates. `bakeKeyFor()`
 * below is the entire mechanism, and `materialization report` exposes which
 * side of the cache each node landed on so the property is measurable rather
 * than asserted.
 */

import { bakedArtifact } from '../bake/io.ts'
import { decodePng, encodePng } from '../gpu/image.ts'
import type { VramAttr, VramScope } from '../gpu/resources.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import { createTexture2D, generateMips, mipLevelsFor, readTexture } from '../gpu/texture.ts'
import type { ParamSchema, ParamValues } from '../harness/params.ts'
import {
  generateMaterializeModule,
  hashText,
  paramFloat,
  paramsUsedBy,
  subtreeOf,
  subtreeSignature,
  uniformLayout,
  type MaterializeModule,
} from './codegen.ts'
import type { ResolvedGraph, ResolvedNode } from './resolve.ts'

/**
 * The only 8-bit format WebGPU core allows as a write-only storage texture.
 * `r8unorm` is NOT storage-capable in core, so a scalar node materializes into
 * four channels and uses one — stated here rather than discovered from a VRAM
 * bar that came out 4x too big.
 */
export const MATERIALIZE_FORMAT: GPUTextureFormat = 'rgba8unorm'

export interface MaterializeKeyInput<S extends ParamSchema> {
  resolved: ResolvedGraph
  nodeId: string
  schema: S
  values: ParamValues<S>
  code: string
  size: readonly [number, number]
}

export interface BakeKey {
  key: string
  /** Everything that went into the hash — logged so a miss is explainable. */
  parts: { subtree: string[]; params: Record<string, number>; size: readonly [number, number] }
}

export function bakeKeyFor<S extends ParamSchema>(input: MaterializeKeyInput<S>): BakeKey {
  const used = new Set(paramsUsedBy(input.schema, input.code))
  // A discrete variantBlend in the subtree reads its selector through the BIND
  // GROUP, not through `u.p.<name>` — so a text scan cannot see it, and a key
  // without it would serve one habitat's blurred AO for another's. Add every
  // selector the subtree actually depends on.
  for (const id of subtreeOf(input.resolved, input.nodeId)) {
    const dep = input.resolved.byId.get(id)
    if (dep?.node.kind === 'variantBlend' && dep.source === 'variant') used.add(dep.node.selector)
  }
  const params: Record<string, number> = {}
  for (const key of used) {
    const def = input.schema[key]
    if (!def) continue
    params[key] = paramFloat(def, (input.values as Record<string, unknown>)[key])
  }
  const signature = subtreeSignature(input.resolved, input.nodeId)
  const material = JSON.stringify({
    subtree: signature,
    params,
    size: input.size,
    format: MATERIALIZE_FORMAT,
    // The generated text subsumes any inline WGSL leaf body in the subtree.
    code: input.code,
  })
  return {
    key: `${input.nodeId}-${hashText(material)}`,
    parts: {
      subtree: signature.map((n) => (n as { id: string }).id),
      params,
      size: input.size,
    },
  }
}

export interface MaterializeInput<S extends ParamSchema> {
  expId: string
  res: VramScope
  device: GPUDevice
  shaders: ShaderRegistry
  frameLayout: GPUBindGroupLayout
  frameBindGroup: GPUBindGroup
  sampler: GPUSampler
  uniformBuffer: GPUBuffer
  resolved: ResolvedGraph
  node: ResolvedNode
  schema: S
  values: ParamValues<S>
  /** Texture for each already-provided dependency, by node id. */
  dependencies: Map<string, GPUTexture>
  attr: VramAttr
  onProgress?: (fraction: number, note?: string) => void
}

export interface MaterializeResult {
  texture: GPUTexture
  key: string
  /** 'baked' = the compute pass ran; 'cache' = the PNG was already there. */
  source: 'baked' | 'cache'
  width: number
  height: number
  ms: number
  module: MaterializeModule
  /** What the cache key was built from — logged so a miss is explainable. */
  keyParts: BakeKey['parts']
}

export async function materializeNode<S extends ParamSchema>(input: MaterializeInput<S>): Promise<MaterializeResult> {
  const { device, res, node, resolved } = input
  const started = performance.now()

  // --- size: declared, else the first texture dependency's -----------------
  let size = node.node.size
  if (!size) {
    for (const [, tex] of input.dependencies) {
      size = [tex.width, tex.height]
      break
    }
  }
  if (!size) {
    throw new Error(
      `material node "${node.node.id}" materializes but has no size: it has no texture dependency to inherit one ` +
        'from, so declare `size: [w, h]` on the node.',
    )
  }
  const [width, height] = size

  const generated = generateMaterializeModule({
    materialId: input.expId,
    schema: input.schema,
    resolved,
    nodeId: node.node.id,
    format: MATERIALIZE_FORMAT,
    shaders: input.shaders,
  })

  const { key, parts } = bakeKeyFor({
    resolved,
    nodeId: node.node.id,
    schema: input.schema,
    values: input.values,
    code: generated.code,
    size,
  })

  const texture = createTexture2D(res, {
    label: `${input.expId}/materialized/${node.node.id}`,
    size: [width, height],
    format: MATERIALIZE_FORMAT,
    mipLevelCount: mipLevelsFor(width, height),
    encoding: node.node.spec.semantics,
    usage: GPUTextureUsage.STORAGE_BINDING,
    attr: input.attr,
  })

  let ran = false
  const png = await bakedArtifact({ expId: input.expId, key, ext: 'png', ...(input.onProgress && { onProgress: input.onProgress }) }, async () => {
    ran = true
    input.onProgress?.(0.3, `materializing ${node.node.id} (${width}x${height})`)
    runMaterializePass(input, generated, texture, width, height)
    const read = await readTexture(device, texture, { exempt: res.exempt.bind(res) })
    const bytes = await encodePng({ width, height, channels: 4, bitDepth: 8, data: read.data })
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
  })

  if (!ran) {
    // Cache (or committed artifact) hit: the PNG IS the node's level 0.
    // rgba8unorm and RGBA8 PNG are the same bytes, so this is exactly what the
    // compute pass would have written.
    const img = await decodePng(new Uint8Array(png))
    if (img.width !== width || img.height !== height || img.channels !== 4 || img.bitDepth !== 8) {
      throw new Error(
        `materialized cache entry ${key}.png is ${img.width}x${img.height}x${img.channels}@${img.bitDepth}, expected ` +
          `${width}x${height}x4@8 — the cache key did not capture something it should have`,
      )
    }
    device.queue.writeTexture(
      { texture },
      img.data as Uint8Array<ArrayBuffer>,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height],
    )
  }

  generateMips(res, input.shaders, texture, { semantics: node.node.spec.semantics })

  return {
    texture,
    key,
    source: ran ? 'baked' : 'cache',
    width,
    height,
    ms: performance.now() - started,
    module: generated,
    keyParts: parts,
  }
}

function runMaterializePass<S extends ParamSchema>(
  input: MaterializeInput<S>,
  generated: MaterializeModule,
  target: GPUTexture,
  width: number,
  height: number,
): void {
  const { device } = input
  const module = input.shaders.module(generated)

  const entries: GPUBindGroupLayoutEntry[] = [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: MATERIALIZE_FORMAT, viewDimension: '2d' },
    },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
  ]
  generated.dependencies.forEach((_, i) => {
    entries.push({ binding: 3 + i, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } })
  })
  const bgl = device.createBindGroupLayout({ label: `${generated.id}/layout`, entries })

  const bindEntries: GPUBindGroupEntry[] = [
    { binding: 0, resource: target.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
    { binding: 1, resource: { buffer: input.uniformBuffer, size: uniformLayout(input.schema).size } },
    { binding: 2, resource: input.sampler },
  ]
  generated.dependencies.forEach((dep, i) => {
    const tex = input.dependencies.get(dep.node.id)
    if (!tex) {
      throw new Error(`materialize "${generated.nodeId}": dependency "${dep.node.id}" has no texture yet`)
    }
    bindEntries.push({ binding: 3 + i, resource: tex.createView() })
  })

  const pipeline = device.createComputePipeline({
    label: generated.id,
    layout: device.createPipelineLayout({
      label: generated.id,
      // @group(0) stays the frozen frame layout — material.wgsl brings
      // frame.wgsl in, and its bindings must land where they always land.
      bindGroupLayouts: [input.frameLayout, bgl],
    }),
    compute: { module, entryPoint: 'cs_materialize' },
  })

  const enc = device.createCommandEncoder({ label: `materialize/${generated.nodeId}` })
  const pass = enc.beginComputePass({ label: `materialize/${generated.nodeId}` })
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, input.frameBindGroup)
  pass.setBindGroup(1, device.createBindGroup({ label: generated.id, layout: bgl, entries: bindEntries }))
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
  pass.end()
  device.queue.submit([enc.finish()])
}
