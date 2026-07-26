/**
 * The material runtime: turn a `MaterialDef` into an `Experiment`.
 *
 * A graph material's `main.ts` is a declaration plus one call to
 * `createMaterialExperiment` — no pipeline, no bind groups, no image loading,
 * no mip plumbing, and no way to forget the debug views.
 *
 * Three behaviours are worth knowing before changing anything here:
 *
 * 1. THE GRAPH IS A FUNCTION OF THE PARAMS. `structuralParams` names the ones
 *    that change the SHAPE of the program (a filter radius, whether there is a
 *    parallax stage at all); changing one rebuilds the graph, regenerates the
 *    WGSL and rebuilds the pipeline. Every other param is only a uniform, so
 *    dragging a slider costs a `writeBuffer`.
 *
 * 2. A DISCRETE variantBlend BINDS ONE SLOT. Only the active variant is
 *    resident, which is what keeps three 48 MiB Sphagnum habitats costing one
 *    habitat. Switching reloads behind a generation guard and keeps drawing
 *    the old texture until the new one is ready — the same behaviour
 *    001-flat-maps hand-wrote.
 *
 * 3. VALIDATION ERRORS THROW at create(). WebGPU's own validation errors do
 *    not (they arrive as a toast), which is exactly why a graph error must not
 *    be one more thing on the pile.
 */

import type { ParamSchema, ParamValues } from '../harness/params.ts'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '../harness/registry.ts'
import type { MaterialContextExt, PreviewMesh } from '../scene/preview.ts'
import { PREVIEW_VERTEX_LAYOUT } from '../scene/preview.ts'
import {
  MATERIAL_FIRST_TEXTURE_BINDING,
  MATERIAL_SAMPLER_BINDING,
  MATERIAL_UNIFORM_BINDING,
  resolveGraph,
  viewUvDependencies,
  type ResolvedGraph,
  type ResolvedNode,
} from './resolve.ts'
import { generateMainModule, subtreeOf, uniformLayout, writeUniforms, type GeneratedModule } from './codegen.ts'
import { loadNodeImage } from './images.ts'
import { materializeNode, type MaterializeResult } from './materialize.ts'
import { clearMaterialInstance, publishMaterialInstance, type MaterialTextureSlot } from './instance.ts'
import { nodeReportsOf, publishMaterialReport, type MaterialReport, type MaterializationRecord } from './report.ts'
import { formatIssues, type MaterialDef, type MaterialGraph, type MaterialIssue } from './types.ts'
import { validateCompilation, validateGeneratedMain, validateMaterial } from './validate.ts'

/**
 * TILES PER METRE — ONE isotropic number, never a per-axis fit.
 *
 * The preview uv is already in metres of surface, so life size is 1/periodM
 * and the tile is square on every object by construction. The single
 * correction is on an axis that CLOSES on itself: a fractional number of tiles
 * around a sphere or a cylinder butts two different parts of the tile together
 * at the seam, so the period is snapped to the nearest exact divisor — and the
 * snapped value is applied to BOTH axes, because scaling one axis alone is
 * exactly the "stretched to fit the polygons" failure the metre convention
 * exists to prevent.
 */
/**
 * THE sampler every graph material binds at @group(1) @binding(1). One
 * declaration, because the portable export has to state it in its manifest and
 * a consumer that creates a different one gets a different picture — `repeat`
 * in particular is not optional for a tiling material.
 */
export const MATERIAL_SAMPLER_DESC = {
  addressModeU: 'repeat',
  addressModeV: 'repeat',
  magFilter: 'linear',
  minFilter: 'linear',
  mipmapFilter: 'linear',
  maxAnisotropy: 16,
} as const satisfies GPUSamplerDescriptor

export function tilesPerMetre(mesh: PreviewMesh, periodM: number, scale = 1): number {
  const wanted = 1 / Math.max(periodM * scale, 1e-4)
  const period = mesh.uvPeriod[0] > 0 ? mesh.uvPeriod[0] : mesh.uvPeriod[1]
  if (period <= 0) return wanted
  return Math.max(1, Math.round(period * wanted)) / period
}

interface TextureSlot {
  node: ResolvedNode
  texture: GPUTexture
  /** For a discrete variantBlend: which option is currently resident. */
  variant?: string
  materialized?: MaterializeResult
}

interface Built {
  graph: MaterialGraph
  resolved: ResolvedGraph
  module: GeneratedModule
  pipeline: GPURenderPipeline
  bgl: GPUBindGroupLayout
  bindGroup: GPUBindGroup
  slots: TextureSlot[]
  issues: MaterialIssue[]
  materializations: MaterializationRecord[]
}

export async function createMaterialExperiment<S extends ParamSchema>(
  ctx: ExperimentContext<S, MaterialContextExt>,
  def: MaterialDef<S>,
): Promise<Experiment> {
  const { device } = ctx
  const species = def.species ?? ctx.id
  const layout = uniformLayout(def.params)
  const uniforms = new Float32Array(layout.size / 4)

  const sampler = device.createSampler({ label: `${ctx.id}/material`, ...MATERIAL_SAMPLER_DESC })

  const uniformBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/material/uniforms`, size: layout.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { species, tag: 'uniforms' },
  )

  let built: Built | null = null
  let generation = 0
  let instanceGeneration = 0
  let disposed = false

  /**
   * Push the current params into the uniform buffer.
   *
   * This MUST run before any materialize pass, not just per frame: a
   * materialized node reads `u.p.<param>` in a compute pass encoded during
   * create(), which is before the harness has ever called `update()`. Without
   * this the first bake sees a ZERO-FILLED uniform — and it is not a crash, it
   * is a plausible-looking texture baked from the wrong numbers, which then
   * sits in the PNG cache under a key that says it is right. (Measured: the
   * occlusion node baked with ao_strength = 0 instead of 1, i.e. no occlusion
   * at all, 99.93% of pixels differing from the live path at `macro`.)
   */
  const pushUniforms = (): void => {
    const mesh = ctx.preview.mesh
    const t = def.uvTransform?.(mesh, ctx.params) ?? { scale: 1 }
    writeUniforms(
      def.params,
      ctx.params,
      { scale: [t.scale, t.scale], rotation: t.rotation ?? 0, min: mesh.uvBounds.min, max: mesh.uvBounds.max },
      uniforms,
    )
    device.queue.writeBuffer(uniformBuffer, 0, uniforms)
  }

  // -------------------------------------------------------------------------
  // Texture provisioning
  // -------------------------------------------------------------------------

  const activeVariantOf = (node: ResolvedNode): string => {
    if (node.node.kind !== 'variantBlend') throw new Error('activeVariantOf: not a variantBlend')
    const value = String((ctx.params as Record<string, unknown>)[node.node.selector])
    return node.node.variants[value] ? value : Object.keys(node.node.variants)[0]!
  }

  /** The node whose texture a slot must hold right now. */
  const sourceNodeOf = (resolved: ResolvedGraph, node: ResolvedNode): ResolvedNode => {
    if (node.source !== 'variant') return node
    const variantId = node.node.kind === 'variantBlend' ? node.node.variants[activeVariantOf(node)]! : node.node.id
    return resolved.byId.get(variantId)!
  }

  const provide = async (
    resolved: ResolvedGraph,
    slotNode: ResolvedNode,
    provided: Map<string, GPUTexture>,
    records: MaterializationRecord[],
  ): Promise<TextureSlot> => {
    const src = sourceNodeOf(resolved, slotNode)
    // The blend's decoder is generated from the blend's spec, and the
    // validator has already proved every variant shares it.
    if (src.node.kind === 'image') {
      const loaded = await loadNodeImage({
        res: ctx.res,
        device,
        shaders: ctx.shaders,
        url: src.node.url,
        label: `${ctx.id}/${src.node.id}`,
        spec: src.node.spec,
        attr: { species, tag: src.node.id },
      })
      provided.set(slotNode.node.id, loaded.texture)
      return {
        node: slotNode,
        texture: loaded.texture,
        ...(slotNode.source === 'variant' && { variant: activeVariantOf(slotNode) }),
      }
    }
    const result = await materializeNode({
      expId: ctx.id,
      res: ctx.res,
      device,
      shaders: ctx.shaders,
      frameLayout: ctx.frame.layout,
      frameBindGroup: ctx.frame.bindGroup,
      sampler,
      uniformBuffer,
      resolved,
      node: src,
      schema: def.params,
      values: ctx.params,
      dependencies: provided,
      attr: { species, tag: src.node.id },
      onProgress: ctx.progress.bind(ctx),
    })
    provided.set(slotNode.node.id, result.texture)
    records.push({
      nodeId: src.node.id,
      key: result.key,
      source: result.source,
      ms: Math.round(result.ms),
      size: [result.width, result.height],
      subtree: result.keyParts.subtree,
      params: result.keyParts.params,
    })
    return {
      node: slotNode,
      texture: result.texture,
      materialized: result,
      ...(slotNode.source === 'variant' && { variant: activeVariantOf(slotNode) }),
    }
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  const makeBindGroup = (bgl: GPUBindGroupLayout, slots: readonly TextureSlot[]): GPUBindGroup =>
    device.createBindGroup({
      label: `${ctx.id}/material`,
      layout: bgl,
      entries: [
        { binding: MATERIAL_UNIFORM_BINDING, resource: { buffer: uniformBuffer } },
        { binding: MATERIAL_SAMPLER_BINDING, resource: sampler },
        ...slots.map((s) => ({ binding: s.node.binding!, resource: s.texture.createView() })),
      ],
    })

  const compile = (resolved: ResolvedGraph): { module: GeneratedModule; gpu: GPUShaderModule } => {
    const module = generateMainModule({
      materialId: ctx.id,
      schema: def.params,
      resolved,
      surface: def.surface,
      shade: def.shade,
      fog: def.fog ?? false,
      shaders: ctx.shaders,
    })
    return { module, gpu: ctx.shaders.module(module) }
  }

  const makePipeline = (gpu: GPUShaderModule, bgl: GPUBindGroupLayout, label: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label,
      layout: device.createPipelineLayout({ label, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module: gpu, entryPoint: 'vs_main', buffers: [PREVIEW_VERTEX_LAYOUT] },
      fragment: { module: gpu, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })

  const build = async (): Promise<Built> => {
    // Before ANY materialize pass — see pushUniforms.
    pushUniforms()
    const graph = def.graph(ctx.params)
    const resolved = resolveGraph(graph)

    const surfaceCode = ctx.shaders.latestCode(def.surface)
    const shadeCode = ctx.shaders.latestCode(def.shade)
    const report = validateMaterial({
      schema: def.params,
      graph,
      resolved,
      surface: def.surface,
      shade: def.shade,
      surfaceCode,
      shadeCode,
    })
    const issues = [...report.issues]
    if (!report.ok) {
      // Publish before throwing. `create()` failing is exactly when someone
      // needs the issue list, and a stack trace in a fatal overlay is a worse
      // place to read it than the material page's report panel.
      publishMaterialReport({
        id: ctx.id,
        ok: false,
        moduleId: '',
        code: '',
        channels: Object.entries(graph.channels).map(([channel, nodeId]) => ({
          channel,
          nodeId,
          mode: resolved.byId.get(nodeId)?.mode ?? 'live',
          type: resolved.byId.get(nodeId)?.node.spec.type ?? 'f32',
        })),
        nodes: nodeReportsOf(resolved),
        viewUv: graph.viewUv.kind,
        viewUvNodes: viewUvDependencies(graph),
        issues,
        materializations: [],
        textureBytes: 0,
      })
      throw new Error(`material "${ctx.id}" failed validation:\n${formatIssues(issues.filter((i) => i.level === 'error'))}`)
    }

    // --- textures, walked in DEPENDENCY order (resolved.order), not binding
    //     order, so a materialized node's inputs already have textures.
    const provided = new Map<string, GPUTexture>()
    const records: MaterializationRecord[] = []
    const slots: TextureSlot[] = []
    const wanted = resolved.order
      .map((id) => resolved.byId.get(id)!)
      .filter((r) => r.mode === 'texture' && r.ownedBy === undefined)
    for (let i = 0; i < wanted.length; i++) {
      const slotNode = wanted[i]!
      ctx.progress(i / Math.max(wanted.length, 1), `${ctx.id}: ${slotNode.node.id}`)
      slots.push(await provide(resolved, slotNode, provided, records))
    }
    // Bindings are assigned sorted by node id (deterministic layout), which is
    // not dependency order — restore it for the bind group.
    slots.sort((a, b) => a.node.binding! - b.node.binding!)
    ctx.progress(1, `${ctx.id}: ready`)

    const bgl = device.createBindGroupLayout({
      label: `${ctx.id}/material`,
      entries: [
        { binding: MATERIAL_UNIFORM_BINDING, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: MATERIAL_SAMPLER_BINDING, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ...slots.map((s) => ({
          binding: s.node.binding!,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' as GPUTextureSampleType },
        })),
      ],
    })

    const { module, gpu } = compile(resolved)
    issues.push(...validateGeneratedMain(module.code))
    void validateCompilation(gpu, module.id).then((compileIssues) => {
      if (compileIssues.length === 0) return
      issues.push(...compileIssues)
      console.error(`[material ${ctx.id}] generated WGSL did not compile cleanly:\n${formatIssues(compileIssues)}`)
    })

    return {
      graph,
      resolved,
      module,
      bgl,
      pipeline: makePipeline(gpu, bgl, `${ctx.id}/material`),
      bindGroup: makeBindGroup(bgl, slots),
      slots,
      issues,
      materializations: records,
    }
  }

  const publish = (b: Built): void => {
    const vram = b.slots.reduce((sum, s) => sum + textureBytesOf(s.texture), 0)
    const report: MaterialReport = {
      id: ctx.id,
      ok: !b.issues.some((i) => i.level === 'error'),
      moduleId: b.module.id,
      code: b.module.code,
      channels: Object.entries(b.graph.channels).map(([channel, nodeId]) => ({
        channel,
        nodeId,
        mode: b.resolved.byId.get(nodeId)?.mode ?? 'live',
        type: b.resolved.byId.get(nodeId)?.node.spec.type ?? 'f32',
      })),
      nodes: nodeReportsOf(b.resolved),
      viewUv: b.graph.viewUv.kind,
      viewUvNodes: viewUvDependencies(b.graph),
      issues: b.issues,
      materializations: b.materializations,
      textureBytes: vram,
    }
    // The GPU handles go out beside the data, never inside it — see
    // instance.ts. An inspector needs both; a headless script needs only the
    // first, and must stay able to JSON it.
    publishMaterialInstance({
      id: ctx.id,
      generation: ++instanceGeneration,
      device,
      shaders: ctx.shaders,
      res: ctx.res,
      frame: ctx.frame,
      sampler,
      uniformBuffer,
      uniformFloats: () => uniforms.slice(),
      schema: def.params,
      values: ctx.params,
      resolved: b.resolved,
      slots: b.slots.map(
        (s): MaterialTextureSlot => ({
          nodeId: s.node.node.id,
          sourceNodeId: sourceNodeOf(b.resolved, s.node).node.id,
          binding: s.node.binding!,
          texture: s.texture,
          ...(s.variant !== undefined && { variant: s.variant }),
        }),
      ),
    })
    publishMaterialReport(report)
    if (import.meta.env.DEV) {
      const warnings = b.issues.filter((i) => i.level === 'warning')
      if (warnings.length > 0) console.warn(`[material ${ctx.id}] validation warnings:\n${formatIssues(warnings)}`)
      for (const m of b.materializations) {
        console.info(
          `[material ${ctx.id}] materialized "${m.nodeId}" ${m.size[0]}x${m.size[1]} from ${m.source} in ${m.ms}ms ` +
            `key=${m.key} subtree=[${m.subtree.join(',')}] params=${JSON.stringify(m.params)}`,
        )
      }
    }
  }

  built = await build()
  publish(built)

  // -------------------------------------------------------------------------
  // Rebuilds
  // -------------------------------------------------------------------------

  const rebuildAll = (): void => {
    const mine = ++generation
    const previous = built
    void build()
      .then((next) => {
        if (disposed || mine !== generation) {
          for (const s of next.slots) s.texture.destroy()
          return
        }
        built = next
        publish(next)
        if (previous) destroySlots(previous.slots, next.slots)
      })
      .catch((err: unknown) => {
        console.error(`[material ${ctx.id}] rebuild failed:`, err)
      })
  }

  /** Only the pipeline: a hot WGSL edit changes the code, not the textures. */
  const rebuildPipeline = (): void => {
    if (!built) return
    try {
      const { module, gpu } = compile(built.resolved)
      built.module = module
      built.pipeline = makePipeline(gpu, built.bgl, `${ctx.id}/material`)
      publish(built)
    } catch (err) {
      console.error(`[material ${ctx.id}] shader rebuild failed:`, err)
    }
  }

  /** A discrete blend's selector moved: reload only that slot. */
  const swapVariants = (): void => {
    const b = built
    if (!b) return
    const mine = ++generation
    const stale = b.slots.filter((s) => s.node.source === 'variant' && s.variant !== activeVariantOf(s.node))
    if (stale.length === 0) return
    const provided = new Map<string, GPUTexture>(b.slots.map((s) => [s.node.node.id, s.texture] as const))
    const records: MaterializationRecord[] = []
    void Promise.all(stale.map((s) => provide(b.resolved, s.node, provided, records)))
      .then((next) => {
        if (disposed || mine !== generation || built !== b) {
          for (const s of next) s.texture.destroy()
          return
        }
        const replaced: GPUTexture[] = []
        for (const slot of next) {
          const index = b.slots.findIndex((s) => s.node.node.id === slot.node.node.id)
          if (index < 0) continue
          replaced.push(b.slots[index]!.texture)
          b.slots[index] = slot
        }
        b.bindGroup = makeBindGroup(b.bgl, b.slots)
        b.materializations = [...b.materializations.filter((m) => !records.some((r) => r.nodeId === m.nodeId)), ...records]
        publish(b)
        for (const t of replaced) t.destroy()
      })
      .catch((err: unknown) => {
        console.error(`[material ${ctx.id}] variant swap failed:`, err)
      })
  }

  const unsubscribe = ctx.shaders.onReload(rebuildPipeline)

  // -------------------------------------------------------------------------
  // The Experiment
  // -------------------------------------------------------------------------

  const structural = new Set<string>(def.structuralParams ?? [])
  const selectors = (b: Built): Set<string> => {
    const out = new Set<string>()
    for (const r of b.resolved.byId.values()) {
      if (r.node.kind === 'variantBlend' && r.source === 'variant') out.add(r.node.selector)
    }
    return out
  }

  return {
    update(_frame: FrameInfo): void {
      pushUniforms()
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const b = built
      if (!b) return
      // The harness base pass cleared colour to the studio backdrop and cleared
      // depth; the preview object is ours to draw, silhouette and all.
      const mesh = ctx.preview.mesh
      const pass = ctx.timing.renderPass(enc, 'material', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(b.pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, b.bindGroup)
      pass.setVertexBuffer(0, mesh.vertices)
      pass.setIndexBuffer(mesh.indices, 'uint32')
      pass.drawIndexed(mesh.indexCount)
      pass.end()
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      const b = built
      if (!b) return
      for (const key of keys) {
        if (structural.has(key)) {
          rebuildAll()
          return
        }
      }
      for (const key of keys) {
        if (!selectors(b).has(key)) continue
        // A materialized node whose subtree contains a variantBlend was baked
        // AGAINST one variant, so a selector change invalidates it — rebind is
        // not enough and would silently show the previous habitat's bake.
        if (materializedDependsOnVariant(b.resolved)) rebuildAll()
        else swapVariants()
        return
      }
    },

    dispose(): void {
      disposed = true
      unsubscribe()
      // The report is data and survives (a headless script may still want it);
      // the GPU handles do not — ctx.res is about to destroy every texture.
      clearMaterialInstance(ctx.id)
      // Textures and buffers are destroyed by the harness via ctx.res.
    },
  }
}

/** True when any materialized node's subtree reaches a discrete variantBlend. */
function materializedDependsOnVariant(resolved: ResolvedGraph): boolean {
  for (const [id, r] of resolved.byId) {
    if (r.source !== 'materialize') continue
    for (const sub of subtreeOf(resolved, id)) {
      const dep = resolved.byId.get(sub)
      if (dep && dep.node.kind === 'variantBlend' && dep.source === 'variant') return true
    }
  }
  return false
}

function destroySlots(previous: readonly TextureSlot[], next: readonly TextureSlot[]): void {
  const keep = new Set(next.map((s) => s.texture))
  for (const s of previous) if (!keep.has(s.texture)) s.texture.destroy()
}

function textureBytesOf(tex: GPUTexture): number {
  let bytes = 0
  let w = tex.width
  let h = tex.height
  const perTexel = tex.format === 'r8unorm' ? 1 : tex.format.includes('32float') ? 16 : 4
  for (let level = 0; level < tex.mipLevelCount; level++) {
    bytes += w * h * perTexel
    w = Math.max(1, w >> 1)
    h = Math.max(1, h >> 1)
  }
  return bytes
}
