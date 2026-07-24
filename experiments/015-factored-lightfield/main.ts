import shaderSrc from './shaders/qlf.wgsl'
import { ATLAS, bakeSpecies, GRID, TILE, type QlfBaked } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Factored quantized light field renderer. Per species the 4D light field is
 * stored factored: a 576-view 8-bit ortho DEPTH atlas (geometry, all the
 * view-dependence) plus view-independent 3D albedo/normal volumes (radiance).
 * Every plant is a single camera-facing card whose fragments reconstruct the
 * eye ray's hit point from the 4 nearest depth views and shade it from the
 * volumes — one appearance fetch, so colours cannot ghost between views.
 * Placement is the scatter WGSL twin over a bounded camera-centred region:
 * per-frame cost never depends on the stand's total plant count.
 */

interface EntryGpu {
  entryIndex: number
  speciesId: string
  baked: QlfBaked
  infoBuffer: GPUBuffer
  bindGroup: GPUBindGroup
}

const CELL = 4 // must equal SCATTER_CELL_SIZE

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  })

  // Bake each unique species once per session (a few seconds; the dev server's
  // SPA fallback poisons committed-bake fetches, as 005 found, so no cache).
  const speciesCache = new Map<string, Promise<QlfBaked>>()
  const loadSpecies = (speciesId: string): Promise<QlfBaked> => {
    let p = speciesCache.get(speciesId)
    if (!p) {
      p = ctx.meshes.load(speciesById(speciesId).meshId).then((mesh) => bakeSpecies(ctx, speciesId, mesh))
      speciesCache.set(speciesId, p)
    }
    return p
  }

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // Bake sequentially — the three species share the GPU during startup and
  // parallel bakes would just interleave submits.
  const entries: EntryGpu[] = []
  for (const [entryIndex, e] of ctx.stand.species.entries()) {
    const baked = await loadSpecies(e.species)
    const infoBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/info-${entryIndex}`, size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: e.species, tag: 'qlf-info' },
    )
    const bindGroup = device.createBindGroup({
      label: `${ctx.id}/bg-${entryIndex}`,
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: baked.depthAtlas.createView() },
        { binding: 2, resource: baked.albedoVol.createView() },
        { binding: 3, resource: baked.normalVol.createView() },
        { binding: 4, resource: sampler },
      ],
    })
    entries.push({ entryIndex, speciesId: e.species, baked, infoBuffer, bindGroup })
  }

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/qlf-cards`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const info = new Float32Array(24)
  let side = 1

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / CELL)
      const originCellZ = Math.floor((cam.z - R) / CELL)
      side = Math.max(1, Math.ceil((2 * R) / CELL) + 1)

      for (const entry of entries) {
        const b = entry.baked
        info[0] = b.center[0]; info[1] = b.center[1]; info[2] = b.center[2]; info[3] = b.radius
        info[4] = b.bmin[0]; info[5] = b.bmin[1]; info[6] = b.bmin[2]; info[7] = GRID
        info[8] = b.bmax[0]; info[9] = b.bmax[1]; info[10] = b.bmax[2]; info[11] = TILE
        info[12] = ATLAS; info[13] = entry.entryIndex; info[14] = originCellX; info[15] = originCellZ
        info[16] = side; info[17] = ctx.seed; info[18] = R; info[19] = ctx.params.coverage
        info[20] = ctx.params.refine ? 1 : 0; info[21] = b.heightM
        info[22] = ctx.params.showViewGrid ? 1 : 0; info[23] = 0
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'qlf-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      const instances = side * side * 128
      for (const entry of entries) {
        pass.setBindGroup(1, entry.bindGroup)
        pass.draw(6, instances)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are destroyed by the harness via ctx.res.
    },
  }
}
