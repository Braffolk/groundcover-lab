import cardShaderSrc from './shaders/card.wgsl'
import { bakeSpecies, type BakedSpecies } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Fourier appearance cards. Each species' view-dependent appearance (opacity,
 * color, depth, normal) is baked once into per-texel truncated Fourier
 * coefficients over view azimuth x E elevation rings — four small rgba8 array
 * textures (~4.2 MB/species), no view atlas. At runtime every plant is a
 * single camera-facing card whose fragments evaluate the basis in closed form
 * (double/triple-angle recurrences); view interpolation is inherently smooth,
 * so there is no 4-view blending and no popping. Placement is the shared
 * scatter WGSL twin over a bounded region around the camera, so per-frame
 * cost is independent of the stand's total plant count.
 */

const CELL = 4 // must equal SCATTER_CELL_SIZE

interface EntryGpu {
  entryIndex: number
  speciesId: string
  baked: BakedSpecies
  metaBuffer: GPUBuffer
  bindGroup: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // Bake each unique species once per session, fully on the GPU (~100 mesh
  // draws + 4 fit dispatches per species, single submit, no readback). The
  // harness bakedArtifact() cache is intentionally not used: the dev server
  // answers missing /mesh/baked files with 200 index.html (see NOTES.md), and
  // the bake is fast enough to not need it.
  const speciesCache = new Map<string, Promise<BakedSpecies>>()
  const loadSpecies = (speciesId: string): Promise<BakedSpecies> => {
    let p = speciesCache.get(speciesId)
    if (!p) {
      p = ctx.meshes.load(speciesById(speciesId).meshId).then((mesh) => bakeSpecies(ctx, mesh, speciesId))
      speciesCache.set(speciesId, p)
    }
    return p
  }

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const baked = await loadSpecies(e.species)
      const metaBuffer = ctx.res.createBuffer(
        { label: `${ctx.id}/meta-${entryIndex}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'meta' },
      )
      const view = (t: GPUTexture): GPUTextureView => t.createView({ dimension: '2d-array' })
      const bindGroup = device.createBindGroup({
        label: `${ctx.id}/bg-${entryIndex}`,
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: metaBuffer } },
          { binding: 1, resource: view(baked.coeffA) },
          { binding: 2, resource: view(baked.coeffB) },
          { binding: 3, resource: view(baked.coeffC) },
          { binding: 4, resource: view(baked.coeffD) },
          { binding: 5, resource: sampler },
        ],
      })
      return { entryIndex, speciesId: e.species, baked, metaBuffer, bindGroup }
    }),
  )

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(cardShaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/cards`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const meta = new Float32Array(16)
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
        meta[0] = b.center[0]; meta[1] = b.center[1]; meta[2] = b.center[2]; meta[3] = b.radius
        meta[4] = originCellX; meta[5] = originCellZ; meta[6] = side; meta[7] = ctx.seed
        meta[8] = R; meta[9] = ctx.params.alphaGain; meta[10] = ctx.params.dither; meta[11] = entry.entryIndex
        meta[12] = ctx.params.colorView; meta[13] = 0; meta[14] = 0; meta[15] = 0
        device.queue.writeBuffer(entry.metaBuffer, 0, meta)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'fourier-cards', {
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
