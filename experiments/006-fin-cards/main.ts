import shaderSrc from './shaders/fins.wgsl'
import { bakeFins, ATLAS_H, ATLAS_W, MIPS, type BakedFins } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Trig-weighted crossed fins. Each species is baked once into an 8-view atlas
 * (6 azimuthal orthographic captures + 2 top-down slab captures, albedo &
 * local-frame normal, full premultiplied mip chains). At runtime every plant
 * is 5 STATIC quads — 3 vertical fins crossed at 60° plus 2 horizontal slab
 * cards — whose per-card content, trigonometric view weights and hashed-alpha
 * dissolve are what make the classic crossing artifacts disappear (see
 * shaders/fins.wgsl). Placement is evaluated procedurally in the vertex
 * shader over a bounded cell region around the camera (the scatter WGSL
 * twin), so per-frame cost is independent of the stand's total plant count.
 */

interface SpeciesGpu {
  albedo: GPUTexture
  normal: GPUTexture
  meta: BakedFins['meta']
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  gpu: SpeciesGpu
  metaBuffer: GPUBuffer
  bindGroup: GPUBindGroup
}

const CELL = 4 // must equal SCATTER_CELL_SIZE
const VERTS_PER_INSTANCE = 30 // 5 quads x 6 vertices
const META_FLOATS = 20

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    lodMaxClamp: MIPS - 1,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- bake one atlas per unique species (fresh per session; see NOTES) -----
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const p = (async (): Promise<SpeciesGpu> => {
      const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
      const baked = await bakeFins(ctx, mesh)
      const mkAtlas = (tag: string): GPUTexture =>
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${speciesId}/${tag}`,
            size: [ATLAS_W, ATLAS_H],
            format: 'rgba8unorm',
            mipLevelCount: MIPS,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          },
          { species: speciesId, tag: `fin-${tag}` },
        )
      const albedo = mkAtlas('albedo')
      const normal = mkAtlas('normal')
      for (let level = 0; level < MIPS; level++) {
        const w = Math.max(1, ATLAS_W >> level)
        const h = Math.max(1, ATLAS_H >> level)
        device.queue.writeTexture(
          { texture: albedo, mipLevel: level },
          baked.albedoMips[level]!,
          { bytesPerRow: w * 4, rowsPerImage: h },
          [w, h],
        )
        device.queue.writeTexture(
          { texture: normal, mipLevel: level },
          baked.normalMips[level]!,
          { bytesPerRow: w * 4, rowsPerImage: h },
          [w, h],
        )
      }
      return { albedo, normal, meta: baked.meta }
    })()
    speciesCache.set(speciesId, p)
    return p
  }

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const gpu = await loadSpecies(e.species)
      const metaBuffer = ctx.res.createBuffer(
        { label: `${ctx.id}/meta-${entryIndex}`, size: META_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'meta' },
      )
      const bindGroup = device.createBindGroup({
        label: `${ctx.id}/bg-${entryIndex}`,
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: metaBuffer } },
          { binding: 1, resource: gpu.albedo.createView() },
          { binding: 2, resource: gpu.normal.createView() },
          { binding: 3, resource: sampler },
        ],
      })
      return { entryIndex, speciesId: e.species, gpu, metaBuffer, bindGroup }
    }),
  )

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/fins`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const metaData = new Float32Array(META_FLOATS)
  let side = 1

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / CELL)
      const originCellZ = Math.floor((cam.z - R) / CELL)
      side = Math.max(1, Math.ceil((2 * R) / CELL) + 1)

      for (const entry of entries) {
        const m = entry.gpu.meta
        metaData[0] = m.cx; metaData[1] = m.yc; metaData[2] = m.cz; metaData[3] = m.halfW
        metaData[4] = m.halfH; metaData[5] = m.yLow; metaData[6] = m.yHigh; metaData[7] = m.radius
        metaData[8] = originCellX; metaData[9] = originCellZ; metaData[10] = side; metaData[11] = ctx.seed
        metaData[12] = R; metaData[13] = entry.entryIndex; metaData[14] = ctx.params.edgeFade; metaData[15] = ctx.params.topBlend
        metaData[16] = ctx.params.alphaSharp; metaData[17] = 0; metaData[18] = 0; metaData[19] = 0
        device.queue.writeBuffer(entry.metaBuffer, 0, metaData)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'fin-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      const instances = side * side * 128
      for (const entry of entries) {
        pass.setBindGroup(1, entry.bindGroup)
        pass.draw(VERTS_PER_INSTANCE, instances)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
