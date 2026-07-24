import shaderSrc from './shaders/impostor.wgsl'
import { bakeAtlas, unpackAtlas, type BakedAtlas } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Octahedral impostors. Each species is baked once into a GRID x GRID
 * hemi-octahedral atlas of orthographic captures (albedo+coverage & a
 * local-frame normal). At runtime every plant is a single camera-facing card;
 * the fragment shader projects the fragment into the 4 atlas views nearest the
 * current viewing direction and blends them, so the plant reads correctly from
 * grazing angles all the way to straight overhead. Placement is evaluated in
 * the vertex shader over a bounded region around the camera (scatter WGSL
 * twin), so per-frame cost is independent of the stand's total plant count.
 */

interface SpeciesGpu {
  albedo: GPUTexture
  normal: GPUTexture
  meta: BakedAtlas['meta']
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  gpu: SpeciesGpu
  metaBuffer: GPUBuffer
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
  })

  // --- bake / load one atlas per unique species -------------------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const p = (async (): Promise<SpeciesGpu> => {
      const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
      // Baked fresh in-browser each session. The harness bakedArtifact() cache
      // is intentionally NOT used: the dev server answers missing /mesh/baked
      // requests with a 200 index.html, which poisons both the OPFS cache and
      // any committed file. A GRID^2 atlas bakes in a couple of seconds.
      const buf = await bakeAtlas(ctx, mesh)
      const atlas = unpackAtlas(buf)
      const px = atlas.meta.atlasPx
      const albedo = ctx.res.createTexture(
        { label: `${ctx.id}/${speciesId}/albedo`, size: [px, px], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
        { species: speciesId, tag: 'impostor-albedo' },
      )
      const normal = ctx.res.createTexture(
        { label: `${ctx.id}/${speciesId}/normal`, size: [px, px], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
        { species: speciesId, tag: 'impostor-normal' },
      )
      const albBytes = new Uint8Array(atlas.albedo)
      const nrmBytes = new Uint8Array(atlas.normal)
      device.queue.writeTexture({ texture: albedo }, albBytes, { bytesPerRow: px * 4, rowsPerImage: px }, [px, px])
      device.queue.writeTexture({ texture: normal }, nrmBytes, { bytesPerRow: px * 4, rowsPerImage: px }, [px, px])
      return { albedo, normal, meta: atlas.meta }
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
        { label: `${ctx.id}/meta-${entryIndex}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
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
      label: `${ctx.id}/impostors`,
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
        const m = entry.gpu.meta
        meta[0] = m.center[0]; meta[1] = m.center[1]; meta[2] = m.center[2]; meta[3] = m.radius
        meta[4] = m.grid; meta[5] = m.tile; meta[6] = m.atlasPx; meta[7] = entry.entryIndex
        meta[8] = originCellX; meta[9] = originCellZ; meta[10] = side; meta[11] = ctx.seed
        meta[12] = R; meta[13] = ctx.params.coverage; meta[14] = 0; meta[15] = 0
        device.queue.writeBuffer(entry.metaBuffer, 0, meta)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'octa-impostors', {
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
    },
  }
}
