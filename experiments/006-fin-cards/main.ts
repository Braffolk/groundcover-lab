import cullSrc from './shaders/cull.wgsl'
import shaderSrc from './shaders/fins.wgsl'
import { BAKE_VERSION, bakeFins, packFins, unpackFins, ATLAS_H, ATLAS_W, MIPS, type BakedFins } from './bake.ts'
import {
  bakedArtifact,
  commitBake,
  speciesById,
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Trig-weighted crossed fins. Each species is baked ONCE into an 8-view atlas
 * (6 azimuthal orthographic captures + 2 top-down slab captures, albedo &
 * local-frame normal, full premultiplied mip chains) and cached by the
 * harness bake flow. At runtime every plant is 5 STATIC quads — 3 vertical
 * fins crossed at 60° plus 2 horizontal slab cards — whose per-card content,
 * trigonometric view weights and hashed-alpha dissolve are what make the
 * classic crossing artifacts disappear (see shaders/fins.wgsl).
 *
 * A compute pass (shaders/cull.wgsl) evaluates the shared scatter hash once
 * per candidate slot of a camera-bounded cell region and compacts the
 * surviving plants into an indirect draw, so per-frame cost is bounded by the
 * region and independent of the stand's total plant count.
 */

interface SpeciesGpu {
  albedo: GPUTexture
  normal: GPUTexture
  meta: BakedFins['meta']
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  density: number
  gpu: SpeciesGpu
  cfgBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  /** Rebuilt only when regionRadius changes (capacity scales with the area). */
  plantBuffer: GPUBuffer
  capacity: number
  cullBg: GPUBindGroup
  renderBg: GPUBindGroup
}

const VERTS_PER_INSTANCE = 30 // 5 quads x 6 vertices
const CFG_FLOATS = 20
const PLANT_BYTES = 24 // struct FinPlant: 6 x f32
/**
 * Compacted-instance capacity. Slot existence is an independent hash test per
 * slot, so the survivor count is binomial with sigma ~160 over the region —
 * a 15% margin plus a flat floor is many hundreds of sigma of headroom, and
 * the shader's arrayLength() guard makes an overflow a no-op rather than a
 * corruption.
 */
const CAP_MARGIN = 1.15
const CAP_FLOOR = 8192

const sideFor = (radius: number): number => Math.max(1, Math.ceil((2 * radius) / SCATTER_CELL_SIZE) + 1)
const capacityFor = (side: number, density: number): number =>
  Math.ceil(
    side * side * SCATTER_MAX_PER_CELL * (Math.min(density, SCATTER_MAX_DENSITY) / SCATTER_MAX_DENSITY) * CAP_MARGIN,
  ) + CAP_FLOOR

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

  // --- bake (or load) one atlas per unique species --------------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const load = async (): Promise<SpeciesGpu> => {
      const key = `fins-v${BAKE_VERSION}-${speciesId}`
      const bake = async (): Promise<ArrayBuffer> => {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        console.log(`[${ctx.id}] baking 8-view fin atlas for ${speciesId}...`)
        const buf = packFins(await bakeFins(ctx, mesh))
        commitBake(ctx.id, key, buf).catch(() => undefined) // static builds: ignore
        return buf
      }
      // A cached/committed artifact can be a poisoned blob (a dev server may
      // answer a missing file with index.html) — validate and rebake.
      let baked = unpackFins(await bakedArtifact({ expId: ctx.id, key }, bake))
      if (!baked) {
        baked = unpackFins(await bake())
        if (!baked) throw new Error(`${ctx.id}: bake produced an invalid fin atlas for ${speciesId}`)
      }
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
    }
    const pending = load()
    speciesCache.set(speciesId, pending)
    return pending
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const renderBgl = device.createBindGroupLayout({
    label: `${ctx.id}/render-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  type EntryBase = Omit<EntryGpu, 'capacity' | 'plantBuffer' | 'cullBg' | 'renderBg'>
  type RegionRes = Pick<EntryGpu, 'capacity' | 'plantBuffer' | 'cullBg' | 'renderBg'>

  /** Allocates the compacted-plant buffer + both bind groups for `side`. */
  const makeRegion = (entry: EntryBase, side: number): RegionRes => {
    const capacity = capacityFor(side, entry.density)
    const plantBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/plants-${entry.entryIndex}`, size: capacity * PLANT_BYTES, usage: GPUBufferUsage.STORAGE },
      { species: entry.speciesId, tag: 'plants' },
    )
    return {
      capacity,
      plantBuffer,
      cullBg: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entry.entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: entry.cfgBuffer } },
          { binding: 1, resource: { buffer: plantBuffer } },
          { binding: 2, resource: { buffer: entry.indirectBuffer } },
        ],
      }),
      renderBg: device.createBindGroup({
        label: `${ctx.id}/render-bg-${entry.entryIndex}`,
        layout: renderBgl,
        entries: [
          { binding: 0, resource: { buffer: entry.cfgBuffer } },
          { binding: 1, resource: { buffer: plantBuffer } },
          { binding: 2, resource: entry.gpu.albedo.createView() },
          { binding: 3, resource: entry.gpu.normal.createView() },
          { binding: 4, resource: sampler },
        ],
      }),
    }
  }

  let side = sideFor(ctx.params.regionRadius)

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const base: EntryBase = {
        entryIndex,
        speciesId: e.species,
        density: e.density,
        gpu: await loadSpecies(e.species),
        cfgBuffer: ctx.res.createBuffer(
          {
            label: `${ctx.id}/cfg-${entryIndex}`,
            size: CFG_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'cfg' },
        ),
        indirectBuffer: ctx.res.createBuffer(
          {
            label: `${ctx.id}/indirect-${entryIndex}`,
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'indirect' },
        ),
      }
      return { ...base, ...makeRegion(base, side) }
    }),
  )

  let cullPipeline!: GPUComputePipeline
  let pipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/fins`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, renderBgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cfgData = new Float32Array(CFG_FLOATS)
  const indirectReset = new Uint32Array([VERTS_PER_INSTANCE, 0, 0, 0])

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / SCATTER_CELL_SIZE)
      const originCellZ = Math.floor((cam.z - R) / SCATTER_CELL_SIZE)
      // Cells overlapping the stand's region — plants exist nowhere else.
      const cellMin = Math.floor(-ctx.stand.radius / SCATTER_CELL_SIZE)
      const cellMax = Math.floor(ctx.stand.radius / SCATTER_CELL_SIZE)

      for (const entry of entries) {
        const m = entry.gpu.meta
        cfgData[0] = m.cx; cfgData[1] = m.yc; cfgData[2] = m.cz; cfgData[3] = m.halfW
        cfgData[4] = m.halfH; cfgData[5] = m.yLow; cfgData[6] = m.yHigh; cfgData[7] = m.radius
        cfgData[8] = originCellX; cfgData[9] = originCellZ; cfgData[10] = side; cfgData[11] = ctx.seed
        cfgData[12] = R; cfgData[13] = entry.entryIndex; cfgData[14] = ctx.params.edgeFade; cfgData[15] = ctx.params.topBlend
        cfgData[16] = ctx.params.alphaSharp; cfgData[17] = cellMin; cfgData[18] = cellMax
        cfgData[19] = ctx.params.cardTint ? 1 : 0
        device.queue.writeBuffer(entry.cfgBuffer, 0, cfgData)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // One thread per candidate slot of the region — the placement hash runs
      // once per candidate instead of once per vertex.
      const groups = Math.ceil((side * side * SCATTER_MAX_PER_CELL) / 64)
      const cull = ctx.timing.computePass(enc, 'fin-cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cull.setBindGroup(1, entry.cullBg)
        cull.dispatchWorkgroups(groups)
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'fin-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.renderBg)
        pass.drawIndirect(entry.indirectBuffer, 0)
      }
      pass.end()
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      if (!keys.has('regionRadius')) return
      side = sideFor(ctx.params.regionRadius)
      for (const entry of entries) {
        if (capacityFor(side, entry.density) === entry.capacity) continue
        entry.plantBuffer.destroy()
        Object.assign(entry, makeRegion(entry, side))
      }
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
