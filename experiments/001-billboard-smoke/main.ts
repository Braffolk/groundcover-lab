import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, loadSpeciesCards, type CardAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Billboard baseline over REAL baked imagery. Startup: each species' card
 * atlas (8 side views + 1 top view; albedo+coverage rgba8, oct normals rg8)
 * is loaded from mesh/baked / OPFS or baked in-browser from the raw mesh,
 * uploaded, and mipped. Per frame: a compute pass evaluates the shared
 * scatter over a camera-centered cell region, frustum-culls, and compacts
 * survivors into an indirect draw. An upright plant is one camera-facing side
 * card + one horizontal top card; a CARPET species (stand carpetDiv > 0, e.g.
 * Sphagnum) is instead a single ground-parallel quad the size of its periodic
 * tile, conformed to the terrain per vertex. Alpha-tested with depth write.
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const MIP_LEVELS = Math.floor(Math.log2(ATLAS)) + 1
const TOP_FRAC = 0.52
const INFO_FLOATS = 44

interface SpeciesGpu {
  atlas: CardAtlas
  albedoTex: GPUTexture
  normalTex: GPUTexture
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  capacity: number
  /** Mat species (stand carpetDiv > 0): one ground-parallel quad, no side card. */
  isCarpet: boolean
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroup: GPUBindGroup
  enumSlots: number
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/atlas-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) --
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesCards(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- bind group layouts / pipelines ---------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // Worst-case region extent in cells, but never more cells than the stand
  // itself owns: outside its radius the scatter is empty, so instance capacity
  // sized for a 128m region on a +/-96m stand is pure dead VRAM. On the
  // +/-128m stands (default, close-quality via its own radius) this is exactly
  // the old number — the clamp only ever removes cells that hold nothing.
  const standSide = Math.floor(ctx.stand.radius / CELL) * 2 + 1
  const sideMax = Math.min(Math.ceil((2 * REGION_MAX) / CELL) + 1, standSide)

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const carpetDiv = standEntry.carpetDiv ?? 0
    const isCarpet = carpetDiv > 0
    // A carpet entry fills carpetDiv^2 of the cell's slots and ignores
    // `density`; a scattered entry uses density/8 of all SCATTER_MAX_PER_CELL.
    // Carpet slots are NOT bounded by the scatter budget: life size needs
    // div 22 = 484 slots, and clamping to 128 renders only the first ~6 rows of
    // every cell, i.e. bands of moss with bare gaps between them.
    // TWO DIFFERENT NUMBERS — conflating them drops plants:
    //  * enumSlots is how many candidate slots the cull must EVALUATE per cell.
    //    Every slot must be visited or plants silently disappear.
    //  * expectedPerCell is how many are expected to SURVIVE, which is all the
    //    instance buffer needs to hold.
    const enumSlots = isCarpet ? carpetDiv * carpetDiv : SCATTER_MAX_PER_CELL
    const bandFraction = Math.min(1, standEntry.wetWidth ?? 1)
    const expectedPerCell = isCarpet
      ? carpetDiv * carpetDiv * Math.min(1, bandFraction * 1.25)
      : SCATTER_MAX_PER_CELL * (Math.min(standEntry.density, 8) / 8)
    const capacity = Math.ceil(sideMax * sideMax * expectedPerCell * 1.06) + 1024
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: capacity * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = device.createBindGroup({
      label: `${ctx.id}/draw-bg-${entryIndex}`,
      layout: drawBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: gpu.albedoTex.createView() },
        { binding: 3, resource: gpu.normalTex.createView() },
        { binding: 4, resource: sampler },
      ],
    })
    return {
      speciesId: standEntry.species,
      gpu,
      capacity,
      isCarpet,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroup,
      enumSlots,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let cardsPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    cardsPipeline = device.createRenderPipeline({
      label: `${ctx.id}/cards`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cards-pl`, bindGroupLayouts: [ctx.frame.layout, drawBgl] }),
      vertex: { module: ctx.shaders.module(cardsSrc), entryPoint: 'vs_main' },
      fragment: { module: ctx.shaders.module(cardsSrc), entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(4)

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      // Region cell rect, clamped to the stand's cell range on the CPU: cells
      // outside the stand hold nothing, so they must not be dispatched (on a
      // small stand — close-quality is ±24m — that is most of the region).
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        // A carpet tile is ONE ground-parallel quad — never a side card, so it
        // never needs the second 6 vertices (and `topCard` does not apply).
        indirectReset[0] = entry.isCarpet ? 6 : ctx.params.topCard ? 12 : 6
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.capacity
        info[32] = a.y0
        info[33] = a.y1
        info[34] = a.rXZ
        info[35] = a.cx
        info[36] = a.cz
        info[37] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info[38] = ctx.params.alphaRef
        info[39] = TOP_FRAC
        info[40] = ctx.params.bottomShade
        info[41] = ctx.params.carpetOverscale
        info[42] = entry.enumSlots
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * entry.enumSlots
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        if (entry.slotsPerFrame === 0) continue // camera outside the stand
        cull.setBindGroup(1, entry.cullBindGroup)
        cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(cardsPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 0)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Upload atlas mip 0 and generate the mip chain on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: CardAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS, ATLAS],
      format: 'rgba8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'card-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [ATLAS, ATLAS],
      format: 'rg8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'card-normal' },
  )
  device.queue.writeTexture(
    { texture: albedoTex },
    atlas.albedo,
    { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS },
    [ATLAS, ATLAS],
  )
  device.queue.writeTexture(
    { texture: normalTex },
    atlas.normalOct,
    { bytesPerRow: ATLAS * 2, rowsPerImage: ATLAS },
    [ATLAS, ATLAS],
  )

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string, format: GPUTextureFormat): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo', 'rgba8unorm')
  const normalPipe = mkPipeline('fs_normal', 'rg8unorm')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < MIP_LEVELS; level++) {
      const srcView = tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 })
      const dstView = tex.createView({ baseMipLevel: level, mipLevelCount: 1 })
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: srcView }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${level}`,
        colorAttachments: [{ view: dstView, loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  genMips(albedoTex, albedoPipe)
  genMips(normalTex, normalPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, normalTex }
}

/** Gribb–Hartmann frustum planes from a column-major view-proj matrix. */
function frustumPlanes(m: Float32Array | number[], out: Float32Array): void {
  const row = (r: number): [number, number, number, number] => [m[r]!, m[4 + r]!, m[8 + r]!, m[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const planes: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r2[0], r2[1], r2[2], r2[3]], // near (WebGPU z >= 0)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // far
  ]
  planes.forEach((p, i) => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1
    out[i * 4] = p[0] / len
    out[i * 4 + 1] = p[1] / len
    out[i * 4 + 2] = p[2] / len
    out[i * 4 + 3] = p[3] / len
  })
}
