import cullSrc from './shaders/cull.wgsl'
import prismsSrc from './shaders/prisms.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, MIP_LEVELS, loadSpeciesPrisms, type PrismAtlas } from './bake.ts'
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
 * Slice prisms. Startup: each species' prism atlas (8 azimuths x 3 vertical
 * depth slabs, 8 merged cards, 3 horizontal height slabs, 1 merged top;
 * albedo+coverage rgba8, oct normals rg8) is loaded from mesh/baked / OPFS or
 * baked in-browser from the raw mesh, uploaded and mipped.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into TWO
 * indirect draws — near plants as three depth prisms placed at their baked
 * centroids, far plants as a single merged card. Per-frame cost is O(visible
 * region), independent of the stand's plant count; the prism list is bounded
 * by a disc of radius slabDist, so it does not grow with the region either.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const SLAB_DIST_MAX = 20 // keep equal to the manifest's slabDist max
const INFO_FLOATS = 256
/**
 * Floats 0..47 of EntryInfo change every frame (frustum, region, params);
 * 48..255 are the baked atlas tables (prism depths, top heights, per-tile UV
 * and geometry boxes) and are uploaded once at startup.
 */
const INFO_DYNAMIC_FLOATS = 48
// Horizontal prisms only earn their fill from genuinely above the canopy top
// (see the signed elevation in prisms.wgsl); below this the canopy is
// described entirely by the vertical prisms and a lid would only read as a
// pale cutout floating among them.
const TOP_ELEV_LO = 0.15
const TOP_ELEV_HI = 0.45

interface SpeciesGpu {
  atlas: PrismAtlas
  albedoTex: GPUTexture
  normalTex: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  nearCapacity: number
  farCapacity: number
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  nearDrawBindGroup: GPUBindGroup
  farDrawBindGroup: GPUBindGroup
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
    const atlas = await loadSpeciesPrisms(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- bind group layouts / pipelines ---------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const farCapacity = Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    // The prism list only ever holds the disc of radius slabDist around the
    // camera — that is what keeps the expensive LOD bounded no matter how far
    // the region reaches or how many plants the stand has.
    const nearCapacity = Math.ceil(Math.PI * (SLAB_DIST_MAX * standEntry.scaleMax) ** 2 * density * 1.3) + 1024

    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const nearBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/near-${entryIndex}`, size: nearCapacity * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'prism-instances' },
    )
    const farBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/far-${entryIndex}`, size: farCapacity * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'card-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 32,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: nearBuffer } },
        { binding: 2, resource: { buffer: farBuffer } },
        { binding: 3, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = (instances: GPUBuffer, tag: string): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/${tag}-bg-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instances } },
          { binding: 2, resource: gpu.albedoTex.createView() },
          { binding: 3, resource: gpu.normalTex.createView() },
          { binding: 4, resource: sampler },
        ],
      })
    // Baked atlas tables never change — upload them once, not every frame.
    const tail = new Float32Array(INFO_FLOATS - INFO_DYNAMIC_FLOATS)
    tail.set(gpu.atlas.slabDepth, 0)
    tail.set(gpu.atlas.topHeight, 12)
    tail.set(gpu.atlas.uvRect, 16)
    tail.set(gpu.atlas.geoRect, 112)
    device.queue.writeBuffer(infoBuffer, INFO_DYNAMIC_FLOATS * 4, tail)

    return {
      gpu,
      nearCapacity,
      farCapacity,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      nearDrawBindGroup: drawBindGroup(nearBuffer, 'near'),
      farDrawBindGroup: drawBindGroup(farBuffer, 'far'),
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(prismsSrc)
    const drawPl = device.createPipelineLayout({
      label: `${ctx.id}/draw-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkDraw = (label: string, entryPoint: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${label}`,
        layout: drawPl,
        vertex: { module, entryPoint },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkDraw('prisms-near', 'vs_near')
    farPipeline = mkDraw('prisms-far', 'vs_far')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_DYNAMIC_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(8)

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      // Region cell rect, clamped to the stand's cell range on the CPU: cells
      // outside the stand hold nothing, so they must not be dispatched.
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      const tops = ctx.params.topCards
      indirectReset[0] = tops ? 36 : 18
      indirectReset[1] = 0
      indirectReset[4] = tops ? 12 : 6
      indirectReset[5] = 0

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.nearCapacity
        info[32] = entry.farCapacity
        info[33] = a.y0
        info[34] = a.y1
        info[35] = a.rXZ
        info[36] = a.cx
        info[37] = a.cz
        info[38] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info[39] = ctx.params.alphaRef
        info[40] = ctx.params.bottomShade
        info[41] = ctx.params.canopyShade
        info[42] = ctx.params.slabDist
        info[43] = TOP_ELEV_LO
        info[44] = TOP_ELEV_HI
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
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

      const pass = ctx.timing.renderPass(enc, 'prisms', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Prisms first: they are the nearest geometry in the frame, so the depth
      // they lay down lets early-z reject the merged cards behind them.
      pass.setPipeline(nearPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.nearDrawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 0)
      }
      pass.setPipeline(farPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.farDrawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 16)
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
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: PrismAtlas): SpeciesGpu {
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
    { species: speciesId, tag: 'prism-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [ATLAS, ATLAS],
      format: 'rg8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'prism-normal' },
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
  const list: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r2[0], r2[1], r2[2], r2[3]], // near (WebGPU z >= 0)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // far
  ]
  list.forEach((p, i) => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1
    out[i * 4] = p[0] / len
    out[i * 4 + 1] = p[1] / len
    out[i * 4 + 2] = p[2] / len
    out[i * 4 + 3] = p[3] / len
  })
}
