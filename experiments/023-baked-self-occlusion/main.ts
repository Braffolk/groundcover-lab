import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS_A, ATLAS_B, loadSpeciesAtlas, type SelfOccAtlas } from './bake.ts'
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
 * Baked self-occlusion cards. Startup: per species a 5x5 atlas of baked views
 * (8 azimuths x 3 elevations + top) is loaded from mesh/baked / OPFS or baked
 * in-browser from the raw GCMESH1 mesh, uploaded and mipped — an albedo +
 * coverage atlas and a geometry atlas holding, per texel, the surface depth
 * behind the tile's near plane, the fraction of sky it can see, and a shading
 * normal leaned toward its openness direction.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into TWO
 * indirect draws — near plants get the depth-warped card (3 taps), far plants
 * get the same card flat (2 taps). One view-aligned quad per plant either way.
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 112 // keep equal to the manifest's regionRadius max
const NEAR_SPLIT_MAX = 44 // keep equal to the manifest's nearSplit max
const MIPS_A = Math.floor(Math.log2(ATLAS_A)) + 1
const MIPS_B = Math.floor(Math.log2(ATLAS_B)) + 1
const INFO_FLOATS = 48

interface SpeciesGpu {
  atlas: SelfOccAtlas
  albedoTex: GPUTexture
  geomTex: GPUTexture
  tileBuffer: GPUBuffer
}

interface EntryGpu {
  gpu: SpeciesGpu
  capNear: number
  capFar: number
  infoBuffer: GPUBuffer
  argsNear: GPUBuffer
  argsFar: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawNear: GPUBindGroup
  drawFar: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const mkSampler = (name: string, aniso: number): GPUSampler =>
    device.createSampler({
      label: `${ctx.id}/${name}`,
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      maxAnisotropy: aniso,
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
  // Coverage/albedo is the sharp channel and earns anisotropy at grazing
  // angles; the geometry atlas is a third of that resolution and does not.
  const albedoSampler = mkSampler('albedo-sampler', 4)
  const geomSampler = mkSampler('geom-sampler', 1)

  // --- species atlases (sequential: the poa bake is transiently heavy) -------
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesAtlas(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    // Region area x density plus slack. The scatter's count over a region this
    // size has a relative spread well under 1%, so 6% + a flat 2k is ample.
    const capNear = Math.ceil(Math.PI * NEAR_SPLIT_MAX ** 2 * density * 1.2) + 1024
    const capFar = Math.ceil(Math.PI * REGION_MAX ** 2 * density * 1.06) + 2048
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const mkInst = (name: string, cap: number): GPUBuffer =>
      ctx.res.createBuffer(
        { label: `${ctx.id}/${name}-${entryIndex}`, size: cap * 16, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: 'culled-instances' },
      )
    const instNear = mkInst('inst-near', capNear)
    const instFar = mkInst('inst-far', capFar)
    const mkArgs = (name: string): GPUBuffer =>
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/${name}-${entryIndex}`,
          size: 16,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        },
        { species: standEntry.species, tag: 'indirect-args' },
      )
    const argsNear = mkArgs('args-near')
    const argsFar = mkArgs('args-far')
    const mkDraw = (name: string, inst: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/${name}-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: inst } },
          { binding: 2, resource: { buffer: gpu.tileBuffer } },
          { binding: 3, resource: gpu.albedoTex.createView() },
          { binding: 4, resource: gpu.geomTex.createView() },
          { binding: 5, resource: albedoSampler },
          { binding: 6, resource: geomSampler },
        ],
      })
    return {
      gpu,
      capNear,
      capFar,
      infoBuffer,
      argsNear,
      argsFar,
      cullBindGroup: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instNear } },
          { binding: 2, resource: { buffer: instFar } },
          { binding: 3, resource: { buffer: argsNear } },
          { binding: 4, resource: { buffer: argsFar } },
        ],
      }),
      drawNear: mkDraw('draw-near', instNear),
      drawFar: mkDraw('draw-far', instFar),
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
    const module = ctx.shaders.module(cardsSrc)
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkCards = (name: string, vs: string, fs: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${name}`,
        layout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkCards('cards-near', 'vs_near', 'fs_near')
    farPipeline = mkCards('cards-far', 'vs_far', 'fs_far')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const argsReset = new Uint32Array([6, 0, 0, 0])

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

      const split = Math.min(ctx.params.nearSplit, NEAR_SPLIT_MAX)
      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        info.set(planes, 0)
        info.set([x0, z0, sideX, sideZ], 24)
        info.set([ctx.seed, entryIndex, R, split], 28)
        info.set([entry.capNear, entry.capFar, ctx.params.alphaRef, a.sphereR], 32)
        info.set([a.aabbC[0], a.aabbC[1], a.aabbC[2], a.aabbC[1] + a.aabbH[1]], 36)
        info.set([a.aabbH[0], a.aabbH[1], a.aabbH[2], 0], 40)
        info.set([ctx.params.occlusion, ctx.params.transmission, ctx.params.canopyDepth, ctx.params.parallax], 44)
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.argsNear, 0, argsReset)
        device.queue.writeBuffer(entry.argsFar, 0, argsReset)
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

      const pass = ctx.timing.renderPass(enc, 'cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near first: coarse front-to-back so early-z rejects the far tier.
      pass.setPipeline(nearPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawNear)
        pass.drawIndirect(entry.argsNear, 0)
      }
      pass.setPipeline(farPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawFar)
        pass.drawIndirect(entry.argsFar, 0)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Upload both atlases plus the tile table, and build the mip chains. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: SelfOccAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS_A, ATLAS_A],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_A,
      usage,
    },
    { species: speciesId, tag: 'view-albedo' },
  )
  const geomTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/geom`,
      size: [ATLAS_B, ATLAS_B],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_B,
      usage,
    },
    { species: speciesId, tag: 'view-geometry' },
  )
  device.queue.writeTexture(
    { texture: albedoTex },
    atlas.albedo,
    { bytesPerRow: ATLAS_A * 4, rowsPerImage: ATLAS_A },
    [ATLAS_A, ATLAS_A],
  )
  device.queue.writeTexture(
    { texture: geomTex },
    atlas.geom,
    { bytesPerRow: ATLAS_B * 4, rowsPerImage: ATLAS_B },
    [ATLAS_B, ATLAS_B],
  )
  const tileBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/tiles`,
      size: 100 * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'tile-table' },
  )
  device.queue.writeBuffer(tileBuffer, 0, atlas.tileTable)

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo')
  const geomPipe = mkPipeline('fs_geom')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, levels: number, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < levels; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${level}`,
        colorAttachments: [
          { view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  genMips(albedoTex, MIPS_A, albedoPipe)
  genMips(geomTex, MIPS_B, geomPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, geomTex, tileBuffer }
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
