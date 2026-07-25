import cullSrc from './shaders/cull.wgsl'
import partsSrc from './shaders/parts.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { BANDS, FAR_MIPS, FAR_TILE, NEAR_MIPS, NEAR_TILE, PARTS, loadSpeciesParts, type PartAtlas } from './bake.ts'
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
 * Exact silhouette by parts. Startup: each species' part atlas (see bake.ts) is
 * loaded from mesh/baked / OPFS or baked in-browser from the raw mesh, uploaded
 * as two texture ARRAYS (near part tiles, far whole-plant tiles) and mipped.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and splits survivors into a near
 * list (plants closer than `partRadius`, drawn as 12 upright part cards + 3
 * horizontal band cards standing at their true 3-D positions inside the plant)
 * and a far list (one billboard pair, i.e. the baseline). Both draws are
 * indirect, hard alpha-tested with depth write, near list first so the far list
 * benefits from early-z. Per-frame cost is O(visible region), collapses with
 * distance, and is independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const PART_R_MAX = 32 // keep equal to the manifest's partRadius max
const INFO_FLOATS = 168
const NEAR_VERTS = (PARTS + BANDS) * 6
const FAR_VERTS = 12
const TOP_FRAC = 0.52

interface SpeciesGpu {
  atlas: PartAtlas
  nearAlbedo: GPUTexture
  nearNormal: GPUTexture
  farAlbedo: GPUTexture
  farNormal: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  capNear: number
  capFar: number
  infoNear: GPUBuffer
  infoFar: GPUBuffer
  indirect: GPUBuffer
  cullBindGroup: GPUBindGroup
  nearBindGroup: GPUBindGroup
  farBindGroup: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/tile-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~500MB) ---
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesParts(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- bind group layouts ---------------------------------------------------
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
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const a = gpu.atlas
    const density = Math.min(standEntry.density, 8)
    const capFar = Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    const capNear = Math.ceil(Math.PI * PART_R_MAX * PART_R_MAX * density * 1.15) + 512

    const mkInfo = (tag: string): GPUBuffer =>
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/info-${tag}-${entryIndex}`,
          size: INFO_FLOATS * 4,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        },
        { species: standEntry.species, tag: 'entry-info' },
      )
    const infoNear = mkInfo('near')
    const infoFar = mkInfo('far')
    // Static half of the uniform: the part/band boxes never change, so the
    // per-frame write only touches the first 48 floats.
    const staticInfo = new Float32Array(INFO_FLOATS)
    staticInfo.set(packBoxes(a), 48)
    device.queue.writeBuffer(infoNear, 0, staticInfo)
    device.queue.writeBuffer(infoFar, 0, staticInfo)

    const nearInst = ctx.res.createBuffer(
      { label: `${ctx.id}/near-inst-${entryIndex}`, size: capNear * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'near-instances' },
    )
    const farInst = ctx.res.createBuffer(
      { label: `${ctx.id}/far-inst-${entryIndex}`, size: capFar * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'far-instances' },
    )
    const indirect = ctx.res.createBuffer(
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
        { binding: 0, resource: { buffer: infoNear } },
        { binding: 1, resource: { buffer: nearInst } },
        { binding: 2, resource: { buffer: farInst } },
        { binding: 3, resource: { buffer: indirect } },
      ],
    })
    const mkDrawBg = (tag: string, info: GPUBuffer, inst: GPUBuffer, alb: GPUTexture, nrm: GPUTexture): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${tag}-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: info } },
          { binding: 1, resource: { buffer: inst } },
          { binding: 2, resource: alb.createView({ dimension: '2d-array' }) },
          { binding: 3, resource: nrm.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: sampler },
        ],
      })

    return {
      gpu,
      capNear,
      capFar,
      infoNear,
      infoFar,
      indirect,
      cullBindGroup,
      nearBindGroup: mkDrawBg('near', infoNear, nearInst, gpu.nearAlbedo, gpu.nearNormal),
      farBindGroup: mkDrawBg('far', infoFar, farInst, gpu.farAlbedo, gpu.farNormal),
      slotsPerFrame: 0,
    }
  })

  // --- pipelines ------------------------------------------------------------
  let cullPipeline!: GPUComputePipeline
  let partsPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    partsPipeline = device.createRenderPipeline({
      label: `${ctx.id}/parts`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/parts-pl`,
        bindGroupLayouts: [ctx.frame.layout, drawBgl],
      }),
      vertex: { module: ctx.shaders.module(partsSrc), entryPoint: 'vs_main' },
      fragment: {
        module: ctx.shaders.module(partsSrc),
        entryPoint: 'fs_main',
        targets: [{ format: ctx.colorFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const dyn = new Float32Array(48)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(8)
  indirectReset[0] = NEAR_VERTS
  indirectReset[4] = FAR_VERTS

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const partR = Math.min(ctx.params.partRadius, PART_R_MAX)
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

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        dyn.set(planes, 0)
        dyn[24] = x0
        dyn[25] = z0
        dyn[26] = sideX
        dyn[27] = sideZ
        dyn[28] = ctx.seed
        dyn[29] = entryIndex
        dyn[30] = R
        dyn[31] = entry.capNear
        dyn[32] = entry.capFar
        dyn[33] = partR
        dyn[34] = ctx.params.alphaRef
        dyn[35] = ctx.params.selfShade
        dyn[36] = ctx.params.bottomShade
        dyn[37] = a.y0
        dyn[38] = a.y1
        dyn[39] = a.rXZ
        dyn[40] = a.cx
        dyn[41] = a.cz
        dyn[42] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        dyn[44] = TOP_FRAC
        dyn[43] = 1 // near draw
        device.queue.writeBuffer(entry.infoNear, 0, dyn)
        dyn[43] = 0 // far draw
        device.queue.writeBuffer(entry.infoFar, 0, dyn)
        device.queue.writeBuffer(entry.indirect, 0, indirectReset)
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

      const pass = ctx.timing.renderPass(enc, 'parts', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(partsPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near plants first: they are the front of the scene, so much of the far
      // list is early-z rejected behind them.
      for (const entry of entries) {
        pass.setBindGroup(1, entry.nearBindGroup)
        pass.drawIndirect(entry.indirect, 0)
      }
      for (const entry of entries) {
        pass.setBindGroup(1, entry.farBindGroup)
        pass.drawIndirect(entry.indirect, 16)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Part/band boxes as the uniform's vec4 pairs (see entry_info.wgsl). */
function packBoxes(a: PartAtlas): Float32Array {
  const out = new Float32Array(PARTS * 8 + BANDS * 8)
  for (let p = 0; p < PARTS; p++) {
    out[p * 8] = a.parts[p * 6]!
    out[p * 8 + 1] = a.parts[p * 6 + 1]!
    out[p * 8 + 2] = a.parts[p * 6 + 2]!
    out[p * 8 + 4] = a.parts[p * 6 + 3]!
    out[p * 8 + 5] = a.parts[p * 6 + 4]!
    out[p * 8 + 6] = a.parts[p * 6 + 5]!
  }
  const bandBase = PARTS * 8
  for (let b = 0; b < BANDS; b++) {
    out[bandBase + b * 8] = a.bands[b * 6]!
    out[bandBase + b * 8 + 1] = a.bands[b * 6 + 1]!
    out[bandBase + b * 8 + 2] = a.bands[b * 6 + 2]!
    out[bandBase + b * 8 + 3] = a.bands[b * 6 + 3]!
    out[bandBase + b * 8 + 4] = a.bands[b * 6 + 4]!
  }
  return out
}

/** Upload both tile sets as texture arrays and build their mip chains. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: PartAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST

  const mkArray = (
    label: string,
    tile: number,
    layers: number,
    mips: number,
    format: GPUTextureFormat,
    data: Uint8Array<ArrayBuffer>,
    bytesPerTexel: number,
  ): GPUTexture => {
    const tex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${label}`,
        size: [tile, tile, layers],
        format,
        mipLevelCount: mips,
        usage,
      },
      { species: speciesId, tag: label },
    )
    device.queue.writeTexture(
      { texture: tex },
      data,
      { bytesPerRow: tile * bytesPerTexel, rowsPerImage: tile },
      [tile, tile, layers],
    )
    return tex
  }

  const nearLayers = atlas.nearAlbedo.byteLength / (NEAR_TILE * NEAR_TILE * 4)
  const farLayers = atlas.farAlbedo.byteLength / (FAR_TILE * FAR_TILE * 4)
  const nearAlbedo = mkArray('near-albedo', NEAR_TILE, nearLayers, NEAR_MIPS, 'rgba8unorm', atlas.nearAlbedo, 4)
  const nearNormal = mkArray('near-normal', NEAR_TILE, nearLayers, NEAR_MIPS, 'rg8unorm', atlas.nearNormal, 2)
  const farAlbedo = mkArray('far-albedo', FAR_TILE, farLayers, FAR_MIPS, 'rgba8unorm', atlas.farAlbedo, 4)
  const farNormal = mkArray('far-normal', FAR_TILE, farLayers, FAR_MIPS, 'rg8unorm', atlas.farNormal, 2)

  // --- mip chains (one pass per layer per level; tiles never bleed) ---------
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
  const genMips = (tex: GPUTexture, layers: number, mips: number, pipeline: GPURenderPipeline): void => {
    for (let layer = 0; layer < layers; layer++) {
      for (let level = 1; level < mips; level++) {
        const srcView = tex.createView({
          dimension: '2d',
          baseArrayLayer: layer,
          arrayLayerCount: 1,
          baseMipLevel: level - 1,
          mipLevelCount: 1,
        })
        const dstView = tex.createView({
          dimension: '2d',
          baseArrayLayer: layer,
          arrayLayerCount: 1,
          baseMipLevel: level,
          mipLevelCount: 1,
        })
        const bg = device.createBindGroup({
          label: `${ctx.id}/mipgen-bg-${layer}-${level}`,
          layout: bgl,
          entries: [{ binding: 0, resource: srcView }],
        })
        const pass = enc.beginRenderPass({
          label: `${ctx.id}/mipgen-${layer}-${level}`,
          colorAttachments: [{ view: dstView, loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bg)
        pass.draw(3)
        pass.end()
      }
    }
  }
  genMips(nearAlbedo, nearLayers, NEAR_MIPS, albedoPipe)
  genMips(nearNormal, nearLayers, NEAR_MIPS, normalPipe)
  genMips(farAlbedo, farLayers, FAR_MIPS, albedoPipe)
  genMips(farNormal, farLayers, FAR_MIPS, normalPipe)
  device.queue.submit([enc.finish()])

  return { atlas, nearAlbedo, nearNormal, farAlbedo, farNormal }
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
