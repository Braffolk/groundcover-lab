import cullSrc from './shaders/cull.wgsl'
import argsSrc from './shaders/args.wgsl'
import patchesSrc from './shaders/patches.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import {
  FAR_MIPS,
  FAR_NRM_MIPS,
  FAR_NRM_RES,
  FAR_RES,
  FAR_SLICES,
  NEAR_MIPS,
  NEAR_NRM_MIPS,
  NEAR_NRM_RES,
  NEAR_RES,
  NEAR_SLICES,
  SLICES,
  loadPatchAtlas,
  type PatchAtlas,
} from './bake.ts'
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
 * Canopy patches: a plant is a STACK of baked canopy patches, not a card.
 *
 * Startup: each species' patch atlas is loaded (mesh/baked, OPFS, or baked
 * in-browser from the raw GCMESH1 mesh) into two texture arrays — a NEAR array
 * of 8 azimuths x 3 depth slabs + 2 horizontal crown slabs, and a much smaller
 * FAR array of 8 azimuth composites + 1 crown composite — and mipped on the GPU.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into two
 * contiguous ranges (patch-stack / composite-card) of one instance array; then a
 * single 64-thread dispatch fans the two counters out into 7 indirect draw slots
 * per stand entry. The draws are issued quad-major (all front slabs of all
 * species first, then all second slabs, ...), so the front slabs prime the depth
 * buffer and early-z eats most of the deeper slabs' fragments.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const PATCH_MAX = 40 // keep equal to the manifest's patchDist max
// Draw slots per stand entry. QUADS/NEAR_QUADS are mirrored by args.wgsl
// (QUADS/FIRST_FAR) and by the quad indices in patches.wgsl — change all three.
const QUADS = 6 // 2 depth slabs + 2 crowns + composite card + composite crown
const NEAR_QUADS = 4
const POS_RANGE = 160 // instance xz quantisation half-range (m)
const INFO_FLOATS = 52 + SLICES * 4 * 2
const INFO_HEAD_FLOATS = 52 // everything before the (constant) slice tables

interface SpeciesGpu {
  atlas: PatchAtlas
  nearAlbedo: GPUTexture
  nearNormal: GPUTexture
  farAlbedo: GPUTexture
  farNormal: GPUTexture
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  capNear: number
  capFar: number
  infoBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  nearBindGroup: GPUBindGroup
  farBindGroup: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/patch-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) ---
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadPatchAtlas(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- bind group layouts ---------------------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const argsBgl = device.createBindGroupLayout({
    label: `${ctx.id}/args-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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

  const entryCount = ctx.stand.species.length
  const countersBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/counters`, size: entryCount * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'lod-counters' },
  )
  const argsBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/draw-args`,
      size: entryCount * QUADS * 16,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
    },
    { tag: 'indirect-args' },
  )
  const argsBindGroup = device.createBindGroup({
    label: `${ctx.id}/args-bg`,
    layout: argsBgl,
    entries: [
      { binding: 0, resource: { buffer: countersBuffer } },
      { binding: 1, resource: { buffer: argsBuffer } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const capNear = Math.ceil(Math.PI * PATCH_MAX * PATCH_MAX * density * 1.15) + 1024
    const capFar = Math.ceil(Math.PI * REGION_MAX * REGION_MAX * density * 1.15) + 4096
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: (capNear + capFar) * 12, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    // Slice tables are pure bake data — uploaded once, never per frame.
    device.queue.writeBuffer(infoBuffer, INFO_HEAD_FLOATS * 4, sliceTables(gpu.atlas))

    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: countersBuffer } },
      ],
    })
    const mkDrawBg = (name: string, albedo: GPUTexture, normal: GPUTexture): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${name}-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer } },
          { binding: 2, resource: albedo.createView({ dimension: '2d-array' }) },
          { binding: 3, resource: normal.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: sampler },
        ],
      })
    return {
      speciesId: standEntry.species,
      gpu,
      capNear,
      capFar,
      infoBuffer,
      cullBindGroup,
      nearBindGroup: mkDrawBg('near', gpu.nearAlbedo, gpu.nearNormal),
      farBindGroup: mkDrawBg('far', gpu.farAlbedo, gpu.farNormal),
      slotsPerFrame: 0,
    }
  })

  // --- pipelines ------------------------------------------------------------
  let cullPipeline!: GPUComputePipeline
  let argsPipeline!: GPUComputePipeline
  let patchPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    argsPipeline = device.createComputePipeline({
      label: `${ctx.id}/args`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/args-pl`,
        bindGroupLayouts: [ctx.frame.layout, argsBgl],
      }),
      compute: { module: ctx.shaders.module(argsSrc), entryPoint: 'cs_args' },
    })
    patchPipeline = device.createRenderPipeline({
      label: `${ctx.id}/patches`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/patches-pl`,
        bindGroupLayouts: [ctx.frame.layout, drawBgl],
      }),
      vertex: { module: ctx.shaders.module(patchesSrc), entryPoint: 'vs_main' },
      fragment: {
        module: ctx.shaders.module(patchesSrc),
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
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const countersZero = new Uint32Array(entryCount * 2)

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
      const patchDist = Math.min(ctx.params.patchDist, PATCH_MAX)

      device.queue.writeBuffer(countersBuffer, 0, countersZero)
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
        info[31] = entry.capNear
        info[32] = entry.capFar
        info[33] = entry.capNear // far_base
        info[34] = patchDist
        info[35] = ctx.params.alphaRef
        info[36] = ctx.params.slabSpread
        info[37] = ctx.params.slabShade
        info[38] = ctx.params.bottomShade
        info[39] = a.y0
        info[40] = a.y1
        info[41] = a.rXZ
        info[42] = a.cx
        info[43] = a.cz
        info[44] = a.crownH
        info[45] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info[46] = cam.x
        info[47] = cam.z
        info[48] = POS_RANGE
        info[49] = ctx.params.topPatches ? 1 : 0
        info[50] = ctx.params.patchTint ? 1 : 0
        device.queue.writeBuffer(entry.infoBuffer, 0, info, 0, INFO_HEAD_FLOATS)
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
      // Same pass: fan the two LOD counters out into the 7 draw slots per entry.
      cull.setPipeline(argsPipeline)
      cull.setBindGroup(1, argsBindGroup)
      cull.dispatchWorkgroups(1)
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'patches', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(patchPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Quad-major: every plant's front slab lands in the depth buffer before
      // any deeper slab is rasterized, which is what keeps the extra layers
      // nearly free at grazing angles.
      for (let q = 0; q < NEAR_QUADS; q++) {
        entries.forEach((entry, entryIndex) => {
          pass.setBindGroup(1, entry.nearBindGroup)
          pass.drawIndirect(argsBuffer, (entryIndex * QUADS + q) * 16)
        })
      }
      for (let q = NEAR_QUADS; q < QUADS; q++) {
        entries.forEach((entry, entryIndex) => {
          pass.setBindGroup(1, entry.farBindGroup)
          pass.drawIndirect(argsBuffer, (entryIndex * QUADS + q) * 16)
        })
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Constant part of the info uniform: per-slice framing rects + plane depths. */
function sliceTables(atlas: PatchAtlas): Float32Array<ArrayBuffer> {
  const out = new Float32Array(SLICES * 4 * 2)
  for (let i = 0; i < SLICES; i++) {
    const m = i * 8
    out[i * 4] = atlas.meta[m]!
    out[i * 4 + 1] = atlas.meta[m + 1]!
    out[i * 4 + 2] = atlas.meta[m + 2]!
    out[i * 4 + 3] = atlas.meta[m + 3]!
    out[SLICES * 4 + i * 4] = atlas.meta[m + 4]!
  }
  return out
}

/** Upload both patch arrays at mip 0 and generate their chains on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: PatchAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST

  const mkTex = (name: string, res: number, layers: number, mips: number, format: GPUTextureFormat): GPUTexture =>
    ctx.res.createTexture(
      { label: `${ctx.id}/${speciesId}/${name}`, size: [res, res, layers], format, mipLevelCount: mips, usage },
      { species: speciesId, tag: `patch-${name}` },
    )
  const nearAlbedo = mkTex('near-albedo', NEAR_RES, NEAR_SLICES, NEAR_MIPS, 'rgba8unorm')
  const nearNormal = mkTex('near-normal', NEAR_NRM_RES, NEAR_SLICES, NEAR_NRM_MIPS, 'rg8unorm')
  const farAlbedo = mkTex('far-albedo', FAR_RES, FAR_SLICES, FAR_MIPS, 'rgba8unorm')
  const farNormal = mkTex('far-normal', FAR_NRM_RES, FAR_SLICES, FAR_NRM_MIPS, 'rg8unorm')

  const write = (tex: GPUTexture, data: Uint8Array<ArrayBuffer>, res: number, layers: number, bpp: number): void => {
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: res * bpp, rowsPerImage: res }, [res, res, layers])
  }
  write(nearAlbedo, atlas.nearAlbedo, NEAR_RES, NEAR_SLICES, 4)
  write(nearNormal, atlas.nearNormal, NEAR_NRM_RES, NEAR_SLICES, 2)
  write(farAlbedo, atlas.farAlbedo, FAR_RES, FAR_SLICES, 4)
  write(farNormal, atlas.farNormal, FAR_NRM_RES, FAR_SLICES, 2)

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
        const viewOpts = { dimension: '2d' as const, baseArrayLayer: layer, arrayLayerCount: 1, mipLevelCount: 1 }
        const srcView = tex.createView({ ...viewOpts, baseMipLevel: level - 1 })
        const dstView = tex.createView({ ...viewOpts, baseMipLevel: level })
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
  genMips(nearAlbedo, NEAR_SLICES, NEAR_MIPS, albedoPipe)
  genMips(nearNormal, NEAR_SLICES, NEAR_NRM_MIPS, normalPipe)
  genMips(farAlbedo, FAR_SLICES, FAR_MIPS, albedoPipe)
  genMips(farNormal, FAR_SLICES, FAR_NRM_MIPS, normalPipe)
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
