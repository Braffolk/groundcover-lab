import binSrc from './shaders/bin.wgsl'
import nearSrc from './shaders/near.wgsl'
import resolveSrc from './shaders/resolve.wgsl'
import { ATLAS, MIPS, bakeSpecies, type BakedSpecies } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * 016-screen-stamp — iterate SCREEN TILES, not plants.
 *
 * Per frame:
 *   1. `bin` (compute, one workgroup per 16x16 tile): reduces the tile's
 *      scene depth into a world footprint, then finds every plant that
 *      projects into the tile by evaluating the procedural scatter twin over
 *      only the footprint's cells (enumerate / column-march / tint-only
 *      strategies), frustum-tests plant spheres, applies wind + fades,
 *      assigns each plant its baked hemi-octa view, and writes a
 *      front-to-back sorted list of <=32 packed 32-byte stamps per tile.
 *   2. `stamp` (fullscreen fragment): each pixel intersects its ray with the
 *      listed impostor cards, samples baked albedo/normal (mipmapped),
 *      composites front-to-back with early termination, then hands off to a
 *      noise-mixed aggregate meadow tint beyond the stamped range.
 *
 * Work is tiles x bounded constants — independent of plant count and stand
 * size. No geometry, no per-plant draws, no per-pixel marching (each pixel
 * does <=K precomputed lookups).
 */

const TILE_W = 16
const TILE_H = 8
const MAX_LIST = 64
const HEADER_U32 = 8
const ENTRY_U32 = 8
const STRIDE_BYTES = (HEADER_U32 + MAX_LIST * ENTRY_U32) * 4
const MAX_ENTRIES = 3 // resolve binds 3 atlas pairs; every shipped stand has <=3
const NEAR_SIDE = 6 // near-field cells per axis (near.wgsl NEAR_SIDE)

interface SpeciesGpu {
  albedo: GPUTexture
  normal: GPUTexture
  baked: BakedSpecies
}

interface ViewRes {
  width: number
  height: number
  tilesX: number
  tilesY: number
  depthTexture: GPUTexture
  buffer: GPUBuffer
  binBG: GPUBindGroup
  resolveBG: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const entries = ctx.stand.species.slice(0, MAX_ENTRIES)
  if (ctx.stand.species.length > MAX_ENTRIES) {
    console.warn(`[${ctx.id}] stand has ${ctx.stand.species.length} entries; rendering first ${MAX_ENTRIES}`)
  }

  // --- bake one atlas pair per unique species (fresh per session; the
  // shared bake cache is bypassed for the same reason 005 documents: the dev
  // server answers missing bake files with 200 index.html). ---
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (!cached) {
      cached = (async (): Promise<SpeciesGpu> => {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        const baked = await bakeSpecies(ctx, mesh)
        const mkTex = (kind: string): GPUTexture =>
          ctx.res.createTexture(
            {
              label: `${ctx.id}/${speciesId}/${kind}`,
              size: [ATLAS, ATLAS],
              format: 'rgba8unorm',
              mipLevelCount: MIPS,
              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            },
            { species: speciesId, tag: `atlas-${kind}` },
          )
        const albedo = mkTex('albedo')
        const normal = mkTex('normal')
        for (let l = 0; l < MIPS; l++) {
          const w = ATLAS >> l
          device.queue.writeTexture(
            { texture: albedo, mipLevel: l },
            baked.albedoMips[l]!,
            { bytesPerRow: w * 4, rowsPerImage: w },
            [w, w],
          )
          device.queue.writeTexture(
            { texture: normal, mipLevel: l },
            baked.normalMips[l]!,
            { bytesPerRow: w * 4, rowsPerImage: w },
            [w, w],
          )
        }
        return { albedo, normal, baked }
      })()
      speciesCache.set(speciesId, cached)
    }
    return cached
  }
  const entryGpu = await Promise.all(entries.map((e) => loadSpecies(e.species)))
  const tex = (i: number): SpeciesGpu => entryGpu[Math.min(i, entryGpu.length - 1)]!

  // --- shared uniform: seed/entry count/params + per-entry impostor meta ---
  const UNI_BYTES = 160
  const uniBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: UNI_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )
  const uniData = new ArrayBuffer(UNI_BYTES)
  const uniU32 = new Uint32Array(uniData, 0, 4)
  const uniF32 = new Float32Array(uniData)
  uniU32[2] = ctx.seed >>> 0
  uniU32[3] = entries.length
  entryGpu.forEach((g, i) => {
    const o = 8 + i * 4
    uniF32[o] = g.baked.center[0]
    uniF32[o + 1] = g.baked.center[1]
    uniF32[o + 2] = g.baked.center[2]
    uniF32[o + 3] = g.baked.radius
    const o2 = 24 + i * 4
    uniF32[o2] = g.baked.avgColor[0]
    uniF32[o2 + 1] = g.baked.avgColor[1]
    uniF32[o2 + 2] = g.baked.avgColor[2]
  })

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  const nearBGL = device.createBindGroupLayout({
    label: `${ctx.id}/near-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const nearBG = device.createBindGroup({
    label: `${ctx.id}/near-bg`,
    layout: nearBGL,
    entries: [
      { binding: 0, resource: { buffer: uniBuf } },
      { binding: 1, resource: tex(0).albedo.createView() },
      { binding: 2, resource: tex(1).albedo.createView() },
      { binding: 3, resource: tex(2).albedo.createView() },
      { binding: 4, resource: tex(0).normal.createView() },
      { binding: 5, resource: tex(1).normal.createView() },
      { binding: 6, resource: tex(2).normal.createView() },
      { binding: 7, resource: sampler },
    ],
  })

  const binBGL = device.createBindGroupLayout({
    label: `${ctx.id}/bin-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
    ],
  })
  const resolveBGL = device.createBindGroupLayout({
    label: `${ctx.id}/resolve-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  let nearPipeline!: GPURenderPipeline
  let binPipeline!: GPUComputePipeline
  let resolvePipeline!: GPURenderPipeline
  const build = (): void => {
    const nearModule = ctx.shaders.module(nearSrc)
    nearPipeline = device.createRenderPipeline({
      label: `${ctx.id}/near`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/near`, bindGroupLayouts: [ctx.frame.layout, nearBGL] }),
      vertex: { module: nearModule, entryPoint: 'vs_main' },
      fragment: { module: nearModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    binPipeline = device.createComputePipeline({
      label: `${ctx.id}/bin`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/bin`, bindGroupLayouts: [ctx.frame.layout, binBGL] }),
      compute: { module: ctx.shaders.module(binSrc), entryPoint: 'cs_bin' },
    })
    const resolveModule = ctx.shaders.module(resolveSrc)
    resolvePipeline = device.createRenderPipeline({
      label: `${ctx.id}/stamp`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/stamp`,
        bindGroupLayouts: [ctx.frame.layout, resolveBGL],
      }),
      vertex: { module: resolveModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: resolveModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: ctx.colorFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- per-view tile-list buffer + bind groups (A/B safe, resize safe) ---
  const viewRes = new Map<string, ViewRes>()
  const getViewRes = (targets: ViewTargets): ViewRes => {
    const prev = viewRes.get(targets.view)
    if (
      prev &&
      prev.width === targets.width &&
      prev.height === targets.height &&
      prev.depthTexture === targets.depthTexture
    ) {
      return prev
    }
    prev?.buffer.destroy()
    const tilesX = Math.ceil(targets.width / TILE_W)
    const tilesY = Math.ceil(targets.height / TILE_H)
    const buffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/tiles-${targets.view}`,
        size: tilesX * tilesY * STRIDE_BYTES,
        usage: GPUBufferUsage.STORAGE,
      },
      { tag: 'tile-lists' },
    )
    const depthView = targets.depthTexture.createView({ label: `${ctx.id}/depth-${targets.view}` })
    const binBG = device.createBindGroup({
      label: `${ctx.id}/bin-${targets.view}`,
      layout: binBGL,
      entries: [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: { buffer } },
        { binding: 2, resource: depthView },
      ],
    })
    const resolveBG = device.createBindGroup({
      label: `${ctx.id}/resolve-${targets.view}`,
      layout: resolveBGL,
      entries: [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: { buffer } },
        { binding: 2, resource: depthView },
        { binding: 3, resource: tex(0).albedo.createView() },
        { binding: 4, resource: tex(1).albedo.createView() },
        { binding: 5, resource: tex(2).albedo.createView() },
        { binding: 6, resource: tex(0).normal.createView() },
        { binding: 7, resource: tex(1).normal.createView() },
        { binding: 8, resource: tex(2).normal.createView() },
        { binding: 9, resource: sampler },
      ],
    })
    const res: ViewRes = {
      width: targets.width,
      height: targets.height,
      tilesX,
      tilesY,
      depthTexture: targets.depthTexture,
      buffer,
      binBG,
      resolveBG,
    }
    viewRes.set(targets.view, res)
    return res
  }

  return {
    update(_frame: FrameInfo): void {
      uniF32[4] = ctx.params.maxDist
      uniF32[5] = ctx.params.tintStrength
      uniF32[6] = ctx.params.debugTiles ? 1 : 0
      device.queue.writeBuffer(uniBuf, 0, uniData)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const vr = getViewRes(targets)

      // Near field first WITH depth write: binning's depth-guided footprint
      // and the stamp resolve's occlusion test then see near plants for free.
      const near = ctx.timing.renderPass(enc, 'near', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: {
          view: targets.depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      })
      near.setPipeline(nearPipeline)
      near.setBindGroup(0, ctx.frame.bindGroup)
      near.setBindGroup(1, nearBG)
      near.draw(6, NEAR_SIDE * NEAR_SIDE * 128 * entries.length)
      near.end()

      const bin = ctx.timing.computePass(enc, 'bin')
      bin.setPipeline(binPipeline)
      bin.setBindGroup(0, ctx.frame.bindGroup)
      bin.setBindGroup(1, vr.binBG)
      bin.dispatchWorkgroups(vr.tilesX, vr.tilesY)
      bin.end()

      const pass = ctx.timing.renderPass(enc, 'stamp', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
      })
      pass.setPipeline(resolvePipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, vr.resolveBG)
      pass.draw(3)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are destroyed by the harness via ctx.res.
    },
  }
}
