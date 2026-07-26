import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import {
  ALBEDO_MIPS,
  ATLAS_H,
  ATLAS_W,
  K_SUB,
  MERGED,
  NORMAL_MIPS,
  NRM_H,
  NRM_W,
  N_UNITS,
  N_VIEWS,
  loadSpeciesClumps,
  tileRect,
  type ClumpAtlas,
} from './bake.ts'
import { ALB_MIPS, AUX_MIPS, AUX_PX, SHELLS, TILE_PX, loadCarpetTile, type CarpetTile } from './carpet.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Clump impostors. Startup: each species' artifact is loaded from mesh/baked /
 * OPFS (or baked in-browser from the raw mesh).
 *
 *  * an upright species gets the CLUMP ATLAS — the plant cut into K_SUB spatial
 *    sub-clumps along whole-blade boundaries, each captured from 8 azimuths +
 *    straight down, plus one merged whole-plant unit for the distance LOD;
 *  * a CARPET species (stand carpetDiv > 0, e.g. Sphagnum) gets the CARPET TILE
 *    instead — one straight-down capture of its periodic tile with the cushion
 *    height field, and no side views at all, because a mat never shows one.
 *    It never loads the clump atlas: 45 side tiles for a 7cm cushion is the
 *    whole 25MB budget spent on views nobody sees.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into near
 * and far lists. Near = K_SUB sub-clump cards standing at their real offsets
 * inside the plant (parallax, mutual occlusion, interleaving), or SHELLS
 * ground-parallel quads through a carpet tile's relief; far = the single merged
 * card, or one ground-parallel tile quad. Per-frame cost is O(visible region),
 * independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const LOD_MAX = 40 // keep equal to the manifest's lodDistance max
const CARPET_LOD_MAX = 24 // keep equal to the manifest's carpetShellDist max
const TOP_FRAC = 0.52
const DYN_FLOATS = 48
const STATIC_FLOATS = 4 + N_UNITS * 8 + 12 + N_UNITS * N_VIEWS * 8
// Front-to-back draw buckets — must match the constants in info.wgsl.
const NEAR_BINS = 3
const FAR_BINS = 4
const N_BINS = NEAR_BINS + FAR_BINS

interface SpeciesGpu {
  /** Upright species: the 45-tile clump atlas. Carpet species: null. */
  atlas: ClumpAtlas | null
  /** Carpet species: the periodic tile + its cushion height field. */
  carpet: CarpetTile | null
  albedoTex: GPUTexture
  normalTex: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  isCarpet: boolean
  /** Candidate slots per scatter cell — carpetDiv², or the scatter budget. */
  enumSlots: number
  nearBinCap: number
  farBinCap: number
  dynBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  /** One per distance ring: the instance buffer bound at that ring's slice. */
  binBindGroups: GPUBindGroup[]
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
  // A carpet tile texture IS periodic — its left edge really does neighbour its
  // right edge, so the bilinear taps at the border must wrap, not clamp.
  const tileSampler = device.createSampler({
    label: `${ctx.id}/tile-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  // --- species artifacts (sequential: the poa bake transiently needs ~500MB) -
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    if ((entry.carpetDiv ?? 0) > 0) {
      speciesGpu.set(entry.species, uploadCarpet(ctx, entry.species, await loadCarpetTile(ctx, entry.species)))
    } else {
      speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, await loadSpeciesClumps(ctx, entry.species)))
    }
  }

  // --- bind group layouts ----------------------------------------------------
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
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const carpetDiv = standEntry.carpetDiv ?? 0
    const isCarpet = carpetDiv > 0
    // TWO DIFFERENT NUMBERS — conflating them silently drops plants:
    //  * enumSlots is how many candidate slots the cull must EVALUATE per cell.
    //    A carpet has carpetDiv² of them (484 at life size), deliberately over
    //    SCATTER_MAX_PER_CELL; visiting only 128 renders the first ~6 of 22
    //    rows in every cell, i.e. 4m-pitch stripes of moss.
    //  * expectedPerCell is how many are expected to SURVIVE, which is all the
    //    instance rings need to hold — for a zone-partitioned carpet that is
    //    roughly its wetness band's width of the slots.
    const enumSlots = standEntrySlots(standEntry)
    const band = Math.min(1, (standEntry.wetWidth ?? 1) * 1.25)
    const expectedPerCell = isCarpet ? enumSlots * band : SCATTER_MAX_PER_CELL * (Math.min(standEntry.density, 8) / 8)
    const perM2 = expectedPerCell / (CELL * CELL)
    // The cull rejects outside a CIRCULAR region, so the circle bound is the
    // honest capacity — no need for the region's square bound. Bins are
    // equal-area, so each holds ~1/BINS of that (plus slack for clustering).
    // Rounded to 16 instances so every ring's byte offset is 256-aligned and
    // can be bound directly (see binBindGroups).
    const cap16 = (n: number): number => Math.ceil(n / 16) * 16
    // The near rings of a carpet are sized for the FULL grid rather than the
    // band (and so need no extra clustering slack): the near disc spans only a
    // few 12m wetness cells, so one state can own all of it, and dropping tiles
    // there would open holes right under the eye. Upright species keep the
    // factors they shipped with.
    const lodMax = isCarpet ? CARPET_LOD_MAX : LOD_MAX
    const nearPerM2 = isCarpet ? enumSlots / (CELL * CELL) : perM2
    const nearSlack = isCarpet ? 1.05 : 1.4
    const farBinCap = cap16(Math.ceil((Math.PI * REGION_MAX * REGION_MAX * perM2 * 1.2) / FAR_BINS) + 512)
    const nearBinCap = cap16(Math.ceil((Math.PI * lodMax * lodMax * nearPerM2 * nearSlack) / NEAR_BINS) + 512)
    const dynBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/dyn-${entryIndex}`,
        size: DYN_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const staticBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/atlas-info-${entryIndex}`,
        size: STATIC_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'atlas-info' },
    )
    device.queue.writeBuffer(staticBuffer, 0, buildStaticInfo(gpu).buffer)
    const nearBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/near-${entryIndex}`, size: nearBinCap * NEAR_BINS * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'near-instances' },
    )
    const farBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/far-${entryIndex}`, size: farBinCap * FAR_BINS * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'far-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: N_BINS * 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: dynBuffer } },
        { binding: 1, resource: { buffer: nearBuffer } },
        { binding: 2, resource: { buffer: farBuffer } },
        { binding: 3, resource: { buffer: indirectBuffer } },
      ],
    })
    // One bind group per distance ring, with the instance buffer bound at that
    // ring's slice. Indirect draws cannot use a non-zero `firstInstance` unless
    // the optional `indirect-first-instance` feature is enabled (it is not), so
    // the ring offset has to live in the BINDING, not in the draw args.
    const binBindGroups = Array.from({ length: N_BINS }, (_unused, b) => {
      const near = b < NEAR_BINS
      const size = (near ? nearBinCap : farBinCap) * 16
      const offset = (near ? b : b - NEAR_BINS) * size
      return device.createBindGroup({
        label: `${ctx.id}/bin${b}-bg-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: dynBuffer } },
          { binding: 1, resource: { buffer: staticBuffer } },
          { binding: 2, resource: { buffer: near ? nearBuffer : farBuffer, offset, size } },
          { binding: 3, resource: gpu.albedoTex.createView() },
          { binding: 4, resource: gpu.normalTex.createView() },
          { binding: 5, resource: isCarpet ? tileSampler : sampler },
        ],
      })
    })
    return {
      gpu,
      isCarpet,
      enumSlots,
      nearBinCap,
      farBinCap,
      dynBuffer,
      indirectBuffer,
      cullBindGroup,
      binBindGroups,
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
  const dyn = new Float32Array(DYN_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(N_BINS * 4)

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

      const tops = ctx.params.topCard ? 2 : 1
      entries.forEach((entry, entryIndex) => {
        // A carpet tile is SHELLS ground-parallel quads near and ONE far; it
        // never gets a camera-facing card or a top card (`topCard` does not
        // apply to a mat — the mat IS the top).
        const nearVerts = entry.isCarpet ? 6 * SHELLS : 6 * K_SUB * tops
        const farVerts = entry.isCarpet ? 6 : 6 * tops
        // Draw args per ring: vertexCount, instanceCount = 0 (the cull counts),
        // firstVertex = 0, firstInstance = 0 (the ring offset is in the binding).
        for (let b = 0; b < N_BINS; b++) {
          indirectReset[b * 4] = b < NEAR_BINS ? nearVerts : farVerts
          indirectReset[b * 4 + 1] = 0
          indirectReset[b * 4 + 2] = 0
          indirectReset[b * 4 + 3] = 0
        }
        const a = entry.gpu.atlas
        const carpet = entry.gpu.carpet
        // Bounding volume for the cull's frustum test, at scale 1.
        const my0 = carpet ? carpet.y0 : a!.unitBox[MERGED * 4 + 3]!
        const my1 = carpet ? carpet.y1 : a!.unitExt[MERGED * 4]!
        const mr = carpet ? carpet.tileM * 0.708 : a!.unitBox[MERGED * 4 + 2]!
        // Separate LOD distances: a 0.18m tile stops resolving its 1cm relief
        // at ~7m, far sooner than a 1m clump stops showing parallax.
        const lodDist = entry.isCarpet
          ? Math.min(ctx.params.carpetShellDist, CARPET_LOD_MAX)
          : Math.min(ctx.params.lodDistance, LOD_MAX)
        dyn.set(planes, 0)
        dyn[24] = x0
        dyn[25] = z0
        dyn[26] = sideX
        dyn[27] = sideZ
        dyn[28] = ctx.seed
        dyn[29] = entryIndex
        dyn[30] = R
        dyn[31] = entry.nearBinCap
        dyn[32] = entry.farBinCap
        dyn[33] = Math.max(mr, (my1 - my0) / 2) * 1.05
        dyn[34] = ctx.params.alphaRef
        dyn[35] = TOP_FRAC
        dyn[36] = ctx.params.bottomShade
        dyn[37] = lodDist
        dyn[38] = my0
        dyn[39] = my1
        dyn[40] = mr
        dyn[41] = ctx.params.swaySpread
        dyn[42] = ctx.params.nearAlphaBias
        dyn[43] = entry.enumSlots
        dyn[44] = ctx.params.carpetShade
        device.queue.writeBuffer(entry.dynBuffer, 0, dyn)
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
      // Front to back, one indirect draw per equal-area distance ring: hard
      // alpha test + depth write means the nearest ring becomes an occluder and
      // early-z rejects most of the deeper rings before they ever sample.
      for (let b = 0; b < N_BINS; b++) {
        for (const entry of entries) {
          pass.setBindGroup(1, entry.binBindGroups[b]!)
          pass.drawIndirect(entry.indirectBuffer, b * 16)
        }
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Atlas layout + per-unit capture boxes — written once, never per frame. */
function buildStaticInfo(gpu: SpeciesGpu): Float32Array {
  const out = new Float32Array(STATIC_FLOATS)
  const boxBase = 4
  const extBase = boxBase + N_UNITS * 4
  const carpetBase = extBase + N_UNITS * 4
  const tileBase = carpetBase + 12
  const carpet = gpu.carpet
  if (carpet) {
    out[0] = carpet.tilePx
    out[1] = carpet.tilePx
    out[carpetBase] = carpet.y0
    out[carpetBase + 1] = carpet.y1
    out[carpetBase + 2] = carpet.farH
    out[carpetBase + 3] = carpet.reliefSpan
    out.set(carpet.shellY, carpetBase + 4)
    out.set(carpet.shellT, carpetBase + 8)
    return out
  }
  const atlas = gpu.atlas!
  out[0] = ATLAS_W
  out[1] = ATLAS_H
  for (let u = 0; u < N_UNITS; u++) {
    out[boxBase + u * 4] = atlas.unitBox[u * 4]!
    out[boxBase + u * 4 + 1] = atlas.unitBox[u * 4 + 1]!
    out[boxBase + u * 4 + 2] = atlas.unitBox[u * 4 + 2]!
    out[boxBase + u * 4 + 3] = atlas.unitBox[u * 4 + 3]!
    out[extBase + u * 4] = atlas.unitExt[u * 4]!
  }
  for (let u = 0; u < N_UNITS; u++) {
    for (let v = 0; v < N_VIEWS; v++) {
      const [tx, ty, tw, th] = tileRect(u, v)
      const t = u * N_VIEWS + v
      const lx0 = atlas.tight[t * 4]!
      const lt0 = atlas.tight[t * 4 + 1]!
      const lx1 = atlas.tight[t * 4 + 2]!
      const lt1 = atlas.tight[t * 4 + 3]!
      const o = tileBase + t * 8
      out[o] = (tx + lx0 * tw) / ATLAS_W
      out[o + 1] = (ty + lt0 * th) / ATLAS_H
      out[o + 2] = ((lx1 - lx0) * tw) / ATLAS_W
      out[o + 3] = ((lt1 - lt0) * th) / ATLAS_H
      out[o + 4] = lx0
      out[o + 5] = lt0
      out[o + 6] = lx1 - lx0
      out[o + 7] = lt1 - lt0
    }
  }
  return out
}

/**
 * Upload a carpet tile. Both chains ship baked — the albedo because its rgb is
 * already coverage-normalised at every level (a GPU box filter over
 * non-premultiplied colour would darken every edge toward the cleared black),
 * and the aux because its normal channels are a hemisphere pair that has to be
 * re-normalised per level.
 */
function uploadCarpet(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, tile: CarpetTile): SpeciesGpu {
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/carpet-albedo`,
      size: [TILE_PX, TILE_PX],
      format: 'rgba8unorm',
      mipLevelCount: ALB_MIPS,
      usage,
    },
    { species: speciesId, tag: 'carpet-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/carpet-aux`,
      size: [AUX_PX, AUX_PX],
      format: 'rgba8unorm',
      mipLevelCount: AUX_MIPS,
      usage,
    },
    { species: speciesId, tag: 'carpet-normal-height' },
  )
  const upload = (tex: GPUTexture, levels: Uint8Array<ArrayBuffer>[], size: number): void => {
    levels.forEach((level, mipLevel) => {
      const w = size >> mipLevel
      ctx.device.queue.writeTexture({ texture: tex, mipLevel }, level, { bytesPerRow: w * 4, rowsPerImage: w }, [w, w])
    })
  }
  upload(albedoTex, tile.albedoLevels, TILE_PX)
  upload(normalTex, tile.auxLevels, AUX_PX)
  return { atlas: null, carpet: tile, albedoTex, normalTex }
}

/** Upload atlas mip 0 and generate the mip chain on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: ClumpAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS_W, ATLAS_H],
      format: 'rgba8unorm',
      mipLevelCount: ALBEDO_MIPS,
      usage,
    },
    { species: speciesId, tag: 'clump-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [NRM_W, NRM_H],
      format: 'rg8unorm',
      mipLevelCount: NORMAL_MIPS,
      usage,
    },
    { species: speciesId, tag: 'clump-normal' },
  )
  // The albedo mip chain is baked, not generated: every level carries its own
  // coverage-calibrated alpha (see bake.ts), which a box filter would destroy.
  atlas.albedoLevels.forEach((level, mipLevel) => {
    const w = ATLAS_W >> mipLevel
    const h = ATLAS_H >> mipLevel
    device.queue.writeTexture(
      { texture: albedoTex, mipLevel },
      level,
      { bytesPerRow: w * 4, rowsPerImage: h },
      [w, h],
    )
  })
  device.queue.writeTexture(
    { texture: normalTex },
    atlas.normalOct,
    { bytesPerRow: NRM_W * 2, rowsPerImage: NRM_H },
    [NRM_W, NRM_H],
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
  const normalPipe = mkPipeline('fs_normal', 'rg8unorm')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, pipeline: GPURenderPipeline, levels: number): void => {
    for (let level = 1; level < levels; level++) {
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
  genMips(normalTex, normalPipe, NORMAL_MIPS)
  device.queue.submit([enc.finish()])

  return { atlas, carpet: null, albedoTex, normalTex }
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
