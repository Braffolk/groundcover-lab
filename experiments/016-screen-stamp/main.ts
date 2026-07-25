import binSrc from './shaders/bin.wgsl'
import carpetSrc from './shaders/carpet.wgsl'
import nearSrc from './shaders/near.wgsl'
import resolveSrc from './shaders/resolve.wgsl'
import {
  ATLAS,
  CARPET_MIPS,
  CARPET_TEX,
  MIPS,
  bakeCarpet,
  bakeSpecies,
  type BakedCarpet,
  type BakedSpecies,
} from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  speciesById,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type StandSpecies,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * 016-screen-stamp — iterate SCREEN TILES, not plants.
 *
 * Per frame:
 *   0. `carpet` (fullscreen, only when the stand has carpet entries): the mat
 *      layer, resolved from the depth buffer — ground point -> grid node ->
 *      wetness state -> that tile's top-view bake, one parallax step for
 *      cushion relief. No list, so a 484-tiles-per-cell life-size carpet costs
 *      the same as a sparse one. See shaders/carpet.wgsl.
 *   1. `near` (render): the constant ring of scatter cells around the camera
 *      as alpha-tested hemi-octa impostor cards with depth write (scattered
 *      species only).
 *   2. `bin` (compute, one workgroup per 16x8 tile): reduces the tile's scene
 *      depth into a world footprint, then finds every scattered plant that
 *      projects into the tile (enumerate / column-march / tint-only), wind +
 *      fades + hemi-octa view snap, writes a sorted list of <=K stamps.
 *   3. `stamp` (fullscreen): each pixel intersects its tile's cards, samples
 *      the baked views, composites front-to-back with early termination, then
 *      hands off to an aggregate meadow tint beyond the stamped range.
 *
 * Work is tiles x bounded constants + screen pixels — independent of plant
 * count and stand size. No geometry, no per-plant draws, no marching.
 */

const TILE_W = 16
const TILE_H = 8
const MAX_LIST = 64
const HEADER_U32 = 8
const ENTRY_U32 = 8
const STRIDE_BYTES = (HEADER_U32 + MAX_LIST * ENTRY_U32) * 4
/** Atlas pairs bound per pass: 3 scattered species + 3 carpet species. */
const MAX_SLOTS = 3
const NEAR_SIDE = 6 // near-field cells per axis (near.wgsl NEAR_SIDE)

interface CardGpu {
  albedo: GPUTexture
  normal: GPUTexture
  baked: BakedSpecies
}

interface CarpetGpu {
  albedo: GPUTexture
  nh: GPUTexture
  baked: BakedCarpet
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
  carpetBG: GPUBindGroup
}

const isCarpet = (e: StandSpecies): boolean => (e.carpetDiv ?? 0) > 0

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  // Two shapes of species, two representations. A carpet is a mat: it is
  // deferred per pixel from the depth buffer, never binned into tile lists
  // (484 life-size tiles per 4m cell would drown any bounded list). Everything
  // else is an upright plant and keeps the hemi-octa impostor path.
  const all = ctx.stand.species
  const cardIdx = all.map((e, i) => ({ e, i })).filter(({ e }) => !isCarpet(e))
  const carpetIdx = all.map((e, i) => ({ e, i })).filter(({ e }) => isCarpet(e))
  if (cardIdx.length > MAX_SLOTS || carpetIdx.length > MAX_SLOTS) {
    console.warn(
      `[${ctx.id}] stand has ${cardIdx.length} scattered / ${carpetIdx.length} carpet entries; rendering the first ${MAX_SLOTS} of each`,
    )
  }
  const cards = cardIdx.slice(0, MAX_SLOTS)
  const carpets = carpetIdx.slice(0, MAX_SLOTS)

  // --- bakes (fresh per session; the shared bake cache is bypassed for the
  // same reason 005 documents: the dev server answers missing bake files with
  // 200 index.html). ---
  const cardGpu: CardGpu[] = []
  for (const { e } of cards) {
    const mesh = await ctx.meshes.load(speciesById(e.species).meshId)
    const baked = await bakeSpecies(ctx, mesh)
    const mkTex = (kind: string): GPUTexture =>
      ctx.res.createTexture(
        {
          label: `${ctx.id}/${e.species}/${kind}`,
          size: [ATLAS, ATLAS],
          format: 'rgba8unorm',
          mipLevelCount: MIPS,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species: e.species, tag: `atlas-${kind}` },
      )
    const albedo = mkTex('albedo')
    const normal = mkTex('normal')
    for (let l = 0; l < MIPS; l++) {
      const w = ATLAS >> l
      device.queue.writeTexture({ texture: albedo, mipLevel: l }, baked.albedoMips[l]!, { bytesPerRow: w * 4 }, [w, w])
      device.queue.writeTexture({ texture: normal, mipLevel: l }, baked.normalMips[l]!, { bytesPerRow: w * 4 }, [w, w])
    }
    cardGpu.push({ albedo, normal, baked })
  }

  const carpetGpu: CarpetGpu[] = []
  for (const { e } of carpets) {
    const species = speciesById(e.species)
    const mesh = await ctx.meshes.load(species.meshId)
    const baked = await bakeCarpet(ctx, mesh, species.tileM ?? 0.18)
    if (species.tileM !== undefined && Math.abs(species.tileM - baked.tileM) > 1e-6) {
      console.warn(`[${ctx.id}] ${e.species}: mesh tile ${baked.tileM} != catalog tileM ${species.tileM}`)
    }
    const mkTex = (kind: string): GPUTexture =>
      ctx.res.createTexture(
        {
          label: `${ctx.id}/${e.species}/${kind}`,
          size: [CARPET_TEX, CARPET_TEX],
          format: 'rgba8unorm',
          mipLevelCount: CARPET_MIPS,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species: e.species, tag: `carpet-${kind}` },
      )
    const albedo = mkTex('albedo')
    const nh = mkTex('normal-height')
    for (let l = 0; l < CARPET_MIPS; l++) {
      const w = CARPET_TEX >> l
      device.queue.writeTexture({ texture: albedo, mipLevel: l }, baked.albedoMips[l]!, { bytesPerRow: w * 4 }, [w, w])
      device.queue.writeTexture({ texture: nh, mipLevel: l }, baked.nhMips[l]!, { bytesPerRow: w * 4 }, [w, w])
    }
    carpetGpu.push({ albedo, nh, baked })
  }

  const card = (i: number): CardGpu | null => cardGpu[Math.min(i, cardGpu.length - 1)] ?? null
  const carpet = (i: number): CarpetGpu | null => carpetGpu[Math.min(i, carpetGpu.length - 1)] ?? null
  const hasCards = cardGpu.length > 0
  const hasCarpet = carpetGpu.length > 0
  // A stand may legitimately have only one of the two shapes; the unused
  // bindings still need something format-compatible (they are never sampled,
  // because the slot count that gates them is 0).
  const cardAlb = (i: number): GPUTexture => (card(i) ?? { albedo: carpet(0)!.albedo }).albedo
  const cardNrm = (i: number): GPUTexture => {
    const c = card(i)
    return c ? c.normal : carpet(0)!.nh
  }
  const carpetAlb = (i: number): GPUTexture => (carpet(i) ?? { albedo: card(0)!.albedo }).albedo
  const carpetNh = (i: number): GPUTexture => {
    const c = carpet(i)
    return c ? c.nh : card(0)!.normal
  }

  // --- shared uniform: seed, slot tables, per-slot species meta, params ---
  const UNI_F32 = 68
  const UNI_BYTES = UNI_F32 * 4
  const uniBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: UNI_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )
  const uniData = new ArrayBuffer(UNI_BYTES)
  const uniU32 = new Uint32Array(uniData)
  const uniF32 = new Float32Array(uniData)
  uniU32[0] = cards.length
  uniU32[1] = carpets.length
  uniU32[2] = ctx.seed >>> 0
  uniU32[3] = all.length
  cards.forEach(({ i }, slot) => {
    uniU32[8 + slot] = i
  })
  carpets.forEach(({ i }, slot) => {
    uniU32[12 + slot] = i
  })
  cardGpu.forEach((g, slot) => {
    const o = 20 + slot * 4
    uniF32[o] = g.baked.center[0]
    uniF32[o + 1] = g.baked.center[1]
    uniF32[o + 2] = g.baked.center[2]
    uniF32[o + 3] = g.baked.radius
    const o2 = 36 + slot * 4
    uniF32[o2] = g.baked.avgColor[0]
    uniF32[o2 + 1] = g.baked.avgColor[1]
    uniF32[o2 + 2] = g.baked.avgColor[2]
  })

  // Carpet grid: one shared lattice for every carpet entry of the stand (the
  // stand contract has them partitioning the wetness axis over ONE grid, which
  // is what makes the mat continuous). carpet_div comes from the stand, never
  // from a guess, and the tile world size is the grid step exactly.
  const div = carpets.length > 0 ? (carpets[0]!.e.carpetDiv ?? 1) : 0
  const step = div > 0 ? SCATTER_CELL_SIZE / div : 0
  if (carpets.some(({ e }) => (e.carpetDiv ?? 0) !== div)) {
    console.warn(`[${ctx.id}] carpet entries disagree on carpetDiv; using ${div}`)
  }
  uniF32[16] = div
  uniF32[17] = step
  uniF32[18] = div > 0 ? CARPET_TEX / step : 0 // 1/texel size (m)
  uniF32[19] = CARPET_MIPS - 1
  carpets.forEach(({ e }, slot) => {
    const g = carpetGpu[slot]!
    const scale = step / g.baked.tileM
    const width = e.wetWidth ?? 0
    const lo = (e.wetCenter ?? 0) - width * 0.5
    const o = 52 + slot * 4
    uniF32[o] = lo
    uniF32[o + 1] = lo + width
    // The layer's own mean height above the ground: the parallax step displaces
    // the lookup by a CONSTANT height per species, not by the per-texel height.
    // A per-texel offset ripples the warp at the relief's own scale and combs
    // the mat into radial streaks up close (measured at 0.3m); the mean height
    // is the honest "flat layer at h" model, is continuous across every tile of
    // a species, and costs no texture tap.
    uniF32[o + 2] = (g.baked.yMin + g.baked.meanH01 * g.baked.yRange) * scale
    uniF32[o + 3] = g.baked.yRange * scale
  })

  // Aggregate-tint coverage. The tint stands in for a full cover of cards; with
  // a carpet already painting the ground it must only add the sparse card layer,
  // or a trace of grass would wash the whole far field. Stands without carpets
  // keep the historical 1.0 exactly.
  const tintCoverage = hasCarpet
    ? 1 -
      Math.exp(
        -cards.reduce((sum, { e, i }) => {
          const fp = speciesById(e.species).tileM ?? 0.35
          const band = e.wetWidth === undefined || e.wetWidth <= 0 ? 1 : Math.min(1, e.wetWidth)
          return sum + e.density * band * fp * fp
        }, 0),
      )
    : 1
  uniF32[7] = tintCoverage

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })
  // The carpet texture IS one period of a periodic field, so wrapping is the
  // correct filter continuation at the tile border (clamping would show a
  // half-texel seam on every tile edge).
  const carpetSampler = device.createSampler({
    label: `${ctx.id}/carpet-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    // The mat is seen at grazing angles most of the time; without anisotropy the
    // long axis of the footprint picks the flattest mip and the carpet reads as
    // paint (see carpet.wgsl).
    maxAnisotropy: 16,
  })

  const texEntry = (binding: number, stage: GPUShaderStageFlags): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: stage,
    texture: { sampleType: 'float' },
  })

  const nearBGL = device.createBindGroupLayout({
    label: `${ctx.id}/near-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4, 5, 6].map((b) => texEntry(b, GPUShaderStage.FRAGMENT)),
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const nearBG = hasCards
    ? device.createBindGroup({
        label: `${ctx.id}/near-bg`,
        layout: nearBGL,
        entries: [
          { binding: 0, resource: { buffer: uniBuf } },
          { binding: 1, resource: cardAlb(0).createView() },
          { binding: 2, resource: cardAlb(1).createView() },
          { binding: 3, resource: cardAlb(2).createView() },
          { binding: 4, resource: cardNrm(0).createView() },
          { binding: 5, resource: cardNrm(1).createView() },
          { binding: 6, resource: cardNrm(2).createView() },
          { binding: 7, resource: sampler },
        ],
      })
    : null

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
      ...[3, 4, 5, 6, 7, 8].map((b) => texEntry(b, GPUShaderStage.FRAGMENT)),
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const carpetBGL = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      ...[2, 3, 4, 5, 6, 7].map((b) => texEntry(b, GPUShaderStage.FRAGMENT)),
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  let nearPipeline: GPURenderPipeline | null = null
  let binPipeline!: GPUComputePipeline
  let resolvePipeline!: GPURenderPipeline
  let carpetPipeline: GPURenderPipeline | null = null
  const overBlend: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  }
  const build = (): void => {
    if (hasCards) {
      const nearModule = ctx.shaders.module(nearSrc)
      nearPipeline = device.createRenderPipeline({
        label: `${ctx.id}/near`,
        layout: device.createPipelineLayout({
          label: `${ctx.id}/near`,
          bindGroupLayouts: [ctx.frame.layout, nearBGL],
        }),
        vertex: { module: nearModule, entryPoint: 'vs_main' },
        fragment: { module: nearModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    }
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
        targets: [{ format: ctx.colorFormat, blend: overBlend }],
      },
      primitive: { topology: 'triangle-list' },
    })
    if (hasCarpet) {
      const carpetModule = ctx.shaders.module(carpetSrc)
      carpetPipeline = device.createRenderPipeline({
        label: `${ctx.id}/carpet`,
        layout: device.createPipelineLayout({
          label: `${ctx.id}/carpet`,
          bindGroupLayouts: [ctx.frame.layout, carpetBGL],
        }),
        vertex: { module: carpetModule, entryPoint: 'vs_fullscreen' },
        fragment: {
          module: carpetModule,
          entryPoint: 'fs_main',
          targets: [{ format: ctx.colorFormat, blend: overBlend }],
        },
        primitive: { topology: 'triangle-list' },
      })
    }
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
        { binding: 3, resource: cardAlb(0).createView() },
        { binding: 4, resource: cardAlb(1).createView() },
        { binding: 5, resource: cardAlb(2).createView() },
        { binding: 6, resource: cardNrm(0).createView() },
        { binding: 7, resource: cardNrm(1).createView() },
        { binding: 8, resource: cardNrm(2).createView() },
        { binding: 9, resource: sampler },
      ],
    })
    const carpetBG = device.createBindGroup({
      label: `${ctx.id}/carpet-${targets.view}`,
      layout: carpetBGL,
      entries: [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: depthView },
        { binding: 2, resource: carpetAlb(0).createView() },
        { binding: 3, resource: carpetAlb(1).createView() },
        { binding: 4, resource: carpetAlb(2).createView() },
        { binding: 5, resource: carpetNh(0).createView() },
        { binding: 6, resource: carpetNh(1).createView() },
        { binding: 7, resource: carpetNh(2).createView() },
        { binding: 8, resource: carpetSampler },
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
      carpetBG,
    }
    viewRes.set(targets.view, res)
    return res
  }

  // The uniform holds seed, slot tables, per-species meta and the three params
  // — nothing that varies per frame. Upload only when a param actually changes.
  const TILE_VIEWS = ['off', 'fill', 'mode'] as const
  let uniDirty = true

  return {
    update(_frame: FrameInfo): void {
      const tileView = TILE_VIEWS.indexOf(ctx.params.tileView)
      if (uniF32[4] !== ctx.params.maxDist || uniF32[5] !== ctx.params.tintStrength || uniF32[6] !== tileView) {
        uniF32[4] = ctx.params.maxDist
        uniF32[5] = ctx.params.tintStrength
        uniF32[6] = tileView
        uniDirty = true
      }
      if (uniDirty) {
        device.queue.writeBuffer(uniBuf, 0, uniData)
        uniDirty = false
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const vr = getViewRes(targets)

      // The mat first, while every non-sky depth texel is still terrain: the
      // carpet is the ground layer, so anything drawn later (near cards, far
      // stamps) simply composites in front of it.
      if (carpetPipeline) {
        const cp = ctx.timing.renderPass(enc, 'carpet', {
          colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        })
        cp.setPipeline(carpetPipeline)
        cp.setBindGroup(0, ctx.frame.bindGroup)
        cp.setBindGroup(1, vr.carpetBG)
        cp.draw(3)
        cp.end()
      }

      // Near field WITH depth write: binning's depth-guided footprint and the
      // stamp resolve's occlusion test then see near plants for free.
      if (nearPipeline && nearBG) {
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
        // Scattered entries only, so every entry has exactly SCATTER_MAX_PER_CELL slots.
        near.draw(6, NEAR_SIDE * NEAR_SIDE * SCATTER_MAX_PER_CELL * cards.length)
        near.end()
      }

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
