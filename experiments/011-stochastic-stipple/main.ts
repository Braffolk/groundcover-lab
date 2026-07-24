import cardsShaderSrc from './shaders/cards.wgsl'
import carpetShaderSrc from './shaders/carpet.wgsl'
import mipShaderSrc from './shaders/mip.wgsl'
import { ATLAS, MIP_LEVELS, loadOrBakeCards, topMeanColor, type CardMeta } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Stochastic thinning cascade — "opacity as statistics".
 *
 * Every plant is one baked coverage card (9-view atlas: 8 azimuths + top;
 * premultiplied albedo·coverage + normals, full mip chain baked once per
 * species). Distance collapse is hierarchical and stochastic:
 *
 *  1. PLANT level — Cook-style aggregate thinning: a plant survives at
 *     distance d iff its scatter slot rank < keep(d) = (R0/d)^2; survivors
 *     widen by 1/sqrt(keep) to conserve expected ensemble frontal area.
 *     Enumeration runs over cell annuli whose area quadruples exactly as the
 *     slot cap quarters, so per-ring work is constant and total per-frame
 *     work is O(rings) — independent of stand plant count.
 *  2. PIXEL level — the mip-averaged coverage alpha is realized by a hashed
 *     alpha test anchored to (plant id, card texel), morphing from a hard
 *     0.5 threshold near the camera to fully stochastic at distance.
 *  3. FIELD level — beyond amplification saturation a mean-field carpet
 *     (species-mottled canopy sheet) dissolves in underneath the cards.
 */

const CELL = 4 // must equal SCATTER_CELL_SIZE
const N0 = 8 // ring 0 half-width in cells (32m)
const MAX_RINGS = 6
const BAKE_VERSION = 1

interface RingRow {
  start: number
  n: number
  h: number
  cap: number
}

/** Ring table: start instance, outer/hole half-width (cells), slot cap. */
function ringTable(detailR: number, ringCount: number): { rows: RingRow[]; total: number } {
  const rows: RingRow[] = []
  let start = 0
  for (let k = 0; k < ringCount; k++) {
    const n = N0 << k
    const h = k === 0 ? 0 : n >> 1
    let cap = 128
    if (k > 0) {
      // Worst-case (nearest) plant distance in this ring, camera anywhere in
      // its own cell: (h-1) cells. keep() there bounds surviving slot ranks.
      const dmin = Math.max(1e-3, (h - 1) * CELL)
      const keep = Math.min(1, (detailR / dmin) ** 2)
      cap = Math.min(128, Math.ceil(128 * keep) + 1)
    }
    const cells = 4 * (n * n - h * h)
    rows.push({ start, n, h, cap })
    start += cells * cap
  }
  return { rows, total: start }
}

interface EntryGpu {
  uniform: GPUBuffer
  bindGroup: GPUBindGroup
  meta: CardMeta
  topMean: [number, number, number]
  speciesId: string
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // ---------------------------------------------------------------- bake ----
  // One atlas pair per unique species; mips generated on the GPU after load.
  const mipModule = ctx.shaders.module(mipShaderSrc)
  const mipSampler = device.createSampler({ label: `${ctx.id}/mip-samp`, magFilter: 'linear', minFilter: 'linear' })
  const mipPipeline = device.createRenderPipeline({
    label: `${ctx.id}/mip`,
    layout: 'auto',
    vertex: { module: mipModule, entryPoint: 'vs' },
    fragment: { module: mipModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  })

  const genMips = (tex: GPUTexture): void => {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/genmips` })
    for (let level = 1; level < MIP_LEVELS; level++) {
      const src = tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 })
      const dst = tex.createView({ baseMipLevel: level, mipLevelCount: 1 })
      const bg = device.createBindGroup({
        label: `${ctx.id}/mip-bg-${level}`,
        layout: mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: src },
          { binding: 1, resource: mipSampler },
        ],
      })
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: dst, loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(mipPipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
    device.queue.submit([enc.finish()])
  }

  interface SpeciesGpu {
    albedo: GPUTexture
    normal: GPUTexture
    meta: CardMeta
    /** Mean visible canopy color from the top-down tile (carpet mean field). */
    topMean: [number, number, number]
  }
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (!cached) {
      cached = (async (): Promise<SpeciesGpu> => {
        const baked = await loadOrBakeCards(ctx, speciesId, speciesById(speciesId).meshId, BAKE_VERSION)
        const mkTex = (name: string, data: Uint8Array): GPUTexture => {
          const tex = ctx.res.createTexture(
            {
              label: `${ctx.id}/${speciesId}/${name}`,
              size: [ATLAS, ATLAS],
              format: 'rgba8unorm',
              mipLevelCount: MIP_LEVELS,
              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            },
            { species: speciesId, tag: `cards-${name}` },
          )
          device.queue.writeTexture({ texture: tex }, data as BufferSource, { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
          genMips(tex)
          return tex
        }
        return {
          albedo: mkTex('albedo', baked.albedo),
          normal: mkTex('normal', baked.normal),
          meta: baked.meta,
          topMean: topMeanColor(baked.albedo),
        }
      })()
      speciesCache.set(speciesId, cached)
    }
    return cached
  }

  // ------------------------------------------------------------- pipeline ----
  const atlasSampler = device.createSampler({
    label: `${ctx.id}/atlas-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    lodMaxClamp: 5,
  })

  const cardBgl = device.createBindGroupLayout({
    label: `${ctx.id}/card-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const carpetBgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
  })

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const gpu = await loadSpecies(e.species)
      const uniform = ctx.res.createBuffer(
        { label: `${ctx.id}/uni-${entryIndex}`, size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'card-uni' },
      )
      const bindGroup = device.createBindGroup({
        label: `${ctx.id}/card-bg-${entryIndex}`,
        layout: cardBgl,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: gpu.albedo.createView() },
          { binding: 2, resource: gpu.normal.createView() },
          { binding: 3, resource: atlasSampler },
        ],
      })
      return { uniform, bindGroup, meta: gpu.meta, topMean: gpu.topMean, speciesId: e.species }
    }),
  )

  const carpetUniform = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-uni`, size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'carpet-uni' },
  )
  const carpetBindGroup = device.createBindGroup({
    label: `${ctx.id}/carpet-bg`,
    layout: carpetBgl,
    entries: [{ binding: 0, resource: { buffer: carpetUniform } }],
  })

  // Card quad as an INDEXED draw: the two triangles share two corners, so the
  // post-transform vertex cache shades 4 vertices per instance instead of 6 —
  // a third fewer runs of the (expensive) placement/thinning vertex stage for
  // exactly the same triangles.
  const quadIndices = ctx.res.createBuffer(
    { label: `${ctx.id}/card-idx`, size: 12, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'card-idx' },
  )
  device.queue.writeBuffer(quadIndices, 0, new Uint16Array([0, 1, 2, 1, 3, 2]))

  let cardPipeline!: GPURenderPipeline
  let carpetPipeline!: GPURenderPipeline
  const build = (): void => {
    const cardModule = ctx.shaders.module(cardsShaderSrc)
    cardPipeline = device.createRenderPipeline({
      label: `${ctx.id}/cards`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cards-pl`, bindGroupLayouts: [ctx.frame.layout, cardBgl] }),
      vertex: { module: cardModule, entryPoint: 'vs_main' },
      fragment: { module: cardModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    const carpetModule = ctx.shaders.module(carpetShaderSrc)
    carpetPipeline = device.createRenderPipeline({
      label: `${ctx.id}/carpet`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/carpet-pl`, bindGroupLayouts: [ctx.frame.layout, carpetBgl] }),
      vertex: { module: carpetModule, entryPoint: 'vs_main' },
      fragment: { module: carpetModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // ------------------------------------------------------------- uniforms ----
  const cardScratch = new ArrayBuffer(160)
  const cardF = new Float32Array(cardScratch)
  const cardU = new Uint32Array(cardScratch)
  const cardI = new Int32Array(cardScratch)
  const carpetScratch = new Float32Array(28)
  const carpetU = new Uint32Array(carpetScratch.buffer)

  let totalInstances = 0
  let drawCarpet = false

  // Carpet mean-field statistics: a pure function of the stand + the baked
  // per-species meta, both frozen for the lifetime of the experiment. Computed
  // ONCE here, never per frame.
  const carpetStats = ((): { colors: number[]; height: number } => {
    const weights = ctx.stand.species.map((e, i) => {
      const meanScale = (e.scaleMin + e.scaleMax) / 2
      return e.density * (entries[i]?.meta.coverTop ?? 0.5) * meanScale * meanScale
    })
    const sum = weights.reduce((a, b) => a + b, 0) || 1
    let cum = 0
    const colors: number[] = []
    for (let i = 0; i < 4; i++) {
      cum += (weights[i] ?? 0) / sum
      const rgb = entries[i]?.topMean ?? [0.2, 0.3, 0.12]
      colors.push(rgb[0], rgb[1], rgb[2], i === ctx.stand.species.length - 1 ? 1.0 : cum)
    }
    let height = 0
    ctx.stand.species.forEach((e, i) => {
      const meanScale = (e.scaleMin + e.scaleMax) / 2
      height += ((weights[i] ?? 0) / sum) * speciesById(e.species).heightScale * meanScale
    })
    return { colors, height }
  })()

  // The card uniforms depend only on (params, camera CELL, stand, bake) and the
  // carpet uniform only on (params, stand, bake) — neither is a per-frame
  // quantity. Both are rebuilt on change only; a still camera writes nothing.
  let ringRows: RingRow[] = []
  let ringKeyR = Number.NaN
  let ringKeyCount = -1
  const ensureRingTable = (detailR: number, ringCount: number): void => {
    if (detailR === ringKeyR && ringCount === ringKeyCount) return
    const { rows, total } = ringTable(detailR, ringCount)
    ringRows = rows
    totalInstances = total
    ringKeyR = detailR
    ringKeyCount = ringCount
  }
  // [detailRadius, rings, widthCap, densify, stochStart, stochFull, debugRings, camCellX, camCellZ]
  const cardKey = new Float64Array(9).fill(Number.NaN)
  const carpetKey = new Float64Array(3).fill(Number.NaN)
  const keyChanged = (key: Float64Array, next: readonly number[]): boolean => {
    let dirty = false
    for (let i = 0; i < key.length; i++) {
      if (key[i] !== next[i]) {
        key[i] = next[i]!
        dirty = true
      }
    }
    return dirty
  }

  return {
    update(frame: FrameInfo): void {
      const p = ctx.params
      const ringCount = Math.round(p.rings)
      ensureRingTable(p.detailRadius, ringCount)
      const camCellX = Math.floor(frame.camera.pose.x / CELL)
      const camCellZ = Math.floor(frame.camera.pose.z / CELL)

      if (
        keyChanged(cardKey, [
          p.detailRadius,
          ringCount,
          p.widthCap,
          p.densify,
          p.stochStart,
          p.stochFull,
          p.debugRings ? 1 : 0,
          camCellX,
          camCellZ,
        ])
      ) {
        for (let k = 0; k < MAX_RINGS; k++) {
          const row = ringRows[Math.min(k, ringRows.length - 1)]!
          cardU[k * 4] = row.start
          cardU[k * 4 + 1] = row.n
          cardU[k * 4 + 2] = row.h
          cardU[k * 4 + 3] = row.cap
        }
        cardU[24] = ringCount
        cardU[25] = ctx.seed >>> 0
        cardU[27] = p.debugRings ? 1 : 0
        cardI[28] = camCellX
        cardI[29] = camCellZ
        cardF[30] = ctx.stand.radius
        cardF[31] = p.detailRadius
        cardF[36] = p.widthCap
        cardF[37] = p.densify
        cardF[38] = p.stochStart
        cardF[39] = p.stochFull
        for (let e = 0; e < entries.length; e++) {
          const entry = entries[e]!
          cardU[26] = e
          cardF[32] = entry.meta.center[0]
          cardF[33] = entry.meta.center[1]
          cardF[34] = entry.meta.center[2]
          cardF[35] = entry.meta.rCard
          device.queue.writeBuffer(entry.uniform, 0, cardScratch)
        }
      }

      // Carpet: from where clump amplification saturates to past the horizon.
      const innerR = Math.max(p.widthCap * p.detailRadius * 0.6, 30)
      drawCarpet = p.carpet && ctx.stand.radius * 1.45 > innerR
      if (drawCarpet && keyChanged(carpetKey, [p.detailRadius, ringCount, p.widthCap])) {
        const cascadeOuter = N0 * (1 << (ringCount - 1)) * CELL
        const outerR = Math.max(800, Math.min(3000, ctx.stand.radius * 1.5))
        carpetScratch.set(carpetStats.colors, 0)
        carpetU[16] = ctx.stand.species.length
        carpetU[17] = ctx.seed >>> 0
        carpetScratch[20] = innerR
        carpetScratch[21] = Math.max(outerR, cascadeOuter)
        carpetScratch[22] = carpetStats.height * 0.45
        carpetScratch[23] = ctx.stand.radius
        carpetScratch[24] = p.widthCap * p.detailRadius * 0.9 // fade band
        device.queue.writeBuffer(carpetUniform, 0, carpetScratch)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'stipple', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Cards first, carpet last. Both are depth-tested opaque (alpha test, no
      // blending), so order cannot change the image — but the near card layer
      // is a solid occluder, and drawing it first lets early-z reject the
      // carpet fragments hidden behind it instead of shading them.
      pass.setPipeline(cardPipeline)
      pass.setIndexBuffer(quadIndices, 'uint16')
      for (const entry of entries) {
        pass.setBindGroup(1, entry.bindGroup)
        pass.drawIndexed(6, totalInstances)
      }
      if (drawCarpet) {
        pass.setPipeline(carpetPipeline)
        pass.setBindGroup(1, carpetBindGroup)
        pass.draw(96 * 20 * 6)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are destroyed by the harness via ctx.res.
    },
  }
}
