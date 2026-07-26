import cardsShaderSrc from './shaders/cards.wgsl'
import carpetShaderSrc from './shaders/carpet.wgsl'
import mipShaderSrc from './shaders/mip.wgsl'
import {
  ATLAS,
  MIP_LEVELS,
  MOSS_MIPS,
  MOSS_TEX,
  heightQuantiles,
  loadOrBakeCards,
  loadOrBakeMoss,
  topMeanColor,
  type CardMeta,
  type MossMeta,
} from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  speciesById,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type StandSpecies,
  type ViewTargets,
} from '@harness'
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

const CELL = SCATTER_CELL_SIZE
const N0 = 8 // ring 0 half-width in cells (32m)
const MAX_RINGS = 6
const MAX_SHELLS = 6
const BAKE_VERSION = 1
/** Uniform size: 56 words (cards + carpet block, 16-aligned) + 6 vec4 shell rows. */
const CARD_UNI_BYTES = 224 + MAX_SHELLS * 16
/** First word of the per-shell (threshold, height) rows. */
const SHELL_TAB_WORD = 56

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
  /** 9-view card meta (scattered species) — null for a carpet entry. */
  meta: CardMeta | null
  /** Periodic tile meta (carpet species) — null for a scattered entry. */
  moss: MossMeta | null
  topMean: [number, number, number]
  speciesId: string
  /** Fractional area the species covers seen from above (mean-field weight). */
  coverTop: number
  /** carpet_div (0 = ordinary scatter). */
  carpetDiv: number
  /** Slots per scatter cell — carpet_div² for a mat, else the scatter budget. */
  slots: number
  /** Constant tile scale of a carpet entry (mesh -> world). */
  tileScale: number
  /** Inverse CDF of the tile's cushion-top height (carpet entries). */
  quantiles: number[]
  /** Per-frame draw shape, rebuilt on param/camera change. */
  instances: number
  indexCount: number
}

/**
 * True scale of a stand entry. A carpet entry's scaleMin/scaleMax in the stand
 * object are placeholders — the harness computes the real constant tile scale
 * so the tile exactly fills its grid step (and only writes it into the GPU
 * stand table), so derive it the same way here.
 */
function entryScale(e: StandSpecies): number {
  const tileM = speciesById(e.species).tileM
  if (e.carpetDiv && e.carpetDiv > 0 && tileM) return SCATTER_CELL_SIZE / (e.carpetDiv * tileM)
  return (e.scaleMin + e.scaleMax) / 2
}

/**
 * Relief shells of a carpet tile, as equal-area bands of its cushion-top height
 * field: shell k keeps the texels above the k/shells area quantile and sits at
 * the median height of its own band. Shell 0's threshold is 0, so it covers the
 * whole tile — the mat can never open, whatever the shell LOD drops.
 */
function shellTable(quantiles: number[], shells: number): { t: number; h: number }[] {
  const at = (f: number): number => {
    const x = Math.min(Math.max(f, 0), 1) * (quantiles.length - 1)
    const i = Math.min(quantiles.length - 2, Math.floor(x))
    return quantiles[i]! + (quantiles[i + 1]! - quantiles[i]!) * (x - i)
  }
  const rows: { t: number; h: number }[] = []
  for (let k = 0; k < shells; k++) rows.push({ t: k === 0 ? 0 : at(k / shells), h: at((k + 0.5) / shells) })
  return rows
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

  const genMips = (tex: GPUTexture, levels: number): void => {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/genmips` })
    for (let level = 1; level < levels; level++) {
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
    /** Card path: 9-view atlas. Carpet path: albedo*cov / cov tile. */
    albedo: GPUTexture
    /** Card path: normal atlas. Carpet path: (normal, cushion height) tile. */
    normal: GPUTexture
    meta: CardMeta | null
    moss: MossMeta | null
    /** Mean visible canopy color from above (carpet mean field). */
    topMean: [number, number, number]
    coverTop: number
    quantiles: number[]
  }

  const mkTex = (
    speciesId: string,
    name: string,
    size: number,
    levels: number,
    data: Uint8Array,
  ): GPUTexture => {
    const tex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${name}`,
        size: [size, size],
        format: 'rgba8unorm',
        mipLevelCount: levels,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      },
      { species: speciesId, tag: `cards-${name}` },
    )
    device.queue.writeTexture({ texture: tex }, data as BufferSource, { bytesPerRow: size * 4, rowsPerImage: size }, [size, size])
    genMips(tex, levels)
    return tex
  }

  // Keyed by species AND shape: a carpet species gets the periodic-tile bake,
  // a scattered one the 9-view card atlas.
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string, carpet: boolean): Promise<SpeciesGpu> => {
    const key = `${speciesId}${carpet ? '|carpet' : ''}`
    let cached = speciesCache.get(key)
    if (!cached) {
      cached = (async (): Promise<SpeciesGpu> => {
        const meshId = speciesById(speciesId).meshId
        if (carpet) {
          const baked = await loadOrBakeMoss(ctx, speciesId, meshId, BAKE_VERSION)
          return {
            albedo: mkTex(speciesId, 'tile-albedo', MOSS_TEX, MOSS_MIPS, baked.albedo),
            normal: mkTex(speciesId, 'tile-nrmh', MOSS_TEX, MOSS_MIPS, baked.nrmh),
            meta: null,
            moss: baked.meta,
            topMean: baked.meta.meanColor,
            coverTop: baked.meta.coverMean,
            quantiles: heightQuantiles(baked),
          }
        }
        const baked = await loadOrBakeCards(ctx, speciesId, meshId, BAKE_VERSION)
        return {
          albedo: mkTex(speciesId, 'albedo', ATLAS, MIP_LEVELS, baked.albedo),
          normal: mkTex(speciesId, 'normal', ATLAS, MIP_LEVELS, baked.normal),
          meta: baked.meta,
          moss: null,
          topMean: topMeanColor(baked.albedo),
          coverTop: baked.meta.coverTop,
          quantiles: [],
        }
      })()
      speciesCache.set(key, cached)
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
  // A carpet tile IS periodic, so its texture wraps: repeat addressing makes
  // the bilinear tap at u=0/1 blend with the geometry that overflows the
  // period, exactly as the mat does in world space.
  const tileSampler = device.createSampler({
    label: `${ctx.id}/tile-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
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
      const carpetDiv = e.carpetDiv ?? 0
      const gpu = await loadSpecies(e.species, carpetDiv > 0)
      const uniform = ctx.res.createBuffer(
        { label: `${ctx.id}/uni-${entryIndex}`, size: CARD_UNI_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'card-uni' },
      )
      const bindGroup = device.createBindGroup({
        label: `${ctx.id}/card-bg-${entryIndex}`,
        layout: cardBgl,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: gpu.albedo.createView() },
          { binding: 2, resource: gpu.normal.createView() },
          { binding: 3, resource: carpetDiv > 0 ? tileSampler : atlasSampler },
        ],
      })
      return {
        uniform,
        bindGroup,
        meta: gpu.meta,
        moss: gpu.moss,
        topMean: gpu.topMean,
        coverTop: gpu.coverTop,
        speciesId: e.species,
        carpetDiv,
        // NEVER SCATTER_MAX_PER_CELL: a carpet has carpet_div² slots per cell
        // (484 for the bog moss), deliberately over the 128-slot budget.
        slots: standEntrySlots(e),
        tileScale: entryScale(e),
        quantiles: gpu.quantiles,
        instances: 0,
        indexCount: 6,
      }
    }),
  )
  const hasCarpet = entries.some((e) => e.carpetDiv > 0)

  const carpetUniform = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-uni`, size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
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
  //
  // The same buffer carries up to MAX_SHELLS quads: vertex index >> 2 is the
  // carpet shell number, so a mat entry asks for 6*shells indices and a card
  // entry for the first 6 (identical values to before, bit for bit).
  const quadIndices = ctx.res.createBuffer(
    { label: `${ctx.id}/card-idx`, size: 12 * MAX_SHELLS, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'card-idx' },
  )
  {
    const idx = new Uint16Array(6 * MAX_SHELLS)
    for (let s = 0; s < MAX_SHELLS; s++) {
      const b = 4 * s
      idx.set([b, b + 1, b + 2, b + 1, b + 3, b + 2], s * 6)
    }
    device.queue.writeBuffer(quadIndices, 0, idx)
  }

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
  const cardScratch = new ArrayBuffer(CARD_UNI_BYTES)
  const cardF = new Float32Array(cardScratch)
  const cardU = new Uint32Array(cardScratch)
  const cardI = new Int32Array(cardScratch)
  const carpetScratch = new Float32Array(44)
  const carpetU = new Uint32Array(carpetScratch.buffer)

  let totalInstances = 0
  let drawCarpet = false

  // Carpet mean-field statistics: a pure function of the stand + the baked
  // per-species meta, both frozen for the lifetime of the experiment. Computed
  // ONCE here, never per frame.
  const carpetStats = ((): { colors: number[]; height: number } => {
    const weights = ctx.stand.species.map((e, i) => {
      const entry = entries[i]
      const scale = entry?.tileScale ?? 1
      // A carpet entry's `density` is nominal (the grid, not the density, sets
      // its coverage) but it is 8 = full, which is also what a closed mat is.
      return e.density * (entry?.coverTop ?? 0.5) * scale * scale
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
      const entry = entries[i]
      const scale = entry?.tileScale ?? 1
      // A mat's height is its baked cushion top, not the species heightScale.
      const h = entry?.moss ? entry.moss.hMin + entry.moss.hRange : speciesById(e.species).heightScale
      height += ((weights[i] ?? 0) / sum) * h * scale
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
  // [detailRadius, rings, widthCap, densify, stochStart, stochFull, debugRings,
  //  camCellX, camCellZ, mossRadius, mossShells, mossShellR, mossCoverRef]
  const cardKey = new Float64Array(16).fill(Number.NaN)
  const carpetKey = new Float64Array(4).fill(Number.NaN)
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

      const shells = Math.round(p.mossShells)
      // Camera height over the ground, quantized so a slow flight does not
      // rewrite the uniforms every frame. Only used as a shell LOD metric.
      const camAbove = hasCarpet
        ? Math.round(
            Math.max(0, frame.camera.pose.y - ctx.scene.terrain.height(frame.camera.pose.x, frame.camera.pose.z)) * 4,
          ) / 4
        : 0
      // A carpet tile block: every slot of every cell within mossRadius must be
      // visited (holes look exactly like a placement bug), so the cell block is
      // sized from the radius and the SLOT COUNT from carpet_div².
      const mossCells = Math.ceil(p.mossRadius / CELL)

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
          p.mossRadius,
          shells,
          p.mossShellR,
          p.mossCoverRef,
          p.mossRelief,
          p.mossLift,
          camAbove,
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
          const moss = entry.moss
          if (moss) {
            cardF[32] = 0
            cardF[33] = 0
            cardF[34] = 0
            cardF[35] = 0
            cardU[40] = 1
            cardU[41] = mossCells
            cardU[42] = entry.slots
            cardU[43] = shells
            cardF[44] = entry.tileScale
            cardF[45] = moss.hMin
            cardF[46] = moss.hRange
            cardF[47] = p.mossRadius
            cardF[48] = p.mossCoverRef
            cardF[49] = p.mossShellR
            cardF[50] = camAbove
            cardF[51] = p.mossRelief
            cardF[52] = p.mossLift
            const rows = shellTable(entry.quantiles, shells)
            for (let k = 0; k < MAX_SHELLS; k++) {
              const row = rows[Math.min(k, rows.length - 1)]!
              cardF[SHELL_TAB_WORD + k * 4] = row.t
              cardF[SHELL_TAB_WORD + k * 4 + 1] = row.h
              cardF[SHELL_TAB_WORD + k * 4 + 2] = 0
              cardF[SHELL_TAB_WORD + k * 4 + 3] = 0
            }
            entry.instances = (2 * mossCells) ** 2 * entry.slots
            entry.indexCount = 6 * shells
          } else {
            const meta = entry.meta!
            cardF[32] = meta.center[0]
            cardF[33] = meta.center[1]
            cardF[34] = meta.center[2]
            cardF[35] = meta.rCard
            cardU[40] = 0
            cardU[41] = 0
            cardU[42] = 0
            cardU[43] = 0
            cardF[44] = 0
            cardF[45] = 0
            cardF[46] = 0
            cardF[47] = 0
            cardF[48] = 0
            cardF[49] = 0
            cardF[50] = 0
            cardF[51] = 0
            cardF[52] = 0
            entry.instances = totalInstances
            entry.indexCount = 6
          }
          device.queue.writeBuffer(entry.uniform, 0, cardScratch)
        }
      } else {
        // The ring table can change without the key (it is keyed separately).
        for (const entry of entries) if (!entry.moss) entry.instances = totalInstances
      }

      // Mean-field sheet. On a carpet stand it is the moss mat's far field, so
      // it takes over just inside where the tiles stop and sits below the
      // cushion tops — the tiles occlude it wherever they are drawn, and past
      // them it is the ensemble's first-order statistics (its whole job).
      const innerR = hasCarpet
        ? Math.max(6, p.mossRadius * 0.62)
        : Math.max(p.widthCap * p.detailRadius * 0.6, 30)
      const fadeBand = hasCarpet ? Math.max(1.5, p.mossRadius * 0.12) : p.widthCap * p.detailRadius * 0.9
      drawCarpet = p.carpet && ctx.stand.radius * 1.45 > innerR
      if (drawCarpet && keyChanged(carpetKey, [p.detailRadius, ringCount, p.widthCap, p.mossRadius])) {
        const cascadeOuter = N0 * (1 << (ringCount - 1)) * CELL
        const outerR = Math.max(800, Math.min(3000, ctx.stand.radius * 1.5))
        carpetScratch.set(carpetStats.colors, 0)
        ctx.stand.species.forEach((e, i) => {
          if (i >= 4) return
          carpetScratch[16 + i * 4] = e.wetCenter ?? 0
          carpetScratch[17 + i * 4] = e.wetWidth ?? 0
          carpetScratch[18 + i * 4] = (e.carpetDiv ?? 0) > 0 ? 1 : 0
        })
        carpetU[32] = ctx.stand.species.length
        carpetU[33] = ctx.seed >>> 0
        carpetU[34] = hasCarpet ? 1 : 0
        carpetScratch[36] = innerR
        carpetScratch[37] = Math.max(outerR, cascadeOuter)
        carpetScratch[38] = carpetStats.height * 0.45
        carpetScratch[39] = ctx.stand.radius
        carpetScratch[40] = fadeBand
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
        pass.drawIndexed(entry.indexCount, entry.instances)
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
