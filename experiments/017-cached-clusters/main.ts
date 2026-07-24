import cardsSrc from './shaders/cards.wgsl'
import compositeSrc from './shaders/composite.wgsl'
import fillSrc from './shaders/fill.wgsl'
import {
  ATLAS_H as CARD_H,
  ATLAS_W as CARD_W,
  BAKE_VERSION,
  MIPS,
  bakeCards,
  buildMips,
  unpackCards,
  type CardsBaked,
} from './bake.ts'
import { DEBUG_VIEW_MODES, asU32, commitBake, hash3, hashF32, speciesById } from '@harness'
import type { DebugViewMode, Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Temporally amortized cluster cache.
 *
 * The world is partitioned (per frame, on CPU, over a bounded window) into a
 * distance-adaptive quadtree of world-anchored clusters: 8m tiles nearby,
 * doubling to 128m toward the horizon, so every cached cluster subtends
 * roughly the same angle (~1/4 rad). A cluster is RECONSTRUCTED only rarely:
 * a compute pass re-derives its plants from the scatter WGSL twin and renders
 * them as baked crossed-card proxies directly into its own 176x176 layer of a
 * shared color+depth cache array, through a frustum fitted from the camera
 * position at refresh time. Every frame, visible slots are drawn as single
 * world-anchored quads whose fragments re-project the cached 16-bit depth
 * through the slot's stored inverse view-proj into the live camera — correct
 * occlusion and parallax at cluster granularity for ~200 textured quads. A slot
 * is invalidated only when the camera has moved further than
 * refreshTau x cluster distance (or on quadtree level change / eviction), and
 * refreshes run under a fixed per-frame budget: that budget — never the plant
 * count — bounds the expensive work. Plants inside directRadius render
 * directly each frame as the same crossed cards with live per-plant wind;
 * cached clusters get an analytic coherent wind shear at composite time so
 * cached imagery never freezes.
 */

const CELL = 4 // scatter cell (m), must match SCATTER_CELL_SIZE
const S0 = 8 // level-0 cluster size (m) = 2x2 scatter cells
const TOP_LEVEL = 4 // cluster sizes 8..128m
const SPLIT_K = 4 // split when nearest-distance < SPLIT_K * size
const MAX_DIST = 1024
const SLOT = 176
const POOL = 224 // cache array layers (WebGPU guarantees >= 256)
const MASK_SIDE = 16 // direct bitmask, in 8m clusters
const NEAR_CAP = 65536
const SCRATCH_CAP = 245760
const MAX_REFRESH = 12
const FAR_SINGLE_LEVEL = 3 // levels >= this use one clamped camera-facing card
const RING_STRIDE = 256
const CELL_BUDGET = 6144

type V3 = [number, number, number]

// ---- minimal column-major mat4 helpers (slot frustum only) -----------------

function matMul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16)
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[k * 4 + r]! * b[c * 4 + k]!
      out[c * 4 + r] = s
    }
  }
  return out
}

/** Standard adjugate-based 4x4 inverse (column-major). */
function matInvert(a: Float32Array): Float32Array {
  const inv = new Float32Array(16)
  inv[0] = a[5]! * a[10]! * a[15]! - a[5]! * a[11]! * a[14]! - a[9]! * a[6]! * a[15]! + a[9]! * a[7]! * a[14]! + a[13]! * a[6]! * a[11]! - a[13]! * a[7]! * a[10]!
  inv[4] = -a[4]! * a[10]! * a[15]! + a[4]! * a[11]! * a[14]! + a[8]! * a[6]! * a[15]! - a[8]! * a[7]! * a[14]! - a[12]! * a[6]! * a[11]! + a[12]! * a[7]! * a[10]!
  inv[8] = a[4]! * a[9]! * a[15]! - a[4]! * a[11]! * a[13]! - a[8]! * a[5]! * a[15]! + a[8]! * a[7]! * a[13]! + a[12]! * a[5]! * a[11]! - a[12]! * a[7]! * a[9]!
  inv[12] = -a[4]! * a[9]! * a[14]! + a[4]! * a[10]! * a[13]! + a[8]! * a[5]! * a[14]! - a[8]! * a[6]! * a[13]! - a[12]! * a[5]! * a[10]! + a[12]! * a[6]! * a[9]!
  inv[1] = -a[1]! * a[10]! * a[15]! + a[1]! * a[11]! * a[14]! + a[9]! * a[2]! * a[15]! - a[9]! * a[3]! * a[14]! - a[13]! * a[2]! * a[11]! + a[13]! * a[3]! * a[10]!
  inv[5] = a[0]! * a[10]! * a[15]! - a[0]! * a[11]! * a[14]! - a[8]! * a[2]! * a[15]! + a[8]! * a[3]! * a[14]! + a[12]! * a[2]! * a[11]! - a[12]! * a[3]! * a[10]!
  inv[9] = -a[0]! * a[9]! * a[15]! + a[0]! * a[11]! * a[13]! + a[8]! * a[1]! * a[15]! - a[8]! * a[3]! * a[13]! - a[12]! * a[1]! * a[11]! + a[12]! * a[3]! * a[9]!
  inv[13] = a[0]! * a[9]! * a[14]! - a[0]! * a[10]! * a[13]! - a[8]! * a[1]! * a[14]! + a[8]! * a[2]! * a[13]! + a[12]! * a[1]! * a[10]! - a[12]! * a[2]! * a[9]!
  inv[2] = a[1]! * a[6]! * a[15]! - a[1]! * a[7]! * a[14]! - a[5]! * a[2]! * a[15]! + a[5]! * a[3]! * a[14]! + a[13]! * a[2]! * a[7]! - a[13]! * a[3]! * a[6]!
  inv[6] = -a[0]! * a[6]! * a[15]! + a[0]! * a[7]! * a[14]! + a[4]! * a[2]! * a[15]! - a[4]! * a[3]! * a[14]! - a[12]! * a[2]! * a[7]! + a[12]! * a[3]! * a[6]!
  inv[10] = a[0]! * a[5]! * a[15]! - a[0]! * a[7]! * a[13]! - a[4]! * a[1]! * a[15]! + a[4]! * a[3]! * a[13]! + a[12]! * a[1]! * a[7]! - a[12]! * a[3]! * a[5]!
  inv[14] = -a[0]! * a[5]! * a[14]! + a[0]! * a[6]! * a[13]! + a[4]! * a[1]! * a[14]! - a[4]! * a[2]! * a[13]! - a[12]! * a[1]! * a[6]! + a[12]! * a[2]! * a[5]!
  inv[3] = -a[1]! * a[6]! * a[11]! + a[1]! * a[7]! * a[10]! + a[5]! * a[2]! * a[11]! - a[5]! * a[3]! * a[10]! - a[9]! * a[2]! * a[7]! + a[9]! * a[3]! * a[6]!
  inv[7] = a[0]! * a[6]! * a[11]! - a[0]! * a[7]! * a[10]! - a[4]! * a[2]! * a[11]! + a[4]! * a[3]! * a[10]! + a[8]! * a[2]! * a[7]! - a[8]! * a[3]! * a[6]!
  inv[11] = -a[0]! * a[5]! * a[11]! + a[0]! * a[7]! * a[9]! + a[4]! * a[1]! * a[11]! - a[4]! * a[3]! * a[9]! - a[8]! * a[1]! * a[7]! + a[8]! * a[3]! * a[5]!
  inv[15] = a[0]! * a[5]! * a[10]! - a[0]! * a[6]! * a[9]! - a[4]! * a[1]! * a[10]! + a[4]! * a[2]! * a[9]! + a[8]! * a[1]! * a[6]! - a[8]! * a[2]! * a[5]!
  let det = a[0]! * inv[0]! + a[1]! * inv[4]! + a[2]! * inv[8]! + a[3]! * inv[12]!
  if (Math.abs(det) < 1e-20) det = 1e-20
  const d = 1 / det
  for (let i = 0; i < 16; i++) inv[i] = inv[i]! * d
  return inv
}

const norm3 = (v: V3): V3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / l, v[1] / l, v[2] / l]
}
const cross3 = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const dot3 = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

interface Aabb {
  minX: number
  minY: number
  minZ: number
  maxX: number
  maxY: number
  maxZ: number
}

/**
 * Quadtree node identity as a plain number (no per-frame string building in
 * the enumeration hot loop). 21 bits per axis, collision-free for any cluster
 * coordinate this method can produce.
 */
const nodeKey = (level: number, cx: number, cz: number): number =>
  (level * 2097152 + (cx & 0x1fffff)) * 2097152 + (cz & 0x1fffff)

interface Leaf {
  key: number
  level: number
  cx: number
  cz: number
  box: Aabb
  dist: number
  solid: number
  sticky: boolean
}

interface Slot {
  key: number | null
  level: number
  cx: number
  cz: number
  camX: number
  camY: number
  camZ: number
  lastVisible: number
  valid: boolean
  /** Debug-view generation this slot's imagery was reconstructed under. */
  epoch: number
  meta: Float32Array // 32 floats, matches SlotMeta in composite.wgsl
  vp: Float32Array
  pxAngle: number
}

interface RefreshJob {
  slotIdx: number
  cellX0: number
  cellZ0: number
  sideCells: number
  level: number
  base: number
  cap: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  if (ctx.stand.species.length > 4) {
    console.warn(`${ctx.id}: stand has ${ctx.stand.species.length} entries; only the first 4 are rendered`)
  }
  const entries = ctx.stand.species.slice(0, 4)
  const entryCount = entries.length

  // ---- baked per-species card proxies -------------------------------------
  const uniqueSpecies = [...new Set(entries.map((e) => e.species))]
  const baked = new Map<string, CardsBaked>()
  for (const speciesId of uniqueSpecies) {
    const key = `${speciesId}-cards-v${BAKE_VERSION}`
    let b: CardsBaked | null = null
    // The dev server answers missing baked files with index.html — validate
    // magic/size ourselves rather than trusting the fetch.
    const committed = await fetch(`/mesh/baked/${ctx.id}/${key}.bin`).catch(() => null)
    if (committed?.ok) b = unpackCards(await committed.arrayBuffer())
    if (!b) {
      const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
      const packed = await bakeCards(ctx, mesh)
      b = unpackCards(packed)
      if (!b) throw new Error(`${ctx.id}: fresh bake for ${speciesId} failed to unpack`)
      commitBake(ctx.id, key, packed).catch(() => undefined) // best effort
    }
    baked.set(speciesId, b)
  }

  const mkCardArray = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [CARD_W, CARD_H, uniqueSpecies.length],
        format: 'rgba8unorm',
        mipLevelCount: MIPS,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { tag: label },
    )
  const cardAlbedo = mkCardArray('card-albedo')
  const cardNormal = mkCardArray('card-normal')
  uniqueSpecies.forEach((speciesId, layer) => {
    const b = baked.get(speciesId)!
    const uploads: [GPUTexture, Uint8Array][] = [
      [cardAlbedo, b.albedo],
      [cardNormal, b.normal],
    ]
    for (const [tex, basePixels] of uploads) {
      buildMips(basePixels).forEach((mip, level) => {
        device.queue.writeTexture(
          { texture: tex, mipLevel: level, origin: [0, 0, layer] },
          mip.data as Uint8Array<ArrayBuffer>,
          { bytesPerRow: mip.w * 4, rowsPerImage: mip.h },
          [mip.w, mip.h],
        )
      })
    }
  })

  // Per-entry plant table (mesh center/extents + species atlas layer).
  const plantTable = ctx.res.createBuffer(
    { label: `${ctx.id}/plant-table`, size: 4 * 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'plant-table' },
  )
  {
    const data = new Float32Array(4 * 12)
    entries.forEach((e, i) => {
      const m = baked.get(e.species)!.meta
      const layer = uniqueSpecies.indexOf(e.species)
      data.set([...m.center, layer, m.sideHalfW, m.sideHalfH, m.topHalfW, m.topHalfH, m.yMin, m.yMax, 0, 0], i * 12)
    })
    device.queue.writeBuffer(plantTable, 0, data)
  }

  const totalDensity = entries.reduce((a, e) => a + Math.min(e.density, 8), 0)
  const maxPlantH = Math.max(...entries.map((e) => speciesById(e.species).heightScale * e.scaleMax)) * 1.1 + 0.6
  const swayAvg = entries.reduce((a, e) => a + e.sway * Math.min(e.density, 8), 0) / Math.max(totalDensity, 1e-4)
  const levelCap = Array.from({ length: TOP_LEVEL + 1 }, (_, l) =>
    Math.min(Math.ceil((S0 << l) * (S0 << l) * totalDensity * 1.25) + 256, SCRATCH_CAP),
  )

  // ---- GPU resources ------------------------------------------------------
  const nearInstances = ctx.res.createBuffer(
    { label: `${ctx.id}/near-instances`, size: NEAR_CAP * 32, usage: GPUBufferUsage.STORAGE },
    { tag: 'instances' },
  )
  const refreshInstances = ctx.res.createBuffer(
    { label: `${ctx.id}/refresh-instances`, size: SCRATCH_CAP * 32, usage: GPUBufferUsage.STORAGE },
    { tag: 'instances' },
  )
  const mkArgs = (label: string): GPUBuffer => {
    const buf = ctx.res.createBuffer(
      {
        label: `${ctx.id}/${label}`,
        size: 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { tag: 'args' },
    )
    device.queue.writeBuffer(buf, 0, new Uint32Array([18, 0, 0, 0]))
    return buf
  }
  const nearArgs = mkArgs('near-args')
  const refreshArgs = Array.from({ length: MAX_REFRESH }, (_, i) => mkArgs(`refresh-args-${i}`))

  const fillRing = ctx.res.createBuffer(
    {
      label: `${ctx.id}/fill-ring`,
      size: RING_STRIDE * (MAX_REFRESH + 1),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'uniforms' },
  )
  const cardsRing = ctx.res.createBuffer(
    {
      label: `${ctx.id}/cards-ring`,
      size: RING_STRIDE * (MAX_REFRESH + 1),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'uniforms' },
  )
  const slotMetaBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/slot-meta`, size: POOL * 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'slot-meta' },
  )

  // The cache is an ARRAY of slots (one layer each), not an atlas grid: a
  // refresh renders straight into its layer, so there is no scratch target and
  // no per-refresh texture copy. Same bytes, one less round trip.
  const mkCacheArray = (label: string, format: GPUTextureFormat): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [SLOT, SLOT, POOL],
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      },
      { tag: 'cluster-cache' },
    )
  const cacheColor = mkCacheArray('cache-color', 'rgba8unorm')
  const cacheAux = mkCacheArray('cache-aux', 'rg8unorm')
  const slotView = (tex: GPUTexture, label: string, layer: number): GPUTextureView =>
    tex.createView({ label: `${ctx.id}/${label}-${layer}`, dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 })
  const cacheColorSlots = Array.from({ length: POOL }, (_, i) => slotView(cacheColor, 'cache-color', i))
  const cacheAuxSlots = Array.from({ length: POOL }, (_, i) => slotView(cacheAux, 'cache-aux', i))
  // One scratch depth buffer shared by every refresh (cleared per pass, never
  // read afterwards).
  const slotDepth = ctx.res.createTexture(
    {
      label: `${ctx.id}/slot-depth`,
      size: [SLOT, SLOT],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'cluster-cache' },
  )
  const slotDepthView = slotDepth.createView()

  const cardSampler = device.createSampler({
    label: `${ctx.id}/card-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })
  const cacheSampler = device.createSampler({
    label: `${ctx.id}/cache-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // ---- bind groups ---------------------------------------------------------
  const fillBgl = device.createBindGroupLayout({
    label: `${ctx.id}/fill-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 80 },
      },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const mkFillBg = (label: string, instances: GPUBuffer, args: GPUBuffer): GPUBindGroup =>
    device.createBindGroup({
      label: `${ctx.id}/${label}`,
      layout: fillBgl,
      entries: [
        { binding: 0, resource: { buffer: fillRing, size: 80 } },
        { binding: 1, resource: { buffer: instances } },
        { binding: 2, resource: { buffer: args } },
      ],
    })
  const nearFillBg = mkFillBg('near-fill-bg', nearInstances, nearArgs)
  const refreshFillBgs = refreshArgs.map((args, i) => mkFillBg(`refresh-fill-bg-${i}`, refreshInstances, args))

  const cardsBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cards-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const mkCardsBg = (label: string, instances: GPUBuffer): GPUBindGroup =>
    device.createBindGroup({
      label: `${ctx.id}/${label}`,
      layout: cardsBgl,
      entries: [
        { binding: 0, resource: { buffer: cardsRing, size: 96 } },
        { binding: 1, resource: { buffer: instances } },
        { binding: 2, resource: { buffer: plantTable } },
        { binding: 3, resource: cardAlbedo.createView({ dimension: '2d-array' }) },
        { binding: 4, resource: cardNormal.createView({ dimension: '2d-array' }) },
        { binding: 5, resource: cardSampler },
      ],
    })
  const nearCardsBg = mkCardsBg('near-cards-bg', nearInstances)
  const refreshCardsBg = mkCardsBg('refresh-cards-bg', refreshInstances)

  const compBgl = device.createBindGroupLayout({
    label: `${ctx.id}/comp-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const compBg = device.createBindGroup({
    label: `${ctx.id}/comp-bg`,
    layout: compBgl,
    entries: [
      { binding: 0, resource: { buffer: slotMetaBuf } },
      { binding: 1, resource: cacheColor.createView({ dimension: '2d-array' }) },
      { binding: 2, resource: cacheAux.createView({ dimension: '2d-array' }) },
      { binding: 3, resource: cacheSampler },
    ],
  })

  // ---- pipelines -----------------------------------------------------------
  let fillPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let refreshPipeline!: GPURenderPipeline
  let compPipeline!: GPURenderPipeline
  const build = (): void => {
    fillPipeline = device.createComputePipeline({
      label: `${ctx.id}/fill`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/fill-pl`, bindGroupLayouts: [ctx.frame.layout, fillBgl] }),
      compute: { module: ctx.shaders.module(fillSrc), entryPoint: 'cs_fill' },
    })
    const cardsModule = ctx.shaders.module(cardsSrc)
    const cardsLayout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, cardsBgl],
    })
    nearPipeline = device.createRenderPipeline({
      label: `${ctx.id}/near`,
      layout: cardsLayout,
      vertex: { module: cardsModule, entryPoint: 'vs_near' },
      fragment: { module: cardsModule, entryPoint: 'fs_near', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    refreshPipeline = device.createRenderPipeline({
      label: `${ctx.id}/refresh`,
      layout: cardsLayout,
      vertex: { module: cardsModule, entryPoint: 'vs_refresh' },
      fragment: {
        module: cardsModule,
        entryPoint: 'fs_refresh',
        targets: [{ format: 'rgba8unorm' }, { format: 'rg8unorm' }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthCompare: 'less', depthWriteEnabled: true },
    })
    const compModule = ctx.shaders.module(compositeSrc)
    compPipeline = device.createRenderPipeline({
      label: `${ctx.id}/composite`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/comp-pl`, bindGroupLayouts: [ctx.frame.layout, compBgl] }),
      vertex: { module: compModule, entryPoint: 'vs_main' },
      fragment: { module: compModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // ---- cache state ---------------------------------------------------------
  const slots: Slot[] = Array.from({ length: POOL }, () => ({
    key: null,
    level: 0,
    cx: 0,
    cz: 0,
    camX: 0,
    camY: 0,
    camZ: 0,
    lastVisible: -1,
    valid: false,
    epoch: -1,
    meta: new Float32Array(32),
    vp: new Float32Array(16),
    pxAngle: 0.01,
  }))
  const byKey = new Map<number, number>()
  const yRangeCache = new Map<number, [number, number]>()
  const standR = ctx.stand.radius

  /**
   * The cache stores SHADED imagery, so a debug view has to be baked into it
   * at refresh time (see shaders/cards.wgsl). The harness exposes the selector
   * only through the frame uniform, so read it off the URL and bump a
   * generation counter when it changes — every cached slot then re-reconstructs
   * under the normal refresh budget while its stale image keeps drawing.
   */
  let lastHash = ''
  let debugMode = 0
  let bakedDebug = 0
  let debugEpoch = 0
  const readDebugMode = (): number => {
    const h = location.hash
    if (h === lastHash) return debugMode
    lastHash = h
    const q = h.indexOf('?')
    const raw = q < 0 ? null : new URLSearchParams(h.slice(q + 1)).get('debug')
    const idx = raw === null ? 0 : DEBUG_VIEW_MODES.indexOf(raw as DebugViewMode)
    debugMode = idx < 0 ? 0 : idx
    return debugMode
  }

  const ringBytes = new ArrayBuffer(RING_STRIDE * (MAX_REFRESH + 1))
  const ringF = new Float32Array(ringBytes)
  const ringI = new Int32Array(ringBytes)
  const ringU = new Uint32Array(ringBytes)
  const cardsData = new Float32Array((RING_STRIDE / 4) * (MAX_REFRESH + 1))
  const metaData = new Float32Array(POOL * 32)

  let refreshJobs: RefreshJob[] = []
  let drawCount = 0
  let nearDispatchX = 0
  let nearDispatchZ = 0
  const refreshVertexCount = refreshArgs.map(() => 18)
  const vcScratch = new Uint32Array(1)

  const clusterYRange = (level: number, cx: number, cz: number): [number, number] => {
    const key = nodeKey(level, cx, cz)
    let r = yRangeCache.get(key)
    if (r) return r
    const s = S0 << level
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i <= 2; i++) {
      for (let j = 0; j <= 2; j++) {
        const h = ctx.scene.terrain.height(cx * s + (i * s) / 2, cz * s + (j * s) / 2)
        lo = Math.min(lo, h)
        hi = Math.max(hi, h)
      }
    }
    r = [lo - 0.5, hi + maxPlantH]
    yRangeCache.set(key, r)
    return r
  }

  const aabbDist = (cam: V3, b: Aabb): number => {
    const dx = Math.max(b.minX - cam[0], 0, cam[0] - b.maxX)
    const dy = Math.max(b.minY - cam[1], 0, cam[1] - b.maxY)
    const dz = Math.max(b.minZ - cam[2], 0, cam[2] - b.maxZ)
    return Math.hypot(dx, dy, dz)
  }

  /** Fit an off-axis frustum from `cam` through the cluster box; write slot meta. */
  const fitSlot = (slot: Slot, slotIdx: number, cam: V3, box: Aabb): void => {
    const center: V3 = [(box.minX + box.maxX) / 2, (box.minY + box.maxY) / 2, (box.minZ + box.maxZ) / 2]
    const fwd = norm3([center[0] - cam[0], center[1] - cam[1], center[2] - cam[2]])
    let right = cross3([0, 1, 0], fwd)
    if (Math.hypot(right[0], right[1], right[2]) < 1e-4) right = cross3([0, 0, 1], fwd)
    right = norm3(right)
    const up = norm3(cross3(fwd, right))

    let l = Infinity
    let r = -Infinity
    let b = Infinity
    let t = -Infinity
    let zMin = Infinity
    let zMax = -Infinity
    for (let i = 0; i < 8; i++) {
      const d: V3 = [
        (i & 1 ? box.maxX : box.minX) - cam[0],
        (i & 2 ? box.maxY : box.minY) - cam[1],
        (i & 4 ? box.maxZ : box.minZ) - cam[2],
      ]
      const z = Math.max(dot3(d, fwd), 0.05)
      zMin = Math.min(zMin, z)
      zMax = Math.max(zMax, z)
      l = Math.min(l, dot3(d, right) / z)
      r = Math.max(r, dot3(d, right) / z)
      b = Math.min(b, dot3(d, up) / z)
      t = Math.max(t, dot3(d, up) / z)
    }
    const near = Math.max(0.05, zMin * 0.98)
    const far = zMax * 1.02 + 0.5

    const view = new Float32Array(16)
    view[0] = right[0]
    view[4] = right[1]
    view[8] = right[2]
    view[12] = -dot3(right, cam)
    view[1] = up[0]
    view[5] = up[1]
    view[9] = up[2]
    view[13] = -dot3(up, cam)
    view[2] = fwd[0]
    view[6] = fwd[1]
    view[10] = fwd[2]
    view[14] = -dot3(fwd, cam)
    view[15] = 1
    const proj = new Float32Array(16)
    proj[0] = 2 / (r - l)
    proj[8] = -(r + l) / (r - l)
    proj[5] = 2 / (t - b)
    proj[9] = -(b + t) / (t - b)
    proj[10] = far / (far - near)
    proj[14] = (-far * near) / (far - near)
    proj[11] = 1
    const vp = matMul(proj, view)
    slot.vp.set(vp)
    slot.pxAngle = (t - b) / SLOT

    const zc = dot3([center[0] - cam[0], center[1] - cam[1], center[2] - cam[2]], fwd)
    const halfR = ((r - l) / 2) * zc
    const halfU = ((t - b) / 2) * zc
    const cxT = (l + r) / 2
    const cyT = (b + t) / 2

    const m = slot.meta
    m[0] = cam[0] + (fwd[0] + right[0] * cxT + up[0] * cyT) * zc
    m[1] = cam[1] + (fwd[1] + right[1] * cxT + up[1] * cyT) * zc
    m[2] = cam[2] + (fwd[2] + right[2] * cxT + up[2] * cyT) * zc
    m[3] = swayAvg
    m[4] = right[0] * halfR
    m[5] = right[1] * halfR
    m[6] = right[2] * halfR
    m[7] = hashF32(hash3(asU32(slot.cx), asU32(slot.cz), asU32(slot.level))) * 6.2831853
    m[8] = up[0] * halfU
    m[9] = up[1] * halfU
    m[10] = up[2] * halfU
    m[11] = 0
    m[12] = slotIdx // cache array layer
    m[13] = slot.level // quadtree level (cluster tint)
    m[14] = 0
    m[15] = 0 // tint mode — rewritten per frame
    m.set(matInvert(vp), 16)
  }

  return {
    update(frame: FrameInfo): void {
      const cam: V3 = [frame.camera.pose.x, frame.camera.pose.y, frame.camera.pose.z]
      const frameIdx = frame.frameIndex
      const tau = ctx.params.refreshTau
      const directR = Math.min(ctx.params.directRadius, 48)
      const dbg = readDebugMode()
      if (dbg !== bakedDebug) {
        bakedDebug = dbg
        debugEpoch++
      }

      // Live-camera frustum planes (left/right/bottom/top, Gribb-Hartmann).
      const vpM = frame.camera.viewProj as Float32Array
      const rowV = (i: number): [number, number, number, number] => [vpM[i]!, vpM[4 + i]!, vpM[8 + i]!, vpM[12 + i]!]
      const r0 = rowV(0)
      const r1 = rowV(1)
      const r3 = rowV(3)
      const planes = [
        [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]],
        [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]],
        [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]],
        [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]],
      ].map((p) => {
        const il = 1 / (Math.hypot(p[0]!, p[1]!, p[2]!) || 1)
        return [p[0]! * il, p[1]! * il, p[2]! * il, p[3]! * il] as [number, number, number, number]
      })
      const boxVisible = (b: Aabb, margin: number): boolean => {
        for (const p of planes) {
          const px = p[0] > 0 ? b.maxX : b.minX
          const py = p[1] > 0 ? b.maxY : b.minY
          const pz = p[2] > 0 ? b.maxZ : b.minZ
          if (p[0] * px + p[1] * py + p[2] * pz + p[3] + margin < 0) return false
        }
        return true
      }

      // ---- enumerate visible leaves (quadtree by distance, bounded window) --
      const leaves: Leaf[] = []
      const direct: [number, number][] = []
      const visit = (level: number, cx: number, cz: number): void => {
        const s = S0 << level
        const minX = cx * s
        const minZ = cz * s
        if (minX > standR || minX + s < -standR || minZ > standR || minZ + s < -standR) return
        const [y0, y1] = clusterYRange(level, cx, cz)
        const box: Aabb = { minX, minY: y0, minZ, maxX: minX + s, maxY: y1, maxZ: minZ + s }
        const dist = aabbDist(cam, box)
        if (dist >= MAX_DIST) return
        const isNearby = dist < directR + 12
        if (!isNearby && !boxVisible(box, 10)) return
        if (level === 0) {
          if (dist < directR) {
            direct.push([cx, cz])
          } else {
            leaves.push({
              key: nodeKey(0, cx, cz),
              level,
              cx,
              cz,
              box,
              dist,
              solid: (s / Math.max(dist, 1)) ** 2,
              sticky: dist < directR + 12,
            })
          }
        } else if (dist < SPLIT_K * s) {
          visit(level - 1, cx * 2, cz * 2)
          visit(level - 1, cx * 2 + 1, cz * 2)
          visit(level - 1, cx * 2, cz * 2 + 1)
          visit(level - 1, cx * 2 + 1, cz * 2 + 1)
        } else {
          leaves.push({
            key: nodeKey(level, cx, cz),
            level,
            cx,
            cz,
            box,
            dist,
            solid: (s / Math.max(dist, 1)) ** 2,
            sticky: false,
          })
        }
      }
      {
        const sTop = S0 << TOP_LEVEL
        const extent = Math.min(MAX_DIST, standR + sTop)
        const c0x = Math.floor((cam[0] - extent) / sTop)
        const c1x = Math.floor((cam[0] + extent) / sTop)
        const c0z = Math.floor((cam[2] - extent) / sTop)
        const c1z = Math.floor((cam[2] + extent) / sTop)
        for (let z = c0z; z <= c1z; z++) for (let x = c0x; x <= c1x; x++) visit(TOP_LEVEL, x, z)
      }

      // ---- classify: draw cached / schedule refresh ------------------------
      const drawSet = new Set<number>()
      const scheduled: { leaf: Leaf; priority: number }[] = []
      for (const leaf of leaves) {
        const si = byKey.get(leaf.key)
        if (si !== undefined && slots[si]!.valid) {
          const slot = slots[si]!
          slot.lastVisible = frameIdx
          drawSet.add(si)
          const err = Math.hypot(cam[0] - slot.camX, cam[1] - slot.camY, cam[2] - slot.camZ) / Math.max(leaf.dist, 1)
          // Stale on parallax error, or because the debug view changed under it.
          const wrongView = slot.epoch !== debugEpoch
          if (err > tau || wrongView) scheduled.push({ leaf, priority: (wrongView ? 4 : err) * leaf.solid })
        } else {
          scheduled.push({ leaf, priority: leaf.solid * 4 })
          // While waiting for the slot: fall back to the parent's stale card.
          const pi = byKey.get(nodeKey(leaf.level + 1, leaf.cx >> 1, leaf.cz >> 1))
          if (pi !== undefined && slots[pi]!.valid) {
            drawSet.add(pi)
            slots[pi]!.lastVisible = frameIdx
          }
          // L0 clusters just past the direct ring stay direct until cached.
          if (leaf.sticky) direct.push([leaf.cx, leaf.cz])
        }
      }

      // ---- pick refresh jobs under budget ----------------------------------
      refreshJobs = []
      if (!ctx.params.freezeCache) {
        scheduled.sort((a, b) => b.priority - a.priority)
        const budget = Math.min(
          MAX_REFRESH,
          scheduled.length > 24 ? Math.max(ctx.params.refreshBudget, 10) : ctx.params.refreshBudget,
        )
        let cells = 0
        let base = 0
        for (const cand of scheduled) {
          if (refreshJobs.length >= budget) break
          const leaf = cand.leaf
          const jobCells = (1 << (leaf.level + 1)) ** 2
          if (refreshJobs.length > 0 && cells + jobCells > CELL_BUDGET) continue
          const cap = levelCap[leaf.level]!
          if (base + cap > SCRATCH_CAP) continue

          let si = byKey.get(leaf.key)
          if (si === undefined) {
            // Acquire: free slot, else evict the least-recently-visible one.
            let bestIdx = -1
            let bestScore = Infinity
            for (let i = 0; i < POOL; i++) {
              const s = slots[i]!
              const score = s.key === null ? -1 : s.lastVisible === frameIdx ? Infinity : s.lastVisible
              if (score < bestScore) {
                bestScore = score
                bestIdx = i
                if (score === -1) break
              }
            }
            if (bestIdx < 0 || bestScore === Infinity) continue // pool exhausted
            si = bestIdx
            const victim = slots[si]!
            if (victim.key !== null) byKey.delete(victim.key)
            byKey.set(leaf.key, si)
            victim.key = leaf.key
            victim.level = leaf.level
            victim.cx = leaf.cx
            victim.cz = leaf.cz
            victim.valid = false
          }
          const slot = slots[si]!
          fitSlot(slot, si, cam, leaf.box)
          slot.camX = cam[0]
          slot.camY = cam[1]
          slot.camZ = cam[2]
          slot.valid = true
          slot.epoch = debugEpoch
          slot.lastVisible = frameIdx
          if (!leaf.sticky) drawSet.add(si)
          const s = S0 << leaf.level
          refreshJobs.push({
            slotIdx: si,
            cellX0: (leaf.cx * s) / CELL,
            cellZ0: (leaf.cz * s) / CELL,
            sideCells: s / CELL,
            level: leaf.level,
            base,
            cap,
          })
          cells += jobCells
          base += cap
        }
      }

      // ---- uniform rings ----------------------------------------------------
      const { height: vh } = ctx.size()
      const screenPxAngle = (2 * Math.tan(((frame.camera.pose.fov * Math.PI) / 180) / 2)) / Math.max(vh, 1)

      ringU.fill(0)
      cardsData.fill(0)
      {
        const camCx = Math.floor(cam[0] / S0)
        const camCz = Math.floor(cam[2] / S0)
        const ox = camCx - MASK_SIDE / 2
        const oz = camCz - MASK_SIDE / 2
        let bx0 = Infinity
        let bx1 = -Infinity
        let bz0 = Infinity
        let bz1 = -Infinity
        for (const [dcx, dcz] of direct) {
          const lx = dcx - ox
          const lz = dcz - oz
          if (lx < 0 || lz < 0 || lx >= MASK_SIDE || lz >= MASK_SIDE) continue
          const bit = lz * MASK_SIDE + lx
          ringU[12 + (bit >> 5)] = ringU[12 + (bit >> 5)]! | (1 << (bit & 31))
          bx0 = Math.min(bx0, dcx)
          bx1 = Math.max(bx1, dcx)
          bz0 = Math.min(bz0, dcz)
          bz1 = Math.max(bz1, dcz)
        }
        // Dispatch over the direct clusters that actually exist, not over the
        // whole 128m bitmask window (directRadius is 34m by default).
        const hasDirect = bx1 >= bx0
        nearDispatchX = hasDirect ? (bx1 - bx0 + 1) * 2 : 0
        nearDispatchZ = hasDirect ? (bz1 - bz0 + 1) * 2 : 0
        ringI[0] = (hasDirect ? bx0 : ox) * 2
        ringI[1] = (hasDirect ? bz0 : oz) * 2
        ringU[2] = nearDispatchX
        ringU[3] = asU32(ctx.seed)
        ringI[4] = ox // bitmask origin stays camera-anchored
        ringI[5] = oz
        ringU[6] = MASK_SIDE
        ringU[7] = 1 // near mode
        ringF[8] = standR
        ringU[9] = NEAR_CAP
        ringU[10] = 0
        cardsData[20] = screenPxAngle
      }
      refreshJobs.forEach((job, j) => {
        const o = ((j + 1) * RING_STRIDE) / 4
        ringI[o] = job.cellX0
        ringI[o + 1] = job.cellZ0
        ringU[o + 2] = job.sideCells
        ringU[o + 3] = asU32(ctx.seed)
        ringU[o + 6] = MASK_SIDE
        ringU[o + 7] = 0 // refresh mode
        ringF[o + 8] = standR
        ringU[o + 9] = job.cap
        ringU[o + 10] = job.base

        const slot = slots[job.slotIdx]!
        const far = job.level >= FAR_SINGLE_LEVEL
        cardsData.set(slot.vp, o)
        cardsData[o + 16] = slot.camX
        cardsData[o + 17] = slot.camY
        cardsData[o + 18] = slot.camZ
        cardsData[o + 19] = far ? 2 : 1
        cardsData[o + 20] = slot.pxAngle
        cardsData[o + 21] = 1.7 // min card px in the slot
        cardsData[o + 22] = job.base

        // Far clusters draw ONE card; the other two quads used to be emitted
        // and immediately clipped — 3x the vertex work on the biggest jobs.
        const vc = far ? 6 : 18
        if (refreshVertexCount[j] !== vc) {
          refreshVertexCount[j] = vc
          vcScratch[0] = vc
          device.queue.writeBuffer(refreshArgs[j]!, 0, vcScratch)
        }
      })
      device.queue.writeBuffer(fillRing, 0, ringBytes, 0, RING_STRIDE * (refreshJobs.length + 1))
      device.queue.writeBuffer(cardsRing, 0, cardsData.buffer, 0, RING_STRIDE * (refreshJobs.length + 1))

      // ---- composite meta ---------------------------------------------------
      const tintMode = ctx.params.clusterTint === 'level' ? 1 : ctx.params.clusterTint === 'slot' ? 2 : 0
      drawCount = 0
      for (const si of drawSet) {
        if (drawCount >= POOL) break
        metaData.set(slots[si]!.meta, drawCount * 32)
        metaData[drawCount * 32 + 15] = tintMode
        drawCount++
      }
      if (drawCount > 0) device.queue.writeBuffer(slotMetaBuf, 0, metaData.buffer, 0, drawCount * 128)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // Reset instance counters inside this encoder (multi-view safe).
      enc.clearBuffer(nearArgs, 4, 4)
      refreshJobs.forEach((_, j) => enc.clearBuffer(refreshArgs[j]!, 4, 4))

      // One compute pass: near fill + every refresh fill (disjoint ranges of
      // the shared scratch, so no inter-dispatch hazards).
      const cpass = ctx.timing.computePass(enc, 'cc-fill')
      cpass.setPipeline(fillPipeline)
      cpass.setBindGroup(0, ctx.frame.bindGroup)
      cpass.setBindGroup(1, nearFillBg, [0])
      if (nearDispatchX > 0) cpass.dispatchWorkgroups(nearDispatchX, nearDispatchZ, entryCount)
      refreshJobs.forEach((job, j) => {
        cpass.setBindGroup(1, refreshFillBgs[j]!, [(j + 1) * RING_STRIDE])
        cpass.dispatchWorkgroups(job.sideCells, job.sideCells, entryCount)
      })
      cpass.end()

      // Amortized reconstruction: each refreshed cluster renders straight into
      // its own cache layer (no scratch target, no blit).
      refreshJobs.forEach((job, j) => {
        const pass = ctx.timing.renderPass(enc, 'cc-refresh', {
          colorAttachments: [
            {
              view: cacheColorSlots[job.slotIdx]!,
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: 'clear',
              storeOp: 'store',
            },
            {
              view: cacheAuxSlots[job.slotIdx]!,
              clearValue: { r: 1, g: 1, b: 0, a: 0 },
              loadOp: 'clear',
              storeOp: 'store',
            },
          ],
          depthStencilAttachment: {
            view: slotDepthView,
            depthClearValue: 1,
            depthLoadOp: 'clear',
            // Scratch depth — never read after the pass.
            depthStoreOp: 'discard',
          },
        })
        pass.setPipeline(refreshPipeline)
        pass.setBindGroup(0, ctx.frame.bindGroup)
        pass.setBindGroup(1, refreshCardsBg, [(j + 1) * RING_STRIDE])
        pass.drawIndirect(refreshArgs[j]!, 0)
        pass.end()
      })

      // Main pass: direct near cards + cached cluster composite.
      const pass = ctx.timing.renderPass(enc, 'cc-main', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setPipeline(nearPipeline)
      pass.setBindGroup(1, nearCardsBg, [0])
      pass.drawIndirect(nearArgs, 0)
      if (drawCount > 0) {
        pass.setPipeline(compPipeline)
        pass.setBindGroup(1, compBg)
        pass.draw(6, drawCount)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are destroyed by the harness via ctx.res.
    },
  }
}
