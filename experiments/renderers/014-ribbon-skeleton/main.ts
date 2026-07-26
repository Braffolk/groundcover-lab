import cellsSrc from './shaders/cells.wgsl'
import expandSrc from './shaders/expand.wgsl'
import ribbonsSrc from './shaders/ribbons.wgsl'
import {
  ATLAS_ROWS,
  LEVEL_BASE,
  ROWS_PER_VARIANT,
  SAMPLES,
  VARIANTS,
  loadRibbonAtlas,
  type RibbonAtlas,
} from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
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
 * Ribbon skeleton — a plant is a handful of curved ribbons whose shape lives
 * in a 12 x 360 rgba16f texture (69 KB/species) and is unrolled into triangle
 * strips in the vertex shader. No source geometry, no impostor images, no
 * raymarching, nothing camera-facing.
 *
 * Per frame, three passes whose cost depends on the visible region, never on
 * the stand's plant count:
 *   1. cells   — one thread per scatter cell of the camera region rect;
 *                region-circle + frustum reject, survivors compacted, and the
 *                counter IS the indirect dispatch size of pass 2.
 *   2. expand  — indirect, one workgroup per visible cell, one thread per
 *                scatter slot; surviving plants are classified by projected
 *                screen extent into 5 LOD buckets and compacted, with the
 *                bucket's instanceCount written straight into its draw args.
 *   3. ribbons — 5 drawIndexedIndirect per stand entry (32/16/8/4/4 ribbons at
 *                12/8/6/4/2 samples). Each ribbon morphs into its parent
 *                ribbon at the next coarser level as it shrinks; the morph
 *                completes exactly where the next bucket takes over, so LOD
 *                transitions are geometrically continuous.
 */

const CELL = SCATTER_CELL_SIZE
/** Keep equal to the manifest's regionRadius max. */
const REGION_MAX = 128
const BUCKETS = 5

interface LodBucket {
  /** Ribbons drawn. */
  ribbons: number
  /** Samples drawn along each ribbon. */
  samples: number
  /** Atlas level supplying the rows. */
  level: number
  /** Level the ribbons morph into as the plant shrinks (`morph` only). */
  parent: number
  /** Whether this bucket morphs at all — false where `parent` == `level`. */
  morph: boolean
}

/** Tufts: wedges halve with distance and each pair converges on its parent. */
const UPRIGHT_BUCKETS: readonly LodBucket[] = [
  { ribbons: 32, samples: 12, level: 0, parent: 1, morph: true },
  { ribbons: 16, samples: 8, level: 1, parent: 2, morph: true },
  { ribbons: 8, samples: 6, level: 2, parent: 3, morph: true },
  { ribbons: 4, samples: 4, level: 3, parent: 3, morph: false },
  { ribbons: 4, samples: 2, level: 3, parent: 3, morph: false },
]
/**
 * Mats: a fan of N wedges covers the tile SQUARE, and the coarser the fan the
 * more it has to overshoot into its neighbours to stay closed (a 4-wedge fan
 * cannot tile a square at all). So a carpet keeps 16 wedges out to mid range and
 * never drops below 8, spending its LOD on radial samples instead. Buckets that
 * only drop samples keep the same level and do not morph.
 */
const MAT_BUCKETS: readonly LodBucket[] = [
  { ribbons: 32, samples: 12, level: 0, parent: 1, morph: true },
  { ribbons: 16, samples: 8, level: 1, parent: 1, morph: false },
  { ribbons: 16, samples: 5, level: 1, parent: 2, morph: true },
  { ribbons: 8, samples: 4, level: 2, parent: 2, morph: false },
  { ribbons: 8, samples: 2, level: 2, parent: 2, morph: false },
]
/** Projected plant extent (px) at which each bucket takes over, before `detail`. */
const PX_THRESHOLDS = [150, 70, 32, 20]
/** Ribbons morph into their parent over [threshold, threshold*SPAN] px. */
const MORPH_SPAN = 1.8
/** Share of an entry's instance budget per bucket (near .. far). */
const CAP_FRACTION = [0.03, 0.07, 0.16, 0.24, 0.5]
/**
 * A carpet's tiles are life size (0.18 m), so essentially all of them land in
 * the farthest bucket: the tuft split would overflow it and drop tiles, which
 * looks exactly like a placement bug. The near buckets still get several times
 * their average-density share, because `entryPerM2` divides the grid by the
 * entry's wetness share while the wetness field is 12 m noise — a camera sitting
 * inside one zone sees the FULL grid density for a few metres around it.
 */
const MAT_CAP_FRACTION = [0.012, 0.03, 0.06, 0.1, 0.9]
/** Fraction of the visible region disc assumed on screen at once. */
const VISIBLE_FRACTION = 0.45
const SHOW_MODES = ['off', 'lod', 'ribbon', 'variant', 'fluff']

/** Plants per m² this entry can produce — a carpet's grid, not its density. */
function entryPerM2(entry: StandSpecies): number {
  if (entry.carpetDiv && entry.carpetDiv > 0) {
    // Carpet entries partition the wetness axis, so an entry only claims about
    // `wetWidth` of its grid nodes.
    const share = entry.wetWidth && entry.wetWidth > 0 ? Math.min(1, entry.wetWidth) : 1
    return (standEntrySlots(entry) / (CELL * CELL)) * share
  }
  return Math.min(entry.density, SCATTER_MAX_DENSITY)
}

interface SpeciesGpu {
  geomTex: GPUTexture
  shadeTex: GPUTexture
  radiusFrac: number
  /** Radial-fan cushion atlas rather than a tuft of blades (see bake.ts). */
  mat: boolean
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  gpu: SpeciesGpu
  /** Scatter slots per cell for THIS entry — carpetDiv², not the 128 budget. */
  slots: number
  buckets: readonly LodBucket[]
  /** Index into `patterns` per bucket. */
  bucketPatterns: number[]
  /** vec4-unit base offset and instance capacity of each LOD bucket. */
  bases: number[]
  caps: number[]
  instances: GPUBuffer
  drawArgs: GPUBuffer
  expandInfo: GPUBuffer
  lodInfo: GPUBuffer
  expandBind: GPUBindGroup
  drawBinds: GPUBindGroup[]
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // --- ribbon atlases (sequential: the poa mesh is 240MB on the wire) -------
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadRibbonAtlas(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- shared index buffer: one strip pattern per distinct (ribbons, samples)
  // combination used by either bucket table -------------------------------
  const patterns: { ribbons: number; samples: number; first: number; count: number }[] = []
  const patternOf = (b: LodBucket): number => {
    const found = patterns.findIndex((p) => p.ribbons === b.ribbons && p.samples === b.samples)
    if (found >= 0) return found
    const first = patterns.length === 0 ? 0 : patterns[patterns.length - 1]!.first + patterns[patterns.length - 1]!.count
    patterns.push({ ribbons: b.ribbons, samples: b.samples, first, count: b.ribbons * (b.samples - 1) * 6 })
    return patterns.length - 1
  }
  const uprightPatterns = UPRIGHT_BUCKETS.map(patternOf)
  const matPatterns = MAT_BUCKETS.map(patternOf)
  const totalIndices = patterns[patterns.length - 1]!.first + patterns[patterns.length - 1]!.count
  const indexData = new Uint16Array(totalIndices)
  for (const p of patterns) {
    let w = p.first
    const stride = 2 * p.samples
    for (let r = 0; r < p.ribbons; r++) {
      for (let q = 0; q < p.samples - 1; q++) {
        const a = r * stride + q * 2
        indexData[w++] = a
        indexData[w++] = a + 1
        indexData[w++] = a + 2
        indexData[w++] = a + 1
        indexData[w++] = a + 3
        indexData[w++] = a + 2
      }
    }
  }
  const indexBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/strip-indices`,
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'strip-indices' },
  )
  device.queue.writeBuffer(indexBuffer, 0, indexData)

  // --- cell list sized for the largest region the params allow -------------
  const cellSideMax = Math.ceil((2 * REGION_MAX) / CELL) + 2
  const cellCapacity = cellSideMax * cellSideMax
  const cellList = ctx.res.createBuffer(
    { label: `${ctx.id}/cell-list`, size: cellCapacity * 8, usage: GPUBufferUsage.STORAGE },
    { tag: 'cell-list' },
  )
  const cellArgs = ctx.res.createBuffer(
    {
      label: `${ctx.id}/cell-args`,
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    },
    { tag: 'indirect-args' },
  )
  const cellUniform = ctx.res.createBuffer(
    { label: `${ctx.id}/cell-info`, size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'uniform' },
  )

  // --- layouts --------------------------------------------------------------
  const cellBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cells-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const expandBgl = device.createBindGroupLayout({
    label: `${ctx.id}/expand-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
    ],
  })

  const cellBind = device.createBindGroup({
    label: `${ctx.id}/cells-bg`,
    layout: cellBgl,
    entries: [
      { binding: 0, resource: { buffer: cellUniform } },
      { binding: 1, resource: { buffer: cellList } },
      { binding: 2, resource: { buffer: cellArgs } },
    ],
  })

  // --- per stand entry state ------------------------------------------------
  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    // A carpet has carpetDiv² slots per cell (484 for the bog moss, over the
    // 128 scatter budget), and the mat atlas wants the mat bucket table.
    const slots = standEntrySlots(standEntry)
    const isMat = gpu.mat
    const buckets = isMat ? MAT_BUCKETS : UPRIGHT_BUCKETS
    const bucketPatterns = isMat ? matPatterns : uprightPatterns
    const fraction = isMat ? MAT_CAP_FRACTION : CAP_FRACTION
    const budget = Math.ceil(Math.PI * REGION_MAX * REGION_MAX * entryPerM2(standEntry) * VISIBLE_FRACTION)
    const caps: number[] = []
    const bases: number[] = []
    let cursor = 0
    for (let b = 0; b < BUCKETS; b++) {
      // Multiples of 8 instances keep every region's byte offset 256-aligned.
      const cap = Math.max(8, Math.ceil((budget * fraction[b]!) / 8) * 8)
      caps.push(cap)
      bases.push(cursor)
      cursor += cap * 2
    }
    const instances = ctx.res.createBuffer(
      {
        label: `${ctx.id}/instances-${entryIndex}`,
        size: cursor * 16,
        usage: GPUBufferUsage.STORAGE,
      },
      { species: standEntry.species, tag: 'lod-instances' },
    )
    const drawArgs = ctx.res.createBuffer(
      {
        label: `${ctx.id}/draw-args-${entryIndex}`,
        size: BUCKETS * 20,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const expandInfo = ctx.res.createBuffer(
      {
        label: `${ctx.id}/expand-info-${entryIndex}`,
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'uniform' },
    )
    const lodInfo = ctx.res.createBuffer(
      {
        label: `${ctx.id}/lod-info-${entryIndex}`,
        size: BUCKETS * 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'uniform' },
    )
    const expandBind = device.createBindGroup({
      label: `${ctx.id}/expand-bg-${entryIndex}`,
      layout: expandBgl,
      entries: [
        { binding: 0, resource: { buffer: expandInfo } },
        { binding: 1, resource: { buffer: cellList } },
        { binding: 2, resource: { buffer: instances } },
        { binding: 3, resource: { buffer: drawArgs } },
      ],
    })
    const drawBinds = buckets.map((_, b) =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}-${b}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: lodInfo, offset: b * 256, size: 64 } },
          { binding: 1, resource: { buffer: instances, offset: bases[b]! * 16, size: caps[b]! * 32 } },
          { binding: 2, resource: gpu.geomTex.createView() },
          { binding: 3, resource: gpu.shadeTex.createView() },
        ],
      }),
    )
    return {
      entryIndex,
      speciesId: standEntry.species,
      gpu,
      slots,
      buckets,
      bucketPatterns,
      bases,
      caps,
      instances,
      drawArgs,
      expandInfo,
      lodInfo,
      expandBind,
      drawBinds,
    }
  })

  // --- pipelines ------------------------------------------------------------
  let cellsPipeline!: GPUComputePipeline
  let expandPipeline!: GPUComputePipeline
  let ribbonsPipeline!: GPURenderPipeline
  const build = (): void => {
    cellsPipeline = device.createComputePipeline({
      label: `${ctx.id}/cells`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cells-pl`,
        bindGroupLayouts: [ctx.frame.layout, cellBgl],
      }),
      compute: { module: ctx.shaders.module(cellsSrc), entryPoint: 'cs_cells' },
    })
    expandPipeline = device.createComputePipeline({
      label: `${ctx.id}/expand`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/expand-pl`,
        bindGroupLayouts: [ctx.frame.layout, expandBgl],
      }),
      compute: { module: ctx.shaders.module(expandSrc), entryPoint: 'cs_expand' },
    })
    const module = ctx.shaders.module(ribbonsSrc)
    ribbonsPipeline = device.createRenderPipeline({
      label: `${ctx.id}/ribbons`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/ribbons-pl`,
        bindGroupLayouts: [ctx.frame.layout, drawBgl],
      }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      // Ribbons are two-sided surfaces, opaque and depth-writing: hard edges,
      // real occluders, no stochastic alpha anywhere.
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- per-frame scratch ----------------------------------------------------
  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const maxPlantHeight = Math.max(
    ...ctx.stand.species.map((e) => speciesById(e.species).heightScale * e.scaleMax),
  )
  const cellData = new Float32Array(32)
  const expandData = new Float32Array(32)
  const lodData = new Float32Array(BUCKETS * 64)
  /** Per entry: drawIndexedIndirect args with this entry's strip patterns. */
  const argsReset = new Map<number, Uint32Array<ArrayBuffer>>()
  for (const entry of entries) {
    const args = new Uint32Array(BUCKETS * 5)
    entry.bucketPatterns.forEach((p, b) => {
      args[b * 5] = patterns[p]!.count
      args[b * 5 + 2] = patterns[p]!.first
    })
    argsReset.set(entry.entryIndex, args)
  }
  const cellArgsReset = new Uint32Array([0, 1, 1, 0])
  let cellCount = 0
  let lodDirty = true

  const writeLodInfo = (): void => {
    const showMode = Math.max(0, SHOW_MODES.indexOf(ctx.params.show))
    for (const entry of entries) {
      entry.buckets.forEach((b, i) => {
        const o = i * 64
        lodData[o] = b.ribbons
        lodData[o + 1] = b.samples
        lodData[o + 2] = LEVEL_BASE[b.level]!
        lodData[o + 3] = LEVEL_BASE[b.parent]!
        lodData[o + 4] = ROWS_PER_VARIANT
        lodData[o + 5] = SAMPLES
        lodData[o + 6] = entry.caps[i]!
        lodData[o + 7] = ctx.params.widthScale
        lodData[o + 8] = ctx.params.bladeCurl
        lodData[o + 9] = ctx.params.flutter
        lodData[o + 10] = showMode
        lodData[o + 11] = entry.entryIndex
        lodData[o + 12] = i
        lodData[o + 13] = ctx.params.canopyLift
      })
      device.queue.writeBuffer(entry.lodInfo, 0, lodData)
    }
  }

  return {
    update(frame: FrameInfo): void {
      const region = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      const { height } = ctx.size()
      // px per metre of world extent at 1 m: proj[1][1] * viewportHeight / 2.
      const projScale = (frame.camera.proj[5] as number) * height * 0.5

      // Cell rect, clamped to the stand's own cell range: cells outside the
      // stand hold nothing and must not be dispatched.
      const x0 = Math.max(cellMin, Math.floor((cam.x - region) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - region) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + region) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + region) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      cellCount = sideX * sideZ

      frustumPlanes(frame.camera.viewProj as Float32Array, cellData)
      cellData[24] = x0
      cellData[25] = z0
      cellData[26] = sideX
      cellData[27] = sideZ
      cellData[28] = region
      cellData[29] = maxPlantHeight
      cellData[30] = cellCapacity
      device.queue.writeBuffer(cellUniform, 0, cellData)
      device.queue.writeBuffer(cellArgs, 0, cellArgsReset)

      const detail = Math.max(0.05, ctx.params.detail)
      for (const entry of entries) {
        expandData[0] = entry.entryIndex
        expandData[1] = ctx.seed
        expandData[2] = region
        expandData[3] = projScale
        expandData[4] = entry.gpu.radiusFrac
        expandData[5] = region * 0.76
        expandData[6] = 0.1
        expandData[7] = 0.5
        for (let i = 0; i < 4; i++) expandData[8 + i] = PX_THRESHOLDS[i]! / detail
        for (let i = 0; i < 4; i++) expandData[12 + i] = entry.caps[i]!
        expandData[16] = entry.caps[4]!
        expandData[17] = MORPH_SPAN
        expandData[18] = cellCapacity
        expandData[19] = VARIANTS
        for (let i = 0; i < 4; i++) expandData[20 + i] = entry.bases[i]!
        expandData[24] = entry.bases[4]!
        // Slots to EVALUATE per cell (carpetDiv² for a carpet — every slot must
        // be visited even though capacity only holds the expected survivors).
        expandData[28] = entry.slots
        // Morph gate: a bucket whose parent level equals its own must not morph
        // (its "parent row" would be a sibling ribbon, not a coarser one).
        for (let i = 0; i < 3; i++) expandData[29 + i] = entry.buckets[i]!.morph ? 1 : 0
        device.queue.writeBuffer(entry.expandInfo, 0, expandData)
        device.queue.writeBuffer(entry.drawArgs, 0, argsReset.get(entry.entryIndex)!)
      }

      if (lodDirty) {
        writeLodInfo()
        lodDirty = false
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      if (cellCount === 0) return // camera outside the stand entirely

      const cells = ctx.timing.computePass(enc, 'cells')
      cells.setPipeline(cellsPipeline)
      cells.setBindGroup(0, ctx.frame.bindGroup)
      cells.setBindGroup(1, cellBind)
      cells.dispatchWorkgroups(Math.ceil(cellCount / 64))
      cells.end()

      const expand = ctx.timing.computePass(enc, 'expand')
      expand.setPipeline(expandPipeline)
      expand.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        expand.setBindGroup(1, entry.expandBind)
        expand.dispatchWorkgroupsIndirect(cellArgs, 0)
      }
      expand.end()

      const pass = ctx.timing.renderPass(enc, 'ribbons', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(ribbonsPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setIndexBuffer(indexBuffer, 'uint16')
      for (const entry of entries) {
        for (let b = 0; b < BUCKETS; b++) {
          pass.setBindGroup(1, entry.drawBinds[b]!)
          pass.drawIndexedIndirect(entry.drawArgs, b * 20)
        }
      }
      pass.end()
    },

    onParamsChanged(): void {
      lodDirty = true
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are reclaimed by the harness via ctx.res.
    },
  }
}

/** Upload the distilled atlas as two tiny rgba16float planes. */
function uploadAtlas(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  atlas: RibbonAtlas,
): SpeciesGpu {
  const mk = (label: string, data: Float32Array): GPUTexture => {
    const tex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${label}`,
        size: [SAMPLES, ATLAS_ROWS],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag: `ribbon-${label}` },
    )
    const half = new Uint16Array(data.length)
    for (let i = 0; i < data.length; i++) half[i] = f32ToF16(data[i]!)
    ctx.device.queue.writeTexture(
      { texture: tex },
      half,
      { bytesPerRow: SAMPLES * 8, rowsPerImage: ATLAS_ROWS },
      [SAMPLES, ATLAS_ROWS],
    )
    return tex
  }
  return {
    geomTex: mk('geom', atlas.geom),
    shadeTex: mk('shade', atlas.shade),
    radiusFrac: atlas.radiusFrac,
    mat: atlas.mat,
  }
}

const f32Scratch = new Float32Array(1)
const u32Scratch = new Uint32Array(f32Scratch.buffer)

/** IEEE binary32 -> binary16 bits, round-to-nearest (values here are tame). */
function f32ToF16(value: number): number {
  f32Scratch[0] = value
  const bits = u32Scratch[0]!
  const sign = (bits >>> 16) & 0x8000
  const exp = (bits >>> 23) & 0xff
  let mant = bits & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant !== 0 ? 0x200 : 0)
  const e = exp - 127 + 15
  if (e >= 0x1f) return sign | 0x7c00
  if (e <= 0) {
    if (e < -10) return sign
    mant = (mant | 0x800000) >>> (1 - e)
    return sign | ((mant + 0x1000) >>> 13)
  }
  return sign | (e << 10) | ((mant + 0x1000) >>> 13)
}

/** Gribb-Hartmann frustum planes from a column-major view-proj matrix. */
function frustumPlanes(m: Float32Array, out: Float32Array): void {
  const row = (r: number): [number, number, number, number] => [m[r]!, m[4 + r]!, m[8 + r]!, m[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const planes: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]],
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]],
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]],
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]],
    [r2[0], r2[1], r2[2], r2[3]],
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]],
  ]
  planes.forEach((p, i) => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1
    out[i * 4] = p[0] / len
    out[i * 4 + 1] = p[1] / len
    out[i * 4 + 2] = p[2] / len
    out[i * 4 + 3] = p[3] / len
  })
}
