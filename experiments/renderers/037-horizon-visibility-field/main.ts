import canopySrc from './shaders/canopy.wgsl'
import fieldSrc from './shaders/field.wgsl'
import { loadEntryTable, N_PHI, N_THETA, RES, TILE_NORM, SIN_MIN, type EntrySpec, type RayTable } from './bake.ts'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * HORIZON VISIBILITY FIELD — a canopy with no canopy geometry.
 *
 * What is rasterized: ONE ground-conformal polar hull, RINGS x SECTORS quads,
 * constant for every stand (49k triangles at 560k plants and at 134M). There is
 * no per-plant, per-clump or per-blade primitive anywhere in the frame.
 *
 * What makes the grass appear: each pixel asks "what does my eye ray meet?" and
 * a baked table answers in a constant number of fetches — the hull fragment
 * gives the exact local ground plane, the scatter-built canopy-hull field gives
 * the local canopy height / species / wind phase, and the 4D visibility table
 * gives where that ray is first blocked, with depth, normal, coverage, albedo and
 * occlusion. Nothing is marched at runtime.
 *
 * Stand fidelity: the field is stamped from ctx.scene.scatter's exact plants
 * (position, scale, species entry, wind phase), so where the canopy is tall,
 * short, bare or which species owns a patch is the stand's answer, not a
 * pattern. Sub-25cm blade structure comes from the periodic baked tile.
 */

const FIELD_RES = 1024
const FIELD_TEXEL = 0.25
const FIELD_SNAP = 32
const H_RANGE = 4 // metres covered by the field's 12-bit height
const RINGS = 128
const SECTORS = 192
/** Iso-clearance shells of the carrier — one instanced draw, fixed count. */
const LEVELS = 6
/** De-tiling warp amplitude, in tile periods. */
const WARP_AMP = 0.5
const MAX_TABLES = 4
const CELL = 4

interface Table {
  spec: EntrySpec
  data: RayTable
  a: GPUTexture
  b: GPUTexture
  maxLod: number
  /** Mean canopy colour + AO over all directions (the coarsest baked level). */
  mean: number[]
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // ---- one ray-answer table per stand entry (species + its density/scales) ---
  // The tile is populated at the stand's TOTAL density: every column carries all
  // entries' plants, so baking each species at its own density alone would leave
  // the canopy transparent. Species identity stays per-column from the scatter.
  const totalDensity = ctx.stand.species.reduce((a, e) => a + e.density, 0)
  const specs: EntrySpec[] = ctx.stand.species.map((e) => ({
    speciesId: e.species,
    density: totalDensity,
    scaleMin: e.scaleMin,
    scaleMax: e.scaleMax,
    heightScale: speciesById(e.species).heightScale,
  }))
  const tables: Table[] = []
  for (const spec of specs.slice(0, MAX_TABLES)) {
    const data = await loadEntryTable(ctx, spec)
    const sizes: number[] = []
    for (let r = data.res; r >= 1; r = r >> 1) {
      sizes.push(r)
      if (r === 1) break
    }
    const layers = data.nPhi * data.nTheta
    const a = ctx.res.createTexture(
      {
        label: `${ctx.id}/rayA/${spec.speciesId}`,
        size: [data.res, data.res, layers],
        mipLevelCount: data.a.length,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: spec.speciesId, tag: 'ray-table' },
    )
    data.a.forEach((bytes, level) => {
      const r = sizes[level]!
      device.queue.writeTexture(
        { texture: a, mipLevel: level },
        bytes,
        { bytesPerRow: r * 4, rowsPerImage: r },
        [r, r, layers],
      )
    })
    const bRes = Math.max(1, data.res >> 1)
    const b = ctx.res.createTexture(
      {
        label: `${ctx.id}/rayB/${spec.speciesId}`,
        size: [bRes, bRes, layers],
        mipLevelCount: data.b.length,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: spec.speciesId, tag: 'ray-shade' },
    )
    data.b.forEach((bytes, level) => {
      const r = sizes[level + 1]!
      device.queue.writeTexture(
        { texture: b, mipLevel: level },
        bytes,
        { bytesPerRow: r * 4, rowsPerImage: r },
        [r, r, layers],
      )
    })
    // The coarsest B level is one texel per direction: its average over all
    // directions is the species' mean canopy colour, used to converge the
    // near-horizontal answer instead of letting it marble.
    const last = data.b[data.b.length - 1]!
    const mean = [0, 0, 0, 0]
    for (let i = 0; i < layers; i++) for (let c = 0; c < 4; c++) mean[c]! += last[i * 4 + c]! / 255 / layers
    tables.push({ spec, data, a, b, maxLod: data.a.length - 1, mean })
  }
  if (tables.length === 0) throw new Error(`[${ctx.id}] stand has no species entries`)
  // One mean canopy colour for the whole stand, not per species: the far-field
  // converged answer is shared by every entry, and a per-species mean would paint
  // the species mosaic across the horizon band as huge magnified blocks (a pixel
  // near the horizon covers metres of the canopy top).
  const standMean = [0, 1, 2, 3].map((c) => tables.reduce((a, t) => a + t.mean[c]!, 0) / tables.length)
  for (const t of tables) t.mean = standMean

  // ---- the canopy-hull field ------------------------------------------------
  const packBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/field-pack`,
      size: FIELD_RES * FIELD_RES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { tag: 'canopy-field' },
  )
  const fieldTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/field`,
      size: [FIELD_RES, FIELD_RES],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    },
    { tag: 'canopy-field' },
  )
  const fieldUniform = ctx.res.createBuffer(
    { label: `${ctx.id}/field-params`, size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )
  const canopyUniform = ctx.res.createBuffer(
    { label: `${ctx.id}/canopy-params`, size: 224, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )

  const fieldBgl = device.createBindGroupLayout({
    label: `${ctx.id}/field-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const fieldOutBgl = device.createBindGroupLayout({
    label: `${ctx.id}/field-out-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float' },
      },
    ],
  })
  const fieldBg = device.createBindGroup({
    label: `${ctx.id}/field-bg`,
    layout: fieldBgl,
    entries: [
      { binding: 0, resource: { buffer: fieldUniform } },
      { binding: 1, resource: { buffer: packBuffer } },
    ],
  })
  const fieldOutBg = device.createBindGroup({
    label: `${ctx.id}/field-out-bg`,
    layout: fieldOutBgl,
    entries: [{ binding: 0, resource: fieldTex.createView() }],
  })

  const canopyBgl = device.createBindGroupLayout({
    label: `${ctx.id}/canopy-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ...[3, 4, 5, 6, 7, 8, 9, 10].map((binding) => ({
        binding,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' as GPUTextureSampleType, viewDimension: '2d-array' as GPUTextureViewDimension },
      })),
      { binding: 11, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  })

  const tabSampler = device.createSampler({
    label: `${ctx.id}/tab-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })
  const fieldSampler = device.createSampler({
    label: `${ctx.id}/field-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })
  const pick = (i: number): Table => tables[Math.min(i, tables.length - 1)]!
  const canopyBg = device.createBindGroup({
    label: `${ctx.id}/canopy-bg`,
    layout: canopyBgl,
    entries: [
      { binding: 0, resource: { buffer: canopyUniform } },
      { binding: 1, resource: fieldTex.createView() },
      { binding: 2, resource: tabSampler },
      { binding: 3, resource: pick(0).a.createView({ dimension: '2d-array' }) },
      { binding: 4, resource: pick(1).a.createView({ dimension: '2d-array' }) },
      { binding: 5, resource: pick(2).a.createView({ dimension: '2d-array' }) },
      { binding: 6, resource: pick(3).a.createView({ dimension: '2d-array' }) },
      { binding: 7, resource: pick(0).b.createView({ dimension: '2d-array' }) },
      { binding: 8, resource: pick(1).b.createView({ dimension: '2d-array' }) },
      { binding: 9, resource: pick(2).b.createView({ dimension: '2d-array' }) },
      { binding: 10, resource: pick(3).b.createView({ dimension: '2d-array' }) },
      { binding: 11, resource: fieldSampler },
    ],
  })

  let splatPipeline!: GPUComputePipeline
  let resolvePipeline!: GPUComputePipeline
  let canopyPipeline!: GPURenderPipeline
  const build = (): void => {
    const fieldModule = ctx.shaders.module(fieldSrc)
    const fieldLayout = device.createPipelineLayout({
      label: `${ctx.id}/field-pl`,
      bindGroupLayouts: [ctx.frame.layout, fieldBgl, fieldOutBgl],
    })
    splatPipeline = device.createComputePipeline({
      label: `${ctx.id}/field-splat`,
      layout: fieldLayout,
      compute: { module: fieldModule, entryPoint: 'cs_splat' },
    })
    resolvePipeline = device.createComputePipeline({
      label: `${ctx.id}/field-resolve`,
      layout: fieldLayout,
      compute: { module: fieldModule, entryPoint: 'cs_resolve' },
    })
    const canopyModule = ctx.shaders.module(canopySrc)
    const canopyLayout = device.createPipelineLayout({
      label: `${ctx.id}/canopy-pl`,
      bindGroupLayouts: [ctx.frame.layout, canopyBgl],
    })
    const common = {
      layout: canopyLayout,
      primitive: { topology: 'triangle-list' as GPUPrimitiveTopology, cullMode: 'none' as GPUCullMode },
      depthStencil: {
        format: ctx.depthFormat,
        depthCompare: 'less' as GPUCompareFunction,
        depthWriteEnabled: true,
      },
    }
    canopyPipeline = device.createRenderPipeline({
      ...common,
      label: `${ctx.id}/canopy-hull`,
      vertex: { module: canopyModule, entryPoint: 'vs_lid' },
      fragment: { module: canopyModule, entryPoint: 'fs_hull', targets: [{ format: ctx.colorFormat }] },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // ---- field bookkeeping ----------------------------------------------------
  const maxHmax = Math.max(...tables.map((t) => t.data.hmax))
  // Top shell = the tallest possible plant; the rest step down evenly, so some
  // shell always sits within HULL_WINDOW of the local canopy top.
  const lidY = maxHmax
  const standCellMin = Math.floor(-ctx.stand.radius / CELL)
  const standCellMax = Math.floor(ctx.stand.radius / CELL)
  const fieldSpan = FIELD_RES * FIELD_TEXEL
  const fieldF32 = new Float32Array(44)
  const fieldU32 = new Uint32Array(fieldF32.buffer)
  const canopyF32 = new Float32Array(56)
  const canopyU32 = new Uint32Array(canopyF32.buffer)

  let activeLevels = 1
  let originX = Number.NaN
  let originZ = Number.NaN
  let dirtyField = true
  let cellsX = 0
  let cellsZ = 0
  let splatGroups: [number, number] = [0, 0]

  const updateField = (camX: number, camZ: number): void => {
    const cx = Math.round(camX / FIELD_SNAP) * FIELD_SNAP
    const cz = Math.round(camZ / FIELD_SNAP) * FIELD_SNAP
    const ox = cx - fieldSpan / 2
    const oz = cz - fieldSpan / 2
    if (ox === originX && oz === originZ) return
    originX = ox
    originZ = oz
    dirtyField = true

    // Splat only the plants that can reach the field, clamped to the stand.
    const margin = Math.max(...tables.map((t) => t.data.radius * t.spec.scaleMax)) + 0.5
    const c0x = Math.max(standCellMin, Math.floor((ox - margin) / CELL))
    const c0z = Math.max(standCellMin, Math.floor((oz - margin) / CELL))
    const c1x = Math.min(standCellMax, Math.floor((ox + fieldSpan + margin) / CELL))
    const c1z = Math.min(standCellMax, Math.floor((oz + fieldSpan + margin) / CELL))
    cellsX = Math.max(0, c1x - c0x + 1)
    cellsZ = Math.max(0, c1z - c0z + 1)

    fieldF32[0] = ox
    fieldF32[1] = oz
    fieldF32[2] = FIELD_TEXEL
    fieldF32[3] = FIELD_RES
    fieldU32[4] = c0x | 0
    fieldU32[5] = c0z | 0
    fieldU32[6] = cellsX
    fieldU32[7] = cellsZ
    fieldU32[8] = ctx.stand.species.length
    fieldU32[9] = ctx.seed >>> 0
    fieldF32[10] = 4095 / H_RANGE
    fieldF32[11] = H_RANGE / 4095
    ctx.stand.species.forEach((entry, i) => {
      if (i >= 8) return
      const t = tables[Math.min(i, tables.length - 1)]!
      fieldF32[12 + i * 4] = t.data.radius
      fieldF32[13 + i * 4] = t.data.hmax
      fieldF32[14 + i * 4] = 0.55
    })
    device.queue.writeBuffer(fieldUniform, 0, fieldF32)

    const threads = cellsX * cellsZ * ctx.stand.species.length * 128
    const wgTotal = Math.ceil(threads / 64)
    const wgX = Math.min(Math.max(wgTotal, 1), 65535)
    splatGroups = [wgX, Math.max(1, Math.ceil(wgTotal / wgX))]
  }

  const taps = (): number => Number(ctx.params.dirTaps)

  return {
    update(frame: FrameInfo): void {
      updateField(frame.camera.pose.x, frame.camera.pose.z)
      const flags =
        (ctx.params.groundContact ? 1 : 0) | (ctx.params.windShear ? 2 : 0) | (ctx.params.refineEntry ? 4 : 0)
      canopyF32[0] = originX
      canopyF32[1] = originZ
      canopyF32[2] = FIELD_TEXEL
      canopyF32[3] = FIELD_RES
      canopyF32[4] = lidY
      canopyF32[5] = Math.min(ctx.params.regionRadius, fieldSpan / 2 - 4)
      canopyF32[6] = ctx.params.alphaRef
      canopyF32[7] = ctx.params.lodBias
      canopyF32[8] = RINGS
      canopyF32[9] = SECTORS
      canopyF32[10] = 0.05
      canopyF32[11] = Math.min(ctx.params.regionRadius, fieldSpan / 2 - 4)
      canopyF32[12] = TILE_NORM
      canopyF32[13] = RES
      canopyF32[14] = SIN_MIN
      canopyF32[15] = N_PHI
      canopyF32[16] = N_THETA
      canopyU32[17] = taps()
      canopyU32[18] = flags
      // Shell stack: a shell above the eye can never be crossed, so when the
      // camera is clear of the canopy ONE shell answers everything; at canopy
      // height the stack fills the range below the eye.
      const clearance = frame.camera.pose.y - ctx.scene.terrain.height(frame.camera.pose.x, frame.camera.pose.z)
      let levels = 1
      let shellLo = maxHmax
      let shellHi = maxHmax
      let window = maxHmax
      if (clearance < maxHmax + 0.15) {
        shellHi = Math.max(0.12 * maxHmax, Math.min(maxHmax, clearance - 0.04))
        shellLo = 0.1 * maxHmax
        if (shellHi > shellLo + 0.06) {
          levels = LEVELS
          window = Math.max(0.3, ((shellHi - shellLo) / (LEVELS - 1)) * 1.75)
        } else {
          shellLo = shellHi
        }
      }
      activeLevels = levels
      canopyF32[19] = levels
      canopyF32[20] = window
      canopyF32[21] = shellLo
      canopyF32[22] = shellHi
      canopyF32[23] = WARP_AMP
      for (let i = 0; i < MAX_TABLES; i++) {
        const t = pick(i)
        const entry = ctx.stand.species[Math.min(i, ctx.stand.species.length - 1)]!
        canopyF32[24 + i * 8] = t.data.hmax
        canopyF32[25 + i * 8] = entry.sway
        canopyF32[26 + i * 8] = t.data.tileNorm * t.data.hmax
        canopyF32[27 + i * 8] = t.maxLod
        canopyF32[28 + i * 8] = t.mean[0]!
        canopyF32[29 + i * 8] = t.mean[1]!
        canopyF32[30 + i * 8] = t.mean[2]!
        canopyF32[31 + i * 8] = t.mean[3]!
      }
      device.queue.writeBuffer(canopyUniform, 0, canopyF32)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      if (dirtyField && cellsX > 0 && cellsZ > 0) {
        dirtyField = false
        enc.clearBuffer(packBuffer)
        const pass = ctx.timing.computePass(enc, 'field-build')
        pass.setPipeline(splatPipeline)
        pass.setBindGroup(0, ctx.frame.bindGroup)
        pass.setBindGroup(1, fieldBg)
        pass.setBindGroup(2, fieldOutBg)
        pass.dispatchWorkgroups(splatGroups[0], splatGroups[1])
        pass.setPipeline(resolvePipeline)
        pass.dispatchWorkgroups(Math.ceil(FIELD_RES / 8), Math.ceil(FIELD_RES / 8))
        pass.end()
      }

      const pass = ctx.timing.renderPass(enc, 'canopy', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(canopyPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, canopyBg)
      // One instanced draw = LEVELS iso-clearance shells. Depth-min between them
      // picks the true nearest hit; each shell only answers where its height
      // brackets the local canopy top.
      pass.draw(RINGS * SECTORS * 6, activeLevels)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
