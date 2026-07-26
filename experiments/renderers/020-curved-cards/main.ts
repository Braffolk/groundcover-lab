import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/curved.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, DISP, SURF, loadSpeciesRelief, type ReliefAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  speciesById,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Curved cards. Startup: each species' relief atlas (8 side views + 1 top view
 * — albedo/coverage, oct normals + front-surface distance + baked sky
 * occlusion, and a hole-free smoothed displacement field) is loaded from
 * mesh/baked / OPFS or baked in-browser from the raw mesh, uploaded and
 * mipped.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into
 * three distance RINGS of one instance buffer, each with its own indirect
 * draw. A ring is a LOD: ring 0 draws a 5x7 curved patch + a 4x4 canopy patch
 * (66 triangles), ring 1 a 3x5 + 3x3 (24 triangles), ring 2 a single flat quad
 * + flat top card (4 triangles — the billboard baseline's exact geometry).
 * Rings are drawn near-to-far so early-z rejects the deep overdraw.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const LOD0_MAX = 24 // keep equal to the manifest's lod0Dist max
const LOD1_MAX = 70 // keep equal to the manifest's lod1Dist max
const MIP_A = Math.floor(Math.log2(ATLAS)) + 1
const MIP_S = Math.floor(Math.log2(SURF)) + 1
const TOP_FRAC = 0.52
const INFO_FLOATS = 64 // 256 B — one dynamic-offset block per LOD
const INFO_STRIDE = INFO_FLOATS * 4
const DRAW_ARG_U32 = 5 // drawIndexedIndirect: indexCount, instanceCount, firstIndex, baseVertex, firstInstance
/**
 * Coverage-weighted means of the baked occlusion / front-distance channels
 * (measured over the committed artifacts, side tiles only). They turn the two
 * occlusion terms into mean-preserving contrast: see the shade_gain write.
 */
const AO_FIELD_MEAN = 0.45
const DIST_FIELD_MEAN = 0.51

/** side lattice (columns x rows), canopy lattice (columns x rows) per ring. */
const LODS: { side: [number, number]; top: [number, number]; curve: number }[] = [
  { side: [5, 7], top: [4, 4], curve: 1 },
  { side: [3, 5], top: [3, 3], curve: 1 },
  { side: [2, 2], top: [2, 2], curve: 0 }, // a 2x2 lattice can only bend at its rim — keep it flat
]

/**
 * A CARPET entry draws one ground-parallel lattice per tile instead of a side
 * patch + canopy patch, so it gets its own ring table. The tile is 0.18m and
 * the baked height field carries ~3cm features, so ring 0's 7x7 (3cm cells) is
 * about the finest lattice that still resolves something real; ring 1 keeps a
 * single centre bump; ring 2 is a flat quad, which is exactly the billboard
 * baseline's geometry.
 */
const CARPET_LODS: { grid: [number, number]; curve: number }[] = [
  { grid: [9, 9], curve: 1 },
  { grid: [3, 3], curve: 1 },
  { grid: [2, 2], curve: 0 },
]
/**
 * Resolution of a mat's own displacement field over its tile square. 16 texels
 * across 0.18m is 1.1cm, blurred once — roughly what a 9x9 near lattice
 * (2.2cm cells) can carry without aliasing, and the whole texture is 256 B.
 */
const CARPET_FIELD = 16
/**
 * A mat's albedo / surface textures are the TILE SQUARE ONLY, resampled out of
 * the top view. The source has ~271 albedo texels across the 0.18m tile, so 320
 * is mild upsampling (0.56mm/texel, no detail lost) and 160 matches the surface
 * atlas's own 136. Both together are 0.44 MB against the 15.7 MB of the full
 * 9-tile atlas set: the 8 side views describe nothing a mat ever shows, and
 * that saving is exactly what pays for a life-size carpet's instance buffer.
 */
const CARPET_ALBEDO_PX = 320
const CARPET_SURF_PX = 160
/** Carpet ring radii (m per unit scale). A 0.18m tile is ~6px at 14m. */
const CARPET_LOD_DIST: [number, number] = [4, 14]
/**
 * Alpha reference for carpet tiles, INSTEAD of the params' alphaRef. A mat is
 * a closed surface (the baked top view is 92-97% covered over the tile square)
 * and must not dissolve with distance, so the reference sits far below the
 * mip-averaged coverage while the genuinely empty texels still open.
 */
const CARPET_ALPHA_REF = 0.06
/**
 * Fraction of a carpet's grid nodes one wetness zone can claim locally. The
 * three Sphagnum entries partition the wetness axis into thirds, but wetness
 * is NOT uniformly distributed — it is damped on slopes — so the middle zone
 * reaches 0.575 of the nodes inside a 120m box on the bog stand (measured
 * against the scatter's own wetness field). Sizing the instance buffer at the
 * nominal 1/3 would silently clamp ~40% of that zone's tiles away.
 */
const CARPET_BAND_HEADROOM = 2.0

interface SpeciesGpu {
  atlas: ReliefAtlas
  albedoTex: GPUTexture
  surfTex: GPUTexture
  dispTex: GPUTexture
  /** Mat species: only the top tile is uploaded, and these describe it. */
  carpet: boolean
  /** Mean canopy height over the tile square, as a fraction of [y0, y1]. */
  canopyMean: number
  /** Coverage-gated mean of the baked height channel — the occlusion gain. */
  heightMean: number
  /** Decode of the carpet displacement texture: hfrac = base + texel * span. */
  dispBase: number
  dispSpan: number
}

interface LatticeRange {
  firstIndex: number
  indexCount: number
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  isCarpet: boolean
  bases: [number, number, number]
  caps: [number, number, number]
  /** Scatter slots per cell to ENUMERATE — carpetDiv², not the scatter budget. */
  enumSlots: number
  lattices: LatticeRange[]
  indirectReset: Uint32Array<ArrayBuffer>
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroup: GPUBindGroup
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

  // A species is a MAT only if every stand entry that uses it says so; then
  // its 8 side views are never sampled and are not worth uploading.
  const carpetOnly = new Set<string>()
  for (const entry of ctx.stand.species) {
    const carpet = (entry.carpetDiv ?? 0) > 0
    if (carpet && !ctx.stand.species.some((e) => e.species === entry.species && (e.carpetDiv ?? 0) <= 0)) {
      carpetOnly.add(entry.species)
    }
  }

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) --
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesRelief(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas, carpetOnly.has(entry.species)))
  }

  // --- static lattice topology ---------------------------------------------
  const { indices, plantLattices, carpetLattices } = buildLattices()
  const indexBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/lattice-indices`,
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'lattice-indices' },
  )
  device.queue.writeBuffer(indexBuffer, 0, indices)

  // --- bind group layouts / pipelines ---------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: INFO_STRIDE } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INFO_STRIDE },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // Cells the stand actually holds — a region disc bigger than the stand
  // cannot produce more candidates, and on the bog that saves a third of the
  // (large) carpet instance buffer.
  const standCells = (Math.floor(ctx.stand.radius / CELL) * 2 + 1) ** 2

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const carpetDiv = standEntry.carpetDiv ?? 0
    const isCarpet = carpetDiv > 0
    // A carpet entry's scaleMin/scaleMax in the stand are placeholders — the
    // harness computes the real (constant) tile scale so a tile exactly fills
    // its grid step, and that is the scale the scatter hands out. Sizing rings
    // from the placeholder would over-allocate by (2.5 / 1.01)^2.
    const scaleMax = isCarpet
      ? CELL / carpetDiv / (speciesById(standEntry.species).tileM ?? 0.18)
      : Math.max(standEntry.scaleMax, 0.1)
    // TWO DIFFERENT NUMBERS, and conflating them drops plants:
    //  * enumSlots — how many candidate slots the cull must EVALUATE per cell.
    //    A carpet has carpetDiv² of them (484 at life size), deliberately over
    //    SCATTER_MAX_PER_CELL; visiting only 128 renders a quarter of the mat.
    //  * perCell — how many are expected to SURVIVE, which is all the instance
    //    buffer has to hold. For a zone-partitioned carpet that is roughly the
    //    entry's share of the wetness axis, not all of its slots.
    const enumSlots = standEntrySlots(standEntry)
    const perCell = isCarpet
      ? enumSlots * Math.min(1, (standEntry.wetWidth ?? 1) * CARPET_BAND_HEADROOM)
      : SCATTER_MAX_PER_CELL * (Math.min(standEntry.density, 8) / 8)
    // Rings are binned by distance/scale, so a ring's world radius is its
    // metric radius times the stand entry's largest scale. Sized for the
    // PARAM MAXIMA, never for the current value: params move at runtime.
    const ring0R = isCarpet ? CARPET_LOD_DIST[0] * scaleMax : LOD0_MAX * scaleMax
    const ring1R = isCarpet ? CARPET_LOD_DIST[1] * scaleMax : LOD1_MAX * scaleMax
    const regionCap = ringCapacity(REGION_MAX, perCell, standCells)
    const caps: [number, number, number] = [
      Math.min(ringCapacity(ring0R, perCell, standCells), regionCap),
      Math.min(ringCapacity(ring1R, perCell, standCells), regionCap),
      regionCap,
    ]
    const bases: [number, number, number] = [0, caps[0], caps[0] + caps[1]]
    const total = bases[2] + caps[2]
    const lattices = isCarpet ? carpetLattices : plantLattices
    const indirectReset = new Uint32Array(16)
    lattices.forEach((l, lod) => {
      indirectReset[lod * DRAW_ARG_U32] = l.indexCount
      indirectReset[lod * DRAW_ARG_U32 + 2] = l.firstIndex
    })

    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_STRIDE * LODS.length,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: total * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 64,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer, size: INFO_STRIDE } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = device.createBindGroup({
      label: `${ctx.id}/draw-bg-${entryIndex}`,
      layout: drawBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer, size: INFO_STRIDE } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: gpu.albedoTex.createView() },
        { binding: 3, resource: gpu.surfTex.createView() },
        { binding: 4, resource: gpu.dispTex.createView() },
        { binding: 5, resource: sampler },
      ],
    })
    return {
      speciesId: standEntry.species,
      gpu,
      isCarpet,
      bases,
      caps,
      enumSlots,
      lattices,
      indirectReset,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroup,
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
  const info = new Float32Array(INFO_FLOATS * LODS.length)
  const planes = new Float32Array(24)

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

      const lod0 = Math.min(ctx.params.lod0Dist, LOD0_MAX)
      const lod1 = Math.max(lod0, Math.min(ctx.params.lod1Dist, LOD1_MAX))

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        const carpet = entry.isCarpet
        // A mat gets its own alpha reference, ring radii and lattice, and its
        // occlusion gain is measured on the ONE tile it actually samples (the
        // top view, whose baked sky occlusion is 1 by construction — folding
        // the side views' 0.45 mean in here would brighten the moss by 38%).
        const crevice = carpet ? ctx.params.carpetCrevice : ctx.params.depthShade
        const gain = carpet
          ? 1 / lerp(1 - crevice, 1, entry.gpu.heightMean)
          : 1 / (lerp(1, AO_FIELD_MEAN, ctx.params.aoStrength) * lerp(1 - crevice, 1, DIST_FIELD_MEAN))
        LODS.forEach((_, lod) => {
          const lodDesc = LODS[lod]!
          const grid = carpet ? CARPET_LODS[lod]!.grid : lodDesc.side
          const curve = (carpet ? CARPET_LODS[lod]!.curve : lodDesc.curve) * ctx.params.curvature
          const o = lod * INFO_FLOATS
          info.set(planes, o)
          info[o + 24] = x0
          info[o + 25] = z0
          info[o + 26] = sideX
          info[o + 27] = sideZ
          info[o + 28] = ctx.seed
          info[o + 29] = entryIndex
          info[o + 30] = R
          info[o + 31] = carpet ? CARPET_ALPHA_REF : ctx.params.alphaRef
          info[o + 32] = a.y0
          info[o + 33] = a.y1
          info[o + 34] = a.rXZ
          info[o + 35] = a.cx
          info[o + 36] = a.cz
          info[o + 37] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
          // For a mat this is the mean canopy height — where a flat far-ring
          // tile sits, and the base the displacement mixes away from.
          info[o + 38] = carpet ? entry.gpu.canopyMean : TOP_FRAC
          info[o + 39] = ctx.params.aoStrength
          info[o + 40] = entry.bases[0]
          info[o + 41] = entry.bases[1]
          info[o + 42] = entry.bases[2]
          info[o + 44] = entry.caps[0]
          info[o + 45] = entry.caps[1]
          info[o + 46] = entry.caps[2]
          info[o + 48] = carpet ? CARPET_LOD_DIST[0] : lod0
          info[o + 49] = carpet ? CARPET_LOD_DIST[1] : lod1
          info[o + 52] = grid[0]
          info[o + 53] = grid[1]
          info[o + 54] = lodDesc.top[0]
          info[o + 55] = lodDesc.top[1]
          info[o + 56] = entry.gpu.dispSpan // carpet height-field decode: span
          info[o + 57] = gain
          info[o + 58] = crevice
          info[o + 59] = curve
          info[o + 60] = entry.bases[lod]!
          info[o + 61] = entry.caps[lod]!
          info[o + 62] = entry.enumSlots
          info[o + 63] = entry.gpu.dispBase // carpet height-field decode: base
        })
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, entry.indirectReset)
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
      pass.setIndexBuffer(indexBuffer, 'uint16')
      // Near ring first: the curved cards write solid depth, so the flat far
      // ring behind them is rejected by early-z instead of shaded.
      for (let lod = 0; lod < LODS.length; lod++) {
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroup, [lod * INFO_STRIDE])
          pass.drawIndexedIndirect(entry.indirectBuffer, lod * DRAW_ARG_U32 * 4)
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

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Worst-case survivors of one ring: `perCell` candidates in every cell of its
 * disc, clamped to the cells the stand actually has.
 */
function ringCapacity(radius: number, perCell: number, standCells: number): number {
  // Grow the disc by a cell's half-diagonal: a small ring can be covered by a
  // few cells whose candidates ALL fall inside it, and the naive area estimate
  // then under-counts them. Cheap insurance — a clamped ring silently drops
  // tiles, which on a mat reads exactly like a placement bug.
  const r = radius + CELL * 0.71
  const cells = Math.min((Math.PI * r * r) / (CELL * CELL), standCells)
  return Math.ceil(cells * perCell * 1.06) + 512
}

/**
 * One index buffer for every lattice: per ring, a plant lattice (side patch
 * then canopy patch) and a carpet lattice (one ground-parallel grid). Vertex
 * ids are local to the lattice — each draw brings its own firstIndex and its
 * own grid dims in the uniform, so ids never have to be globally unique.
 */
function buildLattices(): {
  indices: Uint16Array<ArrayBuffer>
  plantLattices: LatticeRange[]
  carpetLattices: LatticeRange[]
} {
  const out: number[] = []
  const quads = (cols: number, rows: number, base: number): void => {
    for (let j = 0; j < rows - 1; j++) {
      for (let i = 0; i < cols - 1; i++) {
        const a = base + j * cols + i
        out.push(a, a + 1, a + cols, a + 1, a + cols + 1, a + cols)
      }
    }
  }
  const emit = (build: () => void): LatticeRange => {
    const firstIndex = out.length
    build()
    return { firstIndex, indexCount: out.length - firstIndex }
  }
  const plantLattices = LODS.map((lod) =>
    emit(() => {
      quads(lod.side[0], lod.side[1], 0)
      quads(lod.top[0], lod.top[1], lod.side[0] * lod.side[1])
    }),
  )
  const carpetLattices = CARPET_LODS.map((lod) => emit(() => quads(lod.grid[0], lod.grid[1], 0)))
  // Buffer size must be a multiple of 4; u16 indices means an even count.
  if (out.length % 2 !== 0) out.push(0)
  return { indices: new Uint16Array(out), plantLattices, carpetLattices }
}

/**
 * Where the species' periodic tile sits inside its top-view capture, as a uv
 * rect. The capture square is 2*rXZ across, centred on the mesh bounds; the
 * tile is [0, tileM]^2 in the mesh frame (tile origin is (0,0) for every
 * current source mesh). The crop is worth ~3.4x on-screen texel density.
 */
interface TileRect {
  u0: number
  v0: number
  u1: number
  v1: number
}

function tileRect(atlas: ReliefAtlas, tileM: number): TileRect {
  const u0 = 0.5 + (0.5 * (0 - atlas.cx)) / atlas.rXZ
  const u1 = 0.5 + (0.5 * (tileM - atlas.cx)) / atlas.rXZ
  const v0 = 0.5 + (0.5 * (0 - atlas.cz)) / atlas.rXZ
  const v1 = 0.5 + (0.5 * (tileM - atlas.cz)) / atlas.rXZ
  return { u0, v0, u1, v1 }
}

/**
 * Resample the top tile's TILE SQUARE into a standalone rgba8 image.
 *
 * Cropping is not (only) about texel density — it is what keeps the mip chain
 * honest. Only 26-34% of the top tile is inside the periodic tile; the rest is
 * background the mat never samples. Mip level 8 of the uncropped tile therefore
 * averages the whole capture, and the SURF height channel there falls from 0.78
 * to 0.34 — measured, and it showed up as the far field rendering 15% darker
 * than 001-billboard-smoke at `cam=grazing` while `debug=albedo` matched it
 * band for band (the albedo mip filter is coverage-weighted, so it was immune).
 * With the crop every mip level averages moss only, and the crevice term
 * converges to exactly its mean-preserving 1.0 at distance.
 *
 * rgb is resampled unweighted on purpose: the bake already stores it divided by
 * coverage and dilated into the empty texels, so it is defined everywhere and a
 * second division would inflate it (the 017-cached-clusters trap).
 */
function resampleTileSquare(src: Uint8Array, srcPx: number, rect: TileRect, outPx: number): Uint8Array<ArrayBuffer> {
  const tilePx = srcPx / 3
  const base = 2 * tilePx // tile 8 = column 2, row 2 of the 3x3 atlas
  const out = new Uint8Array(outPx * outPx * 4)
  const fetch = (x: number, y: number, c: number): number =>
    src[((base + Math.max(0, Math.min(tilePx - 1, y))) * srcPx + base + Math.max(0, Math.min(tilePx - 1, x))) * 4 + c]!
  for (let j = 0; j < outPx; j++) {
    const v = rect.v0 + (rect.v1 - rect.v0) * ((j + 0.5) / outPx)
    const py = v * tilePx - 0.5
    const y0 = Math.floor(py)
    const fy = py - y0
    for (let i = 0; i < outPx; i++) {
      const u = rect.u0 + (rect.u1 - rect.u0) * ((i + 0.5) / outPx)
      const px = u * tilePx - 0.5
      const x0 = Math.floor(px)
      const fx = px - x0
      for (let c = 0; c < 4; c++) {
        const a = fetch(x0, y0, c) * (1 - fx) + fetch(x0 + 1, y0, c) * fx
        const b = fetch(x0, y0 + 1, c) * (1 - fx) + fetch(x0 + 1, y0 + 1, c) * fx
        out[(j * outPx + i) * 4 + c] = Math.round(a * (1 - fy) + b * fy)
      }
    }
  }
  return out
}

/**
 * The mat's own displacement field: the baked TOP-VIEW front-surface distance
 * — for a straight-down capture that is exactly the cushion's height above the
 * peat — resampled over the tile's own square at the rate the lattice can
 * actually carry, and rescaled to use the full 8-bit range.
 *
 * Not the DISP atlas, which the plant path uses: DISP is 32 texels per tile
 * flood-filled and blurred twice, and at 0.18m that leaves 18 texels across a
 * tile — a field so smooth that 56-68% of its variance is the part that
 * SURVIVES 4-FOLD ROTATION, i.e. the part every tile has in common no matter
 * how the stand rotated it. That component tiles in perfect lockstep across
 * the whole bog, and at a grazing angle a 5mm ridge occludes ~6cm of the tile
 * behind it, so the mat reads as woven basketwork (measured: unmistakable at
 * `cam=grazing`). Two things fix it, both here:
 *   1. resample from the FULL-RATE height channel (136 texels across the tile,
 *      alpha-gated so the empty gaps down to the peat do not pull it), which
 *      drops the rotation-invariant share to ~0.25 and carries MORE relief;
 *   2. subtract the rotation-invariant component outright, mean preserved.
 *      What is left is only the structure that the stand's 90-degree rotations
 *      decorrelate — irregular cushion lumps instead of a lattice. Nothing is
 *      invented: this removes a signal, and specifically the one signal a
 *      periodic tile cannot help repeating.
 */
interface CarpetHeight {
  /** N x N r8, sampled over the tile square at (fu, fv) with no atlas maths. */
  field: Uint8Array<ArrayBuffer>
  n: number
  /** Decode: hfrac = base + texel * span. Both in [y0, y1] fractions. */
  base: number
  span: number
  /** Mean canopy height — where a flat (far-ring) tile sits. */
  mean: number
  /** Coverage-gated mean of the crevice term's input, for the shading gain. */
  heightMean: number
}

function buildCarpetHeight(atlas: ReliefAtlas, tileM: number, n: number): CarpetHeight {
  const { u0, v0, u1, v1 } = tileRect(atlas, tileM)
  const sTile = atlas.surfPx / 3
  const aScale = atlas.atlasPx / atlas.surfPx
  const sample = (x: number, y: number): number => {
    const cx = 2 * sTile + Math.max(0, Math.min(sTile - 1, x))
    const cy = 2 * sTile + Math.max(0, Math.min(sTile - 1, y))
    // Texels the alpha test throws away are gaps, not surface.
    if (atlas.albedo[(cy * aScale * atlas.atlasPx + cx * aScale) * 4 + 3]! / 255 < CARPET_ALPHA_REF) return -1
    return atlas.surf[(cy * atlas.surfPx + cx) * 4 + 2]! / 255
  }

  const h = new Float32Array(n * n)
  let sum = 0
  let count = 0
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const px0 = Math.round((u0 + ((u1 - u0) * i) / n) * sTile)
      const px1 = Math.max(px0 + 1, Math.round((u0 + ((u1 - u0) * (i + 1)) / n) * sTile))
      const py0 = Math.round((v0 + ((v1 - v0) * j) / n) * sTile)
      const py1 = Math.max(py0 + 1, Math.round((v0 + ((v1 - v0) * (j + 1)) / n) * sTile))
      let s = 0
      let k = 0
      for (let y = py0; y < py1; y++) {
        for (let x = px0; x < px1; x++) {
          const v = sample(x, y)
          if (v < 0) continue
          s += v
          k++
        }
      }
      h[j * n + i] = k > 0 ? s / k : -1
      if (k > 0) {
        sum += h[j * n + i]!
        count++
      }
    }
  }
  const rawMean = count > 0 ? sum / count : 0.75
  for (let i = 0; i < n * n; i++) if (h[i]! < 0) h[i] = rawMean

  // One 3x3 blur with WRAP — the tile is periodic, so its own opposite edge is
  // its neighbour. Also band-limits the field to roughly the lattice rate.
  const blurred = new Float32Array(n * n)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) s += h[(((j + dj + n) % n) * n) + ((i + di + n) % n)]!
      }
      blurred[j * n + i] = s / 9
    }
  }

  // Subtract the 4-fold-rotation-invariant part, mean preserved.
  let mean = 0
  for (let i = 0; i < n * n; i++) mean += blurred[i]!
  mean /= n * n
  const out = new Float32Array(n * n)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const rot =
        (blurred[j * n + i]! +
          blurred[i * n + (n - 1 - j)]! +
          blurred[(n - 1 - j) * n + (n - 1 - i)]! +
          blurred[(n - 1 - i) * n + j]!) /
        4
      out[j * n + i] = blurred[j * n + i]! - rot + mean
    }
  }

  let lo = Infinity
  let hi = -Infinity
  for (const v of out) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const span = Math.max(hi - lo, 1e-4)
  const field = new Uint8Array(n * n)
  for (let i = 0; i < n * n; i++) field[i] = Math.round(Math.min(1, Math.max(0, (out[i]! - lo) / span)) * 255)
  return { field, n, base: lo, span, mean, heightMean: rawMean }
}

/**
 * Upload the three atlases and generate the mip chains on the GPU.
 *
 * A CARPET species uploads the TOP TILE ALONE. Budget allocation, not budget
 * overrun: 8 of the 9 tiles are side views, and a mat never shows one — it has
 * no silhouette from the side, and a vertical card through a 7cm cushion is
 * exactly the artifact this pass exists to remove. Dropping them takes the
 * texture cost from 15.7 MB to 1.75 MB, which is what pays for enumerating a
 * life-size carpet's 484 slots per cell instead of the scatter budget's 128.
 */
function uploadAtlas(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  atlas: ReliefAtlas,
  isCarpet: boolean,
): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const tileM = speciesById(speciesId).tileM ?? 0.18
  const aPx = isCarpet ? CARPET_ALBEDO_PX : ATLAS
  const sPx = isCarpet ? CARPET_SURF_PX : SURF
  // A mat replaces the DISP atlas outright with its own tile-square field.
  const carpetH = isCarpet ? buildCarpetHeight(atlas, tileM, CARPET_FIELD) : null
  const dPx = carpetH ? carpetH.n : DISP
  const mipA = Math.floor(Math.log2(aPx)) + 1
  const mipS = Math.floor(Math.log2(sPx)) + 1
  const albedoTex = ctx.res.createTexture(
    { label: `${ctx.id}/${speciesId}/albedo`, size: [aPx, aPx], format: 'rgba8unorm', mipLevelCount: mipA, usage },
    { species: speciesId, tag: 'relief-albedo' },
  )
  const surfTex = ctx.res.createTexture(
    { label: `${ctx.id}/${speciesId}/surf`, size: [sPx, sPx], format: 'rgba8unorm', mipLevelCount: mipS, usage },
    { species: speciesId, tag: 'relief-surface' },
  )
  const dispTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/disp`,
      size: [dPx, dPx],
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { species: speciesId, tag: 'relief-displacement' },
  )
  if (isCarpet) {
    device.queue.writeTexture(
      { texture: albedoTex },
      resampleTileSquare(atlas.albedo, atlas.atlasPx, tileRect(atlas, tileM), aPx),
      { bytesPerRow: aPx * 4, rowsPerImage: aPx },
      [aPx, aPx],
    )
    device.queue.writeTexture(
      { texture: surfTex },
      resampleTileSquare(atlas.surf, atlas.surfPx, tileRect(atlas, tileM), sPx),
      { bytesPerRow: sPx * 4, rowsPerImage: sPx },
      [sPx, sPx],
    )
  } else {
    device.queue.writeTexture({ texture: albedoTex }, atlas.albedo, { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS }, [
      ATLAS,
      ATLAS,
    ])
    device.queue.writeTexture({ texture: surfTex }, atlas.surf, { bytesPerRow: SURF * 4, rowsPerImage: SURF }, [
      SURF,
      SURF,
    ])
  }
  if (carpetH) {
    device.queue.writeTexture({ texture: dispTex }, carpetH.field, { bytesPerRow: dPx, rowsPerImage: dPx }, [dPx, dPx])
  } else {
    device.queue.writeTexture({ texture: dispTex }, atlas.disp, { bytesPerRow: DISP, rowsPerImage: DISP }, [DISP, DISP])
  }

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo')
  const boxPipe = mkPipeline('fs_box')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, levels: number, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < levels; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${level}`,
        colorAttachments: [
          { view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  genMips(albedoTex, mipA, albedoPipe)
  genMips(surfTex, mipS, boxPipe)
  device.queue.submit([enc.finish()])

  return {
    atlas,
    albedoTex,
    surfTex,
    dispTex,
    carpet: isCarpet,
    canopyMean: carpetH ? carpetH.mean : TOP_FRAC,
    heightMean: carpetH ? carpetH.heightMean : DIST_FIELD_MEAN,
    dispBase: carpetH ? carpetH.base : 0,
    dispSpan: carpetH ? carpetH.span : 1,
  }
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
