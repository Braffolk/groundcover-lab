import carpetSrc from './shaders/carpet.wgsl'
import cullSrc from './shaders/cull.wgsl'
import cullCarpetSrc from './shaders/cull_carpet.wgsl'
import impostorSrc from './shaders/impostor.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { GRID_N, N_VIEWS, TILE, buildViewFrames, loadSpeciesViewSet, type ViewSet } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type StandSpecies,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Octahedral view-set impostors.
 *
 * Startup: every species' 121-view hemi-octahedral capture (albedo+coverage
 * rgba8, oct normals rg8) is loaded from mesh/baked / OPFS or baked in-browser
 * from the raw GCMESH1 mesh, uploaded as a mipped texture ARRAY (one layer per
 * view — layers cannot bleed into each other when minified), and its per-view
 * orthographic bases are uploaded once as a small table.
 *
 * Per frame, per stand entry: one compute pass evaluates the shared scatter
 * over the camera-centred cell region (one workgroup per cell, cell-level
 * region/frustum rejects first), frustum-culls, and compacts survivors into
 * four distance buckets; then four indirect draws submit them near-to-far, one
 * quad per plant. Cost is O(visible region), never O(plants in the stand).
 */

const CELL = SCATTER_CELL_SIZE
/** Keep equal to the manifest's regionRadius max — sizes the bucket capacities. */
const REGION_MAX = 128
const MIP_LEVELS = 5 // 128 -> 8 px tiles
const BUCKETS = 4
/** Outer radii (m) of the front-to-back draw buckets; the last is regionRadius. */
const BUCKET_EDGES = [6, 18, 50]
const INFO_FLOATS = 60 // 15 vec4 — keep in sync with struct EntryInfo
/** Lanes per cull workgroup; a carpet cell needs ceil(slots / this) of them. */
const CULL_LANES = 128
/**
 * Fraction of a bucket's grid nodes one carpet entry can own. The three
 * Sphagnum states PARTITION the wetness field, so each owns roughly a third of
 * the mat on average — but the field is not uniform, and a single hummock or
 * hollow can be claimed almost entirely by one state. Measured worst case over
 * six camera positions on the bog stand: 1.00 / 1.00 / 0.61 / 0.68 by bucket.
 * The near buckets are small enough to be pathological; the far ones average
 * over many wetness cells, which is exactly where the memory is.
 */
const CARPET_BUCKET_SHARE = [1, 1, 0.8, 0.85]

interface SpeciesGpu {
  set: ViewSet
  albedoTex: GPUTexture
  normalTex: GPUTexture
  viewTable: GPUBuffer
}

interface EntryGpu {
  speciesId: string
  /** stand_table[i].carpet_div > 0: a tiled mat, drawn by the carpet pipeline. */
  carpet: boolean
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroups: GPUBindGroup[]
  /** EntryInfo staging, with every species-constant slot filled at init. */
  info: Float32Array<ArrayBuffer>
  /**
   * Slot chunks per cell = ceil(standEntrySlots / CULL_LANES). 1 for a
   * scattered entry, 4 for the bog carpet's 484 slots — driving this from
   * SCATTER_MAX_PER_CELL would render a quarter of the mat.
   */
  chunks: number
  /** Workgroups: cells in x, cells in z, slot chunks. */
  dispatch: [number, number, number]
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/view-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- per-species view sets (sequential: a bake transiently needs ~200MB) ---
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const set = await loadSpeciesViewSet(ctx, entry.species)
    if (set.gridN !== GRID_N || set.tilePx !== TILE) {
      throw new Error(`[${ctx.id}] baked view set is ${set.gridN}x${set.tilePx}, shader expects ${GRID_N}x${TILE}`)
    }
    speciesGpu.set(entry.species, uploadViewSet(ctx, entry.species, set))
  }

  // --- layouts ---------------------------------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const carpet = (standEntry.carpetDiv ?? 0) > 0
    // 4 B per carpet tile (a packed grid node), 16 B per scattered plant.
    const stride = carpet ? 4 : 16
    const bucketCap = carpet ? carpetCapacities(standEntry, ctx.stand.radius) : bucketCapacities(Math.min(standEntry.density, 8))
    const bucketBase: number[] = []
    let total = 0
    for (const cap of bucketCap) {
      bucketBase.push(total)
      total += cap
    }
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: total * stride, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: carpet ? 'culled-tiles' : 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: BUCKETS * 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ],
    })
    // One bind group per bucket, each windowing the instance buffer at that
    // bucket's segment — the shader always indexes from 0, so no per-draw
    // uniform or firstInstance semantics are involved.
    const albedoView = gpu.albedoTex.createView({ dimension: '2d-array' })
    const normalView = gpu.normalTex.createView({ dimension: '2d-array' })
    const drawBindGroups = bucketCap.map((cap, b) =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}-${b}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer, offset: bucketBase[b]! * stride, size: cap * stride } },
          { binding: 2, resource: { buffer: gpu.viewTable } },
          { binding: 3, resource: albedoView },
          { binding: 4, resource: normalView },
          { binding: 5, resource: sampler },
        ],
      }),
    )
    // Everything about the plant's baked shape is fixed for the run — fill it
    // once here; update() only rewrites frustum planes, region and params.
    const info = new Float32Array(INFO_FLOATS)
    const [cx, cy, cz] = gpu.set.centre
    const [hx, hy, hz] = gpu.set.half
    info.set([cx, cy, cz, hy * 2], 32)
    info.set([hx, hy, hz, cy - hy], 36)
    info.set(bucketBase, 44)
    info.set(bucketCap, 48)
    // Cull sphere: AABB radius plus the clump's horizontal offset, so the test
    // needs no yaw trig. Core radius drives the camera-inside fade.
    info.set([Math.hypot(hx, hy, hz) + Math.hypot(cx, cz), Math.max(hx, hz), 0, 0], 52)
    info[28] = ctx.seed
    info[29] = entryIndex

    return {
      speciesId: standEntry.species,
      carpet,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroups,
      info,
      chunks: Math.ceil(standEntrySlots(standEntry) / CULL_LANES),
      dispatch: [0, 0, 0],
    }
  })

  const anyCarpet = entries.some((e) => e.carpet)
  let cullPipeline!: GPUComputePipeline
  let cullCarpetPipeline: GPUComputePipeline | null = null
  let blendPipeline!: GPURenderPipeline
  let nearestPipeline!: GPURenderPipeline
  let carpetBlendPipeline: GPURenderPipeline | null = null
  let carpetNearestPipeline: GPURenderPipeline | null = null
  const build = (): void => {
    const cullLayout = device.createPipelineLayout({
      label: `${ctx.id}/cull-pl`,
      bindGroupLayouts: [ctx.frame.layout, cullBgl],
    })
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: cullLayout,
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/draw-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkDraw = (module: GPUShaderModule, vs: string, fs: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${fs}`,
        layout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    const impostorModule = ctx.shaders.module(impostorSrc)
    blendPipeline = mkDraw(impostorModule, 'vs_main', 'fs_blend3')
    nearestPipeline = mkDraw(impostorModule, 'vs_main', 'fs_nearest')
    // Carpet species get their own pipelines rather than a branch in the plant
    // shader: different instance record, different geometry, different alpha
    // rule — and the upright path then stays bit-for-bit what it was.
    if (anyCarpet) {
      cullCarpetPipeline = device.createComputePipeline({
        label: `${ctx.id}/cull-carpet`,
        layout: cullLayout,
        compute: { module: ctx.shaders.module(cullCarpetSrc), entryPoint: 'cs_cull_carpet' },
      })
      const carpetModule = ctx.shaders.module(carpetSrc)
      carpetBlendPipeline = mkDraw(carpetModule, 'vs_carpet', 'fs_carpet_blend3')
      carpetNearestPipeline = mkDraw(carpetModule, 'vs_carpet', 'fs_carpet_nearest')
    }
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(BUCKETS * 4)
  for (let b = 0; b < BUCKETS; b++) indirectReset[b * 4] = 6 // vertexCount

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      // Region cell rect, clamped to the stand's cell range here on the CPU:
      // cells outside the stand hold nothing and must not be dispatched.
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      for (const entry of entries) {
        const info = entry.info
        info.set(planes, 0)
        info.set([x0, z0, sideX, sideZ], 24)
        info[30] = R
        // A mat is a closed surface and must not dissolve with distance: the
        // mip chain pulls a tile's coverage toward its own mean, so the grass
        // reference would drop whole distant tiles out and punch holes in the
        // carpet. The genuinely empty texels (the gaps down to the peat) still
        // open at a low reference.
        info[31] = entry.carpet ? ctx.params.carpetAlphaRef : ctx.params.alphaRef
        info.set([Math.min(BUCKET_EDGES[0]!, R), Math.min(BUCKET_EDGES[1]!, R), Math.min(BUCKET_EDGES[2]!, R), R], 40)
        info[54] = ctx.params.viewTint ? 1 : 0
        info[55] = ctx.params.carpetNormalDetail
        info[56] = ctx.params.carpetCropInset
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.dispatch = [sideX, sideZ, entry.chunks]
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setBindGroup(0, ctx.frame.bindGroup)
      let cullCarpet: boolean | null = null
      for (const entry of entries) {
        if (entry.dispatch[0] === 0 || entry.dispatch[1] === 0) continue
        if (entry.carpet !== cullCarpet) {
          cull.setPipeline(entry.carpet ? cullCarpetPipeline! : cullPipeline)
          cullCarpet = entry.carpet
        }
        cull.setBindGroup(1, entry.cullBindGroup)
        cull.dispatchWorkgroups(entry.dispatch[0], entry.dispatch[1], entry.dispatch[2])
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'impostors', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      const nearest = ctx.params.viewBlend === 'nearest'
      const plantPipeline = nearest ? nearestPipeline : blendPipeline
      const matPipeline = nearest ? carpetNearestPipeline : carpetBlendPipeline
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near bucket first: alpha-tested cards write solid depth, so the
      // nearest plants become occluders for everything drawn after them. The
      // carpet is part of that: a mat at a low alpha reference is nearly
      // solid, so it occludes the tiles behind it instead of leaking overdraw.
      let bound: GPURenderPipeline | null = null
      for (let b = 0; b < BUCKETS; b++) {
        for (const entry of entries) {
          const want = entry.carpet ? matPipeline! : plantPipeline
          if (want !== bound) {
            pass.setPipeline(want)
            bound = want
          }
          pass.setBindGroup(1, entry.drawBindGroups[b]!)
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

/**
 * Instance capacity per distance bucket. Bucket j covers an annulus, and the
 * scatter's density is capped, so density x area is a real bound on what can
 * land there — the slack only covers cell-level clustering.
 */
function bucketCapacities(density: number): number[] {
  const edges = [0, ...BUCKET_EDGES, REGION_MAX]
  const caps: number[] = []
  for (let j = 0; j < BUCKETS; j++) {
    const area = Math.PI * (edges[j + 1]! ** 2 - edges[j]! ** 2)
    // Round to 16 records so every bucket base is 256B aligned (bind offsets).
    caps.push(Math.ceil((density * area * 1.15 + 2048) / 16) * 16)
  }
  return caps
}

/**
 * Tile capacity per distance bucket for a CARPET entry. A carpet's count comes
 * from its GRID, never from `density`: carpetDiv² nodes per scatter cell, of
 * which this entry's wetness interval claims some share. The annulus is also
 * clipped to the stand's own square — on the ±96 m bog the far bucket would
 * otherwise be sized for a 128 m region that holds no plants.
 */
function carpetCapacities(entry: StandSpecies, standRadius: number): number[] {
  const nodesPerM2 = entry.carpetDiv! ** 2 / SCATTER_CELL_SIZE ** 2
  const edges = [0, ...BUCKET_EDGES, REGION_MAX]
  const standArea = (2 * standRadius) ** 2
  const caps: number[] = []
  for (let j = 0; j < BUCKETS; j++) {
    const inner = Math.PI * edges[j]! ** 2
    const area = Math.min(Math.PI * edges[j + 1]! ** 2, standArea) - Math.min(inner, standArea)
    // Round to 64 records: at 4 B each that keeps every bucket base 256B
    // aligned for the per-bucket bind-group window.
    caps.push(Math.ceil((Math.max(area, 0) * nodesPerM2 * CARPET_BUCKET_SHARE[j]! + 4096) / 64) * 64)
  }
  return caps
}

/** Upload the view set as a mipped texture array + its per-view basis table. */
function uploadViewSet(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, set: ViewSet): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [TILE, TILE, N_VIEWS],
      format: 'rgba8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'view-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [TILE, TILE, N_VIEWS],
      format: 'rg8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'view-normal' },
  )
  device.queue.writeTexture({ texture: albedoTex }, set.albedo, { bytesPerRow: TILE * 4, rowsPerImage: TILE }, [
    TILE,
    TILE,
    N_VIEWS,
  ])
  device.queue.writeTexture({ texture: normalTex }, set.normalOct, { bytesPerRow: TILE * 2, rowsPerImage: TILE }, [
    TILE,
    TILE,
    N_VIEWS,
  ])

  // Per-view basis table: the bake's own axes, so runtime reprojection is
  // exact rather than a reconstruction that could drift from it.
  const frames = buildViewFrames(set.half)
  const table = new Float32Array(N_VIEWS * 8)
  frames.forEach((f, k) => {
    table.set([f.right[0], f.right[1], f.right[2], Math.max(set.ext[k * 2]!, 1e-4)], k * 8)
    table.set([f.up[0], f.up[1], f.up[2], Math.max(set.ext[k * 2 + 1]!, 1e-4)], k * 8 + 4)
  })
  const viewTable = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/view-table`,
      size: table.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'view-basis' },
  )
  device.queue.writeBuffer(viewTable, 0, table)

  generateMips(ctx, albedoTex, normalTex)
  return { set, albedoTex, normalTex, viewTable }
}

/** One render pass per (layer, level) writing both maps; layers never bleed. */
function generateMips(ctx: ExperimentContext<typeof PARAMS>, albedoTex: GPUTexture, normalTex: GPUTexture): void {
  const { device } = ctx
  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ],
  })
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/mipgen`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] }),
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rg8unorm' }] },
    primitive: { topology: 'triangle-list' },
  })

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const sub = (tex: GPUTexture, level: number, layer: number): GPUTextureView =>
    tex.createView({
      dimension: '2d',
      baseMipLevel: level,
      mipLevelCount: 1,
      baseArrayLayer: layer,
      arrayLayerCount: 1,
    })
  for (let layer = 0; layer < N_VIEWS; layer++) {
    for (let level = 1; level < MIP_LEVELS; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${layer}-${level}`,
        layout: bgl,
        entries: [
          { binding: 0, resource: sub(albedoTex, level - 1, layer) },
          { binding: 1, resource: sub(normalTex, level - 1, layer) },
        ],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${layer}-${level}`,
        colorAttachments: [
          { view: sub(albedoTex, level, layer), loadOp: 'clear', storeOp: 'store' },
          { view: sub(normalTex, level, layer), loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  device.queue.submit([enc.finish()])
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
