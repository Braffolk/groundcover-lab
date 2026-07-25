import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import canopySrc from './shaders/canopy.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, NORMAL_ATLAS, N_TILES, loadSpeciesAtlas, type ClumpAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Part-cloud cards + an amortised canopy cache.
 *
 * Startup: each species' part atlas (4 sub-clump impostors + 1 whole-plant
 * impostor, each 8 azimuths + a crown view, with per-tile tight coverage boxes)
 * is loaded from mesh/baked / OPFS or baked in-browser from the raw mesh,
 * uploaded and mipped.
 *
 * Per frame:
 *   1. canopy cache — clear + splat the stand's plants inside a 48m window into
 *      a world-anchored toroidal density grid, then re-light ONE QUARTER of the
 *      grid's cells (sun march + sky march). Round robin; the rest of the grid
 *      keeps last frame's values, which are still correct because the grid is
 *      world-anchored and plants do not move.
 *   2. cull — region-bounded scatter evaluation, frustum cull, and a two-bucket
 *      LOD split into a near list (card cloud) and a far list (single card).
 *   3. cards — near clouds first (8 quads/plant, nearest quadrant first so
 *      early-z can work), then the far cards (2 quads/plant). Hard alpha test,
 *      depth write, no blending, no frag_depth.
 *
 * Per-frame cost is O(visible region) + O(cache window), independent of the
 * stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const NEAR_MAX = 40 // keep equal to the manifest's nearRadius max
const MIPS_ALBEDO = 6 // 1792 -> 56 (tile 256 -> 8); deeper mips bleed across tiles
const MIPS_NORMAL = 5 // 896 -> 56
const BOTTOM_SHADE = 0.22
const INFO_FLOATS = 44
const ATLAS_FLOATS = 220 // 5 part boxes + 5 radii + 45 tile coverage boxes

// Canopy cache geometry — keep in sync with the constants in canopy.wgsl.
const VOL_NX = 128
const VOL_NY = 8
const VOL_NZ = 128
const VOL_CELL = 0.375 // m -> 48m window
const VOL_Y_TOP = 1.9 // m above the plant base plane
const VOL_GROUPS = 4
const CANOPY_ENTRY_MAX = 8 // keep equal to CANOPY_MAX_ENTRIES in tables.wgsl
const CANOPY_FLOATS = 48
const DENSITY_SCALE = 2.6
const SUN_STEP = 0.45
const SKY_K = 1.35
const JUMP_DIST = 6 // camera jump (m) that forces a full cache refresh

interface SpeciesGpu {
  atlas: ClumpAtlas
  albedoTex: GPUTexture
  normalTex: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  capNear: number
  capFar: number
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
  // Toroidal cache: xz wrap, y clamps at the ground and above the canopy.
  const canopySampler = device.createSampler({
    label: `${ctx.id}/canopy-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'repeat',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~500MB) ---
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesAtlas(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- canopy cache resources ----------------------------------------------
  const volCells = VOL_NX * VOL_NY * VOL_NZ
  const densityBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/canopy-density`, size: volCells * 4, usage: GPUBufferUsage.STORAGE },
    { tag: 'canopy-density' },
  )
  const volumeTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/canopy-volume`,
      size: [VOL_NX, VOL_NY, VOL_NZ],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    },
    { tag: 'canopy-volume' },
  )
  const canopyBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/canopy-info`,
      size: CANOPY_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'canopy-info' },
  )

  const canopyBgl = device.createBindGroupLayout({
    label: `${ctx.id}/canopy-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '3d' },
      },
    ],
  })
  const canopyBindGroup = device.createBindGroup({
    label: `${ctx.id}/canopy-bg`,
    layout: canopyBgl,
    entries: [
      { binding: 0, resource: { buffer: canopyBuffer } },
      { binding: 1, resource: { buffer: densityBuffer } },
      { binding: 2, resource: volumeTex.createView({ dimension: '3d' }) },
    ],
  })

  // --- bind group layouts ---------------------------------------------------
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
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  })

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL
  const nearSideMax = Math.ceil((2 * NEAR_MAX * 1.15) / CELL) + 1
  const nearSlotsMax = nearSideMax * nearSideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const capNear = Math.ceil(nearSlotsMax * (density / 8) * 1.1) + 512
    const capFar = Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const atlasBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/atlas-info-${entryIndex}`,
        size: ATLAS_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'atlas-info' },
    )
    device.queue.writeBuffer(atlasBuffer, 0, packAtlasInfo(gpu.atlas))
    const instBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/instances-${entryIndex}`,
        size: (capNear + capFar) * 16,
        usage: GPUBufferUsage.STORAGE,
      },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 32,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    return {
      gpu,
      capNear,
      capFar,
      infoBuffer,
      indirectBuffer,
      cullBindGroup: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer } },
          { binding: 2, resource: { buffer: indirectBuffer } },
        ],
      }),
      drawBindGroup: device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: atlasBuffer } },
          { binding: 2, resource: { buffer: instBuffer } },
          { binding: 3, resource: gpu.albedoTex.createView() },
          { binding: 4, resource: gpu.normalTex.createView() },
          { binding: 5, resource: sampler },
          { binding: 6, resource: volumeTex.createView({ dimension: '3d' }) },
          { binding: 7, resource: canopySampler },
          { binding: 8, resource: { buffer: canopyBuffer } },
        ],
      }),
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  let clearPipeline!: GPUComputePipeline
  let splatPipeline!: GPUComputePipeline
  let lightPipeline!: GPUComputePipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const canopyModule = ctx.shaders.module(canopySrc)
    const canopyLayout = device.createPipelineLayout({
      label: `${ctx.id}/canopy-pl`,
      bindGroupLayouts: [ctx.frame.layout, canopyBgl],
    })
    const mkCanopy = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({
        label: `${ctx.id}/${entryPoint}`,
        layout: canopyLayout,
        compute: { module: canopyModule, entryPoint },
      })
    clearPipeline = mkCanopy('cs_clear')
    splatPipeline = mkCanopy('cs_splat')
    lightPipeline = mkCanopy('cs_light')

    const cardsModule = ctx.shaders.module(cardsSrc)
    const cardsLayout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkCards = (entryPoint: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${entryPoint}`,
        layout: cardsLayout,
        vertex: { module: cardsModule, entryPoint },
        fragment: { module: cardsModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkCards('vs_near')
    farPipeline = mkCards('vs_far')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const canopyInfo = new Float32Array(CANOPY_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(8)
  let lastCam: [number, number, number] | null = null
  let cacheTick = 0
  let lightThreads = 0
  let splatSideX = 0
  let splatSideZ = 0
  // Canopy-cache work skipping: the density field is a pure function of (stand,
  // seed, window cell), so it only has to be rebuilt when the window SCROLLS;
  // and the sun does not move, so once every slab has had its turn the cache is
  // fully valid and the light pass has nothing left to do.
  let lastWinCell: [number, number] | null = null
  let densityDirty = true
  let relightLeft = VOL_GROUPS

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

      const topFrac = ctx.params.topCards ? ctx.params.crownHeight : 0
      indirectReset[0] = ctx.params.topCards ? 48 : 24
      indirectReset[4] = ctx.params.topCards ? 12 : 6
      entries.forEach((entry, entryIndex) => {
        const whole = entry.gpu.atlas.parts[4]!
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = Math.min(ctx.params.nearRadius, NEAR_MAX)
        info[32] = entry.capNear
        info[33] = entry.capFar
        info[34] = ctx.params.alphaRef
        info[35] = BOTTOM_SHADE
        info[36] = ctx.params.tintVar
        info[37] = ctx.params.aoStrength
        info[38] = ctx.params.sunShadow
        info[39] = topFrac
        info[40] = Math.max(whole.r, (whole.y1 - whole.y0) / 2) * 1.05
        info[41] = whole.y1
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
      })

      // --- canopy cache window (world-anchored, toroidal) -------------------
      const winCellX = Math.floor(cam.x / VOL_CELL) - VOL_NX / 2
      const winCellZ = Math.floor(cam.z / VOL_CELL) - VOL_NZ / 2
      // The splat needs every plant whose footprint can touch the window, so
      // the scatter rect is the window grown by a plant radius and clamped to
      // the stand (cells outside it hold nothing).
      const margin = 1.2
      const sx0 = Math.max(cellMin, Math.floor((winCellX * VOL_CELL - margin) / CELL))
      const sz0 = Math.max(cellMin, Math.floor((winCellZ * VOL_CELL - margin) / CELL))
      const sx1 = Math.min(cellMax, Math.floor(((winCellX + VOL_NX) * VOL_CELL + margin) / CELL))
      const sz1 = Math.min(cellMax, Math.floor(((winCellZ + VOL_NZ) * VOL_CELL + margin) / CELL))
      splatSideX = Math.max(0, sx1 - sx0 + 1)
      splatSideZ = Math.max(0, sz1 - sz0 + 1)

      const jumped =
        lastCam === null || Math.hypot(cam.x - lastCam[0], cam.y - lastCam[1], cam.z - lastCam[2]) > JUMP_DIST
      lastCam = [cam.x, cam.y, cam.z]
      const scrolled = lastWinCell === null || lastWinCell[0] !== winCellX || lastWinCell[1] !== winCellZ
      lastWinCell = [winCellX, winCellZ]
      // A camera jump invalidates the whole window at once (nothing that just
      // scrolled in is valid), so it is relit in a single frame; steady motion
      // only ever invalidates a rim column and rides the round robin. `slabs`
      // = how much of the grid one frame relights: 1 -> a quarter (4 groups).
      const slabs = Math.min(VOL_GROUPS, Math.max(1, Math.round(ctx.params.cacheSlabs)))
      const groups = jumped ? 1 : slabs === 1 ? 4 : slabs === 2 ? 2 : 1
      const group = cacheTick % groups
      cacheTick = (cacheTick + 1) % 12 // 12 = lcm(1,2,4) * 3, keeps the walk fair
      densityDirty = scrolled
      if (scrolled) relightLeft = groups
      lightThreads = relightLeft > 0 ? (VOL_NX / groups) * VOL_NY * VOL_NZ : 0
      if (relightLeft > 0) relightLeft--

      const cacheEntries = Math.min(ctx.stand.species.length, CANOPY_ENTRY_MAX)
      for (let i = 0; i < cacheEntries; i++) {
        const whole = speciesGpu.get(ctx.stand.species[i]!.species)!.atlas.parts[4]!
        canopyInfo[i * 4] = whole.r
        canopyInfo[i * 4 + 1] = whole.y1
      }
      const o = CANOPY_ENTRY_MAX * 4
      canopyInfo[o] = winCellX
      canopyInfo[o + 1] = winCellZ
      canopyInfo[o + 2] = sx0
      canopyInfo[o + 3] = sz0
      canopyInfo[o + 4] = splatSideX
      canopyInfo[o + 5] = splatSideZ
      canopyInfo[o + 6] = cacheEntries
      canopyInfo[o + 7] = ctx.seed
      canopyInfo[o + 8] = VOL_CELL
      canopyInfo[o + 9] = VOL_Y_TOP
      canopyInfo[o + 10] = DENSITY_SCALE
      canopyInfo[o + 11] = groups
      canopyInfo[o + 12] = group
      canopyInfo[o + 13] = SUN_STEP
      canopyInfo[o + 14] = SKY_K
      device.queue.writeBuffer(canopyBuffer, 0, canopyInfo)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // One compute pass for the cache and the cull: two passes cost two lots
      // of pass overhead (~0.2ms on this machine) for no benefit — nothing here
      // needs a pass boundary, only dispatch ordering.
      const canopyPass = ctx.timing.computePass(enc, 'cache+cull')
      canopyPass.setBindGroup(0, ctx.frame.bindGroup)
      canopyPass.setBindGroup(1, canopyBindGroup)
      // Density: only when the window scrolled (see the flags in update()).
      if (densityDirty && splatSideX > 0 && splatSideZ > 0) {
        canopyPass.setPipeline(clearPipeline)
        canopyPass.dispatchWorkgroups(Math.ceil(volCells / 64))
        canopyPass.setPipeline(splatPipeline)
        const cacheEntries = Math.min(ctx.stand.species.length, CANOPY_ENTRY_MAX)
        const threads = splatSideX * splatSideZ * SCATTER_MAX_PER_CELL * cacheEntries
        canopyPass.dispatchWorkgroups(Math.ceil(threads / 64))
      }
      // Light: one slab per frame while any slab is still stale, then nothing.
      if (lightThreads > 0) {
        canopyPass.setPipeline(lightPipeline)
        canopyPass.dispatchWorkgroups(Math.ceil(lightThreads / 64))
      }
      canopyPass.setPipeline(cullPipeline)
      for (const entry of entries) {
        if (entry.slotsPerFrame === 0) continue // camera outside the stand
        canopyPass.setBindGroup(1, entry.cullBindGroup)
        canopyPass.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
      }
      canopyPass.end()

      const pass = ctx.timing.renderPass(enc, 'cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near clouds first: they own the closest depth, so early-z rejects the
      // far field's hidden fragments instead of shading them.
      pass.setPipeline(nearPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 0)
      }
      pass.setPipeline(farPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 16)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Static per-species uniform: part boxes + per-tile coverage boxes. */
function packAtlasInfo(atlas: ClumpAtlas): Float32Array<ArrayBuffer> {
  const out = new Float32Array(ATLAS_FLOATS)
  atlas.parts.forEach((p, i) => {
    out.set([p.cx, p.cz, p.y0, p.y1], i * 4)
    out[20 + i * 4] = p.r
  })
  for (let t = 0; t < N_TILES; t++) {
    const b = atlas.tiles[t]!
    out.set([b.a0, b.a1, b.b0, b.b1], 40 + t * 4)
  }
  return out
}

/** Upload atlas mip 0 and generate the mip chain on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: ClumpAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS, ATLAS],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_ALBEDO,
      usage,
    },
    { species: speciesId, tag: 'part-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [NORMAL_ATLAS, NORMAL_ATLAS],
      format: 'rg8unorm',
      mipLevelCount: MIPS_NORMAL,
      usage,
    },
    { species: speciesId, tag: 'part-normal' },
  )
  device.queue.writeTexture(
    { texture: albedoTex },
    atlas.albedo,
    { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS },
    [ATLAS, ATLAS],
  )
  device.queue.writeTexture(
    { texture: normalTex },
    atlas.normalOct,
    { bytesPerRow: NORMAL_ATLAS * 2, rowsPerImage: NORMAL_ATLAS },
    [NORMAL_ATLAS, NORMAL_ATLAS],
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
  genMips(albedoTex, MIPS_ALBEDO, mkPipeline('fs_albedo', 'rgba8unorm'))
  genMips(normalTex, MIPS_NORMAL, mkPipeline('fs_normal', 'rg8unorm'))
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, normalTex }
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
