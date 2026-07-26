import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import carpetSrc from './shaders/carpet.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { loadSpeciesSlabs, tileSide, N_AZIM, N_LAYER, N_TILES, TILE_TOP0, type SlabAtlas } from './bake.ts'
import { loadSpeciesCarpet, CARPET_NRM, CARPET_TEX, N_BAND, N_CARPET_LAYER, type CarpetAtlas } from './carpet.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type StandSpecies,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Layered parallax cards.
 *
 * Startup: each species' atlases are loaded from mesh/baked / OPFS or baked
 * in-browser from the raw mesh, uploaded and mipped. An UPRIGHT species gets
 * the depth-slab atlas (bake.ts); a CARPET species — one whose stand entry
 * lays it out as a periodic mat — gets the much smaller height-band atlas
 * (carpet.ts) instead, because 96% of the slab atlas is side views a mat never
 * shows.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region (carpet_div^2 slots per cell for a mat, the 128
 * scatter slots otherwise), frustum-culls, and compacts survivors into indirect
 * draws — for upright plants a near ring running the layered eye-ray
 * reprojection and a far ring sampling one merged tile, for carpets a single
 * ground-parallel tile quad whose fragment shader walks the eye ray down
 * through the cushion's height bands.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const LOD_DIST_MAX = 48 // keep equal to the manifest's lodDistance max
const NEAR_MIPS = 5
const INFO_FLOATS = 64
/** Mip levels the band -> merged dissolve takes (see shaders/carpet.wgsl). */
const CARPET_MERGE_SPAN = 2.5

interface SpeciesGpu {
  atlas: SlabAtlas
  nearAlbedo: GPUTexture
  nearNormal: GPUTexture
  farAlbedo: GPUTexture
  farNormal: GPUTexture
  tileBuffer: GPUBuffer
  sideSpan: number
  topSpan: number
}

interface CarpetGpu {
  atlas: CarpetAtlas
  albedo: GPUTexture
  normal: GPUTexture
}

interface EntryGpu {
  carpet: CarpetGpu | null
  capacity: number
  nearCapacity: number
  /** Scatter slots per cell for THIS entry (carpet_div^2 for a mat). */
  slots: number
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  indirectReset: Uint32Array<ArrayBuffer>
  cullBindGroup: GPUBindGroup
  drawBindGroup: GPUBindGroup
  slotsPerFrame: number
  /** Extra info floats that only depend on the atlas, written once. */
  atlasInfo: (info: Float32Array) => void
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
  // A carpet tile is PERIODIC, and its bake is wrapped, so the image is
  // exactly one period: `repeat` makes the hardware continue the mat for free
  // when the parallax ray walks out of the tile, at every mip level.
  const tileSampler = device.createSampler({
    label: `${ctx.id}/tile-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 8,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  const isCarpet = (entry: StandSpecies): boolean => (entry.carpetDiv ?? 0) > 0

  // --- species atlases (sequential: a bake transiently needs ~400MB) --------
  const speciesGpu = new Map<string, SpeciesGpu>()
  const carpetGpu = new Map<string, CarpetGpu>()
  for (const entry of ctx.stand.species) {
    if (isCarpet(entry)) {
      if (carpetGpu.has(entry.species)) continue
      carpetGpu.set(entry.species, uploadCarpet(ctx, entry.species, await loadSpeciesCarpet(ctx, entry.species)))
    } else {
      if (speciesGpu.has(entry.species)) continue
      speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, await loadSpeciesSlabs(ctx, entry.species)))
    }
  }

  // --- bind group layouts / pipelines ---------------------------------------
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
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      {
        binding: 3,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' },
      },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const carpetBgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d-array' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d-array' },
      },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const carpet = isCarpet(standEntry) ? carpetGpu.get(standEntry.species)! : null
    const slots = standEntrySlots(standEntry)
    const density = Math.min(standEntry.density, 8)
    // Slots to EVALUATE and instances to STORE are different numbers. A carpet
    // fills only the share of the grid its wetness band claims, and that share
    // is far from uniform (the field is damped on slopes, so the dry band is
    // much wider than a third) — measure it instead of guessing, or the buffer
    // either overflows into holes or wastes 3x the memory.
    const capacity = carpet
      ? carpetCapacity(ctx, standEntry, slots)
      : Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    // The near ring is a disc of radius lodDistance * scaleMax, so its worst
    // case is a small fraction of the region — sized for the param maximum.
    // Carpets have no near ring at all.
    const nearR = Math.min(LOD_DIST_MAX * standEntry.scaleMax, REGION_MAX)
    const nearCapacity = carpet
      ? 0
      : Math.min(capacity, Math.ceil(Math.PI * nearR * nearR * density * 1.25) + 2048)
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const mkInstances = (name: string, count: number): GPUBuffer =>
      ctx.res.createBuffer(
        { label: `${ctx.id}/${name}-${entryIndex}`, size: Math.max(count, 16) * 16, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: `${name}-instances` },
      )
    const nearBuffer = mkInstances('near', nearCapacity)
    const farBuffer = mkInstances('far', capacity)
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 32,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: nearBuffer } },
        { binding: 2, resource: { buffer: farBuffer } },
        { binding: 3, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = carpet
      ? device.createBindGroup({
          label: `${ctx.id}/carpet-bg-${entryIndex}`,
          layout: carpetBgl,
          entries: [
            { binding: 0, resource: { buffer: infoBuffer } },
            { binding: 1, resource: { buffer: farBuffer } },
            { binding: 2, resource: carpet.albedo.createView({ dimension: '2d-array' }) },
            { binding: 3, resource: carpet.normal.createView({ dimension: '2d-array' }) },
            { binding: 4, resource: tileSampler },
          ],
        })
      : device.createBindGroup({
          label: `${ctx.id}/draw-bg-${entryIndex}`,
          layout: drawBgl,
          entries: [
            { binding: 0, resource: { buffer: infoBuffer } },
            { binding: 1, resource: { buffer: nearBuffer } },
            { binding: 2, resource: { buffer: farBuffer } },
            { binding: 3, resource: { buffer: speciesGpu.get(standEntry.species)!.tileBuffer } },
            { binding: 4, resource: speciesGpu.get(standEntry.species)!.nearAlbedo.createView() },
            { binding: 5, resource: speciesGpu.get(standEntry.species)!.nearNormal.createView() },
            { binding: 6, resource: speciesGpu.get(standEntry.species)!.farAlbedo.createView() },
            { binding: 7, resource: speciesGpu.get(standEntry.species)!.farNormal.createView() },
            { binding: 8, resource: sampler },
          ],
        })

    const atlasInfo = carpet
      ? (info: Float32Array): void => {
          const a = carpet.atlas
          const h0 = a.heights[0]!
          info[32] = a.yMin
          info[33] = a.yMax
          info[37] = a.tileM * 0.75 // bounding radius of a tile at scale 1
          info.set(
            [h0 - a.heights[1]!, h0 - a.heights[2]!, h0 - a.heights[3]!, h0 - a.heights[N_BAND]!],
            48,
          )
          info[52] = slots
          info[53] = 1
          info[54] = h0
          info[59] = CARPET_TEX
        }
      : (info: Float32Array): void => {
          const a = speciesGpu.get(standEntry.species)!.atlas
          const g = speciesGpu.get(standEntry.species)!
          info[32] = a.y0
          info[33] = a.y1
          info[34] = a.rXZ
          info[35] = a.cx
          info[36] = a.cz
          info[37] = Math.hypot(a.rXZ, (a.y1 - a.y0) / 2) * 1.03
          info[40] = g.sideSpan
          info[41] = g.topSpan
          info[52] = slots
          info[53] = 0
        }

    const indirectReset = new Uint32Array(8)
    return {
      carpet,
      capacity,
      nearCapacity,
      slots,
      infoBuffer,
      indirectBuffer,
      indirectReset,
      cullBindGroup,
      drawBindGroup,
      slotsPerFrame: 0,
      atlasInfo,
    }
  })

  const cardEntries = entries.filter((e) => e.carpet === null)
  const carpetEntries = entries.filter((e) => e.carpet !== null)

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  let carpetPipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(cardsSrc)
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const cardsLayout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkPipe = (
      name: string,
      mod: GPUShaderModule,
      layout: GPUPipelineLayout,
      vs: string,
      fs: string,
    ): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${name}`,
        layout,
        vertex: { module: mod, entryPoint: vs },
        fragment: { module: mod, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkPipe('near-cards', module, cardsLayout, 'vs_near', 'fs_near')
    farPipeline = mkPipe('far-cards', module, cardsLayout, 'vs_far', 'fs_far')
    if (carpetEntries.length > 0) {
      carpetPipeline = mkPipe(
        'carpet-tiles',
        ctx.shaders.module(carpetSrc),
        device.createPipelineLayout({ label: `${ctx.id}/carpet-pl`, bindGroupLayouts: [ctx.frame.layout, carpetBgl] }),
        'vs_carpet',
        'fs_carpet',
      )
    }
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
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

      const verts = ctx.params.topCard ? 12 : 6
      entries.forEach((entry, entryIndex) => {
        info.fill(0)
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.capacity
        info[38] = ctx.params.alphaRef
        info[39] = ctx.params.lodDistance
        info[42] = ctx.params.layerShade
        info[43] = ctx.params.bottomShade
        info[44] = entry.nearCapacity
        info[55] = ctx.params.carpetAlphaRef
        info[56] = ctx.params.carpetMaxSlope
        info[57] = ctx.params.carpetMergeLod
        info[58] = CARPET_MERGE_SPAN
        info[60] = ctx.params.carpetBandRef
        info[61] = ctx.params.carpetDepthShade
        entry.atlasInfo(info)
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        // A carpet draws one 6-vertex ground quad and never uses the near ring.
        entry.indirectReset[0] = entry.carpet ? 0 : verts
        entry.indirectReset[4] = entry.carpet ? 6 : verts
        device.queue.writeBuffer(entry.indirectBuffer, 0, entry.indirectReset)
        entry.slotsPerFrame = sideX * sideZ * entry.slots
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

      const pass = ctx.timing.renderPass(enc, 'slab-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near ring first: it owns the closest pixels, so the depth it writes
      // lets early-z reject the far ring's fragments behind it. The carpet is
      // an opaque ground surface, so it goes next and shields the far ring.
      pass.setPipeline(nearPipeline)
      for (const entry of cardEntries) {
        pass.setBindGroup(1, entry.drawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 0)
      }
      if (carpetEntries.length > 0) {
        pass.setPipeline(carpetPipeline)
        for (const entry of carpetEntries) {
          pass.setBindGroup(1, entry.drawBindGroup)
          pass.drawIndirect(entry.indirectBuffer, 16)
        }
      }
      pass.setPipeline(farPipeline)
      for (const entry of cardEntries) {
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

const mipLevels = (w: number, h: number, cap = 32): number =>
  Math.min(cap, Math.floor(Math.log2(Math.max(w, h))) + 1)

/**
 * Instance capacity for a carpet entry.
 *
 * A carpet has carpet_div^2 slots per cell and the entries sharing the grid
 * PARTITION the wetness axis, so each one gets whatever share of the nodes its
 * band happens to claim. The wetness field is damped on slopes, which pushes
 * the distribution well away from uniform: on the bog the dry band takes
 * roughly half the nodes and the wet band a tenth. Sizing all three for a third
 * would punch holes in one and waste megabytes on another, so this samples the
 * actual field over the stand and takes the worst camera position.
 */
function carpetCapacity(ctx: ExperimentContext<typeof PARAMS>, entry: StandSpecies, slots: number): number {
  const slotsPerM2 = slots / (CELL * CELL)
  const half = ctx.stand.radius
  const R = REGION_MAX
  const N = 96
  const step = (2 * half) / N
  const lo = (entry.wetCenter ?? 0.5) - (entry.wetWidth ?? 1) * 0.5
  const hi = lo + (entry.wetWidth ?? 1)
  const xs = new Float32Array(N)
  for (let i = 0; i < N; i++) xs[i] = -half + (i + 0.5) * step
  const inBand = new Uint8Array(N * N)
  for (let iz = 0; iz < N; iz++) {
    for (let ix = 0; ix < N; ix++) {
      const w = ctx.scene.scatter.wetness(xs[ix]!, xs[iz]!)
      inBand[iz * N + ix] = w >= lo && w < hi ? 1 : 0
    }
  }
  // Worst camera position: a species concentrated in one corner of the stand
  // peaks when the region disc sits on top of it.
  let best = 0
  const C = 13
  for (let cz = 0; cz < C; cz++) {
    for (let cx = 0; cx < C; cx++) {
      const px = -half + ((cx + 0.5) * (2 * half)) / C
      const pz = -half + ((cz + 0.5) * (2 * half)) / C
      let count = 0
      for (let iz = 0; iz < N; iz++) {
        const dz = xs[iz]! - pz
        for (let ix = 0; ix < N; ix++) {
          if (inBand[iz * N + ix] === 0) continue
          const dx = xs[ix]! - px
          if (dx * dx + dz * dz <= R * R) count++
        }
      }
      best = Math.max(best, count)
    }
  }
  return Math.ceil(best * step * step * slotsPerM2 * 1.2) + 4096
}

/** Upload the carpet band array textures and generate their mip chains. */
function uploadCarpet(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: CarpetAtlas): CarpetGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const mk = (name: string, side: number, data: Uint8Array<ArrayBuffer>): GPUTexture => {
    const tex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${name}`,
        size: [side, side, N_CARPET_LAYER],
        format: 'rgba8unorm',
        mipLevelCount: mipLevels(side, side),
        usage,
      },
      { species: speciesId, tag: `carpet-${name}` },
    )
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: side * 4, rowsPerImage: side }, [
      side,
      side,
      N_CARPET_LAYER,
    ])
    return tex
  }
  const albedo = mk('band-albedo', CARPET_TEX, atlas.albedo)
  const normal = mk('band-normal', CARPET_NRM, atlas.normal)

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/carpet-mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/carpet-mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo')
  const normalPipe = mkPipeline('fs_normal_cov')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/carpet-mipgen` })
  const genMips = (tex: GPUTexture, pipeline: GPURenderPipeline): void => {
    for (let layer = 0; layer < N_CARPET_LAYER; layer++) {
      for (let level = 1; level < tex.mipLevelCount; level++) {
        const viewOf = (m: number): GPUTextureView =>
          tex.createView({ dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1, baseMipLevel: m, mipLevelCount: 1 })
        const bg = device.createBindGroup({
          label: `${ctx.id}/carpet-mipgen-bg-${layer}-${level}`,
          layout: bgl,
          entries: [{ binding: 0, resource: viewOf(level - 1) }],
        })
        const pass = enc.beginRenderPass({
          label: `${ctx.id}/carpet-mipgen-${layer}-${level}`,
          colorAttachments: [{ view: viewOf(level), loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bg)
        pass.draw(3)
        pass.end()
      }
    }
  }
  genMips(albedo, albedoPipe)
  genMips(normal, normalPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedo, normal }
}

/** Upload atlas mip 0, generate the mip chains, build the tile table buffer. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: SlabAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST

  const mkTex = (
    name: string,
    w: number,
    h: number,
    format: GPUTextureFormat,
    levels: number,
    data: Uint8Array<ArrayBuffer>,
    bpp: number,
  ): GPUTexture => {
    const tex = ctx.res.createTexture(
      { label: `${ctx.id}/${speciesId}/${name}`, size: [w, h], format, mipLevelCount: levels, usage },
      { species: speciesId, tag: `slab-${name}` },
    )
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: w * bpp, rowsPerImage: h }, [w, h])
    return tex
  }

  const nearAlbedo = mkTex('near-albedo', atlas.nearW, atlas.nearH, 'rgba8unorm', NEAR_MIPS, atlas.nearAlbedo, 4)
  const nearNormal = mkTex(
    'near-normal',
    atlas.nearNW,
    atlas.nearNH,
    'rg8unorm',
    NEAR_MIPS - 1,
    atlas.nearNormal,
    2,
  )
  // The far atlas is small and is what distant plants minify into, so it gets
  // a full mip chain — that is what keeps the far ring cache friendly.
  const farLevels = mipLevels(atlas.farW, atlas.farH)
  const farAlbedo = mkTex('far-albedo', atlas.farW, atlas.farH, 'rgba8unorm', farLevels, atlas.farAlbedo, 4)
  const farNormal = mkTex(
    'far-normal',
    atlas.farNW,
    atlas.farNH,
    'rg8unorm',
    mipLevels(atlas.farNW, atlas.farNH),
    atlas.farNormal,
    2,
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
  const albedoPipe = mkPipeline('fs_albedo', 'rgba8unorm')
  const normalPipe = mkPipeline('fs_normal', 'rg8unorm')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < tex.mipLevelCount; level++) {
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
  for (const tex of [nearAlbedo, farAlbedo]) genMips(tex, albedoPipe)
  for (const tex of [nearNormal, farNormal]) genMips(tex, normalPipe)
  device.queue.submit([enc.finish()])

  // Tile table: two vec4 per tile — rect and (plane depth, texW, texH, isFar).
  const table = new Float32Array(N_TILES * 8)
  atlas.tiles.forEach((t, i) => {
    table.set([t.u0, t.v0, t.du, t.dv], i * 8)
    table.set([t.depth, t.texW, t.texH, t.far ? 1 : 0], i * 8 + 4)
  })
  const tileBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/tiles`,
      size: table.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'tile-table' },
  )
  device.queue.writeBuffer(tileBuffer, 0, table)

  // Slab plane separations drive the proxy quad's margins.
  let sideSpan = 0
  for (let k = 0; k < N_AZIM; k++) {
    sideSpan = Math.max(sideSpan, atlas.tiles[tileSide(k, 0)]!.depth - atlas.tiles[tileSide(k, N_LAYER - 1)]!.depth)
  }
  const topSpan = atlas.tiles[TILE_TOP0]!.depth - atlas.tiles[TILE_TOP0 + N_LAYER - 1]!.depth

  return { atlas, nearAlbedo, nearNormal, farAlbedo, farNormal, tileBuffer, sideSpan, topSpan }
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
  list.forEach((pl, i) => {
    const len = Math.hypot(pl[0], pl[1], pl[2]) || 1
    out[i * 4] = pl[0] / len
    out[i * 4 + 1] = pl[1] / len
    out[i * 4 + 2] = pl[2] / len
    out[i * 4 + 3] = pl[3] / len
  })
}
