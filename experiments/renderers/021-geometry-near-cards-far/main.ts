import cullSrc from './shaders/cull.wgsl'
import plantsSrc from './shaders/plants.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import {
  BANDS,
  CARPET_PX,
  FAR_LAYERS,
  FAR_PX,
  loadSpeciesCloud,
  loadSpeciesTile,
  NEAR_LAYERS,
  NEAR_PX,
  N_CARDS,
  type CardCloud,
  type CarpetTile,
} from './bake.ts'
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
 * Card cloud near, impostor far.
 *
 * Startup: per species one baked artifact is loaded (or baked in-browser from
 * the raw GCMESH1 mesh) holding a NEAR texture array — three vertical card
 * planes carrying one azimuth wedge of the source geometry each, plus a
 * top card — and a FAR array — the usual 8-azimuth + top impostor sheet. Both
 * are cut from one capture box, so a plant is the same size, in the same
 * place, lit the same way, in either LOD.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into TWO
 * indirect draws by distance (jittered per plant). Near plants draw 4 quads of
 * real 3D structure; far plants draw the 2 quads a billboard would. Cost is
 * O(visible region), independent of the stand's plant count, and the near
 * bucket is bounded by cloudRadius no matter how deep the field is.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const CLOUD_R_MAX = 36 // keep equal to the manifest's cloudRadius max
const SHELL_R_MAX = 24 // keep equal to the manifest's shellRadius max
const NEAR_MIPS = Math.floor(Math.log2(NEAR_PX)) + 1
const FAR_MIPS = Math.floor(Math.log2(FAR_PX)) + 1
const CARPET_MIPS = Math.floor(Math.log2(CARPET_PX)) + 1
const TOP_FRAC = 0.52
const SINK_FRAC = 0.05 // card bottom buried, as a fraction of plant height
const INFO_FLOATS = 144
const BOUNDS_OFFSET = 48 // float index of EntryInfo.tile_bounds
const BANDS_OFFSET = 100 // float index of EntryInfo.card_bands
const SHELL_OFFSET = 136 // float index of EntryInfo.shell_span
const CLOUD_VERTS = N_CARDS * BANDS * 6
/**
 * Alpha reference for carpet tiles, INSTEAD of the params' alphaRef. A mat is a
 * closed surface and must not dissolve with distance: tile coverage is ~80%
 * solid up close, but the mip chain pulls it toward the whole tile's mean, and
 * at the 0.4 grass reference entire distant tiles fail the test and punch
 * tile-shaped holes into the carpet. Low reference = MORE hard-edged opaque
 * coverage (a depth-writing occluder), which is the opposite of dithering.
 */
const CARPET_ALPHA_REF = 0.06

interface SpeciesGpu {
  cloud: CardCloud | null
  /** Carpet species: one top-down tile capture with a height channel. */
  tile: CarpetTile | null
  /**
   * Carpet species: the mean shell fraction of the visible surface, i.e. where
   * a tile's height histogram sits inside [hLo, hHi]. The far LOD shades its
   * single quad with it so the flat quad and the shell stack it replaces have
   * the same mean brightness.
   */
  meanShellF: number
  nearAlbedo: GPUTexture
  nearNormal: GPUTexture
  farAlbedo: GPUTexture
  farNormal: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  /** Mat species (stand carpetDiv > 0): relief shells, never azimuth cards. */
  isCarpet: boolean
  /** Candidate slots per cell that the cull MUST evaluate (carpetDiv² for a mat). */
  enumSlots: number
  cloudCapacity: number
  farCapacity: number
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  texBindGroup: GPUBindGroup
  cloudInstBindGroup: GPUBindGroup
  farInstBindGroup: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/card-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 16,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species artifacts (sequential: the poa bake transiently needs ~500MB) -
  // Two artifact kinds. An upright plant gets the card cloud + impostor sheet;
  // a CARPET species gets one top-down capture of its own periodic tile with a
  // height channel, which is both far smaller and far denser per screen texel —
  // for a mat, 8 azimuth views and 3 wedge cards are storage spent on a
  // silhouette that a 7cm cushion does not have.
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    if ((entry.carpetDiv ?? 0) > 0) {
      const tile = await loadSpeciesTile(ctx, entry.species)
      speciesGpu.set(entry.species, uploadTile(ctx, entry.species, tile))
    } else {
      const cloud = await loadSpeciesCloud(ctx, entry.species)
      speciesGpu.set(entry.species, uploadCloud(ctx, entry.species, cloud))
    }
  }

  // --- bind group layouts ----------------------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const texBgl = device.createBindGroupLayout({
    label: `${ctx.id}/tex-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const instBgl = device.createBindGroupLayout({
    label: `${ctx.id}/inst-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
  })

  // Never more cells than the stand itself owns: outside its radius the
  // scatter is empty, so capacity sized for a 128m region on a ±96m stand is
  // dead VRAM. On the ±128m stands this is exactly the old number.
  const standSide = Math.floor(ctx.stand.radius / CELL) * 2 + 1
  const sideMax = Math.min(Math.ceil((2 * REGION_MAX) / CELL) + 1, standSide)
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const isCarpet = (standEntry.carpetDiv ?? 0) > 0
    // TWO DIFFERENT NUMBERS, and conflating them silently drops plants:
    //  * enumSlots — how many candidate slots the cull must EVALUATE per cell.
    //    All of them, always (carpetDiv² = 484 for the bog's life-size moss).
    //  * expected survivors — all the instance buffer has to hold. For a
    //    zone-partitioned carpet that is roughly the entry's wetness share;
    //    sizing for all 484 would waste ~3x the memory.
    const enumSlots = standEntrySlots(standEntry)
    const share = carpetShare(standEntry)
    // Far bucket: worst case is the whole region (cloudRadius = 0).
    const farCapacity = isCarpet
      ? Math.ceil(sideMax * sideMax * enumSlots * share * 1.02) + 1024
      : Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    // Near bucket: a disc of cloudRadius * scaleMax * jitterMax around the
    // camera — or, for a mat, the shell radius times its own tiles per m².
    const nearR = isCarpet ? SHELL_R_MAX * 1.05 : CLOUD_R_MAX * standEntry.scaleMax * 1.15
    const perM2 = isCarpet ? (enumSlots / (CELL * CELL)) * share : density
    const cloudCapacity = Math.ceil(Math.PI * nearR * nearR * perM2 * 1.15) + 2048
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const mkInst = (name: string, capacity: number): GPUBuffer =>
      ctx.res.createBuffer(
        { label: `${ctx.id}/${name}-${entryIndex}`, size: capacity * 16, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: `${name}-instances` },
      )
    const cloudInst = mkInst('cloud', cloudCapacity)
    const farInst = mkInst('far', farCapacity)
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 32,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const mkInstBg = (name: string, buffer: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/${name}-inst-bg-${entryIndex}`,
        layout: instBgl,
        entries: [{ binding: 0, resource: { buffer } }],
      })
    return {
      gpu,
      isCarpet,
      enumSlots,
      cloudCapacity,
      farCapacity,
      infoBuffer,
      indirectBuffer,
      cullBindGroup: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: cloudInst } },
          { binding: 2, resource: { buffer: farInst } },
          { binding: 3, resource: { buffer: indirectBuffer } },
        ],
      }),
      texBindGroup: device.createBindGroup({
        label: `${ctx.id}/tex-bg-${entryIndex}`,
        layout: texBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: gpu.nearAlbedo.createView({ dimension: '2d-array' }) },
          { binding: 2, resource: gpu.nearNormal.createView({ dimension: '2d-array' }) },
          { binding: 3, resource: gpu.farAlbedo.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: gpu.farNormal.createView({ dimension: '2d-array' }) },
          { binding: 5, resource: sampler },
        ],
      }),
      cloudInstBindGroup: mkInstBg('cloud', cloudInst),
      farInstBindGroup: mkInstBg('far', farInst),
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let cloudPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  let carpetPipeline!: GPURenderPipeline
  let carpetFarPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(plantsSrc)
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/plants-pl`,
      bindGroupLayouts: [ctx.frame.layout, texBgl, instBgl],
    })
    const mk = (name: string, vs: string, fs: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${name}`,
        layout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    cloudPipeline = mk('cloud', 'vs_cloud', 'fs_cloud')
    farPipeline = mk('far', 'vs_far', 'fs_far')
    carpetPipeline = mk('carpet', 'vs_carpet', 'fs_carpet')
    carpetFarPipeline = mk('carpet-far', 'vs_carpet_far', 'fs_carpet')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(8)

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

      const withTop = ctx.params.topCard
      const shells = Math.max(2, Math.round(ctx.params.shellCount))
      entries.forEach((entry, entryIndex) => {
        const c = entry.gpu.cloud
        const t = entry.gpu.tile
        info.fill(0)
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.farCapacity
        info[42] = entry.cloudCapacity
        if (t) {
          // Carpet: the capture box is the tile's own square, and the fields the
          // card path uses for its bounding sphere describe the tile instead.
          const rXZ = t.tileM * 0.7072
          info[32] = t.y0
          info[33] = t.y1
          info[34] = rXZ
          info[37] = Math.max(rXZ, (t.y1 - t.y0) / 2) * 1.05
          info[SHELL_OFFSET] = t.hLo
          info[SHELL_OFFSET + 1] = t.hMid
          info[SHELL_OFFSET + 2] = t.hHi
          info[SHELL_OFFSET + 3] = shells
          info[SHELL_OFFSET + 4] = ctx.params.shellRadius
          info[SHELL_OFFSET + 5] = CARPET_ALPHA_REF
          info[SHELL_OFFSET + 6] = ctx.params.cushionShade
          info[SHELL_OFFSET + 7] = entry.gpu.meanShellF
        } else if (c) {
          info[32] = c.y0
          info[33] = c.y1
          info[34] = c.rXZ
          info[35] = c.cx
          info[36] = c.cz
          info[37] = Math.max(c.rXZ, (c.y1 - c.y0) / 2) * 1.05
          info[38] = ctx.params.alphaRef
          info[39] = TOP_FRAC
          info[40] = ctx.params.bottomShade
          info[41] = ctx.params.cloudRadius
          info[43] = ctx.params.translucency
          info[44] = ctx.params.tintJitter
          info[45] = (c.y1 - c.y0) * SINK_FRAC
          info.set(c.bounds, BOUNDS_OFFSET)
          info.set(c.cardBands, BANDS_OFFSET)
        }
        // A mat draws `shells` ground-parallel quads near and ONE far; it never
        // draws a camera-facing card, so `topCard` does not apply to it.
        indirectReset[0] = entry.isCarpet ? shells * 6 : CLOUD_VERTS + (withTop ? 6 : 0)
        indirectReset[4] = entry.isCarpet ? 6 : withTop ? 12 : 6
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
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

      // One pass, near bucket first: the card cloud lays down the depth that
      // early-z then rejects the impostor field against.
      const pass = ctx.timing.renderPass(enc, 'plants', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Four pipelines, one pass: cards near/far for upright species, relief
      // shells near / one flat quad far for mats. Splitting them keeps the card
      // path's interpolant set (and its cost) exactly as it was.
      const bucket = (pipeline: GPURenderPipeline, carpet: boolean, near: boolean): void => {
        let bound = false
        for (const entry of entries) {
          if (entry.isCarpet !== carpet) continue
          if (!bound) {
            pass.setPipeline(pipeline)
            bound = true
          }
          pass.setBindGroup(1, entry.texBindGroup)
          pass.setBindGroup(2, near ? entry.cloudInstBindGroup : entry.farInstBindGroup)
          pass.drawIndirect(entry.indirectBuffer, near ? 0 : 16)
        }
      }
      bucket(cloudPipeline, false, true)
      bucket(carpetPipeline, true, true)
      bucket(farPipeline, false, false)
      bucket(carpetFarPipeline, true, false)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/**
 * Fraction of a carpet entry's grid nodes it can be expected to claim. The
 * three Sphagnum states PARTITION the wetness axis into thirds, but the
 * wetness field is not uniformly distributed and a camera can sit inside one
 * zone: measured over 110m discs on the bog, the worst local share of a
 * nominal 1/3 band is 0.58, so `wetWidth * 1.9` is the honest bound. Non-carpet
 * entries never use this.
 */
function carpetShare(entry: StandSpecies): number {
  return Math.min(1, (entry.wetWidth ?? 1) * 1.9)
}

/**
 * Upload one carpet tile capture (albedo+coverage, normal+height) and mip it.
 * Both far slots are bound to the same textures: a mat's far LOD is the same
 * tile drawn as one quad, so there is no second image to store — which is why
 * a carpet species costs ~3 MB where an upright plant costs ~13 MB here.
 */
function uploadTile(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, tile: CarpetTile): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const mkTex = (name: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${name}`,
        size: [CARPET_PX, CARPET_PX, 1],
        format: 'rgba8unorm',
        mipLevelCount: CARPET_MIPS,
        usage,
      },
      { species: speciesId, tag: name },
    )
  const albedo = mkTex('carpet-albedo')
  const nrmh = mkTex('carpet-normal-height')
  const write = (tex: GPUTexture, data: Uint8Array<ArrayBuffer>): void => {
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: CARPET_PX * 4, rowsPerImage: CARPET_PX }, [
      CARPET_PX,
      CARPET_PX,
      1,
    ])
  }
  write(albedo, tile.albedo)
  write(nrmh, tile.nrmh)

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-tile-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      // Coverage-weighted for albedo; the normal+height plane is a plain box
      // filter, which is exactly right here: the normals are plain unit
      // vectors in one hemisphere (not octahedral, which does NOT average) and
      // the height is a linear quantity.
      fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    })
  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen-tile` })
  const gen = (tex: GPUTexture, pipeline: GPURenderPipeline): void => {
    const view = (level: number): GPUTextureView =>
      tex.createView({ dimension: '2d', baseArrayLayer: 0, arrayLayerCount: 1, baseMipLevel: level, mipLevelCount: 1 })
    for (let level = 1; level < CARPET_MIPS; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-tile-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: view(level - 1) }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-tile-${level}`,
        colorAttachments: [{ view: view(level), loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  gen(albedo, mkPipeline('fs_albedo'))
  gen(nrmh, mkPipeline('fs_normal'))
  device.queue.submit([enc.finish()])

  // Where the visible surface sits inside the shell span, area-weighted: the
  // far LOD's one quad shades itself with this so it matches the mean of the
  // stack it stands in for (histogram, not a guess).
  const span = tile.y1 - tile.y0
  const lo = (tile.hLo - tile.y0) / span
  const hi = (tile.hHi - tile.y0) / span
  let sum = 0
  let covered = 0
  for (let i = 0; i < CARPET_PX * CARPET_PX; i++) {
    if (tile.albedo[i * 4 + 3]! < 8) continue
    covered++
    sum += Math.min(1, Math.max(0, (tile.nrmh[i * 4 + 3]! / 255 - lo) / Math.max(hi - lo, 1e-6)))
  }
  const meanShellF = covered > 0 ? sum / covered : 0.5

  return { cloud: null, tile, meanShellF, nearAlbedo: albedo, nearNormal: nrmh, farAlbedo: albedo, farNormal: nrmh }
}

/** Upload both card arrays' mip 0 and generate their mip chains on the GPU. */
function uploadCloud(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, cloud: CardCloud): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST

  const mkTex = (name: string, px: number, layers: number, format: GPUTextureFormat, mips: number): GPUTexture =>
    ctx.res.createTexture(
      { label: `${ctx.id}/${speciesId}/${name}`, size: [px, px, layers], format, mipLevelCount: mips, usage },
      { species: speciesId, tag: name },
    )
  const nearAlbedo = mkTex('near-albedo', NEAR_PX, NEAR_LAYERS, 'rgba8unorm', NEAR_MIPS)
  const nearNormal = mkTex('near-normal', NEAR_PX, NEAR_LAYERS, 'rg8unorm', NEAR_MIPS)
  const farAlbedo = mkTex('far-albedo', FAR_PX, FAR_LAYERS, 'rgba8unorm', FAR_MIPS)
  const farNormal = mkTex('far-normal', FAR_PX, FAR_LAYERS, 'rg8unorm', FAR_MIPS)

  const write = (tex: GPUTexture, data: Uint8Array<ArrayBuffer>, px: number, layers: number, bpp: number): void => {
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: px * bpp, rowsPerImage: px }, [px, px, layers])
  }
  write(nearAlbedo, cloud.nearAlbedo, NEAR_PX, NEAR_LAYERS, 4)
  write(nearNormal, cloud.nearNormal, NEAR_PX, NEAR_LAYERS, 2)
  write(farAlbedo, cloud.farAlbedo, FAR_PX, FAR_LAYERS, 4)
  write(farNormal, cloud.farNormal, FAR_PX, FAR_LAYERS, 2)

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
  const genMips = (tex: GPUTexture, pipeline: GPURenderPipeline, layers: number, mips: number): void => {
    for (let layer = 0; layer < layers; layer++) {
      const view = (level: number): GPUTextureView =>
        tex.createView({
          dimension: '2d',
          baseArrayLayer: layer,
          arrayLayerCount: 1,
          baseMipLevel: level,
          mipLevelCount: 1,
        })
      for (let level = 1; level < mips; level++) {
        const bg = device.createBindGroup({
          label: `${ctx.id}/mipgen-bg-${layer}-${level}`,
          layout: bgl,
          entries: [{ binding: 0, resource: view(level - 1) }],
        })
        const pass = enc.beginRenderPass({
          label: `${ctx.id}/mipgen-${layer}-${level}`,
          colorAttachments: [{ view: view(level), loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bg)
        pass.draw(3)
        pass.end()
      }
    }
  }
  genMips(nearAlbedo, albedoPipe, NEAR_LAYERS, NEAR_MIPS)
  genMips(nearNormal, normalPipe, NEAR_LAYERS, NEAR_MIPS)
  genMips(farAlbedo, albedoPipe, FAR_LAYERS, FAR_MIPS)
  genMips(farNormal, normalPipe, FAR_LAYERS, FAR_MIPS)
  device.queue.submit([enc.finish()])

  return { cloud, tile: null, meanShellF: 0, nearAlbedo, nearNormal, farAlbedo, farNormal }
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
