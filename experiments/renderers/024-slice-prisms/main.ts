import cullSrc from './shaders/cull.wgsl'
import prismsSrc from './shaders/prisms.wgsl'
import carpetCullSrc from './shaders/carpet_cull.wgsl'
import carpetSrc from './shaders/carpet.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, MIP_LEVELS, loadSpeciesPrisms, type PrismAtlas } from './bake.ts'
import {
  CARPET_H,
  CARPET_MARGIN,
  CARPET_MIPS,
  CARPET_TILE_PX,
  CARPET_W,
  N_SLICE,
  loadSpeciesCarpet,
  type CarpetAtlas,
} from './carpet.ts'
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
 * Slice prisms. Startup: each stand species is loaded (or baked in-browser
 * from the raw mesh) in ONE of two shapes and uploaded + mipped:
 *
 *  * upright plants — the prism atlas (4 baked azimuths x 3 vertical depth
 *    slabs mirrored to 8 views, 8 merged cards, 3 horizontal height slabs,
 *    1 merged top).
 *  * CARPET species (stand carpetDiv > 0, i.e. the bog's Sphagnum) — the
 *    carpet atlas: five horizontal slices of ONE periodic tile plus a merged
 *    top, cropped to the tile square. A mat has no silhouette, so every texel
 *    an azimuth would have cost goes into tile detail instead (0.35mm/texel).
 *
 * Per frame: one compute pass per entry evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into TWO
 * indirect draws — near plants as three depth prisms (or a carpet tile as its
 * five-slice stack), far ones as a single merged card. Per-frame cost is
 * O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const SLAB_DIST_MAX = 20 // keep equal to the manifest's slabDist max
const SLICE_DIST_MAX = 24 // keep equal to the manifest's carpetSliceDist max
const INFO_FLOATS = 256
/**
 * Floats 0..47 of EntryInfo change every frame (frustum, region, params);
 * 48..255 are the baked atlas tables (prism depths, top heights, per-tile UV
 * and geometry boxes) and are uploaded once at startup.
 */
const INFO_DYNAMIC_FLOATS = 48
/** CarpetInfo is small enough to rewrite whole; slice heights live at 52..59. */
const CARPET_INFO_FLOATS = 60
// Horizontal prisms only earn their fill from genuinely above the canopy top
// (see the signed elevation in prisms.wgsl); below this the canopy is
// described entirely by the vertical prisms and a lid would only read as a
// pale cutout floating among them.
const TOP_ELEV_LO = 0.15
const TOP_ELEV_HI = 0.45
/**
 * Alpha-test reference for carpet tiles, INSTEAD of the params' alphaRef. A mat
 * is a closed surface and must not dissolve with distance: tile coverage is
 * ~80% up close, but the mip chain pulls each slice toward its own mean (a
 * fifth of the tile), and at the 0.4 grass reference whole distant tiles then
 * fail the test and punch holes in the carpet. Low reference = MORE hard-edged
 * opaque coverage, which is the opposite of dithering it away.
 */
const CARPET_ALPHA_REF = 0.06

interface SpeciesGpu {
  albedoTex: GPUTexture
  normalTex: GPUTexture
}

interface PrismEntry {
  kind: 'prism'
  nearCapacity: number
  farCapacity: number
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  nearDrawBindGroup: GPUBindGroup
  farDrawBindGroup: GPUBindGroup
  atlas: PrismAtlas
  slotsPerFrame: number
}

interface CarpetEntry {
  kind: 'carpet'
  nearCapacity: number
  farCapacity: number
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  nearDrawBindGroup: GPUBindGroup
  farDrawBindGroup: GPUBindGroup
  atlas: CarpetAtlas
  /** Constant tile scale of this carpet (grid step / tile footprint). */
  tileScale: number
  /** carpetDiv^2 rounded up to a multiple of 64 (see carpet_cull.wgsl). */
  paddedSlots: number
  slotsPerFrame: number
}

type EntryGpu = PrismEntry | CarpetEntry

const isCarpetEntry = (e: StandSpecies): boolean => (e.carpetDiv ?? 0) > 0

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

  // --- species atlases (sequential: a moss bake transiently needs ~500MB) ---
  const prismAtlases = new Map<string, { atlas: PrismAtlas; gpu: SpeciesGpu }>()
  const carpetAtlases = new Map<string, { atlas: CarpetAtlas; gpu: SpeciesGpu }>()
  for (const entry of ctx.stand.species) {
    if (isCarpetEntry(entry)) {
      if (carpetAtlases.has(entry.species)) continue
      const atlas = await loadSpeciesCarpet(ctx, entry.species)
      carpetAtlases.set(entry.species, {
        atlas,
        gpu: uploadAtlas(ctx, entry.species, 'carpet', {
          width: CARPET_W,
          height: CARPET_H,
          mips: CARPET_MIPS,
          albedo: atlas.albedo,
          normalOct: atlas.normalOct,
        }),
      })
    } else {
      if (prismAtlases.has(entry.species)) continue
      const atlas = await loadSpeciesPrisms(ctx, entry.species)
      prismAtlases.set(entry.species, {
        atlas,
        gpu: uploadAtlas(ctx, entry.species, 'prism', {
          width: ATLAS,
          height: ATLAS,
          mips: MIP_LEVELS,
          albedo: atlas.albedo,
          normalOct: atlas.normalOct,
        }),
      })
    }
  }

  // --- bind group layouts / pipelines ---------------------------------------
  // Both paths use the same LAYOUTS; only the WGSL structs behind them differ
  // (16B PlantInst vs a 4B packed grid node).
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
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL
  // Cells the stand itself owns — outside them the scatter is empty, so a
  // carpet's instance capacity must not be sized for a bigger region.
  const standSide = Math.floor(ctx.stand.radius / CELL) * 2 + 1
  const carpetSide = Math.min(sideMax, standSide)

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 32,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const bindGroups = (
      near: GPUBuffer,
      far: GPUBuffer,
      gpu: SpeciesGpu,
    ): Pick<PrismEntry, 'cullBindGroup' | 'nearDrawBindGroup' | 'farDrawBindGroup'> => {
      const draw = (instances: GPUBuffer, tag: string): GPUBindGroup =>
        device.createBindGroup({
          label: `${ctx.id}/${tag}-bg-${entryIndex}`,
          layout: drawBgl,
          entries: [
            { binding: 0, resource: { buffer: infoBuffer } },
            { binding: 1, resource: { buffer: instances } },
            { binding: 2, resource: gpu.albedoTex.createView() },
            { binding: 3, resource: gpu.normalTex.createView() },
            { binding: 4, resource: sampler },
          ],
        })
      return {
        cullBindGroup: device.createBindGroup({
          label: `${ctx.id}/cull-bg-${entryIndex}`,
          layout: cullBgl,
          entries: [
            { binding: 0, resource: { buffer: infoBuffer } },
            { binding: 1, resource: { buffer: near } },
            { binding: 2, resource: { buffer: far } },
            { binding: 3, resource: { buffer: indirectBuffer } },
          ],
        }),
        nearDrawBindGroup: draw(near, 'near'),
        farDrawBindGroup: draw(far, 'far'),
      }
    }

    if (isCarpetEntry(standEntry)) {
      const { atlas, gpu } = carpetAtlases.get(standEntry.species)!
      const carpetDiv = standEntry.carpetDiv!
      // TWO DIFFERENT NUMBERS. `slots` is how many candidate nodes the cull
      // must EVALUATE per cell — carpetDiv^2 (484 at life size), NEVER the
      // 128-slot scatter budget, or three quarters of the mat never exists.
      // Capacity only has to hold the SURVIVORS: the three Sphagnum states
      // partition the wetness axis, so one entry owns roughly `wetWidth` of
      // the nodes (measured on the bog: 0.16 / 0.50 / 0.34 of 1.16M nodes for
      // seed 42). 2.2x that is the headroom.
      const slots = standEntrySlots(standEntry)
      const paddedSlots = Math.ceil(slots / 64) * 64
      const band = Math.min(1, (standEntry.wetWidth ?? 1) * 2.2)
      // A tile is 4 BYTES (cell | slot | quarter turn), not a 16B instance —
      // see carpet_cull.wgsl. That is what lets a 1.1M-tile mat spend its VRAM
      // on the atlas instead of on a list of positions.
      const farCapacity = Math.ceil(carpetSide * carpetSide * slots * band * 1.06) + 4096
      const nodesPerM2 = slots / (CELL * CELL)
      const nearCapacity = Math.ceil(Math.PI * SLICE_DIST_MAX ** 2 * nodesPerM2 * band * 1.3) + 1024
      const nearBuffer = ctx.res.createBuffer(
        { label: `${ctx.id}/near-${entryIndex}`, size: nearCapacity * 4, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: 'stack-tiles' },
      )
      const farBuffer = ctx.res.createBuffer(
        { label: `${ctx.id}/far-${entryIndex}`, size: farCapacity * 4, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: 'card-tiles' },
      )
      return {
        kind: 'carpet',
        nearCapacity,
        farCapacity,
        infoBuffer,
        indirectBuffer,
        atlas,
        // The stand's own constant scale: a tile exactly fills its grid step.
        // (carpetScale() is not exported from @harness, so it is recomputed
        // here from the baked tile size — same expression.)
        tileScale: CELL / carpetDiv / atlas.tileM,
        paddedSlots,
        slotsPerFrame: 0,
        ...bindGroups(nearBuffer, farBuffer, gpu),
      }
    }

    const { atlas, gpu } = prismAtlases.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const farCapacity = Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    // The prism list only ever holds the disc of radius slabDist around the
    // camera — that is what keeps the expensive LOD bounded no matter how far
    // the region reaches or how many plants the stand has.
    const nearCapacity = Math.ceil(Math.PI * (SLAB_DIST_MAX * standEntry.scaleMax) ** 2 * density * 1.3) + 1024
    const nearBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/near-${entryIndex}`, size: nearCapacity * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'prism-instances' },
    )
    const farBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/far-${entryIndex}`, size: farCapacity * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'card-instances' },
    )
    // Baked atlas tables never change — upload them once, not every frame.
    const tail = new Float32Array(INFO_FLOATS - INFO_DYNAMIC_FLOATS)
    tail.set(atlas.slabDepth, 0)
    tail.set(atlas.topHeight, 12)
    tail.set(atlas.uvRect, 16)
    tail.set(atlas.geoRect, 112)
    device.queue.writeBuffer(infoBuffer, INFO_DYNAMIC_FLOATS * 4, tail)

    return {
      kind: 'prism',
      nearCapacity,
      farCapacity,
      infoBuffer,
      indirectBuffer,
      atlas,
      slotsPerFrame: 0,
      ...bindGroups(nearBuffer, farBuffer, gpu),
    }
  })

  const hasPrisms = entries.some((e) => e.kind === 'prism')
  const hasCarpets = entries.some((e) => e.kind === 'carpet')

  let cullPipeline: GPUComputePipeline | null = null
  let nearPipeline: GPURenderPipeline | null = null
  let farPipeline: GPURenderPipeline | null = null
  let carpetCullPipeline: GPUComputePipeline | null = null
  let carpetNearPipeline: GPURenderPipeline | null = null
  let carpetFarPipeline: GPURenderPipeline | null = null
  const build = (): void => {
    const compute = (label: string, src: typeof cullSrc, entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({
        label: `${ctx.id}/${label}`,
        layout: device.createPipelineLayout({
          label: `${ctx.id}/${label}-pl`,
          bindGroupLayouts: [ctx.frame.layout, cullBgl],
        }),
        compute: { module: ctx.shaders.module(src), entryPoint },
      })
    const drawPl = device.createPipelineLayout({
      label: `${ctx.id}/draw-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkDraw = (label: string, src: typeof prismsSrc, vs: string, fs: string): GPURenderPipeline => {
      const module = ctx.shaders.module(src)
      return device.createRenderPipeline({
        label: `${ctx.id}/${label}`,
        layout: drawPl,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    }
    if (hasPrisms) {
      cullPipeline = compute('cull', cullSrc, 'cs_cull')
      nearPipeline = mkDraw('prisms-near', prismsSrc, 'vs_near', 'fs_main')
      farPipeline = mkDraw('prisms-far', prismsSrc, 'vs_far', 'fs_main')
    }
    if (hasCarpets) {
      carpetCullPipeline = compute('carpet-cull', carpetCullSrc, 'cs_carpet_cull')
      carpetNearPipeline = mkDraw('carpet-near', carpetSrc, 'vs_carpet_near', 'fs_carpet')
      carpetFarPipeline = mkDraw('carpet-far', carpetSrc, 'vs_carpet_far', 'fs_carpet')
    }
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_DYNAMIC_FLOATS)
  const carpetInfo = new Float32Array(CARPET_INFO_FLOATS)
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
      const camHeight = cam.y - ctx.scene.terrain.height(cam.x, cam.z)

      const tops = ctx.params.topCards
      entries.forEach((entry, entryIndex) => {
        if (entry.kind === 'carpet') {
          const a = entry.atlas
          const s = entry.tileScale
          carpetInfo.set(planes, 0)
          carpetInfo[24] = x0
          carpetInfo[25] = z0
          carpetInfo[26] = sideX
          carpetInfo[27] = sideZ
          carpetInfo[28] = ctx.seed
          carpetInfo[29] = entryIndex
          carpetInfo[30] = R
          carpetInfo[31] = entry.nearCapacity
          carpetInfo[32] = entry.farCapacity
          carpetInfo[33] = entry.paddedSlots
          carpetInfo[34] = ctx.params.carpetSliceDist
          carpetInfo[35] = CARPET_ALPHA_REF
          carpetInfo[36] = a.tileM
          carpetInfo[37] = s
          carpetInfo[38] = a.y0
          carpetInfo[39] = a.y1
          carpetInfo[40] = CARPET_TILE_PX / CARPET_W
          carpetInfo[41] = CARPET_TILE_PX / CARPET_H
          carpetInfo[42] = CARPET_MARGIN / CARPET_TILE_PX
          carpetInfo[43] = N_SLICE
          carpetInfo[44] = ctx.params.carpetAO
          // Tile bounding sphere from the FOOTPRINT (never from height_scale:
          // a Sphagnum tile is 0.07m tall and 0.24m wide).
          carpetInfo[45] = (a.tileM * 0.7072 + (a.y1 - a.y0)) * s * 1.05
          carpetInfo[46] = camHeight
          carpetInfo[47] = ctx.params.carpetReliefRef
          carpetInfo[48] = ctx.params.carpetNormalGain
          carpetInfo.set(a.sliceH, 52)
          device.queue.writeBuffer(entry.infoBuffer, 0, carpetInfo)
          // 5 relief slices + the base card that guarantees coverage.
          indirectReset[0] = 6 * (N_SLICE + 1)
          indirectReset[1] = 0
          indirectReset[4] = 6
          indirectReset[5] = 0
          device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
          entry.slotsPerFrame = sideX * sideZ * entry.paddedSlots
          return
        }

        const a = entry.atlas
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.nearCapacity
        info[32] = entry.farCapacity
        info[33] = a.y0
        info[34] = a.y1
        info[35] = a.rXZ
        info[36] = a.cx
        info[37] = a.cz
        info[38] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info[39] = ctx.params.alphaRef
        info[40] = ctx.params.bottomShade
        info[41] = ctx.params.canopyShade
        info[42] = ctx.params.slabDist
        info[43] = TOP_ELEV_LO
        info[44] = TOP_ELEV_HI
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        indirectReset[0] = tops ? 36 : 18
        indirectReset[1] = 0
        indirectReset[4] = tops ? 12 : 6
        indirectReset[5] = 0
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setBindGroup(0, ctx.frame.bindGroup)
      if (cullPipeline) {
        cull.setPipeline(cullPipeline)
        for (const entry of entries) {
          if (entry.kind !== 'prism' || entry.slotsPerFrame === 0) continue
          cull.setBindGroup(1, entry.cullBindGroup)
          cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
        }
      }
      if (carpetCullPipeline) {
        cull.setPipeline(carpetCullPipeline)
        for (const entry of entries) {
          if (entry.kind !== 'carpet' || entry.slotsPerFrame === 0) continue
          cull.setBindGroup(1, entry.cullBindGroup)
          cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
        }
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'prisms', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near buckets first: they are the closest geometry in the frame, so the
      // depth they lay down lets early-z reject the merged cards behind them.
      const drawGroup = (
        pipeline: GPURenderPipeline | null,
        kind: EntryGpu['kind'],
        bucket: 'near' | 'far',
      ): void => {
        if (!pipeline) return
        pass.setPipeline(pipeline)
        for (const entry of entries) {
          if (entry.kind !== kind) continue
          pass.setBindGroup(1, bucket === 'near' ? entry.nearDrawBindGroup : entry.farDrawBindGroup)
          pass.drawIndirect(entry.indirectBuffer, bucket === 'near' ? 0 : 16)
        }
      }
      drawGroup(nearPipeline, 'prism', 'near')
      drawGroup(carpetNearPipeline, 'carpet', 'near')
      drawGroup(farPipeline, 'prism', 'far')
      drawGroup(carpetFarPipeline, 'carpet', 'far')
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

interface AtlasUpload {
  width: number
  height: number
  mips: number
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
}

/** Upload atlas mip 0 and generate the mip chain on the GPU. */
function uploadAtlas(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  kind: 'prism' | 'carpet',
  src: AtlasUpload,
): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [src.width, src.height],
      format: 'rgba8unorm',
      mipLevelCount: src.mips,
      usage,
    },
    { species: speciesId, tag: `${kind}-albedo` },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [src.width, src.height],
      format: 'rg8unorm',
      mipLevelCount: src.mips,
      usage,
    },
    { species: speciesId, tag: `${kind}-normal` },
  )
  device.queue.writeTexture(
    { texture: albedoTex },
    src.albedo,
    { bytesPerRow: src.width * 4, rowsPerImage: src.height },
    [src.width, src.height],
  )
  device.queue.writeTexture(
    { texture: normalTex },
    src.normalOct,
    { bytesPerRow: src.width * 2, rowsPerImage: src.height },
    [src.width, src.height],
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
    for (let level = 1; level < src.mips; level++) {
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
  genMips(albedoTex, albedoPipe)
  genMips(normalTex, normalPipe)
  device.queue.submit([enc.finish()])

  return { albedoTex, normalTex }
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
