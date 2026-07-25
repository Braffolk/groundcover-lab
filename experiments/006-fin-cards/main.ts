import cullSrc from './shaders/cull.wgsl'
import shaderSrc from './shaders/fins.wgsl'
import mossSrc from './shaders/moss.wgsl'
import { BAKE_VERSION, bakeFins, packFins, unpackFins, ATLAS_H, ATLAS_W, MIPS, type BakedFins } from './bake.ts'
import {
  MOSS_ALB_MIPS,
  MOSS_BAKE_VERSION,
  MOSS_LEVELS,
  MOSS_NRM_MIPS,
  MOSS_NRM_TILE,
  MOSS_TILE,
  bakeMoss,
  packMoss,
  unpackMoss,
  type BakedMoss,
} from './carpet.ts'
import {
  bakedArtifact,
  commitBake,
  speciesById,
  standEntrySlots,
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type StandSpecies,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Trig-weighted crossed fins. Each species is baked ONCE into an 8-view atlas
 * (6 azimuthal orthographic captures + 2 top-down slab captures, albedo &
 * local-frame normal, full premultiplied mip chains) and cached by the
 * harness bake flow. At runtime every plant is 5 STATIC quads — 3 vertical
 * fins crossed at 60° plus 2 horizontal slab cards — whose per-card content,
 * trigonometric view weights and hashed-alpha dissolve are what make the
 * classic crossing artifacts disappear (see shaders/fins.wgsl).
 *
 * A compute pass (shaders/cull.wgsl) evaluates the shared scatter hash once
 * per candidate slot of a camera-bounded cell region and compacts the
 * surviving plants into an indirect draw, so per-frame cost is bounded by the
 * region and independent of the stand's total plant count.
 *
 * A CARPET entry (stand_table[i].carpet_div > 0 — the bog's Sphagnum) takes a
 * different card set entirely: no fins, but a stack of ground-parallel
 * cross-sections through the cushion, per-vertex terrain conformed. See
 * carpet.ts for the bake and shaders/moss.wgsl for the render.
 */

interface SpeciesGpu {
  albedo: GPUTexture
  normal: GPUTexture
  meta: BakedFins['meta']
}

interface MossGpu {
  albedo: GPUTexture
  normal: GPUTexture
  meta: BakedMoss['meta']
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  density: number
  /** Candidate slots per scatter cell for THIS entry (carpet_div^2 for a mat). */
  slots: number
  /** Expected fraction of those slots that survive placement (capacity sizing). */
  fill: number
  vertexCount: number
  gpu: SpeciesGpu | null
  moss: MossGpu | null
  cfgBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  /**
   * Carpet only (allocated for every entry so one cull bind group layout
   * serves both paths): second draw for the tiles inside MOSS_NEAR_M, the only
   * ones that carry more than a closure card. It runs the same vertex shader
   * with first_vertex = 6, so it emits cards 1..n-1.
   */
  nearIndirect: GPUBuffer
  /** Rebuilt only when regionRadius changes (capacity scales with the area). */
  plantBuffer: GPUBuffer
  nearBuffer: GPUBuffer
  capacity: number
  cullBg: GPUBindGroup
  renderBg: GPUBindGroup
  nearRenderBg: GPUBindGroup | null
}

const VERTS_PER_INSTANCE = 30 // 5 quads x 6 vertices
const MOSS_NEAR_M = 9 // must match MOSS_NEAR_M in shaders/fin_shared.wgsl
const CFG_FLOATS = 32
const PLANT_BYTES = 24 // struct FinPlant: 6 x f32
/**
 * Compacted-instance capacity. Slot existence is an independent hash test per
 * slot, so the survivor count is binomial with sigma ~160 over the region —
 * a 15% margin plus a flat floor is many hundreds of sigma of headroom, and
 * the shader's arrayLength() guard makes an overflow a no-op rather than a
 * corruption.
 */
const CAP_MARGIN = 1.15
const CAP_FLOOR = 8192
/**
 * Only the region CIRCLE can hold survivors (both fades are zero past
 * regionRadius) but the dispatched cell window is a square, so capacity needs
 * the area ratio, not the square: pi*R^2 / (2R+4)^2 = 0.73 at the default
 * radius, taken as 0.78 for margin.
 */
const CAP_CIRCLE = 0.78

const sideFor = (radius: number): number => Math.max(1, Math.ceil((2 * radius) / SCATTER_CELL_SIZE) + 1)

/**
 * Expected fraction of an entry's slots that actually hold a plant.
 *
 * Ordinary scatter: the density test, exactly as before. A carpet PARTITIONS
 * the wetness axis instead, so its share is its wetness interval — but the
 * shared wetness field is skewed (it is damped on slopes), and a nominal 1/3
 * band measures up to 0.59 of the nodes inside a 116 m window on the bog
 * terrain. Hence wetWidth is doubled rather than trusted; an underestimate
 * would drop tiles at the region edge (the shader's arrayLength guard makes an
 * overflow a no-op, not a corruption).
 */
const fillFor = (entry: StandSpecies): number =>
  entry.carpetDiv && entry.carpetDiv > 0
    ? Math.min(1, (entry.wetWidth ?? 1) * 2)
    : Math.min(entry.density, SCATTER_MAX_DENSITY) / SCATTER_MAX_DENSITY

const capacityFor = (side: number, entry: EntryGpu): number =>
  Math.ceil(side * side * entry.slots * entry.fill * CAP_CIRCLE * CAP_MARGIN) + CAP_FLOOR

/** Tiles inside the near band, where the cushion stack is more than one card. */
const nearCapacityFor = (entry: EntryGpu): number =>
  Math.ceil((Math.PI * (MOSS_NEAR_M + 1) ** 2 * entry.slots * entry.fill * CAP_MARGIN) / SCATTER_CELL_SIZE ** 2) + 4096

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    lodMaxClamp: MIPS - 1,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- bake (or load) one atlas per unique species --------------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const load = async (): Promise<SpeciesGpu> => {
      const key = `fins-v${BAKE_VERSION}-${speciesId}`
      const bake = async (): Promise<ArrayBuffer> => {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        console.log(`[${ctx.id}] baking 8-view fin atlas for ${speciesId}...`)
        const buf = packFins(await bakeFins(ctx, mesh))
        commitBake(ctx.id, key, buf).catch(() => undefined) // static builds: ignore
        return buf
      }
      // A cached/committed artifact can be a poisoned blob (a dev server may
      // answer a missing file with index.html) — validate and rebake.
      let baked = unpackFins(await bakedArtifact({ expId: ctx.id, key }, bake))
      if (!baked) {
        baked = unpackFins(await bake())
        if (!baked) throw new Error(`${ctx.id}: bake produced an invalid fin atlas for ${speciesId}`)
      }
      const mkAtlas = (tag: string): GPUTexture =>
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${speciesId}/${tag}`,
            size: [ATLAS_W, ATLAS_H],
            format: 'rgba8unorm',
            mipLevelCount: MIPS,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          },
          { species: speciesId, tag: `fin-${tag}` },
        )
      const albedo = mkAtlas('albedo')
      const normal = mkAtlas('normal')
      for (let level = 0; level < MIPS; level++) {
        const w = Math.max(1, ATLAS_W >> level)
        const h = Math.max(1, ATLAS_H >> level)
        device.queue.writeTexture(
          { texture: albedo, mipLevel: level },
          baked.albedoMips[level]!,
          { bytesPerRow: w * 4, rowsPerImage: h },
          [w, h],
        )
        device.queue.writeTexture(
          { texture: normal, mipLevel: level },
          baked.normalMips[level]!,
          { bytesPerRow: w * 4, rowsPerImage: h },
          [w, h],
        )
      }
      return { albedo, normal, meta: baked.meta }
    }
    const pending = load()
    speciesCache.set(speciesId, pending)
    return pending
  }

  /**
   * Carpet species: MOSS_LEVELS cross-sections in a texture_2d_array, framed to
   * the periodic tile square, sampled with REPEAT addressing (a periodic tile
   * wraps, so that is the correct filter at its boundary — and a per-layer mip
   * chain cannot bleed into a neighbouring view the way an atlas does).
   * Separate artifact key from the fin atlas, so the grass bakes stay valid.
   */
  const mossSampler = device.createSampler({
    label: `${ctx.id}/moss-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    lodMaxClamp: MOSS_ALB_MIPS - 1,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })
  const mossCache = new Map<string, Promise<MossGpu>>()
  /**
   * Moss bakes are SERIALIZED: one Sphagnum mesh is 184 MB of vertices plus a
   * 238 MB index buffer, and the bog stand has three of them — baking them
   * concurrently would hold ~1.3 GB of transient GPU buffers at once.
   */
  let mossQueue: Promise<unknown> = Promise.resolve()
  const loadMoss = (speciesId: string): Promise<MossGpu> => {
    let cached = mossCache.get(speciesId)
    if (cached) return cached
    const load = async (): Promise<MossGpu> => {
      const species = speciesById(speciesId)
      if (species.tileM === undefined) {
        throw new Error(`${ctx.id}: carpet entry "${speciesId}" has no periodic tileM`)
      }
      const key = `moss-v${MOSS_BAKE_VERSION}-${speciesId}`
      const bake = async (): Promise<ArrayBuffer> => {
        const mesh = await ctx.meshes.load(species.meshId)
        console.log(`[${ctx.id}] baking ${MOSS_LEVELS}-level carpet stack for ${speciesId}...`)
        const buf = packMoss(await bakeMoss(ctx, mesh, species.tileM!))
        commitBake(ctx.id, key, buf).catch(() => undefined) // static builds: ignore
        return buf
      }
      let baked = unpackMoss(await bakedArtifact({ expId: ctx.id, key }, bake))
      if (!baked) {
        baked = unpackMoss(await bake())
        if (!baked) throw new Error(`${ctx.id}: bake produced an invalid carpet stack for ${speciesId}`)
      }
      const mkTex = (tag: string, px: number, layers: number, mips: number): GPUTexture =>
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${speciesId}/${tag}`,
            size: [px, px, layers],
            format: 'rgba8unorm',
            mipLevelCount: mips,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          },
          { species: speciesId, tag: `moss-${tag}` },
        )
      // Cross-sections are array layers; the macro-surface map is a single 2D
      // texture shared by all of them.
      const albedo = mkTex('albedo', MOSS_TILE, MOSS_LEVELS, MOSS_ALB_MIPS)
      const normal = mkTex('normal', MOSS_NRM_TILE, 1, MOSS_NRM_MIPS)
      const upload = (texture: GPUTexture, px: number, layer: number, chain: Uint8Array<ArrayBuffer>[]): void => {
        chain.forEach((data, level) => {
          const w = Math.max(1, px >> level)
          device.queue.writeTexture(
            { texture, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
            data,
            { bytesPerRow: w * 4, rowsPerImage: w },
            [w, w, 1],
          )
        })
      }
      baked.albedo.forEach((chain, layer) => upload(albedo, MOSS_TILE, layer, chain))
      upload(normal, MOSS_NRM_TILE, 0, baked.normal)
      return { albedo, normal, meta: baked.meta }
    }
    const pending = mossQueue.then(load)
    mossQueue = pending.catch(() => undefined)
    mossCache.set(speciesId, pending)
    return pending
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const renderBgl = device.createBindGroupLayout({
    label: `${ctx.id}/render-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  // Same shape, but the carpet's cross-sections are array layers.
  const mossBgl = device.createBindGroupLayout({
    label: `${ctx.id}/moss-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  type RegionKeys = 'capacity' | 'plantBuffer' | 'nearBuffer' | 'cullBg' | 'renderBg' | 'nearRenderBg'
  type EntryBase = Omit<EntryGpu, RegionKeys>
  type RegionRes = Pick<EntryGpu, RegionKeys>

  /** Allocates the compacted-plant buffers + bind groups for `side`. */
  const makeRegion = (entry: EntryBase, side: number): RegionRes => {
    const capacity = capacityFor(side, entry as EntryGpu)
    const stack = entry.moss
    const mkPlants = (tag: string, count: number): GPUBuffer =>
      ctx.res.createBuffer(
        { label: `${ctx.id}/${tag}-${entry.entryIndex}`, size: count * PLANT_BYTES, usage: GPUBufferUsage.STORAGE },
        { species: entry.speciesId, tag },
      )
    const plantBuffer = mkPlants('plants', capacity)
    // Non-carpet entries never write the near list; a stub keeps the shared
    // cull bind group layout valid without wasting memory.
    const nearBuffer = mkPlants('plants-near', stack ? nearCapacityFor(entry as EntryGpu) : 2)
    const mkRenderBg = (tag: string, plants: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/${tag}-${entry.entryIndex}`,
        layout: stack ? mossBgl : renderBgl,
        entries: [
          { binding: 0, resource: { buffer: entry.cfgBuffer } },
          { binding: 1, resource: { buffer: plants } },
          { binding: 2, resource: (stack ?? entry.gpu!).albedo.createView() },
          { binding: 3, resource: (stack ?? entry.gpu!).normal.createView() },
          { binding: 4, resource: stack ? mossSampler : sampler },
        ],
      })
    return {
      capacity,
      plantBuffer,
      nearBuffer,
      cullBg: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entry.entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: entry.cfgBuffer } },
          { binding: 1, resource: { buffer: plantBuffer } },
          { binding: 2, resource: { buffer: entry.indirectBuffer } },
          { binding: 3, resource: { buffer: nearBuffer } },
          { binding: 4, resource: { buffer: entry.nearIndirect } },
        ],
      }),
      renderBg: mkRenderBg('render-bg', plantBuffer),
      nearRenderBg: stack ? mkRenderBg('render-bg-near', nearBuffer) : null,
    }
  }

  let side = sideFor(ctx.params.regionRadius)

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      // A carpet entry is a MAT, not a plant: different card set, different
      // bake, different slot count per cell. Everything else is shared.
      const isCarpet = (e.carpetDiv ?? 0) > 0
      const base: EntryBase = {
        entryIndex,
        speciesId: e.species,
        density: e.density,
        slots: standEntrySlots(e),
        fill: fillFor(e),
        // A carpet's main draw is ONE card (6 verts) for every tile; the near
        // list adds the rest of the stack in a second draw.
        vertexCount: isCarpet ? 6 : VERTS_PER_INSTANCE,
        gpu: isCarpet ? null : await loadSpecies(e.species),
        moss: isCarpet ? await loadMoss(e.species) : null,
        cfgBuffer: ctx.res.createBuffer(
          {
            label: `${ctx.id}/cfg-${entryIndex}`,
            size: CFG_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'cfg' },
        ),
        indirectBuffer: ctx.res.createBuffer(
          {
            label: `${ctx.id}/indirect-${entryIndex}`,
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'indirect' },
        ),
        nearIndirect: ctx.res.createBuffer(
          {
            label: `${ctx.id}/indirect-near-${entryIndex}`,
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'indirect' },
        ),
      }
      return { ...base, ...makeRegion(base, side) }
    }),
  )

  let cullPipeline!: GPUComputePipeline
  let pipeline!: GPURenderPipeline
  let mossPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/fins`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, renderBgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    const mossModule = ctx.shaders.module(mossSrc)
    mossPipeline = device.createRenderPipeline({
      label: `${ctx.id}/moss`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/moss-pl`, bindGroupLayouts: [ctx.frame.layout, mossBgl] }),
      vertex: { module: mossModule, entryPoint: 'vs_main' },
      fragment: { module: mossModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      // Cards are ground-parallel and never seen from below in practice, but a
      // mat on a ridge crest can be, so both faces draw.
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cfgData = new Float32Array(CFG_FLOATS)
  const indirectReset = new Uint32Array([VERTS_PER_INSTANCE, 0, 0, 0])
  // Cards 1..MOSS_LEVELS-1 of the near stack. first_vertex = 6 shifts
  // vertex_index so the SAME vertex shader emits them; first_instance stays 0
  // (a non-zero one is silently dropped without indirect-first-instance).
  const nearReset = new Uint32Array([(MOSS_LEVELS - 1) * 6, 0, 6, 0])

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / SCATTER_CELL_SIZE)
      const originCellZ = Math.floor((cam.z - R) / SCATTER_CELL_SIZE)
      // Cells overlapping the stand's region — plants exist nowhere else.
      const cellMin = Math.floor(-ctx.stand.radius / SCATTER_CELL_SIZE)
      const cellMax = Math.floor(ctx.stand.radius / SCATTER_CELL_SIZE)

      for (const entry of entries) {
        const m = entry.gpu?.meta
        cfgData[0] = m?.cx ?? 0; cfgData[1] = m?.yc ?? 0; cfgData[2] = m?.cz ?? 0; cfgData[3] = m?.halfW ?? 0
        cfgData[4] = m?.halfH ?? 0; cfgData[5] = m?.yLow ?? 0; cfgData[6] = m?.yHigh ?? 0; cfgData[7] = m?.radius ?? 0
        cfgData[8] = originCellX; cfgData[9] = originCellZ; cfgData[10] = side; cfgData[11] = ctx.seed
        cfgData[12] = R; cfgData[13] = entry.entryIndex; cfgData[14] = ctx.params.edgeFade; cfgData[15] = ctx.params.topBlend
        cfgData[16] = ctx.params.alphaSharp; cfgData[17] = cellMin; cfgData[18] = cellMax
        cfgData[19] = ctx.params.cardTint ? 1 : 0
        cfgData[20] = entry.slots; cfgData[21] = entry.moss ? 1 : 0
        cfgData[22] = ctx.params.mossLayers; cfgData[23] = ctx.params.mossRelief
        const levels = entry.moss?.meta.levelY
        for (let k = 0; k < MOSS_LEVELS; k++) cfgData[24 + k] = levels?.[k] ?? 0
        cfgData[30] = entry.moss?.meta.apexY ?? 0; cfgData[31] = 0
        device.queue.writeBuffer(entry.cfgBuffer, 0, cfgData)
        indirectReset[0] = entry.vertexCount
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        if (entry.moss) device.queue.writeBuffer(entry.nearIndirect, 0, nearReset)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // One thread per candidate slot of the region — the placement hash runs
      // once per candidate instead of once per vertex. The slot count is PER
      // ENTRY (carpet_div^2 = 484 for the moss, 128 for scattered grass);
      // dispatching the scatter budget for a carpet renders a quarter of it.
      const cull = ctx.timing.computePass(enc, 'fin-cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cull.setBindGroup(1, entry.cullBg)
        cull.dispatchWorkgroups(Math.ceil((side * side * entry.slots) / 64))
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'fin-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Grouped by card set so each pipeline is bound once per frame.
      for (const carpet of [false, true]) {
        let bound = false
        for (const entry of entries) {
          if (!!entry.moss !== carpet) continue
          if (!bound) {
            pass.setPipeline(carpet ? mossPipeline : pipeline)
            bound = true
          }
          pass.setBindGroup(1, entry.renderBg)
          pass.drawIndirect(entry.indirectBuffer, 0)
          if (entry.nearRenderBg) {
            pass.setBindGroup(1, entry.nearRenderBg)
            pass.drawIndirect(entry.nearIndirect, 0)
          }
        }
      }
      pass.end()
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      if (!keys.has('regionRadius')) return
      side = sideFor(ctx.params.regionRadius)
      for (const entry of entries) {
        if (capacityFor(side, entry) === entry.capacity) continue
        entry.plantBuffer.destroy()
        entry.nearBuffer.destroy()
        Object.assign(entry, makeRegion(entry, side))
      }
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
