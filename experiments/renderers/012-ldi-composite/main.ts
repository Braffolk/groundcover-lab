import carpetCullSrc from './shaders/carpet_cull.wgsl'
import carpetSrc from './shaders/carpet.wgsl'
import cullSrc from './shaders/cull.wgsl'
import ldiSrc from './shaders/ldi.wgsl'
import { ATLAS_H, ATLAS_W, BAKE_VERSION, NUM_LAYERS, TILE, bakeLdi, unpackLdi, type LdiBaked, type LdiMeta } from './bake.ts'
import { CARPET_LAYERS, CARPET_N, buildCarpetTiles, type CarpetTiles } from './carpet.ts'
import { SCATTER_CELL_SIZE, SCATTER_MAX_PER_CELL, assetUrl, commitBake, speciesById, standEntrySlots } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, StandSpecies, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Layered depth images, in two shapes.
 *
 * SPECIMEN species (grasses) keep the original method: each is baked once into
 * 5 capture directions x 4 min-separation depth-peeled layers, and a plant
 * draws 12 depth-writing cards — the two side stacks nearest the camera
 * azimuth plus the top stack — with per-texel baked depth via frag_depth, so
 * the depth buffer composites the stack union.
 *
 * CARPET species (`stand_table[i].carpet_div > 0` — the bog's Sphagnum) are a
 * periodic 0.18 m community tile 7-9 cm tall, and want the opposite balance:
 * the side captures see a sliver, while the TOP capture is a 0.84 mm/texel
 * displacement map of the cushion. So a mat tile draws ONE ground-parallel quad
 * exactly one grid step across (two near, for the second peel layer), out of a
 * load-time reshaped periodic + mipped tile texture, terrain-conformed per
 * vertex, with the baked depth turned into real world height in frag_depth.
 * See carpet.ts / shaders/carpet.wgsl.
 *
 * Cost is bounded by the camera-centred region in both shapes; the stand's
 * plant count never enters a loop.
 */

const CELL = SCATTER_CELL_SIZE
const MAX_RADIUS = 128 // must match the regionRadius param max
const NEAR_VERTS = 72 // 3 stacks x 4 layers x 6 verts
const FAR_VERTS = 12 // 2 front-layer cards
const CARPET_NEAR_VERTS = 6 * CARPET_LAYERS
const CARPET_FAR_VERTS = 6
const UNIFORM_FLOATS = 20 + 5 * 16 // five header vec4s + 5 DirRec
const CARPET_UNIFORM_FLOATS = 28 // 7 vec4s, see carpet_common.wgsl
/** Byte offset + float count of the only uniform fields that change per frame. */
const DYN_OFFSET = 32
const CARPET_DYN_OFFSET = 64
const DYN_FLOATS = 8
const CARPET_DYN_FLOATS = 12
const INSPECT_MODES = ['off', 'dir', 'layer', 'path'] as const

interface CarpetGpuTex {
  albedo: GPUTexture
  aux: GPUTexture
  tiles: CarpetTiles
}

interface SpeciesGpu {
  /** Specimen atlases — absent for a species this stand only uses as a carpet. */
  atlas0: GPUTexture | null
  atlas1: GPUTexture | null
  carpet: CarpetGpuTex | null
  /** Header only — the 10.5 MB of atlas bytes are dead once uploaded. */
  meta: LdiMeta
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  capacity: number
  uniform: GPUBuffer
  nearIndirect: GPUBuffer
  farIndirect: GPUBuffer
  cullBind: GPUBindGroup
  renderBind: GPUBindGroup
  /** Extra dispatch dimension: slots per cell / 128 (carpets exceed 128). */
  dispatchZ: number
}

const isCarpet = (e: StandSpecies): boolean => (e.carpetDiv ?? 0) > 0

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })
  // A carpet tile texture is exactly periodic (see carpet.ts), so `repeat`
  // addressing is correct and mips are seamless — that is what lets the mat
  // survive minification instead of speckling. ANISOTROPIC, because a
  // ground-parallel quad at grazing is the textbook anisotropic case: the
  // isotropic LOD of a mat seen edge-on lands near the 1x1 mip and flattens the
  // whole near field to one colour.
  const carpetSampler = device.createSampler({
    label: `${ctx.id}/carpet-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    maxAnisotropy: 16,
  })

  // What each species is used for in THIS stand: specimen atlases, carpet
  // tiles, or both. A carpet species never samples the 5x4 atlas, so it never
  // uploads it — 10.5 MB of VRAM per species that would go entirely to views a
  // cushion does not show.
  interface SpeciesNeed {
    atlas: boolean
    carpet: boolean
    tileM: number
  }
  const needs = new Map<string, SpeciesNeed>()
  for (const e of ctx.stand.species) {
    const need = needs.get(e.species) ?? { atlas: false, carpet: false, tileM: 0 }
    if (isCarpet(e)) {
      need.carpet = true
      need.tileM = speciesById(e.species).tileM ?? 0
    } else {
      need.atlas = true
    }
    needs.set(e.species, need)
  }

  // --- bake or load one LDI set per unique species ---------------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const p = (async (): Promise<SpeciesGpu> => {
      // The dev server answers missing /mesh/baked files with index.html, so
      // validate magic + dims ourselves instead of trusting bakedArtifact().
      // assetUrl() because a production build is served under a base path.
      const key = `${speciesId}-ldi-v${BAKE_VERSION}`
      let baked: LdiBaked | null = null
      const committed = await fetch(assetUrl(`/mesh/baked/${ctx.id}/${key}.bin`)).catch(() => null)
      if (committed?.ok) baked = unpackLdi(await committed.arrayBuffer())
      if (!baked) {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        const packed = await bakeLdi(ctx, mesh)
        baked = unpackLdi(packed)
        if (!baked) throw new Error(`${ctx.id}: fresh bake for ${speciesId} failed to unpack`)
        commitBake(ctx.id, key, packed).catch(() => undefined) // best effort
      }
      const need = needs.get(speciesId)!
      let atlas0: GPUTexture | null = null
      let atlas1: GPUTexture | null = null
      if (need.atlas) {
        const mk = (label: string): GPUTexture =>
          ctx.res.createTexture(
            {
              label: `${ctx.id}/${speciesId}/${label}`,
              size: [ATLAS_W, ATLAS_H],
              format: 'rgba8unorm',
              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            },
            { species: speciesId, tag: `ldi-${label}` },
          )
        atlas0 = mk('albedo')
        atlas1 = mk('aux')
        const layout = { bytesPerRow: ATLAS_W * 4, rowsPerImage: ATLAS_H }
        device.queue.writeTexture({ texture: atlas0 }, baked.atlas0, layout, [ATLAS_W, ATLAS_H])
        device.queue.writeTexture({ texture: atlas1 }, baked.atlas1, layout, [ATLAS_W, ATLAS_H])
      }
      let carpet: CarpetGpuTex | null = null
      if (need.carpet && need.tileM > 0) {
        const tiles = buildCarpetTiles(baked, need.tileM)
        const mips = tiles.layers[0]!.length
        const mkArray = (label: string): GPUTexture =>
          ctx.res.createTexture(
            {
              label: `${ctx.id}/${speciesId}/${label}`,
              size: [CARPET_N, CARPET_N, CARPET_LAYERS],
              format: 'rgba8unorm',
              mipLevelCount: mips,
              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            },
            { species: speciesId, tag: `carpet-${label}` },
          )
        const albedo = mkArray('carpet-albedo')
        const aux = mkArray('carpet-aux')
        tiles.layers.forEach((chain, layer) => {
          chain.forEach((mip, level) => {
            const layout = { bytesPerRow: mip.size * 4, rowsPerImage: mip.size }
            const dst = { origin: { x: 0, y: 0, z: layer }, mipLevel: level }
            device.queue.writeTexture({ texture: albedo, ...dst }, mip.albedo, layout, [mip.size, mip.size, 1])
            device.queue.writeTexture({ texture: aux, ...dst }, mip.aux, layout, [mip.size, mip.size, 1])
          })
        })
        carpet = { albedo, aux, tiles }
      }
      // Keep only the header: atlas0/atlas1 are views into the whole packed
      // artifact, so holding `baked` would pin ~10.5 MB of host memory per
      // species for the rest of the session for nothing.
      return { atlas0, atlas1, carpet, meta: baked.meta }
    })()
    speciesCache.set(speciesId, p)
    return p
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const mkRenderBgl = (label: string, dim: GPUTextureViewDimension): GPUBindGroupLayout =>
    device.createBindGroupLayout({
      label: `${ctx.id}/${label}`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: dim } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: dim } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })
  const renderBgl = mkRenderBgl('render-bgl', '2d')
  const carpetBgl = mkRenderBgl('carpet-bgl', '2d-array')

  const sideMax = Math.ceil((2 * MAX_RADIUS) / CELL) + 1
  // Cells that can hold a plant at all: the region window, clamped to the
  // stand's own cell span (nothing grows outside it).
  const standCells = Math.floor(ctx.stand.radius / CELL) * 2 + 1
  const winCells = Math.min(sideMax, standCells)
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const mkIndirect = (name: string, species: string, verts: number): GPUBuffer => {
    const buf = ctx.res.createBuffer(
      {
        label: `${ctx.id}/${name}`,
        size: 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species, tag: 'ldi-indirect' },
    )
    // vertexCount / firstVertex / firstInstance never change: only the instance
    // count is reset per encode (clearBuffer), so neither draw needs a template
    // copy — and firstInstance stays 0, which is the only value WebGPU accepts
    // without the `indirect-first-instance` feature.
    device.queue.writeBuffer(buf, 0, new Uint32Array([verts, 0, 0, 0]))
    return buf
  }

  const scatterEntries: EntryGpu[] = []
  const carpetEntries: EntryGpu[] = []

  await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<void> => {
      const gpu = await loadSpecies(e.species)
      const carpet = isCarpet(e) ? gpu.carpet : null
      if (isCarpet(e) && !carpet) throw new Error(`${ctx.id}: carpet entry ${e.species} has no periodic tileM`)

      if (carpet) {
        // --- mat entry: 4-byte instances, every carpet_div^2 slot evaluated --
        const div = e.carpetDiv!
        const slots = standEntrySlots(e) // div^2 — NOT SCATTER_MAX_PER_CELL
        const stepM = CELL / div
        const scale = stepM / needs.get(e.species)!.tileM
        const overscale = ctx.params.carpetOverscale
        // Worst case is exact and cheap here: every node of every cell in the
        // window, at 4 B. No overflow is possible, so the mat can never show
        // capacity holes that look like a placement bug.
        const capacity = winCells * winCells * slots
        const uniform = ctx.res.createBuffer(
          {
            label: `${ctx.id}/carpet-uni-${entryIndex}`,
            size: CARPET_UNIFORM_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'carpet-uniform' },
        )
        const instances = ctx.res.createBuffer(
          { label: `${ctx.id}/carpet-inst-${entryIndex}`, size: capacity * 4, usage: GPUBufferUsage.STORAGE },
          { species: e.species, tag: 'carpet-instances' },
        )
        const nearIndirect = mkIndirect(`carpet-near-${entryIndex}`, e.species, CARPET_NEAR_VERTS)
        const farIndirect = mkIndirect(`carpet-far-${entryIndex}`, e.species, CARPET_FAR_VERTS)

        const t = carpet.tiles
        const uniInit = new Float32Array(CARPET_UNIFORM_FLOATS)
        uniInit.set([stepM, 0.5 * stepM * overscale, t.yTop * scale, t.ySpan * scale], 0)
        uniInit.set([t.planeY[0]! * scale, t.planeY[1]! * scale, 0, 0], 4)
        uniInit.set([ctx.seed, entryIndex, capacity, ctx.stand.radius], 8)
        uniInit.set([div, slots, overscale, 0], 12)
        device.queue.writeBuffer(uniform, 0, uniInit)

        carpetEntries.push({
          entryIndex,
          speciesId: e.species,
          capacity,
          uniform,
          nearIndirect,
          farIndirect,
          dispatchZ: Math.ceil(slots / 128),
          cullBind: device.createBindGroup({
            label: `${ctx.id}/carpet-cull-bg-${entryIndex}`,
            layout: cullBgl,
            entries: [
              { binding: 0, resource: { buffer: uniform } },
              { binding: 1, resource: { buffer: instances } },
              { binding: 2, resource: { buffer: nearIndirect } },
              { binding: 3, resource: { buffer: farIndirect } },
            ],
          }),
          renderBind: device.createBindGroup({
            label: `${ctx.id}/carpet-render-bg-${entryIndex}`,
            layout: carpetBgl,
            entries: [
              { binding: 0, resource: { buffer: uniform } },
              { binding: 1, resource: { buffer: instances } },
              { binding: 2, resource: carpet.albedo.createView({ dimension: '2d-array' }) },
              { binding: 3, resource: carpet.aux.createView({ dimension: '2d-array' }) },
              { binding: 4, resource: carpetSampler },
            ],
          }),
        })
        return
      }

      // --- specimen entry: unchanged from the original method ----------------
      const capacity = Math.ceil(((slotsMax * Math.min(e.density, 8)) / 8) * 1.15)
      const uniform = ctx.res.createBuffer(
        {
          label: `${ctx.id}/uni-${entryIndex}`,
          size: UNIFORM_FLOATS * 4,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        },
        { species: e.species, tag: 'ldi-uniform' },
      )
      const instances = ctx.res.createBuffer(
        { label: `${ctx.id}/inst-${entryIndex}`, size: capacity * 32, usage: GPUBufferUsage.STORAGE },
        { species: e.species, tag: 'ldi-instances' },
      )
      const nearIndirect = mkIndirect(`near-args-${entryIndex}`, e.species, NEAR_VERTS)
      const farIndirect = mkIndirect(`far-args-${entryIndex}`, e.species, FAR_VERTS)

      // Everything except `region` and `tune` (floats 8..15) is a bake/stand
      // constant — write the whole block once here, then per frame touch only
      // the 32-byte dynamic window.
      const m = gpu.meta
      const uniInit = new Float32Array(UNIFORM_FLOATS)
      uniInit.set([m.center[0]!, m.center[1]!, m.center[2]!, m.boundR], 0)
      uniInit.set([TILE, ATLAS_W, ATLAS_H, NUM_LAYERS], 4)
      uniInit.set([ctx.seed, entryIndex, capacity, ctx.stand.radius], 16)
      m.dirs.forEach((d, i) => {
        uniInit.set([...d.right, d.halfW, ...d.up, d.halfH, ...d.fwd, d.sMax, ...d.meanD], 20 + i * 16)
      })
      device.queue.writeBuffer(uniform, 0, uniInit)

      scatterEntries.push({
        entryIndex,
        speciesId: e.species,
        capacity,
        uniform,
        nearIndirect,
        farIndirect,
        dispatchZ: 1,
        cullBind: device.createBindGroup({
          label: `${ctx.id}/cull-bg-${entryIndex}`,
          layout: cullBgl,
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: { buffer: instances } },
            { binding: 2, resource: { buffer: nearIndirect } },
            { binding: 3, resource: { buffer: farIndirect } },
          ],
        }),
        renderBind: device.createBindGroup({
          label: `${ctx.id}/render-bg-${entryIndex}`,
          layout: renderBgl,
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: { buffer: instances } },
            { binding: 2, resource: gpu.atlas0!.createView() },
            { binding: 3, resource: gpu.atlas1!.createView() },
            { binding: 4, resource: sampler },
          ],
        }),
      })
    }),
  )
  scatterEntries.sort((a, b) => a.entryIndex - b.entryIndex)
  carpetEntries.sort((a, b) => a.entryIndex - b.entryIndex)

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  let carpetCullPipeline: GPUComputePipeline | null = null
  let carpetNearPipeline: GPURenderPipeline | null = null
  let carpetFarPipeline: GPURenderPipeline | null = null
  const build = (): void => {
    const mkRender = (
      label: string,
      module: GPUShaderModule,
      layout: GPUPipelineLayout,
      vsEntry: string,
      fsEntry: string,
    ): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${label}`,
        layout,
        vertex: { module, entryPoint: vsEntry },
        fragment: { module, entryPoint: fsEntry, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })

    if (scatterEntries.length > 0) {
      cullPipeline = device.createComputePipeline({
        label: `${ctx.id}/cull`,
        layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
        compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
      })
      const module = ctx.shaders.module(ldiSrc)
      const layout = device.createPipelineLayout({ label: `${ctx.id}/pl`, bindGroupLayouts: [ctx.frame.layout, renderBgl] })
      nearPipeline = mkRender('near', module, layout, 'vs_main', 'fs_main')
      farPipeline = mkRender('far', module, layout, 'vs_far', 'fs_far')
    }
    if (carpetEntries.length > 0) {
      carpetCullPipeline = device.createComputePipeline({
        label: `${ctx.id}/carpet-cull`,
        layout: device.createPipelineLayout({
          label: `${ctx.id}/carpet-cull-pl`,
          bindGroupLayouts: [ctx.frame.layout, cullBgl],
        }),
        compute: { module: ctx.shaders.module(carpetCullSrc), entryPoint: 'cs_carpet' },
      })
      const module = ctx.shaders.module(carpetSrc)
      const layout = device.createPipelineLayout({
        label: `${ctx.id}/carpet-pl`,
        bindGroupLayouts: [ctx.frame.layout, carpetBgl],
      })
      carpetNearPipeline = mkRender('carpet-near', module, layout, 'vs_carpet', 'fs_carpet')
      carpetFarPipeline = mkRender('carpet-far', module, layout, 'vs_carpet_far', 'fs_carpet_far')
    }
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // Only `region` + `tune` move per frame; the rest of each block was written
  // at create time. All entries of a kind share these values, so it is one
  // 32-byte scratch reused for every write.
  const dyn = new Float32Array(DYN_FLOATS)
  const carpetDyn = new Float32Array(CARPET_DYN_FLOATS)
  let side = 1
  // Overscale lives in the static half (it scales the card, not the grid), so
  // it is pushed only when the param actually moves.
  let lastOverscale = ctx.params.carpetOverscale
  const pushOverscale = (over: number): void => {
    for (const entry of carpetEntries) {
      const step = CELL / ctx.stand.species[entry.entryIndex]!.carpetDiv!
      device.queue.writeBuffer(entry.uniform, 4, new Float32Array([0.5 * step * over]))
      device.queue.writeBuffer(entry.uniform, 56, new Float32Array([over]))
    }
  }

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      side = Math.max(1, Math.ceil((2 * R) / CELL) + 1)

      const originX = Math.floor((cam.x - R) / CELL)
      const originZ = Math.floor((cam.z - R) / CELL)
      const inspect = Math.max(0, INSPECT_MODES.indexOf(ctx.params.inspect))
      dyn[0] = originX
      dyn[1] = originZ
      dyn[2] = side
      dyn[3] = R
      dyn[4] = ctx.params.layerCullDist
      dyn[5] = ctx.params.parallax ? 1 : 0
      dyn[6] = ctx.params.coverage
      dyn[7] = inspect
      for (const entry of scatterEntries) device.queue.writeBuffer(entry.uniform, DYN_OFFSET, dyn)

      if (carpetEntries.length > 0) {
        if (ctx.params.carpetOverscale !== lastOverscale) {
          lastOverscale = ctx.params.carpetOverscale
          pushOverscale(lastOverscale)
        }
        carpetDyn[0] = originX
        carpetDyn[1] = originZ
        carpetDyn[2] = side
        carpetDyn[3] = R
        carpetDyn[4] = ctx.params.carpetNear
        carpetDyn[5] = ctx.params.parallax ? 1 : 0
        carpetDyn[6] = ctx.params.carpetAlpha
        carpetDyn[7] = inspect
        carpetDyn[8] = ctx.params.carpetSharpen
        for (const entry of carpetEntries) device.queue.writeBuffer(entry.uniform, CARPET_DYN_OFFSET, carpetDyn)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // Reset instance counts inside this encoder so multi-view frames stay correct.
      for (const entry of [...scatterEntries, ...carpetEntries]) {
        enc.clearBuffer(entry.nearIndirect, 4, 4)
        enc.clearBuffer(entry.farIndirect, 4, 4)
      }

      if (carpetEntries.length > 0 && carpetCullPipeline) {
        const cpass = ctx.timing.computePass(enc, 'carpet-cull')
        cpass.setPipeline(carpetCullPipeline)
        cpass.setBindGroup(0, ctx.frame.bindGroup)
        for (const entry of carpetEntries) {
          cpass.setBindGroup(1, entry.cullBind)
          cpass.dispatchWorkgroups(side, side, entry.dispatchZ)
        }
        cpass.end()
      }

      if (scatterEntries.length > 0) {
        const cpass = ctx.timing.computePass(enc, 'ldi-cull')
        cpass.setPipeline(cullPipeline)
        cpass.setBindGroup(0, ctx.frame.bindGroup)
        for (const entry of scatterEntries) {
          cpass.setBindGroup(1, entry.cullBind)
          cpass.dispatchWorkgroups(side, side)
        }
        cpass.end()
      }

      // The mat draws first: it is a solid depth-writing surface, so it becomes
      // an occluder for everything the grass pass would otherwise blend over.
      if (carpetEntries.length > 0 && carpetNearPipeline && carpetFarPipeline) {
        const pass = ctx.timing.renderPass(enc, 'ldi-carpet', {
          colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
          depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
        })
        pass.setBindGroup(0, ctx.frame.bindGroup)
        pass.setPipeline(carpetNearPipeline)
        for (const entry of carpetEntries) {
          pass.setBindGroup(1, entry.renderBind)
          pass.drawIndirect(entry.nearIndirect, 0)
        }
        pass.setPipeline(carpetFarPipeline)
        for (const entry of carpetEntries) {
          pass.setBindGroup(1, entry.renderBind)
          pass.drawIndirect(entry.farIndirect, 0)
        }
        pass.end()
      }

      if (scatterEntries.length > 0) {
        const pass = ctx.timing.renderPass(enc, 'ldi-layers', {
          colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
          depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
        })
        pass.setBindGroup(0, ctx.frame.bindGroup)
        // Near first: it writes the depth that lets early-z reject far fragments.
        pass.setPipeline(nearPipeline)
        for (const entry of scatterEntries) {
          pass.setBindGroup(1, entry.renderBind)
          pass.drawIndirect(entry.nearIndirect, 0)
        }
        pass.setPipeline(farPipeline)
        for (const entry of scatterEntries) {
          pass.setBindGroup(1, entry.renderBind)
          pass.drawIndirect(entry.farIndirect, 0)
        }
        pass.end()
      }
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are destroyed by the harness via ctx.res.
    },
  }
}
