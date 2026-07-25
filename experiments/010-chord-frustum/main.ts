import cullSrc from './shaders/cull.wgsl'
import renderSrc from './shaders/render.wgsl'
import { bakeChordField } from './bake.ts'
import { ATLAS_H, ATLAS_W, unpackChordField, type ChordField } from './chords.ts'
import {
  bakedArtifact,
  commitBake,
  speciesById,
  standEntrySlots,
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
} from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Chord-field frustum proxies. Each plant is a convex cone-frustum proxy
 * volume whose INTERIOR appearance is precomputed: the fragment shader
 * computes the eye ray's chord (entry/exit) through the analytic frustum in
 * closed form and looks up what that chord sees in a baked 4D chord light
 * field. A per-frame compute pass compacts the camera-bounded scatter region
 * into an indirect draw, so cost is independent of total plant count.
 */

const BAKE_VERSION = 6
const MAX_REGION_RADIUS = 80
/**
 * Instance-list ceiling for a carpet entry. A carpet has carpetDiv² slots per
 * 4m cell (484 for the bog moss at life size), so the region-wide slot count is
 * ~4x the scatter budget's and sizing for "every node in the region survives"
 * would cost 20MB per species for a list that never fills: the entries
 * PARTITION the wetness axis (~a third of the nodes each) and the frustum cull
 * keeps only what is on screen. 200k tiles is 6600 m² of closed mat at 30.25
 * tiles/m² — more than any standard camera sees inside a 56m region (topdown
 * from 42m: ~4000 m²) — and the cull drops overflow instead of corrupting.
 */
const CARPET_MAX_INSTANCES = 200_000

/**
 * The 8-sided frustum proxy: 32 triangles over 18 DISTINCT vertices — 8
 * bottom-ring, 8 top-ring, then the top- and bottom-cap fan centers. Drawing
 * it non-indexed would shade every vertex 5.3x over. Winding is CCW-outward
 * (the pipeline culls front faces, keeping the proxy's far side). Must match
 * the vertex_index decode in shaders/render.wgsl.
 */
const PROXY_INDICES: Uint16Array<ArrayBuffer> = (() => {
  const idx: number[] = []
  const bottom = (i: number): number => i % 8
  const top = (i: number): number => 8 + (i % 8)
  const TOP_CENTER = 16
  const BOTTOM_CENTER = 17
  for (let q = 0; q < 8; q++) {
    idx.push(bottom(q), top(q), bottom(q + 1))
    idx.push(bottom(q + 1), top(q), top(q + 1))
  }
  for (let t = 0; t < 8; t++) idx.push(TOP_CENTER, top(t + 1), top(t))
  for (let t = 0; t < 8; t++) idx.push(BOTTOM_CENTER, bottom(t), bottom(t + 1))
  return new Uint16Array(idx)
})()

interface SpeciesGpu {
  field: ChordField
  surfTex: GPUTexture
  geomTex: GPUTexture
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  /** Slots per scatter cell for this entry — carpetDiv², or the scatter budget. */
  slots: number
  isCarpet: boolean
  gpu: SpeciesGpu
  infoBuf: GPUBuffer
  cullBuf: GPUBuffer
  instanceBuf: GPUBuffer
  indirectBuf: GPUBuffer
  cullBg: GPUBindGroup
  renderBg: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- bake / load one chord field per unique species -----------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const load = async (): Promise<SpeciesGpu> => {
      const key = `chords-v${BAKE_VERSION}-${speciesId}`
      const bake = async (): Promise<ArrayBuffer> => {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        console.log(`[${ctx.id}] baking chord field for ${speciesId}...`)
        const buf = await bakeChordField(ctx, mesh, (n) => console.log(`[${ctx.id}] ${speciesId}: ${n}`))
        commitBake(ctx.id, key, buf).catch(() => undefined) // static builds: ignore
        return buf
      }
      // bakedArtifact may hand back a poisoned blob (the dev server answers
      // missing committed files with index.html) — validate and rebake.
      let buf = await bakedArtifact({ expId: ctx.id, key }, bake)
      let field = unpackChordField(buf)
      if (!field) {
        buf = await bake()
        field = unpackChordField(buf)
        if (!field) throw new Error(`${ctx.id}: bake produced an invalid chord field for ${speciesId}`)
      }
      const surfTex = ctx.res.createTexture(
        {
          label: `${ctx.id}/${speciesId}/surf`,
          size: [ATLAS_W, ATLAS_H],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species: speciesId, tag: 'chord-surf' },
      )
      const geomTex = ctx.res.createTexture(
        {
          label: `${ctx.id}/${speciesId}/geom`,
          size: [ATLAS_W, ATLAS_H],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species: speciesId, tag: 'chord-geom' },
      )
      device.queue.writeTexture({ texture: surfTex }, field.surf, { bytesPerRow: ATLAS_W * 4, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
      device.queue.writeTexture({ texture: geomTex }, field.geom, { bytesPerRow: ATLAS_W * 4, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
      return { field, surfTex, geomTex }
    }
    cached = load()
    speciesCache.set(speciesId, cached)
    return cached
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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

  const maxSide = Math.ceil((2 * MAX_REGION_RADIUS) / SCATTER_CELL_SIZE) + 1

  const indexBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/proxy-idx`, size: PROXY_INDICES.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'proxy-index' },
  )
  device.queue.writeBuffer(indexBuf, 0, PROXY_INDICES)

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const gpu = await loadSpecies(e.species)
      // SLOTS TO EVALUATE vs INSTANCES TO STORE are two different numbers.
      // Every slot must be visited (standEntrySlots, never SCATTER_MAX_PER_CELL
      // — a carpet has carpetDiv² of them), but the list only holds survivors.
      // A scattered slot exists with probability density/SCATTER_MAX_DENSITY and
      // over ~2e5 Bernoulli trials the count sits within a few hundred of its
      // mean, so mean + 15% + 4k is unreachable. A carpet node instead exists
      // iff this entry's wetness interval claims it (~wetWidth of the nodes,
      // but the wetness field is smooth over 12m so one zone can dominate a
      // region — hence the 2x share and the hard ceiling).
      const slots = standEntrySlots(e)
      const isCarpet = (e.carpetDiv ?? 0) > 0
      const regionSlots = maxSide * maxSide * slots
      const share = isCarpet
        ? Math.min(1, (e.wetWidth ?? 1) * 2)
        : Math.min(e.density, SCATTER_MAX_DENSITY) / SCATTER_MAX_DENSITY
      const instanceCap = Math.min(
        Math.ceil(regionSlots * share * 1.15) + 4096,
        isCarpet ? CARPET_MAX_INSTANCES : Number.MAX_SAFE_INTEGER,
      )
      const infoBuf = ctx.res.createBuffer(
        { label: `${ctx.id}/info-${entryIndex}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'info' },
      )
      const cullBuf = ctx.res.createBuffer(
        { label: `${ctx.id}/cull-${entryIndex}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'cull-cfg' },
      )
      const instanceBuf = ctx.res.createBuffer(
        { label: `${ctx.id}/instances-${entryIndex}`, size: instanceCap * 32, usage: GPUBufferUsage.STORAGE },
        { species: e.species, tag: 'instances' },
      )
      const indirectBuf = ctx.res.createBuffer(
        {
          label: `${ctx.id}/indirect-${entryIndex}`,
          size: 32, // draw-indexed-indirect args (20B), rounded up
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        },
        { species: e.species, tag: 'indirect' },
      )
      const cullBg = device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: cullBuf } },
          { binding: 1, resource: { buffer: instanceBuf } },
          { binding: 2, resource: { buffer: indirectBuf } },
        ],
      })
      const renderBg = device.createBindGroup({
        label: `${ctx.id}/render-bg-${entryIndex}`,
        layout: renderBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuf } },
          { binding: 1, resource: { buffer: instanceBuf } },
          { binding: 2, resource: gpu.surfTex.createView() },
          { binding: 3, resource: gpu.geomTex.createView() },
          { binding: 4, resource: sampler },
        ],
      })
      return { entryIndex, speciesId: e.species, slots, isCarpet, gpu, infoBuf, cullBuf, instanceBuf, indirectBuf, cullBg, renderBg }
    }),
  )

  let cullPipeline!: GPUComputePipeline
  let renderPipeline!: GPURenderPipeline
  const build = (): void => {
    const cullModule = ctx.shaders.module(cullSrc)
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: cullModule, entryPoint: 'cs' },
    })
    const renderModule = ctx.shaders.module(renderSrc)
    renderPipeline = device.createRenderPipeline({
      label: `${ctx.id}/chords`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/render-pl`, bindGroupLayouts: [ctx.frame.layout, renderBgl] }),
      vertex: { module: renderModule, entryPoint: 'vs_main' },
      fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      // CCW-outward winding + cull 'front' keeps only the proxy's FAR side:
      // no near-plane clipping and camera-inside still rasterizes.
      primitive: { topology: 'triangle-list', cullMode: 'front' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const infoData = new Float32Array(16)
  const cullData = new Float32Array(16)
  // indexCount, instanceCount, firstIndex, baseVertex, firstInstance
  const indirectReset = new Uint32Array([PROXY_INDICES.length, 0, 0, 0, 0])
  const cellMin = Math.floor(-ctx.stand.radius / SCATTER_CELL_SIZE)
  const cellMax = Math.floor(ctx.stand.radius / SCATTER_CELL_SIZE)
  let side = 1
  // The render uniform holds the baked frustum fit + params ONLY — nothing in
  // it varies per frame, so it is uploaded at init and on param changes. The
  // cull uniform additionally carries the region's corner cell, which changes
  // only when the camera crosses a 4m cell boundary.
  let paramsDirty = true
  let lastOriginX = NaN
  let lastOriginZ = NaN

  const writeInfo = (R: number): void => {
    for (const entry of entries) {
      const f = entry.gpu.field.fit
      infoData[0] = f.r0; infoData[1] = f.r1; infoData[2] = f.h; infoData[3] = f.sideLen
      infoData[4] = f.capB; infoData[5] = f.capT; infoData[6] = f.axisY; infoData[7] = R
      infoData[8] = f.axisX; infoData[9] = f.axisZ
      // A carpet wants its own alpha reference: a chord through a moss cushion
      // returns coverage ~1 in the tile's middle and tails off over its last
      // centimetre, so the grass reference eats that centimetre and leaves a
      // hairline crack lattice in ground that should be closed.
      infoData[10] = entry.isCarpet ? ctx.params.carpetCoverage : ctx.params.coverage
      infoData[11] = ctx.params.entrySmooth ? 1 : 0
      infoData[12] = entry.entryIndex
      infoData[13] = ctx.params.debugChart ? 1 : 0
      infoData[14] = f.rb
      // Carpet-only: pull the per-bin leaf normal towards the mat's macro normal.
      infoData[15] = entry.isCarpet ? ctx.params.carpetNormalMix : 0
      device.queue.writeBuffer(entry.infoBuf, 0, infoData)
    }
  }

  const writeCull = (R: number, originCellX: number, originCellZ: number): void => {
    for (const entry of entries) {
      const f = entry.gpu.field.fit
      // Distance from the scatter node to the proxy's own centre: the mesh
      // origin offset, less the half-tile recentering a carpet applies (see
      // render.wgsl). Recentering also shrinks this from 13cm to 8mm.
      const halfTile = entry.isCarpet ? (speciesById(entry.speciesId).tileM ?? 0) * 0.5 : 0
      const cullRad =
        f.rb + Math.hypot(f.axisX - halfTile, f.axisZ - halfTile) + Math.abs(f.axisY)
      cullData[0] = originCellX; cullData[1] = originCellZ; cullData[2] = side; cullData[3] = ctx.seed
      cullData[4] = entry.entryIndex; cullData[5] = R; cullData[6] = cullRad
      cullData[7] = cellMin; cullData[8] = cellMax
      cullData[9] = entry.slots
      // Only a swaying species needs sway headroom on its cull sphere; moss is
      // rigid, and a 0.5m margin on a 0.09m mat is 30x its own radius.
      cullData[10] = ctx.stand.species[entry.entryIndex]!.sway > 0.01 ? 0.5 : 0
      cullData[11] = f.h
      cullData[12] = entry.isCarpet ? ctx.params.carpetLift : 0
      cullData[13] = 0; cullData[14] = 0; cullData[15] = 0
      device.queue.writeBuffer(entry.cullBuf, 0, cullData)
    }
  }

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, MAX_REGION_RADIUS)
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / SCATTER_CELL_SIZE)
      const originCellZ = Math.floor((cam.z - R) / SCATTER_CELL_SIZE)
      side = Math.max(1, Math.ceil((2 * R) / SCATTER_CELL_SIZE) + 1)

      if (paramsDirty) writeInfo(R)
      if (paramsDirty || originCellX !== lastOriginX || originCellZ !== lastOriginZ) {
        writeCull(R, originCellX, originCellZ)
        lastOriginX = originCellX
        lastOriginZ = originCellZ
      }
      paramsDirty = false

      // Only the indirect counter genuinely changes every frame.
      for (const entry of entries) device.queue.writeBuffer(entry.indirectBuf, 0, indirectReset)
    },

    onParamsChanged(): void {
      paramsDirty = true
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // One thread per candidate scatter slot of the camera-bounded region —
      // bounded by regionRadius, never by the stand's plant count. The slot
      // count is PER ENTRY (a carpet has carpetDiv² = 484 of them, 3.8x the
      // scatter budget); driving this from SCATTER_MAX_PER_CELL rendered 6 of
      // the mat's 22 grid rows.
      const cpass = ctx.timing.computePass(enc, 'cull')
      cpass.setPipeline(cullPipeline)
      cpass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cpass.setBindGroup(1, entry.cullBg)
        cpass.dispatchWorkgroups(Math.ceil((side * side * entry.slots) / 64))
      }
      cpass.end()

      const pass = ctx.timing.renderPass(enc, 'chords', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(renderPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setIndexBuffer(indexBuf, 'uint16')
      for (const entry of entries) {
        pass.setBindGroup(1, entry.renderBg)
        pass.drawIndexedIndirect(entry.indirectBuf, 0)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
