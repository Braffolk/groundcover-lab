import cullSrc from './shaders/cull.wgsl'
import renderSrc from './shaders/render.wgsl'
import { bakeChordField } from './bake.ts'
import { ATLAS_H, ATLAS_W, unpackChordField, type ChordField } from './chords.ts'
import { bakedArtifact, commitBake, speciesById, SCATTER_CELL_SIZE } from '@harness'
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
const VERTS_PER_PROXY = 96

interface SpeciesGpu {
  field: ChordField
  surfTex: GPUTexture
  geomTex: GPUTexture
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
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
  const instanceCap = maxSide * maxSide * 128

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const gpu = await loadSpecies(e.species)
      const infoBuf = ctx.res.createBuffer(
        { label: `${ctx.id}/info-${entryIndex}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'info' },
      )
      const cullBuf = ctx.res.createBuffer(
        { label: `${ctx.id}/cull-${entryIndex}`, size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'cull-cfg' },
      )
      const instanceBuf = ctx.res.createBuffer(
        { label: `${ctx.id}/instances-${entryIndex}`, size: instanceCap * 32, usage: GPUBufferUsage.STORAGE },
        { species: e.species, tag: 'instances' },
      )
      const indirectBuf = ctx.res.createBuffer(
        {
          label: `${ctx.id}/indirect-${entryIndex}`,
          size: 16,
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
      return { entryIndex, speciesId: e.species, gpu, infoBuf, cullBuf, instanceBuf, indirectBuf, cullBg, renderBg }
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
  const cullData = new Float32Array(12)
  const indirectReset = new Uint32Array([VERTS_PER_PROXY, 0, 0, 0])
  let side = 1

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, MAX_REGION_RADIUS)
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / SCATTER_CELL_SIZE)
      const originCellZ = Math.floor((cam.z - R) / SCATTER_CELL_SIZE)
      side = Math.max(1, Math.ceil((2 * R) / SCATTER_CELL_SIZE) + 1)
      const cellMin = Math.floor(-ctx.stand.radius / SCATTER_CELL_SIZE)
      const cellMax = Math.floor(ctx.stand.radius / SCATTER_CELL_SIZE)

      for (const entry of entries) {
        const f = entry.gpu.field.fit
        infoData[0] = f.r0; infoData[1] = f.r1; infoData[2] = f.h; infoData[3] = f.sideLen
        infoData[4] = f.capB; infoData[5] = f.capT; infoData[6] = f.axisY; infoData[7] = R
        infoData[8] = f.axisX; infoData[9] = f.axisZ; infoData[10] = ctx.params.coverage
        infoData[11] = ctx.params.entrySmooth ? 1 : 0
        infoData[12] = entry.entryIndex
        infoData[13] = ctx.params.debugChart ? 1 : 0
        infoData[14] = f.rb; infoData[15] = 0
        device.queue.writeBuffer(entry.infoBuf, 0, infoData)

        const cullRad = f.rb + Math.hypot(f.axisX, f.axisZ)
        cullData[0] = originCellX; cullData[1] = originCellZ; cullData[2] = side; cullData[3] = ctx.seed
        cullData[4] = entry.entryIndex; cullData[5] = R; cullData[6] = cullRad
        cullData[7] = cellMin; cullData[8] = cellMax
        cullData[9] = 0; cullData[10] = 0; cullData[11] = 0
        device.queue.writeBuffer(entry.cullBuf, 0, cullData)
        device.queue.writeBuffer(entry.indirectBuf, 0, indirectReset)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const threads = side * side * 128
      const groups = Math.ceil(threads / 64)

      const cpass = ctx.timing.computePass(enc, 'cull')
      cpass.setPipeline(cullPipeline)
      cpass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cpass.setBindGroup(1, entry.cullBg)
        cpass.dispatchWorkgroups(groups)
      }
      cpass.end()

      const pass = ctx.timing.renderPass(enc, 'chords', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(renderPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.renderBg)
        pass.drawIndirect(entry.indirectBuf, 0)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
