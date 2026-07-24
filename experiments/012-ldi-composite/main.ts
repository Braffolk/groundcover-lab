import cullSrc from './shaders/cull.wgsl'
import ldiSrc from './shaders/ldi.wgsl'
import { ATLAS_H, ATLAS_W, BAKE_VERSION, NUM_LAYERS, TILE, bakeLdi, unpackLdi, type LdiBaked, type LdiMeta } from './bake.ts'
import { SCATTER_MAX_PER_CELL, commitBake, speciesById } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Layered depth images. Each species is baked once into 5 capture directions
 * x 4 min-separation depth-peeled layers (see bake.ts). At runtime a compute
 * pass compacts the plants of a bounded camera-centred region (scatter WGSL
 * twin + frustum cull) into an instance list, and a single indirect draw
 * renders 12 depth-writing cards per visible plant: the two side stacks
 * nearest the camera azimuth plus the top stack, each card at its layer's
 * mean capture depth with per-texel baked depth written via frag_depth. The
 * depth buffer composites the stack union — no sorting, no blending. Cost is
 * bounded by the region, never by the stand's plant count.
 */

const CELL = 4 // must equal SCATTER_CELL_SIZE
const MAX_RADIUS = 128 // must match the regionRadius param max
const NEAR_VERTS = 72 // 3 stacks x 4 layers x 6 verts
const FAR_VERTS = 12 // 2 front-layer cards
const UNIFORM_FLOATS = 20 + 5 * 16 // five header vec4s + 5 DirRec
/** Byte offset + float count of the only uniform fields that change per frame. */
const DYN_OFFSET = 32
const DYN_FLOATS = 8
const INSPECT_MODES = ['off', 'dir', 'layer', 'path'] as const

interface SpeciesGpu {
  atlas0: GPUTexture
  atlas1: GPUTexture
  /** Header only — the 10.5 MB of atlas bytes are dead once uploaded. */
  meta: LdiMeta
}

interface EntryGpu {
  entryIndex: number
  speciesId: string
  gpu: SpeciesGpu
  capacity: number
  uniform: GPUBuffer
  instances: GPUBuffer
  nearIndirect: GPUBuffer
  farIndirect: GPUBuffer
  cullBind: GPUBindGroup
  renderBind: GPUBindGroup
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

  // --- bake or load one LDI set per unique species ---------------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const p = (async (): Promise<SpeciesGpu> => {
      // The dev server answers missing /mesh/baked files with index.html, so
      // validate magic + dims ourselves instead of trusting bakedArtifact().
      const key = `${speciesId}-ldi-v${BAKE_VERSION}`
      let baked: LdiBaked | null = null
      const committed = await fetch(`/mesh/baked/${ctx.id}/${key}.bin`).catch(() => null)
      if (committed?.ok) baked = unpackLdi(await committed.arrayBuffer())
      if (!baked) {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        const packed = await bakeLdi(ctx, mesh)
        baked = unpackLdi(packed)
        if (!baked) throw new Error(`${ctx.id}: fresh bake for ${speciesId} failed to unpack`)
        commitBake(ctx.id, key, packed).catch(() => undefined) // best effort
      }
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
      const atlas0 = mk('albedo')
      const atlas1 = mk('aux')
      const layout = { bytesPerRow: ATLAS_W * 4, rowsPerImage: ATLAS_H }
      device.queue.writeTexture({ texture: atlas0 }, baked.atlas0, layout, [ATLAS_W, ATLAS_H])
      device.queue.writeTexture({ texture: atlas1 }, baked.atlas1, layout, [ATLAS_W, ATLAS_H])
      // Keep only the header: atlas0/atlas1 are views into the whole packed
      // artifact, so holding `baked` would pin ~10.5 MB of host memory per
      // species for the rest of the session for nothing.
      return { atlas0, atlas1, meta: baked.meta }
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

  const sideMax = Math.ceil((2 * MAX_RADIUS) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<EntryGpu> => {
      const gpu = await loadSpecies(e.species)
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
      const mkIndirect = (name: string): GPUBuffer =>
        ctx.res.createBuffer(
          {
            label: `${ctx.id}/${name}-${entryIndex}`,
            size: 16,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          },
          { species: e.species, tag: 'ldi-indirect' },
        )
      const nearIndirect = mkIndirect('near-args')
      const farIndirect = mkIndirect('far-args')
      // vertexCount / firstVertex / firstInstance never change: only the
      // instance count is reset per encode (clearBuffer), so neither draw
      // needs a template copy — and firstInstance stays 0, which is the only
      // value WebGPU accepts without the `indirect-first-instance` feature.
      device.queue.writeBuffer(nearIndirect, 0, new Uint32Array([NEAR_VERTS, 0, 0, 0]))
      device.queue.writeBuffer(farIndirect, 0, new Uint32Array([FAR_VERTS, 0, 0, 0]))

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

      const cullBind = device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: instances } },
          { binding: 2, resource: { buffer: nearIndirect } },
          { binding: 3, resource: { buffer: farIndirect } },
        ],
      })
      const renderBind = device.createBindGroup({
        label: `${ctx.id}/render-bg-${entryIndex}`,
        layout: renderBgl,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: instances } },
          { binding: 2, resource: gpu.atlas0.createView() },
          { binding: 3, resource: gpu.atlas1.createView() },
          { binding: 4, resource: sampler },
        ],
      })
      return {
        entryIndex,
        speciesId: e.species,
        gpu,
        capacity,
        uniform,
        instances,
        nearIndirect,
        farIndirect,
        cullBind,
        renderBind,
      }
    }),
  )

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(ldiSrc)
    const layout = device.createPipelineLayout({ label: `${ctx.id}/pl`, bindGroupLayouts: [ctx.frame.layout, renderBgl] })
    const mkRender = (label: string, vsEntry: string, fsEntry: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${label}`,
        layout,
        vertex: { module, entryPoint: vsEntry },
        fragment: { module, entryPoint: fsEntry, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkRender('near', 'vs_main', 'fs_main')
    farPipeline = mkRender('far', 'vs_far', 'fs_far')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // Only `region` + `tune` move per frame; the rest of the block was written
  // at create time. All entries share these values, so it is one 32-byte
  // scratch reused for every entry's write.
  const dyn = new Float32Array(DYN_FLOATS)
  let side = 1

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      side = Math.max(1, Math.ceil((2 * R) / CELL) + 1)

      dyn[0] = Math.floor((cam.x - R) / CELL)
      dyn[1] = Math.floor((cam.z - R) / CELL)
      dyn[2] = side
      dyn[3] = R
      dyn[4] = ctx.params.layerCullDist
      dyn[5] = ctx.params.parallax ? 1 : 0
      dyn[6] = ctx.params.coverage
      dyn[7] = Math.max(0, INSPECT_MODES.indexOf(ctx.params.inspect))
      for (const entry of entries) device.queue.writeBuffer(entry.uniform, DYN_OFFSET, dyn)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // Reset instance counts inside this encoder so multi-view frames stay correct.
      for (const entry of entries) {
        enc.clearBuffer(entry.nearIndirect, 4, 4)
        enc.clearBuffer(entry.farIndirect, 4, 4)
      }

      const cpass = ctx.timing.computePass(enc, 'ldi-cull')
      cpass.setPipeline(cullPipeline)
      cpass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cpass.setBindGroup(1, entry.cullBind)
        cpass.dispatchWorkgroups(side, side)
      }
      cpass.end()

      const pass = ctx.timing.renderPass(enc, 'ldi-layers', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near first: it writes the depth that lets early-z reject far fragments.
      pass.setPipeline(nearPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.renderBind)
        pass.drawIndirect(entry.nearIndirect, 0)
      }
      pass.setPipeline(farPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.renderBind)
        pass.drawIndirect(entry.farIndirect, 0)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are destroyed by the harness via ctx.res.
    },
  }
}
