import cullSrc from './shaders/cull.wgsl'
import reliefSrc from './shaders/relief.wgsl'
import { ATLAS, bakeViews, GRID, TILE, type BakedViews } from './bake.ts'
import { SCATTER_CELL_SIZE, speciesById } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Secant relief cards. Each species is baked once into a 5x5 hemi-octahedral
 * fan of depth-augmented orthographic captures (albedo+coverage, signed
 * heightfield, baked normals). At runtime a GPU cull pass materializes only
 * the plants within maxDist of the camera from the shared scatter twin
 * (fixed-size dispatch — plant-count independent), and every plant is a
 * single card whose fragments intersect the eye ray with the selected view's
 * heightfield using a FIXED 3-tap secant scheme — no loops, no marching —
 * then write the reconstructed hit as real frag_depth.
 */

const MAX_DIST_CAP = 88 // must match PARAMS.maxDist max — sizes the buffers
const RELIEF_MODES = ['flat-1tap', 'linear-2tap', 'secant-3tap'] as const
const VIEW_MODES = ['stochastic', 'nearest'] as const
const DEBUG_MODES = ['lit', 'albedo', 'normal', 'height'] as const

interface EntryState {
  speciesId: string
  views: BakedViews
  capacity: number
  cullUbo: GPUBuffer
  drawUbo: GPUBuffer
  cullBind: GPUBindGroup
  drawBind: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // --- bake the per-species view sets and upload the atlases ----------------
  const uniqueSpecies = [...new Set(ctx.stand.species.map((e) => e.species))]
  const baked = new Map<string, { views: BakedViews; albedo: GPUTexture; geom: GPUTexture }>()
  for (const speciesId of uniqueSpecies) {
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    const views = await bakeViews(ctx, mesh)
    const albedo = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/albedo`,
        size: [ATLAS, ATLAS],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag: 'relief-albedo' },
    )
    const geom = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/geom`,
        size: [ATLAS, ATLAS],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag: 'relief-geom' },
    )
    device.queue.writeTexture(
      { texture: albedo },
      views.albedo as unknown as BufferSource,
      { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS },
      [ATLAS, ATLAS],
    )
    device.queue.writeTexture(
      { texture: geom },
      views.geom as unknown as BufferSource,
      { bytesPerRow: ATLAS * 8, rowsPerImage: ATLAS },
      [ATLAS, ATLAS],
    )
    baked.set(speciesId, { views, albedo, geom })
  }

  // --- shared buffers -------------------------------------------------------
  const entryCount = ctx.stand.species.length
  const drawArgs = ctx.res.createBuffer(
    {
      label: `${ctx.id}/draw-args`,
      size: entryCount * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    },
    { tag: 'indirect' },
  )

  // --- layouts & pipelines --------------------------------------------------
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
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  })

  let cullPipeline!: GPUComputePipeline
  let drawPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    drawPipeline = device.createRenderPipeline({
      label: `${ctx.id}/relief-cards`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/draw`, bindGroupLayouts: [ctx.frame.layout, drawBgl] }),
      vertex: { module: ctx.shaders.module(reliefSrc), entryPoint: 'vs_main' },
      fragment: { module: ctx.shaders.module(reliefSrc), entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- per-entry state ------------------------------------------------------
  const entries: EntryState[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const speciesId = standEntry.species
    const b = baked.get(speciesId)!
    const density = Math.min(standEntry.density, 8)
    const capacity = Math.ceil(Math.PI * MAX_DIST_CAP * MAX_DIST_CAP * density * 1.15)
    const plants = ctx.res.createBuffer(
      { label: `${ctx.id}/plants-${entryIndex}-${speciesId}`, size: capacity * 32, usage: GPUBufferUsage.STORAGE },
      { species: speciesId, tag: 'culled-instances' },
    )
    const cullUbo = ctx.res.createBuffer(
      { label: `${ctx.id}/cull-ubo-${entryIndex}`, size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: speciesId, tag: 'params' },
    )
    const drawUbo = ctx.res.createBuffer(
      { label: `${ctx.id}/draw-ubo-${entryIndex}`, size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: speciesId, tag: 'params' },
    )
    return {
      speciesId,
      views: b.views,
      capacity,
      cullUbo,
      drawUbo,
      cullBind: device.createBindGroup({
        label: `${ctx.id}/cull-bind-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: cullUbo } },
          { binding: 1, resource: { buffer: drawArgs } },
          { binding: 2, resource: { buffer: plants } },
        ],
      }),
      drawBind: device.createBindGroup({
        label: `${ctx.id}/draw-bind-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: plants } },
          { binding: 1, resource: { buffer: drawUbo } },
          { binding: 2, resource: b.albedo.createView() },
          { binding: 3, resource: b.geom.createView() },
        ],
      }),
    }
  })

  const drawArgsReset = new Uint32Array(entryCount * 4)
  for (let i = 0; i < entryCount; i++) drawArgsReset[i * 4] = 6
  const cullData = new ArrayBuffer(48)
  const cullI32 = new Int32Array(cullData)
  const cullU32 = new Uint32Array(cullData)
  const cullF32 = new Float32Array(cullData)
  const drawData = new Float32Array(24)

  let cellsX = 1
  let cellsZ = 1

  return {
    update(frame: FrameInfo): void {
      const maxDist = Math.min(ctx.params.maxDist, MAX_DIST_CAP)
      const cam = frame.camera.pose
      const c0x = Math.floor((cam.x - maxDist) / SCATTER_CELL_SIZE)
      const c0z = Math.floor((cam.z - maxDist) / SCATTER_CELL_SIZE)
      cellsX = Math.floor((cam.x + maxDist) / SCATTER_CELL_SIZE) - c0x + 1
      cellsZ = Math.floor((cam.z + maxDist) / SCATTER_CELL_SIZE) - c0z + 1

      device.queue.writeBuffer(drawArgs, 0, drawArgsReset)
      entries.forEach((entry, entryIndex) => {
        const v = entry.views
        cullI32[0] = c0x
        cullI32[1] = c0z
        cullU32[2] = entryIndex
        cullU32[3] = ctx.seed >>> 0
        cullF32[4] = maxDist
        cullF32[5] = ctx.stand.radius
        cullU32[6] = entry.capacity
        cullF32[7] = v.center[1]
        cullF32[8] = v.radius + Math.hypot(v.center[0], v.center[2])
        device.queue.writeBuffer(entry.cullUbo, 0, cullData)

        drawData[0] = v.center[0]
        drawData[1] = v.center[1]
        drawData[2] = v.center[2]
        drawData[3] = v.radius
        drawData[4] = v.halfExt[0]
        drawData[5] = v.halfExt[1]
        drawData[6] = v.halfExt[2]
        drawData[7] = v.topH
        drawData[8] = GRID
        drawData[9] = TILE
        drawData[10] = ATLAS
        drawData[11] = maxDist
        drawData[12] = ctx.params.fadeBand
        drawData[13] = ctx.params.covThresh
        drawData[14] = ctx.params.ao
        drawData[15] = entryIndex
        drawData[16] = ctx.params.swayMul
        drawData[17] = Math.max(0, RELIEF_MODES.indexOf(ctx.params.reliefMode))
        drawData[18] = Math.max(0, VIEW_MODES.indexOf(ctx.params.viewSelect))
        drawData[19] = entry.capacity
        drawData[20] = Math.max(0, DEBUG_MODES.indexOf(ctx.params.debugView))
        device.queue.writeBuffer(entry.drawUbo, 0, drawData)
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cull.setBindGroup(1, entry.cullBind)
        cull.dispatchWorkgroups(cellsX, cellsZ, 1)
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'relief-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(drawPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      entries.forEach((entry, entryIndex) => {
        pass.setBindGroup(1, entry.drawBind)
        pass.drawIndirect(drawArgs, entryIndex * 16)
      })
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
