import cullSrc from './shaders/cull.wgsl'
import impostorSrc from './shaders/impostor.wgsl'
import { loadRayField, RF_ATLAS, type RayField } from './bake.ts'
import { SCATTER_CELL_SIZE } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Ray-LUT impostors: every species is baked into a 24x24-direction atlas of
 * precomputed raycast answers (albedo/coverage/depth/normal per ray). At
 * runtime a GPU cull pass materializes only the plants within `maxDist` of
 * the camera from the shared scatter twin (region-bounded work, plant-count
 * independent), and each plant is a single quad whose fragments answer the
 * eye ray with ~2 LUT fetches — no source geometry, no marching.
 */

const MAX_DIST_CAP = 96 // must match PARAMS.maxDist max — sizes the buffers
const DEBUG_MODES = ['lit', 'albedo', 'normal', 'coverage'] as const

interface EntryState {
  speciesId: string
  rayField: RayField
  capacity: number
  cullUbo: GPUBuffer
  drawUbo: GPUBuffer
  cullBind: GPUBindGroup
  drawBind: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // --- bake / load the per-species ray-answer fields -----------------------
  const uniqueSpecies = [...new Set(ctx.stand.species.map((e) => e.species))]
  const rayFields = new Map<string, RayField>()
  for (const speciesId of uniqueSpecies) {
    rayFields.set(
      speciesId,
      await loadRayField(ctx, speciesId, (f, note) => {
        if (note) console.log(`[${ctx.id}] bake ${Math.round(f * 100)}% — ${note}`)
      }),
    )
  }

  // --- per-species atlas textures ------------------------------------------
  const atlases = new Map<string, { surf: GPUTexture; geom: GPUTexture }>()
  for (const speciesId of uniqueSpecies) {
    const rf = rayFields.get(speciesId)!
    const mk = (label: string, data: Uint8Array): GPUTexture => {
      const tex = ctx.res.createTexture(
        {
          label: `${ctx.id}/${speciesId}/${label}`,
          size: [RF_ATLAS, RF_ATLAS],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species: speciesId, tag: 'ray-lut' },
      )
      device.queue.writeTexture(
        { texture: tex },
        data as unknown as BufferSource,
        { bytesPerRow: RF_ATLAS * 4 },
        [RF_ATLAS, RF_ATLAS],
      )
      return tex
    }
    atlases.set(speciesId, { surf: mk('atlas-surf', rf.surf), geom: mk('atlas-geom', rf.geom) })
  }

  // --- shared buffers ------------------------------------------------------
  const entryCount = ctx.stand.species.length
  const drawArgs = ctx.res.createBuffer(
    {
      label: `${ctx.id}/draw-args`,
      size: entryCount * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    },
    { tag: 'indirect' },
  )

  // --- bind group layouts & pipelines --------------------------------------
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
      label: `${ctx.id}/impostors`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/draw`, bindGroupLayouts: [ctx.frame.layout, drawBgl] }),
      vertex: { module: ctx.shaders.module(impostorSrc), entryPoint: 'vs_main' },
      fragment: {
        module: ctx.shaders.module(impostorSrc),
        entryPoint: 'fs_main',
        targets: [{ format: ctx.colorFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- per-entry state -----------------------------------------------------
  const entries: EntryState[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const speciesId = standEntry.species
    const rayField = rayFields.get(speciesId)!
    const density = Math.min(standEntry.density, 8)
    const capacity = Math.ceil(Math.PI * MAX_DIST_CAP * MAX_DIST_CAP * density * 1.15)
    const plants = ctx.res.createBuffer(
      {
        label: `${ctx.id}/plants-${entryIndex}-${speciesId}`,
        size: capacity * 32,
        usage: GPUBufferUsage.STORAGE,
      },
      { species: speciesId, tag: 'culled-instances' },
    )
    const cullUbo = ctx.res.createBuffer(
      { label: `${ctx.id}/cull-ubo-${entryIndex}`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: speciesId, tag: 'params' },
    )
    const drawUbo = ctx.res.createBuffer(
      { label: `${ctx.id}/draw-ubo-${entryIndex}`, size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: speciesId, tag: 'params' },
    )
    const atlas = atlases.get(speciesId)!
    return {
      speciesId,
      rayField,
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
          { binding: 2, resource: atlas.surf.createView() },
          { binding: 3, resource: atlas.geom.createView() },
        ],
      }),
    }
  })

  const drawArgsReset = new Uint32Array(entryCount * 4)
  for (let i = 0; i < entryCount; i++) drawArgsReset[i * 4] = 6
  const cullData = new ArrayBuffer(32)
  const cullI32 = new Int32Array(cullData)
  const cullU32 = new Uint32Array(cullData)
  const cullF32 = new Float32Array(cullData)
  const drawData = new Float32Array(12)

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
        cullI32[0] = c0x
        cullI32[1] = c0z
        cullU32[2] = entryIndex
        cullU32[3] = ctx.seed >>> 0
        cullF32[4] = maxDist
        cullF32[5] = ctx.stand.radius
        cullU32[6] = entry.capacity
        cullU32[7] = 0
        device.queue.writeBuffer(entry.cullUbo, 0, cullData)

        const rf = entry.rayField
        drawData[0] = rf.center[0]
        drawData[1] = rf.center[1]
        drawData[2] = rf.center[2]
        drawData[3] = rf.radius
        drawData[4] = rf.topH
        drawData[5] = maxDist
        drawData[6] = ctx.params.fadeBand
        drawData[7] = ctx.params.heightAO
        drawData[8] = entryIndex
        drawData[9] = ctx.params.swayMul
        drawData[10] = Math.max(0, DEBUG_MODES.indexOf(ctx.params.debugView))
        drawData[11] = 0
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

      const pass = ctx.timing.renderPass(enc, 'impostors', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: {
          view: targets.depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
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
