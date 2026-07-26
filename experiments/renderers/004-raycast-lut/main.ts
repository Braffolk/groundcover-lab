import cullSrc from './shaders/cull.wgsl'
import impostorSrc from './shaders/impostor.wgsl'
import { loadRayField, RF_ATLAS, type RayField } from './bake.ts'
import { SCATTER_CELL_SIZE, standEntrySlots } from '@harness'
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
const CULL_UBO_BYTES = 128
const DRAW_UBO_BYTES = 64
const CULL_WG = 128 // must match @workgroup_size in cull.wgsl
/**
 * Upper bound on the share of a carpet's grid nodes one wetness zone can claim
 * inside the culled disc. The bog's three Sphagnum states partition the wetness
 * axis into thirds, but the field is not uniformly distributed: measured over
 * the ±96m bog at seed 42 the global split is 16/50/34% and the worst 80m disc
 * anywhere in the stand reaches 63%. 0.85 leaves room for other seeds and
 * terrains while keeping the instance buffer at 8B/tile inside the budget.
 */
const CARPET_ZONE_SHARE = 0.85

interface EntryState {
  speciesId: string
  rayField: RayField
  capacity: number
  /** u32 words per instance record — 2 for a carpet tile, 8 for a scattered plant. */
  stride: number
  /** Workgroups in z, so that EVERY slot of a cell is visited (carpet_div² of them). */
  dispatchZ: number
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
    const carpetDiv = standEntry.carpetDiv ?? 0
    // Slots per 4m cell — carpet_div² for a mat (484 on the bog), which is
    // deliberately MORE than SCATTER_MAX_PER_CELL. Every one must be visited.
    const slots = standEntrySlots(standEntry)
    const dispatchZ = Math.ceil(slots / CULL_WG)
    // Two different numbers: `slots` drives enumeration, `capacity` only has to
    // hold the expected survivors.
    const density = Math.min(standEntry.density, 8)
    const capacity =
      carpetDiv > 0
        ? Math.ceil(
            Math.PI *
              MAX_DIST_CAP *
              MAX_DIST_CAP *
              ((carpetDiv * carpetDiv) / (SCATTER_CELL_SIZE * SCATTER_CELL_SIZE)) *
              CARPET_ZONE_SHARE,
          )
        : Math.ceil(Math.PI * MAX_DIST_CAP * MAX_DIST_CAP * density * 1.15)
    // A carpet tile stores its grid node (cell + slot + 90° yaw index) in 8
    // bytes and rebuilds everything else; a scattered plant keeps its 32B
    // record. At life size the moss needs 4x as many instances as an oversized
    // mat, and 8B/tile is what keeps that inside the VRAM budget.
    const stride = carpetDiv > 0 ? 2 : 8
    const plants = ctx.res.createBuffer(
      {
        label: `${ctx.id}/plants-${entryIndex}-${speciesId}`,
        size: capacity * stride * 4,
        usage: GPUBufferUsage.STORAGE,
      },
      { species: speciesId, tag: 'culled-instances' },
    )
    const cullUbo = ctx.res.createBuffer(
      {
        label: `${ctx.id}/cull-ubo-${entryIndex}`,
        size: CULL_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: speciesId, tag: 'params' },
    )
    const drawUbo = ctx.res.createBuffer(
      {
        label: `${ctx.id}/draw-ubo-${entryIndex}`,
        size: DRAW_UBO_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: speciesId, tag: 'params' },
    )
    const atlas = atlases.get(speciesId)!
    return {
      speciesId,
      rayField,
      capacity,
      stride,
      dispatchZ,
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
  // Cull UBO: 4 frustum planes, baked sphere, region + entry constants.
  const cullData = new ArrayBuffer(CULL_UBO_BYTES)
  const cullI32 = new Int32Array(cullData)
  const cullU32 = new Uint32Array(cullData)
  const cullF32 = new Float32Array(cullData)
  const drawData = new Float32Array(DRAW_UBO_BYTES / 4)

  let cellsX = 0
  let cellsZ = 0
  // The draw UBO holds only bake constants + params, so it is uploaded once
  // and then only when a param actually changes — never per frame.
  let drawUboDirty = true

  const writeDrawUbos = (): void => {
    const maxDist = Math.min(ctx.params.maxDist, MAX_DIST_CAP)
    entries.forEach((entry, entryIndex) => {
      const rf = entry.rayField
      drawData[0] = rf.center[0]
      drawData[1] = rf.center[1]
      drawData[2] = rf.center[2]
      drawData[3] = rf.radius
      // Baked box half-extents: the per-view slab fit AND the sheared quad bound.
      drawData[4] = rf.half[0]
      drawData[5] = rf.half[1]
      drawData[6] = rf.half[2]
      drawData[7] = rf.topH
      drawData[8] = maxDist
      drawData[9] = ctx.params.fadeBand
      drawData[10] = ctx.params.heightAO
      drawData[11] = entryIndex
      drawData[12] = ctx.params.swayMul
      drawData[13] = ctx.params.showCells ? 1 : 0
      drawData[14] = entry.stride
      drawData[15] = ctx.params.matCov
      device.queue.writeBuffer(entry.drawUbo, 0, drawData)
    })
  }

  return {
    onParamsChanged(): void {
      drawUboDirty = true
    },

    update(frame: FrameInfo): void {
      if (drawUboDirty) {
        writeDrawUbos()
        drawUboDirty = false
      }
      const maxDist = Math.min(ctx.params.maxDist, MAX_DIST_CAP)
      const cam = frame.camera.pose
      // Region walked by the cull dispatch, clamped to the stand's own cell
      // range — the shader rejects out-of-stand cells anyway, so dispatching
      // them is pure waste on small stands.
      const lo = Math.floor(-ctx.stand.radius / SCATTER_CELL_SIZE)
      const hi = Math.floor(ctx.stand.radius / SCATTER_CELL_SIZE)
      const c0x = Math.max(lo, Math.floor((cam.x - maxDist) / SCATTER_CELL_SIZE))
      const c0z = Math.max(lo, Math.floor((cam.z - maxDist) / SCATTER_CELL_SIZE))
      const c1x = Math.min(hi, Math.floor((cam.x + maxDist) / SCATTER_CELL_SIZE))
      const c1z = Math.min(hi, Math.floor((cam.z + maxDist) / SCATTER_CELL_SIZE))
      cellsX = Math.max(0, c1x - c0x + 1)
      cellsZ = Math.max(0, c1z - c0z + 1)

      // Side frustum planes (Gribb–Hartmann) from the shared view-projection,
      // normalized so the cull shader can test them against a plant radius.
      const vp = frame.camera.viewProj
      const rowSum = (r: number, sign: number, out: number): void => {
        const x = vp[3]! + sign * vp[r]!
        const y = vp[7]! + sign * vp[4 + r]!
        const z = vp[11]! + sign * vp[8 + r]!
        const w = vp[15]! + sign * vp[12 + r]!
        const inv = 1 / (Math.hypot(x, y, z) || 1)
        cullF32[out] = x * inv
        cullF32[out + 1] = y * inv
        cullF32[out + 2] = z * inv
        cullF32[out + 3] = w * inv
      }
      rowSum(0, 1, 0) // left
      rowSum(0, -1, 4) // right
      rowSum(1, 1, 8) // bottom
      rowSum(1, -1, 12) // top

      device.queue.writeBuffer(drawArgs, 0, drawArgsReset)
      entries.forEach((entry, entryIndex) => {
        const rf = entry.rayField
        cullF32[16] = rf.center[0]
        cullF32[17] = rf.center[1]
        cullF32[18] = rf.center[2]
        cullF32[19] = rf.radius
        cullI32[20] = c0x
        cullI32[21] = c0z
        cullU32[22] = entryIndex
        cullU32[23] = ctx.seed >>> 0
        cullF32[24] = maxDist
        cullF32[25] = ctx.stand.radius
        cullU32[26] = entry.capacity
        cullF32[27] = rf.topH
        cullF32[28] = ctx.params.fadeBand
        cullF32[29] = ctx.params.swayMul
        cullF32[30] = Math.hypot(rf.half[0], rf.half[1], rf.half[2])
        device.queue.writeBuffer(entry.cullUbo, 0, cullData)
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        cull.setBindGroup(1, entry.cullBind)
        // z spans the entry's slots per cell: 1 workgroup for a scattered
        // entry (128 slots), 4 for the bog carpet (484).
        cull.dispatchWorkgroups(cellsX, cellsZ, entry.dispatchZ)
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
