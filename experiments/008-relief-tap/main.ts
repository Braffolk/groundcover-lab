import cullSrc from './shaders/cull.wgsl'
import reliefSrc from './shaders/relief.wgsl'
import { loadSpeciesBake, type SpeciesBake } from './bake.ts'
import { SCATTER_CELL_SIZE, standEntrySlots } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Secant relief cards.
 *
 * Scattered plants: each species is baked into a 5x5 hemi-octahedral fan of
 * depth-augmented orthographic captures (albedo+coverage, signed heightfield,
 * baked normals) and drawn as one camera-facing card whose fragments intersect
 * the eye ray with the selected view's heightfield using a FIXED 3-tap secant
 * scheme — no loops, no marching — then write the reconstructed hit as real
 * frag_depth.
 *
 * Carpet species (stand carpetDiv > 0, e.g. Sphagnum): the same relief solve,
 * but over ONE zenith capture cropped to the periodic tile, on a ground-
 * parallel tile-sized quad that is terrain-conformed per vertex. See
 * shaders/relief.wgsl and bake.ts for why a mat wants the whole budget spent on
 * that single view.
 *
 * Either way a GPU cull pass materializes only the plants within maxDist of the
 * camera from the shared scatter twin (fixed-size dispatch — plant-count
 * independent).
 */

const MAX_DIST_CAP = 88 // must match PARAMS.maxDist max — sizes the buffers
const RELIEF_MODES = ['flat-1tap', 'linear-2tap', 'secant-3tap'] as const
const VIEW_MODES = ['stochastic', 'nearest'] as const
const INSPECT_MODES = ['off', 'height', 'view-cell'] as const

interface EntryState {
  speciesId: string
  bake: SpeciesBake
  capacity: number
  /** Candidate slots per scatter cell — carpetDiv^2 for a mat, else 128. */
  slots: number
  cullUbo: GPUBuffer
  drawUbo: GPUBuffer
  cullBind: GPUBindGroup
  drawBind: GPUBindGroup
  /** Last bytes uploaded to drawUbo — its contents depend only on the bake and
   *  the params, so it is re-uploaded on change, not every frame. */
  lastDraw: Float32Array
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // --- bake the per-species representations and upload them ------------------
  // A species is baked as a CARPET if the active stand lays it out as one; the
  // two bakes have completely different budget splits (bake.ts).
  const carpetOf = new Map<string, boolean>()
  for (const entry of ctx.stand.species) {
    carpetOf.set(entry.species, (entry.carpetDiv ?? 0) > 0)
  }
  const baked = new Map<string, { bake: SpeciesBake; albedo: GPUTexture; geom: GPUTexture }>()
  for (const [speciesId, isCarpet] of carpetOf) {
    const bake = await loadSpeciesBake(ctx, speciesId, isCarpet)
    const mips = bake.albedoLevels.length
    const albedo = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/albedo`,
        size: [bake.px, bake.px],
        mipLevelCount: mips,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag: 'relief-albedo' },
    )
    const geom = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/geom`,
        size: [bake.px, bake.px],
        mipLevelCount: bake.geomLevels.length,
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag: 'relief-geom' },
    )
    bake.albedoLevels.forEach((data, level) => {
      const n = Math.max(1, bake.px >> level)
      device.queue.writeTexture(
        { texture: albedo, mipLevel: level },
        data as unknown as BufferSource,
        { bytesPerRow: n * 4, rowsPerImage: n },
        [n, n],
      )
    })
    bake.geomLevels.forEach((data, level) => {
      const n = Math.max(1, bake.px >> level)
      device.queue.writeTexture(
        { texture: geom, mipLevel: level },
        data as unknown as BufferSource,
        { bytesPerRow: n * 8, rowsPerImage: n },
        [n, n],
      )
    })
    baked.set(speciesId, { bake, albedo, geom })
  }

  // Carpet tiles only: REPEAT because the cropped tile is exactly periodic;
  // mipmapped because a single view has no neighbouring cells to bleed into;
  // ANISOTROPIC because a ground mat is looked at at grazing angles almost all
  // the time, and an isotropic level chosen for the long axis of that footprint
  // collapses the whole field into per-tile mean colours.
  const carpetSampler = device.createSampler({
    label: `${ctx.id}/carpet-sampler`,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 16,
  })

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
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  let cullPipeline!: GPUComputePipeline
  let cardPipeline!: GPURenderPipeline
  let carpetPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/draw`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const module = ctx.shaders.module(reliefSrc)
    const common = {
      layout,
      primitive: { topology: 'triangle-list' as const, cullMode: 'none' as const },
      depthStencil: {
        format: ctx.depthFormat,
        depthCompare: 'less' as const,
        depthWriteEnabled: true,
      },
    }
    cardPipeline = device.createRenderPipeline({
      ...common,
      label: `${ctx.id}/relief-cards`,
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
    })
    carpetPipeline = device.createRenderPipeline({
      ...common,
      label: `${ctx.id}/relief-carpet`,
      vertex: { module, entryPoint: 'vs_carpet' },
      fragment: { module, entryPoint: 'fs_carpet', targets: [{ format: ctx.colorFormat }] },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- per-entry state ------------------------------------------------------
  const entries: EntryState[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const speciesId = standEntry.species
    const b = baked.get(speciesId)!
    const isCarpet = b.bake.carpet
    // EVERY slot must be evaluated (a carpet has carpet_div^2 of them, over the
    // 128-slot scatter budget), but capacity only has to hold the expected
    // survivors.
    const slots = standEntrySlots(standEntry)
    const area = Math.PI * MAX_DIST_CAP * MAX_DIST_CAP
    let capacity: number
    if (isCarpet) {
      // The bog's three moss states PARTITION the wetness axis, so an entry
      // nominally claims `wetWidth` of the grid nodes. But wetness is damped on
      // slopes, so over steep ground the distribution collapses toward 0 and the
      // driest zone claims far more than its nominal third — hence the flat
      // headroom on top of the width. Sizing all three for all 484 slots instead
      // would waste ~3x (only one entry can own a node).
      const accept = Math.min(1, (standEntry.wetWidth ?? 1) * 1.25 + 0.35)
      capacity = Math.ceil((area * slots * accept) / (SCATTER_CELL_SIZE * SCATTER_CELL_SIZE))
    } else {
      capacity = Math.ceil(area * Math.min(standEntry.density, 8) * 1.15)
    }
    const plants = ctx.res.createBuffer(
      {
        label: `${ctx.id}/plants-${entryIndex}-${speciesId}`,
        size: capacity * (isCarpet ? 16 : 32),
        usage: GPUBufferUsage.STORAGE,
      },
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
      bake: b.bake,
      capacity,
      slots,
      cullUbo,
      drawUbo,
      lastDraw: new Float32Array(24).fill(NaN),
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
          { binding: 4, resource: carpetSampler },
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
      // Cell window = the camera's maxDist box intersected with the stand's
      // cell region. The shader rejects out-of-region cells anyway (same lo/hi
      // test), so clipping here changes nothing except not launching those
      // workgroups at all.
      const lo = Math.floor(-ctx.stand.radius / SCATTER_CELL_SIZE)
      const hi = Math.floor(ctx.stand.radius / SCATTER_CELL_SIZE)
      const c0x = Math.max(Math.floor((cam.x - maxDist) / SCATTER_CELL_SIZE), lo)
      const c0z = Math.max(Math.floor((cam.z - maxDist) / SCATTER_CELL_SIZE), lo)
      cellsX = Math.max(0, Math.min(Math.floor((cam.x + maxDist) / SCATTER_CELL_SIZE), hi) - c0x + 1)
      cellsZ = Math.max(0, Math.min(Math.floor((cam.z + maxDist) / SCATTER_CELL_SIZE), hi) - c0z + 1)

      device.queue.writeBuffer(drawArgs, 0, drawArgsReset)
      entries.forEach((entry, entryIndex) => {
        const v = entry.bake
        // A carpet quad is centred on its grid node and spans the tile square,
        // so its bound is the tile's half diagonal plus its height — not the
        // source mesh's off-origin bounding sphere.
        const sphereY = v.carpet ? v.topH * 0.5 : v.center[1]
        const sphereR = v.carpet
          ? v.radius * Math.SQRT2 + v.topH
          : v.radius + Math.hypot(v.center[0], v.center[2])
        cullI32[0] = c0x
        cullI32[1] = c0z
        cullU32[2] = entryIndex
        cullU32[3] = ctx.seed >>> 0
        cullF32[4] = maxDist
        cullF32[5] = ctx.stand.radius
        cullU32[6] = entry.capacity
        cullF32[7] = sphereY
        cullF32[8] = sphereR
        cullU32[9] = entry.slots
        cullU32[10] = cellsX
        cullU32[11] = cellsZ
        device.queue.writeBuffer(entry.cullUbo, 0, cullData)

        drawData[0] = v.center[0]
        drawData[1] = v.center[1]
        drawData[2] = v.center[2]
        drawData[3] = v.radius
        drawData[4] = v.halfExt[0]
        drawData[5] = v.halfExt[1]
        drawData[6] = v.halfExt[2]
        drawData[7] = v.topH
        drawData[8] = v.grid
        drawData[9] = v.px / v.grid
        drawData[10] = v.px
        drawData[11] = maxDist
        drawData[12] = ctx.params.fadeBand
        drawData[13] = ctx.params.covThresh
        drawData[14] = ctx.params.ao
        drawData[15] = entryIndex
        drawData[16] = ctx.params.swayMul
        drawData[17] = Math.max(0, RELIEF_MODES.indexOf(ctx.params.reliefMode))
        drawData[18] = Math.max(0, VIEW_MODES.indexOf(ctx.params.viewSelect))
        drawData[19] = entry.capacity
        drawData[20] = Math.max(0, INSPECT_MODES.indexOf(ctx.params.inspect))
        drawData[21] = v.hMean
        drawData[22] = v.hSigma
        // Nothing in here depends on the camera or on time — upload only when
        // a param (or the bake) actually changed it.
        let dirty = false
        for (let i = 0; i < drawData.length; i++) {
          if (drawData[i] !== entry.lastDraw[i]) {
            dirty = true
            break
          }
        }
        if (dirty) {
          entry.lastDraw.set(drawData)
          device.queue.writeBuffer(entry.drawUbo, 0, drawData)
        }
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      if (cellsX > 0 && cellsZ > 0) {
        cull.setPipeline(cullPipeline)
        cull.setBindGroup(0, ctx.frame.bindGroup)
        for (const entry of entries) {
          cull.setBindGroup(1, entry.cullBind)
          // One thread per (cell, slot): a carpet entry simply takes more
          // workgroups per cell than a scattered one.
          cull.dispatchWorkgroups(Math.ceil((cellsX * cellsZ * entry.slots) / 128))
        }
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'relief-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      let active: GPURenderPipeline | null = null
      entries.forEach((entry, entryIndex) => {
        const want = entry.bake.carpet ? carpetPipeline : cardPipeline
        if (want !== active) {
          pass.setPipeline(want)
          active = want
        }
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
