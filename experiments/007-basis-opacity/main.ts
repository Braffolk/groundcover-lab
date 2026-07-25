import cardShaderSrc from './shaders/card.wgsl'
import carpetShaderSrc from './shaders/carpet.wgsl'
import { bakeSpecies, type BakedCarpet, type BakedSpecies, type BakedSet } from './bake.ts'
import {
  speciesById,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Fourier appearance cards. Each species' view-dependent appearance (opacity,
 * color, depth, normal) is baked once into per-texel truncated Fourier
 * coefficients over view azimuth x E elevation rings — four small rgba8 array
 * textures (~4.2 MB/species), no view atlas. At runtime every plant is a
 * single camera-facing card whose fragments evaluate the basis in closed form
 * (double/triple-angle recurrences); view interpolation is inherently smooth,
 * so there is no 4-view blending and no popping. Placement is the shared
 * scatter WGSL twin over a bounded region around the camera, so per-frame
 * cost is independent of the stand's total plant count.
 *
 * CARPET entries (`carpetDiv > 0`, the bog's Sphagnum mat) take a different
 * shape: one ground-parallel, terrain-conformed, tile-sized quad per tile over a
 * periodic top-down capture, with the Fourier basis moved from the appearance to
 * the tile's HORIZON (sun self-shadow + view-dependent occlusion). See
 * shaders/carpet.wgsl and NOTES.md.
 */

const CELL = 4 // must equal SCATTER_CELL_SIZE

interface CardEntry {
  entryIndex: number
  speciesId: string
  baked: BakedSpecies
  metaBuffer: GPUBuffer
  bindGroup: GPUBindGroup
  /** Slots per cell, rounded up to a power of two (see slotShift). */
  slots: number
  slotShift: number
}

interface CarpetEntry {
  entryIndex: number
  speciesId: string
  baked: BakedCarpet
  metaBuffer: GPUBuffer
  bindGroup: GPUBindGroup
  slots: number
  slotShift: number
}

/** Slots per cell as a power of two + its shift; the scatter rejects the surplus. */
function slotBits(slots: number): { slots: number; slotShift: number } {
  const shift = Math.ceil(Math.log2(Math.max(slots, 1)))
  return { slots: 1 << shift, slotShift: shift }
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
  // Carpet sampler: the tile is periodic and its capture is exactly periodic, so
  // wrapping is the correct addressing (it also makes an overscale seamless).
  // Mips + anisotropy matter here and only here — a mat is the one surface in
  // this renderer that is genuinely minified and viewed at grazing angles.
  const carpetSampler = device.createSampler({
    label: `${ctx.id}/carpet-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    maxAnisotropy: 4,
  })

  // Bake each unique species once per session, fully on the GPU (single submit,
  // no readback). The harness bakedArtifact() cache is intentionally not used:
  // the dev server answers missing /mesh/baked files with 200 index.html (see
  // NOTES.md), and the bake is fast enough to not need it.
  const needBySpecies = new Map<string, { upright: boolean; carpet: boolean }>()
  for (const e of ctx.stand.species) {
    const isCarpet = (e.carpetDiv ?? 0) > 0
    const cur = needBySpecies.get(e.species) ?? { upright: false, carpet: false }
    needBySpecies.set(e.species, { upright: cur.upright || !isCarpet, carpet: cur.carpet || isCarpet })
  }
  const speciesCache = new Map<string, Promise<BakedSet>>()
  const loadSpecies = (speciesId: string): Promise<BakedSet> => {
    let p = speciesCache.get(speciesId)
    if (!p) {
      const need = needBySpecies.get(speciesId) ?? { upright: true, carpet: false }
      const tileM = speciesById(speciesId).tileM ?? 0
      if (need.carpet && tileM <= 0) {
        throw new Error(`${speciesId} is laid out as a carpet but has no periodic tileM`)
      }
      p = ctx.meshes
        .load(speciesById(speciesId).meshId)
        .then((mesh) => bakeSpecies(ctx, mesh, speciesId, { ...need, tileM }))
      speciesCache.set(speciesId, p)
    }
    return p
  }

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const carpetBgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const cards: CardEntry[] = []
  const carpets: CarpetEntry[] = []
  await Promise.all(
    ctx.stand.species.map(async (e, entryIndex): Promise<void> => {
      const baked = await loadSpecies(e.species)
      const metaBuffer = ctx.res.createBuffer(
        { label: `${ctx.id}/meta-${entryIndex}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'meta' },
      )
      // Every slot of the entry must be enumerated — a carpet has carpet_div^2
      // per cell (484 for the bog moss), NOT SCATTER_MAX_PER_CELL.
      const bits = slotBits(standEntrySlots(e))
      if ((e.carpetDiv ?? 0) > 0) {
        const carpet = baked.carpet
        if (!carpet) throw new Error(`${e.species}: carpet entry without a carpet bake`)
        carpets.push({
          entryIndex,
          speciesId: e.species,
          baked: carpet,
          metaBuffer,
          bindGroup: device.createBindGroup({
            label: `${ctx.id}/carpet-bg-${entryIndex}`,
            layout: carpetBgl,
            entries: [
              { binding: 0, resource: { buffer: metaBuffer } },
              { binding: 1, resource: carpet.albedo.createView() },
              { binding: 2, resource: carpet.relief.createView() },
              { binding: 3, resource: carpet.horizon.createView() },
              { binding: 4, resource: carpetSampler },
            ],
          }),
          ...bits,
        })
        return
      }
      const upright = baked.upright
      if (!upright) throw new Error(`${e.species}: upright entry without a ring bake`)
      const view = (t: GPUTexture): GPUTextureView => t.createView({ dimension: '2d-array' })
      cards.push({
        entryIndex,
        speciesId: e.species,
        baked: upright,
        metaBuffer,
        bindGroup: device.createBindGroup({
          label: `${ctx.id}/bg-${entryIndex}`,
          layout: bgl,
          entries: [
            { binding: 0, resource: { buffer: metaBuffer } },
            { binding: 1, resource: view(upright.coeffA) },
            { binding: 2, resource: view(upright.coeffB) },
            { binding: 3, resource: view(upright.coeffC) },
            { binding: 4, resource: view(upright.coeffD) },
            { binding: 5, resource: sampler },
          ],
        }),
        ...bits,
      })
    }),
  )

  // ONE surface height for every carpet entry sharing the grid, not each
  // species' own. The three Sphagnum states have different tile top heights
  // (9.1 / 6.9 / 7.2 cm), so their own mean-apex planes sit up to 1.7 cm apart —
  // and since a ground quad has no side wall, every zone boundary then showed a
  // step you could see straight through, down onto the dark peat: at grazing the
  // interlocking boundary tiles were outlined in hard dark bands (screenshotted).
  // The mat has to be one continuous surface, so all carpet entries share the
  // mean plane. Purely how high the mat is drawn — position, yaw and scale are
  // still exactly what the scatter gave.
  const carpetPlaneH =
    carpets.length > 0 ? carpets.reduce((s, c) => s + c.baked.planeH, 0) / carpets.length : 0

  let pipeline!: GPURenderPipeline
  let carpetPipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(cardShaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/cards`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    if (carpets.length > 0) {
      const carpetModule = ctx.shaders.module(carpetShaderSrc)
      carpetPipeline = device.createRenderPipeline({
        label: `${ctx.id}/carpet`,
        layout: device.createPipelineLayout({
          label: `${ctx.id}/carpet-pl`,
          bindGroupLayouts: [ctx.frame.layout, carpetBgl],
        }),
        vertex: { module: carpetModule, entryPoint: 'vs_main' },
        fragment: { module: carpetModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
        // A tile quad is 4 vertices as a strip instead of 6 as a list; at ~800k
        // tiles per moss entry that is a third of the vertex work saved.
        primitive: { topology: 'triangle-strip', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    }
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const meta = new Float32Array(16)
  let side = 1

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / CELL)
      const originCellZ = Math.floor((cam.z - R) / CELL)
      side = Math.max(1, Math.ceil((2 * R) / CELL) + 1)

      for (const entry of cards) {
        const b = entry.baked
        meta[0] = b.center[0]; meta[1] = b.center[1]; meta[2] = b.center[2]; meta[3] = b.radius
        meta[4] = originCellX; meta[5] = originCellZ; meta[6] = side; meta[7] = ctx.seed
        meta[8] = R; meta[9] = ctx.params.alphaGain; meta[10] = ctx.params.dither; meta[11] = entry.entryIndex
        meta[12] = ctx.params.colorView; meta[13] = b.extXZ; meta[14] = b.extY; meta[15] = ctx.params.ringTint
        device.queue.writeBuffer(entry.metaBuffer, 0, meta)
      }
      for (const entry of carpets) {
        const b = entry.baked
        meta[0] = originCellX; meta[1] = originCellZ; meta[2] = side; meta[3] = ctx.seed
        meta[4] = R; meta[5] = entry.entryIndex; meta[6] = entry.slots - 1; meta[7] = entry.slotShift
        meta[8] = carpetPlaneH; meta[9] = ctx.params.carpetAlphaRef; meta[10] = ctx.params.sunShadow
        meta[11] = ctx.params.viewOcclusion
        meta[12] = ctx.params.carpetAo; meta[13] = ctx.params.carpetOverscale; meta[14] = b.topH
        meta[15] = ctx.params.reliefTint
        device.queue.writeBuffer(entry.metaBuffer, 0, meta)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'fourier-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of cards) {
        pass.setBindGroup(1, entry.bindGroup)
        pass.draw(6, side * side * entry.slots)
      }
      pass.end()

      if (carpets.length === 0) return
      const carpetPass = ctx.timing.renderPass(enc, 'carpet-tiles', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      carpetPass.setPipeline(carpetPipeline)
      carpetPass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of carpets) {
        carpetPass.setBindGroup(1, entry.bindGroup)
        carpetPass.draw(4, side * side * entry.slots)
      }
      carpetPass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are destroyed by the harness via ctx.res.
    },
  }
}
