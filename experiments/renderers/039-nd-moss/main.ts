import cullSrc from './shaders/cull.wgsl'
import mossSrc from './shaders/moss.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, loadSpeciesCards, type CardAtlas } from './cards.ts'
import { TEX, loadMossTile, type MossTile } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  speciesById,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
  type WgslSource,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Uncharted 4's moss material on a Sphagnum carpet.
 *
 * Startup: every carpet species gets ONE periodic tile baked from its 19.8M-tri
 * source mesh as the paper's four maps (colour / normal / height / dark cavity
 * AO) packed into two rgba8 textures; every upright species keeps the
 * 001-billboard-smoke card atlas unchanged.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centred cell region and compacts survivors into indirect draws. A
 * carpet entry compacts into THREE lists by distance so cost falls with
 * distance — displaced patches near, flat tile quads mid, 2x2 phase-locked
 * merged quads far. Upright entries compact into one card list.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const PATCH_DIST_MAX = 12 // keep equal to the manifest's patchDist max
const MERGE_DIST_MAX = 32 // keep equal to the manifest's mergeDist max
const MERGE_DIST_MIN = 8 //  keep equal to the manifest's mergeDist min
const CARD_MIPS = Math.floor(Math.log2(ATLAS)) + 1
const MOSS_MIPS = Math.floor(Math.log2(TEX)) + 1
const TOP_FRAC = 0.52
const INFO_FLOATS = 76

interface CardGpu {
  kind: 'card'
  atlas: CardAtlas
  texA: GPUTexture
  texB: GPUTexture
}
interface MossGpu {
  kind: 'moss'
  tile: MossTile
  texA: GPUTexture
  texB: GPUTexture
}
type SpeciesGpu = CardGpu | MossGpu

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  isCarpet: boolean
  /** Instance capacity per LOD list (index 0 is the only one cards use). */
  caps: [number, number, number]
  infoBuffer: GPUBuffer
  /** 3 x drawIndirect args (4 u32 each) in one buffer. */
  argsBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  /** One per LOD list (they differ only in the instance buffer binding). */
  drawBindGroups: GPUBindGroup[]
  enumSlots: number
  slotsPerFrame: number
  /** Carpet geometry in world metres (already scaled by the stand's scale). */
  span: number
  scale: number
  planeH: number
  reliefH: number
  baseY: number
  /** This state's habitat wetness (its band centre), for the wetness function. */
  wetCenter: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const cardSampler = device.createSampler({
    label: `${ctx.id}/card-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })
  // The moss tile is exactly periodic (the bake draws the mesh 3x3 so every
  // neighbour's overhang wraps in), so it is sampled with `repeat` and no inset.
  // Anisotropy matters more here than for cards: a carpet is a ground plane and
  // is nearly always seen at a grazing angle.
  const mossSampler = device.createSampler({
    label: `${ctx.id}/moss-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 8,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  // --- per-species artifacts (sequential: a moss bake transiently needs ~500MB)
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    if ((entry.carpetDiv ?? 0) > 0) {
      const tile = await loadMossTile(ctx, entry.species)
      speciesGpu.set(entry.species, uploadMoss(ctx, entry.species, tile))
    } else {
      const atlas = await loadSpeciesCards(ctx, entry.species)
      speciesGpu.set(entry.species, uploadCards(ctx, entry.species, atlas))
    }
  }

  // --- bind group layouts ----------------------------------------------------
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
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // Visible in the vertex stage too: a near patch displaces its vertices by
      // a coarse mip of the height channel.
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // Worst-case region extent in cells, but never more cells than the stand
  // owns: outside its radius the scatter is empty, so capacity sized for a 128m
  // region on a +/-96m stand is dead VRAM.
  const standSide = Math.floor(ctx.stand.radius / CELL) * 2 + 1
  const sideMax = Math.min(Math.ceil((2 * REGION_MAX) / CELL) + 1, standSide)
  const regionEff = Math.min(REGION_MAX, ctx.stand.radius)

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const carpetDiv = standEntry.carpetDiv ?? 0
    const isCarpet = carpetDiv > 0
    // TWO DIFFERENT NUMBERS — conflating them drops plants:
    //  * enumSlots is how many candidate slots the cull must EVALUATE per cell.
    //    Every slot must be visited or tiles silently disappear (484 for a
    //    life-size carpet, deliberately over the 128-slot scatter budget).
    //  * the caps are how many are expected to SURVIVE into each LOD list.
    const enumSlots = standEntrySlots(standEntry)
    let caps: [number, number, number]
    if (isCarpet) {
      // A carpet's three moss states partition the wetness axis, but the field
      // varies on a 12m grid, so LOCALLY one state owns nearly every node of a
      // cell — capacity must be sized for the full node count, not for a third.
      const perM2 = 1 / (CELL / carpetDiv) ** 2
      const disc = (r: number): number => Math.PI * r * r * perM2
      caps = [
        Math.ceil(disc(PATCH_DIST_MAX) * 1.1) + 512,
        // Mid band, plus slack for zone-boundary blocks anywhere in the region
        // that fail to merge and fall back to per-tile quads.
        Math.ceil(disc(MERGE_DIST_MAX) * 1.1 + disc(regionEff) * 0.05) + 512,
        Math.ceil(((disc(regionEff) - disc(MERGE_DIST_MIN)) / 4) * 1.1) + 512,
      ]
    } else {
      const perCell = SCATTER_MAX_PER_CELL * (Math.min(standEntry.density, 8) / 8)
      caps = [Math.ceil(sideMax * sideMax * perCell * 1.06) + 1024, 1, 1]
    }

    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffers = caps.map((cap, lod) =>
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/instances-${entryIndex}-lod${lod}`,
          size: Math.max(256, cap * 16),
          usage: GPUBufferUsage.STORAGE,
        },
        { species: standEntry.species, tag: `instances-lod${lod}` },
      ),
    )
    const argsBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/args-${entryIndex}`,
        size: 48,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffers[0]! } },
        { binding: 2, resource: { buffer: instBuffers[1]! } },
        { binding: 3, resource: { buffer: instBuffers[2]! } },
        { binding: 4, resource: { buffer: argsBuffer } },
      ],
    })
    const drawBindGroups = instBuffers.map((buf, lod) =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}-lod${lod}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: buf } },
          { binding: 2, resource: gpu.texA.createView() },
          { binding: 3, resource: gpu.texB.createView() },
          { binding: 4, resource: isCarpet ? mossSampler : cardSampler },
        ],
      }),
    )

    // A carpet's scale is derived, not authored: the tile must exactly fill its
    // grid step, so scale = cellSize / div / tileM. The harness computes the
    // same thing in carpetScale(), which is not exported from @harness.
    const tileM = speciesById(standEntry.species).tileM ?? 1
    const scale = isCarpet ? CELL / carpetDiv / tileM : 1
    const t = gpu.kind === 'moss' ? gpu.tile : null
    const range = t ? t.y1 - t.y0 : 0
    return {
      speciesId: standEntry.species,
      gpu,
      isCarpet,
      caps,
      infoBuffer,
      argsBuffer,
      cullBindGroup,
      drawBindGroups,
      enumSlots,
      slotsPerFrame: 0,
      span: tileM * scale,
      scale,
      planeH: t ? (t.y0 + t.planeH) * scale : 0,
      reliefH: range * scale,
      baseY: t ? t.y0 * scale : 0,
      wetCenter: standEntry.wetCenter ?? 0.5,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let cardsPipeline!: GPURenderPipeline
  let patchPipeline!: GPURenderPipeline
  let quadPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/draw-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkDraw = (label: string, src: WgslSource, vs: string, fs: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${label}`,
        layout,
        vertex: { module: ctx.shaders.module(src), entryPoint: vs },
        fragment: {
          module: ctx.shaders.module(src),
          entryPoint: fs,
          targets: [{ format: ctx.colorFormat }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    cardsPipeline = mkDraw('cards', cardsSrc, 'vs_main', 'fs_main')
    patchPipeline = mkDraw('moss-patch', mossSrc, 'vs_patch', 'fs_full')
    quadPipeline = mkDraw('moss-quad', mossSrc, 'vs_quad', 'fs_full')
    farPipeline = mkDraw('moss-far', mossSrc, 'vs_quad', 'fs_cheap')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const argsReset = new Uint32Array(12)

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      const patchDiv = Math.max(2, Math.round(ctx.params.patchDiv))
      // Band-limit the vertex displacement to what the patch can resolve: about
      // one heightmap texel per patch quad.
      const patchMip = Math.max(0, Math.min(MOSS_MIPS - 1, Math.round(Math.log2(TEX / patchDiv))))
      const patchDist = Math.min(ctx.params.patchDist, PATCH_DIST_MAX)
      const mergeDist = Math.max(MERGE_DIST_MIN, Math.min(ctx.params.mergeDist, MERGE_DIST_MAX))
      const flags =
        (ctx.params.microShadow ? 1 : 0) |
        (ctx.params.parallaxShadow ? 2 : 0) |
        (ctx.params.parallaxRefine ? 4 : 0)

      entries.forEach((entry, entryIndex) => {
        argsReset.fill(0)
        if (entry.isCarpet) {
          argsReset[0] = 6 * patchDiv * patchDiv
          argsReset[4] = 6
          argsReset[8] = 6
        } else {
          argsReset[0] = ctx.params.topCard ? 12 : 6
        }

        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.enumSlots
        info[32] = entry.caps[0]
        info[33] = entry.caps[1]
        info[34] = entry.caps[2]
        info[36] = patchDist
        info[37] = mergeDist
        info[38] = flags
        info[39] = entry.span
        info[40] = entry.planeH
        info[41] = entry.reliefH
        info[42] = entry.baseY
        info[43] = ctx.params.gapRef
        info[44] = ctx.params.parallax
        info[45] = ctx.params.displace
        info[46] = patchDiv
        info[47] = patchMip
        info[48] = ctx.params.aoStrength
        info[49] = ctx.params.lightWrap
        info[50] = ctx.params.fuzz
        info[51] = ctx.params.aoSaturation
        info[52] = ctx.params.wetness * entry.wetCenter
        info[53] = entry.scale
        info[54] = ctx.params.aoFresnel
        info[61] = ctx.params.alphaRef
        info[62] = TOP_FRAC
        info[63] = ctx.params.bottomShade
        if (entry.gpu.kind === 'card') {
          const a = entry.gpu.atlas
          info[35] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
          info[56] = a.y0
          info[57] = a.y1
          info[58] = a.rXZ
          info[59] = a.cx
          info[60] = a.cz
          info.set([1, 1, 1], 64)
          info.set([1, 1, 1], 68)
          info.set([1, 1, 1], 72)
        } else {
          // Bounding sphere of one tile: half its diagonal plus its thickness.
          info[35] = entry.span * 0.71 + entry.reliefH
          info[56] = entry.baseY
          info[57] = entry.baseY + entry.reliefH
          info[58] = 0
          info[59] = 0
          info[60] = 0
          info.set(entry.gpu.tile.tipColor, 64)
          info.set(entry.gpu.tile.baseColor, 68)
          info.set(entry.gpu.tile.meanColor, 72)
        }
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.argsBuffer, 0, argsReset)
        entry.slotsPerFrame = sideX * sideZ * entry.enumSlots
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        if (entry.slotsPerFrame === 0) continue // camera outside the stand
        cull.setBindGroup(1, entry.cullBindGroup)
        cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'moss', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Carpet first, near to far: the mat is a solid depth-writing occluder, so
      // anything standing in it is early-z rejected below the surface.
      const lodPipelines = [patchPipeline, quadPipeline, farPipeline]
      for (let lod = 0; lod < 3; lod++) {
        pass.setPipeline(lodPipelines[lod]!)
        for (const entry of entries) {
          if (!entry.isCarpet) continue
          pass.setBindGroup(1, entry.drawBindGroups[lod]!)
          pass.drawIndirect(entry.argsBuffer, lod * 16)
        }
      }
      pass.setPipeline(cardsPipeline)
      for (const entry of entries) {
        if (entry.isCarpet) continue
        pass.setBindGroup(1, entry.drawBindGroups[0]!)
        pass.drawIndirect(entry.argsBuffer, 0)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Upload the two moss tile textures and generate their mip chains. */
function uploadMoss(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, tile: MossTile): MossGpu {
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const mk = (tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${tag}`,
        size: [TEX, TEX],
        format: 'rgba8unorm',
        mipLevelCount: MOSS_MIPS,
        usage,
      },
      { species: speciesId, tag },
    )
  const texA = mk('moss-albedo-height')
  const texB = mk('moss-normal-ao')
  const write = (tex: GPUTexture, data: Uint8Array<ArrayBuffer>): void => {
    ctx.device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: TEX * 4, rowsPerImage: TEX }, [TEX, TEX])
  }
  write(texA, tile.albedoHeight)
  write(texB, tile.normalAo)
  genMips(ctx, [
    { tex: texA, levels: MOSS_MIPS, entryPoint: 'fs_moss_a', format: 'rgba8unorm' },
    { tex: texB, levels: MOSS_MIPS, entryPoint: 'fs_normal', format: 'rgba8unorm' },
  ])
  return { kind: 'moss', tile, texA, texB }
}

/** Upload card atlas mip 0 and generate the mip chain (the 001 baseline path). */
function uploadCards(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: CardAtlas): CardGpu {
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const texA = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS, ATLAS],
      format: 'rgba8unorm',
      mipLevelCount: CARD_MIPS,
      usage,
    },
    { species: speciesId, tag: 'card-albedo' },
  )
  const texB = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [ATLAS, ATLAS],
      format: 'rg8unorm',
      mipLevelCount: CARD_MIPS,
      usage,
    },
    { species: speciesId, tag: 'card-normal' },
  )
  ctx.device.queue.writeTexture({ texture: texA }, atlas.albedo, { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS }, [
    ATLAS,
    ATLAS,
  ])
  ctx.device.queue.writeTexture({ texture: texB }, atlas.normalOct, { bytesPerRow: ATLAS * 2, rowsPerImage: ATLAS }, [
    ATLAS,
    ATLAS,
  ])
  genMips(ctx, [
    { tex: texA, levels: CARD_MIPS, entryPoint: 'fs_albedo', format: 'rgba8unorm' },
    { tex: texB, levels: CARD_MIPS, entryPoint: 'fs_normal', format: 'rg8unorm' },
  ])
  return { kind: 'card', atlas, texA, texB }
}

interface MipJob {
  tex: GPUTexture
  levels: number
  entryPoint: string
  format: GPUTextureFormat
}

/** One render pass per mip level, per texture. Init-time only. */
function genMips(ctx: ExperimentContext<typeof PARAMS>, jobs: MipJob[]): void {
  const { device } = ctx
  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  for (const job of jobs) {
    const pipeline = device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${job.entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: job.entryPoint, targets: [{ format: job.format }] },
      primitive: { topology: 'triangle-list' },
    })
    for (let level = 1; level < job.levels; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: job.tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${job.entryPoint}-${level}`,
        colorAttachments: [
          {
            view: job.tex.createView({ baseMipLevel: level, mipLevelCount: 1 }),
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  device.queue.submit([enc.finish()])
}

/** Gribb–Hartmann frustum planes from a column-major view-proj matrix. */
function frustumPlanes(m: Float32Array | number[], out: Float32Array): void {
  const row = (r: number): [number, number, number, number] => [m[r]!, m[4 + r]!, m[8 + r]!, m[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const list: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r2[0], r2[1], r2[2], r2[3]], // near (WebGPU z >= 0)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // far
  ]
  list.forEach((p, i) => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1
    out[i * 4] = p[0] / len
    out[i * 4 + 1] = p[1] / len
    out[i * 4 + 2] = p[2] / len
    out[i * 4 + 3] = p[3] / len
  })
}
