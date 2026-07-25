import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { loadSpeciesSlabs, tileSide, N_AZIM, N_LAYER, N_TILES, TILE_TOP0, type SlabAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Layered parallax cards.
 *
 * Startup: each species' slab atlases are loaded from mesh/baked / OPFS or
 * baked in-browser from the raw mesh, uploaded and mipped; the per-tile rects
 * and slab plane depths become a small storage buffer.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into TWO
 * indirect draws — a near ring that runs the layered eye-ray reprojection and
 * a far ring that samples one merged tile with a single tap (billboard cost).
 * Each plant is one proxy quad plus one horizontal top quad either way.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const LOD_DIST_MAX = 48 // keep equal to the manifest's lodDistance max
const NEAR_MIPS = 5
const INFO_FLOATS = 48

interface SpeciesGpu {
  atlas: SlabAtlas
  nearAlbedo: GPUTexture
  nearNormal: GPUTexture
  farAlbedo: GPUTexture
  farNormal: GPUTexture
  tileBuffer: GPUBuffer
  sideSpan: number
  topSpan: number
}

interface EntryGpu {
  gpu: SpeciesGpu
  capacity: number
  nearCapacity: number
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroup: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/atlas-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) --
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesSlabs(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- bind group layouts / pipelines ---------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      {
        binding: 3,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' },
      },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const capacity = Math.ceil(slotsMax * (density / 8) * 1.06) + 1024
    // The near ring is a disc of radius lodDistance * scaleMax, so its worst
    // case is a small fraction of the region — sized for the param maximum.
    const nearR = Math.min(LOD_DIST_MAX * standEntry.scaleMax, REGION_MAX)
    const nearCapacity = Math.min(capacity, Math.ceil(Math.PI * nearR * nearR * density * 1.25) + 2048)
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const mkInstances = (name: string, count: number): GPUBuffer =>
      ctx.res.createBuffer(
        { label: `${ctx.id}/${name}-${entryIndex}`, size: count * 16, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: `${name}-instances` },
      )
    const nearBuffer = mkInstances('near', nearCapacity)
    const farBuffer = mkInstances('far', capacity)
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 32,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: nearBuffer } },
        { binding: 2, resource: { buffer: farBuffer } },
        { binding: 3, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = device.createBindGroup({
      label: `${ctx.id}/draw-bg-${entryIndex}`,
      layout: drawBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: nearBuffer } },
        { binding: 2, resource: { buffer: farBuffer } },
        { binding: 3, resource: { buffer: gpu.tileBuffer } },
        { binding: 4, resource: gpu.nearAlbedo.createView() },
        { binding: 5, resource: gpu.nearNormal.createView() },
        { binding: 6, resource: gpu.farAlbedo.createView() },
        { binding: 7, resource: gpu.farNormal.createView() },
        { binding: 8, resource: sampler },
      ],
    })
    return {
      gpu,
      capacity,
      nearCapacity,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroup,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(cardsSrc)
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const cardsLayout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkCards = (name: string, vs: string, fs: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${name}`,
        layout: cardsLayout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkCards('near-cards', 'vs_near', 'fs_near')
    farPipeline = mkCards('far-cards', 'vs_far', 'fs_far')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(8)

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      // Region cell rect, clamped to the stand's cell range on the CPU: cells
      // outside the stand hold nothing, so they must not be dispatched.
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      const verts = ctx.params.topCard ? 12 : 6
      indirectReset[0] = verts
      indirectReset[4] = verts
      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = entry.capacity
        info[32] = a.y0
        info[33] = a.y1
        info[34] = a.rXZ
        info[35] = a.cx
        info[36] = a.cz
        info[37] = Math.hypot(a.rXZ, (a.y1 - a.y0) / 2) * 1.03
        info[38] = ctx.params.alphaRef
        info[39] = ctx.params.lodDistance
        info[40] = entry.gpu.sideSpan
        info[41] = entry.gpu.topSpan
        info[42] = ctx.params.layerShade
        info[43] = ctx.params.bottomShade
        info[44] = entry.nearCapacity
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
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

      const pass = ctx.timing.renderPass(enc, 'slab-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      // Near ring first: it owns the closest pixels, so the depth it writes
      // lets early-z reject the far ring's fragments behind it.
      pass.setPipeline(nearPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 0)
      }
      pass.setPipeline(farPipeline)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.drawBindGroup)
        pass.drawIndirect(entry.indirectBuffer, 16)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

const mipLevels = (w: number, h: number, cap = 32): number =>
  Math.min(cap, Math.floor(Math.log2(Math.max(w, h))) + 1)

/** Upload atlas mip 0, generate the mip chains, build the tile table buffer. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: SlabAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST

  const mkTex = (
    name: string,
    w: number,
    h: number,
    format: GPUTextureFormat,
    levels: number,
    data: Uint8Array<ArrayBuffer>,
    bpp: number,
  ): GPUTexture => {
    const tex = ctx.res.createTexture(
      { label: `${ctx.id}/${speciesId}/${name}`, size: [w, h], format, mipLevelCount: levels, usage },
      { species: speciesId, tag: `slab-${name}` },
    )
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: w * bpp, rowsPerImage: h }, [w, h])
    return tex
  }

  const nearAlbedo = mkTex('near-albedo', atlas.nearW, atlas.nearH, 'rgba8unorm', NEAR_MIPS, atlas.nearAlbedo, 4)
  const nearNormal = mkTex(
    'near-normal',
    atlas.nearNW,
    atlas.nearNH,
    'rg8unorm',
    NEAR_MIPS - 1,
    atlas.nearNormal,
    2,
  )
  // The far atlas is small and is what distant plants minify into, so it gets
  // a full mip chain — that is what keeps the far ring cache friendly.
  const farLevels = mipLevels(atlas.farW, atlas.farH)
  const farAlbedo = mkTex('far-albedo', atlas.farW, atlas.farH, 'rgba8unorm', farLevels, atlas.farAlbedo, 4)
  const farNormal = mkTex(
    'far-normal',
    atlas.farNW,
    atlas.farNH,
    'rg8unorm',
    mipLevels(atlas.farNW, atlas.farNH),
    atlas.farNormal,
    2,
  )

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string, format: GPUTextureFormat): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo', 'rgba8unorm')
  const normalPipe = mkPipeline('fs_normal', 'rg8unorm')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < tex.mipLevelCount; level++) {
      const srcView = tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 })
      const dstView = tex.createView({ baseMipLevel: level, mipLevelCount: 1 })
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: srcView }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${level}`,
        colorAttachments: [{ view: dstView, loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  for (const tex of [nearAlbedo, farAlbedo]) genMips(tex, albedoPipe)
  for (const tex of [nearNormal, farNormal]) genMips(tex, normalPipe)
  device.queue.submit([enc.finish()])

  // Tile table: two vec4 per tile — rect and (plane depth, texW, texH, isFar).
  const table = new Float32Array(N_TILES * 8)
  atlas.tiles.forEach((t, i) => {
    table.set([t.u0, t.v0, t.du, t.dv], i * 8)
    table.set([t.depth, t.texW, t.texH, t.far ? 1 : 0], i * 8 + 4)
  })
  const tileBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/tiles`,
      size: table.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'tile-table' },
  )
  device.queue.writeBuffer(tileBuffer, 0, table)

  // Slab plane separations drive the proxy quad's margins.
  let sideSpan = 0
  for (let k = 0; k < N_AZIM; k++) {
    sideSpan = Math.max(sideSpan, atlas.tiles[tileSide(k, 0)]!.depth - atlas.tiles[tileSide(k, N_LAYER - 1)]!.depth)
  }
  const topSpan = atlas.tiles[TILE_TOP0]!.depth - atlas.tiles[TILE_TOP0 + N_LAYER - 1]!.depth

  return { atlas, nearAlbedo, nearNormal, farAlbedo, farNormal, tileBuffer, sideSpan, topSpan }
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
  list.forEach((pl, i) => {
    const len = Math.hypot(pl[0], pl[1], pl[2]) || 1
    out[i * 4] = pl[0] / len
    out[i * 4 + 1] = pl[1] / len
    out[i * 4 + 2] = pl[2] / len
    out[i * 4 + 3] = pl[3] / len
  })
}
