import cullSrc from './shaders/cull.wgsl'
import shellSrc from './shaders/shell.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import thicknessSrc from './shaders/thickness.wgsl'
import { GRID_V, N_VIEWS, TILE_A, TILE_B, loadSpeciesDome, type DomeAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Depth-shell impostors + screen-space canopy thickness.
 *
 * Startup: per species a 13-view baked atlas (8 azimuths at the horizon, 4 at
 * 45deg, 1 from above) is loaded/baked, uploaded as two texture arrays
 * (albedo+coverage, oct normal + burial AO + thickness) and mipped, together
 * with a tiny per-view 9x9 FRONT-DEPTH SHELL.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, picks each plant's baked view
 * and shell LOD, and compacts survivors into three indirect draws. The draw is
 * ONE camera-facing card per plant whose lattice is pushed along the view axis
 * by the shell, so the card is a genuine 3D surface in the depth buffer. Two
 * tiny reductions + a multiply-blend composite turn the per-fragment canopy
 * weight into inter-plant volume and contact darkening.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count,
 * and collapses 8x8 shell -> 4x4 shell -> flat quad with distance.
 */

const CELL = SCATTER_CELL_SIZE
/** Keep equal to the manifest's regionRadius max. */
const REGION_MAX = 112
/** Keep equal to the manifest's lodNear / lodFar maxima. */
const LOD0_MAX = 16
const LOD1_MAX = 48
const MIPS_A = Math.floor(Math.log2(TILE_A)) + 1
const MIPS_B = Math.floor(Math.log2(TILE_B)) + 1
const INFO_FLOATS = 100
const CELLS = GRID_V - 1
const IDX_LOD0 = CELLS * CELLS * 6
const IDX_LOD1 = ((CELLS / 2) * CELLS) / 2 * 6

/** Index ranges over the shared lattice: [indexCount, firstIndex] per LOD. */
const LOD_RANGES: [number, number][] = [
  [IDX_LOD0, 0],
  [IDX_LOD1, IDX_LOD0],
  [6, IDX_LOD0 + IDX_LOD1],
]

interface SpeciesGpu {
  atlas: DomeAtlas
  albedoTex: GPUTexture
  attrTex: GPUTexture
  shellBuffer: GPUBuffer
}

interface EntryGpu {
  gpu: SpeciesGpu
  caps: [number, number, number]
  infoBuffer: GPUBuffer
  argsBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroups: GPUBindGroup[]
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
  const screenSampler = device.createSampler({
    label: `${ctx.id}/canopy-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'nearest',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) --
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesDome(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- shared lattice index buffer -----------------------------------------
  const indices = buildLatticeIndices()
  const indexBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/lattice-indices`,
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'lattice-indices' },
  )
  device.queue.writeBuffer(indexBuffer, 0, indices)

  // --- bind group layouts ---------------------------------------------------
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
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const ssBgl = device.createBindGroupLayout({
    label: `${ctx.id}/ss-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    ],
  })

  // --- per stand entry ------------------------------------------------------
  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, SCATTER_MAX_DENSITY)
    const caps: [number, number, number] = [
      Math.ceil(Math.PI * LOD0_MAX * LOD0_MAX * density * 1.35) + 256,
      Math.ceil(Math.PI * LOD1_MAX * LOD1_MAX * density * 1.3) + 256,
      Math.ceil(Math.PI * REGION_MAX * REGION_MAX * density * 1.25) + 512,
    ]
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
        { label: `${ctx.id}/instances-${entryIndex}-lod${lod}`, size: cap * 16, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: `culled-lod${lod}` },
      ),
    )
    const argsBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 60,
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
    const drawBindGroups = instBuffers.map((buffer, lod) =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}-lod${lod}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer } },
          { binding: 2, resource: { buffer: gpu.shellBuffer } },
          { binding: 3, resource: gpu.albedoTex.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: gpu.attrTex.createView({ dimension: '2d-array' }) },
          { binding: 5, resource: sampler },
        ],
      }),
    )
    return { gpu, caps, infoBuffer, argsBuffer, cullBindGroup, drawBindGroups, slotsPerFrame: 0 }
  })

  const ssParams = ctx.res.createBuffer(
    { label: `${ctx.id}/ss-params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'ss-params' },
  )

  // --- pipelines ------------------------------------------------------------
  let cullPipeline!: GPUComputePipeline
  /** [0] plain colour target, [1] + the screen-space canopy mask target. */
  const shellPipelines: GPURenderPipeline[] = []
  let compositePipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const shellModule = ctx.shaders.module(shellSrc)
    const shellLayout = device.createPipelineLayout({
      label: `${ctx.id}/shell-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    shellPipelines.length = 0
    for (const withMask of [false, true]) {
      shellPipelines.push(
        device.createRenderPipeline({
          label: `${ctx.id}/shell${withMask ? '-canopy' : ''}`,
          layout: shellLayout,
          vertex: { module: shellModule, entryPoint: 'vs_main' },
          fragment: {
            module: shellModule,
            entryPoint: withMask ? 'fs_canopy' : 'fs_main',
            targets: withMask
              ? [{ format: ctx.colorFormat }, { format: 'r8unorm' as GPUTextureFormat }]
              : [{ format: ctx.colorFormat }],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
        }),
      )
    }
    const ssModule = ctx.shaders.module(thicknessSrc)
    const ssLayout = device.createPipelineLayout({
      label: `${ctx.id}/ss-pl`,
      bindGroupLayouts: [ctx.frame.layout, ssBgl],
    })
    compositePipeline = device.createRenderPipeline({
      label: `${ctx.id}/canopy-composite`,
      layout: ssLayout,
      vertex: { module: ssModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: ssModule,
        entryPoint: 'fs_composite',
        targets: [
          {
            format: ctx.colorFormat,
            // dst *= src — no read of the colour target, no copy.
            blend: {
              color: { srcFactor: 'zero', dstFactor: 'src', operation: 'add' },
              alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- screen-space resources (rebuilt on resize / target change) -----------
  interface Screen {
    width: number
    height: number
    depth: GPUTexture
    canopy: GPUTexture
    canopyView: GPUTextureView
    compositeBind: GPUBindGroup
  }
  let screen: Screen | null = null
  const ensureScreen = (targets: ViewTargets): Screen => {
    if (
      screen &&
      screen.width === targets.width &&
      screen.height === targets.height &&
      screen.depth === targets.depthTexture
    ) {
      return screen
    }
    screen?.canopy.destroy()
    const canopy = ctx.res.createTexture(
      {
        label: `${ctx.id}/canopy-mask`,
        size: [targets.width, targets.height],
        format: 'r8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      },
      { tag: 'canopy-mask' },
    )
    const canopyView = canopy.createView()
    screen = {
      width: targets.width,
      height: targets.height,
      depth: targets.depthTexture,
      canopy,
      canopyView,
      compositeBind: device.createBindGroup({
        label: `${ctx.id}/canopy-composite-bg`,
        layout: ssBgl,
        entries: [
          { binding: 0, resource: { buffer: ssParams } },
          { binding: 1, resource: canopyView },
          { binding: 2, resource: screenSampler },
          { binding: 3, resource: targets.depthTexture.createView() },
        ],
      }),
    }
    return screen
  }

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const argsReset = new Uint32Array(15)
  LOD_RANGES.forEach(([indexCount, firstIndex], lod) => {
    argsReset[lod * 5] = indexCount
    argsReset[lod * 5 + 2] = firstIndex
  })
  const ssValues = new Float32Array(4)

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
      const lodNear = Math.min(ctx.params.lodNear, LOD0_MAX)
      const lodFar = Math.max(lodNear, Math.min(ctx.params.lodFar, LOD1_MAX))

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        info.set(planes, 0)
        info.set(a.viewExt, 24)
        info[76] = x0
        info[77] = z0
        info[78] = sideX
        info[79] = sideZ
        info[80] = ctx.seed
        info[81] = entryIndex
        info[82] = R
        info[83] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info[84] = entry.caps[0]
        info[85] = entry.caps[1]
        info[86] = entry.caps[2]
        info[87] = ctx.params.alphaRef
        info[88] = a.y0
        info[89] = a.y1
        info[90] = a.cx
        info[91] = a.cz
        info[92] = lodNear
        info[93] = lodFar
        info[94] = a.rXZ
        info[95] = ctx.params.domeScale
        info[96] = ctx.params.translucency
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.argsBuffer, 0, argsReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
      })

      ssValues[0] = ctx.params.canopyOcclusion
      ssValues[1] = 0.55
      ssValues[2] = 4.5
      ssValues[3] = 11.0
      device.queue.writeBuffer(ssParams, 0, ssValues)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const wantCanopy = ctx.params.canopyOcclusion > 0
      const sc = wantCanopy ? ensureScreen(targets) : null

      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        if (entry.slotsPerFrame === 0) continue // camera outside the stand
        cull.setBindGroup(1, entry.cullBindGroup)
        cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
      }
      cull.end()

      const colorAttachments: GPURenderPassColorAttachment[] = [
        { view: targets.colorView, loadOp: 'load', storeOp: 'store' },
      ]
      if (sc) {
        colorAttachments.push({
          view: sc.canopyView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        })
      }
      const pass = ctx.timing.renderPass(enc, 'shell', {
        colorAttachments,
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(shellPipelines[sc ? 1 : 0]!)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setIndexBuffer(indexBuffer, 'uint16')
      for (const entry of entries) {
        for (let lod = 0; lod < 3; lod++) {
          pass.setBindGroup(1, entry.drawBindGroups[lod]!)
          pass.drawIndexedIndirect(entry.argsBuffer, lod * 20)
        }
      }
      pass.end()

      if (!sc) return

      const comp = ctx.timing.renderPass(enc, 'canopy-occl', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
      })
      comp.setPipeline(compositePipeline)
      comp.setBindGroup(0, ctx.frame.bindGroup)
      comp.setBindGroup(1, sc.compositeBind)
      comp.draw(3)
      comp.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** 9x9 lattice: LOD0 8x8 cells, LOD1 4x4 cells (every 2nd corner), LOD2 quad. */
function buildLatticeIndices(): Uint16Array<ArrayBuffer> {
  const out: number[] = []
  const quad = (a: number, b: number, c: number, d: number): void => {
    out.push(a, b, c, b, d, c)
  }
  for (let cy = 0; cy < GRID_V - 1; cy++) {
    for (let cx = 0; cx < GRID_V - 1; cx++) {
      const v = cy * GRID_V + cx
      quad(v, v + 1, v + GRID_V, v + GRID_V + 1)
    }
  }
  for (let cy = 0; cy < (GRID_V - 1) / 2; cy++) {
    for (let cx = 0; cx < (GRID_V - 1) / 2; cx++) {
      const v = 2 * cy * GRID_V + 2 * cx
      quad(v, v + 2, v + 2 * GRID_V, v + 2 * GRID_V + 2)
    }
  }
  const flat = GRID_V * GRID_V
  quad(flat, flat + 1, flat + 2, flat + 3)
  if (out.length % 2 !== 0) out.push(0) // keep the buffer size a multiple of 4
  return new Uint16Array(out)
}

/** Upload both view arrays + the shell, then generate mips on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: DomeAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [TILE_A, TILE_A, N_VIEWS],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_A,
      usage,
    },
    { species: speciesId, tag: 'view-albedo' },
  )
  const attrTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/attr`,
      size: [TILE_B, TILE_B, N_VIEWS],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_B,
      usage,
    },
    { species: speciesId, tag: 'view-attr' },
  )
  device.queue.writeTexture(
    { texture: albedoTex },
    atlas.texA,
    { bytesPerRow: TILE_A * 4, rowsPerImage: TILE_A },
    [TILE_A, TILE_A, N_VIEWS],
  )
  device.queue.writeTexture(
    { texture: attrTex },
    atlas.texB,
    { bytesPerRow: TILE_B * 4, rowsPerImage: TILE_B },
    [TILE_B, TILE_B, N_VIEWS],
  )
  const shellBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/shell`,
      size: atlas.shell.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'depth-shell' },
  )
  device.queue.writeBuffer(shellBuffer, 0, atlas.shell)

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo')
  const attrPipe = mkPipeline('fs_attr')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, levels: number, pipeline: GPURenderPipeline): void => {
    for (let layer = 0; layer < N_VIEWS; layer++) {
      for (let level = 1; level < levels; level++) {
        const view = (base: number): GPUTextureView =>
          tex.createView({
            dimension: '2d',
            baseMipLevel: base,
            mipLevelCount: 1,
            baseArrayLayer: layer,
            arrayLayerCount: 1,
          })
        const bg = device.createBindGroup({
          label: `${ctx.id}/mipgen-bg-${layer}-${level}`,
          layout: bgl,
          entries: [{ binding: 0, resource: view(level - 1) }],
        })
        const pass = enc.beginRenderPass({
          label: `${ctx.id}/mipgen-${layer}-${level}`,
          colorAttachments: [{ view: view(level), loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bg)
        pass.draw(3)
        pass.end()
      }
    }
  }
  genMips(albedoTex, MIPS_A, albedoPipe)
  genMips(attrTex, MIPS_B, attrPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, attrTex, shellBuffer }
}

/** Gribb-Hartmann frustum planes from a column-major view-proj matrix. */
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
