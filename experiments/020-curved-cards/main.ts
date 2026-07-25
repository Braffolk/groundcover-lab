import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/curved.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS, DISP, SURF, loadSpeciesRelief, type ReliefAtlas } from './bake.ts'
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
 * Curved cards. Startup: each species' relief atlas (8 side views + 1 top view
 * — albedo/coverage, oct normals + front-surface distance + baked sky
 * occlusion, and a hole-free smoothed displacement field) is loaded from
 * mesh/baked / OPFS or baked in-browser from the raw mesh, uploaded and
 * mipped.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into
 * three distance RINGS of one instance buffer, each with its own indirect
 * draw. A ring is a LOD: ring 0 draws a 5x7 curved patch + a 4x4 canopy patch
 * (66 triangles), ring 1 a 3x5 + 3x3 (24 triangles), ring 2 a single flat quad
 * + flat top card (4 triangles — the billboard baseline's exact geometry).
 * Rings are drawn near-to-far so early-z rejects the deep overdraw.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const LOD0_MAX = 24 // keep equal to the manifest's lod0Dist max
const LOD1_MAX = 70 // keep equal to the manifest's lod1Dist max
const MIP_A = Math.floor(Math.log2(ATLAS)) + 1
const MIP_S = Math.floor(Math.log2(SURF)) + 1
const TOP_FRAC = 0.52
const INFO_FLOATS = 64 // 256 B — one dynamic-offset block per LOD
const INFO_STRIDE = INFO_FLOATS * 4
const DRAW_ARG_U32 = 5 // drawIndexedIndirect: indexCount, instanceCount, firstIndex, baseVertex, firstInstance
/**
 * Coverage-weighted means of the baked occlusion / front-distance channels
 * (measured over the committed artifacts, side tiles only). They turn the two
 * occlusion terms into mean-preserving contrast: see the shade_gain write.
 */
const AO_FIELD_MEAN = 0.45
const DIST_FIELD_MEAN = 0.51

/** side lattice (columns x rows), canopy lattice (columns x rows) per ring. */
const LODS: { side: [number, number]; top: [number, number]; curve: number }[] = [
  { side: [5, 7], top: [4, 4], curve: 1 },
  { side: [3, 5], top: [3, 3], curve: 1 },
  { side: [2, 2], top: [2, 2], curve: 0 }, // a 2x2 lattice can only bend at its rim — keep it flat
]

interface SpeciesGpu {
  atlas: ReliefAtlas
  albedoTex: GPUTexture
  surfTex: GPUTexture
  dispTex: GPUTexture
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  bases: [number, number, number]
  caps: [number, number, number]
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
    const atlas = await loadSpeciesRelief(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- static lattice topology ---------------------------------------------
  const { indices, firstIndex, indexCount } = buildLattices()
  const indexBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/lattice-indices`,
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'lattice-indices' },
  )
  device.queue.writeBuffer(indexBuffer, 0, indices)

  // --- bind group layouts / pipelines ---------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: INFO_STRIDE } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INFO_STRIDE },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const scaleMax = Math.max(standEntry.scaleMax, 0.1)
    // Rings are binned by distance/scale, so a ring's world radius is its
    // metric radius times the stand entry's largest scale. Sized for the
    // PARAM MAXIMA, never for the current value: params move at runtime.
    const regionCap = ringCapacity(REGION_MAX, density)
    const caps: [number, number, number] = [
      Math.min(ringCapacity(LOD0_MAX * scaleMax, density), regionCap),
      Math.min(ringCapacity(LOD1_MAX * scaleMax, density), regionCap),
      regionCap,
    ]
    const bases: [number, number, number] = [0, caps[0], caps[0] + caps[1]]
    const total = bases[2] + caps[2]

    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_STRIDE * LODS.length,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: total * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 64,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer, size: INFO_STRIDE } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = device.createBindGroup({
      label: `${ctx.id}/draw-bg-${entryIndex}`,
      layout: drawBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer, size: INFO_STRIDE } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: gpu.albedoTex.createView() },
        { binding: 3, resource: gpu.surfTex.createView() },
        { binding: 4, resource: gpu.dispTex.createView() },
        { binding: 5, resource: sampler },
      ],
    })
    return {
      speciesId: standEntry.species,
      gpu,
      bases,
      caps,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroup,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let cardsPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    cardsPipeline = device.createRenderPipeline({
      label: `${ctx.id}/cards`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cards-pl`, bindGroupLayouts: [ctx.frame.layout, drawBgl] }),
      vertex: { module: ctx.shaders.module(cardsSrc), entryPoint: 'vs_main' },
      fragment: { module: ctx.shaders.module(cardsSrc), entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS * LODS.length)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(16)
  LODS.forEach((_, lod) => {
    indirectReset[lod * DRAW_ARG_U32] = indexCount[lod]!
    indirectReset[lod * DRAW_ARG_U32 + 2] = firstIndex[lod]!
  })

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

      const lod0 = Math.min(ctx.params.lod0Dist, LOD0_MAX)
      const lod1 = Math.max(lod0, Math.min(ctx.params.lod1Dist, LOD1_MAX))

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        LODS.forEach((lodDesc, lod) => {
          const o = lod * INFO_FLOATS
          info.set(planes, o)
          info[o + 24] = x0
          info[o + 25] = z0
          info[o + 26] = sideX
          info[o + 27] = sideZ
          info[o + 28] = ctx.seed
          info[o + 29] = entryIndex
          info[o + 30] = R
          info[o + 31] = ctx.params.alphaRef
          info[o + 32] = a.y0
          info[o + 33] = a.y1
          info[o + 34] = a.rXZ
          info[o + 35] = a.cx
          info[o + 36] = a.cz
          info[o + 37] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
          info[o + 38] = TOP_FRAC
          info[o + 39] = ctx.params.aoStrength
          info[o + 40] = entry.bases[0]
          info[o + 41] = entry.bases[1]
          info[o + 42] = entry.bases[2]
          info[o + 44] = entry.caps[0]
          info[o + 45] = entry.caps[1]
          info[o + 46] = entry.caps[2]
          info[o + 48] = lod0
          info[o + 49] = lod1
          info[o + 52] = lodDesc.side[0]
          info[o + 53] = lodDesc.side[1]
          info[o + 54] = lodDesc.top[0]
          info[o + 55] = lodDesc.top[1]
          info[o + 56] = lod
          info[o + 57] =
            1 /
            (lerp(1, AO_FIELD_MEAN, ctx.params.aoStrength) *
              lerp(1 - ctx.params.depthShade, 1, DIST_FIELD_MEAN))
          info[o + 58] = ctx.params.depthShade
          info[o + 59] = lodDesc.curve * ctx.params.curvature
          info[o + 60] = entry.bases[lod]!
          info[o + 61] = entry.caps[lod]!
        })
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

      const pass = ctx.timing.renderPass(enc, 'cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(cardsPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setIndexBuffer(indexBuffer, 'uint16')
      // Near ring first: the curved cards write solid depth, so the flat far
      // ring behind them is rejected by early-z instead of shaded.
      for (let lod = 0; lod < LODS.length; lod++) {
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroup, [lod * INFO_STRIDE])
          pass.drawIndexedIndirect(entry.indirectBuffer, lod * DRAW_ARG_U32 * 4)
        }
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Worst-case survivors of one ring: every scatter slot inside its disc. */
function ringCapacity(radius: number, density: number): number {
  const cells = (Math.PI * radius * radius) / (CELL * CELL)
  return Math.ceil(cells * SCATTER_MAX_PER_CELL * (density / 8) * 1.15) + 512
}

/**
 * One index buffer for all three rings: a side lattice then a canopy lattice,
 * vertex ids local to the ring (each ring draws with its own firstIndex and
 * its own grid dims in the uniform, so ids never have to be globally unique).
 */
function buildLattices(): { indices: Uint16Array<ArrayBuffer>; firstIndex: number[]; indexCount: number[] } {
  const out: number[] = []
  const firstIndex: number[] = []
  const indexCount: number[] = []
  for (const lod of LODS) {
    const start = out.length
    firstIndex.push(start)
    const quads = (cols: number, rows: number, base: number): void => {
      for (let j = 0; j < rows - 1; j++) {
        for (let i = 0; i < cols - 1; i++) {
          const a = base + j * cols + i
          out.push(a, a + 1, a + cols, a + 1, a + cols + 1, a + cols)
        }
      }
    }
    quads(lod.side[0], lod.side[1], 0)
    quads(lod.top[0], lod.top[1], lod.side[0] * lod.side[1])
    indexCount.push(out.length - start)
  }
  // Buffer size must be a multiple of 4; u16 indices means an even count.
  if (out.length % 2 !== 0) out.push(0)
  return { indices: new Uint16Array(out), firstIndex, indexCount }
}

/** Upload the three atlases and generate the mip chains on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: ReliefAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    { label: `${ctx.id}/${speciesId}/albedo`, size: [ATLAS, ATLAS], format: 'rgba8unorm', mipLevelCount: MIP_A, usage },
    { species: speciesId, tag: 'relief-albedo' },
  )
  const surfTex = ctx.res.createTexture(
    { label: `${ctx.id}/${speciesId}/surf`, size: [SURF, SURF], format: 'rgba8unorm', mipLevelCount: MIP_S, usage },
    { species: speciesId, tag: 'relief-surface' },
  )
  const dispTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/disp`,
      size: [DISP, DISP],
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { species: speciesId, tag: 'relief-displacement' },
  )
  device.queue.writeTexture({ texture: albedoTex }, atlas.albedo, { bytesPerRow: ATLAS * 4, rowsPerImage: ATLAS }, [
    ATLAS,
    ATLAS,
  ])
  device.queue.writeTexture({ texture: surfTex }, atlas.surf, { bytesPerRow: SURF * 4, rowsPerImage: SURF }, [SURF, SURF])
  device.queue.writeTexture({ texture: dispTex }, atlas.disp, { bytesPerRow: DISP, rowsPerImage: DISP }, [DISP, DISP])

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
  const boxPipe = mkPipeline('fs_box')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, levels: number, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < levels; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${level}`,
        colorAttachments: [
          { view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
  genMips(albedoTex, MIP_A, albedoPipe)
  genMips(surfTex, MIP_S, boxPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, surfTex, dispTex }
}

/** Gribb–Hartmann frustum planes from a column-major view-proj matrix. */
function frustumPlanes(m: Float32Array | number[], out: Float32Array): void {
  const row = (r: number): [number, number, number, number] => [m[r]!, m[4 + r]!, m[8 + r]!, m[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const planes: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r2[0], r2[1], r2[2], r2[3]], // near (WebGPU z >= 0)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // far
  ]
  planes.forEach((p, i) => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1
    out[i * 4] = p[0] / len
    out[i * 4 + 1] = p[1] / len
    out[i * 4 + 2] = p[2] / len
    out[i * 4 + 3] = p[3] / len
  })
}
