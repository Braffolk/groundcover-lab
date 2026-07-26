import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS_ALB, ATLAS_GEO, loadSpeciesCards, type CardAtlas } from './bake.ts'
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
 * Near detail injection — depth-reprojected cards.
 *
 * Startup: per species two baked atlases (albedo+coverage; oct normal + signed
 * per-view depth + volumetric sky visibility) are loaded from mesh/baked / OPFS
 * or baked in-browser from the raw mesh, uploaded and mipped.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into SIX
 * disjoint distance buckets of one instance buffer + six indirect draws. The
 * buckets are drawn near-to-far (early-z then rejects most of the far field),
 * and the bucket also selects the fragment shader: 3 taps with two relief steps
 * and transmission up close, 2 taps with one relief step in the middle band, the
 * plain 2-tap billboard shader in the far band. Per-frame cost is O(visible
 * region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const MIPS_ALB = Math.floor(Math.log2(ATLAS_ALB)) + 1
const MIPS_GEO = Math.floor(Math.log2(ATLAS_GEO)) + 1
const TOP_FRAC = 0.52
const INFO_FLOATS = 72
const NB = 6 // distance buckets — must match cull.wgsl / entry_info.wgsl
/** Fragment pipeline per bucket: 0-1 near (3 taps), 2-3 mid, 4-5 far. */
const BUCKET_TIER = [0, 0, 1, 1, 2, 2] as const
/**
 * Frustum share used to size each bucket's slice of the instance buffer. The two
 * nearest buckets get their whole annulus (the camera sits inside them); farther
 * ones are only ever seen through a ~90deg wedge, so 0.45 is already ~2x margin.
 */
const BUCKET_VIEW_SHARE = [1, 1, 0.6, 0.5, 0.45, 0.45] as const

interface SpeciesGpu {
  atlas: CardAtlas
  albedoTex: GPUTexture
  geoTex: GPUTexture
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  density: number
  plan: BucketPlan
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  /** One per bucket — a storage-binding window onto that bucket's slice. */
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
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) --
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesCards(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  // --- bind group layouts / pipelines ---------------------------------------
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
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const plan = bucketPlan(density)
    const capacity = plan.total
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: capacity * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: NB * 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ],
    })
    // drawIndirect cannot set firstInstance without the (optional)
    // indirect-first-instance feature, so each bucket gets its own storage
    // binding window instead and its vertex shader indexes from 0.
    const albedoView = gpu.albedoTex.createView()
    const geoView = gpu.geoTex.createView()
    const drawBindGroups = plan.caps.map((cap, b) =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}-b${b}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer, offset: plan.bases[b]! * 16, size: cap * 16 } },
          { binding: 2, resource: albedoView },
          { binding: 3, resource: geoView },
          { binding: 4, resource: sampler },
        ],
      }),
    )
    return {
      speciesId: standEntry.species,
      gpu,
      density,
      plan,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroups,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  const cardPipelines: GPURenderPipeline[] = []
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(cardsSrc)
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    cardPipelines.length = 0
    const tierEntryPoints = ['fs_near', 'fs_mid', 'fs_far']
    tierEntryPoints.forEach((entryPoint, tier) => {
      cardPipelines.push(
        device.createRenderPipeline({
          label: `${ctx.id}/cards-tier${tier}`,
          layout,
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint, targets: [{ format: ctx.colorFormat }] },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
        }),
      )
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(NB * 4)

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      // Region cell rect, clamped to the stand's cell range on the CPU: cells
      // outside the stand hold nothing, so they must not be dispatched (on a
      // small stand — close-quality is ±24m — that is most of the region).
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      const nearDist = Math.min(ctx.params.nearDist, ctx.params.midDist * 0.5)
      const edges = bucketEdges(nearDist, ctx.params.midDist, R)
      const vertexCount = ctx.params.topCard ? 12 : 6

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        const plan = entry.plan
        info.set(planes, 0)
        for (let b = 0; b < NB; b++) {
          info[24 + b] = edges[b]!
          info[32 + b] = plan.bases[b]!
          info[40 + b] = plan.caps[b]!
          indirectReset[b * 4] = vertexCount
          indirectReset[b * 4 + 1] = 0
          indirectReset[b * 4 + 2] = 0
          indirectReset[b * 4 + 3] = 0
        }
        info[48] = x0
        info[49] = z0
        info[50] = sideX
        info[51] = sideZ
        info[52] = ctx.seed
        info[53] = entryIndex
        info[54] = R
        info[55] = a.y0
        info[56] = a.y1
        info[57] = a.rXZ
        info[58] = a.cx
        info[59] = a.cz
        info[60] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info[61] = ctx.params.alphaRef
        info[62] = TOP_FRAC
        info[63] = ctx.params.aoStrength
        info[64] = ctx.params.translucency
        info[65] = ctx.params.reliefScale
        info[66] = Math.max(0, ctx.params.midDist - 12)
        info[67] = ctx.params.midDist
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
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Buckets in order = near-to-far: the near occluders write depth first and
      // early-z rejects the bands behind them. The bucket also picks the tier.
      let tier = -1
      for (let b = 0; b < NB; b++) {
        if (BUCKET_TIER[b]! !== tier) {
          tier = BUCKET_TIER[b]!
          pass.setPipeline(cardPipelines[tier]!)
        }
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroups[b]!)
          pass.drawIndirect(entry.indirectBuffer, b * 16)
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

/** Outer radius (m) of each distance bucket. */
function bucketEdges(nearDist: number, midDist: number, region: number): number[] {
  return [
    nearDist * 0.5,
    nearDist,
    (nearDist + midDist) * 0.5,
    midDist,
    midDist + Math.max(0, region - midDist) * 0.45,
    1e9,
  ]
}

interface BucketPlan {
  caps: number[]
  bases: number[]
  total: number
}

/** Expected plants per bucket for one edge configuration (annulus x density). */
function bucketDemand(edges: number[], density: number): number[] {
  const out: number[] = []
  let prev = 0
  for (let b = 0; b < NB; b++) {
    const outer = Math.min(edges[b]!, REGION_MAX)
    const area = Math.PI * Math.max(0, outer * outer - prev * prev)
    prev = outer
    out.push(area * density * BUCKET_VIEW_SHARE[b]!)
  }
  return out
}

/**
 * Fixed per-bucket slices of the instance buffer. They must be fixed because
 * each bucket is bound as its own storage window (see the firstInstance note),
 * so the capacity is the worst case over the whole nearDist/midDist param range
 * — the default split and the widest-near-band split. A bucket that still
 * overflows drops its last plants (never scribbles into its neighbour).
 */
function bucketPlan(density: number): BucketPlan {
  const a = bucketDemand(bucketEdges(12, 48, REGION_MAX), density)
  const b2 = bucketDemand(bucketEdges(32, 110, REGION_MAX), density)
  const caps = a.map((v, b) => {
    const n = Math.ceil(Math.max(v, b2[b]!) * 1.3) + 256
    return Math.ceil(n / 16) * 16 // 16 instances = 256B storage offset alignment
  })
  const bases: number[] = []
  let base = 0
  for (let b = 0; b < NB; b++) {
    bases.push(base)
    base += caps[b]!
  }
  return { caps, bases, total: base }
}

/** Upload atlas mip 0 for both atlases and generate the mip chains on the GPU. */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: CardAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS_ALB, ATLAS_ALB],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_ALB,
      usage,
    },
    { species: speciesId, tag: 'card-albedo' },
  )
  const geoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/geo`,
      size: [ATLAS_GEO, ATLAS_GEO],
      format: 'rgba8unorm',
      mipLevelCount: MIPS_GEO,
      usage,
    },
    { species: speciesId, tag: 'card-geo' },
  )
  device.queue.writeTexture(
    { texture: albedoTex },
    atlas.albedo,
    { bytesPerRow: ATLAS_ALB * 4, rowsPerImage: ATLAS_ALB },
    [ATLAS_ALB, ATLAS_ALB],
  )
  device.queue.writeTexture(
    { texture: geoTex },
    atlas.geo,
    { bytesPerRow: ATLAS_GEO * 4, rowsPerImage: ATLAS_GEO },
    [ATLAS_GEO, ATLAS_GEO],
  )

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
  const geoPipe = mkPipeline('fs_geo')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, levels: number, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < levels; level++) {
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
  genMips(albedoTex, MIPS_ALB, albedoPipe)
  genMips(geoTex, MIPS_GEO, geoPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, geoTex }
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
