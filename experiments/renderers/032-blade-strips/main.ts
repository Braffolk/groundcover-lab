import cullSrc from './shaders/cull.wgsl'
import stripsSrc from './shaders/strips.wgsl'
import {
  ATLAS_H,
  ATLAS_W,
  buildMips,
  LEVEL_BASE,
  LEVELS,
  loadSpeciesStrips,
  MIP_LEVELS,
  STRIP_COUNT,
  STRIP_FLOATS,
  type StripSet,
} from './bake.ts'
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
 * Blade strips. Startup: per species a strip set (31 curved ribbons over 5 LOD
 * levels + one top view, with a 1024^2 unrolled-imagery atlas) is loaded from
 * mesh/baked / OPFS or baked in-browser from the raw GCMESH1 mesh.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, classifies each plant by
 * projected diameter into one of five strip-count buckets (16/8/4/2/1 ribbons)
 * and compacts it into that bucket's instance range. Then five
 * drawIndexedIndirect per stand entry, NEAR BUCKET FIRST so early-z has the
 * closest depth already written when the far buckets rasterize. Per-frame cost
 * is O(visible region) and collapses with distance, independent of the stand's
 * plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const N_BUCKETS = LEVELS.length
const INFO_FLOATS = 68
const INFO_STRIDE = 512 // bytes (dynamic uniform offsets must be 256-aligned)
const IND_STRIDE = 32 // bytes per indirect draw slot (5 u32 used)
const TOP_FRAC = 0.55
const TINT_JITTER = 0.09
/**
 * Projected-diameter (px) thresholds between buckets, scaled by `detail`.
 * The last one is deliberately high: beyond it a plant is a single flat ribbon
 * (2 triangles, one texture tap), and that bucket holds most of the meadow.
 */
const PX_THRESHOLDS = [190, 96, 46, 34]
/** Headroom on each bucket's share of the instance pool. */
const BUCKET_SLACK = 1.35
/** How much the coarse buckets force their ribbons to face the camera. */
const FACE_CAM_FLOOR = [0, 0, 0.35, 0.75, 1] as const
const NODES = 5
/** Bucket 3 draws 2-quad ribbons; the last bucket draws one flat quad. */
const COARSE_FROM = 3
const FLAT_FROM = N_BUCKETS - 1
const COARSE_RIBBONS = LEVELS[COARSE_FROM]!
const FINE_INDICES = 6 + LEVELS[0]! * (NODES - 1) * 6
const COARSE_INDICES = 6 + COARSE_RIBBONS * ((NODES - 1) / 2) * 6
const FLAT_INDICES = 6 + 6
/** Below this ratio of camera height to ring distance no canopy card can pass
 *  the fragment shader's elevation gate, so its 6 indices are skipped. */
const TOP_CARD_ELEV_GATE = 0.45

interface SpeciesGpu {
  set: StripSet
  albedoTex: GPUTexture
  normalTex: GPUTexture
  stripBuffer: GPUBuffer
}

interface EntryGpu {
  gpu: SpeciesGpu
  density: number
  capacity: number
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
    // The tile aspect (64 x 512) matches the ribbon's, but a ribbon seen at a
    // slant still compresses hard along its arc — anisotropy is what keeps
    // grazing-angle blades from turning to mush.
    maxAnisotropy: 8,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species strip sets (sequential: the poa bake is memory-hungry) -------
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const set = await loadSpeciesStrips(ctx, entry.species)
    speciesGpu.set(entry.species, upload(ctx, entry.species, set))
  }

  // --- bind group layouts / pipelines --------------------------------------
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
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INFO_FLOATS * 4 },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // Static index buffer, two ranges over the SAME vertex ids:
  //   FINE   [0]: 6 canopy-card indices (so a bucket can skip them with
  //               firstIndex = 6), then 4 quads per ribbon (all 5 nodes).
  //   COARSE: the same, but 2 quads per ribbon through nodes 0/2/4.
  //   FLAT:   one quad from node 0 to node 4 — the last bucket holds most of
  //               the meadow, so its per-plant cost is 4 verts / 2 triangles.
  const maxRibbons = LEVELS[0]!
  const idx = new Uint32Array(FINE_INDICES + COARSE_INDICES + FLAT_INDICES)
  const emitRange = (at: number, ribbons: number, nodeStep: number): void => {
    for (let i = 0; i < 6; i++) idx[at + i] = i
    let o = at + 6
    for (let r = 0; r < ribbons; r++) {
      for (let n = 0; n + nodeStep < NODES; n += nodeStep) {
        const a0 = 6 + r * 10 + n * 2
        const b0 = 6 + r * 10 + (n + nodeStep) * 2
        idx.set([a0, a0 + 1, b0, a0 + 1, b0 + 1, b0], o)
        o += 6
      }
    }
  }
  emitRange(0, maxRibbons, 1)
  emitRange(FINE_INDICES, COARSE_RIBBONS, 2)
  emitRange(FINE_INDICES + COARSE_INDICES, 1, NODES - 1)
  const indexBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/indices`, size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'strip-indices' },
  )
  device.queue.writeBuffer(indexBuffer, 0, idx)

  const sideMax = Math.ceil((2 * REGION_MAX) / CELL) + 1
  const slotsMax = sideMax * sideMax * SCATTER_MAX_PER_CELL

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    // One pool for all five buckets; the per-frame split below hands each
    // bucket a base+capacity from the CURRENT thresholds, so `detail` and the
    // viewport can move the LOD rings without reallocating.
    const capacity = Math.ceil(slotsMax * (density / 8) * 1.06) + 2048
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_STRIDE * N_BUCKETS,
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
        size: IND_STRIDE * N_BUCKETS,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    return {
      gpu,
      density,
      capacity,
      infoBuffer,
      indirectBuffer,
      cullBindGroup: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer, offset: 0, size: INFO_FLOATS * 4 } },
          { binding: 1, resource: { buffer: instBuffer } },
          { binding: 2, resource: { buffer: indirectBuffer } },
        ],
      }),
      drawBindGroup: device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer, offset: 0, size: INFO_FLOATS * 4 } },
          { binding: 1, resource: { buffer: instBuffer } },
          { binding: 2, resource: { buffer: gpu.stripBuffer } },
          { binding: 3, resource: gpu.albedoTex.createView() },
          { binding: 4, resource: gpu.normalTex.createView() },
          { binding: 5, resource: sampler },
        ],
      }),
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let stripPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    stripPipeline = device.createRenderPipeline({
      label: `${ctx.id}/strips`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/strips-pl`,
        bindGroupLayouts: [ctx.frame.layout, drawBgl],
      }),
      vertex: { module: ctx.shaders.module(stripsSrc), entryPoint: 'vs_main' },
      fragment: {
        module: ctx.shaders.module(stripsSrc),
        entryPoint: 'fs_main',
        targets: [{ format: ctx.colorFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array((INFO_STRIDE / 4) * N_BUCKETS)
  const planes = new Float32Array(24)
  const indirect = new Uint32Array((IND_STRIDE / 4) * N_BUCKETS)
  const bases = new Array<number>(N_BUCKETS).fill(0)
  const caps = new Array<number>(N_BUCKETS).fill(0)

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

      const { height } = ctx.size()
      const pxScale = height * (frame.camera.proj[5] ?? 1.7)
      const thresholds = PX_THRESHOLDS.map((t) => t * ctx.params.detail)
      const finestOffset = ctx.params.finest === '16' ? 0 : ctx.params.finest === '8' ? 1 : 2
      const camH = Math.max(0.05, cam.y - ctx.scene.terrain.height(cam.x, cam.z))

      entries.forEach((entry, entryIndex) => {
        const set = entry.gpu.set
        const cullRadius = Math.max(set.rXZ, (set.y1 - set.y0) / 2) * 1.05
        const standEntry = ctx.stand.species[entryIndex]!

        // Split the instance pool across buckets from the CURRENT LOD rings:
        // ring b ends where a plant of the entry's largest scale drops below
        // that bucket's pixel threshold.
        let prevArea = 0
        const raw = thresholds.map((t) => {
          const d = Math.min(R, (cullRadius * standEntry.scaleMax * pxScale) / Math.max(t, 1))
          const area = Math.PI * d * d
          const ring = Math.max(0, area - prevArea)
          prevArea = area
          return ring * entry.density * BUCKET_SLACK + 256
        })
        raw.push(Math.max(0, Math.PI * R * R - prevArea) * entry.density * 1.06 + 256)
        const want = raw.reduce((a, b) => a + b, 0)
        const squeeze = want > entry.capacity ? entry.capacity / want : 1
        let base = 0
        for (let b = 0; b < N_BUCKETS; b++) {
          bases[b] = base
          caps[b] = Math.floor(raw[b]! * squeeze)
          base += caps[b]!
        }

        // Distance at which each bucket's ring begins (bucket 0 starts at the
        // camera, so its plants can always be seen steeply).
        const ringInner = (b: number): number =>
          b === 0 ? 1 : (cullRadius * standEntry.scaleMax * pxScale) / Math.max(thresholds[b - 1]!, 1)

        for (let b = 0; b < N_BUCKETS; b++) {
          const o = (b * INFO_STRIDE) / 4
          const level = Math.min(N_BUCKETS - 1, b + finestOffset)
          const flat = b >= FLAT_FROM
          const coarse = b >= COARSE_FROM && !flat
          // A canopy card that cannot pass the shader's elevation gate anywhere
          // in this ring is not drawn at all — at eye level that is every
          // bucket but the nearest, i.e. almost every plant on screen.
          const topCard = ctx.params.topCard && camH / ringInner(b) > TOP_CARD_ELEV_GATE
          info.set(planes, o)
          info[o + 24] = x0
          info[o + 25] = z0
          info[o + 26] = sideX
          info[o + 27] = sideZ
          info[o + 28] = ctx.seed
          info[o + 29] = entryIndex
          info[o + 30] = R
          info[o + 31] = cullRadius
          info[o + 32] = set.y0
          info[o + 33] = set.y1
          info[o + 34] = set.rXZ
          info[o + 35] = pxScale
          info[o + 36] = set.cx
          info[o + 37] = set.cz
          info[o + 38] = ctx.params.alphaRef
          info[o + 39] = Math.max(ctx.params.faceCam, FACE_CAM_FLOOR[b]!)
          info[o + 40] = ctx.params.bottomShade
          info[o + 41] = topCard ? 1 : 0
          info[o + 42] = TOP_FRAC
          info[o + 43] = TINT_JITTER
          info[o + 44] = LEVEL_BASE[level]!
          info[o + 45] = LEVELS[level]!
          info[o + 46] = bases[b]!
          // flags: 1 = skip the normal-map tap (the flat bucket is ~20px tall,
          // where per-blade normals are far below a pixel).
          info[o + 47] = flat ? 1 : 0
          for (let t = 0; t < 4; t++) info[o + 48 + t] = thresholds[t]!
          for (let k = 0; k < N_BUCKETS; k++) {
            info[o + 52 + k] = bases[k]!
            info[o + 60 + k] = caps[k]!
          }

          const io = (b * IND_STRIDE) / 4
          const quads = flat ? 1 : coarse ? (NODES - 1) / 2 : NODES - 1
          const rangeBase = flat ? FINE_INDICES + COARSE_INDICES : coarse ? FINE_INDICES : 0
          indirect[io] = (topCard ? 6 : 0) + LEVELS[level]! * quads * 6
          indirect[io + 1] = 0 // instanceCount — filled by the cull pass
          indirect[io + 2] = rangeBase + (topCard ? 0 : 6) // firstIndex
          indirect[io + 3] = 0 // baseVertex
          indirect[io + 4] = 0 // firstInstance
        }
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirect)
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

      const pass = ctx.timing.renderPass(enc, 'strips', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(stripPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setIndexBuffer(indexBuffer, 'uint32')
      // Near buckets first: their depth blocks the far buckets' fragments.
      for (let b = 0; b < N_BUCKETS; b++) {
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroup, [b * INFO_STRIDE])
          pass.drawIndexedIndirect(entry.indirectBuffer, b * IND_STRIDE)
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

/** Upload the atlas (CPU mip chain) and the ribbon table. */
function upload(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, set: StripSet): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [ATLAS_W, ATLAS_H],
      format: 'rgba8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'strip-albedo' },
  )
  const normalTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal`,
      size: [ATLAS_W, ATLAS_H],
      format: 'rg8unorm',
      mipLevelCount: MIP_LEVELS,
      usage,
    },
    { species: speciesId, tag: 'strip-normal' },
  )
  const mips = buildMips(set.albedo, set.normal)
  let w = ATLAS_W
  let h = ATLAS_H
  for (let level = 0; level < MIP_LEVELS; level++) {
    device.queue.writeTexture(
      { texture: albedoTex, mipLevel: level },
      mips.albedo[level]!,
      { bytesPerRow: w * 4, rowsPerImage: h },
      [w, h],
    )
    device.queue.writeTexture(
      { texture: normalTex, mipLevel: level },
      mips.normal[level]!,
      { bytesPerRow: w * 2, rowsPerImage: h },
      [w, h],
    )
    w = Math.max(1, w >> 1)
    h = Math.max(1, h >> 1)
  }

  const stripBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/strip-table`,
      size: STRIP_COUNT * STRIP_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'strip-table' },
  )
  device.queue.writeBuffer(stripBuffer, 0, set.strips)

  return { set, albedoTex, normalTex, stripBuffer }
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
