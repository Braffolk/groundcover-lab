import cullSrc from './shaders/cull.wgsl'
import splatsSrc from './shaders/splats.wgsl'
import {
  loadSpeciesSplats,
  LOD_SPLATS,
  MAX_SPLATS,
  STAMP_ATLAS,
  STAMP_MIPS,
  TOTAL_SPLATS,
  type SplatSet,
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
 * Micro-splat renderer.
 *
 * Startup: each species' splat hierarchy (14 kd-tree levels, 8192 -> 1 hard
 * elliptical micro-cards welded to the source surface, plus a 16-tile stamp
 * atlas cut from a real orthographic render) is loaded from mesh/baked / OPFS
 * or baked in-browser from the raw GCMESH1.
 *
 * Per frame:
 *  - one compute pass evaluates the shared scatter over a camera-centered cell
 *    region, frustum-culls, picks each plant's LOD from ONE log2 of its
 *    projected pixel height, resolves the per-plant yaw/wind/fade, and appends
 *    the plant into that LOD's bucket + indirect draw args;
 *  - one render pass issues 12 x entries indexed indirect draws, near LOD
 *    first, so early-z has the nearest opaque geometry already down before the
 *    deep field behind it is rasterized.
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count,
 * and collapses by 2x per sqrt(2) of distance. No source geometry is touched
 * after the bake; there is no per-fragment marching and exactly one texture tap.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const INFO_FLOATS = 84
const LOD_STRIDE = 256 // dynamic-offset stride for the per-LOD uniform
const ARG_STRIDE = 32 // drawIndexedIndirect args (5 u32) padded for alignment
const INST_BYTES = 32
/**
 * Plant pixel height that still earns the finest (8192-splat) level; each
 * further level halves the splat count and moves the switch out by sqrt(2).
 */
const LOD_T0 = 1050
const LOD_RATIO = Math.SQRT2
/**
 * Levels actually used. The bake ships all 14 tree levels, but below 4 splats a
 * plant stops having a silhouette worth looking at (1-2 ellipses read as blobs
 * on the horizon) while saving almost nothing: those splats are ~1px, so they
 * are primitives, not fill. 4 is the floor.
 */
const LOD_USED = 12
if (LOD_USED > LOD_SPLATS.length) throw new Error('LOD_USED exceeds the baked level count')
/** Nominal pixels-per-metre-at-1m used ONCE, to size the LOD buckets. */
const PX_NOMINAL = 0.5 * 900 / Math.tan(Math.PI / 6)

interface SpeciesGpu {
  set: SplatSet
  splatBuffer: GPUBuffer
  stampTex: GPUTexture
}

interface EntryGpu {
  entryIndex: number
  gpu: SpeciesGpu
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroup: GPUBindGroup
  caps: number[]
  bases: number[]
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/stamp-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species splat sets (sequential: the poa bake is memory-hungry) --------
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const set = await loadSpeciesSplats(ctx, entry.species)
    speciesGpu.set(entry.species, uploadSpecies(ctx, entry.species, set))
  }

  // --- shared quad index buffer (4 verts per splat, 2 triangles) ------------
  const indexData = new Uint16Array(MAX_SPLATS * 6)
  for (let i = 0; i < MAX_SPLATS; i++) {
    const v = i * 4
    indexData.set([v, v + 1, v + 2, v + 2, v + 1, v + 3], i * 6)
  }
  const indexBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/quad-indices`,
      size: indexData.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'quad-indices' },
  )
  device.queue.writeBuffer(indexBuffer, 0, indexData)

  // --- bind group layouts ---------------------------------------------------
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
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 },
      },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // --- per-entry resources --------------------------------------------------
  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const plantH = (gpu.set.y1 - gpu.set.y0) * standEntry.scaleMax
    const R = REGION_MAX
    const discCount = Math.ceil(Math.PI * R * R * density) + 1024
    const d0 = (plantH * PX_NOMINAL) / LOD_T0
    const caps: number[] = []
    for (let k = 0; k < LOD_USED; k++) {
      // Ring the level owns, widened both ways so the lodBias slider can move
      // the ladder without starving a bucket.
      const lo = Math.min(R, (d0 * LOD_RATIO ** k) / 2.6)
      const hi = k === LOD_USED - 1 ? R : Math.min(R, d0 * LOD_RATIO ** (k + 1) * 1.6)
      const ring = Math.PI * Math.max(hi * hi - lo * lo, 0)
      caps.push(Math.min(discCount, Math.ceil(ring * density * 0.6) + 1024))
    }
    const bases: number[] = []
    let total = 0
    for (const c of caps) {
      bases.push(total)
      total += c
    }

    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const lodBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/lodinfo-${entryIndex}`,
        size: LOD_STRIDE * LOD_USED,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'lod-info' },
    )
    const lodData = new Float32Array((LOD_STRIDE / 4) * LOD_USED)
    for (let k = 0; k < LOD_USED; k++) {
      const lod = gpu.set.lods[k]!
      const o = (k * LOD_STRIDE) / 4
      lodData[o] = lod.offset
      lodData[o + 1] = bases[k]!
      lodData[o + 2] = lod.maxA
      lodData[o + 3] = lod.maxB
      // solid: only the last two levels give up the stamp cut-out and become
      // plain ellipses. The stamp's mips are coverage-corrected, so the hard
      // alpha test survives minification and keeps far silhouettes feathery
      // instead of blobby — that is what the far field lives or dies on.
      lodData[o + 4] = Math.min(1, Math.max(0, (k - 10) / 2))
      // face_cam: roll the minor axis toward the eye around the (baked) major
      // axis, so a coarse patch can never turn edge-on and vanish.
      lodData[o + 5] = Math.min(1, Math.max(0, (k - 7) / 3))
      // ao_lift: at range you only ever see the canopy's lit outer shell, never
      // its shaded interior, so the baked occlusion is lifted back toward 1.
      lodData[o + 6] = Math.min(1, Math.max(0, (k - 8) / 4))
    }
    device.queue.writeBuffer(lodBuffer, 0, lodData)

    const instBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/instances-${entryIndex}`,
        size: total * INST_BYTES,
        usage: GPUBufferUsage.STORAGE,
      },
      { species: standEntry.species, tag: 'lod-buckets' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: ARG_STRIDE * LOD_USED,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )

    return {
      entryIndex,
      gpu,
      infoBuffer,
      indirectBuffer,
      caps,
      bases,
      slotsPerFrame: 0,
      cullBindGroup: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer } },
          { binding: 2, resource: { buffer: indirectBuffer } },
        ],
      }),
      drawBindGroup: device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: lodBuffer, size: 32 } },
          { binding: 2, resource: { buffer: instBuffer } },
          { binding: 3, resource: { buffer: gpu.splatBuffer } },
          { binding: 4, resource: gpu.stampTex.createView() },
          { binding: 5, resource: sampler },
        ],
      }),
    }
  })

  // --- pipelines ------------------------------------------------------------
  let cullPipeline!: GPUComputePipeline
  let drawPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    drawPipeline = device.createRenderPipeline({
      label: `${ctx.id}/splats`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/splats-pl`,
        bindGroupLayouts: [ctx.frame.layout, drawBgl],
      }),
      vertex: { module: ctx.shaders.module(splatsSrc), entryPoint: 'vs_main' },
      fragment: {
        module: ctx.shaders.module(splatsSrc),
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
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array((ARG_STRIDE / 4) * LOD_USED)
  for (let k = 0; k < LOD_USED; k++) indirectReset[k * 8] = LOD_SPLATS[k]! * 6

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
      const pxScale = 0.5 * ctx.size().height * (frame.camera.proj[5] ?? 1)

      for (const entry of entries) {
        const set = entry.gpu.set
        info.set(planes, 0)
        for (let k = 0; k < LOD_USED; k++) {
          info[24 + k] = entry.bases[k]!
          info[40 + k] = entry.caps[k]!
        }
        info[56] = x0
        info[57] = z0
        info[58] = sideX
        info[59] = sideZ
        info[60] = ctx.seed
        info[61] = entry.entryIndex
        info[62] = R
        info[63] = pxScale
        info[64] = set.y0
        info[65] = set.y1
        info[66] = set.rXZ
        info[67] = Math.max(set.rXZ, (set.y1 - set.y0) / 2) * 1.06
        info[68] = LOD_T0 / ctx.params.lodBias
        info[69] = 1 / Math.log(LOD_RATIO)
        info[70] = LOD_USED
        info[71] = ctx.params.splatScale
        info[72] = ctx.params.minPx
        info[73] = ctx.params.aoStrength
        info[74] = ctx.params.bulge
        info[75] = ctx.params.alphaRef
        info[76] = set.bmin[0]
        info[77] = set.bmin[1]
        info[78] = set.bmin[2]
        info[80] = set.bmax[0] - set.bmin[0]
        info[81] = set.bmax[1] - set.bmin[1]
        info[82] = set.bmax[2] - set.bmin[2]
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
      }
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

      const pass = ctx.timing.renderPass(enc, 'splats', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(drawPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setIndexBuffer(indexBuffer, 'uint16')
      // LOD-major, near ring first across ALL species: the nearest opaque
      // splats lay down depth before the deep field behind them is rasterized.
      for (let k = 0; k < LOD_USED; k++) {
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroup, [k * LOD_STRIDE])
          pass.drawIndexedIndirect(entry.indirectBuffer, k * ARG_STRIDE)
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

/** Upload one species' splat records and its stamp atlas (mips come baked). */
function uploadSpecies(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, set: SplatSet): SpeciesGpu {
  const { device } = ctx
  const splatBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/splats`,
      size: TOTAL_SPLATS * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'splat-hierarchy' },
  )
  device.queue.writeBuffer(splatBuffer, 0, set.splats)

  const stampTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/stamps`,
      size: [STAMP_ATLAS, STAMP_ATLAS],
      format: 'rgba8unorm',
      mipLevelCount: STAMP_MIPS,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { species: speciesId, tag: 'stamp-atlas' },
  )
  let off = 0
  for (let level = 0; level < STAMP_MIPS; level++) {
    const dim = STAMP_ATLAS >> level
    const bytes = dim * dim * 4
    device.queue.writeTexture(
      { texture: stampTex, mipLevel: level },
      set.stamp.subarray(off, off + bytes),
      { bytesPerRow: dim * 4, rowsPerImage: dim },
      [dim, dim],
    )
    off += bytes
  }
  return { set, splatBuffer, stampTex }
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
