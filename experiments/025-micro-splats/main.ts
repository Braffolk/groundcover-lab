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
  speciesById,
  standEntrySlots,
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
const INFO_FLOATS = 88
const LOD_STRIDE = 256 // dynamic-offset stride for the per-LOD uniform
const ARG_STRIDE = 32 // drawIndexedIndirect args (5 u32) padded for alignment
const INST_BYTES = 32
/**
 * Projected pixel size of a plant that still earns the finest (8192-splat)
 * level; each further level halves the splat count and moves the switch out by
 * sqrt(2). Because the ladder is geometric, this constant is exactly "how many
 * pixels one splat covers": splat_px = LOD_T0 / sqrt(8192).
 *
 * An upright plant is measured by its HEIGHT, a carpet tile by its FOOTPRINT
 * (`footprint_m`) — a 0.18m x 0.09m Sphagnum tile measured by height picks a
 * level 2 steps too coarse, i.e. 4x too few splats, and the mat collapses into
 * a handful of fat ellipses. The carpet constant is also lower (finer splats):
 * the total splat count over a field is dominated by the 1-splat floor in the
 * far ring, so near-field detail is close to free (see NOTES.md).
 */
const LOD_T0 = 1050
const LOD_T0_CARPET = 700
const LOD_RATIO = Math.SQRT2
/**
 * Levels actually used, per entry kind. The bake ships all 14 tree levels.
 * For an upright plant, below 4 splats there is no silhouette left worth
 * looking at (1-2 ellipses read as blobs on the horizon) while saving almost
 * nothing — those splats are ~1px, so they are primitives, not fill.
 * A CARPET is the opposite case: its far field is a closed opaque mat, one
 * solid ellipse per tile is exactly the right answer, and there are ~30 tiles
 * per m², so the 4-splat floor would cost 4x for nothing.
 */
const LOD_USED = 12
const LOD_USED_CARPET = 14
const LOD_MAX_USED = Math.max(LOD_USED, LOD_USED_CARPET)
if (LOD_MAX_USED > LOD_SPLATS.length) throw new Error('LOD_USED exceeds the baked level count')
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
  /** carpetDiv² for a mat, SCATTER_MAX_PER_CELL otherwise — NEVER hardcode. */
  slotsPerCell: number
  /** Workgroups along the slot axis of the 2D cull dispatch. */
  slotWorkgroups: number
  lodUsed: number
  lodT0: number
  /** Length (m, at scale 1) whose projected size drives the LOD ladder. */
  lodLen: number
  isCarpet: boolean
  cells: number
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
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 48 },
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
    // A carpet has carpetDiv² slots per cell (484 for the bog moss), NOT the
    // 128-slot scatter budget, and its real scale is the one carpetScale()
    // derives from the grid step — standEntry.scaleMin/Max are placeholders
    // there and are ~2x off.
    const slotsPerCell = standEntrySlots(standEntry)
    const carpetDiv = standEntry.carpetDiv ?? 0
    const isCarpet = carpetDiv > 0
    const tileM = speciesById(standEntry.species).tileM ?? 0
    const scaleMax = isCarpet ? CELL / carpetDiv / tileM : standEntry.scaleMax
    console.log('[025 diag]', standEntry.species, JSON.stringify({ slotsPerCell, carpetDiv, tileM, scaleMax, standScaleMin: standEntry.scaleMin, standScaleMax: standEntry.scaleMax, wetWidth: standEntry.wetWidth, y0: gpu.set.y0, y1: gpu.set.y1, rXZ: gpu.set.rXZ }))
    const lodUsed = isCarpet ? LOD_USED_CARPET : LOD_USED
    const lodT0 = isCarpet ? LOD_T0_CARPET : LOD_T0
    const lodLen = isCarpet ? tileM : gpu.set.y1 - gpu.set.y0
    // Instances per m² this entry can actually produce. For a carpet that is
    // the grid, not `density`: div²/cell² slots, of which the entry's wetness
    // interval claims `wetWidth` (the three moss states partition [0,1)).
    const density = isCarpet ? slotsPerCell / (CELL * CELL) : Math.min(standEntry.density, 8)
    // Fraction of the region disc a bucket must be able to hold. A scattered
    // entry is spatially uniform, so the frustum wedge (~25%) plus slack is
    // enough; a carpet's zones clump on the 12m wetness field, so one state can
    // locally be 2-3x its average share and a full bucket means visible holes.
    const wedge = isCarpet ? 0.35 : 0.6
    const R = REGION_MAX
    const discCount = Math.ceil(Math.PI * R * R * density) + 1024
    const d0 = (lodLen * scaleMax * PX_NOMINAL) / lodT0
    const caps: number[] = []
    for (let k = 0; k < lodUsed; k++) {
      // Ring the level owns, widened both ways so the lodBias slider can move
      // the ladder without starving a bucket.
      const lo = Math.min(R, (d0 * LOD_RATIO ** k) / 2.6)
      const hi = k === lodUsed - 1 ? R : Math.min(R, d0 * LOD_RATIO ** (k + 1) * 1.6)
      const ring = Math.PI * Math.max(hi * hi - lo * lo, 0)
      caps.push(Math.min(discCount, Math.ceil(ring * density * wedge) + 1024))
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
        size: LOD_STRIDE * lodUsed,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'lod-info' },
    )
    const lodData = new Float32Array((LOD_STRIDE / 4) * lodUsed)
    for (let k = 0; k < lodUsed; k++) {
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
      //
      // A CARPET gives it up much earlier. A grass canopy really is porous, so
      // an alpha-cut splat set that covers ~75% of its own footprint is honest;
      // a moss mat is a CLOSED opaque surface, and the same 75% shows as bare
      // peat between the splats. Below ~30 px per tile there is no capitulum
      // left to resolve anyway, so the ellipses go solid and the mat closes.
      lodData[o + 4] = isCarpet
        ? Math.min(1, Math.max(0, (k - 8) / 2))
        : Math.min(1, Math.max(0, (k - 10) / 2))
      // face_cam: roll the minor axis toward the eye around the (baked) major
      // axis, so a coarse patch can never turn edge-on and vanish.
      // NEVER for a carpet: a mat's splats are ground-parallel discs, and
      // rolling them toward the eye stands the whole mat up into camera-facing
      // chips — exactly the billboarding that breaks a tiled carpet.
      lodData[o + 5] = isCarpet ? 0 : Math.min(1, Math.max(0, (k - 7) / 3))
      // ao_lift: at range you only ever see the canopy's lit outer shell, never
      // its shaded interior, so the baked occlusion is lifted back toward 1.
      lodData[o + 6] = Math.min(1, Math.max(0, (k - 8) / 4))
      // neutral: only where `solid` already replaced the cut-out (see LodInfo).
      // Grass keeps its stamp colour at every level, so `default` is untouched.
      lodData[o + 7] = isCarpet ? lodData[o + 4]! : 0
      // shrink_mix: never for a carpet — see LodInfo.
      lodData[o + 8] = isCarpet ? 0 : lodData[o + 4]!
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
        size: ARG_STRIDE * lodUsed,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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
      slotsPerCell,
      slotWorkgroups: Math.ceil(slotsPerCell / 64),
      lodUsed,
      lodT0,
      lodLen,
      isCarpet,
      cells: 0,
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
          { binding: 1, resource: { buffer: lodBuffer, size: 48 } },
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
  let diagFrame = 0
  const diagStage: [EntryGpu, GPUBuffer][] = []
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array((ARG_STRIDE / 4) * LOD_MAX_USED)
  for (let k = 0; k < LOD_MAX_USED; k++) indirectReset[k * 8] = LOD_SPLATS[k]! * 6

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
        for (let k = 0; k < entry.lodUsed; k++) {
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
        info[68] = entry.lodT0 / ctx.params.lodBias
        info[69] = 1 / Math.log(LOD_RATIO)
        info[70] = entry.lodUsed
        info[71] = ctx.params.splatScale
        info[72] = ctx.params.minPx
        info[73] = ctx.params.aoStrength
        info[74] = ctx.params.bulge
        // A carpet needs a CLOSED mat, so its stamps are cut at a much lower
        // reference: the same 0.45 that gives a grass blade a crisp silhouette
        // leaves ~25% of a moss tile as bare peat (001-billboard-smoke hit the
        // identical wall and dropped from 0.4 to 0.06 for its carpet).
        info[75] = ctx.params.alphaRef * (entry.isCarpet ? 0.35 : 1)
        info[76] = set.bmin[0]
        info[77] = set.bmin[1]
        info[78] = set.bmin[2]
        info[80] = set.bmax[0] - set.bmin[0]
        info[81] = set.bmax[1] - set.bmin[1]
        info[82] = set.bmax[2] - set.bmin[2]
        info[84] = entry.isCarpet ? 1 : 0
        info[85] = entry.lodLen
        info[86] = entry.slotsPerCell
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset, 0, (ARG_STRIDE / 4) * entry.lodUsed)
        entry.cells = sideX * sideZ
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      diagFrame++
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        if (entry.cells === 0) continue // camera outside the stand
        cull.setBindGroup(1, entry.cullBindGroup)
        // 2D: x = the entry's own slot count (484 for a carpet, 128 otherwise),
        // y = one cell per workgroup row, so the cell-level rejects stay
        // workgroup-uniform and no integer div/mod is needed in the shader.
        cull.dispatchWorkgroups(entry.slotWorkgroups, entry.cells)
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
      for (let k = 0; k < LOD_MAX_USED; k++) {
        for (const entry of entries) {
          if (k >= entry.lodUsed) continue // carpets run 2 levels deeper
          pass.setBindGroup(1, entry.drawBindGroup, [k * LOD_STRIDE])
          pass.drawIndexedIndirect(entry.indirectBuffer, k * ARG_STRIDE)
        }
      }
      pass.end()
      if (diagFrame === 120) {
        for (const entry of entries) {
          const stage = device.createBuffer({ size: ARG_STRIDE * entry.lodUsed, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
          enc.copyBufferToBuffer(entry.indirectBuffer, 0, stage, 0, ARG_STRIDE * entry.lodUsed)
          diagStage.push([entry, stage])
        }
      }
      if (diagFrame === 126) {
        for (const [entry, stage] of diagStage) {
          void stage.mapAsync(GPUMapMode.READ).then(() => {
            const u = new Uint32Array(stage.getMappedRange().slice(0))
            const counts: number[] = []
            for (let k = 0; k < entry.lodUsed; k++) counts.push(u[k * 8 + 1]!)
            console.log('[025 counts]', entry.entryIndex, JSON.stringify({ caps: entry.caps, counts }))
            stage.unmap()
          })
        }
        diagStage.length = 0
      }
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
