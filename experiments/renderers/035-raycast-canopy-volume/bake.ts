import voxelizeSrc from './shaders/voxelize.wgsl'
import rayanswerSrc from './shaders/rayanswer.wgsl'
import {
  BAND_AZIM,
  BAND_BASE,
  BAND_COUNT,
  BAND_ZENITH_DEG,
  LAYERS,
  PALETTE_BINS,
  TABLE_U,
  TABLE_VERSION,
  packTable,
  tileFor,
  unpackTable,
  type CanopyTable,
} from './layout.ts'
import {
  bakedArtifact,
  commitBake,
  speciesById,
  type ExperimentContext,
  type GcMesh,
  type Scatter,
  type StandSpecies,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * The ray-answer bake. Per stand entry:
 *
 *  1. Build ONE periodic tile of the entry's real scatter plants — the plants
 *     the stand puts inside an LxL window, at their exact positions, yaws and
 *     scales, on a flat local ground.
 *  2. Splat the source mesh (2-8M vertices per instance) into a wrapping 5mm
 *     occupancy bitmask + a 4cm skip mask + a 1.5cm normal volume.
 *  3. March 4 sub-rays per (texel, direction) through that volume and store the
 *     FIRST HIT as (q, coverage, oct normal) — 4.09M answers per entry.
 *  4. Bin the mesh colours by height into a 2-cluster albedo curve.
 *
 * Marching happens ONLY here. At runtime the answer is one texture fetch.
 */

const VOX = 0.005 // occupancy voxel (m)
const COARSE_BLOCK = 8 // fine voxels per coarse skip cell
const ATTR_VOX = 0.015 // normal volume voxel (m)
const ATTR_STRIDE = 4 // every Nth vertex contributes a normal
const MAX_MARCH_STEPS = 2500
const VERT_CHUNK = 4_000_000 // keep each storage binding under 128MiB
const PALETTE_SAMPLES = 300_000

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export interface EntryBakeInput {
  entry: StandSpecies
  entryIndex: number
  scatter: Scatter
  seed: number
}

function slug(x: number): string {
  return String(x).replace('.', 'p')
}

export function tableKey(input: EntryBakeInput): string {
  const e = input.entry
  return `table-v${TABLE_VERSION}-${e.species}-d${slug(e.density)}-s${slug(e.scaleMin)}-${slug(
    e.scaleMax,
  )}-sd${input.seed}`
}

/** Load from OPFS cache / committed artifact, else bake in-browser and commit. */
export async function loadEntryTable(
  ctx: BakeCtx,
  input: EntryBakeInput,
  note: (msg: string) => void,
): Promise<CanopyTable> {
  const key = tableKey(input)
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    note(`baking ${input.entry.species} ray table…`)
    const mesh = await ctx.meshes.load(speciesById(input.entry.species).meshId)
    return bakeEntryTable(ctx, input, mesh, note)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let table = unpackTable(buf)
  if (!table) {
    // A missing /mesh/baked file comes back as index.html at HTTP 200 under the
    // dev server's SPA fallback — never trust an unvalidated artifact.
    buf = await runBake()
    table = unpackTable(buf)
    if (!table) throw new Error(`[${ctx.id}] bake for ${input.entry.species} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return table
}

async function opfsRepair(fullKey: string, data: ArrayBuffer): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle('bake-cache', { create: true })
    const handle = await dir.getFileHandle(`${fullKey}.bin`, { create: true })
    const writable = await handle.createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    /* best effort */
  }
}

interface TileInstance {
  x: number
  z: number
  factor: number
  yaw: number
}

/** The stand's own plants inside one LxL window, mapped into the tile. */
function tileInstances(input: EntryBakeInput, period: number, meshSpanY: number): TileInstance[] {
  const heightScale = speciesById(input.entry.species).heightScale
  const points = input.scatter.region(input.entryIndex, { minX: 0, minZ: 0, maxX: period, maxZ: period })
  const out: TileInstance[] = []
  for (const p of points) {
    if (p.x < 0 || p.x >= period || p.z < 0 || p.z >= period) continue
    out.push({ x: p.x, z: p.z, factor: (heightScale * p.scale) / meshSpanY, yaw: p.yaw })
  }
  return out
}

async function bakeEntryTable(
  ctx: BakeCtx,
  input: EntryBakeInput,
  mesh: GcMesh,
  note: (msg: string) => void,
): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const species = speciesById(input.entry.species)
  const { period, psi } = tileFor(input.entry.species)
  const height = species.heightScale * input.entry.scaleMax * 1.02

  const baseY = Math.min(0, hdr.boundsMin[1])
  const meshSpanY = hdr.boundsMax[1] - baseY
  const pivotX = (hdr.boundsMin[0] + hdr.boundsMax[0]) / 2
  const pivotZ = (hdr.boundsMin[2] + hdr.boundsMax[2]) / 2

  const instances = tileInstances(input, period, meshSpanY)
  if (instances.length === 0) throw new Error(`[${ctx.id}] no plants in the ${period}m tile window`)

  // --- volume dimensions ----------------------------------------------------
  const vx = Math.max(16, Math.round(period / VOX))
  const vy = Math.max(16, Math.ceil(height / VOX))
  const wordsX = Math.ceil(vx / 32)
  const cx = Math.ceil(vx / COARSE_BLOCK)
  const cy = Math.ceil(vy / COARSE_BLOCK)
  const wordsCX = Math.ceil(cx / 32)
  const ax = Math.max(8, Math.ceil(period / ATTR_VOX))
  const ay = Math.max(8, Math.ceil(height / ATTR_VOX))

  const occBytes = vy * vx * wordsX * 4
  const coarseBytes = cy * cx * wordsCX * 4
  const attrBytes = ay * ax * ax * 3 * 4
  note(
    `${input.entry.species}: ${instances.length} plants / ${period}m tile, ` +
      `occ ${vx}³ (${(occBytes / 1e6).toFixed(0)}MB), attr ${(attrBytes / 1e6).toFixed(0)}MB`,
  )

  const scratch: (GPUBuffer | GPUTexture)[] = []
  const mkBuffer = (label: string, size: number, usage: number): GPUBuffer => {
    const b = ctx.res.createBuffer({ label: `${ctx.id}/${label}`, size, usage }, { tag: 'bake-scratch' })
    scratch.push(b)
    return b
  }

  try {
    const vertBytes = mesh.vertices.byteLength
    const vbuf = mkBuffer('bake-verts', vertBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(vbuf, 0, mesh.vertices)

    const instData = new Float32Array(instances.length * 4)
    instances.forEach((p, i) => instData.set([p.x, p.z, p.factor, p.yaw], i * 4))
    const ibuf = mkBuffer('bake-inst', instData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(ibuf, 0, instData)

    const occ = mkBuffer('bake-occ', occBytes, GPUBufferUsage.STORAGE)
    const coarse = mkBuffer('bake-coarse', coarseBytes, GPUBufferUsage.STORAGE)
    const attr = mkBuffer('bake-attr', attrBytes, GPUBufferUsage.STORAGE)

    // ---- stage 1: voxelize -------------------------------------------------
    const voxUni = mkBuffer('bake-vox-uni', 112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const voxModule = ctx.shaders.module(voxelizeSrc)
    const voxBgl = device.createBindGroupLayout({
      label: `${ctx.id}/vox-bgl`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    const voxPipe = device.createComputePipeline({
      label: `${ctx.id}/vox`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/vox-pl`, bindGroupLayouts: [voxBgl] }),
      compute: { module: voxModule, entryPoint: 'splat' },
    })

    const vinfo = new ArrayBuffer(112)
    const vf = new Float32Array(vinfo)
    const vu = new Uint32Array(vinfo)
    vf.set([hdr.boundsMin[0], hdr.boundsMin[1], hdr.boundsMin[2], baseY], 0)
    vf.set(
      [
        (hdr.boundsMax[0] - hdr.boundsMin[0]) / 65535,
        (hdr.boundsMax[1] - hdr.boundsMin[1]) / 65535,
        (hdr.boundsMax[2] - hdr.boundsMin[2]) / 65535,
        0,
      ],
      4,
    )
    vf.set([pivotX, pivotZ, period, height], 8)
    vu.set([vx, vy, vx, wordsX], 12)
    vu.set([ax, ay, ax, hdr.vertexCount], 16)
    vu.set([cx, cy, cx, wordsCX], 20)

    for (let start = 0; start < hdr.vertexCount; start += VERT_CHUNK) {
      const count = Math.min(VERT_CHUNK, hdr.vertexCount - start)
      vu.set([start, count, ATTR_STRIDE, 0], 24)
      device.queue.writeBuffer(voxUni, 0, vinfo)
      const bg = device.createBindGroup({
        label: `${ctx.id}/vox-bg`,
        layout: voxBgl,
        entries: [
          { binding: 0, resource: { buffer: voxUni } },
          { binding: 1, resource: { buffer: vbuf, offset: start * 16, size: count * 16 } },
          { binding: 2, resource: { buffer: ibuf } },
          { binding: 3, resource: { buffer: occ } },
          { binding: 4, resource: { buffer: coarse } },
          { binding: 5, resource: { buffer: attr } },
        ],
      })
      const enc = device.createCommandEncoder({ label: `${ctx.id}/vox-enc` })
      const pass = enc.beginComputePass({ label: `${ctx.id}/vox-pass` })
      pass.setPipeline(voxPipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(Math.ceil(count / 64), instances.length, 1)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    // ---- stage 2: ray answers ---------------------------------------------
    note(`${input.entry.species}: resolving ${((TABLE_U * TABLE_U * LAYERS * 4) / 1e6).toFixed(1)}M rays…`)
    const tableBytes = TABLE_U * TABLE_U * LAYERS * 4
    const out = mkBuffer('bake-out', tableBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
    const rayUni = mkBuffer('bake-ray-uni', 256 * BAND_COUNT, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const rayModule = ctx.shaders.module(rayanswerSrc)
    const rayBgl = device.createBindGroupLayout({
      label: `${ctx.id}/ray-bgl`,
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
        },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    const rayPipe = device.createComputePipeline({
      label: `${ctx.id}/ray`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/ray-pl`, bindGroupLayouts: [rayBgl] }),
      compute: { module: rayModule, entryPoint: 'answer' },
    })
    const rayBg = device.createBindGroup({
      label: `${ctx.id}/ray-bg`,
      layout: rayBgl,
      entries: [
        { binding: 0, resource: { buffer: rayUni, size: 96 } },
        { binding: 1, resource: { buffer: occ } },
        { binding: 2, resource: { buffer: coarse } },
        { binding: 3, resource: { buffer: attr } },
        { binding: 4, resource: { buffer: out } },
      ],
    })

    const rayData = new ArrayBuffer(256 * BAND_COUNT)
    for (let b = 0; b < BAND_COUNT; b++) {
      const f = new Float32Array(rayData, b * 256, 24)
      const u = new Uint32Array(rayData, b * 256, 24)
      u.set([vx, vy, vx, wordsX], 0)
      u.set([ax, ay, ax, TABLE_U], 4)
      u.set([cx, cy, cx, wordsCX], 8)
      f.set([period, height, period / vx, period / cx], 12)
      f.set([Math.tan((BAND_ZENITH_DEG[b]! * Math.PI) / 180), BAND_AZIM[b]!, BAND_BASE[b]!, 0], 16)
      f.set([MAX_MARCH_STEPS, 0, 0, 0], 20)
    }
    device.queue.writeBuffer(rayUni, 0, rayData)

    const groups = Math.ceil(TABLE_U / 8)
    for (let b = 0; b < BAND_COUNT; b++) {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/ray-enc-${b}` })
      const pass = enc.beginComputePass({ label: `${ctx.id}/ray-pass-${b}` })
      pass.setPipeline(rayPipe)
      pass.setBindGroup(0, rayBg, [b * 256])
      pass.dispatchWorkgroups(groups, groups, BAND_AZIM[b]!)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      note(`${input.entry.species}: band ${b + 1}/${BAND_COUNT} resolved`)
    }

    // ---- readback ----------------------------------------------------------
    const rb = mkBuffer('bake-rb', tableBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    const enc = device.createCommandEncoder({ label: `${ctx.id}/ray-copy` })
    enc.copyBufferToBuffer(out, 0, rb, 0, tableBytes)
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const data = new Uint8Array(rb.getMappedRange()).slice()
    rb.unmap()

    const meanQ = bandMeanQ(data)
    const palette = bakePalette(mesh, instances, height, baseY)

    return packTable({
      u: TABLE_U,
      layers: LAYERS,
      period,
      height,
      psi,
      plants: instances.length,
      meanQ,
      palette,
      data,
    })
  } finally {
    for (const r of scratch) r.destroy()
  }
}

/** Coverage-weighted mean q per band — the parallax-correction reference. */
function bandMeanQ(data: Uint8Array): Float32Array {
  const out = new Float32Array(BAND_COUNT)
  const layerTexels = TABLE_U * TABLE_U
  for (let b = 0; b < BAND_COUNT; b++) {
    let wsum = 0
    let qsum = 0
    const from = BAND_BASE[b]! * layerTexels
    const to = from + BAND_AZIM[b]! * layerTexels
    for (let i = from; i < to; i += 7) {
      const cov = data[i * 4 + 1]! / 255
      qsum += (data[i * 4]! / 255) * cov
      wsum += cov
    }
    out[b] = wsum > 1e-3 ? qsum / wsum : 0.25
  }
  return out
}

/**
 * Height-binned albedo curve with TWO colour clusters per bin, split along the
 * red-vs-green axis: for calamagrostis that separates the pink panicles from
 * the green leaves at the same height, so a tuft-scale hash can mix real
 * colours that actually occur up there instead of inventing a tint.
 */
function bakePalette(mesh: GcMesh, instances: TileInstance[], height: number, baseY: number): Float32Array {
  const hdr = mesh.header
  const spanY = hdr.boundsMax[1] - hdr.boundsMin[1]
  const minY = hdr.boundsMin[1]
  const v = mesh.vertices
  const perInstance = Math.max(1, Math.floor(PALETTE_SAMPLES / instances.length))
  const stride = Math.max(1, Math.floor(hdr.vertexCount / perInstance))

  const bins = PALETTE_BINS
  const sampleBin: number[] = []
  const sampleR: number[] = []
  const sampleG: number[] = []
  const sampleB: number[] = []

  for (const inst of instances) {
    for (let i = 0; i < hdr.vertexCount; i += stride) {
      const qy = v[i * 8 + 1]!
      const y = ((minY + (qy / 65535) * spanY - baseY) * inst.factor) / height
      if (y < 0 || y >= 1) continue
      sampleBin.push(Math.min(bins - 1, Math.floor(y * bins)))
      sampleR.push(v[i * 8 + 3]! / 65535)
      sampleG.push(v[i * 8 + 4]! / 65535)
      sampleB.push(v[i * 8 + 5]! / 65535)
    }
  }

  // pass 1: per-bin mean of the split axis
  const axisSum = new Float64Array(bins)
  const count = new Float64Array(bins)
  for (let i = 0; i < sampleBin.length; i++) {
    const b = sampleBin[i]!
    axisSum[b] = (axisSum[b] ?? 0) + (sampleR[i]! - sampleG[i]!)
    count[b] = (count[b] ?? 0) + 1
  }
  // pass 2: mean colour of each side of the split
  const sums = new Float64Array(bins * 2 * 3)
  const counts = new Float64Array(bins * 2)
  for (let i = 0; i < sampleBin.length; i++) {
    const b = sampleBin[i]!
    const mid = count[b]! > 0 ? axisSum[b]! / count[b]! : 0
    const side = sampleR[i]! - sampleG[i]! >= mid ? 1 : 0
    const o = (b * 2 + side) * 3
    sums[o] = (sums[o] ?? 0) + sampleR[i]!
    sums[o + 1] = (sums[o + 1] ?? 0) + sampleG[i]!
    sums[o + 2] = (sums[o + 2] ?? 0) + sampleB[i]!
    const ci = b * 2 + side
    counts[ci] = (counts[ci] ?? 0) + 1
  }

  const out = new Float32Array(bins * 2 * 4)
  let lastA: [number, number, number] = [0.24, 0.3, 0.13]
  let lastB: [number, number, number] = [0.3, 0.34, 0.16]
  for (let b = 0; b < bins; b++) {
    for (let side = 0; side < 2; side++) {
      const n = counts[b * 2 + side]!
      const o = (b * 2 + side) * 3
      let c: [number, number, number]
      if (n >= 8) {
        c = [sums[o]! / n, sums[o + 1]! / n, sums[o + 2]! / n]
      } else {
        c = side === 0 ? lastA : lastB
      }
      if (side === 0) lastA = c
      else lastB = c
      out.set([c[0], c[1], c[2], 1], (b * 2 + side) * 4)
    }
  }
  return out
}
