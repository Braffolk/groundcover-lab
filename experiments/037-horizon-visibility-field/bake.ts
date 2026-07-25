import voxelizeSrc from './shaders/bake_voxelize.wgsl'
import resolveSrc from './shaders/bake_resolve.wgsl'
import traceSrc from './shaders/bake_trace.wgsl'
import { bakedArtifact, commitBake, hash2, hash4, hashF32, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * THE PRECOMPUTED RAYCAST.
 *
 * Per stand entry (species + its density and scale range) we bake a 4D table of
 * ray ANSWERS. It is built in three GPU stages, all offline:
 *
 *   1  voxelize  — a periodic tile of the entry's real GCMESH1 plants, scattered
 *                  at the stand's density, is splatted into a canopy volume as
 *                  surface-area density + mean normal + mean authored colour.
 *   2  resolve   — that volume becomes two filterable 3D textures, and vertical
 *                  transmittance (sky openness = AO) is swept in the same pass.
 *   3  trace     — for every entry point on the canopy top plane x every view
 *                  direction, the ray is integrated ONCE and the answer stored:
 *                  where the canopy first blocks it, with normal, coverage,
 *                  albedo and AO. This is the only ray march in the technique.
 *
 * Mip chains are built on the CPU, coverage-weighted (normals are averaged as
 * vectors, not as oct bytes), so distance means an honestly averaged canopy
 * rather than a shimmering sharp one.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'HVF1', u32 version, u32 res, u32 nPhi, u32 nTheta, u32 levels
 *   f32 tileNorm, hmax, sinMin, radius, meshHeight, plantCount
 *   ... zeros to byte 64
 *   A levels (res, res/2, ...) each res_l^2 * layers * 4 bytes rgba8
 *   B levels (res/2, res/4, ...) each res_l^2 * layers * 4 bytes rgba8
 */

export const RES = 128
export const N_PHI = 16
export const N_THETA = 10
export const TILE_NORM = 1.2
export const SIN_MIN = 0.05
const VOX_XZ = 128
const VOX_Y = 107
const AREA_FIXED = 262144
const MAGIC = 0x31465648 // 'HVF1'
const VERSION = 6
const HEADER_BYTES = 64

export interface RayTable {
  res: number
  nPhi: number
  nTheta: number
  levels: number
  tileNorm: number
  /** World metres per normalized height unit (heightScale * scaleMax). */
  hmax: number
  sinMin: number
  /** Horizontal support radius of one plant at scale 1 (m). */
  radius: number
  plantCount: number
  a: Uint8Array<ArrayBuffer>[]
  b: Uint8Array<ArrayBuffer>[]
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export interface EntrySpec {
  speciesId: string
  /**
   * Plants/m2 the TILE is populated at. The stand's TOTAL density is used, not
   * the entry's own: a column the scatter gives to calamagrostis also has the
   * other entries' plants under it, so a per-entry-density tile would leave the
   * canopy far too thin. Species identity per column still comes from the real
   * scatter via the hull field.
   */
  density: number
  scaleMin: number
  scaleMax: number
  heightScale: number
}

const layers = N_PHI * N_THETA

function levelSizes(res: number): number[] {
  const out: number[] = []
  let r = res
  while (r >= 1) {
    out.push(r)
    if (r === 1) break
    r = Math.max(1, r >> 1)
  }
  return out
}

export function unpackTable(buf: ArrayBuffer): RayTable | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 6)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const res = u[2]!
  const nPhi = u[3]!
  const nTheta = u[4]!
  const levels = u[5]!
  if (res !== RES || nPhi !== N_PHI || nTheta !== N_THETA) return null
  const f = new Float32Array(buf, 24, 6)
  const sizes = levelSizes(res)
  const a: Uint8Array<ArrayBuffer>[] = []
  const b: Uint8Array<ArrayBuffer>[] = []
  let off = HEADER_BYTES
  for (let l = 0; l < levels; l++) {
    const r = sizes[l]!
    const bytes = r * r * layers * 4
    if (off + bytes > buf.byteLength) return null
    a.push(new Uint8Array(buf, off, bytes))
    off += bytes
  }
  for (let l = 1; l < levels; l++) {
    const r = sizes[l]!
    const bytes = r * r * layers * 4
    if (off + bytes > buf.byteLength) return null
    b.push(new Uint8Array(buf, off, bytes))
    off += bytes
  }
  if (off !== buf.byteLength) return null
  return {
    res,
    nPhi,
    nTheta,
    levels,
    tileNorm: f[0]!,
    hmax: f[1]!,
    sinMin: f[2]!,
    radius: f[3]!,
    plantCount: f[5]!,
    a,
    b,
  }
}

function packTable(t: RayTable, meshHeight: number): ArrayBuffer {
  let bytes = HEADER_BYTES
  for (const level of t.a) bytes += level.byteLength
  for (const level of t.b) bytes += level.byteLength
  const buf = new ArrayBuffer(bytes)
  const u = new Uint32Array(buf, 0, 6)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = t.res
  u[3] = t.nPhi
  u[4] = t.nTheta
  u[5] = t.levels
  const f = new Float32Array(buf, 24, 6)
  f[0] = t.tileNorm
  f[1] = t.hmax
  f[2] = t.sinMin
  f[3] = t.radius
  f[4] = meshHeight
  f[5] = t.plantCount
  let off = HEADER_BYTES
  const dst = new Uint8Array(buf)
  for (const level of t.a) {
    dst.set(level, off)
    off += level.byteLength
  }
  for (const level of t.b) {
    dst.set(level, off)
    off += level.byteLength
  }
  return buf
}

export function tableKey(spec: EntrySpec): string {
  const d = spec.density.toFixed(2)
  const s = `${spec.scaleMin.toFixed(2)}-${spec.scaleMax.toFixed(2)}`
  return `hvf-v${VERSION}-${spec.speciesId}-d${d}-s${s}`
}

export async function loadEntryTable(ctx: BakeCtx, spec: EntrySpec): Promise<RayTable> {
  const key = tableKey(spec)
  let fresh = false
  const run = async (): Promise<ArrayBuffer> => {
    fresh = true
    const mesh = await ctx.meshes.load(speciesById(spec.speciesId).meshId)
    return bakeEntryTable(ctx, spec, mesh)
  }
  let buf = await bakedArtifact({ expId: ctx.id, key }, run)
  let table = unpackTable(buf)
  if (!table) {
    // The dev server answers missing /mesh/baked files with index.html at 200,
    // which poisons both stores — rebake and repair rather than trust it.
    buf = await run()
    table = unpackTable(buf)
    if (!table) throw new Error(`[${ctx.id}] bake for ${spec.speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (fresh) {
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
    /* best effort — a cache miss just rebakes */
  }
}

/** Exact horizontal support radius of the source mesh about its xz centre. */
function meshRadius(mesh: GcMesh): { radius: number; cx: number; cz: number } {
  const h = mesh.header
  const [bx0, , bz0] = h.boundsMin
  const [bx1, , bz1] = h.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const sx = (bx1 - bx0) / 65535
  const sz = (bz1 - bz0) / 65535
  const v = mesh.vertices
  let r2 = 0
  for (let i = 0; i < h.vertexCount; i++) {
    const dx = bx0 + v[i * 8]! * sx - cx
    const dz = bz0 + v[i * 8 + 2]! * sz - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  return { radius: Math.sqrt(r2), cx, cz }
}

async function bakeEntryTable(ctx: BakeCtx, spec: EntrySpec, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const hmax = spec.heightScale * spec.scaleMax
  const { radius, cx, cz } = meshRadius(mesh)
  const meshHeight = Math.max(1e-4, hdr.boundsMax[1] - hdr.boundsMin[1])
  const voxel = TILE_NORM / VOX_XZ
  const tileWorld = TILE_NORM * hmax
  const plantCount = Math.max(1, Math.round(spec.density * tileWorld * tileWorld))

  // --- tile instances: the entry's own density and scale range, shared hash ---
  const salt = 0x37f1e1d
  const inst = new Float32Array(plantCount * 4)
  for (let i = 0; i < plantCount; i++) {
    const h = hash4(salt, i, 0, 7)
    inst[i * 4] = hashF32(hash2(h, 1)) * TILE_NORM
    inst[i * 4 + 1] = hashF32(hash2(h, 2)) * TILE_NORM
    inst[i * 4 + 2] = hashF32(hash2(h, 3)) * Math.PI * 2
    inst[i * 4 + 3] = spec.scaleMin + (spec.scaleMax - spec.scaleMin) * hashF32(hash2(h, 4))
  }
  // Instance offsets are in normalized units; the voxelizer works there too.

  const scratch: { destroy(): void }[] = []
  const buffer = (label: string, size: number, usage: number): GPUBuffer => {
    const b = ctx.res.createBuffer({ label: `${ctx.id}/${label}`, size, usage }, { tag: 'bake-scratch' })
    scratch.push(b)
    return b
  }
  const texture = (label: string, desc: GPUTextureDescriptor): GPUTexture => {
    const t = ctx.res.createTexture({ ...desc, label: `${ctx.id}/${label}` }, { tag: 'bake-scratch' })
    scratch.push(t)
    return t
  }

  try {
    // Repack to 12B/vertex (u16 xyz + u8 rgb + u8 oct) and 12B/triangle: poa's
    // raw 16B records exceed maxStorageBufferBindingSize (128MiB) on their own.
    const src = mesh.vertices
    const packedVerts = new Uint32Array(hdr.vertexCount * 3)
    for (let i = 0; i < hdr.vertexCount; i++) {
      const s = i * 8
      packedVerts[i * 3] = src[s]! | (src[s + 1]! << 16)
      packedVerts[i * 3 + 1] =
        src[s + 2]! | ((src[s + 3]! >> 8) << 16) | ((src[s + 4]! >> 8) << 24)
      packedVerts[i * 3 + 2] = (src[s + 5]! >> 8) | ((src[s + 6]! >> 8) << 8) | ((src[s + 7]! >> 8) << 16)
    }
    const vbuf = buffer('bake-verts', packedVerts.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(vbuf, 0, packedVerts)
    const indices = mesh.indices()
    const tbuf = buffer('bake-tris', indices.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(tbuf, 0, indices)
    const ibuf = buffer('bake-inst', inst.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(ibuf, 0, inst)

    const voxelCount = VOX_XZ * VOX_XZ * VOX_Y
    const accU = buffer('bake-accu', voxelCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    const accI = buffer('bake-acci', voxelCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    const accP = buffer('bake-accp', voxelCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)

    const voxUni = buffer('bake-voxuni', 80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    {
      const f = new Float32Array(20)
      const u = new Uint32Array(f.buffer)
      u[0] = VOX_XZ
      u[1] = VOX_Y
      u[2] = VOX_XZ
      u[3] = hdr.triangleCount
      f.set(hdr.boundsMin, 4)
      u[7] = plantCount
      f[8] = hdr.boundsMax[0] - hdr.boundsMin[0]
      f[9] = hdr.boundsMax[1] - hdr.boundsMin[1]
      f[10] = hdr.boundsMax[2] - hdr.boundsMin[2]
      f[11] = voxel
      f[12] = cx
      f[13] = cz
      f[14] = TILE_NORM
      f[15] = spec.heightScale / meshHeight
      f[16] = 1 / hmax
      f[17] = AREA_FIXED / (voxel * voxel)
      f[18] = hdr.boundsMin[1]
      device.queue.writeBuffer(voxUni, 0, f)
    }

    const voxModule = ctx.shaders.module(voxelizeSrc)
    const voxPipeline = device.createComputePipeline({
      label: `${ctx.id}/voxelize`,
      layout: 'auto',
      compute: { module: voxModule, entryPoint: 'cs_voxelize' },
    })
    const voxBg = device.createBindGroup({
      label: `${ctx.id}/voxelize-bg`,
      layout: voxPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vbuf } },
        { binding: 1, resource: { buffer: tbuf } },
        { binding: 2, resource: { buffer: ibuf } },
        { binding: 3, resource: { buffer: accU } },
        { binding: 4, resource: { buffer: accI } },
        { binding: 5, resource: { buffer: voxUni } },
        { binding: 6, resource: { buffer: accP } },
      ],
    })

    const volGeo = texture('bake-volgeo', {
      size: [VOX_XZ, VOX_Y, VOX_XZ],
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const volCol = texture('bake-volcol', {
      size: [VOX_XZ, VOX_Y, VOX_XZ],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const volNrm = texture('bake-volnrm', {
      size: [VOX_XZ, VOX_Y, VOX_XZ],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })

    const resUni = buffer('bake-resuni', 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    {
      const f = new Float32Array(8)
      const u = new Uint32Array(f.buffer)
      u[0] = VOX_XZ
      u[1] = VOX_Y
      u[2] = VOX_XZ
      f[3] = AREA_FIXED
      f[4] = voxel
      device.queue.writeBuffer(resUni, 0, f)
    }
    const resPipeline = device.createComputePipeline({
      label: `${ctx.id}/vol-resolve`,
      layout: 'auto',
      compute: { module: ctx.shaders.module(resolveSrc), entryPoint: 'cs_resolve' },
    })
    const resBg = device.createBindGroup({
      label: `${ctx.id}/vol-resolve-bg`,
      layout: resPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: accU } },
        { binding: 1, resource: { buffer: accI } },
        { binding: 2, resource: volGeo.createView() },
        { binding: 3, resource: volCol.createView() },
        { binding: 4, resource: { buffer: resUni } },
        { binding: 5, resource: { buffer: accP } },
        { binding: 6, resource: volNrm.createView() },
      ],
    })

    const outBytes = RES * RES * layers * 4
    const outA = buffer('bake-outa', outBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
    const outB = buffer('bake-outb', outBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
    const traceUni = buffer('bake-traceuni', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    {
      const f = new Float32Array(12)
      const u = new Uint32Array(f.buffer)
      u[0] = RES
      u[1] = N_PHI
      u[2] = N_THETA
      u[3] = 640
      f[4] = TILE_NORM
      f[5] = VOX_Y * voxel
      f[6] = SIN_MIN
      f[7] = voxel * 0.6
      f[8] = 0.7
      f[9] = voxel * 8
      device.queue.writeBuffer(traceUni, 0, f)
    }
    const volSampler = device.createSampler({
      label: `${ctx.id}/vol-sampler`,
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'repeat',
    })
    const tracePipeline = device.createComputePipeline({
      label: `${ctx.id}/trace`,
      layout: 'auto',
      compute: { module: ctx.shaders.module(traceSrc), entryPoint: 'cs_trace' },
    })
    const traceBg = device.createBindGroup({
      label: `${ctx.id}/trace-bg`,
      layout: tracePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: volGeo.createView() },
        { binding: 1, resource: volCol.createView() },
        { binding: 2, resource: volSampler },
        { binding: 3, resource: { buffer: outA } },
        { binding: 4, resource: { buffer: outB } },
        { binding: 5, resource: { buffer: traceUni } },
        { binding: 6, resource: volNrm.createView() },
      ],
    })

    // --- stage 1+2: voxelize then resolve --------------------------------------
    const total = hdr.triangleCount * plantCount
    const wgTotal = Math.ceil(total / 64)
    const wgX = Math.min(wgTotal, 65535)
    const wgY = Math.ceil(wgTotal / wgX)
    {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-vox` })
      enc.clearBuffer(accU)
      enc.clearBuffer(accI)
      enc.clearBuffer(accP)
      const pass = enc.beginComputePass({ label: `${ctx.id}/voxelize` })
      pass.setPipeline(voxPipeline)
      pass.setBindGroup(0, voxBg)
      pass.dispatchWorkgroups(wgX, wgY)
      pass.end()
      const rp = enc.beginComputePass({ label: `${ctx.id}/vol-resolve` })
      rp.setPipeline(resPipeline)
      rp.setBindGroup(0, resBg)
      rp.dispatchWorkgroups(Math.ceil(VOX_XZ / 8), Math.ceil(VOX_XZ / 8))
      rp.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    // --- stage 3: the precomputed raycast ---------------------------------------
    // One dispatch over every (entry point x direction) — 240 layers is well
    // inside maxComputeWorkgroupsPerDimension.
    {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-trace` })
      const pass = enc.beginComputePass({ label: `${ctx.id}/trace` })
      pass.setPipeline(tracePipeline)
      pass.setBindGroup(0, traceBg)
      pass.dispatchWorkgroups(Math.ceil(RES / 8), Math.ceil(RES / 8), layers)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    // --- readback -------------------------------------------------------------
    const rbA = buffer('bake-rba', outBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    const rbB = buffer('bake-rbb', outBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-readback` })
      enc.copyBufferToBuffer(outA, 0, rbA, 0, outBytes)
      enc.copyBufferToBuffer(outB, 0, rbB, 0, outBytes)
      device.queue.submit([enc.finish()])
    }
    await rbA.mapAsync(GPUMapMode.READ)
    await rbB.mapAsync(GPUMapMode.READ)
    const baseA = new Uint8Array(rbA.getMappedRange()).slice()
    const baseB = new Uint8Array(rbB.getMappedRange()).slice()
    rbA.unmap()
    rbB.unmap()

    const { a, b } = buildMips(baseA, baseB)
    return packTable(
      {
        res: RES,
        nPhi: N_PHI,
        nTheta: N_THETA,
        levels: a.length,
        tileNorm: TILE_NORM,
        hmax,
        sinMin: SIN_MIN,
        radius,
        plantCount,
        a,
        b,
      },
      meshHeight,
    )
  } finally {
    for (const r of scratch) r.destroy()
  }
}

function octDecode(u: number, v: number): [number, number, number] {
  const fu = (u / 255) * 2 - 1
  const fv = (v / 255) * 2 - 1
  let x = fu
  let z = fv
  const y = 1 - Math.abs(fu) - Math.abs(fv)
  if (y < 0) {
    x = (1 - Math.abs(fv)) * (fu >= 0 ? 1 : -1)
    z = (1 - Math.abs(fu)) * (fv >= 0 ? 1 : -1)
  }
  const len = Math.hypot(x, y, z) || 1
  return [x / len, y / len, z / len]
}

function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  if (s < 1e-6) return [128, 128]
  let u = x / s
  let v = z / s
  if (y < 0) {
    const fu = (1 - Math.abs(v)) * (x >= 0 ? 1 : -1)
    const fv = (1 - Math.abs(u)) * (z >= 0 ? 1 : -1)
    u = fu
    v = fv
  }
  return [Math.round((u * 0.5 + 0.5) * 255), Math.round((v * 0.5 + 0.5) * 255)]
}

/**
 * Coverage-weighted mip chains. The ray answer must average as a canopy, not as
 * bytes: distance and albedo are weighted by coverage, normals are averaged as
 * decoded vectors and re-encoded, coverage itself averages linearly.
 */
function buildMips(
  baseA: Uint8Array<ArrayBuffer>,
  baseB: Uint8Array<ArrayBuffer>,
): { a: Uint8Array<ArrayBuffer>[]; b: Uint8Array<ArrayBuffer>[] } {
  const sizes = levelSizes(RES)
  const a: Uint8Array<ArrayBuffer>[] = [baseA]
  const b: Uint8Array<ArrayBuffer>[] = []
  for (let l = 1; l < sizes.length; l++) {
    const rs = sizes[l - 1]!
    const rd = sizes[l]!
    const src = a[l - 1]!
    const dst = new Uint8Array(rd * rd * layers * 4)
    const srcB = l === 1 ? baseB : b[l - 2]!
    const dstB = new Uint8Array(rd * rd * layers * 4)
    for (let layer = 0; layer < layers; layer++) {
      const so = layer * rs * rs * 4
      const doff = layer * rd * rd * 4
      for (let y = 0; y < rd; y++) {
        for (let x = 0; x < rd; x++) {
          let cov = 0
          let e = 0
          let nx = 0
          let ny = 0
          let nz = 0
          let plainE = 0
          let br = 0
          let bg = 0
          let bb = 0
          let ba = 0
          let n = 0
          for (let j = 0; j < 2; j++) {
            for (let i = 0; i < 2; i++) {
              const sx = Math.min(rs - 1, x * 2 + i)
              const sy = Math.min(rs - 1, y * 2 + j)
              const s = so + (sy * rs + sx) * 4
              const c = src[s + 3]! / 255
              cov += c
              plainE += src[s]!
              n++
              if (c > 0) {
                e += src[s]! * c
                const [dx, dy, dz] = octDecode(src[s + 1]!, src[s + 2]!)
                nx += dx * c
                ny += dy * c
                nz += dz * c
                br += srcB[s]! * c
                bg += srcB[s + 1]! * c
                bb += srcB[s + 2]! * c
                ba += srcB[s + 3]! * c
              }
            }
          }
          const d = doff + (y * rd + x) * 4
          const cAvg = cov / n
          dst[d + 3] = Math.round(Math.min(1, cAvg) * 255)
          if (cov > 1e-4) {
            dst[d] = Math.round(e / cov)
            const [ou, ov] = octEncode(nx, ny, nz)
            dst[d + 1] = ou
            dst[d + 2] = ov
            dstB[d] = Math.round(br / cov)
            dstB[d + 1] = Math.round(bg / cov)
            dstB[d + 2] = Math.round(bb / cov)
            dstB[d + 3] = Math.round(ba / cov)
          } else {
            dst[d] = Math.round(plainE / n)
            dst[d + 1] = 128
            dst[d + 2] = 128
            dstB[d + 3] = 255
          }
        }
      }
    }
    if (rd <= DIR_BLUR_FROM) {
      dirBlur(dst, rd)
      dirBlur(dstB, rd)
    }
    a.push(dst)
    b.push(dstB)
  }
  return { a, b }
}

/** Coarse mip levels where the DIRECTION dimension must be prefiltered too. */
const DIR_BLUR_FROM = 8

/**
 * Prefilter across the direction dimension on the coarse levels.
 *
 * Without this, a distant grazing view degenerates into smooth marbled swirls:
 * at the coarsest levels the answer depends only on (azimuth, elevation), so the
 * screen paints the 16x10 direction grid directly, and neighbouring bins differ
 * because each is the average over one finite tile. A 3x3 blur over the
 * direction neighbourhood (azimuth wraps, elevation clamps) makes the far field
 * an honestly averaged canopy instead.
 */
function dirBlur(level: Uint8Array<ArrayBuffer>, res: number): void {
  const src = level.slice()
  const plane = res * res * 4
  for (let pb = 0; pb < N_PHI; pb++) {
    for (let tb = 0; tb < N_THETA; tb++) {
      const dst = (pb * N_THETA + tb) * plane
      for (let i = 0; i < res * res; i++) {
        let wsum = 0
        let r = 0
        let g = 0
        let bl = 0
        let al = 0
        for (let dp = -1; dp <= 1; dp++) {
          const p2 = (pb + dp + N_PHI) % N_PHI
          for (let dt = -1; dt <= 1; dt++) {
            const t2 = Math.min(N_THETA - 1, Math.max(0, tb + dt))
            const s = (p2 * N_THETA + t2) * plane + i * 4
            const w = dp === 0 && dt === 0 ? 2 : 1
            wsum += w
            r += src[s]! * w
            g += src[s + 1]! * w
            bl += src[s + 2]! * w
            al += src[s + 3]! * w
          }
        }
        const d = dst + i * 4
        level[d] = Math.round(r / wsum)
        level[d + 1] = Math.round(g / wsum)
        level[d + 2] = Math.round(bl / wsum)
        level[d + 3] = Math.round(al / wsum)
      }
    }
  }
}
