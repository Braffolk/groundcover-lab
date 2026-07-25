import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Canopy-patch bake.
 *
 * A "canopy patch" is one orthographic capture of the raw GCMESH1 mesh
 * restricted to a SLAB of the plant, framed tightly on the foliage it actually
 * contains. Two families are baked per species:
 *
 *   NEAR (texture array, NEAR_RES²):
 *     - BINS azimuths x SLABS depth slabs. Slab boundaries are depth
 *       percentiles along that azimuth's view axis, so each slab holds ~1/3 of
 *       the foliage; the stored plane depth is the mean depth of its content.
 *       At runtime the three slabs are three quads perpendicular to the bake
 *       azimuth at their own depths — a 3-plane multiplane image of the plant.
 *     - CROWNS horizontal slabs: straight-down captures of the upper / lower
 *       height band, placed at the mean height of their band.
 *   FAR (texture array, FAR_RES²):
 *     - BINS azimuth composites (no slab clipping) + 1 crown composite. These
 *       are the distance LOD: one camera-facing card, i.e. a plain billboard.
 *       Far plants are small on screen, so FAR_RES is deliberately tiny.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'CPA1', u32 version, u32 nearRes, u32 farRes,
 *   u32 bins, u32 slabs, u32 crowns, u32 pad
 *   f32 rXZ, y0, y1, cx, cz, crownH, pad, pad          (metres, unit scale)
 *   ... zeros to byte 128
 *   f32[8] x SLICES  slice metadata: a0, a1, b0, b1, depth, 0, 0, 0
 *   u8 near albedo   rgba8, NEAR_SLICES layers
 *   u8 near normal   oct rg8, NEAR_SLICES layers
 *   u8 far albedo    rgba8, FAR_SLICES layers
 *   u8 far normal    oct rg8, FAR_SLICES layers
 */

export const BINS = 8
export const SLABS = 2
export const CROWNS = 2
export const NEAR_RES = 448
export const FAR_RES = 176
/** Normals are low-frequency next to coverage — half resolution is free detail
 *  spent on the albedo/alpha instead, where silhouette crispness lives. */
export const NEAR_NRM_RES = NEAR_RES / 2
export const FAR_NRM_RES = FAR_RES / 2
export const NEAR_SLICES = BINS * SLABS + CROWNS
export const FAR_SLICES = BINS + 1
export const SLICES = NEAR_SLICES + FAR_SLICES
/** Mip levels kept per array — the near atlas is never minified far. */
export const NEAR_MIPS = 5
export const NEAR_NRM_MIPS = 4
export const FAR_MIPS = 7
export const FAR_NRM_MIPS = 6

const SS = 2
const NRM_SS = 4 // normals reduce 4x from the supersampled render
const MAGIC = 0x31415043 // 'CPA1'
const VERSION = 2
const HEADER_BYTES = 128
const META_FLOATS = 8
const DILATE_PASSES = 5
const ATLAS_MAX = 3584
const UNI_STRIDE = 256

export interface PatchAtlas {
  nearRes: number
  farRes: number
  /** Horizontal support radius of the plant (m at scale 1) — culling. */
  rXZ: number
  y0: number
  y1: number
  /** Clump centre offset in the mesh frame. */
  cx: number
  cz: number
  /** Composite crown-card height (m at scale 1). */
  crownH: number
  /** SLICES x [a0, a1, b0, b1, depth, 0, 0, 0]. */
  meta: Float32Array<ArrayBuffer>
  nearAlbedo: Uint8Array<ArrayBuffer>
  nearNormal: Uint8Array<ArrayBuffer>
  farAlbedo: Uint8Array<ArrayBuffer>
  farNormal: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

// ---------------------------------------------------------------------------
// artifact (de)serialisation
// ---------------------------------------------------------------------------

const nearBytes = (res: number): number => NEAR_SLICES * res * res
const farBytes = (res: number): number => FAR_SLICES * res * res
const nearNrmBytes = (): number => NEAR_SLICES * NEAR_NRM_RES * NEAR_NRM_RES
const farNrmBytes = (): number => FAR_SLICES * FAR_NRM_RES * FAR_NRM_RES

export function unpackPatches(buf: ArrayBuffer): PatchAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const nearRes = u[2]!
  const farRes = u[3]!
  if (nearRes !== NEAR_RES || farRes !== FAR_RES) return null
  if (u[4] !== BINS || u[5] !== SLABS || u[6] !== CROWNS) return null
  const f = new Float32Array(buf, 32, 8)
  const metaBytes = SLICES * META_FLOATS * 4
  const expected =
    HEADER_BYTES + metaBytes + nearBytes(nearRes) * 4 + nearNrmBytes() * 2 + farBytes(farRes) * 4 + farNrmBytes() * 2
  if (buf.byteLength !== expected) return null
  let o = HEADER_BYTES
  const meta = new Float32Array(buf, o, SLICES * META_FLOATS)
  o += metaBytes
  const nearAlbedo = new Uint8Array(buf, o, nearBytes(nearRes) * 4)
  o += nearAlbedo.byteLength
  const nearNormal = new Uint8Array(buf, o, nearNrmBytes() * 2)
  o += nearNormal.byteLength
  const farAlbedo = new Uint8Array(buf, o, farBytes(farRes) * 4)
  o += farAlbedo.byteLength
  const farNormal = new Uint8Array(buf, o, farNrmBytes() * 2)
  return {
    nearRes,
    farRes,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    crownH: f[5]!,
    meta,
    nearAlbedo,
    nearNormal,
    farAlbedo,
    farNormal,
  }
}

function packPatches(a: PatchAtlas): ArrayBuffer {
  const metaBytes = SLICES * META_FLOATS * 4
  const buf = new ArrayBuffer(
    HEADER_BYTES +
      metaBytes +
      a.nearAlbedo.byteLength +
      a.nearNormal.byteLength +
      a.farAlbedo.byteLength +
      a.farNormal.byteLength,
  )
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.nearRes
  u[3] = a.farRes
  u[4] = BINS
  u[5] = SLABS
  u[6] = CROWNS
  const f = new Float32Array(buf, 32, 8)
  f[0] = a.rXZ
  f[1] = a.y0
  f[2] = a.y1
  f[3] = a.cx
  f[4] = a.cz
  f[5] = a.crownH
  let o = HEADER_BYTES
  new Float32Array(buf, o, SLICES * META_FLOATS).set(a.meta)
  o += metaBytes
  for (const part of [a.nearAlbedo, a.nearNormal, a.farAlbedo, a.farNormal]) {
    new Uint8Array(buf, o, part.byteLength).set(part)
    o += part.byteLength
  }
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the patch atlas for a species.
 * Every result is magic+size validated: the dev server answers missing
 * /mesh/baked files with the SPA index.html at status 200, which would
 * otherwise poison both stores — a poisoned entry is rebaked and repaired.
 */
export async function loadPatchAtlas(ctx: BakeCtx, speciesId: string): Promise<PatchAtlas> {
  const key = `patches-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpecies(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackPatches(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackPatches(buf)
    if (!atlas) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return atlas
}

/** Overwrite a poisoned OPFS cache entry (mirrors src/bake/cache.ts naming). */
async function opfsRepair(fullKey: string, data: ArrayBuffer): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle('bake-cache', { create: true })
    const handle = await dir.getFileHandle(`${fullKey}.bin`, { create: true })
    const writable = await handle.createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    /* best effort — cache misses just rebake */
  }
}

// ---------------------------------------------------------------------------
// geometry analysis: slab boundaries, tight framing, plane depths
// ---------------------------------------------------------------------------

interface SliceJob {
  family: 'near' | 'far'
  /** Layer inside its texture array. */
  layer: number
  /** Index into the flat slice metadata table. */
  meta: number
  right: readonly [number, number, number]
  vAxis: readonly [number, number, number]
  fwd: readonly [number, number, number]
  isTop: boolean
  rect: [number, number, number, number]
  lo: number
  hi: number
  depth: number
}

interface Analysis {
  cx: number
  cz: number
  y0: number
  y1: number
  rXZ: number
  crownH: number
  jobs: SliceJob[]
}

const HIST = 2048
/** Depth quantiles that split the foliage into SLABS equal-coverage slabs. */
const SLAB_QUANTILES: readonly number[] = Array.from({ length: SLABS - 1 }, (_, i) => (i + 1) / SLABS)

/** Value at each requested quantile of `vals`, via a fixed histogram. */
function quantiles(vals: Float32Array, count: number, qs: readonly number[]): number[] {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < count; i++) {
    const v = vals[i]!
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const span = Math.max(hi - lo, 1e-6)
  const hist = new Uint32Array(HIST)
  for (let i = 0; i < count; i++) {
    const b = Math.min(HIST - 1, Math.max(0, ((vals[i]! - lo) / span) * HIST) | 0)
    hist[b]!++
  }
  const out: number[] = []
  let acc = 0
  let b = 0
  for (const q of qs) {
    const want = q * count
    while (b < HIST && acc + hist[b]! < want) {
      acc += hist[b]!
      b++
    }
    out.push(lo + ((b + 0.5) / HIST) * span)
  }
  return out
}

function analyze(mesh: GcMesh): Analysis {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const n = hdr.vertexCount
  const verts = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535

  // Positions relative to the clump centre in xz, absolute in y.
  const px = new Float32Array(n)
  const py = new Float32Array(n)
  const pz = new Float32Array(n)
  let r2 = 0
  for (let i = 0; i < n; i++) {
    const x = bx0 + verts[i * 8]! * sx - cx
    const y = by0 + verts[i * 8 + 1]! * sy
    const z = bz0 + verts[i * 8 + 2]! * sz - cz
    px[i] = x
    py[i] = y
    pz[i] = z
    const d = x * x + z * z
    if (d > r2) r2 = d
  }
  const rXZ = Math.sqrt(r2) * 1.02 + 1e-3
  const y0 = Math.min(0, by0)
  const y1 = by1
  const margin = 0.025 * Math.max(2 * rXZ, y1 - y0)

  const jobs: SliceJob[] = []
  const scratch = new Float32Array(n)

  // --- side slices: BINS azimuths x SLABS depth slabs (+ far composite) -----
  for (let bin = 0; bin < BINS; bin++) {
    const a = (bin * 2 * Math.PI) / BINS
    const sa = Math.sin(a)
    const ca = Math.cos(a)
    const fwd = [sa, 0, ca] as const // toward the capture camera
    const right = [ca, 0, -sa] as const

    for (let i = 0; i < n; i++) scratch[i] = px[i]! * sa + pz[i]! * ca
    const cuts = quantiles(scratch, n, SLAB_QUANTILES)
    let tmin = Infinity
    let tmax = -Infinity
    for (let i = 0; i < n; i++) {
      const t = scratch[i]!
      if (t < tmin) tmin = t
      if (t > tmax) tmax = t
    }
    // Slab 0 is the FRONT (largest depth toward the camera); the cuts are
    // coverage percentiles, so every slab holds the same amount of foliage.
    const bounds: [number, number][] = []
    for (let slab = 0; slab < SLABS; slab++) {
      const hi = slab === 0 ? tmax + 1e-3 : cuts[SLABS - 1 - slab]!
      const lo = slab === SLABS - 1 ? tmin - 1e-3 : cuts[SLABS - 2 - slab]!
      bounds.push([lo, hi])
    }

    // Per-slab tight framing + mean depth, plus the union for the composite.
    let cu0 = Infinity
    let cu1 = -Infinity
    let cv0 = Infinity
    let cv1 = -Infinity
    for (let slab = 0; slab < SLABS; slab++) {
      const [lo, hi] = bounds[slab]!
      let u0 = Infinity
      let u1 = -Infinity
      let v0 = Infinity
      let v1 = -Infinity
      let tSum = 0
      let cnt = 0
      for (let i = 0; i < n; i++) {
        const t = scratch[i]!
        if (t < lo || t > hi) continue
        const u = px[i]! * ca - pz[i]! * sa
        const v = py[i]!
        if (u < u0) u0 = u
        if (u > u1) u1 = u
        if (v < v0) v0 = v
        if (v > v1) v1 = v
        tSum += t
        cnt++
      }
      if (cnt === 0) {
        u0 = -rXZ
        u1 = rXZ
        v0 = y0
        v1 = y1
        tSum = (lo + hi) * 0.5
        cnt = 1
      }
      cu0 = Math.min(cu0, u0)
      cu1 = Math.max(cu1, u1)
      cv0 = Math.min(cv0, v0)
      cv1 = Math.max(cv1, v1)
      jobs.push({
        family: 'near',
        layer: bin * SLABS + slab,
        meta: bin * SLABS + slab,
        right,
        vAxis: [0, 1, 0],
        fwd,
        isTop: false,
        rect: [u0 - margin, u1 + margin, Math.max(y0, v0 - margin), v1 + margin],
        lo,
        hi,
        depth: tSum / cnt,
      })
    }
    jobs.push({
      family: 'far',
      layer: bin,
      meta: NEAR_SLICES + bin,
      right,
      vAxis: [0, 1, 0],
      fwd,
      isTop: false,
      rect: [cu0 - margin, cu1 + margin, Math.max(y0, cv0 - margin), cv1 + margin],
      lo: -1e30,
      hi: 1e30,
      depth: 0,
    })
  }

  // --- crown slices: height slabs seen straight down ------------------------
  // Capture basis matches the classic top view: U = +X, V = -Z.
  const rightTop = [1, 0, 0] as const
  const vAxisTop = [0, 0, -1] as const
  // Geometric height split, NOT a vertex quantile: grass meshes put most of
  // their triangles in the panicles, so a count quantile lands both crowns at
  // the top of the plant and there is no vertical parallax left between them.
  const hCut = y0 + 0.6 * (y1 - y0)
  const crownBounds: [number, number][] = [
    [hCut, y1 + 1e-3],
    [y0 - 1e-3, hCut],
  ]
  let allSum = 0
  for (let i = 0; i < n; i++) allSum += py[i]!
  const crownH = allSum / n

  const topRect = (lo: number, hi: number): { rect: [number, number, number, number]; h: number } => {
    let u0 = Infinity
    let u1 = -Infinity
    let v0 = Infinity
    let v1 = -Infinity
    let hSum = 0
    let cnt = 0
    for (let i = 0; i < n; i++) {
      const y = py[i]!
      if (y < lo || y > hi) continue
      const u = px[i]!
      const v = -pz[i]!
      if (u < u0) u0 = u
      if (u > u1) u1 = u
      if (v < v0) v0 = v
      if (v > v1) v1 = v
      hSum += y
      cnt++
    }
    if (cnt === 0) {
      u0 = -rXZ
      u1 = rXZ
      v0 = -rXZ
      v1 = rXZ
      hSum = (lo + hi) * 0.5
      cnt = 1
    }
    return { rect: [u0 - margin, u1 + margin, v0 - margin, v1 + margin], h: hSum / cnt }
  }

  for (let c = 0; c < CROWNS; c++) {
    const [lo, hi] = crownBounds[c]!
    const { rect, h } = topRect(lo, hi)
    jobs.push({
      family: 'near',
      layer: BINS * SLABS + c,
      meta: BINS * SLABS + c,
      right: rightTop,
      vAxis: vAxisTop,
      fwd: [0, 1, 0],
      isTop: true,
      rect,
      lo,
      hi,
      depth: h,
    })
  }
  {
    const { rect } = topRect(y0 - 1e-3, y1 + 1e-3)
    jobs.push({
      family: 'far',
      layer: BINS,
      meta: NEAR_SLICES + BINS,
      right: rightTop,
      vAxis: vAxisTop,
      fwd: [0, 1, 0],
      isTop: true,
      rect,
      lo: -1e30,
      hi: 1e30,
      depth: crownH,
    })
  }

  return { cx, cz, y0, y1, rXZ, crownH, jobs }
}

// ---------------------------------------------------------------------------
// GPU capture
// ---------------------------------------------------------------------------

async function bakeSpecies(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const info = analyze(mesh)
  const hdr = mesh.header

  const verts = mesh.vertices
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 128 },
      },
    ],
  })
  const module = ctx.shaders.module(bakeShaderSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/bake-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-pl`, bindGroupLayouts: [bgl] }),
    vertex: {
      module,
      entryPoint: 'vs',
      buffers: [
        {
          arrayStride: 16,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'uint16x4' },
            { shaderLocation: 1, offset: 8, format: 'uint16x4' },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const out = {
    nearAlbedo: new Uint8Array(nearBytes(NEAR_RES) * 4),
    nearNormal: new Uint8Array(nearNrmBytes() * 2),
    farAlbedo: new Uint8Array(farBytes(FAR_RES) * 4),
    farNormal: new Uint8Array(farNrmBytes() * 2),
  }

  for (const family of ['near', 'far'] as const) {
    const res = family === 'near' ? NEAR_RES : FAR_RES
    const nrmRes = family === 'near' ? NEAR_NRM_RES : FAR_NRM_RES
    const tile = res * SS
    const cols = Math.max(1, Math.floor(ATLAS_MAX / tile))
    const perAtlas = cols * cols
    const jobs = info.jobs.filter((j) => j.family === family)
    for (let start = 0; start < jobs.length; start += perAtlas) {
      const chunk = jobs.slice(start, start + perAtlas)
      const rows = Math.ceil(chunk.length / cols)
      await renderChunk(ctx, {
        chunk,
        info,
        hdr,
        tile,
        res,
        nrmRes,
        cols,
        rows,
        pipeline,
        bgl,
        vbuf,
        ibuf,
        indexCount: indices.length,
        albedoOut: family === 'near' ? out.nearAlbedo : out.farAlbedo,
        normalOut: family === 'near' ? out.nearNormal : out.farNormal,
      })
    }
  }

  for (const r of [vbuf, ibuf]) r.destroy()

  const meta = new Float32Array(SLICES * META_FLOATS)
  for (const job of info.jobs) {
    const o = job.meta * META_FLOATS
    meta[o] = job.rect[0]
    meta[o + 1] = job.rect[1]
    meta[o + 2] = job.rect[2]
    meta[o + 3] = job.rect[3]
    meta[o + 4] = job.depth
  }

  return packPatches({
    nearRes: NEAR_RES,
    farRes: FAR_RES,
    rXZ: info.rXZ,
    y0: info.y0,
    y1: info.y1,
    cx: info.cx,
    cz: info.cz,
    crownH: info.crownH,
    meta,
    ...out,
  })
}

interface ChunkArgs {
  chunk: SliceJob[]
  info: Analysis
  hdr: GcMesh['header']
  tile: number
  res: number
  nrmRes: number
  cols: number
  rows: number
  pipeline: GPURenderPipeline
  bgl: GPUBindGroupLayout
  vbuf: GPUBuffer
  ibuf: GPUBuffer
  indexCount: number
  albedoOut: Uint8Array
  normalOut: Uint8Array
}

async function renderChunk(ctx: BakeCtx, a: ChunkArgs): Promise<void> {
  const { device } = ctx
  const atlasW = a.cols * a.tile
  const atlasH = a.rows * a.tile

  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [atlasW, atlasH],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('bake-albedo')
  const nrmTex = mkTarget('bake-normal')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [atlasW, atlasH],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  const uni = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-uni`,
      size: UNI_STRIDE * a.chunk.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((UNI_STRIDE / 4) * a.chunk.length)
  a.chunk.forEach((job, k) => {
    const o = (k * UNI_STRIDE) / 4
    scratch.set(job.right, o)
    scratch.set(job.vAxis, o + 4)
    scratch.set(job.fwd, o + 8)
    scratch[o + 11] = job.isTop ? 1 : 0
    scratch.set(job.rect, o + 12)
    scratch.set([job.lo, job.hi, a.info.rXZ, 0], o + 16)
    scratch.set([a.info.cx, a.info.cz, a.info.y0, a.info.y1], o + 20)
    scratch.set([a.hdr.boundsMin[0], a.hdr.boundsMin[1], a.hdr.boundsMin[2], 0], o + 24)
    scratch.set(
      [
        a.hdr.boundsMax[0] - a.hdr.boundsMin[0],
        a.hdr.boundsMax[1] - a.hdr.boundsMin[1],
        a.hdr.boundsMax[2] - a.hdr.boundsMin[2],
        0,
      ],
      o + 28,
    )
  })
  device.queue.writeBuffer(uni, 0, scratch)
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: a.bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 128 } }],
  })

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/bake-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  })
  pass.setPipeline(a.pipeline)
  pass.setVertexBuffer(0, a.vbuf)
  pass.setIndexBuffer(a.ibuf, 'uint32')
  a.chunk.forEach((_, k) => {
    const col = k % a.cols
    const row = Math.floor(k / a.cols)
    pass.setViewport(col * a.tile, row * a.tile, a.tile, a.tile, 0, 1)
    pass.setScissorRect(col * a.tile, row * a.tile, a.tile, a.tile)
    pass.setBindGroup(0, bg, [k * UNI_STRIDE])
    pass.drawIndexed(a.indexCount)
  })
  pass.end()

  const bpr = Math.ceil((atlasW * 4) / 256) * 256
  const rbSize = bpr * atlasH
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbNrm = mkReadback('bake-rb-normal')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: atlasH }, [
    atlasW,
    atlasH,
  ])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: atlasH }, [
    atlasW,
    atlasH,
  ])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange())
  const bigNrm = new Uint8Array(rbNrm.getMappedRange())

  const work = makeReduceScratch(a.res, a.nrmRes)
  a.chunk.forEach((job, k) => {
    const col = k % a.cols
    const row = Math.floor(k / a.cols)
    reduceSlice(bigAlb, bigNrm, bpr, col * a.tile, row * a.tile, a.res, a.nrmRes, work)
    dilateAlbedo(work, a.res)
    dilateNormal(work, a.nrmRes)
    writeLayer(work, a.res, a.nrmRes, job.layer, a.albedoOut, a.normalOut)
  })

  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()
}

// ---------------------------------------------------------------------------
// CPU reduction: coverage-weighted 2x downsample, dilation, oct encode
// ---------------------------------------------------------------------------

interface ReduceScratch {
  albedo: Uint8Array
  filledA: Uint8Array
  nextA: Uint8Array
  normal: Float32Array
  filledN: Uint8Array
  nextN: Uint8Array
}

function makeReduceScratch(res: number, nrmRes: number): ReduceScratch {
  return {
    albedo: new Uint8Array(res * res * 4),
    filledA: new Uint8Array(res * res),
    nextA: new Uint8Array(res * res),
    normal: new Float32Array(nrmRes * nrmRes * 3),
    filledN: new Uint8Array(nrmRes * nrmRes),
    nextN: new Uint8Array(nrmRes * nrmRes),
  }
}

/**
 * Coverage-weighted box reduction of one atlas tile: albedo+coverage at SS:1,
 * normals at NRM_SS:1 (they are stored at half the albedo resolution).
 */
function reduceSlice(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  bpr: number,
  tx: number,
  ty: number,
  res: number,
  nrmRes: number,
  w: ReduceScratch,
): void {
  w.albedo.fill(0)
  w.filledA.fill(0)
  for (let y = 0; y < res; y++) {
    for (let x = 0; x < res; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = (ty + y * SS + j) * bpr + (tx + x * SS + i) * 4
          const av = bigAlb[s + 3]!
          if (av === 0) continue
          aSum += av
          r += bigAlb[s]! * av
          g += bigAlb[s + 1]! * av
          b += bigAlb[s + 2]! * av
        }
      }
      if (aSum === 0) continue
      const d = (y * res + x) * 4
      w.albedo[d] = Math.round(r / aSum)
      w.albedo[d + 1] = Math.round(g / aSum)
      w.albedo[d + 2] = Math.round(b / aSum)
      w.albedo[d + 3] = Math.round(aSum / (SS * SS))
      w.filledA[y * res + x] = 1
    }
  }

  w.normal.fill(0)
  w.filledN.fill(0)
  for (let y = 0; y < nrmRes; y++) {
    for (let x = 0; x < nrmRes; x++) {
      let aSum = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < NRM_SS; j++) {
        for (let i = 0; i < NRM_SS; i++) {
          const s = (ty + y * NRM_SS + j) * bpr + (tx + x * NRM_SS + i) * 4
          const av = bigAlb[s + 3]!
          if (av === 0) continue
          aSum += av
          nx += (bigNrm[s]! / 127.5 - 1) * av
          ny += (bigNrm[s + 1]! / 127.5 - 1) * av
          nz += (bigNrm[s + 2]! / 127.5 - 1) * av
        }
      }
      if (aSum === 0) continue
      const dn = (y * nrmRes + x) * 3
      w.normal[dn] = nx
      w.normal[dn + 1] = ny
      w.normal[dn + 2] = nz
      w.filledN[y * nrmRes + x] = 1
    }
  }
}

/** Flood colour into empty texels (alpha stays 0) so filtering and mip
 *  generation never blend toward background black. */
function dilateAlbedo(w: ReduceScratch, res: number): void {
  for (let p = 0; p < DILATE_PASSES; p++) {
    w.nextA.set(w.filledA)
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const idx = y * res + x
        if (w.filledA[idx]! !== 0) continue
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= res) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < 0 || xx >= res) continue
            const nIdx = yy * res + xx
            if (w.filledA[nIdx]! === 0) continue
            count++
            const s4 = nIdx * 4
            r += w.albedo[s4]!
            g += w.albedo[s4 + 1]!
            b += w.albedo[s4 + 2]!
          }
        }
        if (count === 0) continue
        const d4 = idx * 4
        w.albedo[d4] = Math.round(r / count)
        w.albedo[d4 + 1] = Math.round(g / count)
        w.albedo[d4 + 2] = Math.round(b / count)
        w.nextA[idx] = 1
      }
    }
    w.filledA.set(w.nextA)
  }
}

/** Same flood for the (half-resolution) normal sheet. */
function dilateNormal(w: ReduceScratch, res: number): void {
  for (let p = 0; p < DILATE_PASSES; p++) {
    w.nextN.set(w.filledN)
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const idx = y * res + x
        if (w.filledN[idx]! !== 0) continue
        let count = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= res) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < 0 || xx >= res) continue
            const nIdx = yy * res + xx
            if (w.filledN[nIdx]! === 0) continue
            count++
            const s3 = nIdx * 3
            nx += w.normal[s3]!
            ny += w.normal[s3 + 1]!
            nz += w.normal[s3 + 2]!
          }
        }
        if (count === 0) continue
        const d3 = idx * 3
        w.normal[d3] = nx / count
        w.normal[d3 + 1] = ny / count
        w.normal[d3 + 2] = nz / count
        w.nextN[idx] = 1
      }
    }
    w.filledN.set(w.nextN)
  }
}

function writeLayer(
  w: ReduceScratch,
  res: number,
  nrmRes: number,
  layer: number,
  albedoOut: Uint8Array,
  normalOut: Uint8Array,
): void {
  albedoOut.set(w.albedo.subarray(0, res * res * 4), layer * res * res * 4)
  const base = layer * nrmRes * nrmRes * 2
  for (let i = 0; i < nrmRes * nrmRes; i++) {
    const s3 = i * 3
    const [u, v] = octEncode(w.normal[s3]!, w.normal[s3 + 1]!, w.normal[s3 + 2]!)
    normalOut[base + i * 2] = u
    normalOut[base + i * 2 + 1] = v
  }
}

/** Octahedral encode, y-primary — exact inverse of the decode in gcmesh.ts. */
function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  let u: number
  let v: number
  if (s < 1e-6) {
    u = 0
    v = 0
  } else {
    u = x / s
    v = z / s
    if (y < 0) {
      const fu = (1 - Math.abs(v)) * (x >= 0 ? 1 : -1)
      const fv = (1 - Math.abs(u)) * (z >= 0 ? 1 : -1)
      u = fu
      v = fv
    }
  }
  return [Math.round((u * 0.5 + 0.5) * 255), Math.round((v * 0.5 + 0.5) * 255)]
}
