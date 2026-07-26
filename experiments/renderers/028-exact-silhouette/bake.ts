import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Part bake — the precomputation behind "exact silhouette, cheap interior".
 *
 * The source mesh is PARTITIONED (every triangle belongs to exactly one part)
 * into BANDS equal-height bands x SECTORS azimuth sectors around the clump axis.
 * Each part is baked from PART_AZ azimuths using ITS OWN tight box, so every
 * texel covers a part instead of empty margin — that is where the resolution to
 * beat a whole-plant card comes from at the same VRAM.
 *
 * Two extra tile sets ride along:
 *   - one straight-down tile per BAND (the horizontal fill that takes over when
 *     the camera rises — BANDS real layers seen from above), and
 *   - a whole-plant far set (FAR_AZ side views + 1 top) — the distance LOD,
 *     which is exactly a billboard card pair.
 *
 * Layer order (must match parts.wgsl):
 *   near: part p * PART_AZ + k  (p = band * SECTORS + sector),
 *         then PARTS * PART_AZ + band
 *   far:  azimuth k, then FAR_AZ
 *
 * Artifact layout (little-endian):
 *   u32 magic 'PAS1', version, nearTile, farTile, nearLayers, farLayers,
 *       bands, sectors, partAz, farAz
 *   f32 cx, cz, y0, y1, rXZ                         (byte 40)
 *   f32 parts[PARTS][6]  cx,cy,cz,ex,ey,ez          (byte 64, inflated boxes)
 *   f32 bands[BANDS][6]  cx,cz,ex,ez,topY,_         (byte 384)
 *   ...zeros to byte 512
 *   u8  nearAlbedo[nearLayers][nearTile^2][4]   rgba8, a = coverage
 *   u8  nearNormal[nearLayers][nearTile^2][2]   oct rg8, mesh frame
 *   u8  farAlbedo[farLayers][farTile^2][4]
 *   u8  farNormal[farLayers][farTile^2][2]
 */

export const BANDS = 3
export const SECTORS = 4
export const PARTS = BANDS * SECTORS
/**
 * Azimuths per part. Deliberately coarse: a part's POSITION carries the 3-D
 * read, so texel density buys more than angular density does. The per-sector
 * stagger (bakeSpeciesParts) spreads the four sectors' switch angles 22.5deg
 * apart, so an orbit re-quantizes an eighth of the plant at a time instead of
 * flipping the whole card the way an 8-view billboard does.
 */
export const PART_AZ = 4
export const NEAR_TILE = 192
export const NEAR_LAYERS = PARTS * PART_AZ + BANDS
export const FAR_AZ = 8
export const FAR_TILE = 128
export const FAR_LAYERS = FAR_AZ + 1
/** Mip levels kept per atlas (near tiles are never minified past ~6px). */
export const NEAR_MIPS = 6
export const FAR_MIPS = 8

const SS = 2 // bake supersample
const MAGIC = 0x31534150 // 'PAS1'
const VERSION = 3
const HEADER_BYTES = 512
const DILATE_PASSES = 4
/** Box inflation so tile content never touches the clamped tile edge. */
const BOX_MARGIN = 1.03
const BOX_EPS = 0.004

export interface PartAtlas {
  nearTile: number
  farTile: number
  cx: number
  cz: number
  y0: number
  y1: number
  /** Horizontal support radius of the whole plant (m at scale 1). */
  rXZ: number
  /** PARTS * 6: center xyz + half extents xyz, mesh frame, already inflated. */
  parts: Float32Array
  /** BANDS * 6: center xz, half extents xz, horizontal-card height, pad. */
  bands: Float32Array
  nearAlbedo: Uint8Array<ArrayBuffer>
  nearNormal: Uint8Array<ArrayBuffer>
  farAlbedo: Uint8Array<ArrayBuffer>
  farNormal: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

// ---------------------------------------------------------------------------
// load / cache
// ---------------------------------------------------------------------------

/**
 * Load (OPFS cache / committed file) or bake this species' part atlas. The dev
 * server answers missing /mesh/baked files with the SPA index.html at status
 * 200, so every result is magic-validated and a poisoned cache entry is rebaked
 * and repaired in place (same shim as 001).
 */
export async function loadSpeciesParts(ctx: BakeCtx, speciesId: string): Promise<PartAtlas> {
  const key = `parts-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesParts(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackParts(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackParts(buf)
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

export function unpackParts(buf: ArrayBuffer): PartAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 10)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const nearTile = u[2]!
  const farTile = u[3]!
  const nearLayers = u[4]!
  const farLayers = u[5]!
  if (
    nearTile !== NEAR_TILE ||
    farTile !== FAR_TILE ||
    nearLayers !== NEAR_LAYERS ||
    farLayers !== FAR_LAYERS ||
    u[6] !== BANDS ||
    u[7] !== SECTORS ||
    u[8] !== PART_AZ ||
    u[9] !== FAR_AZ
  ) {
    return null
  }
  const nearTexels = nearTile * nearTile * nearLayers
  const farTexels = farTile * farTile * farLayers
  const expected = HEADER_BYTES + nearTexels * 6 + farTexels * 6
  if (buf.byteLength !== expected) return null
  const f = new Float32Array(buf, 40, 5)
  let o = HEADER_BYTES
  const nearAlbedo = new Uint8Array(buf, o, nearTexels * 4)
  o += nearTexels * 4
  const nearNormal = new Uint8Array(buf, o, nearTexels * 2)
  o += nearTexels * 2
  const farAlbedo = new Uint8Array(buf, o, farTexels * 4)
  o += farTexels * 4
  const farNormal = new Uint8Array(buf, o, farTexels * 2)
  return {
    nearTile,
    farTile,
    cx: f[0]!,
    cz: f[1]!,
    y0: f[2]!,
    y1: f[3]!,
    rXZ: f[4]!,
    parts: new Float32Array(buf, 64, PARTS * 6).slice(),
    bands: new Float32Array(buf, 384, BANDS * 6).slice(),
    nearAlbedo,
    nearNormal,
    farAlbedo,
    farNormal,
  }
}

function packParts(a: PartAtlas): ArrayBuffer {
  const nearTexels = a.nearTile * a.nearTile * NEAR_LAYERS
  const farTexels = a.farTile * a.farTile * FAR_LAYERS
  const buf = new ArrayBuffer(HEADER_BYTES + nearTexels * 6 + farTexels * 6)
  const u = new Uint32Array(buf, 0, 10)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.nearTile
  u[3] = a.farTile
  u[4] = NEAR_LAYERS
  u[5] = FAR_LAYERS
  u[6] = BANDS
  u[7] = SECTORS
  u[8] = PART_AZ
  u[9] = FAR_AZ
  new Float32Array(buf, 40, 5).set([a.cx, a.cz, a.y0, a.y1, a.rXZ])
  new Float32Array(buf, 64, PARTS * 6).set(a.parts)
  new Float32Array(buf, 384, BANDS * 6).set(a.bands)
  let o = HEADER_BYTES
  new Uint8Array(buf, o, a.nearAlbedo.byteLength).set(a.nearAlbedo)
  o += a.nearAlbedo.byteLength
  new Uint8Array(buf, o, a.nearNormal.byteLength).set(a.nearNormal)
  o += a.nearNormal.byteLength
  new Uint8Array(buf, o, a.farAlbedo.byteLength).set(a.farAlbedo)
  o += a.farAlbedo.byteLength
  new Uint8Array(buf, o, a.farNormal.byteLength).set(a.farNormal)
  return buf
}

// ---------------------------------------------------------------------------
// partition
// ---------------------------------------------------------------------------

interface Range {
  first: number
  count: number
}

interface Partition {
  cx: number
  cz: number
  y0: number
  y1: number
  rXZ: number
  /** Index buffer re-ordered so every part occupies one contiguous range. */
  indices: Uint32Array<ArrayBuffer>
  /** Per part, in layer order p = band * SECTORS + sector. */
  ranges: Range[]
  /** PARTS * 6 — inflated boxes. */
  parts: Float32Array
  /** BANDS * 6. */
  bands: Float32Array
  /** Per band y range (bake only: the top tile's depth range). */
  bandY: { y0: number; y1: number }[]
}

/** Sector of an in-plane offset, matching parts.wgsl: round(atan2(x,z)/90deg). */
function sectorOf(dx: number, dz: number): number {
  const step = (2 * Math.PI) / SECTORS
  const a = Math.atan2(dx, dz)
  return ((Math.round(a / step) % SECTORS) + SECTORS) % SECTORS
}

function partitionMesh(mesh: GcMesh): Partition {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535
  const verts = mesh.vertices
  const tris = mesh.triangles
  const nTri = hdr.triangleCount

  // Exact horizontal support radius (bounds corners overestimate it a lot on
  // the wide community tiles).
  let r2 = 0
  for (let i = 0; i < hdr.vertexCount; i++) {
    const dx = bx0 + verts[i * 8]! * sx - cx
    const dz = bz0 + verts[i * 8 + 2]! * sz - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  const rXZ = Math.sqrt(r2) * 1.02 + 1e-3
  const y0 = Math.min(0, by0)
  const y1 = by1

  // --- pass 1: centroid height + sector per triangle ------------------------
  const triY = new Float32Array(nTri)
  const triSector = new Uint8Array(nTri)
  for (let t = 0; t < nTri; t++) {
    const i0 = tris[t * 4]! * 8
    const i1 = tris[t * 4 + 1]! * 8
    const i2 = tris[t * 4 + 2]! * 8
    const gy = by0 + ((verts[i0 + 1]! + verts[i1 + 1]! + verts[i2 + 1]!) / 3) * sy
    const gx = bx0 + ((verts[i0]! + verts[i1]! + verts[i2]!) / 3) * sx - cx
    const gz = bz0 + ((verts[i0 + 2]! + verts[i1 + 2]! + verts[i2 + 2]!) / 3) * sz - cz
    triY[t] = gy
    triSector[t] = sectorOf(gx, gz)
  }

  // Bands split the plant into EQUAL HEIGHT slabs. Triangle-count quantiles
  // were tried first and are a trap: calamagrostis carries half its triangles
  // in the top 0.2 m of panicles, so the quantile split gave one band spanning
  // 83% of the height — a part box 0.4 x 0.98 m squeezed into a square tile,
  // i.e. 3.8 mm/texel vertically. Equal height keeps every part box nearly
  // square, which is what keeps texel density isotropic and sharp.
  const splits: number[] = []
  for (let b = 1; b < BANDS; b++) splits.push(y0 + (b / BANDS) * (y1 - y0))

  const bandOf = (y: number): number => {
    let b = 0
    while (b < BANDS - 1 && y >= splits[b]!) b++
    return b
  }

  // --- pass 2: counts -> prefix sums ---------------------------------------
  const counts = new Uint32Array(PARTS)
  const partOf = new Uint8Array(nTri)
  for (let t = 0; t < nTri; t++) {
    const p = bandOf(triY[t]!) * SECTORS + triSector[t]!
    partOf[t] = p
    counts[p]! += 1
  }
  const ranges: Range[] = []
  let off = 0
  for (let p = 0; p < PARTS; p++) {
    ranges.push({ first: off * 3, count: counts[p]! * 3 })
    off += counts[p]!
  }

  // --- pass 3: scatter indices + per-part vertex AABBs ----------------------
  const indices = new Uint32Array(nTri * 3)
  const cursor = new Uint32Array(PARTS)
  const lo = new Float32Array(PARTS * 3).fill(Infinity)
  const hi = new Float32Array(PARTS * 3).fill(-Infinity)
  const bandYSum = new Float64Array(BANDS)
  const bandYCount = new Float64Array(BANDS)
  // One pass over up to 6.5 M triangles: no per-triangle allocation in here (an
  // `of [a, b, c]` loop costs a throwaway array per triangle and is the single
  // slowest thing in the bake).
  const growBox = (l: number, vi: number): void => {
    const o = vi * 8
    const px = bx0 + verts[o]! * sx
    const py = by0 + verts[o + 1]! * sy
    const pz = bz0 + verts[o + 2]! * sz
    if (px < lo[l]!) lo[l] = px
    if (py < lo[l + 1]!) lo[l + 1] = py
    if (pz < lo[l + 2]!) lo[l + 2] = pz
    if (px > hi[l]!) hi[l] = px
    if (py > hi[l + 1]!) hi[l + 1] = py
    if (pz > hi[l + 2]!) hi[l + 2] = pz
  }
  for (let t = 0; t < nTri; t++) {
    const p = partOf[t]!
    const slot = ranges[p]!.first / 3 + cursor[p]!
    cursor[p]! += 1
    const a = tris[t * 4]!
    const b = tris[t * 4 + 1]!
    const c = tris[t * 4 + 2]!
    indices[slot * 3] = a
    indices[slot * 3 + 1] = b
    indices[slot * 3 + 2] = c
    const l = p * 3
    growBox(l, a)
    growBox(l, b)
    growBox(l, c)
    const band = (p / SECTORS) | 0
    bandYSum[band]! += triY[t]!
    bandYCount[band]! += 1
  }

  // --- boxes ---------------------------------------------------------------
  const parts = new Float32Array(PARTS * 6)
  for (let p = 0; p < PARTS; p++) {
    const l = p * 3
    if (counts[p]! === 0) {
      // Empty part (degenerate mesh corner): a tiny box at the axis. Its tile
      // bakes fully transparent, so every fragment of its card discards.
      parts.set([0, 0.01, 0, 0.01, 0.01, 0.01], p * 6)
      continue
    }
    for (let c = 0; c < 3; c++) {
      const center = (lo[l + c]! + hi[l + c]!) / 2
      const half = Math.max((hi[l + c]! - lo[l + c]!) / 2, 1e-4)
      parts[p * 6 + c] = center - (c === 0 ? cx : c === 2 ? cz : 0)
      parts[p * 6 + 3 + c] = half * BOX_MARGIN + BOX_EPS
    }
  }

  const bands = new Float32Array(BANDS * 6)
  const bandY: { y0: number; y1: number }[] = []
  for (let b = 0; b < BANDS; b++) {
    let x0 = Infinity
    let x1 = -Infinity
    let z0 = Infinity
    let z1 = -Infinity
    let by0b = Infinity
    let by1b = -Infinity
    for (let s = 0; s < SECTORS; s++) {
      const p = b * SECTORS + s
      if (counts[p]! === 0) continue
      const cxp = parts[p * 6]!
      const cyp = parts[p * 6 + 1]!
      const czp = parts[p * 6 + 2]!
      const exp = parts[p * 6 + 3]!
      const eyp = parts[p * 6 + 4]!
      const ezp = parts[p * 6 + 5]!
      x0 = Math.min(x0, cxp - exp)
      x1 = Math.max(x1, cxp + exp)
      z0 = Math.min(z0, czp - ezp)
      z1 = Math.max(z1, czp + ezp)
      by0b = Math.min(by0b, cyp - eyp)
      by1b = Math.max(by1b, cyp + eyp)
    }
    if (!Number.isFinite(x0)) {
      x0 = -0.01
      x1 = 0.01
      z0 = -0.01
      z1 = 0.01
      by0b = 0
      by1b = 0.02
    }
    const meanY = bandYCount[b]! > 0 ? bandYSum[b]! / bandYCount[b]! : (by0b + by1b) / 2
    bands.set([(x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (z1 - z0) / 2, meanY, 0], b * 6)
    bandY.push({ y0: by0b, y1: by1b })
  }

  return { cx, cz, y0, y1, rXZ, indices, ranges, parts, bands, bandY }
}

// ---------------------------------------------------------------------------
// GPU tile rendering
// ---------------------------------------------------------------------------

interface TileJob {
  right: [number, number, number]
  up: [number, number, number]
  fwd: [number, number, number]
  center: [number, number, number]
  /** Half extents along right / up / fwd. */
  ext: [number, number, number]
  first: number
  count: number
}

function sideJob(
  azimuth: number,
  center: [number, number, number],
  ext3: [number, number, number],
  range: Range,
): TileJob {
  const ca = Math.cos(azimuth)
  const sa = Math.sin(azimuth)
  const eu = Math.abs(ca) * ext3[0] + Math.abs(sa) * ext3[2]
  const ew = Math.abs(sa) * ext3[0] + Math.abs(ca) * ext3[2]
  return {
    right: [ca, 0, -sa],
    up: [0, 1, 0],
    fwd: [sa, 0, ca],
    center,
    ext: [eu, ext3[1], ew],
    first: range.first,
    count: range.count,
  }
}

function topJob(center: [number, number, number], ext3: [number, number, number], range: Range): TileJob {
  return {
    right: [1, 0, 0],
    up: [0, 0, -1],
    fwd: [0, 1, 0],
    center,
    ext: [ext3[0], ext3[2], ext3[1]],
    first: range.first,
    count: range.count,
  }
}

async function bakeSpeciesParts(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const part = partitionMesh(mesh)
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax

  // --- tile jobs -----------------------------------------------------------
  const allRange: Range = { first: 0, count: part.indices.length }
  const nearJobs: TileJob[] = []
  for (let p = 0; p < PARTS; p++) {
    const center: [number, number, number] = [part.parts[p * 6]!, part.parts[p * 6 + 1]!, part.parts[p * 6 + 2]!]
    const ext3: [number, number, number] = [
      part.parts[p * 6 + 3]!,
      part.parts[p * 6 + 4]!,
      part.parts[p * 6 + 5]!,
    ]
    const sector = p % SECTORS
    const stagger = (sector * 2 * Math.PI) / (PART_AZ * SECTORS)
    for (let k = 0; k < PART_AZ; k++) {
      const azimuth = (k * 2 * Math.PI) / PART_AZ + stagger
      nearJobs.push(sideJob(azimuth, [center[0] + part.cx, center[1], center[2] + part.cz], ext3, part.ranges[p]!))
    }
  }
  for (let b = 0; b < BANDS; b++) {
    const yb = part.bandY[b]!
    const bandRange: Range = {
      first: part.ranges[b * SECTORS]!.first,
      count: part.ranges
        .slice(b * SECTORS, b * SECTORS + SECTORS)
        .reduce((acc, r) => acc + r.count, 0),
    }
    nearJobs.push(
      topJob(
        [part.bands[b * 6]! + part.cx, (yb.y0 + yb.y1) / 2, part.bands[b * 6 + 1]! + part.cz],
        [part.bands[b * 6 + 2]!, Math.max((yb.y1 - yb.y0) / 2, 1e-3), part.bands[b * 6 + 3]!],
        bandRange,
      ),
    )
  }

  const farCenter: [number, number, number] = [part.cx, (part.y0 + part.y1) / 2, part.cz]
  const farExt: [number, number, number] = [part.rXZ, (part.y1 - part.y0) / 2, part.rXZ]
  const farJobs: TileJob[] = []
  for (let k = 0; k < FAR_AZ; k++) {
    farJobs.push(sideJob((k * 2 * Math.PI) / FAR_AZ, farCenter, farExt, allRange))
  }
  farJobs.push(topJob(farCenter, farExt, allRange))

  // --- shared GPU resources ------------------------------------------------
  const vbuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-verts`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const ibuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-idx`,
      size: part.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, part.indices)

  const module = ctx.shaders.module(bakeShaderSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 112 },
      },
    ],
  })
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

  const near = await renderTileSet(ctx, { jobs: nearJobs, tilePx: NEAR_TILE, cols: 8 }, { vbuf, ibuf, pipeline, bgl }, [
    [bx0, by0, bz0],
    [bx1 - bx0, by1 - by0, bz1 - bz0],
  ])
  const far = await renderTileSet(ctx, { jobs: farJobs, tilePx: FAR_TILE, cols: 3 }, { vbuf, ibuf, pipeline, bgl }, [
    [bx0, by0, bz0],
    [bx1 - bx0, by1 - by0, bz1 - bz0],
  ])
  vbuf.destroy()
  ibuf.destroy()

  return packParts({
    nearTile: NEAR_TILE,
    farTile: FAR_TILE,
    cx: part.cx,
    cz: part.cz,
    y0: part.y0,
    y1: part.y1,
    rXZ: part.rXZ,
    parts: part.parts,
    bands: part.bands,
    nearAlbedo: near.albedo,
    nearNormal: near.normal,
    farAlbedo: far.albedo,
    farNormal: far.normal,
  })
}

interface TileSetSpec {
  jobs: TileJob[]
  tilePx: number
  cols: number
}

interface BakePipe {
  vbuf: GPUBuffer
  ibuf: GPUBuffer
  pipeline: GPURenderPipeline
  bgl: GPUBindGroupLayout
}

/** Render one tile set at SSx, read it back, and downsample+dilate per tile. */
async function renderTileSet(
  ctx: BakeCtx,
  spec: TileSetSpec,
  pipe: BakePipe,
  bounds: [[number, number, number], [number, number, number]],
): Promise<{ albedo: Uint8Array<ArrayBuffer>; normal: Uint8Array<ArrayBuffer> }> {
  const { device } = ctx
  const { jobs, tilePx, cols } = spec
  const rows = Math.ceil(jobs.length / cols)
  const big = tilePx * SS
  const width = cols * big
  const height = rows * big

  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-uni`,
      size: STRIDE * jobs.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * jobs.length)
  jobs.forEach((job, k) => {
    const o = (k * STRIDE) / 4
    scratch.set(job.right, o)
    scratch.set(job.up, o + 4)
    scratch.set(job.fwd, o + 8)
    scratch.set(job.center, o + 12)
    scratch.set(job.ext, o + 16)
    scratch.set(bounds[0], o + 20)
    scratch.set(bounds[1], o + 24)
  })
  device.queue.writeBuffer(uni, 0, scratch)
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: pipe.bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
  })

  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [width, height],
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
      size: [width, height],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

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
  pass.setPipeline(pipe.pipeline)
  pass.setVertexBuffer(0, pipe.vbuf)
  pass.setIndexBuffer(pipe.ibuf, 'uint32')
  jobs.forEach((job, k) => {
    if (job.count === 0) return
    const col = k % cols
    const row = Math.floor(k / cols)
    pass.setViewport(col * big, row * big, big, big, 0, 1)
    pass.setScissorRect(col * big, row * big, big, big)
    pass.setBindGroup(0, bg, [k * STRIDE])
    pass.drawIndexed(job.count, 1, job.first)
  })
  pass.end()

  const bpr = width * 4
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbNrm = mkReadback('bake-rb-normal')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: height }, [
    width,
    height,
  ])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNrm = new Uint8Array(rbNrm.getMappedRange()).slice()
  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()

  const albedo = new Uint8Array(jobs.length * tilePx * tilePx * 4)
  const normal = new Uint8Array(jobs.length * tilePx * tilePx * 2)
  for (let k = 0; k < jobs.length; k++) {
    const col = k % cols
    const row = Math.floor(k / cols)
    processTile(bigAlb, bigNrm, width, col * big, row * big, tilePx, albedo, normal, k)
  }
  return { albedo, normal }
}

/**
 * Coverage-weighted SSx downsample of one tile, then a few dilation passes so
 * bilinear/mip filtering never pulls background black in from empty texels
 * (alpha stays 0 there — the silhouette is unaffected).
 */
function processTile(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  bigStride: number,
  ox: number,
  oy: number,
  tilePx: number,
  outAlb: Uint8Array,
  outNrm: Uint8Array,
  layer: number,
): void {
  const n = tilePx * tilePx
  const rgba = new Uint8Array(n * 4)
  const nrm = new Float32Array(n * 3)
  const filled = new Uint8Array(n)

  for (let y = 0; y < tilePx; y++) {
    for (let x = 0; x < tilePx; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((oy + y * SS + j) * bigStride + (ox + x * SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
          nx += (bigNrm[s]! / 127.5 - 1) * a
          ny += (bigNrm[s + 1]! / 127.5 - 1) * a
          nz += (bigNrm[s + 2]! / 127.5 - 1) * a
        }
      }
      const idx = y * tilePx + x
      if (aSum > 0) {
        rgba[idx * 4] = Math.round(r / aSum)
        rgba[idx * 4 + 1] = Math.round(g / aSum)
        rgba[idx * 4 + 2] = Math.round(b / aSum)
        rgba[idx * 4 + 3] = Math.round(aSum / (SS * SS))
        nrm[idx * 3] = nx
        nrm[idx * 3 + 1] = ny
        nrm[idx * 3 + 2] = nz
        filled[idx] = 1
      }
    }
  }

  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const nextFilled = cur.slice()
    for (let y = 0; y < tilePx; y++) {
      for (let x = 0; x < tilePx; x++) {
        const idx = y * tilePx + x
        if (cur[idx]! !== 0) continue
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= tilePx) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < 0 || xx >= tilePx) continue
            const nIdx = yy * tilePx + xx
            if (cur[nIdx]! === 0) continue
            count++
            r += rgba[nIdx * 4]!
            g += rgba[nIdx * 4 + 1]!
            b += rgba[nIdx * 4 + 2]!
            nx += nrm[nIdx * 3]!
            ny += nrm[nIdx * 3 + 1]!
            nz += nrm[nIdx * 3 + 2]!
          }
        }
        if (count === 0) continue
        rgba[idx * 4] = Math.round(r / count)
        rgba[idx * 4 + 1] = Math.round(g / count)
        rgba[idx * 4 + 2] = Math.round(b / count)
        nrm[idx * 3] = nx / count
        nrm[idx * 3 + 1] = ny / count
        nrm[idx * 3 + 2] = nz / count
        nextFilled[idx] = 1
      }
    }
    cur = nextFilled
  }

  outAlb.set(rgba, layer * n * 4)
  for (let i = 0; i < n; i++) {
    const e = octEncode(nrm[i * 3]!, nrm[i * 3 + 1]!, nrm[i * 3 + 2]!)
    outNrm[layer * n * 2 + i * 2] = e[0]
    outNrm[layer * n * 2 + i * 2 + 1] = e[1]
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
