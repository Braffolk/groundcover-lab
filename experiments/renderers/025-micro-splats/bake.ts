import stampShaderSrc from './shaders/stamp.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Micro-splat bake: turns a GCMESH1 source mesh into a hierarchy of TINY
 * oriented surfels that sit exactly on the original surface.
 *
 * 1. Triangles (area, centroid, geometric normal, authored colour) are sorted
 *    into a perfectly balanced kd-tree — 13 median splits along the longest
 *    axis of the current centroid box, so every leaf is a spatially compact,
 *    equal-population patch of surface. Because the splits are balanced, EVERY
 *    tree level is a complete clustering of the plant: level 13 = 8192 patches,
 *    level 0 = 1 patch. That is the LOD ladder, for free, and coarse splats are
 *    the exact union of the fine ones (area moments are additive).
 * 2. Every node's area-weighted moments give a centroid + covariance. The two
 *    largest PCA axes span the splat's plane, the smallest is its normal
 *    (sign-locked to the area-weighted mesh normal). Extents are 1.9 sigma, so
 *    the ellipse hugs the actual patch instead of ballooning around it.
 * 3. The patch's projected area / ellipse area is its coverage ratio. A splat
 *    stores that ratio and an index into a 16-tile stamp atlas cut from a real
 *    orthographic render of the plant, so a 6px splat still shows individual
 *    blades and a blade-shaped silhouette rather than a flat blob.
 * 4. A 24^3 area-density grid gives each splat a baked canopy occlusion term
 *    (how much foliage is above it) — that is what makes the interior read as
 *    self-shadowed rather than uniformly bright.
 *
 * Artifact layout (little-endian):
 *   0   u32 magic 'MSP1', u32 version, u32 lodCount, u32 totalSplats
 *   16  f32[3] bmin, f32[3] bmax           (position dequantization box)
 *   40  f32 y0, f32 y1, f32 rXZ            (support box, unit scale, metres)
 *   52  u32 stampAtlas, u32 stampTile, u32 stampMips
 *   64  lod table x16: u32 offset, u32 count, f32 maxA, f32 maxB
 *   320 splat records, 16B each (see packSplat)
 *   ... stamp atlas mip chain, rgba8, level 0 first
 */

const MAGIC = 0x3150534d // 'MSP1'
const VERSION = 2
const HEADER_BYTES = 320
const LOD_SLOTS = 16

/** kd-tree depth; 2^13 = 8192 finest splats per plant. */
export const TREE_DEPTH = 13
const LEAF_COUNT = 1 << TREE_DEPTH
/** Tree levels used as LODs: 13 (8192 splats) down to 0 (1 splat). */
export const LOD_COUNT = 14
export const LOD_SPLATS: number[] = Array.from({ length: LOD_COUNT }, (_, k) => 1 << (TREE_DEPTH - k))
export const MAX_SPLATS = LOD_SPLATS[0]!
export const TOTAL_SPLATS = LOD_SPLATS.reduce((a, b) => a + b, 0)

export const STAMP_TILE = 64
export const STAMP_GRID = 4
export const STAMP_ATLAS = STAMP_TILE * STAMP_GRID
export const STAMP_MIPS = 4
const STAMP_COUNT = STAMP_GRID * STAMP_GRID
/** Alpha reference the stamp mips are coverage-corrected for. */
const STAMP_ALPHA_REF = 0.45

const RENDER_PX = 2048 // stamp source render (downsampled 2x before windowing)
const SRC_PX = RENDER_PX / 2
const MAX_TRIS = 1_500_000
/** Ellipse half-extent in units of the patch's PCA standard deviation. */
const EXTENT_SIGMA = 1.9

export interface LodDesc {
  offset: number
  count: number
  maxA: number
  maxB: number
}

export interface SplatSet {
  bmin: [number, number, number]
  bmax: [number, number, number]
  y0: number
  y1: number
  rXZ: number
  lods: LodDesc[]
  /** TOTAL_SPLATS * 16 bytes. */
  splats: Uint8Array<ArrayBuffer>
  /** Concatenated rgba8 mip levels of the stamp atlas. */
  stamp: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

// ---------------------------------------------------------------------------
// artifact (de)serialization
// ---------------------------------------------------------------------------

function stampBytes(): number {
  let n = 0
  for (let l = 0; l < STAMP_MIPS; l++) {
    const s = STAMP_ATLAS >> l
    n += s * s * 4
  }
  return n
}

export function unpackSplats(buf: ArrayBuffer): SplatSet | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 16)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const lodCount = u[2]!
  const total = u[3]!
  if (lodCount !== LOD_COUNT || total !== TOTAL_SPLATS) return null
  const f = new Float32Array(buf, 16, 9)
  if (u[13] !== STAMP_ATLAS || u[14] !== STAMP_TILE || u[15] !== STAMP_MIPS) return null
  const expected = HEADER_BYTES + total * 16 + stampBytes()
  if (buf.byteLength !== expected) return null
  const table = new DataView(buf, 64, LOD_SLOTS * 16)
  const lods: LodDesc[] = []
  for (let k = 0; k < lodCount; k++) {
    lods.push({
      offset: table.getUint32(k * 16, true),
      count: table.getUint32(k * 16 + 4, true),
      maxA: table.getFloat32(k * 16 + 8, true),
      maxB: table.getFloat32(k * 16 + 12, true),
    })
  }
  return {
    bmin: [f[0]!, f[1]!, f[2]!],
    bmax: [f[3]!, f[4]!, f[5]!],
    y0: f[6]!,
    y1: f[7]!,
    rXZ: f[8]!,
    lods,
    splats: new Uint8Array(buf, HEADER_BYTES, total * 16),
    stamp: new Uint8Array(buf, HEADER_BYTES + total * 16, stampBytes()),
  }
}

function packSplats(s: SplatSet): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + s.splats.byteLength + s.stamp.byteLength)
  const u = new Uint32Array(buf, 0, 16)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = s.lods.length
  u[3] = TOTAL_SPLATS
  const f = new Float32Array(buf, 16, 9)
  f.set([...s.bmin, ...s.bmax, s.y0, s.y1, s.rXZ])
  u[13] = STAMP_ATLAS
  u[14] = STAMP_TILE
  u[15] = STAMP_MIPS
  const table = new DataView(buf, 64, LOD_SLOTS * 16)
  s.lods.forEach((l, k) => {
    table.setUint32(k * 16, l.offset, true)
    table.setUint32(k * 16 + 4, l.count, true)
    table.setFloat32(k * 16 + 8, l.maxA, true)
    table.setFloat32(k * 16 + 12, l.maxB, true)
  })
  new Uint8Array(buf, HEADER_BYTES, s.splats.byteLength).set(s.splats)
  new Uint8Array(buf, HEADER_BYTES + s.splats.byteLength, s.stamp.byteLength).set(s.stamp)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake this species' splat hierarchy.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, so every result is magic-validated and a poisoned entry rebaked.
 */
export async function loadSpeciesSplats(ctx: BakeCtx, speciesId: string): Promise<SplatSet> {
  const key = `msplat-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesSplats(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let set = unpackSplats(buf)
  if (!set) {
    buf = await runBake()
    set = unpackSplats(buf)
    if (!set) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return set
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
// small math helpers
// ---------------------------------------------------------------------------

/** Octahedral encode, y-primary — exact inverse of octDecode below. */
function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  if (s < 1e-9) return [128, 128]
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

function octDecode(qu: number, qv: number): [number, number, number] {
  const u = (qu / 255) * 2 - 1
  const v = (qv / 255) * 2 - 1
  let x = u
  let z = v
  const y = 1 - Math.abs(u) - Math.abs(v)
  if (y < 0) {
    x = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1)
    z = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1)
  }
  const len = Math.hypot(x, y, z) || 1
  return [x / len, y / len, z / len]
}

/**
 * Branch-free orthonormal basis (Duff et al. 2017). The WGSL twin in
 * splats.wgsl is character-for-character the same formula, evaluated on the
 * SAME dequantized normal, so the stored tangent angle reconstructs exactly.
 */
function onb(nx: number, ny: number, nz: number): [number[], number[]] {
  const sgn = nz >= 0 ? 1 : -1
  const a = -1 / (sgn + nz)
  const b = nx * ny * a
  return [
    [1 + sgn * nx * nx * a, sgn * b, -sgn * nx],
    [b, sgn + ny * ny * a, -ny],
  ]
}

/** Cyclic Jacobi eigen-decomposition of a symmetric 3x3, descending order. */
function eigenSym3(
  xx: number,
  yy: number,
  zz: number,
  xy: number,
  xz: number,
  yz: number,
): { val: number[]; vec: number[][] } {
  const a = [xx, xy, xz, xy, yy, yz, xz, yz, zz]
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1]
  for (let sweep = 0; sweep < 10; sweep++) {
    const off = Math.abs(a[1]!) + Math.abs(a[2]!) + Math.abs(a[5]!)
    if (off < 1e-20) break
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      const apq = a[p * 3 + q]!
      if (Math.abs(apq) < 1e-24) continue
      const theta = (a[q * 3 + q]! - a[p * 3 + p]!) / (2 * apq)
      const sgn = theta >= 0 ? 1 : -1
      const t = sgn / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
      const c = 1 / Math.sqrt(t * t + 1)
      const s = t * c
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p]!
        const akq = a[k * 3 + q]!
        a[k * 3 + p] = c * akp - s * akq
        a[k * 3 + q] = s * akp + c * akq
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k]!
        const aqk = a[q * 3 + k]!
        a[p * 3 + k] = c * apk - s * aqk
        a[q * 3 + k] = s * apk + c * aqk
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p]!
        const vkq = v[k * 3 + q]!
        v[k * 3 + p] = c * vkp - s * vkq
        v[k * 3 + q] = s * vkp + c * vkq
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j * 3 + j]! - a[i * 3 + i]!)
  return {
    val: order.map((i) => a[i * 3 + i]!),
    vec: order.map((i) => [v[i]!, v[3 + i]!, v[6 + i]!]),
  }
}

/** Hoare quickselect: idx[lo..hi) partially ordered so idx[nth] is in place. */
function nthElement(idx: Uint32Array, lo: number, hi: number, nth: number, key: Float32Array): void {
  let l = lo
  let r = hi - 1
  while (l < r) {
    const mid = (l + r) >> 1
    const ka = key[idx[l]!]!
    const kb = key[idx[mid]!]!
    const kc = key[idx[r]!]!
    const pivot = ka < kb ? (kb < kc ? kb : ka < kc ? kc : ka) : ka < kc ? ka : kb < kc ? kc : kb
    let i = l
    let j = r
    while (i <= j) {
      while (key[idx[i]!]! < pivot) i++
      while (key[idx[j]!]! > pivot) j--
      if (i <= j) {
        const t = idx[i]!
        idx[i] = idx[j]!
        idx[j] = t
        i++
        j--
      }
    }
    if (nth <= j) r = j
    else if (nth >= i) l = i
    else return
  }
}

// ---------------------------------------------------------------------------
// the bake
// ---------------------------------------------------------------------------

async function bakeSpeciesSplats(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const verts = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535

  // Exact horizontal support radius (bounds corners overestimate it a lot on
  // wide community tiles) — same measure 001 uses, so both put the plant in
  // the same place at the same size.
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

  // Stamp atlas first: it needs mesh.indices() (large) and nothing else, so it
  // can be finished and freed before the clustering arrays are allocated.
  const stamp = await bakeStampAtlas(ctx, mesh, cx, cz, y0, y1, rXZ)

  // --- triangle soup (sub-sampled; area-unbiased) ---------------------------
  const triCount = hdr.triangleCount
  const stride = Math.max(1, Math.ceil(triCount / MAX_TRIS))
  const m = Math.floor(triCount / stride)
  const tri = mesh.triangles
  const tcx = new Float32Array(m)
  const tcy = new Float32Array(m)
  const tcz = new Float32Array(m)
  const tnx = new Float32Array(m)
  const tny = new Float32Array(m)
  const tnz = new Float32Array(m)
  const tar = new Float32Array(m)
  const tcr = new Float32Array(m)
  const tcg = new Float32Array(m)
  const tcb = new Float32Array(m)

  for (let j = 0; j < m; j++) {
    const t = j * stride
    const i0 = tri[t * 4]! * 8
    const i1 = tri[t * 4 + 1]! * 8
    const i2 = tri[t * 4 + 2]! * 8
    const p0x = bx0 + verts[i0]! * sx
    const p0y = by0 + verts[i0 + 1]! * sy
    const p0z = bz0 + verts[i0 + 2]! * sz
    const p1x = bx0 + verts[i1]! * sx
    const p1y = by0 + verts[i1 + 1]! * sy
    const p1z = bz0 + verts[i1 + 2]! * sz
    const p2x = bx0 + verts[i2]! * sx
    const p2y = by0 + verts[i2 + 1]! * sy
    const p2z = bz0 + verts[i2 + 2]! * sz
    const e1x = p1x - p0x
    const e1y = p1y - p0y
    const e1z = p1z - p0z
    const e2x = p2x - p0x
    const e2y = p2y - p0y
    const e2z = p2z - p0z
    const nx = e1y * e2z - e1z * e2y
    const ny = e1z * e2x - e1x * e2z
    const nz = e1x * e2y - e1y * e2x
    const len = Math.hypot(nx, ny, nz)
    tar[j] = len * 0.5
    const inv = len > 1e-20 ? 1 / len : 0
    tnx[j] = nx * inv
    tny[j] = ny * inv
    tnz[j] = nz * inv
    tcx[j] = (p0x + p1x + p2x) / 3
    tcy[j] = (p0y + p1y + p2y) / 3
    tcz[j] = (p0z + p1z + p2z) / 3
    tcr[j] = (verts[i0 + 3]! + verts[i1 + 3]! + verts[i2 + 3]!) / (3 * 65535)
    tcg[j] = (verts[i0 + 4]! + verts[i1 + 4]! + verts[i2 + 4]!) / (3 * 65535)
    tcb[j] = (verts[i0 + 5]! + verts[i1 + 5]! + verts[i2 + 5]!) / (3 * 65535)
  }

  // --- per-triangle canopy occlusion ----------------------------------------
  // A 24^3 area-density grid, suffix-summed along +y, answers "how much foliage
  // is above this point". Doing it per TRIANGLE (not per splat) means a coarse
  // splat's occlusion is the true mean over its patch, and — crucially — its
  // colour can be occlusion-weighted below, so a distant plant takes the colour
  // of its lit outer shell instead of the average of shell and dark interior.
  const G = 24
  const gsx = G / Math.max(bx1 - bx0, 1e-6)
  const gsy = G / Math.max(by1 - by0, 1e-6)
  const gsz = G / Math.max(bz1 - bz0, 1e-6)
  const grid = new Float32Array(G * G * G)
  const cellOf = (x: number, y: number, z: number): number => {
    const ix = Math.min(G - 1, Math.max(0, Math.floor((x - bx0) * gsx)))
    const iy = Math.min(G - 1, Math.max(0, Math.floor((y - by0) * gsy)))
    const iz = Math.min(G - 1, Math.max(0, Math.floor((z - bz0) * gsz)))
    return (iz * G + iy) * G + ix
  }
  for (let t = 0; t < m; t++) {
    const c = cellOf(tcx[t]!, tcy[t]!, tcz[t]!)
    grid[c] = grid[c]! + tar[t]!
  }
  const aboveGrid = new Float32Array(G * G * G)
  let colMax = 0
  for (let iz = 0; iz < G; iz++) {
    for (let ix = 0; ix < G; ix++) {
      let run = 0
      for (let iy = G - 1; iy >= 0; iy--) {
        const c = (iz * G + iy) * G + ix
        aboveGrid[c] = run
        run += grid[c]!
      }
      if (run > colMax) colMax = run
    }
  }
  const occlDenom = Math.max(colMax * 0.32, 1e-9)
  const invH = 1 / Math.max(y1 - y0, 1e-6)
  const tao = new Float32Array(m)
  for (let t = 0; t < m; t++) {
    const occl = Math.min(1, aboveGrid[cellOf(tcx[t]!, tcy[t]!, tcz[t]!)]! / occlDenom)
    const hf = Math.min(1, Math.max(0, (tcy[t]! - y0) * invH))
    tao[t] = Math.min(1, Math.max(0.22, (1 - 0.62 * occl) * (0.52 + 0.48 * Math.pow(hf, 0.6))))
  }

  // --- balanced kd-tree over triangle centroids ------------------------------
  const idx = new Uint32Array(m)
  for (let i = 0; i < m; i++) idx[i] = i
  const leafLo = new Int32Array(LEAF_COUNT)
  const leafHi = new Int32Array(LEAF_COUNT)
  const axisKey = [tcx, tcy, tcz]

  const build = (lo: number, hi: number, depth: number, node: number): void => {
    if (depth === TREE_DEPTH) {
      leafLo[node - LEAF_COUNT] = lo
      leafHi[node - LEAF_COUNT] = hi
      return
    }
    const mid = lo + ((hi - lo) >> 1)
    if (hi - lo > 1) {
      let lox = Infinity
      let loy = Infinity
      let loz = Infinity
      let hix = -Infinity
      let hiy = -Infinity
      let hiz = -Infinity
      for (let i = lo; i < hi; i++) {
        const t = idx[i]!
        const x = tcx[t]!
        const y = tcy[t]!
        const z = tcz[t]!
        if (x < lox) lox = x
        if (x > hix) hix = x
        if (y < loy) loy = y
        if (y > hiy) hiy = y
        if (z < loz) loz = z
        if (z > hiz) hiz = z
      }
      const ex = hix - lox
      const ey = hiy - loy
      const ez = hiz - loz
      const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2
      nthElement(idx, lo, hi, mid, axisKey[axis]!)
    }
    build(lo, mid, depth + 1, node * 2)
    build(mid, hi, depth + 1, node * 2 + 1)
  }
  build(0, m, 0, 1)

  const leafOf = new Uint16Array(m)
  for (let k = 0; k < LEAF_COUNT; k++) {
    for (let i = leafLo[k]!; i < leafHi[k]!; i++) leafOf[idx[i]!] = k
  }

  // --- area-weighted moments per node (leaves, then summed up the heap) -----
  // A Sx Sy Sz Mxx Myy Mzz Mxy Mxz Myz Nx Ny Nz | Aao CaoR CaoG CaoB.
  // Colour is accumulated PRE-MULTIPLIED by occlusion and divided by Aao, so
  // colour x ao reproduces the patch's true mean radiance at every tree level.
  const F = 17
  const mom = new Float64Array(LEAF_COUNT * 2 * F)
  const acc = new Float64Array(F)
  for (let k = 0; k < LEAF_COUNT; k++) {
    acc.fill(0)
    for (let i = leafLo[k]!; i < leafHi[k]!; i++) {
      const t = idx[i]!
      const a = tar[t]!
      const x = tcx[t]!
      const y = tcy[t]!
      const z = tcz[t]!
      acc[0] = acc[0]! + a
      acc[1] = acc[1]! + a * x
      acc[2] = acc[2]! + a * y
      acc[3] = acc[3]! + a * z
      acc[4] = acc[4]! + a * x * x
      acc[5] = acc[5]! + a * y * y
      acc[6] = acc[6]! + a * z * z
      acc[7] = acc[7]! + a * x * y
      acc[8] = acc[8]! + a * x * z
      acc[9] = acc[9]! + a * y * z
      acc[10] = acc[10]! + a * tnx[t]!
      acc[11] = acc[11]! + a * tny[t]!
      acc[12] = acc[12]! + a * tnz[t]!
      const w = a * tao[t]!
      acc[13] = acc[13]! + w
      acc[14] = acc[14]! + w * tcr[t]!
      acc[15] = acc[15]! + w * tcg[t]!
      acc[16] = acc[16]! + w * tcb[t]!
    }
    mom.set(acc, (LEAF_COUNT + k) * F)
  }
  for (let n = LEAF_COUNT - 1; n >= 1; n--) {
    const o = n * F
    const l = n * 2 * F
    const r = (n * 2 + 1) * F
    for (let c = 0; c < F; c++) mom[o + c] = mom[l + c]! + mom[r + c]!
  }

  // --- PCA frame per node ----------------------------------------------------
  const nodeCount = LEAF_COUNT * 2
  const nrm = new Float32Array(nodeCount * 3)
  const tanA = new Float32Array(nodeCount * 3)
  const extA = new Float32Array(nodeCount)
  const extB = new Float32Array(nodeCount)
  const octU = new Uint8Array(nodeCount)
  const octV = new Uint8Array(nodeCount)
  const angQ = new Uint8Array(nodeCount)

  for (let n = 1; n < nodeCount; n++) {
    const o = n * F
    const A = mom[o]!
    if (A <= 1e-12) continue
    const px = mom[o + 1]! / A
    const py = mom[o + 2]! / A
    const pz = mom[o + 3]! / A
    const cxx = mom[o + 4]! / A - px * px
    const cyy = mom[o + 5]! / A - py * py
    const czz = mom[o + 6]! / A - pz * pz
    const cxy = mom[o + 7]! / A - px * py
    const cxz = mom[o + 8]! / A - px * pz
    const cyz = mom[o + 9]! / A - py * pz
    const { val, vec } = eigenSym3(cxx, cyy, czz, cxy, cxz, cyz)
    let nx = vec[2]![0]!
    let ny = vec[2]![1]!
    let nz = vec[2]![2]!
    if (nx * mom[o + 10]! + ny * mom[o + 11]! + nz * mom[o + 12]! < 0) {
      nx = -nx
      ny = -ny
      nz = -nz
    }
    const [qu, qv] = octEncode(nx, ny, nz)
    const [dnx, dny, dnz] = octDecode(qu, qv)
    octU[n] = qu
    octV[n] = qv
    nrm[n * 3] = dnx
    nrm[n * 3 + 1] = dny
    nrm[n * 3 + 2] = dnz

    // Major PCA axis, projected into the (quantized) splat plane.
    let t1x = vec[0]![0]!
    let t1y = vec[0]![1]!
    let t1z = vec[0]![2]!
    const dp = t1x * dnx + t1y * dny + t1z * dnz
    t1x -= dnx * dp
    t1y -= dny * dp
    t1z -= dnz * dp
    const tl = Math.hypot(t1x, t1y, t1z)
    if (tl > 1e-12) {
      t1x /= tl
      t1y /= tl
      t1z /= tl
    } else {
      t1x = 1
      t1y = 0
      t1z = 0
    }
    const [u0, v0] = onb(dnx, dny, dnz)
    const ang = Math.atan2(
      t1x * v0[0]! + t1y * v0[1]! + t1z * v0[2]!,
      t1x * u0[0]! + t1y * u0[1]! + t1z * u0[2]!,
    )
    const qa = Math.round((((ang / (2 * Math.PI)) % 1) + 1) % 1 * 256) & 255
    angQ[n] = qa
    // Rebuild the tangent EXACTLY as the shader will, so the projected-area
    // pass below measures the frame that actually gets rendered.
    const ra = (qa / 256) * 2 * Math.PI
    const ca = Math.cos(ra)
    const sa = Math.sin(ra)
    tanA[n * 3] = ca * u0[0]! + sa * v0[0]!
    tanA[n * 3 + 1] = ca * u0[1]! + sa * v0[1]!
    tanA[n * 3 + 2] = ca * u0[2]! + sa * v0[2]!

    const a = EXTENT_SIGMA * Math.sqrt(Math.max(val[0]!, 0))
    const b = EXTENT_SIGMA * Math.sqrt(Math.max(val[1]!, 0))
    extA[n] = Math.max(a, 1e-4)
    extB[n] = Math.max(b, a * 0.05, 1e-4)
  }

  // --- projected area per node (one pass, all levels at once) ---------------
  const aproj = new Float64Array(nodeCount)
  for (let t = 0; t < m; t++) {
    const a = tar[t]!
    if (a <= 0) continue
    const nx = tnx[t]!
    const ny = tny[t]!
    const nz = tnz[t]!
    let node = LEAF_COUNT + leafOf[t]!
    for (let d = TREE_DEPTH; d >= 0; d--) {
      aproj[node] =
        aproj[node]! + a * Math.abs(nx * nrm[node * 3]! + ny * nrm[node * 3 + 1]! + nz * nrm[node * 3 + 2]!)
      node >>= 1
    }
  }

  // --- pack splats -----------------------------------------------------------
  const splats = new Uint8Array(TOTAL_SPLATS * 16)
  const view = new DataView(splats.buffer)
  const lods: LodDesc[] = []
  const invExt = [1 / Math.max(bx1 - bx0, 1e-9), 1 / Math.max(by1 - by0, 1e-9), 1 / Math.max(bz1 - bz0, 1e-9)]
  const stampCov = stamp.coverage
  let cursor = 0

  for (let k = 0; k < LOD_COUNT; k++) {
    const level = TREE_DEPTH - k
    const first = 1 << level
    const count = 1 << level
    let maxA = 1e-5
    let maxB = 1e-5
    for (let i = 0; i < count; i++) {
      const n = first + i
      if (mom[n * F]! <= 1e-12) continue
      if (extA[n]! > maxA) maxA = extA[n]!
      if (extB[n]! > maxB) maxB = extB[n]!
    }
    const offset = cursor
    for (let i = 0; i < count; i++) {
      const n = first + i
      const o = n * F
      const A = mom[o]!
      const rec = (offset + i) * 16
      if (A <= 1e-12) {
        // Empty node (degenerate mesh region): a zero-extent splat is
        // rasterized as nothing, which is exactly what we want.
        view.setUint32(rec, 0, true)
        view.setUint32(rec + 4, 0, true)
        view.setUint32(rec + 8, 0, true)
        view.setUint32(rec + 12, 0, true)
        continue
      }
      const px = mom[o + 1]! / A
      const py = mom[o + 2]! / A
      const pz = mom[o + 3]! / A
      const qx = Math.min(65535, Math.max(0, Math.round((px - bx0) * invExt[0]! * 65535)))
      const qy = Math.min(65535, Math.max(0, Math.round((py - by0) * invExt[1]! * 65535)))
      const qz = Math.min(65535, Math.max(0, Math.round((pz - bz0) * invExt[2]! * 65535)))
      const a = extA[n]!
      const b = extB[n]!
      const rho = Math.min(1, aproj[n]! / Math.max(Math.PI * a * b, 1e-12))
      // Stamp whose real coverage is closest to this patch's coverage ratio.
      let best = 0
      let bestErr = Infinity
      for (let s = 0; s < STAMP_COUNT; s++) {
        const e = Math.abs(stampCov[s]! - rho)
        if (e < bestErr) {
          bestErr = e
          best = s
        }
      }
      const rhoQ = Math.min(15, Math.max(0, Math.round(rho * 15)))
      const wao = Math.max(mom[o + 13]!, 1e-12)
      const ao = Math.min(1, Math.max(0.05, wao / A))
      const qa = Math.round(Math.sqrt(Math.min(1, a / maxA)) * 255)
      const qb = Math.round(Math.sqrt(Math.min(1, b / maxB)) * 255)
      view.setUint32(rec, qx | (qy << 16), true)
      view.setUint32(rec + 4, qz | (octU[n]! << 16) | (octV[n]! << 24), true)
      view.setUint32(rec + 8, angQ[n]! | (qa << 8) | (qb << 16) | (best << 24) | (rhoQ << 28), true)
      const cr = Math.min(255, Math.round((mom[o + 14]! / wao) * 255))
      const cg = Math.min(255, Math.round((mom[o + 15]! / wao) * 255))
      const cb = Math.min(255, Math.round((mom[o + 16]! / wao) * 255))
      view.setUint32(rec + 12, cr | (cg << 8) | (cb << 16) | (Math.round(ao * 255) << 24), true)
    }
    lods.push({ offset, count, maxA, maxB })
    cursor += count
  }

  return packSplats({
    bmin: [bx0, by0, bz0],
    bmax: [bx1, by1, bz1],
    y0,
    y1,
    rXZ,
    lods,
    splats,
    stamp: stamp.data,
  })
}

// ---------------------------------------------------------------------------
// stamp atlas — real blade imagery cut out of one orthographic render
// ---------------------------------------------------------------------------

interface StampBake {
  data: Uint8Array<ArrayBuffer>
  /** Measured coverage of each tile (matched against a splat's rho). */
  coverage: number[]
}

async function bakeStampAtlas(
  ctx: BakeCtx,
  mesh: GcMesh,
  cx: number,
  cz: number,
  y0: number,
  y1: number,
  rXZ: number,
): Promise<StampBake> {
  const { device } = ctx
  const hdr = mesh.header
  const scale = Math.max(rXZ, (y1 - y0) / 2)

  const verts = mesh.vertices
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/stamp-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/stamp-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const tex = ctx.res.createTexture(
    {
      label: `${ctx.id}/stamp-render`,
      size: [RENDER_PX, RENDER_PX],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    },
    { tag: 'bake-scratch' },
  )
  const depth = ctx.res.createTexture(
    {
      label: `${ctx.id}/stamp-depth`,
      size: [RENDER_PX, RENDER_PX],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/stamp-uni`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const u = new Float32Array(16)
  u.set([cx, (y0 + y1) / 2, cz, scale], 0)
  u.set([hdr.boundsMin[0], hdr.boundsMin[1], hdr.boundsMin[2], 0], 4)
  u.set([
    hdr.boundsMax[0] - hdr.boundsMin[0],
    hdr.boundsMax[1] - hdr.boundsMin[1],
    hdr.boundsMax[2] - hdr.boundsMin[2],
    0,
  ], 8)
  device.queue.writeBuffer(uni, 0, u)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/stamp-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/stamp-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni } }],
  })
  const module = ctx.shaders.module(stampShaderSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/stamp-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/stamp-pl`, bindGroupLayouts: [bgl] }),
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
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const enc = device.createCommandEncoder({ label: `${ctx.id}/stamp-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/stamp-pass`,
    colorAttachments: [
      { view: tex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: {
      view: depth.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  })
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  pass.drawIndexed(indices.length)
  pass.end()

  const bpr = RENDER_PX * 4
  const rb = ctx.res.createBuffer(
    { label: `${ctx.id}/stamp-rb`, size: bpr * RENDER_PX, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
    { tag: 'bake-scratch' },
  )
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: RENDER_PX }, [
    RENDER_PX,
    RENDER_PX,
  ])
  device.queue.submit([enc.finish()])
  await rb.mapAsync(GPUMapMode.READ)
  const big = new Uint8Array(rb.getMappedRange()).slice()
  rb.unmap()
  for (const r of [vbuf, ibuf, tex, depth, uni, rb]) r.destroy()

  return buildStampAtlas(big)
}

/** Downsample 2x, window-search 16 tiles by coverage, build corrected mips. */
function buildStampAtlas(big: Uint8Array): StampBake {
  const N = SRC_PX
  const src = new Float32Array(N * N * 4)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          const s = ((y * 2 + j) * RENDER_PX + (x * 2 + i)) * 4
          const a = big[s + 3]!
          if (a === 0) continue
          aSum += a
          r += big[s]! * a
          g += big[s + 1]! * a
          b += big[s + 2]! * a
        }
      }
      const d = (y * N + x) * 4
      if (aSum > 0) {
        src[d] = r / aSum / 255
        src[d + 1] = g / aSum / 255
        src[d + 2] = b / aSum / 255
        src[d + 3] = aSum / (4 * 255)
      }
    }
  }

  // Summed-area table of alpha for O(1) window coverage.
  const sat = new Float64Array((N + 1) * (N + 1))
  for (let y = 0; y < N; y++) {
    let row = 0
    for (let x = 0; x < N; x++) {
      row += src[(y * N + x) * 4 + 3]!
      sat[(y + 1) * (N + 1) + (x + 1)] = sat[y * (N + 1) + (x + 1)]! + row
    }
  }
  const winCov = (x: number, y: number): number => {
    const s =
      sat[(y + STAMP_TILE) * (N + 1) + x + STAMP_TILE]! -
      sat[y * (N + 1) + x + STAMP_TILE]! -
      sat[(y + STAMP_TILE) * (N + 1) + x]! +
      sat[y * (N + 1) + x]!
    return s / (STAMP_TILE * STAMP_TILE)
  }

  const picks: { x: number; y: number; cov: number }[] = []
  const STEP = 8
  for (let s = 0; s < STAMP_COUNT; s++) {
    const target = (s + 0.5) / STAMP_COUNT
    let bx = 0
    let by = 0
    let bcov = 0
    let bestScore = Infinity
    for (let y = 0; y + STAMP_TILE <= N; y += STEP) {
      for (let x = 0; x + STAMP_TILE <= N; x += STEP) {
        const c = winCov(x, y)
        // Spread the picks apart so the 16 tiles are not 16 crops of the same
        // leaf; the penalty is tiny next to a coverage mismatch.
        let near = 0
        for (const p of picks) {
          const d = Math.hypot(p.x - x, p.y - y)
          if (d < STAMP_TILE * 2) near += (STAMP_TILE * 2 - d) / (STAMP_TILE * 2)
        }
        const score = Math.abs(c - target) + near * 0.02
        if (score < bestScore) {
          bestScore = score
          bx = x
          by = y
          bcov = c
        }
      }
    }
    picks.push({ x: bx, y: by, cov: bcov })
  }

  // Assemble mip 0: each tile transposed so the source's mostly-vertical
  // blades run along the tile's U axis — which is the splat's major PCA axis.
  const atlas = new Float32Array(STAMP_ATLAS * STAMP_ATLAS * 4)
  const coverage: number[] = []
  for (let s = 0; s < STAMP_COUNT; s++) {
    const p = picks[s]!
    const tx = (s % STAMP_GRID) * STAMP_TILE
    const ty = Math.floor(s / STAMP_GRID) * STAMP_TILE
    let mr = 0
    let mg = 0
    let mb = 0
    let mw = 0
    for (let v = 0; v < STAMP_TILE; v++) {
      for (let u = 0; u < STAMP_TILE; u++) {
        const sxi = p.x + v
        const syi = p.y + u
        const s4 = (syi * N + sxi) * 4
        const d4 = ((ty + v) * STAMP_ATLAS + tx + u) * 4
        const a = src[s4 + 3]!
        atlas[d4] = src[s4]!
        atlas[d4 + 1] = src[s4 + 1]!
        atlas[d4 + 2] = src[s4 + 2]!
        atlas[d4 + 3] = a
        mr += src[s4]! * a
        mg += src[s4 + 1]! * a
        mb += src[s4 + 2]! * a
        mw += a
      }
    }
    coverage.push(p.cov)
    // Store colour as a modulation around the tile mean (0.5 = neutral), so a
    // stamp transplanted onto any splat keeps that splat's own mean colour.
    const mean = [mr, mg, mb].map((c) => (mw > 1e-6 ? c / mw : 0.5))
    for (let v = 0; v < STAMP_TILE; v++) {
      for (let u = 0; u < STAMP_TILE; u++) {
        const d4 = ((ty + v) * STAMP_ATLAS + tx + u) * 4
        for (let c = 0; c < 3; c++) {
          const ref = Math.max(mean[c]!, 1e-3)
          atlas[d4 + c] = Math.min(1, Math.max(0, (atlas[d4 + c]! / ref) * 0.5))
        }
      }
    }
    dilateTile(atlas, tx, ty)
  }

  // Mip chain, per tile (never across tile borders), with Castano-style alpha
  // rescaling so alpha-tested coverage survives minification instead of
  // dissolving the far field away.
  const total = stampBytes()
  const out = new Uint8Array(total)
  let off = 0
  let level = new Float32Array(atlas)
  let tile = STAMP_TILE
  const refCov: number[] = []
  for (let l = 0; l < STAMP_MIPS; l++) {
    const dim = STAMP_ATLAS >> l
    if (l === 0) {
      for (let s = 0; s < STAMP_COUNT; s++) refCov.push(tileTestedCoverage(level, dim, tile, s))
    } else {
      for (let s = 0; s < STAMP_COUNT; s++) rescaleTileAlpha(level, dim, tile, s, refCov[s]!)
    }
    for (let i = 0; i < dim * dim * 4; i++) out[off + i] = Math.round(Math.min(1, Math.max(0, level[i]!)) * 255)
    off += dim * dim * 4
    if (l === STAMP_MIPS - 1) break
    level = downsampleTiles(level, dim, tile)
    tile >>= 1
  }

  return { data: out, coverage }
}

/** Bleed colour (not alpha) into empty texels so filtering never pulls black. */
function dilateTile(atlas: Float32Array, tx: number, ty: number): void {
  const filled = new Uint8Array(STAMP_TILE * STAMP_TILE)
  for (let v = 0; v < STAMP_TILE; v++) {
    for (let u = 0; u < STAMP_TILE; u++) {
      filled[v * STAMP_TILE + u] = atlas[((ty + v) * STAMP_ATLAS + tx + u) * 4 + 3]! > 0 ? 1 : 0
    }
  }
  for (let pass = 0; pass < 4; pass++) {
    const next = filled.slice()
    for (let v = 0; v < STAMP_TILE; v++) {
      for (let u = 0; u < STAMP_TILE; u++) {
        if (filled[v * STAMP_TILE + u] !== 0) continue
        let n = 0
        const acc = [0, 0, 0]
        for (let dv = -1; dv <= 1; dv++) {
          for (let du = -1; du <= 1; du++) {
            const vv = v + dv
            const uu = u + du
            if (vv < 0 || uu < 0 || vv >= STAMP_TILE || uu >= STAMP_TILE) continue
            if (filled[vv * STAMP_TILE + uu] === 0) continue
            const s4 = ((ty + vv) * STAMP_ATLAS + tx + uu) * 4
            for (let c = 0; c < 3; c++) acc[c]! += atlas[s4 + c]!
            n++
          }
        }
        if (n === 0) continue
        const d4 = ((ty + v) * STAMP_ATLAS + tx + u) * 4
        for (let c = 0; c < 3; c++) atlas[d4 + c] = acc[c]! / n
        next[v * STAMP_TILE + u] = 1
      }
    }
    filled.set(next)
  }
}

function tileTestedCoverage(level: Float32Array, dim: number, tile: number, s: number): number {
  const tx = (s % STAMP_GRID) * tile
  const ty = Math.floor(s / STAMP_GRID) * tile
  let n = 0
  for (let v = 0; v < tile; v++) {
    for (let u = 0; u < tile; u++) {
      if (level[((ty + v) * dim + tx + u) * 4 + 3]! >= STAMP_ALPHA_REF) n++
    }
  }
  return n / (tile * tile)
}

function rescaleTileAlpha(level: Float32Array, dim: number, tile: number, s: number, target: number): void {
  const tx = (s % STAMP_GRID) * tile
  const ty = Math.floor(s / STAMP_GRID) * tile
  let lo = 0.05
  let hi = 20
  const fraction = (k: number): number => {
    let n = 0
    for (let v = 0; v < tile; v++) {
      for (let u = 0; u < tile; u++) {
        if (level[((ty + v) * dim + tx + u) * 4 + 3]! * k >= STAMP_ALPHA_REF) n++
      }
    }
    return n / (tile * tile)
  }
  for (let it = 0; it < 14; it++) {
    const mid = (lo + hi) * 0.5
    if (fraction(mid) < target) lo = mid
    else hi = mid
  }
  const k = (lo + hi) * 0.5
  for (let v = 0; v < tile; v++) {
    for (let u = 0; u < tile; u++) {
      const i = ((ty + v) * dim + tx + u) * 4 + 3
      level[i] = Math.min(1, level[i]! * k)
    }
  }
}

function downsampleTiles(level: Float32Array, dim: number, tile: number): Float32Array<ArrayBuffer> {
  const nd = dim >> 1
  const nt = tile >> 1
  const out = new Float32Array(nd * nd * 4)
  for (let s = 0; s < STAMP_COUNT; s++) {
    const sx = (s % STAMP_GRID) * tile
    const sy = Math.floor(s / STAMP_GRID) * tile
    const dx = (s % STAMP_GRID) * nt
    const dy = Math.floor(s / STAMP_GRID) * nt
    for (let v = 0; v < nt; v++) {
      for (let u = 0; u < nt; u++) {
        let aSum = 0
        const acc = [0, 0, 0]
        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 2; i++) {
            const s4 = ((sy + v * 2 + j) * dim + sx + u * 2 + i) * 4
            const a = level[s4 + 3]!
            aSum += a
            for (let c = 0; c < 3; c++) acc[c]! += level[s4 + c]! * Math.max(a, 1e-3)
          }
        }
        const d4 = ((dy + v) * nd + dx + u) * 4
        const w = Math.max(aSum, 4e-3)
        for (let c = 0; c < 3; c++) out[d4 + c] = acc[c]! / w
        out[d4 + 3] = aSum / 4
      }
    }
  }
  return out
}
