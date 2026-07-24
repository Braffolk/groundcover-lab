import type { GcMesh } from '@harness'

/**
 * Bake: source mesh (millions of tris) -> a few thousand anisotropic
 * "covariance splats", 16 bytes each, in 5 concatenated LOD levels.
 *
 * Method: vertices are sorted once along a 30-bit Morton curve, then each LOD
 * level chops the sorted order into K equal-count chunks. A chunk becomes one
 * splat: mean position/color, mean-normal blended with a radial "shell"
 * normal (stable when blade normals cancel out), and the covariance
 * eigenvectors give an elongation axis + two semi-radii — so blades bake into
 * slim oriented splats and fluffy heads into round ones. Coarser LODs are the
 * same curve with bigger chunks, so radii grow automatically.
 *
 * Record (16B = 4 u32):
 *   w0: posX u16 | posY u16 << 16          (quantized to header bounds)
 *   w1: posZ u16 | r8 << 16 | g8 << 24
 *   w2: b8 | colorVar8 << 8 | octNu8 << 16 | octNv8 << 24
 *   w3: octAu8 | octAv8 << 8 | ra8 << 16 | rb8 << 24   (radii sqrt-encoded)
 */

export const LOD_COUNTS = [2048, 384, 96, 24, 6] as const
export const LOD_LEVELS = LOD_COUNTS.length
export const LOD_OFFSETS = LOD_COUNTS.map((_, i) => LOD_COUNTS.slice(0, i).reduce((a, b) => a + b, 0))
export const TOTAL_SPLATS = LOD_COUNTS.reduce((a: number, b) => a + b, 0)

/** Per-(entry,lod) plant-record capacities for the cull buckets. */
export const BUCKET_CAPS = [1024, 4096, 16384, 32768, 131072] as const
export const BUCKET_BASES = BUCKET_CAPS.map((_, i) => BUCKET_CAPS.slice(0, i).reduce((a: number, b) => a + b, 0))
export const TOTAL_RECORDS = BUCKET_CAPS.reduce((a: number, b) => a + b, 0)

const MAGIC = 0x31505347 // 'GSP1'
const HEADER_BYTES = 80
const ACC_STRIDE = 20 // per-cluster accumulator doubles

export interface SplatBake {
  boundsMin: [number, number, number]
  boundsExt: [number, number, number]
  topH: number
  maxRa: number
  maxRb: number
  levelCounts: number[]
  /** 4 u32 per splat, levels concatenated in LOD_COUNTS order. */
  records: Uint32Array<ArrayBuffer>
}

function part1by2(x: number): number {
  let v = x & 0x3ff
  v = (v | (v << 16)) & 0x30000ff
  v = (v | (v << 8)) & 0x300f00f
  v = (v | (v << 4)) & 0x30c30c3
  v = (v | (v << 2)) & 0x9249249
  return v
}

function octEncode8(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1
  let px = x / s
  let pz = z / s
  if (y < 0) {
    const ox = px
    px = (1 - Math.abs(pz)) * (ox >= 0 ? 1 : -1)
    pz = (1 - Math.abs(ox)) * (pz >= 0 ? 1 : -1)
  }
  return [Math.max(0, Math.min(255, Math.round((px * 0.5 + 0.5) * 255))), Math.max(0, Math.min(255, Math.round((pz * 0.5 + 0.5) * 255)))]
}

/** Largest-eigenvalue direction of a symmetric 3x3 via power iteration. */
function principalAxis(
  cxx: number, cxy: number, cxz: number, cyy: number, cyz: number, czz: number,
): [number, number, number, number] {
  let vx = 1
  let vy = 0.31
  let vz = 0.7 // deliberately non-axis-aligned start
  let lambda = 0
  for (let it = 0; it < 14; it++) {
    const nx = cxx * vx + cxy * vy + cxz * vz
    const ny = cxy * vx + cyy * vy + cyz * vz
    const nz = cxz * vx + cyz * vy + czz * vz
    const len = Math.hypot(nx, ny, nz)
    if (len < 1e-12) return [1, 0, 0, 0]
    vx = nx / len
    vy = ny / len
    vz = nz / len
    lambda = len
  }
  return [vx, vy, vz, lambda]
}

export function bakeSpeciesSplats(mesh: GcMesh): ArrayBuffer {
  const h = mesh.header
  const n = h.vertexCount
  const verts = mesh.vertices
  const [bx, by, bz] = h.boundsMin
  const ex = h.boundsMax[0] - bx
  const ey = h.boundsMax[1] - by
  const ez = h.boundsMax[2] - bz
  // Center: periodic tiles rotate around the tile center; specimens around
  // their bounds center. Y stays absolute (ground = 0).
  const tiled = h.tileSize[0] > 0 && h.tileSize[1] > 0
  const cx = tiled ? h.tileOrigin[0] + h.tileSize[0] / 2 : bx + ex / 2
  const cz = tiled ? h.tileOrigin[1] + h.tileSize[1] / 2 : bz + ez / 2

  // --- Morton order over quantized positions (10 bits per axis) ------------
  const codes = new Uint32Array(n)
  for (let i = 0; i < n; i++) {
    const b = i * 8
    codes[i] = part1by2(verts[b]! >> 6) | (part1by2(verts[b + 1]! >> 6) << 1) | (part1by2(verts[b + 2]! >> 6) << 2)
  }
  let order = new Uint32Array(n)
  for (let i = 0; i < n; i++) order[i] = i
  let tmp = new Uint32Array(n)
  const counts = new Uint32Array(1024)
  for (let pass = 0; pass < 3; pass++) {
    const shift = pass * 10
    counts.fill(0)
    for (let i = 0; i < n; i++) counts[(codes[order[i]!]! >> shift) & 1023]!++
    let sum = 0
    for (let d = 0; d < 1024; d++) {
      const c = counts[d]!
      counts[d] = sum
      sum += c
    }
    for (let i = 0; i < n; i++) {
      const idx = order[i]!
      const d = (codes[idx]! >> shift) & 1023
      tmp[counts[d]!++] = idx
    }
    const swap = order
    order = tmp
    tmp = swap
  }

  // --- One decoding pass accumulating all LOD levels -----------------------
  const acc = LOD_COUNTS.map((k) => new Float64Array(k * ACC_STRIDE))
  const step = Math.max(1, Math.floor(n / 2_000_000))
  for (let r = 0; r < n; r += step) {
    const i = order[r]!
    const b = i * 8
    const px = bx + (verts[b]! / 65535) * ex - cx
    const py = by + (verts[b + 1]! / 65535) * ey
    const pz = bz + (verts[b + 2]! / 65535) * ez - cz
    const cr = verts[b + 3]! / 65535
    const cg = verts[b + 4]! / 65535
    const cb = verts[b + 5]! / 65535
    // Octahedral decode (y-up), matching GcMesh.normalAt.
    const ou = (verts[b + 6]! / 65535) * 2 - 1
    const ov = (verts[b + 7]! / 65535) * 2 - 1
    let nx = ou
    let nz = ov
    let ny = 1 - Math.abs(ou) - Math.abs(ov)
    if (ny < 0) {
      nx = (1 - Math.abs(ov)) * (ou >= 0 ? 1 : -1)
      nz = (1 - Math.abs(ou)) * (ov >= 0 ? 1 : -1)
    }
    const nl = Math.hypot(nx, ny, nz) || 1
    nx /= nl
    ny /= nl
    nz /= nl
    const lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
    for (let L = 0; L < LOD_LEVELS; L++) {
      const k = LOD_COUNTS[L]!
      const a = acc[L]!
      const o = Math.min(k - 1, Math.floor((r * k) / n)) * ACC_STRIDE
      a[o]! += 1
      a[o + 1]! += px
      a[o + 2]! += py
      a[o + 3]! += pz
      a[o + 4]! += px * px
      a[o + 5]! += px * py
      a[o + 6]! += px * pz
      a[o + 7]! += py * py
      a[o + 8]! += py * pz
      a[o + 9]! += pz * pz
      a[o + 10]! += cr
      a[o + 11]! += cg
      a[o + 12]! += cb
      a[o + 13]! += nx
      a[o + 14]! += ny
      a[o + 15]! += nz
      a[o + 16]! += lum
      a[o + 17]! += lum * lum
    }
  }

  // --- Reduce clusters to splat parameters ---------------------------------
  interface Cluster {
    x: number; y: number; z: number
    r: number; g: number; b: number
    cvar: number
    nx: number; ny: number; nz: number
    ax: number; ay: number; az: number
    ra: number; rb: number
  }
  const KR = 1.9
  const clusters: Cluster[] = []
  for (let L = 0; L < LOD_LEVELS; L++) {
    const k = LOD_COUNTS[L]!
    const a = acc[L]!
    for (let c = 0; c < k; c++) {
      const o = c * ACC_STRIDE
      const m = a[o]! || 1
      const mx = a[o + 1]! / m
      const my = a[o + 2]! / m
      const mz = a[o + 3]! / m
      const cxx = a[o + 4]! / m - mx * mx
      const cxy = a[o + 5]! / m - mx * my
      const cxz = a[o + 6]! / m - mx * mz
      const cyy = a[o + 7]! / m - my * my
      const cyz = a[o + 8]! / m - my * mz
      const czz = a[o + 9]! / m - mz * mz
      const [e1x, e1y, e1z, l1] = principalAxis(cxx, cxy, cxz, cyy, cyz, czz)
      // Deflate and find the second axis.
      const d = l1
      const [e2x, e2y, e2z, l2raw] = principalAxis(
        cxx - d * e1x * e1x, cxy - d * e1x * e1y, cxz - d * e1x * e1z,
        cyy - d * e1y * e1y, cyz - d * e1y * e1z, czz - d * e1z * e1z,
      )
      void e2x; void e2y; void e2z
      const l2 = Math.max(0, Math.min(l2raw, l1))
      const ra = Math.min(0.6, Math.max(0.006, KR * Math.sqrt(Math.max(l1, 0))))
      const rb = Math.min(0.6, Math.max(0.005, KR * Math.sqrt(l2)))
      // Normal: mean vertex normal when coherent, radial shell when noisy.
      const mnx = a[o + 13]! / m
      const mny = a[o + 14]! / m
      const mnz = a[o + 15]! / m
      const coh = Math.hypot(mnx, mny, mnz)
      let shx = mx
      let shz = mz
      const shl = Math.hypot(shx, shz)
      if (shl < 1e-4) {
        shx = 0
        shz = 0
      } else {
        shx /= shl
        shz /= shl
      }
      let shy = 0.6
      const sl = Math.hypot(shx, shy, shz) || 1
      const w = Math.min(coh * 1.4, 0.8)
      let fnx = (shx / sl) * (1 - w) + (coh > 1e-4 ? (mnx / coh) * w : 0)
      let fny = (shy / sl) * (1 - w) + (coh > 1e-4 ? (mny / coh) * w : w)
      let fnz = (shz / sl) * (1 - w) + (coh > 1e-4 ? (mnz / coh) * w : 0)
      const fnl = Math.hypot(fnx, fny, fnz) || 1
      fnx /= fnl
      fny /= fnl
      fnz /= fnl
      const meanLum = a[o + 16]! / m
      const lumVar = Math.sqrt(Math.max(a[o + 17]! / m - meanLum * meanLum, 0))
      clusters.push({
        x: mx, y: my, z: mz,
        r: a[o + 10]! / m, g: a[o + 11]! / m, b: a[o + 12]! / m,
        cvar: Math.min(1, lumVar * 2.5),
        nx: fnx, ny: fny, nz: fnz,
        ax: e1x, ay: e1y, az: e1z,
        ra, rb,
      })
    }
  }

  // --- Quantize into the GSP1 artifact -------------------------------------
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
  let maxRa = 0
  let maxRb = 0
  for (const c of clusters) {
    minX = Math.min(minX, c.x); minY = Math.min(minY, c.y); minZ = Math.min(minZ, c.z)
    maxX = Math.max(maxX, c.x); maxY = Math.max(maxY, c.y); maxZ = Math.max(maxZ, c.z)
    maxRa = Math.max(maxRa, c.ra)
    maxRb = Math.max(maxRb, c.rb)
  }
  const extX = Math.max(maxX - minX, 1e-5)
  const extY = Math.max(maxY - minY, 1e-5)
  const extZ = Math.max(maxZ - minZ, 1e-5)

  const out = new ArrayBuffer(HEADER_BYTES + TOTAL_SPLATS * 16)
  const dv = new DataView(out)
  dv.setUint32(0, MAGIC, true)
  dv.setUint32(4, 1, true)
  dv.setUint32(8, LOD_LEVELS, true)
  dv.setUint32(12, TOTAL_SPLATS, true)
  dv.setFloat32(16, minX, true)
  dv.setFloat32(20, minY, true)
  dv.setFloat32(24, minZ, true)
  dv.setFloat32(28, maxX, true)
  dv.setFloat32(32, maxY, true)
  dv.setFloat32(36, maxZ, true)
  dv.setFloat32(40, h.boundsMax[1], true) // topH used for wind height fraction
  dv.setFloat32(44, maxRa, true)
  dv.setFloat32(48, maxRb, true)
  LOD_COUNTS.forEach((k, i) => dv.setUint32(52 + i * 4, k, true))

  const rec = new Uint32Array(out, HEADER_BYTES, TOTAL_SPLATS * 4)
  const q16 = (v: number): number => Math.max(0, Math.min(65535, Math.round(v * 65535)))
  const q8 = (v: number): number => Math.max(0, Math.min(255, Math.round(v * 255)))
  clusters.forEach((c, i) => {
    const px = q16((c.x - minX) / extX)
    const py = q16((c.y - minY) / extY)
    const pz = q16((c.z - minZ) / extZ)
    const [nu, nv] = octEncode8(c.nx, c.ny, c.nz)
    const [au, av] = octEncode8(c.ax, c.ay, c.az)
    const ra8 = q8(Math.sqrt(c.ra / maxRa))
    const rb8 = q8(Math.sqrt(c.rb / maxRb))
    rec[i * 4] = px | (py << 16)
    rec[i * 4 + 1] = pz | (q8(c.r) << 16) | (q8(c.g) << 24)
    rec[i * 4 + 2] = q8(c.b) | (q8(c.cvar) << 8) | (nu << 16) | (nv << 24)
    rec[i * 4 + 3] = au | (av << 8) | (ra8 << 16) | (rb8 << 24)
  })
  return out
}

/**
 * The dev server answers missing /mesh/baked/ files with the SPA index.html
 * (status 200), so artifact bytes must be validated before trusting them.
 */
export function isValidSplatBake(data: ArrayBuffer): boolean {
  if (data.byteLength < HEADER_BYTES) return false
  const dv = new DataView(data)
  return dv.getUint32(0, true) === MAGIC && dv.getUint32(4, true) === 1
}

export function parseSplatBake(data: ArrayBuffer): SplatBake {
  const dv = new DataView(data)
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('GSP1: bad magic')
  if (dv.getUint32(4, true) !== 1) throw new Error('GSP1: unsupported version')
  const levels = dv.getUint32(8, true)
  const total = dv.getUint32(12, true)
  const minX = dv.getFloat32(16, true)
  const minY = dv.getFloat32(20, true)
  const minZ = dv.getFloat32(24, true)
  const levelCounts: number[] = []
  for (let i = 0; i < levels; i++) levelCounts.push(dv.getUint32(52 + i * 4, true))
  return {
    boundsMin: [minX, minY, minZ],
    boundsExt: [
      Math.max(dv.getFloat32(28, true) - minX, 1e-5),
      Math.max(dv.getFloat32(32, true) - minY, 1e-5),
      Math.max(dv.getFloat32(36, true) - minZ, 1e-5),
    ],
    topH: dv.getFloat32(40, true),
    maxRa: dv.getFloat32(44, true),
    maxRb: dv.getFloat32(48, true),
    levelCounts,
    records: new Uint32Array(data, HEADER_BYTES, total * 4),
  }
}
