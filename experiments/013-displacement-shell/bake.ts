import type { GcMesh } from '@harness'

/**
 * Bake a GCMESH1 source mesh into a strand vector-displacement field ("VDF1").
 *
 * The runtime geometry is a single flat sheet of patches; these textures are
 * WHERE the plant surfaces are. We reconstruct the plant statistically as S
 * "strands" (blade-like ribbons): each strand is a polyline of T stations
 * traced through the point-sampled surface density of the source mesh,
 * carrying position offset, ribbon half-width, mean normal, mean color and a
 * baked vertical-occlusion factor per station. Rows are ordered so that any
 * prefix of strands is a spatially balanced, importance-sorted subset — the
 * runtime LOD is literally "use the first N rows".
 *
 * Texture layout (T wide × S tall, one layer per species):
 *   posW : xyz = offset from plant origin (mesh metres), w = ribbon half-width
 *   norAo: xyz = mean surface normal,                    w = occlusion [0..1]
 *   color: rgb = mean albedo,                            w = 1
 */

export const BAKE_VERSION = 1
export const STRANDS = 96
export const STATIONS = 16
const MAGIC = 0x31464456 // 'VDF1'
const POINT_TARGET = 320_000
const FLOATS_PER_STATION = 12

export interface StrandField {
  strands: number
  stations: number
  topH: number
  /** Mean albedo of the upper canopy (far-shell color). */
  canopy: [number, number, number]
  /** S×T×12 floats: [ox,oy,oz,halfW, nx,ny,nz,ao, r,g,b,1] */
  data: Float32Array
}

// ---------------------------------------------------------------------------
// Point sampling (area-weighted over all triangles)
// ---------------------------------------------------------------------------

interface PointCloud {
  n: number
  px: Float32Array
  py: Float32Array
  pz: Float32Array
  cr: Float32Array
  cg: Float32Array
  cb: Float32Array
  nx: Float32Array
  ny: Float32Array
  nz: Float32Array
}

function octDecode(u16u: number, u16v: number, out: [number, number, number]): void {
  const u = (u16u / 65535) * 2 - 1
  const v = (u16v / 65535) * 2 - 1
  let x = u
  let z = v
  const y = 1 - Math.abs(u) - Math.abs(v)
  if (y < 0) {
    x = (1 - Math.abs(v)) * Math.sign(u)
    z = (1 - Math.abs(u)) * Math.sign(v)
  }
  const len = Math.hypot(x, y, z) || 1
  out[0] = x / len
  out[1] = y / len
  out[2] = z / len
}

/** Deterministic LCG so bakes are reproducible. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

function samplePoints(mesh: GcMesh, cx: number, cz: number): PointCloud {
  const h = mesh.header
  const verts = mesh.vertices
  const tris = mesh.triangles
  const triCount = h.triangleCount
  const q = 1 / 65535
  const e0 = (h.boundsMax[0] - h.boundsMin[0]) * q
  const e1 = (h.boundsMax[1] - h.boundsMin[1]) * q
  const e2 = (h.boundsMax[2] - h.boundsMin[2]) * q
  const m0 = h.boundsMin[0] - cx
  const m1 = h.boundsMin[1]
  const m2 = h.boundsMin[2] - cz

  // Pass 1: total area.
  let totalArea = 0
  for (let t = 0; t < triCount; t++) {
    const b = t * 4
    const i0 = tris[b]! * 8
    const i1 = tris[b + 1]! * 8
    const i2 = tris[b + 2]! * 8
    const ax = (verts[i1]! - verts[i0]!) * e0
    const ay = (verts[i1 + 1]! - verts[i0 + 1]!) * e1
    const az = (verts[i1 + 2]! - verts[i0 + 2]!) * e2
    const bx = (verts[i2]! - verts[i0]!) * e0
    const by = (verts[i2 + 1]! - verts[i0 + 1]!) * e1
    const bz = (verts[i2 + 2]! - verts[i0 + 2]!) * e2
    const cxx = ay * bz - az * by
    const cyy = az * bx - ax * bz
    const czz = ax * by - ay * bx
    totalArea += 0.5 * Math.sqrt(cxx * cxx + cyy * cyy + czz * czz)
  }

  const cap = POINT_TARGET + 4096
  const out: PointCloud = {
    n: 0,
    px: new Float32Array(cap),
    py: new Float32Array(cap),
    pz: new Float32Array(cap),
    cr: new Float32Array(cap),
    cg: new Float32Array(cap),
    cb: new Float32Array(cap),
    nx: new Float32Array(cap),
    ny: new Float32Array(cap),
    nz: new Float32Array(cap),
  }
  const perArea = POINT_TARGET / Math.max(totalArea, 1e-9)
  const rng = makeRng(0x9e3779b9)
  const nrm: [number, number, number] = [0, 0, 0]
  let acc = 0

  for (let t = 0; t < triCount && out.n < cap; t++) {
    const b = t * 4
    const i0 = tris[b]! * 8
    const i1 = tris[b + 1]! * 8
    const i2 = tris[b + 2]! * 8
    const p0x = m0 + verts[i0]! * e0
    const p0y = m1 + verts[i0 + 1]! * e1
    const p0z = m2 + verts[i0 + 2]! * e2
    const p1x = m0 + verts[i1]! * e0
    const p1y = m1 + verts[i1 + 1]! * e1
    const p1z = m2 + verts[i1 + 2]! * e2
    const p2x = m0 + verts[i2]! * e0
    const p2y = m1 + verts[i2 + 1]! * e1
    const p2z = m2 + verts[i2 + 2]! * e2
    const ax = p1x - p0x
    const ay = p1y - p0y
    const az = p1z - p0z
    const bx = p2x - p0x
    const by = p2y - p0y
    const bz = p2z - p0z
    const cxx = ay * bz - az * by
    const cyy = az * bx - ax * bz
    const czz = ax * by - ay * bx
    const area = 0.5 * Math.sqrt(cxx * cxx + cyy * cyy + czz * czz)
    if (area === 0) continue
    acc += area * perArea
    const take = Math.floor(acc)
    if (take === 0) continue
    acc -= take

    const cr = (verts[i0 + 3]! + verts[i1 + 3]! + verts[i2 + 3]!) * q * (1 / 3)
    const cg = (verts[i0 + 4]! + verts[i1 + 4]! + verts[i2 + 4]!) * q * (1 / 3)
    const cb = (verts[i0 + 5]! + verts[i1 + 5]! + verts[i2 + 5]!) * q * (1 / 3)
    octDecode(verts[i0 + 6]!, verts[i0 + 7]!, nrm)
    let nvx = nrm[0]
    let nvy = nrm[1]
    let nvz = nrm[2]
    octDecode(verts[i1 + 6]!, verts[i1 + 7]!, nrm)
    nvx += nrm[0]
    nvy += nrm[1]
    nvz += nrm[2]
    octDecode(verts[i2 + 6]!, verts[i2 + 7]!, nrm)
    nvx += nrm[0]
    nvy += nrm[1]
    nvz += nrm[2]
    const nl = Math.hypot(nvx, nvy, nvz) || 1

    for (let k = 0; k < take && out.n < cap; k++) {
      let u = rng()
      let v = rng()
      if (u + v > 1) {
        u = 1 - u
        v = 1 - v
      }
      const w0 = 1 - u - v
      const i = out.n++
      out.px[i] = w0 * p0x + u * p1x + v * p2x
      out.py[i] = w0 * p0y + u * p1y + v * p2y
      out.pz[i] = w0 * p0z + u * p1z + v * p2z
      out.cr[i] = cr
      out.cg[i] = cg
      out.cb[i] = cb
      out.nx[i] = nvx / nl
      out.ny[i] = nvy / nl
      out.nz[i] = nvz / nl
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// Spatial hash grid over the point cloud
// ---------------------------------------------------------------------------

class PointGrid {
  readonly cell: number
  readonly minX: number
  readonly minY: number
  readonly minZ: number
  readonly dx: number
  readonly dy: number
  readonly dz: number
  private start: Uint32Array
  private items: Uint32Array

  constructor(private pts: PointCloud) {
    let minX = Infinity
    let minY = Infinity
    let minZ = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    let maxZ = -Infinity
    for (let i = 0; i < pts.n; i++) {
      minX = Math.min(minX, pts.px[i]!)
      minY = Math.min(minY, pts.py[i]!)
      minZ = Math.min(minZ, pts.pz[i]!)
      maxX = Math.max(maxX, pts.px[i]!)
      maxY = Math.max(maxY, pts.py[i]!)
      maxZ = Math.max(maxZ, pts.pz[i]!)
    }
    const ext = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.01)
    this.cell = Math.max(0.025, ext / 96)
    this.minX = minX
    this.minY = minY
    this.minZ = minZ
    this.dx = Math.max(1, Math.floor((maxX - minX) / this.cell) + 1)
    this.dy = Math.max(1, Math.floor((maxY - minY) / this.cell) + 1)
    this.dz = Math.max(1, Math.floor((maxZ - minZ) / this.cell) + 1)
    const nCells = this.dx * this.dy * this.dz
    const counts = new Uint32Array(nCells + 1)
    const cellOf = (i: number): number =>
      Math.min(this.dx - 1, Math.floor((pts.px[i]! - minX) / this.cell)) +
      this.dx *
        (Math.min(this.dy - 1, Math.floor((pts.py[i]! - minY) / this.cell)) +
          this.dy * Math.min(this.dz - 1, Math.floor((pts.pz[i]! - minZ) / this.cell)))
    for (let i = 0; i < pts.n; i++) counts[cellOf(i) + 1]!++
    for (let c = 0; c < nCells; c++) counts[c + 1]! += counts[c]!
    this.start = counts
    this.items = new Uint32Array(pts.n)
    const cursor = counts.slice(0, nCells)
    for (let i = 0; i < pts.n; i++) {
      const c = cellOf(i)
      this.items[cursor[c]!++] = i
    }
  }

  /** Visit point indices within `r` of (x,y,z). */
  query(x: number, y: number, z: number, r: number, visit: (i: number) => void): void {
    const p = this.pts
    const r2 = r * r
    const x0 = Math.max(0, Math.floor((x - r - this.minX) / this.cell))
    const y0 = Math.max(0, Math.floor((y - r - this.minY) / this.cell))
    const z0 = Math.max(0, Math.floor((z - r - this.minZ) / this.cell))
    const x1 = Math.min(this.dx - 1, Math.floor((x + r - this.minX) / this.cell))
    const y1 = Math.min(this.dy - 1, Math.floor((y + r - this.minY) / this.cell))
    const z1 = Math.min(this.dz - 1, Math.floor((z + r - this.minZ) / this.cell))
    for (let cz = z0; cz <= z1; cz++) {
      for (let cy = y0; cy <= y1; cy++) {
        for (let cx = x0; cx <= x1; cx++) {
          const c = cx + this.dx * (cy + this.dy * cz)
          const end = this.start[c + 1]!
          for (let k = this.start[c]!; k < end; k++) {
            const i = this.items[k]!
            const ddx = p.px[i]! - x
            const ddy = p.py[i]! - y
            const ddz = p.pz[i]! - z
            if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) visit(i)
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Strand tracing
// ---------------------------------------------------------------------------

interface Gather {
  count: number
  mx: number
  my: number
  mz: number
  cr: number
  cg: number
  cb: number
  nx: number
  ny: number
  nz: number
  /** RMS distance of members to the axis through (x,y,z) along (dx,dy,dz). */
  rms: number
}

function gatherAt(
  grid: PointGrid,
  pts: PointCloud,
  x: number,
  y: number,
  z: number,
  r: number,
  dx: number,
  dy: number,
  dz: number,
): Gather {
  const g: Gather = { count: 0, mx: 0, my: 0, mz: 0, cr: 0, cg: 0, cb: 0, nx: 0, ny: 0, nz: 0, rms: 0 }
  grid.query(x, y, z, r, (i) => {
    g.count++
    g.mx += pts.px[i]!
    g.my += pts.py[i]!
    g.mz += pts.pz[i]!
    g.cr += pts.cr[i]!
    g.cg += pts.cg[i]!
    g.cb += pts.cb[i]!
    g.nx += pts.nx[i]!
    g.ny += pts.ny[i]!
    g.nz += pts.nz[i]!
    const ox = pts.px[i]! - x
    const oy = pts.py[i]! - y
    const oz = pts.pz[i]! - z
    const along = ox * dx + oy * dy + oz * dz
    const perp2 = ox * ox + oy * oy + oz * oz - along * along
    g.rms += Math.max(perp2, 0)
  })
  if (g.count > 0) {
    const inv = 1 / g.count
    g.mx *= inv
    g.my *= inv
    g.mz *= inv
    g.cr *= inv
    g.cg *= inv
    g.cb *= inv
    g.rms = Math.sqrt(g.rms * inv)
    const nl = Math.hypot(g.nx, g.ny, g.nz) || 1
    g.nx /= nl
    g.ny /= nl
    g.nz /= nl
  }
  return g
}

export function bakeStrandField(mesh: GcMesh, tileSize: [number, number] | null): StrandField {
  const h = mesh.header
  const cx = tileSize ? h.tileOrigin[0] + tileSize[0] / 2 : (h.boundsMin[0] + h.boundsMax[0]) / 2
  const cz = tileSize ? h.tileOrigin[1] + tileSize[1] / 2 : (h.boundsMin[2] + h.boundsMax[2]) / 2
  const topH = h.topH > 0.05 ? h.topH : h.boundsMax[1]

  const pts = samplePoints(mesh, cx, cz)
  const grid = new PointGrid(pts)
  const rng = makeRng(0x51ed270b)

  // Roots: dart-thrown low points with a relaxing min XZ separation.
  const ys = Float32Array.from(pts.py.subarray(0, pts.n)).sort()
  const yLow = ys[Math.floor(pts.n * 0.3)]!
  const lowIdx: number[] = []
  for (let i = 0; i < pts.n; i++) if (pts.py[i]! <= yLow) lowIdx.push(i)
  const footprint = tileSize ? Math.min(tileSize[0], tileSize[1]) : Math.min(h.boundsMax[0] - h.boundsMin[0], h.boundsMax[2] - h.boundsMin[2])
  let minSep = Math.sqrt((footprint * footprint) / STRANDS) * 0.7
  const rootX: number[] = []
  const rootY: number[] = []
  const rootZ: number[] = []
  let fails = 0
  while (rootX.length < STRANDS && lowIdx.length > 0) {
    const i = lowIdx[Math.floor(rng() * lowIdx.length)]!
    const x = pts.px[i]!
    const z = pts.pz[i]!
    let ok = true
    for (let k = 0; k < rootX.length; k++) {
      const dx = rootX[k]! - x
      const dz = rootZ[k]! - z
      if (dx * dx + dz * dz < minSep * minSep) {
        ok = false
        break
      }
    }
    if (ok) {
      rootX.push(x)
      rootY.push(pts.py[i]!)
      rootZ.push(z)
      fails = 0
    } else if (++fails > 40) {
      minSep *= 0.82
      fails = 0
    }
  }

  // Trace each strand upward through the point density.
  const step = (topH / (STATIONS - 1)) * 0.95
  const rGather = Math.max(0.03, topH * 0.032)
  const nStrands = rootX.length
  const raw = new Float32Array(nStrands * STATIONS * FLOATS_PER_STATION)
  const importance = new Float32Array(nStrands)

  for (let s = 0; s < nStrands; s++) {
    let x = rootX[s]!
    let y = rootY[s]!
    let z = rootZ[s]!
    let dx = 0
    let dy = 1
    let dz = 0
    let alive = true
    let lastW = 0.004
    for (let t = 0; t < STATIONS; t++) {
      const base = (s * STATIONS + t) * FLOATS_PER_STATION
      if (t > 0 && alive) {
        const tx = x + dx * step
        const ty = y + dy * step
        const tz = z + dz * step
        let g = gatherAt(grid, pts, tx, ty, tz, rGather, dx, dy, dz)
        if (g.count < 6) g = gatherAt(grid, pts, tx, ty, tz, rGather * 1.7, dx, dy, dz)
        if (g.count < 4) {
          alive = false
        } else {
          // Pull toward the local surface centroid, capped to keep arc-length.
          let px = g.mx
          let py = g.my
          let pz = g.mz
          const ox = px - tx
          const oy = py - ty
          const oz = pz - tz
          const od = Math.hypot(ox, oy, oz)
          const cap = step * 0.8
          if (od > cap) {
            px = tx + (ox / od) * cap
            py = ty + (oy / od) * cap
            pz = tz + (oz / od) * cap
          }
          let ndx = px - x
          let ndy = py - y
          let ndz = pz - z
          const nl = Math.hypot(ndx, ndy, ndz) || 1
          ndx /= nl
          ndy /= nl
          ndz /= nl
          ndy = Math.max(ndy, 0.15)
          const rl = Math.hypot(ndx, ndy, ndz)
          dx = ndx / rl
          dy = ndy / rl
          dz = ndz / rl
          x = px
          y = py
          z = pz
        }
      }
      const g = gatherAt(grid, pts, x, y, z, t === 0 ? rGather * 1.4 : rGather, dx, dy, dz)
      const hasPts = g.count >= 3
      let w = hasPts ? Math.min(Math.max(1.15 * g.rms, 0.0025), 0.075) : 0
      if (!alive && t > 0) w = 0 // tapered dead tail
      if (t === STATIONS - 1) w *= 0.4
      if (t === 0) w *= 0.85
      const hN = Math.min(Math.max(y / topH, 0), 1)
      raw[base] = x
      raw[base + 1] = y
      raw[base + 2] = z
      raw[base + 3] = w
      raw[base + 4] = hasPts ? g.nx : 0
      raw[base + 5] = hasPts ? g.ny : 1
      raw[base + 6] = hasPts ? g.nz : 0
      raw[base + 7] = 0.35 + 0.65 * Math.pow(hN, 0.75)
      raw[base + 8] = hasPts ? g.cr : 0.2
      raw[base + 9] = hasPts ? g.cg : 0.25
      raw[base + 10] = hasPts ? g.cb : 0.1
      raw[base + 11] = 1
      importance[s] = importance[s]! + w * step
      if (!alive && t > 0) {
        // Freeze remaining stations at the tip (zero-width tail).
        for (let tt = t + 1; tt < STATIONS; tt++) {
          const bb = (s * STATIONS + tt) * FLOATS_PER_STATION
          raw.copyWithin(bb, base, base + FLOATS_PER_STATION)
          raw[bb + 3] = 0
        }
        break
      }
    }
  }

  // Order strands: interleave 4 spatial quadrants, each sorted by importance,
  // so every prefix is a balanced high-importance subset (the LOD contract).
  const quadrant = (s: number): number => (rootX[s]! >= 0 ? 1 : 0) + (rootZ[s]! >= 0 ? 2 : 0)
  const buckets: number[][] = [[], [], [], []]
  for (let s = 0; s < nStrands; s++) buckets[quadrant(s)]!.push(s)
  for (const b of buckets) b.sort((a, bb) => importance[bb]! - importance[a]!)
  const order: number[] = []
  for (let k = 0; order.length < nStrands; k++) {
    const b = buckets[k % 4]!
    const idx = Math.floor(k / 4)
    if (idx < b.length) order.push(b[idx]!)
  }

  const data = new Float32Array(STRANDS * STATIONS * FLOATS_PER_STATION)
  for (let s = 0; s < STRANDS; s++) {
    const src = order[s % Math.max(nStrands, 1)] ?? 0
    data.set(
      raw.subarray(src * STATIONS * FLOATS_PER_STATION, (src + 1) * STATIONS * FLOATS_PER_STATION),
      s * STATIONS * FLOATS_PER_STATION,
    )
  }

  // Canopy mean albedo (upper half, width-weighted) for the far shell.
  let wr = 0
  let wg = 0
  let wb = 0
  let wsum = 0
  for (let s = 0; s < STRANDS; s++) {
    for (let t = 0; t < STATIONS; t++) {
      const b = (s * STATIONS + t) * FLOATS_PER_STATION
      if (data[b + 1]! < topH * 0.45) continue
      const w = data[b + 3]!
      wr += data[b + 8]! * w
      wg += data[b + 9]! * w
      wb += data[b + 10]! * w
      wsum += w
    }
  }
  const inv = wsum > 0 ? 1 / wsum : 0
  return {
    strands: STRANDS,
    stations: STATIONS,
    topH,
    canopy: wsum > 0 ? [wr * inv, wg * inv, wb * inv] : [0.2, 0.26, 0.11],
    data,
  }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

export function serializeField(f: StrandField): ArrayBuffer {
  const buf = new ArrayBuffer(64 + f.data.byteLength)
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, 16)
  u32[0] = MAGIC
  u32[1] = BAKE_VERSION
  u32[2] = f.strands
  u32[3] = f.stations
  f32[4] = f.topH
  f32[5] = f.canopy[0]
  f32[6] = f.canopy[1]
  f32[7] = f.canopy[2]
  new Float32Array(buf, 64).set(f.data)
  return buf
}

/** True iff buf is a valid VDF1 artifact of the current version and shape. */
export function isField(buf: ArrayBuffer | null): buf is ArrayBuffer {
  if (!buf || buf.byteLength < 64) return false
  const u32 = new Uint32Array(buf, 0, 16)
  if (u32[0] !== MAGIC || u32[1] !== BAKE_VERSION) return false
  if (u32[2] !== STRANDS || u32[3] !== STATIONS) return false
  return buf.byteLength === 64 + STRANDS * STATIONS * FLOATS_PER_STATION * 4
}

export function parseField(buf: ArrayBuffer): StrandField {
  if (!isField(buf)) throw new Error('VDF1: bad artifact')
  const f32 = new Float32Array(buf, 0, 16)
  return {
    strands: STRANDS,
    stations: STATIONS,
    topH: f32[4]!,
    canopy: [f32[5]!, f32[6]!, f32[7]!],
    data: new Float32Array(buf, 64),
  }
}

const f16Scratch = new Float32Array(1)
const f16Bits = new Uint32Array(f16Scratch.buffer)

/** Scalar f32 -> f16 bits (round-to-nearest-even not needed at our precision). */
export function toF16(v: number): number {
  const f32 = f16Scratch
  const u32 = f16Bits
  f32[0] = v
  const x = u32[0]!
  const sign = (x >>> 16) & 0x8000
  let exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0)
  const e = exp - 127 + 15
  if (e >= 31) return sign | 0x7c00
  if (e <= 0) {
    if (e < -10) return sign
    mant |= 0x800000
    const shift = 14 - e
    return sign | (mant >> shift)
  }
  return sign | (e << 10) | (mant >> 13)
}
