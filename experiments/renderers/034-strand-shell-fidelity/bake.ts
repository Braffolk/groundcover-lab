import type { GcMesh } from '@harness'

/**
 * BLD1 — a REAL BLADE LIBRARY distilled from a GCMESH1 source mesh.
 *
 * 013-displacement-shell traced 96 "strands" upward from dart-thrown low
 * points with a ~4 cm gather radius and an RMS width. At that radius a gather
 * spans ~10 neighbouring blades, so every strand is the *average* of a bundle
 * (a 4-8 cm paddle), never a blade. That is exactly the fidelity that this
 * bake recovers:
 *
 *  1. TIP SEEDING. A leaf tip is the one point on a blade that is
 *     unambiguous: its local neighbourhood sits entirely on one side of it.
 *     We score every point by "extremality" (distance from its own local
 *     centroid), suppress non-maxima on a fine grid, and stratify the
 *     survivors over 4 xz quadrants x 3 height bands so the understory is
 *     sampled as well as the canopy.
 *  2. TIP -> BASE WALK WITH A LOCAL FRAME. From each tip we walk *down* the
 *     blade with a blade-scale gather radius (5-25 mm, adapted to the measured
 *     width). At every step a weighted 3x3 covariance of the gathered points
 *     is decomposed by power iteration + deflation into
 *        T = tangent (largest axis), B = width axis, N = blade-plane normal.
 *     So every station carries a REAL ORIENTED FRAME with the blade's own
 *     twist and curvature — the thing 013 threw away by making ribbons
 *     camera-facing.
 *  3. GAP-DETECTED WIDTH. Half-width is not an RMS. Perpendicular offsets
 *     inside a thin slab are sorted and walked outward until a 3 mm gap
 *     appears: that is the real blade edge, and a neighbouring blade in the
 *     gather cannot inflate it.
 *  4. REAL AO. Neighbour-point density in a 5 cm ball (periodically wrapped
 *     for community tiles) instead of a fake height curve, so the inside of a
 *     tuft is genuinely darker than an exposed tip.
 *  5. SILHOUETTE CALIBRATION. 014-ribbon-skeleton's headline failure was
 *     coverage 4x short of ground truth. We rasterize the source point cloud
 *     and the reconstructed ribbon set from 4 azimuths and bisect one global
 *     width multiplier until the ribbon silhouette area matches the mesh's.
 *     The number ends up in the artifact header and in NOTES.md.
 *
 * Texture layout (STATIONS wide x BLADES tall, one layer per species index):
 *   posW : xyz = offset from plant origin (m), w = half-width (m)
 *   norAo: xyz = blade-plane normal,           w = ambient occlusion [0..1]
 *   colFl: rgb = mean albedo,                  w = isotropy ("fluff") [0..1]
 */

export const BAKE_VERSION = 10
/**
 * Calamagrostis and elymus leaves run 4-10 mm wide, poa 2-5 mm, and a
 * calamagrostis spikelet ~1 mm. So a half-width of 6 mm is already the widest
 * leaf these species have.
 * Inside a fluffy panicle the gap detector finds no gap at all (the volume is
 * dense in every direction), so without a physical cap the width runs away to
 * the gather radius and the plume rebuilds 013's origami paddles. Coverage
 * that a cap costs us must be bought with MORE BLADES, never wider ones.
 */
const MAX_HALF_WIDTH = 0.007
const MAX_HALF_WIDTH_CALIBRATED = 0.006
/**
 * ...and inside a panicle it is far tighter still. A calamagrostis spikelet is
 * ~1 mm across; the reference billboard bake of the same mesh reads as fine
 * pink hairs over green stems, not tan flakes. The cap therefore tightens with
 * the measured isotropy, so a plume becomes MANY THIN strands (see-through,
 * with the stems visible behind them) instead of a few opaque plates. `min`,
 * not a multiplier: a genuinely thin culm keeps its own measured width.
 */
const MAX_HALF_WIDTH_FLUFF = 0.0016
/**
 * Baked blade rows. The reference source meshes carry 203+54 leaves (elymus),
 * 120+18 leaves + 369 panicle branches (poa) and a 2.2 M-tri panicle
 * (calamagrostis), so a few hundred traced blades is the same order as the
 * real organ count rather than a 96-strand statistical proxy.
 */
export const BLADES = 384
export const STATIONS = 16
/** Rows are emitted round-robin over this many spatial buckets (see order()). */
export const BUCKETS = 8
const MAGIC = 0x31444c42 // 'BLD1'
const FLOATS_PER_STATION = 12
const POINT_TARGET = 400_000
const HEADER_BYTES = 64

export interface BladeField {
  blades: number
  stations: number
  topH: number
  /** Width-weighted mean albedo of the upper canopy (far-shell colour). */
  canopy: [number, number, number]
  /** Global width multiplier that matched the mesh silhouette area. */
  widthCalib: number
  /** BLADES*STATIONS*12 floats: [x,y,z,halfW, nx,ny,nz,ao, r,g,b,fluff] */
  data: Float32Array
}

// ---------------------------------------------------------------------------
// Point sampling (area-weighted over every triangle of the source mesh)
// ---------------------------------------------------------------------------

interface PointCloud {
  n: number
  px: Float32Array
  py: Float32Array
  pz: Float32Array
  cr: Float32Array
  cg: Float32Array
  cb: Float32Array
}

/** Deterministic LCG so bakes are byte-reproducible. */
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
    const c0 = ay * bz - az * by
    const c1 = az * bx - ax * bz
    const c2 = ax * by - ay * bx
    totalArea += 0.5 * Math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
  }

  const cap = POINT_TARGET + 8192
  const out: PointCloud = {
    n: 0,
    px: new Float32Array(cap),
    py: new Float32Array(cap),
    pz: new Float32Array(cap),
    cr: new Float32Array(cap),
    cg: new Float32Array(cap),
    cb: new Float32Array(cap),
  }
  const perArea = POINT_TARGET / Math.max(totalArea, 1e-9)
  const rng = makeRng(0x9e3779b9)
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
    const c0 = ay * bz - az * by
    const c1 = az * bx - ax * bz
    const c2 = ax * by - ay * bx
    const area = 0.5 * Math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    if (area === 0) continue
    acc += area * perArea
    const take = Math.floor(acc)
    if (take === 0) continue
    acc -= take

    const cr = (verts[i0 + 3]! + verts[i1 + 3]! + verts[i2 + 3]!) * q * (1 / 3)
    const cg = (verts[i0 + 4]! + verts[i1 + 4]! + verts[i2 + 4]!) * q * (1 / 3)
    const cb = (verts[i0 + 5]! + verts[i1 + 5]! + verts[i2 + 5]!) * q * (1 / 3)

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
    }
  }
  // No vertex normals are read: they come from PCA, not the mesh, because a
  // two-sided thin blade averages its own normals to ~zero.
  return out
}

// ---------------------------------------------------------------------------
// Spatial hash grid, optionally periodic in xz (community tiles)
// ---------------------------------------------------------------------------

/** Fills scratch arrays with the neighbourhood of a query point. */
interface Gathered {
  n: number
  x: Float32Array
  y: Float32Array
  z: Float32Array
  idx: Int32Array
}

const GATHER_CAP = 1536

class PointGrid {
  readonly cell: number
  private minX: number
  private minY: number
  private minZ: number
  private maxX: number
  private maxZ: number
  private dx: number
  private dy: number
  private dz: number
  private start: Uint32Array
  private items: Uint32Array
  /** Tile size for periodic wrapping, or 0 for a finite specimen. */
  private tx = 0
  private tz = 0

  constructor(
    private pts: PointCloud,
    cellSize: number,
    tile: [number, number] | null,
  ) {
    if (tile) {
      this.tx = tile[0]
      this.tz = tile[1]
    }
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
    this.cell = cellSize
    this.minX = minX
    this.minY = minY
    this.minZ = minZ
    this.maxX = maxX
    this.maxZ = maxZ
    this.dx = Math.max(1, Math.floor((maxX - minX) / cellSize) + 1)
    this.dy = Math.max(1, Math.floor((maxY - minY) / cellSize) + 1)
    this.dz = Math.max(1, Math.floor((maxZ - minZ) / cellSize) + 1)
    const nCells = this.dx * this.dy * this.dz
    const counts = new Uint32Array(nCells + 1)
    for (let i = 0; i < pts.n; i++) counts[this.cellOf(i) + 1]!++
    for (let c = 0; c < nCells; c++) counts[c + 1]! += counts[c]!
    this.start = counts
    this.items = new Uint32Array(pts.n)
    const cursor = counts.slice(0, nCells)
    for (let i = 0; i < pts.n; i++) this.items[cursor[this.cellOf(i)]!++] = i
  }

  private cellOf(i: number): number {
    const p = this.pts
    return (
      Math.min(this.dx - 1, Math.floor((p.px[i]! - this.minX) / this.cell)) +
      this.dx *
        (Math.min(this.dy - 1, Math.floor((p.py[i]! - this.minY) / this.cell)) +
          this.dy * Math.min(this.dz - 1, Math.floor((p.pz[i]! - this.minZ) / this.cell)))
    )
  }

  /** One non-periodic box scan; `ox/oz` is added to every reported position. */
  private scan(g: Gathered, qx: number, qy: number, qz: number, r: number, ox: number, oz: number): void {
    const p = this.pts
    const r2 = r * r
    const sx = qx - ox
    const sz = qz - oz
    const x0 = Math.max(0, Math.floor((sx - r - this.minX) / this.cell))
    const y0 = Math.max(0, Math.floor((qy - r - this.minY) / this.cell))
    const z0 = Math.max(0, Math.floor((sz - r - this.minZ) / this.cell))
    const x1 = Math.min(this.dx - 1, Math.floor((sx + r - this.minX) / this.cell))
    const y1 = Math.min(this.dy - 1, Math.floor((qy + r - this.minY) / this.cell))
    const z1 = Math.min(this.dz - 1, Math.floor((sz + r - this.minZ) / this.cell))
    for (let cz = z0; cz <= z1; cz++) {
      for (let cy = y0; cy <= y1; cy++) {
        const rowBase = this.dx * (cy + this.dy * cz)
        for (let cx = x0; cx <= x1; cx++) {
          const c = cx + rowBase
          const end = this.start[c + 1]!
          for (let k = this.start[c]!; k < end; k++) {
            if (g.n >= GATHER_CAP) return
            const i = this.items[k]!
            const ex = p.px[i]! + ox
            const ey = p.py[i]!
            const ez = p.pz[i]! + oz
            const ddx = ex - qx
            const ddy = ey - qy
            const ddz = ez - qz
            if (ddx * ddx + ddy * ddy + ddz * ddz > r2) continue
            const j = g.n++
            g.x[j] = ex
            g.y[j] = ey
            g.z[j] = ez
            g.idx[j] = i
          }
        }
      }
    }
  }

  /**
   * Neighbours within `r`. For a periodic community tile the eight wrapped
   * copies are scanned too, so a blade crossing the tile seam keeps its points
   * and edge AO is not artificially bright.
   */
  query(g: Gathered, qx: number, qy: number, qz: number, r: number): void {
    g.n = 0
    if (this.tx <= 0) {
      this.scan(g, qx, qy, qz, r, 0, 0)
      return
    }
    for (let i = -1; i <= 1; i++) {
      const ox = i * this.tx
      if (qx - ox + r < this.minX || qx - ox - r > this.maxX) continue
      for (let j = -1; j <= 1; j++) {
        const oz = j * this.tz
        if (qz - oz + r < this.minZ || qz - oz - r > this.maxZ) continue
        this.scan(g, qx, qy, qz, r, ox, oz)
      }
    }
  }
}

function makeGathered(): Gathered {
  return {
    n: 0,
    x: new Float32Array(GATHER_CAP),
    y: new Float32Array(GATHER_CAP),
    z: new Float32Array(GATHER_CAP),
    idx: new Int32Array(GATHER_CAP),
  }
}

// ---------------------------------------------------------------------------
// Symmetric 3x3 covariance: power iteration + deflation
// ---------------------------------------------------------------------------

type Cov = Float64Array // [xx, yy, zz, xy, xz, yz]

function covMul(m: Cov, vx: number, vy: number, vz: number, out: Float64Array): void {
  out[0] = m[0]! * vx + m[3]! * vy + m[4]! * vz
  out[1] = m[3]! * vx + m[1]! * vy + m[5]! * vz
  out[2] = m[4]! * vx + m[5]! * vy + m[2]! * vz
}

const powScratch = new Float64Array(3)

/** Dominant eigenvector of `m`, seeded at (sx,sy,sz) and sign-aligned to it. */
function dominantAxis(m: Cov, sx: number, sy: number, sz: number, out: Float64Array): number {
  let vx = sx
  let vy = sy
  let vz = sz
  let lambda = 0
  for (let k = 0; k < 14; k++) {
    covMul(m, vx, vy, vz, powScratch)
    const len = Math.hypot(powScratch[0]!, powScratch[1]!, powScratch[2]!)
    if (len < 1e-18) break
    lambda = len
    vx = powScratch[0]! / len
    vy = powScratch[1]! / len
    vz = powScratch[2]! / len
  }
  if (vx * sx + vy * sy + vz * sz < 0) {
    vx = -vx
    vy = -vy
    vz = -vz
  }
  out[0] = vx
  out[1] = vy
  out[2] = vz
  return lambda
}

/** m -= lambda * a a^T (removes the axis so the next power iteration finds #2). */
function deflate(m: Cov, a: Float64Array, lambda: number): void {
  const ax = a[0]!
  const ay = a[1]!
  const az = a[2]!
  m[0]! -= lambda * ax * ax
  m[1]! -= lambda * ay * ay
  m[2]! -= lambda * az * az
  m[3]! -= lambda * ax * ay
  m[4]! -= lambda * ax * az
  m[5]! -= lambda * ay * az
}

// ---------------------------------------------------------------------------
// Blade tracing
// ---------------------------------------------------------------------------

interface TracedStation {
  x: number
  y: number
  z: number
  w: number
  nx: number
  ny: number
  nz: number
  r: number
  g: number
  b: number
  fluff: number
}

const axT = new Float64Array(3)
const axB = new Float64Array(3)
const cov: Cov = new Float64Array(6)
const perp = new Float64Array(GATHER_CAP)

/**
 * Half-width by GAP DETECTION: perpendicular offsets inside a thin slab are
 * sorted and walked outward until a `gap`-sized hole appears. A neighbouring
 * blade sitting 1 cm away therefore cannot inflate the width, which is exactly
 * how an RMS estimate turns 3 mm blades into 5 cm paddles.
 */
function gapWidth(
  g: Gathered,
  cx: number,
  cy: number,
  cz: number,
  tx: number,
  ty: number,
  tz: number,
  bx: number,
  by: number,
  bz: number,
  nx: number,
  ny: number,
  nz: number,
  slab: number,
  thick: number,
  gap: number,
  rMax: number,
): number {
  let m = 0
  for (let i = 0; i < g.n; i++) {
    const ox = g.x[i]! - cx
    const oy = g.y[i]! - cy
    const oz = g.z[i]! - cz
    if (Math.abs(ox * tx + oy * ty + oz * tz) > slab) continue
    if (Math.abs(ox * nx + oy * ny + oz * nz) > thick) continue
    perp[m++] = Math.abs(ox * bx + oy * by + oz * bz)
    if (m >= GATHER_CAP) break
  }
  if (m < 3) return 0
  const view = perp.subarray(0, m)
  view.sort()
  let edge = view[0]!
  for (let i = 1; i < m; i++) {
    const v = view[i]!
    if (v - edge > gap) break
    edge = v
  }
  return Math.min(Math.max(edge, 0.0006), rMax)
}

interface TraceParams {
  step: number
  maxSteps: number
  gapTol: number
  minLength: number
  /** Height above ground below which a near-horizontal run is a rhizome. */
  groundBand: number
}

function traceBlade(
  grid: PointGrid,
  pts: PointCloud,
  g: Gathered,
  tipX: number,
  tipY: number,
  tipZ: number,
  groundY: number,
  prm: TraceParams,
): TracedStation[] {
  // Initial direction: from the tip toward its own neighbourhood centroid,
  // i.e. down the blade.
  grid.query(g, tipX, tipY, tipZ, prm.step * 2.2)
  if (g.n < 4) return []
  let mx = 0
  let my = 0
  let mz = 0
  for (let i = 0; i < g.n; i++) {
    mx += g.x[i]!
    my += g.y[i]!
    mz += g.z[i]!
  }
  mx = mx / g.n - tipX
  my = my / g.n - tipY
  mz = mz / g.n - tipZ
  let dl = Math.hypot(mx, my, mz)
  if (dl < 1e-6) return []
  let dx = mx / dl
  let dy = my / dl
  let dz = mz / dl

  let x = tipX
  let y = tipY
  let z = tipZ
  let rLocal = prm.step * 1.1
  let nPrevX = 0
  let nPrevY = 1
  let nPrevZ = 0
  let havePrevN = false
  const path: TracedStation[] = []

  for (let s = 0; s < prm.maxSteps; s++) {
    const tX = x + dx * prm.step
    const tY = y + dy * prm.step
    const tZ = z + dz * prm.step
    grid.query(g, tX, tY, tZ, rLocal)
    if (g.n < 5) {
      grid.query(g, tX, tY, tZ, rLocal * 1.8)
      if (g.n < 5) break
    }

    // Weighted covariance about the gathered centroid.
    let cxm = 0
    let cym = 0
    let czm = 0
    for (let i = 0; i < g.n; i++) {
      cxm += g.x[i]!
      cym += g.y[i]!
      czm += g.z[i]!
    }
    cxm /= g.n
    cym /= g.n
    czm /= g.n
    cov.fill(0)
    for (let i = 0; i < g.n; i++) {
      const ox = g.x[i]! - cxm
      const oy = g.y[i]! - cym
      const oz = g.z[i]! - czm
      cov[0]! += ox * ox
      cov[1]! += oy * oy
      cov[2]! += oz * oz
      cov[3]! += ox * oy
      cov[4]! += ox * oz
      cov[5]! += oy * oz
    }
    const trace = cov[0]! + cov[1]! + cov[2]!
    if (trace < 1e-12) break
    const l0 = dominantAxis(cov, dx, dy, dz, axT)
    // Deflate and find the second axis -> the blade's width direction.
    deflate(cov, axT, l0)
    // Seed orthogonal to T (prefer the previous normal's cross product).
    let sx = nPrevY * axT[2]! - nPrevZ * axT[1]!
    let sy = nPrevZ * axT[0]! - nPrevX * axT[2]!
    let sz = nPrevX * axT[1]! - nPrevY * axT[0]!
    let sl = Math.hypot(sx, sy, sz)
    if (sl < 1e-6) {
      sx = axT[1]!
      sy = -axT[0]!
      sz = 0
      sl = Math.hypot(sx, sy, sz) || 1
    }
    const l1 = dominantAxis(cov, sx / sl, sy / sl, sz / sl, axB)
    const l2 = Math.max(trace - l0 - l1, 0)
    // Isotropy ("fluff"): a flat blade has l2 << l1; a panicle plume does not.
    const fluff = Math.min(1, Math.sqrt(l2 / Math.max(l1, 1e-12)) * 1.15)

    // N = B x T, re-orthogonalised and sign-propagated along the blade.
    let nx = axB[1]! * axT[2]! - axB[2]! * axT[1]!
    let ny = axB[2]! * axT[0]! - axB[0]! * axT[2]!
    let nz = axB[0]! * axT[1]! - axB[1]! * axT[0]!
    const nl = Math.hypot(nx, ny, nz) || 1
    nx /= nl
    ny /= nl
    nz /= nl
    const ref = havePrevN ? nx * nPrevX + ny * nPrevY + nz * nPrevZ : ny
    if (ref < 0) {
      nx = -nx
      ny = -ny
      nz = -nz
    }
    nPrevX = nx
    nPrevY = ny
    nPrevZ = nz
    havePrevN = true

    // Re-derive B so (T, B, N) is exactly orthonormal.
    let bx = axT[1]! * nz - axT[2]! * ny
    let by = axT[2]! * nx - axT[0]! * nz
    let bz = axT[0]! * ny - axT[1]! * nx
    const bl = Math.hypot(bx, by, bz) || 1
    bx /= bl
    by /= bl
    bz /= bl

    // Recentre on the local surface, but only perpendicular to the tangent so
    // arc length is preserved.
    const ox = cxm - tX
    const oy = cym - tY
    const oz = czm - tZ
    const along = ox * axT[0]! + oy * axT[1]! + oz * axT[2]!
    let px = ox - along * axT[0]!
    let py = oy - along * axT[1]!
    let pz = oz - along * axT[2]!
    const pl = Math.hypot(px, py, pz)
    const cap = rLocal * 0.75
    if (pl > cap) {
      px = (px / pl) * cap
      py = (py / pl) * cap
      pz = (pz / pl) * cap
    }
    const nX = tX + px
    const nY = tY + py
    const nZ = tZ + pz

    const w = gapWidth(
      g,
      nX,
      nY,
      nZ,
      axT[0]!,
      axT[1]!,
      axT[2]!,
      bx,
      by,
      bz,
      nx,
      ny,
      nz,
      prm.step * 0.7,
      Math.max(0.004, rLocal * 0.55),
      prm.gapTol,
      Math.min(rLocal, MAX_HALF_WIDTH),
    )

    // Mean albedo of the slab.
    let cr = 0
    let cg = 0
    let cb = 0
    let cn = 0
    for (let i = 0; i < g.n; i++) {
      const j = g.idx[i]!
      cr += pts.cr[j]!
      cg += pts.cg[j]!
      cb += pts.cb[j]!
      cn++
    }
    const inv = cn > 0 ? 1 / cn : 0

    path.push({
      x: nX,
      y: nY,
      z: nZ,
      w: w > 0 ? w : 0.0008,
      nx,
      ny,
      nz,
      r: cr * inv,
      g: cg * inv,
      b: cb * inv,
      fluff,
    })

    // Smooth the direction toward the measured tangent.
    dx = dx * 0.55 + axT[0]! * 0.45
    dy = dy * 0.55 + axT[1]! * 0.45
    dz = dz * 0.55 + axT[2]! * 0.45
    dl = Math.hypot(dx, dy, dz) || 1
    dx /= dl
    dy /= dl
    dz /= dl

    x = nX
    y = nY
    z = nZ
    rLocal = Math.min(Math.max(w * 2.6, prm.step * 0.7), prm.step * 2.0, 0.016)
    if (nY <= groundY + 0.004) break
  }

  // Length gate: a 2 cm stub is noise, not a blade.
  let len = 0
  let ySum = 0
  let yMin = Infinity
  let yMax = -Infinity
  for (let i = 0; i < path.length; i++) {
    const b = path[i]!
    ySum += b.y
    yMin = Math.min(yMin, b.y)
    yMax = Math.max(yMax, b.y)
    if (i > 0) {
      const a = path[i - 1]!
      len += Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z)
    }
  }
  if (len < prm.minLength || path.length < 3) return []
  // RHIZOME REJECT. Both community meshes model explicit rhizomes (elymus 12,
  // poa 4) — horizontal creepers that in the real plant lie in the soil. Traced
  // as blades they render as a tangle of brown noodles lying on the terrain,
  // which was the one clearly WRONG feature in the first close-ups. A run that
  // stays low and barely climbs is a rhizome, not a leaf.
  // A leaf rises: even a drooping one covers most of its arc length in Y. A
  // run that stays inside the litter band AND barely climbs is a creeper.
  if (yMax < groundY + prm.groundBand && yMax - yMin < len * 0.45) return []
  if (ySum / path.length < groundY + prm.groundBand * 0.35 && yMax - yMin < len * 0.6) return []
  return path
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/**
 * Reverse (tip-first -> base-first), optionally extend the base to the ground
 * so blades never float, and resample to STATIONS by arc length.
 */
function resample(path: TracedStation[], groundY: number): Float32Array {
  const src = path.slice().reverse() // now base -> tip
  const base = src[0]!
  if (base.y > groundY + 0.03 && base.y < groundY + 0.28) {
    // Short tapered stem down to the soil — real grass emerges from a tiller,
    // and a blade hanging in mid-air reads as a bug from every angle.
    src.unshift({
      ...base,
      y: groundY + 0.002,
      w: base.w * 0.75,
      nx: base.nx,
      ny: base.ny,
      nz: base.nz,
    })
  }

  const cum = new Float64Array(src.length)
  for (let i = 1; i < src.length; i++) {
    const a = src[i - 1]!
    const b = src[i]!
    cum[i] = cum[i - 1]! + Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z)
  }
  const total = cum[src.length - 1]!
  const out = new Float32Array(STATIONS * FLOATS_PER_STATION)
  let cursor = 0
  for (let t = 0; t < STATIONS; t++) {
    const target = (t / (STATIONS - 1)) * total
    while (cursor < src.length - 2 && cum[cursor + 1]! < target) cursor++
    const a = src[cursor]!
    const b = src[Math.min(cursor + 1, src.length - 1)]!
    const span = Math.max(cum[Math.min(cursor + 1, src.length - 1)]! - cum[cursor]!, 1e-9)
    const f = Math.min(Math.max((target - cum[cursor]!) / span, 0), 1)
    const o = t * FLOATS_PER_STATION
    const lerp = (u: number, v: number): number => u + (v - u) * f
    out[o] = lerp(a.x, b.x)
    out[o + 1] = lerp(a.y, b.y)
    out[o + 2] = lerp(a.z, b.z)
    out[o + 3] = lerp(a.w, b.w)
    let nx = lerp(a.nx, b.nx)
    let ny = lerp(a.ny, b.ny)
    let nz = lerp(a.nz, b.nz)
    const nl = Math.hypot(nx, ny, nz) || 1
    nx /= nl
    ny /= nl
    nz /= nl
    out[o + 4] = nx
    out[o + 5] = ny
    out[o + 6] = nz
    out[o + 7] = 1 // AO filled in later
    out[o + 8] = lerp(a.r, b.r)
    out[o + 9] = lerp(a.g, b.g)
    out[o + 10] = lerp(a.b, b.b)
    out[o + 11] = lerp(a.fluff, b.fluff)
  }
  // Taper the very tip so blades end in a point, not a chisel.
  out[(STATIONS - 1) * FLOATS_PER_STATION + 3]! *= 0.25
  out[(STATIONS - 2) * FLOATS_PER_STATION + 3]! *= 0.7
  return out
}

// ---------------------------------------------------------------------------
// Silhouette calibration (the fix for 014's 4x coverage shortfall)
// ---------------------------------------------------------------------------

const CAL_RES = 192
const CAL_VIEWS = 4

/**
 * The one place blade half-width is finalized: scale by the calibration
 * multiplier, then clamp to a physical maximum that tightens with isotropy.
 * Used both inside the calibration rasterizer and on the stored data, so the
 * bisection solves for exactly the geometry that ships.
 */
function finalWidth(raw: number, fluff: number, mul: number): number {
  const cap = MAX_HALF_WIDTH_CALIBRATED + (MAX_HALF_WIDTH_FLUFF - MAX_HALF_WIDTH_CALIBRATED) * Math.min(fluff, 1)
  return Math.min(raw * mul, cap)
}

interface CalView {
  rx: number
  rz: number
}

function calViews(): CalView[] {
  const out: CalView[] = []
  for (let v = 0; v < CAL_VIEWS; v++) {
    const a = (v / CAL_VIEWS) * Math.PI
    out.push({ rx: Math.cos(a), rz: Math.sin(a) })
  }
  return out
}

interface CalFrame {
  u0: number
  v0: number
  scale: number
}

function calFrame(pts: PointCloud, topH: number): CalFrame {
  let ext = 0
  for (let i = 0; i < pts.n; i++) ext = Math.max(ext, Math.abs(pts.px[i]!), Math.abs(pts.pz[i]!))
  const half = Math.max(ext * 1.05, 0.05)
  const height = Math.max(topH * 1.05, 0.1)
  const span = Math.max(half * 2, height)
  return { u0: -span / 2, v0: -0.02, scale: CAL_RES / span }
}

function meshCoverage(pts: PointCloud, f: CalFrame, views: CalView[]): number {
  let hit = 0
  const mask = new Uint8Array(CAL_RES * CAL_RES)
  for (const view of views) {
    mask.fill(0)
    for (let i = 0; i < pts.n; i++) {
      const u = (pts.px[i]! * view.rx + pts.pz[i]! * view.rz - f.u0) * f.scale
      const v = (pts.py[i]! - f.v0) * f.scale
      const iu = u | 0
      const iv = v | 0
      if (iu < 0 || iv < 0 || iu >= CAL_RES || iv >= CAL_RES) continue
      const c = iv * CAL_RES + iu
      if (mask[c] === 0) {
        mask[c] = 1
        hit++
      }
    }
  }
  return hit / (CAL_RES * CAL_RES * views.length)
}

function ribbonCoverage(
  data: Float32Array,
  count: number,
  widthMul: number,
  f: CalFrame,
  views: CalView[],
): number {
  let hit = 0
  const mask = new Uint8Array(CAL_RES * CAL_RES)
  const stride = STATIONS * FLOATS_PER_STATION
  for (const view of views) {
    mask.fill(0)
    // The camera looks along (-rz, 0, rx); the ribbon's projected half-extent
    // is its half-width times |B . right| in the view basis.
    for (let s = 0; s < count; s++) {
      const b0 = s * stride
      for (let t = 0; t + 1 < STATIONS; t++) {
        const oa = b0 + t * FLOATS_PER_STATION
        const ob = oa + FLOATS_PER_STATION
        const ax = data[oa]!
        const ay = data[oa + 1]!
        const az = data[oa + 2]!
        const bx = data[ob]!
        const by = data[ob + 1]!
        const bz = data[ob + 2]!
        // Blade frame at the segment midpoint.
        let tx = bx - ax
        let ty = by - ay
        let tz = bz - az
        const tl = Math.hypot(tx, ty, tz) || 1
        tx /= tl
        ty /= tl
        tz /= tl
        const nx = data[oa + 4]!
        const ny = data[oa + 5]!
        const nz = data[oa + 6]!
        // width axis W = T x N
        const wx = ty * nz - tz * ny
        const wy = tz * nx - tx * nz
        const wz = tx * ny - ty * nx
        // The width axis projected into the view plane — a ribbon lying flat
        // on its side must lose its coverage, that is the whole point of
        // baked orientation, so the calibration has to see the same thing.
        const pu = wx * view.rx + wz * view.rz
        const pv = wy
        const plen = Math.hypot(pu, pv)
        if (plen < 1e-9) continue
        const du = pu / plen
        const dv = pv / plen
        const wA = finalWidth(data[oa + 3]!, data[oa + 11]!, widthMul)
        const wB = finalWidth(data[ob + 3]!, data[ob + 11]!, widthMul)
        const steps = Math.max(2, Math.ceil((tl * f.scale) / 0.7))
        for (let k = 0; k <= steps; k++) {
          const gt = k / steps
          const px = ax + (bx - ax) * gt
          const py = ay + (by - ay) * gt
          const pz = az + (bz - az) * gt
          const hwPix = (wA + (wB - wA) * gt) * plen * f.scale
          const u = (px * view.rx + pz * view.rz - f.u0) * f.scale
          const v = (py - f.v0) * f.scale
          const across = Math.max(1, Math.ceil(hwPix))
          for (let q = -across; q <= across; q++) {
            const t2 = (q / across) * hwPix
            const iu = (u + du * t2) | 0
            const iv = (v + dv * t2) | 0
            if (iu < 0 || iv < 0 || iu >= CAL_RES || iv >= CAL_RES) continue
            const c = iv * CAL_RES + iu
            if (mask[c] === 0) {
              mask[c] = 1
              hit++
            }
          }
        }
      }
    }
  }
  return hit / (CAL_RES * CAL_RES * views.length)
}

/** In-place 1-2-1 pass over the station axis for `count` floats at `off`. */
function smoothChannel(data: Float32Array, blade: number, stride: number, off: number, count: number, mix: number): void {
  const tmp = new Float32Array(STATIONS * count)
  for (let t = 0; t < STATIONS; t++) {
    const a = blade * stride + Math.max(t - 1, 0) * FLOATS_PER_STATION + off
    const b = blade * stride + t * FLOATS_PER_STATION + off
    const c = blade * stride + Math.min(t + 1, STATIONS - 1) * FLOATS_PER_STATION + off
    for (let k = 0; k < count; k++) {
      const smoothed = 0.25 * data[a + k]! + 0.5 * data[b + k]! + 0.25 * data[c + k]!
      tmp[t * count + k] = data[b + k]! + (smoothed - data[b + k]!) * mix
    }
  }
  for (let t = 0; t < STATIONS; t++) {
    const b = blade * stride + t * FLOATS_PER_STATION + off
    for (let k = 0; k < count; k++) data[b + k] = tmp[t * count + k]!
  }
}

function smoothBlades(data: Float32Array, stride: number): void {
  for (let b = 0; b < BLADES; b++) {
    // Inside a panicle the neighbourhood is isotropic, so the PCA tangent is
    // near-arbitrary and the walk wanders: the strand comes out lumpy instead
    // of a clean radiating hair. Fluffy blades therefore get extra position
    // smoothing (a real spikelet branch IS straight); leaves keep their curve.
    let fluffSum = 0
    for (let t = 0; t < STATIONS; t++) fluffSum += data[b * stride + t * FLOATS_PER_STATION + 11]!
    const extra = Math.round((fluffSum / STATIONS) * 4)
    for (let pass = 0; pass < extra; pass++) smoothChannel(data, b, stride, 0, 3, 1)
    smoothChannel(data, b, stride, 0, 3, 0.55) // position (light: keep curvature)
    for (let pass = 0; pass < 2; pass++) {
      smoothChannel(data, b, stride, 3, 1, 1) // half-width
      smoothChannel(data, b, stride, 8, 3, 1) // albedo
      smoothChannel(data, b, stride, 11, 1, 1) // fluff
      smoothChannel(data, b, stride, 4, 3, 1) // normal
    }
    for (let t = 0; t < STATIONS; t++) {
      const o = b * stride + t * FLOATS_PER_STATION + 4
      const l = Math.hypot(data[o]!, data[o + 1]!, data[o + 2]!)
      if (l > 1e-6) {
        data[o] = data[o]! / l
        data[o + 1] = data[o + 1]! / l
        data[o + 2] = data[o + 2]! / l
      } else {
        data[o] = 0
        data[o + 1] = 1
        data[o + 2] = 0
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The bake
// ---------------------------------------------------------------------------

export function bakeBladeField(mesh: GcMesh, tileSize: [number, number] | null): BladeField {
  const h = mesh.header
  const cx = tileSize ? h.tileOrigin[0] + tileSize[0] / 2 : (h.boundsMin[0] + h.boundsMax[0]) / 2
  const cz = tileSize ? h.tileOrigin[1] + tileSize[1] / 2 : (h.boundsMin[2] + h.boundsMax[2]) / 2
  const topH = h.topH > 0.05 ? h.topH : h.boundsMax[1]
  const groundY = h.boundsMin[1]

  const pts = samplePoints(mesh, cx, cz)
  const gridCell = Math.max(0.008, topH * 0.009)
  const grid = new PointGrid(pts, gridCell, tileSize)
  const g = makeGathered()

  // -- 1. tip candidates ----------------------------------------------------
  const rTip = Math.max(0.012, topH * 0.014)
  const stride = 3
  const candIdx: number[] = []
  const candScore: number[] = []
  for (let i = 0; i < pts.n; i += stride) {
    grid.query(g, pts.px[i]!, pts.py[i]!, pts.pz[i]!, rTip)
    if (g.n < 5) continue
    let mx = 0
    let my = 0
    let mz = 0
    for (let k = 0; k < g.n; k++) {
      mx += g.x[k]!
      my += g.y[k]!
      mz += g.z[k]!
    }
    mx = mx / g.n - pts.px[i]!
    my = my / g.n - pts.py[i]!
    mz = mz / g.n - pts.pz[i]!
    const score = Math.hypot(mx, my, mz) / rTip
    if (score < 0.22) continue
    candIdx.push(i)
    candScore.push(score)
  }

  // -- 2. non-maximum suppression on a fine grid, stratified ----------------
  const nmsCell = Math.max(0.012, topH * 0.018)
  const best = new Map<number, number>() // cell -> candidate slot
  const key = (x: number, y: number, z: number): number => {
    const a = Math.floor(x / nmsCell) + 512
    const b = Math.floor(y / nmsCell) + 512
    const c = Math.floor(z / nmsCell) + 512
    return (a * 1024 + b) * 1024 + c
  }
  for (let s = 0; s < candIdx.length; s++) {
    const i = candIdx[s]!
    const k = key(pts.px[i]!, pts.py[i]!, pts.pz[i]!)
    const prev = best.get(k)
    if (prev === undefined || candScore[s]! > candScore[prev]!) best.set(k, s)
  }
  // Stratify over 4 xz quadrants x 3 height bands so the understory survives.
  const strata: number[][] = Array.from({ length: 12 }, () => [])
  const bandOf = (y: number): number => Math.min(2, Math.max(0, Math.floor((y / Math.max(topH, 1e-3)) * 3)))
  for (const s of best.values()) {
    const i = candIdx[s]!
    const quad = (pts.px[i]! >= 0 ? 1 : 0) + (pts.pz[i]! >= 0 ? 2 : 0)
    strata[quad * 3 + bandOf(pts.py[i]!)]!.push(s)
  }
  for (const st of strata) st.sort((a, b) => candScore[b]! - candScore[a]!)

  // -- 3. trace, round-robin over strata so every prefix stays balanced -----
  const prm: TraceParams = {
    step: Math.max(0.006, topH * 0.0068),
    maxSteps: 170,
    gapTol: 0.003,
    minLength: Math.max(0.035, topH * 0.05),
    groundBand: Math.max(0.08, topH * 0.2),
  }
  const stationsFloats = STATIONS * FLOATS_PER_STATION
  const traced: Float32Array[] = []
  const tracedKey: { quad: number; band: number; imp: number }[] = []
  const cursors = new Array<number>(12).fill(0)
  // Weighted round-robin. Area-weighted point sampling puts most of its points
  // (and therefore most extremal "tips") inside the panicle, so an even split
  // spent ~a third of the library on plume strands and the meadow read as a
  // pink crust with the green understory buried. The two lower height bands are
  // visited twice as often, which is also the right call visually: a plume is
  // supposed to be see-through.
  const bandWeight = [2, 2, 1]
  const visitOrder: number[] = []
  for (let q = 0; q < 4; q++) {
    for (let b = 0; b < 3; b++) {
      for (let w = 0; w < bandWeight[b]!; w++) visitOrder.push(q * 3 + b)
    }
  }
  let exhausted = 0
  let sIdx = 0
  const traceTarget = Math.round(BLADES * 1.5)
  while (traced.length < traceTarget && exhausted < visitOrder.length) {
    const si = visitOrder[sIdx % visitOrder.length]!
    sIdx++
    const st = strata[si]!
    const c = cursors[si]!
    if (c >= st.length) {
      exhausted++
      continue
    }
    exhausted = 0
    cursors[si] = c + 1
    const i = candIdx[st[c]!]!
    const path = traceBlade(grid, pts, g, pts.px[i]!, pts.py[i]!, pts.pz[i]!, groundY, prm)
    if (path.length === 0) continue
    const row = resample(path, groundY)
    let imp = 0
    for (let t = 0; t + 1 < STATIONS; t++) {
      const oa = t * FLOATS_PER_STATION
      const ob = oa + FLOATS_PER_STATION
      const seg = Math.hypot(row[ob]! - row[oa]!, row[ob + 1]! - row[oa + 1]!, row[ob + 2]! - row[oa + 2]!)
      imp += seg * (row[oa + 3]! + row[ob + 3]!)
    }
    traced.push(row)
    tracedKey.push({
      quad: (pts.px[i]! >= 0 ? 1 : 0) + (pts.pz[i]! >= 0 ? 2 : 0),
      band: bandOf(pts.py[i]!),
      imp,
    })
  }

  if (traced.length === 0) throw new Error('BLD1: no blade survived tracing')

  // -- 4. order: round-robin over BUCKETS spatial buckets, each sorted by
  //       importance, so ANY window of consecutive rows is spatially balanced.
  const buckets: number[][] = Array.from({ length: BUCKETS }, () => [])
  for (let i = 0; i < traced.length; i++) {
    const k = tracedKey[i]!
    buckets[(k.quad * 2 + (k.band >= 2 ? 1 : 0)) % BUCKETS]!.push(i)
  }
  for (const b of buckets) b.sort((a, c) => tracedKey[c]!.imp - tracedKey[a]!.imp)
  const order: number[] = []
  for (let k = 0; order.length < traced.length; k++) {
    const b = buckets[k % BUCKETS]!
    const idx = Math.floor(k / BUCKETS)
    if (idx < b.length) order.push(b[idx]!)
    if (k > traced.length * BUCKETS + BUCKETS) break
  }

  const data = new Float32Array(BLADES * stationsFloats)
  for (let s = 0; s < BLADES; s++) {
    const src = traced[order[s % order.length]!]!
    data.set(src, s * stationsFloats)
  }

  // -- 4b. smooth each blade along its stations ----------------------------
  // The raw walk measures every station independently, so width and colour
  // jump between neighbours and the ribbon rasterizes with a notched,
  // torn-paper silhouette (very visible at 1 m). A 1-2-1 filter along the
  // station axis turns them back into leaves. Positions get ONE light pass so
  // real curvature and droop survive; width/colour/normal get two.
  smoothBlades(data, stationsFloats)

  // -- 5. real AO from neighbour density (periodically wrapped for tiles) ---
  const rAo = Math.max(0.035, topH * 0.045)
  const counts = new Float32Array(BLADES * STATIONS)
  for (let s = 0; s < BLADES; s++) {
    for (let t = 0; t < STATIONS; t++) {
      const o = s * stationsFloats + t * FLOATS_PER_STATION
      grid.query(g, data[o]!, data[o + 1]!, data[o + 2]!, rAo)
      counts[s * STATIONS + t] = g.n
    }
  }
  const sorted = Float32Array.from(counts).sort()
  const ref = Math.max(sorted[Math.floor(sorted.length * 0.8)]!, 1)
  for (let s = 0; s < BLADES; s++) {
    for (let t = 0; t < STATIONS; t++) {
      const o = s * stationsFloats + t * FLOATS_PER_STATION
      const occ = counts[s * STATIONS + t]! / ref
      data[o + 7] = Math.min(1, Math.exp(-0.85 * occ) * 1.35)
    }
  }

  // -- 6. silhouette calibration -------------------------------------------
  const views = calViews()
  const frame = calFrame(pts, topH)
  const target = meshCoverage(pts, frame, views)
  let lo = 0.4
  let hi = 14
  for (let it = 0; it < 16; it++) {
    const mid = Math.sqrt(lo * hi)
    if (ribbonCoverage(data, BLADES, mid, frame, views) < target) lo = mid
    else hi = mid
  }
  // Clamped hard. A runaway multiplier just rebuilds 013's paddles, and on
  // calamagrostis the target is unreachable by construction: a 2.2 M-tri
  // panicle's silhouette is near-solid at any raster resolution, while its
  // honest look is see-through. So the calibration is a NUDGE toward the
  // mesh's density, not a solver; the raw number goes into NOTES.md.
  const widthCalibRaw = Math.sqrt(lo * hi)
  const widthCalib = Math.min(Math.max(widthCalibRaw, 0.6), 1.8)
  let wSum = 0
  let fluffy = 0
  for (let s = 0; s < BLADES; s++) {
    for (let t = 0; t < STATIONS; t++) {
      const o = s * stationsFloats + t * FLOATS_PER_STATION + 3
      const w = finalWidth(data[o]!, data[o + 8]!, widthCalib)
      data[o] = w
      wSum += w
      if (data[o + 8]! > 0.6) fluffy++
    }
  }
  console.log(
    `[BLD1] traced ${traced.length} blades, widthCalib ${widthCalibRaw.toFixed(2)}` +
      ` -> ${widthCalib.toFixed(2)}, mean half-width ${((wSum / (BLADES * STATIONS)) * 1000).toFixed(2)}mm,` +
      ` fluff stations ${((fluffy / (BLADES * STATIONS)) * 100).toFixed(0)}%,` +
      ` mesh cover ${(target * 100).toFixed(1)}% vs ribbons ${(ribbonCoverage(data, BLADES, 1, frame, views) * 100).toFixed(1)}%`,
  )

  // -- 7. canopy colour for the far shell ----------------------------------
  let wr = 0
  let wg = 0
  let wb = 0
  let wsum = 0
  for (let s = 0; s < BLADES; s++) {
    for (let t = 0; t < STATIONS; t++) {
      const o = s * stationsFloats + t * FLOATS_PER_STATION
      if (data[o + 1]! < topH * 0.45) continue
      const w = data[o + 3]!
      wr += data[o + 8]! * w
      wg += data[o + 9]! * w
      wb += data[o + 10]! * w
      wsum += w
    }
  }
  const inv = wsum > 0 ? 1 / wsum : 0

  return {
    blades: BLADES,
    stations: STATIONS,
    topH,
    canopy: wsum > 0 ? [wr * inv, wg * inv, wb * inv] : [0.2, 0.26, 0.11],
    widthCalib,
    data,
  }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

export function serializeField(f: BladeField): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + f.data.byteLength)
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, 16)
  u32[0] = MAGIC
  u32[1] = BAKE_VERSION
  u32[2] = f.blades
  u32[3] = f.stations
  f32[4] = f.topH
  f32[5] = f.canopy[0]
  f32[6] = f.canopy[1]
  f32[7] = f.canopy[2]
  f32[8] = f.widthCalib
  new Float32Array(buf, HEADER_BYTES).set(f.data)
  return buf
}

/** True iff `buf` is a BLD1 artifact of exactly the current version and shape. */
export function isField(buf: ArrayBuffer | null): buf is ArrayBuffer {
  if (!buf || buf.byteLength < HEADER_BYTES) return false
  const u32 = new Uint32Array(buf, 0, 16)
  if (u32[0] !== MAGIC || u32[1] !== BAKE_VERSION) return false
  if (u32[2] !== BLADES || u32[3] !== STATIONS) return false
  return buf.byteLength === HEADER_BYTES + BLADES * STATIONS * FLOATS_PER_STATION * 4
}

export function parseField(buf: ArrayBuffer): BladeField {
  if (!isField(buf)) throw new Error('BLD1: bad artifact')
  const f32 = new Float32Array(buf, 0, 16)
  return {
    blades: BLADES,
    stations: STATIONS,
    topH: f32[4]!,
    canopy: [f32[5]!, f32[6]!, f32[7]!],
    widthCalib: f32[8]!,
    data: new Float32Array(buf, HEADER_BYTES),
  }
}

const f16Scratch = new Float32Array(1)
const f16Bits = new Uint32Array(f16Scratch.buffer)

/** Scalar f32 -> f16 bits (round-to-nearest not needed at our precision). */
export function toF16(v: number): number {
  f16Scratch[0] = v
  const x = f16Bits[0]!
  const sign = (x >>> 16) & 0x8000
  const exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0)
  const e = exp - 127 + 15
  if (e >= 31) return sign | 0x7c00
  if (e <= 0) {
    if (e < -10) return sign
    mant |= 0x800000
    return sign | (mant >> (14 - e))
  }
  return sign | (e << 10) | (mant >> 13)
}
