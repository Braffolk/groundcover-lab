/**
 * Ribbon-skeleton distiller.
 *
 * Distills an arbitrary-complexity GCMESH1 source mesh into a handful of
 * curved ribbons per plant. No connectivity is used (the meshes range from
 * one giant welded component to 1.7M micro-fragments) — instead we trace
 * STRANDS directly through the vertex point cloud:
 *
 *   1. pick a tuft center (a low vertex for community tiles; the origin for
 *      single specimens) and gather the local point cloud,
 *   2. repeatedly seed at the most prominent still-uncovered point (tips
 *      score high) and walk a curve down/inward through the cloud with a
 *      direction-cone neighbor filter, giving a centerline polyline,
 *   3. stamp the tube around the polyline as covered; the stamped points
 *      yield per-bin width (PCA across the local tangent), twist azimuth and
 *      mean authored color,
 *   4. leftover uncovered high points (fluffy panicles) are binned into a few
 *      fat azimuth-sector "plume" ribbons,
 *   5. one extra "aggregate" ribbon stores the whole plant's vertical
 *      silhouette profile — the far-LOD card.
 *
 * Output: per species, VARIANTS plant variants x (RIBBONS + 1 aggregate)
 * rows x K_COLS samples, packed into two tiny rgba16f-bound textures:
 *   T0 = centerline xyz (m, plant-local) + half-width
 *   T1 = authored rgb color + lateral azimuth (rad)
 * The GPU's bilinear filter along the K axis is the curve interpolator.
 *
 * This file is dependency-free (no @harness imports) so it can also run
 * under node for offline tuning.
 */

export const ATLAS_MAGIC = 0x4b534252 // 'RBSK'
export const ATLAS_VERSION = 3
export const K_COLS = 12
export const RIBBONS = 48
export const ROWS_PER_VARIANT = RIBBONS + 1 // last row = aggregate silhouette
export const VARIANTS = 4

/** Structural subset of GcMesh that bake needs (node-testable). */
export interface MeshLike {
  vertexCount: number
  boundsMin: [number, number, number]
  boundsMax: [number, number, number]
  /** Periodic tile size or [0,0] for finite specimens. */
  tileSize: [number, number]
  topH: number
  /** Interleaved u16 records: x y z r g b octU octV. */
  vertices: Uint16Array
}

export interface DistillOptions {
  seed: number
  /** XZ radius of the cloud gathered around a tuft center (tiles). */
  cloudRadius: number
  maxCloudPoints: number
  traceStep: number
  neighborRadius: number
  coverRadius: number
  strands: number
  plumes: number
  widthScale: number
  /** Smallest-eigenvalue share of the local covariance trace above which a
   *  cell counts as fluff (thin sheets/lines score near zero). */
  fluffThreshold: number
  /** Grid cell size (m) — the scale of the thin-vs-fluff classifier. */
  cellSize: number
  /** Fluff clusters above this size are split into azimuth sectors. */
  maxClusterPoints: number
  /** Log distillation statistics to the console. */
  debug?: boolean
}

export const DEFAULT_OPTIONS: DistillOptions = {
  seed: 1,
  cloudRadius: 0.3,
  maxCloudPoints: 140000,
  traceStep: 0.014,
  neighborRadius: 0.024,
  coverRadius: 0.013,
  strands: RIBBONS - 8,
  plumes: 8,
  widthScale: 1.0,
  fluffThreshold: 0.09,
  cellSize: 0.02,
  maxClusterPoints: 1500,
}

/** Per-species tuning on top of the defaults. */
export const SPECIES_OPTIONS: Record<string, Partial<DistillOptions>> = {
  'calamagrostis-canescens': {},
  'elymus-repens': {},
  'poa-pratensis': {
    cloudRadius: 0.25,
    traceStep: 0.01,
    neighborRadius: 0.017,
    coverRadius: 0.009,
    cellSize: 0.012,
    strands: RIBBONS - 12,
    plumes: 12,
  },
}

// ---------------------------------------------------------------------------
// Small deterministic PRNG (bake-side only).
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ---------------------------------------------------------------------------
// Point cloud with a uniform grid for radius queries.
// ---------------------------------------------------------------------------

interface Cloud {
  n: number
  pos: Float32Array // 3n, plant-local
  col: Float32Array // 3n
  stamp: Int32Array // -1 = uncovered, else ribbon id
  /** 1 = locally line-like (blade/stem), 0 = fluff (panicle). */
  linear: Uint8Array
  cellSize: number
  grid: Map<number, number[]>
}

function gridKey(cs: number, x: number, y: number, z: number): number {
  const ix = Math.floor(x / cs) + 512
  const iy = Math.floor(y / cs) + 512
  const iz = Math.floor(z / cs) + 512
  return ix + iy * 1024 + iz * 1048576
}

function buildCloud(pos: Float32Array, col: Float32Array, cellSize: number): Cloud {
  const n = pos.length / 3
  const grid = new Map<number, number[]>()
  for (let i = 0; i < n; i++) {
    const k = gridKey(cellSize, pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!)
    let arr = grid.get(k)
    if (!arr) {
      arr = []
      grid.set(k, arr)
    }
    arr.push(i)
  }
  return { n, pos, col, stamp: new Int32Array(n).fill(-1), linear: new Uint8Array(n).fill(1), cellSize, grid }
}

/**
 * Classify points as THIN (blade/stem surfaces — locally 1D or 2D) vs FLUFF
 * (panicle — locally 3D), per grid cell: for each occupied cell, take the
 * covariance of the points in its 3x3x3 cell block and measure the smallest
 * eigenvalue's share of the trace. Thin sheets/lines have a near-zero
 * smallest eigenvalue; volumetric fluff does not.
 */
function classifyCloud(cloud: Cloud, threshold: number): void {
  const cs = cloud.cellSize
  for (const [key, members] of cloud.grid) {
    const kx = key & 1023
    const ky = (key >>> 10) & 1023
    const kz = key >>> 20
    let sx = 0
    let sy = 0
    let sz = 0
    let sxx = 0
    let syy = 0
    let szz = 0
    let sxy = 0
    let syz = 0
    let sxz = 0
    let cnt = 0
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const arr = cloud.grid.get(kx + dx + (ky + dy) * 1024 + (kz + dz) * 1048576)
          if (!arr) continue
          for (const i of arr) {
            const x = cloud.pos[i * 3]!
            const y = cloud.pos[i * 3 + 1]!
            const z = cloud.pos[i * 3 + 2]!
            sx += x
            sy += y
            sz += z
            sxx += x * x
            syy += y * y
            szz += z * z
            sxy += x * y
            syz += y * z
            sxz += x * z
            cnt++
          }
        }
      }
    }
    if (cnt < 8) continue // sparse: keep default thin
    const mx = sx / cnt
    const my = sy / cnt
    const mz = sz / cnt
    const cxx = sxx / cnt - mx * mx
    const cyy = syy / cnt - my * my
    const czz = szz / cnt - mz * mz
    const cxy = sxy / cnt - mx * my
    const cyz = syz / cnt - my * mz
    const cxz = sxz / cnt - mx * mz
    const trace = cxx + cyy + czz
    if (trace < 1e-10) continue
    // smallest eigenvalue via power iteration on (trace*I - C)
    let vx = 1
    let vy = 0.5
    let vz = 0.25
    for (let it = 0; it < 8; it++) {
      const nx = (trace - cxx) * vx - cxy * vy - cxz * vz
      const ny = -cxy * vx + (trace - cyy) * vy - cyz * vz
      const nz = -cxz * vx - cyz * vy + (trace - czz) * vz
      const l = Math.hypot(nx, ny, nz) || 1
      vx = nx / l
      vy = ny / l
      vz = nz / l
    }
    const dom =
      vx * ((trace - cxx) * vx - cxy * vy - cxz * vz) +
      vy * (-cxy * vx + (trace - cyy) * vy - cyz * vz) +
      vz * (-cxz * vx - cyz * vy + (trace - czz) * vz)
    const lMin = trace - dom
    const thin = lMin / trace < threshold
    if (!thin) {
      for (const i of members) cloud.linear[i] = 0
    }
    void cs
  }
}

function forNeighbors(cloud: Cloud, x: number, y: number, z: number, r: number, cb: (i: number) => void): void {
  const cs = cloud.cellSize
  const r2 = r * r
  const x0 = Math.floor((x - r) / cs)
  const x1 = Math.floor((x + r) / cs)
  const y0 = Math.floor((y - r) / cs)
  const y1 = Math.floor((y + r) / cs)
  const z0 = Math.floor((z - r) / cs)
  const z1 = Math.floor((z + r) / cs)
  for (let iy = y0; iy <= y1; iy++) {
    for (let iz = z0; iz <= z1; iz++) {
      for (let ix = x0; ix <= x1; ix++) {
        const arr = cloud.grid.get(ix + 512 + (iy + 512) * 1024 + (iz + 512) * 1048576)
        if (!arr) continue
        for (const i of arr) {
          const dx = cloud.pos[i * 3]! - x
          const dy = cloud.pos[i * 3 + 1]! - y
          const dz = cloud.pos[i * 3 + 2]! - z
          if (dx * dx + dy * dy + dz * dz <= r2) cb(i)
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Ribbon fitting
// ---------------------------------------------------------------------------

/** One fitted ribbon: K samples of centerline/halfwidth/color/azimuth. */
export interface Ribbon {
  /** K*4: x y z halfWidth */
  posw: Float32Array
  /** K*4: r g b azimuth */
  cola: Float32Array
  importance: number
}

function emptyRibbon(): Ribbon {
  return { posw: new Float32Array(K_COLS * 4), cola: new Float32Array(K_COLS * 4), importance: 0 }
}

/**
 * Fit K bins along a given per-point parameter using an explicit centerline.
 * `center` is K*3 (already resampled); points contribute width (PCA across
 * the tangent), color and azimuth to their bin.
 */
function fitBins(cloud: Cloud, ids: number[], param: Float32Array, centerIn: Float32Array, widthScale: number): Ribbon {
  const K = K_COLS
  // smooth the centerline (2 gentle Laplacian passes, endpoints pinned)
  const center = new Float32Array(centerIn)
  for (let pass = 0; pass < 2; pass++) {
    const prev = new Float32Array(center)
    for (let k = 1; k < K - 1; k++) {
      for (let c = 0; c < 3; c++) {
        center[k * 3 + c] = 0.25 * prev[(k - 1) * 3 + c]! + 0.5 * prev[k * 3 + c]! + 0.25 * prev[(k + 1) * 3 + c]!
      }
    }
  }
  const cnt = new Int32Array(K)
  const colAcc = new Float32Array(K * 3)
  // covariance accumulators in the tangent-perpendicular plane
  const cxx = new Float64Array(K)
  const cxy = new Float64Array(K)
  const cyy = new Float64Array(K)
  // per-bin plane basis
  const basis = new Float32Array(K * 6)
  for (let k = 0; k < K; k++) {
    const ka = Math.max(0, k - 1)
    const kb = Math.min(K - 1, k + 1)
    let tx = center[kb * 3]! - center[ka * 3]!
    let ty = center[kb * 3 + 1]! - center[ka * 3 + 1]!
    let tz = center[kb * 3 + 2]! - center[ka * 3 + 2]!
    const tl = Math.hypot(tx, ty, tz) || 1
    tx /= tl
    ty /= tl
    tz /= tl
    // e1 = arbitrary unit vector perpendicular to tangent
    let e1x = -tz
    let e1y = 0
    let e1z = tx
    const e1l = Math.hypot(e1x, e1y, e1z)
    if (e1l < 1e-4) {
      e1x = 1
      e1y = 0
      e1z = 0
    } else {
      e1x /= e1l
      e1z /= e1l
    }
    // e2 = tangent x e1
    const e2x = ty * e1z - tz * e1y
    const e2y = tz * e1x - tx * e1z
    const e2z = tx * e1y - ty * e1x
    basis.set([e1x, e1y, e1z, e2x, e2y, e2z], k * 6)
  }
  for (let j = 0; j < ids.length; j++) {
    const i = ids[j]!
    const k = Math.min(K - 1, Math.max(0, Math.floor(param[j]! * K)))
    const ox = cloud.pos[i * 3]! - center[k * 3]!
    const oy = cloud.pos[i * 3 + 1]! - center[k * 3 + 1]!
    const oz = cloud.pos[i * 3 + 2]! - center[k * 3 + 2]!
    const b = k * 6
    const px = ox * basis[b]! + oy * basis[b + 1]! + oz * basis[b + 2]!
    const py = ox * basis[b + 3]! + oy * basis[b + 4]! + oz * basis[b + 5]!
    cxx[k] = cxx[k]! + px * px
    cxy[k] = cxy[k]! + px * py
    cyy[k] = cyy[k]! + py * py
    colAcc[k * 3] = colAcc[k * 3]! + cloud.col[i * 3]!
    colAcc[k * 3 + 1] = colAcc[k * 3 + 1]! + cloud.col[i * 3 + 1]!
    colAcc[k * 3 + 2] = colAcc[k * 3 + 2]! + cloud.col[i * 3 + 2]!
    cnt[k] = cnt[k]! + 1
  }
  const out = emptyRibbon()
  const filled = new Uint8Array(K)
  for (let k = 0; k < K; k++) {
    out.posw[k * 4] = center[k * 3]!
    out.posw[k * 4 + 1] = center[k * 3 + 1]!
    out.posw[k * 4 + 2] = center[k * 3 + 2]!
    const c = cnt[k]!
    if (c < 2) continue
    filled[k] = 1
    // dominant eigenvector of the 2x2 covariance
    const xx = cxx[k]! / c
    const xy = cxy[k]! / c
    const yy = cyy[k]! / c
    const tr = xx + yy
    const det = xx * yy - xy * xy
    const disc = Math.sqrt(Math.max(tr * tr * 0.25 - det, 0))
    const l1 = tr * 0.5 + disc
    let vx = xy
    let vy = l1 - xx
    const vl = Math.hypot(vx, vy)
    if (vl < 1e-9) {
      vx = 1
      vy = 0
    } else {
      vx /= vl
      vy /= vl
    }
    const b = k * 6
    const lx = vx * basis[b]! + vy * basis[b + 3]!
    const ly = vx * basis[b + 1]! + vy * basis[b + 4]!
    const lz = vx * basis[b + 2]! + vy * basis[b + 5]!
    const halfw = Math.min(Math.max(1.7 * Math.sqrt(Math.max(l1, 0)) * widthScale, 0.0025), 0.09)
    out.posw[k * 4 + 3] = halfw
    out.cola[k * 4] = colAcc[k * 3]! / c
    out.cola[k * 4 + 1] = colAcc[k * 3 + 1]! / c
    out.cola[k * 4 + 2] = colAcc[k * 3 + 2]! / c
    out.cola[k * 4 + 3] = Math.atan2(lz, lx)
    void ly // azimuth encoding keeps only the horizontal direction
  }
  // fill empty bins from neighbors
  let last = -1
  for (let k = 0; k < K; k++) {
    if (filled[k]) {
      if (last >= 0 && last < k - 1) {
        for (let m = last + 1; m < k; m++) {
          const t = (m - last) / (k - last)
          for (let c = 3; c < 4; c++) {
            out.posw[m * 4 + c] = out.posw[last * 4 + c]! * (1 - t) + out.posw[k * 4 + c]! * t
          }
          for (let c = 0; c < 4; c++) {
            out.cola[m * 4 + c] = out.cola[last * 4 + c]! * (1 - t) + out.cola[k * 4 + c]! * t
          }
        }
      } else if (last < 0) {
        for (let m = 0; m < k; m++) {
          out.posw[m * 4 + 3] = out.posw[k * 4 + 3]!
          out.cola.set(out.cola.subarray(k * 4, k * 4 + 4), m * 4)
        }
      }
      last = k
    }
  }
  if (last < 0) return out // nothing filled — degenerate
  for (let m = last + 1; m < K; m++) {
    out.posw[m * 4 + 3] = out.posw[last * 4 + 3]!
    out.cola.set(out.cola.subarray(last * 4, last * 4 + 4), m * 4)
  }
  // smooth widths and azimuths along k. Azimuth is a direction mod pi, so
  // smooth in doubled-angle space where phi and phi+pi coincide.
  {
    const w0 = new Float32Array(K)
    const a2c = new Float32Array(K)
    const a2s = new Float32Array(K)
    for (let k = 0; k < K; k++) {
      w0[k] = out.posw[k * 4 + 3]!
      a2c[k] = Math.cos(2 * out.cola[k * 4 + 3]!)
      a2s[k] = Math.sin(2 * out.cola[k * 4 + 3]!)
    }
    for (let k = 0; k < K; k++) {
      const ka = Math.max(0, k - 1)
      const kb = Math.min(K - 1, k + 1)
      out.posw[k * 4 + 3] = 0.25 * w0[ka]! + 0.5 * w0[k]! + 0.25 * w0[kb]!
      const c = 0.25 * a2c[ka]! + 0.5 * a2c[k]! + 0.25 * a2c[kb]!
      const s = 0.25 * a2s[ka]! + 0.5 * a2s[k]! + 0.25 * a2s[kb]!
      out.cola[k * 4 + 3] = Math.atan2(s, c) / 2
    }
    // unwrap so the shader's linear interpolation along k never crosses a
    // pi-flip (which would twist the strip 180 degrees between samples)
    for (let k = 1; k < K; k++) {
      let phi = out.cola[k * 4 + 3]!
      const prev = out.cola[(k - 1) * 4 + 3]!
      while (phi - prev > Math.PI / 2) phi -= Math.PI
      while (phi - prev < -Math.PI / 2) phi += Math.PI
      out.cola[k * 4 + 3] = phi
    }
  }
  // taper the ends slightly so blades end in tips, not chopped rectangles
  out.posw[(K - 1) * 4 + 3] = out.posw[(K - 1) * 4 + 3]! * 0.35
  // importance = ribbon area
  let area = 0
  for (let k = 1; k < K; k++) {
    const dx = out.posw[k * 4]! - out.posw[(k - 1) * 4]!
    const dy = out.posw[k * 4 + 1]! - out.posw[(k - 1) * 4 + 1]!
    const dz = out.posw[k * 4 + 2]! - out.posw[(k - 1) * 4 + 2]!
    area += Math.hypot(dx, dy, dz) * (out.posw[k * 4 + 3]! + out.posw[(k - 1) * 4 + 3]!)
  }
  out.importance = area
  return out
}

/** Resample a polyline (flat xyz) to K points by arc length. */
function resample(poly: number[], K: number): Float32Array {
  const n = poly.length / 3
  const acc = new Float32Array(n)
  for (let i = 1; i < n; i++) {
    const dx = poly[i * 3]! - poly[(i - 1) * 3]!
    const dy = poly[i * 3 + 1]! - poly[(i - 1) * 3 + 1]!
    const dz = poly[i * 3 + 2]! - poly[(i - 1) * 3 + 2]!
    acc[i] = acc[i - 1]! + Math.hypot(dx, dy, dz)
  }
  const total = acc[n - 1]! || 1
  const out = new Float32Array(K * 3)
  let j = 1
  for (let k = 0; k < K; k++) {
    const target = (k / (K - 1)) * total
    while (j < n - 1 && acc[j]! < target) j++
    const t = (target - acc[j - 1]!) / (acc[j]! - acc[j - 1]! || 1)
    for (let c = 0; c < 3; c++) {
      out[k * 3 + c] = poly[(j - 1) * 3 + c]! * (1 - t) + poly[j * 3 + c]! * t
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// Strand tracing
// ---------------------------------------------------------------------------

function traceStrand(cloud: Cloud, seedIdx: number, dir0: [number, number, number], opts: DistillOptions): number[] {
  let px = cloud.pos[seedIdx * 3]!
  let py = cloud.pos[seedIdx * 3 + 1]!
  let pz = cloud.pos[seedIdx * 3 + 2]!
  let dx = dir0[0]
  let dy = dir0[1]
  let dz = dir0[2]
  const poly: number[] = [px, py, pz]
  for (let step = 0; step < 70; step++) {
    let ax = 0
    let ay = 0
    let az = 0
    let cnt = 0
    forNeighbors(cloud, px, py, pz, opts.neighborRadius, (i) => {
      if (cloud.linear[i] === 0) return
      const ox = cloud.pos[i * 3]! - px
      const oy = cloud.pos[i * 3 + 1]! - py
      const oz = cloud.pos[i * 3 + 2]! - pz
      const l = Math.hypot(ox, oy, oz)
      if (l < 0.003) return
      const d = (ox * dx + oy * dy + oz * dz) / l
      if (d < 0.45) return
      ax += ox / l
      ay += oy / l
      az += oz / l
      cnt++
    })
    if (cnt === 0) {
      // one-shot gap jump: retry with a wider radius before giving up
      forNeighbors(cloud, px, py, pz, opts.neighborRadius * 1.8, (i) => {
        if (cloud.linear[i] === 0) return
        const ox = cloud.pos[i * 3]! - px
        const oy = cloud.pos[i * 3 + 1]! - py
        const oz = cloud.pos[i * 3 + 2]! - pz
        const l = Math.hypot(ox, oy, oz)
        if (l < 0.003) return
        const d = (ox * dx + oy * dy + oz * dz) / l
        if (d < 0.55) return
        ax += ox / l
        ay += oy / l
        az += oz / l
        cnt++
      })
      if (cnt === 0) break
    }
    let nx = ax / cnt
    let ny = ay / cnt
    let nz = az / cnt
    // blend with previous direction for smoothness
    nx = nx * 0.72 + dx * 0.28
    ny = ny * 0.72 + dy * 0.28
    nz = nz * 0.72 + dz * 0.28
    const l = Math.hypot(nx, ny, nz) || 1
    dx = nx / l
    dy = ny / l
    dz = nz / l
    px += dx * opts.traceStep
    py += dy * opts.traceStep
    pz += dz * opts.traceStep
    poly.push(px, py, pz)
    if (py < 0.006) break
    if (px * px + pz * pz > opts.cloudRadius * opts.cloudRadius) break
  }
  return poly
}

/** Dominant local direction at point i (unit), from linear neighbors only. */
function localAxis(cloud: Cloud, i: number, r: number): [number, number, number] {
  const cov = new Float64Array(6)
  const mean = new Float64Array(3)
  let cnt = 0
  const x = cloud.pos[i * 3]!
  const y = cloud.pos[i * 3 + 1]!
  const z = cloud.pos[i * 3 + 2]!
  forNeighbors(cloud, x, y, z, r, (j) => {
    if (cloud.linear[j] === 0) return
    mean[0] = mean[0]! + cloud.pos[j * 3]!
    mean[1] = mean[1]! + cloud.pos[j * 3 + 1]!
    mean[2] = mean[2]! + cloud.pos[j * 3 + 2]!
    cnt++
  })
  if (cnt < 3) return [0, -1, 0]
  mean[0] = mean[0]! / cnt
  mean[1] = mean[1]! / cnt
  mean[2] = mean[2]! / cnt
  forNeighbors(cloud, x, y, z, r, (j) => {
    if (cloud.linear[j] === 0) return
    const ox = cloud.pos[j * 3]! - mean[0]!
    const oy = cloud.pos[j * 3 + 1]! - mean[1]!
    const oz = cloud.pos[j * 3 + 2]! - mean[2]!
    cov[0] = cov[0]! + ox * ox
    cov[1] = cov[1]! + oy * oy
    cov[2] = cov[2]! + oz * oz
    cov[3] = cov[3]! + ox * oy
    cov[4] = cov[4]! + oy * oz
    cov[5] = cov[5]! + ox * oz
  })
  let vx = 1
  let vy = 0.5
  let vz = 0.25
  for (let it = 0; it < 8; it++) {
    const nx = cov[0]! * vx + cov[3]! * vy + cov[5]! * vz
    const ny = cov[3]! * vx + cov[1]! * vy + cov[4]! * vz
    const nz = cov[5]! * vx + cov[4]! * vy + cov[2]! * vz
    const l = Math.hypot(nx, ny, nz) || 1
    vx = nx / l
    vy = ny / l
    vz = nz / l
  }
  return [vx, vy, vz]
}

/** Stamp the tube around a polyline; returns stamped ids + their parameters. */
function stampTube(
  cloud: Cloud,
  poly: number[],
  ribbonId: number,
  radius: number,
): { ids: number[]; param: Float32Array } {
  const n = poly.length / 3
  const ids: number[] = []
  const params: number[] = []
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0
    forNeighbors(cloud, poly[i * 3]!, poly[i * 3 + 1]!, poly[i * 3 + 2]!, radius, (j) => {
      if (cloud.stamp[j]! >= 0) return
      cloud.stamp[j] = ribbonId
      ids.push(j)
      params.push(t)
    })
  }
  return { ids, param: new Float32Array(params) }
}

/**
 * Flood-fill fluff points into proximity clusters (~ individual panicles).
 * Cell-based: classification is per grid cell, so fluff cells are flood-
 * filled across 26-connectivity and each cluster collects its cells' points.
 */
function clusterFluff(cloud: Cloud, maxClusterPoints: number): number[][] {
  const fluffCells = new Map<number, number[]>()
  for (const [key, members] of cloud.grid) {
    const pts = members.filter((i) => cloud.linear[i] === 0 && cloud.stamp[i]! < 0)
    if (pts.length > 0) fluffCells.set(key, pts)
  }
  const seen = new Set<number>()
  const clusters: number[][] = []
  for (const startKey of fluffCells.keys()) {
    if (seen.has(startKey)) continue
    const members: number[] = []
    const queue = [startKey]
    seen.add(startKey)
    while (queue.length > 0) {
      const key = queue.pop()!
      members.push(...fluffCells.get(key)!)
      for (let dz = -1; dz <= 1; dz++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nk = key + dx + dy * 1024 + dz * 1048576
            if (nk === key || seen.has(nk) || !fluffCells.has(nk)) continue
            seen.add(nk)
            queue.push(nk)
          }
        }
      }
    }
    if (members.length <= maxClusterPoints) {
      clusters.push(members)
      continue
    }
    // a big blob (dense understory) is not one panicle: split it into
    // azimuth sectors around its own centroid so each part stays plume-sized
    let mx = 0
    let mz = 0
    for (const i of members) {
      mx += cloud.pos[i * 3]!
      mz += cloud.pos[i * 3 + 2]!
    }
    mx /= members.length
    mz /= members.length
    const sectors = Math.min(8, Math.ceil(members.length / maxClusterPoints))
    const parts: number[][] = Array.from({ length: sectors }, () => [])
    for (const i of members) {
      const a = Math.atan2(cloud.pos[i * 3 + 2]! - mz, cloud.pos[i * 3]! - mx) + Math.PI
      const s = Math.min(sectors - 1, Math.floor((a / (2 * Math.PI)) * sectors))
      parts[s]!.push(i)
    }
    clusters.push(...parts)
  }
  return clusters
}

/** Fit one fat ribbon along a fluff cluster's principal axis. */
function plumeRibbon(cloud: Cloud, ids: number[], opts: DistillOptions): Ribbon | null {
  // principal axis of the cluster
  const mean = [0, 0, 0]
  for (const i of ids) {
    mean[0]! += cloud.pos[i * 3]!
    mean[1]! += cloud.pos[i * 3 + 1]!
    mean[2]! += cloud.pos[i * 3 + 2]!
  }
  mean[0]! /= ids.length
  mean[1]! /= ids.length
  mean[2]! /= ids.length
  const cov = new Float64Array(6)
  for (const i of ids) {
    const ox = cloud.pos[i * 3]! - mean[0]!
    const oy = cloud.pos[i * 3 + 1]! - mean[1]!
    const oz = cloud.pos[i * 3 + 2]! - mean[2]!
    cov[0] = cov[0]! + ox * ox
    cov[1] = cov[1]! + oy * oy
    cov[2] = cov[2]! + oz * oz
    cov[3] = cov[3]! + ox * oy
    cov[4] = cov[4]! + oy * oz
    cov[5] = cov[5]! + ox * oz
  }
  let vx = 0.2
  let vy = 1
  let vz = 0.1
  for (let it = 0; it < 8; it++) {
    const nx = cov[0]! * vx + cov[3]! * vy + cov[5]! * vz
    const ny = cov[3]! * vx + cov[1]! * vy + cov[4]! * vz
    const nz = cov[5]! * vx + cov[4]! * vy + cov[2]! * vz
    const l = Math.hypot(nx, ny, nz) || 1
    vx = nx / l
    vy = ny / l
    vz = nz / l
  }
  if (vy < 0) {
    vx = -vx
    vy = -vy
    vz = -vz
  }
  let minP = Infinity
  let maxP = -Infinity
  const param = new Float32Array(ids.length)
  ids.forEach((i, j) => {
    const p =
      (cloud.pos[i * 3]! - mean[0]!) * vx + (cloud.pos[i * 3 + 1]! - mean[1]!) * vy + (cloud.pos[i * 3 + 2]! - mean[2]!) * vz
    param[j] = p
    if (p < minP) minP = p
    if (p > maxP) maxP = p
  })
  const span = maxP - minP
  if (span < 0.03) return null
  const K = K_COLS
  const cnt = new Int32Array(K)
  const acc = new Float32Array(K * 3)
  ids.forEach((i, j) => {
    const t = Math.min(0.999, (param[j]! - minP) / span)
    param[j] = t
    const k = Math.floor(t * K)
    acc[k * 3] = acc[k * 3]! + cloud.pos[i * 3]!
    acc[k * 3 + 1] = acc[k * 3 + 1]! + cloud.pos[i * 3 + 1]!
    acc[k * 3 + 2] = acc[k * 3 + 2]! + cloud.pos[i * 3 + 2]!
    cnt[k] = cnt[k]! + 1
  })
  const center = new Float32Array(K * 3)
  let prev: [number, number, number] | null = null
  for (let k = 0; k < K; k++) {
    if (cnt[k]! > 0) {
      center[k * 3] = acc[k * 3]! / cnt[k]!
      center[k * 3 + 1] = acc[k * 3 + 1]! / cnt[k]!
      center[k * 3 + 2] = acc[k * 3 + 2]! / cnt[k]!
      prev = [center[k * 3]!, center[k * 3 + 1]!, center[k * 3 + 2]!]
    } else if (prev) {
      center.set(prev, k * 3)
    }
  }
  const r = fitBins(cloud, ids, param, center, opts.widthScale * 1.15)
  // Plumes are volumetric fluff, not oriented sheets: re-derive width as the
  // RADIAL spread about the centerline and mark it camera-facing by sign.
  {
    const radAcc = new Float64Array(K)
    const radCnt = new Int32Array(K)
    ids.forEach((i, j) => {
      const k = Math.min(K - 1, Math.floor(param[j]! * K))
      const dx = cloud.pos[i * 3]! - r.posw[k * 4]!
      const dy = cloud.pos[i * 3 + 1]! - r.posw[k * 4 + 1]!
      const dz = cloud.pos[i * 3 + 2]! - r.posw[k * 4 + 2]!
      radAcc[k] = radAcc[k]! + dx * dx + dy * dy + dz * dz
      radCnt[k] = radCnt[k]! + 1
    })
    let lastW = 0.02
    for (let k = 0; k < K; k++) {
      if (radCnt[k]! > 2) lastW = Math.min(Math.max(1.35 * Math.sqrt(radAcc[k]! / radCnt[k]!), 0.004), 0.07)
      r.posw[k * 4 + 3] = -lastW * opts.widthScale
    }
    // keep the tapered tip
    r.posw[(K - 1) * 4 + 3] = r.posw[(K - 1) * 4 + 3]! * 0.4
    // recompute importance with the new widths
    let area = 0
    for (let k = 1; k < K; k++) {
      const dx = r.posw[k * 4]! - r.posw[(k - 1) * 4]!
      const dy = r.posw[k * 4 + 1]! - r.posw[(k - 1) * 4 + 1]!
      const dz = r.posw[k * 4 + 2]! - r.posw[(k - 1) * 4 + 2]!
      area += Math.hypot(dx, dy, dz) * (Math.abs(r.posw[k * 4 + 3]!) + Math.abs(r.posw[(k - 1) * 4 + 3]!))
    }
    r.importance = area
  }
  return r
}

// ---------------------------------------------------------------------------
// Species distillation
// ---------------------------------------------------------------------------

function decodePosition(mesh: MeshLike, i: number, out: [number, number, number]): [number, number, number] {
  for (let c = 0; c < 3; c++) {
    out[c] =
      mesh.boundsMin[c]! + (mesh.vertices[i * 8 + c]! / 65535) * (mesh.boundsMax[c]! - mesh.boundsMin[c]!)
  }
  return out
}

/** Distill one species into VARIANTS x ROWS_PER_VARIANT ribbon rows. */
export function distillSpecies(mesh: MeshLike, options: Partial<DistillOptions> = {}): Ribbon[][] {
  const opts: DistillOptions = { ...DEFAULT_OPTIONS, ...options }
  const rng = mulberry32(opts.seed)
  const isTile = mesh.tileSize[0] > 0
  const tmp: [number, number, number] = [0, 0, 0]

  // Variant centers: low vertices spread apart (tiles) / the origin (specimen)
  const centers: [number, number][] = []
  if (isTile) {
    let tries = 0
    while (centers.length < VARIANTS && tries < 4000) {
      tries++
      const i = Math.floor(rng() * mesh.vertexCount)
      decodePosition(mesh, i, tmp)
      if (tmp[1] > 0.08) continue
      let ok = true
      for (const [cx, cz] of centers) {
        if (Math.hypot(tmp[0] - cx, tmp[2] - cz) < 0.15) ok = false
      }
      if (ok) centers.push([tmp[0], tmp[2]])
    }
    while (centers.length < VARIANTS) centers.push([rng() * mesh.tileSize[0]!, rng() * mesh.tileSize[1]!])
  } else {
    for (let v = 0; v < VARIANTS; v++) centers.push([0, 0])
  }

  const variants: Ribbon[][] = []
  for (let v = 0; v < VARIANTS; v++) {
    const [cx, cz] = centers[v]!
    // Gather the cloud; for tiles map each vertex through its NEAREST
    // periodic copy (foliage wrapping the tile edge stays with the tuft).
    // Subsampling is density-EQUALIZING — at most `cap` points per 2cm grid
    // cell — so sparse blade surfaces keep full detail while the very dense
    // panicle fluff is decimated hard.
    const r2max = opts.cloudRadius * opts.cloudRadius
    const cap = 10
    const cellCount = new Map<number, number>()
    const posArr: number[] = []
    const colArr: number[] = []
    for (let i = 0; i < mesh.vertexCount && posArr.length / 3 < opts.maxCloudPoints; i++) {
      decodePosition(mesh, i, tmp)
      let dx = tmp[0] - cx
      let dz = tmp[2] - cz
      if (isTile) {
        dx -= Math.round(dx / mesh.tileSize[0]!) * mesh.tileSize[0]!
        dz -= Math.round(dz / mesh.tileSize[1]!) * mesh.tileSize[1]!
        if (dx * dx + dz * dz > r2max) continue
      }
      const key = gridKey(opts.cellSize, dx, tmp[1], dz)
      const c = cellCount.get(key) ?? 0
      if (c >= cap) continue
      cellCount.set(key, c + 1)
      posArr.push(dx, tmp[1], dz)
      colArr.push(
        mesh.vertices[i * 8 + 3]! / 65535,
        mesh.vertices[i * 8 + 4]! / 65535,
        mesh.vertices[i * 8 + 5]! / 65535,
      )
    }
    const cloud = buildCloud(new Float32Array(posArr), new Float32Array(colArr), opts.cellSize)
    classifyCloud(cloud, opts.fluffThreshold)

    // seed pool: ONE representative per occupied thin cell, so compact dense
    // regions don't hog the strand budget by sheer point count
    const seedPool: number[] = []
    for (const members of cloud.grid.values()) {
      const i = members[0]!
      if (cloud.linear[i] === 1) seedPool.push(i)
    }

    const ribbons: Ribbon[] = []
    let rejectShort = 0
    let rejectSparse = 0
    // --- strands over thin points (blades, stems) ---
    let attempts = 0
    while (ribbons.length < opts.strands && attempts < opts.strands * 6) {
      attempts++
      // random uncovered thin seed from the pool
      let seed = -1
      while (seedPool.length > 0) {
        const j = Math.floor(rng() * seedPool.length)
        const i = seedPool[j]!
        if (cloud.stamp[i]! >= 0) {
          seedPool[j] = seedPool[seedPool.length - 1]!
          seedPool.pop()
          continue
        }
        seed = i
        break
      }
      if (seed < 0) break
      const axis = localAxis(cloud, seed, opts.neighborRadius)
      // trace both ways: "down" toward the root, "up" toward the tip
      const down: [number, number, number] = axis[1] <= 0 ? axis : [-axis[0], -axis[1], -axis[2]]
      const up: [number, number, number] = [-down[0], -down[1], -down[2]]
      const polyDown = traceStrand(cloud, seed, down, opts)
      const polyUp = traceStrand(cloud, seed, up, opts)
      // stitch: reverse(down-run) + up-run (skip the duplicated seed)
      const poly: number[] = []
      for (let i = polyDown.length / 3 - 1; i >= 0; i--) {
        poly.push(polyDown[i * 3]!, polyDown[i * 3 + 1]!, polyDown[i * 3 + 2]!)
      }
      for (let i = 1; i < polyUp.length / 3; i++) {
        poly.push(polyUp[i * 3]!, polyUp[i * 3 + 1]!, polyUp[i * 3 + 2]!)
      }
      const { ids, param } = stampTube(cloud, poly, ribbons.length, opts.coverRadius)
      if (poly.length / 3 < 6) {
        rejectShort++
        continue
      }
      if (ids.length < 10) {
        rejectSparse++
        continue
      }
      const center = resample(poly, K_COLS)
      const r = fitBins(cloud, ids, param, center, opts.widthScale)
      if (r.importance > 0) ribbons.push(r)
    }
    const strandCount = ribbons.length

    // --- plumes: cluster the fluff by proximity (one cluster ~ one panicle) ---
    const clusters = clusterFluff(cloud, opts.maxClusterPoints)
    clusters.sort((a, b) => b.length - a.length)
    let plumeCount = 0
    for (const ids of clusters) {
      if (plumeCount >= opts.plumes) break
      if (ids.length < 60) break
      // keep only panicles whose centroid is reasonably close to the tuft
      let mx = 0
      let mz = 0
      for (const i of ids) {
        mx += cloud.pos[i * 3]!
        mz += cloud.pos[i * 3 + 2]!
      }
      mx /= ids.length
      mz /= ids.length
      if (isTile && Math.hypot(mx, mz) > opts.cloudRadius * 0.75) continue
      const r = plumeRibbon(cloud, ids, opts)
      if (r && r.importance > 0) {
        ribbons.push(r)
        plumeCount++
        for (const i of ids) cloud.stamp[i] = 900 // include in aggregate
      }
    }

    if (opts.debug) {
      let thin = 0
      for (let i = 0; i < cloud.n; i++) thin += cloud.linear[i]!
      console.log(
        `[ribbon-skeleton] variant ${v}: cloud=${cloud.n} thin=${thin} seeds=${seedPool.length} ` +
          `attempts=${attempts} strands=${strandCount} plumes=${plumeCount} ` +
          `rejects(short=${rejectShort} sparse=${rejectSparse})`,
      )
    }
    // --- sort by importance, cap at RIBBONS, pad with degenerates ---
    ribbons.sort((a, b) => b.importance - a.importance)
    ribbons.length = Math.min(ribbons.length, RIBBONS)
    while (ribbons.length < RIBBONS) ribbons.push(emptyRibbon())

    // --- aggregate silhouette over everything covered ---
    {
      const ids: number[] = []
      for (let i = 0; i < cloud.n; i++) if (cloud.stamp[i]! >= 0) ids.push(i)
      const agg = emptyRibbon()
      if (ids.length > 50) {
        let minY = Infinity
        let maxY = -Infinity
        for (const i of ids) {
          const y = cloud.pos[i * 3 + 1]!
          if (y < minY) minY = y
          if (y > maxY) maxY = y
        }
        minY = Math.max(0, minY)
        const K = K_COLS
        const cnt = new Int32Array(K)
        const accR = new Float64Array(K)
        const accC = new Float32Array(K * 3)
        for (const i of ids) {
          const y = cloud.pos[i * 3 + 1]!
          const t = Math.min(0.999, Math.max(0, (y - minY) / (maxY - minY || 1)))
          const k = Math.floor(t * K)
          const r = Math.hypot(cloud.pos[i * 3]!, cloud.pos[i * 3 + 2]!)
          accR[k] = accR[k]! + r * r
          accC[k * 3] = accC[k * 3]! + cloud.col[i * 3]!
          accC[k * 3 + 1] = accC[k * 3 + 1]! + cloud.col[i * 3 + 1]!
          accC[k * 3 + 2] = accC[k * 3 + 2]! + cloud.col[i * 3 + 2]!
          cnt[k] = cnt[k]! + 1
        }
        let lastW = 0.03
        let lastC: [number, number, number] = [0.3, 0.35, 0.15]
        for (let k = 0; k < K; k++) {
          const y = minY + ((maxY - minY) * k) / (K - 1)
          agg.posw[k * 4] = 0
          agg.posw[k * 4 + 1] = y
          agg.posw[k * 4 + 2] = 0
          if (cnt[k]! > 3) {
            lastW = 1.3 * Math.sqrt(accR[k]! / cnt[k]!)
            lastC = [accC[k * 3]! / cnt[k]!, accC[k * 3 + 1]! / cnt[k]!, accC[k * 3 + 2]! / cnt[k]!]
          }
          // negative width = camera-facing (the far-LOD card)
          agg.posw[k * 4 + 3] = -lastW
          agg.cola[k * 4] = lastC[0]
          agg.cola[k * 4 + 1] = lastC[1]
          agg.cola[k * 4 + 2] = lastC[2]
          agg.cola[k * 4 + 3] = 0
        }
        // ground the card and taper its top
        agg.posw[3] = agg.posw[3]! * 0.8
        agg.posw[(K - 1) * 4 + 3] = agg.posw[(K - 1) * 4 + 3]! * 0.5
      }
      ribbons.push(agg)
    }
    variants.push(ribbons)
  }
  return variants
}

// ---------------------------------------------------------------------------
// Atlas (de)serialization
// ---------------------------------------------------------------------------

export interface ParsedAtlas {
  kCols: number
  rowsTotal: number
  rowsPerVariant: number
  variants: number
  /** Row offset per global species catalog index (8 slots). */
  rowOffsets: Uint32Array
  t0: Float32Array
  t1: Float32Array
}

/** speciesRows: map from global species catalog index to that species' rows. */
export function serializeAtlas(speciesRows: Map<number, Ribbon[][]>): ArrayBuffer {
  const rowsPerSpecies = VARIANTS * ROWS_PER_VARIANT
  const indices = [...speciesRows.keys()].sort((a, b) => a - b)
  const rowsTotal = indices.length * rowsPerSpecies
  const headerWords = 16
  const texels = rowsTotal * K_COLS * 4
  const buf = new ArrayBuffer(headerWords * 4 + texels * 4 * 2)
  const u32 = new Uint32Array(buf, 0, headerWords)
  u32[0] = ATLAS_MAGIC
  u32[1] = ATLAS_VERSION
  u32[2] = K_COLS
  u32[3] = rowsTotal
  u32[4] = VARIANTS
  u32[5] = ROWS_PER_VARIANT
  u32[6] = indices.length
  const t0 = new Float32Array(buf, headerWords * 4, texels)
  const t1 = new Float32Array(buf, headerWords * 4 + texels * 4, texels)
  indices.forEach((speciesIndex, slot) => {
    u32[8 + speciesIndex] = slot * rowsPerSpecies
    const rows = speciesRows.get(speciesIndex)!
    rows.forEach((variant, v) => {
      variant.forEach((ribbon, r) => {
        const row = slot * rowsPerSpecies + v * ROWS_PER_VARIANT + r
        t0.set(ribbon.posw, row * K_COLS * 4)
        t1.set(ribbon.cola, row * K_COLS * 4)
      })
    })
  })
  return buf
}

export function parseAtlas(buf: ArrayBuffer): ParsedAtlas | null {
  if (buf.byteLength < 64) return null
  const u32 = new Uint32Array(buf, 0, 16)
  if (u32[0] !== ATLAS_MAGIC || u32[1] !== ATLAS_VERSION) return null
  const kCols = u32[2]!
  const rowsTotal = u32[3]!
  const texels = rowsTotal * kCols * 4
  if (buf.byteLength !== 64 + texels * 8) return null
  return {
    kCols,
    rowsTotal,
    rowsPerVariant: u32[5]!,
    variants: u32[4]!,
    rowOffsets: new Uint32Array(buf.slice(32, 64)),
    t0: new Float32Array(buf, 64, texels),
    t1: new Float32Array(buf, 64 + texels * 4, texels),
  }
}

/** f32 -> f16 bits (round-to-nearest-even not needed; truncation is fine here). */
export function toF16(data: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(new ArrayBuffer(data.length * 2))
  const f32 = new Float32Array(1)
  const u32 = new Uint32Array(f32.buffer)
  for (let i = 0; i < data.length; i++) {
    f32[0] = data[i]!
    const x = u32[0]!
    const sign = (x >>> 16) & 0x8000
    let exp = (x >>> 23) & 0xff
    const mant = x & 0x7fffff
    if (exp === 0xff) {
      out[i] = sign | 0x7c00
      continue
    }
    let e = exp - 127 + 15
    if (e >= 31) {
      out[i] = sign | 0x7bff // clamp to max finite f16
    } else if (e <= 0) {
      out[i] = sign // flush denormals to zero (values ~<6e-5 — irrelevant here)
    } else {
      out[i] = sign | (e << 10) | (mant >>> 13)
    }
    void exp
  }
  return out
}
