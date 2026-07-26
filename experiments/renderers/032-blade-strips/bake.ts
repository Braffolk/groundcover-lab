import bakeShaderSrc from './shaders/bake_unroll.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Blade-strip bake: turn a raw GCMESH1 plant into a nested set of CURVED
 * RIBBONS plus one unrolled imagery tile per ribbon.
 *
 *  1. subsample the mesh vertices and k-means them into LEAF_CLUSTERS blade
 *     bundles (vertical weight < 1, so bundles come out as tall columns of
 *     foliage rather than horizontal slabs);
 *  2. pair bundles bottom-up by centroid distance into a perfect binary merge
 *     tree, and order the leaves so that every level's cluster is a CONTIGUOUS
 *     leaf range — one reordered index buffer then serves all levels;
 *  3. fit a ribbon per cluster of every level: a 5-node polyline through the
 *     cluster's height-binned centroids (its real mean blade arc), a lateral
 *     axis from the horizontal residual covariance, per-node half-widths that
 *     hug the real lateral spread, and the residual thickness that is being
 *     flattened away;
 *  4. render each cluster's real triangles into its own atlas tile with the
 *     unrolling vertex transform in shaders/bake_unroll.wgsl, 2x supersampled,
 *     coverage-weighted downsample + tile-clamped dilation;
 *  5. one extra tile holds the straight-down top view for the canopy card.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'BST1', u32 version, u32 atlasPx, u32 tilePxU, u32 tilePxV,
 *   u32 stripCount, u32 nodeCount, u32 pad
 *   f32 rXZ, y0, y1, cx, cz                (capture box, unit scale, metres)
 *   ...zeros to byte 128
 *   f32[stripCount * 24]   ribbon table (5 x vec4 nodes + lat/depth vec4)
 *   u8 [atlasPx^2 * 4]     albedo rgba8 (straight colour, a = coverage)
 *   u8 [atlasPx^2 * 2]     tangent-frame normal rg8 (face component implied)
 */

/** Strip counts per LOD level, finest first. Sum = strip table size. */
export const LEVELS = [16, 8, 4, 2, 1] as const
export const LEVEL_BASE = ((): number[] => {
  const out: number[] = []
  let acc = 0
  for (const n of LEVELS) {
    out.push(acc)
    acc += n
  }
  return out
})()
export const STRIP_COUNT = LEVELS.reduce((a, b) => a + b, 0) // 31
/** Atlas slot of the straight-down top view (last slot). */
export const TOP_SLOT = STRIP_COUNT
export const SLOTS = STRIP_COUNT + 1 // 32
export const NODES = 5
// Tile aspect follows the ribbon's, but LATERAL resolution is what sets
// perceived crispness: blades run along the arc, so their silhouette edges run
// along the arc too and are resolved across it. 128 x 512 over a ~0.25m x 0.5m
// bundle is ~2mm/texel both ways — the billboard card's texel density, at 1/16
// of the plant. (64 x 1024 was tried: blade EDGES went soft, and the flat card
// read as the crisper image up close, which is exactly what must not happen.)
export const TILE_U = 128 // lateral texels
export const TILE_V = 512 // arc texels
export const GRID_U = 16 // slots per atlas row
export const GRID_V = 2
export const ATLAS_W = TILE_U * GRID_U // 2048
export const ATLAS_H = TILE_V * GRID_V // 1024
// Deep chain on purpose: a far plant minifies a 512-tall tile by 25x, and
// stopping the chain early turns that into a texture-cache stall per fragment
// (measured: the far field, not the near ribbons, was the cost).
export const MIP_LEVELS = 7
export const STRIP_FLOATS = NODES * 4 + 4

const LEAF_CLUSTERS = LEVELS[0]
const SS = 2
const BIG_W = ATLAS_W * SS
const BIG_H = ATLAS_H * SS
const MAGIC = 0x31545342 // 'BST1'
const VERSION = 7
const HEADER_BYTES = 128
const DILATE_PASSES = 3
const VERTEX_STRIDE_TARGET = 900_000
/** Occupancy voxel edge (m) — clustering works on voxels, not vertices. */
const VOXEL = 0.02
const KMEANS_ITERS = 16
/**
 * Vertical distance weight in the clustering metric. 0.5 lets bundles split
 * into ~2 height bands as well as columns, and that is deliberate: a band gets
 * its own tile, so a 0.5m band spends 512 arc texels on 0.5m (1mm/texel)
 * instead of on the whole 1.15m plant. Tried 0.22 (full-height columns only):
 * every ribbon halved its arc resolution and the panicles smeared into
 * diagonal streaks — clearly worse. Continuity across a band boundary is
 * handled by fitting each ribbon's arc to its ASSIGNED TRIANGLES' height range
 * (see fitRibbon), so neighbouring bands overlap instead of cutting blades.
 */
const Y_WEIGHT = 0.5

export interface StripSet {
  atlasPx: number
  stripCount: number
  nodeCount: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump centre offset in the mesh frame (the capture was centred here). */
  cx: number
  cz: number
  /** STRIP_COUNT * STRIP_FLOATS: [x,y,z,halfWidth] x5, [latX,latZ,depth,pad]. */
  strips: Float32Array<ArrayBuffer>
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackStrips(buf: ArrayBuffer): StripSet | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const atlasPx = u[2]!
  const stripCount = u[5]!
  const nodeCount = u[6]!
  if (atlasPx !== ATLAS_W || stripCount !== STRIP_COUNT || nodeCount !== NODES) return null
  const stripBytes = stripCount * STRIP_FLOATS * 4
  const albBytes = ATLAS_W * ATLAS_H * 4
  const nrmBytes = ATLAS_W * ATLAS_H * 2
  if (buf.byteLength !== HEADER_BYTES + stripBytes + albBytes + nrmBytes) return null
  const f = new Float32Array(buf, 32, 5)
  return {
    atlasPx,
    stripCount,
    nodeCount,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    strips: new Float32Array(buf.slice(HEADER_BYTES, HEADER_BYTES + stripBytes)),
    albedo: new Uint8Array(buf, HEADER_BYTES + stripBytes, albBytes),
    normal: new Uint8Array(buf, HEADER_BYTES + stripBytes + albBytes, nrmBytes),
  }
}

function packStrips(s: StripSet): ArrayBuffer {
  const stripBytes = s.strips.byteLength
  const buf = new ArrayBuffer(HEADER_BYTES + stripBytes + s.albedo.byteLength + s.normal.byteLength)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = s.atlasPx
  u[3] = TILE_U
  u[4] = TILE_V
  u[5] = s.stripCount
  u[6] = s.nodeCount
  const f = new Float32Array(buf, 32, 5)
  f[0] = s.rXZ
  f[1] = s.y0
  f[2] = s.y1
  f[3] = s.cx
  f[4] = s.cz
  new Float32Array(buf, HEADER_BYTES, s.strips.length).set(s.strips)
  new Uint8Array(buf, HEADER_BYTES + stripBytes, s.albedo.byteLength).set(s.albedo)
  new Uint8Array(buf, HEADER_BYTES + stripBytes + s.albedo.byteLength, s.normal.byteLength).set(s.normal)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the strip set for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesStrips(ctx: BakeCtx, speciesId: string): Promise<StripSet> {
  const key = `strips-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesStrips(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let set = unpackStrips(buf)
  if (!set) {
    buf = await runBake()
    set = unpackStrips(buf)
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
// Ribbon fitting (CPU)
// ---------------------------------------------------------------------------

interface Ribbon {
  /** NODES * 3 positions in the clump frame (x/z centred, y raw). */
  nodes: Float32Array
  halfWidth: Float32Array
  latX: number
  latZ: number
  depth: number
}

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0
  const sorted = values.slice().sort((a, b) => a - b)
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))
  return sorted[i]!
}

/** Closest point on a polyline, measured perpendicular to `lat` (xz unit). */
function closestOnRibbon(
  nodes: Float32Array,
  px: number,
  py: number,
  pz: number,
  latX: number,
  latZ: number,
): { seg: number; t: number; qx: number; qy: number; qz: number } {
  let bestD2 = Infinity
  let out = { seg: 0, t: 0, qx: nodes[0]!, qy: nodes[1]!, qz: nodes[2]! }
  for (let i = 0; i < NODES - 1; i++) {
    const ax = nodes[i * 3]!
    const ay = nodes[i * 3 + 1]!
    const az = nodes[i * 3 + 2]!
    const dx = nodes[i * 3 + 3]! - ax
    const dy = nodes[i * 3 + 4]! - ay
    const dz = nodes[i * 3 + 5]! - az
    const dd = dx * dx + dy * dy + dz * dz
    let t = dd > 1e-12 ? ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / dd : 0
    t = t < 0 ? 0 : t > 1 ? 1 : t
    const qx = ax + dx * t
    const qy = ay + dy * t
    const qz = az + dz * t
    const rx = px - qx
    const ry = py - qy
    const rz = pz - qz
    const lat = rx * latX + rz * latZ
    const d2 = rx * rx + ry * ry + rz * rz - lat * lat
    if (d2 < bestD2) {
      bestD2 = d2
      out = { seg: i, t, qx, qy, qz }
    }
  }
  return out
}

/**
 * Fit one ribbon to a set of sample points: mean blade arc through
 * height-binned centroids, lateral axis from the horizontal residual
 * covariance, per-node half-widths that hug the real spread.
 */
function fitRibbon(
  xs: Float32Array,
  ys: Float32Array,
  zs: Float32Array,
  idx: Int32Array,
  from: number,
  to: number,
  /** Height range of the TRIANGLES assigned to this bundle (they reach past
   *  the sampled voxel centres; clipping them would cut blades). */
  triY: [number, number],
): Ribbon {
  const n = to - from
  const nodes = new Float32Array(NODES * 3)
  const halfWidth = new Float32Array(NODES)
  if (n < 8) {
    for (let k = 0; k < NODES; k++) {
      nodes[k * 3 + 1] = (k / (NODES - 1)) * 0.5
      halfWidth[k] = 0.02
    }
    return { nodes, halfWidth, latX: 1, latZ: 0, depth: 0.02 }
  }

  // The bundle's OWN height band — never stretched to the ground. A canopy-top
  // bundle that got a ground-anchored ribbon would spend four fifths of its
  // tile on empty space and rasterize as a tall empty sliver.
  let yMin = Infinity
  let yMax = -Infinity
  for (let i = from; i < to; i++) {
    const y = ys[idx[i]!]!
    if (y < yMin) yMin = y
    if (y > yMax) yMax = y
  }
  // Cover the assigned triangles, but do not let one stray triangle stretch the
  // arc (and with it the tile's texel density) by more than 15% of the span.
  const margin = Math.max((yMax - yMin) * 0.15, 4e-3)
  yMin = Math.max(0, Math.max(triY[0] - 1e-3, yMin - margin))
  yMax = Math.min(triY[1] + 1e-3, yMax + margin)
  const hSpan = Math.max(yMax - yMin, 1e-3)

  // Height-binned XZ centroids -> the bundle's real mean arc. Bins are centred
  // on the node heights k/(NODES-1), so the polyline spans the full height.
  const cx = new Float64Array(NODES)
  const cz = new Float64Array(NODES)
  const cw = new Float64Array(NODES)
  for (let i = from; i < to; i++) {
    const s = idx[i]!
    const f = (ys[s]! - yMin) / hSpan
    let k = Math.round(f * (NODES - 1))
    k = k < 0 ? 0 : k > NODES - 1 ? NODES - 1 : k
    cx[k] = cx[k]! + xs[s]!
    cz[k] = cz[k]! + zs[s]!
    cw[k] = cw[k]! + 1
  }
  // Normalize, then fill empty bins by interpolating between filled neighbours
  // (copying the nearest one would kink or collapse the arc).
  const mx = new Float64Array(NODES)
  const mz = new Float64Array(NODES)
  for (let k = 0; k < NODES; k++) {
    if (cw[k]! <= 0) continue
    mx[k] = cx[k]! / cw[k]!
    mz[k] = cz[k]! / cw[k]!
  }
  for (let k = 0; k < NODES; k++) {
    if (cw[k]! > 0) continue
    let lo = -1
    let hi = -1
    for (let j = k - 1; j >= 0; j--) {
      if (cw[j]! > 0) {
        lo = j
        break
      }
    }
    for (let j = k + 1; j < NODES; j++) {
      if (cw[j]! > 0) {
        hi = j
        break
      }
    }
    if (lo >= 0 && hi >= 0) {
      const t = (k - lo) / (hi - lo)
      mx[k] = mx[lo]! + (mx[hi]! - mx[lo]!) * t
      mz[k] = mz[lo]! + (mz[hi]! - mz[lo]!) * t
    } else if (lo >= 0) {
      mx[k] = mx[lo]!
      mz[k] = mz[lo]!
    } else if (hi >= 0) {
      mx[k] = mx[hi]!
      mz[k] = mz[hi]!
    }
  }
  for (let k = 0; k < NODES; k++) {
    nodes[k * 3] = mx[k]!
    nodes[k * 3 + 1] = yMin + (k / (NODES - 1)) * hSpan
    nodes[k * 3 + 2] = mz[k]!
  }

  // Lateral axis: principal direction of the horizontal residuals.
  let sxx = 0
  let sxz = 0
  let szz = 0
  for (let i = from; i < to; i++) {
    const s = idx[i]!
    const c = closestOnRibbon(nodes, xs[s]!, ys[s]!, zs[s]!, 1, 0)
    const rx = xs[s]! - c.qx
    const rz = zs[s]! - c.qz
    sxx += rx * rx
    sxz += rx * rz
    szz += rz * rz
  }
  let latX = 1
  let latZ = 0
  {
    const tr = (sxx + szz) / 2
    const df = (sxx - szz) / 2
    const lam = tr + Math.sqrt(df * df + sxz * sxz)
    if (Math.abs(sxz) > 1e-12) {
      const vx = sxz
      const vz = lam - sxx
      const len = Math.hypot(vx, vz) || 1
      latX = vx / len
      latZ = vz / len
    } else if (szz > sxx) {
      latX = 0
      latZ = 1
    }
  }

  // Half-widths (per node) and flattened thickness, from lateral/face residuals.
  const latBins: number[][] = Array.from({ length: NODES }, () => [])
  const faceAbs: number[] = []
  for (let i = from; i < to; i++) {
    const s = idx[i]!
    const c = closestOnRibbon(nodes, xs[s]!, ys[s]!, zs[s]!, latX, latZ)
    const rx = xs[s]! - c.qx
    const ry = ys[s]! - c.qy
    const rz = zs[s]! - c.qz
    const lateral = rx * latX + rz * latZ
    // Face normal = normalize(cross(lat, tangent)) with lat horizontal.
    const tx = nodes[c.seg * 3 + 3]! - nodes[c.seg * 3]!
    const ty = nodes[c.seg * 3 + 4]! - nodes[c.seg * 3 + 1]!
    const tz = nodes[c.seg * 3 + 5]! - nodes[c.seg * 3 + 2]!
    let fx = -latZ * ty
    let fy = latZ * tx - latX * tz
    let fz = latX * ty
    const fl = Math.hypot(fx, fy, fz) || 1
    fx /= fl
    fy /= fl
    fz /= fl
    faceAbs.push(Math.abs(rx * fx + ry * fy + rz * fz))
    const arc = (c.seg + c.t) / (NODES - 1)
    let k = Math.round(arc * (NODES - 1))
    k = k < 0 ? 0 : k > NODES - 1 ? NODES - 1 : k
    latBins[k]!.push(Math.abs(lateral))
  }
  // Per-node width from the local lateral spread, floored against the bundle's
  // overall spread so a thin bin cannot pinch the ribbon into a sliver.
  const allLat: number[] = []
  for (const bin of latBins) for (const v of bin) allLat.push(v)
  // Keep ribbons TIGHT: every extra centimetre of half-width spends the tile's
  // lateral texels on empty space, adds overdraw, and (because the ribbons of a
  // level tile the plant) buys nothing. The 90th percentile plus 20% covers the
  // bundle; the stray tips past it are cheaper to lose than to resolve.
  const globalW = percentile(allLat, 0.9) * 1.2 + 0.004
  const raw = new Float32Array(NODES)
  for (let k = 0; k < NODES; k++) {
    raw[k] = Math.max(percentile(latBins[k]!, 0.9) * 1.2 + 0.004, globalW * 0.4)
  }
  // Smooth so a sparse bin cannot pinch the ribbon, and keep a sane minimum.
  for (let k = 0; k < NODES; k++) {
    const a = raw[Math.max(0, k - 1)]!
    const b = raw[k]!
    const c = raw[Math.min(NODES - 1, k + 1)]!
    halfWidth[k] = Math.max(0.012, (a + 2 * b + c) / 4)
  }
  return { nodes, halfWidth, latX, latZ, depth: Math.max(0.01, percentile(faceAbs, 0.92) * 1.3 + 0.004) }
}

// ---------------------------------------------------------------------------
// Bake
// ---------------------------------------------------------------------------

async function bakeSpeciesStrips(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const verts = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535

  // --- 1. occupancy voxels in the clump frame ------------------------------
  // Clustering on raw vertices follows TESSELLATION density, not shape: the
  // fluffy panicle carries most of the mesh's vertices and would swallow most
  // of the bundles. One sample per occupied 2cm voxel is spatially uniform,
  // which is what "split the plant into 16 bundles" actually means.
  const stride = Math.max(1, Math.floor(hdr.vertexCount / VERTEX_STRIDE_TARGET))
  const gx = Math.max(1, Math.ceil((bx1 - bx0) / VOXEL))
  const gy = Math.max(1, Math.ceil((by1 - by0) / VOXEL))
  const gz = Math.max(1, Math.ceil((bz1 - bz0) / VOXEL))
  const cellCount = gx * gy * gz
  const accX = new Float32Array(cellCount)
  const accY = new Float32Array(cellCount)
  const accZ = new Float32Array(cellCount)
  const accN = new Int32Array(cellCount)
  let r2 = 0
  let occupied = 0
  for (let i = 0; i < hdr.vertexCount; i += stride) {
    const x = bx0 + verts[i * 8]! * sx - cx
    const y = by0 + verts[i * 8 + 1]! * sy
    const z = bz0 + verts[i * 8 + 2]! * sz - cz
    const d = x * x + z * z
    if (d > r2) r2 = d
    const ix = Math.min(gx - 1, Math.max(0, Math.floor((x + cx - bx0) / VOXEL)))
    const iy = Math.min(gy - 1, Math.max(0, Math.floor((y - by0) / VOXEL)))
    const iz = Math.min(gz - 1, Math.max(0, Math.floor((z + cz - bz0) / VOXEL)))
    const c = (iy * gz + iz) * gx + ix
    if (accN[c] === 0) occupied++
    accX[c] = accX[c]! + x
    accY[c] = accY[c]! + y
    accZ[c] = accZ[c]! + z
    accN[c] = accN[c]! + 1
  }
  const nSamples = occupied
  const xs = new Float32Array(nSamples)
  const ys = new Float32Array(nSamples)
  const zs = new Float32Array(nSamples)
  for (let c = 0, s = 0; c < cellCount; c++) {
    const n = accN[c]!
    if (n === 0) continue
    xs[s] = accX[c]! / n
    ys[s] = accY[c]! / n
    zs[s] = accZ[c]! / n
    s++
  }
  const rXZ = Math.sqrt(r2) * 1.03 + 1e-3
  const y0 = Math.min(0, by0)
  const y1 = by1

  // --- 2. k-means into blade bundles ---------------------------------------
  const K = LEAF_CLUSTERS
  const ccx = new Float64Array(K)
  const ccy = new Float64Array(K)
  const ccz = new Float64Array(K)
  let rng = 0x9e3779b9
  const rand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0
    return rng / 4294967296
  }
  for (let k = 0; k < K; k++) {
    const s = Math.floor(rand() * nSamples)
    ccx[k] = xs[s]!
    ccy[k] = ys[s]!
    ccz[k] = zs[s]!
  }
  const assign = new Int32Array(nSamples)
  const nearest = (x: number, y: number, z: number): number => {
    let best = 0
    let bestD = Infinity
    for (let k = 0; k < K; k++) {
      const dx = x - ccx[k]!
      const dy = (y - ccy[k]!) * Y_WEIGHT
      const dz = z - ccz[k]!
      const d = dx * dx + dy * dy + dz * dz
      if (d < bestD) {
        bestD = d
        best = k
      }
    }
    return best
  }
  for (let iter = 0; iter < KMEANS_ITERS; iter++) {
    const ax = new Float64Array(K)
    const ay = new Float64Array(K)
    const az = new Float64Array(K)
    const an = new Float64Array(K)
    for (let s = 0; s < nSamples; s++) {
      const k = nearest(xs[s]!, ys[s]!, zs[s]!)
      assign[s] = k
      ax[k] = ax[k]! + xs[s]!
      ay[k] = ay[k]! + ys[s]!
      az[k] = az[k]! + zs[s]!
      an[k] = an[k]! + 1
    }
    for (let k = 0; k < K; k++) {
      if (an[k]! < 1) {
        // Re-seed an empty cluster on a random sample so levels stay populated.
        const s = Math.floor(rand() * nSamples)
        ccx[k] = xs[s]!
        ccy[k] = ys[s]!
        ccz[k] = zs[s]!
        continue
      }
      ccx[k] = ax[k]! / an[k]!
      ccy[k] = ay[k]! / an[k]!
      ccz[k] = az[k]! / an[k]!
    }
  }
  for (let s = 0; s < nSamples; s++) assign[s] = nearest(xs[s]!, ys[s]!, zs[s]!)

  // --- 3. merge tree -> leaf order (level clusters are contiguous ranges) ---
  const leafOrder = buildMergeOrder(K, ccx, ccy, ccz)
  const leafRank = new Int32Array(K)
  leafOrder.forEach((cluster, rank) => {
    leafRank[cluster] = rank
  })
  for (let s = 0; s < nSamples; s++) assign[s] = leafRank[assign[s]!]!

  // Samples sorted by leaf, with per-leaf ranges (used by the ribbon fits).
  const leafCount = new Int32Array(K)
  for (let s = 0; s < nSamples; s++) {
    const k = assign[s]!
    leafCount[k] = leafCount[k]! + 1
  }
  const leafStart = new Int32Array(K + 1)
  for (let k = 0; k < K; k++) leafStart[k + 1] = leafStart[k]! + leafCount[k]!
  const sampleByLeaf = new Int32Array(nSamples)
  {
    const cursor = leafStart.slice()
    for (let s = 0; s < nSamples; s++) {
      const k = assign[s]!
      sampleByLeaf[cursor[k]!] = s
      cursor[k] = cursor[k]! + 1
    }
  }

  // --- 4. per-triangle cluster assignment, reordered index buffer -----------
  const triCount = hdr.triangleCount
  const tris = mesh.triangles
  const triCluster = new Uint8Array(triCount)
  const triPerLeaf = new Int32Array(K)
  const leafYLo = new Float32Array(K).fill(Infinity)
  const leafYHi = new Float32Array(K).fill(-Infinity)
  for (let t = 0; t < triCount; t++) {
    const i0 = tris[t * 4]!
    const i1 = tris[t * 4 + 1]!
    const i2 = tris[t * 4 + 2]!
    const y0q = verts[i0 * 8 + 1]!
    const y1q = verts[i1 * 8 + 1]!
    const y2q = verts[i2 * 8 + 1]!
    const mx = (verts[i0 * 8]! + verts[i1 * 8]! + verts[i2 * 8]!) / 3
    const my = (y0q + y1q + y2q) / 3
    const mz = (verts[i0 * 8 + 2]! + verts[i1 * 8 + 2]! + verts[i2 * 8 + 2]!) / 3
    const k = leafRank[nearest(bx0 + mx * sx - cx, by0 + my * sy, bz0 + mz * sz - cz)]!
    triCluster[t] = k
    triPerLeaf[k] = triPerLeaf[k]! + 1
    const lo = by0 + Math.min(y0q, y1q, y2q) * sy
    const hi = by0 + Math.max(y0q, y1q, y2q) * sy
    if (lo < leafYLo[k]!) leafYLo[k] = lo
    if (hi > leafYHi[k]!) leafYHi[k] = hi
  }
  const triStart = new Int32Array(K + 1)
  for (let k = 0; k < K; k++) triStart[k + 1] = triStart[k]! + triPerLeaf[k]!
  const indices = new Uint32Array(triCount * 3)
  {
    const cursor = triStart.slice()
    for (let t = 0; t < triCount; t++) {
      const k = triCluster[t]!
      const o = cursor[k]! * 3
      cursor[k] = cursor[k]! + 1
      indices[o] = tris[t * 4]!
      indices[o + 1] = tris[t * 4 + 1]!
      indices[o + 2] = tris[t * 4 + 2]!
    }
  }

  // --- 5. ribbon fits for every level --------------------------------------
  const strips = new Float32Array(STRIP_COUNT * STRIP_FLOATS)
  const ranges: { first: number; count: number }[] = []
  LEVELS.forEach((count, level) => {
    const span = K / count
    for (let j = 0; j < count; j++) {
      const leafFrom = j * span
      const leafTo = leafFrom + span
      let triLo = Infinity
      let triHi = -Infinity
      for (let leaf = leafFrom; leaf < leafTo; leaf++) {
        if (leafYLo[leaf]! < triLo) triLo = leafYLo[leaf]!
        if (leafYHi[leaf]! > triHi) triHi = leafYHi[leaf]!
      }
      const rib = fitRibbon(xs, ys, zs, sampleByLeaf, leafStart[leafFrom]!, leafStart[leafTo]!, [triLo, triHi])
      const o = (LEVEL_BASE[level]! + j) * STRIP_FLOATS
      for (let k = 0; k < NODES; k++) {
        strips[o + k * 4] = rib.nodes[k * 3]!
        strips[o + k * 4 + 1] = rib.nodes[k * 3 + 1]!
        strips[o + k * 4 + 2] = rib.nodes[k * 3 + 2]!
        strips[o + k * 4 + 3] = Math.min(rib.halfWidth[k]!, rXZ * 0.85)
      }
      strips[o + NODES * 4] = rib.latX
      strips[o + NODES * 4 + 1] = rib.latZ
      strips[o + NODES * 4 + 2] = rib.depth
      ranges.push({ first: triStart[leafFrom]! * 3, count: (triStart[leafTo]! - triStart[leafFrom]!) * 3 })
    }
  })

  // --- 6. GPU capture ------------------------------------------------------
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const ibuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-idx`,
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [BIG_W, BIG_H],
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
      size: [BIG_W, BIG_H],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // Per-tile uniforms, dynamic offsets (36 floats used of a 256B stride).
  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * SLOTS)
  for (let slot = 0; slot < SLOTS; slot++) {
    const o = (slot * STRIDE) / 4
    if (slot < STRIP_COUNT) {
      const s = slot * STRIP_FLOATS
      for (let k = 0; k < NODES; k++) {
        scratch[o + k * 4] = strips[s + k * 4]!
        scratch[o + k * 4 + 1] = strips[s + k * 4 + 1]!
        scratch[o + k * 4 + 2] = strips[s + k * 4 + 2]!
        scratch[o + k * 4 + 3] = strips[s + k * 4 + 3]!
      }
      scratch[o + 20] = strips[s + NODES * 4]! // lat.x
      scratch[o + 21] = strips[s + NODES * 4 + 1]! // lat.z
      scratch[o + 24] = strips[s + NODES * 4 + 2]! // depth range
      scratch[o + 25] = 0 // is_top
    } else {
      scratch[o + 20] = 1
      scratch[o + 21] = 0
      scratch[o + 24] = 1
      scratch[o + 25] = 1 // is_top
    }
    scratch[o + 26] = y0
    scratch[o + 27] = y1
    scratch.set([bx0, by0, bz0, 0], o + 28)
    scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 32)
    scratch.set([cx, cz, rXZ, 0], o + 36)
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const UNI_SIZE = 160
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: UNI_SIZE },
      },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: UNI_SIZE } }],
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

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/bake-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  const bigU = TILE_U * SS
  const bigV = TILE_V * SS
  for (let slot = 0; slot < SLOTS; slot++) {
    const col = slot % GRID_U
    const row = Math.floor(slot / GRID_U)
    pass.setViewport(col * bigU, row * bigV, bigU, bigV, 0, 1)
    pass.setScissorRect(col * bigU, row * bigV, bigU, bigV)
    pass.setBindGroup(0, bg, [slot * STRIDE])
    const range = ranges[slot]
    if (range) {
      if (range.count > 0) pass.drawIndexed(range.count, 1, range.first)
    } else {
      pass.drawIndexed(triCount * 3) // top view: the whole plant
    }
  }
  pass.end()

  const bpr = BIG_W * 4
  const rbSize = bpr * BIG_H
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbNrm = mkReadback('bake-rb-normal')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: BIG_H }, [BIG_W, BIG_H])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: BIG_H }, [BIG_W, BIG_H])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNrm = new Uint8Array(rbNrm.getMappedRange()).slice()
  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()

  const { albedo, normal } = postProcess(bigAlb, bigNrm)
  return packStrips({
    atlasPx: ATLAS_W,
    stripCount: STRIP_COUNT,
    nodeCount: NODES,
    rXZ,
    y0,
    y1,
    cx,
    cz,
    strips,
    albedo,
    normal,
  })
}

/**
 * Bottom-up greedy pairing of clusters by centroid distance, returning the
 * leaf ORDER of the resulting perfect binary tree — so at every level a
 * cluster is exactly one contiguous run of leaves.
 */
function buildMergeOrder(K: number, ccx: Float64Array, ccy: Float64Array, ccz: Float64Array): number[] {
  interface Node {
    x: number
    y: number
    z: number
    leaves: number[]
  }
  let nodes: Node[] = []
  for (let k = 0; k < K; k++) nodes.push({ x: ccx[k]!, y: ccy[k]!, z: ccz[k]!, leaves: [k] })
  while (nodes.length > 1) {
    const next: Node[] = []
    const used = new Set<number>()
    for (let i = 0; i < nodes.length; i++) {
      if (used.has(i)) continue
      let best = -1
      let bestD = Infinity
      for (let j = i + 1; j < nodes.length; j++) {
        if (used.has(j)) continue
        const a = nodes[i]!
        const b = nodes[j]!
        const d = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 * Y_WEIGHT + (a.z - b.z) ** 2
        if (d < bestD) {
          bestD = d
          best = j
        }
      }
      used.add(i)
      if (best < 0) {
        next.push(nodes[i]!)
        continue
      }
      used.add(best)
      const a = nodes[i]!
      const b = nodes[best]!
      next.push({
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        z: (a.z + b.z) / 2,
        leaves: [...a.leaves, ...b.leaves],
      })
    }
    nodes = next
  }
  return nodes[0]!.leaves
}

/** Coverage-weighted 2x downsample, then tile-clamped dilation, then pack. */
function postProcess(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
): { albedo: Uint8Array<ArrayBuffer>; normal: Uint8Array<ArrayBuffer> } {
  const W = ATLAS_W
  const H = ATLAS_H
  const albedo = new Uint8Array(W * H * 4)
  const normal = new Uint8Array(W * H * 2)
  const filled = new Uint8Array(W * H)

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nu = 0
      let nv = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * BIG_W + (x * SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
          nu += bigNrm[s]! * a
          nv += bigNrm[s + 1]! * a
        }
      }
      const d = (y * W + x) * 4
      if (aSum > 0) {
        albedo[d] = Math.round(r / aSum)
        albedo[d + 1] = Math.round(g / aSum)
        albedo[d + 2] = Math.round(b / aSum)
        albedo[d + 3] = Math.round(aSum / (SS * SS))
        normal[(y * W + x) * 2] = Math.round(nu / aSum)
        normal[(y * W + x) * 2 + 1] = Math.round(nv / aSum)
        filled[y * W + x] = 1
      } else {
        normal[(y * W + x) * 2] = 128
        normal[(y * W + x) * 2 + 1] = 128
      }
    }
  }

  // Dilate colour/normal into empty texels (alpha stays 0) so filtering never
  // blends toward background black. Clamped to tile boundaries.
  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (let y = 0; y < H; y++) {
      const ty0 = Math.floor(y / TILE_V) * TILE_V
      const ty1 = ty0 + TILE_V - 1
      for (let x = 0; x < W; x++) {
        const idx = y * W + x
        if (cur[idx]! !== 0) continue
        const tx0 = Math.floor(x / TILE_U) * TILE_U
        const tx1 = tx0 + TILE_U - 1
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nu = 0
        let nv = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const nIdx = yy * W + xx
            if (cur[nIdx]! === 0) continue
            count++
            r += albedo[nIdx * 4]!
            g += albedo[nIdx * 4 + 1]!
            b += albedo[nIdx * 4 + 2]!
            nu += normal[nIdx * 2]!
            nv += normal[nIdx * 2 + 1]!
          }
        }
        if (count === 0) continue
        albedo[idx * 4] = Math.round(r / count)
        albedo[idx * 4 + 1] = Math.round(g / count)
        albedo[idx * 4 + 2] = Math.round(b / count)
        normal[idx * 2] = Math.round(nu / count)
        normal[idx * 2 + 1] = Math.round(nv / count)
        next[idx] = 1
      }
    }
    cur = next
  }
  return { albedo, normal }
}

/** Alpha reference the mip chain preserves coverage against (params default). */
const MIP_ALPHA_REF = 0.42 * 255

/**
 * Rescale one tile's alpha at one mip level so the fraction of texels passing
 * the alpha test matches mip 0. Without this the chain is a coverage killer:
 * a strip tile is only ~10% covered, so plain box-filtered alpha falls under
 * the alpha reference within two levels and mid-distance plants dissolve into
 * the terrain. (Standard "alpha-preserving mipmaps" — the reason foliage in
 * shipped engines does not thin out with distance.)
 */
function matchCoverage(dst: Uint8Array, width: number, tileW: number, tileH: number, target: number[]): void {
  for (let ty = 0; ty < GRID_V; ty++) {
    for (let tx = 0; tx < GRID_U; tx++) {
      const want = target[ty * GRID_U + tx]!
      if (want <= 0) continue
      const total = tileW * tileH
      const cover = (scale: number): number => {
        let n = 0
        for (let y = 0; y < tileH; y++) {
          const row = (ty * tileH + y) * width + tx * tileW
          for (let x = 0; x < tileW; x++) {
            if (dst[(row + x) * 4 + 3]! * scale >= MIP_ALPHA_REF) n++
          }
        }
        return n / total
      }
      if (cover(1) >= want) continue
      let lo = 1
      let hi = 64
      for (let it = 0; it < 12; it++) {
        const mid = (lo + hi) / 2
        if (cover(mid) < want) lo = mid
        else hi = mid
      }
      const scale = (lo + hi) / 2
      for (let y = 0; y < tileH; y++) {
        const row = (ty * tileH + y) * width + tx * tileW
        for (let x = 0; x < tileW; x++) {
          const i = (row + x) * 4 + 3
          dst[i] = Math.min(255, Math.round(dst[i]! * scale))
        }
      }
    }
  }
}

/** Per-tile fraction of texels that pass the alpha test at mip 0. */
function tileCoverage(albedo: Uint8Array): number[] {
  const out: number[] = []
  for (let ty = 0; ty < GRID_V; ty++) {
    for (let tx = 0; tx < GRID_U; tx++) {
      let n = 0
      for (let y = 0; y < TILE_V; y++) {
        const row = (ty * TILE_V + y) * ATLAS_W + tx * TILE_U
        for (let x = 0; x < TILE_U; x++) {
          if (albedo[(row + x) * 4 + 3]! >= MIP_ALPHA_REF) n++
        }
      }
      out.push(n / (TILE_U * TILE_V))
    }
  }
  return out
}

/**
 * Coverage-weighted, coverage-PRESERVING mip chain for the atlas, built on the
 * CPU at load time. A 2x2 box never straddles a tile boundary (tiles are even
 * and aligned), so tiles stay independent at every level.
 */
export function buildMips(
  albedo: Uint8Array,
  normal: Uint8Array,
): { albedo: Uint8Array<ArrayBuffer>[]; normal: Uint8Array<ArrayBuffer>[] } {
  const target = tileCoverage(albedo)
  const albLevels: Uint8Array<ArrayBuffer>[] = [new Uint8Array(albedo)]
  const nrmLevels: Uint8Array<ArrayBuffer>[] = [new Uint8Array(normal)]
  let w = ATLAS_W
  let h = ATLAS_H
  for (let level = 1; level < MIP_LEVELS; level++) {
    const srcW = w
    const dstW = Math.max(1, w >> 1)
    const dstH = Math.max(1, h >> 1)
    const srcA = albLevels[level - 1]!
    const srcN = nrmLevels[level - 1]!
    const dstA = new Uint8Array(dstW * dstH * 4)
    const dstN = new Uint8Array(dstW * dstH * 2)
    for (let y = 0; y < dstH; y++) {
      for (let x = 0; x < dstW; x++) {
        let aSum = 0
        let r = 0
        let g = 0
        let b = 0
        let nu = 0
        let nv = 0
        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 2; i++) {
            const s = (y * 2 + j) * srcW + (x * 2 + i)
            const a = srcA[s * 4 + 3]!
            aSum += a
            r += srcA[s * 4]! * a
            g += srcA[s * 4 + 1]! * a
            b += srcA[s * 4 + 2]! * a
            nu += srcN[s * 2]! * a
            nv += srcN[s * 2 + 1]! * a
          }
        }
        const d = (y * dstW + x) * 4
        const wt = Math.max(aSum, 1e-4)
        dstA[d] = Math.round(r / wt)
        dstA[d + 1] = Math.round(g / wt)
        dstA[d + 2] = Math.round(b / wt)
        dstA[d + 3] = Math.round(aSum / 4)
        dstN[(y * dstW + x) * 2] = aSum > 0 ? Math.round(nu / wt) : 128
        dstN[(y * dstW + x) * 2 + 1] = aSum > 0 ? Math.round(nv / wt) : 128
      }
    }
    matchCoverage(dstA, dstW, Math.max(1, TILE_U >> level), Math.max(1, TILE_V >> level), target)
    albLevels.push(dstA)
    nrmLevels.push(dstN)
    w = dstW
    h = dstH
  }
  return { albedo: albLevels, normal: nrmLevels }
}
