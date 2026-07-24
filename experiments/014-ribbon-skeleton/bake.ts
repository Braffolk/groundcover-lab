import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Ribbon-skeleton distiller.
 *
 * A plant is reduced to a nested hierarchy of CURVED RIBBONS. The source mesh
 * is swept into a cylindrical histogram around a tuft centre — SECTORS azimuth
 * wedges x HBINS height bins — accumulating area-weighted position, authored
 * colour, radially-projected area and radial spread. One ribbon per wedge is
 * then fitted: its centreline is the area-centroid curve of the wedge, and its
 * half-width is COVERAGE-CONSERVING (the wedge's real silhouette area divided
 * by the centreline arc length), capped by the wedge's own arc. That single
 * rule is what makes the representation work at every scale: at 32 wedges each
 * ribbon is far thinner than its wedge and reads as a blade; at 4 wedges each
 * ribbon saturates its arc and reads as a solid tuft.
 *
 * Levels are built by merging adjacent wedges (32 -> 16 -> 8 -> 4), so ribbon
 * i of level L always has parent i>>1 at level L+1. The renderer morphs a
 * ribbon into its parent with distance; when the morph completes, ribbon pairs
 * are coincident and dropping half of them is invisible — continuous LOD with
 * no popping and no stochastic alpha anywhere.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'RIB2', u32 version, u32 variants, u32 levels
 *   u32 samples, u32 rowsPerVariant, f32 radiusFrac, f32 heightM
 *   u32 levelRibbons[4], u32 levelBase[4]
 *   ...zeros to byte 64
 *   f32[rows*samples*4]  geom  : centreline xyz + half-width (nominal heights)
 *   f32[rows*samples*4]  shade : authored rgb + packed(2*fluffLevel + ao)
 * where rows = variants * rowsPerVariant, row = variant*rowsPerVariant +
 * levelBase[level] + ribbon, and the sample axis is the texture's x axis.
 */

export const SECTORS = 32
export const HBINS = 96
export const SAMPLES = 12
export const VARIANTS = 6
export const LEVEL_RIBBONS = [32, 16, 8, 4] as const
export const LEVEL_BASE = [0, 32, 48, 56] as const
export const ROWS_PER_VARIANT = 60
export const ATLAS_ROWS = VARIANTS * ROWS_PER_VARIANT
export const ATLAS_VERSION = 4

const MAGIC = 0x32424952 // 'RIB2'
const HEADER_BYTES = 64
/** Accumulator fields per (variant, sector, height bin). */
const ACC = 10
const A_AREA = 0
const A_X = 1
const A_Y = 2
const A_Z = 3
const A_R = 4
const A_G = 5
const A_B = 6
const A_PROJ = 7
const A_RAD = 8
const A_RAD2 = 9

/** Sample window height / sample spacing — windows overlap, nothing is missed. */
const WINDOW_SCALE = 1.5
/** Blade surfaces are modelled front+back in these meshes; halve the coverage. */
const COVER = 0.5
/** Ambient-occlusion falloff against the foliage area above a sample. */
const AO_K = 0.7
/**
 * How much of its own wedge arc a ribbon may fill, per level. Fine levels stay
 * under 1 so blades keep gaps between them; coarse levels close up into a solid
 * tuft, which is what a plant looks like once it is a few pixels tall.
 */
const ARC_CAP = [0.8, 0.95, 1.05, 1.12]
/**
 * Fluffy wedges (panicles) get a much tighter cap: their silhouette estimate
 * sums every floret's projected area and ignores self-occlusion, so it wildly
 * overstates a structure that is mostly air.
 */
const FLUFF_CAP = 0.62
/** Surface area per unit wedge volume (1/m) at which foliage reads as a volume. */
const FLUFF_LO = 12
const FLUFF_HI = 45

export interface RibbonAtlas {
  /** rgba per (row, sample): centreline xyz + half-width, in nominal heights. */
  geom: Float32Array
  /** rgba per (row, sample): authored rgb + packed(2*fluffLevel + ao). */
  shade: Float32Array
  /** Largest drawn XZ radius / nominal height — feeds the screen-size LOD. */
  radiusFrac: number
  /** Nominal foliage height of the distilled tuft, metres (informational). */
  heightM: number
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'meshes'>

// ---------------------------------------------------------------------------
// artifact io
// ---------------------------------------------------------------------------

export function unpackAtlas(buf: ArrayBuffer): RibbonAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 16)
  const f = new Float32Array(buf, 0, 16)
  if (u[0] !== MAGIC || u[1] !== ATLAS_VERSION) return null
  if (u[2] !== VARIANTS || u[3] !== LEVEL_RIBBONS.length) return null
  if (u[4] !== SAMPLES || u[5] !== ROWS_PER_VARIANT) return null
  const floats = ATLAS_ROWS * SAMPLES * 4
  if (buf.byteLength !== HEADER_BYTES + floats * 8) return null
  return {
    geom: new Float32Array(buf, HEADER_BYTES, floats),
    shade: new Float32Array(buf, HEADER_BYTES + floats * 4, floats),
    radiusFrac: f[6]!,
    heightM: f[7]!,
  }
}

function packAtlas(a: RibbonAtlas): ArrayBuffer {
  const floats = ATLAS_ROWS * SAMPLES * 4
  const buf = new ArrayBuffer(HEADER_BYTES + floats * 8)
  const u = new Uint32Array(buf, 0, 16)
  const f = new Float32Array(buf, 0, 16)
  u[0] = MAGIC
  u[1] = ATLAS_VERSION
  u[2] = VARIANTS
  u[3] = LEVEL_RIBBONS.length
  u[4] = SAMPLES
  u[5] = ROWS_PER_VARIANT
  f[6] = a.radiusFrac
  f[7] = a.heightM
  for (let i = 0; i < 4; i++) {
    u[8 + i] = LEVEL_RIBBONS[i]!
    u[12 + i] = LEVEL_BASE[i]!
  }
  new Float32Array(buf, HEADER_BYTES, floats).set(a.geom)
  new Float32Array(buf, HEADER_BYTES + floats * 4, floats).set(a.shade)
  return buf
}

/**
 * Load the committed/cached atlas, or distil it in-browser from the raw
 * GCMESH1 mesh. Every result is magic-validated before it is trusted: the dev
 * server answers a missing /mesh/baked file with index.html at status 200,
 * which would otherwise poison the caches.
 */
export async function loadRibbonAtlas(ctx: BakeCtx, speciesId: string): Promise<RibbonAtlas> {
  const key = `ribbons-v${ATLAS_VERSION}-s${SECTORS}-k${SAMPLES}-va${VARIANTS}-${speciesId}`
  let freshlyBaked = false
  const runBake = async (): Promise<ArrayBuffer> => {
    freshlyBaked = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    const t0 = performance.now()
    const atlas = distill(mesh)
    console.info(`[${ctx.id}] distilled ${speciesId} in ${Math.round(performance.now() - t0)}ms`)
    return packAtlas(atlas)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackAtlas(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackAtlas(buf)
    if (!atlas) throw new Error(`[${ctx.id}] ${speciesId}: distiller produced an invalid artifact`)
  }
  if (freshlyBaked) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] could not commit ${speciesId} bake:`, err)
    }
  }
  return atlas
}

// ---------------------------------------------------------------------------
// distillation
// ---------------------------------------------------------------------------

const TWO_PI = Math.PI * 2
/** R2 low-discrepancy sequence — deterministic, well-spread variant centres. */
const R2A = 0.7548776662466927
const R2B = 0.5698402909980532
const PHI = 0.6180339887498949

/** Stride-subsample huge meshes; every bin still receives hundreds of triangles. */
function pickStride(triangles: number): number {
  for (const p of [1, 3, 7, 11, 13, 17, 19, 23, 29]) {
    if (triangles / p <= 1.2e6) return p
  }
  return 31
}

export function distill(mesh: GcMesh): RibbonAtlas {
  const h = mesh.header
  const verts = mesh.vertices
  const tris = mesh.triangles
  const bx = h.boundsMin[0]
  const by = h.boundsMin[1]
  const bz = h.boundsMin[2]
  const sx = (h.boundsMax[0] - bx) / 65535
  const sy = (h.boundsMax[1] - by) / 65535
  const sz = (h.boundsMax[2] - bz) / 65535
  const yMax = Math.max(h.boundsMax[1], 1e-3)
  const invYMax = 1 / yMax

  const tiled = h.tileSize[0] > 1e-4 && h.tileSize[1] > 1e-4
  const tileX = h.tileSize[0]
  const tileZ = h.tileSize[1]

  // --- variant frames -------------------------------------------------------
  const cX = new Float64Array(VARIANTS)
  const cZ = new Float64Array(VARIANTS)
  const rot = new Float64Array(VARIANTS)
  let gatherR: number
  if (tiled) {
    // One plant = roughly one community tile of foliage, gathered around a
    // different centre per variant (periodic folding keeps edge-crossing
    // foliage with its tuft), so the six variants are genuinely different
    // pieces of the source mesh.
    gatherR = 0.58 * Math.min(tileX, tileZ)
    for (let v = 0; v < VARIANTS; v++) {
      cX[v] = h.tileOrigin[0] + (((v + 1) * R2A) % 1) * tileX
      cZ[v] = h.tileOrigin[1] + (((v + 1) * R2B) % 1) * tileZ
      rot[v] = (((v + 1) * PHI) % 1) * TWO_PI
    }
  } else {
    const mx = (h.boundsMin[0] + h.boundsMax[0]) * 0.5
    const mz = (h.boundsMin[2] + h.boundsMax[2]) * 0.5
    const specimenR = 0.5 * Math.max(h.boundsMax[0] - h.boundsMin[0], h.boundsMax[2] - h.boundsMin[2])
    gatherR = specimenR * 1.25
    for (let v = 0; v < VARIANTS; v++) {
      cX[v] = mx + ((((v + 1) * R2A) % 1) - 0.5) * 0.24 * specimenR
      cZ[v] = mz + ((((v + 1) * R2B) % 1) - 0.5) * 0.24 * specimenR
      rot[v] = (((v + 1) * PHI) % 1) * TWO_PI
    }
  }
  const gatherR2 = gatherR * gatherR

  // --- sweep the mesh into cylindrical histograms ---------------------------
  const acc = new Float64Array(VARIANTS * SECTORS * HBINS * ACC)
  const stride = pickStride(h.triangleCount)
  const sectorScale = SECTORS / TWO_PI

  for (let t = 0; t < h.triangleCount; t += stride) {
    const o = t * 4
    const a0 = tris[o]! * 8
    const a1 = tris[o + 1]! * 8
    const a2 = tris[o + 2]! * 8

    const x0 = bx + verts[a0]! * sx
    const y0 = by + verts[a0 + 1]! * sy
    const z0 = bz + verts[a0 + 2]! * sz
    const x1 = bx + verts[a1]! * sx
    const y1 = by + verts[a1 + 1]! * sy
    const z1 = bz + verts[a1 + 2]! * sz
    const x2 = bx + verts[a2]! * sx
    const y2 = by + verts[a2 + 1]! * sy
    const z2 = bz + verts[a2 + 2]! * sz

    const ex1 = x1 - x0
    const ey1 = y1 - y0
    const ez1 = z1 - z0
    const ex2 = x2 - x0
    const ey2 = y2 - y0
    const ez2 = z2 - z0
    const nx = ey1 * ez2 - ez1 * ey2
    const ny = ez1 * ex2 - ex1 * ez2
    const nz = ex1 * ey2 - ey1 * ex2
    const nl = Math.sqrt(nx * nx + ny * ny + nz * nz)
    if (nl < 1e-12) continue
    const area = 0.5 * nl * stride
    const inx = nx / nl
    const inz = nz / nl

    const px = (x0 + x1 + x2) / 3
    const py = (y0 + y1 + y2) / 3
    const pz = (z0 + z1 + z2) / 3

    const cr = (verts[a0 + 3]! + verts[a1 + 3]! + verts[a2 + 3]!) / 196605
    const cg = (verts[a0 + 4]! + verts[a1 + 4]! + verts[a2 + 4]!) / 196605
    const cb = (verts[a0 + 5]! + verts[a1 + 5]! + verts[a2 + 5]!) / 196605

    let hb = (py * invYMax * HBINS) | 0
    if (hb < 0) hb = 0
    else if (hb >= HBINS) hb = HBINS - 1

    for (let v = 0; v < VARIANTS; v++) {
      let dx = px - cX[v]!
      let dz = pz - cZ[v]!
      if (tiled) {
        dx -= Math.round(dx / tileX) * tileX
        dz -= Math.round(dz / tileZ) * tileZ
      }
      const r2 = dx * dx + dz * dz
      if (r2 > gatherR2) continue
      const rr = Math.sqrt(r2)

      let ang = Math.atan2(dz, dx) - rot[v]!
      ang -= Math.floor(ang / TWO_PI) * TWO_PI
      let sec = (ang * sectorScale) | 0
      if (sec < 0) sec = 0
      else if (sec >= SECTORS) sec = SECTORS - 1

      // Silhouette area seen from outside, along this triangle's own radial.
      const proj = rr > 1e-6 ? Math.abs((inx * dx + inz * dz) / rr) * area : area * 0.5

      const base = ((v * SECTORS + sec) * HBINS + hb) * ACC
      acc[base + A_AREA]! += area
      acc[base + A_X]! += area * dx
      acc[base + A_Y]! += area * py
      acc[base + A_Z]! += area * dz
      acc[base + A_R]! += area * cr
      acc[base + A_G]! += area * cg
      acc[base + A_B]! += area * cb
      acc[base + A_PROJ]! += proj
      acc[base + A_RAD]! += area * rr
      acc[base + A_RAD2]! += area * rr * rr
    }
  }

  // --- fit ribbons ----------------------------------------------------------
  const geom = new Float32Array(ATLAS_ROWS * SAMPLES * 4)
  const shade = new Float32Array(ATLAS_ROWS * SAMPLES * 4)
  const merged = new Float64Array(HBINS * ACC)
  const areaProfile = new Float64Array(HBINS)
  const aoProfile = new Float64Array(HBINS)
  const win = new Float64Array(ACC)
  const cx = new Float64Array(SAMPLES)
  const cy = new Float64Array(SAMPLES)
  const cz = new Float64Array(SAMPLES)
  const sProj = new Float64Array(SAMPLES)
  const sArea = new Float64Array(SAMPLES)
  const sSigma = new Float64Array(SAMPLES)
  const colR = new Float64Array(SAMPLES)
  const colG = new Float64Array(SAMPLES)
  const colB = new Float64Array(SAMPLES)
  const valid = new Uint8Array(SAMPLES)
  let radiusFrac = 0
  let heightM = 0

  for (let v = 0; v < VARIANTS; v++) {
    // Whole-plant vertical area profile: sets the nominal height and the AO.
    areaProfile.fill(0)
    for (let s = 0; s < SECTORS; s++) {
      for (let b = 0; b < HBINS; b++) {
        areaProfile[b]! += acc[((v * SECTORS + s) * HBINS + b) * ACC + A_AREA]!
      }
    }
    let totalArea = 0
    let maxBin = 0
    for (let b = 0; b < HBINS; b++) {
      totalArea += areaProfile[b]!
      maxBin = Math.max(maxBin, areaProfile[b]!)
    }
    if (totalArea <= 0) continue
    let topBin = 0
    for (let b = HBINS - 1; b >= 0; b--) {
      if (areaProfile[b]! > 0.0015 * maxBin) {
        topBin = b
        break
      }
    }
    const plantH = Math.max(((topBin + 1) / HBINS) * yMax, 1e-3)
    heightM = Math.max(heightM, plantH)

    // ao(y) = exp(-K * fraction of the plant's foliage area above y)
    let above = 0
    for (let b = HBINS - 1; b >= 0; b--) {
      above += areaProfile[b]!
      aoProfile[b] = Math.exp(-AO_K * Math.min(1, above / totalArea))
    }

    for (let level = 0; level < LEVEL_RIBBONS.length; level++) {
      const n = LEVEL_RIBBONS[level]!
      const span = SECTORS / n
      const arc = TWO_PI / n
      for (let ribbon = 0; ribbon < n; ribbon++) {
        merged.fill(0)
        for (let k = 0; k < span; k++) {
          const s = ribbon * span + k
          for (let b = 0; b < HBINS; b++) {
            const src = ((v * SECTORS + s) * HBINS + b) * ACC
            const dst = b * ACC
            for (let f = 0; f < ACC; f++) merged[dst + f]! += acc[src + f]!
          }
        }

        const row = v * ROWS_PER_VARIANT + LEVEL_BASE[level]! + ribbon
        let ribbonMax = 0
        let ribbonTotal = 0
        for (let b = 0; b < HBINS; b++) {
          ribbonMax = Math.max(ribbonMax, merged[b * ACC + A_AREA]!)
          ribbonTotal += merged[b * ACC + A_AREA]!
        }
        if (ribbonTotal <= 0) continue // empty wedge: zero row = invisible ribbon

        // This wedge's own top — short blades still get the full sample budget.
        let rTop = 0
        for (let b = HBINS - 1; b >= 0; b--) {
          if (merged[b * ACC + A_AREA]! > 0.004 * ribbonMax) {
            rTop = b
            break
          }
        }
        const topY = Math.max(((rTop + 1) / HBINS) * yMax, 1e-3)
        const dy = topY / (SAMPLES - 1)
        const halfWin = dy * 0.5 * WINDOW_SCALE

        // pass 1: centreline, colour, projected area, radial spread
        for (let j = 0; j < SAMPLES; j++) {
          const yc = dy * j
          win.fill(0)
          const b0 = Math.max(0, Math.floor((yc - halfWin) * invYMax * HBINS))
          const b1 = Math.min(HBINS - 1, Math.floor((yc + halfWin) * invYMax * HBINS))
          for (let b = b0; b <= b1; b++) {
            const src = b * ACC
            for (let f = 0; f < ACC; f++) win[f]! += merged[src + f]!
          }
          const a = win[A_AREA]!
          valid[j] = a > 0 ? 1 : 0
          if (a > 0) {
            cx[j] = win[A_X]! / a
            cy[j] = win[A_Y]! / a
            cz[j] = win[A_Z]! / a
            colR[j] = win[A_R]! / a
            colG[j] = win[A_G]! / a
            colB[j] = win[A_B]! / a
            sProj[j] = win[A_PROJ]!
            sArea[j] = a
            const meanR = win[A_RAD]! / a
            sSigma[j] = Math.sqrt(Math.max(0, win[A_RAD2]! / a - meanR * meanR))
          } else {
            sProj[j] = 0
            sArea[j] = 0
            sSigma[j] = 0
          }
        }

        // fill gaps from the nearest valid neighbours (width stays 0 there)
        for (let j = 0; j < SAMPLES; j++) {
          if (valid[j]) continue
          let lo = j - 1
          while (lo >= 0 && !valid[lo]) lo--
          let hi = j + 1
          while (hi < SAMPLES && !valid[hi]) hi++
          if (lo < 0 && hi >= SAMPLES) break
          if (lo < 0) {
            cx[j] = cx[hi]! * 0.5
            cz[j] = cz[hi]! * 0.5
            colR[j] = colR[hi]!
            colG[j] = colG[hi]!
            colB[j] = colB[hi]!
          } else if (hi >= SAMPLES) {
            cx[j] = cx[lo]!
            cz[j] = cz[lo]!
            colR[j] = colR[lo]!
            colG[j] = colG[lo]!
            colB[j] = colB[lo]!
          } else {
            const f = (j - lo) / (hi - lo)
            cx[j] = cx[lo]! + (cx[hi]! - cx[lo]!) * f
            cz[j] = cz[lo]! + (cz[hi]! - cz[lo]!) * f
            colR[j] = colR[lo]! + (colR[hi]! - colR[lo]!) * f
            colG[j] = colG[lo]! + (colG[hi]! - colG[lo]!) * f
            colB[j] = colB[lo]! + (colB[hi]! - colB[lo]!) * f
          }
          cy[j] = dy * j
        }
        // Ribbons emanate from the tuft base: ground the root sample and pull
        // it toward the axis, so every level reads as one plant, not a ring.
        cx[0]! *= 0.25
        cz[0]! *= 0.25
        cy[0] = 0

        // pass 2: coverage-conserving widths, fluffiness, ambient occlusion
        const invH = 1 / plantH
        for (let j = 0; j < SAMPLES; j++) {
          const jp = Math.max(0, j - 1)
          const jn = Math.min(SAMPLES - 1, j + 1)
          const dPrev = Math.hypot(cx[j]! - cx[jp]!, cy[j]! - cy[jp]!, cz[j]! - cz[jp]!)
          const dNext = Math.hypot(cx[jn]! - cx[j]!, cy[jn]! - cy[j]!, cz[jn]! - cz[j]!)
          const segLen = Math.max(0.5 * (dPrev + dNext), 1e-4)
          const rj = Math.hypot(cx[j]!, cz[j]!)
          // Volumetric-ness: surface area packed into the wedge cell this
          // sample owns (m^2 per m^3). A blade is a sheet crossing the cell;
          // a panicle fills it with hundreds of florets.
          const cellVol =
            Math.max(2 * sSigma[j]!, 0.002) * arc * Math.max(rj, 0.05 * gatherR) * (2 * halfWin)
          const fluff = smoothstep(FLUFF_LO, FLUFF_HI, sArea[j]! / Math.max(cellVol, 1e-9))
          const cap = ARC_CAP[level]! * (1 - FLUFF_CAP * fluff)
          let w = (COVER * sProj[j]!) / (2 * segLen * WINDOW_SCALE)
          w = Math.min(w, 0.5 * arc * Math.max(rj, 0.18 * gatherR) * cap, 0.42 * gatherR)
          // A panicle is feathery, not a strap: ragged the width of fluffy
          // samples (mean-preserving, deterministic) so plumes get a broken
          // silhouette instead of one flat quad per wedge.
          w *= 1 + fluff * (0.45 + 1.1 * ragged(row, j) - 1)
          const yb = Math.min(HBINS - 1, Math.max(0, Math.floor(cy[j]! * invYMax * HBINS)))
          const ao = Math.min(0.99, Math.max(0.3, aoProfile[yb]!))
          const fluffLevel = Math.round(fluff * 7)

          const o = (row * SAMPLES + j) * 4
          geom[o] = cx[j]! * invH
          geom[o + 1] = cy[j]! * invH
          geom[o + 2] = cz[j]! * invH
          geom[o + 3] = w * invH
          shade[o] = colR[j]!
          shade[o + 1] = colG[j]!
          shade[o + 2] = colB[j]!
          shade[o + 3] = 2 * fluffLevel + ao
          radiusFrac = Math.max(radiusFrac, (rj + w) * invH)
        }
      }
    }
  }

  return { geom, shade, radiusFrac: Math.max(radiusFrac, 0.05), heightM }
}

/** Deterministic [0,1) jitter per (atlas row, sample) — no Math.random anywhere. */
function ragged(row: number, sample: number): number {
  let h = Math.imul(row * 73856093 + sample * 19349663 + 0x9e3779b9, 2246822519) >>> 0
  h ^= h >>> 13
  h = Math.imul(h, 3266489917) >>> 0
  return (h >>> 8) / 16777216
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}
