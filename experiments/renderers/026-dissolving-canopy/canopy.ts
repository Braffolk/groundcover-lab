import { SCATTER_CELL_SIZE, hash2, hash4, hashF32, type Stand } from '@harness'
import { ALB_W, CT_PX, TILE_H, TILE_W, TOP_PX, type CanopyBake, type CarpetTileBake } from './bake.ts'

/**
 * The continuous end of the dissolve, built at init (not baked): ONE tileable
 * canopy texture for the ACTIVE STAND, composited on the CPU out of the baked
 * per-species straight-down captures.
 *
 * Why composite instead of bake: the shell has to represent the stand's actual
 * species mix and densities, and those are not known at bake time. Stamping the
 * top captures into a torus-wrapped tile is cheap (a few M texel ops, once) and
 * makes the far field agree with the near field for whatever stand is loaded.
 *
 * Compositing rules, in the order they matter:
 *   * position/yaw/scale of every copy come from the shared PCG hash + seed, so
 *     the texture is deterministic;
 *   * yaw is one of 8 free orientations (4 rotations x mirror), so rotation
 *     introduces no resampling blur;
 *   * COVERAGE accumulates as independent occluders: 1 - prod(1 - c_i). Overlaps
 *     therefore close up like real foliage instead of clobbering each other;
 *   * COLOUR / NORMAL / HEIGHT are composited front-to-back: pass 1 finds the
 *     top of the canopy per texel, pass 2 weights every copy by its coverage
 *     attenuated by how far below that top it sits. A wispy panicle above a
 *     solid leaf therefore contributes exactly its own coverage instead of
 *     replacing what is under it.
 */

export const CANOPY_TEX = 768
export const CANOPY_TILE_M = 12 // 64 texels/m; the repeat is 12 m, i.e. ~17deg at 40 m
export const CANOPY_MIPS = 9

export interface CanopyTile {
  /** rgb = albedo, a = coverage. */
  albedo: Uint8Array<ArrayBuffer>
  /** rg = oct normal, b = canopy height / heightMax. */
  aux: Uint8Array<ArrayBuffer>
  /** Metres corresponding to aux.b = 1 (the canopy's 97th-percentile top). */
  heightMax: number
  /** 85th-percentile canopy top (m) — the shell surface's base height. */
  baseHeight: number
  /** Density-weighted mean wind response of the stand. */
  meanSway: number
  /** Mean coverage of the tile (sanity/diagnostics). */
  meanCoverage: number
  /**
   * Coverage-weighted mean colours of the SIDE views, split into the flower-head
   * band and everything below it, plus this tile's mean luminance. At grazing the
   * shell rebuilds its albedo as mix(low, top, canopyHeight) * localLuma/meanLuma:
   * a meadow's colour from above is not its colour from the side (the heads cover
   * little of the footprint from above but a wide band of the silhouette from the
   * side), and using the baked canopy HEIGHT to pick between the two bands puts
   * the flower colour exactly where the tall plants are. No extra taps.
   */
  sideTopColor: [number, number, number]
  sideLowColor: [number, number, number]
  meanLuma: number
  /**
   * Multiplier turning the aux height channel into the flower-band mix factor.
   * Chosen so that a fully mip-averaged sample lands on the flower band's real
   * share of the silhouette — the far field then averages to the right hue while
   * mid distances still put the flower colour on the tall texels only.
   */
  hueGain: number
}

/** How much less the plant's base counts in the side-view mean (it is hidden). */
const SIDE_TOP_BIAS = 0.55
/** Fraction of a plant's silhouette height counted as the flower/seed-head band. */
const FLOWER_BAND = 0.3
/** Mean flower-band weight of the fully averaged far field. */
const FAR_FLOWER_MIX = 0.72
/** Self-shadowing of the canopy interior seen edge-on. */
const CANOPY_INTERIOR = 0.7
/** Below this coverage a copy cannot define where the canopy top is. */
const TOP_MIN_COV = 0.15
/** Front-to-back falloff (1/m) under the canopy top. */
const DEPTH_FALLOFF = 3

interface Stamp {
  /** rgba8 source planes: rgb + coverage, and oct normal + height. */
  srcA: Uint8Array
  srcX: Uint8Array
  srcPx: number
  px: number
  pz: number
  sizePx: number
  plantH: number
  orient: number
}

/** Downsample a carpet tile to `n` px, coverage-weighted, for stamping. */
function shrinkTile(tile: CarpetTileBake, n: number): { a: Uint8Array; x: Uint8Array } {
  const S = CT_PX
  const k = Math.floor(S / n)
  const a = new Uint8Array(n * n * 4)
  const x = new Uint8Array(n * n * 4)
  for (let y = 0; y < n; y++) {
    for (let i = 0; i < n; i++) {
      let cov = 0
      let w = 0
      let cr = 0
      let cg = 0
      let cb = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let h = 0
      for (let j = 0; j < k; j++) {
        for (let m = 0; m < k; m++) {
          const s = ((y * k + j) * S + i * k + m) * 4
          const av = tile.albedo[s + 3]!
          cov += av
          if (av === 0) continue
          w += av
          cr += tile.albedo[s]! * av
          cg += tile.albedo[s + 1]! * av
          cb += tile.albedo[s + 2]! * av
          // Oct pairs are NOT box-filterable: decode, average, re-encode.
          const d = octDecodeOriented((tile.aux[s]! / 255) * 2 - 1, (tile.aux[s + 1]! / 255) * 2 - 1, 0)
          nx += d[0] * av
          ny += d[1] * av
          nz += d[2] * av
          h += tile.aux[s + 2]! * av
        }
      }
      const o = (y * n + i) * 4
      a[o + 3] = Math.round(cov / (k * k))
      if (w === 0) continue
      a[o] = clamp255(cr / w)
      a[o + 1] = clamp255(cg / w)
      a[o + 2] = clamp255(cb / w)
      const [u, v] = octEncode(nx / w, ny / w, nz / w)
      x[o] = u
      x[o + 1] = v
      x[o + 2] = Math.round(h / w)
    }
  }
  return { a, x }
}

export function compositeCanopy(
  stand: Stand,
  bakes: Map<string, CanopyBake>,
  seed: number,
  tiles?: Map<string, CarpetTileBake>,
): CanopyTile {
  const N = CANOPY_TEX
  const pxPerM = N / CANOPY_TILE_M
  /** Constant tile scale of a carpet entry, or null for a scattered entry. */
  const carpetOf = (e: Stand['species'][number]): { tile: CarpetTileBake; scale: number } | null => {
    const div = e.carpetDiv ?? 0
    const tile = div > 0 ? tiles?.get(e.species) : undefined
    return tile ? { tile, scale: SCATTER_CELL_SIZE / div / tile.tileM } : null
  }

  let heightMax = 0.1
  let swayNum = 0
  let swayDen = 0
  const sideTop: [number, number, number] = [0, 0, 0]
  const sideLow: [number, number, number] = [0, 0, 0]

  for (const e of stand.species) {
    const b = bakes.get(e.species)
    if (!b) continue
    const carpet = carpetOf(e)
    // A carpet's scaleMin/scaleMax in the stand are placeholders — the real
    // scale is the one that makes a tile fill its grid step (life size, ~1.01).
    // Taking the placeholders literally composited the moss at ~2x height and
    // width, which put the shell's surface 20cm above a 7cm bog.
    heightMax = Math.max(heightMax, carpet ? (carpet.tile.y1 - carpet.tile.y0) * carpet.scale : b.plantH * e.scaleMax)
    swayNum += e.density * e.sway
    swayDen += e.density
    const top = meanSideColor(b, 0, FLOWER_BAND)
    const low = meanSideColor(b, FLOWER_BAND, 1)
    for (let c = 0; c < 3; c++) {
      sideTop[c] = sideTop[c]! + e.density * top[c]!
      sideLow[c] = sideLow[c]! + e.density * low[c]!
    }
  }

  if (swayDen > 0) {
    for (let c = 0; c < 3; c++) {
      // INTERIOR: at grazing the shell is opaque, so its albedo has to stand for
      // the whole canopy volume behind the first blade — and that volume is
      // self-shadowed. Without this the far field reads brighter than the splat
      // field in front of it and the hand-off shows as a tone step.
      sideTop[c] = (sideTop[c]! / swayDen) * CANOPY_INTERIOR
      sideLow[c] = (sideLow[c]! / swayDen) * CANOPY_INTERIOR
    }
  }

  const stamps: Stamp[] = []
  // Carpet entries share ONE grid and partition it, exactly as they partition
  // the wetness axis in the stand: a node is claimed by one of them, so the
  // composite gets the mat's real mix instead of three overlapping mats.
  const carpetEntries = stand.species.filter((e) => carpetOf(e) !== null)
  stand.species.forEach((e, entryIndex) => {
    const bake = bakes.get(e.species)
    if (!bake) return
    const carpet = carpetOf(e)
    if (carpet) {
      const { tile, scale } = carpet
      const step = tile.tileM * scale
      const n = Math.max(1, Math.round(CANOPY_TILE_M / step))
      const stepPx = (CANOPY_TILE_M / n) * pxPerM
      // 512px stamped into ~12px would alias badly; pre-reduce once instead.
      const small = shrinkTile(tile, 32)
      const ord = carpetEntries.indexOf(e)
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          // NODE-only hash — hashing the entry index in would let the three
          // states disagree about who owns a node, which is exactly how the
          // scatter's own zoning once grew holes and double stacks.
          const h = hash4(seed, 0x5eed, j * n + i, 0x51ed2701)
          if (carpetEntries.length > 1 && hash2(h, 7) % carpetEntries.length !== ord) continue
          stamps.push({
            srcA: small.a,
            srcX: small.x,
            srcPx: 32,
            px: i * stepPx,
            pz: j * stepPx,
            sizePx: stepPx,
            plantH: (tile.y1 - tile.y0) * scale,
            // Quarter turns only: a mat's tiles must agree with their neighbours.
            orient: hash2(h, 4) & 3,
          })
        }
      }
      return
    }
    const count = Math.max(1, Math.round(e.density * CANOPY_TILE_M * CANOPY_TILE_M))
    for (let k = 0; k < count; k++) {
      const h = hash4(seed, entryIndex, k, 0x9e3779b9)
      const scale = e.scaleMin + (e.scaleMax - e.scaleMin) * hashF32(hash2(h, 3))
      stamps.push({
        srcA: bake.topAlbedo,
        srcX: bake.topAux,
        srcPx: TOP_PX,
        px: hashF32(hash2(h, 1)) * CANOPY_TILE_M * pxPerM,
        pz: hashF32(hash2(h, 2)) * CANOPY_TILE_M * pxPerM,
        sizePx: 2 * bake.rootHalfW * scale * pxPerM,
        plantH: bake.plantH * scale,
        orient: hash2(h, 4) & 7,
      })
    }
  })

  // Pass 1: coverage (independent occluders) and the canopy top per texel.
  const open = new Float32Array(N * N).fill(1)
  const top = new Float32Array(N * N)
  for (const s of stamps) {
    visit(s, (idx, cov, hm) => {
      open[idx] = open[idx]! * (1 - cov)
      if (cov >= TOP_MIN_COV && hm > top[idx]!) top[idx] = hm
    })
  }

  // Pass 2: front-to-back weighted colour / normal / height.
  const acc = new Float32Array(N * N * 7)
  for (const s of stamps) {
    visit(s, (idx, cov, hm, r, g, b, nx, ny, nz) => {
      const w = cov * Math.exp(-(top[idx]! - hm) * DEPTH_FALLOFF)
      const o = idx * 7
      acc[o] = acc[o]! + r * w
      acc[o + 1] = acc[o + 1]! + g * w
      acc[o + 2] = acc[o + 2]! + b * w
      acc[o + 3] = acc[o + 3]! + nx * w
      acc[o + 4] = acc[o + 4]! + ny * w
      acc[o + 5] = acc[o + 5]! + nz * w
      acc[o + 6] = acc[o + 6]! + w
    })
  }

  // The perceived canopy surface is the top of the dense mass, not its mean: a
  // top-down capture of grass is dominated by the wide lower leaves, so the mean
  // of `top` sits about a third of the way up the plants. Take percentiles.
  const BINS = 128
  const hist = new Int32Array(BINS)
  let hCount = 0
  for (let i = 0; i < N * N; i++) {
    if (top[i]! <= 0.05) continue
    hist[Math.min(BINS - 1, Math.floor((top[i]! / heightMax) * BINS))]!++
    hCount++
  }
  const percentile = (frac: number): number => {
    let acc = 0
    for (let b = 0; b < BINS; b++) {
      acc += hist[b]!
      if (acc >= frac * hCount) return ((b + 0.5) / BINS) * heightMax
    }
    return heightMax
  }
  const canopyTop = hCount > 0 ? Math.max(0.15, percentile(0.97)) : heightMax
  const baseHeight = hCount > 0 ? Math.max(0.12, percentile(0.85)) : 0.5

  let auxZSum = 0
  const albedo = new Uint8Array(N * N * 4)
  const aux = new Uint8Array(N * N * 4)
  let cSum = 0
  let mr = 0
  let mg = 0
  let mb = 0
  for (let i = 0; i < N * N; i++) {
    const cov = Math.round((1 - open[i]!) * 255)
    albedo[i * 4 + 3] = cov
    const o = i * 7
    const w = acc[o + 6]!
    if (w > 0) {
      const r = acc[o]! / w
      const g = acc[o + 1]! / w
      const b = acc[o + 2]! / w
      albedo[i * 4] = clamp255(r)
      albedo[i * 4 + 1] = clamp255(g)
      albedo[i * 4 + 2] = clamp255(b)
      const [u, v] = octEncode(acc[o + 3]! / w, acc[o + 4]! / w, acc[o + 5]! / w)
      aux[i * 4] = u
      aux[i * 4 + 1] = v
      mr += r * cov
      mg += g * cov
      mb += b * cov
    } else {
      aux[i * 4] = 128
      aux[i * 4 + 1] = 128
    }
    const az = Math.min(1, top[i]! / canopyTop)
    aux[i * 4 + 2] = Math.round(az * 255)
    auxZSum += az
    cSum += cov
  }
  const meanLuma = cSum > 0 ? (mr + mg + mb) / 3 / cSum / 255 : 0.25

  return {
    albedo: albedo as Uint8Array<ArrayBuffer>,
    aux: aux as Uint8Array<ArrayBuffer>,
    heightMax: canopyTop,
    baseHeight,
    meanSway: swayDen > 0 ? swayNum / swayDen : 0.6,
    meanCoverage: cSum / (N * N) / 255,
    sideTopColor: sideTop,
    sideLowColor: sideLow,
    meanLuma,
    // Not the flower band's share of the silhouette: at grazing the heads are the
    // OUTERMOST layer of the canopy, so a horizontal ray hits them first. That is
    // also why a distant meadow reads as its flower colour and not as its leaf
    // colour (and why alpha-tested cards keep their heads and lose their stems).
    hueGain: FAR_FLOWER_MIX / Math.max(auxZSum / (N * N), 1e-3),
  }
}

type Visitor = (
  idx: number,
  cov: number,
  hm: number,
  r: number,
  g: number,
  b: number,
  nx: number,
  ny: number,
  nz: number,
) => void

/** Walk one plant copy's destination texels, torus-wrapped and box-filtered. */
function visit(s: Stamp, fn: Visitor): void {
  const N = CANOPY_TEX
  const S = s.srcPx
  const span = Math.max(1, Math.ceil(s.sizePx))
  const taps = Math.min(2, Math.max(1, Math.round(S / Math.max(s.sizePx, 1))))
  const srcA = s.srcA
  const srcX = s.srcX
  const x0 = Math.floor(s.px - s.sizePx * 0.5)
  const z0 = Math.floor(s.pz - s.sizePx * 0.5)

  for (let j = 0; j < span; j++) {
    const dz = (((z0 + j) % N) + N) % N
    for (let i = 0; i < span; i++) {
      const dx = (((x0 + i) % N) + N) % N
      let cAcc = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let hAcc = 0
      let n = 0
      for (let tj = 0; tj < taps; tj++) {
        const v = (j + (tj + 0.5) / taps) / s.sizePx
        if (v < 0 || v >= 1) continue
        for (let ti = 0; ti < taps; ti++) {
          const u = (i + (ti + 0.5) / taps) / s.sizePx
          if (u < 0 || u >= 1) continue
          const [su, sv] = orientUv(u, v, s.orient)
          const sx = Math.min(S - 1, Math.max(0, Math.floor(su * S)))
          const sy = Math.min(S - 1, Math.max(0, Math.floor(sv * S)))
          const s4 = (sy * S + sx) * 4
          const cov = srcA[s4 + 3]! / 255
          n++
          cAcc += cov
          if (cov <= 0) continue
          r += srcA[s4]! * cov
          g += srcA[s4 + 1]! * cov
          b += srcA[s4 + 2]! * cov
          const nrm = octDecodeOriented(
            (srcX[s4]! / 255) * 2 - 1,
            (srcX[s4 + 1]! / 255) * 2 - 1,
            s.orient,
          )
          nx += nrm[0] * cov
          ny += nrm[1] * cov
          nz += nrm[2] * cov
          hAcc += (srcX[s4 + 2]! / 255) * s.plantH * cov
        }
      }
      if (n === 0) continue
      const cov = cAcc / n
      if (cov <= 0.002) continue
      const w = Math.max(cAcc, 1e-6)
      fn(dz * N + dx, cov, hAcc / w, r / w, g / w, b / w, nx / w, ny / w, nz / w)
    }
  }
}

/**
 * Coverage-weighted mean colour of the whole-plant SIDE views (atlas column 0,
 * all 8 azimuths, full height). This is the colour a distant viewer averages
 * when looking along the canopy: flower heads, stems and shaded lower leaves in
 * the proportions they actually occupy the silhouette. Weighted toward the top
 * (SIDE_TOP_BIAS), because the lower band is what neighbouring plants hide.
 */
function meanSideColor(bake: CanopyBake, v0: number, v1: number): [number, number, number, number] {
  const tw = TILE_W >> 1
  const th = TILE_H >> 1
  const y0 = Math.floor(th * v0)
  const rows = Math.max(1, Math.floor(th * v1))
  let a = 0
  let r = 0
  let g = 0
  let b = 0
  for (let az = 0; az < 8; az++) {
    for (let y = y0; y < rows; y++) {
      // tile v = 0 is the plant top: bias the average toward what stays visible.
      const hw = 1 - SIDE_TOP_BIAS * (y / th)
      const row = (az * th + y) * ALB_W
      for (let x = 0; x < tw; x++) {
        const o = (row + x) * 4
        const w = bake.albedo[o + 3]! * hw
        if (w === 0) continue
        a += w
        r += bake.albedo[o]! * w
        g += bake.albedo[o + 1]! * w
        b += bake.albedo[o + 2]! * w
      }
    }
  }
  if (a === 0) return [0.3, 0.35, 0.2, 0]
  return [r / a / 255, g / a / 255, b / a / 255, a]
}

function clamp255(v: number): number {
  return Math.min(255, Math.max(0, Math.round(v)))
}

/** 8 free orientations of the source tile: 4 rotations x mirror. */
function orientUv(u: number, v: number, orient: number): [number, number] {
  let a = orient & 4 ? 1 - u : u
  let b = v
  switch (orient & 3) {
    case 1:
      ;[a, b] = [b, 1 - a]
      break
    case 2:
      ;[a, b] = [1 - a, 1 - b]
      break
    case 3:
      ;[a, b] = [1 - b, a]
      break
    default:
      break
  }
  return [a, b]
}

/** Decode the source normal and rotate it the way the imagery was rotated. */
function octDecodeOriented(u: number, v: number, orient: number): [number, number, number] {
  let x = u
  let z = v
  const y = 1 - Math.abs(u) - Math.abs(v)
  if (y < 0) {
    x = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1)
    z = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1)
  }
  const len = Math.hypot(x, y, z) || 1
  let nx = orient & 4 ? -x / len : x / len
  let nz = z / len
  switch (orient & 3) {
    case 1:
      ;[nx, nz] = [nz, -nx]
      break
    case 2:
      ;[nx, nz] = [-nx, -nz]
      break
    case 3:
      ;[nx, nz] = [-nz, nx]
      break
    default:
      break
  }
  return [nx, y / len, nz]
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
