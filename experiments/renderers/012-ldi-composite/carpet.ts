import { ATLAS_W, NUM_DIRS, TILE, type LdiBaked } from './bake.ts'

/**
 * Carpet (mat) tiles, derived at LOAD TIME from the existing LDI bake — no
 * re-bake, no new artifact, no bake-version bump.
 *
 * A carpet species (`stand_table[i].carpet_div > 0`) is a PERIODIC community
 * tile, not a specimen: 0.18 m across, 0.07-0.09 m tall, laid out grid-snapped
 * with 90-degree-only rotations. For that shape the LDI's 4 side captures are
 * nearly worthless (a 7 cm cushion seen from 20 degrees of elevation is a
 * sliver) while the TOP capture is a 0.84 mm/texel heightfield of the cushion:
 * albedo + coverage + per-texel normal + per-texel capture depth, i.e. exactly
 * a displacement map. So the carpet path uses the top capture only, and this
 * module reshapes it into what a mat renderer actually wants:
 *
 * 1. **Cropped to the tile's own square** `[0, tileM]^2` in the mesh frame
 *    (the tile origin is (0,0) for every current source mesh, so the tile
 *    CENTRE is (tileM/2, tileM/2) — not the bbox centre the specimen path
 *    rotates about). One texture = exactly one grid step.
 * 2. **Periodic wrap-around composite.** The source mesh's geometry overflows
 *    its own period (0.24 m of moss inside a 0.18 m tile), and in an infinite
 *    mat that overflow reappears on the far side. So each output texel takes
 *    the FRONTMOST (min capture depth) of the 9 candidate samples at
 *    (x + dx*tileM, z + dz*tileM) — the overhangs a single-tile capture misses
 *    come back, and the result is exactly periodic, which means `repeat`
 *    addressing and box-filtered mips are both seamless.
 * 3. **A mip chain**, which the atlas cannot have (its 5x4 tiles would bleed
 *    into each other). Without it a 0.18 m tile at 40 m samples one texel out
 *    of 65k and the mat renders as shimmering speckle. Filtering conventions,
 *    which the shader must match:
 *      - `rgb` is COVERAGE-WEIGHTED at every level (sum(rgb*a)/sum(a)), i.e.
 *        already normalised — the shader must NOT divide by alpha again.
 *      - `a` is the plain mean, i.e. true fractional coverage. That is what
 *        lets a distant tile stay a solid occluder under a low alpha
 *        reference instead of dissolving.
 *      - normals are averaged as DECODED UNIT VECTORS and re-encoded
 *        (octahedral encoding is not mip-averageable — box-filtering it lands
 *        near (0,0), which decodes to exactly straight up).
 *      - depth is coverage-weighted, so relief flattens toward the mean with
 *        distance instead of being dragged by empty texels.
 */

/** Resampled tile resolution. The top capture is 256 px over ~0.215 m. */
export const CARPET_N = 256
/** Peel layers the carpet path draws: 0 = cushion surface, 1 = ~2 cm inside. */
export const CARPET_LAYERS = 2
const TOP_DIR = NUM_DIRS - 1

export interface CarpetMip {
  size: number
  /** rgba8: albedo.rgb + coverage. */
  albedo: Uint8Array<ArrayBuffer>
  /** rgba8: oct normal (rg) + 16-bit capture depth (b = hi, a = lo). */
  aux: Uint8Array<ArrayBuffer>
}

export interface CarpetTiles {
  /** [layer][mipLevel]. */
  layers: CarpetMip[][]
  /** Mesh-frame y (m) of stored depth 0, and the span stored depth 1 covers. */
  yTop: number
  ySpan: number
  /** Mesh-frame y (m) of each layer's mean-depth card plane. */
  planeY: number[]
}

type V3 = [number, number, number]

function octDecode(ex: number, ey: number): V3 {
  const y = 1 - Math.abs(ex) - Math.abs(ey)
  let x = ex
  let z = ey
  if (y < 0) {
    x = (1 - Math.abs(ey)) * (ex >= 0 ? 1 : -1)
    z = (1 - Math.abs(ex)) * (ey >= 0 ? 1 : -1)
  }
  const l = Math.hypot(x, y, z) || 1
  return [x / l, y / l, z / l]
}

function octEncode(n: V3): [number, number] {
  const s = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]) || 1
  let px = n[0] / s
  let py = n[2] / s
  if (n[1] < 0) {
    const t = px
    px = (1 - Math.abs(py)) * (t >= 0 ? 1 : -1)
    py = (1 - Math.abs(t)) * (py >= 0 ? 1 : -1)
  }
  return [px * 0.5 + 0.5, py * 0.5 + 0.5]
}

const clamp255 = (v: number): number => Math.max(0, Math.min(255, Math.round(v)))

/** Crop + periodic composite of one peel layer of the top capture. */
function buildLevel0(baked: LdiBaked, tileM: number, layer: number): CarpetMip {
  const dir = baked.meta.dirs[TOP_DIR]!
  const cx = baked.meta.center[0]
  const cz = baked.meta.center[2]
  const N = CARPET_N
  const albedo = new Uint8Array(N * N * 4)
  const aux = new Uint8Array(N * N * 4)
  const ax0 = TOP_DIR * TILE
  const ay0 = layer * TILE

  // Mesh xz -> atlas texel base index for this (dir, layer) tile, or -1 when
  // the point lies outside the ortho capture box.
  const srcAt = (mx: number, mz: number): number => {
    const rx = mx - cx
    const rz = mz - cz
    const cu = (rx * dir.right[0] + rz * dir.right[2]) / dir.halfW
    const cv = (rx * dir.up[0] + rz * dir.up[2]) / dir.halfH
    const u = 0.5 + 0.5 * cu
    const v = 0.5 - 0.5 * cv // NDC +y is texture row 0
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return -1
    const tx = Math.min(TILE - 1, Math.floor(u * TILE))
    const ty = Math.min(TILE - 1, Math.floor(v * TILE))
    return ((ay0 + ty) * ATLAS_W + ax0 + tx) * 4
  }

  for (let j = 0; j < N; j++) {
    const mz = ((j + 0.5) / N) * tileM
    for (let i = 0; i < N; i++) {
      const mx = ((i + 0.5) / N) * tileM
      let best = -1
      let bestD = Infinity
      let fallback = -1
      for (let dz = -1; dz <= 1; dz++) {
        for (let dx = -1; dx <= 1; dx++) {
          const s = srcAt(mx + dx * tileM, mz + dz * tileM)
          if (s < 0) continue
          if (baked.atlas0[s + 3]! === 0) {
            if (fallback < 0 || (dx === 0 && dz === 0)) fallback = s
            continue
          }
          const dep = baked.atlas1[s + 2]! * 256 + baked.atlas1[s + 3]!
          if (dep < bestD) {
            bestD = dep
            best = s
          }
        }
      }
      const o = (j * N + i) * 4
      const s = best >= 0 ? best : fallback
      if (s < 0) {
        // No capture data at all: empty, flat-up normal, far depth.
        aux[o] = 128
        aux[o + 1] = 128
        aux[o + 2] = 255
        aux[o + 3] = 255
        continue
      }
      albedo[o] = baked.atlas0[s]!
      albedo[o + 1] = baked.atlas0[s + 1]!
      albedo[o + 2] = baked.atlas0[s + 2]!
      albedo[o + 3] = best >= 0 ? 255 : 0
      // Flip the normal into the CAPTURE HEMISPHERE (+y, straight up) before
      // storing. The bake keeps raw two-sided mesh normals, so a leaf's front
      // and back faces store nearly opposite vectors — averaging those for a
      // mip cancels them out and normalize() then amplifies whatever sideways
      // residue is left. Measured: without this flip, whole tiles came out with
      // horizontal normals in `debug=normals`, in four groups matching the four
      // yaw quadrants, and the mat lit as a patchwork of bright and dark
      // squares. One hemisphere is also what the shading wants anyway: the mat
      // is seen from above, so the visible face is the up-facing one.
      const n = octDecode((baked.atlas1[s]! / 255) * 2 - 1, (baked.atlas1[s + 1]! / 255) * 2 - 1)
      const e = octEncode(n[1] < 0 ? [-n[0], -n[1], -n[2]] : n)
      aux[o] = clamp255(e[0] * 255)
      aux[o + 1] = clamp255(e[1] * 255)
      aux[o + 2] = baked.atlas1[s + 2]!
      aux[o + 3] = baked.atlas1[s + 3]!
    }
  }
  return { size: N, albedo, aux }
}

/** One 2x2 box step, in the conventions documented at the top of this file. */
function downsample(prev: CarpetMip): CarpetMip {
  const n = prev.size >> 1
  const albedo = new Uint8Array(n * n * 4)
  const aux = new Uint8Array(n * n * 4)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let rp = 0
      let gp = 0
      let bp = 0
      let depth = 0
      let depthPlain = 0
      const nrm: V3 = [0, 0, 0]
      const nrmPlain: V3 = [0, 0, 0]
      for (let dj = 0; dj < 2; dj++) {
        for (let di = 0; di < 2; di++) {
          const s = ((j * 2 + dj) * prev.size + i * 2 + di) * 4
          const a = prev.albedo[s + 3]! / 255
          aSum += a
          r += prev.albedo[s]! * a
          g += prev.albedo[s + 1]! * a
          b += prev.albedo[s + 2]! * a
          rp += prev.albedo[s]!
          gp += prev.albedo[s + 1]!
          bp += prev.albedo[s + 2]!
          const dep = prev.aux[s + 2]! * 256 + prev.aux[s + 3]!
          depth += dep * a
          depthPlain += dep
          const un = octDecode((prev.aux[s]! / 255) * 2 - 1, (prev.aux[s + 1]! / 255) * 2 - 1)
          for (let k = 0; k < 3; k++) {
            nrm[k]! += un[k]! * a
            nrmPlain[k]! += un[k]!
          }
        }
      }
      const o = (j * n + i) * 4
      const covered = aSum > 1e-6
      albedo[o] = clamp255(covered ? r / aSum : rp / 4)
      albedo[o + 1] = clamp255(covered ? g / aSum : gp / 4)
      albedo[o + 2] = clamp255(covered ? b / aSum : bp / 4)
      albedo[o + 3] = clamp255((aSum / 4) * 255)
      const src = covered ? nrm : nrmPlain
      const len = Math.hypot(src[0], src[1], src[2])
      const unit: V3 = len > 1e-5 ? [src[0] / len, src[1] / len, src[2] / len] : [0, 1, 0]
      const e = octEncode(unit)
      aux[o] = clamp255(e[0] * 255)
      aux[o + 1] = clamp255(e[1] * 255)
      const dep = Math.round(covered ? depth / aSum : depthPlain / 4)
      const hi = Math.min(255, dep >> 8)
      aux[o + 2] = hi
      aux[o + 3] = Math.max(0, Math.min(255, dep - hi * 256))
    }
  }
  return { size: n, albedo, aux }
}

export function buildCarpetTiles(baked: LdiBaked, tileM: number): CarpetTiles {
  const dir = baked.meta.dirs[TOP_DIR]!
  const layers: CarpetMip[][] = []
  const planeY: number[] = []
  for (let L = 0; L < CARPET_LAYERS; L++) {
    const chain: CarpetMip[] = [buildLevel0(baked, tileM, L)]
    while (chain[chain.length - 1]!.size > 1) chain.push(downsample(chain[chain.length - 1]!))
    layers.push(chain)
    const mean = dir.meanD[L]!
    planeY.push(baked.meta.center[1] + dir.sMax - (mean >= 0 ? mean : dir.sMax))
  }
  return { layers, yTop: baked.meta.center[1] + dir.sMax, ySpan: 2 * dir.sMax, planeY }
}
