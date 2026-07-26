import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Two bakes, because an upright plant and a ground carpet want completely
 * different storage — see NOTES.md ("budget split").
 *
 * UPRIGHT PLANTS — `views-v1-g5t256-<species>.bin`
 *   A GRID x GRID hemi-octahedral fan of orthographic captures over the whole
 *   bounding sphere, each a TILE x TILE cell in two atlases:
 *     albedo  rgba8unorm  — authored rgb + coverage
 *     geom    rgba16float — signed relief height h = dot(unit_pos, fwd),
 *                           oct-encoded local normal (flipped toward capture)
 *   25 views because a tall plant looks different from every azimuth.
 *
 * CARPET TILES — `carpet-v1-t512-<species>.bin`
 *   A periodic mat is one lumpy surface seen from above, so 24 of those 25
 *   views are grazing captures of a 7 cm cushion and store almost nothing.
 *   The carpet bake spends the whole budget on the ONE view that matters:
 *   a single zenith capture CROPPED to the species' periodic tile square
 *   ([0, tileM]^2 in the mesh frame — the tile origin is (0,0) for every
 *   current source mesh), at 512^2. That is 512 texels across 0.18 m
 *   (0.35 mm/texel) versus the 5x5 atlas's ~140 (1.29 mm/texel), for 4.2 MB
 *   instead of 18.75 MB.
 *   Two consequences fall out of the crop:
 *     - the mesh overflows its tile (0.24 m of geometry in a 0.18 m period),
 *       so the crop edges are the periodic continuation of each other and the
 *       texture can simply be sampled with address mode `repeat` — the relief
 *       march walking off one edge is correct, not an artifact;
 *     - one view has no neighbouring view cells to bleed into, so unlike the
 *       25-view atlas it CAN carry a mip chain. It is built on the CPU at load
 *       (coverage-weighted colour and height, normals averaged as unit vectors
 *       and re-encoded — oct is not mip-averageable, see CLAUDE.md).
 *
 * After readback, empty texels bordering covered ones are DILATED (records
 * copied, coverage kept 0) so runtime parallax reprojections that land just
 * outside a silhouette read a plausible height/colour instead of background.
 *
 * Both go through the standard harness bake flow (OPFS cache -> committed
 * mesh/baked/<exp>/<key>.bin -> in-browser bake), so a normal load never
 * touches the raw source meshes at all.
 *
 * Artifact (little-endian, 64-byte header, identical layout for both):
 *   u32 magic ('R8TP' views / 'R8CP' carpet), u32 version, u32 px, u32 grid
 *   f32 centre.xyz, f32 radius, f32 halfExt.xyz, f32 topH, zeros to byte 64
 *   u8  [px^2 * 4]   albedo rgba8   (a = coverage)
 *   u8  [px^2 * 8]   geom rgba16float (h, oct normal .yz, spare)
 */

export const GRID = 5
export const TILE = 256
export const ATLAS = GRID * TILE // 1280

/** Carpet tile capture resolution — 512 texels across one 0.18 m mesh tile. */
export const CARPET_PX = 512

const MAGIC_VIEWS = 0x50543852 // 'R8TP'
const MAGIC_CARPET = 0x50433852 // 'R8CP'
const VERSION = 1
/** v2 composites the eight periodic neighbours into the crop (see bakeCarpetTile). */
const CARPET_VERSION = 2
const HEADER_BYTES = 64

/** One baked representation, ready to upload: level 0 first. */
export interface SpeciesBake {
  /** true = single cropped zenith tile with mips (carpet), false = 5x5 view fan. */
  carpet: boolean
  px: number
  grid: number
  center: [number, number, number]
  radius: number
  halfExt: [number, number, number]
  topH: number
  albedoLevels: Uint8Array[]
  geomLevels: Uint8Array[]
  /** Carpet only: coverage-weighted mean / std-dev of the height channel. */
  hMean: number
  hSigma: number
}

type V3 = [number, number, number]
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

// --- f16, for reading/writing the geom atlas's raw bytes on the CPU ---------
const f32scratch = new Float32Array(1)
const u32scratch = new Uint32Array(f32scratch.buffer)

function f16ToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exp = (bits >> 10) & 0x1f
  const mant = bits & 0x3ff
  if (exp === 0) return sign * mant * 2 ** -24
  if (exp === 31) return mant === 0 ? sign * Infinity : NaN
  return sign * (1024 + mant) * 2 ** (exp - 25)
}

function f32ToF16(value: number): number {
  f32scratch[0] = value
  const x = u32scratch[0]!
  const sign = (x >>> 16) & 0x8000
  const exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant !== 0 ? 0x200 : 0)
  let e = exp - 127 + 15
  if (e >= 0x1f) return sign | 0x7c00
  if (e <= 0) {
    if (e < -10) return sign
    mant |= 0x800000
    const shift = 14 - e
    let m = mant >>> shift
    if ((mant >>> (shift - 1)) & 1) m += 1
    return sign | m
  }
  let m = mant >>> 13
  if ((mant >>> 12) & 1) {
    m += 1
    if (m === 0x400) {
      m = 0
      e += 1
      if (e >= 0x1f) return sign | 0x7c00
    }
  }
  return sign | (e << 10) | m
}

// --- octahedral normals — MUST match oct_encode/oct_decode in bake.wgsl ----
function octDecode(ex: number, ey: number): V3 {
  const y = 1 - Math.abs(ex) - Math.abs(ey)
  let x = ex
  let z = ey
  if (y < 0) {
    x = (1 - Math.abs(ey)) * Math.sign(ex)
    z = (1 - Math.abs(ex)) * Math.sign(ey)
  }
  return norm([x, y, z])
}

function octEncode(n: V3): [number, number] {
  const s = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2])
  let ex = n[0] / Math.max(s, 1e-6)
  let ey = n[2] / Math.max(s, 1e-6)
  if (n[1] < 0) {
    const ax = 1 - Math.abs(ey)
    const ay = 1 - Math.abs(ex)
    const sx = Math.sign(ex) || 1
    const sy = Math.sign(ey) || 1
    ex = ax * sx
    ey = ay * sy
  }
  return [ex, ey]
}

/** Hemi-octahedral decode — MUST match hemioct_decode() in relief.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const px = (ex + ey) * 0.5
  const pz = (ex - ey) * 0.5
  const y = 1 - Math.abs(px) - Math.abs(pz)
  return norm([px, y, pz])
}

/** Capture basis for grid node (i, j) — MUST match node_basis() in relief.wgsl. */
export function nodeBasis(i: number, j: number, grid: number): { f: V3; x: V3; y: V3 } {
  const ex = grid > 1 ? (i / (grid - 1)) * 2 - 1 : 0
  const ey = grid > 1 ? (j / (grid - 1)) * 2 - 1 : 0
  const f = hemioctDecode(ex, ey)
  const upRef: V3 = Math.abs(f[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0]
  const x = norm(cross(upRef, f))
  const y = norm(cross(f, x))
  return { f, x, y }
}

/**
 * Dilate covered texel records into empty neighbours, per tile (never across
 * tile seams), `iters` rings. Copies raw bytes — no f16 decode needed.
 * Coverage (albedo alpha) stays 0 so alpha-testing is unaffected.
 */
function dilate(albedo: Uint8Array, geom: Uint8Array, iters: number, px: number, grid: number): void {
  const tile = px / grid
  const covered = new Uint8Array(px * px)
  for (let i = 0; i < px * px; i++) covered[i] = albedo[i * 4 + 3]! > 0 ? 1 : 0
  const next = new Uint8Array(px * px)
  const geo16 = new Uint16Array(geom.buffer, geom.byteOffset, px * px * 4)
  for (let it = 0; it < iters; it++) {
    next.set(covered)
    for (let tj = 0; tj < grid; tj++) {
      for (let ti = 0; ti < grid; ti++) {
        const x0 = ti * tile
        const y0 = tj * tile
        for (let y = y0; y < y0 + tile; y++) {
          for (let x = x0; x < x0 + tile; x++) {
            const idx = y * px + x
            if (covered[idx]) continue
            // First covered 4-neighbour inside the tile wins.
            let src = -1
            if (x > x0 && covered[idx - 1]) src = idx - 1
            else if (x < x0 + tile - 1 && covered[idx + 1]) src = idx + 1
            else if (y > y0 && covered[idx - px]) src = idx - px
            else if (y < y0 + tile - 1 && covered[idx + px]) src = idx + px
            if (src < 0) continue
            albedo[idx * 4] = albedo[src * 4]!
            albedo[idx * 4 + 1] = albedo[src * 4 + 1]!
            albedo[idx * 4 + 2] = albedo[src * 4 + 2]!
            // alpha stays 0
            geo16[idx * 4] = geo16[src * 4]!
            geo16[idx * 4 + 1] = geo16[src * 4 + 1]!
            geo16[idx * 4 + 2] = geo16[src * 4 + 2]!
            geo16[idx * 4 + 3] = geo16[src * 4 + 3]!
            next[idx] = 1
          }
        }
      }
    }
    covered.set(next)
  }
}

/**
 * Mip chain for the single-view carpet tile, built on the CPU.
 *
 * Colour and height are COVERAGE-WEIGHTED (a half-empty texel must not drag
 * the canopy height toward the -1 background), so the stored rgb is already
 * normalised by coverage at every level and the shader must NOT divide by
 * alpha again. Normals are decoded to unit vectors, averaged, renormalised and
 * re-encoded: octahedral codes are not linear, and box-filtering them lands
 * near (0,0) which decodes to exactly straight up (CLAUDE.md).
 */
function buildCarpetMips(px: number, albedo0: Uint8Array, geom0: Uint8Array): {
  albedoLevels: Uint8Array[]
  geomLevels: Uint8Array[]
  hMean: number
  hSigma: number
} {
  const albedoLevels = [albedo0]
  const geomLevels = [geom0]
  let size = px
  let srcAlb = albedo0
  let srcGeo = new Uint16Array(geom0.buffer, geom0.byteOffset, size * size * 4)

  // Coverage-weighted mean and spread of the canopy height. The mean is where
  // the relief card plane goes (a plane at the MAXIMUM would make every pixel
  // march ~2 cm inward, and on a surface this steep a three-tap solve turns
  // that march into noise); the spread scales the crevice darkening.
  let wAcc = 0
  let hAcc = 0
  let h2Acc = 0
  for (let i = 0; i < size * size; i++) {
    const a = srcAlb[i * 4 + 3]! / 255
    if (a <= 0) continue
    const h = f16ToF32(srcGeo[i * 4]!)
    wAcc += a
    hAcc += h * a
    h2Acc += h * h * a
  }
  const hMean = wAcc > 0 ? hAcc / wAcc : 0
  const hSigma = wAcc > 0 ? Math.sqrt(Math.max(h2Acc / wAcc - hMean * hMean, 1e-8)) : 0.1
  while (size > 1) {
    const half = size >> 1
    const dstAlb = new Uint8Array(half * half * 4)
    const dstGeoBytes = new Uint8Array(half * half * 8)
    const dstGeo = new Uint16Array(dstGeoBytes.buffer)
    for (let y = 0; y < half; y++) {
      for (let x = 0; x < half; x++) {
        let wsum = 0
        let r = 0
        let g = 0
        let b = 0
        let rp = 0
        let gp = 0
        let bp = 0
        let h = 0
        let hp = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const si = (y * 2 + dy) * size + (x * 2 + dx)
            const a = srcAlb[si * 4 + 3]! / 255
            wsum += a
            rp += srcAlb[si * 4]!
            gp += srcAlb[si * 4 + 1]!
            bp += srcAlb[si * 4 + 2]!
            r += srcAlb[si * 4]! * a
            g += srcAlb[si * 4 + 1]! * a
            b += srcAlb[si * 4 + 2]! * a
            const hv = f16ToF32(srcGeo[si * 4]!)
            hp += hv
            h += hv * a
            const n = octDecode(f16ToF32(srcGeo[si * 4 + 1]!), f16ToF32(srcGeo[si * 4 + 2]!))
            nx += n[0] * a
            ny += n[1] * a
            nz += n[2] * a
          }
        }
        const di = y * half + x
        if (wsum > 1e-4) {
          dstAlb[di * 4] = Math.round(r / wsum)
          dstAlb[di * 4 + 1] = Math.round(g / wsum)
          dstAlb[di * 4 + 2] = Math.round(b / wsum)
          dstGeo[di * 4] = f32ToF16(h / wsum)
          const e = octEncode(norm([nx, ny, nz]))
          dstGeo[di * 4 + 1] = f32ToF16(e[0])
          dstGeo[di * 4 + 2] = f32ToF16(e[1])
        } else {
          // Fully empty: keep the dilated colour and the background height so a
          // reprojection landing here still reads something sane.
          dstAlb[di * 4] = Math.round(rp / 4)
          dstAlb[di * 4 + 1] = Math.round(gp / 4)
          dstAlb[di * 4 + 2] = Math.round(bp / 4)
          dstGeo[di * 4] = f32ToF16(hp / 4)
          dstGeo[di * 4 + 1] = f32ToF16(0)
          dstGeo[di * 4 + 2] = f32ToF16(0)
        }
        dstAlb[di * 4 + 3] = Math.round((wsum / 4) * 255)
        dstGeo[di * 4 + 3] = f32ToF16(1)
      }
    }
    albedoLevels.push(dstAlb)
    geomLevels.push(dstGeoBytes)
    srcAlb = dstAlb
    srcGeo = dstGeo
    size = half
  }
  return { albedoLevels, geomLevels, hMean, hSigma }
}

interface RawCapture {
  px: number
  grid: number
  center: V3
  radius: number
  halfExt: V3
  topH: number
  albedo: Uint8Array
  geom: Uint8Array
}

function packCapture(magic: number, version: number, v: RawCapture): ArrayBuffer {
  const albedoBytes = v.px * v.px * 4
  const geomBytes = v.px * v.px * 8
  const buf = new ArrayBuffer(HEADER_BYTES + albedoBytes + geomBytes)
  const u32 = new Uint32Array(buf, 0, 4)
  u32[0] = magic
  u32[1] = version
  u32[2] = v.px
  u32[3] = v.grid
  const f32 = new Float32Array(buf, 16, 8)
  f32.set([...v.center, v.radius, ...v.halfExt, v.topH])
  new Uint8Array(buf, HEADER_BYTES, albedoBytes).set(v.albedo)
  new Uint8Array(buf, HEADER_BYTES + albedoBytes, geomBytes).set(v.geom)
  return buf
}

/** Validate + view an artifact in place (null = not one of ours -> rebake). */
function unpackCapture(
  buf: ArrayBuffer,
  magic: number,
  version: number,
  px: number,
  grid: number,
): RawCapture | null {
  const albedoBytes = px * px * 4
  const geomBytes = px * px * 8
  if (buf.byteLength !== HEADER_BYTES + albedoBytes + geomBytes) return null
  const u32 = new Uint32Array(buf, 0, 4)
  if (u32[0] !== magic || u32[1] !== version || u32[2] !== px || u32[3] !== grid) return null
  const f = new Float32Array(buf, 16, 8)
  return {
    px,
    grid,
    center: [f[0]!, f[1]!, f[2]!],
    radius: f[3]!,
    halfExt: [f[4]!, f[5]!, f[6]!],
    topH: f[7]!,
    albedo: new Uint8Array(buf, HEADER_BYTES, albedoBytes),
    geom: new Uint8Array(buf, HEADER_BYTES + albedoBytes, geomBytes),
  }
}

/**
 * The one entry point the renderer uses: cached artifact if there is one,
 * otherwise load the source mesh and bake (then offer the result back to the
 * repo). The mesh load lives INSIDE the bake closure on purpose — a cache hit
 * must not fetch hundreds of MB of triangles it will never look at.
 *
 * `carpet` picks the storage split, not a rendering option: it is true exactly
 * when the active stand lays this species out as a mat (carpetDiv > 0).
 */
export async function loadSpeciesBake(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  carpet: boolean,
): Promise<SpeciesBake> {
  const px = carpet ? CARPET_PX : ATLAS
  const grid = carpet ? 1 : GRID
  const magic = carpet ? MAGIC_CARPET : MAGIC_VIEWS
  const version = carpet ? CARPET_VERSION : VERSION
  const key = carpet
    ? `carpet-v${CARPET_VERSION}-t${CARPET_PX}-${speciesId}`
    : `views-v${VERSION}-g${GRID}t${TILE}-${speciesId}`

  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const species = speciesById(speciesId)
    const mesh = await ctx.meshes.load(species.meshId)
    if (carpet) {
      const tileM = species.tileM
      if (tileM === undefined) throw new Error(`[${ctx.id}] carpet bake needs a periodic tileM for ${speciesId}`)
      return packCapture(magic, version, await bakeCarpetTile(ctx, mesh, tileM))
    }
    return packCapture(magic, version, await bakeViewFan(ctx, mesh))
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let raw = unpackCapture(buf, magic, version, px, grid)
  if (!raw) {
    console.warn(`[${ctx.id}] cached artifact for ${speciesId} (${key}) is invalid — rebaking`)
    buf = await runBake()
    raw = unpackCapture(buf, magic, version, px, grid)
    if (!raw) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }

  const levels = carpet
    ? buildCarpetMips(px, raw.albedo, raw.geom)
    : { albedoLevels: [raw.albedo], geomLevels: [raw.geom], hMean: 0, hSigma: 0 }
  return {
    carpet,
    px,
    grid,
    center: raw.center,
    radius: raw.radius,
    halfExt: raw.halfExt,
    topH: raw.topH,
    ...levels,
  }
}

interface CaptureTile {
  /** Atlas pixel origin. */
  x: number
  y: number
  size: number
  basis: { f: V3; x: V3; y: V3 }
  /**
   * Projection centre override. Shifting the centre by -o is exactly the same
   * as translating the mesh by +o, which is how the carpet bake composites the
   * periodic neighbours into one window (see bakeCarpetTile).
   */
  center?: V3
}

interface CaptureConfig {
  px: number
  grid: number
  /** Projection centre in mesh units; also the origin for the height channel. */
  center: V3
  /** Projection half-extent in mesh units; also the height normalisation. */
  radius: number
  tiles: CaptureTile[]
  dilateRings: number
}

/** The whole 5x5 hemi-octahedral fan over the bounding sphere. */
async function bakeViewFan(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<RawCapture> {
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const tiles: CaptureTile[] = []
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      tiles.push({ x: i * TILE, y: j * TILE, size: TILE, basis: nodeBasis(i, j, GRID) })
    }
  }
  return runCaptures(ctx, mesh, { px: ATLAS, grid: GRID, center, radius, tiles, dilateRings: 4 })
}

/**
 * ONE zenith capture, cropped to the species' periodic tile square. The tile
 * origin is (0, 0) in the mesh frame for every current source mesh, so the
 * crop window is centred on (tileM/2, ., tileM/2) with half-extent tileM/2 —
 * exactly the period, which is what makes the result tileable and samplable
 * with address mode `repeat`. `radius` also normalises the height channel, so
 * h lands in about +/-0.52 for these cushions (well inside the bake's [-1, 1]
 * depth window).
 */
async function bakeCarpetTile(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  tileM: number,
): Promise<RawCapture> {
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [tileM / 2, (bmin[1] + bmax[1]) / 2, tileM / 2]
  const radius = tileM / 2
  // The source mesh holds ONE tile's geometry, and that geometry overflows its
  // period (0.24 m of cushion in a 0.18 m tile). So the periodic image inside
  // the crop window is the tile PLUS the overflow of its eight neighbours, and
  // rendering a single copy leaves a band along every edge under-covered —
  // measured at 0.73-0.87 coverage against 0.95 in the interior, which shows up
  // as a faint dark grid over the whole mat in debug=coverage. Compositing all
  // nine copies (depth test resolves which is on top) makes the tile genuinely
  // periodic, and is what lets the runtime sample it with address mode repeat.
  const basis = nodeBasis(0, 0, 1)
  const tiles: CaptureTile[] = []
  for (const dz of [-tileM, 0, tileM]) {
    for (const dx of [-tileM, 0, tileM]) {
      tiles.push({
        x: 0,
        y: 0,
        size: CARPET_PX,
        basis,
        center: [center[0] - dx, center[1], center[2] - dz],
      })
    }
  }
  return runCaptures(ctx, mesh, { px: CARPET_PX, grid: 1, center, radius, tiles, dilateRings: 4 })
}

/** Render every requested capture on the GPU and read both atlases back. */
async function runCaptures(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  cfg: CaptureConfig,
): Promise<RawCapture> {
  const { device } = ctx
  const { px, center, radius } = cfg
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax

  // --- transient GPU resources (destroyed at the end, not VRAM-tracked) ---
  const vbuf = device.createBuffer({
    label: `${ctx.id}/bake-verts`,
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = device.createBuffer({
    label: `${ctx.id}/bake-idx`,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(ibuf, 0, indices)

  const albTex = device.createTexture({
    label: `${ctx.id}/bake-alb`,
    size: [px, px],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const geoTex = device.createTexture({
    label: `${ctx.id}/bake-geo`,
    size: [px, px],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const depthTex = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [px, px],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  // Per-tile uniform ring (96B used, 256B stride for dynamic offsets).
  const STRIDE = 256
  const nTiles = cfg.tiles.length
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * nTiles,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * nTiles)
  const fv = new Float32Array(scratch)
  cfg.tiles.forEach((tile, index) => {
    const { f, x, y } = tile.basis
    const c = tile.center ?? center
    const o = (index * STRIDE) / 4
    fv[o + 0] = c[0]; fv[o + 1] = c[1]; fv[o + 2] = c[2]; fv[o + 3] = 1 / radius
    fv[o + 4] = x[0]; fv[o + 5] = x[1]; fv[o + 6] = x[2]
    fv[o + 8] = y[0]; fv[o + 9] = y[1]; fv[o + 10] = y[2]
    fv[o + 12] = f[0]; fv[o + 13] = f[1]; fv[o + 14] = f[2]
    fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
    fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
  })
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 96 } }],
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
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/bake-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      // Empty background height = -1 (far side) so rays passing through gaps
      // keep travelling instead of stopping at the mid-plane.
      { view: geoTex.createView(), clearValue: { r: -1, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  cfg.tiles.forEach((tile, index) => {
    pass.setViewport(tile.x, tile.y, tile.size, tile.size, 0, 1)
    pass.setScissorRect(tile.x, tile.y, tile.size, tile.size)
    pass.setBindGroup(0, bg, [index * STRIDE])
    pass.drawIndexed(indices.length)
  })
  pass.end()

  const bprAlb = px * 4 // multiple of 256 for px >= 64
  const bprGeo = px * 8
  const rbAlb = device.createBuffer({
    label: `${ctx.id}/rb-alb`,
    size: bprAlb * px,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const rbGeo = device.createBuffer({
    label: `${ctx.id}/rb-geo`,
    size: bprGeo * px,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bprAlb, rowsPerImage: px }, [px, px])
  enc.copyTextureToBuffer({ texture: geoTex }, { buffer: rbGeo, bytesPerRow: bprGeo, rowsPerImage: px }, [px, px])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbGeo.mapAsync(GPUMapMode.READ)
  const albedo = new Uint8Array(bprAlb * px)
  const geom = new Uint8Array(bprGeo * px)
  albedo.set(new Uint8Array(rbAlb.getMappedRange()))
  geom.set(new Uint8Array(rbGeo.getMappedRange()))
  rbAlb.unmap()
  rbGeo.unmap()
  for (const r of [vbuf, ibuf, albTex, geoTex, depthTex, uni, rbAlb, rbGeo]) r.destroy()

  dilate(albedo, geom, cfg.dilateRings, px, cfg.grid)

  const halfExt: V3 = [(bmax[0] - bmin[0]) / 2, (bmax[1] - bmin[1]) / 2, (bmax[2] - bmin[2]) / 2]
  return {
    px,
    grid: cfg.grid,
    center,
    radius,
    halfExt,
    topH: mesh.header.topH || bmax[1],
    albedo,
    geom,
  }
}
