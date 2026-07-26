/**
 * Baked ray-answer table layout — shared by the bake and the runtime.
 *
 * The table answers ONE question per texel: "a ray that crosses the canopy-top
 * plane of a periodic tile at (u,v), travelling in baked direction (band,
 * azimuth) — what does it hit first?" The answer is stored as
 *
 *   R = q      normalized vertical drop from the top plane to the hit (0..1)
 *   G = cov    fraction of the texel's sub-rays that hit canopy at all
 *   B,A = oct  octahedral world-frame surface normal at the hit (tile frame)
 *
 * Everything else the shader needs (hit position, along-ray distance, height
 * fraction, AO, albedo) is closed-form from q + the ray, so ONE texture fetch
 * per band tap resolves a pixel.
 *
 * Direction quantization is elevation-adaptive: near-vertical rays barely care
 * about azimuth (at exactly vertical they don't care at all -> 1 layer), while
 * grazing rays need fine azimuth steps. Layers are therefore allocated per
 * band, not as a uniform azimuth x elevation grid.
 */

export const TABLE_VERSION = 3
/** Table spatial resolution per direction layer (texels per tile side). */
export const TABLE_U = 192
/** Band centre zenith angles (degrees from straight down). */
export const BAND_ZENITH_DEG = [0, 26, 48, 66, 81] as const
/** Azimuth bins per band. */
export const BAND_AZIM = [1, 12, 20, 30, 48] as const
export const BAND_COUNT = BAND_ZENITH_DEG.length
export const LAYERS = BAND_AZIM.reduce((a, b) => a + b, 0)
export const BAND_BASE = BAND_AZIM.reduce<number[]>((acc, n, i) => {
  acc.push(i === 0 ? 0 : acc[i - 1]! + BAND_AZIM[i - 1]!)
  return acc
}, [])
/** Height bins of the baked albedo curve; 2 colour clusters per bin. */
export const PALETTE_BINS = 32

export const MIP_LEVELS = ((): number => {
  let n = 1
  let s = TABLE_U
  while (s > 1) {
    s = s >> 1
    n++
  }
  return n
})()

/** Tile period (m) and lattice rotation (rad) per species — deliberately
 * incommensurate so the three periodic canopies never repeat in step. */
export const SPECIES_TILE: Record<string, { period: number; psi: number }> = {
  'calamagrostis-canescens': { period: 2.6, psi: 0.0 },
  'elymus-repens': { period: 2.15, psi: 0.62 },
  'poa-pratensis': { period: 1.9, psi: 1.31 },
}

export function tileFor(speciesId: string): { period: number; psi: number } {
  return SPECIES_TILE[speciesId] ?? { period: 2.4, psi: 0.35 }
}

const MAGIC = 0x31564352 // 'RCV1'
const PALETTE_OFFSET = 256
const PALETTE_FLOATS = PALETTE_BINS * 2 * 4
export const DATA_OFFSET = PALETTE_OFFSET + PALETTE_FLOATS * 4 // 1280

export interface CanopyTable {
  u: number
  layers: number
  /** Tile period (m). */
  period: number
  /** Canopy top plane height above the local ground (m). */
  height: number
  /** Tile lattice rotation (rad). */
  psi: number
  /** Plants of the stand entry inside one tile. */
  plants: number
  /** Mean q per band — the parallax-correction reference depth. */
  meanQ: Float32Array
  /** PALETTE_BINS * 2 * vec4 — height-binned albedo, two colour clusters. */
  palette: Float32Array
  /** TABLE_U^2 * layers * 4 bytes, layer-major. */
  data: Uint8Array<ArrayBuffer>
}

export function packTable(t: CanopyTable): ArrayBuffer {
  const buf = new ArrayBuffer(DATA_OFFSET + t.data.byteLength)
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, 64)
  u32[0] = MAGIC
  u32[1] = TABLE_VERSION
  u32[2] = t.u
  u32[3] = t.layers
  u32[4] = BAND_COUNT
  u32[5] = t.plants
  f32[6] = t.period
  f32[7] = t.height
  f32[8] = t.psi
  f32[9] = 0
  for (let b = 0; b < BAND_COUNT; b++) {
    f32[10 + b] = t.meanQ[b] ?? 0.25
    u32[20 + b] = BAND_AZIM[b]!
    u32[28 + b] = BAND_BASE[b]!
  }
  new Float32Array(buf, PALETTE_OFFSET, PALETTE_FLOATS).set(t.palette)
  new Uint8Array(buf, DATA_OFFSET, t.data.byteLength).set(t.data)
  return buf
}

/** Strict validation — the dev server answers missing files with index.html. */
export function unpackTable(buf: ArrayBuffer): CanopyTable | null {
  if (buf.byteLength < DATA_OFFSET + 4) return null
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, 64)
  if (u32[0] !== MAGIC || u32[1] !== TABLE_VERSION) return null
  const u = u32[2]!
  const layers = u32[3]!
  if (u !== TABLE_U || layers !== LAYERS) return null
  const bytes = u * u * layers * 4
  if (buf.byteLength !== DATA_OFFSET + bytes) return null
  const meanQ = new Float32Array(BAND_COUNT)
  for (let b = 0; b < BAND_COUNT; b++) meanQ[b] = f32[10 + b]!
  return {
    u,
    layers,
    period: f32[6]!,
    height: f32[7]!,
    psi: f32[8]!,
    plants: u32[5]!,
    meanQ,
    palette: new Float32Array(buf, PALETTE_OFFSET, PALETTE_FLOATS),
    data: new Uint8Array(buf, DATA_OFFSET, bytes) as Uint8Array<ArrayBuffer>,
  }
}
