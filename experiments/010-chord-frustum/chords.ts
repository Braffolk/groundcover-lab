import type { GcMesh } from '@harness'

/**
 * Chord-field constants + frustum proxy fitting + artifact packing.
 *
 * The baked representation is a 4D "chord light field" native to a convex
 * cone-frustum proxy: for every (entry point on the proxy surface, exit point
 * on the proxy surface) bin, we store what a ray traversing that chord of the
 * proxy INTERIOR would see (premultiplied albedo, coverage, first-hit fraction
 * along the chord, mesh normal). The proxy surface is charted by
 * (azimuth phi, meridian m) where m runs bottom-cap-center -> bottom rim ->
 * side wall -> top rim -> top-cap-center. Exit azimuth is stored RELATIVE to
 * entry azimuth so the inner tile is a pure chord-shape parameterization.
 *
 * These constants are mirrored as `const` in the WGSL shaders — keep in sync.
 */

// Entry bins (outer atlas grid).
export const A_E = 48 // entry azimuth bins
export const M_E = 32 // entry meridian bins
// Exit samples (inner tile). +1 gutter column duplicates dphi=0 at dphi=2pi
// so hardware bilinear wraps seamlessly.
export const A_X = 32 // exit relative-azimuth samples (tile width A_X+1)
export const M_X = 24 // exit meridian samples
export const ATLAS_W = A_E * (A_X + 1) // 1584
export const ATLAS_H = M_E * M_X // 768

// Bake-time ortho view set: full-sphere octahedral grid of directions.
// Rendered in row-batches so the transient G-buffer stays ~130MB.
export const VIEW_GRID = 20 // 400 views
export const VIEW_TILE = 256 // px per view (~5.5mm texels — fine fluff survives)
export const VIEW_BATCH_ROWS = 10 // view-grid rows per batch (2 batches)
export const VIEW_ATLAS_W = VIEW_GRID * VIEW_TILE // 5120
export const VIEW_ATLAS_H = VIEW_BATCH_ROWS * VIEW_TILE // 2560

export const MAGIC = 0x44524843 // 'CHRD'
export const HEADER_U32 = 32

export interface FrustumFit {
  /** Radius at the base (y=0 of the proxy frame). */
  r0: number
  /** Radius at the top (y=h). */
  r1: number
  /** Proxy height. */
  h: number
  /** Mesh-frame offset of the proxy origin: proxy = mesh - axis. */
  axisX: number
  axisY: number // = boundsMin.y
  axisZ: number
  /** Slant length of the side wall. */
  sideLen: number
  /** Meridian chart extents: bottom cap [-capB,0], side [0,1], top [1,1+capT]. */
  capB: number
  capT: number
  /** Bounding-sphere radius of the frustum about (0, h/2, 0). */
  rb: number
}

export interface ChordField {
  fit: FrustumFit
  /** rgba8 ATLAS_W x ATLAS_H: premultiplied albedo rgb + coverage. */
  surf: Uint8Array<ArrayBuffer>
  /** rgba8: r = first-hit fraction along chord, gb = oct normal, a unused. */
  geom: Uint8Array<ArrayBuffer>
}

/**
 * Fit a containing cone frustum to the mesh: per-height-band max horizontal
 * radius about the bounds-center axis, least-squares line through the band
 * maxima, then inflated so every band is contained.
 */
export function fitFrustum(mesh: GcMesh): FrustumFit {
  const { boundsMin, boundsMax, vertexCount } = mesh.header
  const axisX = (boundsMin[0] + boundsMax[0]) / 2
  const axisZ = (boundsMin[2] + boundsMax[2]) / 2
  const axisY = boundsMin[1]
  const h = Math.max(boundsMax[1] - axisY, 0.05)

  const BANDS = 24
  const bandMax = new Float64Array(BANDS)
  const verts = mesh.vertices
  const sx = (boundsMax[0] - boundsMin[0]) / 65535
  const sy = (boundsMax[1] - boundsMin[1]) / 65535
  const sz = (boundsMax[2] - boundsMin[2]) / 65535
  const stride = Math.max(1, Math.floor(vertexCount / 400_000))
  for (let i = 0; i < vertexCount; i += stride) {
    const b = i * 8
    const x = boundsMin[0] + verts[b]! * sx - axisX
    const y = boundsMin[1] + verts[b + 1]! * sy - axisY
    const z = boundsMin[2] + verts[b + 2]! * sz - axisZ
    const r = Math.hypot(x, z)
    const band = Math.min(BANDS - 1, Math.max(0, Math.floor((y / h) * BANDS)))
    if (r > bandMax[band]!) bandMax[band] = r
  }

  // Least squares r(yf) = c0 + c1*yf over band maxima (yf = band center).
  let sw = 0, swy = 0, swyy = 0, swr = 0, swyr = 0
  for (let i = 0; i < BANDS; i++) {
    if (bandMax[i]! <= 0) continue
    const yf = (i + 0.5) / BANDS
    sw += 1; swy += yf; swyy += yf * yf; swr += bandMax[i]!; swyr += yf * bandMax[i]!
  }
  let c0 = 0.2, c1 = 0
  const det = sw * swyy - swy * swy
  if (sw >= 2 && Math.abs(det) > 1e-9) {
    c1 = (sw * swyr - swy * swr) / det
    c0 = (swr - c1 * swy) / sw
  } else if (sw >= 1) {
    c0 = swr / sw
  }
  // Inflate so the line contains every band, then pad.
  let viol = 0
  for (let i = 0; i < BANDS; i++) {
    if (bandMax[i]! <= 0) continue
    const yf = (i + 0.5) / BANDS
    viol = Math.max(viol, bandMax[i]! - (c0 + c1 * yf))
  }
  let r0 = (c0 + viol) * 1.03
  let r1 = (c0 + c1 + viol) * 1.03
  r0 = Math.max(r0, 0.02)
  r1 = Math.max(r1, 0.02)

  const sideLen = Math.hypot(h, r1 - r0)
  const capB = r0 / sideLen
  const capT = r1 / sideLen
  const rb = Math.hypot(Math.max(r0, r1), h / 2) * 1.02
  return { r0, r1, h, axisX, axisY, axisZ, sideLen, capB, capT, rb }
}

export function packChordField(f: ChordField): ArrayBuffer {
  const blob = ATLAS_W * ATLAS_H * 4
  const buf = new ArrayBuffer(HEADER_U32 * 4 + blob * 2)
  const u = new Uint32Array(buf, 0, HEADER_U32)
  const fl = new Float32Array(buf, 0, HEADER_U32)
  u[0] = MAGIC
  u[1] = 1 // version
  u[2] = A_E; u[3] = M_E; u[4] = A_X; u[5] = M_X
  u[6] = ATLAS_W; u[7] = ATLAS_H
  const t = f.fit
  fl[8] = t.r0; fl[9] = t.r1; fl[10] = t.h
  fl[11] = t.axisX; fl[12] = t.axisY; fl[13] = t.axisZ
  fl[14] = t.sideLen; fl[15] = t.capB; fl[16] = t.capT; fl[17] = t.rb
  new Uint8Array(buf, HEADER_U32 * 4, blob).set(f.surf)
  new Uint8Array(buf, HEADER_U32 * 4 + blob, blob).set(f.geom)
  return buf
}

export function unpackChordField(buf: ArrayBuffer): ChordField | null {
  if (buf.byteLength < HEADER_U32 * 4) return null
  const u = new Uint32Array(buf, 0, HEADER_U32)
  if (u[0] !== MAGIC || u[1] !== 1) return null
  if (u[2] !== A_E || u[3] !== M_E || u[4] !== A_X || u[5] !== M_X) return null
  const blob = ATLAS_W * ATLAS_H * 4
  if (buf.byteLength !== HEADER_U32 * 4 + blob * 2) return null
  const fl = new Float32Array(buf, 0, HEADER_U32)
  return {
    fit: {
      r0: fl[8]!, r1: fl[9]!, h: fl[10]!,
      axisX: fl[11]!, axisY: fl[12]!, axisZ: fl[13]!,
      sideLen: fl[14]!, capB: fl[15]!, capT: fl[16]!, rb: fl[17]!,
    },
    surf: new Uint8Array(buf, HEADER_U32 * 4, blob),
    geom: new Uint8Array(buf, HEADER_U32 * 4 + blob, blob),
  }
}

// --- full-sphere octahedral direction grid (mirrored in gather.wgsl) --------

type V3 = [number, number, number]

export function octDecodeDir(ex: number, ey: number): V3 {
  let x = ex
  let z = ey
  const y = 1 - Math.abs(ex) - Math.abs(ey)
  if (y < 0) {
    x = (1 - Math.abs(ey)) * Math.sign(ex)
    z = (1 - Math.abs(ex)) * Math.sign(ey)
  }
  const l = Math.hypot(x, y, z) || 1
  return [x / l, y / l, z / l]
}

export function viewDir(i: number, j: number): V3 {
  const ex = ((i + 0.5) / VIEW_GRID) * 2 - 1
  const ey = ((j + 0.5) / VIEW_GRID) * 2 - 1
  return octDecodeDir(ex, ey)
}

/** View basis — MUST mirror view_basis() in gather.wgsl. */
export function viewBasis(fwd: V3): { right: V3; up: V3 } {
  const ref: V3 = Math.abs(fwd[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0]
  const rx = ref[1] * fwd[2] - ref[2] * fwd[1]
  const ry = ref[2] * fwd[0] - ref[0] * fwd[2]
  const rz = ref[0] * fwd[1] - ref[1] * fwd[0]
  const rl = Math.hypot(rx, ry, rz) || 1
  const right: V3 = [rx / rl, ry / rl, rz / rl]
  const up: V3 = [
    fwd[1] * right[2] - fwd[2] * right[1],
    fwd[2] * right[0] - fwd[0] * right[2],
    fwd[0] * right[1] - fwd[1] * right[0],
  ]
  return { right, up }
}
