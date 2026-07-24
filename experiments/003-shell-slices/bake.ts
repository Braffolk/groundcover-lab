import type { GcMesh } from '@harness'

/**
 * Bake a GCMESH1 source mesh into a small 3D voxel volume ("SSV1"):
 *   tex0 rgba8: premultiplied albedo (rgb) + coverage (a)
 *   tex1 rgba8: octahedral mean normal (rg) + precomputed sky visibility (b)
 * Sky visibility is the precomputed-raycast loophole: per-voxel top-down
 * transmittance, baked once, used as canopy self-occlusion at runtime.
 *
 * The volume is centered on the community tile (periodic species) or the
 * specimen footprint (poa), so a scatter point stamps one tile as "a plant".
 */

export const BAKE_VERSION = 3
export const VOXEL_BUDGET = 1_300_000
export const TOP_RES = 128
const MAGIC = 0x31565353 // 'SSV1'

export interface VolumeHeader {
  nx: number
  ny: number
  nz: number
  /** Local-space bounds (mesh space minus tile center). */
  lmin: [number, number, number]
  lsize: [number, number, number]
  /** Mean voxel edge length (local units). */
  voxel: number
  /** Tiling period (m, unscaled) used by the far canopy shell. */
  tile: [number, number]
}

export interface Volume {
  header: VolumeHeader
  tex0: Uint8Array
  tex1: Uint8Array
}

// ---------------------------------------------------------------------------
// Voxelization
// ---------------------------------------------------------------------------

function chooseGrid(ext: [number, number, number]): [number, number, number] {
  const vol = ext[0] * ext[1] * ext[2]
  let v = Math.cbrt(vol / VOXEL_BUDGET)
  for (let iter = 0; iter < 64; iter++) {
    const nx = Math.max(8, Math.min(224, Math.round(ext[0] / v)))
    const ny = Math.max(8, Math.min(224, Math.round(ext[1] / v)))
    const nz = Math.max(8, Math.min(224, Math.round(ext[2] / v)))
    if (nx * ny * nz <= VOXEL_BUDGET * 1.1) return [nx, ny, nz]
    v *= 1.03
  }
  return [64, 64, 64]
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

/** Standard octahedral encode — exact inverse family of GcMesh.normalAt. */
function octEncode(x: number, y: number, z: number): [number, number] {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1
  let px = x / l1
  let pz = z / l1
  if (y < 0) {
    const ox = px
    px = (1 - Math.abs(pz)) * (ox >= 0 ? 1 : -1)
    pz = (1 - Math.abs(ox)) * (pz >= 0 ? 1 : -1)
  }
  return [px * 0.5 + 0.5, pz * 0.5 + 0.5]
}

export function bakeVolume(mesh: GcMesh, tileSize: [number, number] | null): Volume {
  const h = mesh.header
  const bmin = h.boundsMin
  const bmax = h.boundsMax
  const ext: [number, number, number] = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]]

  // Center: periodic tile center for community meshes, footprint mid for specimens.
  const cx = tileSize ? h.tileOrigin[0] + tileSize[0] / 2 : (bmin[0] + bmax[0]) / 2
  const cz = tileSize ? h.tileOrigin[1] + tileSize[1] / 2 : (bmin[2] + bmax[2]) / 2
  const lmin: [number, number, number] = [bmin[0] - cx, bmin[1], bmin[2] - cz]
  const tile: [number, number] = tileSize ?? [
    Math.min(ext[0], ext[2]) * 0.9,
    Math.min(ext[0], ext[2]) * 0.9,
  ]

  const [nx, ny, nz] = chooseGrid(ext)
  const n = nx * ny * nz
  const voxel = Math.cbrt((ext[0] / nx) * (ext[1] / ny) * (ext[2] / nz))
  const vmin = Math.min(ext[0] / nx, ext[1] / ny, ext[2] / nz)

  const wSum = new Float32Array(n)
  const wr = new Float32Array(n)
  const wg = new Float32Array(n)
  const wb = new Float32Array(n)
  const nxA = new Float32Array(n)
  const nyA = new Float32Array(n)
  const nzA = new Float32Array(n)

  const verts = mesh.vertices
  const tris = mesh.triangles
  const triCount = h.triangleCount
  const sx = nx / ext[0]
  const sy = ny / ext[1]
  const sz = nz / ext[2]
  const q = 1 / 65535
  const e0 = ext[0] * q
  const e1 = ext[1] * q
  const e2 = ext[2] * q
  const invSub = 1 / (vmin * 0.7)
  const nrm: [number, number, number] = [0, 0, 0]

  for (let t = 0; t < triCount; t++) {
    const b = t * 4
    const i0 = tris[b]! * 8
    const i1 = tris[b + 1]! * 8
    const i2 = tris[b + 2]! * 8

    const p0x = lmin[0] + verts[i0]! * e0
    const p0y = lmin[1] + verts[i0 + 1]! * e1
    const p0z = lmin[2] + verts[i0 + 2]! * e2
    const p1x = lmin[0] + verts[i1]! * e0
    const p1y = lmin[1] + verts[i1 + 1]! * e1
    const p1z = lmin[2] + verts[i1 + 2]! * e2
    const p2x = lmin[0] + verts[i2]! * e0
    const p2y = lmin[1] + verts[i2 + 1]! * e1
    const p2z = lmin[2] + verts[i2 + 2]! * e2

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

    const e01 = ax * ax + ay * ay + az * az
    const e02 = bx * bx + by * by + bz * bz
    const dx = p2x - p1x
    const dy = p2y - p1y
    const dz = p2z - p1z
    const e12 = dx * dx + dy * dy + dz * dz
    const maxEdge = Math.sqrt(Math.max(e01, e02, e12))
    const S = Math.max(1, Math.min(6, Math.ceil(maxEdge * invSub)))

    // Vertex colors (mean is fine at blade scale) and mean normal.
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

    const w = area / (S * S)
    const inv = 1 / S
    for (let a = 0; a < S; a++) {
      for (let bb = 0; bb < S - a; bb++) {
        // Centroids of the "up" sub-triangles plus mirrored "down" ones.
        for (let k = 0; k < (a + bb < S - 1 ? 2 : 1); k++) {
          const u = (a + (k === 0 ? 1 / 3 : 2 / 3)) * inv
          const v = (bb + (k === 0 ? 1 / 3 : 2 / 3)) * inv
          const w0 = 1 - u - v
          const px = w0 * p0x + u * p1x + v * p2x
          const py = w0 * p0y + u * p1y + v * p2y
          const pz = w0 * p0z + u * p1z + v * p2z
          let ix = ((px - lmin[0]) * sx) | 0
          let iy = ((py - lmin[1]) * sy) | 0
          let iz = ((pz - lmin[2]) * sz) | 0
          if (ix < 0) ix = 0
          else if (ix >= nx) ix = nx - 1
          if (iy < 0) iy = 0
          else if (iy >= ny) iy = ny - 1
          if (iz < 0) iz = 0
          else if (iz >= nz) iz = nz - 1
          const idx = ix + nx * (iy + ny * iz)
          wSum[idx] = wSum[idx]! + w
          wr[idx] = wr[idx]! + cr * w
          wg[idx] = wg[idx]! + cg * w
          wb[idx] = wb[idx]! + cb * w
          nxA[idx] = nxA[idx]! + nvx * w
          nyA[idx] = nyA[idx]! + nvy * w
          nzA[idx] = nzA[idx]! + nvz * w
        }
      }
    }
  }

  // Finalize: coverage from area density, premultiplied albedo, oct normals.
  const tex0 = new Uint8Array(n * 4)
  const tex1 = new Uint8Array(n * 4)
  const crossSec = voxel * voxel
  const cov = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const a = wSum[i]!
    if (a === 0) continue
    const c = 1 - Math.exp((-1.2 * a) / crossSec)
    cov[i] = c
    const invA = 1 / a
    tex0[i * 4] = Math.min(255, wr[i]! * invA * c * 255 + 0.5)
    tex0[i * 4 + 1] = Math.min(255, wg[i]! * invA * c * 255 + 0.5)
    tex0[i * 4 + 2] = Math.min(255, wb[i]! * invA * c * 255 + 0.5)
    tex0[i * 4 + 3] = Math.min(255, c * 255 + 0.5)
    let mx = nxA[i]!
    let my = nyA[i]!
    let mz = nzA[i]!
    const len = Math.hypot(mx, my, mz)
    if (len < 1e-6) {
      mx = 0
      my = 1
      mz = 0
    } else {
      mx /= len
      my /= len
      mz /= len
    }
    const [ou, ov] = octEncode(mx, my, mz)
    tex1[i * 4] = Math.min(255, ou * 255 + 0.5)
    tex1[i * 4 + 1] = Math.min(255, ov * 255 + 0.5)
  }

  // Sky visibility: per-column top-down transmittance (precomputed raycast).
  for (let iz = 0; iz < nz; iz++) {
    for (let ix = 0; ix < nx; ix++) {
      let trans = 1
      for (let iy = ny - 1; iy >= 0; iy--) {
        const idx = ix + nx * (iy + ny * iz)
        tex1[idx * 4 + 2] = Math.min(255, trans * 255 + 0.5)
        trans *= 1 - 0.6 * cov[idx]!
      }
    }
  }

  return {
    header: { nx, ny, nz, lmin, lsize: ext, voxel, tile },
    tex0,
    tex1,
  }
}

// ---------------------------------------------------------------------------
// Serialization (the baked-artifact byte format is entirely ours)
// ---------------------------------------------------------------------------

export function serializeVolume(v: Volume): ArrayBuffer {
  const n = v.header.nx * v.header.ny * v.header.nz
  const buf = new ArrayBuffer(64 + n * 8)
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, 16)
  u32[0] = MAGIC
  u32[1] = BAKE_VERSION
  u32[2] = v.header.nx
  u32[3] = v.header.ny
  u32[4] = v.header.nz
  f32[5] = v.header.lmin[0]
  f32[6] = v.header.lmin[1]
  f32[7] = v.header.lmin[2]
  f32[8] = v.header.lsize[0]
  f32[9] = v.header.lsize[1]
  f32[10] = v.header.lsize[2]
  f32[11] = v.header.voxel
  f32[12] = v.header.tile[0]
  f32[13] = v.header.tile[1]
  new Uint8Array(buf, 64, n * 4).set(v.tex0)
  new Uint8Array(buf, 64 + n * 4, n * 4).set(v.tex1)
  return buf
}

/** True iff buf looks like a valid SSV1 artifact of the current version. */
export function isVolume(buf: ArrayBuffer | null): buf is ArrayBuffer {
  if (!buf || buf.byteLength < 64) return false
  const u32 = new Uint32Array(buf, 0, 16)
  if (u32[0] !== MAGIC || u32[1] !== BAKE_VERSION) return false
  const n = u32[2]! * u32[3]! * u32[4]!
  return buf.byteLength === 64 + n * 8
}

export function parseVolume(buf: ArrayBuffer): Volume {
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, 16)
  if (u32[0] !== MAGIC) throw new Error('SSV1: bad magic')
  if (u32[1] !== BAKE_VERSION) throw new Error(`SSV1: version ${u32[1]} != ${BAKE_VERSION}`)
  const nx = u32[2]!
  const ny = u32[3]!
  const nz = u32[4]!
  const n = nx * ny * nz
  return {
    header: {
      nx,
      ny,
      nz,
      lmin: [f32[5]!, f32[6]!, f32[7]!],
      lsize: [f32[8]!, f32[9]!, f32[10]!],
      voxel: f32[11]!,
      tile: [f32[12]!, f32[13]!],
    },
    tex0: new Uint8Array(buf, 64, n * 4),
    tex1: new Uint8Array(buf, 64 + n * 4, n * 4),
  }
}

// ---------------------------------------------------------------------------
// Load-time derived data: 3D mips (slab integration) + far-shell top view
// ---------------------------------------------------------------------------

export interface MipLevel {
  nx: number
  ny: number
  nz: number
  data: Uint8Array
}

/** Average-of-8 mips. Premultiplied albedo makes plain averaging correct. */
export function buildMips3D(nx: number, ny: number, nz: number, base: Uint8Array, levels: number): MipLevel[] {
  const out: MipLevel[] = []
  let sw = nx
  let sh = ny
  let sd = nz
  let src = base
  for (let level = 1; level < levels; level++) {
    const dw = Math.max(1, sw >> 1)
    const dh = Math.max(1, sh >> 1)
    const dd = Math.max(1, sd >> 1)
    const dst = new Uint8Array(dw * dh * dd * 4)
    for (let z = 0; z < dd; z++) {
      for (let y = 0; y < dh; y++) {
        for (let x = 0; x < dw; x++) {
          let r = 0
          let g = 0
          let b = 0
          let a = 0
          let cnt = 0
          for (let oz = 0; oz < 2; oz++) {
            const szz = Math.min(z * 2 + oz, sd - 1)
            for (let oy = 0; oy < 2; oy++) {
              const syy = Math.min(y * 2 + oy, sh - 1)
              for (let ox = 0; ox < 2; ox++) {
                const sxx = Math.min(x * 2 + ox, sw - 1)
                const si = (sxx + sw * (syy + sh * szz)) * 4
                r += src[si]!
                g += src[si + 1]!
                b += src[si + 2]!
                a += src[si + 3]!
                cnt++
              }
            }
          }
          const di = (x + dw * (y + dh * z)) * 4
          dst[di] = r / cnt
          dst[di + 1] = g / cnt
          dst[di + 2] = b / cnt
          dst[di + 3] = a / cnt
        }
      }
    }
    out.push({ nx: dw, ny: dh, nz: dd, data: dst })
    sw = dw
    sh = dh
    sd = dd
    src = dst
  }
  return out
}

/**
 * Top-down orthographic composite over one tiling period, with periodic
 * wrap of overhanging foliage — the texture the far canopy shell tiles.
 * Output premultiplied rgba8, TOP_RES².
 */
export function buildTopView(vol: Volume): Uint8Array {
  const { nx, ny, nz, lmin, lsize, tile } = vol.header
  const t0 = vol.tex0
  const out = new Uint8Array(TOP_RES * TOP_RES * 4)
  const sx = nx / lsize[0]
  const sz = nz / lsize[2]
  for (let j = 0; j < TOP_RES; j++) {
    const lz = -tile[1] / 2 + ((j + 0.5) / TOP_RES) * tile[1]
    for (let i = 0; i < TOP_RES; i++) {
      const lx = -tile[0] / 2 + ((i + 0.5) / TOP_RES) * tile[0]
      let r = 0
      let g = 0
      let b = 0
      let trans = 1
      for (let oz = -1; oz <= 1; oz++) {
        const pz = lz + oz * tile[1]
        const iz = Math.floor((pz - lmin[2]) * sz)
        if (iz < 0 || iz >= nz) continue
        for (let ox = -1; ox <= 1; ox++) {
          const px = lx + ox * tile[0]
          const ix = Math.floor((px - lmin[0]) * sx)
          if (ix < 0 || ix >= nx) continue
          for (let iy = ny - 1; iy >= 0; iy--) {
            const idx = (ix + nx * (iy + ny * iz)) * 4
            const a = t0[idx + 3]! / 255
            if (a === 0) continue
            r += (trans * t0[idx]!) / 255
            g += (trans * t0[idx + 1]!) / 255
            b += (trans * t0[idx + 2]!) / 255
            trans *= 1 - a
            if (trans < 0.02) break
          }
        }
      }
      const o = (i + j * TOP_RES) * 4
      out[o] = Math.min(255, r * 255 + 0.5)
      out[o + 1] = Math.min(255, g * 255 + 0.5)
      out[o + 2] = Math.min(255, b * 255 + 0.5)
      out[o + 3] = Math.min(255, (1 - trans) * 255 + 0.5)
    }
  }
  return out
}

/** Average-of-4 2D mips for the top view (premultiplied). */
export function buildMips2D(res: number, base: Uint8Array, levels: number): { res: number; data: Uint8Array }[] {
  const out: { res: number; data: Uint8Array }[] = []
  let sr = res
  let src = base
  for (let level = 1; level < levels; level++) {
    const dr = Math.max(1, sr >> 1)
    const dst = new Uint8Array(dr * dr * 4)
    for (let y = 0; y < dr; y++) {
      for (let x = 0; x < dr; x++) {
        for (let c = 0; c < 4; c++) {
          const s0 = src[(Math.min(x * 2, sr - 1) + Math.min(y * 2, sr - 1) * sr) * 4 + c]!
          const s1 = src[(Math.min(x * 2 + 1, sr - 1) + Math.min(y * 2, sr - 1) * sr) * 4 + c]!
          const s2 = src[(Math.min(x * 2, sr - 1) + Math.min(y * 2 + 1, sr - 1) * sr) * 4 + c]!
          const s3 = src[(Math.min(x * 2 + 1, sr - 1) + Math.min(y * 2 + 1, sr - 1) * sr) * 4 + c]!
          dst[(x + y * dr) * 4 + c] = (s0 + s1 + s2 + s3) / 4
        }
      }
    }
    out.push({ res: dr, data: dst })
    sr = dr
    src = dst
  }
  return out
}
