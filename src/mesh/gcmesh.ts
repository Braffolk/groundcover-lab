/**
 * GCMESH1 parser — the canonical source-mesh format, spec in mesh/README.md.
 * Vertex record (16B): u16 quantized XYZ (against header bounds), u16 RGB
 * (UNORM, no colorspace conversion), u16x2 octahedral normal. Triangle record
 * (16B): three u32 indices + reserved u32.
 */

export interface GcMeshHeader {
  vertexCount: number
  triangleCount: number
  boundsMin: [number, number, number]
  boundsMax: [number, number, number]
  tileOrigin: [number, number]
  /** Periodic tile size (m) — repeat copies by integer multiples. */
  tileSize: [number, number]
  topH: number
  profileId: number
  flags: number
}

const MAGIC = 'GCMESH1\0'
const HEADER_BYTES = 192

export class GcMesh {
  private packedIndices: Uint32Array<ArrayBuffer> | null = null

  constructor(
    readonly header: GcMeshHeader,
    /** Interleaved quantized vertex records — 8 u16 per vertex. */
    readonly vertices: Uint16Array<ArrayBuffer>,
    /** Raw triangle records — 4 u32 per triangle (3 indices + pad). */
    readonly triangles: Uint32Array<ArrayBuffer>,
  ) {}

  positionAt(i: number, out: [number, number, number] = [0, 0, 0]): [number, number, number] {
    const { boundsMin, boundsMax } = this.header
    const base = i * 8
    for (let c = 0; c < 3; c++) {
      out[c] = boundsMin[c]! + (this.vertices[base + c]! / 65535) * (boundsMax[c]! - boundsMin[c]!)
    }
    return out
  }

  colorAt(i: number, out: [number, number, number] = [0, 0, 0]): [number, number, number] {
    const base = i * 8 + 3
    for (let c = 0; c < 3; c++) out[c] = this.vertices[base + c]! / 65535
    return out
  }

  /**
   * Octahedral decode for GCMESH1 source normals. See `decodeGcMeshNormal`.
   */
  normalAt(i: number, out: [number, number, number] = [0, 0, 0]): [number, number, number] {
    const base = i * 8 + 6
    return decodeGcMeshNormal(this.vertices[base]!, this.vertices[base + 1]!, out)
  }

  /** Tightly-packed 3-per-triangle index array (pad u32 stripped), cached. */
  indices(): Uint32Array<ArrayBuffer> {
    if (!this.packedIndices) {
      const n = this.header.triangleCount
      const out = new Uint32Array(n * 3)
      for (let t = 0; t < n; t++) {
        out[t * 3] = this.triangles[t * 4]!
        out[t * 3 + 1] = this.triangles[t * 4 + 1]!
        out[t * 3 + 2] = this.triangles[t * 4 + 2]!
      }
      this.packedIndices = out
    }
    return this.packedIndices
  }
}

/**
 * Decode one GCMESH1 octahedral normal from its two raw u16 components — the TS
 * twin of `gcmesh_normal_decode_u16()` in src/wgsl/gcmesh.wgsl, and the single
 * place the convention lives on this side. Allocation-free with `out`, so a bake
 * looping over two million vertices can call it instead of copying the
 * arithmetic. Copies are exactly how the wrong convention survived: **40** of
 * them (36 WGSL `oct_decode*` in experiment bakes, one in the harness's own mesh
 * inspector, and 4 in TS bakes), all y-derived, and not one would have been
 * reached by fixing `normalAt` alone. 36 experiment ids read the source normal
 * through this function or its WGSL twin now, and nothing else may.
 *
 * **The stored pair is (x, y) and the reconstructed component is z, positive.**
 * Both halves were measured, and the sign took two attempts:
 *
 * - AXIS. Decode three ways and compare against face normals computed from raw
 *   dequantized positions (no convention involved) over 217,113 triangles: mean
 *   |cos| 0.4225 x-derived, 0.5354 y-derived (what this used to do), 0.7871
 *   z-derived. On the other two species z-derived wins 0.85 and 0.94. A vertex
 *   normal is a smoothed average, so |cos| stays under 1 even when right;
 *   0.79 against 0.54 is not ambiguous.
 * - SIGN. Do NOT take it from the winding: the mean SIGNED cos against those
 *   face normals is -0.73 on calamagrostis but +0.71 on elymus and +0.89 on poa,
 *   so the three meshes are not wound alike, and the signed-volume test that
 *   would settle winding is meaningless here because all three are OPEN shells
 *   (their signed volume moves when you translate the mesh). Settle it on
 *   geometry instead: fire a 4 mm ray from a vertex along +n and -n, skipping
 *   the triangles that touch it. A surface has material on one side only, so the
 *   outward normal is the one with free space ahead. +n hits geometry 65/60/58%
 *   of the time against -n's 77/93/69% (calamagrostis/elymus/poa), and has the
 *   greater free space on every mesh. +n is outward — do not negate.
 *
 * Re-run `tools/probe-oct-normal.ts` rather than trusting this comment.
 */
export function decodeGcMeshNormal(
  nu16: number,
  nv16: number,
  out: [number, number, number] = [0, 0, 0],
): [number, number, number] {
  const u = (nu16 / 65535) * 2 - 1
  const v = (nv16 / 65535) * 2 - 1
  let x = u
  let y = v
  const z = 1 - Math.abs(u) - Math.abs(v)
  if (z < 0) {
    x = (1 - Math.abs(v)) * Math.sign(u)
    y = (1 - Math.abs(u)) * Math.sign(v)
  }
  const len = Math.hypot(x, y, z) || 1
  out[0] = x / len
  out[1] = y / len
  out[2] = z / len
  return out
}

export function parseGcMesh(buffer: ArrayBuffer): GcMesh {
  if (buffer.byteLength < HEADER_BYTES) throw new Error('GCMESH1: file shorter than header')
  const view = new DataView(buffer)
  const magic = new TextDecoder('latin1').decode(new Uint8Array(buffer, 0, 8))
  if (magic !== MAGIC) throw new Error(`GCMESH1: bad magic ${JSON.stringify(magic)}`)
  const u32 = (o: number): number => view.getUint32(o, true)
  const f32 = (o: number): number => view.getFloat32(o, true)
  const u64 = (o: number): number => Number(view.getBigUint64(o, true))
  if (u32(8) !== 1) throw new Error(`GCMESH1: unsupported version ${u32(8)}`)
  if (u32(12) !== HEADER_BYTES) throw new Error('GCMESH1: unexpected header size')
  if (u32(16) !== 16 || u32(20) !== 16) throw new Error('GCMESH1: unexpected record sizes')

  const vertexCount = u32(24)
  const triangleCount = u32(28)
  const vertexOffset = u64(88)
  const triangleOffset = u64(96)
  const totalBytes = u64(104)
  if (totalBytes !== buffer.byteLength) {
    throw new Error(`GCMESH1: header says ${totalBytes} bytes, file is ${buffer.byteLength}`)
  }

  const header: GcMeshHeader = {
    vertexCount,
    triangleCount,
    boundsMin: [f32(40), f32(44), f32(48)],
    boundsMax: [f32(52), f32(56), f32(60)],
    tileOrigin: [f32(64), f32(68)],
    tileSize: [f32(72), f32(76)],
    topH: f32(80),
    profileId: u32(32),
    flags: u32(36),
  }
  return new GcMesh(
    header,
    new Uint16Array(buffer, vertexOffset, vertexCount * 8),
    new Uint32Array(buffer, triangleOffset, triangleCount * 4),
  )
}
