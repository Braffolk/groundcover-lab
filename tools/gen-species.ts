/**
 * Generate simple procedural species meshes (grass-blade, moss-patch) as
 * GCMESH1 binaries in mesh/raw/<id>/, so multi-species overlap is testable
 * without authored assets. Usage: npm run species:gen
 */
import { createHash } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')

interface Vert {
  p: [number, number, number]
  c: [number, number, number]
  n: [number, number, number]
}

interface Mesh {
  verts: Vert[]
  tris: [number, number, number][]
  tile: [number, number]
}

// Deterministic PRNG (mulberry32) — generation must be reproducible.
function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function grassBlade(): Mesh {
  const r = rng(1234)
  const verts: Vert[] = []
  const tris: [number, number, number][] = []
  const TILE = 0.5
  for (let b = 0; b < 90; b++) {
    const bx = r() * TILE
    const bz = r() * TILE
    const height = 0.25 + r() * 0.45
    const lean = (r() - 0.5) * 0.35
    const yaw = r() * Math.PI * 2
    const dir: [number, number] = [Math.cos(yaw), Math.sin(yaw)]
    const width = 0.006 + r() * 0.006
    const green = 0.75 + r() * 0.5
    const segments = 4
    const base = verts.length
    for (let s = 0; s <= segments; s++) {
      const t = s / segments
      const bend = lean * t * t
      const w = width * (1 - t * 0.9)
      const cx = bx + dir[0] * bend
      const cz = bz + dir[1] * bend
      const y = height * t
      // Blade cross vector (perpendicular to growth dir in xz).
      const px = -dir[1] * w
      const pz = dir[0] * w
      const shade = 0.55 + 0.45 * t
      const color: [number, number, number] = [0.13 * green * shade, 0.3 * green * shade, 0.08 * shade]
      const normal: [number, number, number] = [dir[0] * 0.35, 0.9, dir[1] * 0.35]
      verts.push({ p: [cx - px, y, cz - pz], c: color, n: normal })
      verts.push({ p: [cx + px, y, cz + pz], c: color, n: normal })
    }
    for (let s = 0; s < segments; s++) {
      const i = base + s * 2
      tris.push([i, i + 1, i + 2], [i + 1, i + 3, i + 2])
    }
  }
  return { verts, tris, tile: [0.5, 0.5] }
}

function mossPatch(): Mesh {
  const r = rng(4321)
  const verts: Vert[] = []
  const tris: [number, number, number][] = []
  const N = 24
  const TILE = 0.5
  // Bumpy low heightfield with color mottling.
  const h: number[][] = []
  for (let z = 0; z <= N; z++) {
    h.push([])
    for (let x = 0; x <= N; x++) {
      const bump = Math.sin((x / N) * Math.PI * 6 + 1.3) * Math.sin((z / N) * Math.PI * 5.2)
      h[z]!.push(0.012 + 0.02 * (bump * 0.5 + 0.5) + r() * 0.008)
    }
  }
  for (let z = 0; z <= N; z++) {
    for (let x = 0; x <= N; x++) {
      const y = h[z]![x]!
      const dx = ((h[z]![Math.min(x + 1, N)] ?? y) - (h[z]![Math.max(x - 1, 0)] ?? y)) * N
      const dz = ((h[Math.min(z + 1, N)]![x] ?? y) - (h[Math.max(z - 1, 0)]![x] ?? y)) * N
      const g = 0.7 + r() * 0.6
      verts.push({
        p: [(x / N) * TILE, y, (z / N) * TILE],
        c: [0.1 * g, 0.17 * g, 0.06 * g],
        n: [-dx * 0.5, 1, -dz * 0.5],
      })
    }
  }
  for (let z = 0; z < N; z++) {
    for (let x = 0; x < N; x++) {
      const i = z * (N + 1) + x
      tris.push([i, i + 1, i + N + 1], [i + 1, i + N + 2, i + N + 1])
    }
  }
  return { verts, tris, tile: [TILE, TILE] }
}

function octEncode(n: [number, number, number]): [number, number] {
  const len = Math.hypot(...n) || 1
  const [x, y, z] = [n[0] / len, n[1] / len, n[2] / len]
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  let px = x / s
  let pz = z / s
  if (y < 0) {
    const [ox, oz] = [px, pz]
    px = (1 - Math.abs(oz)) * Math.sign(ox === 0 ? 1 : ox)
    pz = (1 - Math.abs(ox)) * Math.sign(oz === 0 ? 1 : oz)
  }
  const q = (v: number): number => Math.max(0, Math.min(65535, Math.round((v * 0.5 + 0.5) * 65535)))
  return [q(px), q(pz)]
}

function writeGcMesh(id: string, mesh: Mesh): void {
  const { verts, tris, tile } = mesh
  const min: number[] = [Infinity, Infinity, Infinity]
  const max: number[] = [-Infinity, -Infinity, -Infinity]
  for (const v of verts) {
    for (let c = 0; c < 3; c++) {
      min[c] = Math.min(min[c]!, v.p[c]!)
      max[c] = Math.max(max[c]!, v.p[c]!)
    }
  }

  const HEADER = 192
  const vertexBytes = verts.length * 16
  const triangleOffset = HEADER + vertexBytes
  const total = triangleOffset + tris.length * 16
  const buf = Buffer.alloc(total)

  const payload = buf.subarray(HEADER)
  const vertView = new DataView(buf.buffer, buf.byteOffset + HEADER, vertexBytes)
  verts.forEach((v, i) => {
    const o = i * 16
    for (let c = 0; c < 3; c++) {
      const range = max[c]! - min[c]! || 1
      vertView.setUint16(o + c * 2, Math.round(((v.p[c]! - min[c]!) / range) * 65535), true)
    }
    for (let c = 0; c < 3; c++) {
      vertView.setUint16(o + 6 + c * 2, Math.max(0, Math.min(65535, Math.round(v.c[c]! * 65535))), true)
    }
    const [ou, ov] = octEncode(v.n)
    vertView.setUint16(o + 12, ou, true)
    vertView.setUint16(o + 14, ov, true)
  })
  const triView = new DataView(buf.buffer, buf.byteOffset + triangleOffset, tris.length * 16)
  tris.forEach((t, i) => {
    triView.setUint32(i * 16, t[0], true)
    triView.setUint32(i * 16 + 4, t[1], true)
    triView.setUint32(i * 16 + 8, t[2], true)
  })

  buf.write('GCMESH1\0', 0, 'latin1')
  buf.writeUInt32LE(1, 8)
  buf.writeUInt32LE(HEADER, 12)
  buf.writeUInt32LE(16, 16)
  buf.writeUInt32LE(16, 20)
  buf.writeUInt32LE(verts.length, 24)
  buf.writeUInt32LE(tris.length, 28)
  buf.writeUInt32LE(0, 32) // profile id: procedural
  buf.writeUInt32LE(7, 36) // flags: RGB + oct normals + indexed
  for (let c = 0; c < 3; c++) buf.writeFloatLE(min[c]!, 40 + c * 4)
  for (let c = 0; c < 3; c++) buf.writeFloatLE(max[c]!, 52 + c * 4)
  buf.writeFloatLE(0, 64)
  buf.writeFloatLE(0, 68)
  buf.writeFloatLE(tile[0], 72)
  buf.writeFloatLE(tile[1], 76)
  buf.writeFloatLE(max[1]!, 80) // profile top H
  buf.writeUInt32LE(65535, 84)
  buf.writeBigUInt64LE(BigInt(HEADER), 88)
  buf.writeBigUInt64LE(BigInt(triangleOffset), 96)
  buf.writeBigUInt64LE(BigInt(total), 104)
  buf.writeBigUInt64LE(0n, 112) // no source GCRP
  createHash('sha256').update(payload).digest().copy(buf, 152)

  const dir = path.join(root, 'mesh', 'raw', id)
  mkdirSync(dir, { recursive: true })
  const binary = `${id}-mesh-v1.bin`
  writeFileSync(path.join(dir, binary), buf)
  const manifest = {
    schema: 'procedural-species-mesh/v1',
    binary,
    outputSha256: createHash('sha256').update(buf).digest('hex'),
    payloadSha256: createHash('sha256').update(payload).digest('hex'),
    outputBytes: total,
    vertexCount: verts.length,
    triangleCount: tris.length,
    boundsMin: min,
    boundsMax: max,
    tileOrigin: [0, 0],
    tileSize: tile,
    topH: max[1],
    offsets: { vertexOffset: HEADER, triangleOffset },
    records: { vertexBytes: 16, triangleBytes: 16 },
  }
  writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2))
  console.log(`wrote mesh/raw/${id}/ — ${verts.length} verts, ${tris.length} tris, ${(total / 1024).toFixed(0)}KB`)
}

writeGcMesh('grass-blade', grassBlade())
writeGcMesh('moss-patch', mossPatch())
