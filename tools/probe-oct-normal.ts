/**
 * Settle the open GCMESH1 octahedral question.
 *
 * `GcMesh.normalAt()` derives Y from the two stored components;
 * 004-raycast-lut claims the data is z-derived and applies a y<->z swap,
 * reporting mean |cos| 0.75 vs 0.57 against face normals. If that holds, every
 * experiment using the shared decoder has subtly wrong normals.
 *
 * Test: decode every candidate convention, compare against the face normal
 * computed from raw dequantized POSITIONS (which involve no convention at all),
 * and report mean |cos| plus mean signed cos. Ground truth is the geometry.
 */
import { readFileSync } from 'node:fs'

const REPO = '/Users/sebastian/IdeaProjects/groundcover-experiments'
const MESH = `${REPO}/mesh/raw/calamagrostis-canescens/calamagrostis-canescens-mesh-v1.bin`
const HEADER_BYTES = 192

const buf = readFileSync(MESH)
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
const dv = new DataView(ab)

const magic = new TextDecoder().decode(new Uint8Array(ab, 0, 8))
if (!magic.startsWith('GCMESH1')) throw new Error(`bad magic ${JSON.stringify(magic)}`)

// Header layout per mesh/README.md: counts then bounds as f32.
// Offsets per mesh/README.md, not guessed.
const vertexCount = dv.getUint32(24, true)
const triangleCount = dv.getUint32(28, true)
const boundsMin: [number, number, number] = [dv.getFloat32(40, true), dv.getFloat32(44, true), dv.getFloat32(48, true)]
const boundsMax: [number, number, number] = [dv.getFloat32(52, true), dv.getFloat32(56, true), dv.getFloat32(60, true)]
const vertexOffset = Number(dv.getBigUint64(88, true))
const triangleOffset = Number(dv.getBigUint64(96, true))
console.log(`verts ${vertexCount.toLocaleString()}  tris ${triangleCount.toLocaleString()}`)
console.log(`bounds min ${boundsMin.map((v) => v.toFixed(3)).join(', ')}  max ${boundsMax.map((v) => v.toFixed(3)).join(', ')}`)

const vertices = new Uint16Array(ab, vertexOffset, vertexCount * 8)
const triangles = new Uint32Array(ab, triangleOffset, triangleCount * 4)

const pos = (i: number): [number, number, number] => {
  const b = i * 8
  return [
    boundsMin[0] + (vertices[b]! / 65535) * (boundsMax[0] - boundsMin[0]),
    boundsMin[1] + (vertices[b + 1]! / 65535) * (boundsMax[1] - boundsMin[1]),
    boundsMin[2] + (vertices[b + 2]! / 65535) * (boundsMax[2] - boundsMin[2]),
  ]
}

/** Octahedral decode with a configurable "derived" axis. */
function decode(i: number, derived: 'x' | 'y' | 'z'): [number, number, number] {
  const b = i * 8 + 6
  const u = (vertices[b]! / 65535) * 2 - 1
  const v = (vertices[b + 1]! / 65535) * 2 - 1
  let a = u
  let c = v
  const d = 1 - Math.abs(u) - Math.abs(v)
  if (d < 0) {
    a = (1 - Math.abs(v)) * Math.sign(u)
    c = (1 - Math.abs(u)) * Math.sign(v)
  }
  // (a, c) are the two stored axes; `d` is the reconstructed one.
  const out: [number, number, number] =
    derived === 'y' ? [a, d, c] : derived === 'z' ? [a, c, d] : [d, a, c]
  const len = Math.hypot(out[0], out[1], out[2]) || 1
  return [out[0] / len, out[1] / len, out[2] / len]
}

const sub = (p: number[], q: number[]): number[] => [p[0]! - q[0]!, p[1]! - q[1]!, p[2]! - q[2]!]
const cross = (p: number[], q: number[]): number[] => [
  p[1]! * q[2]! - p[2]! * q[1]!,
  p[2]! * q[0]! - p[0]! * q[2]!,
  p[0]! * q[1]! - p[1]! * q[0]!,
]

const CONVENTIONS = ['x', 'y', 'z'] as const
const stats = new Map<string, { absSum: number; sgnSum: number; n: number }>()
for (const c of CONVENTIONS) stats.set(c, { absSum: 0, sgnSum: 0, n: 0 })

// Sample widely rather than taking a prefix: a mesh is spatially sorted, so the
// first N triangles are one corner of one plant.
const SAMPLE = 200_000
const stride = Math.max(1, Math.floor(triangleCount / SAMPLE))
let degenerate = 0

for (let t = 0; t < triangleCount; t += stride) {
  const i0 = triangles[t * 4]!
  const i1 = triangles[t * 4 + 1]!
  const i2 = triangles[t * 4 + 2]!
  if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue
  const p0 = pos(i0)
  const p1 = pos(i1)
  const p2 = pos(i2)
  const fn = cross(sub(p1, p0), sub(p2, p0))
  const flen = Math.hypot(fn[0]!, fn[1]!, fn[2]!)
  if (flen < 1e-12) {
    degenerate++
    continue
  }
  const f = [fn[0]! / flen, fn[1]! / flen, fn[2]! / flen]
  for (const conv of CONVENTIONS) {
    const s = stats.get(conv)!
    for (const vi of [i0, i1, i2]) {
      const n = decode(vi, conv)
      const dot = n[0] * f[0]! + n[1] * f[1]! + n[2] * f[2]!
      s.absSum += Math.abs(dot)
      s.sgnSum += dot
      s.n++
    }
  }
}

console.log(`\nsampled ${(stats.get('y')!.n / 3).toLocaleString()} triangles (stride ${stride}), ${degenerate} degenerate\n`)
console.log('derived axis   mean |cos|   mean cos    verdict')
for (const conv of CONVENTIONS) {
  const s = stats.get(conv)!
  const abs = s.absSum / s.n
  const sgn = s.sgnSum / s.n
  const label = conv === 'y' ? '(gcmesh.ts)' : conv === 'z' ? '(004 claim)' : ''
  console.log(`  ${conv}          ${abs.toFixed(4)}      ${sgn.toFixed(4)}    ${label}`)
}
console.log(
  '\nA vertex normal is a smoothed average, so |cos| against a face normal is well under 1 even when correct.\nThe convention that is RIGHT should win clearly, and its mean SIGNED cos should be positive\n(normals point the same way as the winding, not the opposite).',
)
