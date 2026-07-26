/**
 * The GCMESH1 octahedral convention, measured. SETTLED 2026-07-26: the stored
 * pair is (x, y) and the reconstructed component is z, POSITIVE — no negation.
 * `src/mesh/gcmesh.ts` (decodeGcMeshNormal) and `src/wgsl/gcmesh.wgsl` do that;
 * this file is why, and it is cheap to re-run before believing either.
 *
 *   npx tsx tools/probe-oct-normal.ts [species]   (default calamagrostis)
 *
 * TEST 1 — AXIS. Decode every candidate convention and compare against the face
 * normal computed from raw dequantized POSITIONS, which involve no convention at
 * all. Ground truth is the geometry. z-derived wins 0.79 to 0.54 on mean |cos|
 * for calamagrostis, and by more on the other two species (0.85, 0.94).
 *
 * TEST 2 — SIGN, from occlusion. The tempting shortcut is test 1's mean SIGNED
 * cos: -0.73 says the un-negated normal is ANTIPARALLEL to the face normals, so
 * negate. DO NOT USE IT. It only means something if (p1-p0)x(p2-p0) points out of
 * the surface, and (a) the same figure is +0.71 on elymus and +0.89 on poa, so
 * the three meshes are not wound alike, and (b) the signed-volume test that would
 * settle winding is invalid here — all three are OPEN shells, which the test
 * checks by translating the mesh and watching the "volume" move. An earlier fix
 * negated on this evidence and inverted every source normal in the repo.
 *
 * What settles it is geometry: a surface has material on ONE side. Fire a 4 mm
 * ray from a sampled vertex along +n and along -n, skipping the triangles that
 * touch that vertex, and the outward normal is the one with free space ahead. +n
 * hits geometry 65% / 60% / 58% of the time against -n's 77% / 93% / 69%
 * (calamagrostis / elymus / poa), and has the greater free space on all three.
 * +n is outward on every mesh, so there is no negation.
 *
 * TEST 3 — the weak test, kept because it is what misled the first attempt. The
 * topmost vertex of each 4 mm xz column sits on an upward-facing surface, so its
 * n.y should be positive. On calamagrostis that favours negating (66% up) — but
 * the mean |n.y| is only 0.08, i.e. these are near-horizontal normals and the
 * split is nearly a coin toss, while on elymus, where the same test is
 * unambiguous (mean |n.y| 0.72), it says the opposite: 93.5% of them face up
 * WITHOUT negation. A test that weak on one mesh cannot outvote test 2 on three.
 */
import { readFileSync } from 'node:fs'

const REPO = '/Users/sebastian/IdeaProjects/groundcover-experiments'
const SPECIES = process.argv[2] ?? 'calamagrostis-canescens'
const MESH = `${REPO}/mesh/raw/${SPECIES}/${SPECIES}-mesh-v1.bin`
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
  const label = conv === 'y' ? '(the old, wrong decode)' : conv === 'z' ? '(the decode in use)' : ''
  console.log(`  ${conv}          ${abs.toFixed(4)}      ${sgn.toFixed(4)}    ${label}`)
}
console.log(
  '\nA vertex normal is a smoothed average, so |cos| against a face normal is well under 1 even when correct.\nThe convention that is RIGHT should win clearly on |cos|. Its mean SIGNED cos is NOT the sign test —\nsee the header, and test 2 below.',
)

// ---------------------------------------------------------------------------
// TEST 2 — the sign, from short-ray occlusion
//
// A surface has material on one side. Fire a short ray along +n and -n from a
// vertex, skipping the triangles incident to it: the outward normal is the one
// with free space ahead. No winding, no closedness, no "up" assumed.
// ---------------------------------------------------------------------------

const REACH = 0.004
const RAY_SAMPLES = 4000
const RAY_EPS = 2e-5
{
  // Triangle grid keyed on centroid; a REACH-long ray stays inside a 3x3x3 block.
  const cell = REACH
  const nx = Math.max(1, Math.ceil((boundsMax[0] - boundsMin[0]) / cell))
  const ny = Math.max(1, Math.ceil((boundsMax[1] - boundsMin[1]) / cell))
  const nz = Math.max(1, Math.ceil((boundsMax[2] - boundsMin[2]) / cell))
  const cellOfTri = new Int32Array(triangleCount)
  for (let t = 0; t < triangleCount; t++) {
    const a = pos(triangles[t * 4]!)
    const b = pos(triangles[t * 4 + 1]!)
    const c = pos(triangles[t * 4 + 2]!)
    const cx = Math.min(nx - 1, Math.max(0, Math.floor(((a[0] + b[0] + c[0]) / 3 - boundsMin[0]) / cell)))
    const cy = Math.min(ny - 1, Math.max(0, Math.floor(((a[1] + b[1] + c[1]) / 3 - boundsMin[1]) / cell)))
    const cz = Math.min(nz - 1, Math.max(0, Math.floor(((a[2] + b[2] + c[2]) / 3 - boundsMin[2]) / cell)))
    cellOfTri[t] = (cz * ny + cy) * nx + cx
  }
  const starts = new Int32Array(nx * ny * nz + 1)
  for (let t = 0; t < triangleCount; t++) starts[cellOfTri[t]! + 1]!++
  for (let c = 0; c < starts.length - 1; c++) starts[c + 1]! += starts[c]!
  const bucket = new Int32Array(triangleCount)
  const fill = starts.slice(0, starts.length - 1)
  for (let t = 0; t < triangleCount; t++) bucket[fill[cellOfTri[t]!]!++] = t

  /** Moller-Trumbore, two-sided; nearest positive t or Infinity. */
  const rayTri = (o: [number, number, number], d: [number, number, number], t: number): number => {
    const a = pos(triangles[t * 4]!)
    const b = pos(triangles[t * 4 + 1]!)
    const c = pos(triangles[t * 4 + 2]!)
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
    const p = [d[1] * e2[2]! - d[2] * e2[1]!, d[2] * e2[0]! - d[0] * e2[2]!, d[0] * e2[1]! - d[1] * e2[0]!]
    const det = e1[0]! * p[0]! + e1[1]! * p[1]! + e1[2]! * p[2]!
    if (Math.abs(det) < 1e-18) return Infinity
    const inv = 1 / det
    const s = [o[0] - a[0], o[1] - a[1], o[2] - a[2]]
    const u = (s[0]! * p[0]! + s[1]! * p[1]! + s[2]! * p[2]!) * inv
    if (u < -1e-6 || u > 1 + 1e-6) return Infinity
    const q = [s[1]! * e1[2]! - s[2]! * e1[1]!, s[2]! * e1[0]! - s[0]! * e1[2]!, s[0]! * e1[1]! - s[1]! * e1[0]!]
    const v = (d[0] * q[0]! + d[1] * q[1]! + d[2] * q[2]!) * inv
    if (v < -1e-6 || u + v > 1 + 1e-6) return Infinity
    const tt = (e2[0]! * q[0]! + e2[1]! * q[1]! + e2[2]! * q[2]!) * inv
    return tt > RAY_EPS ? tt : Infinity
  }

  const step = Math.max(1, Math.floor(vertexCount / RAY_SAMPLES))
  let sampled = 0
  let hitPlus = 0
  let hitMinus = 0
  let freerPlus = 0
  let freerMinus = 0
  for (let i = 0; i < vertexCount; i += step) {
    const o = pos(i)
    const n = decode(i, 'z')
    const cx = Math.min(nx - 1, Math.max(0, Math.floor((o[0] - boundsMin[0]) / cell)))
    const cy = Math.min(ny - 1, Math.max(0, Math.floor((o[1] - boundsMin[1]) / cell)))
    const cz = Math.min(nz - 1, Math.max(0, Math.floor((o[2] - boundsMin[2]) / cell)))
    let tp = Infinity
    let tm = Infinity
    for (let dz = -1; dz <= 1; dz++) {
      if (cz + dz < 0 || cz + dz >= nz) continue
      for (let dy = -1; dy <= 1; dy++) {
        if (cy + dy < 0 || cy + dy >= ny) continue
        for (let dx = -1; dx <= 1; dx++) {
          if (cx + dx < 0 || cx + dx >= nx) continue
          const c = ((cz + dz) * ny + (cy + dy)) * nx + (cx + dx)
          for (let k = starts[c]!; k < starts[c + 1]!; k++) {
            const t = bucket[k]!
            const i0 = triangles[t * 4]!
            const i1 = triangles[t * 4 + 1]!
            const i2 = triangles[t * 4 + 2]!
            if (i0 === i || i1 === i || i2 === i) continue
            tp = Math.min(tp, rayTri(o, n, t))
            tm = Math.min(tm, rayTri(o, [-n[0], -n[1], -n[2]], t))
          }
        }
      }
    }
    sampled++
    if (tp <= REACH) hitPlus++
    if (tm <= REACH) hitMinus++
    const fp = Math.min(tp, REACH)
    const fm = Math.min(tm, REACH)
    if (fp > fm) freerPlus++
    else if (fm > fp) freerMinus++
  }
  console.log(`\nsign test — ${REACH * 1000}mm rays from ${sampled.toLocaleString()} vertices (z-derived, un-negated n):`)
  console.log(`  +n hits geometry ${((hitPlus / sampled) * 100).toFixed(1)}% of the time, -n ${((hitMinus / sampled) * 100).toFixed(1)}%`)
  console.log(`  more free space along +n for ${((freerPlus / sampled) * 100).toFixed(1)}% of vertices, along -n for ${((freerMinus / sampled) * 100).toFixed(1)}%`)
  console.log(
    `  => the material is on the ${freerPlus > freerMinus ? '-n' : '+n'} side, so the un-negated normal points ${freerPlus > freerMinus ? 'OUTWARD: do NOT negate, which is what the repo does' : 'INWARD: negate'}.`,
  )
}

// ---------------------------------------------------------------------------
// TEST 3 — extreme vertex per 4mm xz column (no winding involved)
// ---------------------------------------------------------------------------

const CELL = 0.004
const gx = Math.max(1, Math.ceil((boundsMax[0] - boundsMin[0]) / CELL))
const gz = Math.max(1, Math.ceil((boundsMax[2] - boundsMin[2]) / CELL))
const topOf = new Int32Array(gx * gz).fill(-1)
for (let i = 0; i < vertexCount; i++) {
  const p = pos(i)
  const c =
    Math.min(gz - 1, Math.floor((p[2] - boundsMin[2]) / CELL)) * gx +
    Math.min(gx - 1, Math.floor((p[0] - boundsMin[0]) / CELL))
  if (topOf[c] === -1 || p[1] > pos(topOf[c]!)[1]) topOf[c] = i
}
let topN = 0
let topSum = 0
let topUp = 0
for (const i of topOf) {
  if (i === -1) continue
  const n = decode(i, 'z')
  topN++
  topSum += n[1]
  if (n[1] > 0) topUp++
}
const topMean = topSum / topN
console.log(
  `\ntopmost vertex of each ${CELL * 1000}mm column: ${topN.toLocaleString()} verts, un-negated mean n.y ${topMean.toFixed(4)} (${((topUp / topN) * 100).toFixed(1)}% up)`,
)
console.log(
  `  => ${
    Math.abs(topMean) < 0.2
      ? `|mean n.y| is only ${Math.abs(topMean).toFixed(2)}: these are near-horizontal normals and this test is near-random on this mesh. Ignore it — run it on elymus-repens, where |mean| is 0.72, to see it read clearly.`
      : `clear on this mesh: upward-facing geometry faces up ${topMean > 0 ? 'WITHOUT negation' : 'only when NEGATED'}.`
  }`,
)
