import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Part-atlas bake for the card cloud.
 *
 * The mesh's triangles are split (by centroid, against the median x/z of the
 * vertex cloud) into FOUR spatial quadrants — "sub-clumps". Splitting per
 * TRIANGLE, not per vertex, is what keeps blades intact: a blade always lands
 * whole on one card instead of being cut in half and torn apart by parallax.
 *
 * Each part gets its own impostor: 8 side views at 45deg azimuth steps plus one
 * straight-down crown view, captured orthographically against the part's own
 * tight box (so a 0.3m sub-clump gets the same texel density a 0.75m plant
 * would). Part 4 is the whole plant — the far LOD, and the reason the distant
 * field looks exactly like a good billboard.
 *
 * Every tile also stores the TIGHT BOUNDING BOX of its coverage, in
 * card-normalized coordinates. The runtime quad spans exactly that box, which
 * is what keeps 4 overlapping cards from costing 4x the fragments.
 *
 * Artifact layout (little-endian):
 *   0   u32 magic 'TDC1', u32 version, u32 atlasPx, u32 tilePx
 *   16  u32 grid, u32 nParts, u32 nViews, u32 normalPx
 *   32  f32[5][4] part box: cx, cz, y0, y1      (unit scale, metres)
 *   112 f32[5]    part radius r
 *   144 f32[45][4] tile coverage box: a0, a1, b0, b1
 *   1024 u8[atlasPx^2*4]   albedo rgba8 (straight colour, a = coverage)
 *   ...  u8[normalPx^2*2]  oct-encoded mesh-frame normals (y-primary)
 */

export const TILE = 256
export const GRID = 7
export const ATLAS = TILE * GRID // 1792
export const NORMAL_ATLAS = ATLAS / 2 // 896
export const N_SIDE = 8
export const N_VIEWS = 9 // 8 side + 1 crown
export const N_PARTS = 5 // 4 sub-clumps + whole plant
export const N_TILES = N_PARTS * N_VIEWS // 45 <= 49 grid slots

const SS = 2
const BIG = ATLAS * SS
const NRM_DOWN = SS * 2 // big -> normal atlas
const MAGIC = 0x31434454 // 'TDC1'
const VERSION = 3
const HEADER_BYTES = 1024
const DILATE_PASSES = 6
const COVERAGE_MIN = 6 // alpha (0..255) that counts as covered for tile boxes

export interface PartBox {
  cx: number
  cz: number
  y0: number
  y1: number
  r: number
}

/** Coverage box of one tile, in card-normalized coords (see TileBox usage). */
export interface TileBox {
  a0: number
  a1: number
  b0: number
  b1: number
}

export interface ClumpAtlas {
  parts: PartBox[]
  tiles: TileBox[]
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackAtlas(buf: ArrayBuffer): ClumpAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== ATLAS || u[3] !== TILE || u[4] !== GRID) return null
  if (u[5] !== N_PARTS || u[6] !== N_VIEWS || u[7] !== NORMAL_ATLAS) return null
  const albedoBytes = ATLAS * ATLAS * 4
  const normalBytes = NORMAL_ATLAS * NORMAL_ATLAS * 2
  if (buf.byteLength !== HEADER_BYTES + albedoBytes + normalBytes) return null
  const pf = new Float32Array(buf, 32, N_PARTS * 4)
  const rf = new Float32Array(buf, 112, N_PARTS)
  const tf = new Float32Array(buf, 144, N_TILES * 4)
  const parts: PartBox[] = []
  for (let i = 0; i < N_PARTS; i++) {
    parts.push({ cx: pf[i * 4]!, cz: pf[i * 4 + 1]!, y0: pf[i * 4 + 2]!, y1: pf[i * 4 + 3]!, r: rf[i]! })
  }
  const tiles: TileBox[] = []
  for (let i = 0; i < N_TILES; i++) {
    tiles.push({ a0: tf[i * 4]!, a1: tf[i * 4 + 1]!, b0: tf[i * 4 + 2]!, b1: tf[i * 4 + 3]! })
  }
  return {
    parts,
    tiles,
    albedo: new Uint8Array(buf, HEADER_BYTES, albedoBytes),
    normalOct: new Uint8Array(buf, HEADER_BYTES + albedoBytes, normalBytes),
  }
}

function packAtlas(a: ClumpAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.normalOct.byteLength)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = ATLAS
  u[3] = TILE
  u[4] = GRID
  u[5] = N_PARTS
  u[6] = N_VIEWS
  u[7] = NORMAL_ATLAS
  const pf = new Float32Array(buf, 32, N_PARTS * 4)
  const rf = new Float32Array(buf, 112, N_PARTS)
  a.parts.forEach((p, i) => {
    pf.set([p.cx, p.cz, p.y0, p.y1], i * 4)
    rf[i] = p.r
  })
  const tf = new Float32Array(buf, 144, N_TILES * 4)
  a.tiles.forEach((t, i) => tf.set([t.a0, t.a1, t.b0, t.b1], i * 4))
  new Uint8Array(buf, HEADER_BYTES, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + a.albedo.byteLength, a.normalOct.byteLength).set(a.normalOct)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the part atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesAtlas(ctx: BakeCtx, speciesId: string): Promise<ClumpAtlas> {
  const key = `parts-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesAtlas(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackAtlas(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackAtlas(buf)
    if (!atlas) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return atlas
}

/** Overwrite a poisoned OPFS cache entry (mirrors src/bake/cache.ts naming). */
async function opfsRepair(fullKey: string, data: ArrayBuffer): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle('bake-cache', { create: true })
    const handle = await dir.getFileHandle(`${fullKey}.bin`, { create: true })
    const writable = await handle.createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    /* best effort — cache misses just rebake */
  }
}

interface Partition {
  /** Part boxes 0..3 (quadrants) and 4 (whole plant). */
  boxes: PartBox[]
  /** Triangle indices reordered so each part occupies one contiguous range. */
  order: Uint32Array<ArrayBuffer>
  /** Index-buffer range per part (part 4 spans everything). */
  ranges: { first: number; count: number }[]
}

/**
 * Split triangles into four spatial quadrants by CENTROID (so blades stay
 * whole), against the median x/z of the vertex cloud so the four parts carry
 * comparable mass. Also computes each part's exact capture box.
 */
function partitionMesh(mesh: GcMesh): Partition {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535
  const v = mesh.vertices
  const tri = mesh.triangles
  const nTri = hdr.triangleCount
  const nVert = hdr.vertexCount

  // --- median split point (histogram over quantized vertex x/z) -------------
  const BINS = 512
  const hx = new Uint32Array(BINS)
  const hz = new Uint32Array(BINS)
  const binOf = BINS / 65536
  for (let i = 0; i < nVert; i++) {
    hx[(v[i * 8]! * binOf) | 0]!++
    hz[(v[i * 8 + 2]! * binOf) | 0]!++
  }
  const median = (h: Uint32Array): number => {
    let acc = 0
    for (let b = 0; b < BINS; b++) {
      acc += h[b]!
      if (acc * 2 >= nVert) return (b + 0.5) / BINS
    }
    return 0.5
  }
  const mx = bx0 + median(hx) * (bx1 - bx0)
  const mz = bz0 + median(hz) * (bz1 - bz0)

  // --- pass A: classify + per-part bounds -----------------------------------
  const part = new Uint8Array(nTri)
  const counts = new Uint32Array(4)
  const lo = [Infinity, Infinity, Infinity, Infinity]
  const hi = [-Infinity, -Infinity, -Infinity, -Infinity]
  const loZ = [Infinity, Infinity, Infinity, Infinity]
  const hiZ = [-Infinity, -Infinity, -Infinity, -Infinity]
  const loY = [Infinity, Infinity, Infinity, Infinity]
  const hiY = [-Infinity, -Infinity, -Infinity, -Infinity]
  for (let t = 0; t < nTri; t++) {
    const o = t * 4
    const i0 = tri[o]! * 8
    const i1 = tri[o + 1]! * 8
    const i2 = tri[o + 2]! * 8
    const x0 = bx0 + v[i0]! * sx
    const x1 = bx0 + v[i1]! * sx
    const x2 = bx0 + v[i2]! * sx
    const y0 = by0 + v[i0 + 1]! * sy
    const y1 = by0 + v[i1 + 1]! * sy
    const y2 = by0 + v[i2 + 1]! * sy
    const z0 = bz0 + v[i0 + 2]! * sz
    const z1 = bz0 + v[i1 + 2]! * sz
    const z2 = bz0 + v[i2 + 2]! * sz
    const cx = (x0 + x1 + x2) / 3
    const cz = (z0 + z1 + z2) / 3
    const p = (cx > mx ? 1 : 0) + (cz > mz ? 2 : 0)
    part[t] = p
    counts[p]!++
    if (x0 < lo[p]!) lo[p] = x0
    if (x1 < lo[p]!) lo[p] = x1
    if (x2 < lo[p]!) lo[p] = x2
    if (x0 > hi[p]!) hi[p] = x0
    if (x1 > hi[p]!) hi[p] = x1
    if (x2 > hi[p]!) hi[p] = x2
    if (z0 < loZ[p]!) loZ[p] = z0
    if (z1 < loZ[p]!) loZ[p] = z1
    if (z2 < loZ[p]!) loZ[p] = z2
    if (z0 > hiZ[p]!) hiZ[p] = z0
    if (z1 > hiZ[p]!) hiZ[p] = z1
    if (z2 > hiZ[p]!) hiZ[p] = z2
    if (y0 < loY[p]!) loY[p] = y0
    if (y1 < loY[p]!) loY[p] = y1
    if (y2 < loY[p]!) loY[p] = y2
    if (y0 > hiY[p]!) hiY[p] = y0
    if (y1 > hiY[p]!) hiY[p] = y1
    if (y2 > hiY[p]!) hiY[p] = y2
  }

  const boxes: PartBox[] = []
  for (let p = 0; p < 4; p++) {
    if (counts[p]! === 0) {
      boxes.push({ cx: mx, cz: mz, y0: 0, y1: Math.max(by1, 0.01), r: 0.01 })
      continue
    }
    boxes.push({
      cx: (lo[p]! + hi[p]!) / 2,
      cz: (loZ[p]! + hiZ[p]!) / 2,
      y0: Math.min(0, loY[p]!),
      y1: hiY[p]!,
      r: 0, // exact radius filled in pass B
    })
  }

  // --- pass B: exact radii + counting-sort the index buffer by part ---------
  const order = new Uint32Array(nTri * 3)
  const write = new Uint32Array(4)
  let acc = 0
  const ranges: { first: number; count: number }[] = []
  for (let p = 0; p < 4; p++) {
    write[p] = acc
    ranges.push({ first: acc, count: counts[p]! * 3 })
    acc += counts[p]! * 3
  }
  const r2 = [0, 0, 0, 0]
  for (let t = 0; t < nTri; t++) {
    const o = t * 4
    const p = part[t]!
    const box = boxes[p]!
    let w = write[p]!
    for (let k = 0; k < 3; k++) {
      const idx = tri[o + k]!
      order[w++] = idx
      const dx = bx0 + v[idx * 8]! * sx - box.cx
      const dz = bz0 + v[idx * 8 + 2]! * sz - box.cz
      const d = dx * dx + dz * dz
      if (d > r2[p]!) r2[p] = d
    }
    write[p] = w
  }
  for (let p = 0; p < 4; p++) boxes[p]!.r = Math.sqrt(r2[p]!) * 1.02 + 1e-3

  // --- part 4: the whole plant ---------------------------------------------
  const wcx = (bx0 + bx1) / 2
  const wcz = (bz0 + bz1) / 2
  let wr2 = 0
  for (let i = 0; i < nVert; i++) {
    const dx = bx0 + v[i * 8]! * sx - wcx
    const dz = bz0 + v[i * 8 + 2]! * sz - wcz
    const d = dx * dx + dz * dz
    if (d > wr2) wr2 = d
  }
  boxes.push({
    cx: wcx,
    cz: wcz,
    y0: Math.min(0, by0),
    y1: by1,
    r: Math.sqrt(wr2) * 1.02 + 1e-3,
  })
  ranges.push({ first: 0, count: nTri * 3 })

  return { boxes, order, ranges }
}

async function bakeSpeciesAtlas(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const { boxes, order, ranges } = partitionMesh(mesh)

  // --- transient GPU resources ---------------------------------------------
  const verts = mesh.vertices
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: order.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, order)

  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [BIG, BIG],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('bake-albedo')
  const nrmTex = mkTarget('bake-normal')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // Per-tile uniforms in one buffer with dynamic offsets (28 floats used).
  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_TILES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_TILES)
  for (let p = 0; p < N_PARTS; p++) {
    const box = boxes[p]!
    for (let view = 0; view < N_VIEWS; view++) {
      const o = ((p * N_VIEWS + view) * STRIDE) / 4
      if (view === N_SIDE) {
        scratch.set([1, 0, 0], o) // right = +X
        scratch.set([0, 0, -1], o + 4) // card V axis = -Z
        scratch.set([0, 1, 0, 1], o + 8) // fwd = +Y, w = is_top
      } else {
        const a = (view * 2 * Math.PI) / N_SIDE
        scratch.set([Math.cos(a), 0, -Math.sin(a)], o)
        scratch.set([0, 1, 0], o + 4)
        scratch.set([Math.sin(a), 0, Math.cos(a), 0], o + 8)
      }
      scratch.set([box.cx, box.cz, box.y0, box.y1], o + 12)
      scratch.set([box.r, 0, 0, 0], o + 16)
      scratch.set([bx0, by0, bz0, 0], o + 20)
      scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 24)
    }
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 112 },
      },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
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
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/bake-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  const bigTile = TILE * SS
  for (let p = 0; p < N_PARTS; p++) {
    const range = ranges[p]!
    if (range.count === 0) continue
    for (let view = 0; view < N_VIEWS; view++) {
      const tile = p * N_VIEWS + view
      const col = tile % GRID
      const row = Math.floor(tile / GRID)
      pass.setViewport(col * bigTile, row * bigTile, bigTile, bigTile, 0, 1)
      pass.setScissorRect(col * bigTile, row * bigTile, bigTile, bigTile)
      pass.setBindGroup(0, bg, [tile * STRIDE])
      pass.drawIndexed(range.count, 1, range.first)
    }
  }
  pass.end()

  const bpr = BIG * 4
  const rbSize = bpr * BIG
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbNrm = mkReadback('bake-rb-normal')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNrm = new Uint8Array(rbNrm.getMappedRange()).slice()
  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()

  const albedo = downsampleAlbedo(bigAlb)
  const normalOct = downsampleNormals(bigNrm, bigAlb)
  const tiles = coverageBoxes(albedo)
  return packAtlas({ parts: boxes, tiles, albedo, normalOct })
}

/** Coverage-weighted 2x downsample of the albedo, then tile-clamped dilation. */
function downsampleAlbedo(big: Uint8Array): Uint8Array<ArrayBuffer> {
  const N = ATLAS
  const out = new Uint8Array(N * N * 4)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * BIG + (x * SS + i)) * 4
          const a = big[s + 3]!
          if (a === 0) continue
          aSum += a
          r += big[s]! * a
          g += big[s + 1]! * a
          b += big[s + 2]! * a
        }
      }
      const d = (y * N + x) * 4
      if (aSum > 0) {
        out[d] = Math.round(r / aSum)
        out[d + 1] = Math.round(g / aSum)
        out[d + 2] = Math.round(b / aSum)
        out[d + 3] = Math.round(aSum / (SS * SS))
      }
    }
  }
  dilateRgb(out, N, TILE, DILATE_PASSES)
  return out
}

/**
 * Coverage-weighted downsample of the normal target to half the albedo
 * resolution (lighting is low frequency; this buys back 4.8MB per species),
 * then oct-encode. Coverage comes from the albedo target's alpha.
 */
function downsampleNormals(bigNrm: Uint8Array, bigAlb: Uint8Array): Uint8Array<ArrayBuffer> {
  const N = NORMAL_ATLAS
  const rgb = new Uint8Array(N * N * 4) // xyz in 0..255 + coverage flag in a
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < NRM_DOWN; j++) {
        for (let i = 0; i < NRM_DOWN; i++) {
          const s = ((y * NRM_DOWN + j) * BIG + (x * NRM_DOWN + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          nx += (bigNrm[s]! / 127.5 - 1) * a
          ny += (bigNrm[s + 1]! / 127.5 - 1) * a
          nz += (bigNrm[s + 2]! / 127.5 - 1) * a
        }
      }
      const d = (y * N + x) * 4
      if (aSum > 0) {
        const len = Math.hypot(nx, ny, nz) || 1
        rgb[d] = Math.round(((nx / len) * 0.5 + 0.5) * 255)
        rgb[d + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255)
        rgb[d + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255)
        rgb[d + 3] = 255
      }
    }
  }
  dilateRgb(rgb, N, TILE / 2, 3)
  const oct = new Uint8Array(N * N * 2)
  for (let i = 0; i < N * N; i++) {
    const s = i * 4
    const x = rgb[s]! / 127.5 - 1
    const y = rgb[s + 1]! / 127.5 - 1
    const z = rgb[s + 2]! / 127.5 - 1
    oct.set(octEncode(x, y, z), i * 2)
  }
  return oct
}

/**
 * Dilate rgb into texels with alpha 0 (alpha itself is never touched) so
 * bilinear/mip filtering can never pull background black into a silhouette.
 * Clamped to tile boundaries so parts never bleed into each other.
 */
function dilateRgb(buf: Uint8Array, N: number, tile: number, passes: number): void {
  let filled = new Uint8Array(N * N)
  for (let i = 0; i < N * N; i++) filled[i] = buf[i * 4 + 3]! > 0 ? 1 : 0
  for (let pass = 0; pass < passes; pass++) {
    const next = filled.slice()
    for (let y = 0; y < N; y++) {
      const ty0 = Math.floor(y / tile) * tile
      const ty1 = ty0 + tile - 1
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (filled[idx]! !== 0) continue
        const tx0 = Math.floor(x / tile) * tile
        const tx1 = tx0 + tile - 1
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < tx0 || xx > tx1) continue
            const n = yy * N + xx
            if (filled[n]! === 0) continue
            count++
            r += buf[n * 4]!
            g += buf[n * 4 + 1]!
            b += buf[n * 4 + 2]!
          }
        }
        if (count === 0) continue
        buf[idx * 4] = Math.round(r / count)
        buf[idx * 4 + 1] = Math.round(g / count)
        buf[idx * 4 + 2] = Math.round(b / count)
        next[idx] = 1
      }
    }
    filled = next
  }
}

/**
 * The TIGHT COVERAGE BOX of every tile, in card-normalized coordinates:
 *   side view: a = u in [-1,1] across the card, b = height fraction in [0,1]
 *   crown view: a = u, b = v, both in [-1,1] across the card's footprint
 * The runtime quad spans exactly this box — that is what keeps four overlapping
 * cards from costing four cards' worth of fragments. Typical boxes come out
 * around 0.7 of the card, i.e. ~50% of the naive quad's area.
 */
function coverageBoxes(albedo: Uint8Array): TileBox[] {
  const out: TileBox[] = []
  for (let tile = 0; tile < N_TILES; tile++) {
    const col = tile % GRID
    const row = Math.floor(tile / GRID)
    const ox = col * TILE
    const oy = row * TILE
    let x0 = TILE
    let x1 = -1
    let y0 = TILE
    let y1 = -1
    for (let y = 0; y < TILE; y++) {
      const base = ((oy + y) * ATLAS + ox) * 4
      for (let x = 0; x < TILE; x++) {
        if (albedo[base + x * 4 + 3]! < COVERAGE_MIN) continue
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
    if (x1 < 0) {
      out.push({ a0: 0, a1: 0, b0: 0, b1: 0 }) // empty tile — runtime skips it
      continue
    }
    // One texel of margin, then texel centers -> card-normalized coords.
    const pad = 1.5
    const u0 = ((Math.max(0, x0 - pad) + 0.5) / TILE) * 2 - 1
    const u1 = ((Math.min(TILE - 1, x1 + pad) + 0.5) / TILE) * 2 - 1
    const isTop = tile % N_VIEWS === N_SIDE
    if (isTop) {
      // Card V axis points -Z and the viewport's +y is texture row 0.
      const v1 = 1 - ((Math.max(0, y0 - pad) + 0.5) / TILE) * 2
      const v0 = 1 - ((Math.min(TILE - 1, y1 + pad) + 0.5) / TILE) * 2
      out.push({ a0: u0, a1: u1, b0: v0, b1: v1 })
    } else {
      const h1 = 1 - (Math.max(0, y0 - pad) + 0.5) / TILE
      const h0 = 1 - (Math.min(TILE - 1, y1 + pad) + 0.5) / TILE
      out.push({ a0: u0, a1: u1, b0: h0, b1: h1 })
    }
  }
  return out
}

/**
 * Octahedral encode, y-primary — this atlas's OWN convention, the exact inverse
 * of `oct_decode` in mipgen.wgsl and `oct_decode_card` in cards.wgsl.
 */
function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  let u: number
  let v: number
  if (s < 1e-6) {
    u = 0
    v = 0
  } else {
    u = x / s
    v = z / s
    if (y < 0) {
      const fu = (1 - Math.abs(v)) * (x >= 0 ? 1 : -1)
      const fv = (1 - Math.abs(u)) * (z >= 0 ? 1 : -1)
      u = fu
      v = fv
    }
  }
  return [Math.round((u * 0.5 + 0.5) * 255), Math.round((v * 0.5 + 0.5) * 255)]
}
