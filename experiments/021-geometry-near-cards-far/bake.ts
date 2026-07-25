import bakeShaderSrc from './shaders/bake.wgsl'
import bakeTileSrc from './shaders/bake_tile.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Two-level bake: a NEAR card cloud and a FAR impostor atlas, both cut from
 * exactly the same capture box so the LOD handover cannot move, resize or
 * re-light a plant.
 *
 * NEAR (texture_2d_array, N_CARDS + 1 layers of NEAR_PX):
 *   The mesh is partitioned by the azimuth of each triangle's centroid around
 *   the clump axis into N_CARDS wedges of half-width WEDGE_DEG. Wedge k is
 *   rendered orthographically onto the vertical plane through the axis that
 *   CONTAINS its own direction, so every triangle moves at most
 *   r*sin(WEDGE_DEG) along the plane normal — and the card that turns edge-on
 *   is always the one holding the most foreshortened geometry.
 *   Layer N_CARDS is a straight-down top view of the whole mesh.
 *
 * FAR (texture_2d_array, N_AZI + 1 layers of FAR_PX):
 *   The classic impostor sheet — 8 azimuth side views of the whole mesh plus
 *   a top view, same box, same lighting-agnostic albedo/normal encoding.
 *
 * Artifact layout (little-endian):
 *   u32  magic 'GNC1', version, nearPx, nearLayers, farPx, farLayers,
 *        nCards, nAzi                                          (32B)
 *   f32  rXZ, y0, y1, cx, cz, wedgeDeg, bands, pad             (32B)
 *   f32  tileBounds[nearLayers + farLayers][4]  (u0,v0,u1,v1)  (208B)
 *   f32  cardBands[nCards][BANDS][2]  (u0,u1 per height band)  (144B)
 *   u8   near albedo rgba8   [layer][y][x]
 *   u8   near normal  rg8    (oct, mesh frame)
 *   u8   far  albedo rgba8
 *   u8   far  normal  rg8
 *
 * A card carries only 4-10% ink over its rect, so almost every rasterized
 * fragment is an alpha-test reject. cardBands is the cheap fix: each card is
 * drawn as BANDS stacked quads, each only as wide as the ink in its own height
 * band, which removes ~35% of the empty area for free (the vertex mapping is
 * linear in height, so bands are pixel-identical to one tall quad).
 */

export const NEAR_PX = 512
export const N_CARDS = 3
export const NEAR_LAYERS = N_CARDS + 1
export const FAR_PX = 256
export const N_AZI = 8
export const FAR_LAYERS = N_AZI + 1
/** Height bands per near card — keep equal to BANDS in entry_info.wgsl. */
export const BANDS = 6
/** Half-width of a card's azimuth wedge; 180/(2*N_CARDS) is an exact partition. */
const WEDGE_DEG = 30
const SS = 2
const MAGIC = 0x31434e47 // 'GNC1'
const VERSION = 3
const BOUNDS_FLOATS = (NEAR_LAYERS + FAR_LAYERS) * 4
const BAND_FLOATS = N_CARDS * BANDS * 2
const HEADER_BYTES = 64 + BOUNDS_FLOATS * 4 + BAND_FLOATS * 4
const DILATE_PASSES = 6

export interface CardCloud {
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  /** Per-layer tight content rect in tile UV: near layers first, then far. */
  bounds: Float32Array
  /** Per near card, per height band: (u0, u1) of the ink in that band. */
  cardBands: Float32Array
  nearAlbedo: Uint8Array<ArrayBuffer>
  nearNormal: Uint8Array<ArrayBuffer>
  farAlbedo: Uint8Array<ArrayBuffer>
  farNormal: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

const NEAR_BYTES = NEAR_PX * NEAR_PX * NEAR_LAYERS
const FAR_BYTES = FAR_PX * FAR_PX * FAR_LAYERS
const TOTAL_BYTES = HEADER_BYTES + NEAR_BYTES * 6 + FAR_BYTES * 6

export function unpackCloud(buf: ArrayBuffer): CardCloud | null {
  if (buf.byteLength !== TOTAL_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== NEAR_PX || u[3] !== NEAR_LAYERS || u[4] !== FAR_PX || u[5] !== FAR_LAYERS) return null
  const f = new Float32Array(buf, 32, 8)
  let o = HEADER_BYTES
  const nearAlbedo = new Uint8Array(buf, o, NEAR_BYTES * 4)
  o += NEAR_BYTES * 4
  const nearNormal = new Uint8Array(buf, o, NEAR_BYTES * 2)
  o += NEAR_BYTES * 2
  const farAlbedo = new Uint8Array(buf, o, FAR_BYTES * 4)
  o += FAR_BYTES * 4
  const farNormal = new Uint8Array(buf, o, FAR_BYTES * 2)
  const boundsEnd = 64 + BOUNDS_FLOATS * 4
  return {
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    bounds: new Float32Array(buf.slice(64, boundsEnd)),
    cardBands: new Float32Array(buf.slice(boundsEnd, boundsEnd + BAND_FLOATS * 4)),
    nearAlbedo,
    nearNormal,
    farAlbedo,
    farNormal,
  }
}

function packCloud(c: CardCloud): ArrayBuffer {
  const buf = new ArrayBuffer(TOTAL_BYTES)
  const u = new Uint32Array(buf, 0, 8)
  u.set([MAGIC, VERSION, NEAR_PX, NEAR_LAYERS, FAR_PX, FAR_LAYERS, N_CARDS, N_AZI])
  const f = new Float32Array(buf, 32, 8)
  f.set([c.rXZ, c.y0, c.y1, c.cx, c.cz, WEDGE_DEG, BANDS, 0])
  new Float32Array(buf, 64, BOUNDS_FLOATS).set(c.bounds)
  new Float32Array(buf, 64 + BOUNDS_FLOATS * 4, BAND_FLOATS).set(c.cardBands)
  let o = HEADER_BYTES
  const put = (src: Uint8Array): void => {
    new Uint8Array(buf, o, src.byteLength).set(src)
    o += src.byteLength
  }
  put(c.nearAlbedo)
  put(c.nearNormal)
  put(c.farAlbedo)
  put(c.farNormal)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake this species' card cloud. The dev
 * server answers missing /mesh/baked files with the SPA index.html at status
 * 200, which can poison bakedArtifact's stores — every result is validated,
 * and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesCloud(ctx: BakeCtx, speciesId: string): Promise<CardCloud> {
  const key = `cloud-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesCloud(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let cloud = unpackCloud(buf)
  if (!cloud) {
    buf = await runBake()
    cloud = unpackCloud(buf)
    if (!cloud) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return cloud
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

/**
 * Split the mesh into N_CARDS azimuth wedges around the clump axis.
 * Card k's plane contains the horizontal direction a_k = k*pi/N (and
 * a_k + pi), so a triangle belongs to card k when its centroid azimuth is
 * within WEDGE_DEG of that plane's own direction — i.e. the geometry a card
 * carries is the geometry that lies IN it, which is also the geometry that is
 * most foreshortened exactly when the card turns edge-on.
 * Returns one concatenated index buffer plus per-card [offset, count] ranges.
 */
function partitionWedges(mesh: GcMesh): { indices: Uint32Array<ArrayBuffer>; ranges: { offset: number; count: number }[] } {
  const hdr = mesh.header
  const [bx0, , bz0] = hdr.boundsMin
  const [bx1, , bz1] = hdr.boundsMax
  const cx = (bx0! + bx1!) / 2
  const cz = (bz0! + bz1!) / 2
  const sx = (bx1! - bx0!) / 65535
  const sz = (bz1! - bz0!) / 65535
  const tris = mesh.triangles
  const verts = mesh.vertices
  const n = hdr.triangleCount
  const wedge = (WEDGE_DEG * Math.PI) / 180
  const cardDir = new Float64Array(N_CARDS)
  for (let k = 0; k < N_CARDS; k++) cardDir[k] = (k * Math.PI) / N_CARDS

  const mask = new Uint8Array(n)
  const counts = new Uint32Array(N_CARDS)
  for (let t = 0; t < n; t++) {
    const b = t * 4
    const i0 = tris[b]! * 8
    const i1 = tris[b + 1]! * 8
    const i2 = tris[b + 2]! * 8
    const qx = (verts[i0]! + verts[i1]! + verts[i2]!) / 3
    const qz = (verts[i0 + 2]! + verts[i1 + 2]! + verts[i2 + 2]!) / 3
    const dx = bx0! + qx * sx - cx
    const dz = bz0! + qz * sz - cz
    // atan2(-dz, dx): matches dot(off, right_k) = |off| * cos(ang - a_k) with
    // right_k = (cos a_k, 0, -sin a_k), the bake's card U axis.
    const ang = Math.atan2(-dz, dx)
    let bits = 0
    for (let k = 0; k < N_CARDS; k++) {
      let d = ang - cardDir[k]!
      d -= Math.PI * Math.round(d / Math.PI) // fold onto [-pi/2, pi/2]
      if (Math.abs(d) <= wedge) {
        bits |= 1 << k
        counts[k]!++
      }
    }
    mask[t] = bits
  }

  const ranges: { offset: number; count: number }[] = []
  let total = 0
  for (let k = 0; k < N_CARDS; k++) {
    ranges.push({ offset: total * 3, count: counts[k]! * 3 })
    total += counts[k]!
  }
  const indices = new Uint32Array(total * 3)
  const cursor = new Uint32Array(N_CARDS)
  for (let k = 0; k < N_CARDS; k++) cursor[k] = ranges[k]!.offset
  for (let t = 0; t < n; t++) {
    const bits = mask[t]!
    if (bits === 0) continue
    const b = t * 4
    const a = tris[b]!
    const c = tris[b + 1]!
    const d = tris[b + 2]!
    for (let k = 0; k < N_CARDS; k++) {
      if ((bits & (1 << k)) === 0) continue
      let w = cursor[k]!
      indices[w++] = a
      indices[w++] = c
      indices[w++] = d
      cursor[k] = w
    }
  }
  return { indices, ranges }
}

interface ViewSpec {
  /** Card U axis in world/mesh space. */
  right: [number, number, number]
  /** Card V axis (top view only). */
  up: [number, number, number]
  /** Toward the bake camera. */
  fwd: [number, number, number]
  isTop: boolean
  /** Index range to draw. */
  offset: number
  count: number
  col: number
  row: number
}

async function bakeSpeciesCloud(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0! + bx1!) / 2
  const cz = (bz0! + bz1!) / 2

  // Exact horizontal support radius from the vertices (bounds corners
  // overestimate it noticeably on wide community tiles).
  const verts = mesh.vertices
  const sx = (bx1! - bx0!) / 65535
  const sz = (bz1! - bz0!) / 65535
  let r2 = 0
  for (let i = 0; i < hdr.vertexCount; i++) {
    const dx = bx0! + verts[i * 8]! * sx - cx
    const dz = bz0! + verts[i * 8 + 2]! * sz - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  const rXZ = Math.sqrt(r2) * 1.02 + 1e-3
  const y0 = Math.min(0, by0!)
  const y1 = by1!

  const { indices, ranges } = partitionWedges(mesh)
  const allOffset = 0
  const allCount = indices.length

  // --- transient GPU resources ---------------------------------------------
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  // Near grid: 2x2 tiles of NEAR_PX*SS. Far grid: 3x3 tiles of FAR_PX*SS.
  const nearTile = NEAR_PX * SS
  const nearGrid = 2
  const nearBig = nearTile * nearGrid
  const farTile = FAR_PX * SS
  const farGrid = 3
  const farBig = farTile * farGrid

  const sideAxes = (a: number): Pick<ViewSpec, 'right' | 'up' | 'fwd'> => ({
    right: [Math.cos(a), 0, -Math.sin(a)],
    up: [0, 1, 0],
    fwd: [Math.sin(a), 0, Math.cos(a)],
  })
  const topAxes: Pick<ViewSpec, 'right' | 'up' | 'fwd'> = { right: [1, 0, 0], up: [0, 0, -1], fwd: [0, 1, 0] }

  const nearViews: ViewSpec[] = []
  for (let k = 0; k < N_CARDS; k++) {
    // The bake's `right` axis IS the card's in-plane horizontal direction, so
    // the bake azimuth of card k is exactly its wedge direction a_k = k*pi/N
    // (the camera then automatically sits on the plane normal).
    const a = (k * Math.PI) / N_CARDS
    nearViews.push({
      ...sideAxes(a),
      isTop: false,
      offset: ranges[k]!.offset,
      count: ranges[k]!.count,
      col: k % nearGrid,
      row: Math.floor(k / nearGrid),
    })
  }
  nearViews.push({
    ...topAxes,
    isTop: true,
    offset: allOffset,
    count: allCount,
    col: N_CARDS % nearGrid,
    row: Math.floor(N_CARDS / nearGrid),
  })

  const farViews: ViewSpec[] = []
  for (let k = 0; k < N_AZI; k++) {
    farViews.push({
      ...sideAxes((k * 2 * Math.PI) / N_AZI),
      isTop: false,
      offset: allOffset,
      count: allCount,
      col: k % farGrid,
      row: Math.floor(k / farGrid),
    })
  }
  farViews.push({
    ...topAxes,
    isTop: true,
    offset: allOffset,
    count: allCount,
    col: N_AZI % farGrid,
    row: Math.floor(N_AZI / farGrid),
  })

  // Per-view uniforms with dynamic offsets (28 floats used of a 256B stride).
  const STRIDE = 256
  const allViews = [...nearViews, ...farViews]
  const uni = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-uni`,
      size: STRIDE * allViews.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * allViews.length)
  allViews.forEach((v, i) => {
    const o = (i * STRIDE) / 4
    scratch.set(v.right, o)
    scratch.set(v.up, o + 4)
    scratch.set([...v.fwd, v.isTop ? 1 : 0], o + 8)
    scratch.set([cx, cz, y0, y1], o + 12)
    scratch.set([rXZ, 0, 0, 0], o + 16)
    scratch.set([bx0!, by0!, bz0!, 0], o + 20)
    scratch.set([bx1! - bx0!, by1! - by0!, bz1! - bz0!, 0], o + 24)
  })
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

  const renderGrid = (
    label: string,
    big: number,
    tile: number,
    views: ViewSpec[],
    viewBase: number,
  ): { alb: GPUTexture; nrm: GPUTexture; depth: GPUTexture } => {
    const mk = (name: string, format: GPUTextureFormat, usage: number): GPUTexture =>
      ctx.res.createTexture({ label: `${ctx.id}/${name}`, size: [big, big], format, usage }, { tag: 'bake-scratch' })
    const rt = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    const alb = mk(`${label}-albedo`, 'rgba8unorm', rt)
    const nrm = mk(`${label}-normal`, 'rgba8unorm', rt)
    const depth = mk(`${label}-depth`, 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT)
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/${label}-pass`,
      colorAttachments: [
        { view: alb.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: nrm.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: depth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    views.forEach((v, i) => {
      pass.setViewport(v.col * tile, v.row * tile, tile, tile, 0, 1)
      pass.setScissorRect(v.col * tile, v.row * tile, tile, tile)
      pass.setBindGroup(0, bg, [(viewBase + i) * STRIDE])
      pass.drawIndexed(v.count, 1, v.offset)
    })
    pass.end()
    return { alb, nrm, depth }
  }

  const nearRt = renderGrid('bake-near', nearBig, nearTile, nearViews, 0)
  const farRt = renderGrid('bake-far', farBig, farTile, farViews, nearViews.length)

  const readback = (tex: GPUTexture, big: number, label: string): GPUBuffer => {
    const buf = ctx.res.createBuffer(
      {
        label: `${ctx.id}/${label}`,
        size: big * 4 * big,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      },
      { tag: 'bake-scratch' },
    )
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: big * 4, rowsPerImage: big }, [big, big])
    return buf
  }
  const rbNearAlb = readback(nearRt.alb, nearBig, 'rb-near-albedo')
  const rbNearNrm = readback(nearRt.nrm, nearBig, 'rb-near-normal')
  const rbFarAlb = readback(farRt.alb, farBig, 'rb-far-albedo')
  const rbFarNrm = readback(farRt.nrm, farBig, 'rb-far-normal')
  device.queue.submit([enc.finish()])

  const rbs = [rbNearAlb, rbNearNrm, rbFarAlb, rbFarNrm]
  await Promise.all(rbs.map((b) => b.mapAsync(GPUMapMode.READ)))
  const grab = (b: GPUBuffer): Uint8Array => new Uint8Array(b.getMappedRange()).slice()
  const bigNearAlb = grab(rbNearAlb)
  const bigNearNrm = grab(rbNearNrm)
  const bigFarAlb = grab(rbFarAlb)
  const bigFarNrm = grab(rbFarNrm)
  for (const b of rbs) b.unmap()
  for (const r of [vbuf, ibuf, uni, nearRt.alb, nearRt.nrm, nearRt.depth, farRt.alb, farRt.nrm, farRt.depth, ...rbs]) {
    r.destroy()
  }

  const near = processGrid(bigNearAlb, bigNearNrm, nearBig, NEAR_PX, nearViews)
  const far = processGrid(bigFarAlb, bigFarNrm, farBig, FAR_PX, farViews)
  const bounds = new Float32Array(BOUNDS_FLOATS)
  bounds.set(near.bounds, 0)
  bounds.set(far.bounds, NEAR_LAYERS * 4)
  // Only the N_CARDS vertical near cards are banded (the top cards and the
  // impostor keep one quad — their plant counts are what must stay cheap).
  const cardBands = new Float32Array(near.bands.subarray(0, BAND_FLOATS))

  return packCloud({
    rXZ,
    y0,
    y1,
    cx,
    cz,
    bounds,
    cardBands,
    nearAlbedo: near.albedo,
    nearNormal: near.normalOct,
    farAlbedo: far.albedo,
    farNormal: far.normalOct,
  })
}

/**
 * Coverage-weighted SSxSS downsample of one tile grid into layer-major array
 * data, then dilation into empty texels (so filtering never blends toward
 * background black), then a tight content rect per layer. Side tiles keep the
 * full vertical range: their bottom edge is deliberately sunk into the ground
 * at draw time and both LODs must agree on where that edge is.
 */
function processGrid(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  bigW: number,
  tilePx: number,
  views: ViewSpec[],
): {
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
  bounds: Float32Array
  bands: Float32Array
} {
  const T = tilePx
  const L = views.length
  const albedo = new Uint8Array(L * T * T * 4)
  const nrm = new Float32Array(L * T * T * 3)
  const filled = new Uint8Array(L * T * T)
  const bounds = new Float32Array(L * 4)
  const bands = new Float32Array(L * BANDS * 2)

  for (let l = 0; l < L; l++) {
    const ox = views[l]!.col * T * SS
    const oy = views[l]!.row * T * SS
    const layerBase = l * T * T
    for (let y = 0; y < T; y++) {
      for (let x = 0; x < T; x++) {
        let aSum = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = 0; j < SS; j++) {
          for (let i = 0; i < SS; i++) {
            const s = ((oy + y * SS + j) * bigW + (ox + x * SS + i)) * 4
            const a = bigAlb[s + 3]!
            if (a === 0) continue
            aSum += a
            r += bigAlb[s]! * a
            g += bigAlb[s + 1]! * a
            b += bigAlb[s + 2]! * a
            nx += (bigNrm[s]! / 127.5 - 1) * a
            ny += (bigNrm[s + 1]! / 127.5 - 1) * a
            nz += (bigNrm[s + 2]! / 127.5 - 1) * a
          }
        }
        if (aSum === 0) continue
        const idx = layerBase + y * T + x
        const d = idx * 4
        albedo[d] = Math.round(r / aSum)
        albedo[d + 1] = Math.round(g / aSum)
        albedo[d + 2] = Math.round(b / aSum)
        albedo[d + 3] = Math.round(aSum / (SS * SS))
        const dn = idx * 3
        nrm[dn] = nx
        nrm[dn + 1] = ny
        nrm[dn + 2] = nz
        filled[idx] = 1
      }
    }
  }

  // Content rect (from coverage, before dilation) with a 2-texel margin.
  for (let l = 0; l < L; l++) {
    const layerBase = l * T * T
    let x0 = T
    let y0 = T
    let x1 = -1
    let y1 = -1
    for (let y = 0; y < T; y++) {
      for (let x = 0; x < T; x++) {
        if (filled[layerBase + y * T + x] === 0) continue
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
    if (x1 < 0) {
      bounds.set([0, 0, 1, 1], l * 4)
      continue
    }
    const m = 2
    let u0 = Math.max(0, x0 - m) / T
    let u1 = Math.min(T, x1 + 1 + m) / T
    let v0 = Math.max(0, y0 - m) / T
    let v1 = Math.min(T, y1 + 1 + m) / T
    if (!views[l]!.isTop) {
      // Side cards span the whole capture height in both LODs.
      v0 = 0
      v1 = 1
    }
    bounds.set([u0, v0, u1, v1], l * 4)

    // Per-height-band horizontal hull: the same ink, ~35% less empty raster.
    for (let b = 0; b < BANDS; b++) {
      const ry0 = Math.floor((b * T) / BANDS)
      const ry1 = Math.floor(((b + 1) * T) / BANDS)
      let bx0 = T
      let bx1 = -1
      for (let y = ry0; y < ry1; y++) {
        for (let x = 0; x < T; x++) {
          if (filled[layerBase + y * T + x] === 0) continue
          if (x < bx0) bx0 = x
          if (x > bx1) bx1 = x
        }
      }
      const o = (l * BANDS + b) * 2
      if (bx1 < 0) {
        bands[o] = 0
        bands[o + 1] = 0
      } else {
        bands[o] = Math.max(0, bx0 - m) / T
        bands[o + 1] = Math.min(T, bx1 + 1 + m) / T
      }
    }
  }

  // Dilate color/normal into empty texels (alpha stays 0), per layer.
  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (let l = 0; l < L; l++) {
      const layerBase = l * T * T
      for (let y = 0; y < T; y++) {
        for (let x = 0; x < T; x++) {
          const idx = layerBase + y * T + x
          if (cur[idx]! !== 0) continue
          let count = 0
          let r = 0
          let g = 0
          let b = 0
          let nx = 0
          let ny = 0
          let nz = 0
          for (let j = -1; j <= 1; j++) {
            const yy = y + j
            if (yy < 0 || yy >= T) continue
            for (let i = -1; i <= 1; i++) {
              const xx = x + i
              if ((i === 0 && j === 0) || xx < 0 || xx >= T) continue
              const nIdx = layerBase + yy * T + xx
              if (cur[nIdx]! === 0) continue
              count++
              const s4 = nIdx * 4
              r += albedo[s4]!
              g += albedo[s4 + 1]!
              b += albedo[s4 + 2]!
              const s3 = nIdx * 3
              nx += nrm[s3]!
              ny += nrm[s3 + 1]!
              nz += nrm[s3 + 2]!
            }
          }
          if (count === 0) continue
          const d4 = idx * 4
          albedo[d4] = Math.round(r / count)
          albedo[d4 + 1] = Math.round(g / count)
          albedo[d4 + 2] = Math.round(b / count)
          const d3 = idx * 3
          nrm[d3] = nx / count
          nrm[d3 + 1] = ny / count
          nrm[d3 + 2] = nz / count
          next[idx] = 1
        }
      }
    }
    cur = next
  }

  const normalOct = new Uint8Array(L * T * T * 2)
  for (let i = 0; i < L * T * T; i++) {
    const s3 = i * 3
    normalOct.set(octEncode(nrm[s3]!, nrm[s3 + 1]!, nrm[s3 + 2]!), i * 2)
  }
  return { albedo, normalOct, bounds, bands }
}

// ---------------------------------------------------------------------------
// CARPET TILE bake — the second artifact kind, for `carpet_div > 0` species.
//
// A Sphagnum tile is not a plant with a silhouette; it is 0.18m of periodic
// cushion, 0.09m tall, that lies on the ground. Azimuth cards are the wrong
// primitive for it entirely (see NOTES), so a carpet species gets ONE
// straight-down capture of exactly its own period, plus a HEIGHT channel. The
// near LOD stacks that one texture into N ground-parallel shells, each keeping
// only the texels whose surface reaches its height, so the 3.3cm of capitulum
// relief becomes real geometry instead of a flat photo.
//
// Artifact layout (little-endian):
//   u32  magic 'GNCC', version, px, pad                        (16B)
//   f32  tileM, y0, y1, hLo, hMid, hHi, pad, pad, pad, pad, pad, pad (48B)
//   u8   albedo rgba8 [px*px]   (rgb, a = coverage)
//   u8   nrmh   rgba8 [px*px]   (n*0.5+0.5, a = surface height, normalized)
// ---------------------------------------------------------------------------

export const CARPET_PX = 512
const CARPET_SS = 2
const CARPET_MAGIC = 0x43434e47 // 'GNCC'
const CARPET_VERSION = 1
const CARPET_HEADER = 64
const CARPET_PLANE = CARPET_PX * CARPET_PX * 4
const CARPET_BYTES = CARPET_HEADER + CARPET_PLANE * 2
/** Height percentiles the shell stack spans (of the visible surface). */
const SHELL_LO_PCT = 0.05
const SHELL_HI_PCT = 0.97

export interface CarpetTile {
  /** Periodic tile size (m) — the capture is exactly [0, tileM]^2. */
  tileM: number
  y0: number
  y1: number
  /** Height (m, mesh frame) of the lowest / mean / highest shell plane. */
  hLo: number
  hMid: number
  hHi: number
  albedo: Uint8Array<ArrayBuffer>
  nrmh: Uint8Array<ArrayBuffer>
}

export function unpackTile(buf: ArrayBuffer): CarpetTile | null {
  if (buf.byteLength !== CARPET_BYTES) return null
  const u = new Uint32Array(buf, 0, 4)
  if (u[0] !== CARPET_MAGIC || u[1] !== CARPET_VERSION || u[2] !== CARPET_PX) return null
  const f = new Float32Array(buf, 16, 12)
  return {
    tileM: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    hLo: f[3]!,
    hMid: f[4]!,
    hHi: f[5]!,
    albedo: new Uint8Array(buf, CARPET_HEADER, CARPET_PLANE),
    nrmh: new Uint8Array(buf, CARPET_HEADER + CARPET_PLANE, CARPET_PLANE),
  }
}

function packTile(t: CarpetTile): ArrayBuffer {
  const buf = new ArrayBuffer(CARPET_BYTES)
  new Uint32Array(buf, 0, 4).set([CARPET_MAGIC, CARPET_VERSION, CARPET_PX, 0])
  new Float32Array(buf, 16, 12).set([t.tileM, t.y0, t.y1, t.hLo, t.hMid, t.hHi])
  new Uint8Array(buf, CARPET_HEADER, CARPET_PLANE).set(t.albedo)
  new Uint8Array(buf, CARPET_HEADER + CARPET_PLANE, CARPET_PLANE).set(t.nrmh)
  return buf
}

/** Load (OPFS cache / committed file) or bake this species' carpet tile. */
export async function loadSpeciesTile(ctx: BakeCtx, speciesId: string): Promise<CarpetTile> {
  const key = `carpet-v${CARPET_VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesTile(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let tile = unpackTile(buf)
  if (!tile) {
    buf = await runBake()
    tile = unpackTile(buf)
    if (!tile) throw new Error(`[${ctx.id}] carpet bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return tile
}

async function bakeSpeciesTile(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const tileM = hdr.tileSize[0]
  if (!(tileM > 0)) throw new Error(`[${ctx.id}] carpet bake needs a periodic mesh (tileSize = 0)`)
  const y0 = Math.min(0, hdr.boundsMin[1]!)
  const y1 = hdr.boundsMax[1]!
  const big = CARPET_PX * CARPET_SS

  const verts = mesh.vertices
  const indices = mesh.indices()
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/tile-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/tile-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/tile-uni`, size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  device.queue.writeBuffer(
    uni,
    0,
    new Float32Array([bx0!, by0!, bz0!, 0, bx1! - bx0!, by1! - by0!, bz1! - bz0!, 0, tileM, y0, y1, 0]),
  )

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/tile-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/tile-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni } }],
  })
  const module = ctx.shaders.module(bakeTileSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/tile-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/tile-pl`, bindGroupLayouts: [bgl] }),
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

  const mk = (name: string, format: GPUTextureFormat, usage: number): GPUTexture =>
    ctx.res.createTexture({ label: `${ctx.id}/${name}`, size: [big, big], format, usage }, { tag: 'bake-scratch' })
  const rt = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
  const albTex = mk('tile-albedo', 'rgba8unorm', rt)
  const nrmTex = mk('tile-nrmh', 'rgba8unorm', rt)
  const depthTex = mk('tile-depth', 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT)

  const enc = device.createCommandEncoder({ label: `${ctx.id}/tile-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/tile-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 1, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  pass.setBindGroup(0, bg)
  // 3x3 copies: the mesh overflows its period, so a neighbour's overhang has
  // to be inside our square for the mat to close. Everything outside the tile
  // square is clipped away by the ortho projection.
  pass.drawIndexed(indices.length, 9)
  pass.end()

  const readback = (tex: GPUTexture, label: string): GPUBuffer => {
    const buf = ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: big * big * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: big * 4, rowsPerImage: big }, [big, big])
    return buf
  }
  const rbAlb = readback(albTex, 'rb-tile-albedo')
  const rbNrm = readback(nrmTex, 'rb-tile-nrmh')
  device.queue.submit([enc.finish()])
  await Promise.all([rbAlb.mapAsync(GPUMapMode.READ), rbNrm.mapAsync(GPUMapMode.READ)])
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNrm = new Uint8Array(rbNrm.getMappedRange()).slice()
  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [vbuf, ibuf, uni, albTex, nrmTex, depthTex, rbAlb, rbNrm]) r.destroy()

  const tile = processTile(bigAlb, bigNrm, big)
  const span = y1 - y0
  return packTile({
    tileM,
    y0,
    y1,
    hLo: y0 + tile.hLo * span,
    hMid: y0 + tile.hMid * span,
    hHi: y0 + tile.hHi * span,
    albedo: tile.albedo,
    nrmh: tile.nrmh,
  })
}

/**
 * Coverage-weighted SSxSS downsample of the tile capture, dilation into empty
 * texels, and the height percentiles the shell stack spans. The normal is
 * stored as a plain unit vector in the upper hemisphere (NOT octahedral):
 * every normal here faces up, so a box-filtered mip of the raw vector is a
 * genuine average, while oct-encoded values average into nonsense.
 */
function processTile(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  bigW: number,
): { albedo: Uint8Array<ArrayBuffer>; nrmh: Uint8Array<ArrayBuffer>; hLo: number; hMid: number; hHi: number } {
  const T = CARPET_PX
  const albedo = new Uint8Array(T * T * 4)
  const nrmh = new Uint8Array(T * T * 4)
  const filled = new Uint8Array(T * T)
  const hist = new Float64Array(256)
  let covered = 0
  let hSum = 0

  for (let y = 0; y < T; y++) {
    for (let x = 0; x < T; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let h = 0
      for (let j = 0; j < CARPET_SS; j++) {
        for (let i = 0; i < CARPET_SS; i++) {
          const s = ((y * CARPET_SS + j) * bigW + (x * CARPET_SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
          nx += (bigNrm[s]! / 127.5 - 1) * a
          ny += (bigNrm[s + 1]! / 127.5 - 1) * a
          nz += (bigNrm[s + 2]! / 127.5 - 1) * a
          h += (bigNrm[s + 3]! / 255) * a
        }
      }
      if (aSum === 0) continue
      const idx = y * T + x
      const d = idx * 4
      albedo[d] = Math.round(r / aSum)
      albedo[d + 1] = Math.round(g / aSum)
      albedo[d + 2] = Math.round(b / aSum)
      albedo[d + 3] = Math.round(aSum / (CARPET_SS * CARPET_SS))
      const len = Math.hypot(nx, ny, nz) || 1
      nrmh[d] = Math.round(((nx / len) * 0.5 + 0.5) * 255)
      nrmh[d + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255)
      nrmh[d + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255)
      const hn = h / aSum
      nrmh[d + 3] = Math.round(hn * 255)
      filled[idx] = 1
      covered++
      hSum += hn
      const bin = Math.min(255, Math.round(hn * 255))
      hist[bin] = hist[bin]! + 1
    }
  }

  // Height percentiles over the visible surface only.
  const pct = (p: number): number => {
    const want = covered * p
    let acc = 0
    for (let i = 0; i < 256; i++) {
      acc += hist[i]!
      if (acc >= want) return i / 255
    }
    return 1
  }
  const hLo = covered > 0 ? pct(SHELL_LO_PCT) : 0
  const hHi = covered > 0 ? pct(SHELL_HI_PCT) : 1
  const hMid = covered > 0 ? hSum / covered : 0.5

  // Dilate colour / normal / height into empty texels (coverage stays 0), so
  // bilinear filtering and the mip chain never blend toward background black.
  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (let y = 0; y < T; y++) {
      for (let x = 0; x < T; x++) {
        const idx = y * T + x
        if (cur[idx]! !== 0) continue
        let count = 0
        let ar = 0
        let ag = 0
        let ab = 0
        let nx = 0
        let ny = 0
        let nz = 0
        let nh = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= T) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < 0 || xx >= T) continue
            const n = yy * T + xx
            if (cur[n]! === 0) continue
            count++
            const s = n * 4
            ar += albedo[s]!
            ag += albedo[s + 1]!
            ab += albedo[s + 2]!
            nx += nrmh[s]!
            ny += nrmh[s + 1]!
            nz += nrmh[s + 2]!
            nh += nrmh[s + 3]!
          }
        }
        if (count === 0) continue
        const d = idx * 4
        albedo[d] = Math.round(ar / count)
        albedo[d + 1] = Math.round(ag / count)
        albedo[d + 2] = Math.round(ab / count)
        nrmh[d] = Math.round(nx / count)
        nrmh[d + 1] = Math.round(ny / count)
        nrmh[d + 2] = Math.round(nz / count)
        nrmh[d + 3] = Math.round(nh / count)
        next[idx] = 1
      }
    }
    cur = next
  }

  return { albedo, nrmh, hLo, hMid, hHi }
}

/** Octahedral encode, y-primary — exact inverse of the decode in gcmesh.ts. */
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
