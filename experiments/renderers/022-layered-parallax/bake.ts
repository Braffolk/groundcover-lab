import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Depth-slab bake for layered parallax.
 *
 * Per species the source mesh is captured orthographically from 8 azimuths
 * plus straight down. Each capture is split into N_LAYER = 3 SLABS along that
 * view's own axis (front / middle / back for the side views, top / middle /
 * bottom bands for the down view); every slab becomes its own atlas tile and
 * records the MEAN DEPTH of the geometry it holds. Those means are the planes
 * the runtime shader intersects the eye ray with — that is where the parallax
 * comes from.
 *
 * Two atlases, because the near and far renderers want opposite things:
 *   NEAR atlas — the 27 slab tiles, in 9 view-group BLOCKS of (TW + TW/2) x TH:
 *     the front slab (where most fragments resolve) takes the full TW x TH tile
 *     and the two slabs behind it stack beside it at quarter area. A fragment
 *     that falls through to a deeper slab therefore stays inside the same
 *     texture pages. Only 5 mip levels: near cards never minify far.
 *   FAR atlas  — the 9 merged tiles (8 azimuths + top) at quarter area with a
 *     FULL mip chain, in its own small, cache-friendly texture. Beyond
 *     lodDistance that is all the far path touches, so distant plants cost
 *     exactly what a billboard costs.
 *
 * Bake canvas (texels), one supersampled render target that holds both:
 *   x 0        .. 4.5*TW   3x3 grid of view-group blocks — the near atlas
 *   x 4.5*TW   .. 6*TW     3x3 grid of half tiles        — the far atlas
 *
 * Artifact layout (little-endian):
 *   u32 magic 'LPX1', version, nearW, nearH, nearNW, nearNH,
 *       farW, farH, farNW, farNH, nAzim, nLayer, nTiles, tileW, tileH, pad
 *   f32 R, y0, y1, cx, cz                              (capture box, unit scale)
 *   ...zeros to byte 128
 *   f32[nTiles*8]  per tile: u0, v0, du, dv (normalized content rect inside
 *                  ITS OWN atlas), plane depth (m), texW, texH, isFar
 *   u8[nearW*nearH*4]  near albedo rgba8 (straight colour, a = coverage)
 *   u8[nearNW*nearNH*2] near normals, oct-encoded rg8 (mesh frame, y-primary)
 *   u8[farW*farH*4]    far albedo
 *   u8[farNW*farNH*2]  far normals
 */

export const N_AZIM = 8
export const N_LAYER = 3
/** 8*3 side slabs + 3 top slabs + 8 merged side + 1 merged top. */
export const N_TILES = N_AZIM * N_LAYER + N_LAYER + N_AZIM + 1

/** Tile ids — must match the constants in shaders/cards.wgsl. */
export const tileSide = (k: number, layer: number): number => k * N_LAYER + layer
export const TILE_TOP0 = N_AZIM * N_LAYER // 24
export const TILE_MERGED_SIDE0 = TILE_TOP0 + N_LAYER // 27
export const TILE_MERGED_TOP = TILE_MERGED_SIDE0 + N_AZIM // 35

const MAGIC = 0x3158504c // 'LPX1'
const VERSION = 3
const HEADER_BYTES = 128
const TILE_FLOATS = 8
const SS = 2 // supersampling factor of the bake render
const DILATE_PASSES = 5
/** Content is rendered into the inner (1 - 2*INSET) of each tile: mip guard. */
const INSET = 0.02
const TARGET_TILE_TEXELS = 200_000
const MAX_TILE_H = 560
const BIG_T = 1e9

export interface TileRec {
  /** Normalized content rect inside the tile's own atlas: u0, v0, du, dv. */
  u0: number
  v0: number
  du: number
  dv: number
  /** Slab plane: depth along the view axis (side) or world height (top). */
  depth: number
  texW: number
  texH: number
  far: boolean
}

export interface SlabAtlas {
  nearW: number
  nearH: number
  nearNW: number
  nearNH: number
  farW: number
  farH: number
  farNW: number
  farNH: number
  tileW: number
  tileH: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  tiles: TileRec[]
  nearAlbedo: Uint8Array<ArrayBuffer>
  nearNormal: Uint8Array<ArrayBuffer>
  farAlbedo: Uint8Array<ArrayBuffer>
  farNormal: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

// ---------------------------------------------------------------------------
// artifact (de)serialization
// ---------------------------------------------------------------------------

export function unpackSlabs(buf: ArrayBuffer): SlabAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 16)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const [, , nearW, nearH, nearNW, nearNH, farW, farH, farNW, farNH, nAzim, nLayer, nTiles, tileW, tileH] = u as
    unknown as number[]
  if (nAzim !== N_AZIM || nLayer !== N_LAYER || nTiles !== N_TILES) return null
  const f = new Float32Array(buf, 64, 5)
  const tableBytes = N_TILES * TILE_FLOATS * 4
  const nearABytes = nearW! * nearH! * 4
  const nearNBytes = nearNW! * nearNH! * 2
  const farABytes = farW! * farH! * 4
  const farNBytes = farNW! * farNH! * 2
  const total = HEADER_BYTES + tableBytes + nearABytes + nearNBytes + farABytes + farNBytes
  if (buf.byteLength !== total) return null
  const table = new Float32Array(buf, HEADER_BYTES, N_TILES * TILE_FLOATS)
  const tiles: TileRec[] = []
  for (let i = 0; i < N_TILES; i++) {
    const o = i * TILE_FLOATS
    tiles.push({
      u0: table[o]!,
      v0: table[o + 1]!,
      du: table[o + 2]!,
      dv: table[o + 3]!,
      depth: table[o + 4]!,
      texW: table[o + 5]!,
      texH: table[o + 6]!,
      far: table[o + 7]! > 0.5,
    })
  }
  let off = HEADER_BYTES + tableBytes
  const take = (bytes: number): Uint8Array<ArrayBuffer> => {
    const a = new Uint8Array(buf, off, bytes)
    off += bytes
    return a
  }
  return {
    nearW: nearW!,
    nearH: nearH!,
    nearNW: nearNW!,
    nearNH: nearNH!,
    farW: farW!,
    farH: farH!,
    farNW: farNW!,
    farNH: farNH!,
    tileW: tileW!,
    tileH: tileH!,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    tiles,
    nearAlbedo: take(nearABytes),
    nearNormal: take(nearNBytes),
    farAlbedo: take(farABytes),
    farNormal: take(farNBytes),
  }
}

function packSlabs(a: SlabAtlas): ArrayBuffer {
  const tableBytes = N_TILES * TILE_FLOATS * 4
  const payload = [a.nearAlbedo, a.nearNormal, a.farAlbedo, a.farNormal]
  const payloadBytes = payload.reduce((s, p) => s + p.byteLength, 0)
  const buf = new ArrayBuffer(HEADER_BYTES + tableBytes + payloadBytes)
  const u = new Uint32Array(buf, 0, 16)
  u.set([
    MAGIC,
    VERSION,
    a.nearW,
    a.nearH,
    a.nearNW,
    a.nearNH,
    a.farW,
    a.farH,
    a.farNW,
    a.farNH,
    N_AZIM,
    N_LAYER,
    N_TILES,
    a.tileW,
    a.tileH,
    0,
  ])
  new Float32Array(buf, 64, 5).set([a.rXZ, a.y0, a.y1, a.cx, a.cz])
  const table = new Float32Array(buf, HEADER_BYTES, N_TILES * TILE_FLOATS)
  a.tiles.forEach((t, i) => {
    table.set([t.u0, t.v0, t.du, t.dv, t.depth, t.texW, t.texH, t.far ? 1 : 0], i * TILE_FLOATS)
  })
  let off = HEADER_BYTES + tableBytes
  for (const p of payload) {
    new Uint8Array(buf, off, p.byteLength).set(p)
    off += p.byteLength
  }
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the slab atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesSlabs(ctx: BakeCtx, speciesId: string): Promise<SlabAtlas> {
  const key = `slabs-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesSlabs(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackSlabs(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackSlabs(buf)
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

// ---------------------------------------------------------------------------
// layout
// ---------------------------------------------------------------------------

interface Rect {
  x: number
  y: number
  w: number
  h: number
}

const round16 = (v: number): number => Math.max(64, Math.round(v / 16) * 16)

interface Layout {
  tileW: number
  tileH: number
  canvasW: number
  canvasH: number
  nearW: number
  nearH: number
  farW: number
  farH: number
  farX: number
  /** Rect of every tile inside the bake canvas. */
  canvasRects: Rect[]
  isFar: boolean[]
}

function buildLayout(rXZ: number, y0: number, y1: number): Layout {
  const aspect = Math.max(0.15, (2 * rXZ) / Math.max(y1 - y0, 1e-4))
  const th = round16(Math.min(MAX_TILE_H, Math.sqrt(TARGET_TILE_TEXELS / aspect)))
  const tw = round16(th * aspect)
  const hw = tw / 2
  const hh = th / 2
  // A view group's three slabs live in ONE contiguous block — a fragment that
  // falls through to slab 1 or 2 then stays inside the same texture pages
  // instead of chasing three far-apart tiles across the atlas.
  const blockW = tw + hw
  const nearW = blockW * 3
  const nearH = th * 3
  const farW = hw * 3
  const farH = hh * 3
  const farX = nearW

  const group = (slot: number, layer: number): Rect => {
    const bx = (slot % 3) * blockW
    const by = Math.floor(slot / 3) * th
    if (layer === 0) return { x: bx, y: by, w: tw, h: th }
    return { x: bx + tw, y: by + (layer - 1) * hh, w: hw, h: hh }
  }
  const far = (slot: number): Rect => ({
    x: farX + (slot % 3) * hw,
    y: Math.floor(slot / 3) * hh,
    w: hw,
    h: hh,
  })

  const canvasRects: Rect[] = new Array<Rect>(N_TILES)
  const isFar: boolean[] = new Array<boolean>(N_TILES).fill(false)
  for (let k = 0; k < N_AZIM; k++) {
    for (let l = 0; l < N_LAYER; l++) canvasRects[tileSide(k, l)] = group(k, l)
    canvasRects[TILE_MERGED_SIDE0 + k] = far(k)
    isFar[TILE_MERGED_SIDE0 + k] = true
  }
  for (let l = 0; l < N_LAYER; l++) canvasRects[TILE_TOP0 + l] = group(N_AZIM, l)
  canvasRects[TILE_MERGED_TOP] = far(N_AZIM)
  isFar[TILE_MERGED_TOP] = true

  return {
    tileW: tw,
    tileH: th,
    canvasW: nearW + farW,
    canvasH: nearH,
    nearW,
    nearH,
    farW,
    farH,
    farX,
    canvasRects,
    isFar,
  }
}

// ---------------------------------------------------------------------------
// the bake itself
// ---------------------------------------------------------------------------

async function bakeSpeciesSlabs(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2

  // Exact horizontal support radius from the vertices (bounds corners
  // overestimate it noticeably on wide community tiles).
  const verts = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sz = (bz1 - bz0) / 65535
  let r2 = 0
  for (let i = 0; i < hdr.vertexCount; i++) {
    const dx = bx0 + verts[i * 8]! * sx - cx
    const dz = bz0 + verts[i * 8 + 2]! * sz - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  const rXZ = Math.sqrt(r2) * 1.02 + 1e-3
  const y0 = Math.min(0, by0)
  const y1 = by1
  const layout = buildLayout(rXZ, y0, y1)
  const { tileW, tileH, canvasW, canvasH, nearW, nearH, farW, farH, farX } = layout

  const depths = slabPlanes(mesh, cx, cz, y0, y1, rXZ)

  // Tile table: rects normalized inside each tile's OWN atlas.
  const tiles: TileRec[] = layout.canvasRects.map((r, i) => {
    const far = layout.isFar[i]!
    const ax = far ? farX : 0
    const ay = 0
    const aw = far ? farW : nearW
    const ah = far ? farH : nearH
    const ix = r.x - ax + r.w * INSET
    const iy = r.y - ay + r.h * INSET
    const iw = r.w * (1 - 2 * INSET)
    const ih = r.h * (1 - 2 * INSET)
    return { u0: ix / aw, v0: iy / ah, du: iw / aw, dv: ih / ah, depth: depths[i]!, texW: iw, texH: ih, far }
  })

  // --- transient GPU resources ---------------------------------------------
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const bigW = canvasW * SS
  const bigH = canvasH * SS
  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [bigW, bigH],
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
      size: [bigW, bigH],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // --- per-instance tile view records (9 view groups x 4 instances) ---------
  const N_GROUPS = N_AZIM + 1
  const REC_FLOATS = 20
  const recs = new Float32Array(N_GROUPS * 4 * REC_FLOATS)
  const setRec = (slot: number, view: number[], tileId: number, slabMin: number, slabMax: number): void => {
    const r = layout.canvasRects[tileId]!
    const u0 = (r.x + r.w * INSET) / canvasW
    const v0 = (r.y + r.h * INSET) / canvasH
    const du = (r.w * (1 - 2 * INSET)) / canvasW
    const dv = (r.h * (1 - 2 * INSET)) / canvasH
    const o = slot * REC_FLOATS
    recs.set(view, o) // right(4) + up(4) + fwd(4)
    recs[o + 12] = slabMin
    recs[o + 13] = slabMax
    recs[o + 16] = 2 * u0 + du - 1
    recs[o + 17] = 1 - 2 * v0 - dv
    recs[o + 18] = du
    recs[o + 19] = dv
  }
  const bandY = (f: number): number => y0 + (y1 - y0) * f
  for (let g = 0; g < N_GROUPS; g++) {
    const isTop = g === N_AZIM
    let view: number[]
    if (isTop) {
      view = [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
    } else {
      const a = (g * 2 * Math.PI) / N_AZIM
      view = [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0]
    }
    const bounds = isTop
      ? [bandY(2 / 3), BIG_T, bandY(1 / 3), bandY(2 / 3), -BIG_T, bandY(1 / 3)]
      : [rXZ / 3, BIG_T, -rXZ / 3, rXZ / 3, -BIG_T, -rXZ / 3]
    for (let l = 0; l < N_LAYER; l++) {
      const tileId = isTop ? TILE_TOP0 + l : tileSide(g, l)
      setRec(g * 4 + l, view, tileId, bounds[l * 2]!, bounds[l * 2 + 1]!)
    }
    setRec(g * 4 + 3, view, isTop ? TILE_MERGED_TOP : TILE_MERGED_SIDE0 + g, -BIG_T, BIG_T)
  }
  const recBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-recs`, size: recs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(recBuf, 0, recs)

  const constants = new Float32Array(16)
  constants.set([cx, cz, y0, y1], 0)
  constants.set([rXZ, 0, 0, 0], 4)
  constants.set([bx0, by0, bz0, 0], 8)
  constants.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], 12)
  const constBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-const`,
      size: constants.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(constBuf, 0, constants)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: recBuf } },
      { binding: 1, resource: { buffer: constBuf } },
    ],
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

  // One submit per view group keeps any single command buffer small enough for
  // the 6.5M-triangle poa mesh (4 instances each) not to trip a GPU watchdog.
  const albView = albTex.createView()
  const nrmView = nrmTex.createView()
  const depthView = depthTex.createView()
  for (let g = 0; g < N_GROUPS; g++) {
    const first = g === 0
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc-${g}` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-pass-${g}`,
      colorAttachments: [
        { view: albView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: first ? 'clear' : 'load', storeOp: 'store' },
        {
          view: nrmView,
          clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 },
          loadOp: first ? 'clear' : 'load',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: first ? 'clear' : 'load',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    pass.drawIndexed(indices.length, 4, 0, 0, g * 4)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }
  // Free the source geometry before allocating readback staging (poa's vertex
  // + index buffers alone are ~210MB).
  vbuf.destroy()
  ibuf.destroy()
  recBuf.destroy()
  constBuf.destroy()
  depthTex.destroy()

  // --- readback -------------------------------------------------------------
  const bpr = Math.ceil((bigW * 4) / 256) * 256
  const rbSize = bpr * bigH
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbNrm = mkReadback('bake-rb-normal')
  const encCopy = device.createCommandEncoder({ label: `${ctx.id}/bake-copy` })
  encCopy.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: bigH }, [
    bigW,
    bigH,
  ])
  encCopy.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: bigH }, [
    bigW,
    bigH,
  ])
  device.queue.submit([encCopy.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNrm = new Uint8Array(rbNrm.getMappedRange()).slice()
  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [albTex, nrmTex, rbAlb, rbNrm]) r.destroy()

  const out = postProcess(bigAlb, bigNrm, bpr, layout)
  return packSlabs({
    nearW,
    nearH,
    nearNW: nearW >> 1,
    nearNH: nearH >> 1,
    farW,
    farH,
    farNW: farW >> 1,
    farNH: farH >> 1,
    tileW,
    tileH,
    rXZ,
    y0,
    y1,
    cx,
    cz,
    tiles,
    ...out,
  })
}

/**
 * Mean depth of the geometry inside each slab — the plane the runtime shader
 * intersects. Sampled over the vertices (a stride keeps it linear-ish on the
 * 8.5M-vertex poa mesh); empty slabs fall back to their geometric centre.
 */
function slabPlanes(mesh: GcMesh, cx: number, cz: number, y0: number, y1: number, rXZ: number): number[] {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535
  const verts = mesh.vertices
  const stride = Math.max(1, Math.floor(hdr.vertexCount / 250_000))

  const sums = new Float64Array(N_TILES)
  const counts = new Float64Array(N_TILES)
  const cosA: number[] = []
  const sinA: number[] = []
  for (let k = 0; k < N_AZIM; k++) {
    const a = (k * 2 * Math.PI) / N_AZIM
    sinA.push(Math.sin(a))
    cosA.push(Math.cos(a))
  }
  const yb1 = y0 + (y1 - y0) / 3
  const yb2 = y0 + (2 * (y1 - y0)) / 3

  for (let i = 0; i < hdr.vertexCount; i += stride) {
    const px = bx0 + verts[i * 8]! * sx - cx
    const py = by0 + verts[i * 8 + 1]! * sy
    const pz = bz0 + verts[i * 8 + 2]! * sz - cz
    for (let k = 0; k < N_AZIM; k++) {
      const t = px * sinA[k]! + pz * cosA[k]!
      const l = t > rXZ / 3 ? 0 : t > -rXZ / 3 ? 1 : 2
      const id = tileSide(k, l)
      sums[id]! += t
      counts[id]!++
      sums[TILE_MERGED_SIDE0 + k]! += t
      counts[TILE_MERGED_SIDE0 + k]!++
    }
    const lt = py > yb2 ? 0 : py > yb1 ? 1 : 2
    sums[TILE_TOP0 + lt]! += py
    counts[TILE_TOP0 + lt]!++
    sums[TILE_MERGED_TOP]! += py
    counts[TILE_MERGED_TOP]!++
  }

  const out = new Array<number>(N_TILES).fill(0)
  const fallbackSide = [(2 * rXZ) / 3, 0, (-2 * rXZ) / 3]
  const fallbackTop = [y0 + (5 * (y1 - y0)) / 6, y0 + (y1 - y0) / 2, y0 + (y1 - y0) / 6]
  for (let i = 0; i < N_TILES; i++) {
    if (counts[i]! > 0) {
      out[i] = sums[i]! / counts[i]!
    } else if (i < TILE_TOP0) {
      out[i] = fallbackSide[i % N_LAYER]!
    } else if (i < TILE_MERGED_SIDE0) {
      out[i] = fallbackTop[i - TILE_TOP0]!
    } else if (i < TILE_MERGED_TOP) {
      out[i] = 0
    } else {
      out[i] = (y0 + y1) * 0.5
    }
  }
  return out
}

interface PostOut {
  nearAlbedo: Uint8Array<ArrayBuffer>
  nearNormal: Uint8Array<ArrayBuffer>
  farAlbedo: Uint8Array<ArrayBuffer>
  farNormal: Uint8Array<ArrayBuffer>
}

/** Coverage-weighted 2x downsample, per-tile dilation, then split the canvas. */
function postProcess(bigAlb: Uint8Array, bigNrm: Uint8Array, bpr: number, layout: Layout): PostOut {
  const { canvasW: W, canvasH: H, tileW: TW, tileH: TH, nearW, farW, farH, farX } = layout
  const hw = TW / 2
  const hh = TH / 2
  const albedo = new Uint8Array(W * H * 4)
  const nrm = new Float32Array(W * H * 3)
  const filled = new Uint8Array(W * H)

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = (y * SS + j) * bpr + (x * SS + i) * 4
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
      const d = (y * W + x) * 4
      if (aSum > 0) {
        albedo[d] = Math.round(r / aSum)
        albedo[d + 1] = Math.round(g / aSum)
        albedo[d + 2] = Math.round(b / aSum)
        albedo[d + 3] = Math.round(aSum / (SS * SS))
        const dn = (y * W + x) * 3
        nrm[dn] = nx
        nrm[dn + 1] = ny
        nrm[dn + 2] = nz
        filled[y * W + x] = 1
      }
    }
  }

  // Tile bounds of a texel, so dilation never bleeds across a tile border.
  const blockW = TW + hw
  const tileBounds = (x: number, y: number): [number, number, number, number] => {
    if (x >= farX) {
      const bx = farX + Math.floor((x - farX) / hw) * hw
      const by = Math.floor(y / hh) * hh
      return [bx, by, bx + hw - 1, by + hh - 1]
    }
    const gx = Math.floor(x / blockW) * blockW
    const gy = Math.floor(y / TH) * TH
    if (x - gx < TW) return [gx, gy, gx + TW - 1, gy + TH - 1]
    const bx = gx + TW
    const by = gy + (y - gy < hh ? 0 : hh)
    return [bx, by, bx + hw - 1, by + hh - 1]
  }

  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x
        if (cur[idx]! !== 0) continue
        const [tx0, ty0, tx1, ty1] = tileBounds(x, y)
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const nIdx = yy * W + xx
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
    cur = next
  }

  /** Copy a sub-rectangle of the canvas out as its own rgba8 image. */
  const cropAlbedo = (x0: number, y0: number, w: number, h: number): Uint8Array<ArrayBuffer> => {
    const out = new Uint8Array(w * h * 4)
    for (let y = 0; y < h; y++) {
      const src = ((y0 + y) * W + x0) * 4
      out.set(albedo.subarray(src, src + w * 4), y * w * 4)
    }
    return out
  }

  /**
   * Normals live at half the albedo resolution: they are low frequency next to
   * coverage, and the texels bought back go into the front slab.
   */
  const cropNormal = (x0: number, y0: number, w: number, h: number): Uint8Array<ArrayBuffer> => {
    const nw = w >> 1
    const nh = h >> 1
    const out = new Uint8Array(nw * nh * 2)
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        let wsum = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 2; i++) {
            const si = (y0 + y * 2 + j) * W + (x0 + x * 2 + i)
            const wgt = albedo[si * 4 + 3]! / 255 + 0.002
            nx += nrm[si * 3]! * wgt
            ny += nrm[si * 3 + 1]! * wgt
            nz += nrm[si * 3 + 2]! * wgt
            wsum += wgt
          }
        }
        out.set(octEncode(nx / wsum, ny / wsum, nz / wsum), (y * nw + x) * 2)
      }
    }
    return out
  }

  return {
    nearAlbedo: cropAlbedo(0, 0, nearW, H),
    nearNormal: cropNormal(0, 0, nearW, H),
    farAlbedo: cropAlbedo(farX, 0, farW, farH),
    farNormal: cropNormal(farX, 0, farW, farH),
  }
}

/** Octahedral encode, y-primary — exact inverse of `world_normal` in cards.wgsl. */
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
