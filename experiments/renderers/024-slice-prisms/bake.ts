import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Prism-atlas bake ("SLP2").
 *
 * A plant is cut into PRISMS — slices of the plant volume — and every prism
 * gets its own orthographic tile. The atlas is deliberately NOT uniform: it is
 * an 8x8 grid of 192px cells, and texels go where the eye is.
 *
 *   12 tiles @384  4 baked azimuths x 3 vertical depth slabs (slab 0 = front).
 *                  Cut planes are per-azimuth equal-mass quantiles of the
 *                  depth distribution, so each slab carries a third of the
 *                  geometry; the slab's CENTROID DEPTH is stored with it and
 *                  is what places the card in space at render time.
 *                  The other four view azimuths reuse these tiles MIRRORED:
 *                  view a+180 sees the same slabs with u flipped, depth
 *                  negated and the slab order reversed, which is exact except
 *                  for which surface wins inside one thin slab. Four azimuths
 *                  of storage, eight view directions, 384px each.
 *    3 tiles @192  3 horizontal height slabs seen straight down (0 = top).
 *    8 tiles @192  merged (all three slabs) per VIEW azimuth — the far LOD
 *                  card, i.e. a plain billboard. Resolved on the CPU from the
 *                  slab renders (front-most covered wins = exactly the
 *                  z-buffered result), never re-rendered. Only ever seen
 *                  beyond slabDist, where a plant is <120px tall, so 192px is
 *                  already oversampled.
 *    1 tile  @192  merged top view — the far LOD top card.
 *
 * EXCLUSIVE OWNERSHIP is the subtle part. A hard alpha test is non-linear, so
 * if a texel's coverage is split across two prisms both halves can fail the
 * test that the merged texel would have passed — the plant comes out thinner
 * and darker than the billboard, which is the opposite of the goal. So each
 * final texel is assigned to exactly ONE prism (the one owning most of its
 * front-most-covered subsamples) and given the MERGED colour and coverage.
 * The three prisms are then a perfect partition of the billboard image: same
 * silhouette, same colours, every texel drawn exactly once — but each at its
 * own depth in the world. That is also why the LOD switch is invisible.
 *
 * Rendered at 2x and downsampled with coverage weighting, then albedo/normals
 * are dilated a few texels into empty space so bilinear/mip sampling never
 * pulls in background black. Per tile a tight coverage box is stored twice:
 * as atlas UV (what to sample) and as tile-local UV (what geometry that maps
 * to), so the draw can shrink its quad to the texels that actually exist.
 *
 * Artifact layout (little-endian), header 1024B:
 *   0    u32 magic 'SLP2', u32 version, u32 atlasPx, u32 cellPx
 *   16   u32 nAzBaked, u32 nSlab, u32 nTop, u32 nTiles
 *   32   f32 rXZ, y0, y1, cx, cz        (capture box, unit scale, metres)
 *   64   f32[nAzBaked*nSlab] slab centroid depth along the baked axis
 *        (metres, + = toward the bake camera)
 *   112  f32[4] height of top slab 0..2 + [3] merged top height
 *   128  f32[nTiles*4] tight coverage box in ATLAS uv (u0, v0, u1, v1)
 *   512  f32[nTiles*4] the same box in TILE-LOCAL uv
 *   1024 u8[atlasPx^2*4] albedo rgba8 (straight colour, a = coverage)
 *        u8[atlasPx^2*2] normals, oct-encoded rg8 (mesh frame, y-primary)
 */

export const CELL = 192
export const GRID_CELLS = 8
export const ATLAS = CELL * GRID_CELLS // 1536
export const SLAB_PX = 384
export const SMALL_PX = 192
export const N_AZ_BAKED = 4
export const N_AZ_VIEW = 8
export const N_SLAB = 3
export const N_TOP = 3
export const N_TILES = N_AZ_BAKED * N_SLAB + N_TOP + N_AZ_VIEW + 1 // 24
/** Mip levels that keep every tile boundary texel-aligned (192 = 64*3). */
export const MIP_LEVELS = 7

const SS = 2
const BIG = ATLAS * SS
const MAGIC = 0x3250_4c53 // 'SLP2'
const VERSION = 3
const HEADER_BYTES = 1024
const DILATE_PASSES = 5
const COVER_EPS = 8 / 255

export const tileSlab = (azBaked: number, slab: number): number => azBaked * N_SLAB + slab
export const tileTopSlab = (j: number): number => N_AZ_BAKED * N_SLAB + j
export const tileMerged = (azView: number): number => N_AZ_BAKED * N_SLAB + N_TOP + azView
export const TILE_TOP_MERGED = N_TILES - 1

export interface TileRect {
  x: number
  y: number
  size: number
}

/**
 * THE atlas layout. Slab tiles take 2x2 cells in the top six rows; every
 * small tile takes one cell in the bottom two rows.
 */
export function tileRects(): TileRect[] {
  const rects: TileRect[] = new Array<TileRect>(N_TILES)
  for (let t = 0; t < N_AZ_BAKED * N_SLAB; t++) {
    rects[t] = { x: (t % 4) * 2 * CELL, y: Math.floor(t / 4) * 2 * CELL, size: SLAB_PX }
  }
  const small = (i: number): TileRect => ({
    x: (i % GRID_CELLS) * CELL,
    y: (6 + Math.floor(i / GRID_CELLS)) * CELL,
    size: SMALL_PX,
  })
  for (let j = 0; j < N_TOP; j++) rects[tileTopSlab(j)] = small(j)
  for (let a = 0; a < N_AZ_VIEW; a++) rects[tileMerged(a)] = small(N_TOP + a)
  rects[TILE_TOP_MERGED] = small(N_TOP + N_AZ_VIEW)
  return rects
}

export interface PrismAtlas {
  atlasPx: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  /** [azBaked * N_SLAB + slab] centroid depth, metres, + = toward the camera. */
  slabDepth: Float32Array
  /** [0..2] top-slab centre heights, [3] merged top card height (m). */
  topHeight: Float32Array
  /** [tile * 4] tight coverage box in atlas UV. */
  uvRect: Float32Array
  /** [tile * 4] the same box in tile-local UV. */
  geoRect: Float32Array
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackPrisms(buf: ArrayBuffer): PrismAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== ATLAS || u[3] !== CELL) return null
  if (u[4] !== N_AZ_BAKED || u[5] !== N_SLAB || u[6] !== N_TOP || u[7] !== N_TILES) return null
  if (buf.byteLength !== HEADER_BYTES + ATLAS * ATLAS * 6) return null
  const f = new Float32Array(buf, 32, 5)
  return {
    atlasPx: ATLAS,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    slabDepth: new Float32Array(buf, 64, N_AZ_BAKED * N_SLAB),
    topHeight: new Float32Array(buf, 112, 4),
    uvRect: new Float32Array(buf, 128, N_TILES * 4),
    geoRect: new Float32Array(buf, 512, N_TILES * 4),
    albedo: new Uint8Array(buf, HEADER_BYTES, ATLAS * ATLAS * 4),
    normalOct: new Uint8Array(buf, HEADER_BYTES + ATLAS * ATLAS * 4, ATLAS * ATLAS * 2),
  }
}

function packPrisms(a: PrismAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.normalOct.byteLength)
  new Uint32Array(buf, 0, 8).set([MAGIC, VERSION, ATLAS, CELL, N_AZ_BAKED, N_SLAB, N_TOP, N_TILES])
  new Float32Array(buf, 32, 5).set([a.rXZ, a.y0, a.y1, a.cx, a.cz])
  new Float32Array(buf, 64, N_AZ_BAKED * N_SLAB).set(a.slabDepth)
  new Float32Array(buf, 112, 4).set(a.topHeight)
  new Float32Array(buf, 128, N_TILES * 4).set(a.uvRect)
  new Float32Array(buf, 512, N_TILES * 4).set(a.geoRect)
  new Uint8Array(buf, HEADER_BYTES, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + a.albedo.byteLength, a.normalOct.byteLength).set(a.normalOct)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the prism atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesPrisms(ctx: BakeCtx, speciesId: string): Promise<PrismAtlas> {
  const key = `prisms-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesPrisms(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackPrisms(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackPrisms(buf)
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

interface SliceSplit {
  /** Cut positions in normalized depth d (increasing, length n-1). */
  cuts: number[]
  /** Slice centroids in normalized depth d (length n). */
  centers: number[]
}

/**
 * Equal-mass split of a sample set into `n` slices, from a histogram over
 * [0,1]. Quantiles rather than equal thirds: every prism then carries the
 * same amount of geometry, and its centroid is where the card belongs.
 */
function equalMassSplit(hist: Float64Array, n: number): SliceSplit {
  const bins = hist.length
  let total = 0
  for (let i = 0; i < bins; i++) total += hist[i]!
  const cuts: number[] = []
  if (total <= 0) {
    for (let k = 1; k < n; k++) cuts.push(k / n)
  } else {
    let acc = 0
    let next = 1
    for (let i = 0; i < bins && next < n; i++) {
      acc += hist[i]!
      while (next < n && acc >= (total * next) / n) {
        cuts.push((i + 1) / bins)
        next++
      }
    }
    while (cuts.length < n - 1) cuts.push(1)
  }
  const edges = [0, ...cuts, 1]
  const centers: number[] = []
  for (let k = 0; k < n; k++) {
    const lo = edges[k]!
    const hi = edges[k + 1]!
    let mass = 0
    let sum = 0
    for (let i = 0; i < bins; i++) {
      const c = (i + 0.5) / bins
      if (c < lo || c >= hi) continue
      mass += hist[i]!
      sum += hist[i]! * c
    }
    centers.push(mass > 0 ? sum / mass : (lo + hi) / 2)
  }
  return { cuts, centers }
}

async function bakeSpeciesPrisms(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
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
  const sy = (by1 - by0) / 65535
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

  // --- prism cut planes: equal-mass quantiles of the depth distribution -----
  // Subsampled (quantiles do not need every vertex) so this stays ~ms even on
  // the 8.5M-vertex poa specimen.
  const BINS = 512
  const stride = Math.max(1, Math.floor(hdr.vertexCount / 400_000))
  const depthCuts: number[][] = []
  const slabDepth = new Float32Array(N_AZ_BAKED * N_SLAB)
  const hist = new Float64Array(BINS)
  for (let az = 0; az < N_AZ_BAKED; az++) {
    const a = (az * 2 * Math.PI) / N_AZ_VIEW
    const fx = Math.sin(a)
    const fz = Math.cos(a)
    hist.fill(0)
    for (let i = 0; i < hdr.vertexCount; i += stride) {
      const px = bx0 + verts[i * 8]! * sx - cx
      const pz = bz0 + verts[i * 8 + 2]! * sz - cz
      const d = 0.5 - (0.5 * (px * fx + pz * fz)) / rXZ
      hist[Math.min(BINS - 1, Math.max(0, Math.floor(d * BINS)))]! += 1
    }
    const split = equalMassSplit(hist, N_SLAB)
    depthCuts.push(split.cuts)
    // d -> metres along the bake axis (+ = toward the bake camera).
    for (let k = 0; k < N_SLAB; k++) slabDepth[az * N_SLAB + k] = (1 - 2 * split.centers[k]!) * rXZ
  }

  // --- height cut planes for the top view (d = 0 at the top) ---------------
  hist.fill(0)
  for (let i = 0; i < hdr.vertexCount; i += stride) {
    const py = by0 + verts[i * 8 + 1]! * sy
    const d = (y1 - py) / (y1 - y0)
    hist[Math.min(BINS - 1, Math.max(0, Math.floor(d * BINS)))]! += 1
  }
  const topSplit = equalMassSplit(hist, N_TOP)
  const topHeight = new Float32Array(4)
  for (let k = 0; k < N_TOP; k++) topHeight[k] = y1 - topSplit.centers[k]! * (y1 - y0)
  {
    let mass = 0
    let sum = 0
    for (let i = 0; i < BINS; i++) {
      mass += hist[i]!
      sum += hist[i]! * ((i + 0.5) / BINS)
    }
    topHeight[3] = y1 - (mass > 0 ? sum / mass : 0.5) * (y1 - y0)
  }

  // --- transient GPU resources ---------------------------------------------
  const rects = tileRects()
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

  // One uniform slot per rendered tile, addressed with dynamic offsets.
  const STRIDE = 256
  const rendered: number[] = []
  const scratch: number[] = []
  const pushView = (tile: number, fill: (o: number[]) => void): void => {
    const o = new Array<number>(STRIDE / 4).fill(0)
    fill(o)
    o[12] = cx
    o[13] = cz
    o[14] = y0
    o[15] = y1
    o[16] = rXZ
    o[20] = bx0
    o[21] = by0
    o[22] = bz0
    o[24] = bx1 - bx0
    o[25] = by1 - by0
    o[26] = bz1 - bz0
    scratch.push(...o)
    rendered.push(tile)
  }
  for (let az = 0; az < N_AZ_BAKED; az++) {
    const a = (az * 2 * Math.PI) / N_AZ_VIEW
    const edges = [0, ...depthCuts[az]!, 1]
    for (let k = 0; k < N_SLAB; k++) {
      pushView(tileSlab(az, k), (o) => {
        o[0] = Math.cos(a)
        o[2] = -Math.sin(a)
        o[5] = 1 // up = +Y (unused for side views)
        o[8] = Math.sin(a)
        o[10] = Math.cos(a)
        o[11] = 0 // not a top view
        o[17] = k === 0 ? -1e-4 : edges[k]!
        o[18] = k === N_SLAB - 1 ? 1 + 1e-4 : edges[k + 1]!
      })
    }
  }
  {
    const edges = [0, ...topSplit.cuts, 1]
    for (let k = 0; k < N_TOP; k++) {
      pushView(tileTopSlab(k), (o) => {
        o[0] = 1 // right = +X
        o[6] = -1 // card V axis = -Z
        o[9] = 1 // fwd = +Y
        o[11] = 1 // is_top
        o[17] = k === 0 ? -1e-4 : edges[k]!
        o[18] = k === N_TOP - 1 ? 1 + 1e-4 : edges[k + 1]!
      })
    }
  }

  const uni = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-uni`,
      size: STRIDE * rendered.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(uni, 0, new Float32Array(scratch))

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
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
  rendered.forEach((tile, i) => {
    const r = rects[tile]!
    pass.setViewport(r.x * SS, r.y * SS, r.size * SS, r.size * SS, 0, 1)
    pass.setScissorRect(r.x * SS, r.y * SS, r.size * SS, r.size * SS)
    pass.setBindGroup(0, bg, [i * STRIDE])
    pass.drawIndexed(indices.length)
  })
  pass.end()

  const bpr = BIG * 4
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: bpr * BIG, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
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

  const { albedo, normalOct, uvRect, geoRect } = postProcess(bigAlb, bigNrm, rects)
  return packPrisms({
    atlasPx: ATLAS,
    rXZ,
    y0,
    y1,
    cx,
    cz,
    slabDepth,
    topHeight,
    uvRect,
    geoRect,
    albedo,
    normalOct,
  })
}

/**
 * Resolve every final atlas texel from the supersampled prism renders.
 *
 * Vertical / horizontal prism tiles use EXCLUSIVE OWNERSHIP: for each final
 * texel the subsamples vote for the prism that is front-most there, the
 * winner takes the MERGED colour and coverage, and the losers are cleared.
 * The prisms therefore partition the billboard image exactly — identical
 * silhouette and colour, one draw per texel, three different depths.
 *
 * Merged tiles are resolved straight from the same subsamples (front-most
 * covered wins = what a z-buffer would have produced), so the far LOD is
 * guaranteed to match the union of the prisms and 9 bake passes are saved.
 * For view azimuths >= N_AZ_BAKED the subsamples are read MIRRORED (u
 * flipped, slab order reversed) and their mesh-frame normals get their
 * HORIZONTAL components negated — mirroring the geometry about the tile's U
 * axis and then re-facing the normal toward the opposite camera composes to
 * exactly `n.xz = -n.xz` for any azimuth, and unlike negating the whole
 * normal it leaves n.y alone, so the hemisphere ambient does not step across
 * the 180-degree seam.
 */
function resolveAtlas(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  rects: TileRect[],
): { albedo: Uint8Array<ArrayBuffer>; nrm: Float32Array; filled: Uint8Array } {
  const N = ATLAS
  const albedo = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const filled = new Uint8Array(N * N)
  const dec = (v: number): number => v / 127.5 - 1

  interface Sample {
    cov: number
    r: number
    g: number
    b: number
    nx: number
    ny: number
    nz: number
  }

  /**
   * Front-most covered subsample over `src` (already in draw order) at the
   * supersampled coordinate; returns the owning source index or -1.
   */
  const frontMost = (src: TileRect[], sx: number, sy: number, out: Sample): number => {
    for (let k = 0; k < src.length; k++) {
      const rect = src[k]!
      const s = ((rect.y * SS + sy) * BIG + rect.x * SS + sx) * 4
      if (bigAlb[s + 3] === 0) continue
      out.cov++
      out.r += bigAlb[s]!
      out.g += bigAlb[s + 1]!
      out.b += bigAlb[s + 2]!
      out.nx += dec(bigNrm[s]!)
      out.ny += dec(bigNrm[s + 1]!)
      out.nz += dec(bigNrm[s + 2]!)
      return k
    }
    return -1
  }

  const write = (rect: TileRect, x: number, y: number, acc: Sample, sub: number, flip: number): void => {
    const idx = (rect.y + y) * N + rect.x + x
    if (acc.cov === 0) {
      albedo[idx * 4 + 3] = 0
      return
    }
    albedo[idx * 4] = Math.round(acc.r / acc.cov)
    albedo[idx * 4 + 1] = Math.round(acc.g / acc.cov)
    albedo[idx * 4 + 2] = Math.round(acc.b / acc.cov)
    albedo[idx * 4 + 3] = Math.round((acc.cov * 255) / sub)
    nrm[idx * 3] = (acc.nx / acc.cov) * flip
    nrm[idx * 3 + 1] = acc.ny / acc.cov
    nrm[idx * 3 + 2] = (acc.nz / acc.cov) * flip
    filled[idx] = 1
  }

  const blank: Sample = { cov: 0, r: 0, g: 0, b: 0, nx: 0, ny: 0, nz: 0 }

  /** Exclusive-ownership resolve of one prism group (same tile size). */
  const resolveGroup = (group: TileRect[]): void => {
    const size = group[0]!.size
    const votes = new Int32Array(group.length)
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const acc: Sample = { cov: 0, r: 0, g: 0, b: 0, nx: 0, ny: 0, nz: 0 }
        votes.fill(0)
        for (let j = 0; j < SS; j++) {
          for (let i = 0; i < SS; i++) {
            const owner = frontMost(group, x * SS + i, y * SS + j, acc)
            if (owner >= 0) votes[owner]! += 1
          }
        }
        let winner = 0
        for (let k = 1; k < group.length; k++) if (votes[k]! > votes[winner]!) winner = k
        group.forEach((rect, k) => {
          write(rect, x, y, k === winner ? acc : blank, SS * SS, 1)
        })
      }
    }
  }

  for (let az = 0; az < N_AZ_BAKED; az++) {
    resolveGroup(Array.from({ length: N_SLAB }, (_, k) => rects[tileSlab(az, k)]!))
  }
  resolveGroup(Array.from({ length: N_TOP }, (_, k) => rects[tileTopSlab(k)]!))

  // --- merged (far LOD) tiles ------------------------------------------------
  for (let azView = 0; azView < N_AZ_VIEW; azView++) {
    const mirrored = azView >= N_AZ_BAKED
    const azBaked = azView % N_AZ_BAKED
    const order = Array.from({ length: N_SLAB }, (_, k) => (mirrored ? N_SLAB - 1 - k : k))
    const src = order.map((k) => rects[tileSlab(azBaked, k)]!)
    const dst = rects[tileMerged(azView)]!
    const sSize = src[0]!.size * SS
    const step = sSize / dst.size // 4: 768 subsamples across a 192px tile
    for (let y = 0; y < dst.size; y++) {
      for (let x = 0; x < dst.size; x++) {
        const acc: Sample = { cov: 0, r: 0, g: 0, b: 0, nx: 0, ny: 0, nz: 0 }
        for (let j = 0; j < step; j++) {
          for (let i = 0; i < step; i++) {
            const raw = x * step + i
            frontMost(src, mirrored ? sSize - 1 - raw : raw, y * step + j, acc)
          }
        }
        write(dst, x, y, acc, step * step, mirrored ? -1 : 1)
      }
    }
  }

  // Merged top view: same size as the height slabs, front-most (highest) wins.
  {
    const dst = rects[TILE_TOP_MERGED]!
    const src = Array.from({ length: N_TOP }, (_, k) => rects[tileTopSlab(k)]!)
    for (let y = 0; y < dst.size; y++) {
      for (let x = 0; x < dst.size; x++) {
        const acc: Sample = { cov: 0, r: 0, g: 0, b: 0, nx: 0, ny: 0, nz: 0 }
        for (let j = 0; j < SS; j++) {
          for (let i = 0; i < SS; i++) frontMost(src, x * SS + i, y * SS + j, acc)
        }
        write(dst, x, y, acc, SS * SS, 1)
      }
    }
  }

  return { albedo, nrm, filled }
}

/** Dilation (tile-clamped) + tight boxes over the resolved atlas. */
function postProcess(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  rects: TileRect[],
): {
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
  uvRect: Float32Array
  geoRect: Float32Array
} {
  const N = ATLAS
  const { albedo, nrm, filled } = resolveAtlas(bigAlb, bigNrm, rects)

  const { uvRect, geoRect } = tightBoxes(albedo, rects)

  // Dilate colour/normal into empty texels (alpha stays 0) so filtering never
  // blends toward background black. Clamped to each tile's own rect.
  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (const rect of rects) {
      const x0 = rect.x
      const y0 = rect.y
      const x1 = rect.x + rect.size - 1
      const y1 = rect.y + rect.size - 1
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const idx = y * N + x
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
            if (yy < y0 || yy > y1) continue
            for (let i = -1; i <= 1; i++) {
              if (i === 0 && j === 0) continue
              const xx = x + i
              if (xx < x0 || xx > x1) continue
              const nIdx = yy * N + xx
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

  const normalOct = new Uint8Array(N * N * 2)
  for (let i = 0; i < N * N; i++) {
    const s3 = i * 3
    normalOct.set(octEncode(nrm[s3]!, nrm[s3 + 1]!, nrm[s3 + 2]!), i * 2)
  }
  return { albedo, normalOct, uvRect, geoRect }
}

/**
 * Per-tile bounding box of the texels that actually carry coverage, returned
 * both as atlas UV (what to sample) and as tile-local UV (what geometry that
 * box maps to). The draw shrinks its quad to this, so a prism only rasterizes
 * where its slice of the plant exists — for a front slab seen corner-on that
 * is a small fraction of the full card.
 */
function tightBoxes(
  albedo: Uint8Array,
  rects: TileRect[],
): { uvRect: Float32Array; geoRect: Float32Array } {
  const N = ATLAS
  const thr = Math.round(COVER_EPS * 255)
  const uvRect = new Float32Array(N_TILES * 4)
  const geoRect = new Float32Array(N_TILES * 4)
  rects.forEach((rect, tile) => {
    const size = rect.size
    let x0 = size
    let y0 = size
    let x1 = -1
    let y1 = -1
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (albedo[((rect.y + y) * N + rect.x + x) * 4 + 3]! <= thr) continue
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
    if (x1 < 0) {
      // Empty prism (possible for a degenerate slice) — a zero-area box makes
      // the quad collapse instead of rasterizing a guaranteed-empty card.
      geoRect.set([0.5, 0.5, 0.5, 0.5], tile * 4)
      const u = (rect.x + size * 0.5) / N
      const v = (rect.y + size * 0.5) / N
      uvRect.set([u, v, u, v], tile * 4)
      return
    }
    // Room for bilinear taps at the box edge, kept inside the tile.
    const margin = Math.max(2, Math.round(size / 96))
    const clampT = (v: number): number => Math.min(size - 1.5, Math.max(1.5, v))
    const bx0 = clampT(x0 - margin)
    const by0 = clampT(y0 - margin)
    const bx1 = clampT(x1 + 1 + margin)
    const by1 = clampT(y1 + 1 + margin)
    geoRect.set([bx0 / size, by0 / size, bx1 / size, by1 / size], tile * 4)
    uvRect.set([(rect.x + bx0) / N, (rect.y + by0) / N, (rect.x + bx1) / N, (rect.y + by1) / N], tile * 4)
  })
  return { uvRect, geoRect }
}

/** Octahedral encode, y-primary — exact inverse of `oct_decode_tile` in prisms.wgsl. */
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
