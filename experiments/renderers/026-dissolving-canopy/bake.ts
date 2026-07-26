import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake for the dissolving canopy.
 *
 * Per species, ONE artifact holding:
 *
 *  A. A node atlas. The mesh's TRIANGLES are partitioned by centroid into the
 *     four xz quadrants of the plant (nodes 1..4); node 0 is the whole plant.
 *     Each node is captured orthographically from AZ azimuths into a tile.
 *     Because the partition is exact, the four quadrant tiles reassemble the
 *     plant — but they are placed at four different world positions at draw
 *     time, which is where the parallax, self-occlusion and view-dependent
 *     silhouette come from. Reprojection error scales with a node's radius, so
 *     quarter-plant nodes are ~2x more faithful off-axis than one whole-plant
 *     card.
 *
 *     Channels are split by how much resolution they actually need:
 *       coverage  r8    full res  — this is the alpha-tested silhouette
 *       albedo    rgba8 half res  — colour is low frequency (a = coverage, so
 *                                   its mip chain stays coverage-weighted)
 *       normals   rg8   quarter   — oct-encoded, smooth by nature
 *     That buys ~500 texels/m of silhouette for a third of the bytes a plain
 *     rgba8 atlas would need.
 *
 *  B. One straight-down capture of the whole plant (albedo+coverage, oct
 *     normal + height fraction). main.ts composites the active stand's species
 *     mix out of these into the single tileable canopy texture the continuous
 *     shell is drawn with.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'DCV1', u32 version, u32 nodes, u32 az, u32 tileW, u32 tileH,
 *   u32 topPx, u32 pad, f32 plantH, f32 rootHalfW, ...zeros to byte 64
 *   f32[nodes*8]  node table: cx, cy, cz, halfW, y0, y1, azOffset, pad
 *   u8[COV_W*COV_H]      coverage
 *   u8[ALB_W*ALB_H*4]    albedo + coverage
 *   u8[NRM_W*NRM_H*2]    oct normals
 *   u8[TOP_PX^2*4]       top view albedo + coverage
 *   u8[TOP_PX^2*4]       top view oct normal (rg) + height (b)
 */

export const AZ = 8
export const NODES = 5 // 0 = whole plant, 1..4 = xz quadrants
export const TILE_W = 256
export const TILE_H = 512
export const COV_W = TILE_W * NODES // 1280 (256B-aligned rows for writeTexture)
export const COV_H = TILE_H * AZ // 4096
export const ALB_W = COV_W >> 1
export const ALB_H = COV_H >> 1
export const NRM_W = COV_W >> 2
export const NRM_H = COV_H >> 2
export const COV_MIPS = 6 // tile 256 -> 8 px
export const ALB_MIPS = 5 // tile 128 -> 8 px
export const NRM_MIPS = 4 // tile  64 -> 8 px
export const TOP_PX = 160
export const TOP_MIPS = 4

const SS = 2 // supersample factor of the bake render
const AZ_CHUNK = 2 // azimuth rows rendered per chunk (caps transient VRAM)
const WS = COV_W * SS
const HS = TILE_H * SS * AZ_CHUNK
const MAGIC = 0x31564344 // 'DCV1'
const VERSION = 2
const HEADER_BYTES = 64
const NODE_STRIDE = 8
const BOX_MARGIN = 1.03
const DILATE_ALBEDO = 3
const DILATE_NORMAL = 2
const DILATE_TOP = 4

export interface NodeDesc {
  /** Node centre in the mesh frame (the sprite quad's anchor). */
  cx: number
  cy: number
  cz: number
  /** Horizontal support radius — the quad's half width at scale 1. */
  halfW: number
  /** Vertical span of the tile in mesh-frame metres. */
  y0: number
  y1: number
  /** Per-node rotation of the azimuth grid (0 — see the note in bakeSpecies). */
  azOffset: number
}

export interface CanopyBake {
  nodes: NodeDesc[]
  /** Whole-plant height (m at scale 1) — wind + shell height reference. */
  plantH: number
  /** Whole-plant horizontal support radius (m at scale 1). */
  rootHalfW: number
  coverage: Uint8Array<ArrayBuffer>
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
  topAlbedo: Uint8Array<ArrayBuffer>
  topAux: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

const COV_BYTES = COV_W * COV_H
const ALB_BYTES = ALB_W * ALB_H * 4
const NRM_BYTES = NRM_W * NRM_H * 2
const TOP_BYTES = TOP_PX * TOP_PX * 4
const TABLE_BYTES = NODES * NODE_STRIDE * 4
const PAYLOAD_OFFSET = HEADER_BYTES + TABLE_BYTES
const TOTAL_BYTES = PAYLOAD_OFFSET + COV_BYTES + ALB_BYTES + NRM_BYTES + 2 * TOP_BYTES

export function unpackBake(buf: ArrayBuffer): CanopyBake | null {
  if (buf.byteLength !== TOTAL_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== NODES || u[3] !== AZ || u[4] !== TILE_W || u[5] !== TILE_H || u[6] !== TOP_PX) return null
  const f = new Float32Array(buf, 32, 2)
  const table = new Float32Array(buf, HEADER_BYTES, NODES * NODE_STRIDE)
  const nodes: NodeDesc[] = []
  for (let i = 0; i < NODES; i++) {
    const o = i * NODE_STRIDE
    nodes.push({
      cx: table[o]!,
      cy: table[o + 1]!,
      cz: table[o + 2]!,
      halfW: table[o + 3]!,
      y0: table[o + 4]!,
      y1: table[o + 5]!,
      azOffset: table[o + 6]!,
    })
  }
  let off = PAYLOAD_OFFSET
  const take = (n: number): Uint8Array<ArrayBuffer> => {
    const a = new Uint8Array(buf, off, n) as Uint8Array<ArrayBuffer>
    off += n
    return a
  }
  return {
    nodes,
    plantH: f[0]!,
    rootHalfW: f[1]!,
    coverage: take(COV_BYTES),
    albedo: take(ALB_BYTES),
    normal: take(NRM_BYTES),
    topAlbedo: take(TOP_BYTES),
    topAux: take(TOP_BYTES),
  }
}

function packBake(b: CanopyBake): ArrayBuffer {
  const buf = new ArrayBuffer(TOTAL_BYTES)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = NODES
  u[3] = AZ
  u[4] = TILE_W
  u[5] = TILE_H
  u[6] = TOP_PX
  const f = new Float32Array(buf, 32, 2)
  f[0] = b.plantH
  f[1] = b.rootHalfW
  const table = new Float32Array(buf, HEADER_BYTES, NODES * NODE_STRIDE)
  b.nodes.forEach((n, i) => {
    const o = i * NODE_STRIDE
    table[o] = n.cx
    table[o + 1] = n.cy
    table[o + 2] = n.cz
    table[o + 3] = n.halfW
    table[o + 4] = n.y0
    table[o + 5] = n.y1
    table[o + 6] = n.azOffset
  })
  let off = PAYLOAD_OFFSET
  const put = (src: Uint8Array): void => {
    new Uint8Array(buf, off, src.byteLength).set(src)
    off += src.byteLength
  }
  put(b.coverage)
  put(b.albedo)
  put(b.normal)
  put(b.topAlbedo)
  put(b.topAux)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake this species' artifact. Every
 * result is magic+size validated: the dev server answers missing
 * /mesh/baked files with the SPA index.html at status 200, which would
 * otherwise poison both the cache and the committed-file path.
 */
export async function loadSpeciesBake(ctx: BakeCtx, speciesId: string): Promise<CanopyBake> {
  const key = `canopy-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpecies(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let data = unpackBake(buf)
  if (!data) {
    buf = await runBake()
    data = unpackBake(buf)
    if (!data) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return data
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

async function bakeSpecies(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax

  // Dequantize positions once — the partition and the node bounds walk them
  // several times and the u16 decode is the expensive part.
  const vcount = hdr.vertexCount
  const pos = new Float32Array(vcount * 3)
  {
    const v = mesh.vertices
    const sx = (bx1 - bx0) / 65535
    const sy = (by1 - by0) / 65535
    const sz = (bz1 - bz0) / 65535
    for (let i = 0; i < vcount; i++) {
      pos[i * 3] = bx0 + v[i * 8]! * sx
      pos[i * 3 + 1] = by0 + v[i * 8 + 1]! * sy
      pos[i * 3 + 2] = bz0 + v[i * 8 + 2]! * sz
    }
  }

  // Plant centre: a periodic community tile is centred on its tile, a finite
  // specimen on its bounding box.
  const periodic = hdr.tileSize[0] > 0 && hdr.tileSize[1] > 0
  const cx = periodic ? hdr.tileOrigin[0] + hdr.tileSize[0] / 2 : (bx0 + bx1) / 2
  const cz = periodic ? hdr.tileOrigin[1] + hdr.tileSize[1] / 2 : (bz0 + bz1) / 2

  // --- triangle partition into xz quadrants --------------------------------
  const tcount = hdr.triangleCount
  const tri = mesh.triangles
  const quad = new Uint8Array(tcount)
  const counts = [0, 0, 0, 0]
  for (let t = 0; t < tcount; t++) {
    const a = tri[t * 4]! * 3
    const b = tri[t * 4 + 1]! * 3
    const c = tri[t * 4 + 2]! * 3
    const mx = (pos[a]! + pos[b]! + pos[c]!) / 3
    const mz = (pos[a + 2]! + pos[b + 2]! + pos[c + 2]!) / 3
    const q = (mx > cx ? 1 : 0) + (mz > cz ? 2 : 0)
    quad[t] = q
    counts[q]!++
  }
  const starts = [0, counts[0]!, counts[0]! + counts[1]!, counts[0]! + counts[1]! + counts[2]!]
  const ranges = starts.map((s, i) => ({ first: s * 3, count: counts[i]! * 3 }))
  const indices = new Uint32Array(tcount * 3)
  {
    const cursor = starts.slice()
    for (let t = 0; t < tcount; t++) {
      const q = quad[t]!
      const w = cursor[q]!++ * 3
      indices[w] = tri[t * 4]!
      indices[w + 1] = tri[t * 4 + 1]!
      indices[w + 2] = tri[t * 4 + 2]!
    }
  }

  // --- node table -----------------------------------------------------------
  const nodes: NodeDesc[] = []
  const rootY0 = Math.min(0, by0)
  const rootBox = boxOf(pos, indices, 0, tcount * 3, cx, cz, rootY0, by1)
  nodes.push({ ...rootBox, azOffset: 0 })
  for (let q = 0; q < 4; q++) {
    const r = ranges[q]!
    const box =
      r.count > 0
        ? boxOfAuto(pos, indices, r.first, r.count)
        : { cx, cy: (rootY0 + by1) / 2, cz, halfW: rootBox.halfW, y0: rootY0, y1: by1 }
    // azOffset stays 0: rotating each node's azimuth grid spread the view-switch
    // pop nicely, but it also meant two adjacent quadrants reprojected their
    // shared blades from azimuths up to 9deg apart, which breaks those blades
    // visibly at close range. A coherent pop is the cheaper artifact.
    nodes.push({ ...box, azOffset: 0 })
  }
  const plantH = rootBox.y1 - rootBox.y0
  const rootHalfW = rootBox.halfW

  // --- transient GPU resources ---------------------------------------------
  const vbuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-verts`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (label: string, w: number, h: number): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('bake-albedo', WS, HS)
  const auxTex = mkTarget('bake-aux', WS, HS)
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [WS, HS],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // Per-draw uniforms with dynamic offsets: NODES*AZ side views + 1 top view.
  const STRIDE = 256
  const N_VIEWS = NODES * AZ + 1
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_VIEWS)
  const writeView = (
    slot: number,
    right: [number, number, number],
    up: [number, number, number],
    fwd: [number, number, number],
    isTop: number,
    node: NodeDesc,
  ): void => {
    const o = (slot * STRIDE) / 4
    scratch.set([right[0], right[1], right[2], isTop], o)
    scratch.set([up[0], up[1], up[2], 0], o + 4)
    scratch.set([fwd[0], fwd[1], fwd[2], 0], o + 8)
    scratch.set([node.cx, node.cy, node.cz, node.halfW], o + 12)
    scratch.set([node.y0, node.y1, 0, 0], o + 16)
    scratch.set([bx0, by0, bz0, 0], o + 20)
    scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 24)
  }
  for (let n = 0; n < NODES; n++) {
    const node = nodes[n]!
    for (let k = 0; k < AZ; k++) {
      const a = (k * 2 * Math.PI) / AZ + node.azOffset
      // fwd points toward the bake camera; right is fwd rotated -90deg in xz.
      writeView(
        n * AZ + k,
        [Math.cos(a), 0, -Math.sin(a)],
        [0, 1, 0],
        [Math.sin(a), 0, Math.cos(a)],
        0,
        node,
      )
    }
  }
  writeView(NODES * AZ, [1, 0, 0], [0, 0, -1], [0, 1, 0], 1, nodes[0]!)
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

  const bpr = WS * 4
  const mkReadback = (label: string, size: number): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo', bpr * HS)
  const rbAux = mkReadback('bake-rb-aux', bpr * HS)

  // --- output accumulators --------------------------------------------------
  const coverage = new Uint8Array(COV_BYTES)
  const albedo = new Uint8Array(ALB_BYTES)
  const nrmAcc = new Float32Array(NRM_W * NRM_H * 3)
  const nrmFilled = new Uint8Array(NRM_W * NRM_H)

  const ssTileW = TILE_W * SS
  const ssTileH = TILE_H * SS
  for (let a0 = 0; a0 < AZ; a0 += AZ_CHUNK) {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc-${a0}` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-pass-${a0}`,
      colorAttachments: [
        { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: auxTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
    for (let n = 0; n < NODES; n++) {
      const r = n === 0 ? { first: 0, count: tcount * 3 } : ranges[n - 1]!
      if (r.count === 0) continue
      for (let k = a0; k < a0 + AZ_CHUNK; k++) {
        const vx = n * ssTileW
        const vy = (k - a0) * ssTileH
        pass.setViewport(vx, vy, ssTileW, ssTileH, 0, 1)
        pass.setScissorRect(vx, vy, ssTileW, ssTileH)
        pass.setBindGroup(0, bg, [(n * AZ + k) * STRIDE])
        pass.drawIndexed(r.count, 1, r.first)
      }
    }
    pass.end()
    enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: HS }, [WS, HS])
    enc.copyTextureToBuffer({ texture: auxTex }, { buffer: rbAux, bytesPerRow: bpr, rowsPerImage: HS }, [WS, HS])
    device.queue.submit([enc.finish()])

    await rbAlb.mapAsync(GPUMapMode.READ)
    await rbAux.mapAsync(GPUMapMode.READ)
    const src = new Uint8Array(rbAlb.getMappedRange())
    const aux = new Uint8Array(rbAux.getMappedRange())
    reduceChunk(src, aux, a0, coverage, albedo, nrmAcc, nrmFilled)
    rbAlb.unmap()
    rbAux.unmap()
  }

  // --- top view (feeds the shell's tileable canopy composite) --------------
  const topSs = TOP_PX * SS
  const topAlbTex = mkTarget('bake-top-albedo', topSs, topSs)
  const topAuxTex = mkTarget('bake-top-aux', topSs, topSs)
  const topDepth = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-top-depth`,
      size: [topSs, topSs],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )
  const topBpr = topSs * 4
  const rbTopAlb = mkReadback('bake-rb-top-albedo', topBpr * topSs)
  const rbTopAux = mkReadback('bake-rb-top-aux', topBpr * topSs)
  {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-top-enc` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-top-pass`,
      colorAttachments: [
        { view: topAlbTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: topAuxTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: topDepth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    pass.setBindGroup(0, bg, [NODES * AZ * STRIDE])
    pass.drawIndexed(tcount * 3)
    pass.end()
    enc.copyTextureToBuffer(
      { texture: topAlbTex },
      { buffer: rbTopAlb, bytesPerRow: topBpr, rowsPerImage: topSs },
      [topSs, topSs],
    )
    enc.copyTextureToBuffer(
      { texture: topAuxTex },
      { buffer: rbTopAux, bytesPerRow: topBpr, rowsPerImage: topSs },
      [topSs, topSs],
    )
    device.queue.submit([enc.finish()])
    await rbTopAlb.mapAsync(GPUMapMode.READ)
    await rbTopAux.mapAsync(GPUMapMode.READ)
  }
  const { topAlbedo, topAux } = reduceTop(
    new Uint8Array(rbTopAlb.getMappedRange()),
    new Uint8Array(rbTopAux.getMappedRange()),
  )
  rbTopAlb.unmap()
  rbTopAux.unmap()

  for (const r of [
    vbuf,
    ibuf,
    albTex,
    auxTex,
    depthTex,
    uni,
    rbAlb,
    rbAux,
    topAlbTex,
    topAuxTex,
    topDepth,
    rbTopAlb,
    rbTopAux,
  ]) {
    r.destroy()
  }

  dilateRgba(albedo, ALB_W, ALB_H, TILE_W >> 1, TILE_H >> 1, DILATE_ALBEDO)
  const normal = finishNormals(nrmAcc, nrmFilled)

  return packBake({ nodes, plantH, rootHalfW, coverage, albedo, normal, topAlbedo, topAux })
}

/** Node box with an explicit vertical range (used for the whole-plant node). */
function boxOf(
  pos: Float32Array,
  indices: Uint32Array,
  first: number,
  count: number,
  cx: number,
  cz: number,
  y0: number,
  y1: number,
): Omit<NodeDesc, 'azOffset'> {
  let r2 = 0
  for (let i = first; i < first + count; i++) {
    const v = indices[i]! * 3
    const dx = pos[v]! - cx
    const dz = pos[v + 2]! - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  const half = ((y1 - y0) / 2) * BOX_MARGIN
  const mid = (y0 + y1) / 2
  return { cx, cy: mid, cz, halfW: Math.sqrt(r2) * BOX_MARGIN + 1e-3, y0: mid - half, y1: mid + half }
}

/** Node box derived from the node's own contents (used for the quadrants). */
function boxOfAuto(pos: Float32Array, indices: Uint32Array, first: number, count: number): Omit<NodeDesc, 'azOffset'> {
  let x0 = Infinity
  let x1 = -Infinity
  let z0 = Infinity
  let z1 = -Infinity
  let y0 = Infinity
  let y1 = -Infinity
  for (let i = first; i < first + count; i++) {
    const v = indices[i]! * 3
    const x = pos[v]!
    const y = pos[v + 1]!
    const z = pos[v + 2]!
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
    if (z < z0) z0 = z
    if (z > z1) z1 = z
  }
  const cx = (x0 + x1) / 2
  const cz = (z0 + z1) / 2
  return boxOf(pos, indices, first, count, cx, cz, Math.min(0, y0), y1)
}

/**
 * Fold one chunk of the supersampled render into the three output resolutions:
 * coverage 2x2, albedo 4x4 (coverage-weighted), normals 8x8 (coverage-weighted).
 */
function reduceChunk(
  src: Uint8Array,
  aux: Uint8Array,
  a0: number,
  coverage: Uint8Array,
  albedo: Uint8Array,
  nrmAcc: Float32Array,
  nrmFilled: Uint8Array,
): void {
  const covRows = TILE_H * AZ_CHUNK
  const covRow0 = a0 * TILE_H
  for (let y = 0; y < covRows; y++) {
    const dRow = (covRow0 + y) * COV_W
    for (let x = 0; x < COV_W; x++) {
      const s = ((y * 2) * WS + x * 2) * 4
      const a = src[s + 3]! + src[s + 7]! + src[s + WS * 4 + 3]! + src[s + WS * 4 + 7]!
      coverage[dRow + x] = (a + 2) >> 2
    }
  }

  const albRows = covRows >> 1
  const albRow0 = covRow0 >> 1
  for (let y = 0; y < albRows; y++) {
    const dRow = (albRow0 + y) * ALB_W
    for (let x = 0; x < ALB_W; x++) {
      let as = 0
      let r = 0
      let g = 0
      let b = 0
      for (let j = 0; j < 4; j++) {
        const row = (y * 4 + j) * WS
        for (let i = 0; i < 4; i++) {
          const s = (row + x * 4 + i) * 4
          const a = src[s + 3]!
          if (a === 0) continue
          as += a
          r += src[s]! * a
          g += src[s + 1]! * a
          b += src[s + 2]! * a
        }
      }
      const d = (dRow + x) * 4
      if (as > 0) {
        albedo[d] = Math.round(r / as)
        albedo[d + 1] = Math.round(g / as)
        albedo[d + 2] = Math.round(b / as)
        albedo[d + 3] = Math.round(as / 16)
      }
    }
  }

  const nrmRows = covRows >> 2
  const nrmRow0 = covRow0 >> 2
  for (let y = 0; y < nrmRows; y++) {
    const dRow = (nrmRow0 + y) * NRM_W
    for (let x = 0; x < NRM_W; x++) {
      let as = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < 8; j++) {
        const row = (y * 8 + j) * WS
        for (let i = 0; i < 8; i++) {
          const s = (row + x * 8 + i) * 4
          const a = src[s + 3]!
          if (a === 0) continue
          as += a
          nx += (aux[s]! / 127.5 - 1) * a
          ny += (aux[s + 1]! / 127.5 - 1) * a
          nz += (aux[s + 2]! / 127.5 - 1) * a
        }
      }
      if (as > 0) {
        const d = (dRow + x) * 3
        nrmAcc[d] = nx
        nrmAcc[d + 1] = ny
        nrmAcc[d + 2] = nz
        nrmFilled[dRow + x] = 1
      }
    }
  }
}

/** 2x downsample of the top view into albedo+coverage and octN+height. */
function reduceTop(
  src: Uint8Array,
  aux: Uint8Array,
): { topAlbedo: Uint8Array<ArrayBuffer>; topAux: Uint8Array<ArrayBuffer> } {
  const N = TOP_PX
  const S = TOP_PX * SS
  const topAlbedo = new Uint8Array(N * N * 4)
  const topAux = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const filled = new Uint8Array(N * N)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let as = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let h = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * S + x * SS + i) * 4
          const a = src[s + 3]!
          if (a === 0) continue
          as += a
          r += src[s]! * a
          g += src[s + 1]! * a
          b += src[s + 2]! * a
          nx += (aux[s]! / 127.5 - 1) * a
          ny += (aux[s + 1]! / 127.5 - 1) * a
          nz += (aux[s + 2]! / 127.5 - 1) * a
          h += aux[s + 3]! * a
        }
      }
      const i2 = y * N + x
      if (as === 0) continue
      topAlbedo[i2 * 4] = Math.round(r / as)
      topAlbedo[i2 * 4 + 1] = Math.round(g / as)
      topAlbedo[i2 * 4 + 2] = Math.round(b / as)
      topAlbedo[i2 * 4 + 3] = Math.round(as / (SS * SS))
      topAux[i2 * 4 + 2] = Math.round(h / as)
      nrm[i2 * 3] = nx
      nrm[i2 * 3 + 1] = ny
      nrm[i2 * 3 + 2] = nz
      filled[i2] = 1
    }
  }
  // Dilate colour/normal/height into empty texels so the composite's bilinear
  // stamping never pulls background black across a silhouette.
  dilateRgba(topAlbedo, N, N, N, N, DILATE_TOP)
  dilateTopAux(topAux, nrm, filled, N, DILATE_TOP)
  for (let i = 0; i < N * N; i++) {
    const [u, v] = octEncode(nrm[i * 3]!, nrm[i * 3 + 1]!, nrm[i * 3 + 2]!)
    topAux[i * 4] = u
    topAux[i * 4 + 1] = v
  }
  return { topAlbedo: topAlbedo as Uint8Array<ArrayBuffer>, topAux: topAux as Uint8Array<ArrayBuffer> }
}

/** Spread rgb into alpha==0 texels (alpha itself stays 0), clamped per tile. */
function dilateRgba(buf: Uint8Array, w: number, h: number, tw: number, th: number, passes: number): void {
  let filled = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) filled[i] = buf[i * 4 + 3]! > 0 ? 1 : 0
  for (let p = 0; p < passes; p++) {
    const next = filled.slice()
    for (let y = 0; y < h; y++) {
      const ty0 = Math.floor(y / th) * th
      const ty1 = ty0 + th - 1
      for (let x = 0; x < w; x++) {
        const idx = y * w + x
        if (filled[idx] !== 0) continue
        const tx0 = Math.floor(x / tw) * tw
        const tx1 = tx0 + tw - 1
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const n = yy * w + xx
            if (filled[n] === 0) continue
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

/** Same dilation for the top view's normal accumulator + height byte. */
function dilateTopAux(aux: Uint8Array, nrm: Float32Array, filled0: Uint8Array, n: number, passes: number): void {
  let filled = filled0.slice()
  for (let p = 0; p < passes; p++) {
    const next = filled.slice()
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const idx = y * n + x
        if (filled[idx] !== 0) continue
        let count = 0
        let nx = 0
        let ny = 0
        let nz = 0
        let h = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= n) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < 0 || xx >= n) continue
            const m = yy * n + xx
            if (filled[m] === 0) continue
            count++
            nx += nrm[m * 3]!
            ny += nrm[m * 3 + 1]!
            nz += nrm[m * 3 + 2]!
            h += aux[m * 4 + 2]!
          }
        }
        if (count === 0) continue
        nrm[idx * 3] = nx / count
        nrm[idx * 3 + 1] = ny / count
        nrm[idx * 3 + 2] = nz / count
        aux[idx * 4 + 2] = Math.round(h / count)
        next[idx] = 1
      }
    }
    filled = next
  }
}

/** Dilate the node normals, then oct-encode them into the rg8 atlas. */
function finishNormals(nrmAcc: Float32Array, filled0: Uint8Array): Uint8Array<ArrayBuffer> {
  const w = NRM_W
  const h = NRM_H
  const tw = TILE_W >> 2
  const th = TILE_H >> 2
  let filled = filled0.slice()
  for (let p = 0; p < DILATE_NORMAL; p++) {
    const next = filled.slice()
    for (let y = 0; y < h; y++) {
      const ty0 = Math.floor(y / th) * th
      const ty1 = ty0 + th - 1
      for (let x = 0; x < w; x++) {
        const idx = y * w + x
        if (filled[idx] !== 0) continue
        const tx0 = Math.floor(x / tw) * tw
        const tx1 = tx0 + tw - 1
        let count = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const m = yy * w + xx
            if (filled[m] === 0) continue
            count++
            nx += nrmAcc[m * 3]!
            ny += nrmAcc[m * 3 + 1]!
            nz += nrmAcc[m * 3 + 2]!
          }
        }
        if (count === 0) continue
        nrmAcc[idx * 3] = nx / count
        nrmAcc[idx * 3 + 1] = ny / count
        nrmAcc[idx * 3 + 2] = nz / count
        next[idx] = 1
      }
    }
    filled = next
  }
  const out = new Uint8Array(w * h * 2)
  for (let i = 0; i < w * h; i++) {
    const [u, v] = octEncode(nrmAcc[i * 3]!, nrmAcc[i * 3 + 1]!, nrmAcc[i * 3 + 2]!)
    out[i * 2] = u
    out[i * 2 + 1] = v
  }
  return out as Uint8Array<ArrayBuffer>
}

// ---------------------------------------------------------------------------
// Carpet tile bake — a SECOND, separate artifact, loaded only for stand entries
// that are carpets (stand carpet_div > 0, i.e. the periodic Sphagnum tiles).
//
// Why not more of the node atlas: the node atlas is 5 nodes x 8 azimuths of
// SIDE views at 256x512 per tile. For a 0.24m wide, 0.09m tall cushion that
// spends 512 texels of vertical resolution on 9cm (~5700 texels/m up, ~850
// across) and 40 tiles on a silhouette a mat never shows — the budget lands
// almost entirely where a low carpet has nothing to say. A mat's whole
// appearance is its surface from above, so a carpet species trades the entire
// 14.8 MiB node atlas (never uploaded — see main.ts) for ONE top-down capture
// of the tile square at 512px, i.e. 2844 texels/m across the 0.18m tile: 6x the
// density of the same tile inside the 160px whole-mesh top capture, and ~1.9x
// the billboard baseline's cropped top view.
//
// The capture is genuinely SEAMLESS: the source mesh is a periodic community
// tile whose geometry overflows its 0.18m period (0.21 x 0.23m of bounds), so
// the bake draws the mesh 9 times at the 3x3 wrap offsets and keeps only the
// tile square. What leaves one edge comes back on the other, exactly as it does
// in the real mat.
//
// Channels: rgb albedo + coverage, oct normal, and the HEIGHT of the topmost
// surface — the last one is what lets the renderer express the cushion's 3.3cm
// of capitulum relief (parallax, cavity occlusion, self-shadowing) instead of
// drawing a flat picture of moss.
// ---------------------------------------------------------------------------

export const CT_PX = 512
export const CT_MIPS = 7 // 512 -> 8 px
const CT_SS = 2
const CT_MAGIC = 0x31544344 // 'DCT1'
const CT_VERSION = 2
const CT_HEADER = 64
const CT_PLANE_BYTES = CT_PX * CT_PX * 4
const CT_TOTAL = CT_HEADER + 2 * CT_PLANE_BYTES
const CT_DILATE = 4

export interface CarpetTileBake {
  /** Periodic tile size (m at scale 1) — matches stand_table.footprint_m. */
  tileM: number
  /** Vertical range of the capture, mesh frame (m at scale 1). */
  y0: number
  y1: number
  /** Coverage-weighted mean of the height channel: the mat's mean surface. */
  planeFrac: number
  /** Mean coverage of the tile (diagnostics — a closed mat is ~0.8+). */
  meanCov: number
  /** rgba8: rgb = albedo, a = coverage. */
  albedo: Uint8Array<ArrayBuffer>
  /**
   * rgba8: rg = oct normal, b = height fraction of [y0,y1] of the top surface,
   * a = cavity occlusion of that surface (1 = open sky, 0 = fully enclosed).
   *
   * The occlusion is BAKED rather than derived from `b` in the shader for one
   * specific reason: a term computed from the height is a non-linear function
   * of it, so it does NOT survive mip filtering — as the chain flattens the
   * height toward its mean, the derived occlusion collapses to "open" and the
   * mat gets brighter the further away it is. That is the classic
   * distance-dependent drift CLAUDE.md warns about. Occlusion stored per texel
   * mip-averages linearly, so the far field keeps the mat's real mean darkness.
   */
  aux: Uint8Array<ArrayBuffer>
}

function unpackCarpet(buf: ArrayBuffer): CarpetTileBake | null {
  if (buf.byteLength !== CT_TOTAL) return null
  const u = new Uint32Array(buf, 0, 4)
  if (u[0] !== CT_MAGIC || u[1] !== CT_VERSION || u[2] !== CT_PX) return null
  const f = new Float32Array(buf, 16, 5)
  return {
    tileM: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    planeFrac: f[3]!,
    meanCov: f[4]!,
    albedo: new Uint8Array(buf, CT_HEADER, CT_PLANE_BYTES) as Uint8Array<ArrayBuffer>,
    aux: new Uint8Array(buf, CT_HEADER + CT_PLANE_BYTES, CT_PLANE_BYTES) as Uint8Array<ArrayBuffer>,
  }
}

function packCarpet(t: CarpetTileBake): ArrayBuffer {
  const buf = new ArrayBuffer(CT_TOTAL)
  const u = new Uint32Array(buf, 0, 4)
  u[0] = CT_MAGIC
  u[1] = CT_VERSION
  u[2] = CT_PX
  const f = new Float32Array(buf, 16, 5)
  f[0] = t.tileM
  f[1] = t.y0
  f[2] = t.y1
  f[3] = t.planeFrac
  f[4] = t.meanCov
  new Uint8Array(buf, CT_HEADER, CT_PLANE_BYTES).set(t.albedo)
  new Uint8Array(buf, CT_HEADER + CT_PLANE_BYTES, CT_PLANE_BYTES).set(t.aux)
  return buf
}

/** Load (OPFS / committed) or bake the carpet tile artifact for one species. */
export async function loadCarpetTile(ctx: BakeCtx, speciesId: string): Promise<CarpetTileBake> {
  const key = `carpettile-v${CT_VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeCarpetTile(ctx, mesh)
  }
  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let data = unpackCarpet(buf)
  if (!data) {
    buf = await runBake()
    data = unpackCarpet(buf)
    if (!data) throw new Error(`[${ctx.id}] carpet tile bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return data
}

async function bakeCarpetTile(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const tileM = hdr.tileSize[0]
  if (!(tileM > 0) || hdr.tileSize[1] !== tileM) {
    throw new Error(`[${ctx.id}] carpet tile bake needs a square periodic mesh (tileSize=${hdr.tileSize})`)
  }
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = hdr.tileOrigin[0] + tileM / 2
  const cz = hdr.tileOrigin[1] + tileM / 2
  const y0 = Math.min(0, by0)
  const y1 = by1

  const indices = mesh.indices()
  const vbuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/ct-verts`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/ct-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  // One view: straight down over exactly the tile square. b_min.w carries the
  // wrap period, which the bake shader turns into the 3x3 instance offsets.
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/ct-uni`, size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const view = new Float32Array(64)
  view.set([1, 0, 0, 1], 0) // right axis (mesh +x), w = 1 -> top view
  view.set([0, 0, -1, 0], 4) // up axis: framebuffer row 0 is the tile's min z
  view.set([0, 1, 0, 0], 8) // toward the bake camera
  view.set([cx, (y0 + y1) / 2, cz, tileM / 2], 12)
  view.set([y0, y1, 0, 0], 16)
  view.set([bx0, by0, bz0, tileM], 20)
  view.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], 24)
  device.queue.writeBuffer(uni, 0, view)

  const S = CT_PX * CT_SS
  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [S, S],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('ct-albedo')
  const auxTex = mkTarget('ct-aux')
  const depthTex = ctx.res.createTexture(
    { label: `${ctx.id}/ct-depth`, size: [S, S], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT },
    { tag: 'bake-scratch' },
  )

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/ct-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', minBindingSize: 112 } }],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/ct-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
  })
  const module = ctx.shaders.module(bakeShaderSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/ct-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/ct-pl`, bindGroupLayouts: [bgl] }),
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

  const bpr = S * 4
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: bpr * S, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('ct-rb-albedo')
  const rbAux = mkReadback('ct-rb-aux')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/ct-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/ct-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: auxTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  // 9 wrapped copies -> the tile square receives its neighbours' overflow, so
  // the capture tiles exactly (0.24m of geometry inside a 0.18m period).
  pass.drawIndexed(indices.length, 9)
  pass.end()
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: S }, [S, S])
  enc.copyTextureToBuffer({ texture: auxTex }, { buffer: rbAux, bytesPerRow: bpr, rowsPerImage: S }, [S, S])
  device.queue.submit([enc.finish()])
  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbAux.mapAsync(GPUMapMode.READ)
  const reduced = reduceCarpet(new Uint8Array(rbAlb.getMappedRange()), new Uint8Array(rbAux.getMappedRange()))
  rbAlb.unmap()
  rbAux.unmap()
  for (const r of [vbuf, ibuf, uni, albTex, auxTex, depthTex, rbAlb, rbAux]) r.destroy()
  bakeCavityAo(reduced.aux, tileM, y1 - y0)

  return packCarpet({ tileM, y0, y1, ...reduced })
}

/** Supersample reduce of the tile capture, then dilate colour/normal/height. */
function reduceCarpet(
  src: Uint8Array,
  aux: Uint8Array,
): Pick<CarpetTileBake, 'albedo' | 'aux' | 'planeFrac' | 'meanCov'> {
  const N = CT_PX
  const S = CT_PX * CT_SS
  const albedo = new Uint8Array(N * N * 4)
  const out = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const filled = new Uint8Array(N * N)
  let covSum = 0
  let hSum = 0
  let hDen = 0
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let as = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let h = 0
      for (let j = 0; j < CT_SS; j++) {
        for (let i = 0; i < CT_SS; i++) {
          const s = ((y * CT_SS + j) * S + x * CT_SS + i) * 4
          const a = src[s + 3]!
          if (a === 0) continue
          as += a
          r += src[s]! * a
          g += src[s + 1]! * a
          b += src[s + 2]! * a
          nx += (aux[s]! / 127.5 - 1) * a
          ny += (aux[s + 1]! / 127.5 - 1) * a
          nz += (aux[s + 2]! / 127.5 - 1) * a
          h += aux[s + 3]! * a
        }
      }
      const i2 = y * N + x
      covSum += as / (CT_SS * CT_SS * 255)
      if (as === 0) continue
      albedo[i2 * 4] = Math.round(r / as)
      albedo[i2 * 4 + 1] = Math.round(g / as)
      albedo[i2 * 4 + 2] = Math.round(b / as)
      albedo[i2 * 4 + 3] = Math.round(as / (CT_SS * CT_SS))
      out[i2 * 4 + 2] = Math.round(h / as)
      hSum += (h / as / 255) * as
      hDen += as
      nrm[i2 * 3] = nx
      nrm[i2 * 3 + 1] = ny
      nrm[i2 * 3 + 2] = nz
      filled[i2] = 1
    }
  }
  dilateRgba(albedo, N, N, N, N, CT_DILATE)
  dilateTopAux(out, nrm, filled, N, CT_DILATE)
  for (let i = 0; i < N * N; i++) {
    const [u, v] = octEncode(nrm[i * 3]!, nrm[i * 3 + 1]!, nrm[i * 3 + 2]!)
    out[i * 4] = u
    out[i * 4 + 1] = v
  }
  return {
    albedo: albedo as Uint8Array<ArrayBuffer>,
    aux: out as Uint8Array<ArrayBuffer>,
    planeFrac: hDen > 0 ? hSum / hDen : 0.7,
    meanCov: covSum / (N * N),
  }
}

/**
 * Cavity occlusion of the captured surface, written into aux.a in place.
 *
 * Horizon obscurance over the tile's own height channel: for each texel, how
 * far the surface rises around it, sampled on 8 azimuths at three radii that
 * bracket the capitulum scale (a capitulum is ~1cm, the relief 3.3cm). This is
 * what makes a cushion read as a mass of separate heads rather than a printed
 * texture — the crevices between capitula go dark because they genuinely see
 * less sky. Torus-wrapped, because the tile is periodic.
 */
function bakeCavityAo(aux: Uint8Array, tileM: number, spanM: number): void {
  const N = CT_PX
  const mPerTexel = tileM / N
  const radii = [8, 20, 44] // ~2.8mm, 7mm, 15mm at 512px / 0.18m
  const dirs = 8
  const cos: number[] = []
  const sin: number[] = []
  for (let k = 0; k < dirs; k++) {
    const a = (k * 2 * Math.PI) / dirs
    cos.push(Math.cos(a))
    sin.push(Math.sin(a))
  }
  const out = new Uint8Array(N * N)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const hc = aux[(y * N + x) * 4 + 2]! / 255
      let occ = 0
      for (let k = 0; k < dirs; k++) {
        for (const r of radii) {
          const sx = (((x + Math.round(cos[k]! * r)) % N) + N) % N
          const sy = (((y + Math.round(sin[k]! * r)) % N) + N) % N
          const dh = (aux[(sy * N + sx) * 4 + 2]! / 255 - hc) * spanM
          // tan(elevation) of the neighbour, clamped to a 45-degree horizon.
          occ += Math.min(1, Math.max(0, dh / (r * mPerTexel)))
        }
      }
      out[y * N + x] = Math.round(255 * Math.max(0, 1 - occ / (dirs * radii.length)))
    }
  }
  for (let i = 0; i < N * N; i++) aux[i * 4 + 3] = out[i]!
}

/** Octahedral encode, y-primary — exact inverse of the decode in gcmesh.ts. */
export function octEncode(x: number, y: number, z: number): [number, number] {
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
