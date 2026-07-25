import bakeShaderSrc from './shaders/bake.wgsl'
import mipShaderSrc from './shaders/mip.wgsl'
import mossBakeShaderSrc from './shaders/moss-bake.wgsl'
import { assetUrl, bakedArtifact, commitBake, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake one species into a 3x3 "coverage card" atlas: 8 azimuthal orthographic
 * views (elevation 0) + 1 top-down view, each a TILE x TILE cell. Two atlases:
 *   - albedo: PREMULTIPLIED authored color in rgb, coverage in alpha
 *   - normal: premultiplied (n*0.5+0.5), plant-local, faceforwarded per view
 * Premultiplied storage matters: the runtime builds the full mip chain, and
 * box-filtered premultiplied texels are exactly the ensemble statistics we
 * want at distance — alpha becomes fractional COVERAGE, rgb the mean radiance
 * of covered area. The runtime then *realizes* that coverage per pixel with a
 * hashed alpha test instead of blending.
 */

export const TILE = 256
export const GRID = 3
export const ATLAS = TILE * GRID // 768
export const N_AZIMUTH = 8
export const MIP_LEVELS = 7 // 768 .. 12

const MAGIC = 'GCSTIP1\0'
const HEADER_BYTES = 96
const BLOB = ATLAS * ATLAS * 4

export interface CardMeta {
  /** Card/frame center in mesh-local space (offset from the plant origin). */
  center: [number, number, number]
  /** Ortho half-size of every view frame (card half-size in metres). */
  rCard: number
  /** Ortho half-depth used at bake (informational). */
  rDepth: number
  /** Mean authored vertex color (carpet mean-field albedo). */
  meanColor: [number, number, number]
  /** Mean coverage of the top-down view (carpet density statistics). */
  coverTop: number
}

export interface BakedCards {
  meta: CardMeta
  albedo: Uint8Array // ATLAS*ATLAS*4, mip level 0
  normal: Uint8Array // ATLAS*ATLAS*4, mip level 0
}

export function packCards(b: BakedCards): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + BLOB * 2)
  new Uint8Array(buf, 0, 8).set(new TextEncoder().encode(MAGIC))
  const u = new Uint32Array(buf, 8, 4)
  u[0] = 1 // version
  u[1] = ATLAS
  u[2] = TILE
  u[3] = N_AZIMUTH
  const f = new Float32Array(buf, 24, 12)
  f.set(b.meta.center, 0)
  f[3] = b.meta.rCard
  f[4] = b.meta.rDepth
  f.set(b.meta.meanColor, 5)
  f[8] = b.meta.coverTop
  new Uint8Array(buf, HEADER_BYTES, BLOB).set(b.albedo)
  new Uint8Array(buf, HEADER_BYTES + BLOB, BLOB).set(b.normal)
  return buf
}

export function unpackCards(buf: ArrayBuffer): BakedCards | null {
  if (buf.byteLength !== HEADER_BYTES + BLOB * 2) return null
  const magic = new TextDecoder('latin1').decode(new Uint8Array(buf, 0, 8))
  if (magic !== MAGIC) return null
  const u = new Uint32Array(buf, 8, 4)
  if (u[0] !== 1 || u[1] !== ATLAS || u[2] !== TILE || u[3] !== N_AZIMUTH) return null
  const f = new Float32Array(buf, 24, 12)
  return {
    meta: {
      center: [f[0]!, f[1]!, f[2]!],
      rCard: f[3]!,
      rDepth: f[4]!,
      meanColor: [f[5]!, f[6]!, f[7]!],
      coverTop: f[8]!,
    },
    albedo: new Uint8Array(buf, HEADER_BYTES, BLOB),
    normal: new Uint8Array(buf, HEADER_BYTES + BLOB, BLOB),
  }
}

type V3 = [number, number, number]

/**
 * Mean visible canopy color of the top-down tile — the premultiplied rgb sum
 * over the alpha sum, i.e. exactly the color the ensemble converges to from
 * above. Drives the mean-field carpet.
 */
export function topMeanColor(albedo: Uint8Array): V3 {
  const tx0 = (N_AZIMUTH % GRID) * TILE
  const ty0 = Math.floor(N_AZIMUTH / GRID) * TILE
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  for (let y = 0; y < TILE; y += 2) {
    for (let x = 0; x < TILE; x += 2) {
      const o = ((ty0 + y) * ATLAS + tx0 + x) * 4
      r += albedo[o]!
      g += albedo[o + 1]!
      b += albedo[o + 2]!
      a += albedo[o + 3]!
    }
  }
  if (a < 1) return [0.2, 0.3, 0.12]
  return [r / a, g / a, b / a]
}

/**
 * View basis — MUST match view_basis() in cards.wgsl:
 *  azimuth i: v=(cos, 0, sin), right=(v.z, 0, -v.x), up=(0,1,0)
 *  top      : v=(0,1,0), right=(1,0,0), up=(0,0,-1)
 */
function viewBasis(i: number): { v: V3; right: V3; up: V3 } {
  if (i === N_AZIMUTH) return { v: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1] }
  const phi = (i * 2 * Math.PI) / N_AZIMUTH
  const v: V3 = [Math.cos(phi), 0, Math.sin(phi)]
  return { v, right: [v[2], 0, -v[0]], up: [0, 1, 0] }
}

/** GPU-render the 9-view atlas pair and read it back (one-time, transient). */
export async function bakeCards(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const ex = bmax[0] - bmin[0]
  const ey = bmax[1] - bmin[1]
  const ez = bmax[2] - bmin[2]
  // Half-size covering the plant from any azimuth (XZ diagonal) and vertically.
  const rCard = 0.5 * Math.max(Math.hypot(ex, ez), ey)
  const rDepth = 0.5 * Math.hypot(ex, ey, ez) + 1e-3

  // Mean authored color (carpet mean field) — strided CPU pass over vertices.
  const meanColor: V3 = [0, 0, 0]
  let n = 0
  for (let i = 0; i < mesh.header.vertexCount; i += 97) {
    const base = i * 8 + 3
    meanColor[0] += mesh.vertices[base]! / 65535
    meanColor[1] += mesh.vertices[base + 1]! / 65535
    meanColor[2] += mesh.vertices[base + 2]! / 65535
    n++
  }
  meanColor[0] /= n
  meanColor[1] /= n
  meanColor[2] /= n

  // --- transient GPU resources ---
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-transient' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-transient' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (name: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/bake-${name}`,
        size: [ATLAS, ATLAS],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-transient' },
    )
  const albTex = mkTarget('alb')
  const nrmTex = mkTarget('nrm')
  const depthTex = ctx.res.createTexture(
    { label: `${ctx.id}/bake-depth`, size: [ATLAS, ATLAS], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT },
    { tag: 'bake-transient' },
  )

  // Per-view uniforms, 256-aligned for dynamic offsets. Struct = 96 bytes.
  const STRIDE = 256
  const N_VIEWS = N_AZIMUTH + 1
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-transient' },
  )
  const scratch = new Float32Array((STRIDE * N_VIEWS) / 4)
  for (let i = 0; i < N_VIEWS; i++) {
    const { v, right, up } = viewBasis(i)
    const o = (i * STRIDE) / 4
    scratch.set(center, o); scratch[o + 3] = rCard
    scratch.set(right, o + 4); scratch[o + 7] = rDepth
    scratch.set(up, o + 8)
    scratch.set(v, o + 12)
    scratch.set(bmin, o + 16)
    scratch.set(bmax, o + 20)
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 96 } }],
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
      { view: nrmTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  for (let i = 0; i < N_VIEWS; i++) {
    const tx = (i % GRID) * TILE
    const ty = Math.floor(i / GRID) * TILE
    pass.setViewport(tx, ty, TILE, TILE, 0, 1)
    pass.setScissorRect(tx, ty, TILE, TILE)
    pass.setBindGroup(0, bg, [i * STRIDE])
    pass.drawIndexed(indices.length)
  }
  pass.end()

  const bpr = ATLAS * 4 // 3072, multiple of 256
  const mkRb = (name: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/bake-rb-${name}`, size: BLOB, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-transient' },
    )
  const rbAlb = mkRb('alb')
  const rbNrm = mkRb('nrm')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const albedo = new Uint8Array(BLOB)
  const normal = new Uint8Array(BLOB)
  albedo.set(new Uint8Array(rbAlb.getMappedRange()))
  normal.set(new Uint8Array(rbNrm.getMappedRange()))
  rbAlb.unmap()
  rbNrm.unmap()

  // Mean coverage of the top-down tile (view 8 = grid cell 2,2).
  let cover = 0
  const tx0 = (N_AZIMUTH % GRID) * TILE
  const ty0 = Math.floor(N_AZIMUTH / GRID) * TILE
  for (let y = 0; y < TILE; y += 2) {
    for (let x = 0; x < TILE; x += 2) {
      cover += albedo[((ty0 + y) * ATLAS + tx0 + x) * 4 + 3]!
    }
  }
  cover /= 255 * (TILE / 2) * (TILE / 2)

  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()

  return packCards({ meta: { center, rCard, rDepth, meanColor, coverTop: cover }, albedo, normal })
}

// ---------------------------------------------------------------------------
// Carpet tile bake (Sphagnum): ONE top-down capture of the tile's own period,
// with a height field, instead of nine views of a 7cm-tall mat.
// ---------------------------------------------------------------------------

/** Stored tile resolution. 512 texels over a 0.18m tile = ~2800 px/m. */
export const MOSS_TEX = 512
/** Rendered at 2x and box-filtered down — thin branch leaves alias badly. */
const MOSS_SS = MOSS_TEX * 2
/** 512 .. 4 */
export const MOSS_MIPS = 8

const MOSS_MAGIC = 'GCMOSS1\0'
const MOSS_HEADER_BYTES = 64
const MOSS_BLOB = MOSS_TEX * MOSS_TEX * 4

export interface MossMeta {
  /** Periodic tile size (m) at scale 1 — the mesh's own period. */
  tileM: number
  /** Height range mapped into the height channel (m, mesh frame). */
  hMin: number
  hRange: number
  /** Mean coverage of the tile at level 0 (how closed the mat is). */
  coverMean: number
  /** Coverage-weighted mean albedo — the colour the mat converges to. */
  meanColor: [number, number, number]
}

export interface BakedMoss {
  meta: MossMeta
  /** rgb = albedo*cov, a = cov. */
  albedo: Uint8Array
  /** rgb = (n*0.5+0.5)*cov, a = height_norm*cov. */
  nrmh: Uint8Array
}

export function packMoss(b: BakedMoss): ArrayBuffer {
  const buf = new ArrayBuffer(MOSS_HEADER_BYTES + MOSS_BLOB * 2)
  new Uint8Array(buf, 0, 8).set(new TextEncoder().encode(MOSS_MAGIC))
  const u = new Uint32Array(buf, 8, 2)
  u[0] = 1 // version
  u[1] = MOSS_TEX
  const f = new Float32Array(buf, 16, 8)
  f[0] = b.meta.tileM
  f[1] = b.meta.hMin
  f[2] = b.meta.hRange
  f[3] = b.meta.coverMean
  f[4] = b.meta.meanColor[0]
  f[5] = b.meta.meanColor[1]
  f[6] = b.meta.meanColor[2]
  new Uint8Array(buf, MOSS_HEADER_BYTES, MOSS_BLOB).set(b.albedo)
  new Uint8Array(buf, MOSS_HEADER_BYTES + MOSS_BLOB, MOSS_BLOB).set(b.nrmh)
  return buf
}

export function unpackMoss(buf: ArrayBuffer): BakedMoss | null {
  if (buf.byteLength !== MOSS_HEADER_BYTES + MOSS_BLOB * 2) return null
  const magic = new TextDecoder('latin1').decode(new Uint8Array(buf, 0, 8))
  if (magic !== MOSS_MAGIC) return null
  const u = new Uint32Array(buf, 8, 2)
  if (u[0] !== 1 || u[1] !== MOSS_TEX) return null
  const f = new Float32Array(buf, 16, 8)
  return {
    meta: {
      tileM: f[0]!,
      hMin: f[1]!,
      hRange: f[2]!,
      coverMean: f[3]!,
      meanColor: [f[4]!, f[5]!, f[6]!],
    },
    albedo: new Uint8Array(buf, MOSS_HEADER_BYTES, MOSS_BLOB),
    nrmh: new Uint8Array(buf, MOSS_HEADER_BYTES + MOSS_BLOB, MOSS_BLOB),
  }
}

/** GPU-render the periodic tile (albedo+coverage, normal+height) and read back. */
export async function bakeMossTile(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const tileM = mesh.header.tileSize[0]
  if (!(tileM > 0)) throw new Error(`${ctx.id}: carpet bake needs a periodic mesh (tileSize = 0)`)
  const hMin = bmin[1]
  const hRange = Math.max(bmax[1] - bmin[1], 1e-4)

  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/moss-verts`, size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-transient' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/moss-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-transient' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  // Two mip levels: level 0 is the 2x supersampled render, level 1 the stored
  // 512 tile (a 2x2 box filter over premultiplied values = the right average).
  const mkTarget = (name: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/moss-${name}`,
        size: [MOSS_SS, MOSS_SS],
        format: 'rgba8unorm',
        mipLevelCount: 2,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-transient' },
    )
  const albTex = mkTarget('alb')
  const nrmTex = mkTarget('nrmh')
  const depthTex = ctx.res.createTexture(
    { label: `${ctx.id}/moss-depth`, size: [MOSS_SS, MOSS_SS], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT },
    { tag: 'bake-transient' },
  )

  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/moss-uni`, size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-transient' },
  )
  const scratch = new Float32Array(12)
  scratch.set(bmin, 0)
  scratch[3] = tileM
  scratch.set(bmax, 4)
  scratch[7] = hMin
  scratch[8] = mesh.header.tileOrigin[0]
  scratch[9] = mesh.header.tileOrigin[1]
  scratch[10] = hRange
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/moss-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  })
  const bg = device.createBindGroup({ label: `${ctx.id}/moss-bg`, layout: bgl, entries: [{ binding: 0, resource: { buffer: uni } }] })
  const module = ctx.shaders.module(mossBakeShaderSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/moss-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/moss-pl`, bindGroupLayouts: [bgl] }),
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

  const enc = device.createCommandEncoder({ label: `${ctx.id}/moss-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/moss-pass`,
    colorAttachments: [
      { view: albTex.createView({ baseMipLevel: 0, mipLevelCount: 1 }), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nrmTex.createView({ baseMipLevel: 0, mipLevelCount: 1 }), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  pass.setBindGroup(0, bg)
  // 9 instances = the 3x3 periodic wrap; only the centre one is fully inside.
  pass.drawIndexed(indices.length, 9)
  pass.end()

  // 2x -> 1x box filter into level 1.
  const mipModule = ctx.shaders.module(mipShaderSrc)
  const mipSampler = device.createSampler({ label: `${ctx.id}/moss-mip-samp`, magFilter: 'linear', minFilter: 'linear' })
  const mipPipeline = device.createRenderPipeline({
    label: `${ctx.id}/moss-mip`,
    layout: 'auto',
    vertex: { module: mipModule, entryPoint: 'vs' },
    fragment: { module: mipModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  })
  for (const tex of [albTex, nrmTex]) {
    const dpass = enc.beginRenderPass({
      colorAttachments: [{ view: tex.createView({ baseMipLevel: 1, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }],
    })
    dpass.setPipeline(mipPipeline)
    dpass.setBindGroup(
      0,
      device.createBindGroup({
        label: `${ctx.id}/moss-mip-bg`,
        layout: mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tex.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
          { binding: 1, resource: mipSampler },
        ],
      }),
    )
    dpass.draw(3)
    dpass.end()
  }

  const mkRb = (name: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/moss-rb-${name}`, size: MOSS_BLOB, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-transient' },
    )
  const rbAlb = mkRb('alb')
  const rbNrm = mkRb('nrmh')
  enc.copyTextureToBuffer(
    { texture: albTex, mipLevel: 1 },
    { buffer: rbAlb, bytesPerRow: MOSS_TEX * 4, rowsPerImage: MOSS_TEX },
    [MOSS_TEX, MOSS_TEX],
  )
  enc.copyTextureToBuffer(
    { texture: nrmTex, mipLevel: 1 },
    { buffer: rbNrm, bytesPerRow: MOSS_TEX * 4, rowsPerImage: MOSS_TEX },
    [MOSS_TEX, MOSS_TEX],
  )
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const albedo = new Uint8Array(MOSS_BLOB)
  const nrmh = new Uint8Array(MOSS_BLOB)
  albedo.set(new Uint8Array(rbAlb.getMappedRange()))
  nrmh.set(new Uint8Array(rbNrm.getMappedRange()))
  rbAlb.unmap()
  rbNrm.unmap()

  // Coverage-weighted mean colour + mean coverage (strided).
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  let n = 0
  for (let y = 0; y < MOSS_TEX; y += 2) {
    for (let x = 0; x < MOSS_TEX; x += 2) {
      const o = (y * MOSS_TEX + x) * 4
      r += albedo[o]!
      g += albedo[o + 1]!
      b += albedo[o + 2]!
      a += albedo[o + 3]!
      n++
    }
  }
  const meanColor: V3 = a < 1 ? [0.2, 0.3, 0.12] : [r / a, g / a, b / a]

  for (const res of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) res.destroy()

  return packMoss({
    meta: { tileM, hMin, hRange, coverMean: a / (255 * n), meanColor },
    albedo,
    nrmh,
  })
}

/**
 * Inverse CDF of the tile's cushion-top height, sampled at (j / (n-1)) of the
 * COVERED area — i.e. `q[j]` is the normalized height below which j/(n-1) of the
 * mat's visible surface lies. Computed at load (a 512² pass, ~5ms) rather than
 * baked, so the artifact format stays put.
 *
 * This is what makes a shell stack worth anything on real moss: the cushion top
 * occupies a narrow band near the top of the mesh's height range (2nd..98th
 * percentile = 4.6cm..8.5cm of a 9.3cm range), so evenly spaced shells put
 * three coincident planes in empty air below the cushions and express the
 * relief with one. Equal-AREA shells put every shell where surface actually is.
 */
export function heightQuantiles(b: BakedMoss, n = 17): number[] {
  const BINS = 256
  const hist = new Float64Array(BINS)
  let total = 0
  for (let i = 0; i < MOSS_TEX * MOSS_TEX; i++) {
    const cov = b.albedo[i * 4 + 3]!
    if (cov < 8) continue
    // Stored premultiplied: recover the height, then weight by coverage.
    const bin = Math.min(BINS - 1, Math.max(0, Math.round((b.nrmh[i * 4 + 3]! / cov) * (BINS - 1))))
    const w = cov / 255
    hist[bin] = hist[bin]! + w
    total += w
  }
  const out: number[] = []
  if (total <= 0) {
    for (let j = 0; j < n; j++) out.push(j / (n - 1))
    return out
  }
  let acc = 0
  let bin = 0
  for (let j = 0; j < n; j++) {
    const want = (j / (n - 1)) * total
    while (bin < BINS - 1 && acc + hist[bin]! < want) {
      acc += hist[bin]!
      bin++
    }
    out.push(bin / (BINS - 1))
  }
  return out
}

/** Committed carpet tile, else bake on the GPU and best-effort commit. */
export async function loadOrBakeMoss(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  meshId: string,
  bakeVersion: number,
): Promise<BakedMoss> {
  const key = `${speciesId}-carpet-v${bakeVersion}`
  const fresh = async (): Promise<ArrayBuffer> => {
    const mesh = await ctx.meshes.load(meshId)
    const packed = await bakeMossTile(ctx, mesh)
    commitBake(ctx.id, key, packed).catch((err: unknown) => console.warn(`[${ctx.id}] commitBake failed:`, err))
    return packed
  }
  const buf = await bakedArtifact({ expId: ctx.id, key }, fresh)
  const unpacked = unpackMoss(buf)
  if (unpacked) return unpacked
  // Stale/poisoned artifact (the dev server answers missing files with
  // index.html at 200) — bake it again rather than trusting the bytes.
  const rebaked = unpackMoss(await fresh())
  if (!rebaked) throw new Error(`${ctx.id}: freshly baked carpet tile failed validation`)
  return rebaked
}

/**
 * Load a species' baked cards: committed file (validated — the dev server
 * answers missing files with index.html, so blind trust would poison us),
 * else bake on the GPU and best-effort commit for next time.
 */
export async function loadOrBakeCards(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  meshId: string,
  bakeVersion: number,
): Promise<BakedCards> {
  const key = `${speciesId}-v${bakeVersion}`
  try {
    const res = await fetch(assetUrl(`/mesh/baked/${ctx.id}/${key}.bin`))
    if (res.ok) {
      const unpacked = unpackCards(await res.arrayBuffer())
      if (unpacked) return unpacked
    }
  } catch {
    /* fall through to bake */
  }
  const mesh = await ctx.meshes.load(meshId)
  const packed = await bakeCards(ctx, mesh)
  commitBake(ctx.id, key, packed).catch((err: unknown) => console.warn(`[${ctx.id}] commitBake failed:`, err))
  const unpacked = unpackCards(packed)
  if (!unpacked) throw new Error(`${ctx.id}: freshly baked cards failed validation`)
  return unpacked
}
