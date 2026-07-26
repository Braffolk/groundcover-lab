import bakeShaderSrc from './shaders/cards_bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Card-atlas bake for the billboard baseline.
 *
 * Per species, one 3x3 atlas of TILE px tiles rendered orthographically from
 * the raw GCMESH1 mesh: tiles 0..7 are side views at 45deg azimuth steps
 * (camera on the horizon), tile 8 is the straight-down top view. Rendered at
 * 2x and downsampled with coverage weighting, then the albedo/normals are
 * dilated a few texels into empty space so bilinear/mip sampling never pulls
 * in background black.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'BBC1', u32 version, u32 atlasPx, u32 tilePx, u32 nSide, u32 pad
 *   f32 rXZ, y0, y1, cx, cz            (capture box, unit scale, metres)
 *   ...zeros to byte 64
 *   u8[atlasPx^2*4]  albedo rgba8 (straight color, a = coverage)
 *   u8[atlasPx^2*2]  normals, oct-encoded rg8 (mesh frame, y-primary)
 */

export const TILE = 512
export const GRID = 3
export const ATLAS = TILE * GRID
export const N_SIDE = 8
const SS = 2
const BIG = ATLAS * SS
const MAGIC = 0x31434242 // 'BBC1'
const VERSION = 1
const HEADER_BYTES = 64
const DILATE_PASSES = 6

export interface CardAtlas {
  atlasPx: number
  tilePx: number
  nSide: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackCards(buf: ArrayBuffer): CardAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 6)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const atlasPx = u[2]!
  const tilePx = u[3]!
  const nSide = u[4]!
  const expected = HEADER_BYTES + atlasPx * atlasPx * 6
  if (atlasPx !== TILE * GRID || buf.byteLength !== expected) return null
  const f = new Float32Array(buf, 24, 5)
  return {
    atlasPx,
    tilePx,
    nSide,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    albedo: new Uint8Array(buf, HEADER_BYTES, atlasPx * atlasPx * 4),
    normalOct: new Uint8Array(buf, HEADER_BYTES + atlasPx * atlasPx * 4, atlasPx * atlasPx * 2),
  }
}

function packCards(a: CardAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.normalOct.byteLength)
  const u = new Uint32Array(buf, 0, 6)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.atlasPx
  u[3] = a.tilePx
  u[4] = a.nSide
  const f = new Float32Array(buf, 24, 5)
  f[0] = a.rXZ
  f[1] = a.y0
  f[2] = a.y1
  f[3] = a.cx
  f[4] = a.cz
  new Uint8Array(buf, HEADER_BYTES, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + a.albedo.byteLength, a.normalOct.byteLength).set(a.normalOct)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the card atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesCards(ctx: BakeCtx, speciesId: string): Promise<CardAtlas> {
  const key = `cards-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesCards(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackCards(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackCards(buf)
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

async function bakeSpeciesCards(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
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

  // Per-view uniforms in one buffer with dynamic offsets (28 floats used).
  const STRIDE = 256
  const N_VIEWS = N_SIDE + 1
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_VIEWS)
  for (let k = 0; k < N_VIEWS; k++) {
    const o = (k * STRIDE) / 4
    const isTop = k === N_SIDE
    if (isTop) {
      scratch.set([1, 0, 0], o) // right = +X
      scratch.set([0, 0, -1], o + 4) // card V axis = -Z
      scratch.set([0, 1, 0, 1], o + 8) // fwd = +Y, w = is_top
    } else {
      const a = (k * 2 * Math.PI) / N_SIDE
      scratch.set([Math.cos(a), 0, -Math.sin(a)], o)
      scratch.set([0, 1, 0], o + 4)
      scratch.set([Math.sin(a), 0, Math.cos(a), 0], o + 8)
    }
    scratch.set([cx, cz, y0, y1], o + 12)
    scratch.set([rXZ, 0, 0, 0], o + 16)
    scratch.set([bx0, by0, bz0, 0], o + 20)
    scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 24)
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
  for (let k = 0; k < N_VIEWS; k++) {
    const col = k % GRID
    const row = Math.floor(k / GRID)
    pass.setViewport(col * bigTile, row * bigTile, bigTile, bigTile, 0, 1)
    pass.setScissorRect(col * bigTile, row * bigTile, bigTile, bigTile)
    pass.setBindGroup(0, bg, [k * STRIDE])
    pass.drawIndexed(indices.length)
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

  const { albedo, normalOct } = postProcess(bigAlb, bigNrm)
  return packCards({ atlasPx: ATLAS, tilePx: TILE, nSide: N_SIDE, rXZ, y0, y1, cx, cz, albedo, normalOct })
}

/** Coverage-weighted 2x downsample, then dilation (tile-clamped), then pack. */
function postProcess(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
): { albedo: Uint8Array<ArrayBuffer>; normalOct: Uint8Array<ArrayBuffer> } {
  const N = ATLAS
  const albedo = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const filled = new Uint8Array(N * N)

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          const s = ((y * 2 + j) * BIG + (x * 2 + i)) * 4
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
      const d = (y * N + x) * 4
      if (aSum > 0) {
        albedo[d] = Math.round(r / aSum)
        albedo[d + 1] = Math.round(g / aSum)
        albedo[d + 2] = Math.round(b / aSum)
        albedo[d + 3] = Math.round(aSum / 4)
        const dn = (y * N + x) * 3
        nrm[dn] = nx
        nrm[dn + 1] = ny
        nrm[dn + 2] = nz
        filled[y * N + x] = 1
      }
    }
  }

  // Dilate color/normal into empty texels (alpha stays 0) so filtering never
  // blends toward background black. Clamped to tile boundaries.
  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (let y = 0; y < N; y++) {
      const ty0 = Math.floor(y / TILE) * TILE
      const ty1 = ty0 + TILE - 1
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (cur[idx]! !== 0) continue
        const tx0 = Math.floor(x / TILE) * TILE
        const tx1 = tx0 + TILE - 1
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
    cur = next
  }

  const normalOct = new Uint8Array(N * N * 2)
  for (let i = 0; i < N * N; i++) {
    const s3 = i * 3
    normalOct.set(octEncode(nrm[s3]!, nrm[s3 + 1]!, nrm[s3 + 2]!), i * 2)
  }
  return { albedo, normalOct }
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
