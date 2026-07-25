import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Relief-card bake.
 *
 * Per species one 3x3 atlas is rendered orthographically from the raw GCMESH1
 * mesh: tiles 0..7 are side views at 45deg azimuth steps (camera on the
 * horizon), tile 8 is the straight-down top view. Beyond the usual
 * albedo/coverage/normal an extra channel is kept — the FRONT-SURFACE
 * DISTANCE of the visible plant along the view axis. That is what lets the
 * runtime bend the card into the plant's actual frontal relief.
 *
 * Three artifacts come out of it, at three deliberately different rates:
 *   ALBEDO  (atlasPx^2, rgba8)  rgb = colour, a = coverage.  Silhouette detail
 *                               — needs the full resolution.
 *   SURF    (surfPx^2,  rgba8)  rg = oct normal, b = front distance,
 *                               a = baked sky occlusion. Half resolution —
 *                               lighting and occlusion are low frequency by
 *                               nature, and halving it buys back the bytes the
 *                               extra channels cost.
 *   DISP    (dispPx^2,  r8)     hole-free, heavily smoothed front distance.
 *                               32 texels per tile — this is what the VERTEX
 *                               shader reads to curve the card, so it must be
 *                               defined everywhere (including outside the
 *                               silhouette) and free of high-frequency noise.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'CRV1', u32 version, u32 atlasPx, u32 tilePx,
 *   u32 nSide, u32 surfPx, u32 dispPx, u32 pad
 *   f32 rXZ, y0, y1, cx, cz             (capture box, unit scale, metres)
 *   ...zeros to byte 64
 *   u8[atlasPx^2*4]  albedo rgba8
 *   u8[surfPx^2*4]   oct normal rg + front distance b + sky occlusion a
 *   u8[dispPx^2]     smoothed front distance
 */

export const TILE = 512
export const GRID = 3
export const ATLAS = TILE * GRID // 1536
export const SURF_TILE = 256
export const SURF = SURF_TILE * GRID // 768
export const DISP_TILE = 32
export const DISP = DISP_TILE * GRID // 96
export const N_SIDE = 8

const SS = 2
const BIG = ATLAS * SS // 3072
const BIG_TILE = TILE * SS // 1024
const MAGIC = 0x31565243 // 'CRV1'
const VERSION = 2
const HEADER_BYTES = 64
const DILATE_ALBEDO = 6
const DILATE_SURF = 5
/**
 * Sky-occlusion falloff per unit of accumulated coverage above a texel. A
 * typical column inside these clumps integrates to ~6 units of coverage, so
 * the constant is picked to put that at roughly the middle of the ramp — at
 * 0.55 the whole interior pinned to the floor and the term degenerated into a
 * flat dimmer instead of a shading cue.
 */
const AO_K = 0.12
const AO_FLOOR = 0.35

export interface ReliefAtlas {
  atlasPx: number
  tilePx: number
  surfPx: number
  dispPx: number
  nSide: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  albedo: Uint8Array<ArrayBuffer>
  surf: Uint8Array<ArrayBuffer>
  disp: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackRelief(buf: ArrayBuffer): ReliefAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const atlasPx = u[2]!
  const tilePx = u[3]!
  const nSide = u[4]!
  const surfPx = u[5]!
  const dispPx = u[6]!
  if (atlasPx !== ATLAS || surfPx !== SURF || dispPx !== DISP) return null
  const expected = HEADER_BYTES + atlasPx * atlasPx * 4 + surfPx * surfPx * 4 + dispPx * dispPx
  if (buf.byteLength !== expected) return null
  const f = new Float32Array(buf, 32, 5)
  let o = HEADER_BYTES
  const albedo = new Uint8Array(buf, o, atlasPx * atlasPx * 4)
  o += atlasPx * atlasPx * 4
  const surf = new Uint8Array(buf, o, surfPx * surfPx * 4)
  o += surfPx * surfPx * 4
  const disp = new Uint8Array(buf, o, dispPx * dispPx)
  return { atlasPx, tilePx, surfPx, dispPx, nSide, rXZ: f[0]!, y0: f[1]!, y1: f[2]!, cx: f[3]!, cz: f[4]!, albedo, surf, disp }
}

function packRelief(a: ReliefAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.surf.byteLength + a.disp.byteLength)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.atlasPx
  u[3] = a.tilePx
  u[4] = a.nSide
  u[5] = a.surfPx
  u[6] = a.dispPx
  const f = new Float32Array(buf, 32, 5)
  f[0] = a.rXZ
  f[1] = a.y0
  f[2] = a.y1
  f[3] = a.cx
  f[4] = a.cz
  let o = HEADER_BYTES
  new Uint8Array(buf, o, a.albedo.byteLength).set(a.albedo)
  o += a.albedo.byteLength
  new Uint8Array(buf, o, a.surf.byteLength).set(a.surf)
  o += a.surf.byteLength
  new Uint8Array(buf, o, a.disp.byteLength).set(a.disp)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the relief atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesRelief(ctx: BakeCtx, speciesId: string): Promise<ReliefAtlas> {
  const key = `relief-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesRelief(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackRelief(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackRelief(buf)
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

async function bakeSpeciesRelief(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
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
  const ndTex = mkTarget('bake-nrm-dist')
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
      { view: ndTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  for (let k = 0; k < N_VIEWS; k++) {
    const col = k % GRID
    const row = Math.floor(k / GRID)
    pass.setViewport(col * BIG_TILE, row * BIG_TILE, BIG_TILE, BIG_TILE, 0, 1)
    pass.setScissorRect(col * BIG_TILE, row * BIG_TILE, BIG_TILE, BIG_TILE)
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
  const rbNd = mkReadback('bake-rb-nrm-dist')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  enc.copyTextureToBuffer({ texture: ndTex }, { buffer: rbNd, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNd.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNd = new Uint8Array(rbNd.getMappedRange()).slice()
  rbAlb.unmap()
  rbNd.unmap()
  for (const res of [vbuf, ibuf, albTex, ndTex, depthTex, uni, rbAlb, rbNd]) res.destroy()

  const out = postProcess(bigAlb, bigNd)
  return packRelief({
    atlasPx: ATLAS,
    tilePx: TILE,
    surfPx: SURF,
    dispPx: DISP,
    nSide: N_SIDE,
    rXZ,
    y0,
    y1,
    cx,
    cz,
    ...out,
  })
}

interface PostOut {
  albedo: Uint8Array<ArrayBuffer>
  surf: Uint8Array<ArrayBuffer>
  disp: Uint8Array<ArrayBuffer>
}

/**
 * One walk over the supersampled render produces all three rates at once:
 * 2x2 -> ALBEDO, 4x4 -> SURF, 32x32 -> DISP, every reduction weighted by
 * coverage so empty background never pulls the mean. Then: dilation (so
 * bilinear/mip filtering never reaches into unwritten texels), a hole-free
 * flood fill of DISP (the vertex shader reads it outside the silhouette too),
 * and a column-integral sky-occlusion term.
 */
function postProcess(bigAlb: Uint8Array, bigNd: Uint8Array): PostOut {
  const A = ATLAS
  const S = SURF
  const D = DISP
  const albedo = new Uint8Array(A * A * 4)
  const albFilled = new Uint8Array(A * A)
  const aRgb = new Float32Array(A * A * 3)
  const aCov = new Float32Array(A * A)

  const sNrm = new Float32Array(S * S * 3)
  const sDist = new Float32Array(S * S)
  const sCov = new Float32Array(S * S)
  const sFilled = new Uint8Array(S * S)

  const dDist = new Float32Array(D * D)
  const dW = new Float32Array(D * D)

  const aStep = BIG / A // 2
  const sStep = BIG / S // 4
  const dStep = BIG / D // 32

  for (let y = 0; y < BIG; y++) {
    const ay = (y / aStep) | 0
    const sy = (y / sStep) | 0
    const dy = (y / dStep) | 0
    for (let x = 0; x < BIG; x++) {
      const s = (y * BIG + x) * 4
      const a = bigAlb[s + 3]!
      if (a === 0) continue
      const w = a / 255

      const ai = ay * A + ((x / aStep) | 0)
      aRgb[ai * 3] = aRgb[ai * 3]! + bigAlb[s]! * w
      aRgb[ai * 3 + 1] = aRgb[ai * 3 + 1]! + bigAlb[s + 1]! * w
      aRgb[ai * 3 + 2] = aRgb[ai * 3 + 2]! + bigAlb[s + 2]! * w
      aCov[ai] = aCov[ai]! + w

      const si = sy * S + ((x / sStep) | 0)
      sNrm[si * 3] = sNrm[si * 3]! + (bigNd[s]! / 127.5 - 1) * w
      sNrm[si * 3 + 1] = sNrm[si * 3 + 1]! + (bigNd[s + 1]! / 127.5 - 1) * w
      sNrm[si * 3 + 2] = sNrm[si * 3 + 2]! + (bigNd[s + 2]! / 127.5 - 1) * w
      sDist[si] = sDist[si]! + (bigNd[s + 3]! / 255) * w
      sCov[si] = sCov[si]! + w

      const di = dy * D + ((x / dStep) | 0)
      dDist[di] = dDist[di]! + (bigNd[s + 3]! / 255) * w
      dW[di] = dW[di]! + w
    }
  }

  const aSub = aStep * aStep
  for (let i = 0; i < A * A; i++) {
    const c = aCov[i]!
    if (c <= 0) continue
    albedo[i * 4] = Math.round(aRgb[i * 3]! / c)
    albedo[i * 4 + 1] = Math.round(aRgb[i * 3 + 1]! / c)
    albedo[i * 4 + 2] = Math.round(aRgb[i * 3 + 2]! / c)
    albedo[i * 4 + 3] = Math.round(Math.min(1, c / aSub) * 255)
    albFilled[i] = 1
  }

  for (let i = 0; i < S * S; i++) {
    const c = sCov[i]!
    if (c <= 0) continue
    sNrm[i * 3] = sNrm[i * 3]! / c
    sNrm[i * 3 + 1] = sNrm[i * 3 + 1]! / c
    sNrm[i * 3 + 2] = sNrm[i * 3 + 2]! / c
    sDist[i] = sDist[i]! / c
    sCov[i] = Math.min(1, c / (sStep * sStep))
    sFilled[i] = 1
  }
  for (let i = 0; i < D * D; i++) {
    if (dW[i]! > 0) dDist[i] = dDist[i]! / dW[i]!
  }

  dilateAlbedo(albedo, albFilled)
  dilateSurf(sNrm, sDist, sFilled)
  const dFilled = new Uint8Array(D * D)
  for (let i = 0; i < D * D; i++) dFilled[i] = dW[i]! > 0 ? 1 : 0
  floodFillDisp(dDist, dFilled)

  const surf = new Uint8Array(S * S * 4)
  const ao = skyOcclusion(sCov)
  for (let i = 0; i < S * S; i++) {
    const e = octEncode(sNrm[i * 3]!, sNrm[i * 3 + 1]!, sNrm[i * 3 + 2]!)
    surf[i * 4] = e[0]
    surf[i * 4 + 1] = e[1]
    surf[i * 4 + 2] = Math.round(clamp01(sDist[i]!) * 255)
    surf[i * 4 + 3] = Math.round(ao[i]! * 255)
  }
  const disp = new Uint8Array(D * D)
  for (let i = 0; i < D * D; i++) disp[i] = Math.round(clamp01(dDist[i]!) * 255)

  return { albedo, surf, disp }
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

/** Dilate colour into empty texels (alpha stays 0), clamped to tile bounds. */
function dilateAlbedo(albedo: Uint8Array, filled: Uint8Array): void {
  const N = ATLAS
  let cur = filled
  for (let p = 0; p < DILATE_ALBEDO; p++) {
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
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const n = yy * N + xx
            if (cur[n]! === 0) continue
            count++
            r += albedo[n * 4]!
            g += albedo[n * 4 + 1]!
            b += albedo[n * 4 + 2]!
          }
        }
        if (count === 0) continue
        albedo[idx * 4] = Math.round(r / count)
        albedo[idx * 4 + 1] = Math.round(g / count)
        albedo[idx * 4 + 2] = Math.round(b / count)
        next[idx] = 1
      }
    }
    cur = next
  }
}

/** Same dilation for the surface atlas (normals + front distance). */
function dilateSurf(nrm: Float32Array, dist: Float32Array, filled: Uint8Array): void {
  const N = SURF
  let cur = filled
  for (let p = 0; p < DILATE_SURF; p++) {
    const next = cur.slice()
    for (let y = 0; y < N; y++) {
      const ty0 = Math.floor(y / SURF_TILE) * SURF_TILE
      const ty1 = ty0 + SURF_TILE - 1
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (cur[idx]! !== 0) continue
        const tx0 = Math.floor(x / SURF_TILE) * SURF_TILE
        const tx1 = tx0 + SURF_TILE - 1
        let count = 0
        let nx = 0
        let ny = 0
        let nz = 0
        let dd = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const n = yy * N + xx
            if (cur[n]! === 0) continue
            count++
            nx += nrm[n * 3]!
            ny += nrm[n * 3 + 1]!
            nz += nrm[n * 3 + 2]!
            dd += dist[n]!
          }
        }
        if (count === 0) continue
        nrm[idx * 3] = nx / count
        nrm[idx * 3 + 1] = ny / count
        nrm[idx * 3 + 2] = nz / count
        dist[idx] = dd / count
        next[idx] = 1
      }
    }
    cur = next
  }
}

/**
 * Flood the smoothed displacement field until every texel of every tile has a
 * value, then blur it twice. The vertex grid samples this OUTSIDE the plant's
 * silhouette as well (the card's rim), so a hole there would fold the card
 * backwards; a smooth extrapolation of the nearest real surface is exactly the
 * "curved card" rim we want.
 */
function floodFillDisp(dist: Float32Array, filled: Uint8Array): void {
  const N = DISP
  const T = DISP_TILE
  let cur = filled
  for (let pass = 0; pass < T; pass++) {
    let remaining = 0
    const next = cur.slice()
    for (let y = 0; y < N; y++) {
      const ty0 = Math.floor(y / T) * T
      const ty1 = ty0 + T - 1
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (cur[idx]! !== 0) continue
        const tx0 = Math.floor(x / T) * T
        const tx1 = tx0 + T - 1
        let count = 0
        let acc = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const n = yy * N + xx
            if (cur[n]! === 0) continue
            count++
            acc += dist[n]!
          }
        }
        if (count === 0) {
          remaining++
          continue
        }
        dist[idx] = acc / count
        next[idx] = 1
      }
    }
    cur = next
    if (remaining === 0) break
  }
  // Two 3x3 box blurs, tile-clamped: the vertex grid must not see steps.
  for (let pass = 0; pass < 2; pass++) {
    const src = dist.slice()
    for (let y = 0; y < N; y++) {
      const ty0 = Math.floor(y / T) * T
      const ty1 = ty0 + T - 1
      for (let x = 0; x < N; x++) {
        const tx0 = Math.floor(x / T) * T
        const tx1 = tx0 + T - 1
        let count = 0
        let acc = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            count++
            acc += src[yy * N + xx]!
          }
        }
        dist[y * N + x] = acc / count
      }
    }
  }
}

/**
 * Sky occlusion by column integral. Inside a side tile, texture row 0 is the
 * TOP of the plant, so walking rows downward accumulates exactly the canopy
 * that stands between a texel and the sky. The top tile looks straight down
 * onto the canopy surface — nothing is above it, so it stays fully lit.
 */
function skyOcclusion(cov: Float32Array): Float32Array {
  const N = SURF
  const T = SURF_TILE
  const ao = new Float32Array(N * N).fill(1)
  for (let tile = 0; tile < N_SIDE; tile++) {
    const tx0 = (tile % GRID) * T
    const ty0 = Math.floor(tile / GRID) * T
    for (let x = tx0; x < tx0 + T; x++) {
      let acc = 0
      for (let y = ty0; y < ty0 + T; y++) {
        const i = y * N + x
        acc += cov[i]!
        ao[i] = AO_FLOOR + (1 - AO_FLOOR) * Math.exp(-AO_K * acc)
      }
    }
  }
  return ao
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
