import bakeShaderSrc from './shaders/moss_bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * The Uncharted-4 moss material set, measured off the 19.8M-triangle Sphagnum
 * source mesh instead of authored by hand.
 *
 * The talk's moss shader eats four maps — Colour, Normal, Heightmap, Ambient
 * Occlusion — and notes that its AO map is unusually dark. All four are baked
 * here for exactly ONE periodic tile, straight down over the tile square:
 *
 *   texture A  rgba8   rgb = albedo, a = HEIGHT of the topmost surface
 *   texture B  rgba8   rgb = mesh normal (upper hemisphere), a = cavity AO
 *
 * Height comes free: an orthographic top-down render with a depth test keeps the
 * topmost surface per texel, so its y IS the heightfield. AO is then computed
 * from that heightfield by a wrapping horizon sweep — the crevices between
 * capitula genuinely go black, which is the "very dark AO" the paper describes.
 *
 * Normals are stored as plain xyz, NOT octahedral: this atlas gets a mip chain,
 * and octahedral pairs are not mip-averageable (they filter toward straight up).
 *
 * The bake draws the mesh 3x3 times, offset by the tile period, so a tile
 * receives its neighbours' overhang and the result is exactly periodic — the
 * runtime can then sample it with `repeat` addressing and no seam.
 */

export const TEX = 2048
const SS = 2
const BIG = TEX * SS
const MAGIC = 0x314d444e // 'NDM1'
const VERSION = 4
const HEADER_BYTES = 128
const DILATE_PASSES = 5
/**
 * Horizon-sweep AO: 8 azimuths x these radii, in MILLIMETRES (so the AO does
 * not change when the texture resolution does). The sweep deliberately starts
 * at ~1mm: below that the cushion's own sub-texel roughness makes the horizon
 * tangent enormous everywhere, and radii of a texel or two produce a noise
 * field (mean AO 0.26) instead of a cavity map. The top end, ~17mm, is a few
 * capitulum widths — the scale of a real inter-cushion crevice.
 */
const AO_RADII_MM = [1.05, 2.1, 4.2, 8.4, 16.9]

export interface MossTile {
  texPx: number
  /** Periodic tile period (m, at mesh scale 1). */
  tileM: number
  /** Capture range: height 0 is world y0, height 1 is y1 (m, mesh scale 1). */
  y0: number
  y1: number
  /** Coverage-weighted mean surface height (m above y0) — the mat's mean plane. */
  planeH: number
  /** Tallest surface (m above y0). */
  topH: number
  /** Mean height of the top decile — the capitulum apex plane. */
  apexH: number
  /** Fraction of the tile that has any moss above the peat. */
  cover: number
  /** Mean albedo of the capitulum tips (paper: "lighter at the tips"). */
  tipColor: [number, number, number]
  /** Mean albedo deep in the cushion — the light-wrap / bounce tint. */
  baseColor: [number, number, number]
  meanColor: [number, number, number]
  /** rgb = albedo, a = height. */
  albedoHeight: Uint8Array<ArrayBuffer>
  /** rgb = normal*0.5+0.5, a = cavity AO. */
  normalAo: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackMoss(buf: ArrayBuffer): MossTile | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 4)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const texPx = u[2]!
  const expected = HEADER_BYTES + texPx * texPx * 8
  if (texPx !== TEX || buf.byteLength !== expected) return null
  const f = new Float32Array(buf, 16, 20)
  return {
    texPx,
    tileM: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    planeH: f[3]!,
    topH: f[4]!,
    cover: f[5]!,
    tipColor: [f[6]!, f[7]!, f[8]!],
    baseColor: [f[9]!, f[10]!, f[11]!],
    meanColor: [f[12]!, f[13]!, f[14]!],
    apexH: f[15]!,
    albedoHeight: new Uint8Array(buf, HEADER_BYTES, texPx * texPx * 4),
    normalAo: new Uint8Array(buf, HEADER_BYTES + texPx * texPx * 4, texPx * texPx * 4),
  }
}

function packMoss(t: MossTile): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + t.albedoHeight.byteLength + t.normalAo.byteLength)
  const u = new Uint32Array(buf, 0, 4)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = t.texPx
  const f = new Float32Array(buf, 16, 20)
  f[0] = t.tileM
  f[1] = t.y0
  f[2] = t.y1
  f[3] = t.planeH
  f[4] = t.topH
  f[5] = t.cover
  f.set(t.tipColor, 6)
  f.set(t.baseColor, 9)
  f.set(t.meanColor, 12)
  f[15] = t.apexH
  new Uint8Array(buf, HEADER_BYTES, t.albedoHeight.byteLength).set(t.albedoHeight)
  new Uint8Array(buf, HEADER_BYTES + t.albedoHeight.byteLength, t.normalAo.byteLength).set(t.normalAo)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the moss tile for a species.
 * The dev server answers a missing /mesh/baked file with the SPA index.html at
 * status 200, which would poison both stores — every result is magic-validated
 * and a poisoned cache entry is rebaked and repaired in place.
 */
export async function loadMossTile(ctx: BakeCtx, speciesId: string): Promise<MossTile> {
  const key = `moss-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    const tileM = speciesById(speciesId).tileM
    if (tileM === undefined) throw new Error(`[${ctx.id}] ${speciesId} has no periodic tileM`)
    return bakeMossTile(ctx, mesh, tileM)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let tile = unpackMoss(buf)
  if (!tile) {
    buf = await runBake()
    tile = unpackMoss(buf)
    if (!tile) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
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
    /* best effort — a cache miss just rebakes */
  }
}

async function bakeMossTile(ctx: BakeCtx, mesh: GcMesh, tileM: number): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const y0 = Math.min(0, by0)
  const y1 = by1

  const verts = mesh.vertices
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
  const nrmTex = mkTarget('bake-normal-height')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(
    uni,
    0,
    new Float32Array([bx0, by0, bz0, 0, bx1 - bx0, by1 - by0, bz1 - bz0, 0, tileM, y0, y1, 0]),
  )

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni } }],
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
  pass.setBindGroup(0, bg)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  // 9 instances = the tile plus its 8 periodic neighbours, so overhanging
  // geometry wraps in and the captured tile is exactly periodic.
  pass.drawIndexed(indices.length, 9)
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

  return packMoss(postProcess(bigAlb, bigNrm, tileM, y0, y1))
}

/**
 * Coverage-weighted 2x downsample, wrapping horizon-sweep AO, wrapping
 * dilation, and the handful of scalars the shader needs (mean plane, apex
 * plane, tip/base colours).
 */
function postProcess(bigAlb: Uint8Array, bigNrm: Uint8Array, tileM: number, y0: number, y1: number): MossTile {
  const N = TEX
  const albedoHeight = new Uint8Array(N * N * 4)
  const normalAo = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const cover = new Float32Array(N * N)
  // Height in [0,1] of the capture range, box-filtered over the "premultiplied"
  // height (empty subsamples count as 0). One channel is then both the
  // heightfield and the coverage signal, and a mip chain pulls a gap toward the
  // surrounding surface height instead of dissolving the mat.
  const height = new Float32Array(N * N)

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let hSum = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * BIG + (x * SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
          nx += (bigNrm[s]! / 127.5 - 1) * a
          ny += (bigNrm[s + 1]! / 127.5 - 1) * a
          nz += (bigNrm[s + 2]! / 127.5 - 1) * a
          hSum += bigNrm[s + 3]! / 255
        }
      }
      const idx = y * N + x
      height[idx] = hSum / (SS * SS)
      if (aSum > 0) {
        const d = idx * 4
        albedoHeight[d] = Math.round(r / aSum)
        albedoHeight[d + 1] = Math.round(g / aSum)
        albedoHeight[d + 2] = Math.round(b / aSum)
        const dn = idx * 3
        nrm[dn] = nx
        nrm[dn + 1] = ny
        nrm[dn + 2] = nz
        cover[idx] = aSum / (255 * SS * SS)
      }
    }
  }

  // --- wrapping dilation of colour + normal into the gaps -------------------
  // Height stays 0 there (it is the gap), but filtering must never pull
  // background black into a capitulum edge. The tile is periodic, so the
  // dilation wraps instead of clamping.
  let filled = new Uint8Array(N * N)
  for (let i = 0; i < N * N; i++) filled[i] = cover[i]! > 0 ? 1 : 0
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = filled.slice()
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (filled[idx] !== 0) continue
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = (y + j + N) % N
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const nIdx = yy * N + ((x + i + N) % N)
            if (filled[nIdx] === 0) continue
            count++
            const s4 = nIdx * 4
            r += albedoHeight[s4]!
            g += albedoHeight[s4 + 1]!
            b += albedoHeight[s4 + 2]!
            const s3 = nIdx * 3
            nx += nrm[s3]!
            ny += nrm[s3 + 1]!
            nz += nrm[s3 + 2]!
          }
        }
        if (count === 0) continue
        const d4 = idx * 4
        albedoHeight[d4] = Math.round(r / count)
        albedoHeight[d4 + 1] = Math.round(g / count)
        albedoHeight[d4 + 2] = Math.round(b / count)
        const d3 = idx * 3
        nrm[d3] = nx / count
        nrm[d3 + 1] = ny / count
        nrm[d3 + 2] = nz / count
        next[idx] = 1
      }
    }
    filled = next
  }

  // --- cavity AO by a wrapping horizon sweep over the heightfield -----------
  // 8 azimuths; per direction the occluded fraction is sin^2 of the largest
  // horizon angle, which is the cosine-weighted visible fraction of a hemisphere
  // cut off at that horizon (the classic horizon-based AO form). A clamped
  // tangent instead saturates the moment any neighbour is higher, which on a
  // cushion is nearly everywhere and made even the capitulum tops read as holes.
  // The paper's moss AO "goes very dark" and a real cushion's inter-capitulum
  // crevices genuinely do — but its tops have to come out near white.
  const range = y1 - y0
  const texelM = tileM / N
  const aoRadii = AO_RADII_MM.map((mm) => Math.max(1, Math.round(mm / 1000 / texelM)))
  const dirs: [number, number][] = []
  for (let k = 0; k < 8; k++) {
    const a = (k * Math.PI) / 4
    dirs.push([Math.cos(a), Math.sin(a)])
  }
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx = y * N + x
      const h0 = height[idx]! * range
      let occ = 0
      for (const [dx, dz] of dirs) {
        let best = 0
        for (const r of aoRadii) {
          const sx = (Math.round(x + dx * r) % N + N) % N
          const sy = (Math.round(y + dz * r) % N + N) % N
          const dh = height[sy * N + sx]! * range - h0
          if (dh <= 0) continue
          const t = dh / (r * texelM)
          const sin2 = (t * t) / (1 + t * t)
          if (sin2 > best) best = sin2
        }
        occ += best
      }
      normalAo[idx * 4 + 3] = Math.round(255 * Math.max(0, 1 - occ / 8))
    }
  }

  // --- pack height + normals, gather statistics ----------------------------
  const hist = new Uint32Array(256)
  let coverSum = 0
  let topH = 0
  let hCovSum = 0
  let hCovW = 0
  for (let i = 0; i < N * N; i++) {
    const q = Math.min(255, Math.round(height[i]! * 255))
    albedoHeight[i * 4 + 3] = q
    hist[q]!++
    const s3 = i * 3
    const len = Math.hypot(nrm[s3]!, nrm[s3 + 1]!, nrm[s3 + 2]!) || 1
    normalAo[i * 4] = Math.round((nrm[s3]! / len) * 127.5 + 127.5)
    normalAo[i * 4 + 1] = Math.round((nrm[s3 + 1]! / len) * 127.5 + 127.5)
    normalAo[i * 4 + 2] = Math.round((nrm[s3 + 2]! / len) * 127.5 + 127.5)
    coverSum += cover[i]!
    if (height[i]! * range > topH) topH = height[i]! * range
    if (cover[i]! > 0) {
      hCovSum += height[i]! * range * cover[i]!
      hCovW += cover[i]!
    }
  }

  // Height quantiles from the histogram: tips = top 12%, base = bottom 45%.
  const total = N * N
  const quantile = (frac: number): number => {
    let acc = 0
    for (let q = 0; q < 256; q++) {
      acc += hist[q]!
      if (acc >= frac * total) return q
    }
    return 255
  }
  const qTip = quantile(0.88)
  const qBase = quantile(0.45)

  const accum = (test: (q: number) => boolean): [number, number, number] => {
    let r = 0
    let g = 0
    let b = 0
    let w = 0
    for (let i = 0; i < total; i++) {
      if (cover[i]! <= 0) continue
      if (!test(albedoHeight[i * 4 + 3]!)) continue
      const c = cover[i]!
      r += albedoHeight[i * 4]! * c
      g += albedoHeight[i * 4 + 1]! * c
      b += albedoHeight[i * 4 + 2]! * c
      w += c
    }
    if (w <= 0) return [0.5, 0.5, 0.5]
    return [r / w / 255, g / w / 255, b / w / 255]
  }
  const tipColor = accum((q) => q >= qTip)
  const baseColor = accum((q) => q <= qBase)
  const meanColor = accum(() => true)

  // Mean height of the top decile — the capitulum apex plane.
  let apexSum = 0
  let apexW = 0
  const qApex = quantile(0.9)
  for (let i = 0; i < total; i++) {
    const q = albedoHeight[i * 4 + 3]!
    if (q < qApex) continue
    apexSum += (q / 255) * range
    apexW++
  }

  return {
    texPx: N,
    tileM,
    y0,
    y1,
    planeH: hCovW > 0 ? hCovSum / hCovW : range * 0.5,
    topH,
    apexH: apexW > 0 ? apexSum / apexW : topH,
    cover: coverSum / total,
    tipColor,
    baseColor,
    meanColor,
    albedoHeight,
    normalAo,
  }
}
