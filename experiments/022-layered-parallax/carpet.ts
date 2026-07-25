import carpetBakeSrc from './shaders/carpet_bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Carpet bake — the layered-parallax representation, re-targeted at a mat.
 *
 * A carpet species is a PERIODIC COMMUNITY TILE (Sphagnum palustre: 0.18m
 * across, 0.07-0.09m tall), not a plant. The slab bake in bake.ts is built for
 * an upright plant and spends its atlas accordingly: 8 side azimuths x 3 depth
 * slabs that a mat never shows, and one top view captured over the mesh's full
 * support radius at the SIDE view's aspect ratio (3.3:1 for the moss). Only
 * ~4% of that atlas is ever sampled by a carpet, and the part that is has ~130
 * texels across the tile in v.
 *
 * So a carpet gets its own, much smaller capture:
 *
 *   - straight down over EXACTLY the tile square [0, tileM]^2 in the mesh
 *     frame (tile origin is (0,0) for every current source mesh), square and
 *     isotropic, TEX x TEX per band;
 *   - WRAPPED: the mesh is drawn once per 3x3 neighbour offset, so everything
 *     that hangs over one edge of the period re-enters on the opposite one.
 *     The image is then exactly one period and can be sampled with `repeat`
 *     addressing and mipped to 1x1 without ever growing a seam;
 *   - split into N_BAND HEIGHT BANDS at quartiles of the geometry's own height
 *     distribution, each recording the MEAN HEIGHT of the geometry it holds —
 *     those are the planes the runtime intersects the eye ray with, and they
 *     are what turns a flat mat into a cushion with interior parallax;
 *   - plus one MERGED tile (no band restriction) for the far ring.
 *
 * All N_BAND + 1 tiles live as layers of one array texture, so the runtime
 * needs a single binding and the hardware does the wrapping and the mip
 * selection.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'CRP1', version, tex, nrmTex, layers, nBand, pad x 10
 *   f32 tileM, yMin, yMax, then `layers` plane heights (mesh metres)
 *   ...zeros to byte 128
 *   u8[layers * tex * tex * 4]        albedo rgba8 (straight colour, a = coverage)
 *   u8[layers * nrmTex * nrmTex * 4]  normals rgb8 (mesh frame, +Y hemisphere),
 *                                     a = coverage. Plain vectors, NOT oct —
 *                                     these are mipped, and oct is not
 *                                     mip-averageable (see CLAUDE.md).
 */

/** Height bands per tile. Layer N_BAND is the merged (all-heights) tile. */
export const N_BAND = 4
export const N_CARPET_LAYER = N_BAND + 1
/** Albedo texels across the tile — 0.18m / 512 = 0.35mm at life size. */
export const CARPET_TEX = 512
/** Normals at half resolution: low frequency next to coverage. */
export const CARPET_NRM = CARPET_TEX >> 1

const MAGIC = 0x31505243 // 'CRP1'
const VERSION = 1
const HEADER_BYTES = 128
const SS = 2 // supersampling of the bake render
const DILATE_PASSES = 4

export interface CarpetAtlas {
  /** Periodic tile size (m) at scale 1. */
  tileM: number
  tex: number
  nrmTex: number
  layers: number
  /** Plane height (mesh metres) per layer; index N_BAND is the merged tile. */
  heights: number[]
  yMin: number
  yMax: number
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

// ---------------------------------------------------------------------------
// artifact (de)serialization
// ---------------------------------------------------------------------------

export function unpackCarpet(buf: ArrayBuffer): CarpetAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 16)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const tex = u[2]!
  const nrmTex = u[3]!
  const layers = u[4]!
  if (u[5] !== N_BAND || layers !== N_CARPET_LAYER || tex !== CARPET_TEX || nrmTex !== CARPET_NRM) return null
  const f = new Float32Array(buf, 64, 3 + layers)
  const albBytes = layers * tex * tex * 4
  const nrmBytes = layers * nrmTex * nrmTex * 4
  if (buf.byteLength !== HEADER_BYTES + albBytes + nrmBytes) return null
  return {
    tileM: f[0]!,
    yMin: f[1]!,
    yMax: f[2]!,
    heights: Array.from(f.subarray(3, 3 + layers)),
    tex,
    nrmTex,
    layers,
    albedo: new Uint8Array(buf, HEADER_BYTES, albBytes),
    normal: new Uint8Array(buf, HEADER_BYTES + albBytes, nrmBytes),
  }
}

function packCarpet(a: CarpetAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.normal.byteLength)
  new Uint32Array(buf, 0, 16).set([MAGIC, VERSION, a.tex, a.nrmTex, a.layers, N_BAND])
  new Float32Array(buf, 64, 3 + a.layers).set([a.tileM, a.yMin, a.yMax, ...a.heights])
  new Uint8Array(buf, HEADER_BYTES, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + a.albedo.byteLength, a.normal.byteLength).set(a.normal)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the carpet atlas for a species.
 * Same magic-validation shim as the slab loader: the dev server answers a
 * missing /mesh/baked file with the SPA index.html at status 200.
 */
export async function loadSpeciesCarpet(ctx: BakeCtx, speciesId: string): Promise<CarpetAtlas> {
  const key = `carpet-v${VERSION}-${speciesId}`
  const species = speciesById(speciesId)
  if (species.tileM === undefined) throw new Error(`[${ctx.id}] ${speciesId} is a carpet entry but has no periodic tileM`)
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(species.meshId)
    return bakeSpeciesCarpet(ctx, mesh, species.tileM!)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackCarpet(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackCarpet(buf)
    if (!atlas) throw new Error(`[${ctx.id}] carpet bake for ${speciesId} produced an invalid artifact`)
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
// the bake itself
// ---------------------------------------------------------------------------

async function bakeSpeciesCarpet(ctx: BakeCtx, mesh: GcMesh, tileM: number): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const yMin = Math.min(0, by0!)
  const yMax = by1!

  const { bounds, heights } = bandSplit(mesh, yMin, yMax)

  // --- transient GPU resources ---------------------------------------------
  const verts = mesh.vertices
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const bigW = N_CARPET_LAYER * CARPET_TEX * SS
  const bigH = CARPET_TEX * SS
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
  const albTex = mkTarget('carpet-bake-albedo')
  const nrmTex = mkTarget('carpet-bake-normal')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/carpet-bake-depth`,
      size: [bigW, bigH],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // --- per-instance records: 9 wrap offsets x (N_BAND bands + merged) -------
  const REC_FLOATS = 8
  const N_WRAP = 9
  const recs = new Float32Array(N_WRAP * N_CARPET_LAYER * REC_FLOATS)
  const slotScaleX = 1 / N_CARPET_LAYER
  for (let w = 0; w < N_WRAP; w++) {
    const ox = ((w % 3) - 1) * tileM
    const oz = (Math.floor(w / 3) - 1) * tileM
    for (let l = 0; l < N_CARPET_LAYER; l++) {
      const o = (w * N_CARPET_LAYER + l) * REC_FLOATS
      const lo = l < N_BAND ? bounds[l + 1]! : -1e9
      const hi = l < N_BAND ? bounds[l]! : 1e9
      recs.set([ox, oz, lo, hi], o)
      recs.set([-1 + (2 * l + 1) * slotScaleX, 0, slotScaleX, 1], o + 4)
    }
  }
  const recBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-recs`, size: recs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(recBuf, 0, recs)

  const constants = new Float32Array(16)
  constants.set([0, 0, tileM, 0], 0)
  constants.set([yMin, yMax, 0, 0], 4)
  constants.set([bx0!, by0!, bz0!, 0], 8)
  constants.set([bx1! - bx0!, by1! - by0!, bz1! - bz0!, 0], 12)
  const constBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/carpet-const`,
      size: constants.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(constBuf, 0, constants)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bake-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/carpet-bake-bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: recBuf } },
      { binding: 1, resource: { buffer: constBuf } },
    ],
  })

  const module = ctx.shaders.module(carpetBakeSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/carpet-bake-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/carpet-bake-pl`, bindGroupLayouts: [bgl] }),
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

  // One submit per wrap offset keeps a single command buffer to the same size
  // the slab bake already runs at on these 19.8M-triangle meshes.
  const albView = albTex.createView()
  const nrmView = nrmTex.createView()
  const depthView = depthTex.createView()
  for (let w = 0; w < N_WRAP; w++) {
    const first = w === 0
    const enc = device.createCommandEncoder({ label: `${ctx.id}/carpet-enc-${w}` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/carpet-pass-${w}`,
      colorAttachments: [
        { view: albView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: first ? 'clear' : 'load', storeOp: 'store' },
        {
          view: nrmView,
          clearValue: { r: 0.5, g: 1, b: 0.5, a: 0 },
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
    pass.drawIndexed(indices.length, N_CARPET_LAYER, 0, 0, w * N_CARPET_LAYER)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }
  // Free the source geometry before allocating readback staging.
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
  const rbAlb = mkReadback('carpet-rb-albedo')
  const rbNrm = mkReadback('carpet-rb-normal')
  const encCopy = device.createCommandEncoder({ label: `${ctx.id}/carpet-copy` })
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

  const out = postProcess(bigAlb, bigNrm, bpr)
  return packCarpet({ tileM, tex: CARPET_TEX, nrmTex: CARPET_NRM, layers: N_CARPET_LAYER, heights, yMin, yMax, ...out })
}

/**
 * Height band boundaries + the plane each band lives on.
 *
 * Boundaries are QUANTILES of the geometry's own height distribution rather
 * than equal slices of the bounding box: a Sphagnum cushion carries almost all
 * of its structure in the top ~4cm, so equal slices would put three of four
 * bands in nearly empty air. Equal-mass bands put a plane where there is
 * something to see. The plane itself is the MEAN height inside the band.
 *
 * Returns `bounds` (N_BAND + 1 descending values, bounds[0] = +inf) and
 * `heights` (N_BAND band planes + the merged tile's mean height).
 */
function bandSplit(mesh: GcMesh, yMin: number, yMax: number): { bounds: number[]; heights: number[] } {
  const hdr = mesh.header
  const by0 = hdr.boundsMin[1]!
  const sy = (hdr.boundsMax[1]! - by0) / 65535
  const verts = mesh.vertices
  const stride = Math.max(1, Math.floor(hdr.vertexCount / 400_000))

  const sample: number[] = []
  for (let i = 0; i < hdr.vertexCount; i += stride) sample.push(by0 + verts[i * 8 + 1]! * sy)
  sample.sort((a, b) => a - b)

  const bounds: number[] = [1e9]
  for (let l = 1; l < N_BAND; l++) {
    const q = 1 - l / N_BAND
    bounds.push(sample[Math.min(sample.length - 1, Math.floor(q * sample.length))]!)
  }
  bounds.push(-1e9)

  const sums = new Float64Array(N_BAND)
  const counts = new Float64Array(N_BAND)
  let allSum = 0
  for (const y of sample) {
    for (let l = 0; l < N_BAND; l++) {
      if (y > bounds[l + 1]! && y <= bounds[l]!) {
        sums[l] = sums[l]! + y
        counts[l] = counts[l]! + 1
        break
      }
    }
    allSum += y
  }
  const heights: number[] = []
  for (let l = 0; l < N_BAND; l++) {
    heights.push(counts[l]! > 0 ? sums[l]! / counts[l]! : yMin + ((yMax - yMin) * (N_BAND - 0.5 - l)) / N_BAND)
  }
  heights.push(sample.length > 0 ? allSum / sample.length : (yMin + yMax) * 0.5)
  return { bounds, heights }
}

interface PostOut {
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
}

/**
 * Coverage-weighted downsample of the supersampled canvas, then a few wrapped
 * dilation passes per layer (the tile is periodic, so dilation wraps too — a
 * clamped dilation would build a slightly different fringe on each edge and
 * break the seam it exists to protect), then split into layers.
 */
function postProcess(bigAlb: Uint8Array, bigNrm: Uint8Array, bpr: number): PostOut {
  const T = CARPET_TEX
  const W = N_CARPET_LAYER * T
  const albedo = new Uint8Array(W * T * 4)
  const nrm = new Float32Array(W * T * 3)
  const filled = new Uint8Array(W * T)

  for (let y = 0; y < T; y++) {
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
      const idx = y * W + x
      if (aSum > 0) {
        albedo[idx * 4] = Math.round(r / aSum)
        albedo[idx * 4 + 1] = Math.round(g / aSum)
        albedo[idx * 4 + 2] = Math.round(b / aSum)
        albedo[idx * 4 + 3] = Math.round(aSum / (SS * SS))
        nrm[idx * 3] = nx
        nrm[idx * 3 + 1] = ny
        nrm[idx * 3 + 2] = nz
        filled[idx] = 1
      }
    }
  }

  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (let y = 0; y < T; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x
        if (cur[idx]! !== 0) continue
        const layer = Math.floor(x / T)
        const lx = x - layer * T
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            // Wrap inside the layer's own tile — the tile is periodic.
            const sx = layer * T + ((lx + i + T) % T)
            const sy = (y + j + T) % T
            const nIdx = sy * W + sx
            if (cur[nIdx]! === 0) continue
            count++
            r += albedo[nIdx * 4]!
            g += albedo[nIdx * 4 + 1]!
            b += albedo[nIdx * 4 + 2]!
            nx += nrm[nIdx * 3]!
            ny += nrm[nIdx * 3 + 1]!
            nz += nrm[nIdx * 3 + 2]!
          }
        }
        if (count === 0) continue
        albedo[idx * 4] = Math.round(r / count)
        albedo[idx * 4 + 1] = Math.round(g / count)
        albedo[idx * 4 + 2] = Math.round(b / count)
        nrm[idx * 3] = nx / count
        nrm[idx * 3 + 1] = ny / count
        nrm[idx * 3 + 2] = nz / count
        next[idx] = 1
      }
    }
    cur = next
  }

  const outAlb = new Uint8Array(N_CARPET_LAYER * T * T * 4)
  for (let l = 0; l < N_CARPET_LAYER; l++) {
    for (let y = 0; y < T; y++) {
      const src = (y * W + l * T) * 4
      outAlb.set(albedo.subarray(src, src + T * 4), (l * T * T + y * T) * 4)
    }
  }

  // Normals at half resolution, coverage weighted, as plain unit vectors in
  // the +Y hemisphere: they are mipped on the GPU, and octahedral encoding is
  // not mip-averageable (a box filter over oct pairs decodes to straight up).
  const N = CARPET_NRM
  const outNrm = new Uint8Array(N_CARPET_LAYER * N * N * 4)
  for (let l = 0; l < N_CARPET_LAYER; l++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        let nx = 0
        let ny = 0
        let nz = 0
        let cov = 0
        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 2; i++) {
            const si = (y * 2 + j) * W + l * T + x * 2 + i
            const a = albedo[si * 4 + 3]! / 255
            const wgt = a + 0.002
            nx += nrm[si * 3]! * wgt
            ny += nrm[si * 3 + 1]! * wgt
            nz += nrm[si * 3 + 2]! * wgt
            cov += a
          }
        }
        const len = Math.hypot(nx, ny, nz) || 1
        const d = (l * N * N + y * N + x) * 4
        outNrm[d] = Math.round(((nx / len) * 0.5 + 0.5) * 255)
        outNrm[d + 1] = Math.round(((ny / len) * 0.5 + 0.5) * 255)
        outNrm[d + 2] = Math.round(((nz / len) * 0.5 + 0.5) * 255)
        outNrm[d + 3] = Math.round(Math.min(1, cov / 4) * 255)
      }
    }
  }

  return { albedo: outAlb, normal: outNrm }
}
