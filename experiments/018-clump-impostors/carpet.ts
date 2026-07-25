import carpetBakeSrc from './shaders/carpet_bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Carpet-tile bake — the moss half of this experiment.
 *
 * A Sphagnum entry is not a plant, it is a 0.18m periodic community tile that
 * is 0.07-0.09m tall with 3.3cm of capitulum relief. Four camera-facing
 * sub-clump cards are the wrong shape for that twice over (vertical cards
 * slicing through the ground, and the whole 45-tile side-view atlas spent on
 * views a mat never shows), so a carpet species gets its own artifact and
 * skips the clump atlas entirely.
 *
 * What is captured: ONE straight-down orthographic view of exactly the tile
 * square, with the mesh drawn 9x at +/- one tile step so the periodic overflow
 * wraps correctly (see carpet_bake.wgsl). Every texel of the whole 25MB budget
 * therefore lands on the one view a mat actually shows, at 1024px across
 * 0.18m — 5.7k px/m, ~3.5x the linear texel density 001-billboard-smoke gets
 * for the same tile out of its 3x3 card atlas.
 *
 * What makes it 3D rather than a decal: the capture also keeps the HEIGHT of
 * the topmost surface per texel (the depth test resolves it for free). The
 * renderer draws the near field as SHELLS — four ground-parallel quads at four
 * heights taken from the quantiles of that height field, each keeping only the
 * texels whose cushion reaches its level. The result is a stepped, self-
 * occluding, depth-writing approximation of the real relief instead of a decal
 * on a plane; past `carpetShellDist` it collapses to a single quad.
 *
 * Artifact layout (little-endian), header 256B:
 *   u32 magic 'CPT1', version, tilePx, auxPx, albMips, auxMips, 0, 0
 *   f32[4] @32  tileM, y0, y1, meanH        (metres / fraction of [y0,y1])
 *   f32[4] @48  shell heights, descending    (fraction of [y0,y1])
 *   f32[4] @64  shell lower thresholds       (fraction; last = -1)
 *   f32[4] @80  farH, reliefSpan, coverage, 0
 *   u8[...]     albedo rgba8 (rgb = coverage-weighted colour, a = coverage),
 *               all mip levels concatenated
 *   u8[...]     aux rgba8 (r,g = mesh-frame normal xz, b = height, a = coverage),
 *               all mip levels concatenated
 *
 * Colour convention: rgb is ALREADY normalised by coverage at every level
 * (rgb = sum(rgb*a)/sum(a)), so the shader must NOT divide by alpha.
 */

export const TILE_PX = 1024
export const AUX_PX = 512
/** Ground-parallel shells per tile in the near LOD. */
export const SHELLS = 4

const SS = 2
const BIG = TILE_PX * SS
const AUX_SS = BIG / AUX_PX
const MAGIC = 0x31545043 // 'CPT1'
const VERSION = 2
const HEADER_BYTES = 256
const DILATE_PASSES = 4
/**
 * Shell heights, as "fraction of the covered area that lies ABOVE this shell".
 * Sphagnum's height histogram is strongly top-heavy (a closed cushion with a
 * few deep gaps), so evenly spaced heights would put three shells in empty air;
 * quantiles put them where the capitula actually are.
 */
const SHELL_QUANTILES = [0.08, 0.35, 0.68]
/** Lowest shell: the mat's skirt, just above the peat, so it closes from the side. */
const SHELL_FLOOR = 0.05
/**
 * How much of the captured leaf-scale normal survives into the stored normal.
 * The raw capture is per-leaf facets at 0.35mm — genuine, but at any real
 * viewing distance it is white noise that mips straight to "up" and lights the
 * cushion perfectly flat. The structure a viewer actually sees is the CUSHION,
 * and that lives in the height field, so the stored normal is mostly the
 * gradient of the (smoothed) apex height with a little of the leaf noise left
 * on top for close-range texture.
 */
const MICRO_NORMAL_WEIGHT = 0.35

export const ALB_MIPS = Math.log2(TILE_PX) + 1
export const AUX_MIPS = Math.log2(AUX_PX) + 1

export interface CarpetTile {
  tilePx: number
  auxPx: number
  /** Tile period (m) in the mesh frame at scale 1. */
  tileM: number
  /** Capture box in metres at scale 1. */
  y0: number
  y1: number
  /** Coverage-weighted mean apex height (fraction of [y0, y1]). */
  meanH: number
  /** Shell heights, descending (fraction of [y0, y1]). */
  shellY: Float32Array
  /** Per-shell lower height threshold (fraction); the last is -1 (draws all). */
  shellT: Float32Array
  /** Height of the single far-LOD quad — the shell the near band converges to. */
  farH: number
  /** shellY[0] - shellY[SHELLS-2]: the apex band the cavity shade normalises by. */
  reliefSpan: number
  /** Fraction of the tile covered by geometry from above. */
  coverage: number
  albedoLevels: Uint8Array<ArrayBuffer>[]
  auxLevels: Uint8Array<ArrayBuffer>[]
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

function chainTexels(size: number, mips: number): number {
  let total = 0
  for (let l = 0; l < mips; l++) total += (size >> l) * (size >> l)
  return total
}

export function unpackCarpet(buf: ArrayBuffer): CarpetTile | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const tilePx = u[2]!
  const auxPx = u[3]!
  const albMips = u[4]!
  const auxMips = u[5]!
  if (tilePx !== TILE_PX || auxPx !== AUX_PX || albMips !== ALB_MIPS || auxMips !== AUX_MIPS) return null
  const albBytes = chainTexels(tilePx, albMips) * 4
  const auxBytes = chainTexels(auxPx, auxMips) * 4
  if (buf.byteLength !== HEADER_BYTES + albBytes + auxBytes) return null
  const f = new Float32Array(buf, 32, 16)
  const albedoLevels: Uint8Array<ArrayBuffer>[] = []
  let o = HEADER_BYTES
  for (let l = 0; l < albMips; l++) {
    const bytes = (tilePx >> l) * (tilePx >> l) * 4
    albedoLevels.push(new Uint8Array(buf, o, bytes))
    o += bytes
  }
  const auxLevels: Uint8Array<ArrayBuffer>[] = []
  for (let l = 0; l < auxMips; l++) {
    const bytes = (auxPx >> l) * (auxPx >> l) * 4
    auxLevels.push(new Uint8Array(buf, o, bytes))
    o += bytes
  }
  return {
    tilePx,
    auxPx,
    tileM: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    meanH: f[3]!,
    shellY: new Float32Array(buf.slice(48, 48 + 16)),
    shellT: new Float32Array(buf.slice(64, 64 + 16)),
    farH: f[12]!,
    reliefSpan: f[13]!,
    coverage: f[14]!,
    albedoLevels,
    auxLevels,
  }
}

function packCarpet(t: CarpetTile): ArrayBuffer {
  const albBytes = t.albedoLevels.reduce((n, l) => n + l.byteLength, 0)
  const auxBytes = t.auxLevels.reduce((n, l) => n + l.byteLength, 0)
  const buf = new ArrayBuffer(HEADER_BYTES + albBytes + auxBytes)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = t.tilePx
  u[3] = t.auxPx
  u[4] = t.albedoLevels.length
  u[5] = t.auxLevels.length
  const f = new Float32Array(buf, 32, 16)
  f.set([t.tileM, t.y0, t.y1, t.meanH], 0)
  f.set(t.shellY, 4)
  f.set(t.shellT, 8)
  f.set([t.farH, t.reliefSpan, t.coverage, 0], 12)
  let o = HEADER_BYTES
  for (const level of [...t.albedoLevels, ...t.auxLevels]) {
    new Uint8Array(buf, o, level.byteLength).set(level)
    o += level.byteLength
  }
  return buf
}

/** Load (OPFS cache / committed file) or bake the carpet tile for a species. */
export async function loadCarpetTile(ctx: BakeCtx, speciesId: string): Promise<CarpetTile> {
  const key = `carpet-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeCarpetTile(ctx, mesh, speciesById(speciesId).tileM ?? mesh.header.tileSize[0])
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let tile = unpackCarpet(buf)
  if (!tile) {
    // The dev server answers a missing /mesh/baked file with index.html at 200,
    // which poisons the OPFS entry — rebake and repair in place.
    buf = await runBake()
    tile = unpackCarpet(buf)
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
// GPU capture
// ---------------------------------------------------------------------------

async function bakeCarpetTile(ctx: BakeCtx, mesh: GcMesh, tileM: number): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const y0 = Math.min(0, hdr.boundsMin[1])
  const y1 = Math.max(hdr.boundsMax[1], hdr.topH)
  const originX = hdr.tileOrigin[0]
  const originZ = hdr.tileOrigin[1]

  const vbuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/carpet-verts`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
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
  const albTex = mkTarget('carpet-albedo')
  const auxTex = mkTarget('carpet-aux')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/carpet-depth`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // One 256B-strided uniform slot per wrap copy.
  const STRIDE = 256
  const COPIES = 9
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-uni`, size: STRIDE * COPIES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * COPIES)
  for (let c = 0; c < COPIES; c++) {
    const o = (c * STRIDE) / 4
    scratch.set([originX, originZ, tileM, 0], o)
    scratch.set([y0, y1, 0, 0], o + 4)
    scratch.set([hdr.boundsMin[0], hdr.boundsMin[1], hdr.boundsMin[2], 0], o + 8)
    scratch.set(
      [hdr.boundsMax[0] - hdr.boundsMin[0], hdr.boundsMax[1] - hdr.boundsMin[1], hdr.boundsMax[2] - hdr.boundsMin[2], 0],
      o + 12,
    )
    scratch.set([((c % 3) - 1) * tileM, 0, (Math.floor(c / 3) - 1) * tileM, 0], o + 16)
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 80 },
      },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/carpet-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 80 } }],
  })

  const module = ctx.shaders.module(carpetBakeSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/carpet-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/carpet-pl`, bindGroupLayouts: [bgl] }),
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

  // One submission per wrap copy — ~20M triangles each; nine of them in one
  // command buffer is a watchdog risk on weaker GPUs.
  const albView = albTex.createView()
  const auxView = auxTex.createView()
  const depthView = depthTex.createView()
  for (let c = 0; c < COPIES; c++) {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/carpet-enc-${c}` })
    const load = c === 0 ? 'clear' : 'load'
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/carpet-pass-${c}`,
      colorAttachments: [
        { view: albView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: load, storeOp: 'store' },
        { view: auxView, clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: load, storeOp: 'store' },
      ],
      depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: load, depthStoreOp: 'store' },
    })
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    pass.setBindGroup(0, bg, [c * STRIDE])
    pass.drawIndexed(indices.length, 1, 0, 0, 0)
    pass.end()
    device.queue.submit([enc.finish()])
  }
  depthTex.destroy()
  vbuf.destroy()
  ibuf.destroy()

  const bigAlb = await readback(ctx, albTex, 'albedo')
  albTex.destroy()
  const bigAux = await readback(ctx, auxTex, 'aux')
  auxTex.destroy()

  const tile = resolve(bigAlb, bigAux, tileM, y0, y1)
  console.info(
    `[${ctx.id}] carpet tile ${tileM.toFixed(3)}m x ${((y1 - y0) * 1000).toFixed(0)}mm:` +
      ` coverage ${(tile.coverage * 100).toFixed(1)}%, mean apex ${(tile.meanH * (y1 - y0) * 1000).toFixed(1)}mm,` +
      ` shells ${Array.from(tile.shellY, (h) => (h * (y1 - y0) * 1000).toFixed(1)).join('/')}mm,` +
      ` far quad ${(tile.farH * (y1 - y0) * 1000).toFixed(1)}mm`,
  )
  return packCarpet(tile)
}

async function readback(ctx: BakeCtx, tex: GPUTexture, label: string): Promise<Uint8Array> {
  const bpr = BIG * 4
  const rb = ctx.res.createBuffer(
    {
      label: `${ctx.id}/carpet-rb-${label}`,
      size: bpr * BIG,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    },
    { tag: 'bake-scratch' },
  )
  const enc = ctx.device.createCommandEncoder({ label: `${ctx.id}/carpet-copy-${label}` })
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  ctx.device.queue.submit([enc.finish()])
  await rb.mapAsync(GPUMapMode.READ)
  const out = new Uint8Array(rb.getMappedRange()).slice()
  rb.unmap()
  rb.destroy()
  return out
}

// ---------------------------------------------------------------------------
// Resolve: downsample, wrap-aware dilation, mip chains, shell heights
// ---------------------------------------------------------------------------

function resolve(bigAlb: Uint8Array, bigAux: Uint8Array, tileM: number, y0: number, y1: number): CarpetTile {
  // --- albedo level 0: coverage-weighted 2x2 box ----------------------------
  const alb0 = new Uint8Array(TILE_PX * TILE_PX * 4)
  for (let y = 0; y < TILE_PX; y++) {
    for (let x = 0; x < TILE_PX; x++) {
      let n = 0
      let r = 0
      let g = 0
      let b = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * BIG + (x * SS + i)) * 4
          if (bigAlb[s + 3]! === 0) continue
          n++
          r += bigAlb[s]!
          g += bigAlb[s + 1]!
          b += bigAlb[s + 2]!
        }
      }
      const d = (y * TILE_PX + x) * 4
      if (n > 0) {
        alb0[d] = Math.round(r / n)
        alb0[d + 1] = Math.round(g / n)
        alb0[d + 2] = Math.round(b / n)
        alb0[d + 3] = Math.round((n / (SS * SS)) * 255)
      }
    }
  }

  // --- aux level 0: normals + apex height -----------------------------------
  // The leaf-scale normal is averaged as a VECTOR (never as an octahedral pair,
  // which mips toward garbage) and kept only as a minority term; the dominant
  // normal is rebuilt below from the gradient of the height field, which is
  // where the cushion's actual shape lives.
  const aux0 = new Uint8Array(AUX_PX * AUX_PX * 4)
  const micro = new Float32Array(AUX_PX * AUX_PX * 3)
  const height = new Float32Array(AUX_PX * AUX_PX)
  for (let y = 0; y < AUX_PX; y++) {
    for (let x = 0; x < AUX_PX; x++) {
      let n = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let h = 0
      for (let j = 0; j < AUX_SS; j++) {
        for (let i = 0; i < AUX_SS; i++) {
          const s = ((y * AUX_SS + j) * BIG + (x * AUX_SS + i)) * 4
          if (bigAlb[s + 3]! === 0) continue
          n++
          nx += bigAux[s]! / 127.5 - 1
          ny += bigAux[s + 1]! / 127.5 - 1
          nz += bigAux[s + 2]! / 127.5 - 1
          h += bigAux[s + 3]! / 255
        }
      }
      const p = y * AUX_PX + x
      if (n > 0) {
        const len = Math.hypot(nx, ny, nz) || 1
        micro[p * 3] = nx / len
        micro[p * 3 + 1] = ny / len
        micro[p * 3 + 2] = nz / len
        height[p] = h / n
        aux0[p * 4 + 3] = Math.round((n / (AUX_SS * AUX_SS)) * 255)
      } else {
        micro[p * 3 + 1] = 1
      }
    }
  }

  // Cushion normals from the height gradient. Two wrapped binomial passes take
  // the speckle out (a single 0.35mm texel can sit in a gap between two leaves
  // and would otherwise read as a cliff), and the shell test uses the same
  // smoothed field so its boundaries follow capitula instead of noise.
  const smooth = blurWrapped(blurWrapped(height, AUX_PX), AUX_PX)
  const dx = tileM / AUX_PX
  const yScale = y1 - y0
  for (let y = 0; y < AUX_PX; y++) {
    for (let x = 0; x < AUX_PX; x++) {
      const p = y * AUX_PX + x
      const xp = (x + 1) % AUX_PX
      const xm = (x + AUX_PX - 1) % AUX_PX
      const yp = (y + 1) % AUX_PX
      const ym = (y + AUX_PX - 1) % AUX_PX
      const gx = ((smooth[y * AUX_PX + xp]! - smooth[y * AUX_PX + xm]!) * yScale) / (2 * dx)
      const gz = ((smooth[yp * AUX_PX + x]! - smooth[ym * AUX_PX + x]!) * yScale) / (2 * dx)
      let nx = -gx + micro[p * 3]! * MICRO_NORMAL_WEIGHT
      let ny = 1 + micro[p * 3 + 1]! * MICRO_NORMAL_WEIGHT
      let nz = -gz + micro[p * 3 + 2]! * MICRO_NORMAL_WEIGHT
      const len = Math.hypot(nx, ny, nz) || 1
      nx /= len
      ny /= len
      nz /= len
      // ny stays positive by construction, so (nx, nz) reconstructs it exactly.
      aux0[p * 4] = Math.round(Math.max(-1, Math.min(1, nx)) * 127.5 + 127.5)
      aux0[p * 4 + 1] = Math.round(Math.max(-1, Math.min(1, nz)) * 127.5 + 127.5)
      aux0[p * 4 + 2] = Math.round(smooth[p]! * 255)
    }
  }

  dilateWrapped(alb0, TILE_PX, [0, 1, 2])
  dilateWrapped(aux0, AUX_PX, [0, 1, 2])

  // --- shell heights from the height field ----------------------------------
  const hist = new Float64Array(257)
  let covSum = 0
  let hSum = 0
  for (let i = 0; i < AUX_PX * AUX_PX; i++) {
    const cov = aux0[i * 4 + 3]! / 255
    if (cov <= 0) continue
    hist[aux0[i * 4 + 2]!] = hist[aux0[i * 4 + 2]!]! + cov
    covSum += cov
    hSum += (aux0[i * 4 + 2]! / 255) * cov
  }
  const meanH = covSum > 0 ? hSum / covSum : 0.7
  const quantile = (q: number): number => {
    // q = fraction of covered area ABOVE the returned height.
    let acc = 0
    for (let bin = 256; bin >= 0; bin--) {
      acc += hist[bin]!
      if (acc >= q * covSum) return bin / 255
    }
    return 0
  }
  const shellY = new Float32Array(SHELLS)
  for (let s = 0; s < SHELLS - 1; s++) shellY[s] = quantile(SHELL_QUANTILES[s]!)
  shellY[SHELLS - 1] = SHELL_FLOOR
  // Keep them strictly descending even if the height field is degenerate.
  for (let s = 1; s < SHELLS; s++) shellY[s] = Math.min(shellY[s]!, shellY[s - 1]! - 0.02)
  const shellT = new Float32Array(SHELLS)
  for (let s = 0; s < SHELLS - 1; s++) shellT[s] = (shellY[s]! + shellY[s + 1]!) * 0.5
  shellT[SHELLS - 1] = -1
  // The far LOD quad sits on the shell the mip-averaged near band converges to,
  // so the collapse at carpetShellDist moves nothing.
  let farH = shellY[SHELLS - 1]!
  for (let s = 0; s < SHELLS; s++) {
    if (meanH >= shellT[s]!) {
      farH = shellY[s]!
      break
    }
  }

  return {
    tilePx: TILE_PX,
    auxPx: AUX_PX,
    tileM,
    y0,
    y1,
    meanH,
    shellY,
    shellT,
    farH,
    reliefSpan: Math.max(shellY[0]! - shellY[SHELLS - 2]!, 1e-3),
    coverage: covSum / (AUX_PX * AUX_PX),
    albedoLevels: mipChain(alb0, TILE_PX, ALB_MIPS, false),
    auxLevels: mipChain(aux0, AUX_PX, AUX_MIPS, true),
  }
}

/** Wrapped 3x3 binomial blur of a scalar field. */
function blurWrapped(src: Float32Array, size: number): Float32Array {
  const out = new Float32Array(size * size)
  const w = [1, 2, 1]
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let acc = 0
      for (let j = -1; j <= 1; j++) {
        const yy = (y + j + size) % size
        for (let i = -1; i <= 1; i++) {
          acc += src[yy * size + ((x + i + size) % size)]! * w[i + 1]! * w[j + 1]!
        }
      }
      out[y * size + x] = acc / 16
    }
  }
  return out
}

/**
 * Push colour/normal/height outward into empty texels so bilinear taps at a
 * coverage edge never pull the cleared value in. Wrapped, because the texture
 * is periodic and its left edge really does neighbour its right edge.
 */
function dilateWrapped(img: Uint8Array, size: number, channels: number[]): void {
  let filled = new Uint8Array(size * size)
  for (let i = 0; i < size * size; i++) filled[i] = img[i * 4 + 3]! > 0 ? 1 : 0
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = filled.slice()
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (filled[y * size + x]! !== 0) continue
        let n = 0
        const acc = [0, 0, 0]
        for (let j = -1; j <= 1; j++) {
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const yy = (y + j + size) % size
            const xx = (x + i + size) % size
            if (filled[yy * size + xx]! === 0) continue
            n++
            for (let c = 0; c < channels.length; c++) acc[c] = acc[c]! + img[(yy * size + xx) * 4 + channels[c]!]!
          }
        }
        if (n === 0) continue
        for (let c = 0; c < channels.length; c++) img[(y * size + x) * 4 + channels[c]!] = Math.round(acc[c]! / n)
        next[y * size + x] = 1
      }
    }
    filled = next
  }
}

/**
 * Box mip chain. A 2x2 block never crosses the periodic boundary, so no wrap
 * handling is needed here. rgb (or nx/nz/h) is averaged weighted by coverage —
 * i.e. the stored colour is ALREADY normalised at every level and the shader
 * must not divide by alpha — and alpha is the plain mean coverage. For `renorm`
 * (the aux chain) the averaged normal is re-normalised, which is why normals
 * are stored as a plain hemisphere pair rather than octahedral: an octahedral
 * pair averages toward garbage, this does not.
 */
function mipChain(level0: Uint8Array, size: number, mips: number, renorm: boolean): Uint8Array<ArrayBuffer>[] {
  const levels: Uint8Array<ArrayBuffer>[] = [new Uint8Array(level0)]
  for (let l = 1; l < mips; l++) {
    const pw = size >> (l - 1)
    const w = size >> l
    const prev = levels[l - 1]!
    const out = new Uint8Array(w * w * 4)
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        let aSum = 0
        let c0 = 0
        let c1 = 0
        let c2 = 0
        let p0 = 0
        let p1 = 0
        let p2 = 0
        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 2; i++) {
            const s = ((y * 2 + j) * pw + x * 2 + i) * 4
            const a = prev[s + 3]!
            p0 += prev[s]!
            p1 += prev[s + 1]!
            p2 += prev[s + 2]!
            if (a === 0) continue
            aSum += a
            c0 += prev[s]! * a
            c1 += prev[s + 1]! * a
            c2 += prev[s + 2]! * a
          }
        }
        const d = (y * w + x) * 4
        if (aSum > 0) {
          out[d] = Math.round(c0 / aSum)
          out[d + 1] = Math.round(c1 / aSum)
          out[d + 2] = Math.round(c2 / aSum)
          out[d + 3] = Math.round(aSum / 4)
        } else {
          out[d] = Math.round(p0 / 4)
          out[d + 1] = Math.round(p1 / 4)
          out[d + 2] = Math.round(p2 / 4)
        }
        if (renorm) {
          const nx = out[d]! / 127.5 - 1
          const nz = out[d + 1]! / 127.5 - 1
          const r2 = nx * nx + nz * nz
          if (r2 > 1) {
            const k = 1 / Math.sqrt(r2)
            out[d] = Math.round(nx * k * 127.5 + 127.5)
            out[d + 1] = Math.round(nz * k * 127.5 + 127.5)
          }
        }
      }
    }
    levels.push(out)
  }
  return levels
}
