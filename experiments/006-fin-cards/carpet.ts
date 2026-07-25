import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Carpet bake — the fin atlas re-spent for a periodic ground mat.
 *
 * A Sphagnum tile is 0.18 m across and 0.07-0.09 m tall, so the 6 azimuthal
 * fin views are nearly worthless for it: each 512px tile would hold a 512x135
 * sliver of content (74% of the tile empty) showing a 7 cm-tall fence, and
 * three of those crossed at 60 degrees is exactly the "vertical cards through a
 * mat" failure. The same VRAM buys something a mat actually shows:
 *
 *   MOSS_LEVELS horizontal CROSS-SECTIONS through the cushion. Level k is a
 *   top-down orthographic capture of everything ABOVE height y_k, and it is
 *   drawn at y_k. Stacked with a hard alpha test and depth writes, the highest
 *   level that is opaque at a given xz *is* the cushion surface there, so the
 *   stack is a terraced height field of the real capitulum relief (3.3 cm on
 *   the wet-vigorous state) — thickness, silhouette and parallax that a single
 *   ground quad cannot express. Level 0 sits just above the peat and holds the
 *   full top view, which makes the mat a closed, depth-writing occluder.
 *
 * Framing is the tile's own square [0, tileM]^2 in the mesh frame (tile origin
 * is (0,0) for every source mesh), NOT the bounds diagonal: 512px over 0.18 m
 * is 0.35 mm/texel, ~4x the on-screen texel density of a diagonal-framed
 * capture cropped to the tile. Levels live in a texture_2d_array with REPEAT
 * addressing, not an atlas — the tile is periodic, so wrapping is the correct
 * filter at its boundary, and per-layer mip chains never bleed between views.
 * The mesh's own overflow (bounds run -0.013..0.197 for a 0.18 m period) is the
 * periodic image of the opposite edge, so cutting at the square loses nothing.
 *
 * The second texture is NOT the mesh's authored normals. Sampled over 200k
 * vertices, a Sphagnum tile's normals are isotropic — mean n.y = -0.001, mean
 * |n.y| = 0.49 (a uniform sphere gives 0.5), 35% of them pointing DOWN — because
 * the geometry is a mass of tiny leaves facing every direction. Averaging that
 * per texel cancels to near zero, and normalising the remainder yields noise:
 * the mat lit as blotchy dark speckle, half of it shaded from behind. What a
 * texel of aggregate cushion actually needs is the normal of the cushion's own
 * MACRO surface, and the capture already contains it — the bake's depth buffer
 * IS the height field of the visible surface at that level. So the normal
 * texture stores (nx, nz) of the smoothed height gradient plus a baked
 * crevice-occlusion term, and n.y is recovered as sqrt(1 - nx^2 - nz^2) (a
 * height-field normal always points up). Storing the gradient rather than an
 * octahedral pair also makes the chain honestly mip-averageable.
 *
 * There is exactly ONE normal map for the whole stack, at half resolution.
 * That is not a saving, it is correctness: wherever cross-section k is opaque
 * there IS geometry above y_k, so the topmost surface at that texel is the same
 * surface level 0 sees — the levels differ only in WHICH texels they cover.
 * Shading each card from its own level made the close-up worse, because a high
 * level's height field is mostly capitulum crowns and its occlusion term
 * flattens out; one macro-normal map from the complete surface keeps every card
 * shaded like the cushion it belongs to. (By the same argument the per-level
 * albedo is redundant too — it is kept because reading colour and mask from one
 * tap is cheaper than a third texture fetch, and the budget has room.)
 */

/** Bump only when the carpet artifact layout changes. */
export const MOSS_BAKE_VERSION = 3

export const MOSS_LEVELS = 6
export const MOSS_TILE = 512
export const MOSS_NRM_TILE = 256
export const MOSS_ALB_MIPS = 5 // 512 -> 32
export const MOSS_NRM_MIPS = 4 // 256 -> 32
const COLS = 3
const ROWS = 2
const ATLAS_W = COLS * MOSS_TILE // 1536
const ATLAS_H = ROWS * MOSS_TILE // 1024

/**
 * Cross-section heights as a fraction of the mesh's Y extent.
 *
 * From the source manifests, all three states put their capitulum apices
 * between 0.55 and 0.92 of the tile height (mean 0.74), so that band is where
 * the relief lives and gets five of the six levels. Level 0 sits at 0.12 — just
 * above the peat, below every capitulum — so its mask is the full top view and
 * the mat cannot open a hole.
 */
const LEVEL_FRACS = [0.12, 0.5, 0.62, 0.72, 0.82, 0.91] as const
/** Mean capitulum apex height (manifest canopy.capitulumApexMeanH / topH). */
const APEX_FRAC = 0.74

export interface MossMeta {
  /** Periodic tile size (m) the capture was framed to. */
  tileM: number
  /** Mesh-frame Y of each cross-section (mesh y=0 is the ground plane). */
  levelY: number[]
  /** Mesh-frame Y of the mean capitulum apex — the far-LOD surface height. */
  apexY: number
}

export interface BakedMoss {
  meta: MossMeta
  /** [cross-section][mip], mip 0 = MOSS_TILE^2, premultiplied rgb. */
  albedo: Uint8Array<ArrayBuffer>[][]
  /** One shared macro-surface map, [mip], mip 0 = MOSS_NRM_TILE^2. */
  normal: Uint8Array<ArrayBuffer>[]
}

type V3 = [number, number, number]

const MAGIC = 0x31_53_53_4d // 'MSS1'
const HEADER_BYTES = 64

const chainBytes = (px: number, levels: number): number => {
  let total = 0
  for (let l = 0; l < levels; l++) total += Math.max(1, px >> l) ** 2 * 4
  return total
}
const ARTIFACT_BYTES =
  HEADER_BYTES + MOSS_LEVELS * chainBytes(MOSS_TILE, MOSS_ALB_MIPS) + chainBytes(MOSS_NRM_TILE, MOSS_NRM_MIPS)

export function packMoss(baked: BakedMoss): ArrayBuffer {
  const buf = new ArrayBuffer(ARTIFACT_BYTES)
  new Uint32Array(buf, 0, 2).set([MAGIC, MOSS_BAKE_VERSION])
  new Float32Array(buf, 8, 8).set([baked.meta.tileM, ...baked.meta.levelY, baked.meta.apexY])
  const bytes = new Uint8Array(buf)
  let at = HEADER_BYTES
  for (const chain of [...baked.albedo, baked.normal]) {
    for (const mip of chain) {
      bytes.set(mip, at)
      at += mip.byteLength
    }
  }
  return buf
}

/** Null on any size/magic mismatch (e.g. an SPA-fallback HTML response). */
export function unpackMoss(buf: ArrayBuffer): BakedMoss | null {
  if (buf.byteLength !== ARTIFACT_BYTES) return null
  const head = new Uint32Array(buf, 0, 2)
  if (head[0] !== MAGIC || head[1] !== MOSS_BAKE_VERSION) return null
  const f = new Float32Array(buf, 8, 8)
  let at = HEADER_BYTES
  const take = (px: number, levels: number): Uint8Array<ArrayBuffer>[] => {
    const out: Uint8Array<ArrayBuffer>[] = []
    for (let l = 0; l < levels; l++) {
      const n = Math.max(1, px >> l) ** 2 * 4
      out.push(new Uint8Array(buf, at, n))
      at += n
    }
    return out
  }
  const albedo: Uint8Array<ArrayBuffer>[][] = []
  for (let k = 0; k < MOSS_LEVELS; k++) albedo.push(take(MOSS_TILE, MOSS_ALB_MIPS))
  const normal = take(MOSS_NRM_TILE, MOSS_NRM_MIPS)
  return {
    meta: { tileM: f[0]!, levelY: [f[1]!, f[2]!, f[3]!, f[4]!, f[5]!, f[6]!], apexY: f[7]! },
    albedo,
    normal,
  }
}

/**
 * Premultiplied mip chain of one square tile lifted out of the readback.
 * Level 0 rgb is multiplied by its 0/1 coverage; each next level is a plain
 * 2x2 box average of premultiplied rgb + alpha, so the stored rgb is already
 * coverage-normalised and the shader un-premultiplies with a single divide.
 * `skip` drops the first `skip` levels (that is how the half-res normal chain
 * is produced without a second render pass — the two MRT targets must share a
 * viewport, so both are rendered full-res and the normal starts at level 1).
 */
function tileMips(
  src: Uint8Array,
  srcW: number,
  x0: number,
  y0: number,
  px: number,
  levels: number,
  skip: number,
): Uint8Array<ArrayBuffer>[] {
  let cur = new Float32Array(px * px * 4)
  for (let y = 0; y < px; y++) {
    for (let x = 0; x < px; x++) {
      const s = ((y0 + y) * srcW + x0 + x) * 4
      const d = (y * px + x) * 4
      const a = src[s + 3]! / 255
      cur[d] = (src[s]! / 255) * a
      cur[d + 1] = (src[s + 1]! / 255) * a
      cur[d + 2] = (src[s + 2]! / 255) * a
      cur[d + 3] = a
    }
  }
  const quantize = (v: Float32Array, n: number): Uint8Array<ArrayBuffer> => {
    const out = new Uint8Array(n * 4)
    for (let i = 0; i < n * 4; i++) out[i] = Math.round(Math.min(1, Math.max(0, v[i]!)) * 255)
    return out
  }
  const out: Uint8Array<ArrayBuffer>[] = []
  let w = px
  if (skip === 0) out.push(quantize(cur, w * w))
  for (let l = 1; l < levels + skip; l++) {
    const nw = Math.max(1, w >> 1)
    const next = new Float32Array(nw * nw * 4)
    for (let y = 0; y < nw; y++) {
      for (let x = 0; x < nw; x++) {
        const s0 = (y * 2 * w + x * 2) * 4
        const s1 = s0 + 4
        const s2 = s0 + w * 4
        const s3 = s2 + 4
        const d = (y * nw + x) * 4
        for (let c = 0; c < 4; c++) next[d + c] = (cur[s0 + c]! + cur[s1 + c]! + cur[s2 + c]! + cur[s3 + c]!) * 0.25
      }
    }
    cur = next
    w = nw
    if (l >= skip) out.push(quantize(cur, w * w))
  }
  return out
}

/**
 * Macro-surface normal + crevice occlusion of one captured level, from the
 * capture's own depth buffer.
 *
 * `depth` is the bake pass's depth tile: 0.5 - (y - yc)/halfH * 0.5, i.e. the
 * height of the topmost surface at or above this level, and 1.0 where nothing
 * was hit. All stencils WRAP, because the tile is periodic — that is what keeps
 * the normal field seamless across tile boundaries.
 *
 * Returns level 0 of the normal texture: rg = (nx, nz) of the smoothed height
 * gradient, b = occlusion, a = coverage.
 */
function surfaceNormalTile(
  depth: Float32Array,
  cover: Uint8Array,
  px: number,
  yc: number,
  halfH: number,
  texelM: number,
): Uint8Array<ArrayBuffer> {
  const wrap = (i: number): number => (i + px) % px
  const at = (x: number, z: number): number => wrap(z) * px + wrap(x)
  const h = new Float32Array(px * px)
  const w = new Float32Array(px * px)
  for (let i = 0; i < px * px; i++) {
    w[i] = cover[i * 4 + 3]! > 127 ? 1 : 0
    h[i] = yc + (1 - 2 * depth[i]!) * halfH
  }
  // Coverage-weighted 5x5 box: kills leaf-scale roughness (a leaf is 1-3 texels)
  // while keeping the capitulum dome, which spans ~28 texels at 0.35 mm/texel.
  const smooth = (src: Float32Array, radius: number): Float32Array => {
    const out = new Float32Array(px * px)
    const tmp = new Float32Array(px * px)
    const tmpW = new Float32Array(px * px)
    for (let z = 0; z < px; z++) {
      for (let x = 0; x < px; x++) {
        let s = 0
        let sw = 0
        for (let d = -radius; d <= radius; d++) {
          const i = at(x + d, z)
          s += src[i]! * w[i]!
          sw += w[i]!
        }
        tmp[z * px + x] = s
        tmpW[z * px + x] = sw
      }
    }
    for (let z = 0; z < px; z++) {
      for (let x = 0; x < px; x++) {
        let s = 0
        let sw = 0
        for (let d = -radius; d <= radius; d++) {
          const i = at(x, z + d)
          s += tmp[i]!
          sw += tmpW[i]!
        }
        out[z * px + x] = sw > 0 ? s / sw : 0
      }
    }
    return out
  }
  const hs = smooth(h, 2)
  // Wide mean (~1.1 cm) as the local surface datum: below it is a crevice
  // between capitula, above it is a capitulum crown.
  const hWide = smooth(hs, 16)

  const MAX_SLOPE = 3.73 // tan(75deg): a terrace cliff must not flip the normal
  const out = new Uint8Array(px * px * 4)
  for (let z = 0; z < px; z++) {
    for (let x = 0; x < px; x++) {
      const i = z * px + x
      // Central differences at +-2 texels, matching the smoothing radius.
      const dx = (hs[at(x + 2, z)]! - hs[at(x - 2, z)]!) / (4 * texelM)
      const dz = (hs[at(x, z + 2)]! - hs[at(x, z - 2)]!) / (4 * texelM)
      const gx = Math.max(-MAX_SLOPE, Math.min(MAX_SLOPE, dx))
      const gz = Math.max(-MAX_SLOPE, Math.min(MAX_SLOPE, dz))
      const inv = 1 / Math.hypot(gx, 1, gz)
      const rel = (hs[i]! - hWide[i]!) / Math.max(halfH, 1e-4)
      const ao = Math.max(0.42, Math.min(1, 0.78 + 2.2 * rel))
      out[i * 4] = Math.round((-gx * inv * 0.5 + 0.5) * 255)
      out[i * 4 + 1] = Math.round((-gz * inv * 0.5 + 0.5) * 255)
      out[i * 4 + 2] = Math.round(ao * 255)
      out[i * 4 + 3] = w[i]! > 0 ? 255 : 0
    }
  }
  return out
}

/**
 * Render the MOSS_LEVELS cross-sections of a periodic tile and build their mip
 * chains. `tileM` is the species' periodic footprint (speciesById().tileM) —
 * the capture is framed to that square, never to the mesh bounds.
 */
export async function bakeMoss(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  tileM: number,
): Promise<BakedMoss> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const dy = bmax[1] - bmin[1]
  const halfTile = tileM * 0.5
  const halfH = 0.5 * dy * 1.02
  const yc = (bmin[1] + bmax[1]) / 2
  const meta: MossMeta = {
    tileM,
    levelY: LEVEL_FRACS.map((f) => bmin[1] + f * dy),
    apexY: bmin[1] + APEX_FRAC * dy,
  }

  // --- transient GPU resources (destroyed at the end, not budget-tracked) ---
  const vbuf = device.createBuffer({
    label: `${ctx.id}/moss-verts`,
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = device.createBuffer({
    label: `${ctx.id}/moss-idx`,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (label: string, format: GPUTextureFormat, usage: number): GPUTexture =>
    device.createTexture({ label, size: [ATLAS_W, ATLAS_H], format, usage })
  const albTex = mkTarget(`${ctx.id}/moss-alb`, 'rgba8unorm', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC)
  const nrmTex = mkTarget(`${ctx.id}/moss-nrm`, 'rgba8unorm', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC)
  // The depth buffer is read back too: it is the height field of each level's
  // visible surface, which is where the macro normals come from.
  const depthTex = mkTarget(
    `${ctx.id}/moss-depth`,
    'depth32float',
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  )

  // Per-level uniform ring: 96 bytes of data padded to 256 for dynamic offsets.
  const STRIDE = 256
  const uni = device.createBuffer({
    label: `${ctx.id}/moss-uni`,
    size: STRIDE * MOSS_LEVELS,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * MOSS_LEVELS)
  const fv = new Float32Array(scratch)
  // Top-down capture of the tile square. up = (0,0,-1) so texture v maps to
  // +local z; fwd = +Y so nearer-the-sky geometry wins the depth test; the
  // slab filter keeps everything at or above the level's height.
  const right: V3 = [1, 0, 0]
  const up: V3 = [0, 0, -1]
  const fwd: V3 = [0, 1, 0]
  for (let k = 0; k < MOSS_LEVELS; k++) {
    const o = (k * STRIDE) / 4
    fv[o + 0] = halfTile; fv[o + 1] = yc; fv[o + 2] = halfTile; fv[o + 3] = 1 / halfTile
    fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]; fv[o + 7] = 1 / halfTile
    fv[o + 8] = fwd[0]; fv[o + 9] = fwd[1]; fv[o + 10] = fwd[2]; fv[o + 11] = 1 / halfH
    fv[o + 12] = up[0]; fv[o + 13] = up[1]; fv[o + 14] = up[2]; fv[o + 15] = 0
    fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]; fv[o + 19] = meta.levelY[k]!
    fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]; fv[o + 23] = bmax[1] + 1
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/moss-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/moss-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 96 } }],
  })

  const module = ctx.shaders.module(bakeShaderSrc)
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
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  for (let k = 0; k < MOSS_LEVELS; k++) {
    const col = k % COLS
    const row = Math.floor(k / COLS)
    pass.setViewport(col * MOSS_TILE, row * MOSS_TILE, MOSS_TILE, MOSS_TILE, 0, 1)
    pass.setScissorRect(col * MOSS_TILE, row * MOSS_TILE, MOSS_TILE, MOSS_TILE)
    pass.setBindGroup(0, bg, [k * STRIDE])
    pass.drawIndexed(indices.length)
  }
  pass.end()

  const bpr = ATLAS_W * 4 // 6144, multiple of 256
  const rbSize = bpr * ATLAS_H
  const mkReadback = (label: string): GPUBuffer =>
    device.createBuffer({ label, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const rbAlb = mkReadback(`${ctx.id}/moss-rb-alb`)
  const rbDepth = mkReadback(`${ctx.id}/moss-rb-depth`)
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
  enc.copyTextureToBuffer(
    { texture: depthTex, aspect: 'depth-only' },
    { buffer: rbDepth, bytesPerRow: bpr, rowsPerImage: ATLAS_H },
    [ATLAS_W, ATLAS_H],
  )
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbDepth.mapAsync(GPUMapMode.READ)
  const albedo0 = new Uint8Array(rbAlb.getMappedRange().slice(0))
  const depth0 = new Float32Array(rbDepth.getMappedRange().slice(0))
  rbAlb.unmap()
  rbDepth.unmap()

  const albedo: Uint8Array<ArrayBuffer>[][] = []
  for (let k = 0; k < MOSS_LEVELS; k++) {
    albedo.push(tileMips(albedo0, ATLAS_W, (k % COLS) * MOSS_TILE, Math.floor(k / COLS) * MOSS_TILE, MOSS_TILE, MOSS_ALB_MIPS, 0))
  }
  // The macro surface comes from cross-section 0 — the complete height field of
  // the cushion. Every card shades from it (see the header comment).
  const depth = new Float32Array(MOSS_TILE * MOSS_TILE)
  const cover = new Uint8Array(MOSS_TILE * MOSS_TILE * 4)
  for (let y = 0; y < MOSS_TILE; y++) {
    const src = y * ATLAS_W
    depth.set(depth0.subarray(src, src + MOSS_TILE), y * MOSS_TILE)
    cover.set(albedo0.subarray(src * 4, (src + MOSS_TILE) * 4), y * MOSS_TILE * 4)
  }
  const nrm0 = surfaceNormalTile(depth, cover, MOSS_TILE, yc, halfH, tileM / MOSS_TILE)
  const normal = tileMips(nrm0, MOSS_TILE, 0, 0, MOSS_TILE, MOSS_NRM_MIPS, 1)

  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbDepth]) r.destroy()
  return { meta, albedo, normal }
}
