import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake the 8-view fin atlas from a GCMESH1 source mesh.
 *
 * Layout: one 2048x1024 atlas of 4x2 tiles, 512px each.
 *   tiles 0..5  azimuthal orthographic captures at yaw = k*60 deg
 *   tile  6     top-down capture of the LOWER vertical slab
 *   tile  7     top-down capture of the UPPER vertical slab
 * Two atlases (albedo+coverage, mesh-local normal), each with a full CPU-built
 * premultiplied mip chain (MIPS levels) so distant cards don't shimmer.
 */

export const TILE = 512
export const COLS = 4
export const ROWS = 2
export const ATLAS_W = COLS * TILE // 2048
export const ATLAS_H = ROWS * TILE // 1024
export const MIPS = 5 // 512px tile -> 32px tile
const N_TILES = 8
const MARGIN = 1.05 // transparent border so mip filtering never clips content
const SPLIT_FRAC = 0.42 // slab split height (fraction of plant height)
const Y_LOW_FRAC = 0.2 // resting height of the lower horizontal card
const Y_HIGH_FRAC = 0.62 // resting height of the upper horizontal card

export interface FinMeta {
  /** Local-space framing center (XZ of bounds center, Y mid-height). */
  cx: number
  yc: number
  cz: number
  /** Margined half extents: fins are 2*halfW wide, 2*halfH tall. */
  halfW: number
  halfH: number
  /** Local heights of the two horizontal slab cards. */
  yLow: number
  yHigh: number
  /** Bounding radius for near/far fades. */
  radius: number
}

export interface BakedFins {
  meta: FinMeta
  /** MIPS levels each, level 0 = ATLAS_W x ATLAS_H, premultiplied rgb. */
  albedoMips: Uint8Array<ArrayBuffer>[]
  normalMips: Uint8Array<ArrayBuffer>[]
}

type V3 = [number, number, number]

/**
 * Premultiplied mip chain. Level 0 rgb is multiplied by its 0/1 coverage;
 * every next level is a plain 2x2 box average of premultiplied rgb + alpha.
 * True (unboosted) alpha is stored — the runtime `alphaSharp` param sharpens
 * coverage at test time, and rgb is un-premultiplied in the shader.
 */
function buildMips(level0: Uint8Array, w: number, h: number): Uint8Array<ArrayBuffer>[] {
  let cur = new Float32Array(w * h * 4)
  for (let i = 0; i < w * h; i++) {
    const a = level0[i * 4 + 3]! / 255
    cur[i * 4] = (level0[i * 4]! / 255) * a
    cur[i * 4 + 1] = (level0[i * 4 + 1]! / 255) * a
    cur[i * 4 + 2] = (level0[i * 4 + 2]! / 255) * a
    cur[i * 4 + 3] = a
  }
  const out: Uint8Array<ArrayBuffer>[] = []
  let cw = w
  let ch = h
  const quantize = (src: Float32Array, n: number): Uint8Array<ArrayBuffer> => {
    const bytes = new Uint8Array(n * 4)
    for (let i = 0; i < n * 4; i++) bytes[i] = Math.round(Math.min(1, Math.max(0, src[i]!)) * 255)
    return bytes
  }
  out.push(quantize(cur, cw * ch))
  for (let level = 1; level < MIPS; level++) {
    const nw = Math.max(1, cw >> 1)
    const nh = Math.max(1, ch >> 1)
    const next = new Float32Array(nw * nh * 4)
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        const s0 = (y * 2 * cw + x * 2) * 4
        const s1 = s0 + 4
        const s2 = s0 + cw * 4
        const s3 = s2 + 4
        const d = (y * nw + x) * 4
        for (let c = 0; c < 4; c++) {
          next[d + c] = (cur[s0 + c]! + cur[s1 + c]! + cur[s2 + c]! + cur[s3 + c]!) * 0.25
        }
      }
    }
    out.push(quantize(next, nw * nh))
    cur = next
    cw = nw
    ch = nh
  }
  return out
}

/** Render the 8 views on the GPU, read back, build mips. */
export async function bakeFins(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<BakedFins> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const dx = bmax[0] - bmin[0]
  const dy = bmax[1] - bmin[1]
  const dz = bmax[2] - bmin[2]
  const cx = (bmin[0] + bmax[0]) / 2
  const cz = (bmin[2] + bmax[2]) / 2
  const yc = (bmin[1] + bmax[1]) / 2
  // Half the XZ diagonal fits the content along ANY horizontal axis.
  const halfW = 0.5 * Math.hypot(dx, dz) * MARGIN
  const halfH = 0.5 * dy * MARGIN
  const ySplit = bmin[1] + SPLIT_FRAC * dy
  const meta: FinMeta = {
    cx,
    yc,
    cz,
    halfW,
    halfH,
    yLow: bmin[1] + Y_LOW_FRAC * dy,
    yHigh: bmin[1] + Y_HIGH_FRAC * dy,
    radius: Math.hypot(halfW, halfH),
  }

  // --- transient GPU resources (destroyed at the end, not budget-tracked) ---
  const vbuf = device.createBuffer({
    label: `${ctx.id}/bake-verts`,
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = device.createBuffer({
    label: `${ctx.id}/bake-idx`,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (label: string, format: GPUTextureFormat, usage: number): GPUTexture =>
    device.createTexture({ label, size: [ATLAS_W, ATLAS_H], format, usage })
  const albTex = mkTarget(`${ctx.id}/bake-alb`, 'rgba8unorm', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC)
  const nrmTex = mkTarget(`${ctx.id}/bake-nrm`, 'rgba8unorm', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC)
  const depthTex = mkTarget(`${ctx.id}/bake-depth`, 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT)

  // Per-tile uniform ring: 96 bytes of data padded to 256 for dynamic offsets.
  const STRIDE = 256
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * N_TILES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * N_TILES)
  const fv = new Float32Array(scratch)
  const writeTile = (
    t: number,
    right: V3,
    invHu: number,
    up: V3,
    invHv: number,
    fwd: V3,
    invHd: number,
    slabY0: number,
    slabY1: number,
  ): void => {
    const o = (t * STRIDE) / 4
    fv[o + 0] = cx; fv[o + 1] = yc; fv[o + 2] = cz; fv[o + 3] = invHu
    fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]; fv[o + 7] = invHv
    fv[o + 8] = fwd[0]; fv[o + 9] = fwd[1]; fv[o + 10] = fwd[2]; fv[o + 11] = invHd
    fv[o + 12] = up[0]; fv[o + 13] = up[1]; fv[o + 14] = up[2]; fv[o + 15] = 0
    fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]; fv[o + 19] = slabY0
    fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]; fv[o + 23] = slabY1
  }
  for (let k = 0; k < 6; k++) {
    const phi = (k * Math.PI) / 3
    const fwd: V3 = [Math.sin(phi), 0, Math.cos(phi)]
    const right: V3 = [Math.cos(phi), 0, -Math.sin(phi)]
    // Full y range — no slab filter for azimuth views.
    writeTile(k, right, 1 / halfW, [0, 1, 0], 1 / halfH, fwd, 1 / halfW, bmin[1] - 1, bmax[1] + 1)
  }
  // Top-down captures. up = (0,0,-1) so runtime texture v maps to +local z.
  writeTile(6, [1, 0, 0], 1 / halfW, [0, 0, -1], 1 / halfW, [0, 1, 0], 1 / halfH, bmin[1] - 1, ySplit)
  writeTile(7, [1, 0, 0], 1 / halfW, [0, 0, -1], 1 / halfW, [0, 1, 0], 1 / halfH, ySplit, bmax[1] + 1)
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
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
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
    ],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  })
  pass.setPipeline(pipeline)
  pass.setVertexBuffer(0, vbuf)
  pass.setIndexBuffer(ibuf, 'uint32')
  for (let t = 0; t < N_TILES; t++) {
    const col = t % COLS
    const row = Math.floor(t / COLS)
    pass.setViewport(col * TILE, row * TILE, TILE, TILE, 0, 1)
    pass.setScissorRect(col * TILE, row * TILE, TILE, TILE)
    pass.setBindGroup(0, bg, [t * STRIDE])
    pass.drawIndexed(indices.length)
  }
  pass.end()

  const bpr = ATLAS_W * 4 // 8192, multiple of 256
  const rbSize = bpr * ATLAS_H
  const rbAlb = device.createBuffer({ label: `${ctx.id}/rb-alb`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const rbNrm = device.createBuffer({ label: `${ctx.id}/rb-nrm`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const albedo0 = new Uint8Array(rbAlb.getMappedRange().slice(0))
  const normal0 = new Uint8Array(rbNrm.getMappedRange().slice(0))
  rbAlb.unmap()
  rbNrm.unmap()

  const baked: BakedFins = {
    meta,
    albedoMips: buildMips(albedo0, ATLAS_W, ATLAS_H),
    normalMips: buildMips(normal0, ATLAS_W, ATLAS_H),
  }

  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()
  return baked
}
