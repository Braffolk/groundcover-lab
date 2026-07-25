import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'

/**
 * Per-species card-proxy bake: two orthographic captures of the source mesh
 * (one SIDE view along +Z, one TOP view along -Y), each 256x256, into two
 * atlases: albedo (rgb + coverage) and normal (plant-local UNIT NORMAL in rgb,
 * n * 0.5 + 0.5, flipped into the hemisphere around that view's capture axis).
 * These two tiles are the ONLY per-species imagery the runtime ever touches —
 * the source mesh is never processed again after this one-time bake.
 *
 * v4 replaced v3's raw octahedral rg normal. Averaging octahedral codes of a
 * blade's two opposing faces cancels to (0,0), which decodes to exactly
 * (0,1,0) — and since every cached cluster samples mip 1.5-3.5, that made the
 * entire far field light with a straight-up normal (flat, blown out, no sun
 * gradient). A flipped plain vector averages linearly and cannot cancel.
 *
 * Covered texels are dilated into empty neighbours on the CPU so runtime
 * bilinear taps never blend toward black. A CPU mip chain (alpha-weighted)
 * feeds the far-cluster reconstruction where a whole plant is ~1 slot texel.
 */

export const TILE = 256
export const ATLAS_W = TILE * 2
export const ATLAS_H = TILE
export const MIPS = 6
export const BAKE_VERSION = 4
const MAGIC = 0x31434343 // 'CCC1'
const HEADER_BYTES = 64

export interface CardsMeta {
  center: [number, number, number]
  /** Side view half extents (m) about the centre: X (right) and Y (up). */
  sideHalfW: number
  sideHalfH: number
  /** Top view half extents (m): X (right) and Z (v axis, -Z up). */
  topHalfW: number
  topHalfH: number
  yMin: number
  yMax: number
}

export interface CardsBaked {
  meta: CardsMeta
  albedo: Uint8Array // ATLAS_W * ATLAS_H * 4 — rgb = albedo, a = coverage
  /** ATLAS_W * ATLAS_H * 4 — rgb = plant-local unit normal * 0.5 + 0.5. */
  normal: Uint8Array
}

export function packCards(b: CardsBaked): ArrayBuffer {
  const blob = ATLAS_W * ATLAS_H * 4
  const buf = new ArrayBuffer(HEADER_BYTES + blob * 2)
  const u = new Uint32Array(buf, 0, 4)
  u[0] = MAGIC
  u[1] = BAKE_VERSION
  u[2] = TILE
  const f = new Float32Array(buf, 16, 12)
  const m = b.meta
  f.set([...m.center, m.sideHalfW, m.sideHalfH, m.topHalfW, m.topHalfH, m.yMin, m.yMax], 0)
  new Uint8Array(buf, HEADER_BYTES, blob).set(b.albedo)
  new Uint8Array(buf, HEADER_BYTES + blob, blob).set(b.normal)
  return buf
}

/** Returns null unless the buffer is a valid artifact of this exact version. */
export function unpackCards(buf: ArrayBuffer): CardsBaked | null {
  const blob = ATLAS_W * ATLAS_H * 4
  if (buf.byteLength !== HEADER_BYTES + blob * 2) return null
  const u = new Uint32Array(buf, 0, 4)
  if (u[0] !== MAGIC || u[1] !== BAKE_VERSION || u[2] !== TILE) return null
  const f = new Float32Array(buf, 16, 12)
  return {
    meta: {
      center: [f[0]!, f[1]!, f[2]!],
      sideHalfW: f[3]!,
      sideHalfH: f[4]!,
      topHalfW: f[5]!,
      topHalfH: f[6]!,
      yMin: f[7]!,
      yMax: f[8]!,
    },
    albedo: new Uint8Array(buf, HEADER_BYTES, blob),
    normal: new Uint8Array(buf, HEADER_BYTES + blob, blob),
  }
}

/**
 * Grow covered texels into empty ones (alpha stays 0) so bilinear sampling at
 * silhouettes blends between real colours instead of toward black.
 */
function dilate(albedo: Uint8Array, normal: Uint8Array, x0: number, iterations = 8): void {
  const mask = new Uint8Array(TILE * TILE)
  for (let y = 0; y < TILE; y++) {
    for (let x = 0; x < TILE; x++) {
      mask[y * TILE + x] = albedo[(y * ATLAS_W + x0 + x) * 4 + 3]! > 0 ? 1 : 0
    }
  }
  for (let it = 0; it < iterations; it++) {
    const grown = mask.slice()
    for (let y = 0; y < TILE; y++) {
      for (let x = 0; x < TILE; x++) {
        if (mask[y * TILE + x]!) continue
        for (let n = 0; n < 4; n++) {
          const nx = x + [1, -1, 0, 0][n]!
          const ny = y + [0, 0, 1, -1][n]!
          if (nx < 0 || ny < 0 || nx >= TILE || ny >= TILE || !mask[ny * TILE + nx]!) continue
          const src = ((ny * ATLAS_W) + x0 + nx) * 4
          const dst = ((y * ATLAS_W) + x0 + x) * 4
          for (let c = 0; c < 3; c++) {
            albedo[dst + c] = albedo[src + c]!
            normal[dst + c] = normal[src + c]!
          }
          grown[y * TILE + x] = 1
          break
        }
      }
    }
    mask.set(grown)
  }
}

/** Alpha-weighted 2x box downsample chain. Returns levels 0..MIPS-1. */
export function buildMips(base: Uint8Array): { data: Uint8Array; w: number; h: number }[] {
  const levels: { data: Uint8Array; w: number; h: number }[] = [{ data: base, w: ATLAS_W, h: ATLAS_H }]
  for (let l = 1; l < MIPS; l++) {
    const prev = levels[l - 1]!
    const w = Math.max(1, prev.w >> 1)
    const h = Math.max(1, prev.h >> 1)
    const data = new Uint8Array(w * h * 4)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0
        let g = 0
        let b = 0
        let a = 0
        for (let s = 0; s < 4; s++) {
          const i = ((y * 2 + (s >> 1)) * prev.w + x * 2 + (s & 1)) * 4
          const al = prev.data[i + 3]!
          r += prev.data[i]! * al
          g += prev.data[i + 1]! * al
          b += prev.data[i + 2]! * al
          a += al
        }
        const o = (y * w + x) * 4
        if (a > 0) {
          data[o] = Math.round(r / a)
          data[o + 1] = Math.round(g / a)
          data[o + 2] = Math.round(b / a)
        }
        data[o + 3] = Math.round(a / 4)
      }
    }
    levels.push({ data, w, h })
  }
  return levels
}

/** One-time GPU render of the source mesh into the two capture tiles. */
export async function bakeCards(ctx: ExperimentContext, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: [number, number, number] = [
    (bmin[0] + bmax[0]) / 2,
    (bmin[1] + bmax[1]) / 2,
    (bmin[2] + bmax[2]) / 2,
  ]
  const hx = (bmax[0] - bmin[0]) / 2
  const hy = (bmax[1] - bmin[1]) / 2
  const hz = (bmax[2] - bmin[2]) / 2
  const meta: CardsMeta = {
    center,
    sideHalfW: Math.max(hx, hz) * 1.02,
    sideHalfH: hy * 1.02,
    topHalfW: Math.max(hx, hz) * 1.02,
    topHalfH: Math.max(hx, hz) * 1.02,
    yMin: bmin[1],
    yMax: bmax[1],
  }

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

  const mkTarget = (label: string): GPUTexture =>
    device.createTexture({
      label: `${ctx.id}/${label}`,
      size: [ATLAS_W, ATLAS_H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    })
  const albTex = mkTarget('bake-albedo')
  const nrmTex = mkTarget('bake-normal')
  const depthTex = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [ATLAS_W, ATLAS_H],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  // Per-view basis: [right | halfW, up | halfH, fwd | sMax, center, bmin, bmax]
  const STRIDE = 256
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * 2,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new Float32Array((STRIDE / 4) * 2)
  const views = [
    // side: capture along +Z (camera at -Z looking +Z... fwd = axis toward camera)
    { right: [1, 0, 0], up: [0, 1, 0], fwd: [0, 0, 1], hw: meta.sideHalfW, hh: meta.sideHalfH, s: hz * 1.05 },
    // top: camera above looking down; card u = +X, v axis = -Z
    { right: [1, 0, 0], up: [0, 0, -1], fwd: [0, 1, 0], hw: meta.topHalfW, hh: meta.topHalfH, s: hy * 1.05 },
  ]
  views.forEach((v, i) => {
    const o = (i * STRIDE) / 4
    scratch.set([...v.right, v.hw], o)
    scratch.set([...v.up, v.hh], o + 4)
    scratch.set([...v.fwd, v.s], o + 8)
    scratch.set([...center, 0], o + 12)
    scratch.set([bmin[0], bmin[1], bmin[2], 0], o + 16)
    scratch.set([bmax[0], bmax[1], bmax[2], 0], o + 20)
  })
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
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
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  for (let i = 0; i < 2; i++) {
    pass.setViewport(i * TILE, 0, TILE, TILE, 0, 1)
    pass.setScissorRect(i * TILE, 0, TILE, TILE)
    pass.setBindGroup(0, bg, [i * STRIDE])
    pass.drawIndexed(indices.length)
  }
  pass.end()

  const bpr = ATLAS_W * 4 // 2048, multiple of 256
  const rbSize = bpr * ATLAS_H
  const rbs = [albTex, nrmTex].map((tex) => {
    const buf = device.createBuffer({
      label: `${ctx.id}/bake-rb`,
      size: rbSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: ATLAS_H }, [
      ATLAS_W,
      ATLAS_H,
    ])
    return buf
  })
  device.queue.submit([enc.finish()])

  const albedo = new Uint8Array(rbSize)
  const normal = new Uint8Array(rbSize)
  await rbs[0]!.mapAsync(GPUMapMode.READ)
  albedo.set(new Uint8Array(rbs[0]!.getMappedRange()))
  rbs[0]!.unmap()
  await rbs[1]!.mapAsync(GPUMapMode.READ)
  normal.set(new Uint8Array(rbs[1]!.getMappedRange()))
  rbs[1]!.unmap()

  dilate(albedo, normal, 0)
  dilate(albedo, normal, TILE)

  const packed = packCards({ meta, albedo, normal })
  for (const r of [vbuf, ibuf, uni, albTex, nrmTex, depthTex, ...rbs]) r.destroy()
  return packed
}
