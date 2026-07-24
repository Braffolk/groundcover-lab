import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake a hemi-octahedral impostor atlas from a GCMESH1 source mesh.
 *
 * The upper hemisphere of viewing directions is parameterised by a GRID x GRID
 * hemi-octahedral grid; each node is one orthographic capture rendered into a
 * TILE x TILE cell of two atlases: albedo+coverage and a local-frame normal.
 * The result is packed into a single ArrayBuffer (small header + both blobs)
 * so it can be cached / committed and re-uploaded without touching geometry.
 */

export const GRID = 12
export const TILE = 128
export const ATLAS = GRID * TILE // 1536
const HEADER_FLOATS = 8

export interface AtlasMeta {
  grid: number
  tile: number
  atlasPx: number
  center: [number, number, number]
  radius: number
}

export interface BakedAtlas {
  meta: AtlasMeta
  albedo: Uint8Array // ATLAS*ATLAS*4
  normal: Uint8Array // ATLAS*ATLAS*4
}

// --- tiny vec helpers (avoid pulling in a matrix lib) -----------------------
type V3 = [number, number, number]
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** Hemi-octahedral decode — MUST match hemioct_decode() in impostor.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const px = (ex + ey) * 0.5
  const pz = (ex - ey) * 0.5
  const y = 1 - Math.abs(px) - Math.abs(pz)
  return norm([px, y, pz])
}

export function packAtlas(a: BakedAtlas): ArrayBuffer {
  const headerBytes = HEADER_FLOATS * 4
  const blob = a.albedo.byteLength
  const buf = new ArrayBuffer(headerBytes + blob * 2)
  const h = new Float32Array(buf, 0, HEADER_FLOATS)
  h[0] = a.meta.grid
  h[1] = a.meta.tile
  h[2] = a.meta.atlasPx
  h[3] = a.meta.center[0]
  h[4] = a.meta.center[1]
  h[5] = a.meta.center[2]
  h[6] = a.meta.radius
  h[7] = 0
  new Uint8Array(buf, headerBytes, blob).set(a.albedo)
  new Uint8Array(buf, headerBytes + blob, blob).set(a.normal)
  return buf
}

export function unpackAtlas(buf: ArrayBuffer): BakedAtlas {
  const h = new Float32Array(buf, 0, HEADER_FLOATS)
  const headerBytes = HEADER_FLOATS * 4
  const atlasPx = h[2]!
  const blob = atlasPx * atlasPx * 4
  return {
    meta: { grid: h[0]!, tile: h[1]!, atlasPx, center: [h[3]!, h[4]!, h[5]!], radius: h[6]! },
    albedo: new Uint8Array(buf, headerBytes, blob),
    normal: new Uint8Array(buf, headerBytes + blob, blob),
  }
}

/** Render the atlas on the GPU and read it back to CPU. */
export async function bakeAtlas(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const invR = 1 / radius

  // --- transient GPU resources (destroyed at the end) ---
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

  const albTex = device.createTexture({
    label: `${ctx.id}/bake-alb`,
    size: [ATLAS, ATLAS],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const nrmTex = device.createTexture({
    label: `${ctx.id}/bake-nrm`,
    size: [ATLAS, ATLAS],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const depthTex = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [ATLAS, ATLAS],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  // Per-tile uniform ring: 96 bytes of data padded to 256 for dynamic offset.
  const STRIDE = 256
  const nTiles = GRID * GRID
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * nTiles,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * nTiles)
  const fv = new Float32Array(scratch)
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      const ex = (i / (GRID - 1)) * 2 - 1
      const ey = (j / (GRID - 1)) * 2 - 1
      const camDir = hemioctDecode(ex, ey) // z axis (points toward camera)
      const upRef: V3 = Math.abs(camDir[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0]
      const right = norm(cross(upRef, camDir))
      const up = norm(cross(camDir, right))
      const o = ((j * GRID + i) * STRIDE) / 4
      // Tile struct: center,inv_r | right | up | fwd | bmin | bmax
      fv[o + 0] = center[0]; fv[o + 1] = center[1]; fv[o + 2] = center[2]; fv[o + 3] = invR
      fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]
      fv[o + 8] = up[0]; fv[o + 9] = up[1]; fv[o + 10] = up[2]
      fv[o + 12] = camDir[0]; fv[o + 13] = camDir[1]; fv[o + 14] = camDir[2]
      fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
      fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
    }
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } }],
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
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
    },
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
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      pass.setViewport(i * TILE, j * TILE, TILE, TILE, 0, 1)
      pass.setScissorRect(i * TILE, j * TILE, TILE, TILE)
      pass.setBindGroup(0, bg, [(j * GRID + i) * STRIDE])
      pass.drawIndexed(indices.length)
    }
  }
  pass.end()

  // Copy both atlases into readback buffers (bytesPerRow = ATLAS*4 = 6144, /256 ok).
  const bpr = ATLAS * 4
  const rbSize = bpr * ATLAS
  const rbAlb = device.createBuffer({ label: `${ctx.id}/rb-alb`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const rbNrm = device.createBuffer({ label: `${ctx.id}/rb-nrm`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
  enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const albedo = new Uint8Array(rbSize)
  const normal = new Uint8Array(rbSize)
  albedo.set(new Uint8Array(rbAlb.getMappedRange()))
  normal.set(new Uint8Array(rbNrm.getMappedRange()))
  rbAlb.unmap()
  rbNrm.unmap()

  const packed = packAtlas({ meta: { grid: GRID, tile: TILE, atlasPx: ATLAS, center, radius }, albedo, normal })

  // Release transient GPU memory.
  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()
  return packed
}
