import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake the depth-augmented view set for one species from its GCMESH1 source
 * mesh: a GRID x GRID hemi-octahedral fan of orthographic captures, each a
 * TILE x TILE cell in two atlases:
 *
 *   albedo  rgba8unorm  — authored rgb + coverage
 *   geom    rgba16float — signed relief height h = dot(unit_pos, fwd),
 *                         oct-encoded local normal (flipped toward capture)
 *
 * After readback, empty texels bordering covered ones are DILATED (records
 * copied, coverage kept 0) so runtime parallax reprojections that land just
 * outside a silhouette read a plausible height/color instead of background.
 * Baked fresh in-browser per session (~25 draws over the source mesh); the
 * harness bakedArtifact() cache is intentionally not used — the dev server's
 * SPA fallback answers missing /mesh/baked files with 200 index.html, which
 * poisons the cache (see 004/005 NOTES; harness limitation).
 */

export const GRID = 5
export const TILE = 256
export const ATLAS = GRID * TILE // 1280

export interface BakedViews {
  center: [number, number, number]
  radius: number
  halfExt: [number, number, number]
  topH: number
  albedo: Uint8Array // ATLAS*ATLAS*4 rgba8
  geom: Uint8Array // ATLAS*ATLAS*8 rgba16float raw
}

type V3 = [number, number, number]
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** Hemi-octahedral decode — MUST match hemioct_decode() in relief.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const px = (ex + ey) * 0.5
  const pz = (ex - ey) * 0.5
  const y = 1 - Math.abs(px) - Math.abs(pz)
  return norm([px, y, pz])
}

/** Capture basis for grid node (i, j) — MUST match node_basis() in relief.wgsl. */
export function nodeBasis(i: number, j: number, grid: number): { f: V3; x: V3; y: V3 } {
  const ex = (i / (grid - 1)) * 2 - 1
  const ey = (j / (grid - 1)) * 2 - 1
  const f = hemioctDecode(ex, ey)
  const upRef: V3 = Math.abs(f[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0]
  const x = norm(cross(upRef, f))
  const y = norm(cross(f, x))
  return { f, x, y }
}

/**
 * Dilate covered texel records into empty neighbours, per tile (never across
 * tile seams), `iters` rings. Copies raw bytes — no f16 decode needed.
 * Coverage (albedo alpha) stays 0 so alpha-testing is unaffected.
 */
function dilate(albedo: Uint8Array, geom: Uint8Array, iters: number): void {
  const covered = new Uint8Array(ATLAS * ATLAS)
  for (let i = 0; i < ATLAS * ATLAS; i++) covered[i] = albedo[i * 4 + 3]! > 0 ? 1 : 0
  const next = new Uint8Array(ATLAS * ATLAS)
  const geo16 = new Uint16Array(geom.buffer, geom.byteOffset, ATLAS * ATLAS * 4)
  for (let it = 0; it < iters; it++) {
    next.set(covered)
    for (let tj = 0; tj < GRID; tj++) {
      for (let ti = 0; ti < GRID; ti++) {
        const x0 = ti * TILE
        const y0 = tj * TILE
        for (let y = y0; y < y0 + TILE; y++) {
          for (let x = x0; x < x0 + TILE; x++) {
            const idx = y * ATLAS + x
            if (covered[idx]) continue
            // First covered 4-neighbour inside the tile wins.
            let src = -1
            if (x > x0 && covered[idx - 1]) src = idx - 1
            else if (x < x0 + TILE - 1 && covered[idx + 1]) src = idx + 1
            else if (y > y0 && covered[idx - ATLAS]) src = idx - ATLAS
            else if (y < y0 + TILE - 1 && covered[idx + ATLAS]) src = idx + ATLAS
            if (src < 0) continue
            albedo[idx * 4] = albedo[src * 4]!
            albedo[idx * 4 + 1] = albedo[src * 4 + 1]!
            albedo[idx * 4 + 2] = albedo[src * 4 + 2]!
            // alpha stays 0
            geo16[idx * 4] = geo16[src * 4]!
            geo16[idx * 4 + 1] = geo16[src * 4 + 1]!
            geo16[idx * 4 + 2] = geo16[src * 4 + 2]!
            geo16[idx * 4 + 3] = geo16[src * 4 + 3]!
            next[idx] = 1
          }
        }
      }
    }
    covered.set(next)
  }
}

/** Render all GRID^2 captures on the GPU and read both atlases back. */
export async function bakeViews(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<BakedViews> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])

  // --- transient GPU resources (destroyed at the end, not VRAM-tracked) ---
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
  const geoTex = device.createTexture({
    label: `${ctx.id}/bake-geo`,
    size: [ATLAS, ATLAS],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const depthTex = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [ATLAS, ATLAS],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  // Per-tile uniform ring (96B used, 256B stride for dynamic offsets).
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
      const { f, x, y } = nodeBasis(i, j, GRID)
      const o = ((j * GRID + i) * STRIDE) / 4
      fv[o + 0] = center[0]; fv[o + 1] = center[1]; fv[o + 2] = center[2]; fv[o + 3] = 1 / radius
      fv[o + 4] = x[0]; fv[o + 5] = x[1]; fv[o + 6] = x[2]
      fv[o + 8] = y[0]; fv[o + 9] = y[1]; fv[o + 10] = y[2]
      fv[o + 12] = f[0]; fv[o + 13] = f[1]; fv[o + 14] = f[2]
      fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
      fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
    }
  }
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
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/bake-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      // Empty background height = -1 (far side) so rays passing through gaps
      // keep travelling instead of stopping at the mid-plane.
      { view: geoTex.createView(), clearValue: { r: -1, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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

  const bprAlb = ATLAS * 4 // 5120, /256 ok
  const bprGeo = ATLAS * 8 // 10240, /256 ok
  const rbAlb = device.createBuffer({
    label: `${ctx.id}/rb-alb`,
    size: bprAlb * ATLAS,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const rbGeo = device.createBuffer({
    label: `${ctx.id}/rb-geo`,
    size: bprGeo * ATLAS,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bprAlb, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
  enc.copyTextureToBuffer({ texture: geoTex }, { buffer: rbGeo, bytesPerRow: bprGeo, rowsPerImage: ATLAS }, [ATLAS, ATLAS])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbGeo.mapAsync(GPUMapMode.READ)
  const albedo = new Uint8Array(bprAlb * ATLAS)
  const geom = new Uint8Array(bprGeo * ATLAS)
  albedo.set(new Uint8Array(rbAlb.getMappedRange()))
  geom.set(new Uint8Array(rbGeo.getMappedRange()))
  rbAlb.unmap()
  rbGeo.unmap()
  for (const r of [vbuf, ibuf, albTex, geoTex, depthTex, uni, rbAlb, rbGeo]) r.destroy()

  dilate(albedo, geom, 4)

  const halfExt: V3 = [(bmax[0] - bmin[0]) / 2, (bmax[1] - bmin[1]) / 2, (bmax[2] - bmin[2]) / 2]
  return { center, radius, halfExt, topH: mesh.header.topH || bmax[1], albedo, geom }
}
