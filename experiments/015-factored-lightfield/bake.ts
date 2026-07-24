import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake one species into the factored light-field representation:
 *
 *  1. GPU: a 24x24 hemi-octahedral grid of orthographic DEPTH+COVERAGE
 *     captures. Each view is rasterized 2x2 supersampled (256px) into a
 *     one-row strip, then downsampled to its final 128px tile storing
 *     min-depth (8-bit, 1.0 = miss sentinel) and fractional coverage in one
 *     rg8unorm 3072x3072 atlas (18.9MB). Fractional coverage is what keeps
 *     sub-texel fluff (calamagrostis seed heads) wispy instead of ballooning
 *     to its convex hull. Work is chunked into several submits so huge
 *     meshes (poa is 6.5M tris) never risk a device timeout.
 *  2. CPU: view-INDEPENDENT appearance volumes — every source vertex (2M-6.5M
 *     of them; denser than the voxel grid) is splatted into a <=96^3
 *     aspect-fit voxel grid, averaged, then dilated so any quantized hit
 *     point reconstructed at runtime lands on valid albedo/normal data.
 *     albedo.a stores per-voxel luminance sigma, re-injected at runtime as
 *     deterministic speckle so voxel-averaged interiors aren't flat.
 *
 * Naive 4D storage at the same angular/spatial resolution (albedo+normal+
 * depth+coverage, ~10B/ray) would be ~94MB; the factorization stores ~21MB.
 */

export const GRID = 24
export const TILE = 128
export const ATLAS = GRID * TILE // 3072
const SS = 2 // supersampling factor of the bake rasterization
const TILE_SS = TILE * SS
const STRIP_W = GRID * TILE_SS // 6144
const VOL_MAX = 96
const DILATE_PASSES = 4
/** Cap triangles per GPU submit during the bake (watchdog safety). */
const TRIS_PER_SUBMIT = 1.6e8

export interface QlfBaked {
  depthAtlas: GPUTexture
  albedoVol: GPUTexture
  normalVol: GPUTexture
  center: [number, number, number]
  radius: number
  bmin: [number, number, number]
  bmax: [number, number, number]
  heightM: number
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

/** Hemi-octahedral decode — MUST match hemioct_decode() in qlf.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const px = (ex + ey) * 0.5
  const pz = (ex - ey) * 0.5
  const y = 1 - Math.abs(px) - Math.abs(pz)
  return norm([px, y, pz])
}

// ---------------------------------------------------------------------------
// CPU half: view-independent appearance volumes from the raw vertices.
// ---------------------------------------------------------------------------

interface Volumes {
  dims: [number, number, number]
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
}

const ACC = 9 // r,g,b, nx,ny,nz, count, lum, lum^2

function bakeVolumes(mesh: GcMesh): Volumes {
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const ext: V3 = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]]
  const maxExt = Math.max(ext[0], ext[1], ext[2])
  const dims = ext.map((e) => Math.max(8, Math.min(VOL_MAX, Math.round((e / maxExt) * VOL_MAX)))) as [
    number,
    number,
    number,
  ]
  const [dx, dy, dz] = dims
  const n = dx * dy * dz

  let acc = new Float32Array(n * ACC)
  const verts = mesh.vertices
  const count = mesh.header.vertexCount
  let sumR = 0
  let sumG = 0
  let sumB = 0
  for (let v = 0; v < count; v++) {
    const b = v * 8
    // Quantized position is already normalized against the bounds — voxel
    // coordinates come straight from the u16s, no decode needed.
    const ix = Math.min(dx - 1, ((verts[b]! / 65535) * dx) | 0)
    const iy = Math.min(dy - 1, ((verts[b + 1]! / 65535) * dy) | 0)
    const iz = Math.min(dz - 1, ((verts[b + 2]! / 65535) * dz) | 0)
    const o = ((iz * dy + iy) * dx + ix) * ACC
    const r = verts[b + 3]! / 65535
    const g = verts[b + 4]! / 65535
    const bl = verts[b + 5]! / 65535
    // Octahedral normal decode (mirrors GcMesh.normalAt, allocation-free).
    const ou = (verts[b + 6]! / 65535) * 2 - 1
    const ov = (verts[b + 7]! / 65535) * 2 - 1
    let nx = ou
    let nz = ov
    const nyRaw = 1 - Math.abs(ou) - Math.abs(ov)
    if (nyRaw < 0) {
      nx = (1 - Math.abs(ov)) * Math.sign(ou)
      nz = (1 - Math.abs(ou)) * Math.sign(ov)
    }
    const nl = Math.hypot(nx, nyRaw, nz) || 1
    const lum = 0.299 * r + 0.587 * g + 0.114 * bl
    acc[o] = acc[o]! + r
    acc[o + 1] = acc[o + 1]! + g
    acc[o + 2] = acc[o + 2]! + bl
    acc[o + 3] = acc[o + 3]! + nx / nl
    acc[o + 4] = acc[o + 4]! + nyRaw / nl
    acc[o + 5] = acc[o + 5]! + nz / nl
    acc[o + 6] = acc[o + 6]! + 1
    acc[o + 7] = acc[o + 7]! + lum
    acc[o + 8] = acc[o + 8]! + lum * lum
    sumR += r
    sumG += g
    sumB += bl
  }
  const avg: V3 = count > 0 ? [sumR / count, sumG / count, sumB / count] : [0.3, 0.4, 0.2]

  // Dilate: fill empty voxels with the mean of occupied 6-neighbours, a few
  // passes, so trilinear samples near quantized hit points never read holes.
  const stride: V3 = [ACC, dx * ACC, dx * dy * ACC]
  for (let pass = 0; pass < DILATE_PASSES; pass++) {
    const next = acc.slice()
    for (let iz = 0; iz < dz; iz++) {
      for (let iy = 0; iy < dy; iy++) {
        for (let ix = 0; ix < dx; ix++) {
          const o = ((iz * dy + iy) * dx + ix) * ACC
          if (acc[o + 6]! > 0) continue
          let k = 0
          const sums = [0, 0, 0, 0, 0, 0, 0, 0] // r,g,b,nx,ny,nz,lum,lum2
          for (let axis = 0; axis < 3; axis++) {
            const idx = axis === 0 ? ix : axis === 1 ? iy : iz
            const lim = axis === 0 ? dx : axis === 1 ? dy : dz
            for (const s of [-1, 1]) {
              if (idx + s < 0 || idx + s >= lim) continue
              const no = o + s * stride[axis]!
              const c = acc[no + 6]!
              if (c <= 0) continue
              for (let f = 0; f < 6; f++) sums[f]! += acc[no + f]! / c
              sums[6]! += acc[no + 7]! / c
              sums[7]! += acc[no + 8]! / c
              k++
            }
          }
          if (k > 0) {
            for (let f = 0; f < 6; f++) next[o + f] = sums[f]!
            next[o + 6] = k
            next[o + 7] = sums[6]!
            next[o + 8] = sums[7]!
          }
        }
      }
    }
    acc = next
  }

  const albedo = new Uint8Array(n * 4)
  const normal = new Uint8Array(n * 4)
  for (let i = 0; i < n; i++) {
    const o = i * ACC
    const c = acc[o + 6]!
    const t = i * 4
    if (c > 0) {
      albedo[t] = Math.min(255, (acc[o]! / c) * 255) | 0
      albedo[t + 1] = Math.min(255, (acc[o + 1]! / c) * 255) | 0
      albedo[t + 2] = Math.min(255, (acc[o + 2]! / c) * 255) | 0
      const meanL = acc[o + 7]! / c
      const sigma = Math.sqrt(Math.max(acc[o + 8]! / c - meanL * meanL, 0))
      albedo[t + 3] = Math.min(255, sigma * 2 * 255) | 0 // a = sigma * 2
      const nx = acc[o + 3]!
      const ny = acc[o + 4]!
      const nz = acc[o + 5]!
      const l = Math.hypot(nx, ny, nz)
      // Averaged two-sided foliage normals can cancel; fall back to up.
      const ux = l > 1e-4 ? nx / l : 0
      const uy = l > 1e-4 ? ny / l : 1
      const uz = l > 1e-4 ? nz / l : 0
      normal[t] = ((ux * 0.5 + 0.5) * 255) | 0
      normal[t + 1] = ((uy * 0.5 + 0.5) * 255) | 0
      normal[t + 2] = ((uz * 0.5 + 0.5) * 255) | 0
      normal[t + 3] = 255
    } else {
      albedo[t] = (avg[0] * 255) | 0
      albedo[t + 1] = (avg[1] * 255) | 0
      albedo[t + 2] = (avg[2] * 255) | 0
      albedo[t + 3] = 50
      normal[t] = 128
      normal[t + 1] = 255
      normal[t + 2] = 128
      normal[t + 3] = 0
    }
  }
  return { dims, albedo, normal }
}

// ---------------------------------------------------------------------------
// GPU half: the 576-view supersampled ortho depth+coverage light field.
// ---------------------------------------------------------------------------

export async function bakeSpecies(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  mesh: GcMesh,
): Promise<QlfBaked> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const invR = 1 / radius

  const t0 = performance.now()

  // Persistent runtime textures (counted against the species budget).
  const depthAtlas = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/depth-lf`,
      size: [ATLAS, ATLAS],
      format: 'rg8unorm', // r = min depth (1 = miss), g = fractional coverage
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    },
    { species: speciesId, tag: 'depth-lightfield' },
  )

  const vols = bakeVolumes(mesh)
  const [dx, dy, dz] = vols.dims
  const albedoVol = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo-vol`,
      size: [dx, dy, dz],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { species: speciesId, tag: 'albedo-volume' },
  )
  const normalVol = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/normal-vol`,
      size: [dx, dy, dz],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { species: speciesId, tag: 'normal-volume' },
  )
  device.queue.writeTexture({ texture: albedoVol }, vols.albedo, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz])
  device.queue.writeTexture({ texture: normalVol }, vols.normal, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz])

  // Transient bake resources (destroyed below; destroy() un-counts them).
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'bake-transient' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'bake-transient' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)
  // One-row strip render targets at supersampled resolution.
  const stripColor = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-strip`,
      size: [STRIP_W, TILE_SS],
      format: 'r8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    },
    { species: speciesId, tag: 'bake-transient' },
  )
  const stripDepth = ctx.res.createTexture(
    { label: `${ctx.id}/bake-strip-depth`, size: [STRIP_W, TILE_SS], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT },
    { species: speciesId, tag: 'bake-transient' },
  )

  // Per-tile basis uniforms (dynamic offset ring, 256B stride).
  const STRIDE = 256
  const nTiles = GRID * GRID
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * nTiles, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'bake-transient' },
  )
  const scratch = new ArrayBuffer(STRIDE * nTiles)
  const fv = new Float32Array(scratch)
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      const ex = (i / (GRID - 1)) * 2 - 1
      const ey = (j / (GRID - 1)) * 2 - 1
      const fwd = hemioctDecode(ex, ey)
      const upRef: V3 = Math.abs(fwd[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0]
      const right = norm(cross(upRef, fwd))
      const up = norm(cross(fwd, right))
      const o = ((j * GRID + i) * STRIDE) / 4
      fv[o] = center[0]; fv[o + 1] = center[1]; fv[o + 2] = center[2]; fv[o + 3] = invR
      fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]
      fv[o + 8] = up[0]; fv[o + 9] = up[1]; fv[o + 10] = up[2]
      fv[o + 12] = fwd[0]; fv[o + 13] = fwd[1]; fv[o + 14] = fwd[2]
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
      buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'uint16x4' }] }],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'r8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  // Downsample: strip (2x2 supersampled) -> final atlas row (min depth + coverage).
  const downBgl = device.createBindGroupLayout({
    label: `${ctx.id}/down-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
  })
  const downBg = device.createBindGroup({
    label: `${ctx.id}/down-bg`,
    layout: downBgl,
    entries: [{ binding: 0, resource: stripColor.createView() }],
  })
  const downPipeline = device.createRenderPipeline({
    label: `${ctx.id}/down-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/down-pl`, bindGroupLayouts: [downBgl] }),
    vertex: { module, entryPoint: 'vs_down' },
    fragment: { module, entryPoint: 'fs_down', targets: [{ format: 'rg8unorm' }] },
    primitive: { topology: 'triangle-list' },
  })

  const tilesPerSubmit = Math.max(2, Math.min(GRID, Math.floor(TRIS_PER_SUBMIT / mesh.header.triangleCount)))
  const stripColorView = stripColor.createView()
  const stripDepthView = stripDepth.createView()
  const atlasView = depthAtlas.createView()
  for (let j = 0; j < GRID; j++) {
    // 1. Rasterize row j's 24 views into the supersampled strip (chunked).
    for (let start = 0; start < GRID; start += tilesPerSubmit) {
      const end = Math.min(GRID, start + tilesPerSubmit)
      const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/bake-strip-pass`,
        colorAttachments: [
          {
            view: stripColorView,
            // r=1 is the "miss" sentinel — hits write <= 254/255.
            clearValue: { r: 1, g: 1, b: 1, a: 1 },
            loadOp: start === 0 ? 'clear' : 'load',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: stripDepthView,
          depthClearValue: 1,
          depthLoadOp: start === 0 ? 'clear' : 'load',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      for (let i = start; i < end; i++) {
        pass.setViewport(i * TILE_SS, 0, TILE_SS, TILE_SS, 0, 1)
        pass.setScissorRect(i * TILE_SS, 0, TILE_SS, TILE_SS)
        pass.setBindGroup(0, bg, [(j * GRID + i) * STRIDE])
        pass.drawIndexed(indices.length)
      }
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    // 2. Downsample the strip into atlas row j.
    const enc = device.createCommandEncoder({ label: `${ctx.id}/down-enc` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/down-pass`,
      colorAttachments: [
        {
          view: atlasView,
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: j === 0 ? 'clear' : 'load',
          storeOp: 'store',
        },
      ],
    })
    pass.setPipeline(downPipeline)
    pass.setViewport(0, j * TILE, ATLAS, TILE, 0, 1)
    pass.setScissorRect(0, j * TILE, ATLAS, TILE)
    pass.setBindGroup(0, downBg)
    pass.draw(3)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }

  for (const r of [vbuf, ibuf, stripColor, stripDepth, uni]) r.destroy()
  console.log(
    `[${ctx.id}] baked ${speciesId}: ${nTiles} views (${SS}x${SS} ss) + ${dx}x${dy}x${dz} volumes in ${(performance.now() - t0).toFixed(0)}ms`,
  )

  return {
    depthAtlas,
    albedoVol,
    normalVol,
    center,
    radius,
    bmin: [bmin[0], bmin[1], bmin[2]],
    bmax: [bmax[0], bmax[1], bmax[2]],
    heightM: bmax[1] - bmin[1],
  }
}
