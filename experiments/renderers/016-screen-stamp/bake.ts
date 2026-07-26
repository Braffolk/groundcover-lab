import bakeShaderSrc from './shaders/bake.wgsl'
import bakeCarpetSrc from './shaders/bake-carpet.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake one species into a hemi-octahedral impostor atlas: GRID x GRID
 * orthographic captures of the source mesh, each TILE px, into two rgba8
 * atlases (albedo+coverage, local-frame normal). A CPU box-filtered,
 * coverage-weighted mip chain is built from the readback (tiles are
 * power-of-two so mips never bleed across views), plus the coverage-weighted
 * average albedo used by the far-field tint.
 *
 * Capture pipeline adapted from 005-octa-impostors (rated working there);
 * everything downstream of the readback is new.
 */

export const GRID = 10
export const TILE = 128
export const ATLAS = GRID * TILE // 1280
export const MIPS = 4

export interface BakedSpecies {
  center: [number, number, number]
  radius: number
  /** MIPS levels each, level 0 is ATLAS x ATLAS rgba8. */
  albedoMips: Uint8Array<ArrayBuffer>[]
  normalMips: Uint8Array<ArrayBuffer>[]
  avgColor: [number, number, number]
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

/** Hemi-octahedral decode — MUST match hemioct_decode() in resolve.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const px = (ex + ey) * 0.5
  const pz = (ex - ey) * 0.5
  const y = 1 - Math.abs(px) - Math.abs(pz)
  return norm([px, y, pz])
}

/**
 * 2x box downsample. `premult` treats rgb as coverage-premultiplied (albedo:
 * empty texels are black, plain averaging IS the premultiplied filter);
 * otherwise rgb is coverage-weighted (normals).
 */
function downsample(src: Uint8Array, w: number, premult: boolean): Uint8Array<ArrayBuffer> {
  const hw = w >> 1
  const dst = new Uint8Array(hw * hw * 4)
  for (let y = 0; y < hw; y++) {
    for (let x = 0; x < hw; x++) {
      let r = 0
      let g = 0
      let b = 0
      let a = 0
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const i = ((y * 2 + dy) * w + (x * 2 + dx)) * 4
          const al = premult ? 255 : src[i + 3]!
          r += src[i]! * al
          g += src[i + 1]! * al
          b += src[i + 2]! * al
          a += src[i + 3]!
        }
      }
      const o = (y * hw + x) * 4
      const div = premult ? 255 * 4 : Math.max(a, 1)
      dst[o] = Math.round(r / div)
      dst[o + 1] = a > 0 || premult ? Math.round(g / div) : 128
      dst[o + 2] = Math.round(b / div)
      dst[o + 3] = Math.round(a / 4)
    }
  }
  return dst
}

function mipChain(level0: Uint8Array<ArrayBuffer>, premult: boolean): Uint8Array<ArrayBuffer>[] {
  const out = [level0]
  let w = ATLAS
  for (let l = 1; l < MIPS; l++) {
    out.push(downsample(out[l - 1]!, w, premult))
    w >>= 1
  }
  return out
}

function averageColor(albedo: Uint8Array): [number, number, number] {
  let r = 0
  let g = 0
  let b = 0
  let a = 0
  for (let i = 0; i < albedo.length; i += 4) {
    const al = albedo[i + 3]!
    r += albedo[i]! * al
    g += albedo[i + 1]! * al
    b += albedo[i + 2]! * al
    a += al
  }
  if (a === 0) return [0.3, 0.4, 0.2]
  return [r / a / 255, g / a / 255, b / a / 255]
}

// ---------------------------------------------------------------------------
// CARPET BAKE — a mat is a different problem from an upright plant, so it gets
// a different budget split: ONE straight-down capture of the tile's own square
// at high resolution instead of 100 hemi-octa views of a sphere-fitted card.
// See shaders/bake-carpet.wgsl for why the draw is instanced 9x.
// ---------------------------------------------------------------------------

export const CARPET_TEX = 1024
/** 1024 -> 2px. The deepest levels are what a distant mat actually samples. */
export const CARPET_MIPS = 10

export interface BakedCarpet {
  /** Periodic tile size (m, mesh frame) the capture is cropped to. */
  tileM: number
  /** Cushion base y and y extent in the mesh frame (scale 1). */
  yMin: number
  yRange: number
  /** rgb = albedo (coverage-weighted, NOT premultiplied), a = coverage. */
  albedoMips: Uint8Array<ArrayBuffer>[]
  /** rg = mesh-frame normal xz (biased), b = height/yRange, a = coverage. */
  nhMips: Uint8Array<ArrayBuffer>[]
  avgColor: [number, number, number]
  /** Coverage-weighted mean of the stored height (0..1) — the layer's own height. */
  meanH01: number
  /** Mean coverage of level 0 / of the deepest level — the mat-solidity check. */
  coverage0: number
  coverageN: number
}

/**
 * Coverage-weighted 2x box downsample: rgb = sum(rgb*a)/sum(a), a = mean(a).
 * The output colour is therefore ALREADY normalised by coverage at every level
 * — the runtime must NOT divide by alpha again (the 017 inflation trap).
 * Uncovered texels fall back to a neutral value so a fully empty texel decodes
 * to a flat up-normal / zero height rather than garbage.
 */
function downsampleWeighted(src: Uint8Array, w: number, empty: [number, number, number]): Uint8Array<ArrayBuffer> {
  const hw = w >> 1
  const dst = new Uint8Array(hw * hw * 4)
  for (let y = 0; y < hw; y++) {
    for (let x = 0; x < hw; x++) {
      let r = 0
      let g = 0
      let b = 0
      let a = 0
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const i = ((y * 2 + dy) * w + (x * 2 + dx)) * 4
          const al = src[i + 3]!
          r += src[i]! * al
          g += src[i + 1]! * al
          b += src[i + 2]! * al
          a += al
        }
      }
      const o = (y * hw + x) * 4
      if (a > 0) {
        dst[o] = Math.round(r / a)
        dst[o + 1] = Math.round(g / a)
        dst[o + 2] = Math.round(b / a)
      } else {
        dst[o] = empty[0]
        dst[o + 1] = empty[1]
        dst[o + 2] = empty[2]
      }
      dst[o + 3] = Math.round(a / 4)
    }
  }
  return dst
}

/** Coverage-weighted mean of one channel, 0..1. */
function meanChannel(px: Uint8Array, ch: number): number {
  let s = 0
  let a = 0
  for (let i = 0; i < px.length; i += 4) {
    s += px[i + ch]! * px[i + 3]!
    a += px[i + 3]!
  }
  return a === 0 ? 0 : s / a / 255
}

function meanAlpha(px: Uint8Array): number {
  let a = 0
  for (let i = 3; i < px.length; i += 4) a += px[i]!
  return a / (px.length / 4) / 255
}

/** Mean luma of the stored rgb per level — flat means "no divide belongs here". */
function meanLuma(px: Uint8Array): number {
  let s = 0
  let n = 0
  for (let i = 0; i < px.length; i += 4) {
    if (px[i + 3]! > 8) {
      s += 0.299 * px[i]! + 0.587 * px[i + 1]! + 0.114 * px[i + 2]!
      n++
    }
  }
  return n === 0 ? 0 : s / n / 255
}

export async function bakeCarpet(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  fallbackTileM: number,
): Promise<BakedCarpet> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const tileM = mesh.header.tileSize[0] > 0 ? mesh.header.tileSize[0] : fallbackTileM
  const tileMin: [number, number] = [mesh.header.tileOrigin[0], mesh.header.tileOrigin[1]]
  const yMin = bmin[1]
  const yRange = Math.max(bmax[1] - bmin[1], 1e-4)

  const vbuf = device.createBuffer({
    label: `${ctx.id}/carpet-verts`,
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = device.createBuffer({
    label: `${ctx.id}/carpet-idx`,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (label: string): GPUTexture =>
    device.createTexture({
      label: `${ctx.id}/${label}`,
      size: [CARPET_TEX, CARPET_TEX],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    })
  const albTex = mkTarget('carpet-alb')
  const nhTex = mkTarget('carpet-nh')
  const depthTex = device.createTexture({
    label: `${ctx.id}/carpet-depth`,
    size: [CARPET_TEX, CARPET_TEX],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  const uni = device.createBuffer({
    label: `${ctx.id}/carpet-uni`,
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const fv = new Float32Array(16)
  fv[0] = tileMin[0]
  fv[1] = tileMin[1]
  fv[2] = 1 / tileM
  fv[3] = tileM
  fv[4] = yMin
  fv[5] = 1 / yRange
  fv[8] = bmin[0]; fv[9] = bmin[1]; fv[10] = bmin[2]
  fv[12] = bmax[0]; fv[13] = bmax[1]; fv[14] = bmax[2]
  device.queue.writeBuffer(uni, 0, fv)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bake-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/carpet-bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni } }],
  })
  const module = ctx.shaders.module(bakeCarpetSrc)
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

  const enc = device.createCommandEncoder({ label: `${ctx.id}/carpet-enc` })
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/carpet-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: nhTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  pass.setBindGroup(0, bg)
  // 9 periodic copies in ONE draw — the neighbours fill this tile's window.
  pass.drawIndexed(indices.length, 9)
  pass.end()

  const bpr = CARPET_TEX * 4
  const rbSize = bpr * CARPET_TEX
  const mkRb = (label: string): GPUBuffer =>
    device.createBuffer({
      label: `${ctx.id}/${label}`,
      size: rbSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
  const rbAlb = mkRb('carpet-rb-alb')
  const rbNh = mkRb('carpet-rb-nh')
  const copy = (tex: GPUTexture, buf: GPUBuffer): void => {
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: CARPET_TEX }, [
      CARPET_TEX,
      CARPET_TEX,
    ])
  }
  copy(albTex, rbAlb)
  copy(nhTex, rbNh)
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNh.mapAsync(GPUMapMode.READ)
  const albedo = new Uint8Array(rbSize)
  const nh = new Uint8Array(rbSize)
  albedo.set(new Uint8Array(rbAlb.getMappedRange()))
  nh.set(new Uint8Array(rbNh.getMappedRange()))
  rbAlb.unmap()
  rbNh.unmap()
  for (const r of [vbuf, ibuf, albTex, nhTex, depthTex, uni, rbAlb, rbNh]) r.destroy()

  const albedoMips: Uint8Array<ArrayBuffer>[] = [albedo as Uint8Array<ArrayBuffer>]
  const nhMips: Uint8Array<ArrayBuffer>[] = [nh as Uint8Array<ArrayBuffer>]
  let w = CARPET_TEX
  for (let l = 1; l < CARPET_MIPS; l++) {
    albedoMips.push(downsampleWeighted(albedoMips[l - 1]!, w, [0, 0, 0]))
    nhMips.push(downsampleWeighted(nhMips[l - 1]!, w, [128, 128, 0]))
    w >>= 1
  }
  // Self-check for the premultiply trap: coverage-weighted mips keep the stored
  // luma flat across levels, so the runtime must not divide by alpha.
  console.log(
    `[${ctx.id}] carpet bake tile=${tileM.toFixed(3)}m yRange=${yRange.toFixed(4)}m meanH=${(
      meanChannel(nhMips[0]!, 2) * yRange * 100
    ).toFixed(2)}cm cov L0=${meanAlpha(
      albedoMips[0]!,
    ).toFixed(3)} L${CARPET_MIPS - 1}=${meanAlpha(albedoMips[CARPET_MIPS - 1]!).toFixed(3)} luma=` +
      albedoMips.map((m) => meanLuma(m).toFixed(3)).join('/'),
  )

  return {
    tileM,
    yMin,
    yRange,
    albedoMips,
    nhMips,
    avgColor: averageColor(albedo),
    meanH01: meanChannel(nhMips[0]!, 2),
    coverage0: meanAlpha(albedoMips[0]!),
    coverageN: meanAlpha(albedoMips[CARPET_MIPS - 1]!),
  }
}

/** Render the atlas on the GPU, read it back, build mips + average color. */
export async function bakeSpecies(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<BakedSpecies> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const invR = 1 / radius

  // --- transient GPU resources (destroyed at the end, never in the budget) ---
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
      size: [ATLAS, ATLAS],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    })
  const albTex = mkTarget('bake-alb')
  const nrmTex = mkTarget('bake-nrm')
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
      { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      pass.setViewport(i * TILE, j * TILE, TILE, TILE, 0, 1)
      pass.setScissorRect(i * TILE, j * TILE, TILE, TILE)
      pass.setBindGroup(0, bg, [(j * GRID + i) * STRIDE])
      pass.drawIndexed(indices.length)
    }
  }
  pass.end()

  const bpr = ATLAS * 4
  const rbSize = bpr * ATLAS
  const rbAlb = device.createBuffer({
    label: `${ctx.id}/rb-alb`,
    size: rbSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const rbNrm = device.createBuffer({
    label: `${ctx.id}/rb-nrm`,
    size: rbSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
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

  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()

  return {
    center,
    radius,
    albedoMips: mipChain(albedo, true),
    normalMips: mipChain(normal, false),
    avgColor: averageColor(albedo),
  }
}
