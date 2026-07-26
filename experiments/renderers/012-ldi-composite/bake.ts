import bakeShaderSrc from './shaders/bake.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake a layered depth image (LDI) set from a GCMESH1 source mesh.
 *
 * 5 key capture directions (4 side views at azimuth 0/90/180/270 deg,
 * elevation 20 deg, plus one straight-down top view) x 4 depth-peeled layers.
 * Peeling enforces a MINIMUM SEPARATION between layers: layer L keeps the
 * nearest surface at least SEP metres behind layer L-1 at that pixel. Without
 * it, thin foliage burns all layers on near-coincident blade surfaces of the
 * frontmost tuft and the crown interior is never captured — the classic way
 * LDIs break on grass.
 *
 * Output artifact: header + two atlases (5 dir columns x 4 layer rows of
 * TILE^2 tiles): atlas0 = albedo.rgb + coverage, atlas1 = oct normal (rg) +
 * 16-bit linear capture depth (ba). Per (dir, layer) the header records the
 * mean covered depth — the runtime card plane — computed here on readback.
 */

export const TILE = 256
export const NUM_DIRS = 5
export const NUM_LAYERS = 4
export const ATLAS_W = NUM_DIRS * TILE // 1280
export const ATLAS_H = NUM_LAYERS * TILE // 1024
export const BAKE_VERSION = 2
const SIDE_ELEV = (20 * Math.PI) / 180
const MAGIC = 0x3149444c // 'LDI1'
const HEADER_BYTES = 32 + 16 + NUM_DIRS * 16 * 4 // 368

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
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

export interface DirMeta {
  right: V3
  up: V3
  fwd: V3 // toward the capture camera
  halfW: number
  halfH: number
  sMax: number
  /** Mean covered capture depth (m) per layer; -1 = layer empty. */
  meanD: number[]
}

export interface LdiMeta {
  center: V3
  boundR: number
  tile: number
  dirs: DirMeta[]
}

export interface LdiBaked {
  meta: LdiMeta
  atlas0: Uint8Array<ArrayBuffer> // ATLAS_W * ATLAS_H * 4, albedo + coverage
  atlas1: Uint8Array<ArrayBuffer> // ATLAS_W * ATLAS_H * 4, oct normal + depth16
}

/** Capture bases in plant-local space. Order: 4 side (azimuth k*90deg), then top. */
function captureDirs(): { right: V3; up: V3; fwd: V3 }[] {
  const out: { right: V3; up: V3; fwd: V3 }[] = []
  for (let k = 0; k < 4; k++) {
    const az = (k * Math.PI) / 2
    const fwd: V3 = [Math.cos(SIDE_ELEV) * Math.cos(az), Math.sin(SIDE_ELEV), Math.cos(SIDE_ELEV) * Math.sin(az)]
    const right = norm(cross([0, 1, 0], fwd))
    const up = norm(cross(fwd, right))
    out.push({ right, up, fwd })
  }
  const fwd: V3 = [0, 1, 0]
  const right = norm(cross([0, 0, 1], fwd))
  const up = norm(cross(fwd, right))
  out.push({ right, up, fwd })
  return out
}

export function packLdi(b: LdiBaked): ArrayBuffer {
  const blob = ATLAS_W * ATLAS_H * 4
  const buf = new ArrayBuffer(HEADER_BYTES + blob * 2)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = BAKE_VERSION
  u[2] = b.meta.tile
  u[3] = NUM_LAYERS
  u[4] = NUM_DIRS
  u[5] = ATLAS_W
  u[6] = ATLAS_H
  const f = new Float32Array(buf, 32, 4 + NUM_DIRS * 16)
  f[0] = b.meta.center[0]
  f[1] = b.meta.center[1]
  f[2] = b.meta.center[2]
  f[3] = b.meta.boundR
  b.meta.dirs.forEach((d, i) => {
    const o = 4 + i * 16
    f.set([...d.right, d.halfW, ...d.up, d.halfH, ...d.fwd, d.sMax, ...d.meanD], o)
  })
  new Uint8Array(buf, HEADER_BYTES, blob).set(b.atlas0)
  new Uint8Array(buf, HEADER_BYTES + blob, blob).set(b.atlas1)
  return buf
}

/** Returns null when the buffer is not a valid LDI1 artifact of this version. */
export function unpackLdi(buf: ArrayBuffer): LdiBaked | null {
  const blob = ATLAS_W * ATLAS_H * 4
  if (buf.byteLength !== HEADER_BYTES + blob * 2) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== BAKE_VERSION || u[2] !== TILE) return null
  if (u[3] !== NUM_LAYERS || u[4] !== NUM_DIRS || u[5] !== ATLAS_W || u[6] !== ATLAS_H) return null
  const f = new Float32Array(buf, 32, 4 + NUM_DIRS * 16)
  const dirs: DirMeta[] = []
  for (let i = 0; i < NUM_DIRS; i++) {
    const o = 4 + i * 16
    dirs.push({
      right: [f[o]!, f[o + 1]!, f[o + 2]!],
      halfW: f[o + 3]!,
      up: [f[o + 4]!, f[o + 5]!, f[o + 6]!],
      halfH: f[o + 7]!,
      fwd: [f[o + 8]!, f[o + 9]!, f[o + 10]!],
      sMax: f[o + 11]!,
      meanD: [f[o + 12]!, f[o + 13]!, f[o + 14]!, f[o + 15]!],
    })
  }
  return {
    meta: { center: [f[0]!, f[1]!, f[2]!], boundR: f[3]!, tile: TILE, dirs },
    atlas0: new Uint8Array(buf, HEADER_BYTES, blob),
    atlas1: new Uint8Array(buf, HEADER_BYTES + blob, blob),
  }
}

/**
 * Flood albedo/normal/depth from covered texels into empty neighbours (alpha
 * stays 0). Without this, bilinear taps at silhouettes blend toward black
 * albedo and FAR depth — close up that reads as dark fringes and "beads" of
 * geometry scattered behind the plant, the dominant close-range LDI artifact
 * for thin foliage. When several covered neighbours compete, the frontmost
 * (min depth) wins.
 */
function dilateTile(atlas0: Uint8Array, atlas1: Uint8Array, x0: number, y0: number, iterations = 6): void {
  const mask = new Uint8Array(TILE * TILE)
  for (let y = 0; y < TILE; y++) {
    for (let x = 0; x < TILE; x++) {
      mask[y * TILE + x] = atlas0[((y0 + y) * ATLAS_W + x0 + x) * 4 + 3]! > 0 ? 1 : 0
    }
  }
  for (let it = 0; it < iterations; it++) {
    const grown = mask.slice()
    for (let y = 0; y < TILE; y++) {
      for (let x = 0; x < TILE; x++) {
        if (mask[y * TILE + x]!) continue
        let bestI = -1
        let bestD = Infinity
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx
            const ny = y + dy
            if (nx < 0 || ny < 0 || nx >= TILE || ny >= TILE || !mask[ny * TILE + nx]!) continue
            const ni = ((y0 + ny) * ATLAS_W + x0 + nx) * 4
            const d = atlas1[ni + 2]! * 256 + atlas1[ni + 3]!
            if (d < bestD) {
              bestD = d
              bestI = ni
            }
          }
        }
        if (bestI < 0) continue
        const i = ((y0 + y) * ATLAS_W + x0 + x) * 4
        atlas0[i] = atlas0[bestI]!
        atlas0[i + 1] = atlas0[bestI + 1]!
        atlas0[i + 2] = atlas0[bestI + 2]!
        atlas1[i] = atlas1[bestI]!
        atlas1[i + 1] = atlas1[bestI + 1]!
        atlas1[i + 2] = atlas1[bestI + 2]!
        atlas1[i + 3] = atlas1[bestI + 3]!
        grown[y * TILE + x] = 1
      }
    }
    mask.set(grown)
  }
}

/** Render the peeled layer set on the GPU and read it back. */
export async function bakeLdi(ctx: ExperimentContext<typeof PARAMS>, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const boundR = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const sep = Math.max(0.02, 0.04 * boundR)

  // Tight ortho framing per direction: project bbox corners onto the basis.
  const corners: V3[] = []
  for (let i = 0; i < 8; i++) {
    corners.push([
      (i & 1 ? bmax[0] : bmin[0]) - center[0],
      (i & 2 ? bmax[1] : bmin[1]) - center[1],
      (i & 4 ? bmax[2] : bmin[2]) - center[2],
    ])
  }
  const dirs: DirMeta[] = captureDirs().map((b) => {
    let halfW = 0
    let halfH = 0
    let sMax = 0
    for (const c of corners) {
      halfW = Math.max(halfW, Math.abs(dot(c, b.right)))
      halfH = Math.max(halfH, Math.abs(dot(c, b.up)))
      sMax = Math.max(sMax, Math.abs(dot(c, b.fwd)))
    }
    return { ...b, halfW: halfW * 1.02, halfH: halfH * 1.02, sMax: sMax * 1.02, meanD: [-1, -1, -1, -1] }
  })

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

  const stripW = ATLAS_W
  const stripH = TILE
  const mkStrip = (label: string): GPUTexture =>
    device.createTexture({
      label: `${ctx.id}/${label}`,
      size: [stripW, stripH],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    })
  const albStrips = Array.from({ length: NUM_LAYERS }, (_, l) => mkStrip(`bake-alb-${l}`))
  const auxStrips = Array.from({ length: NUM_LAYERS }, (_, l) => mkStrip(`bake-aux-${l}`))
  const depthTex = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [stripW, stripH],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  // Per-(layer, dir) uniform ring with dynamic offsets. Stride 256, size 96.
  const STRIDE = 256
  const nTiles = NUM_LAYERS * NUM_DIRS
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * nTiles,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new Float32Array((STRIDE / 4) * nTiles)
  for (let l = 0; l < NUM_LAYERS; l++) {
    for (let di = 0; di < NUM_DIRS; di++) {
      const d = dirs[di]!
      const o = ((l * NUM_DIRS + di) * STRIDE) / 4
      scratch.set([...d.right, d.halfW], o)
      scratch.set([...d.up, d.halfH], o + 4)
      scratch.set([...d.fwd, d.sMax], o + 8)
      scratch.set([...center, l === 0 ? 1 : 0], o + 12)
      scratch.set([bmin[0], bmin[1], bmin[2], sep / (2 * d.sMax)], o + 16)
      scratch.set([bmax[0], bmax[1], bmax[2], 0], o + 20)
    }
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl0 = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl0`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
    ],
  })
  const bgl1 = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl1`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
  })
  const bg0 = device.createBindGroup({
    label: `${ctx.id}/bake-bg0`,
    layout: bgl0,
    entries: [{ binding: 0, resource: { buffer: uni, size: 96 } }],
  })

  const module = ctx.shaders.module(bakeShaderSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/bake-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-pl`, bindGroupLayouts: [bgl0, bgl1] }),
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
  for (let l = 0; l < NUM_LAYERS; l++) {
    // Peel input: previous layer's aux strip. Layer 0 skips the read (flag in
    // the tile uniform); it binds the (zero-initialized) last strip only to
    // satisfy the layout.
    const prev = l === 0 ? auxStrips[NUM_LAYERS - 1]! : auxStrips[l - 1]!
    const bg1 = device.createBindGroup({
      label: `${ctx.id}/bake-bg1-${l}`,
      layout: bgl1,
      entries: [{ binding: 0, resource: prev.createView() }],
    })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-layer-${l}`,
      colorAttachments: [
        { view: albStrips[l]!.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        // aux clear = depth 1.0 (far): once a pixel column is exhausted, every
        // deeper layer stays empty there.
        { view: auxStrips[l]!.createView(), clearValue: { r: 0.5, g: 0.5, b: 1, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    })
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    pass.setBindGroup(1, bg1)
    for (let di = 0; di < NUM_DIRS; di++) {
      pass.setViewport(di * TILE, 0, TILE, TILE, 0, 1)
      pass.setScissorRect(di * TILE, 0, TILE, TILE)
      pass.setBindGroup(0, bg0, [(l * NUM_DIRS + di) * STRIDE])
      pass.drawIndexed(indices.length)
    }
    pass.end()
  }

  const bpr = stripW * 4 // 5120, multiple of 256
  const rbSize = bpr * stripH
  const rbs: GPUBuffer[] = []
  const readbacks: { tex: GPUTexture; buf: GPUBuffer }[] = []
  for (let l = 0; l < NUM_LAYERS; l++) {
    for (const tex of [albStrips[l]!, auxStrips[l]!]) {
      const buf = device.createBuffer({
        label: `${ctx.id}/bake-rb`,
        size: rbSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: stripH }, [stripW, stripH])
      rbs.push(buf)
      readbacks.push({ tex, buf })
    }
  }
  device.queue.submit([enc.finish()])

  const atlas0 = new Uint8Array(ATLAS_W * ATLAS_H * 4)
  const atlas1 = new Uint8Array(ATLAS_W * ATLAS_H * 4)
  for (let l = 0; l < NUM_LAYERS; l++) {
    const albBuf = readbacks[l * 2]!.buf
    const auxBuf = readbacks[l * 2 + 1]!.buf
    await albBuf.mapAsync(GPUMapMode.READ)
    await auxBuf.mapAsync(GPUMapMode.READ)
    atlas0.set(new Uint8Array(albBuf.getMappedRange()), l * TILE * bpr)
    atlas1.set(new Uint8Array(auxBuf.getMappedRange()), l * TILE * bpr)
    albBuf.unmap()
    auxBuf.unmap()

    for (let di = 0; di < NUM_DIRS; di++) {
      // Mean covered capture depth per dir -> the runtime card plane.
      let sum = 0
      let count = 0
      for (let y = l * TILE; y < (l + 1) * TILE; y++) {
        for (let x = di * TILE; x < (di + 1) * TILE; x++) {
          const i = (y * ATLAS_W + x) * 4
          if (atlas0[i + 3]! === 0) continue
          sum += (atlas1[i + 2]! * 256 + atlas1[i + 3]!) / 65535
          count++
        }
      }
      if (count >= 16) dirs[di]!.meanD[l] = (sum / count) * 2 * dirs[di]!.sMax

      dilateTile(atlas0, atlas1, di * TILE, l * TILE)
    }
  }

  const packed = packLdi({ meta: { center, boundR, tile: TILE, dirs }, atlas0, atlas1 })
  for (const r of [vbuf, ibuf, uni, depthTex, ...albStrips, ...auxStrips, ...rbs]) r.destroy()
  return packed
}
