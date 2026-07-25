import bakeShaderSrc from './shaders/bake.wgsl'
import { assetUrl, asU32, commitBake, hash2, hashF32, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Canopy BTF bake.
 *
 * The species' periodic community tile is captured ORTHOGRAPHICALLY along a
 * 5x5 hemi-octahedral grid of view directions. Unlike a per-plant impostor,
 * each capture is parameterised by the view ray's intersection with the GROUND
 * PLANE of the tile: texel (u,v) of bin d stores what an observer looking
 * along -d through ground point (u*T, 0, v*T) would see — after the ray has
 * passed through the (periodically wrapped) canopy above and upstream of that
 * point. Because the tile is periodic, the same texture wraps over infinite
 * ground; a billion plants and a hundred cost the same.
 *
 * Each capture is rendered 4x supersampled and reduced on CPU to an aggregate
 * per texel: mean albedo, coverage (fraction of subsample rays that hit
 * anything), mean hit height, mean normal, and luminance sigma. Content is
 * dilated into empty texels, wrapped borders are added per bin cell, and a
 * 4-level mip chain is built (alpha-weighted) so distant ground filters
 * correctly. The packed artifact is committed via the dev server.
 *
 * poa-pratensis is a finite specimen, not a community tile: a synthetic
 * periodic tile is composed at bake time from 3 hashed copies (fixed bake
 * seed, independent of the runtime placement seed — the whole point of this
 * experiment is that per-plant identity is given up).
 */

export const BINS = 5
export const TILE_RES = 256
export const BORDER = 8
export const LEVELS = 4
export const CELL_PX = TILE_RES + 2 * BORDER // 272
export const ATLAS_PX = BINS * CELL_PX // 1360
export const BAKE_VERSION = 1

const SS = 4 // supersample factor per axis
const CAPTURE_RES = TILE_RES * SS // 1024
const MIN_ELEVATION_DEG = 10
const MAX_TRIS_PER_SUBMIT = 250e6
const MAGIC = 0x54424347 // 'GCBT' little-endian
const HEADER_BYTES = 512

type V3 = [number, number, number]

export interface BtfMeta {
  bins: number
  tileRes: number
  border: number
  levels: number
  atlasPx: number
  /** Periodic tile size in metres (of the UNSCALED capture). */
  tileSize: number
  /** Canopy top height in metres used to normalise stored hit heights. */
  topH: number
  /** Coverage-weighted mean albedo over all bins (distant fade target). */
  avgColor: V3
  /** bins*bins capture directions (xyz, toward viewer), vec4-strided. */
  dirs: Float32Array
}

export interface BtfArtifact {
  meta: BtfMeta
  /** Albedo+coverage atlas, LEVELS mips, rgba8. */
  levelsA: Uint8Array<ArrayBuffer>[]
  /** Height + oct-normal + sigma atlas, LEVELS mips, rgba8. */
  levelsB: Uint8Array<ArrayBuffer>[]
}

const norm3 = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** Hemi-octahedral decode — must match hemioct code in canopy.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const u = (ex + ey) * 0.5
  const w = (ex - ey) * 0.5
  const y = 1 - Math.abs(u) - Math.abs(w)
  return norm3([u, y, w])
}

/** Full-sphere octahedral decode — twin of GcMesh.normalAt(). */
function octDecode(u01: number, v01: number, out: V3): V3 {
  const u = u01 * 2 - 1
  const v = v01 * 2 - 1
  let x = u
  let z = v
  const y = 1 - Math.abs(u) - Math.abs(v)
  if (y < 0) {
    x = (1 - Math.abs(v)) * Math.sign(u)
    z = (1 - Math.abs(u)) * Math.sign(v)
  }
  const l = Math.hypot(x, y, z) || 1
  out[0] = x / l
  out[1] = y / l
  out[2] = z / l
  return out
}

/** Full-sphere octahedral encode to [0,1]^2 — inverse of octDecode. */
function octEncode(n: V3): [number, number] {
  const s = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]) || 1
  let x = n[0] / s
  const y = n[1] / s
  let z = n[2] / s
  if (y < 0) {
    const ox = (1 - Math.abs(z)) * Math.sign(x)
    const oz = (1 - Math.abs(x)) * Math.sign(z)
    x = ox
    z = oz
  }
  return [(x + 1) * 0.5, (z + 1) * 0.5]
}

/** The 25 capture directions: hemi-oct grid nodes, elevation-clamped. */
export function binDirections(): V3[] {
  const minY = Math.sin((MIN_ELEVATION_DEG * Math.PI) / 180)
  const dirs: V3[] = []
  for (let j = 0; j < BINS; j++) {
    for (let i = 0; i < BINS; i++) {
      const ex = (i / (BINS - 1)) * 2 - 1
      const ey = (j / (BINS - 1)) * 2 - 1
      let d = hemioctDecode(ex, ey)
      if (d[1] < minY) {
        const hl = Math.hypot(d[0], d[2]) || 1
        const s = Math.sqrt(1 - minY * minY) / hl
        d = [d[0] * s, minY, d[2] * s]
      }
      dirs.push(d)
    }
  }
  return dirs
}

// ---------------------------------------------------------------------------
// Artifact packing
// ---------------------------------------------------------------------------

function levelBytes(level: number): number {
  const px = ATLAS_PX >> level
  return px * px * 4
}

export function packArtifact(a: BtfArtifact): ArrayBuffer {
  let blob = 0
  for (let l = 0; l < LEVELS; l++) blob += levelBytes(l)
  const buf = new ArrayBuffer(HEADER_BYTES + blob * 2)
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, HEADER_BYTES / 4)
  u32[0] = MAGIC
  u32[1] = BAKE_VERSION
  u32[2] = a.meta.bins
  u32[3] = a.meta.tileRes
  u32[4] = a.meta.border
  u32[5] = a.meta.levels
  u32[6] = a.meta.atlasPx
  f32[7] = a.meta.tileSize
  f32[8] = a.meta.topH
  f32[9] = a.meta.avgColor[0]
  f32[10] = a.meta.avgColor[1]
  f32[11] = a.meta.avgColor[2]
  f32.set(a.meta.dirs, 16)
  let off = HEADER_BYTES
  for (const levels of [a.levelsA, a.levelsB]) {
    for (let l = 0; l < LEVELS; l++) {
      new Uint8Array(buf, off, levelBytes(l)).set(levels[l]!)
      off += levelBytes(l)
    }
  }
  return buf
}

export function unpackArtifact(buf: ArrayBuffer): BtfArtifact {
  const u32 = new Uint32Array(buf, 0, 16)
  const f32 = new Float32Array(buf, 0, HEADER_BYTES / 4)
  if (u32[0] !== MAGIC || u32[1] !== BAKE_VERSION) throw new Error('GCBTF: bad magic/version')
  const meta: BtfMeta = {
    bins: u32[2]!,
    tileRes: u32[3]!,
    border: u32[4]!,
    levels: u32[5]!,
    atlasPx: u32[6]!,
    tileSize: f32[7]!,
    topH: f32[8]!,
    avgColor: [f32[9]!, f32[10]!, f32[11]!],
    dirs: new Float32Array(buf.slice(64, 64 + BINS * BINS * 16)),
  }
  const levelsA: Uint8Array<ArrayBuffer>[] = []
  const levelsB: Uint8Array<ArrayBuffer>[] = []
  let off = HEADER_BYTES
  for (const levels of [levelsA, levelsB]) {
    for (let l = 0; l < LEVELS; l++) {
      levels.push(new Uint8Array(buf, off, levelBytes(l)))
      off += levelBytes(l)
    }
  }
  return { meta, levelsA, levelsB }
}

// ---------------------------------------------------------------------------
// Tile composition (community tile as-is; synthetic tile for finite specimens)
// ---------------------------------------------------------------------------

interface TileSpec {
  tileT: number
  topH: number
  /** Copies of the mesh laid into one periodic tile: x, z, yaw, scale. */
  copies: { x: number; z: number; yaw: number; scale: number }[]
  /** Conservative world-space bounds of one tile's geometry (for instancing). */
  gMin: [number, number]
  gMax: [number, number]
  /**
   * The tile is a SURFACE (wider than it is tall — a cushion/mat) rather than
   * foliage standing off the ground. Set from the mesh alone: topH < tileT.
   * Sphagnum is 0.09m tall over a 0.18m tile (0.5); the grasses are 1.18m over
   * 0.52m (2.3). For a surface the stored normal is derived from the captured
   * HEIGHT FIELD instead of from the mean of the first-hit leaf normals — see
   * surfaceNormals().
   */
  surfaceLike: boolean
}

function tileSpecFor(mesh: GcMesh): TileSpec {
  const h = mesh.header
  if (h.tileSize[0] > 0 && h.tileSize[1] > 0) {
    const topH = Math.max(h.topH, h.boundsMax[1])
    return {
      tileT: h.tileSize[0],
      topH,
      copies: [{ x: 0, z: 0, yaw: 0, scale: 1 }],
      gMin: [h.boundsMin[0], h.boundsMin[2]],
      gMax: [h.boundsMax[0], h.boundsMax[2]],
      surfaceLike: topH < h.tileSize[0],
    }
  }
  // Finite specimen (poa): compose a synthetic periodic tile. Fixed bake seed:
  // the aggregate field deliberately has no per-plant identity.
  const T = 0.9
  const copies = []
  for (let c = 0; c < 3; c++) {
    copies.push({
      x: hashF32(hash2(asU32(c), 101)) * T,
      z: hashF32(hash2(asU32(c), 202)) * T,
      yaw: hashF32(hash2(asU32(c), 303)) * Math.PI * 2,
      scale: 0.85 + 0.45 * hashF32(hash2(asU32(c), 404)),
    })
  }
  const r = Math.max(
    Math.abs(h.boundsMin[0]),
    Math.abs(h.boundsMax[0]),
    Math.abs(h.boundsMin[2]),
    Math.abs(h.boundsMax[2]),
  ) * 1.35
  return {
    tileT: T,
    topH: h.boundsMax[1] * 1.3,
    copies,
    gMin: [-r, -r],
    gMax: [T + r, T + r],
    surfaceLike: h.boundsMax[1] * 1.3 < T,
  }
}

// ---------------------------------------------------------------------------
// The bake proper
// ---------------------------------------------------------------------------

export async function bakeSpecies(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  onNote: (s: string) => void,
): Promise<ArrayBuffer> {
  const { device } = ctx
  const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
  const spec = tileSpecFor(mesh)
  const h = mesh.header
  const dirs = binDirections()

  // --- transient GPU resources (destroyed at the end, never in ctx.res) ---
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
    size: [CAPTURE_RES, CAPTURE_RES],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const attrTex = device.createTexture({
    label: `${ctx.id}/bake-attr`,
    size: [CAPTURE_RES, CAPTURE_RES],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })
  const depthTex = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [CAPTURE_RES, CAPTURE_RES],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  const MAX_INSTANCES = 8192
  const instBuf = device.createBuffer({
    label: `${ctx.id}/bake-inst`,
    size: MAX_INSTANCES * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const uniBuf = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniBuf } },
      { binding: 1, resource: { buffer: instBuf } },
    ],
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

  const bpr = CAPTURE_RES * 4
  const rbSize = bpr * CAPTURE_RES
  const rbAlb = device.createBuffer({ label: `${ctx.id}/rb-a`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const rbAttr = device.createBuffer({ label: `${ctx.id}/rb-b`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

  const uniforms = new Float32Array(16)
  const albBytes = new Uint8Array(rbSize)
  const attrBytes = new Uint8Array(rbSize)

  // Per-bin reduced content at mip 0 (TILE_RES^2), two rgba8 planes each.
  const binA: Uint8Array<ArrayBuffer>[] = []
  const binB: Uint8Array<ArrayBuffer>[] = []

  for (let bin = 0; bin < BINS * BINS; bin++) {
    const d = dirs[bin]!
    // Which tile instances can project into the [0,T]^2 footprint? Content at
    // height y lands at q = w.xz - (w.y/d.y)*d.xz, so needed w-range is the
    // footprint expanded upstream by topH/d.y * d.xz.
    const sx = (spec.topH / d[1]) * d[0]
    const sz = (spec.topH / d[1]) * d[2]
    const needMinX = Math.min(0, sx)
    const needMaxX = spec.tileT + Math.max(0, sx)
    const needMinZ = Math.min(0, sz)
    const needMaxZ = spec.tileT + Math.max(0, sz)
    const aMin = Math.floor((needMinX - spec.gMax[0]) / spec.tileT)
    const aMax = Math.ceil((needMaxX - spec.gMin[0]) / spec.tileT)
    const bMin = Math.floor((needMinZ - spec.gMax[1]) / spec.tileT)
    const bMax = Math.ceil((needMaxZ - spec.gMin[1]) / spec.tileT)

    const inst: number[] = []
    for (let b = bMin; b <= bMax; b++) {
      for (let a = aMin; a <= aMax; a++) {
        for (const c of spec.copies) {
          inst.push(a * spec.tileT + c.x, b * spec.tileT + c.z, c.yaw, c.scale)
        }
      }
    }
    const count = Math.min(inst.length / 4, MAX_INSTANCES)
    device.queue.writeBuffer(instBuf, 0, new Float32Array(inst.slice(0, count * 4)))

    uniforms[0] = d[0]; uniforms[1] = d[1]; uniforms[2] = d[2]; uniforms[3] = 0
    uniforms[4] = h.boundsMin[0]; uniforms[5] = h.boundsMin[1]; uniforms[6] = h.boundsMin[2]; uniforms[7] = spec.tileT
    uniforms[8] = h.boundsMax[0]; uniforms[9] = h.boundsMax[1]; uniforms[10] = h.boundsMax[2]; uniforms[11] = spec.topH
    device.queue.writeBuffer(uniBuf, 0, uniforms)

    // Chunk instance draws so a single submit never exceeds the TDR budget.
    const perChunk = Math.max(1, Math.floor(MAX_TRIS_PER_SUBMIT / mesh.header.triangleCount))
    for (let first = 0; first < count; first += perChunk) {
      const n = Math.min(perChunk, count - first)
      const enc = device.createCommandEncoder()
      const load: GPULoadOp = first === 0 ? 'clear' : 'load'
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: load, storeOp: 'store' },
          { view: attrTex.createView(), clearValue: { r: 0, g: 0.5, b: 0.5, a: 0 }, loadOp: load, storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1,
          depthLoadOp: first === 0 ? 'clear' : 'load',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      pass.drawIndexed(indices.length, n, 0, 0, first)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    const enc = device.createCommandEncoder()
    enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: CAPTURE_RES }, [CAPTURE_RES, CAPTURE_RES])
    enc.copyTextureToBuffer({ texture: attrTex }, { buffer: rbAttr, bytesPerRow: bpr, rowsPerImage: CAPTURE_RES }, [CAPTURE_RES, CAPTURE_RES])
    device.queue.submit([enc.finish()])
    await rbAlb.mapAsync(GPUMapMode.READ)
    await rbAttr.mapAsync(GPUMapMode.READ)
    albBytes.set(new Uint8Array(rbAlb.getMappedRange()))
    attrBytes.set(new Uint8Array(rbAttr.getMappedRange()))
    rbAlb.unmap()
    rbAttr.unmap()

    const { a, b } = reduceCapture(albBytes, attrBytes)
    if (spec.surfaceLike) surfaceNormals(b, spec, d)
    binA.push(a)
    binB.push(b)
    onNote(`${speciesId}: bin ${bin + 1}/${BINS * BINS} (${count} tile instances)`)
  }

  for (const r of [vbuf, ibuf, albTex, attrTex, depthTex, instBuf, uniBuf, rbAlb, rbAttr]) r.destroy()

  onNote(`${speciesId}: assembling atlas`)
  const artifact = assembleArtifact(binA, binB, spec, dirs)
  return packArtifact(artifact)
}

/** 4x4 supersample reduction: coverage, mean albedo/height/normal, sigma. */
function reduceCapture(alb: Uint8Array, attr: Uint8Array): { a: Uint8Array<ArrayBuffer>; b: Uint8Array<ArrayBuffer> } {
  const a = new Uint8Array(TILE_RES * TILE_RES * 4)
  const b = new Uint8Array(TILE_RES * TILE_RES * 4)
  const n: V3 = [0, 0, 0]
  const lums = new Float32Array(SS * SS)
  for (let ty = 0; ty < TILE_RES; ty++) {
    for (let tx = 0; tx < TILE_RES; tx++) {
      let cnt = 0
      let cr = 0, cg = 0, cb = 0, ch = 0
      let nx = 0, ny = 0, nz = 0
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const px = ((ty * SS + sy) * CAPTURE_RES + tx * SS + sx) * 4
          if (alb[px + 3]! === 0) continue
          cr += alb[px]!
          cg += alb[px + 1]!
          cb += alb[px + 2]!
          ch += attr[px]!
          octDecode(attr[px + 1]! / 255, attr[px + 2]! / 255, n)
          nx += n[0]; ny += n[1]; nz += n[2]
          lums[cnt] = 0.299 * alb[px]! + 0.587 * alb[px + 1]! + 0.114 * alb[px + 2]!
          cnt++
        }
      }
      const o = (ty * TILE_RES + tx) * 4
      if (cnt === 0) {
        a[o + 3] = 0
        b[o] = 0
        b[o + 1] = 128
        b[o + 2] = 128
        continue
      }
      const inv = 1 / cnt
      a[o] = Math.round(cr * inv)
      a[o + 1] = Math.round(cg * inv)
      a[o + 2] = Math.round(cb * inv)
      a[o + 3] = Math.round((cnt / (SS * SS)) * 255)
      b[o] = Math.round(ch * inv)
      const [eu, ev] = octEncode(norm3([nx, ny, nz]))
      b[o + 1] = Math.round(eu * 255)
      b[o + 2] = Math.round(ev * 255)
      const mean = (cr * 0.299 + cg * 0.587 + cb * 0.114) * inv
      let varSum = 0
      for (let i = 0; i < cnt; i++) varSum += (lums[i]! - mean) ** 2
      b[o + 3] = Math.min(255, Math.round(Math.sqrt(varSum / cnt) * 3))
    }
  }
  dilate(a, b)
  return { a, b }
}

/**
 * Replace the stored normal of a SURFACE-LIKE tile (a cushion/mat) with the
 * normal of its captured HEIGHT FIELD, and store the lightly smoothed height.
 *
 * Why: at 0.18m/256 texels the Sphagnum capture is 0.70mm per texel, which is
 * roughly ONE branch leaf. The 16 supersampled first-hit normals inside a texel
 * therefore belong to 16 differently-oriented leaves and their mean is
 * ill-conditioned: measured over the top bin, normal.y had mean 0.002 (a
 * ground-hugging cushion averaging to no up component at all) and was
 * decorrelated from its neighbour at 82% of the white-noise level — i.e. the
 * shading normal was static, so `light_surface()` produced per-pixel noise and
 * the mat read as flat grain with no cushion relief whatsoever.
 *
 * The HEIGHT channel, by contrast, is real structure: sd 8.2mm with only 48% of
 * white-noise neighbour difference, and 71% of that sd survives an 8x8 (5.6mm)
 * box — capitulum scale. So the aggregate surface is well defined even though
 * the individual leaf normals are not, and its gradient is the normal that
 * makes the cushions visible.
 *
 * The capture is parameterised by the ray's GROUND intersection, so a texel's
 * world position is p = uv*tileT + (h/d.y)*d.xz: the (u,v) grid is sheared by
 * the bin direction, and the shear is undone here before differencing (for the
 * top bin d.xz = 0 and this reduces to the plain height gradient).
 *
 * Foliage tiles (topH >= tileT: all three grasses) are left alone — a blade
 * canopy has no aggregate surface, its height field IS noise (calamagrostis:
 * sd 285mm, 28% lag-1) and the first-hit leaf normal is the right answer there.
 */
function surfaceNormals(b: Uint8Array, spec: TileSpec, d: V3): void {
  const R = TILE_RES
  const texel = spec.tileT / R
  const wrap = (i: number): number => ((i % R) + R) % R
  // Height in metres, then two [1,2,1] passes (sigma ~1 texel = 0.7mm, far
  // below the 5-10mm capitulum scale) to remove the residual per-texel jitter.
  const h = new Float32Array(R * R)
  for (let i = 0; i < R * R; i++) h[i] = (b[i * 4]! / 255) * spec.topH
  const tmp = new Float32Array(R * R)
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 0; y < R; y++) {
      for (let x = 0; x < R; x++) {
        tmp[y * R + x] = (h[y * R + wrap(x - 1)]! + 2 * h[y * R + x]! + h[y * R + wrap(x + 1)]!) / 4
      }
    }
    for (let y = 0; y < R; y++) {
      for (let x = 0; x < R; x++) {
        h[y * R + x] = (tmp[wrap(y - 1) * R + x]! + 2 * tmp[y * R + x]! + tmp[wrap(y + 1) * R + x]!) / 4
      }
    }
  }
  // Central difference over +/-2 texels (1.4mm baseline) — a facet of a
  // capitulum, not a single leaf.
  const L = 2
  const inv = 1 / (2 * L * texel)
  for (let y = 0; y < R; y++) {
    for (let x = 0; x < R; x++) {
      const o = (y * R + x) * 4
      const dhu = (h[y * R + wrap(x + L)]! - h[y * R + wrap(x - L)]!) * inv
      const dhv = (h[wrap(y + L) * R + x]! - h[wrap(y - L) * R + x]!) * inv
      // Un-shear: dp/du = (1 + dh/du * d.x/d.y, dh/du, dh/du * d.z/d.y) * texel
      const kx = d[0] / d[1]
      const kz = d[2] / d[1]
      const tu: V3 = [1 + dhu * kx, dhu, dhu * kz]
      const tv: V3 = [dhv * kx, dhv, 1 + dhv * kz]
      // Where the surface is steeper than the view elevation the (u,v) -> world
      // map FOLDS (that is a silhouette: the ray grazes past a bump onto the
      // patch behind it) and the Jacobian determinant, which reduces to
      // 1 + grad(h).d.xz/d.y, flips sign — so the cross product comes out
      // inverted, not the surface. Restoring the orientation with sign(det)
      // takes the grazing bins from "normal.y mean 0.53, sd 0.60, 20% of texels
      // facing DOWNWARD" to "mean 0.75, sd 0.27, none below zero".
      const sg = 1 + dhu * kx + dhv * kz < 0 ? -1 : 1
      // tv x tu (this order points +Y up for a flat tile).
      const n = norm3([
        sg * (tv[1] * tu[2] - tv[2] * tu[1]),
        sg * (tv[2] * tu[0] - tv[0] * tu[2]),
        sg * (tv[0] * tu[1] - tv[1] * tu[0]),
      ])
      const [eu, ev] = octEncode(n)
      b[o] = Math.round(clamp01(h[y * R + x]! / spec.topH) * 255)
      b[o + 1] = Math.round(eu * 255)
      b[o + 2] = Math.round(ev * 255)
    }
  }
}

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v))

/** Flood covered colors/attrs into empty texels (keeps alpha 0) — stops halo
 * artifacts when linear filtering crosses coverage edges. */
function dilate(a: Uint8Array, b: Uint8Array): void {
  const R = TILE_RES
  for (let pass = 0; pass < 3; pass++) {
    const filled: number[] = []
    for (let y = 0; y < R; y++) {
      for (let x = 0; x < R; x++) {
        const o = (y * R + x) * 4
        if (a[o + 3]! > 0 || a[o]! + a[o + 1]! + a[o + 2]! > 0) continue
        let cr = 0, cg = 0, cb = 0, hh = 0, cnt = 0
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const q = (((y + dy + R) % R) * R + ((x + dx + R) % R)) * 4
          if (a[q + 3]! === 0 && a[q]! + a[q + 1]! + a[q + 2]! === 0) continue
          cr += a[q]!; cg += a[q + 1]!; cb += a[q + 2]!; hh += b[q]!
          cnt++
        }
        if (cnt > 0) filled.push(o, Math.round(cr / cnt), Math.round(cg / cnt), Math.round(cb / cnt), Math.round(hh / cnt))
      }
    }
    for (let i = 0; i < filled.length; i += 5) {
      const o = filled[i]!
      a[o] = filled[i + 1]!
      a[o + 1] = filled[i + 2]!
      a[o + 2] = filled[i + 3]!
      b[o] = filled[i + 4]!
    }
    if (filled.length === 0) break
  }
}

/** Build per-bin mips, add wrapped borders, lay out the atlas, pack meta. */
function assembleArtifact(binA: Uint8Array<ArrayBuffer>[], binB: Uint8Array<ArrayBuffer>[], spec: TileSpec, dirs: V3[]): BtfArtifact {
  // Per-bin mip chains (alpha-weighted for A; A-alpha-weighted attrs for B).
  const chainsA: Uint8Array<ArrayBuffer>[][] = []
  const chainsB: Uint8Array<ArrayBuffer>[][] = []
  for (let bin = 0; bin < BINS * BINS; bin++) {
    const ca = [binA[bin]!]
    const cb = [binB[bin]!]
    for (let l = 1; l < LEVELS; l++) {
      const rs = TILE_RES >> (l - 1)
      const rd = TILE_RES >> l
      const pa = ca[l - 1]!
      const pb = cb[l - 1]!
      const da = new Uint8Array(rd * rd * 4)
      const db = new Uint8Array(rd * rd * 4)
      for (let y = 0; y < rd; y++) {
        for (let x = 0; x < rd; x++) {
          let wSum = 0, aSum = 0
          let cr = 0, cg = 0, cb2 = 0, hh = 0, sg = 0
          let pr = 0, pg = 0, pb2 = 0
          const n: V3 = [0, 0, 0]
          const nAcc: V3 = [0, 0, 0]
          for (let sy = 0; sy < 2; sy++) {
            for (let sx = 0; sx < 2; sx++) {
              const q = ((y * 2 + sy) * rs + x * 2 + sx) * 4
              const al = pa[q + 3]!
              aSum += al
              pr += pa[q]!; pg += pa[q + 1]!; pb2 += pa[q + 2]!
              if (al === 0) continue
              wSum += al
              cr += pa[q]! * al; cg += pa[q + 1]! * al; cb2 += pa[q + 2]! * al
              hh += pb[q]! * al
              sg += pb[q + 3]! * al
              octDecode(pb[q + 1]! / 255, pb[q + 2]! / 255, n)
              nAcc[0] += n[0] * al; nAcc[1] += n[1] * al; nAcc[2] += n[2] * al
            }
          }
          const o = (y * rd + x) * 4
          da[o + 3] = Math.round(aSum / 4)
          if (wSum > 0) {
            da[o] = Math.round(cr / wSum)
            da[o + 1] = Math.round(cg / wSum)
            da[o + 2] = Math.round(cb2 / wSum)
            db[o] = Math.round(hh / wSum)
            const [eu, ev] = octEncode(norm3(nAcc))
            db[o + 1] = Math.round(eu * 255)
            db[o + 2] = Math.round(ev * 255)
            db[o + 3] = Math.round(sg / wSum)
          } else {
            da[o] = Math.round(pr / 4)
            da[o + 1] = Math.round(pg / 4)
            da[o + 2] = Math.round(pb2 / 4)
            db[o + 1] = 128
            db[o + 2] = 128
          }
        }
      }
      ca.push(da)
      cb.push(db)
    }
    chainsA.push(ca)
    chainsB.push(cb)
  }

  // Atlas assembly with periodic borders per bin cell.
  const levelsA: Uint8Array<ArrayBuffer>[] = []
  const levelsB: Uint8Array<ArrayBuffer>[] = []
  for (let l = 0; l < LEVELS; l++) {
    const apx = ATLAS_PX >> l
    const cell = CELL_PX >> l
    const border = BORDER >> l
    const res = TILE_RES >> l
    const outA = new Uint8Array(apx * apx * 4)
    const outB = new Uint8Array(apx * apx * 4)
    for (let bin = 0; bin < BINS * BINS; bin++) {
      const bi = bin % BINS
      const bj = Math.floor(bin / BINS)
      const ca = chainsA[bin]![l]!
      const cb = chainsB[bin]![l]!
      for (let py = 0; py < cell; py++) {
        const sy = (((py - border) % res) + res) % res
        for (let px = 0; px < cell; px++) {
          const sx = (((px - border) % res) + res) % res
          const src = (sy * res + sx) * 4
          const dst = ((bj * cell + py) * apx + bi * cell + px) * 4
          outA[dst] = ca[src]!
          outA[dst + 1] = ca[src + 1]!
          outA[dst + 2] = ca[src + 2]!
          outA[dst + 3] = ca[src + 3]!
          outB[dst] = cb[src]!
          outB[dst + 1] = cb[src + 1]!
          outB[dst + 2] = cb[src + 2]!
          outB[dst + 3] = cb[src + 3]!
        }
      }
    }
    levelsA.push(outA)
    levelsB.push(outB)
  }

  // Coverage-weighted average albedo (distant fade target).
  let wr = 0, wg = 0, wb = 0, w = 0
  for (let bin = 0; bin < BINS * BINS; bin++) {
    const ca = chainsA[bin]![LEVELS - 1]!
    for (let i = 0; i < ca.length; i += 4) {
      const al = ca[i + 3]!
      wr += ca[i]! * al; wg += ca[i + 1]! * al; wb += ca[i + 2]! * al
      w += al
    }
  }
  const avg: V3 = w > 0 ? [wr / w / 255, wg / w / 255, wb / w / 255] : [0.2, 0.3, 0.12]

  const dirsF = new Float32Array(BINS * BINS * 4)
  dirs.forEach((d, i) => dirsF.set(d, i * 4))

  return {
    meta: {
      bins: BINS,
      tileRes: TILE_RES,
      border: BORDER,
      levels: LEVELS,
      atlasPx: ATLAS_PX,
      tileSize: spec.tileT,
      topH: spec.topH,
      avgColor: avg,
      dirs: dirsF,
    },
    levelsA,
    levelsB,
  }
}

/**
 * Committed-artifact load with in-browser bake fallback. The harness
 * bakedArtifact() cache is bypassed on purpose: the dev server answers missing
 * /mesh/baked files with 200 index.html, so we validate the magic ourselves
 * and commit the artifact after a fresh bake.
 */
export async function loadOrBakeSpecies(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  onNote: (s: string) => void,
): Promise<BtfArtifact> {
  const key = `btf-v${BAKE_VERSION}-${speciesId}`
  // Committed artifacts are ~19.7MB each and a stand can want five of them, and
  // a single flaky read used to fall straight through to a fresh bake — which
  // for a 19.8M-triangle Sphagnum tile is minutes of GPU per species, silently.
  // So: retry, and validate the FULL length (a truncated body would otherwise
  // pass the magic check and then throw out of unpackArtifact's views).
  const expected = HEADER_BYTES + 2 * [0, 1, 2, 3].reduce((s, l) => s + levelBytes(l), 0)
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      // assetUrl(): production builds live under a base path, so a root-absolute
      // fetch would 404 there (CLAUDE.md).
      const res = await fetch(assetUrl(`/mesh/baked/${ctx.id}/${key}.bin`))
      if (res.ok) {
        const buf = await res.arrayBuffer()
        if (buf.byteLength === expected && new Uint32Array(buf, 0, 1)[0] === MAGIC) {
          return unpackArtifact(buf)
        }
        // The dev server answers a missing file with 200 + index.html, which is
        // the expected miss on the first ever run — not worth retrying.
        if (buf.byteLength < HEADER_BYTES || new Uint32Array(buf, 0, 1)[0] !== MAGIC) break
        onNote(`${speciesId}: artifact truncated (${buf.byteLength}/${expected}) — retry ${attempt + 1}/3`)
      } else {
        onNote(`${speciesId}: artifact fetch ${res.status} — retry ${attempt + 1}/3`)
      }
    } catch (err) {
      onNote(`${speciesId}: artifact fetch failed (${String(err)}) — retry ${attempt + 1}/3`)
    }
  }
  onNote(`${speciesId}: no committed artifact — baking in-browser`)
  const packed = await bakeSpecies(ctx, speciesId, onNote)
  try {
    const saved = await commitBake(ctx.id, key, packed)
    onNote(`${speciesId}: committed ${saved}`)
  } catch (err) {
    onNote(`${speciesId}: commitBake unavailable (${String(err)}) — using in-memory bake`)
  }
  return unpackArtifact(packed)
}
