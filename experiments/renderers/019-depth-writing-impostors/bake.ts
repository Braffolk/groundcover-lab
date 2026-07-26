import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * View-set bake for the true-depth impostor.
 *
 * Per species one texture-array atlas of TILE px layers, rendered
 * orthographically from the raw GCMESH1 mesh: one row of azimuths per
 * elevation, plus a single straight-down top view as the last layer.
 *
 * Each layer stores, per texel:
 *   albedo rgba8  rgb = authored colour, a = coverage
 *   geo    rgba8  rg = octahedral mesh-frame normal, b = SIGNED DEPTH along
 *                 the view axis, a = baked ambient occlusion
 *
 * The depth channel is what makes this method different from a billboard: the
 * runtime turns (u, v, depth) back into a real 3D point in the plant's frame.
 *
 * Every view carries its own orthonormal basis, its own half-extents (the
 * local AABB projected onto that basis) AND the tight bounding rectangle of
 * its actual coverage. The runtime builds the card from that rectangle, so no
 * quad rasterizes the empty margin an AABB would force (~20% of the area on
 * these plants, and every one of those fragments would cost a texture tap).
 *
 * Post-processing, all on the CPU so the alpha chain can be done properly:
 *   - coverage-weighted 2x downsample (depth takes the NEAREST subsample, not
 *     the average, so silhouette edges do not grow ghost surfaces)
 *   - a few dilation passes so bilinear/mip taps never pull in background
 *   - the full mip chain with Castano alpha rescaling: every level is scaled
 *     so the fraction of texels passing the alpha reference matches level 0.
 *     Without it a hard alpha test dissolves distant grass into nothing.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'DWI1', version, tilePx, layers, nRows, mipLevels, topLayer, pad
 *   f32 cx, cy, cz, hx, hy, hz, y0, y1            (local frame, metres)
 *   f32 rowElev[4] (radians), rowAz[4], rowOffset[4]
 *   ...zeros to byte 128
 *   f32[layers*16]  view table: R.xyz,eu | U.xyz,ev | F.xyz,ew | u0,u1,v0,v1
 *   u8[...]         albedo mip chain, level-major then layer-major
 *   u8[...]         geo mip chain, same layout
 */

export const TILE = 352
/** Azimuth count per elevation row, and the row's elevation in degrees. */
export const ROWS: { elDeg: number; az: number }[] = [
  { elDeg: 0, az: 10 },
  { elDeg: 40, az: 6 },
]
export const N_SIDE = ROWS.reduce((a, r) => a + r.az, 0)
export const N_LAYERS = N_SIDE + 1
export const TOP_LAYER = N_SIDE
export const MIP_LEVELS = 9 // 352 -> 1
export const BAKE_ALPHA_REF = 0.4

const SS = 2
const BIG = TILE * SS
const MAGIC = 0x31495744 // 'DWI1'
const VERSION = 2
const HEADER_BYTES = 128
const DILATE_PASSES = 5
const VIEW_FLOATS = 16
/** Coverage level a texel must reach to count toward the tight card rect. */
const RECT_COVER = 0.12
const RECT_MARGIN = 3

// Canopy density grid used for baked AO.
const GX = 24
const GY = 40
const GZ = 24
const AO_DIRS = 12
const AO_STEPS = 6
const AO_STEP_VOX = 1.5

export interface ViewSet {
  tilePx: number
  layers: number
  nRows: number
  mipLevels: number
  topLayer: number
  /** Local capture centre relative to the plant base, metres at scale 1. */
  cx: number
  cy: number
  cz: number
  /** Local AABB half-extents about that centre. */
  hx: number
  hy: number
  hz: number
  y0: number
  y1: number
  rowElev: number[]
  rowAz: number[]
  rowOffset: number[]
  /** layers*16 floats: R.xyz,eu | U.xyz,ev | F.xyz,ew | u0,u1,v0,v1 */
  viewTable: Float32Array<ArrayBuffer>
  albedo: Uint8Array<ArrayBuffer>
  geo: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

/** Texel count of the whole mip chain for one layer. */
export function mipTexels(tile: number, levels: number): number {
  let n = 0
  let s = tile
  for (let i = 0; i < levels; i++) {
    n += s * s
    s = Math.max(1, s >> 1)
  }
  return n
}

/** Texel offset of a level inside a level-major mip chain. */
export function levelOffsetTexels(tile: number, level: number, layers: number): number {
  let off = 0
  let s = tile
  for (let i = 0; i < level; i++) {
    off += s * s * layers
    s = Math.max(1, s >> 1)
  }
  return off
}

function planeBytes(tile: number, levels: number, layers: number): number {
  return mipTexels(tile, levels) * layers * 4
}

export function unpackViewSet(buf: ArrayBuffer): ViewSet | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const tilePx = u[2]!
  const layers = u[3]!
  const nRows = u[4]!
  const mipLevels = u[5]!
  const topLayer = u[6]!
  if (tilePx !== TILE || layers !== N_LAYERS || mipLevels !== MIP_LEVELS) return null
  const f = new Float32Array(buf, 32, 20)
  const tableBytes = layers * VIEW_FLOATS * 4
  const plane = planeBytes(tilePx, mipLevels, layers)
  if (buf.byteLength !== HEADER_BYTES + tableBytes + plane * 2) return null
  return {
    tilePx,
    layers,
    nRows,
    mipLevels,
    topLayer,
    cx: f[0]!,
    cy: f[1]!,
    cz: f[2]!,
    hx: f[3]!,
    hy: f[4]!,
    hz: f[5]!,
    y0: f[6]!,
    y1: f[7]!,
    rowElev: [f[8]!, f[9]!, f[10]!, f[11]!],
    rowAz: [f[12]!, f[13]!, f[14]!, f[15]!],
    rowOffset: [f[16]!, f[17]!, f[18]!, f[19]!],
    viewTable: new Float32Array(buf.slice(HEADER_BYTES, HEADER_BYTES + tableBytes)),
    albedo: new Uint8Array(buf, HEADER_BYTES + tableBytes, plane),
    geo: new Uint8Array(buf, HEADER_BYTES + tableBytes + plane, plane),
  }
}

function packViewSet(v: ViewSet): ArrayBuffer {
  const tableBytes = v.layers * VIEW_FLOATS * 4
  const plane = planeBytes(v.tilePx, v.mipLevels, v.layers)
  const buf = new ArrayBuffer(HEADER_BYTES + tableBytes + plane * 2)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = v.tilePx
  u[3] = v.layers
  u[4] = v.nRows
  u[5] = v.mipLevels
  u[6] = v.topLayer
  const f = new Float32Array(buf, 32, 20)
  f.set([v.cx, v.cy, v.cz, v.hx, v.hy, v.hz, v.y0, v.y1], 0)
  for (let i = 0; i < 4; i++) {
    f[8 + i] = v.rowElev[i] ?? 0
    f[12 + i] = v.rowAz[i] ?? 0
    f[16 + i] = v.rowOffset[i] ?? 0
  }
  new Float32Array(buf, HEADER_BYTES, v.layers * VIEW_FLOATS).set(v.viewTable)
  new Uint8Array(buf, HEADER_BYTES + tableBytes, plane).set(v.albedo)
  new Uint8Array(buf, HEADER_BYTES + tableBytes + plane, plane).set(v.geo)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the view set for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesViews(ctx: BakeCtx, speciesId: string): Promise<ViewSet> {
  const key = `views-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesViews(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let set = unpackViewSet(buf)
  if (!set) {
    buf = await runBake()
    set = unpackViewSet(buf)
    if (!set) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return set
}

/** Overwrite a poisoned OPFS cache entry (mirrors src/bake/cache.ts naming). */
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
// view set geometry
// ---------------------------------------------------------------------------

interface ViewBasis {
  r: [number, number, number]
  u: [number, number, number]
  f: [number, number, number]
}

/** (R, U, F) for a view at azimuth/elevation; F points toward the camera. */
function basisFor(az: number, el: number): ViewBasis {
  const ca = Math.cos(az)
  const sa = Math.sin(az)
  const ce = Math.cos(el)
  const se = Math.sin(el)
  return {
    r: [ca, 0, -sa],
    u: [-sa * se, ce, -ca * se],
    f: [sa * ce, se, ca * ce],
  }
}

export function viewBases(): ViewBasis[] {
  const out: ViewBasis[] = []
  for (const row of ROWS) {
    const el = (row.elDeg * Math.PI) / 180
    for (let k = 0; k < row.az; k++) out.push(basisFor((k * 2 * Math.PI) / row.az, el))
  }
  out.push(basisFor(0, Math.PI / 2))
  return out
}

// ---------------------------------------------------------------------------
// baked ambient occlusion from a canopy density grid
// ---------------------------------------------------------------------------

/**
 * Splat the source vertices into a coarse density grid, then trace a few short
 * hemisphere rays through it per voxel. Grass is a chaotic volume, so a
 * density-transmittance estimate is a much better occlusion cue than anything
 * derivable from a single view — and it is what makes the canopy interior read
 * as interior instead of as flat lit paint.
 */
function buildAoGrid(
  mesh: GcMesh,
  gmin: [number, number, number],
  gmax: [number, number, number],
): Uint8Array<ArrayBuffer> {
  const hdr = mesh.header
  const verts = mesh.vertices
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535
  const inv = [GX / (gmax[0] - gmin[0]), GY / (gmax[1] - gmin[1]), GZ / (gmax[2] - gmin[2])]

  const counts = new Float32Array(GX * GY * GZ)
  for (let i = 0; i < hdr.vertexCount; i++) {
    const b = i * 8
    const gx = Math.floor((bx0 + verts[b]! * sx - gmin[0]) * inv[0]!)
    const gy = Math.floor((by0 + verts[b + 1]! * sy - gmin[1]) * inv[1]!)
    const gz = Math.floor((bz0 + verts[b + 2]! * sz - gmin[2]) * inv[2]!)
    if (gx < 0 || gy < 0 || gz < 0 || gx >= GX || gy >= GY || gz >= GZ) continue
    counts[(gy * GZ + gz) * GX + gx]! += 1
  }

  // Normalize against the mean of the occupied voxels so the extinction scale
  // is mesh-density independent.
  let sum = 0
  let occupied = 0
  for (let i = 0; i < counts.length; i++) {
    if (counts[i]! > 0) {
      sum += counts[i]!
      occupied++
    }
  }
  const ref = occupied > 0 ? sum / occupied : 1
  const opacity = new Float32Array(counts.length)
  for (let i = 0; i < counts.length; i++) opacity[i] = Math.min(1, counts[i]! / (ref * 1.6))

  // Fixed hemisphere directions, cosine weighted.
  const dirs: [number, number, number, number][] = []
  for (let i = 0; i < AO_DIRS; i++) {
    const t = (i + 0.5) / AO_DIRS
    const y = Math.sqrt(t) * 0.95 + 0.05
    const rr = Math.sqrt(Math.max(0, 1 - y * y))
    const ang = i * 2.399963
    dirs.push([Math.cos(ang) * rr, y, Math.sin(ang) * rr, y])
  }

  const at = (x: number, y: number, z: number): number => {
    const ix = Math.round(x)
    const iy = Math.round(y)
    const iz = Math.round(z)
    if (ix < 0 || iy < 0 || iz < 0 || ix >= GX || iy >= GY || iz >= GZ) return 0
    return opacity[(iy * GZ + iz) * GX + ix]!
  }

  const ao = new Float32Array(counts.length)
  for (let j = 0; j < GY; j++) {
    for (let k = 0; k < GZ; k++) {
      for (let i = 0; i < GX; i++) {
        let acc = 0
        let wsum = 0
        for (const [dx, dy, dz, w] of dirs) {
          let t = 1
          for (let s = 1; s <= AO_STEPS; s++) {
            const d = s * AO_STEP_VOX
            t *= 1 - at(i + dx * d, j + dy * d, k + dz * d)
          }
          acc += w * t
          wsum += w
        }
        ao[(j * GZ + k) * GX + i] = acc / wsum
      }
    }
  }

  // One 3x3x3 box smoothing pass — the raw trace is noisy at grid resolution.
  const out = new Uint8Array(counts.length)
  for (let j = 0; j < GY; j++) {
    for (let k = 0; k < GZ; k++) {
      for (let i = 0; i < GX; i++) {
        let s = 0
        let n = 0
        for (let dj = -1; dj <= 1; dj++) {
          const jj = j + dj
          if (jj < 0 || jj >= GY) continue
          for (let dk = -1; dk <= 1; dk++) {
            const kk = k + dk
            if (kk < 0 || kk >= GZ) continue
            for (let di = -1; di <= 1; di++) {
              const ii = i + di
              if (ii < 0 || ii >= GX) continue
              s += ao[(jj * GZ + kk) * GX + ii]!
              n++
            }
          }
        }
        out[(j * GZ + k) * GX + i] = Math.round(Math.min(1, s / n) * 255)
      }
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// the bake itself
// ---------------------------------------------------------------------------

async function bakeSpeciesViews(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax

  const y0 = Math.min(0, by0)
  const y1 = by1
  const cx = (bx0 + bx1) / 2
  const cy = (y0 + y1) / 2
  const cz = (bz0 + bz1) / 2
  const hx = (bx1 - bx0) / 2
  const hy = (y1 - y0) / 2
  const hz = (bz1 - bz0) / 2

  const bases = viewBases()
  const viewTable = new Float32Array(N_LAYERS * VIEW_FLOATS)
  const proj = (a: [number, number, number]): number =>
    Math.abs(a[0]) * hx + Math.abs(a[1]) * hy + Math.abs(a[2]) * hz
  bases.forEach((b, i) => {
    const o = i * VIEW_FLOATS
    const eu = proj(b.r) * 1.02 + 1e-3
    const ev = proj(b.u) * 1.02 + 1e-3
    const ew = proj(b.f) * 1.02 + 1e-3
    viewTable.set([b.r[0], b.r[1], b.r[2], eu], o)
    viewTable.set([b.u[0], b.u[1], b.u[2], ev], o + 4)
    viewTable.set([b.f[0], b.f[1], b.f[2], ew], o + 8)
    viewTable.set([-eu, eu, -ev, ev], o + 12) // replaced by the coverage rect
  })

  // --- AO grid --------------------------------------------------------------
  const pad = 0.06
  const gmin: [number, number, number] = [cx - hx * 1.05 - pad, cy - hy * 1.05 - pad, cz - hz * 1.05 - pad]
  const gmax: [number, number, number] = [cx + hx * 1.05 + pad, cy + hy * 1.05 + pad, cz + hz * 1.05 + pad]
  const aoBytes = buildAoGrid(mesh, gmin, gmax)
  const aoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-ao`,
      size: [GX, GY, GZ],
      dimension: '3d',
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeTexture({ texture: aoTex }, aoBytes, { bytesPerRow: GX, rowsPerImage: GY }, [GX, GY, GZ])
  const aoSampler = device.createSampler({
    label: `${ctx.id}/bake-ao-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  })

  // --- transient GPU resources ---------------------------------------------
  const verts = mesh.vertices
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkArray = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [BIG, BIG, N_LAYERS],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkArray('bake-albedo')
  const geoTex = mkArray('bake-geo')
  const depthBuf = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-zbuffer`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_LAYERS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_LAYERS)
  for (let k = 0; k < N_LAYERS; k++) {
    const o = (k * STRIDE) / 4
    const v = k * VIEW_FLOATS
    scratch.set(viewTable.subarray(v, v + 12), o)
    scratch.set([cx, cy, cz, 0], o + 12)
    scratch.set([bx0, by0, bz0, 0], o + 16)
    scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 20)
    scratch.set([gmin[0], gmin[1], gmin[2], 0], o + 24)
    scratch.set([1 / (gmax[0] - gmin[0]), 1 / (gmax[1] - gmin[1]), 1 / (gmax[2] - gmin[2]), 0], o + 28)
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 128 },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, sampler: { type: 'filtering' } },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uni, size: 128 } },
      { binding: 1, resource: aoTex.createView({ dimension: '3d' }) },
      { binding: 2, resource: aoSampler },
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

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
  const layerView = (t: GPUTexture, k: number): GPUTextureView =>
    t.createView({ dimension: '2d', baseArrayLayer: k, arrayLayerCount: 1 })
  for (let k = 0; k < N_LAYERS; k++) {
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-view-${k}`,
      colorAttachments: [
        { view: layerView(albTex, k), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        {
          view: layerView(geoTex, k),
          clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthBuf.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    pass.setBindGroup(0, bg, [k * STRIDE])
    pass.drawIndexed(indices.length)
    pass.end()
  }

  const rbSize = BIG * 4 * BIG * N_LAYERS
  const rbAlb = mkReadback(ctx, 'bake-rb-albedo', rbSize)
  const rbGeo = mkReadback(ctx, 'bake-rb-geo', rbSize)
  const copy = { bytesPerRow: BIG * 4, rowsPerImage: BIG }
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, ...copy }, [BIG, BIG, N_LAYERS])
  enc.copyTextureToBuffer({ texture: geoTex }, { buffer: rbGeo, ...copy }, [BIG, BIG, N_LAYERS])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbGeo.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigGeo = new Uint8Array(rbGeo.getMappedRange()).slice()
  rbAlb.unmap()
  rbGeo.unmap()
  for (const r of [vbuf, ibuf, albTex, geoTex, depthBuf, uni, aoTex, rbAlb, rbGeo]) r.destroy()

  const { albedo, geo, rects } = postProcess(bigAlb, bigGeo)
  // Replace the AABB rectangles with the measured coverage rectangles.
  for (let k = 0; k < N_LAYERS; k++) {
    const o = k * VIEW_FLOATS
    const eu = viewTable[o + 3]!
    const ev = viewTable[o + 7]!
    const r = rects[k]!
    viewTable[o + 12] = (r[0] / TILE) * 2 * eu - eu
    viewTable[o + 13] = (r[1] / TILE) * 2 * eu - eu
    viewTable[o + 14] = ev - (r[3] / TILE) * 2 * ev
    viewTable[o + 15] = ev - (r[2] / TILE) * 2 * ev
  }

  const rowElev = ROWS.map((r) => (r.elDeg * Math.PI) / 180)
  const rowAz = ROWS.map((r) => r.az)
  const rowOffset: number[] = []
  let acc = 0
  for (const r of ROWS) {
    rowOffset.push(acc)
    acc += r.az
  }
  return packViewSet({
    tilePx: TILE,
    layers: N_LAYERS,
    nRows: ROWS.length,
    mipLevels: MIP_LEVELS,
    topLayer: TOP_LAYER,
    cx,
    cy,
    cz,
    hx,
    hy,
    hz,
    y0,
    y1,
    rowElev,
    rowAz,
    rowOffset,
    viewTable,
    albedo,
    geo,
  })
}

function mkReadback(ctx: BakeCtx, label: string, size: number): GPUBuffer {
  return ctx.res.createBuffer(
    { label: `${ctx.id}/${label}`, size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
    { tag: 'bake-scratch' },
  )
}

// ---------------------------------------------------------------------------
// CPU post-processing
// ---------------------------------------------------------------------------

interface Level {
  size: number
  /** rgb + coverage, 4 per texel. */
  alb: Uint8Array
  /** decoded normal xyz, 3 per texel. */
  nrm: Float32Array
  /** depth 0..1, 1 per texel. */
  dep: Float32Array
  /** ao 0..1, 1 per texel. */
  ao: Float32Array
}

/** Pixel-space coverage rect per layer: [x0, x1, y0, y1]. */
type Rect = [number, number, number, number]

function postProcess(
  bigAlb: Uint8Array,
  bigGeo: Uint8Array,
): { albedo: Uint8Array<ArrayBuffer>; geo: Uint8Array<ArrayBuffer>; rects: Rect[] } {
  const chainTexels = mipTexels(TILE, MIP_LEVELS)
  const albedo = new Uint8Array(chainTexels * N_LAYERS * 4)
  const geo = new Uint8Array(chainTexels * N_LAYERS * 4)
  const rects: Rect[] = []

  for (let layer = 0; layer < N_LAYERS; layer++) {
    const l0 = downsampleLayer(bigAlb, bigGeo, layer)
    rects.push(coverageRect(l0))
    dilate(l0)
    const chain: Level[] = [l0]
    for (let m = 1; m < MIP_LEVELS; m++) chain.push(reduce(chain[m - 1]!))
    rescaleAlpha(chain)
    for (let m = 0; m < MIP_LEVELS; m++) {
      const lv = chain[m]!
      const base = (levelOffsetTexels(TILE, m, N_LAYERS) + lv.size * lv.size * layer) * 4
      writeLevel(lv, albedo, geo, base)
    }
  }
  return { albedo, geo, rects }
}

/** Coverage-weighted 2x box downsample; depth takes the NEAREST subsample. */
function downsampleLayer(bigAlb: Uint8Array, bigGeo: Uint8Array, layer: number): Level {
  const N = TILE
  const alb = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const dep = new Float32Array(N * N)
  const ao = new Float32Array(N * N)
  const base = layer * BIG * BIG * 4
  const n = [0, 0, 0] as [number, number, number]

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let aoSum = 0
      let dBest = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = base + ((y * SS + j) * BIG + (x * SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
          octDecode((bigGeo[s]! / 255) * 2 - 1, (bigGeo[s + 1]! / 255) * 2 - 1, n)
          nx += n[0] * a
          ny += n[1] * a
          nz += n[2] * a
          aoSum += (bigGeo[s + 3]! / 255) * a
          const d = bigGeo[s + 2]! / 255
          if (d > dBest) dBest = d
        }
      }
      const o = (y * N + x) * 4
      const o3 = (y * N + x) * 3
      if (aSum > 0) {
        alb[o] = Math.round(r / aSum)
        alb[o + 1] = Math.round(g / aSum)
        alb[o + 2] = Math.round(b / aSum)
        alb[o + 3] = Math.round(aSum / (SS * SS))
        nrm[o3] = nx / aSum
        nrm[o3 + 1] = ny / aSum
        nrm[o3 + 2] = nz / aSum
        dep[y * N + x] = dBest
        ao[y * N + x] = aoSum / aSum
      } else {
        nrm[o3 + 1] = 1
        dep[y * N + x] = 0.5
        ao[y * N + x] = 1
      }
    }
  }
  return { size: N, alb, nrm, dep, ao }
}

/** Tight bounding box (pixels) of texels that actually carry coverage. */
function coverageRect(lv: Level): Rect {
  const N = lv.size
  const cut = RECT_COVER * 255
  let x0 = N
  let x1 = -1
  let y0 = N
  let y1 = -1
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (lv.alb[(y * N + x) * 4 + 3]! < cut) continue
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
    }
  }
  if (x1 < 0) return [0, N, 0, N]
  return [
    Math.max(0, x0 - RECT_MARGIN),
    Math.min(N, x1 + 1 + RECT_MARGIN),
    Math.max(0, y0 - RECT_MARGIN),
    Math.min(N, y1 + 1 + RECT_MARGIN),
  ]
}

/** Push colour/normal/depth/ao a few texels into empty space (alpha stays 0). */
function dilate(lv: Level): void {
  const N = lv.size
  const filled = new Uint8Array(N * N)
  for (let i = 0; i < N * N; i++) filled[i] = lv.alb[i * 4 + 3]! > 0 ? 1 : 0
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = filled.slice()
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (filled[idx] !== 0) continue
        let n = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        let d = 0
        let a = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= N) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < 0 || xx >= N || (i === 0 && j === 0)) continue
            const s = yy * N + xx
            if (filled[s] === 0) continue
            n++
            r += lv.alb[s * 4]!
            g += lv.alb[s * 4 + 1]!
            b += lv.alb[s * 4 + 2]!
            nx += lv.nrm[s * 3]!
            ny += lv.nrm[s * 3 + 1]!
            nz += lv.nrm[s * 3 + 2]!
            d += lv.dep[s]!
            a += lv.ao[s]!
          }
        }
        if (n === 0) continue
        lv.alb[idx * 4] = Math.round(r / n)
        lv.alb[idx * 4 + 1] = Math.round(g / n)
        lv.alb[idx * 4 + 2] = Math.round(b / n)
        lv.nrm[idx * 3] = nx / n
        lv.nrm[idx * 3 + 1] = ny / n
        lv.nrm[idx * 3 + 2] = nz / n
        lv.dep[idx] = d / n
        lv.ao[idx] = a / n
        next[idx] = 1
      }
    }
    filled.set(next)
  }
}

/** Coverage-weighted 2x reduction of a level (colours already dilated). */
function reduce(src: Level): Level {
  const N = Math.max(1, src.size >> 1)
  const alb = new Uint8Array(N * N * 4)
  const nrm = new Float32Array(N * N * 3)
  const dep = new Float32Array(N * N)
  const ao = new Float32Array(N * N)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let aSum = 0
      let wSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let d = 0
      let aoS = 0
      let rF = 0
      let gF = 0
      let bF = 0
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          const sx = Math.min(src.size - 1, x * 2 + i)
          const sy = Math.min(src.size - 1, y * 2 + j)
          const s = sy * src.size + sx
          const a = src.alb[s * 4 + 3]! / 255
          aSum += a
          rF += src.alb[s * 4]!
          gF += src.alb[s * 4 + 1]!
          bF += src.alb[s * 4 + 2]!
          const w = Math.max(a, 1e-4)
          wSum += w
          r += src.alb[s * 4]! * w
          g += src.alb[s * 4 + 1]! * w
          b += src.alb[s * 4 + 2]! * w
          nx += src.nrm[s * 3]! * w
          ny += src.nrm[s * 3 + 1]! * w
          nz += src.nrm[s * 3 + 2]! * w
          d += src.dep[s]! * w
          aoS += src.ao[s]! * w
        }
      }
      const o = (y * N + x) * 4
      alb[o] = Math.round(aSum > 0 ? r / wSum : rF / 4)
      alb[o + 1] = Math.round(aSum > 0 ? g / wSum : gF / 4)
      alb[o + 2] = Math.round(aSum > 0 ? b / wSum : bF / 4)
      alb[o + 3] = Math.round((aSum / 4) * 255)
      const o3 = (y * N + x) * 3
      nrm[o3] = nx / wSum
      nrm[o3 + 1] = ny / wSum
      nrm[o3 + 2] = nz / wSum
      dep[y * N + x] = d / wSum
      ao[y * N + x] = aoS / wSum
    }
  }
  return { size: N, alb, nrm, dep, ao }
}

/**
 * Castano alpha rescaling: pick, per mip level, the scale that preserves the
 * fraction of texels above the alpha reference. A plain box-filtered mip chain
 * makes a hard alpha test erase distant grass; this keeps far plants solid.
 */
function rescaleAlpha(chain: Level[]): void {
  const l0 = chain[0]!
  const n0 = l0.size * l0.size
  let covered = 0
  for (let i = 0; i < n0; i++) if (l0.alb[i * 4 + 3]! / 255 >= BAKE_ALPHA_REF) covered++
  const target = covered / n0
  if (target <= 0) return

  for (let m = 1; m < chain.length; m++) {
    const lv = chain[m]!
    const n = lv.size * lv.size
    let lo = 1
    let hi = 6
    for (let it = 0; it < 12; it++) {
      const s = (lo + hi) / 2
      let c = 0
      for (let i = 0; i < n; i++) if (Math.min(1, (lv.alb[i * 4 + 3]! / 255) * s) >= BAKE_ALPHA_REF) c++
      if (c / n < target) lo = s
      else hi = s
    }
    const s = (lo + hi) / 2
    for (let i = 0; i < n; i++) lv.alb[i * 4 + 3] = Math.round(Math.min(255, lv.alb[i * 4 + 3]! * s))
  }
}

function writeLevel(lv: Level, albedo: Uint8Array, geo: Uint8Array, base: number): void {
  const n = lv.size * lv.size
  for (let i = 0; i < n; i++) {
    const o = base + i * 4
    albedo[o] = lv.alb[i * 4]!
    albedo[o + 1] = lv.alb[i * 4 + 1]!
    albedo[o + 2] = lv.alb[i * 4 + 2]!
    albedo[o + 3] = lv.alb[i * 4 + 3]!
    const [ou, ov] = octEncode(lv.nrm[i * 3]!, lv.nrm[i * 3 + 1]!, lv.nrm[i * 3 + 2]!)
    geo[o] = ou
    geo[o + 1] = ov
    geo[o + 2] = Math.round(Math.max(0, Math.min(1, lv.dep[i]!)) * 255)
    geo[o + 3] = Math.round(Math.max(0, Math.min(1, lv.ao[i]!)) * 255)
  }
}

/** Octahedral decode, y-primary — mirrors oct_decode_mesh in bake.wgsl. */
function octDecode(u: number, v: number, out: [number, number, number]): void {
  let x = u
  let z = v
  const y = 1 - Math.abs(u) - Math.abs(v)
  if (y < 0) {
    x = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1)
    z = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1)
  }
  const len = Math.hypot(x, y, z) || 1
  out[0] = x / len
  out[1] = y / len
  out[2] = z / len
}

/** Octahedral encode, y-primary — exact inverse of octDecode. */
function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  let u: number
  let v: number
  if (s < 1e-6) {
    u = 0 // decodes to +Y
    v = 0
  } else {
    u = x / s
    v = z / s
    if (y < 0) {
      const fu = (1 - Math.abs(v)) * (x >= 0 ? 1 : -1)
      const fv = (1 - Math.abs(u)) * (z >= 0 ? 1 : -1)
      u = fu
      v = fv
    }
  }
  return [Math.round((u * 0.5 + 0.5) * 255), Math.round((v * 0.5 + 0.5) * 255)]
}
