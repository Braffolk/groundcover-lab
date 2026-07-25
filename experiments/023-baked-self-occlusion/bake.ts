import bakeViewsSrc from './shaders/bake_views.wgsl'
import bakeVisSrc from './shaders/bake_vis.wgsl'
import bakeResolveSrc from './shaders/bake_resolve.wgsl'
import bakeDilateSrc from './shaders/bake_dilate.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake for "baked self-occlusion cards".
 *
 * Two atlases per species, both indexed by the SAME 5x5 grid of baked view
 * directions (8 azimuths x 3 elevations + 1 straight-down top):
 *
 *   albedo atlas  384 px/tile — rgb = authored colour, a = coverage
 *   geometry atlas 128 px/tile — [octN.x, octN.y, ao, q]
 *       q    normalized depth of the first surface along the baked view,
 *            0 at the tile's near plane, 1 at its far plane. This is what
 *            makes the flat card parallax.
 *       ao   fraction of the upper hemisphere visible from that texel,
 *            solved at bake time from 32 orthographic visibility maps.
 *       octN shading normal = the geometric normal leaned a third of the way
 *            toward the mean unoccluded direction (the "bent normal"), so
 *            lighting follows the clump's shape as well as blade facets.
 *
 * Visibility is solved against the plant PLUS its 8 canopy neighbours (offset
 * by the mesh's own periodic tile size), which is what actually produces dark
 * bases and lit tips — an isolated tuft sees ~87% of the sky everywhere.
 *
 * Each tile uses its OWN tight silhouette box (computed per view direction),
 * so no tile area is wasted on empty margin and the runtime can map a fragment
 * ray straight into tile UV space.
 *
 * Artifact layout (little-endian):
 *   0  u32 magic 'BSO1', u32 version, u32 atlasA, u32 atlasB
 *   16 u32 tileA, u32 tileB, u32 grid, u32 nTiles
 *   32 f32 sphereR, f32 aabbCx, aabbCy, aabbCz
 *   48 f32 aabbHx, aabbHy, aabbHz, f32 pad
 *   64 f32[25*16] tile table  (r.xyz,boxCx)(u.xyz,boxCy)(b.xyz,dNear)(hx,hy,D,0)
 *   1664 u8[atlasA^2*4] albedo rgba8
 *   ...  u8[atlasB^2*4] geometry rgba8
 */

export const GRID = 5
export const TILE_A = 384
export const ATLAS_A = TILE_A * GRID // 1920
export const TILE_B = 128
export const ATLAS_B = TILE_B * GRID // 640
export const N_TILES = 25
/**
 * The 25 baked view directions. Azimuth resolution follows foreshortening:
 * near the horizon the silhouette swings hard with azimuth and needs 8 steps,
 * but at 52-74deg elevation the plant is compressed into its footprint and
 * azimuth barely matters — so those rows spend their tiles on ELEVATION
 * instead. That buys a worst-case elevation error of ~11deg (a uniform
 * 8x3+top grid leaves 21deg, and the visible symptom is a density ring in the
 * top-down view where the tile switches).
 */
export const ROW_ELEV_DEG = [6, 30, 52, 74]
export const ROW_AZ = [8, 8, 4, 4]
export const ROW_BASE = [0, 8, 16, 20]
export const TOP_TILE = 24

const SS = 2
const SS_TILE = TILE_A * SS // 768
const ROW_W = SS_TILE * GRID // 3840
const ROW_H = SS_TILE
const N_VIS = 32
const VIS_TILE = 256
const VIS_COLS = 8
const VIS_ROWS = 4
const MAGIC = 0x314f5342 // 'BSO1'
const VERSION = 5
const HEADER_BYTES = 64
const TILE_TABLE_FLOATS = N_TILES * 16
const TABLE_BYTES = TILE_TABLE_FLOATS * 4
const PIXELS_OFFSET = HEADER_BYTES + TABLE_BYTES
/** Vertices sampled when fitting per-view silhouette boxes (bounded by AABB). */
const BBOX_SAMPLES = 400_000

export interface SelfOccAtlas {
  atlasA: number
  atlasB: number
  /** Bounding-sphere radius at scale 1 (m), around aabbC. */
  sphereR: number
  /** Plant-local AABB (origin = clump centre in XZ, ground at y=0). */
  aabbC: [number, number, number]
  aabbH: [number, number, number]
  /** 25 x 16 floats — uploaded verbatim as the runtime tile uniform. */
  tileTable: Float32Array<ArrayBuffer>
  albedo: Uint8Array<ArrayBuffer>
  geom: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

type V3 = [number, number, number]

function cross(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
function norm(a: V3): V3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** Baked view direction (toward the camera) for tile `k`, plant-local frame. */
export function viewDir(k: number): V3 {
  if (k >= TOP_TILE) return [0, 1, 0]
  let row = 0
  while (row < ROW_BASE.length - 1 && k >= ROW_BASE[row + 1]!) row++
  const az = ((k - ROW_BASE[row]!) * 2 * Math.PI) / ROW_AZ[row]!
  const el = (ROW_ELEV_DEG[row]! * Math.PI) / 180
  return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)]
}

/** Orthonormal capture frame (right, up) for a direction, y-up biased. */
function frameFor(d: V3): { r: V3; u: V3 } {
  if (Math.abs(d[1]) > 0.999) {
    const r: V3 = [1, 0, 0]
    return { r, u: cross(d, r) }
  }
  const r = norm(cross([0, 1, 0], d))
  return { r, u: cross(d, r) }
}

/** 32 directions over the upper hemisphere, uniform in solid angle. */
export function probeDirs(): V3[] {
  const out: V3[] = []
  const golden = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < N_VIS; i++) {
    const cosT = (i + 0.5) / N_VIS
    const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT))
    const phi = i * golden
    out.push([sinT * Math.cos(phi), cosT, sinT * Math.sin(phi)])
  }
  return out
}

export function unpackAtlas(buf: ArrayBuffer): SelfOccAtlas | null {
  if (buf.byteLength < PIXELS_OFFSET) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const atlasA = u[2]!
  const atlasB = u[3]!
  if (atlasA !== ATLAS_A || atlasB !== ATLAS_B || u[6] !== GRID || u[7] !== N_TILES) return null
  const expected = PIXELS_OFFSET + atlasA * atlasA * 4 + atlasB * atlasB * 4
  if (buf.byteLength !== expected) return null
  const f = new Float32Array(buf, 32, 7)
  return {
    atlasA,
    atlasB,
    sphereR: f[0]!,
    aabbC: [f[1]!, f[2]!, f[3]!],
    aabbH: [f[4]!, f[5]!, f[6]!],
    tileTable: new Float32Array(buf.slice(HEADER_BYTES, HEADER_BYTES + TABLE_BYTES)),
    albedo: new Uint8Array(buf, PIXELS_OFFSET, atlasA * atlasA * 4),
    geom: new Uint8Array(buf, PIXELS_OFFSET + atlasA * atlasA * 4, atlasB * atlasB * 4),
  }
}

function packAtlas(a: SelfOccAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(PIXELS_OFFSET + a.albedo.byteLength + a.geom.byteLength)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.atlasA
  u[3] = a.atlasB
  u[4] = TILE_A
  u[5] = TILE_B
  u[6] = GRID
  u[7] = N_TILES
  const f = new Float32Array(buf, 32, 7)
  f.set([a.sphereR, ...a.aabbC, ...a.aabbH])
  new Float32Array(buf, HEADER_BYTES, TILE_TABLE_FLOATS).set(a.tileTable)
  new Uint8Array(buf, PIXELS_OFFSET, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, PIXELS_OFFSET + a.albedo.byteLength, a.geom.byteLength).set(a.geom)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake this species' atlases. The dev
 * server answers missing /mesh/baked files with index.html at status 200, so
 * every result is magic-validated and a poisoned cache entry is repaired.
 */
export async function loadSpeciesAtlas(ctx: BakeCtx, speciesId: string): Promise<SelfOccAtlas> {
  const key = `selfocc-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpecies(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackAtlas(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackAtlas(buf)
    if (!atlas) throw new Error(`[${ctx.id}] bake for ${speciesId} produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return atlas
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

/**
 * Fit the plant-local frame, bounding sphere and one tight silhouette box per
 * baked view. Boxes come from a strided vertex sample (400k points is plenty
 * for an extent), inflated 8%, then clamped to the EXACT projection of the
 * mesh AABB so a box can never clip real geometry.
 */
function fitGeometry(mesh: GcMesh): {
  anchor: V3
  aabbC: V3
  aabbH: V3
  sphereR: number
  tileTable: Float32Array<ArrayBuffer>
} {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const yGround = Math.min(0, by0)
  const anchor: V3 = [cx, yGround, cz]

  const aabbLo: V3 = [bx0 - cx, by0 - yGround, bz0 - cz]
  const aabbHi: V3 = [bx1 - cx, by1 - yGround, bz1 - cz]
  const aabbC: V3 = [0, (aabbLo[1] + aabbHi[1]) / 2, 0]
  const aabbH: V3 = [(aabbHi[0] - aabbLo[0]) / 2, (aabbHi[1] - aabbLo[1]) / 2, (aabbHi[2] - aabbLo[2]) / 2]
  // AABB centre in x/z is 0 by construction; keep the true offset if not.
  aabbC[0] = (aabbLo[0] + aabbHi[0]) / 2
  aabbC[2] = (aabbLo[2] + aabbHi[2]) / 2

  const n = hdr.vertexCount
  const stride = Math.max(1, Math.floor(n / BBOX_SAMPLES))
  const verts = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535
  const m = Math.ceil(n / stride)
  const pts = new Float32Array(m * 3)
  let sphereR2 = 0
  let w = 0
  for (let i = 0; i < n; i += stride) {
    const px = bx0 + verts[i * 8]! * sx - cx
    const py = by0 + verts[i * 8 + 1]! * sy - yGround
    const pz = bz0 + verts[i * 8 + 2]! * sz - cz
    pts[w * 3] = px
    pts[w * 3 + 1] = py
    pts[w * 3 + 2] = pz
    w++
    const dx = px - aabbC[0]
    const dy = py - aabbC[1]
    const dz = pz - aabbC[2]
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 > sphereR2) sphereR2 = d2
  }
  const aabbDiag = Math.hypot(aabbH[0], aabbH[1], aabbH[2])
  const sphereR = Math.min(Math.sqrt(sphereR2) * 1.04 + 1e-3, aabbDiag)

  const table = new Float32Array(TILE_TABLE_FLOATS)
  const support = (d: V3): number => Math.abs(d[0]) * aabbH[0] + Math.abs(d[1]) * aabbH[1] + Math.abs(d[2]) * aabbH[2]
  const centerAlong = (d: V3): number => d[0] * aabbC[0] + d[1] * aabbC[1] + d[2] * aabbC[2]
  for (let k = 0; k < N_TILES; k++) {
    const b = viewDir(k)
    const { r, u } = frameFor(b)
    let rMin = Infinity
    let rMax = -Infinity
    let uMin = Infinity
    let uMax = -Infinity
    let bMin = Infinity
    let bMax = -Infinity
    for (let i = 0; i < w; i++) {
      const px = pts[i * 3]!
      const py = pts[i * 3 + 1]!
      const pz = pts[i * 3 + 2]!
      const pr = px * r[0] + py * r[1] + pz * r[2]
      const pu = px * u[0] + py * u[1] + pz * u[2]
      const pb = px * b[0] + py * b[1] + pz * b[2]
      if (pr < rMin) rMin = pr
      if (pr > rMax) rMax = pr
      if (pu < uMin) uMin = pu
      if (pu > uMax) uMax = pu
      if (pb < bMin) bMin = pb
      if (pb > bMax) bMax = pb
    }
    // Inflate for the unsampled vertices, then clamp to the exact AABB extent.
    const grow = (lo: number, hi: number, d: V3): [number, number] => {
      const c = (lo + hi) / 2
      const h = Math.min(((hi - lo) / 2) * 1.08 + 0.008, support(d))
      const cc = Math.max(centerAlong(d) - support(d) + h, Math.min(centerAlong(d) + support(d) - h, c))
      return [cc, h]
    }
    const [boxCx, hx] = grow(rMin, rMax, r)
    const [boxCy, hy] = grow(uMin, uMax, u)
    const bPad = 0.01 + 0.04 * (bMax - bMin)
    const dNear = Math.min(bMax + bPad, centerAlong(b) + support(b))
    const dFar = Math.max(bMin - bPad, centerAlong(b) - support(b))
    const o = k * 16
    table.set([r[0], r[1], r[2], boxCx], o)
    table.set([u[0], u[1], u[2], boxCy], o + 4)
    table.set([b[0], b[1], b[2], dNear], o + 8)
    table.set([hx, hy, Math.max(dNear - dFar, 1e-3), 0], o + 12)
  }
  return { anchor, aabbC, aabbH, sphereR, tileTable: table }
}

async function bakeSpecies(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const fit = fitGeometry(mesh)
  const hdr = mesh.header
  const bMin = hdr.boundsMin
  const bRange: V3 = [
    hdr.boundsMax[0] - bMin[0],
    hdr.boundsMax[1] - bMin[1],
    hdr.boundsMax[2] - bMin[2],
  ]

  // --- mesh on the GPU -------------------------------------------------------
  const vbuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-verts`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)
  const vertexLayout: GPUVertexBufferLayout = {
    arrayStride: 16,
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'uint16x4' },
      { shaderLocation: 1, offset: 8, format: 'uint16x4' },
    ],
  }

  // --- pass A: 32 orthographic visibility maps -------------------------------
  const visTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-vis`,
      size: [VIS_TILE * VIS_COLS, VIS_TILE * VIS_ROWS],
      format: 'r32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    },
    { tag: 'bake-scratch' },
  )
  const visDepth = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-vis-depth`,
      size: [VIS_TILE * VIS_COLS, VIS_TILE * VIS_ROWS],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )
  const UNI_STRIDE = 256
  const probes = probeDirs()
  // Canopy spacing for the neighbour ring in the visibility bake. Community
  // tiles are periodic — their own tile size IS the natural spacing. A finite
  // specimen (poa) gets its footprint x1.6, which lands near the ~0.58 m
  // spacing the default stand's 3 plants/m^2 actually produces.
  const tile = hdr.tileSize
  const spacing: [number, number] =
    tile[0] > 1e-4 && tile[1] > 1e-4
      ? [tile[0], tile[1]]
      : [fit.aabbH[0] * 3.2, fit.aabbH[2] * 3.2]
  const visUni = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-vis-uni`,
      size: UNI_STRIDE * N_VIS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  {
    const data = new Float32Array((UNI_STRIDE / 4) * N_VIS)
    probes.forEach((d, i) => {
      const { r, u } = frameFor(d)
      const o = (i * UNI_STRIDE) / 4
      data.set([r[0], r[1], r[2], 0], o)
      data.set([u[0], u[1], u[2], 0], o + 4)
      data.set([d[0], d[1], d[2], 0], o + 8)
      data.set([...fit.aabbC, fit.sphereR], o + 12)
      data.set([...fit.anchor, 0], o + 16)
      data.set([...bMin, 0], o + 20)
      data.set([...bRange, 0], o + 24)
      data.set([spacing[0], spacing[1], 0, 0], o + 28)
    })
    device.queue.writeBuffer(visUni, 0, data)
  }
  const uniBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-uni-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 128 },
      },
    ],
  })
  const visBg = device.createBindGroup({
    label: `${ctx.id}/bake-vis-bg`,
    layout: uniBgl,
    entries: [{ binding: 0, resource: { buffer: visUni, size: UNI_STRIDE } }],
  })
  const visModule = ctx.shaders.module(bakeVisSrc)
  const visPipeline = device.createRenderPipeline({
    label: `${ctx.id}/bake-vis-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-vis-pl`, bindGroupLayouts: [uniBgl] }),
    vertex: { module: visModule, entryPoint: 'vs', buffers: [vertexLayout] },
    fragment: { module: visModule, entryPoint: 'fs', targets: [{ format: 'r32float' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })
  // 9 instances (the plant + its 8 canopy neighbours) x 32 probes is a lot of
  // vertex work for one submission — chunk it so no single command buffer runs
  // long enough to worry a driver watchdog.
  const VIS_CHUNK = 8
  for (let start = 0; start < N_VIS; start += VIS_CHUNK) {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-vis-enc-${start}` })
    const first = start === 0
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-vis-pass-${start}`,
      colorAttachments: [
        {
          view: visTex.createView(),
          clearValue: { r: -1e9, g: 0, b: 0, a: 0 },
          loadOp: first ? 'clear' : 'load',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: visDepth.createView(),
        depthClearValue: 1,
        depthLoadOp: first ? 'clear' : 'load',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(visPipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    for (let k = start; k < Math.min(N_VIS, start + VIS_CHUNK); k++) {
      const col = k % VIS_COLS
      const row = Math.floor(k / VIS_COLS)
      pass.setViewport(col * VIS_TILE, row * VIS_TILE, VIS_TILE, VIS_TILE, 0, 1)
      pass.setScissorRect(col * VIS_TILE, row * VIS_TILE, VIS_TILE, VIS_TILE)
      pass.setBindGroup(0, visBg, [k * UNI_STRIDE])
      pass.drawIndexed(indices.length, 9)
    }
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // --- pass B: per-row view capture + resolve --------------------------------
  const rowAlbedo = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-row-albedo`,
      size: [ROW_W, ROW_H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    },
    { tag: 'bake-scratch' },
  )
  const rowNrm = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-row-nrm`,
      size: [ROW_W, ROW_H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    },
    { tag: 'bake-scratch' },
  )
  const rowDepth = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-row-depth`,
      size: [ROW_W, ROW_H],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )
  const viewUni = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-view-uni`,
      size: UNI_STRIDE * N_TILES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  {
    const data = new Float32Array((UNI_STRIDE / 4) * N_TILES)
    for (let k = 0; k < N_TILES; k++) {
      const t = fit.tileTable.subarray(k * 16, k * 16 + 16)
      const o = (k * UNI_STRIDE) / 4
      data.set(t.subarray(0, 4), o) // r.xyz, boxCx
      data.set(t.subarray(4, 8), o + 4) // u.xyz, boxCy
      data.set(t.subarray(8, 12), o + 8) // b.xyz, dNear
      data.set(t.subarray(12, 16), o + 12) // hx, hy, D, 0
      data.set([...fit.anchor, 0], o + 16)
      data.set([...bMin, 0], o + 20)
      data.set([...bRange, 0], o + 24)
    }
    device.queue.writeBuffer(viewUni, 0, data)
  }
  const viewBg = device.createBindGroup({
    label: `${ctx.id}/bake-view-bg`,
    layout: uniBgl,
    entries: [{ binding: 0, resource: { buffer: viewUni, size: UNI_STRIDE } }],
  })
  const viewModule = ctx.shaders.module(bakeViewsSrc)
  const viewPipeline = device.createRenderPipeline({
    label: `${ctx.id}/bake-view-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-view-pl`, bindGroupLayouts: [uniBgl] }),
    vertex: { module: viewModule, entryPoint: 'vs', buffers: [vertexLayout] },
    fragment: {
      module: viewModule,
      entryPoint: 'fs',
      targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const mkAtlas = (label: string, size: number, extra: number): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [size, size],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | extra,
      },
      { tag: 'bake-scratch' },
    )
  const rawA = mkAtlas('bake-raw-a', ATLAS_A, 0)
  const rawB = mkAtlas('bake-raw-b', ATLAS_B, 0)
  const finalA = mkAtlas('bake-final-a', ATLAS_A, GPUTextureUsage.COPY_SRC)
  const finalB = mkAtlas('bake-final-b', ATLAS_B, GPUTextureUsage.COPY_SRC)
  const maskB = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-mask-b`,
      size: [ATLAS_B, ATLAS_B],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    },
    { tag: 'bake-scratch' },
  )

  // Resolve tables: tile table (100 vec4) + probe frames (96 vec4) + sphere.
  const tablesBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-tables`,
      size: (100 + 96 + 2) * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  {
    const data = new Float32Array((100 + 96 + 2) * 4)
    data.set(fit.tileTable, 0)
    probes.forEach((d, i) => {
      const { r, u } = frameFor(d)
      const o = 400 + i * 12
      data.set([r[0], r[1], r[2], 0], o)
      data.set([u[0], u[1], u[2], 0], o + 4)
      data.set([d[0], d[1], d[2], 0], o + 8)
    })
    data.set([...fit.aabbC, fit.sphereR], 400 + 96 * 4)
    data.set([fit.aabbC[1] + fit.aabbH[1], 0, 0, 0], 404 + 96 * 4)
    device.queue.writeBuffer(tablesBuf, 0, data)
  }
  const rowUniBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-row-uni`, size: UNI_STRIDE * GRID, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  {
    const data = new Float32Array((UNI_STRIDE / 4) * GRID)
    for (let r = 0; r < GRID; r++) data[(r * UNI_STRIDE) / 4] = r
    device.queue.writeBuffer(rowUniBuf, 0, data)
  }

  const resolveBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-resolve-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
    ],
  })
  const resolveBg = device.createBindGroup({
    label: `${ctx.id}/bake-resolve-bg`,
    layout: resolveBgl,
    entries: [
      { binding: 0, resource: { buffer: tablesBuf } },
      { binding: 1, resource: { buffer: rowUniBuf, size: 16 } },
      { binding: 2, resource: rowAlbedo.createView() },
      { binding: 3, resource: rowNrm.createView() },
      { binding: 4, resource: visTex.createView() },
      { binding: 5, resource: rawA.createView() },
      { binding: 6, resource: rawB.createView() },
      { binding: 7, resource: maskB.createView() },
    ],
  })
  const resolveModule = ctx.shaders.module(bakeResolveSrc)
  const resolvePl = device.createPipelineLayout({
    label: `${ctx.id}/bake-resolve-pl`,
    bindGroupLayouts: [resolveBgl],
  })
  const resolveAlbedo = device.createComputePipeline({
    label: `${ctx.id}/bake-resolve-albedo`,
    layout: resolvePl,
    compute: { module: resolveModule, entryPoint: 'cs_albedo' },
  })
  const resolveGeom = device.createComputePipeline({
    label: `${ctx.id}/bake-resolve-geom`,
    layout: resolvePl,
    compute: { module: resolveModule, entryPoint: 'cs_geom' },
  })

  for (let row = 0; row < GRID; row++) {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-row-${row}` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-row-pass-${row}`,
      colorAttachments: [
        {
          view: rowAlbedo.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: rowNrm.createView(),
          clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: rowDepth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(viewPipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    for (let col = 0; col < GRID; col++) {
      const k = row * GRID + col
      pass.setViewport(col * SS_TILE, 0, SS_TILE, SS_TILE, 0, 1)
      pass.setScissorRect(col * SS_TILE, 0, SS_TILE, SS_TILE)
      pass.setBindGroup(0, viewBg, [k * UNI_STRIDE])
      pass.drawIndexed(indices.length)
    }
    pass.end()

    const cs = enc.beginComputePass({ label: `${ctx.id}/bake-resolve-${row}` })
    cs.setBindGroup(0, resolveBg, [row * UNI_STRIDE])
    cs.setPipeline(resolveAlbedo)
    cs.dispatchWorkgroups(Math.ceil((GRID * TILE_A) / 8), Math.ceil(TILE_A / 8))
    cs.setPipeline(resolveGeom)
    cs.dispatchWorkgroups(Math.ceil((GRID * TILE_B) / 8), Math.ceil(TILE_B / 8))
    cs.end()
    device.queue.submit([enc.finish()])
  }

  // --- pass C: dilation ------------------------------------------------------
  const dilateBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-dilate-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
    ],
  })
  const dilateBg = device.createBindGroup({
    label: `${ctx.id}/bake-dilate-bg`,
    layout: dilateBgl,
    entries: [
      { binding: 0, resource: rawA.createView() },
      { binding: 1, resource: rawB.createView() },
      { binding: 2, resource: maskB.createView() },
      { binding: 3, resource: finalA.createView() },
      { binding: 4, resource: finalB.createView() },
    ],
  })
  const dilateModule = ctx.shaders.module(bakeDilateSrc)
  const dilatePl = device.createPipelineLayout({ label: `${ctx.id}/bake-dilate-pl`, bindGroupLayouts: [dilateBgl] })
  {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-dilate` })
    const cs = enc.beginComputePass({ label: `${ctx.id}/bake-dilate-pass` })
    cs.setBindGroup(0, dilateBg)
    cs.setPipeline(
      device.createComputePipeline({
        label: `${ctx.id}/bake-dilate-a`,
        layout: dilatePl,
        compute: { module: dilateModule, entryPoint: 'cs_albedo' },
      }),
    )
    cs.dispatchWorkgroups(Math.ceil(ATLAS_A / 8), Math.ceil(ATLAS_A / 8))
    cs.setPipeline(
      device.createComputePipeline({
        label: `${ctx.id}/bake-dilate-b`,
        layout: dilatePl,
        compute: { module: dilateModule, entryPoint: 'cs_geom' },
      }),
    )
    cs.dispatchWorkgroups(Math.ceil(ATLAS_B / 8), Math.ceil(ATLAS_B / 8))
    cs.end()

    const rbA = ctx.res.createBuffer(
      {
        label: `${ctx.id}/bake-rb-a`,
        size: ATLAS_A * ATLAS_A * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      },
      { tag: 'bake-scratch' },
    )
    const rbB = ctx.res.createBuffer(
      {
        label: `${ctx.id}/bake-rb-b`,
        size: ATLAS_B * ATLAS_B * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      },
      { tag: 'bake-scratch' },
    )
    enc.copyTextureToBuffer({ texture: finalA }, { buffer: rbA, bytesPerRow: ATLAS_A * 4 }, [ATLAS_A, ATLAS_A])
    enc.copyTextureToBuffer({ texture: finalB }, { buffer: rbB, bytesPerRow: ATLAS_B * 4 }, [ATLAS_B, ATLAS_B])
    device.queue.submit([enc.finish()])

    await rbA.mapAsync(GPUMapMode.READ)
    await rbB.mapAsync(GPUMapMode.READ)
    const albedo = new Uint8Array(rbA.getMappedRange()).slice()
    const geom = new Uint8Array(rbB.getMappedRange()).slice()
    rbA.unmap()
    rbB.unmap()
    for (const r of [
      vbuf,
      ibuf,
      visTex,
      visDepth,
      visUni,
      rowAlbedo,
      rowNrm,
      rowDepth,
      viewUni,
      rawA,
      rawB,
      finalA,
      finalB,
      maskB,
      tablesBuf,
      rowUniBuf,
      rbA,
      rbB,
    ]) {
      r.destroy()
    }
    return packAtlas({
      atlasA: ATLAS_A,
      atlasB: ATLAS_B,
      sphereR: fit.sphereR,
      aabbC: fit.aabbC,
      aabbH: fit.aabbH,
      tileTable: fit.tileTable,
      albedo,
      geom,
    })
  }
}
