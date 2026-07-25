import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Two-atlas bake for the depth-reprojected cards.
 *
 * Per species, 16 orthographic captures of the raw GCMESH1 mesh (15 side views
 * at 24deg azimuth steps + 1 straight-down top view) go into a 4x4 grid, twice:
 *
 *   albedo atlas   1536^2 rgba8 (384px tiles) — authored colour, a = coverage
 *   geometry atlas  768^2 rgba8 (192px tiles) — rg = oct normal (mesh frame),
 *                   b = signed depth along the capture axis (0 = "no surface"),
 *                   a = volumetric sky visibility
 *
 * 15 azimuths rather than 8 is a deliberate trade against tile resolution: the
 * reprojection error of a single depth layer scales with the angle between the
 * selected capture and the true eye ray, so halving that angle (22.5deg -> 12deg)
 * halves the disocclusion tearing on close-up foliage — worth much more than the
 * extra 128 texels per tile it costs, at identical VRAM.
 *
 * The depth channel is what turns a card into geometry at runtime: the fragment
 * shader walks the eye ray onto that height field analytically. The sky
 * visibility channel is a real volumetric term — the bake voxelizes the mesh
 * into a triangle-area extinction grid (periodic in xz, so a community tile is
 * occluded by its own neighbours) and integrates transmittance along 13
 * cosine-weighted directions per voxel, which is where the in-clump darkening
 * and the ground contact come from.
 *
 * Both atlases are rendered 2x supersampled (3072^2), coverage-weighted
 * downsampled (2x2 for albedo, 4x4 for geometry) and dilated a few texels into
 * empty space so bilinear/mip sampling and reprojections that land just past a
 * silhouette never read background.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'NDI1', version, atlasAlb, atlasGeo, tileAlb, tileGeo, nSide, pad
 *   f32 rXZ, y0, y1, cx, cz                (capture box, unit scale, metres)
 *   ...zeros to byte 64
 *   u8[atlasAlb^2*4]  albedo rgba8
 *   u8[atlasGeo^2*4]  geometry rgba8
 */

export const TILE_ALB = 384
export const GRID = 4
export const ATLAS_ALB = TILE_ALB * GRID // 1536
export const TILE_GEO = 192
export const ATLAS_GEO = TILE_GEO * GRID // 768
/** 15 azimuths + 1 top view = exactly the 4x4 grid. */
export const N_SIDE = 15

const SS = 2
const BIG = ATLAS_ALB * SS // 3072
const GEO_DOWN = BIG / ATLAS_GEO // 4
const MAGIC = 0x3149444e // 'NDI1'
const VERSION = 3
const HEADER_BYTES = 64
const DILATE_ALB = 6
/**
 * Geometry dilation rings. This is exactly how far a reprojection may legally
 * grow a silhouette: past the dilated ring the depth channel is the 0 sentinel
 * ("this view saw nothing here"), which makes the runtime step nowhere instead
 * of dragging colour in from a texel that has no business being visible.
 */
const DILATE_GEO = 8

// Sky-visibility grid. Spans exactly one periodic tile in xz (or the support box
// for a finite specimen) and [y0, y1] in y, so marching wraps by construction.
const AO_NX = 32
const AO_NY = 48
const AO_NZ = 32
const AO_STEPS = 32
const AO_GROWTH = 1.1
/** Extinction per unit projected area density (1/m); tuned by eye on the bake. */
const AO_SIGMA_SCALE = 1.6

export interface CardAtlas {
  atlasAlb: number
  atlasGeo: number
  tileAlb: number
  tileGeo: number
  nSide: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  albedo: Uint8Array<ArrayBuffer>
  geo: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackCards(buf: ArrayBuffer): CardAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const atlasAlb = u[2]!
  const atlasGeo = u[3]!
  if (atlasAlb !== ATLAS_ALB || atlasGeo !== ATLAS_GEO) return null
  const albBytes = atlasAlb * atlasAlb * 4
  const geoBytes = atlasGeo * atlasGeo * 4
  if (buf.byteLength !== HEADER_BYTES + albBytes + geoBytes) return null
  const f = new Float32Array(buf, 32, 5)
  return {
    atlasAlb,
    atlasGeo,
    tileAlb: u[4]!,
    tileGeo: u[5]!,
    nSide: u[6]!,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    albedo: new Uint8Array(buf, HEADER_BYTES, albBytes),
    geo: new Uint8Array(buf, HEADER_BYTES + albBytes, geoBytes),
  }
}

function packCards(a: CardAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.geo.byteLength)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.atlasAlb
  u[3] = a.atlasGeo
  u[4] = a.tileAlb
  u[5] = a.tileGeo
  u[6] = a.nSide
  const f = new Float32Array(buf, 32, 5)
  f[0] = a.rXZ
  f[1] = a.y0
  f[2] = a.y1
  f[3] = a.cx
  f[4] = a.cz
  new Uint8Array(buf, HEADER_BYTES, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + a.albedo.byteLength, a.geo.byteLength).set(a.geo)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the atlases for a species. The dev
 * server answers missing /mesh/baked files with the SPA index.html at status
 * 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesCards(ctx: BakeCtx, speciesId: string): Promise<CardAtlas> {
  const key = `ndi-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesCards(ctx, mesh, speciesId)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackCards(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackCards(buf)
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

interface CaptureBox {
  cx: number
  cz: number
  rXZ: number
  y0: number
  y1: number
}

interface AoGrid {
  data: Uint8Array<ArrayBuffer>
  org: [number, number, number]
  ext: [number, number, number]
}

/** Exact horizontal support radius + capture box from the vertex cloud. */
function captureBox(mesh: GcMesh): CaptureBox {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const verts = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sz = (bz1 - bz0) / 65535
  let r2 = 0
  for (let i = 0; i < hdr.vertexCount; i++) {
    const dx = bx0 + verts[i * 8]! * sx - cx
    const dz = bz0 + verts[i * 8 + 2]! * sz - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  return { cx, cz, rXZ: Math.sqrt(r2) * 1.02 + 1e-3, y0: Math.min(0, by0), y1: by1 }
}

/**
 * Volumetric sky visibility. Triangle area is binned into an extinction grid
 * (wrapped in xz over the mesh's periodic tile, or over the support box for a
 * finite specimen), then transmittance is integrated along 13 cosine-weighted
 * upper-hemisphere directions per voxel. Deep-in-the-clump texels and texels
 * near the soil come out dark; canopy tips come out open.
 */
function buildAoGrid(mesh: GcMesh, box: CaptureBox, label: string): AoGrid {
  const hdr = mesh.header
  const [tsx, tsz] = hdr.tileSize
  const periodic = tsx > 0.01 && tsz > 0.01
  const extX = periodic ? tsx : 2 * box.rXZ
  const extZ = periodic ? tsz : 2 * box.rXZ
  const orgX = periodic ? hdr.tileOrigin[0] : box.cx - box.rXZ
  const orgZ = periodic ? hdr.tileOrigin[1] : box.cz - box.rXZ
  const orgY = box.y0
  const extY = box.y1 - box.y0

  const sigma = new Float32Array(AO_NX * AO_NY * AO_NZ)
  const verts = mesh.vertices
  const tris = mesh.triangles
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const qx = (bx1 - bx0) / 65535
  const qy = (by1 - by0) / 65535
  const qz = (bz1 - bz0) / 65535

  const wrap = (f: number): number => {
    const w = f - Math.floor(f)
    return w < 0 ? w + 1 : w
  }

  for (let t = 0; t < hdr.triangleCount; t++) {
    const i0 = tris[t * 4]! * 8
    const i1 = tris[t * 4 + 1]! * 8
    const i2 = tris[t * 4 + 2]! * 8
    const ax = bx0 + verts[i0]! * qx
    const ay = by0 + verts[i0 + 1]! * qy
    const az = bz0 + verts[i0 + 2]! * qz
    const b1x = bx0 + verts[i1]! * qx - ax
    const b1y = by0 + verts[i1 + 1]! * qy - ay
    const b1z = bz0 + verts[i1 + 2]! * qz - az
    const b2x = bx0 + verts[i2]! * qx - ax
    const b2y = by0 + verts[i2 + 1]! * qy - ay
    const b2z = bz0 + verts[i2 + 2]! * qz - az
    const nx = b1y * b2z - b1z * b2y
    const ny = b1z * b2x - b1x * b2z
    const nz = b1x * b2y - b1y * b2x
    const area = 0.5 * Math.hypot(nx, ny, nz)
    if (area <= 0) continue
    const px = ax + (b1x + b2x) / 3
    const py = ay + (b1y + b2y) / 3
    const pz = az + (b1z + b2z) / 3
    const gx = Math.min(AO_NX - 1, (wrap((px - orgX) / extX) * AO_NX) | 0)
    const gy = Math.min(AO_NY - 1, Math.max(0, (((py - orgY) / extY) * AO_NY) | 0))
    const gz = Math.min(AO_NZ - 1, (wrap((pz - orgZ) / extZ) * AO_NZ) | 0)
    const vi = (gz * AO_NY + gy) * AO_NX + gx
    sigma[vi] = sigma[vi]! + area
  }

  const voxVol = (extX / AO_NX) * (extY / AO_NY) * (extZ / AO_NZ)
  // Extinction of randomly oriented flat plates ~ 0.5 * area density.
  const k = (0.5 * AO_SIGMA_SCALE) / voxVol
  for (let i = 0; i < sigma.length; i++) sigma[i]! *= k

  const dirs: [number, number, number, number][] = [[0, 1, 0, 1]]
  for (const [theta, phi0] of [
    [(35 * Math.PI) / 180, 0],
    [(65 * Math.PI) / 180, Math.PI / 6],
  ] as [number, number][]) {
    const dy = Math.cos(theta)
    const rho = Math.sin(theta)
    for (let i = 0; i < 6; i++) {
      const phi = phi0 + (i * Math.PI) / 3
      dirs.push([rho * Math.cos(phi), dy, rho * Math.sin(phi), dy])
    }
  }
  const wSum = dirs.reduce((s, d) => s + d[3], 0)

  const stepBase = 0.75 * Math.min(extX / AO_NX, extY / AO_NY, extZ / AO_NZ)
  const sigmaAt = (x: number, y: number, z: number): number => {
    const gy = ((y - orgY) / extY) * AO_NY
    if (gy < 0 || gy >= AO_NY) return 0
    const gx = Math.min(AO_NX - 1, (wrap((x - orgX) / extX) * AO_NX) | 0)
    const gz = Math.min(AO_NZ - 1, (wrap((z - orgZ) / extZ) * AO_NZ) | 0)
    return sigma[((gz * AO_NY + (gy | 0)) * AO_NX + gx)]!
  }

  const data = new Uint8Array(AO_NX * AO_NY * AO_NZ)
  let aoMin = 1
  let aoSum = 0
  for (let iz = 0; iz < AO_NZ; iz++) {
    const wz = orgZ + ((iz + 0.5) / AO_NZ) * extZ
    for (let iy = 0; iy < AO_NY; iy++) {
      const wy = orgY + ((iy + 0.5) / AO_NY) * extY
      for (let ix = 0; ix < AO_NX; ix++) {
        const wx = orgX + ((ix + 0.5) / AO_NX) * extX
        let vis = 0
        for (const [dx, dy, dz, w] of dirs) {
          let x = wx
          let y = wy
          let z = wz
          let tau = 0
          let step = stepBase
          for (let s = 0; s < AO_STEPS; s++) {
            x += dx * step
            y += dy * step
            z += dz * step
            if (y > box.y1) break
            tau += sigmaAt(x, y, z) * step
            if (tau > 6) break
            step *= AO_GROWTH
          }
          vis += w * Math.exp(-tau)
        }
        const ao = vis / wSum
        aoMin = Math.min(aoMin, ao)
        aoSum += ao
        data[(iz * AO_NY + iy) * AO_NX + ix] = Math.round(Math.min(1, ao) * 255)
      }
    }
  }
  console.info(
    `[030] ${label} sky-visibility grid ${AO_NX}x${AO_NY}x${AO_NZ}` +
      ` periodic=${periodic} min=${aoMin.toFixed(3)} mean=${(aoSum / data.length).toFixed(3)}`,
  )
  return { data, org: [orgX, orgY, orgZ], ext: [extX, extY, extZ] }
}

async function bakeSpeciesCards(ctx: BakeCtx, mesh: GcMesh, speciesId: string): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const box = captureBox(mesh)
  const { cx, cz, rXZ, y0, y1 } = box
  const yMid = (y0 + y1) / 2
  const hHalf = (y1 - y0) / 2
  const ao = buildAoGrid(mesh, box, speciesId)

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

  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [BIG, BIG],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('bake-albedo')
  const geoTex = mkTarget('bake-geo')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )
  const aoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-ao-grid`,
      size: [AO_NX, AO_NY, AO_NZ],
      dimension: '3d',
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeTexture({ texture: aoTex }, ao.data, { bytesPerRow: AO_NX, rowsPerImage: AO_NY }, [
    AO_NX,
    AO_NY,
    AO_NZ,
  ])
  const aoSampler = device.createSampler({
    label: `${ctx.id}/bake-ao-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'repeat',
  })

  // Per-view uniforms in one buffer with dynamic offsets (32 floats used).
  const STRIDE = 256
  const N_VIEWS = N_SIDE + 1
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_VIEWS)
  for (let k = 0; k < N_VIEWS; k++) {
    const o = (k * STRIDE) / 4
    const isTop = k === N_SIDE
    if (isTop) {
      scratch.set([1 / rXZ, 0, 0], o) // u = +X
      scratch.set([0, 0, 1 / rXZ], o + 4) // v = +Z
      scratch.set([0, 1 / hHalf, 0, 1], o + 8) // depth axis = +Y, w = is_top
    } else {
      const a = (k * 2 * Math.PI) / N_SIDE
      scratch.set([Math.cos(a) / rXZ, 0, -Math.sin(a) / rXZ], o)
      scratch.set([0, -1 / hHalf, 0], o + 4)
      scratch.set([Math.sin(a) / rXZ, 0, Math.cos(a) / rXZ, 0], o + 8)
    }
    scratch.set([cx, yMid, cz, isTop ? 1 : 0], o + 12)
    scratch.set([ao.org[0], ao.org[1], ao.org[2], 0], o + 16)
    scratch.set([1 / ao.ext[0], 1 / ao.ext[1], 1 / ao.ext[2], 0], o + 20)
    scratch.set([hdr.boundsMin[0], hdr.boundsMin[1], hdr.boundsMin[2], 0], o + 24)
    scratch.set(
      [
        hdr.boundsMax[0] - hdr.boundsMin[0],
        hdr.boundsMax[1] - hdr.boundsMin[1],
        hdr.boundsMax[2] - hdr.boundsMin[2],
        0,
      ],
      o + 28,
    )
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
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uni, size: 128 } },
      { binding: 1, resource: aoTex.createView() },
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
  const pass = enc.beginRenderPass({
    label: `${ctx.id}/bake-pass`,
    colorAttachments: [
      { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      { view: geoTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1 }, loadOp: 'clear', storeOp: 'store' },
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
  const bigTile = TILE_ALB * SS
  for (let k = 0; k < N_VIEWS; k++) {
    const col = k % GRID
    const row = Math.floor(k / GRID)
    pass.setViewport(col * bigTile, row * bigTile, bigTile, bigTile, 0, 1)
    pass.setScissorRect(col * bigTile, row * bigTile, bigTile, bigTile)
    pass.setBindGroup(0, bg, [k * STRIDE])
    pass.drawIndexed(indices.length)
  }
  pass.end()

  const bpr = BIG * 4
  const rbSize = bpr * BIG
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbGeo = mkReadback('bake-rb-geo')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  enc.copyTextureToBuffer({ texture: geoTex }, { buffer: rbGeo, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbGeo.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigGeo = new Uint8Array(rbGeo.getMappedRange()).slice()
  rbAlb.unmap()
  rbGeo.unmap()
  for (const r of [vbuf, ibuf, albTex, geoTex, depthTex, aoTex, uni, rbAlb, rbGeo]) r.destroy()

  const albedo = resolveAlbedo(bigAlb)
  const geo = resolveGeo(bigAlb, bigGeo)
  return packCards({
    atlasAlb: ATLAS_ALB,
    atlasGeo: ATLAS_GEO,
    tileAlb: TILE_ALB,
    tileGeo: TILE_GEO,
    nSide: N_SIDE,
    rXZ,
    y0,
    y1,
    cx,
    cz,
    albedo,
    geo,
  })
}

/** Octahedral decode/encode pair, y-primary — matches the WGSL convention. */
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

function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  if (s < 1e-6) return [128, 128]
  let u = x / s
  let v = z / s
  if (y < 0) {
    const fu = (1 - Math.abs(v)) * (x >= 0 ? 1 : -1)
    const fv = (1 - Math.abs(u)) * (z >= 0 ? 1 : -1)
    u = fu
    v = fv
  }
  return [Math.round((u * 0.5 + 0.5) * 255), Math.round((v * 0.5 + 0.5) * 255)]
}

/**
 * Generic tile-clamped dilation of float planes over a filled mask. Returns the
 * grown mask (which texels now hold a plausible value).
 */
function dilate(n: number, tile: number, passes: number, filled: Uint8Array, planes: Float32Array[]): Uint8Array {
  let cur = filled
  const acc = new Float32Array(planes.length)
  for (let p = 0; p < passes; p++) {
    const next = cur.slice()
    for (let y = 0; y < n; y++) {
      const ty0 = Math.floor(y / tile) * tile
      const ty1 = ty0 + tile - 1
      for (let x = 0; x < n; x++) {
        const idx = y * n + x
        if (cur[idx]! !== 0) continue
        const tx0 = Math.floor(x / tile) * tile
        const tx1 = tx0 + tile - 1
        let count = 0
        acc.fill(0)
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const nIdx = yy * n + xx
            if (cur[nIdx]! === 0) continue
            count++
            for (let c = 0; c < planes.length; c++) acc[c]! += planes[c]![nIdx]!
          }
        }
        if (count === 0) continue
        for (let c = 0; c < planes.length; c++) planes[c]![idx] = acc[c]! / count
        next[idx] = 1
      }
    }
    cur = next
  }
  return cur
}

/** Coverage-weighted 2x2 downsample of the albedo target, then colour dilation. */
function resolveAlbedo(bigAlb: Uint8Array): Uint8Array<ArrayBuffer> {
  const n = ATLAS_ALB
  const out = new Uint8Array(n * n * 4)
  const r = new Float32Array(n * n)
  const g = new Float32Array(n * n)
  const b = new Float32Array(n * n)
  const filled = new Uint8Array(n * n)
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let aSum = 0
      let rs = 0
      let gs = 0
      let bs = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * BIG + (x * SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          rs += bigAlb[s]! * a
          gs += bigAlb[s + 1]! * a
          bs += bigAlb[s + 2]! * a
        }
      }
      const idx = y * n + x
      if (aSum > 0) {
        r[idx] = rs / aSum
        g[idx] = gs / aSum
        b[idx] = bs / aSum
        out[idx * 4 + 3] = Math.round(aSum / (SS * SS))
        filled[idx] = 1
      }
    }
  }
  dilate(n, TILE_ALB, DILATE_ALB, filled, [r, g, b])
  for (let i = 0; i < n * n; i++) {
    out[i * 4] = Math.round(r[i]!)
    out[i * 4 + 1] = Math.round(g[i]!)
    out[i * 4 + 2] = Math.round(b[i]!)
  }
  return out
}

/**
 * Coverage-weighted 4x4 downsample of the geometry target (normals averaged as
 * vectors), then dilation. Never-covered texels end up with depth 0.5 (= the
 * plant's mid-plane, so a reprojection starting there simply does not move) and
 * full sky visibility.
 */
function resolveGeo(bigAlb: Uint8Array, bigGeo: Uint8Array): Uint8Array<ArrayBuffer> {
  const n = ATLAS_GEO
  const nx = new Float32Array(n * n)
  const ny = new Float32Array(n * n)
  const nz = new Float32Array(n * n)
  const depth = new Float32Array(n * n)
  const sky = new Float32Array(n * n)
  const filled = new Uint8Array(n * n)
  const tmp: [number, number, number] = [0, 0, 0]
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let aSum = 0
      let vx = 0
      let vy = 0
      let vz = 0
      let ds = 0
      let ss = 0
      for (let j = 0; j < GEO_DOWN; j++) {
        for (let i = 0; i < GEO_DOWN; i++) {
          const p = (y * GEO_DOWN + j) * BIG + (x * GEO_DOWN + i)
          const a = bigAlb[p * 4 + 3]!
          if (a === 0) continue
          aSum += a
          octDecode(bigGeo[p * 4]! / 127.5 - 1, bigGeo[p * 4 + 1]! / 127.5 - 1, tmp)
          vx += tmp[0] * a
          vy += tmp[1] * a
          vz += tmp[2] * a
          ds += bigGeo[p * 4 + 2]! * a
          ss += bigGeo[p * 4 + 3]! * a
        }
      }
      const idx = y * n + x
      if (aSum > 0) {
        nx[idx] = vx / aSum
        ny[idx] = vy / aSum
        nz[idx] = vz / aSum
        depth[idx] = ds / aSum / 255
        sky[idx] = ss / aSum / 255
        filled[idx] = 1
      } else {
        nx[idx] = 0
        ny[idx] = 1
        nz[idx] = 0
        depth[idx] = 0 // sentinel: no surface along this view ray
        sky[idx] = 1
      }
    }
  }
  // Dilation carries a plausible depth a few texels past every silhouette, so a
  // reprojection that lands just outside it still reads a real surface.
  const covered = dilate(n, TILE_GEO, DILATE_GEO, filled, [nx, ny, nz, depth, sky])
  const out = new Uint8Array(n * n * 4)
  for (let i = 0; i < n * n; i++) {
    const [eu, ev] = octEncode(nx[i]!, ny[i]!, nz[i]!)
    out[i * 4] = eu
    out[i * 4 + 1] = ev
    // 0 is the "no surface" sentinel; real depths start one code point in.
    out[i * 4 + 2] = covered[i] === 0 ? 0 : Math.max(1, Math.round(Math.min(1, Math.max(0, depth[i]!)) * 255))
    out[i * 4 + 3] = Math.round(Math.min(1, Math.max(0, sky[i]!)) * 255)
  }
  return out
}
