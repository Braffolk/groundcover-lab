import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Relief-lit card atlas bake.
 *
 * Per species ONE 3x3 atlas of TILE px tiles is rendered orthographically from
 * the raw GCMESH1 mesh (tiles 0..7 = side views at 45deg azimuth steps, tile 8
 * = the straight-down canopy view), in two co-registered layers:
 *
 *   A: rgb = authored albedo, a = coverage
 *   G: rg  = octahedral mesh-frame normal
 *      b   = HEIGHTFIELD s (front surface depth along the capture axis,
 *            0 = "no data" sentinel so the runtime warp can switch itself off)
 *      a   = canopy occlusion (how deeply the texel sits under the plant's own
 *            canopy silhouette, derived from tile 8's heightfield)
 *
 * Everything after the GPU capture happens on the CPU, once: 2x coverage-
 * weighted downsample, canopy-occlusion solve, dilation into empty texels, and
 * a PER-TILE mip chain (so no mip level ever blends one view into another —
 * the usual atlas-bleed that makes distant impostors shift colour). The chain
 * ships inside the artifact, so load = fetch + writeTexture, no GPU mipgen.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'NDR1', u32 version, u32 atlasPx, u32 tilePx, u32 mips, u32 nSide
 *   f32 rXZ, y0, y1, cx, cz              (capture box, unit scale, metres)
 *   ...zeros to byte 64
 *   u8[chainBytes] A mip chain (rgba8, level 0 first)
 *   u8[chainBytes] G mip chain (rgba8, level 0 first)
 */

export const TILE = 448
export const GRID = 3
export const ATLAS = TILE * GRID
/** Levels down to a 7px tile — deeper mips would cross tile borders. */
export const MIPS = 7
export const N_SIDE = 8
const N_VIEWS = N_SIDE + 1
const SS = 2
const BIG = ATLAS * SS
const MAGIC = 0x3152444e // 'NDR1'
const VERSION = 2
const HEADER_BYTES = 64
const DILATE_PASSES = 6
/** Canopy occlusion: ao = AO_MIN + (1-AO_MIN) * exp(-AO_K * depth_under_canopy). */
const AO_MIN = 0.3
const AO_K = 1.3
/**
 * Radius (texels) of the coverage-weighted blur applied to the heightfield.
 * Grass depth is NOT a heightfield: neighbouring texels can be a blade at the
 * front of the clump and bare ground 30 cm behind it, and a one-step relief
 * solve against that discontinuous field warps colour to places the surface
 * never was (measured: visibly muddier than a plain billboard). Blurring to the
 * canopy ENVELOPE — coarser than a blade, finer than the clump — keeps the
 * parallax coherent, which is what the eye actually reads as depth.
 */
const HEIGHT_BLUR = 10
/** Radius (m) of the max filter that turns the canopy heightfield into a lid. */
const CANOPY_SMOOTH_M = 0.07

export interface ReliefAtlas {
  atlasPx: number
  tilePx: number
  mips: number
  nSide: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  /** rgba8 mip chains, level 0 first. */
  albedo: Uint8Array<ArrayBuffer>
  geom: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function levelPx(level: number): number {
  return ATLAS >> level
}

export function chainBytes(): number {
  let total = 0
  for (let l = 0; l < MIPS; l++) total += levelPx(l) * levelPx(l) * 4
  return total
}

export function unpackRelief(buf: ArrayBuffer): ReliefAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 6)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== ATLAS || u[3] !== TILE || u[4] !== MIPS) return null
  const bytes = chainBytes()
  if (buf.byteLength !== HEADER_BYTES + bytes * 2) return null
  const f = new Float32Array(buf, 24, 5)
  return {
    atlasPx: u[2]!,
    tilePx: u[3]!,
    mips: u[4]!,
    nSide: u[5]!,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    albedo: new Uint8Array(buf, HEADER_BYTES, bytes),
    geom: new Uint8Array(buf, HEADER_BYTES + bytes, bytes),
  }
}

function packRelief(a: ReliefAtlas): ArrayBuffer {
  const bytes = chainBytes()
  const buf = new ArrayBuffer(HEADER_BYTES + bytes * 2)
  const u = new Uint32Array(buf, 0, 6)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.atlasPx
  u[3] = a.tilePx
  u[4] = a.mips
  u[5] = a.nSide
  const f = new Float32Array(buf, 24, 5)
  f[0] = a.rXZ
  f[1] = a.y0
  f[2] = a.y1
  f[3] = a.cx
  f[4] = a.cz
  new Uint8Array(buf, HEADER_BYTES, bytes).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + bytes, bytes).set(a.geom)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the relief atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, so every result is magic+size validated; a poisoned cache entry
 * is rebaked and repaired instead of rendering garbage.
 */
export async function loadSpeciesRelief(ctx: BakeCtx, speciesId: string): Promise<ReliefAtlas> {
  const key = `relief-v${VERSION}-t${TILE}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesRelief(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackRelief(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackRelief(buf)
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

// ---------------------------------------------------------------------------
// GPU capture
// ---------------------------------------------------------------------------

async function bakeSpeciesRelief(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2

  // Exact horizontal support radius from the vertices (bounds corners
  // overestimate it noticeably on wide community tiles).
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
  const rXZ = Math.sqrt(r2) * 1.02 + 1e-3
  const y0 = Math.min(0, by0)
  const y1 = by1
  const yMid = (y0 + y1) / 2
  const H = y1 - y0

  // --- transient GPU resources ---------------------------------------------
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
  const geoTex = mkTarget('bake-geom')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // Per-view uniforms in one buffer with dynamic offsets (24 floats used).
  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_VIEWS)
  for (let k = 0; k < N_VIEWS; k++) {
    const o = (k * STRIDE) / 4
    if (k === N_SIDE) {
      scratch.set([1, 0, 0, rXZ], o) // R = +X
      scratch.set([0, 0, 1, rXZ], o + 4) // V = +Z (texture v grows toward +Z)
      scratch.set([0, 1, 0, H / 2], o + 8) // F = +Y
    } else {
      const a = (k * 2 * Math.PI) / N_SIDE
      scratch.set([Math.cos(a), 0, -Math.sin(a), rXZ], o)
      scratch.set([0, -1, 0, H / 2], o + 4)
      scratch.set([Math.sin(a), 0, Math.cos(a), rXZ], o + 8)
    }
    scratch.set([cx, yMid, cz, 0], o + 12)
    scratch.set([bx0, by0, bz0, 0], o + 16)
    scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 20)
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
      { view: geoTex.createView(), clearValue: { r: 0.5, g: 1, b: 0.5, a: 0 }, loadOp: 'clear', storeOp: 'store' },
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
  const bigTile = TILE * SS
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
  const rbGeo = mkReadback('bake-rb-geom')
  enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  enc.copyTextureToBuffer({ texture: geoTex }, { buffer: rbGeo, bytesPerRow: bpr, rowsPerImage: BIG }, [BIG, BIG])
  device.queue.submit([enc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbGeo.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigGeo = new Uint8Array(rbGeo.getMappedRange()).slice()
  rbAlb.unmap()
  rbGeo.unmap()
  for (const r of [vbuf, ibuf, albTex, geoTex, depthTex, uni, rbAlb, rbGeo]) r.destroy()

  const geo = { rXZ, y0, y1, cx, cz, yMid, H }
  const level0 = reduceCapture(bigAlb, bigGeo)
  solveCanopyOcclusion(level0, geo)
  blurHeight(level0)
  dilate(level0)
  const { albedo, geom } = buildChain(level0)
  return packRelief({ atlasPx: ATLAS, tilePx: TILE, mips: MIPS, nSide: N_SIDE, rXZ, y0, y1, cx, cz, albedo, geom })
}

// ---------------------------------------------------------------------------
// CPU post-process
// ---------------------------------------------------------------------------

interface Level {
  px: number
  tile: number
  /** 0..255 albedo. */
  rgb: Float32Array
  /** unnormalized mesh-frame normal. */
  nrm: Float32Array
  /** heightfield in [0,1]. */
  s: Float32Array
  /** canopy occlusion in [0,1]. */
  ao: Float32Array
  /** coverage in [0,1]. */
  cov: Float32Array
  /** 1 = has usable rgb/nrm/s (covered OR dilated). */
  filled: Uint8Array
}

function newLevel(px: number, tile: number): Level {
  const n = px * px
  return {
    px,
    tile,
    rgb: new Float32Array(n * 3),
    nrm: new Float32Array(n * 3),
    s: new Float32Array(n),
    ao: new Float32Array(n).fill(1),
    cov: new Float32Array(n),
    filled: new Uint8Array(n),
  }
}

/** Coverage-weighted SSxSS reduction of the capture into level 0. */
function reduceCapture(bigAlb: Uint8Array, bigGeo: Uint8Array): Level {
  const L = newLevel(ATLAS, TILE)
  for (let y = 0; y < ATLAS; y++) {
    for (let x = 0; x < ATLAS; x++) {
      let w = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let hs = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const o = ((y * SS + j) * BIG + (x * SS + i)) * 4
          const a = bigAlb[o + 3]!
          if (a === 0) continue
          w += a
          r += bigAlb[o]! * a
          g += bigAlb[o + 1]! * a
          b += bigAlb[o + 2]! * a
          nx += (bigGeo[o]! / 127.5 - 1) * a
          ny += (bigGeo[o + 1]! / 127.5 - 1) * a
          nz += (bigGeo[o + 2]! / 127.5 - 1) * a
          hs += (bigGeo[o + 3]! / 255) * a
        }
      }
      const d = y * ATLAS + x
      if (w === 0) {
        L.nrm[d * 3 + 1] = 1
        continue
      }
      L.rgb[d * 3] = r / w
      L.rgb[d * 3 + 1] = g / w
      L.rgb[d * 3 + 2] = b / w
      L.nrm[d * 3] = nx / w
      L.nrm[d * 3 + 1] = ny / w
      L.nrm[d * 3 + 2] = nz / w
      L.s[d] = hs / w
      L.cov[d] = w / (255 * SS * SS)
      L.filled[d] = 1
    }
  }
  return L
}

interface Geo {
  rXZ: number
  y0: number
  y1: number
  cx: number
  cz: number
  yMid: number
  H: number
}

/**
 * Canopy occlusion. Tile 8 (straight down) IS a canopy-top heightfield: for
 * any (x,z) it gives the highest surface of the plant. Max-filter it into a
 * smooth "lid", then every texel of every tile — whose full 3D position is
 * known from its own (u, v, heightfield) triple — is occluded by how deep it
 * sits under that lid. That is a real sky-visibility estimate for grass, from
 * data the bake already produced, in one pass and no ray casting.
 */
function solveCanopyOcclusion(L: Level, geo: Geo): void {
  const T = L.tile
  const topCol = (N_SIDE % GRID) * T
  const topRow = Math.floor(N_SIDE / GRID) * T

  // Canopy lid in metres, tile-local (T x T).
  const lid = new Float32Array(T * T)
  for (let j = 0; j < T; j++) {
    for (let i = 0; i < T; i++) {
      const src = (topRow + j) * L.px + (topCol + i)
      lid[j * T + i] = L.filled[src] === 1 ? geo.y0 + L.s[src]! * geo.H : geo.y0
    }
  }
  const rad = Math.max(2, Math.min(64, Math.round((CANOPY_SMOOTH_M / (2 * geo.rXZ)) * T)))
  maxFilterSeparable(lid, T, rad)

  const lidAt = (x: number, z: number): number => {
    const u = 0.5 + (0.5 * (x - geo.cx)) / geo.rXZ
    const v = 0.5 + (0.5 * (z - geo.cz)) / geo.rXZ
    const i = Math.min(T - 1, Math.max(0, Math.round(u * T - 0.5)))
    const j = Math.min(T - 1, Math.max(0, Math.round(v * T - 0.5)))
    return lid[j * T + i]!
  }

  for (let k = 0; k < N_VIEWS; k++) {
    const col = (k % GRID) * T
    const row = Math.floor(k / GRID) * T
    const isTop = k === N_SIDE
    const a = (k * 2 * Math.PI) / N_SIDE
    const ck = Math.cos(a)
    const sk = Math.sin(a)
    for (let j = 0; j < T; j++) {
      for (let i = 0; i < T; i++) {
        const idx = (row + j) * L.px + (col + i)
        if (L.filled[idx] !== 1) continue
        const uu = (i + 0.5) / T
        const vv = (j + 0.5) / T
        const s = L.s[idx]!
        let px: number
        let py: number
        let pz: number
        if (isTop) {
          // R=+X, V=+Z, F=+Y with half-extents (rXZ, rXZ, H/2).
          px = geo.cx + (2 * uu - 1) * geo.rXZ
          pz = geo.cz + (2 * vv - 1) * geo.rXZ
          py = geo.yMid + (2 * s - 1) * (geo.H / 2)
        } else {
          // R=(cos,0,-sin), V=(0,-1,0), F=(sin,0,cos); (rXZ, H/2, rXZ).
          const xl = (2 * uu - 1) * geo.rXZ
          const w = (2 * s - 1) * geo.rXZ
          px = geo.cx + xl * ck + w * sk
          pz = geo.cz - xl * sk + w * ck
          py = geo.yMid - (2 * vv - 1) * (geo.H / 2)
        }
        const depth = Math.max(0, lidAt(px, pz) - py)
        L.ao[idx] = AO_MIN + (1 - AO_MIN) * Math.exp(-AO_K * depth)
      }
    }
  }
}

/**
 * Coverage-weighted separable box blur of the heightfield, per tile. Empty
 * texels contribute nothing and keep their sentinel, so the blur never drags a
 * "no data" zero into the envelope.
 */
function blurHeight(L: Level): void {
  const N = L.px
  const T = L.tile
  const rad = HEIGHT_BLUR
  const tmpS = new Float32Array(N * N)
  const tmpW = new Float32Array(N * N)
  // horizontal
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const tx0 = Math.floor(x / T) * T
      const tx1 = tx0 + T - 1
      let sum = 0
      let w = 0
      for (let t = Math.max(tx0, x - rad); t <= Math.min(tx1, x + rad); t++) {
        const i = y * N + t
        const c = L.cov[i]!
        if (c <= 0) continue
        sum += L.s[i]! * c
        w += c
      }
      tmpS[y * N + x] = sum
      tmpW[y * N + x] = w
    }
  }
  // vertical
  const out = new Float32Array(N * N)
  for (let y = 0; y < N; y++) {
    const ty0 = Math.floor(y / T) * T
    const ty1 = ty0 + T - 1
    for (let x = 0; x < N; x++) {
      let sum = 0
      let w = 0
      for (let t = Math.max(ty0, y - rad); t <= Math.min(ty1, y + rad); t++) {
        const i = t * N + x
        sum += tmpS[i]!
        w += tmpW[i]!
      }
      out[y * N + x] = w > 0 ? sum / w : 0
    }
  }
  for (let i = 0; i < N * N; i++) {
    if (L.cov[i]! > 0) L.s[i] = out[i]!
  }
}

/** In-place separable max filter with a square kernel of radius `rad`. */
function maxFilterSeparable(field: Float32Array, n: number, rad: number): void {
  const tmp = new Float32Array(n * n)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let m = -Infinity
      const lo = Math.max(0, i - rad)
      const hi = Math.min(n - 1, i + rad)
      for (let t = lo; t <= hi; t++) m = Math.max(m, field[j * n + t]!)
      tmp[j * n + i] = m
    }
  }
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let m = -Infinity
      const lo = Math.max(0, j - rad)
      const hi = Math.min(n - 1, j + rad)
      for (let t = lo; t <= hi; t++) m = Math.max(m, tmp[t * n + i]!)
      field[j * n + i] = m
    }
  }
}

/**
 * Grow albedo / normal / height / occlusion a few texels into empty space,
 * clamped to tile borders, leaving coverage at 0. Bilinear taps and warped
 * taps that land just outside a silhouette then read plausible values instead
 * of background black — and, crucially, a heightfield that still points at the
 * nearby surface instead of the sentinel.
 */
function dilate(L: Level): void {
  const N = L.px
  const T = L.tile
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = L.filled.slice()
    for (let y = 0; y < N; y++) {
      const ty0 = Math.floor(y / T) * T
      const ty1 = ty0 + T - 1
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (L.filled[idx] !== 0) continue
        const tx0 = Math.floor(x / T) * T
        const tx1 = tx0 + T - 1
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        let hs = 0
        let ao = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < ty0 || yy > ty1) continue
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue
            const xx = x + i
            if (xx < tx0 || xx > tx1) continue
            const s = yy * N + xx
            if (L.filled[s] === 0) continue
            count++
            r += L.rgb[s * 3]!
            g += L.rgb[s * 3 + 1]!
            b += L.rgb[s * 3 + 2]!
            nx += L.nrm[s * 3]!
            ny += L.nrm[s * 3 + 1]!
            nz += L.nrm[s * 3 + 2]!
            hs += L.s[s]!
            ao += L.ao[s]!
          }
        }
        if (count === 0) continue
        L.rgb[idx * 3] = r / count
        L.rgb[idx * 3 + 1] = g / count
        L.rgb[idx * 3 + 2] = b / count
        L.nrm[idx * 3] = nx / count
        L.nrm[idx * 3 + 1] = ny / count
        L.nrm[idx * 3 + 2] = nz / count
        L.s[idx] = hs / count
        L.ao[idx] = ao / count
        next[idx] = 1
      }
    }
    L.filled = next
  }
}

/** Quantize one level into its two rgba8 planes. */
function quantize(L: Level, aOut: Uint8Array, gOut: Uint8Array, byteOffset: number): void {
  const n = L.px * L.px
  for (let i = 0; i < n; i++) {
    const o = byteOffset + i * 4
    const filled = L.filled[i] === 1
    aOut[o] = clamp255(L.rgb[i * 3]!)
    aOut[o + 1] = clamp255(L.rgb[i * 3 + 1]!)
    aOut[o + 2] = clamp255(L.rgb[i * 3 + 2]!)
    aOut[o + 3] = clamp255(L.cov[i]! * 255)
    const [ou, ov] = octEncode(L.nrm[i * 3]!, L.nrm[i * 3 + 1]!, L.nrm[i * 3 + 2]!)
    gOut[o] = ou
    gOut[o + 1] = ov
    // b = heightfield; 0 is the "no data" sentinel that switches the runtime
    // warp off, so a real surface never quantizes to 0.
    gOut[o + 2] = filled ? Math.max(1, clamp255(L.s[i]! * 255)) : 0
    gOut[o + 3] = filled ? clamp255(L.ao[i]! * 255) : 255
  }
}

/** Per-tile coverage-weighted 2x reduction — never blends across tiles. */
function reduceLevel(L: Level): Level {
  const out = newLevel(L.px >> 1, L.tile >> 1)
  const T = out.tile
  for (let k = 0; k < N_VIEWS; k++) {
    const dCol = (k % GRID) * T
    const dRow = Math.floor(k / GRID) * T
    const sCol = dCol * 2
    const sRow = dRow * 2
    for (let j = 0; j < T; j++) {
      for (let i = 0; i < T; i++) {
        // Two accumulators: coverage-weighted (what a mip should be) and a
        // plain average over merely-dilated texels (so coarse levels still hold
        // plausible colour/height where nothing was ever covered).
        let w = 0
        let raw = 0
        let covSum = 0
        let wr = 0
        let wg = 0
        let wb = 0
        let wnx = 0
        let wny = 0
        let wnz = 0
        let ws = 0
        let wao = 0
        let rr = 0
        let rg = 0
        let rb = 0
        let rnx = 0
        let rny = 0
        let rnz = 0
        let rs = 0
        let rao = 0
        for (let dj = 0; dj < 2; dj++) {
          for (let di = 0; di < 2; di++) {
            const s = (sRow + j * 2 + dj) * L.px + (sCol + i * 2 + di)
            const rgb0 = L.rgb[s * 3]!
            const rgb1 = L.rgb[s * 3 + 1]!
            const rgb2 = L.rgb[s * 3 + 2]!
            const n0 = L.nrm[s * 3]!
            const n1 = L.nrm[s * 3 + 1]!
            const n2 = L.nrm[s * 3 + 2]!
            const sv = L.s[s]!
            const av = L.ao[s]!
            if (L.filled[s] === 1) {
              raw++
              rr += rgb0
              rg += rgb1
              rb += rgb2
              rnx += n0
              rny += n1
              rnz += n2
              rs += sv
              rao += av
            }
            const c = L.cov[s]!
            covSum += c
            if (c <= 0) continue
            w += c
            wr += rgb0 * c
            wg += rgb1 * c
            wb += rgb2 * c
            wnx += n0 * c
            wny += n1 * c
            wnz += n2 * c
            ws += sv * c
            wao += av * c
          }
        }
        const d = (dRow + j) * out.px + (dCol + i)
        out.cov[d] = covSum / 4
        out.filled[d] = raw > 0 ? 1 : 0
        const useW = w > 0
        const norm = useW ? w : raw
        if (norm === 0) {
          out.nrm[d * 3 + 1] = 1
          out.ao[d] = 1
          continue
        }
        out.rgb[d * 3] = (useW ? wr : rr) / norm
        out.rgb[d * 3 + 1] = (useW ? wg : rg) / norm
        out.rgb[d * 3 + 2] = (useW ? wb : rb) / norm
        out.nrm[d * 3] = (useW ? wnx : rnx) / norm
        out.nrm[d * 3 + 1] = (useW ? wny : rny) / norm
        out.nrm[d * 3 + 2] = (useW ? wnz : rnz) / norm
        out.s[d] = (useW ? ws : rs) / norm
        out.ao[d] = (useW ? wao : rao) / norm
      }
    }
  }
  return out
}

function buildChain(level0: Level): { albedo: Uint8Array<ArrayBuffer>; geom: Uint8Array<ArrayBuffer> } {
  const bytes = chainBytes()
  const albedo = new Uint8Array(bytes)
  const geom = new Uint8Array(bytes)
  let off = 0
  let level = level0
  for (let l = 0; l < MIPS; l++) {
    quantize(level, albedo, geom, off)
    off += level.px * level.px * 4
    if (l + 1 < MIPS) level = reduceLevel(level)
  }
  return { albedo, geom }
}

function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)))
}

/**
 * Octahedral encode, y-primary — this atlas's OWN convention, the exact inverse of
 * oct_decode() in relight.wgsl. Unrelated to the GCMESH1 source convention
 * (src/wgsl/gcmesh.wgsl): what it encodes has already been decoded from the source
 * and flipped toward the capture axis by the GPU bake.
 */
function octEncode(x: number, y: number, z: number): [number, number] {
  const len = Math.hypot(x, y, z)
  if (len < 1e-9) return [128, 128] // degenerate -> +Y
  const nx = x / len
  const ny = y / len
  const nz = z / len
  const s = Math.abs(nx) + Math.abs(ny) + Math.abs(nz)
  let u = nx / s
  let v = nz / s
  if (ny < 0) {
    const fu = (1 - Math.abs(v)) * (nx >= 0 ? 1 : -1)
    const fv = (1 - Math.abs(u)) * (nz >= 0 ? 1 : -1)
    u = fu
    v = fv
  }
  return [clamp255((u * 0.5 + 0.5) * 255), clamp255((v * 0.5 + 0.5) * 255)]
}
