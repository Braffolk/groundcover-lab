import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * The view set: one plant -> GRID_N x GRID_N orthographic captures covering
 * the whole upper hemisphere of viewing directions, stored as layers of a
 * mipped texture ARRAY (not one big atlas — layers cannot bleed into each
 * other when minified, which is what forces atlas impostors to either waste
 * texels on gutters or skip mips entirely).
 *
 * Node (i, j) sits at hemi-octahedral uv (i/(N-1), j/(N-1)); the square's
 * boundary is the horizon and its centre is straight down, so the corners are
 * the +-X / +-Z horizon views and there is an exact top view (N is odd).
 * Layer index is j * GRID_N + i.
 *
 * Every view gets its OWN orthographic extents — the plant's local AABB
 * projected onto that view's basis — so a tall thin side view and a squat top
 * view both fill their tile completely. The runtime reprojects a fragment into
 * any view using those extents, which is what makes the 3-view blend sharp
 * instead of a cross-fade of misaligned images.
 *
 * Artifact (little-endian):
 *   u32 magic 'OCTI', u32 version, u32 gridN, u32 tilePx
 *   f32 centre.xyz, pad          (mesh-frame capture centre)
 *   f32 half.xyz, pad            (mesh-frame AABB half extents)
 *   ... zeros to byte 64
 *   f32[gridN^2 * 2]             per-view (extU, extV), metres at unit scale
 *   u8 [gridN^2 * tile^2 * 4]    albedo rgba8, a = coverage   (layer-major)
 *   u8 [gridN^2 * tile^2 * 2]    normals, oct-encoded rg8 (plant local frame)
 */

export const GRID_N = 11
export const N_VIEWS = GRID_N * GRID_N
export const TILE = 256
/** Tile supersampling factor for the capture (resolved on the CPU). */
const SS = 2
/** Views rendered per submit — keeps transient bake VRAM at a few MB. */
const CHUNK_SIDE = 3
const CHUNK_VIEWS = CHUNK_SIDE * CHUNK_SIDE
const BIG = TILE * SS
const CHUNK_PX = BIG * CHUNK_SIDE
/** Empty margin inside every tile so clamp-to-edge never smears a silhouette. */
const TILE_PAD = 1.04
const DILATE_PASSES = 4

const MAGIC = 0x4954434f // 'OCTI'
const VERSION = 2
const HEADER_BYTES = 64
const EXT_BYTES = N_VIEWS * 2 * 4
const ALBEDO_BYTES = N_VIEWS * TILE * TILE * 4
const NORMAL_BYTES = N_VIEWS * TILE * TILE * 2
const ARTIFACT_BYTES = HEADER_BYTES + EXT_BYTES + ALBEDO_BYTES + NORMAL_BYTES

export interface ViewSet {
  gridN: number
  tilePx: number
  /** Mesh-frame centre the captures were taken around. */
  centre: [number, number, number]
  /** Mesh-frame AABB half extents. */
  half: [number, number, number]
  /** Per-view orthographic half extents (u, v), metres at unit scale. */
  ext: Float32Array<ArrayBuffer>
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
}

type Vec3 = [number, number, number]

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

// ---------------------------------------------------------------------------
// Hemi-octahedral view grid — the TypeScript twin of impostor_info.wgsl.
// ---------------------------------------------------------------------------

/** Node (i, j) -> unit view direction (from the plant toward the camera). */
export function nodeDirection(i: number, j: number, gridN = GRID_N): Vec3 {
  const fx = (i / (gridN - 1)) * 2 - 1
  const fy = (j / (gridN - 1)) * 2 - 1
  const nx = (fx - fy) * 0.5
  const nz = (fx + fy) * 0.5
  const ny = 1 - Math.abs(nx) - Math.abs(nz)
  const len = Math.hypot(nx, ny, nz) || 1
  return [nx / len, ny / len, nz / len]
}

/** cross(+Y, d), normalized, with the same vertical fallback as view_right(). */
export function viewRight(d: Vec3): Vec3 {
  const hx = d[2]
  const hz = -d[0]
  const len = Math.hypot(hx, hz)
  if (len < 1e-4) return [1, 0, 0]
  return [hx / len, 0, hz / len]
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/** Support of the AABB half extents along an axis = |a| . h. */
function support(a: Vec3, h: Vec3): number {
  return Math.abs(a[0]) * h[0] + Math.abs(a[1]) * h[1] + Math.abs(a[2]) * h[2]
}

export interface ViewFrame {
  dir: Vec3
  right: Vec3
  up: Vec3
  extU: number
  extV: number
  extD: number
}

/** Every view's basis + padded extents for a plant with AABB half extents h. */
export function buildViewFrames(half: Vec3, gridN = GRID_N): ViewFrame[] {
  const out: ViewFrame[] = []
  for (let j = 0; j < gridN; j++) {
    for (let i = 0; i < gridN; i++) {
      const dir = nodeDirection(i, j, gridN)
      const right = viewRight(dir)
      const up = cross(dir, right)
      out.push({
        dir,
        right,
        up,
        extU: support(right, half) * TILE_PAD,
        extV: support(up, half) * TILE_PAD,
        extD: support(dir, half) * TILE_PAD,
      })
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// Artifact pack / unpack
// ---------------------------------------------------------------------------

export function unpackViewSet(buf: ArrayBuffer): ViewSet | null {
  if (buf.byteLength !== ARTIFACT_BYTES) return null
  const u = new Uint32Array(buf, 0, 4)
  if (u[0] !== MAGIC || u[1] !== VERSION || u[2] !== GRID_N || u[3] !== TILE) return null
  const f = new Float32Array(buf, 16, 8)
  return {
    gridN: GRID_N,
    tilePx: TILE,
    centre: [f[0]!, f[1]!, f[2]!],
    half: [f[4]!, f[5]!, f[6]!],
    ext: new Float32Array(buf, HEADER_BYTES, N_VIEWS * 2),
    albedo: new Uint8Array(buf, HEADER_BYTES + EXT_BYTES, ALBEDO_BYTES),
    normalOct: new Uint8Array(buf, HEADER_BYTES + EXT_BYTES + ALBEDO_BYTES, NORMAL_BYTES),
  }
}

function packViewSet(v: ViewSet): ArrayBuffer {
  const buf = new ArrayBuffer(ARTIFACT_BYTES)
  const u = new Uint32Array(buf, 0, 4)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = GRID_N
  u[3] = TILE
  const f = new Float32Array(buf, 16, 8)
  f.set(v.centre, 0)
  f.set(v.half, 4)
  new Float32Array(buf, HEADER_BYTES, N_VIEWS * 2).set(v.ext)
  new Uint8Array(buf, HEADER_BYTES + EXT_BYTES, ALBEDO_BYTES).set(v.albedo)
  new Uint8Array(buf, HEADER_BYTES + EXT_BYTES + ALBEDO_BYTES, NORMAL_BYTES).set(v.normalOct)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake this species' view set. Every
 * result is magic+size validated: the dev server answers a missing
 * /mesh/baked file with the SPA index.html at HTTP 200, which has poisoned
 * other experiments' caches, so a bad artifact is rebaked and the cache entry
 * repaired rather than trusted.
 */
export async function loadSpeciesViewSet(ctx: BakeCtx, speciesId: string): Promise<ViewSet> {
  const key = `views-v${VERSION}-${GRID_N}x${TILE}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesViewSet(ctx, mesh)
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
    /* best effort — a cache miss just rebakes */
  }
}

// ---------------------------------------------------------------------------
// The bake itself
// ---------------------------------------------------------------------------

async function bakeSpeciesViewSet(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [x0, y0, z0] = hdr.boundsMin
  const [x1, y1, z1] = hdr.boundsMax
  const centre: Vec3 = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
  const half: Vec3 = [(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]
  const frames = buildViewFrames(half)

  // --- transient GPU resources (destroyed before returning) ------------------
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
        size: [CHUNK_PX, CHUNK_PX],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('bake-albedo')
  const nrmTex = mkTarget('bake-normal')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [CHUNK_PX, CHUNK_PX],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // One dynamic-offset uniform slot per view (24 floats used of 64).
  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const uniData = new Float32Array((STRIDE / 4) * N_VIEWS)
  frames.forEach((f, k) => {
    const o = (k * STRIDE) / 4
    uniData.set([f.right[0], f.right[1], f.right[2], Math.max(f.extU, 1e-4)], o)
    uniData.set([f.up[0], f.up[1], f.up[2], Math.max(f.extV, 1e-4)], o + 4)
    uniData.set([f.dir[0], f.dir[1], f.dir[2], Math.max(f.extD, 1e-4)], o + 8)
    uniData.set([centre[0], centre[1], centre[2], 0], o + 12)
    uniData.set([x0, y0, z0, 0], o + 16)
    uniData.set([x1 - x0, y1 - y0, z1 - z0, 0], o + 20)
  })
  device.queue.writeBuffer(uni, 0, uniData)

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
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const bpr = CHUNK_PX * 4
  const rbBytes = bpr * CHUNK_PX
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: rbBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('bake-rb-albedo')
  const rbNrm = mkReadback('bake-rb-normal')

  const albedo = new Uint8Array(ALBEDO_BYTES)
  const normalOct = new Uint8Array(NORMAL_BYTES)
  const scratch = makeResolveScratch()

  for (let first = 0; first < N_VIEWS; first += CHUNK_VIEWS) {
    const count = Math.min(CHUNK_VIEWS, N_VIEWS - first)
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-chunk-${first}` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-pass-${first}`,
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
    for (let k = 0; k < count; k++) {
      const col = k % CHUNK_SIDE
      const row = Math.floor(k / CHUNK_SIDE)
      pass.setViewport(col * BIG, row * BIG, BIG, BIG, 0, 1)
      pass.setScissorRect(col * BIG, row * BIG, BIG, BIG)
      pass.setBindGroup(0, bg, [(first + k) * STRIDE])
      pass.drawIndexed(indices.length)
    }
    pass.end()
    const copySize: [number, number] = [CHUNK_PX, CHUNK_PX]
    enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: CHUNK_PX }, copySize)
    enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: CHUNK_PX }, copySize)
    device.queue.submit([enc.finish()])

    await rbAlb.mapAsync(GPUMapMode.READ)
    await rbNrm.mapAsync(GPUMapMode.READ)
    const bigAlb = new Uint8Array(rbAlb.getMappedRange())
    const bigNrm = new Uint8Array(rbNrm.getMappedRange())
    for (let k = 0; k < count; k++) {
      const col = k % CHUNK_SIDE
      const row = Math.floor(k / CHUNK_SIDE)
      resolveTile(bigAlb, bigNrm, col * BIG, row * BIG, scratch, albedo, normalOct, first + k)
    }
    rbAlb.unmap()
    rbNrm.unmap()
  }

  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, uni, rbAlb, rbNrm]) r.destroy()

  const ext = new Float32Array(N_VIEWS * 2)
  frames.forEach((f, k) => {
    ext[k * 2] = f.extU
    ext[k * 2 + 1] = f.extV
  })
  return packViewSet({ gridN: GRID_N, tilePx: TILE, centre, half, ext, albedo, normalOct })
}

// ---------------------------------------------------------------------------
// CPU resolve: supersample downsample -> dilate -> oct-encode, per tile.
// ---------------------------------------------------------------------------

interface ResolveScratch {
  rgba: Float32Array
  nrm: Float32Array
  filled: Uint8Array
  next: Uint8Array
}

function makeResolveScratch(): ResolveScratch {
  const n = TILE * TILE
  return {
    rgba: new Float32Array(n * 4),
    nrm: new Float32Array(n * 3),
    filled: new Uint8Array(n),
    next: new Uint8Array(n),
  }
}

/**
 * Resolve one supersampled tile of the chunk target into layer `layer` of the
 * output arrays: coverage-weighted colour/normal average, then a few dilation
 * passes so bilinear and mip filtering at the silhouette never pull the
 * cleared background in. Dilated texels keep alpha 0, so they feed the filter
 * but contribute nothing to the runtime's coverage-weighted blend.
 */
function resolveTile(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  ox: number,
  oy: number,
  s: ResolveScratch,
  outAlbedo: Uint8Array,
  outNormal: Uint8Array,
  layer: number,
): void {
  const { rgba, nrm } = s
  s.filled.fill(0)
  for (let ty = 0; ty < TILE; ty++) {
    for (let tx = 0; tx < TILE; tx++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const src = ((oy + ty * SS + j) * CHUNK_PX + (ox + tx * SS + i)) * 4
          const a = bigAlb[src + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[src]! * a
          g += bigAlb[src + 1]! * a
          b += bigAlb[src + 2]! * a
          nx += (bigNrm[src]! / 127.5 - 1) * a
          ny += (bigNrm[src + 1]! / 127.5 - 1) * a
          nz += (bigNrm[src + 2]! / 127.5 - 1) * a
        }
      }
      const d = ty * TILE + tx
      if (aSum === 0) {
        rgba[d * 4] = 0
        rgba[d * 4 + 1] = 0
        rgba[d * 4 + 2] = 0
        rgba[d * 4 + 3] = 0
        nrm[d * 3] = 0
        nrm[d * 3 + 1] = 1
        nrm[d * 3 + 2] = 0
        continue
      }
      rgba[d * 4] = r / aSum
      rgba[d * 4 + 1] = g / aSum
      rgba[d * 4 + 2] = b / aSum
      rgba[d * 4 + 3] = aSum / (SS * SS)
      nrm[d * 3] = nx
      nrm[d * 3 + 1] = ny
      nrm[d * 3 + 2] = nz
      s.filled[d] = 1
    }
  }

  let cur = s.filled
  let next = s.next
  for (let p = 0; p < DILATE_PASSES; p++) {
    next.set(cur)
    for (let y = 0; y < TILE; y++) {
      for (let x = 0; x < TILE; x++) {
        const idx = y * TILE + x
        if (cur[idx] !== 0) continue
        let count = 0
        let r = 0
        let g = 0
        let b = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= TILE) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if (xx < 0 || xx >= TILE || (i === 0 && j === 0)) continue
            const n = yy * TILE + xx
            if (cur[n] === 0) continue
            count++
            r += rgba[n * 4]!
            g += rgba[n * 4 + 1]!
            b += rgba[n * 4 + 2]!
            nx += nrm[n * 3]!
            ny += nrm[n * 3 + 1]!
            nz += nrm[n * 3 + 2]!
          }
        }
        if (count === 0) continue
        rgba[idx * 4] = r / count
        rgba[idx * 4 + 1] = g / count
        rgba[idx * 4 + 2] = b / count
        nrm[idx * 3] = nx / count
        nrm[idx * 3 + 1] = ny / count
        nrm[idx * 3 + 2] = nz / count
        next[idx] = 1
      }
    }
    const swap = cur
    cur = next
    next = swap
  }

  const albBase = layer * TILE * TILE * 4
  const nrmBase = layer * TILE * TILE * 2
  for (let i = 0; i < TILE * TILE; i++) {
    outAlbedo[albBase + i * 4] = Math.round(rgba[i * 4]!)
    outAlbedo[albBase + i * 4 + 1] = Math.round(rgba[i * 4 + 1]!)
    outAlbedo[albBase + i * 4 + 2] = Math.round(rgba[i * 4 + 2]!)
    outAlbedo[albBase + i * 4 + 3] = Math.round(rgba[i * 4 + 3]!)
    const [u, v] = octEncode(nrm[i * 3]!, nrm[i * 3 + 1]!, nrm[i * 3 + 2]!)
    outNormal[nrmBase + i * 2] = u
    outNormal[nrmBase + i * 2 + 1] = v
  }
}

/** Octahedral encode, y-primary — exact inverse of oct_decode_y in impostor_info.wgsl. */
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
