import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Depth-shell impostor bake.
 *
 * Per species, 13 orthographic views of the raw GCMESH1 mesh:
 *   views 0..7   azimuth k*45deg at elevation 0   (the grazing workhorses)
 *   views 8..11  azimuth k*90deg at elevation 45deg
 *   view  12     straight down
 * Each view stores
 *   texA (TILE_A^2, rgba8)  albedo.rgb + coverage
 *   texB (TILE_B^2, rgba8)  oct normal xy + burial AO + thickness
 *   shell (GRID_V^2 f32)    coverage-weighted FRONT depth in metres at the
 *                           lattice corners, + = toward the bake camera
 * The shell is what turns the runtime card into a real 3D surface; the AO
 * channel is "how far behind my local front surface am I" (a baked thickness
 * cue) and thickness is (back - front) along the view ray.
 *
 * Artifact layout (little-endian):
 *   u32 magic 'DOM1', version, nViews, tileA, tileB, gridV
 *   f32 rXZ, y0, y1, cx, cz          (capture box, unit scale, metres)
 *   ...zeros to byte 64
 *   f32[nViews*4]            su, sv, sd, ring
 *   f32[nViews*gridV*gridV]  front-depth shell (metres)
 *   u8 [nViews*tileA^2*4]    texA
 *   u8 [nViews*tileB^2*4]    texB
 */

export const N_AZ0 = 8
export const N_AZ1 = 4
export const N_VIEWS = N_AZ0 + N_AZ1 + 1
export const TILE_A = 480
export const TILE_B = 240
export const GRID_V = 9
const SS = 2
const BIG = TILE_A * SS
const MAGIC = 0x314d4f44 // 'DOM1'
const VERSION = 3
const HEADER_BYTES = 64
const DILATE_A = 5
const DILATE_B = 4
/** Burial AO: darkening at full burial, and the burial depth that reaches it. */
const AO_STRENGTH = 0.45
const AO_RANGE_FRAC = 0.55
/** Local canopy density that counts as fully occluding for the burial term. */
const AO_DENSITY_FULL = 0.4
/** Grounding gradient: light multiplier at the very bottom of the plant. */
const GROUND_MIN = 0.6
/** Occlusion/thickness blur radius (texels of TILE_B) — kills blade speckle. */
const AO_BLUR = 3

export interface DomeView {
  su: number
  sv: number
  sd: number
  ring: number
  fwd: [number, number, number]
  right: [number, number, number]
  up: [number, number, number]
}

export interface DomeAtlas {
  tileA: number
  tileB: number
  gridV: number
  nViews: number
  /** Horizontal support radius of the capture box (m at scale 1). */
  rXZ: number
  y0: number
  y1: number
  /** Clump center offset in the mesh frame (the capture was centered here). */
  cx: number
  cz: number
  /** su, sv, sd, ring per view — flat, 4 floats each. */
  viewExt: Float32Array<ArrayBuffer>
  /** Front-depth shell, metres, [view][gy*gridV+gx]. */
  shell: Float32Array<ArrayBuffer>
  texA: Uint8Array<ArrayBuffer>
  texB: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

/** Half-extents of the capture box for a view at `elev`, cylinder support. */
function extents(rXZ: number, height: number, elev: number): [number, number, number] {
  const se = Math.sin(elev)
  const ce = Math.cos(elev)
  return [rXZ, rXZ * se + (height / 2) * ce, rXZ * ce + (height / 2) * se]
}

/** The 13 view frames. Azimuth 0 looks along +Z (matches atan2(x, z)). */
export function buildViews(rXZ: number, height: number): DomeView[] {
  const out: DomeView[] = []
  const push = (elev: number, az: number, ring: number): void => {
    const se = Math.sin(elev)
    const ce = Math.cos(elev)
    const sa = Math.sin(az)
    const ca = Math.cos(az)
    const [su, sv, sd] = extents(rXZ, height, elev)
    const fwd: [number, number, number] = [ce * sa, se, ce * ca]
    const right: [number, number, number] = ring === 2 ? [1, 0, 0] : [ca, 0, -sa]
    const up: [number, number, number] = ring === 2 ? [0, 0, -1] : [-se * sa, ce, -se * ca]
    out.push({ su, sv, sd, ring, fwd, right, up })
  }
  for (let k = 0; k < N_AZ0; k++) push(0, (k * 2 * Math.PI) / N_AZ0, 0)
  for (let k = 0; k < N_AZ1; k++) push(Math.PI / 4, (k * 2 * Math.PI) / N_AZ1, 1)
  push(Math.PI / 2, 0, 2)
  return out
}

export function unpackDome(buf: ArrayBuffer): DomeAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 6)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const nViews = u[2]!
  const tileA = u[3]!
  const tileB = u[4]!
  const gridV = u[5]!
  if (nViews !== N_VIEWS || tileA !== TILE_A || tileB !== TILE_B || gridV !== GRID_V) return null
  const extBytes = nViews * 4 * 4
  const shellBytes = nViews * gridV * gridV * 4
  const aBytes = nViews * tileA * tileA * 4
  const bBytes = nViews * tileB * tileB * 4
  if (buf.byteLength !== HEADER_BYTES + extBytes + shellBytes + aBytes + bBytes) return null
  const f = new Float32Array(buf, 24, 5)
  let o = HEADER_BYTES
  const viewExt = new Float32Array(buf, o, nViews * 4)
  o += extBytes
  const shell = new Float32Array(buf, o, nViews * gridV * gridV)
  o += shellBytes
  const texA = new Uint8Array(buf, o, aBytes)
  o += aBytes
  const texB = new Uint8Array(buf, o, bBytes)
  return {
    tileA,
    tileB,
    gridV,
    nViews,
    rXZ: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    cx: f[3]!,
    cz: f[4]!,
    viewExt,
    shell,
    texA,
    texB,
  }
}

function packDome(a: DomeAtlas): ArrayBuffer {
  const extBytes = a.nViews * 4 * 4
  const shellBytes = a.shell.byteLength
  const buf = new ArrayBuffer(HEADER_BYTES + extBytes + shellBytes + a.texA.byteLength + a.texB.byteLength)
  const u = new Uint32Array(buf, 0, 6)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.nViews
  u[3] = a.tileA
  u[4] = a.tileB
  u[5] = a.gridV
  const f = new Float32Array(buf, 24, 5)
  f[0] = a.rXZ
  f[1] = a.y0
  f[2] = a.y1
  f[3] = a.cx
  f[4] = a.cz
  let o = HEADER_BYTES
  new Float32Array(buf, o, a.nViews * 4).set(a.viewExt)
  o += extBytes
  new Float32Array(buf, o, a.shell.length).set(a.shell)
  o += shellBytes
  new Uint8Array(buf, o, a.texA.byteLength).set(a.texA)
  o += a.texA.byteLength
  new Uint8Array(buf, o, a.texB.byteLength).set(a.texB)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the depth-shell atlas.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesDome(ctx: BakeCtx, speciesId: string): Promise<DomeAtlas> {
  const key = `dome-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesDome(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackDome(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackDome(buf)
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

async function bakeSpeciesDome(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2

  // Exact horizontal support radius from the vertices (bounds corners
  // overestimate it noticeably on wide community tiles).
  const verts = mesh.vertices
  const sxq = (bx1 - bx0) / 65535
  const szq = (bz1 - bz0) / 65535
  let r2 = 0
  for (let i = 0; i < hdr.vertexCount; i++) {
    const dx = bx0 + verts[i * 8]! * sxq - cx
    const dz = bz0 + verts[i * 8 + 2]! * szq - cz
    const d = dx * dx + dz * dz
    if (d > r2) r2 = d
  }
  const rXZ = Math.sqrt(r2) * 1.02 + 1e-3
  const y0 = Math.min(0, by0)
  const y1 = by1
  const height = y1 - y0
  const cy = (y0 + y1) / 2
  const views = buildViews(rXZ, height)

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

  const mkTarget = (label: string, format: GPUTextureFormat): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [BIG, BIG],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('bake-albedo', 'rgba8unorm')
  const nrmTex = mkTarget('bake-normal', 'rgba8unorm')
  const backTex = mkTarget('bake-back', 'r8unorm')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [BIG, BIG],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // Per-view uniforms in one buffer with dynamic offsets (28 floats used).
  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_VIEWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_VIEWS)
  views.forEach((v, k) => {
    const o = (k * STRIDE) / 4
    scratch.set(v.right, o)
    scratch.set(v.up, o + 4)
    scratch.set(v.fwd, o + 8)
    scratch.set([cx, cy, cz, 0], o + 12)
    scratch.set([v.su, v.sv, v.sd, 0], o + 16)
    scratch.set([bx0, by0, bz0, 0], o + 20)
    scratch.set([bx1 - bx0, by1 - by0, bz1 - bz0, 0], o + 24)
  })
  device.queue.writeBuffer(uni, 0, scratch)

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 112 },
      },
    ],
  })
  const bg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
  })

  const module = ctx.shaders.module(bakeShaderSrc)
  const layout = device.createPipelineLayout({ label: `${ctx.id}/bake-pl`, bindGroupLayouts: [bgl] })
  const vertexState: GPUVertexState = {
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
  }
  const frontPipe = device.createRenderPipeline({
    label: `${ctx.id}/bake-front`,
    layout,
    vertex: vertexState,
    fragment: {
      module,
      entryPoint: 'fs_front',
      targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })
  const backPipe = device.createRenderPipeline({
    label: `${ctx.id}/bake-back`,
    layout,
    vertex: vertexState,
    fragment: { module, entryPoint: 'fs_back', targets: [{ format: 'r8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'greater', depthWriteEnabled: true },
  })

  // Readback buffers, reused for every view (one submit + map per view keeps
  // the peak footprint at ~8MB instead of ~110MB for a 13-view batch).
  const rowRgba = BIG * 4
  const rowR8 = Math.ceil(BIG / 256) * 256
  const mkReadback = (label: string, size: number): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('rb-albedo', rowRgba * BIG)
  const rbNrm = mkReadback('rb-normal', rowRgba * BIG)
  const rbBack = mkReadback('rb-back', rowR8 * BIG)

  const texA = new Uint8Array(N_VIEWS * TILE_A * TILE_A * 4)
  const texB = new Uint8Array(N_VIEWS * TILE_B * TILE_B * 4)
  const shell = new Float32Array(N_VIEWS * GRID_V * GRID_V)
  const viewExt = new Float32Array(N_VIEWS * 4)

  for (let k = 0; k < N_VIEWS; k++) {
    const v = views[k]!
    viewExt.set([v.su, v.sv, v.sd, v.ring], k * 4)

    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-${k}` })
    const front = enc.beginRenderPass({
      label: `${ctx.id}/bake-front-${k}`,
      colorAttachments: [
        { view: albTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: nrmTex.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    front.setPipeline(frontPipe)
    front.setVertexBuffer(0, vbuf)
    front.setIndexBuffer(ibuf, 'uint32')
    front.setBindGroup(0, bg, [k * STRIDE])
    front.drawIndexed(indices.length)
    front.end()

    const back = enc.beginRenderPass({
      label: `${ctx.id}/bake-back-${k}`,
      colorAttachments: [
        { view: backTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthClearValue: 0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    back.setPipeline(backPipe)
    back.setVertexBuffer(0, vbuf)
    back.setIndexBuffer(ibuf, 'uint32')
    back.setBindGroup(0, bg, [k * STRIDE])
    back.drawIndexed(indices.length)
    back.end()

    enc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: rowRgba, rowsPerImage: BIG }, [BIG, BIG])
    enc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: rowRgba, rowsPerImage: BIG }, [BIG, BIG])
    enc.copyTextureToBuffer({ texture: backTex }, { buffer: rbBack, bytesPerRow: rowR8, rowsPerImage: BIG }, [BIG, BIG])
    device.queue.submit([enc.finish()])

    await rbAlb.mapAsync(GPUMapMode.READ)
    await rbNrm.mapAsync(GPUMapMode.READ)
    await rbBack.mapAsync(GPUMapMode.READ)
    const bigAlb = new Uint8Array(rbAlb.getMappedRange())
    const bigNrm = new Uint8Array(rbNrm.getMappedRange())
    const bigBack = new Uint8Array(rbBack.getMappedRange())
    postProcessView(bigAlb, bigNrm, bigBack, rowR8, v, { y0, height, cy }, k, texA, texB, shell)
    rbAlb.unmap()
    rbNrm.unmap()
    rbBack.unmap()
  }

  for (const r of [vbuf, ibuf, albTex, nrmTex, backTex, depthTex, uni, rbAlb, rbNrm, rbBack]) r.destroy()

  return packDome({
    tileA: TILE_A,
    tileB: TILE_B,
    gridV: GRID_V,
    nViews: N_VIEWS,
    rXZ,
    y0,
    y1,
    cx,
    cz,
    viewExt,
    shell,
    texA,
    texB,
  })
}

interface Box {
  y0: number
  height: number
  cy: number
}

/**
 * One view: coverage-weighted 2x downsample to TILE_A, a second reduction to
 * TILE_B, dilation into empty texels, the front-depth shell at the lattice
 * corners, and the burial-AO / thickness channels.
 */
function postProcessView(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  bigBack: Uint8Array,
  rowR8: number,
  view: DomeView,
  box: Box,
  viewIndex: number,
  texA: Uint8Array,
  texB: Uint8Array,
  shell: Float32Array,
): void {
  const A = TILE_A
  const B = TILE_B
  // --- 2x downsample (960 -> 480), coverage weighted -------------------------
  const covA = new Float32Array(A * A)
  const depA = new Float32Array(A * A)
  const bckA = new Float32Array(A * A)
  const nrmA = new Float32Array(A * A * 3)
  const albA = new Uint8Array(A * A * 4)
  for (let y = 0; y < A; y++) {
    for (let x = 0; x < A; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let dep = 0
      let bck = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const sx = x * SS + i
          const sy = y * SS + j
          const s = (sy * BIG + sx) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
          nx += (bigNrm[s]! / 127.5 - 1) * a
          ny += (bigNrm[s + 1]! / 127.5 - 1) * a
          nz += (bigNrm[s + 2]! / 127.5 - 1) * a
          dep += (bigNrm[s + 3]! / 255) * a
          bck += (bigBack[sy * rowR8 + sx]! / 255) * a
        }
      }
      const d = y * A + x
      if (aSum === 0) continue
      albA[d * 4] = Math.round(r / aSum)
      albA[d * 4 + 1] = Math.round(g / aSum)
      albA[d * 4 + 2] = Math.round(b / aSum)
      albA[d * 4 + 3] = Math.round(aSum / (SS * SS))
      covA[d] = aSum / (255 * SS * SS)
      depA[d] = dep / aSum
      bckA[d] = bck / aSum
      nrmA[d * 3] = nx / aSum
      nrmA[d * 3 + 1] = ny / aSum
      nrmA[d * 3 + 2] = nz / aSum
    }
  }

  // --- front-depth shell at the (GRID_V x GRID_V) lattice corners -----------
  // Corner (i,j) sits at texture uv (i/8, j/8) — j runs top-down like the
  // image, which is exactly how the runtime indexes it.
  const cells = GRID_V - 1
  const half = Math.max(2, Math.round(A / (2 * cells)))
  const sd = view.sd
  const shellBase = viewIndex * GRID_V * GRID_V
  const filledC = new Uint8Array(GRID_V * GRID_V)
  for (let j = 0; j < GRID_V; j++) {
    const cyT = Math.min(A - 1, Math.round((j / cells) * A))
    for (let i = 0; i < GRID_V; i++) {
      const cxT = Math.min(A - 1, Math.round((i / cells) * A))
      let wSum = 0
      let dSum = 0
      for (let yy = Math.max(0, cyT - half); yy <= Math.min(A - 1, cyT + half); yy++) {
        for (let xx = Math.max(0, cxT - half); xx <= Math.min(A - 1, cxT + half); xx++) {
          const s = yy * A + xx
          const w = covA[s]!
          if (w <= 0) continue
          wSum += w
          dSum += depA[s]! * w
        }
      }
      const idx = j * GRID_V + i
      if (wSum > 1e-4) {
        shell[shellBase + idx] = (0.5 - dSum / wSum) * 2 * sd
        filledC[idx] = 1
      }
    }
  }
  // Fill corners with no coverage from their filled neighbours so the lattice
  // stays a smooth envelope instead of snapping to the centre plane.
  for (let pass = 0; pass < 6; pass++) {
    let changed = false
    const next = filledC.slice()
    for (let j = 0; j < GRID_V; j++) {
      for (let i = 0; i < GRID_V; i++) {
        const idx = j * GRID_V + i
        if (filledC[idx] !== 0) continue
        let n = 0
        let sum = 0
        for (const [di, dj] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ] as const) {
          const ii = i + di
          const jj = j + dj
          if (ii < 0 || jj < 0 || ii >= GRID_V || jj >= GRID_V) continue
          const nIdx = jj * GRID_V + ii
          if (filledC[nIdx] === 0) continue
          n++
          sum += shell[shellBase + nIdx]!
        }
        if (n === 0) continue
        shell[shellBase + idx] = sum / n
        next[idx] = 1
        changed = true
      }
    }
    filledC.set(next)
    if (!changed) break
  }

  // --- dilate albedo into empty texels (alpha stays 0) ----------------------
  dilateRgb(albA, covA, A, DILATE_A)

  // --- reduce normals / depth / back to TILE_B ------------------------------
  const step = A / B
  const covB = new Float32Array(B * B)
  const depB = new Float32Array(B * B)
  const bckB = new Float32Array(B * B)
  const nrmB = new Float32Array(B * B * 3)
  for (let y = 0; y < B; y++) {
    for (let x = 0; x < B; x++) {
      let w = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let dep = 0
      let bck = 0
      for (let j = 0; j < step; j++) {
        for (let i = 0; i < step; i++) {
          const s = (y * step + j) * A + (x * step + i)
          const c = covA[s]!
          if (c <= 0) continue
          w += c
          nx += nrmA[s * 3]! * c
          ny += nrmA[s * 3 + 1]! * c
          nz += nrmA[s * 3 + 2]! * c
          dep += depA[s]! * c
          bck += bckA[s]! * c
        }
      }
      const d = y * B + x
      if (w <= 0) continue
      covB[d] = w / (step * step)
      nrmB[d * 3] = nx / w
      nrmB[d * 3 + 1] = ny / w
      nrmB[d * 3 + 2] = nz / w
      depB[d] = dep / w
      bckB[d] = bck / w
    }
  }
  dilateChannels(covB, B, DILATE_B, [nrmB, depB, bckB], [3, 1, 1])
  // Local canopy density at the shell-cell scale: burial only darkens where
  // there is actually material in front of the texel, so a sparse fluffy head
  // at the back of the clump stays bright instead of turning to mud.
  const densB = boxBlur(covB, B, Math.max(2, Math.round(B / (2 * cells))))

  // --- AO (burial + grounding) and thickness, then pack --------------------
  const aoRange = Math.max(0.02, AO_RANGE_FRAC * sd)
  const aoRaw = new Float32Array(B * B)
  const thickRaw = new Float32Array(B * B)
  for (let y = 0; y < B; y++) {
    const v = 1 - 2 * ((y + 0.5) / B)
    for (let x = 0; x < B; x++) {
      const d = y * B + x
      const dep = depB[d]!
      const dm = (0.5 - dep) * 2 * sd
      const front = sampleShell(shell, shellBase, (x + 0.5) / B, (y + 0.5) / B)
      const buried = Math.max(0, front - dm)
      const dens = Math.min(1, densB[d]! / AO_DENSITY_FULL)
      const burialOccl = AO_STRENGTH * smoothstep(0.004, aoRange, buried) * dens
      // World height of this texel in the mesh frame, from the reconstructed
      // 3D point — correct for tilted and top-down views too.
      const yMesh = box.cy + v * view.sv * view.up[1] + dm * view.fwd[1]
      const hf = Math.min(1, Math.max(0, (yMesh - box.y0) / Math.max(1e-4, box.height)))
      const groundOccl = (1 - GROUND_MIN) * (1 - Math.pow(hf, 0.55))
      // max, not product: near the soil both terms describe the same shadow.
      aoRaw[d] = 1 - Math.max(burialOccl, groundOccl)
      thickRaw[d] = Math.min(1, Math.max(0, bckB[d]! - dep))
    }
  }
  // Occlusion is a clump-scale quantity: per-texel burial swings by centimetres
  // between neighbouring blades, and leaving that in reads as dirt on the
  // fluffy heads. Blur it (and thickness) back to the scale it belongs to.
  const aoS = boxBlur(aoRaw, B, AO_BLUR)
  const thickS = boxBlur(thickRaw, B, AO_BLUR)
  const outB = viewIndex * B * B * 4
  for (let d = 0; d < B * B; d++) {
    const [ou, ov] = octEncode(nrmB[d * 3]!, nrmB[d * 3 + 1]!, nrmB[d * 3 + 2]!)
    texB[outB + d * 4] = ou
    texB[outB + d * 4 + 1] = ov
    texB[outB + d * 4 + 2] = Math.round(255 * Math.min(1, Math.max(0, aoS[d]!)))
    texB[outB + d * 4 + 3] = Math.round(255 * Math.min(1, Math.max(0, thickS[d]!)))
  }
  texA.set(albA, viewIndex * A * A * 4)
}

/** Separable sliding-window box blur (radius r, clamped edges). */
function boxBlur(src: Float32Array, N: number, r: number): Float32Array {
  const tmp = new Float32Array(N * N)
  const out = new Float32Array(N * N)
  const w = 2 * r + 1
  for (let y = 0; y < N; y++) {
    let sum = 0
    for (let i = -r; i <= r; i++) sum += src[y * N + Math.min(N - 1, Math.max(0, i))]!
    for (let x = 0; x < N; x++) {
      tmp[y * N + x] = sum / w
      const add = src[y * N + Math.min(N - 1, x + r + 1)]!
      const sub = src[y * N + Math.min(N - 1, Math.max(0, x - r))]!
      sum += add - sub
    }
  }
  for (let x = 0; x < N; x++) {
    let sum = 0
    for (let i = -r; i <= r; i++) sum += tmp[Math.min(N - 1, Math.max(0, i)) * N + x]!
    for (let y = 0; y < N; y++) {
      out[y * N + x] = sum / w
      const add = tmp[Math.min(N - 1, y + r + 1) * N + x]!
      const sub = tmp[Math.min(N - 1, Math.max(0, y - r)) * N + x]!
      sum += add - sub
    }
  }
  return out
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / Math.max(1e-6, e1 - e0)))
  return t * t * (3 - 2 * t)
}

/** Bilinear lookup into one view's front-depth lattice (uv in texture space). */
function sampleShell(shell: Float32Array, base: number, u: number, v: number): number {
  const cells = GRID_V - 1
  const gx = Math.min(cells - 1e-4, Math.max(0, u * cells))
  const gy = Math.min(cells - 1e-4, Math.max(0, v * cells))
  const x0 = Math.floor(gx)
  const y0 = Math.floor(gy)
  const fx = gx - x0
  const fy = gy - y0
  const at = (x: number, y: number): number => shell[base + y * GRID_V + x]!
  const a = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
  const b = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
  return a * (1 - fy) + b * fy
}

/** Spread rgb into uncovered texels so filtering never pulls in background. */
function dilateRgb(rgba: Uint8Array, cov: Float32Array, N: number, passes: number): void {
  let filled = Uint8Array.from(cov, (c) => (c > 0 ? 1 : 0))
  for (let p = 0; p < passes; p++) {
    const next = filled.slice()
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (filled[idx] !== 0) continue
        let n = 0
        let r = 0
        let g = 0
        let b = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= N) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < 0 || xx >= N) continue
            const s = yy * N + xx
            if (filled[s] === 0) continue
            n++
            r += rgba[s * 4]!
            g += rgba[s * 4 + 1]!
            b += rgba[s * 4 + 2]!
          }
        }
        if (n === 0) continue
        rgba[idx * 4] = Math.round(r / n)
        rgba[idx * 4 + 1] = Math.round(g / n)
        rgba[idx * 4 + 2] = Math.round(b / n)
        next[idx] = 1
      }
    }
    filled = next
  }
}

/** Same dilation over a set of float channel arrays. */
function dilateChannels(cov: Float32Array, N: number, passes: number, chans: Float32Array[], comps: number[]): void {
  let filled = Uint8Array.from(cov, (c) => (c > 0 ? 1 : 0))
  for (let p = 0; p < passes; p++) {
    const next = filled.slice()
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x
        if (filled[idx] !== 0) continue
        let n = 0
        const acc = chans.map((_, ci) => new Float64Array(comps[ci]!))
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= N) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < 0 || xx >= N) continue
            const s = yy * N + xx
            if (filled[s] === 0) continue
            n++
            chans.forEach((arr, ci) => {
              const c = comps[ci]!
              for (let k = 0; k < c; k++) acc[ci]![k]! += arr[s * c + k]!
            })
          }
        }
        if (n === 0) continue
        chans.forEach((arr, ci) => {
          const c = comps[ci]!
          for (let k = 0; k < c; k++) arr[idx * c + k] = acc[ci]![k]! / n
        })
        next[idx] = 1
      }
    }
    filled = next
  }
}

/** Octahedral encode, y-primary — exact inverse of the decode in gcmesh.ts. */
function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.abs(x) + Math.abs(y) + Math.abs(z)
  let u: number
  let v: number
  if (s < 1e-6) {
    u = 0
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
