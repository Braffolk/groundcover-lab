import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Carpet-atlas bake ("SLPC") — the prism idea rotated 90 degrees for a mat.
 *
 * A Sphagnum tile is 0.18m across and 0.09m tall: it has no silhouette worth a
 * camera-facing card, so every texel of the prism atlas spent on azimuths is
 * dead weight for it. Instead the whole atlas holds HORIZONTAL slices of one
 * periodic tile, seen straight down:
 *
 *   5 tiles @512  height slices (0 = top), cut at equal-VISIBLE-AREA quantiles
 *                 and stored with the mean height of the surface inside them.
 *                 At draw time slice k is a ground-parallel quad at that
 *                 height, so the stack is a stepped relief of the cushion
 *                 instead of one flat plane: the layers slide against each
 *                 other as the view tilts, they occlude each other with real
 *                 depth writes, and the mat has thickness at a ridge.
 *   1 tile  @512  merged top view (front-most = highest wins) — the far LOD
 *                 card, i.e. exactly what a flat billboard mat would draw.
 *
 * Equal VISIBLE AREA, not equal vertex mass, is deliberate. Sphagnum puts most
 * of its geometry in the capitulum layer, so equal-mass quantiles bunch four of
 * the five cuts inside a single centimetre (measured on late-season: 81% of the
 * tile inside 1cm of height) and the relief the eye gets is nil. A depth-probe
 * render of the tile gives the height distribution of the surface you actually
 * see, and quantiling THAT spreads the five layers over the full ~3.5cm.
 *
 * EXCLUSIVE OWNERSHIP, same as the prism bake: each final texel is assigned to
 * the ONE slice that owns the front-most covered subsample and given the MERGED
 * colour and coverage; the losers are cleared. The slices are therefore an
 * exact partition of the merged top view — seen from straight above their union
 * is bit-identical to the far card, so the LOD switch cannot pop, and no texel
 * is ever shaded twice.
 *
 * CUSHION NORMALS (v3). The per-texel mesh normal is real but it is *leaflet*
 * frequency: measured on the wet-vigorous tile its median tilt is 52 degrees,
 * yet neighbouring texels disagree completely, so one screen pixel — already
 * ~4 texels wide at the carpet-close bookmark — averages the whole field to
 * straight up. `debug=normals` came out a flat pale green and `debug=lighting`
 * a flat white: the mat was lit as if it were a sheet of paper, which is why
 * every renderer of this species reads as a photograph of moss rather than as
 * moss. The signal that DOES survive filtering is the one that is spatially
 * coherent — the shape of the capitula themselves — and the bake gets it for
 * free: the normal target's alpha now carries the depth of whatever won the
 * depth test, i.e. a full-resolution HEIGHT FIELD of the visible surface. Blur
 * it to ~3mm (capitulum scale), take its gradient, and add that slope to the
 * leaflet slope. Mip 0 keeps the fine detail; from mip 2 on the cushions are
 * what is left, which is exactly the right thing to be left with.
 *
 * Stored as the plain tangent (n.x, n.z) rather than octahedrally, because
 * this map is *meant* to be filtered: every normal is in the upper hemisphere
 * (the bake flips them toward the camera), so (x, z) is linear, its box-filtered
 * average is the true mean tangent, and y = sqrt(1 - x^2 - z^2) flattens
 * gracefully as the average shortens. The shader can also scale the tangent at
 * runtime, which is what the `carpetNormalGain` param does — 0 reproduces the
 * old flat-lit mat exactly, without a rebake.
 *
 * The ortho box is the TILE SQUARE ([0, tileM]^2 in the mesh frame — the tile
 * origin is (0,0) for every current source mesh), not the mesh bounds: the
 * overhang outside the tile belongs to the neighbouring tile of the periodic
 * community and would be drawn twice. Cropping there is what multiplies the
 * on-screen texel density: 512px over 0.18m = 0.35mm/texel, against ~108px of
 * the prism atlas' whole-mesh top tile.
 *
 * Each slice is rendered over the 3x3 block of PERIODIC COPIES of the mesh.
 * A single copy leaves the tile with a sparse fringe — the moss that should
 * cover it belongs to the neighbours' overhang — and every tile boundary then
 * draws itself as a dark line across the mat (measured before the fix: edge
 * coverage 0.37-0.65 against 0.91 in the middle). Plus a 4-texel margin so
 * bilinear taps at the tile edge have something to read.
 *
 * Artifact layout (little-endian), header 256B:
 *   0   u32 magic 'SLPC', u32 version, u32 atlasW, u32 atlasH
 *   16  u32 tilePx, u32 nSlice, u32 margin, u32 _pad
 *   32  f32 tileM, y0, y1, _pad
 *   48  f32[8] slice centroid heights (metres, mesh frame), [nSlice] = the
 *       merged card's height (coverage-weighted mean of the winning slices)
 *   256 u8[w*h*4] albedo rgba8 (straight colour, a = coverage)
 *       u8[w*h*2] normals rg8 = (n.x, n.z) * 0.5 + 0.5, mesh frame, upper
 *       hemisphere (n.y = sqrt(1 - x^2 - z^2)); linear, so it mips correctly
 */

export const CARPET_TILE_PX = 512
export const CARPET_COLS = 2
export const CARPET_ROWS = 3
export const CARPET_W = CARPET_TILE_PX * CARPET_COLS // 1024
export const CARPET_H = CARPET_TILE_PX * CARPET_ROWS // 1536
/** Height slices per tile; +1 merged tile fills the 2x3 grid exactly. */
export const N_SLICE = 5
/** Texels of real overhang geometry baked around the tile square. */
export const CARPET_MARGIN = 4
/** 512px tiles stay texel-aligned through level 6 (tile = 8px). */
export const CARPET_MIPS = 7

const SS = 2
const BIG_W = CARPET_W * SS
const BIG_H = CARPET_H * SS
const MAGIC = 0x4350_4c53 // 'SLPC'
const VERSION = 3
const HEADER_BYTES = 256
const DILATE_PASSES = 5

/**
 * Cushion-normal filter, in texels of the 512px tile (0.357 mm each).
 * BLUR is the half-width of the box that smooths the raw height field —
 * ±4 texels ≈ 1.4mm, so leaflet-scale noise goes and capitulum-scale shape
 * stays. DIFF is the half-baseline of the central difference, ~2.9mm end to
 * end, which is about one capitulum.
 */
const HEIGHT_BLUR = 4
const HEIGHT_DIFF = 4
/** Slope clamps: 2.0 = 63 degrees. Keeps a single bad texel from spiking. */
const SLOPE_CLAMP = 2.0

export interface CarpetRect {
  x: number
  y: number
}

export function carpetRects(): CarpetRect[] {
  return Array.from({ length: N_SLICE + 1 }, (_, k) => ({
    x: (k % CARPET_COLS) * CARPET_TILE_PX,
    y: Math.floor(k / CARPET_COLS) * CARPET_TILE_PX,
  }))
}

export interface CarpetAtlas {
  tileM: number
  /** Mesh-frame vertical bounds of the capture (metres at scale 1). */
  y0: number
  y1: number
  /** [0..N_SLICE-1] slice centroid heights, [N_SLICE] merged card height. */
  sliceH: Float32Array
  albedo: Uint8Array<ArrayBuffer>
  normalOct: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackCarpet(buf: ArrayBuffer): CarpetAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== CARPET_W || u[3] !== CARPET_H) return null
  if (u[4] !== CARPET_TILE_PX || u[5] !== N_SLICE || u[6] !== CARPET_MARGIN) return null
  if (buf.byteLength !== HEADER_BYTES + CARPET_W * CARPET_H * 6) return null
  const f = new Float32Array(buf, 32, 4)
  return {
    tileM: f[0]!,
    y0: f[1]!,
    y1: f[2]!,
    sliceH: new Float32Array(buf, 48, 8),
    albedo: new Uint8Array(buf, HEADER_BYTES, CARPET_W * CARPET_H * 4),
    normalOct: new Uint8Array(buf, HEADER_BYTES + CARPET_W * CARPET_H * 4, CARPET_W * CARPET_H * 2),
  }
}

function packCarpet(a: CarpetAtlas): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_BYTES + a.albedo.byteLength + a.normalOct.byteLength)
  new Uint32Array(buf, 0, 8).set([MAGIC, VERSION, CARPET_W, CARPET_H, CARPET_TILE_PX, N_SLICE, CARPET_MARGIN, 0])
  new Float32Array(buf, 32, 4).set([a.tileM, a.y0, a.y1, 0])
  new Float32Array(buf, 48, 8).set(a.sliceH)
  new Uint8Array(buf, HEADER_BYTES, a.albedo.byteLength).set(a.albedo)
  new Uint8Array(buf, HEADER_BYTES + a.albedo.byteLength, a.normalOct.byteLength).set(a.normalOct)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the carpet atlas for a species.
 * Same magic-validation shim as the prism loader: the dev server answers a
 * missing /mesh/baked file with the SPA index.html at status 200.
 */
export async function loadSpeciesCarpet(ctx: BakeCtx, speciesId: string): Promise<CarpetAtlas> {
  const key = `carpet-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const desc = speciesById(speciesId)
    if (desc.tileM === undefined) throw new Error(`[${ctx.id}] ${speciesId} is a carpet but has no periodic tileM`)
    const mesh = await ctx.meshes.load(desc.meshId)
    return bakeSpeciesCarpet(ctx, mesh, desc.tileM)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackCarpet(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackCarpet(buf)
    if (!atlas) throw new Error(`[${ctx.id}] carpet bake for ${speciesId} produced an invalid artifact`)
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

/** Equal-mass split of a height histogram into `n` slices (see bake.ts). */
function equalMassSplit(hist: Float64Array, n: number): { cuts: number[]; centers: number[] } {
  const bins = hist.length
  let total = 0
  for (let i = 0; i < bins; i++) total += hist[i]!
  const cuts: number[] = []
  if (total <= 0) {
    for (let k = 1; k < n; k++) cuts.push(k / n)
  } else {
    let acc = 0
    let next = 1
    for (let i = 0; i < bins && next < n; i++) {
      acc += hist[i]!
      while (next < n && acc >= (total * next) / n) {
        cuts.push((i + 1) / bins)
        next++
      }
    }
    while (cuts.length < n - 1) cuts.push(1)
  }
  const edges = [0, ...cuts, 1]
  const centers: number[] = []
  for (let k = 0; k < n; k++) {
    const lo = edges[k]!
    const hi = edges[k + 1]!
    let mass = 0
    let sum = 0
    for (let i = 0; i < bins; i++) {
      const c = (i + 0.5) / bins
      if (c < lo || c >= hi) continue
      mass += hist[i]!
      sum += hist[i]! * c
    }
    centers.push(mass > 0 ? sum / mass : (lo + hi) / 2)
  }
  return { cuts, centers }
}

async function bakeSpeciesCarpet(ctx: BakeCtx, mesh: GcMesh, tileM: number): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const verts = mesh.vertices
  const y0 = by0
  const y1 = by1

  // --- transient GPU resources ---------------------------------------------
  const rects = carpetRects()
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/cbake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/cbake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)

  const mkTarget = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [BIG_W, BIG_H],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    )
  const albTex = mkTarget('cbake-albedo')
  const nrmTex = mkTarget('cbake-normal')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/cbake-depth`,
      size: [BIG_W, BIG_H],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // Ortho box: the tile square plus CARPET_MARGIN texels of margin, so
  // bilinear taps at the tile edge read geometry instead of the neighbouring
  // atlas tile.
  const halfBox = (tileM / 2) * (CARPET_TILE_PX / (CARPET_TILE_PX - 2 * CARPET_MARGIN))
  const STRIDE = 256
  /**
   * One ortho view of the tile. `shift` translates the mesh by whole tile
   * periods: the community is PERIODIC, so the moss that overhangs the tile
   * square is matched by its neighbours' overhang coming the other way. Baking
   * only the centre copy leaves every tile with a sparse fringe (measured: edge
   * coverage 0.37-0.65 against 0.91 in the middle), and the mat then draws a
   * dark line along every tile boundary. Drawing the 3x3 block of periodic
   * copies into the same tile is what makes the seam disappear.
   */
  const view = (dLo: number, dHi: number, shift: [number, number], probe: boolean): number[] => {
    const o = new Array<number>(STRIDE / 4).fill(0)
    o[0] = 1 // right = +X
    o[6] = -1 // card V axis = -Z
    o[9] = 1 // fwd = +Y
    o[11] = 1 // is_top
    o[12] = tileM / 2 - shift[0] // capture centre = the TILE centre, shifted
    o[13] = tileM / 2 - shift[1]
    o[14] = y0
    o[15] = y1
    o[16] = halfBox
    o[17] = dLo
    o[18] = dHi
    o[19] = probe ? 1 : 0
    o[20] = bx0
    o[21] = by0
    o[22] = bz0
    o[24] = bx1 - bx0
    o[25] = by1 - by0
    o[26] = bz1 - bz0
    return o
  }
  const PERIOD: [number, number][] = []
  for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) PERIOD.push([i * tileM, j * tileM])

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/cbake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 112 },
      },
    ],
  })

  const module = ctx.shaders.module(bakeShaderSrc)
  const pipeline = device.createRenderPipeline({
    label: `${ctx.id}/cbake-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/cbake-pl`, bindGroupLayouts: [bgl] }),
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

  const bpr = BIG_W * 4
  /** Render `views` (one dynamic-offset uniform slot each) into their tiles. */
  const renderViews = (views: number[][], tileOf: (i: number) => CarpetRect): void => {
    const uni = ctx.res.createBuffer(
      {
        label: `${ctx.id}/cbake-uni`,
        size: STRIDE * views.length,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { tag: 'bake-scratch' },
    )
    device.queue.writeBuffer(uni, 0, new Float32Array(views.flat()))
    const bg = device.createBindGroup({
      label: `${ctx.id}/cbake-bg`,
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
    })
    const enc = device.createCommandEncoder({ label: `${ctx.id}/cbake-enc` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/cbake-pass`,
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
    views.forEach((_, i) => {
      const r = tileOf(i)
      pass.setViewport(r.x * SS, r.y * SS, CARPET_TILE_PX * SS, CARPET_TILE_PX * SS, 0, 1)
      pass.setScissorRect(r.x * SS, r.y * SS, CARPET_TILE_PX * SS, CARPET_TILE_PX * SS)
      pass.setBindGroup(0, bg, [i * STRIDE])
      pass.drawIndexed(indices.length)
    })
    pass.end()
    device.queue.submit([enc.finish()])
    uni.destroy()
  }

  const readTile = async (tex: GPUTexture, rect: CarpetRect): Promise<Uint8Array> => {
    const size = CARPET_TILE_PX * SS
    const rowBytes = size * 4
    const rb = ctx.res.createBuffer(
      {
        label: `${ctx.id}/cbake-rb-tile`,
        size: rowBytes * size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      },
      { tag: 'bake-scratch' },
    )
    const enc = device.createCommandEncoder({ label: `${ctx.id}/cbake-read-tile` })
    enc.copyTextureToBuffer(
      { texture: tex, origin: { x: rect.x * SS, y: rect.y * SS } },
      { buffer: rb, bytesPerRow: rowBytes, rowsPerImage: size },
      [size, size],
    )
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const out = new Uint8Array(rb.getMappedRange()).slice()
    rb.unmap()
    rb.destroy()
    return out
  }

  // --- pass A: where does the VISIBLE surface actually sit? -----------------
  // Slicing by equal vertex mass puts four of five cuts inside the capitulum
  // layer, where the geometry is dense but the surface barely moves (measured:
  // 81% of the visible tile inside 1cm of height). What the relief stack needs
  // is equal VISIBLE AREA, so the layers spread across the whole 3.5cm the eye
  // can actually see. One depth-probe render of the whole tile gives exactly
  // that distribution.
  renderViews([view(-1e-4, 1 + 1e-4, [0, 0], true)], () => rects[0]!)
  const probe = await readTile(albTex, rects[0]!)
  const BINS = 512
  const hist = new Float64Array(BINS)
  const probeSize = CARPET_TILE_PX * SS
  const skip = CARPET_MARGIN * SS * 2 // ignore the margin band
  for (let y = skip; y < probeSize - skip; y++) {
    for (let x = skip; x < probeSize - skip; x++) {
      const i = (y * probeSize + x) * 4
      if (probe[i + 3] === 0) continue
      const d = probe[i]! / 255
      hist[Math.min(BINS - 1, Math.max(0, Math.floor(d * BINS)))]! += 1
    }
  }
  const split = equalMassSplit(hist, N_SLICE)
  const sliceH = new Float32Array(8)
  for (let k = 0; k < N_SLICE; k++) sliceH[k] = y1 - split.centers[k]! * (y1 - y0)

  // --- pass B: the slices, each over the 3x3 block of periodic copies -------
  const edges = [0, ...split.cuts, 1]
  const sliceViews: number[][] = []
  const sliceTile: number[] = []
  for (let k = 0; k < N_SLICE; k++) {
    for (const shift of PERIOD) {
      sliceViews.push(view(k === 0 ? -1e-4 : edges[k]!, k === N_SLICE - 1 ? 1 + 1e-4 : edges[k + 1]!, shift, false))
      sliceTile.push(k)
    }
  }
  renderViews(sliceViews, (i) => rects[sliceTile[i]!]!)
  const mkReadback = (label: string): GPUBuffer =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/${label}`, size: bpr * BIG_H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    )
  const rbAlb = mkReadback('cbake-rb-albedo')
  const rbNrm = mkReadback('cbake-rb-normal')
  const readEnc = device.createCommandEncoder({ label: `${ctx.id}/cbake-read` })
  readEnc.copyTextureToBuffer({ texture: albTex }, { buffer: rbAlb, bytesPerRow: bpr, rowsPerImage: BIG_H }, [
    BIG_W,
    BIG_H,
  ])
  readEnc.copyTextureToBuffer({ texture: nrmTex }, { buffer: rbNrm, bytesPerRow: bpr, rowsPerImage: BIG_H }, [
    BIG_W,
    BIG_H,
  ])
  device.queue.submit([readEnc.finish()])

  await rbAlb.mapAsync(GPUMapMode.READ)
  await rbNrm.mapAsync(GPUMapMode.READ)
  const bigAlb = new Uint8Array(rbAlb.getMappedRange()).slice()
  const bigNrm = new Uint8Array(rbNrm.getMappedRange()).slice()
  rbAlb.unmap()
  rbNrm.unmap()
  for (const r of [vbuf, ibuf, albTex, nrmTex, depthTex, rbAlb, rbNrm]) r.destroy()

  const { albedo, normalOct, mergedH } = postProcess(bigAlb, bigNrm, rects, sliceH, tileM, y0, y1)
  sliceH[N_SLICE] = mergedH
  return packCarpet({ tileM, y0, y1, sliceH, albedo, normalOct })
}

/**
 * Exclusive-ownership resolve + merged tile + cushion normals + dilation.
 *
 * For every final texel the SS x SS subsamples vote for the slice that is
 * front-most (highest) there; the winner takes the merged colour and coverage
 * and the losers are cleared, so the slices partition the top view exactly.
 * The merged tile gets the same accumulated sample, which is precisely what a
 * z-buffer over all slices would have produced.
 *
 * The same accumulation reads the depth the bake wrote into the normal
 * target's alpha, which gives a full-resolution height field of the visible
 * surface. Blurring that to capitulum scale and differencing it yields the one
 * normal component that is spatially coherent enough to survive minification —
 * see the file header for why that matters more than the leaflet normals do.
 */
function postProcess(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
  rects: CarpetRect[],
  sliceH: Float32Array,
  tileM: number,
  y0: number,
  y1: number,
): { albedo: Uint8Array<ArrayBuffer>; normalOct: Uint8Array<ArrayBuffer>; mergedH: number } {
  const W = CARPET_W
  const H = CARPET_H
  const TP = CARPET_TILE_PX
  const albedo = new Uint8Array(W * H * 4)
  const filled = new Uint8Array(W * H)
  const dec = (v: number): number => v / 127.5 - 1
  const votes = new Int32Array(N_SLICE)
  let heightSum = 0
  let heightWeight = 0

  // Per-texel resolve of the tile, kept flat so the height field can be
  // filtered before anything is written into the atlas.
  const detail = new Float32Array(TP * TP * 3) // mean leaflet normal
  const height = new Float32Array(TP * TP) // visible surface height (m)
  const covTex = new Float32Array(TP * TP) // subsample coverage 0..1
  const winTex = new Uint8Array(TP * TP)
  const rgbTex = new Float32Array(TP * TP * 3)
  let heightMean = 0
  let heightCount = 0

  for (let y = 0; y < TP; y++) {
    for (let x = 0; x < TP; x++) {
      let cov = 0
      let r = 0
      let g = 0
      let b = 0
      let nx = 0
      let ny = 0
      let nz = 0
      let dSum = 0
      votes.fill(0)
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const sx = x * SS + i
          const sy = y * SS + j
          for (let k = 0; k < N_SLICE; k++) {
            const rect = rects[k]!
            const s = ((rect.y * SS + sy) * BIG_W + rect.x * SS + sx) * 4
            if (bigAlb[s + 3] === 0) continue
            cov++
            r += bigAlb[s]!
            g += bigAlb[s + 1]!
            b += bigAlb[s + 2]!
            nx += dec(bigNrm[s]!)
            ny += dec(bigNrm[s + 1]!)
            nz += dec(bigNrm[s + 2]!)
            dSum += bigNrm[s + 3]! / 255 // normalized depth, 0 = tile top
            votes[k]! += 1
            break // front-most covered subsample wins (slice 0 is the top)
          }
        }
      }
      let winner = 0
      for (let k = 1; k < N_SLICE; k++) if (votes[k]! > votes[winner]!) winner = k

      const t = y * TP + x
      winTex[t] = winner
      covTex[t] = cov / (SS * SS)
      if (cov > 0) {
        rgbTex[t * 3] = r / cov
        rgbTex[t * 3 + 1] = g / cov
        rgbTex[t * 3 + 2] = b / cov
        detail[t * 3] = nx / cov
        detail[t * 3 + 1] = ny / cov
        detail[t * 3 + 2] = nz / cov
        const h = y1 - (dSum / cov) * (y1 - y0)
        height[t] = h
        heightMean += h
        heightCount++
        // The far card's height: what the visible surface actually sits at,
        // weighted by how much of the tile each slice ended up owning.
        heightSum += sliceH[winner]! * cov
        heightWeight += cov
      } else {
        height[t] = Number.NaN
      }
    }
  }
  heightMean = heightCount > 0 ? heightMean / heightCount : 0
  for (let i = 0; i < TP * TP; i++) if (Number.isNaN(height[i]!)) height[i] = heightMean

  // --- cushion normals from the height field --------------------------------
  // The tile is PERIODIC, so the filter wraps over the tile square (the baked
  // margin is real overhang from the neighbouring copy, i.e. already the wrap).
  // Wrapping rather than clamping is what keeps the shading continuous across
  // every tile boundary of the mat.
  const M = CARPET_MARGIN
  const P = TP - 2 * M
  const wrap = (i: number): number => M + (((i - M) % P) + P) % P
  const smooth = boxBlurWrapped(height, TP, M, P, HEIGHT_BLUR)
  const mPerTexel = tileM / P
  const normal = new Float32Array(TP * TP * 3)
  let tiltSum = 0
  for (let y = 0; y < TP; y++) {
    for (let x = 0; x < TP; x++) {
      const t = y * TP + x
      const dhdx =
        (smooth[y * TP + wrap(x + HEIGHT_DIFF)]! - smooth[y * TP + wrap(x - HEIGHT_DIFF)]!) /
        (2 * HEIGHT_DIFF * mPerTexel)
      // Framebuffer row +1 is +Z: the bake's V axis is -Z and WebGPU flips y.
      const dhdz =
        (smooth[wrap(y + HEIGHT_DIFF) * TP + x]! - smooth[wrap(y - HEIGHT_DIFF) * TP + x]!) /
        (2 * HEIGHT_DIFF * mPerTexel)
      // Slopes add: for any height field n.xz / n.y is exactly -grad(h), so the
      // leaflet tilt and the cushion tilt compose by summing their gradients.
      const dy = Math.max(detail[t * 3 + 1]!, 0.25)
      let sx = detail[t * 3]! / dy
      let sz = detail[t * 3 + 2]! / dy
      const dl = Math.hypot(sx, sz)
      if (dl > SLOPE_CLAMP) {
        sx *= SLOPE_CLAMP / dl
        sz *= SLOPE_CLAMP / dl
      }
      let hx = -dhdx
      let hz = -dhdz
      const hl = Math.hypot(hx, hz)
      if (hl > SLOPE_CLAMP) {
        hx *= SLOPE_CLAMP / hl
        hz *= SLOPE_CLAMP / hl
      }
      tiltSum += Math.atan(hl)
      const gx = sx + hx
      const gz = sz + hz
      const inv = 1 / Math.hypot(gx, 1, gz)
      normal[t * 3] = gx * inv
      normal[t * 3 + 1] = inv
      normal[t * 3 + 2] = gz * inv
    }
  }
  console.log(
    `[024-slice-prisms] cushion normals: mean |grad| tilt ${((tiltSum / (TP * TP)) * 180) / Math.PI} deg`,
  )

  // --- write the atlas ------------------------------------------------------
  // Colour and coverage are per-owner; the NORMAL goes to every tile at the
  // same texel, because all six tiles show the same physical surface there.
  // That also means the normal map needs no dilation and has no tile seams.
  for (let y = 0; y < TP; y++) {
    for (let x = 0; x < TP; x++) {
      const t = y * TP + x
      const cov = covTex[t]!
      for (let k = 0; k <= N_SLICE; k++) {
        const rect = rects[k]!
        const idx = (rect.y + y) * W + rect.x + x
        const take = k === N_SLICE || k === winTex[t]
        if (take && cov > 0) {
          albedo[idx * 4] = Math.round(rgbTex[t * 3]!)
          albedo[idx * 4 + 1] = Math.round(rgbTex[t * 3 + 1]!)
          albedo[idx * 4 + 2] = Math.round(rgbTex[t * 3 + 2]!)
          albedo[idx * 4 + 3] = Math.round(cov * 255)
          filled[idx] = 1
        } else {
          albedo[idx * 4 + 3] = 0
        }
      }
    }
  }

  // Dilate colour into empty texels (alpha stays 0) so filtering never blends
  // toward background black. Clamped to each tile's own rect.
  let cur = filled
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = cur.slice()
    for (const rect of rects) {
      const rx0 = rect.x
      const ry0 = rect.y
      const rx1 = rect.x + TP - 1
      const ry1 = rect.y + TP - 1
      for (let y = ry0; y <= ry1; y++) {
        for (let x = rx0; x <= rx1; x++) {
          const idx = y * W + x
          if (cur[idx]! !== 0) continue
          let count = 0
          let r = 0
          let g = 0
          let b = 0
          for (let j = -1; j <= 1; j++) {
            const yy = y + j
            if (yy < ry0 || yy > ry1) continue
            for (let i = -1; i <= 1; i++) {
              if (i === 0 && j === 0) continue
              const xx = x + i
              if (xx < rx0 || xx > rx1) continue
              const nIdx = yy * W + xx
              if (cur[nIdx]! === 0) continue
              count++
              const s4 = nIdx * 4
              r += albedo[s4]!
              g += albedo[s4 + 1]!
              b += albedo[s4 + 2]!
            }
          }
          if (count === 0) continue
          const d4 = idx * 4
          albedo[d4] = Math.round(r / count)
          albedo[d4 + 1] = Math.round(g / count)
          albedo[d4 + 2] = Math.round(b / count)
          next[idx] = 1
        }
      }
    }
    cur = next
  }

  // Plain tangent storage, upper hemisphere — linear, so the mip chain
  // averages what it should (see the file header).
  const normalOct = new Uint8Array(W * H * 2)
  for (let y = 0; y < TP; y++) {
    for (let x = 0; x < TP; x++) {
      const t = y * TP + x
      const ex = Math.max(0, Math.min(255, Math.round((normal[t * 3]! * 0.5 + 0.5) * 255)))
      const ez = Math.max(0, Math.min(255, Math.round((normal[t * 3 + 2]! * 0.5 + 0.5) * 255)))
      for (const rect of rects) {
        const idx = (rect.y + y) * W + rect.x + x
        normalOct[idx * 2] = ex
        normalOct[idx * 2 + 1] = ez
      }
    }
  }
  return {
    albedo,
    normalOct,
    mergedH: heightWeight > 0 ? heightSum / heightWeight : sliceH[0]!,
  }
}

/** Separable box blur over the tile square, wrapping on its period. */
function boxBlurWrapped(src: Float32Array, tp: number, margin: number, period: number, radius: number): Float32Array {
  const wrap = (i: number): number => margin + (((i - margin) % period) + period) % period
  const inv = 1 / (2 * radius + 1)
  const tmp = new Float32Array(tp * tp)
  for (let y = 0; y < tp; y++) {
    for (let x = 0; x < tp; x++) {
      let s = 0
      for (let i = -radius; i <= radius; i++) s += src[y * tp + wrap(x + i)]!
      tmp[y * tp + x] = s * inv
    }
  }
  const out = new Float32Array(tp * tp)
  for (let y = 0; y < tp; y++) {
    for (let x = 0; x < tp; x++) {
      let s = 0
      for (let j = -radius; j <= radius; j++) s += tmp[wrap(y + j) * tp + x]!
      out[y * tp + x] = s * inv
    }
  }
  return out
}
