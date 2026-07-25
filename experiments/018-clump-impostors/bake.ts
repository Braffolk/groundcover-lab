import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Clump-impostor bake.
 *
 * The source mesh is first cut into STRANDS (whole blades/culms). The
 * generators emit one blade at a time, so a strand is exactly a maximal run of
 * consecutive vertices with no big positional jump — verified on all three
 * meshes (calamagrostis 158, elymus 851, poa 221 strands; < 0.3% of triangles
 * straddle a run, and those follow their first vertex). Cutting on strand
 * boundaries rather than triangles is what makes the split invisible: no blade
 * is ever sliced in half between two cards.
 *
 * Strands are then k-means clustered (weighted by vertex count, deterministic
 * seeding) into K_SUB spatial sub-clumps. Each sub-clump plus one MERGED unit
 * (the whole plant, for the distance LOD) is captured orthographically from 8
 * azimuths + straight down into one atlas. After the coverage-weighted
 * downsample every tile is scanned for its exact alpha bounding box, and that
 * TIGHT rect is what the runtime quad spans — no empty margin is ever
 * rasterized (the merged card benefits from this too).
 *
 * Artifact layout (little-endian), header 1024B:
 *   u32 magic 'CLM1', u32 version, u32 atlasW, atlasH, nrmW, nrmH, kSub, nViews
 *   f32[K_SUB+1][4] unit box: cx, cz, rXZ, y0   (mesh frame, metres, scale 1)
 *   f32[K_SUB+1][4] unit ext: y1, 0, 0, 0
 *   f32[(K_SUB+1)*9][4] tight rect per tile: lx0, ly0, lx1, ly1 in [0,1]
 *   u8[...]              albedo rgba8, ALL mip levels concatenated
 *   u8[nrmW*nrmH*2]      oct-encoded mesh-frame normals (half resolution)
 *
 * The albedo mip chain is built here on the CPU rather than on the GPU because
 * every level needs COVERAGE-PRESERVING alpha: splitting a plant into K
 * sub-clumps splits each texel's partial coverage K ways, and a hard alpha test
 * is non-linear, so two half-covered sub-clump texels both vanish where the one
 * merged texel survived. Every tile therefore gets its true geometric coverage
 * measured from the supersampled capture, and every level's alpha is rescaled
 * (histogram quantile -> one scalar) so the fraction of texels passing the
 * alpha reference matches that truth. Sub-clump and merged cards then have the
 * same density, the LOD switch stops popping, and grass no longer thins out
 * with distance — all without a single dithered pixel.
 */

export const K_SUB = 4
export const MERGED = K_SUB
export const N_UNITS = K_SUB + 1
export const N_SIDE = 8
export const N_VIEWS = N_SIDE + 1
export const N_TILES = N_UNITS * N_VIEWS

// Atlas layout. Every tile origin and size is a multiple of 32 so a 6-level
// mip chain never mixes two tiles into one texel (2x2 blocks stay inside a
// tile at every level) — no cross-tile bleeding at any distance.
const SUB_W = 160
const SUB_H = 512
const MRG_W = 160
const MRG_H = 256
const TOP_S = 160
export const ATLAS_W = 1280
export const ATLAS_H = 2464 // 4*512 sub sides + 256 merged sides + 160 tops
export const NRM_W = ATLAS_W / 2
export const NRM_H = ATLAS_H / 2
export const ALBEDO_MIPS = 6
export const NORMAL_MIPS = 5

const SS = 2
const BIG_W = ATLAS_W * SS
const BIG_H = ATLAS_H * SS
const MAGIC = 0x324d4c43 // 'CLM2'
const VERSION = 2
const HEADER_BYTES = 1024
const DILATE_PASSES = 4
/** Coverage below this is treated as empty when measuring the tight rect. */
const TIGHT_ALPHA = 6
/** Strand break threshold (m) in vertex order. */
const STRAND_JUMP = 0.06
/**
 * The alpha reference the coverage calibration is tuned for — matches the
 * manifest's alphaRef default. Moving the runtime slider off this value trades
 * some of the density match back.
 */
const CALIBRATION_ALPHA = 0.4
const ALPHA_TH = Math.round(CALIBRATION_ALPHA * 255)

/** Total texels in the albedo mip chain. */
function albedoChainTexels(): number {
  let total = 0
  for (let l = 0; l < ALBEDO_MIPS; l++) total += (ATLAS_W >> l) * (ATLAS_H >> l)
  return total
}

/** Tile rect in atlas pixels for unit `u`, view `v` (v == N_SIDE is the top). */
export function tileRect(u: number, v: number): [number, number, number, number] {
  if (v === N_SIDE) return [u * TOP_S, 2304, TOP_S, TOP_S]
  if (u === MERGED) return [v * MRG_W, 2048, MRG_W, MRG_H]
  return [v * SUB_W, u * SUB_H, SUB_W, SUB_H]
}

export interface ClumpAtlas {
  atlasW: number
  atlasH: number
  nrmW: number
  nrmH: number
  kSub: number
  /** Per unit: cx, cz, rXZ, y0 (mesh frame, metres at scale 1). */
  unitBox: Float32Array
  /** Per unit: y1, 0, 0, 0. */
  unitExt: Float32Array
  /** Per tile (unit*N_VIEWS + view): lx0, ly0, lx1, ly1 in local [0,1]. */
  tight: Float32Array
  /** rgba8 albedo, one entry per mip level (level 0 first). */
  albedoLevels: Uint8Array<ArrayBuffer>[]
  normalOct: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res'>

export function unpackClumps(buf: ArrayBuffer): ClumpAtlas | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const atlasW = u[2]!
  const atlasH = u[3]!
  const nrmW = u[4]!
  const nrmH = u[5]!
  const kSub = u[6]!
  const nViews = u[7]!
  if (atlasW !== ATLAS_W || atlasH !== ATLAS_H || kSub !== K_SUB || nViews !== N_VIEWS) return null
  const albedoBytes = albedoChainTexels() * 4
  const expected = HEADER_BYTES + albedoBytes + nrmW * nrmH * 2
  if (buf.byteLength !== expected) return null
  let o = 32
  const unitBox = new Float32Array(buf.slice(o, o + N_UNITS * 16))
  o += N_UNITS * 16
  const unitExt = new Float32Array(buf.slice(o, o + N_UNITS * 16))
  o += N_UNITS * 16
  const tight = new Float32Array(buf.slice(o, o + N_TILES * 16))
  const albedoLevels: Uint8Array<ArrayBuffer>[] = []
  let ao = HEADER_BYTES
  for (let l = 0; l < ALBEDO_MIPS; l++) {
    const bytes = (ATLAS_W >> l) * (ATLAS_H >> l) * 4
    albedoLevels.push(new Uint8Array(buf, ao, bytes))
    ao += bytes
  }
  return {
    atlasW,
    atlasH,
    nrmW,
    nrmH,
    kSub,
    unitBox,
    unitExt,
    tight,
    albedoLevels,
    normalOct: new Uint8Array(buf, HEADER_BYTES + albedoBytes, nrmW * nrmH * 2),
  }
}

function packClumps(a: ClumpAtlas): ArrayBuffer {
  const albedoBytes = a.albedoLevels.reduce((n, l) => n + l.byteLength, 0)
  const buf = new ArrayBuffer(HEADER_BYTES + albedoBytes + a.normalOct.byteLength)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = a.atlasW
  u[3] = a.atlasH
  u[4] = a.nrmW
  u[5] = a.nrmH
  u[6] = a.kSub
  u[7] = N_VIEWS
  new Float32Array(buf, 32, N_UNITS * 4).set(a.unitBox)
  new Float32Array(buf, 32 + N_UNITS * 16, N_UNITS * 4).set(a.unitExt)
  new Float32Array(buf, 32 + N_UNITS * 32, N_TILES * 4).set(a.tight)
  let o = HEADER_BYTES
  for (const level of a.albedoLevels) {
    new Uint8Array(buf, o, level.byteLength).set(level)
    o += level.byteLength
  }
  new Uint8Array(buf, o, a.normalOct.byteLength).set(a.normalOct)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the clump atlas for a species.
 * The dev server answers missing /mesh/baked files with the SPA index.html at
 * status 200, which can poison bakedArtifact's stores — every result is
 * magic-validated, and a poisoned cache entry is rebaked and repaired.
 */
export async function loadSpeciesClumps(ctx: BakeCtx, speciesId: string): Promise<ClumpAtlas> {
  const key = `clumps-v${VERSION}-${speciesId}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
    return bakeSpeciesClumps(ctx, mesh)
  }

  let buf = await bakedArtifact({ expId: ctx.id, key }, runBake)
  let atlas = unpackClumps(buf)
  if (!atlas) {
    buf = await runBake()
    atlas = unpackClumps(buf)
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
// Strand segmentation + clustering
// ---------------------------------------------------------------------------

interface Split {
  /** Cluster index per vertex (0..K_SUB-1). */
  clusterOfVertex: Int32Array
  /** Reordered triangle indices, cluster-major. */
  indices: Uint32Array<ArrayBuffer>
  /** Index range per cluster. */
  first: number[]
  count: number[]
  strands: number
}

/**
 * Cut the mesh into whole-blade strands, then k-means the strands into K_SUB
 * spatial clumps. Deterministic: seeds are fixed angular positions, no RNG.
 */
function splitIntoClumps(mesh: GcMesh): Split {
  const { vertexCount, triangleCount, boundsMin, boundsMax } = mesh.header
  const verts = mesh.vertices
  const sx = (boundsMax[0] - boundsMin[0]) / 65535
  const sy = (boundsMax[1] - boundsMin[1]) / 65535
  const sz = (boundsMax[2] - boundsMin[2]) / 65535

  // --- strands: maximal runs of consecutive vertices without a big jump -----
  const strandOfVertex = new Int32Array(vertexCount)
  let strands = 0
  let px = 0
  let py = 0
  let pz = 0
  for (let i = 0; i < vertexCount; i++) {
    const x = boundsMin[0] + verts[i * 8]! * sx
    const y = boundsMin[1] + verts[i * 8 + 1]! * sy
    const z = boundsMin[2] + verts[i * 8 + 2]! * sz
    if (i === 0 || Math.hypot(x - px, y - py, z - pz) > STRAND_JUMP) strands++
    strandOfVertex[i] = strands - 1
    px = x
    py = y
    pz = z
  }

  // --- strand centroids (vertex-count weighted) -----------------------------
  const sumX = new Float64Array(strands)
  const sumZ = new Float64Array(strands)
  const weight = new Float64Array(strands)
  for (let i = 0; i < vertexCount; i++) {
    const s = strandOfVertex[i]!
    sumX[s] = sumX[s]! + boundsMin[0] + verts[i * 8]! * sx
    sumZ[s] = sumZ[s]! + boundsMin[2] + verts[i * 8 + 2]! * sz
    weight[s] = weight[s]! + 1
  }
  const cx = new Float64Array(strands)
  const cz = new Float64Array(strands)
  for (let s = 0; s < strands; s++) {
    const w = Math.max(weight[s]!, 1)
    cx[s] = sumX[s]! / w
    cz[s] = sumZ[s]! / w
  }

  // --- k-means (Lloyd), deterministic angular seeding -----------------------
  let mx = 0
  let mz = 0
  let wTotal = 0
  for (let s = 0; s < strands; s++) {
    mx += cx[s]! * weight[s]!
    mz += cz[s]! * weight[s]!
    wTotal += weight[s]!
  }
  mx /= Math.max(wTotal, 1)
  mz /= Math.max(wTotal, 1)
  let meanR = 0
  for (let s = 0; s < strands; s++) meanR += Math.hypot(cx[s]! - mx, cz[s]! - mz) * weight[s]!
  meanR /= Math.max(wTotal, 1)

  const kx = new Float64Array(K_SUB)
  const kz = new Float64Array(K_SUB)
  for (let k = 0; k < K_SUB; k++) {
    const a = (k * 2 * Math.PI) / K_SUB + Math.PI / 4
    kx[k] = mx + Math.cos(a) * meanR
    kz[k] = mz + Math.sin(a) * meanR
  }
  const clusterOfStrand = new Int32Array(strands)
  for (let iter = 0; iter < 32; iter++) {
    let moved = 0
    for (let s = 0; s < strands; s++) {
      let best = 0
      let bestD = Infinity
      for (let k = 0; k < K_SUB; k++) {
        const d = (cx[s]! - kx[k]!) ** 2 + (cz[s]! - kz[k]!) ** 2
        if (d < bestD) {
          bestD = d
          best = k
        }
      }
      if (clusterOfStrand[s] !== best) moved++
      clusterOfStrand[s] = best
    }
    const ax = new Float64Array(K_SUB)
    const az = new Float64Array(K_SUB)
    const aw = new Float64Array(K_SUB)
    for (let s = 0; s < strands; s++) {
      const k = clusterOfStrand[s]!
      ax[k] = ax[k]! + cx[s]! * weight[s]!
      az[k] = az[k]! + cz[s]! * weight[s]!
      aw[k] = aw[k]! + weight[s]!
    }
    for (let k = 0; k < K_SUB; k++) {
      if (aw[k]! > 0) {
        kx[k] = ax[k]! / aw[k]!
        kz[k] = az[k]! / aw[k]!
      } else {
        // Empty cluster: reseed on the strand farthest from its own center.
        let far = 0
        let farD = -1
        for (let s = 0; s < strands; s++) {
          const c = clusterOfStrand[s]!
          const d = (cx[s]! - kx[c]!) ** 2 + (cz[s]! - kz[c]!) ** 2
          if (d > farD) {
            farD = d
            far = s
          }
        }
        kx[k] = cx[far]!
        kz[k] = cz[far]!
      }
    }
    if (moved === 0 && iter > 0) break
  }

  const clusterOfVertex = new Int32Array(vertexCount)
  for (let i = 0; i < vertexCount; i++) clusterOfVertex[i] = clusterOfStrand[strandOfVertex[i]!]!

  // --- cluster-major triangle reorder ---------------------------------------
  const tris = mesh.triangles
  const count = new Array<number>(K_SUB).fill(0)
  for (let t = 0; t < triangleCount; t++) count[clusterOfVertex[tris[t * 4]!]!]! += 3
  const first = new Array<number>(K_SUB).fill(0)
  for (let k = 1; k < K_SUB; k++) first[k] = first[k - 1]! + count[k - 1]!
  const cursor = first.slice()
  const indices = new Uint32Array(triangleCount * 3)
  for (let t = 0; t < triangleCount; t++) {
    const a = tris[t * 4]!
    const k = clusterOfVertex[a]!
    const w = cursor[k]!
    indices[w] = a
    indices[w + 1] = tris[t * 4 + 1]!
    indices[w + 2] = tris[t * 4 + 2]!
    cursor[k] = w + 3
  }
  return { clusterOfVertex, indices, first, count, strands }
}

/** Per-unit capture box: bbox center in xz, exact max horizontal radius, y range. */
function unitBoxes(mesh: GcMesh, clusterOfVertex: Int32Array): { box: Float32Array; ext: Float32Array } {
  const { vertexCount, boundsMin, boundsMax } = mesh.header
  const verts = mesh.vertices
  const sx = (boundsMax[0] - boundsMin[0]) / 65535
  const sy = (boundsMax[1] - boundsMin[1]) / 65535
  const sz = (boundsMax[2] - boundsMin[2]) / 65535
  const minX = new Float64Array(N_UNITS).fill(Infinity)
  const maxX = new Float64Array(N_UNITS).fill(-Infinity)
  const minY = new Float64Array(N_UNITS).fill(Infinity)
  const maxY = new Float64Array(N_UNITS).fill(-Infinity)
  const minZ = new Float64Array(N_UNITS).fill(Infinity)
  const maxZ = new Float64Array(N_UNITS).fill(-Infinity)
  const touch = (u: number, x: number, y: number, z: number): void => {
    if (x < minX[u]!) minX[u] = x
    if (x > maxX[u]!) maxX[u] = x
    if (y < minY[u]!) minY[u] = y
    if (y > maxY[u]!) maxY[u] = y
    if (z < minZ[u]!) minZ[u] = z
    if (z > maxZ[u]!) maxZ[u] = z
  }
  for (let i = 0; i < vertexCount; i++) {
    const x = boundsMin[0] + verts[i * 8]! * sx
    const y = boundsMin[1] + verts[i * 8 + 1]! * sy
    const z = boundsMin[2] + verts[i * 8 + 2]! * sz
    touch(clusterOfVertex[i]!, x, y, z)
    touch(MERGED, x, y, z)
  }
  const box = new Float32Array(N_UNITS * 4)
  const ext = new Float32Array(N_UNITS * 4)
  const centerX = new Float64Array(N_UNITS)
  const centerZ = new Float64Array(N_UNITS)
  for (let u = 0; u < N_UNITS; u++) {
    centerX[u] = (minX[u]! + maxX[u]!) / 2
    centerZ[u] = (minZ[u]! + maxZ[u]!) / 2
  }
  const r2 = new Float64Array(N_UNITS)
  for (let i = 0; i < vertexCount; i++) {
    const x = boundsMin[0] + verts[i * 8]! * sx
    const z = boundsMin[2] + verts[i * 8 + 2]! * sz
    const u = clusterOfVertex[i]!
    const du = (x - centerX[u]!) ** 2 + (z - centerZ[u]!) ** 2
    if (du > r2[u]!) r2[u] = du
    const dm = (x - centerX[MERGED]!) ** 2 + (z - centerZ[MERGED]!) ** 2
    if (dm > r2[MERGED]!) r2[MERGED] = dm
  }
  for (let u = 0; u < N_UNITS; u++) {
    box[u * 4] = centerX[u]!
    box[u * 4 + 1] = centerZ[u]!
    box[u * 4 + 2] = Math.sqrt(r2[u]!) * 1.02 + 1e-3
    box[u * 4 + 3] = Math.min(0, minY[u]!)
    ext[u * 4] = maxY[u]!
  }
  return { box, ext }
}

// ---------------------------------------------------------------------------
// GPU capture
// ---------------------------------------------------------------------------

async function bakeSpeciesClumps(ctx: BakeCtx, mesh: GcMesh): Promise<ArrayBuffer> {
  const { device } = ctx
  const hdr = mesh.header
  const split = splitIntoClumps(mesh)
  const { box, ext } = unitBoxes(mesh, split.clusterOfVertex)
  console.info(
    `[${ctx.id}] split ${hdr.triangleCount} tris into ${split.strands} strands / ${K_SUB} clumps` +
      ` (tris ${split.count.map((c) => c / 3).join('/')})`,
  )

  const verts = mesh.vertices
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(vbuf, 0, verts)
  const ibuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-idx`,
      size: split.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'bake-scratch' },
  )
  device.queue.writeBuffer(ibuf, 0, split.indices)

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
  const albTex = mkTarget('bake-albedo')
  const nrmTex = mkTarget('bake-normal')
  const depthTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-depth`,
      size: [BIG_W, BIG_H],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    },
    { tag: 'bake-scratch' },
  )

  // One 256B-strided uniform slot per (unit, view).
  const STRIDE = 256
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * N_TILES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'bake-scratch' },
  )
  const scratch = new Float32Array((STRIDE / 4) * N_TILES)
  for (let u = 0; u < N_UNITS; u++) {
    for (let v = 0; v < N_VIEWS; v++) {
      const o = ((u * N_VIEWS + v) * STRIDE) / 4
      const isTop = v === N_SIDE
      if (isTop) {
        scratch.set([1, 0, 0], o) // card U axis = +X
        scratch.set([0, 0, -1], o + 4) // card V axis = -Z
        scratch.set([0, 1, 0, 1], o + 8) // toward camera = +Y, w = is_top
      } else {
        const a = (v * 2 * Math.PI) / N_SIDE
        scratch.set([Math.cos(a), 0, -Math.sin(a)], o)
        scratch.set([0, 1, 0], o + 4)
        scratch.set([Math.sin(a), 0, Math.cos(a), 0], o + 8)
      }
      scratch.set([box[u * 4]!, box[u * 4 + 1]!, box[u * 4 + 3]!, ext[u * 4]!], o + 12)
      scratch.set([box[u * 4 + 2]!, 0, 0, 0], o + 16)
      scratch.set([hdr.boundsMin[0], hdr.boundsMin[1], hdr.boundsMin[2], 0], o + 20)
      scratch.set(
        [
          hdr.boundsMax[0] - hdr.boundsMin[0],
          hdr.boundsMax[1] - hdr.boundsMin[1],
          hdr.boundsMax[2] - hdr.boundsMin[2],
          0,
        ],
        o + 24,
      )
    }
  }
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

  // One submission per unit — 45 tiles of a multi-million-triangle mesh in a
  // single command buffer is a watchdog risk on weaker GPUs.
  const albView = albTex.createView()
  const nrmView = nrmTex.createView()
  const depthView = depthTex.createView()
  for (let u = 0; u < N_UNITS; u++) {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc-${u}` })
    const load = u === 0 ? 'clear' : 'load'
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/bake-pass-${u}`,
      colorAttachments: [
        { view: albView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: load, storeOp: 'store' },
        { view: nrmView, clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0 }, loadOp: load, storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: load,
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    const first = u === MERGED ? 0 : split.first[u]!
    const count = u === MERGED ? hdr.triangleCount * 3 : split.count[u]!
    for (let v = 0; v < N_VIEWS; v++) {
      const [tx, ty, tw, th] = tileRect(u, v)
      pass.setViewport(tx * SS, ty * SS, tw * SS, th * SS, 0, 1)
      pass.setScissorRect(tx * SS, ty * SS, tw * SS, th * SS)
      pass.setBindGroup(0, bg, [(u * N_VIEWS + v) * STRIDE])
      if (count > 0) pass.drawIndexed(count, 1, first, 0, 0)
    }
    pass.end()
    device.queue.submit([enc.finish()])
  }
  depthTex.destroy()
  vbuf.destroy()
  ibuf.destroy()

  const bigAlb = await readback(ctx, albTex, 'albedo')
  albTex.destroy()
  const bigNrm = await readback(ctx, nrmTex, 'normal')
  nrmTex.destroy()

  const { albedoLevels, normalOct, tight } = postProcess(bigAlb, bigNrm)
  return packClumps({
    atlasW: ATLAS_W,
    atlasH: ATLAS_H,
    nrmW: NRM_W,
    nrmH: NRM_H,
    kSub: K_SUB,
    unitBox: box,
    unitExt: ext,
    tight,
    albedoLevels,
    normalOct,
  })
}

async function readback(ctx: BakeCtx, tex: GPUTexture, label: string): Promise<Uint8Array> {
  const bpr = BIG_W * 4
  const rb = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-rb-${label}`, size: bpr * BIG_H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
    { tag: 'bake-scratch' },
  )
  const enc = ctx.device.createCommandEncoder({ label: `${ctx.id}/bake-copy-${label}` })
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: BIG_H }, [BIG_W, BIG_H])
  ctx.device.queue.submit([enc.finish()])
  await rb.mapAsync(GPUMapMode.READ)
  const out = new Uint8Array(rb.getMappedRange()).slice()
  rb.unmap()
  rb.destroy()
  return out
}

// ---------------------------------------------------------------------------
// Post-process: downsample, tile-clamped dilation, tight rects, oct encode
// ---------------------------------------------------------------------------

function postProcess(
  bigAlb: Uint8Array,
  bigNrm: Uint8Array,
): { albedoLevels: Uint8Array<ArrayBuffer>[]; normalOct: Uint8Array<ArrayBuffer>; tight: Float32Array } {
  // Level 0: coverage-weighted 2x downsample. rgb and the RAW (uncalibrated)
  // alpha are kept apart — the mip chain is built from raw alpha, and only the
  // emitted levels get the coverage calibration applied.
  const rgb0 = new Uint8Array(ATLAS_W * ATLAS_H * 3)
  const alpha0 = new Float32Array(ATLAS_W * ATLAS_H)
  for (let y = 0; y < ATLAS_H; y++) {
    for (let x = 0; x < ATLAS_W; x++) {
      let aSum = 0
      let r = 0
      let g = 0
      let b = 0
      for (let j = 0; j < SS; j++) {
        for (let i = 0; i < SS; i++) {
          const s = ((y * SS + j) * BIG_W + (x * SS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          aSum += a
          r += bigAlb[s]! * a
          g += bigAlb[s + 1]! * a
          b += bigAlb[s + 2]! * a
        }
      }
      const d = (y * ATLAS_W + x) * 3
      if (aSum > 0) {
        rgb0[d] = Math.round(r / aSum)
        rgb0[d + 1] = Math.round(g / aSum)
        rgb0[d + 2] = Math.round(b / aSum)
        alpha0[y * ATLAS_W + x] = aSum / (SS * SS)
      }
    }
  }

  // Normals straight from the supersampled buffer to half atlas resolution
  // (4x4 coverage-weighted box) — lighting tolerates it, coverage does not.
  const nrm = new Float32Array(NRM_W * NRM_H * 3)
  const nrmFilled = new Uint8Array(NRM_W * NRM_H)
  const NS = SS * 2
  for (let y = 0; y < NRM_H; y++) {
    for (let x = 0; x < NRM_W; x++) {
      let w = 0
      let nx = 0
      let ny = 0
      let nz = 0
      for (let j = 0; j < NS; j++) {
        for (let i = 0; i < NS; i++) {
          const s = ((y * NS + j) * BIG_W + (x * NS + i)) * 4
          const a = bigAlb[s + 3]!
          if (a === 0) continue
          w += a
          nx += (bigNrm[s]! / 127.5 - 1) * a
          ny += (bigNrm[s + 1]! / 127.5 - 1) * a
          nz += (bigNrm[s + 2]! / 127.5 - 1) * a
        }
      }
      if (w > 0) {
        const d = (y * NRM_W + x) * 3
        nrm[d] = nx
        nrm[d + 1] = ny
        nrm[d + 2] = nz
        nrmFilled[y * NRM_W + x] = 1
      }
    }
  }

  // Per-tile: exact geometric coverage from the supersampled capture, the
  // tight alpha rect, and dilation clamped to the tile so filtering never
  // pulls a neighbouring tile's colour in.
  const tight = new Float32Array(N_TILES * 4)
  const trueCov = new Float32Array(N_TILES)
  for (let u = 0; u < N_UNITS; u++) {
    for (let v = 0; v < N_VIEWS; v++) {
      const [tx, ty, tw, th] = tileRect(u, v)
      const t = u * N_VIEWS + v
      let covered = 0
      for (let y = 0; y < th * SS; y++) {
        const row = (ty * SS + y) * BIG_W + tx * SS
        for (let x = 0; x < tw * SS; x++) {
          if (bigAlb[(row + x) * 4 + 3]! > 0) covered++
        }
      }
      trueCov[t] = covered / (tw * th * SS * SS)

      let x0 = tw
      let x1 = -1
      let y0 = th
      let y1 = -1
      for (let y = 0; y < th; y++) {
        for (let x = 0; x < tw; x++) {
          if (alpha0[(ty + y) * ATLAS_W + tx + x]! < TIGHT_ALPHA) continue
          if (x < x0) x0 = x
          if (x > x1) x1 = x
          if (y < y0) y0 = y
          if (y > y1) y1 = y
        }
      }
      if (x1 < 0) {
        tight.fill(0, t * 4, t * 4 + 4)
      } else {
        // 2 texels of slack so bilinear taps stay inside real coverage, and at
        // least 1 texel inside the tile so they never reach the neighbour.
        x0 = Math.max(1, x0 - 2)
        y0 = Math.max(1, y0 - 2)
        x1 = Math.min(tw - 2, x1 + 2)
        y1 = Math.min(th - 2, y1 + 2)
        tight[t * 4] = x0 / tw
        tight[t * 4 + 1] = y0 / th
        tight[t * 4 + 2] = (x1 + 1) / tw
        tight[t * 4 + 3] = (y1 + 1) / th
      }
      dilateRect(rgb0, alpha0, ATLAS_W, tx, ty, tw, th)
      dilateNormals(nrm, nrmFilled, tx / 2, ty / 2, tw / 2, th / 2)
    }
  }

  const albedoLevels = buildAlbedoChain(rgb0, alpha0, trueCov)

  const normalOct = new Uint8Array(NRM_W * NRM_H * 2)
  for (let i = 0; i < NRM_W * NRM_H; i++) {
    const s = i * 3
    const e = octEncode(nrm[s]!, nrm[s + 1]!, nrm[s + 2]!)
    normalOct[i * 2] = e[0]
    normalOct[i * 2 + 1] = e[1]
  }
  return { albedoLevels, normalOct, tight }
}

/**
 * Build the rgba8 albedo mip chain, per tile, with coverage-preserving alpha.
 *
 * Colour halves with a coverage-weighted box (falling back to a plain mean in
 * fully transparent 2x2 blocks so the dilated colour keeps spreading outward).
 * Alpha halves with a plain mean, and each emitted level is then multiplied by
 * ONE scalar per tile, chosen so that the number of texels at or above the
 * alpha reference equals the tile's true geometric coverage. The scalar is a
 * histogram quantile, not a search: find the alpha level whose tail holds
 * exactly `trueCov * area` texels and scale that level onto the reference.
 */
function buildAlbedoChain(rgb0: Uint8Array, alpha0: Float32Array, trueCov: Float32Array): Uint8Array<ArrayBuffer>[] {
  const levels: Uint8Array<ArrayBuffer>[] = []
  let rgb = rgb0
  let alpha = alpha0
  for (let l = 0; l < ALBEDO_MIPS; l++) {
    const lw = ATLAS_W >> l
    const lh = ATLAS_H >> l
    if (l > 0) {
      const pw = ATLAS_W >> (l - 1)
      const nextRgb = new Uint8Array(lw * lh * 3)
      const nextAlpha = new Float32Array(lw * lh)
      for (let y = 0; y < lh; y++) {
        for (let x = 0; x < lw; x++) {
          let aSum = 0
          let r = 0
          let g = 0
          let b = 0
          let rp = 0
          let gp = 0
          let bp = 0
          for (let j = 0; j < 2; j++) {
            for (let i = 0; i < 2; i++) {
              const si = (y * 2 + j) * pw + x * 2 + i
              const a = alpha[si]!
              rp += rgb[si * 3]!
              gp += rgb[si * 3 + 1]!
              bp += rgb[si * 3 + 2]!
              if (a <= 0) continue
              aSum += a
              r += rgb[si * 3]! * a
              g += rgb[si * 3 + 1]! * a
              b += rgb[si * 3 + 2]! * a
            }
          }
          const d = (y * lw + x) * 3
          if (aSum > 0) {
            nextRgb[d] = Math.round(r / aSum)
            nextRgb[d + 1] = Math.round(g / aSum)
            nextRgb[d + 2] = Math.round(b / aSum)
            nextAlpha[y * lw + x] = aSum / 4
          } else {
            nextRgb[d] = Math.round(rp / 4)
            nextRgb[d + 1] = Math.round(gp / 4)
            nextRgb[d + 2] = Math.round(bp / 4)
          }
        }
      }
      rgb = nextRgb
      alpha = nextAlpha
    }

    const out = new Uint8Array(lw * lh * 4)
    const hist = new Int32Array(257)
    for (let u = 0; u < N_UNITS; u++) {
      for (let v = 0; v < N_VIEWS; v++) {
        const t = u * N_VIEWS + v
        const [tx0, ty0, tw0, th0] = tileRect(u, v)
        const tx = tx0 >> l
        const ty = ty0 >> l
        const tw = tw0 >> l
        const th = th0 >> l
        hist.fill(0)
        for (let y = 0; y < th; y++) {
          for (let x = 0; x < tw; x++) {
            const bin = Math.min(256, Math.round(alpha[(ty + y) * lw + tx + x]!))
            hist[bin] = hist[bin]! + 1
          }
        }
        const target = Math.round(trueCov[t]! * tw * th)
        let acc = 0
        let aStar = 1
        for (let a = 256; a >= 1; a--) {
          acc += hist[a]!
          if (acc >= target) {
            aStar = a
            break
          }
        }
        const k = target <= 0 ? 1 : Math.min(4, Math.max(0.6, ALPHA_TH / aStar))
        for (let y = 0; y < th; y++) {
          for (let x = 0; x < tw; x++) {
            const si = (ty + y) * lw + tx + x
            const d = si * 4
            out[d] = rgb[si * 3]!
            out[d + 1] = rgb[si * 3 + 1]!
            out[d + 2] = rgb[si * 3 + 2]!
            out[d + 3] = Math.min(255, Math.round(alpha[si]! * k))
          }
        }
      }
    }
    levels.push(out)
  }
  return levels
}

/** Push colour (never coverage) outward into empty texels inside one tile. */
function dilateRect(
  rgb: Uint8Array,
  alpha: Float32Array,
  stride: number,
  tx: number,
  ty: number,
  tw: number,
  th: number,
): void {
  let filled = new Uint8Array(tw * th)
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      if (alpha[(ty + y) * stride + tx + x]! > 0) filled[y * tw + x] = 1
    }
  }
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = filled.slice()
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        if (filled[y * tw + x]! !== 0) continue
        let n = 0
        let r = 0
        let g = 0
        let b = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= th) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < 0 || xx >= tw) continue
            if (filled[yy * tw + xx]! === 0) continue
            const s = ((ty + yy) * stride + tx + xx) * 3
            n++
            r += rgb[s]!
            g += rgb[s + 1]!
            b += rgb[s + 2]!
          }
        }
        if (n === 0) continue
        const d = ((ty + y) * stride + tx + x) * 3
        rgb[d] = Math.round(r / n)
        rgb[d + 1] = Math.round(g / n)
        rgb[d + 2] = Math.round(b / n)
        next[y * tw + x] = 1
      }
    }
    filled = next
  }
}

function dilateNormals(nrm: Float32Array, filledAtlas: Uint8Array, tx: number, ty: number, tw: number, th: number): void {
  let filled = new Uint8Array(tw * th)
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) filled[y * tw + x] = filledAtlas[(ty + y) * NRM_W + tx + x]!
  }
  for (let p = 0; p < DILATE_PASSES; p++) {
    const next = filled.slice()
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        if (filled[y * tw + x]! !== 0) continue
        let n = 0
        let nx = 0
        let ny = 0
        let nz = 0
        for (let j = -1; j <= 1; j++) {
          const yy = y + j
          if (yy < 0 || yy >= th) continue
          for (let i = -1; i <= 1; i++) {
            const xx = x + i
            if ((i === 0 && j === 0) || xx < 0 || xx >= tw) continue
            if (filled[yy * tw + xx]! === 0) continue
            const s = ((ty + yy) * NRM_W + tx + xx) * 3
            n++
            nx += nrm[s]!
            ny += nrm[s + 1]!
            nz += nrm[s + 2]!
          }
        }
        if (n === 0) continue
        const d = ((ty + y) * NRM_W + tx + x) * 3
        nrm[d] = nx / n
        nrm[d + 1] = ny / n
        nrm[d + 2] = nz / n
        next[y * tw + x] = 1
      }
    }
    filled = next
  }
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) filledAtlas[(ty + y) * NRM_W + tx + x] = filled[y * tw + x]!
  }
}

/** Octahedral encode, y-primary — exact inverse of the decode in the shader. */
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
