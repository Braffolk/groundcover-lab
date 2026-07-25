import bakeShaderSrc from './shaders/bake.wgsl'
import { bakedArtifact, commitBake, type ExperimentContext, type GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake one species into the factored light-field representation:
 *
 *  1. GPU: a GRIDxGRID hemi-octahedral grid of orthographic DEPTH+COVERAGE
 *     captures. Each view is rasterized 2x2 supersampled into a one-row strip,
 *     then downsampled to its final TILE px tile storing min-depth (8-bit,
 *     1.0 = miss sentinel) and fractional coverage in one rg8unorm 3072x3072
 *     atlas (18.9MB). Fractional coverage is what keeps sub-texel fluff
 *     (calamagrostis seed heads) wispy instead of ballooning to its convex
 *     hull. Work is chunked into several submits so huge meshes (the 19.8M-tri
 *     Sphagnum tiles) never risk a device timeout.
 *  2. CPU: view-INDEPENDENT appearance volumes — every source vertex (2M-11.5M
 *     of them; denser than the voxel grid) is splatted into a <=96^3
 *     aspect-fit voxel grid, averaged, then dilated so any quantized hit
 *     point reconstructed at runtime lands on valid albedo/normal data.
 *     albedo.a stores per-voxel luminance sigma, re-injected at runtime as
 *     deterministic speckle so voxel-averaged interiors aren't flat.
 *
 * Naive 4D storage at the same angular/spatial resolution (albedo+normal+
 * depth+coverage, ~10B/ray) would be ~94MB; the factorization stores ~21MB.
 *
 * The result is packed into ONE gzipped artifact per species and cached /
 * committed through the harness bake flow: the Sphagnum meshes are 19.8M tris
 * / 479MB each, so a fresh bake costs minutes and nobody should pay it twice.
 */

/** Atlas is always this square, whatever (grid, tile) split a profile picks. */
export const ATLAS = 3072

/**
 * How a species' light field is split between angular and spatial resolution.
 * grid*tile == ATLAS, so every profile costs the same 18.9MB.
 */
export interface QlfProfile {
  /** Hemi-oct views per axis (grid^2 views). */
  grid: number
  /** Pixels per view tile. */
  tile: number
  /** Cache-key fragment — bump when the bake maths changes. */
  key: string
  /**
   * Commit a fresh bake into mesh/baked/<exp>/. True for the Sphagnum tiles:
   * fetching and parsing 479MB of source mesh per species dwarfs the bake
   * itself, and nobody should pay it on a page load.
   */
  commit: boolean
}

/** Upright plants: fine angular sampling for thin stems seen edge-on. */
export const PROFILE_PLANT: QlfProfile = { grid: 24, tile: 128, key: 'v2-g24t128', commit: false }
/**
 * Carpet tiles (Sphagnum): trade angular for SPATIAL resolution. A cushion is
 * nearly a height field — 3.3cm of relief, no thin stems standing off it — so
 * the parallax between neighbouring views is ~2mm and 144 views are plenty,
 * while 256px tiles put 0.9mm on a texel and finally resolve single capitula.
 * Same 3072 atlas, same 18.9MB.
 */
export const PROFILE_CARPET: QlfProfile = { grid: 12, tile: 256, key: 'v2-g12t256', commit: true }

const SS = 2 // supersampling factor of the bake rasterization
/**
 * Every ortho view is framed on the BOX SUPPORT along its own axes rather than
 * on the bounding sphere, times this margin. The margin buys a few empty texels
 * around the silhouette, so a runtime query that projects outside the frame
 * clamps onto a miss texel instead of smearing the edge outward. The win is
 * large for anything non-cubic: a Sphagnum slab (0.21 x 0.09 x 0.23 m) inside
 * its 0.33 m bounding sphere used to spend 70% of every tile on empty space.
 */
const FRAME_PAD = 1.03
const VOL_MAX = 96
const DILATE_PASSES = 4
/** Cap triangles per GPU submit during the bake (watchdog safety). */
const TRIS_PER_SUBMIT = 6e7
const MAGIC = 0x314c5147 // 'QLF1'
const HEADER_F32 = 24 // 96B header

export interface QlfBaked {
  depthAtlas: GPUTexture
  albedoVol: GPUTexture
  normalVol: GPUTexture
  grid: number
  tile: number
  center: [number, number, number]
  /** Padded bbox half-extents in unit-sphere space — the per-view frame sizes. */
  halfU: [number, number, number]
  radius: number
  bmin: [number, number, number]
  bmax: [number, number, number]
  heightM: number
}

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

/** Hemi-octahedral decode — MUST match hemioct_decode() in qlf.wgsl. */
function hemioctDecode(ex: number, ey: number): V3 {
  const px = (ex + ey) * 0.5
  const pz = (ex - ey) * 0.5
  const y = 1 - Math.abs(px) - Math.abs(pz)
  return norm([px, y, pz])
}

// ---------------------------------------------------------------------------
// CPU half: view-independent appearance volumes from the raw vertices.
// ---------------------------------------------------------------------------

interface Volumes {
  dims: [number, number, number]
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
}

const ACC = 9 // r,g,b, nx,ny,nz, count, lum, lum^2

function bakeVolumes(mesh: GcMesh): Volumes {
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const ext: V3 = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]]
  const maxExt = Math.max(ext[0], ext[1], ext[2])
  const dims = ext.map((e) => Math.max(8, Math.min(VOL_MAX, Math.round((e / maxExt) * VOL_MAX)))) as [
    number,
    number,
    number,
  ]
  const [dx, dy, dz] = dims
  const n = dx * dy * dz

  let acc = new Float32Array(n * ACC)
  const verts = mesh.vertices
  const count = mesh.header.vertexCount
  let sumR = 0
  let sumG = 0
  let sumB = 0
  for (let v = 0; v < count; v++) {
    const b = v * 8
    // Quantized position is already normalized against the bounds — voxel
    // coordinates come straight from the u16s, no decode needed.
    const ix = Math.min(dx - 1, ((verts[b]! / 65535) * dx) | 0)
    const iy = Math.min(dy - 1, ((verts[b + 1]! / 65535) * dy) | 0)
    const iz = Math.min(dz - 1, ((verts[b + 2]! / 65535) * dz) | 0)
    const o = ((iz * dy + iy) * dx + ix) * ACC
    const r = verts[b + 3]! / 65535
    const g = verts[b + 4]! / 65535
    const bl = verts[b + 5]! / 65535
    // Octahedral normal decode (mirrors GcMesh.normalAt, allocation-free).
    const ou = (verts[b + 6]! / 65535) * 2 - 1
    const ov = (verts[b + 7]! / 65535) * 2 - 1
    let nx = ou
    let nz = ov
    const nyRaw = 1 - Math.abs(ou) - Math.abs(ov)
    if (nyRaw < 0) {
      nx = (1 - Math.abs(ov)) * Math.sign(ou)
      nz = (1 - Math.abs(ou)) * Math.sign(ov)
    }
    const nl = Math.hypot(nx, nyRaw, nz) || 1
    const lum = 0.299 * r + 0.587 * g + 0.114 * bl
    acc[o] = acc[o]! + r
    acc[o + 1] = acc[o + 1]! + g
    acc[o + 2] = acc[o + 2]! + bl
    acc[o + 3] = acc[o + 3]! + nx / nl
    acc[o + 4] = acc[o + 4]! + nyRaw / nl
    acc[o + 5] = acc[o + 5]! + nz / nl
    acc[o + 6] = acc[o + 6]! + 1
    acc[o + 7] = acc[o + 7]! + lum
    acc[o + 8] = acc[o + 8]! + lum * lum
    sumR += r
    sumG += g
    sumB += bl
  }
  const avg: V3 = count > 0 ? [sumR / count, sumG / count, sumB / count] : [0.3, 0.4, 0.2]

  // Dilate: fill empty voxels with the mean of occupied 6-neighbours, a few
  // passes, so trilinear samples near quantized hit points never read holes.
  const stride: V3 = [ACC, dx * ACC, dx * dy * ACC]
  for (let pass = 0; pass < DILATE_PASSES; pass++) {
    const next = acc.slice()
    for (let iz = 0; iz < dz; iz++) {
      for (let iy = 0; iy < dy; iy++) {
        for (let ix = 0; ix < dx; ix++) {
          const o = ((iz * dy + iy) * dx + ix) * ACC
          if (acc[o + 6]! > 0) continue
          let k = 0
          const sums = [0, 0, 0, 0, 0, 0, 0, 0] // r,g,b,nx,ny,nz,lum,lum2
          for (let axis = 0; axis < 3; axis++) {
            const idx = axis === 0 ? ix : axis === 1 ? iy : iz
            const lim = axis === 0 ? dx : axis === 1 ? dy : dz
            for (const s of [-1, 1]) {
              if (idx + s < 0 || idx + s >= lim) continue
              const no = o + s * stride[axis]!
              const c = acc[no + 6]!
              if (c <= 0) continue
              for (let f = 0; f < 6; f++) sums[f]! += acc[no + f]! / c
              sums[6]! += acc[no + 7]! / c
              sums[7]! += acc[no + 8]! / c
              k++
            }
          }
          if (k > 0) {
            for (let f = 0; f < 6; f++) next[o + f] = sums[f]!
            next[o + 6] = k
            next[o + 7] = sums[6]!
            next[o + 8] = sums[7]!
          }
        }
      }
    }
    acc = next
  }

  const albedo = new Uint8Array(n * 4)
  const normal = new Uint8Array(n * 4)
  for (let i = 0; i < n; i++) {
    const o = i * ACC
    const c = acc[o + 6]!
    const t = i * 4
    if (c > 0) {
      albedo[t] = Math.min(255, (acc[o]! / c) * 255) | 0
      albedo[t + 1] = Math.min(255, (acc[o + 1]! / c) * 255) | 0
      albedo[t + 2] = Math.min(255, (acc[o + 2]! / c) * 255) | 0
      const meanL = acc[o + 7]! / c
      const sigma = Math.sqrt(Math.max(acc[o + 8]! / c - meanL * meanL, 0))
      albedo[t + 3] = Math.min(255, sigma * 2 * 255) | 0 // a = sigma * 2
      const nx = acc[o + 3]!
      const ny = acc[o + 4]!
      const nz = acc[o + 5]!
      const l = Math.hypot(nx, ny, nz)
      // Averaged two-sided foliage normals can cancel; fall back to up.
      const ux = l > 1e-4 ? nx / l : 0
      const uy = l > 1e-4 ? ny / l : 1
      const uz = l > 1e-4 ? nz / l : 0
      normal[t] = ((ux * 0.5 + 0.5) * 255) | 0
      normal[t + 1] = ((uy * 0.5 + 0.5) * 255) | 0
      normal[t + 2] = ((uz * 0.5 + 0.5) * 255) | 0
      normal[t + 3] = 255
    } else {
      albedo[t] = (avg[0] * 255) | 0
      albedo[t + 1] = (avg[1] * 255) | 0
      albedo[t + 2] = (avg[2] * 255) | 0
      albedo[t + 3] = 50
      normal[t] = 128
      normal[t + 1] = 255
      normal[t + 2] = 128
      normal[t + 3] = 0
    }
  }
  return { dims, albedo, normal }
}

// ---------------------------------------------------------------------------
// Artifact container: header + rg8 atlas + rgba8 albedo/normal volumes, gzipped.
// ---------------------------------------------------------------------------

async function gzip(data: ArrayBuffer): Promise<ArrayBuffer> {
  const s = new Blob([data]).stream().pipeThrough(new CompressionStream('gzip'))
  return await new Response(s).arrayBuffer()
}

async function gunzipIfNeeded(data: ArrayBuffer): Promise<ArrayBuffer> {
  const head = new Uint8Array(data, 0, Math.min(2, data.byteLength))
  if (head[0] !== 0x1f || head[1] !== 0x8b) return data
  const s = new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'))
  return await new Response(s).arrayBuffer()
}

interface Artifact {
  grid: number
  tile: number
  dims: [number, number, number]
  center: V3
  halfU: V3
  radius: number
  bmin: V3
  bmax: V3
  heightM: number
  atlas: Uint8Array<ArrayBuffer>
  albedo: Uint8Array<ArrayBuffer>
  normal: Uint8Array<ArrayBuffer>
}

function packArtifact(a: Artifact): ArrayBuffer {
  const atlasBytes = ATLAS * ATLAS * 2
  const volBytes = a.dims[0] * a.dims[1] * a.dims[2] * 4
  const total = HEADER_F32 * 4 + atlasBytes + volBytes * 2
  const buf = new ArrayBuffer(total)
  const u32 = new Uint32Array(buf, 0, HEADER_F32)
  const f32 = new Float32Array(buf, 0, HEADER_F32)
  u32[0] = MAGIC
  u32[1] = a.grid
  u32[2] = a.tile
  u32[3] = a.dims[0]
  u32[4] = a.dims[1]
  u32[5] = a.dims[2]
  f32[6] = a.radius
  f32[7] = a.heightM
  f32.set(a.center, 8)
  f32.set(a.bmin, 12)
  f32.set(a.bmax, 16)
  f32.set(a.halfU, 20)
  const bytes = new Uint8Array(buf)
  let o = HEADER_F32 * 4
  bytes.set(a.atlas, o)
  o += atlasBytes
  bytes.set(a.albedo, o)
  o += volBytes
  bytes.set(a.normal, o)
  return buf
}

function unpackArtifact(buf: ArrayBuffer): Artifact {
  const u32 = new Uint32Array(buf, 0, HEADER_F32)
  const f32 = new Float32Array(buf, 0, HEADER_F32)
  if (u32[0] !== MAGIC) throw new Error('015: baked artifact has bad magic (stale or HTML fallback)')
  const dims: [number, number, number] = [u32[3]!, u32[4]!, u32[5]!]
  const atlasBytes = ATLAS * ATLAS * 2
  const volBytes = dims[0] * dims[1] * dims[2] * 4
  let o = HEADER_F32 * 4
  const atlas = new Uint8Array(buf, o, atlasBytes)
  o += atlasBytes
  const albedo = new Uint8Array(buf, o, volBytes)
  o += volBytes
  const normal = new Uint8Array(buf, o, volBytes)
  return {
    grid: u32[1]!,
    tile: u32[2]!,
    dims,
    center: [f32[8]!, f32[9]!, f32[10]!],
    halfU: [f32[20]!, f32[21]!, f32[22]!],
    radius: f32[6]!,
    bmin: [f32[12]!, f32[13]!, f32[14]!],
    bmax: [f32[16]!, f32[17]!, f32[18]!],
    heightM: f32[7]!,
    atlas,
    albedo,
    normal,
  }
}

/**
 * Load (cache -> committed file -> fresh bake) and upload one species' light
 * field. A fresh bake that cost real time is committed back into
 * mesh/baked/<exp>/ so neither the owner nor a later run pays it again.
 */
export async function loadOrBakeSpecies(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  profile: QlfProfile,
  loadMesh: () => Promise<GcMesh>,
): Promise<QlfBaked> {
  const key = `qlf-${profile.key}-${speciesId}`
  let bakeMs = 0
  const raw = await bakedArtifact({ expId: ctx.id, key }, async () => {
    const t0 = performance.now()
    const mesh = await loadMesh()
    const packed = packArtifact(await bakeArtifact(ctx, speciesId, mesh, profile))
    bakeMs = performance.now() - t0
    return await gzip(packed)
  })
  const artifact = unpackArtifact(await gunzipIfNeeded(raw))
  // Only the profiles that ask for it (see QlfProfile.commit); the grasses bake
  // in ~1s from a 64MB mesh and stay in the OPFS cache.
  if (bakeMs > 0 && profile.commit) {
    void commitBake(ctx.id, key, raw)
      .then((saved) => console.log(`[${ctx.id}] committed ${saved} (${(bakeMs / 1000).toFixed(0)}s bake)`))
      .catch((e: unknown) => console.warn(`[${ctx.id}] commit failed`, e))
  }
  return uploadBaked(ctx, speciesId, artifact)
}

function uploadBaked(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, a: Artifact): QlfBaked {
  const { device } = ctx
  const depthAtlas = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/depth-lf`,
      size: [ATLAS, ATLAS],
      format: 'rg8unorm', // r = min depth (1 = miss), g = fractional coverage
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { species: speciesId, tag: 'depth-lightfield' },
  )
  device.queue.writeTexture({ texture: depthAtlas }, a.atlas, { bytesPerRow: ATLAS * 2 }, [ATLAS, ATLAS])

  const [dx, dy, dz] = a.dims
  const volTex = (label: string, data: Uint8Array<ArrayBuffer>, tag: string): GPUTexture => {
    const tex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${label}`,
        size: [dx, dy, dz],
        dimension: '3d',
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag },
    )
    device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz])
    return tex
  }
  return {
    depthAtlas,
    albedoVol: volTex('albedo-vol', a.albedo, 'albedo-volume'),
    normalVol: volTex('normal-vol', a.normal, 'normal-volume'),
    grid: a.grid,
    tile: a.tile,
    center: a.center,
    halfU: a.halfU,
    radius: a.radius,
    bmin: a.bmin,
    bmax: a.bmax,
    heightM: a.heightM,
  }
}

// ---------------------------------------------------------------------------
// GPU half: the supersampled ortho depth+coverage light field.
// ---------------------------------------------------------------------------

async function bakeArtifact(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  mesh: GcMesh,
  profile: QlfProfile,
): Promise<Artifact> {
  const { device } = ctx
  const { grid: GRID, tile: TILE } = profile
  const TILE_SS = TILE * SS
  const STRIP_W = GRID * TILE_SS
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  // Padded half-extents (metres) — every view's ortho frame is the support of
  // this box along its own axes, which is what stops an oblate cushion or a
  // thin grass stem from paying for the empty corners of a sphere.
  const halfM: V3 = [
    ((bmax[0] - bmin[0]) / 2) * FRAME_PAD,
    ((bmax[1] - bmin[1]) / 2) * FRAME_PAD,
    ((bmax[2] - bmin[2]) / 2) * FRAME_PAD,
  ]
  const halfU: V3 = [halfM[0] / radius, halfM[1] / radius, halfM[2] / radius]
  const support = (a: V3): number => Math.abs(a[0]) * halfM[0] + Math.abs(a[1]) * halfM[1] + Math.abs(a[2]) * halfM[2]

  const t0 = performance.now()
  const vols = bakeVolumes(mesh)

  // Transient bake resources (destroyed below; destroy() un-counts them).
  const atlas = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/bake-atlas`,
      size: [ATLAS, ATLAS],
      format: 'rg8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    },
    { species: speciesId, tag: 'bake-transient' },
  )
  const vbuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-verts`, size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'bake-transient' },
  )
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'bake-transient' },
  )
  device.queue.writeBuffer(ibuf, 0, indices)
  // One-row strip render targets at supersampled resolution.
  const stripColor = ctx.res.createTexture(
    {
      label: `${ctx.id}/bake-strip`,
      size: [STRIP_W, TILE_SS],
      format: 'r8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    },
    { species: speciesId, tag: 'bake-transient' },
  )
  const stripDepth = ctx.res.createTexture(
    { label: `${ctx.id}/bake-strip-depth`, size: [STRIP_W, TILE_SS], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT },
    { species: speciesId, tag: 'bake-transient' },
  )

  // Per-tile basis uniforms (dynamic offset ring, 256B stride).
  const STRIDE = 256
  const nTiles = GRID * GRID
  const uni = ctx.res.createBuffer(
    { label: `${ctx.id}/bake-uni`, size: STRIDE * nTiles, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'bake-transient' },
  )
  const scratch = new ArrayBuffer(STRIDE * nTiles)
  const fv = new Float32Array(scratch)
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      const ex = (i / (GRID - 1)) * 2 - 1
      const ey = (j / (GRID - 1)) * 2 - 1
      const fwd = hemioctDecode(ex, ey)
      const upRef: V3 = Math.abs(fwd[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0]
      const right = norm(cross(upRef, fwd))
      const up = norm(cross(fwd, right))
      const o = ((j * GRID + i) * STRIDE) / 4
      fv[o] = center[0]; fv[o + 1] = center[1]; fv[o + 2] = center[2]; fv[o + 3] = 0
      fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]; fv[o + 7] = 1 / support(right)
      fv[o + 8] = up[0]; fv[o + 9] = up[1]; fv[o + 10] = up[2]; fv[o + 11] = 1 / support(up)
      fv[o + 12] = fwd[0]; fv[o + 13] = fwd[1]; fv[o + 14] = fwd[2]; fv[o + 15] = 1 / support(fwd)
      fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
      fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
    }
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
      buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'uint16x4' }] }],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format: 'r8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  // Downsample: strip (2x2 supersampled) -> final atlas row (min depth + coverage).
  const downBgl = device.createBindGroupLayout({
    label: `${ctx.id}/down-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
  })
  const downBg = device.createBindGroup({
    label: `${ctx.id}/down-bg`,
    layout: downBgl,
    entries: [{ binding: 0, resource: stripColor.createView() }],
  })
  const downPipeline = device.createRenderPipeline({
    label: `${ctx.id}/down-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/down-pl`, bindGroupLayouts: [downBgl] }),
    vertex: { module, entryPoint: 'vs_down' },
    fragment: {
      module,
      entryPoint: 'fs_down',
      targets: [{ format: 'rg8unorm' }],
      // The strip holds ONE row of views, so the shader has to fold the atlas
      // row offset out of @builtin(position) — it needs the tile height.
      constants: { tile_px: TILE },
    },
    primitive: { topology: 'triangle-list' },
  })

  const tilesPerSubmit = Math.max(1, Math.min(GRID, Math.floor(TRIS_PER_SUBMIT / mesh.header.triangleCount)))
  const stripColorView = stripColor.createView()
  const stripDepthView = stripDepth.createView()
  const atlasView = atlas.createView()
  for (let j = 0; j < GRID; j++) {
    // 1. Rasterize row j's views into the supersampled strip (chunked).
    for (let start = 0; start < GRID; start += tilesPerSubmit) {
      const end = Math.min(GRID, start + tilesPerSubmit)
      const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-enc` })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/bake-strip-pass`,
        colorAttachments: [
          {
            view: stripColorView,
            // r=1 is the "miss" sentinel — hits write <= 254/255.
            clearValue: { r: 1, g: 1, b: 1, a: 1 },
            loadOp: start === 0 ? 'clear' : 'load',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: stripDepthView,
          depthClearValue: 1,
          depthLoadOp: start === 0 ? 'clear' : 'load',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      for (let i = start; i < end; i++) {
        pass.setViewport(i * TILE_SS, 0, TILE_SS, TILE_SS, 0, 1)
        pass.setScissorRect(i * TILE_SS, 0, TILE_SS, TILE_SS)
        pass.setBindGroup(0, bg, [(j * GRID + i) * STRIDE])
        pass.drawIndexed(indices.length)
      }
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    // 2. Downsample the strip into atlas row j.
    const enc = device.createCommandEncoder({ label: `${ctx.id}/down-enc` })
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/down-pass`,
      colorAttachments: [
        {
          view: atlasView,
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: j === 0 ? 'clear' : 'load',
          storeOp: 'store',
        },
      ],
    })
    pass.setPipeline(downPipeline)
    pass.setViewport(0, j * TILE, ATLAS, TILE, 0, 1)
    pass.setScissorRect(0, j * TILE, ATLAS, TILE)
    pass.setBindGroup(0, downBg)
    pass.draw(3)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }

  // Read the atlas back so it can be cached and committed.
  const rowBytes = ATLAS * 2
  const readback = ctx.res.createBuffer(
    {
      label: `${ctx.id}/bake-readback`,
      size: rowBytes * ATLAS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    },
    { species: speciesId, tag: 'bake-transient' },
  )
  {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/readback-enc` })
    enc.copyTextureToBuffer({ texture: atlas }, { buffer: readback, bytesPerRow: rowBytes, rowsPerImage: ATLAS }, [
      ATLAS,
      ATLAS,
    ])
    device.queue.submit([enc.finish()])
  }
  await readback.mapAsync(GPUMapMode.READ)
  const atlasBytes = new Uint8Array(readback.getMappedRange().slice(0))
  readback.unmap()

  for (const r of [vbuf, ibuf, stripColor, stripDepth, uni, atlas, readback]) r.destroy()
  console.log(
    `[${ctx.id}] baked ${speciesId}: ${nTiles} views @${TILE}px (${SS}x${SS} ss) + ${vols.dims.join('x')} volumes in ${(
      (performance.now() - t0) /
      1000
    ).toFixed(1)}s`,
  )

  return {
    grid: GRID,
    tile: TILE,
    dims: vols.dims,
    center,
    halfU,
    radius,
    bmin: [bmin[0], bmin[1], bmin[2]],
    bmax: [bmax[0], bmax[1], bmax[2]],
    heightM: bmax[1] - bmin[1],
    atlas: atlasBytes,
    albedo: vols.albedo,
    normal: vols.normal,
  }
}
