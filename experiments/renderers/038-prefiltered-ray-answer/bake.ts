import plantOrthoSrc from './shaders/plant_ortho.wgsl'
import compositeSrc from './shaders/composite.wgsl'
import resolveSrc from './shaders/resolve_ss.wgsl'
import {
  bakedArtifact,
  commitBake,
  SCATTER_CELL_SIZE,
  speciesById,
  type ExperimentContext,
  type GcMesh,
  type ScatterPoint,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * THE BAKE — a table of precomputed ray ANSWERS for the whole stand canopy.
 *
 * Stage 1 (per species): 144 + 1 orthographic renders of the raw GCMESH1 mesh,
 * one per baked direction. An orthographic render with a depth test along the
 * view axis *is* a first-hit ray solve, resolved offline by the rasterizer for
 * every pixel of the bundle at once.
 *
 * Stage 2 (once): the patch answer. The patch is a TILE_L x TILE_L periodic
 * window of the ACTIVE STAND's own scatter — real cells, real positions, real
 * yaws (snapped to the 15 deg azimuth grid), real scales, real species mix. For
 * each direction, every plant's stage-1 tile is drawn as one sheared quad into
 * "entry point" space:
 *
 *     E(X) = X.xz - d.xz * (X.y - H) / d.y
 *
 * which is invariant along d, hence affine in the tile coordinates. Depth =
 * hit height, so the depth test resolves the union of all plants EXACTLY (first
 * hit along a descending ray is the greatest y). Rendered at 2x and resolved
 * coverage-weighted, so a texel of the final table is the honest average of the
 * ray answers over its footprint of entry points.
 *
 * Artifact (little-endian):
 *   u32 magic 'PRA1', u32 version, u32 res, u32 slices, u32 nAz, u32 nEl
 *   f32 tileL, f32 canopyH, f32 lc0, f32 lcStep, f32 aoRange
 *   ... zeros to byte 64
 *   u8[slices * res^2 * 4]  surf: rgb = albedo * cov, a = AO * cov
 *   u8[slices * res^2 * 4]  geom: r = height01 * cov, gb = oct' * cov, a = cov
 * Layer order is band-major (band = azimuth / 8) so each of the three runtime
 * texture pairs uploads as one contiguous block.
 */

export const RES = 192
export const N_AZ = 24
export const N_EL = 6
export const SLICES = N_AZ * N_EL
export const AZ_PER_BAND = 8
export const BANDS = N_AZ / AZ_PER_BAND
export const LAYERS_PER_BAND = AZ_PER_BAND * N_EL
export const TILE_L = 6
export const LC0 = 2.8
export const LC_STEP = 1.02
export const AO_RANGE = 0.6
/** 85th-percentile plant top: the canopy is truncated here (see NOTES). */
export const HEIGHT_PERCENTILE = 0.85

const SS = 2
const PLANT_GRID = 13
const TOP_TILE = SLICES
const MAGIC = 0x31415250
const VERSION = 1
const HEADER = 64
const UNI_STRIDE = 256

export interface RayTable {
  res: number
  slices: number
  tileL: number
  canopyH: number
  /** surf and geom, layer-major (band-major layer order). */
  surf: Uint8Array<ArrayBuffer>
  geom: Uint8Array<ArrayBuffer>
}

type BakeCtx = Pick<ExperimentContext<typeof PARAMS>, 'id' | 'device' | 'meshes' | 'shaders' | 'res' | 'scene' | 'stand'>

/** Elevation (radians below horizontal) of bin `el` — uniform in ln(cot). */
export function elTheta(el: number): number {
  return Math.atan2(1, Math.exp(LC0 - LC_STEP * el))
}

/** Staging/runtime layer index for a slice: band-major, then azimuth, then elevation. */
export function layerIndex(az: number, el: number): number {
  return Math.floor(az / AZ_PER_BAND) * LAYERS_PER_BAND + (az % AZ_PER_BAND) * N_EL + el
}

/** Canopy height the table is baked (and truncated) to, from the stand alone. */
export function canopyHeight(stand: BakeCtx['stand']): number {
  let h = 0
  for (const e of stand.species) {
    const s = speciesById(e.species)
    h = Math.max(h, s.heightScale * (e.scaleMin + HEIGHT_PERCENTILE * (e.scaleMax - e.scaleMin)))
  }
  return h
}

/** Density-weighted mean wind response of the stand (one shear for one table). */
export function meanSway(stand: BakeCtx['stand']): number {
  let num = 0
  let den = 0
  for (const e of stand.species) {
    num += e.sway * e.density
    den += e.density
  }
  return den > 0 ? num / den : 0
}

function mixKey(ctx: BakeCtx, seed: number): string {
  const parts = ctx.stand.species
    .map((e) => `${e.species}:${e.density}:${e.scaleMin}:${e.scaleMax}:${e.wetCenter ?? -1}:${e.wetWidth ?? -1}`)
    .join('|')
  let h = 0x811c9dc5
  const s = `${parts}|H${canopyHeight(ctx.stand).toFixed(4)}|L${TILE_L}|${RES}x${N_AZ}x${N_EL}|s${seed}`
  for (let i = 0; i < s.length; i++) {
    h = ((h ^ s.charCodeAt(i)) * 0x01000193) >>> 0
  }
  return h.toString(16).padStart(8, '0')
}

// ---------------------------------------------------------------------------
// artifact
// ---------------------------------------------------------------------------

export function unpackTable(buf: ArrayBuffer): RayTable | null {
  if (buf.byteLength < HEADER) return null
  const u = new Uint32Array(buf, 0, 6)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  const res = u[2]!
  const slices = u[3]!
  if (res !== RES || slices !== SLICES || u[4] !== N_AZ || u[5] !== N_EL) return null
  const plane = res * res * 4 * slices
  if (buf.byteLength !== HEADER + plane * 2) return null
  const f = new Float32Array(buf, 24, 5)
  return {
    res,
    slices,
    tileL: f[0]!,
    canopyH: f[1]!,
    surf: new Uint8Array(buf, HEADER, plane),
    geom: new Uint8Array(buf, HEADER + plane, plane),
  }
}

function packTable(t: { canopyH: number; surf: Uint8Array; geom: Uint8Array }): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER + t.surf.byteLength + t.geom.byteLength)
  const u = new Uint32Array(buf, 0, 6)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = RES
  u[3] = SLICES
  u[4] = N_AZ
  u[5] = N_EL
  const f = new Float32Array(buf, 24, 5)
  f[0] = TILE_L
  f[1] = t.canopyH
  f[2] = LC0
  f[3] = LC_STEP
  f[4] = AO_RANGE
  new Uint8Array(buf, HEADER, t.surf.byteLength).set(t.surf)
  new Uint8Array(buf, HEADER + t.surf.byteLength, t.geom.byteLength).set(t.geom)
  return buf
}

/**
 * Load (OPFS cache / committed file) or bake the answer table. Every result is
 * magic-validated: the dev server answers a missing /mesh/baked file with
 * index.html at status 200, which would otherwise poison both stores.
 */
export async function loadRayTable(
  ctx: BakeCtx,
  seed: number,
  onProgress?: (f: number, note?: string) => void,
): Promise<RayTable> {
  const key = `answer-v${VERSION}-mix${mixKey(ctx, seed)}-s${seed}`
  let bakedFresh = false
  const runBake = async (): Promise<ArrayBuffer> => {
    bakedFresh = true
    return bakeTable(ctx, seed, onProgress)
  }
  let buf = await bakedArtifact({ expId: ctx.id, key, ...(onProgress && { onProgress }) }, runBake)
  let table = unpackTable(buf)
  if (!table) {
    buf = await runBake()
    table = unpackTable(buf)
    if (!table) throw new Error(`[${ctx.id}] bake produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (bakedFresh) {
    try {
      await commitBake(ctx.id, key, buf)
    } catch (err) {
      console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
    }
  }
  return table
}

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
// the patch: a periodic window of the stand's own scatter
// ---------------------------------------------------------------------------

interface PatchPlant extends ScatterPoint {
  slot: number
  /** heightScale * scale (m). */
  height: number
}

function collectPatchPlants(ctx: BakeCtx, slots: Map<string, number>): PatchPlant[] {
  const cells = Math.ceil(TILE_L / SCATTER_CELL_SIZE)
  const out: PatchPlant[] = []
  ctx.stand.species.forEach((entry, entryIndex) => {
    const species = speciesById(entry.species)
    for (let cz = 0; cz < cells; cz++) {
      for (let cx = 0; cx < cells; cx++) {
        for (const pt of ctx.scene.scatter.cell(entryIndex, cx, cz)) {
          if (pt.x < 0 || pt.x >= TILE_L || pt.z < 0 || pt.z >= TILE_L) continue
          out.push({ ...pt, slot: slots.get(entry.species)!, height: species.heightScale * pt.scale })
        }
      }
    }
  })
  return out
}

// ---------------------------------------------------------------------------
// stage 1 — per-plant directional first-hit atlas
// ---------------------------------------------------------------------------

interface MeshFrame {
  cx: number
  cz: number
  y0: number
  y1: number
  /** Bounding-sphere radius around the capture centre (mesh units). */
  radius: number
}

function meshFrame(mesh: GcMesh): MeshFrame {
  const hdr = mesh.header
  const [bx0, by0, bz0] = hdr.boundsMin
  const [bx1, by1, bz1] = hdr.boundsMax
  const cx = (bx0 + bx1) / 2
  const cz = (bz0 + bz1) / 2
  const y0 = Math.min(0, by0)
  const y1 = by1
  const cy = (y0 + y1) / 2
  const v = mesh.vertices
  const sx = (bx1 - bx0) / 65535
  const sy = (by1 - by0) / 65535
  const sz = (bz1 - bz0) / 65535
  let r2 = 0
  for (let i = 0; i < hdr.vertexCount; i++) {
    const dx = bx0 + v[i * 8]! * sx - cx
    const dy = by0 + v[i * 8 + 1]! * sy - cy
    const dz = bz0 + v[i * 8 + 2]! * sz - cz
    const d = dx * dx + dy * dy + dz * dz
    if (d > r2) r2 = d
  }
  return { cx, cz, y0, y1, radius: Math.sqrt(r2) * 1.02 + 1e-3 }
}

/** Ortho basis of a baked direction: `a` horizontal, `b` = d x a (up-ish). */
function sliceBasis(az: number, el: number): { d: [number, number, number]; a: [number, number, number]; b: [number, number, number] } {
  if (el < 0) {
    return { d: [0, -1, 0], a: [1, 0, 0], b: [0, 0, 1] }
  }
  const phi = (2 * Math.PI * az) / N_AZ
  const th = elTheta(el)
  const d: [number, number, number] = [Math.cos(phi) * Math.cos(th), -Math.sin(th), Math.sin(phi) * Math.cos(th)]
  const a: [number, number, number] = [Math.sin(phi), 0, -Math.cos(phi)]
  const b: [number, number, number] = [
    d[1] * a[2] - d[2] * a[1],
    d[2] * a[0] - d[0] * a[2],
    d[0] * a[1] - d[1] * a[0],
  ]
  return { d, a, b }
}

async function bakeTable(
  ctx: BakeCtx,
  seed: number,
  onProgress?: (f: number, note?: string) => void,
): Promise<ArrayBuffer> {
  const { device } = ctx
  const canopyH = canopyHeight(ctx.stand)

  const speciesIds = [...new Set(ctx.stand.species.map((e) => e.species))]
  const slots = new Map(speciesIds.map((id, i) => [id, i]))
  const plants = collectPatchPlants(ctx, slots)
  if (plants.length === 0) throw new Error(`[${ctx.id}] the stand put no plants in the ${TILE_L}m patch window`)
  const plantRes = speciesIds.length <= 3 ? 192 : 128
  const atlasPx = plantRes * PLANT_GRID

  onProgress?.(0.02, `patch: ${plants.length} plants, ${speciesIds.length} species`)

  const scratch: (GPUBuffer | GPUTexture)[] = []
  const tex = (
    label: string,
    size: [number, number] | [number, number, number],
    format: GPUTextureFormat,
    usage: number,
  ): GPUTexture => {
    const t = ctx.res.createTexture({ label: `${ctx.id}/${label}`, size, format, usage }, { tag: 'bake-scratch' })
    scratch.push(t)
    return t
  }
  const buf = (label: string, size: number, usage: number): GPUBuffer => {
    const b = ctx.res.createBuffer({ label: `${ctx.id}/${label}`, size, usage }, { tag: 'bake-scratch' })
    scratch.push(b)
    return b
  }

  try {
    // ---- stage 1 -----------------------------------------------------------
    const atlasUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const atlasSurf = tex('atlas-surf', [atlasPx, atlasPx, speciesIds.length], 'rgba8unorm', atlasUsage)
    const atlasGeom = tex('atlas-geom', [atlasPx, atlasPx, speciesIds.length], 'rgba8unorm', atlasUsage)
    const atlasDepth = tex('atlas-depth', [atlasPx, atlasPx], 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT)

    const orthoUni = buf('ortho-uni', UNI_STRIDE * (SLICES + 1), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const orthoBgl = device.createBindGroupLayout({
      label: `${ctx.id}/ortho-bgl`,
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
        },
      ],
    })
    const orthoBg = device.createBindGroup({
      label: `${ctx.id}/ortho-bg`,
      layout: orthoBgl,
      entries: [{ binding: 0, resource: { buffer: orthoUni, size: 96 } }],
    })
    const orthoModule = ctx.shaders.module(plantOrthoSrc)
    const orthoPipe = device.createRenderPipeline({
      label: `${ctx.id}/ortho`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/ortho`, bindGroupLayouts: [orthoBgl] }),
      vertex: {
        module: orthoModule,
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
      fragment: {
        module: orthoModule,
        entryPoint: 'fs',
        targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
    })

    const frames: MeshFrame[] = []
    for (let s = 0; s < speciesIds.length; s++) {
      const id = speciesIds[s]!
      onProgress?.(0.04 + 0.5 * (s / speciesIds.length), `ray-solving ${id} (${SLICES + 1} directions)`)
      const mesh = await ctx.meshes.load(speciesById(id).meshId)
      const frame = meshFrame(mesh)
      frames.push(frame)

      const uni = new Float32Array(((SLICES + 1) * UNI_STRIDE) / 4)
      for (let t = 0; t <= SLICES; t++) {
        const isTop = t === TOP_TILE
        const az = isTop ? 0 : t % N_AZ
        const el = isTop ? -1 : Math.floor(t / N_AZ)
        const { d, a, b } = sliceBasis(az, el)
        const o = (t * UNI_STRIDE) / 4
        uni.set([a[0], a[1], a[2], frame.radius], o)
        uni.set([b[0], b[1], b[2], 0], o + 4)
        uni.set([d[0], d[1], d[2], 0], o + 8)
        uni.set([frame.cx, (frame.y0 + frame.y1) / 2, frame.cz, 1 / (frame.y1 - frame.y0)], o + 12)
        uni.set(
          [mesh.header.boundsMin[0], mesh.header.boundsMin[1], mesh.header.boundsMin[2], frame.y0],
          o + 16,
        )
        uni.set(
          [
            mesh.header.boundsMax[0] - mesh.header.boundsMin[0],
            mesh.header.boundsMax[1] - mesh.header.boundsMin[1],
            mesh.header.boundsMax[2] - mesh.header.boundsMin[2],
            0,
          ],
          o + 20,
        )
      }
      device.queue.writeBuffer(orthoUni, 0, uni)

      const vbuf = device.createBuffer({
        label: `${ctx.id}/mesh-v`,
        size: mesh.vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      const indices = mesh.indices()
      const ibuf = device.createBuffer({
        label: `${ctx.id}/mesh-i`,
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(vbuf, 0, mesh.vertices)
      device.queue.writeBuffer(ibuf, 0, indices)

      const enc = device.createCommandEncoder({ label: `${ctx.id}/ortho-${id}` })
      const layerView = (t: GPUTexture): GPUTextureView =>
        t.createView({ dimension: '2d', baseArrayLayer: s, arrayLayerCount: 1 })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/ortho-${id}`,
        colorAttachments: [
          { view: layerView(atlasSurf), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          { view: layerView(atlasGeom), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: atlasDepth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(orthoPipe)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      for (let t = 0; t <= SLICES; t++) {
        const col = t % PLANT_GRID
        const row = Math.floor(t / PLANT_GRID)
        pass.setViewport(col * plantRes, row * plantRes, plantRes, plantRes, 0, 1)
        pass.setScissorRect(col * plantRes, row * plantRes, plantRes, plantRes)
        pass.setBindGroup(0, orthoBg, [t * UNI_STRIDE])
        pass.drawIndexed(indices.length)
      }
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      vbuf.destroy()
      ibuf.destroy()
    }

    // ---- stage 2 -----------------------------------------------------------
    const instData = new Float32Array(plants.length * 8)
    let maxR = 0
    plants.forEach((p, i) => {
      const frame = frames[p.slot]!
      const k = p.height / (frame.y1 - frame.y0)
      const rWorld = frame.radius * k
      maxR = Math.max(maxR, rWorld)
      const azYaw = Math.round((p.yaw / (2 * Math.PI)) * N_AZ) % N_AZ
      const psi = (2 * Math.PI * azYaw) / N_AZ
      instData.set([p.x, p.z, p.height, rWorld], i * 8)
      instData.set([Math.cos(psi), Math.sin(psi), p.slot, azYaw], i * 8 + 4)
    })
    const instBuf = buf('instances', instData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(instBuf, 0, instData)

    const ssPx = RES * SS
    const rt = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const ssSurf = tex('ss-surf', [ssPx, ssPx], 'rgba8unorm', rt)
    const ssGeom = tex('ss-geom', [ssPx, ssPx], 'rgba8unorm', rt)
    const ssDepth = tex('ss-depth', [ssPx, ssPx], 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT)
    const store = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    const stagSurf = tex('stage-surf', [RES, RES, SLICES], 'rgba8unorm', store)
    const stagGeom = tex('stage-geom', [RES, RES, SLICES], 'rgba8unorm', store)
    const topSurf = tex('top-surf', [RES, RES, 1], 'rgba8unorm', store)
    const topGeom = tex('top-geom', [RES, RES, 1], 'rgba8unorm', store)

    const compUni = buf('comp-uni', UNI_STRIDE * (SLICES + 1), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const wrapSide: number[] = []
    const compData = new Float32Array(((SLICES + 1) * UNI_STRIDE) / 4)
    for (let t = 0; t <= SLICES; t++) {
      const isTop = t === TOP_TILE
      const az = isTop ? 0 : t % N_AZ
      const el = isTop ? -1 : Math.floor(t / N_AZ)
      const { d, a, b } = sliceBasis(az, el)
      const th = isTop ? Math.PI / 2 : elTheta(el)
      const sinT = Math.sin(th)
      const cosT = Math.cos(th)
      const extent = maxR * (1 + sinT + (cosT * cosT) / Math.max(sinT, 1e-3))
      const w = Math.min(2 * Math.ceil(extent / TILE_L) + 1, 41)
      wrapSide.push(w)
      const o = (t * UNI_STRIDE) / 4
      compData.set([a[0], a[1], a[2], 1 / PLANT_GRID], o)
      compData.set([b[0], b[1], b[2], ssPx], o + 4)
      compData.set([d[0], d[1], d[2], canopyH], o + 8)
      compData.set([az, Math.max(el, 0), w, isTop ? 1 : 0], o + 12)
      compData.set([TILE_L, canopyH, AO_RANGE, PLANT_GRID], o + 16)
      compData.set([atlasPx, 0, 0, 0], o + 20)
    }
    device.queue.writeBuffer(compUni, 0, compData)

    const atlasSampler = device.createSampler({
      label: `${ctx.id}/atlas-samp`,
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
    const compBgl = device.createBindGroupLayout({
      label: `${ctx.id}/comp-bgl`,
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
        },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      ],
    })
    const compBg = device.createBindGroup({
      label: `${ctx.id}/comp-bg`,
      layout: compBgl,
      entries: [
        { binding: 0, resource: { buffer: compUni, size: 96 } },
        { binding: 1, resource: { buffer: instBuf } },
        { binding: 2, resource: atlasSurf.createView({ dimension: '2d-array' }) },
        { binding: 3, resource: atlasGeom.createView({ dimension: '2d-array' }) },
        { binding: 4, resource: atlasSampler },
        { binding: 5, resource: topGeom.createView({ dimension: '2d-array' }) },
      ],
    })
    const compModule = ctx.shaders.module(compositeSrc)
    const compPipe = device.createRenderPipeline({
      label: `${ctx.id}/composite`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/composite`, bindGroupLayouts: [compBgl] }),
      vertex: { module: compModule, entryPoint: 'vs' },
      fragment: {
        module: compModule,
        entryPoint: 'fs',
        targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
    })

    // resolve (2x2 coverage-weighted, premultiplying) into the staging layers
    const resUni = buf('res-uni', UNI_STRIDE * (SLICES + 1), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const resData = new Uint32Array(((SLICES + 1) * UNI_STRIDE) / 4)
    for (let t = 0; t <= SLICES; t++) {
      const isTop = t === TOP_TILE
      const az = isTop ? 0 : t % N_AZ
      const el = isTop ? 0 : Math.floor(t / N_AZ)
      const o = (t * UNI_STRIDE) / 4
      resData[o] = isTop ? 0 : layerIndex(az, el)
      resData[o + 1] = RES
    }
    device.queue.writeBuffer(resUni, 0, resData)
    const resBgl = device.createBindGroupLayout({
      label: `${ctx.id}/res-bgl`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d-array' },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d-array' },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 },
        },
      ],
    })
    const mkResBg = (dstS: GPUTexture, dstG: GPUTexture): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/res-bg`,
        layout: resBgl,
        entries: [
          { binding: 0, resource: ssSurf.createView() },
          { binding: 1, resource: ssGeom.createView() },
          { binding: 2, resource: dstS.createView({ dimension: '2d-array' }) },
          { binding: 3, resource: dstG.createView({ dimension: '2d-array' }) },
          { binding: 4, resource: { buffer: resUni, size: 16 } },
        ],
      })
    const resBgMain = mkResBg(stagSurf, stagGeom)
    const resBgTop = mkResBg(topSurf, topGeom)
    const resPipe = device.createComputePipeline({
      label: `${ctx.id}/resolve`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/resolve`, bindGroupLayouts: [resBgl] }),
      compute: { module: ctx.shaders.module(resolveSrc), entryPoint: 'main' },
    })

    const groups = Math.ceil(RES / 8)
    const compositeSlice = (enc: GPUCommandEncoder, t: number): void => {
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/comp-${t}`,
        colorAttachments: [
          { view: ssSurf.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          { view: ssGeom.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: ssDepth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(compPipe)
      pass.setBindGroup(0, compBg, [t * UNI_STRIDE])
      const w = wrapSide[t]!
      pass.draw(6, plants.length * w * w)
      pass.end()
      const cpass = enc.beginComputePass({ label: `${ctx.id}/resolve-${t}` })
      cpass.setPipeline(resPipe)
      cpass.setBindGroup(0, t === TOP_TILE ? resBgTop : resBgMain, [t * UNI_STRIDE])
      cpass.dispatchWorkgroups(groups, groups, 1)
      cpass.end()
    }

    // The straight-down height map first: the slices sample it for baked AO.
    {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/comp-top` })
      compositeSlice(enc, TOP_TILE)
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }
    const CHUNK = 24
    for (let base = 0; base < SLICES; base += CHUNK) {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/comp-${base}` })
      for (let t = base; t < Math.min(base + CHUNK, SLICES); t++) compositeSlice(enc, t)
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      onProgress?.(0.56 + 0.36 * ((base + CHUNK) / SLICES), `compositing answers ${base}/${SLICES}`)
    }

    // ---- readback ----------------------------------------------------------
    const plane = RES * RES * 4 * SLICES
    const rbSurf = buf('rb-surf', plane, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    const rbGeom = buf('rb-geom', plane, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    const enc = device.createCommandEncoder({ label: `${ctx.id}/readback` })
    enc.copyTextureToBuffer(
      { texture: stagSurf },
      { buffer: rbSurf, bytesPerRow: RES * 4, rowsPerImage: RES },
      [RES, RES, SLICES],
    )
    enc.copyTextureToBuffer(
      { texture: stagGeom },
      { buffer: rbGeom, bytesPerRow: RES * 4, rowsPerImage: RES },
      [RES, RES, SLICES],
    )
    device.queue.submit([enc.finish()])
    await rbSurf.mapAsync(GPUMapMode.READ)
    await rbGeom.mapAsync(GPUMapMode.READ)
    const surf = new Uint8Array(rbSurf.getMappedRange()).slice()
    const geom = new Uint8Array(rbGeom.getMappedRange()).slice()
    rbSurf.unmap()
    rbGeom.unmap()
    onProgress?.(0.98, 'packing table')
    return packTable({ canopyH, surf, geom })
  } finally {
    for (const r of scratch) r.destroy()
  }
}
