import bakeViewSrc from './shaders/bake-view.wgsl'
import downsampleSrc from './shaders/bake-downsample.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext } from '@harness'

/**
 * Bakes the "ray-answer field" for one species: RF_VIEWS^2 orthographic
 * rasterizations of the raw source mesh (each one a bundle of precomputed
 * raycasts), supersampled SS x SS and resolved into a RF_ATLAS^2 pair of
 * rgba8 atlases:
 *   surf = premultiplied albedo rgb, coverage
 *   geom = premultiplied depth01 (along the view dir), oct normal, height01
 * The result is cached via the harness bake flow (OPFS + mesh/baked/).
 */

export const RF_VIEWS = 24
export const RF_SLAB = 64
export const RF_ATLAS = RF_VIEWS * RF_SLAB
const SS = 4
const SRC_RES = RF_SLAB * SS
const BAKE_VERSION = 1
const MAGIC = 0x31544c52 // 'RLT1'
const HEADER_BYTES = 64
const ATLAS_BYTES = RF_ATLAS * RF_ATLAS * 4

export interface RayField {
  center: [number, number, number]
  radius: number
  topH: number
  surf: Uint8Array
  geom: Uint8Array
}

/** TS mirror of rf_oct_decode() in rayfield-common.wgsl. */
export function octDecode(ex: number, ey: number): [number, number, number] {
  const fx = ex * 2 - 1
  const fy = ey * 2 - 1
  let x = fx
  let z = fy
  const y = 1 - Math.abs(fx) - Math.abs(fy)
  if (y < 0) {
    x = (1 - Math.abs(fy)) * (fx >= 0 ? 1 : -1)
    z = (1 - Math.abs(fx)) * (fy >= 0 ? 1 : -1)
  }
  const len = Math.hypot(x, y, z) || 1
  return [x / len, y / len, z / len]
}

/** TS mirror of rf_view_right()/up in rayfield-common.wgsl. */
export function viewBasis(dv: [number, number, number]): {
  r: [number, number, number]
  u: [number, number, number]
} {
  let r: [number, number, number]
  if (Math.abs(dv[1]) > 0.999) {
    r = [1, 0, 0]
  } else {
    // normalize(cross((0,1,0), dv))
    const cx = dv[2]
    const cz = -dv[0]
    const l = Math.hypot(cx, cz) || 1
    r = [cx / l, 0, cz / l]
  }
  // u = normalize(cross(dv, r))
  const ux = dv[1] * r[2] - dv[2] * r[1]
  const uy = dv[2] * r[0] - dv[0] * r[2]
  const uz = dv[0] * r[1] - dv[1] * r[0]
  const l = Math.hypot(ux, uy, uz) || 1
  return { r, u: [ux / l, uy / l, uz / l] }
}

export async function loadRayField(
  ctx: ExperimentContext,
  speciesId: string,
  onProgress?: (f: number, note?: string) => void,
): Promise<RayField> {
  const key = `${speciesId}-v${BAKE_VERSION}-${RF_VIEWS}x${RF_SLAB}`
  let bakedNow = false
  const data = await bakedArtifact({ expId: ctx.id, key, ...(onProgress && { onProgress }) }, async () => {
    bakedNow = true
    return bakeRayField(ctx, speciesId, onProgress)
  })
  try {
    const rf = parseRayField(data)
    if (bakedNow) await commitArtifact(ctx.id, key, data)
    return rf
  } catch {
    // The harness bake flow can cache the dev server's SPA-fallback HTML when
    // the committed file does not exist yet (fetch returns 200 + index.html).
    // Rebake for real, repair the poisoned OPFS entry, and commit.
    console.warn(`[${ctx.id}] cached artifact for ${key} is invalid — rebaking`)
    const fresh = await bakeRayField(ctx, speciesId, onProgress)
    await repairOpfsCache(`${ctx.id}__${key}`, fresh)
    await commitArtifact(ctx.id, key, fresh)
    return parseRayField(fresh)
  }
}

async function commitArtifact(expId: string, key: string, data: ArrayBuffer): Promise<void> {
  try {
    await commitBake(expId, key, data)
  } catch (err) {
    console.warn(`[${expId}] commitBake failed (static build?):`, err)
  }
}

/** Overwrite the harness OPFS bake-cache entry (same layout as src/bake/cache.ts). */
async function repairOpfsCache(fullKey: string, data: ArrayBuffer): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle('bake-cache', { create: true })
    const handle = await dir.getFileHandle(`${fullKey}.bin`, { create: true })
    const writable = await handle.createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    // No OPFS — cache stays poisoned but every load self-heals by rebaking.
  }
}

function parseRayField(data: ArrayBuffer): RayField {
  if (data.byteLength !== HEADER_BYTES + ATLAS_BYTES * 2) throw new Error('ray-field artifact: bad size')
  const dv = new DataView(data)
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('ray-field artifact: bad magic')
  if (dv.getUint32(4, true) !== BAKE_VERSION) throw new Error('ray-field artifact: version mismatch')
  if (dv.getUint32(8, true) !== RF_VIEWS || dv.getUint32(12, true) !== RF_SLAB) {
    throw new Error('ray-field artifact: layout mismatch')
  }
  return {
    center: [dv.getFloat32(16, true), dv.getFloat32(20, true), dv.getFloat32(24, true)],
    radius: dv.getFloat32(28, true),
    topH: dv.getFloat32(32, true),
    surf: new Uint8Array(data, HEADER_BYTES, ATLAS_BYTES),
    geom: new Uint8Array(data, HEADER_BYTES + ATLAS_BYTES, ATLAS_BYTES),
  }
}

async function bakeRayField(
  ctx: ExperimentContext,
  speciesId: string,
  onProgress?: (f: number, note?: string) => void,
): Promise<ArrayBuffer> {
  const { device } = ctx
  const meshId = speciesById(speciesId).meshId
  const info = ctx.meshes.info(meshId)
  const mesh = await ctx.meshes.load(meshId)
  const h = mesh.header

  // Instance pivot: tile center for periodic community tiles, bounds center
  // for finite specimens. Baked space puts this pivot at xz origin.
  const anchor: [number, number] = info.tileSize
    ? [h.tileOrigin[0] + info.tileSize[0] / 2, h.tileOrigin[1] + info.tileSize[1] / 2]
    : [(h.boundsMin[0] + h.boundsMax[0]) / 2, (h.boundsMin[2] + h.boundsMax[2]) / 2]
  const bmin: [number, number, number] = [h.boundsMin[0] - anchor[0], h.boundsMin[1], h.boundsMin[2] - anchor[1]]
  const bmax: [number, number, number] = [h.boundsMax[0] - anchor[0], h.boundsMax[1], h.boundsMax[2] - anchor[1]]
  const center: [number, number, number] = [
    (bmin[0] + bmax[0]) / 2,
    (bmin[1] + bmax[1]) / 2,
    (bmin[2] + bmax[2]) / 2,
  ]
  // Tight bounding-sphere radius from actual vertices (quantized decode).
  let r2max = 0
  const verts = mesh.vertices
  const sx = (bmax[0] - bmin[0]) / 65535
  const sy = (bmax[1] - bmin[1]) / 65535
  const sz = (bmax[2] - bmin[2]) / 65535
  for (let i = 0; i < h.vertexCount; i++) {
    const b = i * 8
    const dx = bmin[0] + verts[b]! * sx - center[0]
    const dy = bmin[1] + verts[b + 1]! * sy - center[1]
    const dz = bmin[2] + verts[b + 2]! * sz - center[2]
    const r2 = dx * dx + dy * dy + dz * dz
    if (r2 > r2max) r2max = r2
  }
  const radius = Math.sqrt(r2max) * 1.005

  // --- transient GPU resources (destroyed before returning) ---
  const temp: (GPUBuffer | GPUTexture)[] = []
  const tag = { species: speciesId, tag: 'bake-temp' }
  const track = <T extends GPUBuffer | GPUTexture>(r: T): T => {
    temp.push(r)
    return r
  }
  try {
    const vtx = track(
      ctx.res.createBuffer(
        { label: `${ctx.id}/bake-vtx`, size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
        tag,
      ),
    )
    device.queue.writeBuffer(vtx, 0, mesh.vertices)
    const indices = mesh.indices()
    const idx = track(
      ctx.res.createBuffer(
        { label: `${ctx.id}/bake-idx`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
        tag,
      ),
    )
    device.queue.writeBuffer(idx, 0, indices)

    const viewUbo = track(
      ctx.res.createBuffer(
        { label: `${ctx.id}/bake-view-ubo`, size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        tag,
      ),
    )
    const dsUbo = track(
      ctx.res.createBuffer(
        { label: `${ctx.id}/bake-ds-ubo`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        tag,
      ),
    )

    const mkTarget = (label: string): GPUTexture =>
      track(
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${label}`,
            size: [SRC_RES, SRC_RES],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
          },
          tag,
        ),
      )
    const srcSurf = mkTarget('bake-src-surf')
    const srcGeom = mkTarget('bake-src-geom')
    const depthTex = track(
      ctx.res.createTexture(
        {
          label: `${ctx.id}/bake-depth`,
          size: [SRC_RES, SRC_RES],
          format: 'depth24plus',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        },
        tag,
      ),
    )
    const mkAtlas = (label: string): GPUTexture =>
      track(
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${label}`,
            size: [RF_ATLAS, RF_ATLAS],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
          },
          tag,
        ),
      )
    const dstSurf = mkAtlas('bake-atlas-surf')
    const dstGeom = mkAtlas('bake-atlas-geom')
    const readback = track(
      ctx.res.createBuffer(
        { label: `${ctx.id}/bake-readback`, size: ATLAS_BYTES * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST },
        tag,
      ),
    )

    const bakeModule = ctx.shaders.module(bakeViewSrc)
    const dsModule = ctx.shaders.module(downsampleSrc)
    const renderPipeline = device.createRenderPipeline({
      label: `${ctx.id}/bake-view`,
      layout: 'auto',
      vertex: {
        module: bakeModule,
        entryPoint: 'vs_bake',
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
        module: bakeModule,
        entryPoint: 'fs_bake',
        targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthCompare: 'less', depthWriteEnabled: true },
    })
    const renderBind = device.createBindGroup({
      label: `${ctx.id}/bake-view-bind`,
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: viewUbo } }],
    })
    const dsPipeline = device.createComputePipeline({
      label: `${ctx.id}/bake-ds`,
      layout: 'auto',
      compute: { module: dsModule, entryPoint: 'cs_downsample' },
    })
    const dsBind = device.createBindGroup({
      label: `${ctx.id}/bake-ds-bind`,
      layout: dsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcSurf.createView() },
        { binding: 1, resource: srcGeom.createView() },
        { binding: 2, resource: dstSurf.createView() },
        { binding: 3, resource: dstGeom.createView() },
        { binding: 4, resource: { buffer: dsUbo } },
      ],
    })

    const ubo = new ArrayBuffer(112)
    const uf = new Float32Array(ubo)
    const dsData = new Uint32Array(4)
    const srcSurfView = srcSurf.createView()
    const srcGeomView = srcGeom.createView()
    const depthView = depthTex.createView()

    for (let vy = 0; vy < RF_VIEWS; vy++) {
      for (let vx = 0; vx < RF_VIEWS; vx++) {
        const dv = octDecode((vx + 0.5) / RF_VIEWS, (vy + 0.5) / RF_VIEWS)
        const { r, u } = viewBasis(dv)
        uf.set(bmin, 0)
        uf[3] = radius
        uf.set(bmax, 4)
        uf.set(r, 8)
        uf.set(u, 12)
        uf.set(dv, 16)
        uf.set(center, 20)
        uf.set(anchor, 24)
        device.queue.writeBuffer(viewUbo, 0, ubo)
        dsData[0] = vx * RF_SLAB
        dsData[1] = vy * RF_SLAB
        dsData[2] = SS
        device.queue.writeBuffer(dsUbo, 0, dsData)

        const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-${vx}-${vy}` })
        const pass = enc.beginRenderPass({
          colorAttachments: [
            { view: srcSurfView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
            { view: srcGeomView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
          ],
          depthStencilAttachment: {
            view: depthView,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            depthClearValue: 1,
          },
        })
        pass.setPipeline(renderPipeline)
        pass.setBindGroup(0, renderBind)
        pass.setVertexBuffer(0, vtx)
        pass.setIndexBuffer(idx, 'uint32')
        pass.drawIndexed(indices.length)
        pass.end()
        const cp = enc.beginComputePass()
        cp.setPipeline(dsPipeline)
        cp.setBindGroup(0, dsBind)
        cp.dispatchWorkgroups(RF_SLAB / 8, RF_SLAB / 8)
        cp.end()
        device.queue.submit([enc.finish()])

        const done = vy * RF_VIEWS + vx + 1
        if (done % 48 === 0 || done === RF_VIEWS * RF_VIEWS) {
          await device.queue.onSubmittedWorkDone()
          onProgress?.(done / (RF_VIEWS * RF_VIEWS), `${speciesId}: ${done}/${RF_VIEWS * RF_VIEWS} views`)
        }
      }
    }

    const enc = device.createCommandEncoder()
    enc.copyTextureToBuffer(
      { texture: dstSurf },
      { buffer: readback, offset: 0, bytesPerRow: RF_ATLAS * 4 },
      [RF_ATLAS, RF_ATLAS],
    )
    enc.copyTextureToBuffer(
      { texture: dstGeom },
      { buffer: readback, offset: ATLAS_BYTES, bytesPerRow: RF_ATLAS * 4 },
      [RF_ATLAS, RF_ATLAS],
    )
    device.queue.submit([enc.finish()])
    await readback.mapAsync(GPUMapMode.READ)

    const out = new ArrayBuffer(HEADER_BYTES + ATLAS_BYTES * 2)
    const head = new DataView(out)
    head.setUint32(0, MAGIC, true)
    head.setUint32(4, BAKE_VERSION, true)
    head.setUint32(8, RF_VIEWS, true)
    head.setUint32(12, RF_SLAB, true)
    head.setFloat32(16, center[0], true)
    head.setFloat32(20, center[1], true)
    head.setFloat32(24, center[2], true)
    head.setFloat32(28, radius, true)
    head.setFloat32(32, h.topH, true)
    new Uint8Array(out, HEADER_BYTES).set(new Uint8Array(readback.getMappedRange()))
    readback.unmap()
    return out
  } finally {
    for (const r of temp) r.destroy()
  }
}
