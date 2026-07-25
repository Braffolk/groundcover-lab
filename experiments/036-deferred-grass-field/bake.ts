import stampSrc from './shaders/stamp.wgsl'
import compositeSrc from './shaders/composite.wgsl'
import reduceSrc from './shaders/reduce.wgsl'
import { bakedArtifact, commitBake, speciesById, type ExperimentContext, type GcMesh } from '@harness'
import {
  AZ,
  ELEV,
  GEOM_DIV,
  GEOM_PX,
  SS,
  TILE_M,
  TILE_PX,
  VERSION,
  YAWS,
  packField,
  shearOfCell,
  unpackField,
  type FieldTable,
} from './field.ts'

/**
 * The bake: turn the ACTIVE STAND into a table of ray answers.
 *
 * Stage A — stamp library. Each species' source mesh is rasterized once per
 * (yaw, elevation) with a SHEARED orthographic projection, i.e. one image per
 * ray bundle. 3 species x 16 yaws x 6 elevations = 288 mesh draws.
 *
 * Stage B — field composite. For each of ELEV x AZ directions, the stand's own
 * scatter (a genuine TILE_M x TILE_M realisation, terrain flattened) is
 * composited from those stamps with a depth test on the HIT HEIGHT, at integer
 * tile offsets so the answer is periodic. 96 directions, ~14k stamp quads each.
 *
 * The result is reduced (coverage-weighted, premultiplied) into two slabs per
 * elevation and read back. Nothing about the runtime cost depends on how many
 * plants the stand has; only the bake sees plants at all.
 */

const STAMP_TEXEL = TILE_M / (TILE_PX * SS)
const GUTTER = 2
const MAX_ATLAS = 8192

type BakeCtx = Pick<ExperimentContext, 'id' | 'device' | 'meshes' | 'shaders' | 'res' | 'scene' | 'stand' | 'seed'>

interface SpeciesGeom {
  id: string
  /** Mesh centre in xz (the stamp is centred here). */
  cx: number
  cz: number
  /** Exact horizontal support radius (m, scale 1). */
  r: number
  topH: number
  y0: number
  qmin: [number, number, number]
  qsize: [number, number, number]
}

interface StampAtlas {
  surf: GPUTexture
  geom: GPUTexture
  tileW: number
  tileH: number
  atlasW: number
  atlasH: number
  /** Tiles per atlas row — as many as fit; the low-elevation strips are long. */
  cols: number
  /** Sheared-ortho rect in the plant frame: u0, u1, v0, v1 (m at scale 1). */
  rect: [number, number, number, number]
}

export function bakeKey(ctx: Pick<BakeCtx, 'stand' | 'seed'>): string {
  return `field-v${VERSION}-${ctx.stand.id}-s${ctx.seed}-t${TILE_PX}a${AZ}e${ELEV}`
}

export async function loadFieldTable(
  ctx: BakeCtx,
  onProgress?: (note: string) => void,
): Promise<FieldTable> {
  const key = bakeKey(ctx)
  let freshlyBaked = false
  const run = async (): Promise<ArrayBuffer> => {
    freshlyBaked = true
    return bakeField(ctx, onProgress)
  }
  let buf = await bakedArtifact({ expId: ctx.id, key }, run)
  let table = unpackField(buf)
  if (!table) {
    // A missing committed file answers 200 with index.html under the SPA
    // fallback, which poisons the OPFS cache — rebake and repair.
    buf = await run()
    table = unpackField(buf)
    if (!table) throw new Error(`[${ctx.id}] bake produced an invalid artifact`)
    await opfsRepair(`${ctx.id}__${key}`, buf)
  }
  if (freshlyBaked) {
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
    /* best effort */
  }
}

/** Canopy height the table normalizes hit heights by (tallest plant possible). */
export function canopyHeight(ctx: Pick<BakeCtx, 'stand'>): number {
  let h = 0.2
  for (const e of ctx.stand.species) {
    h = Math.max(h, speciesById(e.species).heightScale * e.scaleMax)
  }
  return h
}

function measureMesh(mesh: GcMesh, id: string): SpeciesGeom {
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
  return {
    id,
    cx,
    cz,
    r: Math.sqrt(r2) * 1.01 + 1e-3,
    topH: Math.max(by1, 0.02),
    y0: Math.min(0, by0),
    qmin: [bx0, by0, bz0],
    qsize: [bx1 - bx0, by1 - by0, bz1 - bz0],
  }
}

async function bakeField(ctx: BakeCtx, onProgress?: (note: string) => void): Promise<ArrayBuffer> {
  const { device } = ctx
  const H = canopyHeight(ctx)
  const note = (s: string): void => {
    onProgress?.(s)
    console.info(`[${ctx.id}] bake: ${s}`)
  }

  // --- the stand's own plants, over one periodic tile --------------------
  const entries = ctx.stand.species.map((entry, i) => {
    const pts = ctx.scene.scatter
      .region(i, { minX: 0, minZ: 0, maxX: TILE_M, maxZ: TILE_M })
      .filter((p) => p.x >= 0 && p.x < TILE_M && p.z >= 0 && p.z < TILE_M)
    const data = new Float32Array(Math.max(pts.length, 1) * 4)
    pts.forEach((p, k) => data.set([p.x, p.z, p.scale, p.yaw], k * 4))
    return { speciesId: entry.species, count: pts.length, data, scaleMax: entry.scaleMax }
  })
  note(`${entries.map((e) => `${e.speciesId}:${e.count}`).join(' ')} plants in a ${TILE_M}m tile`)

  const speciesIds = [...new Set(entries.map((e) => e.speciesId))]
  const geoms = new Map<string, SpeciesGeom>()
  for (const id of speciesIds) {
    const mesh = await ctx.meshes.load(speciesById(id).meshId)
    geoms.set(id, measureMesh(mesh, id))
  }

  const scratch: (GPUBuffer | GPUTexture)[] = []
  const track = <T extends GPUBuffer | GPUTexture>(r: T): T => {
    scratch.push(r)
    return r
  }

  // --- Stage A: stamp library ------------------------------------------------
  const stampModule = ctx.shaders.module(stampSrc)
  const stampBgl = device.createBindGroupLayout({
    label: `${ctx.id}/stamp-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
    ],
  })
  const stampPipeline = device.createRenderPipeline({
    label: `${ctx.id}/stamp`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/stamp-pl`, bindGroupLayouts: [stampBgl] }),
    vertex: {
      module: stampModule,
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
      module: stampModule,
      entryPoint: 'fs',
      targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const stamps = new Map<string, StampAtlas[]>()
  for (const id of speciesIds) {
    const g = geoms.get(id)!
    const mesh = await ctx.meshes.load(speciesById(id).meshId)
    const vbuf = track(
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/bake-verts-${id}`,
          size: mesh.vertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        },
        { tag: 'bake-scratch' },
      ),
    )
    device.queue.writeBuffer(vbuf, 0, mesh.vertices)
    const indices = mesh.indices()
    const ibuf = track(
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/bake-idx-${id}`,
          size: indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        },
        { tag: 'bake-scratch' },
      ),
    )
    device.queue.writeBuffer(ibuf, 0, indices)

    const perSpecies: StampAtlas[] = []
    for (let e = 0; e < ELEV; e++) {
      const s = shearOfCell(e)
      const rect: [number, number, number, number] = [
        -g.r + s * g.y0,
        g.r + s * g.topH,
        -g.r,
        g.r,
      ]
      const tileW = Math.ceil((rect[1] - rect[0]) / STAMP_TEXEL)
      const tileH = Math.ceil((rect[3] - rect[2]) / STAMP_TEXEL)
      // Pack the yaw variants into as square an atlas as the device allows: at
      // the grazing cells one stamp is a 12 m streak, i.e. >2000 texels wide.
      const cols = Math.max(
        1,
        Math.min(
          YAWS,
          Math.floor(MAX_ATLAS / (tileW + GUTTER * 2)),
          Math.max(1, Math.round(Math.sqrt((YAWS * (tileH + GUTTER * 2)) / (tileW + GUTTER * 2)))),
        ),
      )
      const atlasW = cols * (tileW + GUTTER * 2)
      const atlasH = Math.ceil(YAWS / cols) * (tileH + GUTTER * 2)
      if (atlasW > MAX_ATLAS || atlasH > MAX_ATLAS) {
        throw new Error(`[${ctx.id}] stamp atlas ${atlasW}x${atlasH} exceeds ${MAX_ATLAS} (species ${id}, e${e})`)
      }
      const mk = (label: string): GPUTexture =>
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${label}`,
            size: [atlasW, atlasH],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
          },
          { tag: 'bake-scratch' },
        )
      const atlas: StampAtlas = {
        surf: track(mk(`stamp-surf-${id}-e${e}`)),
        geom: track(mk(`stamp-geom-${id}-e${e}`)),
        tileW,
        tileH,
        atlasW,
        atlasH,
        cols,
        rect,
      }
      const depth = ctx.res.createTexture(
        {
          label: `${ctx.id}/stamp-depth`,
          size: [atlasW, atlasH],
          format: 'depth32float',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        },
        { tag: 'bake-scratch' },
      )

      // One uniform per yaw, dynamic-offset addressed.
      const stride = 256
      const uni = ctx.res.createBuffer(
        {
          label: `${ctx.id}/stamp-uni`,
          size: stride * YAWS,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        },
        { tag: 'bake-scratch' },
      )
      const cfg = new Float32Array((stride / 4) * YAWS)
      for (let y = 0; y < YAWS; y++) {
        const o = (y * stride) / 4
        const yaw = (y * 2 * Math.PI) / YAWS
        cfg.set(rect, o)
        cfg.set([s, yaw, Math.cos(yaw), Math.sin(yaw)], o + 4)
        cfg.set([g.topH, g.y0, 1 / g.topH, 0], o + 8)
        cfg.set([...g.qmin, 0], o + 12)
        cfg.set([...g.qsize, 0], o + 16)
        cfg.set([g.cx, g.cz, 0, 0], o + 20)
      }
      device.queue.writeBuffer(uni, 0, cfg)
      const bg = device.createBindGroup({
        label: `${ctx.id}/stamp-bg`,
        layout: stampBgl,
        entries: [{ binding: 0, resource: { buffer: uni, size: 96 } }],
      })

      const enc = device.createCommandEncoder({ label: `${ctx.id}/stamp-${id}-e${e}` })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/stamp-pass`,
        colorAttachments: [
          { view: atlas.surf.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          { view: atlas.geom.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: depth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(stampPipeline)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      for (let y = 0; y < YAWS; y++) {
        const col = y % cols
        const row = Math.floor(y / cols)
        const x0 = col * (tileW + GUTTER * 2) + GUTTER
        const y0 = row * (tileH + GUTTER * 2) + GUTTER
        pass.setViewport(x0, y0, tileW, tileH, 0, 1)
        pass.setScissorRect(x0, y0, tileW, tileH)
        pass.setBindGroup(0, bg, [y * stride])
        pass.drawIndexed(indices.length)
      }
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      depth.destroy()
      uni.destroy()
      perSpecies.push(atlas)
      note(`stamps ${id} e${e} (${tileW}x${tileH} x${YAWS})`)
    }
    stamps.set(id, perSpecies)
    vbuf.destroy()
    ibuf.destroy()
  }

  // --- Stage B: field composite --------------------------------------------
  const compModule = ctx.shaders.module(compositeSrc)
  const compBgl = device.createBindGroupLayout({
    label: `${ctx.id}/comp-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', minBindingSize: 64 } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const compPipeline = device.createRenderPipeline({
    label: `${ctx.id}/composite`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/comp-pl`, bindGroupLayouts: [compBgl] }),
    vertex: { module: compModule, entryPoint: 'vs' },
    fragment: {
      module: compModule,
      entryPoint: 'fs',
      targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  const SSPX = TILE_PX * SS
  const compSurf = track(
    ctx.res.createTexture(
      {
        label: `${ctx.id}/comp-surf`,
        size: [SSPX, SSPX],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      },
      { tag: 'bake-scratch' },
    ),
  )
  const compGeom = track(
    ctx.res.createTexture(
      {
        label: `${ctx.id}/comp-geom`,
        size: [SSPX, SSPX],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      },
      { tag: 'bake-scratch' },
    ),
  )
  const compDepth = track(
    ctx.res.createTexture(
      {
        label: `${ctx.id}/comp-depth`,
        size: [SSPX, SSPX],
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      },
      { tag: 'bake-scratch' },
    ),
  )

  const slabSurf = track(
    ctx.res.createTexture(
      {
        label: `${ctx.id}/slab-surf`,
        size: [TILE_PX, TILE_PX, AZ],
        dimension: '3d',
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    ),
  )
  const slabGeom = track(
    ctx.res.createTexture(
      {
        label: `${ctx.id}/slab-geom`,
        size: [GEOM_PX, GEOM_PX, AZ],
        dimension: '3d',
        format: 'rgba8snorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
      },
      { tag: 'bake-scratch' },
    ),
  )
  // Direction uniforms (one per elevation x azimuth), dynamic-offset addressed.
  const dirStride = 256
  const dirBuf = track(
    ctx.res.createBuffer(
      {
        label: `${ctx.id}/dir-uni`,
        size: dirStride * ELEV * AZ,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { tag: 'bake-scratch' },
    ),
  )
  const perAxis: number[] = []
  {
    const dirData = new Float32Array((dirStride / 4) * ELEV * AZ)
    for (let e = 0; e < ELEV; e++) {
      const s = shearOfCell(e)
      let extent = 0
      for (const ent of entries) {
        const g = geoms.get(ent.speciesId)!
        extent = Math.max(extent, (2 * g.r + s * g.topH) * ent.scaleMax)
      }
      const w = Math.ceil(extent / TILE_M)
      perAxis.push(2 * w + 1)
      for (let a = 0; a < AZ; a++) {
        const az = (a * 2 * Math.PI) / AZ
        const o = ((e * AZ + a) * dirStride) / 4
        dirData.set([Math.cos(az), Math.sin(az), s, H], o)
        dirData.set([TILE_M, 2 * w + 1, w, 0], o + 4)
      }
    }
    device.queue.writeBuffer(dirBuf, 0, dirData)
  }

  // Per (entry, elevation) uniforms + bind groups.
  const entryStride = 256
  const entryBuf = track(
    ctx.res.createBuffer(
      {
        label: `${ctx.id}/entry-uni`,
        size: entryStride * entries.length * ELEV,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { tag: 'bake-scratch' },
    ),
  )
  const plantBufs = entries.map((ent, i) => {
    const buf = track(
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/plants-${i}`,
          size: ent.data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        },
        { tag: 'bake-scratch' },
      ),
    )
    device.queue.writeBuffer(buf, 0, ent.data)
    return buf
  })
  const stampSampler = device.createSampler({
    label: `${ctx.id}/stamp-samp`,
    magFilter: 'linear',
    minFilter: 'linear',
  })
  const entryData = new Float32Array((entryStride / 4) * entries.length * ELEV)
  const compBgs: GPUBindGroup[][] = []
  for (let e = 0; e < ELEV; e++) {
    const row: GPUBindGroup[] = []
    entries.forEach((ent, i) => {
      const atlas = stamps.get(ent.speciesId)![e]!
      const g = geoms.get(ent.speciesId)!
      const o = ((e * entries.length + i) * entryStride) / 4
      entryData.set(atlas.rect, o)
      entryData.set([atlas.tileW, atlas.tileH, atlas.atlasW, atlas.atlasH], o + 4)
      entryData.set([Math.max(ent.count, 1), g.topH, YAWS, 0], o + 8)
      entryData.set([atlas.cols, GUTTER, 0, 0], o + 12)
      row.push(
        device.createBindGroup({
          label: `${ctx.id}/comp-bg-e${e}-${i}`,
          layout: compBgl,
          entries: [
            { binding: 0, resource: { buffer: dirBuf, size: 32 } },
            {
              binding: 1,
              resource: { buffer: entryBuf, offset: (e * entries.length + i) * entryStride, size: 64 },
            },
            { binding: 2, resource: { buffer: plantBufs[i]! } },
            { binding: 3, resource: atlas.surf.createView() },
            { binding: 4, resource: atlas.geom.createView() },
            { binding: 5, resource: stampSampler },
          ],
        }),
      )
    })
    compBgs.push(row)
  }
  device.queue.writeBuffer(entryBuf, 0, entryData)

  // Reduce pipelines.
  const reduceModule = ctx.shaders.module(reduceSrc)
  const reduceBgl = device.createBindGroupLayout({
    label: `${ctx.id}/reduce-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 },
      },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '3d' },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8snorm', viewDimension: '3d' },
      },
    ],
  })
  const reduceLayout = device.createPipelineLayout({
    label: `${ctx.id}/reduce-pl`,
    bindGroupLayouts: [reduceBgl],
  })
  const mkReduce = (entryPoint: string): GPUComputePipeline =>
    device.createComputePipeline({
      label: `${ctx.id}/${entryPoint}`,
      layout: reduceLayout,
      compute: { module: reduceModule, entryPoint },
    })
  const pipeSurf = mkReduce('reduce_surf')
  const pipeGeom = mkReduce('reduce_geom')

  const rcStride = 256
  const rcBuf = track(
    ctx.res.createBuffer(
      { label: `${ctx.id}/reduce-cfg`, size: rcStride * AZ * 2, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'bake-scratch' },
    ),
  )
  {
    const rc = new Uint32Array((rcStride / 4) * AZ * 2)
    for (let a = 0; a < AZ; a++) {
      rc.set([a, SS, 0, 0], ((a * 2 + 0) * rcStride) / 4)
      rc.set([a, SS * GEOM_DIV, 0, 0], ((a * 2 + 1) * rcStride) / 4)
    }
    device.queue.writeBuffer(rcBuf, 0, rc)
  }
  const reduceBg = device.createBindGroup({
    label: `${ctx.id}/reduce-bg`,
    layout: reduceBgl,
    entries: [
      { binding: 0, resource: { buffer: rcBuf, size: 16 } },
      { binding: 1, resource: compSurf.createView() },
      { binding: 2, resource: compGeom.createView() },
      { binding: 3, resource: slabSurf.createView({ dimension: '3d' }) },
      { binding: 4, resource: slabGeom.createView({ dimension: '3d' }) },
    ],
  })

  // Readback buffers.
  const surfBytesTotal = TILE_PX * TILE_PX * AZ * 4
  const geomBytesTotal = GEOM_PX * GEOM_PX * AZ * 4
  const rbSurf = track(
    ctx.res.createBuffer(
      { label: `${ctx.id}/rb-surf`, size: surfBytesTotal, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    ),
  )
  const rbGeom = track(
    ctx.res.createBuffer(
      { label: `${ctx.id}/rb-geom`, size: geomBytesTotal, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ },
      { tag: 'bake-scratch' },
    ),
  )
  const surfSlabs: Uint8Array<ArrayBuffer>[] = []
  const geomSlabs: Uint8Array<ArrayBuffer>[] = []

  for (let e = 0; e < ELEV; e++) {
    const enc = device.createCommandEncoder({ label: `${ctx.id}/field-e${e}` })
    for (let a = 0; a < AZ; a++) {
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/comp-e${e}-a${a}`,
        colorAttachments: [
          { view: compSurf.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          { view: compGeom.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: compDepth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(compPipeline)
      entries.forEach((ent, i) => {
        if (ent.count === 0) return
        pass.setBindGroup(0, compBgs[e]![i]!, [(e * AZ + a) * dirStride])
        pass.draw(6, ent.count * perAxis[e]! * perAxis[e]!)
      })
      pass.end()

      const cp = enc.beginComputePass({ label: `${ctx.id}/reduce-e${e}-a${a}` })
      cp.setPipeline(pipeSurf)
      cp.setBindGroup(0, reduceBg, [(a * 2 + 0) * rcStride])
      cp.dispatchWorkgroups(Math.ceil(TILE_PX / 8), Math.ceil(TILE_PX / 8))
      cp.setPipeline(pipeGeom)
      cp.setBindGroup(0, reduceBg, [(a * 2 + 1) * rcStride])
      cp.dispatchWorkgroups(Math.ceil(GEOM_PX / 8), Math.ceil(GEOM_PX / 8))
      cp.end()
    }
    enc.copyTextureToBuffer(
      { texture: slabSurf },
      { buffer: rbSurf, bytesPerRow: TILE_PX * 4, rowsPerImage: TILE_PX },
      [TILE_PX, TILE_PX, AZ],
    )
    enc.copyTextureToBuffer(
      { texture: slabGeom },
      { buffer: rbGeom, bytesPerRow: GEOM_PX * 4, rowsPerImage: GEOM_PX },
      [GEOM_PX, GEOM_PX, AZ],
    )
    device.queue.submit([enc.finish()])

    await rbSurf.mapAsync(GPUMapMode.READ)
    surfSlabs.push(new Uint8Array(rbSurf.getMappedRange()).slice() as Uint8Array<ArrayBuffer>)
    rbSurf.unmap()
    await rbGeom.mapAsync(GPUMapMode.READ)
    geomSlabs.push(new Uint8Array(rbGeom.getMappedRange()).slice() as Uint8Array<ArrayBuffer>)
    rbGeom.unmap()
    note(`field elevation ${e + 1}/${ELEV} (shear ${shearOfCell(e).toFixed(2)}, ${perAxis[e]! ** 2} wrap copies)`)
  }

  for (const r of scratch) r.destroy()
  for (const list of stamps.values()) for (const a of list) { a.surf.destroy(); a.geom.destroy() }

  return packField({
    tilePx: TILE_PX,
    geomPx: GEOM_PX,
    az: AZ,
    elev: ELEV,
    canopyH: H,
    tileM: TILE_M,
    surf: surfSlabs,
    geom: geomSlabs,
  })
}
