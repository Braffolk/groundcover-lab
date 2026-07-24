import bakeViewsSrc from './shaders/bake_views.wgsl'
import gatherSrc from './shaders/gather.wgsl'
import {
  ATLAS_H,
  ATLAS_W,
  fitFrustum,
  M_E,
  packChordField,
  VIEW_ATLAS_H,
  VIEW_ATLAS_W,
  VIEW_BATCH_ROWS,
  VIEW_GRID,
  VIEW_TILE,
  viewBasis,
  viewDir,
  type FrustumFit,
} from './chords.ts'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Two-stage GPU bake of the chord field:
 *
 *  1. Rasterize the raw GCMESH1 mesh into a full-sphere octahedral grid of
 *     VIEW_GRID^2 orthographic G-buffer captures (albedo+mask, oct normal,
 *     depth) at ~5.5mm texels. This is the "precomputed raycast": the
 *     rasterizer answers every ray query offline. Views are rendered in two
 *     row-batches so the transient G-buffer stays ~130MB.
 *  2. After each batch, a compute pass resamples the resident captures into
 *     the 4D chord atlas: for each (entry bin, exit coord) chord of the
 *     fitted cone frustum it picks the nearest capture direction, runs
 *     parallax-correction iterations against the capture's depth image, and
 *     prefilters with 8 jittered parallel chords (fractional coverage).
 *
 * All resources here are transient (raw device allocations, destroyed at the
 * end); the persistent artifact is the packed chord field.
 */
export async function bakeChordField(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  onNote?: (note: string) => void,
): Promise<ArrayBuffer> {
  const { device } = ctx
  const fit: FrustumFit = fitFrustum(mesh)
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax

  // --- shared transient GPU resources ---------------------------------------
  const vbuf = device.createBuffer({
    label: `${ctx.id}/bake-verts`,
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = device.createBuffer({
    label: `${ctx.id}/bake-idx`,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(ibuf, 0, indices)

  const viewAlbedo = device.createTexture({
    label: `${ctx.id}/bake-view-albedo`,
    size: [VIEW_ATLAS_W, VIEW_ATLAS_H],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })
  const viewOct = device.createTexture({
    label: `${ctx.id}/bake-view-oct`,
    size: [VIEW_ATLAS_W, VIEW_ATLAS_H],
    format: 'rg8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })
  const viewDepth = device.createTexture({
    label: `${ctx.id}/bake-view-depth`,
    size: [VIEW_ATLAS_W, VIEW_ATLAS_H],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })

  const STRIDE = 256
  const nTiles = VIEW_GRID * VIEW_GRID
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * nTiles,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * nTiles)
  const fv = new Float32Array(scratch)
  for (let j = 0; j < VIEW_GRID; j++) {
    for (let i = 0; i < VIEW_GRID; i++) {
      const fwd = viewDir(i, j)
      const { right, up } = viewBasis(fwd)
      const o = ((j * VIEW_GRID + i) * STRIDE) / 4
      fv[o + 0] = right[0]; fv[o + 1] = right[1]; fv[o + 2] = right[2]; fv[o + 3] = fit.rb
      fv[o + 4] = up[0]; fv[o + 5] = up[1]; fv[o + 6] = up[2]
      fv[o + 8] = fwd[0]; fv[o + 9] = fwd[1]; fv[o + 10] = fwd[2]
      fv[o + 12] = 0; fv[o + 13] = fit.h / 2; fv[o + 14] = 0
      fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
      fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
      fv[o + 24] = fit.axisX; fv[o + 25] = fit.axisY; fv[o + 26] = fit.axisZ
    }
  }
  device.queue.writeBuffer(uni, 0, scratch)

  const viewBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-view-bgl`,
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 112 },
    }],
  })
  const viewBg = device.createBindGroup({
    label: `${ctx.id}/bake-view-bg`,
    layout: viewBgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
  })

  const viewModule = ctx.shaders.module(bakeViewsSrc)
  const viewPipeline = device.createRenderPipeline({
    label: `${ctx.id}/bake-view-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-view-pl`, bindGroupLayouts: [viewBgl] }),
    vertex: {
      module: viewModule,
      entryPoint: 'vs',
      buffers: [{
        arrayStride: 16,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'uint16x4' },
          { shaderLocation: 1, offset: 8, format: 'uint16x4' },
        ],
      }],
    },
    fragment: { module: viewModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rg8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  // --- chord field targets + gather pipeline --------------------------------
  const chordSurf = device.createTexture({
    label: `${ctx.id}/bake-chord-surf`,
    size: [ATLAS_W, ATLAS_H],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  })
  const chordGeom = device.createTexture({
    label: `${ctx.id}/bake-chord-geom`,
    size: [ATLAS_W, ATLAS_H],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  })

  const mRange = 1 + fit.capB + fit.capT
  const jr = 0.4 * (fit.sideLen * mRange) / M_E
  const cfgBuf = device.createBuffer({
    label: `${ctx.id}/bake-gather-cfg`,
    size: 256 * 2, // one 256B-aligned slot per batch
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const gatherBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-gather-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
    ],
  })
  const gatherModule = ctx.shaders.module(gatherSrc)
  const gatherPipeline = device.createComputePipeline({
    label: `${ctx.id}/bake-gather-pipe`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-gather-pl`, bindGroupLayouts: [gatherBgl] }),
    compute: { module: gatherModule, entryPoint: 'cs' },
  })

  const nBatches = VIEW_GRID / VIEW_BATCH_ROWS
  for (let batch = 0; batch < nBatches; batch++) {
    const row0 = batch * VIEW_BATCH_ROWS

    // Render this batch's views in chunks so huge meshes cannot TDR.
    const tiles = VIEW_GRID * VIEW_BATCH_ROWS
    const CHUNK = 25
    for (let start = 0; start < tiles; start += CHUNK) {
      const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-views-${batch}-${start}` })
      const first = start === 0
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/bake-views`,
        colorAttachments: [
          { view: viewAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: first ? 'clear' : 'load', storeOp: 'store' },
          { view: viewOct.createView(), clearValue: { r: 0.5, g: 0.5, b: 0, a: 0 }, loadOp: first ? 'clear' : 'load', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: viewDepth.createView(),
          depthClearValue: 1,
          depthLoadOp: first ? 'clear' : 'load',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(viewPipeline)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      for (let t = start; t < Math.min(start + CHUNK, tiles); t++) {
        const i = t % VIEW_GRID
        const j = row0 + Math.floor(t / VIEW_GRID)
        const ty = (j - row0) * VIEW_TILE
        pass.setViewport(i * VIEW_TILE, ty, VIEW_TILE, VIEW_TILE, 0, 1)
        pass.setScissorRect(i * VIEW_TILE, ty, VIEW_TILE, VIEW_TILE)
        pass.setBindGroup(0, viewBg, [(j * VIEW_GRID + i) * STRIDE])
        pass.drawIndexed(indices.length)
      }
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      onNote?.(`batch ${batch + 1}/${nBatches}: views ${Math.min(start + CHUNK, tiles)}/${tiles}`)
    }

    // Gather this batch's share of the chord field.
    const cfgOff = batch * 256
    device.queue.writeBuffer(
      cfgBuf,
      cfgOff,
      new Float32Array([fit.r0, fit.r1, fit.h, fit.sideLen, fit.capB, fit.capT, fit.rb, jr, row0, 0, 0, 0]),
    )
    const gatherBg = device.createBindGroup({
      label: `${ctx.id}/bake-gather-bg-${batch}`,
      layout: gatherBgl,
      entries: [
        { binding: 0, resource: { buffer: cfgBuf, offset: cfgOff, size: 48 } },
        { binding: 1, resource: viewAlbedo.createView() },
        { binding: 2, resource: viewOct.createView() },
        { binding: 3, resource: viewDepth.createView() },
        { binding: 4, resource: chordSurf.createView() },
        { binding: 5, resource: chordGeom.createView() },
      ],
    })
    const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-gather-${batch}` })
    const cpass = enc.beginComputePass({ label: `${ctx.id}/bake-gather` })
    cpass.setPipeline(gatherPipeline)
    cpass.setBindGroup(0, gatherBg)
    cpass.dispatchWorkgroups(Math.ceil(ATLAS_W / 8), Math.ceil(ATLAS_H / 8))
    cpass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    onNote?.(`batch ${batch + 1}/${nBatches}: chords gathered`)
  }

  // --- readback + pack ------------------------------------------------------
  const bpr = ATLAS_W * 4
  const bprAligned = Math.ceil(bpr / 256) * 256
  const rbSize = bprAligned * ATLAS_H
  const rbSurf = device.createBuffer({ label: `${ctx.id}/rb-surf`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const rbGeom = device.createBuffer({ label: `${ctx.id}/rb-geom`, size: rbSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-readback` })
  enc.copyTextureToBuffer({ texture: chordSurf }, { buffer: rbSurf, bytesPerRow: bprAligned, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
  enc.copyTextureToBuffer({ texture: chordGeom }, { buffer: rbGeom, bytesPerRow: bprAligned, rowsPerImage: ATLAS_H }, [ATLAS_W, ATLAS_H])
  device.queue.submit([enc.finish()])

  await rbSurf.mapAsync(GPUMapMode.READ)
  await rbGeom.mapAsync(GPUMapMode.READ)
  const surf = new Uint8Array(ATLAS_W * ATLAS_H * 4)
  const geom = new Uint8Array(ATLAS_W * ATLAS_H * 4)
  const surfRows = new Uint8Array(rbSurf.getMappedRange())
  const geomRows = new Uint8Array(rbGeom.getMappedRange())
  for (let y = 0; y < ATLAS_H; y++) {
    surf.set(surfRows.subarray(y * bprAligned, y * bprAligned + bpr), y * bpr)
    geom.set(geomRows.subarray(y * bprAligned, y * bprAligned + bpr), y * bpr)
  }
  rbSurf.unmap()
  rbGeom.unmap()

  const packed = packChordField({ fit, surf, geom })
  for (const r of [vbuf, ibuf, viewAlbedo, viewOct, viewDepth, uni, cfgBuf, chordSurf, chordGeom, rbSurf, rbGeom]) {
    r.destroy()
  }
  return packed
}
