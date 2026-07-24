import captureShaderSrc from './shaders/capture.wgsl'
import fitShaderSrc from './shaders/fit.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake a species into Fourier coefficient textures, entirely on the GPU.
 *
 * For each of E elevation rings, the source mesh is rendered orthographically
 * from V azimuths into a V*TILE x TILE ring atlas (albedo+coverage, mesh-frame
 * normal + relative depth). A compute pass then projects, per card texel, the
 * angular functions onto a truncated Fourier basis and writes the packed
 * coefficients straight into one layer of four persistent rgba8 array
 * textures. Nothing is ever read back to the CPU; a full species bake is one
 * queue submit of E * V mesh draws + E fit dispatches.
 */

// Must match fit.wgsl (V, TILE) and card.wgsl (RINGS, PHI_MIN/MAX).
export const V_AZIMUTHS = 24
export const TILE = 256
export const RINGS = 4
export const PHI_MIN = 0.13962634 //  8 deg
export const PHI_MAX = 1.43116999 // 82 deg

export interface BakedSpecies {
  coeffA: GPUTexture
  coeffB: GPUTexture
  coeffC: GPUTexture
  coeffD: GPUTexture
  center: [number, number, number]
  radius: number
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

export function bakeSpecies(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  speciesId: string,
): BakedSpecies {
  const { device } = ctx
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const invR = 1 / radius

  // --- persistent coefficient array textures (the whole species footprint) --
  const coeffTex = (tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${tag}`,
        size: [TILE, TILE, RINGS],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      },
      { species: speciesId, tag },
    )
  const coeffA = coeffTex('fourier-a')
  const coeffB = coeffTex('fourier-b')
  const coeffC = coeffTex('fourier-c')
  const coeffD = coeffTex('fourier-d')

  // --- transient bake resources (destroyed right after submit) --------------
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

  const atlasW = V_AZIMUTHS * TILE
  const capColor = device.createTexture({
    label: `${ctx.id}/bake-color`,
    size: [atlasW, TILE],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })
  const capNormal = device.createTexture({
    label: `${ctx.id}/bake-normal`,
    size: [atlasW, TILE],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })
  const capDepth = device.createTexture({
    label: `${ctx.id}/bake-depth`,
    size: [atlasW, TILE],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  // Per-view uniforms: 96B of data padded to 256B for dynamic offsets.
  const STRIDE = 256
  const nViews = RINGS * V_AZIMUTHS
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * nViews,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * nViews)
  const fv = new Float32Array(scratch)
  for (let e = 0; e < RINGS; e++) {
    const phi = PHI_MIN + (e / (RINGS - 1)) * (PHI_MAX - PHI_MIN)
    for (let j = 0; j < V_AZIMUTHS; j++) {
      // MUST match fit.wgsl (theta_j) and card.wgsl (basis construction).
      const theta = (2 * Math.PI * j) / V_AZIMUTHS
      const dir: V3 = [Math.cos(theta) * Math.cos(phi), Math.sin(phi), Math.sin(theta) * Math.cos(phi)]
      const right = norm(cross([0, 1, 0], dir))
      const up = cross(dir, right) as V3
      const o = ((e * V_AZIMUTHS + j) * STRIDE) / 4
      fv[o + 0] = center[0]; fv[o + 1] = center[1]; fv[o + 2] = center[2]; fv[o + 3] = invR
      fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]
      fv[o + 8] = up[0]; fv[o + 9] = up[1]; fv[o + 10] = up[2]
      fv[o + 12] = dir[0]; fv[o + 13] = dir[1]; fv[o + 14] = dir[2]
      fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
      fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
    }
  }
  device.queue.writeBuffer(uni, 0, scratch)

  // --- capture pipeline -----------------------------------------------------
  const capBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 },
      },
    ],
  })
  const capBg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: capBgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 96 } }],
  })
  const capModule = ctx.shaders.module(captureShaderSrc)
  const capPipeline = device.createRenderPipeline({
    label: `${ctx.id}/bake-capture`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/bake-pl`, bindGroupLayouts: [capBgl] }),
    vertex: {
      module: capModule,
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
    fragment: { module: capModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
  })

  // --- fit pipeline ---------------------------------------------------------
  const fitBgl = device.createBindGroupLayout({
    label: `${ctx.id}/fit-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
    ],
  })
  const fitPipeline = device.createComputePipeline({
    label: `${ctx.id}/fit`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/fit-pl`, bindGroupLayouts: [fitBgl] }),
    compute: { module: ctx.shaders.module(fitShaderSrc), entryPoint: 'cs' },
  })
  const layerView = (tex: GPUTexture, e: number): GPUTextureView =>
    tex.createView({ dimension: '2d', baseArrayLayer: e, arrayLayerCount: 1 })
  const fitBindGroups = Array.from({ length: RINGS }, (_, e) =>
    device.createBindGroup({
      label: `${ctx.id}/fit-bg-${e}`,
      layout: fitBgl,
      entries: [
        { binding: 0, resource: capColor.createView() },
        { binding: 1, resource: capNormal.createView() },
        { binding: 2, resource: layerView(coeffA, e) },
        { binding: 3, resource: layerView(coeffB, e) },
        { binding: 4, resource: layerView(coeffC, e) },
        { binding: 5, resource: layerView(coeffD, e) },
      ],
    }),
  )

  // --- one submit: (capture ring -> fit ring) x RINGS -----------------------
  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-${speciesId}` })
  for (let e = 0; e < RINGS; e++) {
    const pass = enc.beginRenderPass({
      label: `${ctx.id}/capture-ring-${e}`,
      colorAttachments: [
        { view: capColor.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: capNormal.createView(), clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0.5 }, loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: {
        view: capDepth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(capPipeline)
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    for (let j = 0; j < V_AZIMUTHS; j++) {
      pass.setViewport(j * TILE, 0, TILE, TILE, 0, 1)
      pass.setScissorRect(j * TILE, 0, TILE, TILE)
      pass.setBindGroup(0, capBg, [(e * V_AZIMUTHS + j) * STRIDE])
      pass.drawIndexed(indices.length)
    }
    pass.end()

    const fit = enc.beginComputePass({ label: `${ctx.id}/fit-ring-${e}` })
    fit.setPipeline(fitPipeline)
    fit.setBindGroup(0, fitBindGroups[e]!)
    fit.dispatchWorkgroups(TILE / 8, TILE / 8)
    fit.end()
  }
  device.queue.submit([enc.finish()])

  // Safe immediately after submit — WebGPU defers actual release.
  for (const r of [vbuf, ibuf, capColor, capNormal, capDepth, uni]) r.destroy()

  return { coeffA, coeffB, coeffC, coeffD, center, radius }
}
