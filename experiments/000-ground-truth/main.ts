import shaderSrc from './shaders/ground-truth.wgsl'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Brute-force reference render of the raw Calamagrostis GCMESH1 tile,
 * periodically repeated (its native semantic) and placed on the shared
 * terrain, with the shared wind applied per vertex. Deliberately violates
 * every performance rule — it exists to be the visual truth in A/B.
 */
export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const speciesId = 'calamagrostis-canescens'
  const mesh = await ctx.meshes.load(speciesId)

  const vertexBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/vertices`, size: mesh.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'source-mesh' },
  )
  device.queue.writeBuffer(vertexBuffer, 0, mesh.vertices)
  const indices = mesh.indices()
  const indexBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/indices`, size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST },
    { species: speciesId, tag: 'source-mesh' },
  )
  device.queue.writeBuffer(indexBuffer, 0, indices)

  const paramsBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: 12 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { species: speciesId },
  )
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bindings`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  })
  const bindGroup = device.createBindGroup({
    label: `${ctx.id}/bindings`,
    layout: bgl,
    entries: [{ binding: 0, resource: { buffer: paramsBuffer } }],
  })

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/mesh`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
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
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const uniforms = new Float32Array(12)

  return {
    update(_frame: FrameInfo): void {
      const h = mesh.header
      uniforms.set(h.boundsMin, 0)
      uniforms[3] = Math.round(ctx.params.tiles)
      uniforms.set(h.boundsMax, 4)
      uniforms[7] = ctx.params.sway
      uniforms.set(h.tileSize, 8)
      uniforms[10] = h.topH
      device.queue.writeBuffer(paramsBuffer, 0, uniforms)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const tiles = Math.round(ctx.params.tiles)
      const pass = ctx.timing.renderPass(enc, 'ground-truth', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, bindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.setIndexBuffer(indexBuffer, 'uint32')
      pass.drawIndexed(indices.length, tiles * tiles)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
