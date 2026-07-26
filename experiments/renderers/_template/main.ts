import shaderSrc from './shaders/main.wgsl'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Template experiment: renders the ACTIVE STAND — every species entry, at the
 * stand's exact placement — as camera-facing billboard blades. Not a serious
 * technique; a working skeleton showing the wiring: stand-driven scatter,
 * frame group, wind, timed passes, VRAM scope, render-only params, shader hot
 * reload. Replace the technique, keep the contract.
 */
export function create(ctx: ExperimentContext<typeof PARAMS>): Experiment {
  const { device } = ctx

  // One instance buffer per stand entry — the stand defines what exists.
  const entries = ctx.stand.species.map((_, entryIndex) =>
    ctx.scene.scatter.instanceBuffer(ctx.res, device.queue, entryIndex),
  )

  const paramsBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )

  const bglLayout = device.createBindGroupLayout({
    label: `${ctx.id}/bindings`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  })
  const bindGroups = entries.map((entry, i) =>
    device.createBindGroup({
      label: `${ctx.id}/entry-${i}`,
      layout: bglLayout,
      entries: [
        { binding: 0, resource: { buffer: entry.buffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    }),
  )

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/billboards`,
      layout: device.createPipelineLayout({
        label: ctx.id,
        bindGroupLayouts: [ctx.frame.layout, bglLayout],
      }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const uniforms = new Float32Array(4)

  return {
    update(_frame: FrameInfo): void {
      uniforms[0] = ctx.params.widthRatio
      device.queue.writeBuffer(paramsBuffer, 0, uniforms)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'billboards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: {
          view: targets.depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      entries.forEach((entry, i) => {
        if (entry.count === 0) return
        pass.setBindGroup(1, bindGroups[i]!)
        pass.draw(6, entry.count)
      })
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are destroyed by the harness via ctx.res.
    },
  }
}
