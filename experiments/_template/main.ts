import shaderSrc from './shaders/main.wgsl'
import { speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Template experiment: camera-facing billboard blades fed by the shared
 * scatter service. Not a serious technique — a working skeleton showing every
 * harness facility (scatter, frame group, wind, timed passes, VRAM scope,
 * params, shader hot reload). Replace the technique, keep the wiring.
 */
export function create(ctx: ExperimentContext<typeof PARAMS>): Experiment {
  const { device } = ctx
  const species = speciesById('grass-blade')
  // Honor the shared workload: cover ctx.coverage.radius at the shared
  // density scale (times this experiment's own density param).
  const r = ctx.coverage.radius
  const region = { minX: -r, minZ: -r, maxX: r, maxZ: r }
  const densityScale = (): number => ctx.params.density * ctx.coverage.densityScale

  let instances = ctx.scene.scatter.instanceBuffer(ctx.res, device.queue, species, region, densityScale())

  const paramsBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { species: species.id },
  )

  const bglLayout = device.createBindGroupLayout({
    label: `${ctx.id}/bindings`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  })

  let bindGroup!: GPUBindGroup
  const makeBindGroup = (): void => {
    bindGroup = device.createBindGroup({
      label: `${ctx.id}/bindings`,
      layout: bglLayout,
      entries: [
        { binding: 0, resource: { buffer: instances.buffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    })
  }
  makeBindGroup()

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
      uniforms[0] = ctx.params.height
      uniforms[1] = ctx.params.sway
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
      pass.setBindGroup(1, bindGroup)
      pass.draw(6, instances.count)
      pass.end()
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      if (keys.has('density')) {
        instances.buffer.destroy()
        instances = ctx.scene.scatter.instanceBuffer(ctx.res, device.queue, species, region, densityScale())
        makeBindGroup()
      }
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are destroyed by the harness via ctx.res.
    },
  }
}
