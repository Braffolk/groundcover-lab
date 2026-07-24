import terrainDrawSrc from './terrain-draw.wgsl'
import skySrc from './sky.wgsl'
import { createFullscreenPipeline } from '../gpu/screenPass.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import type { PassTimer } from '../gpu/timing.ts'

const TERRAIN_QUADS = 256 // must match QUADS in terrain-draw.wgsl

/** Fog/horizon clear color — matches sky_horizon_color() in lighting.wgsl. */
export const CLEAR_COLOR: GPUColor = { r: 0.62, g: 0.72, b: 0.85, a: 1 }

/**
 * The harness-owned base pass: clears color+depth, draws terrain then sky.
 * Identical for every experiment — experiments never touch it, which is what
 * makes A/B comparisons honest.
 */
export class BasePass {
  private terrainPipeline!: GPURenderPipeline
  private skyPipeline!: GPURenderPipeline

  constructor(
    private device: GPUDevice,
    private shaders: ShaderRegistry,
    private frameLayout: GPUBindGroupLayout,
    private colorFormat: GPUTextureFormat,
    private depthFormat: GPUTextureFormat,
  ) {
    this.rebuild()
  }

  /** (Re)create pipelines — called on construction and on shader hot reload. */
  rebuild(): void {
    const module = this.shaders.module(terrainDrawSrc)
    this.terrainPipeline = this.device.createRenderPipeline({
      label: 'base/terrain',
      layout: this.device.createPipelineLayout({ label: 'base/terrain', bindGroupLayouts: [this.frameLayout] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: this.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    this.skyPipeline = createFullscreenPipeline(this.device, this.shaders, skySrc, {
      label: 'base/sky',
      format: this.colorFormat,
      bindGroupLayouts: [this.frameLayout],
      depth: { format: this.depthFormat, depthCompare: 'less-equal', depthWriteEnabled: false },
    })
  }

  encode(
    enc: GPUCommandEncoder,
    timing: PassTimer,
    frameBindGroup: GPUBindGroup,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
  ): void {
    const pass = timing.renderPass(enc, 'base', {
      colorAttachments: [{ view: colorView, loadOp: 'clear', clearValue: CLEAR_COLOR, storeOp: 'store' }],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: 'clear',
        depthClearValue: 1,
        depthStoreOp: 'store',
      },
    })
    pass.setBindGroup(0, frameBindGroup)
    pass.setPipeline(this.terrainPipeline)
    pass.draw(TERRAIN_QUADS * TERRAIN_QUADS * 6)
    pass.setPipeline(this.skyPipeline)
    pass.draw(3)
    pass.end()
  }
}
