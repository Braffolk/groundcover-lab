import compareSrc from './compare.wgsl'
import diffStatsSrc from './diff-stats.wgsl'
import { createFullscreenPipeline } from '../gpu/screenPass.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import type { PassTimer } from '../gpu/timing.ts'
import type { Compositor, ViewSlot } from './loop.ts'

export type CompareMode = 'wipe' | 'flicker' | 'diff'

export interface DiffStats {
  /** Mean per-pixel max-channel |A-B| (0..1). */
  mean: number
  /** Fraction of pixels with |A-B| above ~1/255. */
  overFrac: number
}

/**
 * A/B compositor. Both experiments render every frame in every mode (so
 * timings stay symmetric and flicker swaps are instant); this pass picks the
 * pixels. Diff mode also runs a compute reduction whose result surfaces as
 * DiffStats for the HUD — mean 0 on a paused frame is the determinism proof.
 */
export class CompareCompositor implements Compositor {
  mode: CompareMode = 'wipe'
  split = 0.5
  /** Flicker side (0 = A, 1 = B); toggled by Space or auto at 2Hz. */
  side = 0
  autoFlicker = true
  diffGain = 8
  lastDiff: DiffStats | null = null

  private renderPipeline!: GPURenderPipeline
  private statsPipeline!: GPUComputePipeline
  private renderLayout: GPUBindGroupLayout
  private statsLayout: GPUBindGroupLayout
  private sampler: GPUSampler
  private uniform: GPUBuffer
  private statsBuffer: GPUBuffer
  private statsReadback: GPUBuffer
  private statsInFlight = false
  private renderBindGroup: GPUBindGroup | null = null
  private statsBindGroup: GPUBindGroup | null = null
  private pixels = 1
  private frame = 0

  constructor(
    private device: GPUDevice,
    private shaders: ShaderRegistry,
    private format: GPUTextureFormat,
  ) {
    this.renderLayout = device.createBindGroupLayout({
      label: 'compare/render',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })
    this.statsLayout = device.createBindGroupLayout({
      label: 'compare/stats',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.sampler = device.createSampler({ label: 'compare', magFilter: 'linear', minFilter: 'linear' })
    this.uniform = device.createBuffer({
      label: 'compare/uniform',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.statsBuffer = device.createBuffer({
      label: 'compare/stats',
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.statsReadback = device.createBuffer({
      label: 'compare/stats-readback',
      size: 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    this.rebuild()
  }

  rebuild(): void {
    this.renderPipeline = createFullscreenPipeline(this.device, this.shaders, compareSrc, {
      label: 'compare/composite',
      format: this.format,
      bindGroupLayouts: [this.renderLayout],
    })
    this.statsPipeline = this.device.createComputePipeline({
      label: 'compare/stats',
      layout: this.device.createPipelineLayout({ label: 'compare/stats', bindGroupLayouts: [this.statsLayout] }),
      compute: { module: this.shaders.module(diffStatsSrc), entryPoint: 'cs_main' },
    })
  }

  targetsChanged(views: ViewSlot[]): void {
    const [a, b] = views
    if (!a || !b) {
      this.renderBindGroup = null
      this.statsBindGroup = null
      return
    }
    this.pixels = a.color.width * a.color.height
    this.renderBindGroup = this.device.createBindGroup({
      label: 'compare/render',
      layout: this.renderLayout,
      entries: [
        { binding: 0, resource: a.colorView },
        { binding: 1, resource: b.colorView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.uniform } },
      ],
    })
    this.statsBindGroup = this.device.createBindGroup({
      label: 'compare/stats',
      layout: this.statsLayout,
      entries: [
        { binding: 0, resource: a.colorView },
        { binding: 1, resource: b.colorView },
        { binding: 2, resource: { buffer: this.statsBuffer } },
      ],
    })
  }

  composite(enc: GPUCommandEncoder, timer: PassTimer, views: ViewSlot[], canvasView: GPUTextureView): void {
    this.frame++
    if (this.mode === 'flicker' && this.autoFlicker && this.frame % 30 === 0) this.side = 1 - this.side

    const first = views[0]
    if (this.renderBindGroup && first) {
      const modeIndex = this.mode === 'wipe' ? 0 : this.mode === 'flicker' ? 1 : 2
      this.device.queue.writeBuffer(
        this.uniform,
        0,
        new Float32Array([modeIndex, this.split, this.side, this.diffGain]),
      )

      // Diff reduction every 10th frame (readback ~2 frames later).
      if (this.mode === 'diff' && this.statsBindGroup && !this.statsInFlight) {
        enc.clearBuffer(this.statsBuffer)
        const stats = timer.computePass(enc, 'diff-stats')
        stats.setPipeline(this.statsPipeline)
        stats.setBindGroup(0, this.statsBindGroup)
        stats.dispatchWorkgroups(Math.ceil(first.color.width / 8), Math.ceil(first.color.height / 8))
        stats.end()
        enc.copyBufferToBuffer(this.statsBuffer, 0, this.statsReadback, 0, 8)
        this.statsInFlight = true
        queueMicrotask(() => {
          this.statsReadback
            .mapAsync(GPUMapMode.READ)
            .then(() => {
              const [sum, over] = new Uint32Array(this.statsReadback.getMappedRange())
              this.lastDiff = { mean: sum! / 255 / this.pixels, overFrac: over! / this.pixels }
              this.statsReadback.unmap()
              this.statsInFlight = false
            })
            .catch(() => {
              this.statsInFlight = false
            })
        })
      }
    }

    const pass = timer.renderPass(enc, 'composite', {
      colorAttachments: [{ view: canvasView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    })
    if (this.renderBindGroup) {
      pass.setPipeline(this.renderPipeline)
      pass.setBindGroup(0, this.renderBindGroup)
      pass.draw(3)
    }
    pass.end()
  }

  dispose(): void {
    this.uniform.destroy()
    this.statsBuffer.destroy()
    this.statsReadback.destroy()
  }
}
