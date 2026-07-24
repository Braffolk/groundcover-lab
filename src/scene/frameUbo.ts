import type { Mat4 } from 'wgpu-matrix'
import type { VramScope } from '../gpu/resources.ts'
import type { WindParams } from './wind.ts'
import type { Terrain } from './terrain.ts'

/**
 * The shared @group(0) every pass binds — layout mirrors `struct Frame` in
 * src/wgsl/frame.wgsl (352 bytes / 88 floats).
 */

export interface FrameData {
  view: Mat4
  proj: Mat4
  viewProj: Mat4
  invViewProj: Mat4
  cameraPos: [number, number, number]
  time: number
  dt: number
  frameIndex: number
  wind: WindParams
  viewport: [number, number]
  /** Index into DEBUG_VIEW_MODES — see src/wgsl/debug.wgsl. */
  debugMode: number
}

export const SUN_DIR: [number, number, number] = (() => {
  const v: [number, number, number] = [0.35, 0.75, 0.3]
  const len = Math.hypot(...v)
  return [v[0] / len, v[1] / len, v[2] / len]
})()
export const SUN_COLOR: [number, number, number] = [1.15, 1.02, 0.82]
export const AMBIENT: [number, number, number] = [0.21, 0.25, 0.32]

// 89 used + 3 pad — struct alignment rounds to a multiple of 16 bytes.
const FLOATS = 92

export class FrameGroup {
  readonly layout: GPUBindGroupLayout
  readonly bindGroup: GPUBindGroup
  readonly buffer: GPUBuffer
  /** WGSL declarations for this group — `#include "src/wgsl/frame.wgsl"`. */
  readonly wgslInclude = 'src/wgsl/frame.wgsl'
  private data = new Float32Array(FLOATS)

  constructor(
    device: GPUDevice,
    scope: VramScope,
    private terrain: Terrain,
    standBuffer: GPUBuffer,
  ) {
    this.buffer = scope.createBuffer(
      { label: 'scene/frame-ubo', size: FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'frame-ubo' },
    )
    const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE
    this.layout = device.createBindGroupLayout({
      label: 'scene/frame',
      entries: [
        { binding: 0, visibility, buffer: { type: 'uniform' } },
        { binding: 1, visibility, texture: { sampleType: 'float' } },
        { binding: 2, visibility, sampler: { type: 'filtering' } },
        { binding: 3, visibility, buffer: { type: 'read-only-storage' } },
      ],
    })
    this.bindGroup = device.createBindGroup({
      label: 'scene/frame',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.buffer } },
        { binding: 1, resource: terrain.texture.createView() },
        { binding: 2, resource: device.createSampler({ label: 'scene/linear', magFilter: 'linear', minFilter: 'linear' }) },
        { binding: 3, resource: { buffer: standBuffer } },
      ],
    })
  }

  write(queue: GPUQueue, f: FrameData): void {
    const d = this.data
    d.set(f.view, 0)
    d.set(f.proj, 16)
    d.set(f.viewProj, 32)
    d.set(f.invViewProj, 48)
    d.set(f.cameraPos, 64)
    d[67] = f.time
    d.set(SUN_DIR, 68)
    d[71] = f.dt
    d.set(SUN_COLOR, 72)
    d[75] = f.wind.strength
    d.set(AMBIENT, 76)
    d[79] = f.wind.gustFreq
    d.set(f.wind.dir, 80)
    d.set(f.viewport, 82)
    d[84] = this.terrain.desc.size
    d[85] = this.terrain.desc.heightScale
    d[86] = this.terrain.desc.resolution
    d[87] = f.frameIndex
    d[88] = f.debugMode
    queue.writeBuffer(this.buffer, 0, d)
  }
}
