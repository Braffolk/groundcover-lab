import type { Mat4 } from 'wgpu-matrix'
import type { VramScope } from '../gpu/resources.ts'
import type { WindParams } from './wind.ts'

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

/**
 * Geometry of the heightfield bound at @group(0) @binding(1), as the Frame
 * struct carries it. A stage that has no terrain still supplies dims for its
 * dummy heightmap — the struct fields are frozen with the layout.
 */
export interface HeightfieldDims {
  /** World extent in metres (square, centred on the origin). */
  size: number
  heightScale: number
  /** Texels per side of the heightmap texture. */
  resolution: number
}

const FRAME_VISIBILITY = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE

/**
 * THE @group(0) bind group layout — ONE declaration, FROZEN.
 *
 * Do not add, remove or reorder bindings, and never introduce a second layout
 * for a different kind of Stage. This layout is a WGSL contract:
 * `src/wgsl/frame.wgsl` declares bindings 0-3 by name and lighting.wgsl,
 * debug.wgsl and terrain.wgsl all `#include` it through an include mechanism
 * (tools/vite-plugin-wgsl.ts) that has NO conditionals — a second layout forks
 * those three shared files and they drift apart. Every experiment's pipelines
 * are built against it too, so a change here invalidates all of them at once.
 *
 * Only the CONTENTS vary per stage. A stage with no terrain supplies a tiny
 * dummy heightmap and a one-row stand buffer (<1KB) rather than change the
 * layout; that is the sanctioned way to add a stage kind.
 */
export const FRAME_LAYOUT_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
  { binding: 0, visibility: FRAME_VISIBILITY, buffer: { type: 'uniform' } },
  { binding: 1, visibility: FRAME_VISIBILITY, texture: { sampleType: 'float' } },
  { binding: 2, visibility: FRAME_VISIBILITY, sampler: { type: 'filtering' } },
  { binding: 3, visibility: FRAME_VISIBILITY, buffer: { type: 'read-only-storage' } },
]

/**
 * Created once per device and shared by the stage (for its base pass), the
 * FrameGroup and every experiment — one object, so pipeline compatibility is
 * identity rather than a structural-equivalence argument.
 */
export function createFrameLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({ label: 'scene/frame', entries: FRAME_LAYOUT_ENTRIES })
}

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
    layout: GPUBindGroupLayout,
    /** Stage-supplied @binding(1) — r=height, g=normal.x, b=normal.z. */
    heightmapView: GPUTextureView,
    /** Stage-supplied @binding(3) — the StandEntry rows. */
    standBuffer: GPUBuffer,
    private dims: HeightfieldDims,
  ) {
    this.buffer = scope.createBuffer(
      { label: 'scene/frame-ubo', size: FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'frame-ubo' },
    )
    this.layout = layout
    this.bindGroup = device.createBindGroup({
      label: 'scene/frame',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.buffer } },
        { binding: 1, resource: heightmapView },
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
    d[84] = this.dims.size
    d[85] = this.dims.heightScale
    d[86] = this.dims.resolution
    d[87] = f.frameIndex
    d[88] = f.debugMode
    queue.writeBuffer(this.buffer, 0, d)
  }
}
