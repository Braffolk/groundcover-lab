import type { ShaderRegistry, WgslSource } from './shaders.ts'

/**
 * Helper for fullscreen-triangle passes (sky, composite, diff, blit). The
 * shader must `#include "src/wgsl/fullscreen.wgsl"` and define `fs_main`.
 */
export function createFullscreenPipeline(
  device: GPUDevice,
  shaders: ShaderRegistry,
  src: WgslSource,
  opts: {
    label: string
    format: GPUTextureFormat
    bindGroupLayouts: GPUBindGroupLayout[]
    depth?: { format: GPUTextureFormat; depthCompare: GPUCompareFunction; depthWriteEnabled: boolean }
    constants?: Record<string, number>
  },
): GPURenderPipeline {
  const module = shaders.module(src)
  return device.createRenderPipeline({
    label: opts.label,
    layout: device.createPipelineLayout({ label: opts.label, bindGroupLayouts: opts.bindGroupLayouts }),
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [{ format: opts.format }],
      ...(opts.constants && { constants: opts.constants }),
    },
    primitive: { topology: 'triangle-list' },
    ...(opts.depth && {
      depthStencil: {
        format: opts.depth.format,
        depthCompare: opts.depth.depthCompare,
        depthWriteEnabled: opts.depth.depthWriteEnabled,
      },
    }),
  })
}
