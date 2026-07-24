export interface GpuInfo {
  vendor: string
  architecture: string
  description: string
  /** Normalized short id for filenames, e.g. "apple-common-3". */
  slug: string
}

export interface Gpu {
  adapter: GPUAdapter
  device: GPUDevice
  /** timestamp-query granted — per-pass GPU timing available. */
  hasTimestamps: boolean
  /** Preferred canvas format; also used for all offscreen color targets. */
  format: GPUTextureFormat
  info: GpuInfo
}

export interface GpuHandlers {
  onError: (message: string) => void
  onDeviceLost: (reason: string) => void
}

export async function createGpu(handlers: GpuHandlers): Promise<Gpu> {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser (need Chrome/Edge 113+, or enable the flag).')
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('No WebGPU adapter available.')

  const hasTimestamps = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamps ? ['timestamp-query'] : [],
  })

  device.onuncapturederror = (ev) => handlers.onError(ev.error.message)
  void device.lost.then((info) => {
    if (info.reason !== 'destroyed') handlers.onDeviceLost(`${info.reason}: ${info.message}`)
  })

  const raw = adapter.info
  const info: GpuInfo = {
    vendor: raw?.vendor ?? 'unknown',
    architecture: raw?.architecture ?? 'unknown',
    description: raw?.description ?? '',
    slug:
      `${raw?.vendor ?? 'unknown'}-${raw?.architecture ?? 'unknown'}`
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'unknown',
  }

  return { adapter, device, hasTimestamps, format: navigator.gpu.getPreferredCanvasFormat(), info }
}

export function configureCanvas(gpu: Gpu, canvas: HTMLCanvasElement): GPUCanvasContext {
  const context = canvas.getContext('webgpu')
  if (!context) throw new Error('Failed to get WebGPU canvas context.')
  context.configure({ device: gpu.device, format: gpu.format, alphaMode: 'opaque' })
  return context
}
