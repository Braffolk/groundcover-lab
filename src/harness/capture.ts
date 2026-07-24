import type { LabApp } from './loop.ts'

/**
 * Deterministic frame capture — reads back the offscreen color target (NOT
 * the canvas, whose buffer is cleared after presentation) and encodes a PNG.
 */
export async function captureViewPng(app: LabApp, viewIndex = 0): Promise<Blob> {
  const view = app.views[viewIndex]
  if (!view) throw new Error('no view to capture')
  const { device } = app.gpu
  const { width, height } = view.color
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256

  const readback = device.createBuffer({
    label: 'capture/readback',
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const enc = device.createCommandEncoder({ label: 'capture' })
  enc.copyTextureToBuffer({ texture: view.color }, { buffer: readback, bytesPerRow }, [width, height])
  device.queue.submit([enc.finish()])
  await readback.mapAsync(GPUMapMode.READ)
  const src = new Uint8Array(readback.getMappedRange())

  const bgra = app.gpu.format.startsWith('bgra')
  const image = new ImageData(width, height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = y * bytesPerRow + x * 4
      const d = (y * width + x) * 4
      image.data[d] = src[bgra ? s + 2 : s]!
      image.data[d + 1] = src[s + 1]!
      image.data[d + 2] = src[bgra ? s : s + 2]!
      image.data[d + 3] = 255
    }
  }
  readback.unmap()
  readback.destroy()

  const canvas = new OffscreenCanvas(width, height)
  canvas.getContext('2d')!.putImageData(image, 0, 0)
  return canvas.convertToBlob({ type: 'image/png' })
}

export async function uploadCapture(expId: string, blob: Blob, golden?: string): Promise<string> {
  const url = `/__thumb?exp=${encodeURIComponent(expId)}${golden ? `&cam=${encodeURIComponent(golden)}` : ''}`
  const res = await fetch(url, { method: 'POST', body: blob })
  if (!res.ok) throw new Error(`upload failed: ${res.status} ${await res.text()}`)
  return ((await res.json()) as { saved: string }).saved
}

export function downloadBlob(blob: Blob, filename: string): void {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}
