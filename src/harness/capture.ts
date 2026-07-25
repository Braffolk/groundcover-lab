import { readTextureRgba8 } from '../gpu/image.ts'
import { HAS_DEV_SINK } from '../util/env.ts'
import type { LabApp } from './loop.ts'

/**
 * Deterministic frame capture — reads back the offscreen color target (NOT
 * the canvas, whose buffer is cleared after presentation) and encodes a PNG.
 * `maxWidth` downscales (thumbnails); omit for full resolution (goldens).
 *
 * The readback itself (256-byte row alignment, unpadding, BGRA swizzle) lives
 * in src/gpu/image.ts so bakes and materials share one copy of it.
 */
export async function captureViewPng(app: LabApp, viewIndex = 0, maxWidth?: number): Promise<Blob> {
  const view = app.views[viewIndex]
  if (!view) throw new Error('no view to capture')
  const { device } = app.gpu
  const { width, height } = view.color

  const image = await readTextureRgba8(device, view.color, {
    forceOpaque: true,
    exempt: (fn) => app.tracker.exempt(fn),
  })

  const canvas = new OffscreenCanvas(width, height)
  canvas.getContext('2d')!.putImageData(image, 0, 0)
  if (maxWidth !== undefined && width > maxWidth) {
    const scale = maxWidth / width
    const scaled = new OffscreenCanvas(Math.round(width * scale), Math.round(height * scale))
    const ctx2d = scaled.getContext('2d')!
    ctx2d.imageSmoothingQuality = 'high'
    ctx2d.drawImage(canvas, 0, 0, scaled.width, scaled.height)
    return scaled.convertToBlob({ type: 'image/png' })
  }
  return canvas.convertToBlob({ type: 'image/png' })
}

/** Writes into the repo — dev server only; the buttons are hidden otherwise. */
export async function uploadCapture(expId: string, blob: Blob, golden?: string): Promise<string> {
  if (!HAS_DEV_SINK) throw new Error('captures cannot be saved on a static build')
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
