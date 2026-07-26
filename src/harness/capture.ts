import { readTextureRgba8 } from '../gpu/image.ts'
import { HAS_DEV_SINK } from '../util/env.ts'
import type { LabApp } from './loop.ts'

/**
 * Encode an ImageData as a PNG blob, optionally downscaled to `maxWidth`.
 * Shared by the frame/texture capture paths and by anything that already has
 * pixels (the material inspector builds its own from a node's mip level).
 */
export async function imageDataToPngBlob(image: ImageData, maxWidth?: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(image.width, image.height)
  canvas.getContext('2d')!.putImageData(image, 0, 0)
  if (maxWidth !== undefined && image.width > maxWidth) {
    const scale = maxWidth / image.width
    const scaled = new OffscreenCanvas(Math.round(image.width * scale), Math.round(image.height * scale))
    const ctx2d = scaled.getContext('2d')!
    ctx2d.imageSmoothingQuality = 'high'
    ctx2d.drawImage(canvas, 0, 0, scaled.width, scaled.height)
    return scaled.convertToBlob({ type: 'image/png' })
  }
  return canvas.convertToBlob({ type: 'image/png' })
}

export interface CaptureTextureOptions {
  /** Downscale to this width (thumbnails); omit for full resolution. */
  maxWidth?: number
  mipLevel?: number
  /** Force alpha to 255 — a frame target's alpha is not meaningful. */
  forceOpaque?: boolean
  /** Suppress the VRAM tracker's warning for the staging buffer. */
  exempt?: <T>(fn: () => T) => T
}

/**
 * Read any 8-bit colour texture back and encode it as a PNG.
 *
 * The readback itself (256-byte row alignment, unpadding, BGRA swizzle) lives
 * in src/gpu/image.ts so bakes and materials share one copy of it — which also
 * means this is rgba8/bgra8 only. A material node's texture can be r8unorm or
 * float; the material inspector decodes those itself and comes back through
 * `imageDataToPngBlob`.
 */
export async function captureTexturePng(
  device: GPUDevice,
  texture: GPUTexture,
  opts: CaptureTextureOptions = {},
): Promise<Blob> {
  const image = await readTextureRgba8(device, texture, {
    ...(opts.mipLevel !== undefined && { mipLevel: opts.mipLevel }),
    ...(opts.forceOpaque !== undefined && { forceOpaque: opts.forceOpaque }),
    ...(opts.exempt && { exempt: opts.exempt }),
  })
  return imageDataToPngBlob(image, opts.maxWidth)
}

/**
 * Deterministic frame capture — reads back the offscreen color target (NOT
 * the canvas, whose buffer is cleared after presentation) and encodes a PNG.
 * `maxWidth` downscales (thumbnails); omit for full resolution (goldens).
 */
export async function captureViewPng(app: LabApp, viewIndex = 0, maxWidth?: number): Promise<Blob> {
  const view = app.views[viewIndex]
  if (!view) throw new Error('no view to capture')
  return captureTexturePng(app.gpu.device, view.color, {
    ...(maxWidth !== undefined && { maxWidth }),
    forceOpaque: true,
    exempt: (fn) => app.tracker.exempt(fn),
  })
}

/**
 * Writes into the repo — dev server only; the buttons are hidden otherwise.
 *
 * `dir` is the experiment's repo-relative directory (`experiments/...`), since
 * a nested experiment's id is no longer its path. Thumbnails land beside the
 * manifest; goldens land in `goldens/<id>/`, and the sink derives that id from
 * the directory's last segment.
 */
export async function uploadCapture(dir: string, blob: Blob, golden?: string): Promise<string> {
  if (!HAS_DEV_SINK) throw new Error('captures cannot be saved on a static build')
  const url = `/__thumb?exp=${encodeURIComponent(dir)}${golden ? `&cam=${encodeURIComponent(golden)}` : ''}`
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
