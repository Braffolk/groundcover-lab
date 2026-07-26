/**
 * Loading an `image` node's PNG into the texture its ChannelSpec describes.
 *
 * The format is chosen from the PNG's OWN header, not from a hopeful default:
 *
 *   gray8   -> r8unorm      (a 2048^2 AO map is 5.6 MB with mips, not 22.4)
 *   8-bit   -> rgba8unorm[-srgb], via the ImageBitmap path
 *   16-bit  -> r/rg/rgba32float, via the hand-written codec, because
 *              createImageBitmap SILENTLY truncates 16-bit samples to 8
 *
 * and the mip chain is built with the node's declared `TexelSemantics`, which
 * is the same declaration the generated decoder is emitted from.
 */

import { decodePng, fetchPngBytes, readPngHeader, textureFromPng16 } from '../gpu/image.ts'
import type { VramAttr, VramScope } from '../gpu/resources.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import { createTexture2D, generateMips, mipLevelsFor, uploadImageBitmap } from '../gpu/texture.ts'
import type { ChannelSpec } from './types.ts'

/**
 * lib.dom types BlobPart/writeTexture sources as ArrayBuffer-backed only,
 * while a plain Uint8Array is `Uint8Array<ArrayBufferLike>`. Narrow, copying
 * only in the (never taken here) shared case.
 */
function ownBuffer(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const buf = bytes.buffer
  return buf instanceof ArrayBuffer ? new Uint8Array(buf, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes)
}

export interface LoadNodeImageOptions {
  res: VramScope
  device: GPUDevice
  shaders: ShaderRegistry
  url: string
  label: string
  spec: ChannelSpec
  attr: VramAttr
}

export interface LoadedImage {
  texture: GPUTexture
  width: number
  height: number
}

export async function loadNodeImage(opts: LoadNodeImageOptions): Promise<LoadedImage> {
  const { res, device, shaders, spec } = opts
  const bytes = await fetchPngBytes(opts.url)
  const header = readPngHeader(bytes)
  const levels = mipLevelsFor(header.width, header.height)
  const mips = { semantics: spec.semantics, levels }

  if (header.bitDepth === 16) {
    const texture = textureFromPng16(res, device, await decodePng(bytes), {
      label: opts.label,
      // float32 rather than uint16: the shared mip filter is float-only, and
      // a 16-bit integer is exact in f32.
      precision: 'float32',
      mips,
      shaders,
      attr: opts.attr,
    })
    return { texture, width: header.width, height: header.height }
  }

  if (header.channels === 1) {
    const img = await decodePng(bytes)
    const texture = createTexture2D(res, {
      label: opts.label,
      size: [img.width, img.height],
      format: 'r8unorm',
      mipLevelCount: levels,
      encoding: spec.semantics,
      attr: opts.attr,
    })
    device.queue.writeTexture(
      { texture },
      ownBuffer(img.data as Uint8Array),
      { bytesPerRow: img.width, rowsPerImage: img.height },
      [img.width, img.height],
    )
    generateMips(res, shaders, texture, mips)
    return { texture, width: img.width, height: img.height }
  }

  const bitmap = await createImageBitmap(new Blob([ownBuffer(bytes)], { type: 'image/png' }), {
    // The defaults premultiply and apply the display profile; both silently
    // rewrite texel values.
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
  })
  try {
    const texture = createTexture2D(res, {
      label: opts.label,
      size: [bitmap.width, bitmap.height],
      format: spec.srgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
      mipLevelCount: levels,
      encoding: spec.semantics,
      attr: opts.attr,
    })
    uploadImageBitmap(device, texture, bitmap)
    generateMips(res, shaders, texture, mips)
    return { texture, width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}
