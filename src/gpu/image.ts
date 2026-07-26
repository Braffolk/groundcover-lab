/**
 * PNG in, PNG out, and the GPU textures on either end.
 *
 * Before this module the lab could only ENCODE PNG (via canvas, in
 * src/harness/capture.ts) and had no image loading at all. Four things are
 * silent-failure traps and are handled here once, so nobody rediscovers them:
 *
 *  1. The dev server answers a MISSING file with index.html at HTTP 200 (see
 *     src/bake/io.ts — three agents poisoned their caches this way). Every
 *     load here checks content-type AND the PNG magic before decoding.
 *  2. `createImageBitmap` defaults to premultiplying alpha and applying the
 *     display colour profile. Both silently change texel values — the 017
 *     colour-convention bug arriving through a new door. Always
 *     `{ premultiplyAlpha: 'none', colorSpaceConversion: 'none' }`.
 *  3. `copyExternalImageToTexture` needs COPY_DST | RENDER_ATTACHMENT on the
 *     destination. Omitting it is a validation error, and WebGPU validation
 *     errors do NOT throw — they arrive as a toast you may not be looking at.
 *  4. The browser cannot encode 16-bit PNG through a canvas, and
 *     `createImageBitmap` SILENTLY TRUNCATES a 16-bit PNG to 8 bits. So the
 *     16-bit path is a hand-written codec, and the ImageBitmap path reads IHDR
 *     first and refuses bit depth 16 rather than halving your precision behind
 *     your back.
 *
 * 16-bit PNG samples are BIG-ENDIAN. Get that wrong and a heightmap looks like
 * noise with a perfectly plausible histogram, so DEV runs an encode → decode →
 * byte-compare round trip (plus a raw byte-order assertion) on first use.
 *
 * We deliberately do NOT pack 16 bits across two 8-bit channels: the file
 * stops being viewable as an image, which deletes the entire reason to prefer
 * PNG over the .bin format the lab already has.
 */

import { bakedArtifact, type BakeContext } from '../bake/io.ts'
import { assetUrl } from '../util/paths.ts'
import type { VramAttr, VramScope } from './resources.ts'
import type { ShaderRegistry } from './shaders.ts'
import { createTexture2D, generateMips, uploadImageBitmap, type MipPlan } from './texture.ts'

// ---------------------------------------------------------------------------
// PNG codec
// ---------------------------------------------------------------------------

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const

/** 1 = gray, 2 = gray+alpha, 3 = RGB, 4 = RGBA. */
export type PngChannels = 1 | 2 | 3 | 4

export interface PngHeader {
  width: number
  height: number
  bitDepth: 8 | 16
  /** PNG colour type: 0 gray, 2 RGB, 4 gray+alpha, 6 RGBA. */
  colorType: 0 | 2 | 4 | 6
  channels: PngChannels
  interlace: number
}

export interface PngImage extends PngHeader {
  /** Row-major, tightly packed, NATIVE-endian samples. */
  data: Uint8Array | Uint16Array
}

const COLOR_TYPE_CHANNELS: Record<number, PngChannels> = { 0: 1, 2: 3, 4: 2, 6: 4 }
const CHANNELS_COLOR_TYPE: Record<PngChannels, 0 | 2 | 4 | 6> = { 1: 0, 2: 4, 3: 2, 4: 6 }

const CRC_TABLE = (() => {
  const t = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    t[n] = c >>> 0
  }
  return t
})()

/** CRC-32 (IEEE, the PNG/ZIP polynomial). Exported because ZIP needs it too. */
export function crc32(buf: Uint8Array): number {
  let c = 0xffffffff
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]!) & 0xff]! ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length)
  const dv = new DataView(out.buffer)
  dv.setUint32(0, data.length)
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i)
  out.set(data, 8)
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)))
  return out
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c
  const pa = Math.abs(p - a)
  const pb = Math.abs(p - b)
  const pc = Math.abs(p - c)
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c
}

/**
 * lib.dom types BlobPart as ArrayBuffer-backed only, while a plain Uint8Array
 * is `Uint8Array<ArrayBufferLike>` (it might sit on a SharedArrayBuffer).
 * Narrow, copying only in the shared case.
 */
export function blobPart(bytes: Uint8Array): BlobPart {
  const buf = bytes.buffer
  return buf instanceof ArrayBuffer ? new Uint8Array(buf, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes)
}

/**
 * `CompressionStream('deflate')` emits exactly the zlib-wrapped (RFC 1950)
 * stream an IDAT wants — 'deflate-raw' would NOT.
 */
async function zlibDeflate(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([blobPart(bytes)]).stream().pipeThrough(new CompressionStream('deflate'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

async function zlibInflate(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([blobPart(bytes)]).stream().pipeThrough(new DecompressionStream('deflate'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

export function hasPngMagic(bytes: Uint8Array): boolean {
  if (bytes.length < 8) return false
  for (let i = 0; i < 8; i++) if (bytes[i] !== PNG_MAGIC[i]) return false
  return true
}

/**
 * Parse IHDR. Cheap and synchronous — call it BEFORE handing bytes to
 * `createImageBitmap`, which would silently truncate a 16-bit image.
 */
export function readPngHeader(bytes: Uint8Array): PngHeader {
  if (!hasPngMagic(bytes)) {
    throw new Error(`not a PNG: magic is ${[...bytes.subarray(0, 8)].map((b) => b.toString(16).padStart(2, '0')).join(' ')}`)
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (dv.getUint32(12) !== 0x49484452) throw new Error('PNG: first chunk is not IHDR')
  const width = dv.getUint32(16)
  const height = dv.getUint32(20)
  const bitDepth = bytes[24]!
  const colorType = bytes[25]!
  const interlace = bytes[28]!
  if (bitDepth !== 8 && bitDepth !== 16) {
    throw new Error(`PNG bit depth ${bitDepth} is not supported (8 and 16 only; sub-byte depths and palettes are not used in this lab)`)
  }
  const channels = COLOR_TYPE_CHANNELS[colorType]
  if (channels === undefined) {
    throw new Error(
      `PNG colour type ${colorType} is not supported (0 gray, 2 RGB, 4 gray+alpha, 6 RGBA). ` +
        `Type 3 is palette — re-export without a palette, or load it through loadImageTexture (ImageBitmap) instead.`,
    )
  }
  if (interlace !== 0) throw new Error('interlaced (Adam7) PNG is not supported — re-export non-interlaced')
  return { width, height, bitDepth: bitDepth as 8 | 16, colorType: colorType as 0 | 2 | 4 | 6, channels, interlace }
}

interface PngUnfiltered {
  header: PngHeader
  /** Unfiltered scanline bytes, tightly packed, still BIG-endian for depth 16. */
  raw: Uint8Array
}

async function pngUnfilter(bytes: Uint8Array): Promise<PngUnfiltered> {
  const header = readPngHeader(bytes)
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  const idats: Uint8Array[] = []
  let at = 8
  while (at + 8 <= bytes.length) {
    const len = dv.getUint32(at)
    const type = String.fromCharCode(bytes[at + 4]!, bytes[at + 5]!, bytes[at + 6]!, bytes[at + 7]!)
    const body = at + 8
    if (body + len > bytes.length) throw new Error(`PNG: chunk ${type} runs past the end of the file (truncated?)`)
    if (type === 'IDAT') idats.push(bytes.subarray(body, body + len))
    if (type === 'IEND') break
    at = body + len + 4
  }
  if (idats.length === 0) throw new Error('PNG: no IDAT chunks')

  let compressed: Uint8Array
  if (idats.length === 1) {
    compressed = idats[0]!
  } else {
    const total = idats.reduce((n, c) => n + c.length, 0)
    compressed = new Uint8Array(total)
    let o = 0
    for (const c of idats) {
      compressed.set(c, o)
      o += c.length
    }
  }
  const filtered = await zlibInflate(compressed)

  const sampleBytes = header.bitDepth / 8
  const bpp = header.channels * sampleBytes
  const stride = header.width * bpp
  const expected = (stride + 1) * header.height
  if (filtered.length < expected) {
    throw new Error(`PNG: inflated ${filtered.length} bytes, expected ${expected} (${header.width}x${header.height}x${header.channels}@${header.bitDepth})`)
  }

  const raw = new Uint8Array(stride * header.height)
  for (let y = 0; y < header.height; y++) {
    const src = y * (stride + 1)
    const ft = filtered[src]!
    const cur = y * stride
    const prev = cur - stride
    for (let x = 0; x < stride; x++) {
      const v = filtered[src + 1 + x]!
      const a = x >= bpp ? raw[cur + x - bpp]! : 0
      const b = y > 0 ? raw[prev + x]! : 0
      const c = x >= bpp && y > 0 ? raw[prev + x - bpp]! : 0
      let out: number
      switch (ft) {
        case 0:
          out = v
          break
        case 1:
          out = v + a
          break
        case 2:
          out = v + b
          break
        case 3:
          out = v + ((a + b) >> 1)
          break
        case 4:
          out = v + paeth(a, b, c)
          break
        default:
          throw new Error(`PNG: unknown filter type ${ft} on row ${y}`)
      }
      raw[cur + x] = out & 0xff
    }
  }
  return { header, raw }
}

/**
 * Decode a PNG the browser cannot be trusted with. Bit depth 8 yields a
 * Uint8Array; bit depth 16 yields a Uint16Array with the BIG-endian file
 * samples converted to native order.
 */
export async function decodePng(source: Uint8Array | ArrayBuffer): Promise<PngImage> {
  await codecSelfCheck()
  return decodePngRaw(source)
}

async function decodePngRaw(source: Uint8Array | ArrayBuffer): Promise<PngImage> {
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source)
  const { header, raw } = await pngUnfilter(bytes)
  if (header.bitDepth === 8) return { ...header, data: raw }

  const n = header.width * header.height * header.channels
  const data = new Uint16Array(n)
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  for (let i = 0; i < n; i++) data[i] = dv.getUint16(i * 2, false) // BIG-endian, per the PNG spec
  return { ...header, data }
}

export interface EncodePngInput {
  width: number
  height: number
  channels: PngChannels
  bitDepth: 8 | 16
  /** Tightly packed, native-endian. Uint16Array is required for bitDepth 16. */
  data: Uint8Array | Uint16Array | Uint8ClampedArray
}

/**
 * Encode a PNG, including 16-bit — which the browser's canvas encoder cannot
 * do at all. Filter type is chosen per scanline by the standard
 * minimum-sum-of-absolute-differences heuristic.
 */
export async function encodePng(img: EncodePngInput): Promise<Uint8Array> {
  await codecSelfCheck()
  return encodePngRaw(img)
}

async function encodePngRaw(img: EncodePngInput): Promise<Uint8Array> {
  const { width, height, channels, bitDepth } = img
  const sampleBytes = bitDepth / 8
  const bpp = channels * sampleBytes
  const stride = width * bpp
  const expected = width * height * channels
  if (img.data.length < expected) {
    throw new Error(`encodePng: got ${img.data.length} samples, need ${expected} (${width}x${height}x${channels})`)
  }
  if (bitDepth === 16 && !(img.data instanceof Uint16Array)) {
    throw new Error('encodePng: bitDepth 16 needs a Uint16Array — an 8-bit view would truncate every sample')
  }

  // Native samples -> big-endian scanline bytes.
  const scan = new Uint8Array(stride * height)
  if (bitDepth === 16) {
    const src = img.data as Uint16Array
    const dv = new DataView(scan.buffer)
    for (let i = 0; i < expected; i++) dv.setUint16(i * 2, src[i]!, false) // BIG-endian
  } else {
    scan.set((img.data as Uint8Array).subarray(0, expected))
  }

  const raw = new Uint8Array((stride + 1) * height)
  const cand = [new Uint8Array(stride), new Uint8Array(stride), new Uint8Array(stride), new Uint8Array(stride), new Uint8Array(stride)]
  for (let y = 0; y < height; y++) {
    const cur = y * stride
    const prev = cur - stride
    const scores = [0, 0, 0, 0, 0]
    for (let x = 0; x < stride; x++) {
      const v = scan[cur + x]!
      const a = x >= bpp ? scan[cur + x - bpp]! : 0
      const b = y > 0 ? scan[prev + x]! : 0
      const c = x >= bpp && y > 0 ? scan[prev + x - bpp]! : 0
      const f = [v, v - a, v - b, v - ((a + b) >> 1), v - paeth(a, b, c)]
      for (let t = 0; t < 5; t++) {
        const byte = f[t]! & 0xff
        cand[t]![x] = byte
        scores[t] = scores[t]! + (byte < 128 ? byte : 256 - byte)
      }
    }
    let best = 0
    for (let t = 1; t < 5; t++) if (scores[t]! < scores[best]!) best = t
    const o = y * (stride + 1)
    raw[o] = best
    raw.set(cand[best]!, o + 1)
  }

  const ihdr = new Uint8Array(13)
  const dv = new DataView(ihdr.buffer)
  dv.setUint32(0, width)
  dv.setUint32(4, height)
  ihdr[8] = bitDepth
  ihdr[9] = CHANNELS_COLOR_TYPE[channels]
  // [10] compression = 0, [11] filter = 0, [12] interlace = 0

  const idat = await zlibDeflate(raw)
  const parts = [
    new Uint8Array(PNG_MAGIC),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', idat),
    pngChunk('IEND', new Uint8Array(0)),
  ]
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
  let o = 0
  for (const p of parts) {
    out.set(p, o)
    o += p.length
  }
  return out
}

// ---------------------------------------------------------------------------
// DEV codec self-check — endianness and filters, on first use
// ---------------------------------------------------------------------------

let selfCheckPromise: Promise<void> | null = null

function codecSelfCheck(): Promise<void> {
  if (!import.meta.env.DEV) return Promise.resolve()
  if (!selfCheckPromise) selfCheckPromise = runCodecSelfCheck()
  return selfCheckPromise
}

async function runCodecSelfCheck(): Promise<void> {
  // 16-bit gray, values chosen so a byte-swap cannot survive the comparison.
  const w = 5
  const h = 3
  const gray16 = new Uint16Array(w * h)
  gray16[0] = 0x1234
  gray16[1] = 0
  gray16[2] = 0xffff
  gray16[3] = 0x00ff
  gray16[4] = 0xff00
  for (let i = 5; i < gray16.length; i++) gray16[i] = (i * 7919) & 0xffff
  const png16 = await encodePngRaw({ width: w, height: h, channels: 1, bitDepth: 16, data: gray16 })

  // (a) Independent byte-order assertion: a round trip alone would pass even
  //     if BOTH sides swapped, so check the file's own bytes.
  const { raw } = await pngUnfilter(png16)
  if (raw[0] !== 0x12 || raw[1] !== 0x34) {
    throw new Error(
      `[png] 16-bit samples are not big-endian: first sample stored as ${raw[0]!.toString(16)} ${raw[1]!.toString(16)}, expected 12 34. ` +
        'Use DataView.setUint16(o, v, false) / getUint16(o, false).',
    )
  }

  // (b) Round trip.
  const back16 = await decodePngRaw(png16)
  if (back16.bitDepth !== 16 || back16.channels !== 1 || back16.width !== w || back16.height !== h) {
    throw new Error(`[png] 16-bit header round trip failed: ${JSON.stringify({ ...back16, data: undefined })}`)
  }
  for (let i = 0; i < gray16.length; i++) {
    if ((back16.data as Uint16Array)[i] !== gray16[i]) {
      throw new Error(`[png] 16-bit round trip mismatch at ${i}: ${(back16.data as Uint16Array)[i]} != ${gray16[i]}`)
    }
  }

  // (c) 8-bit RGBA, with enough structure that the per-row filter heuristic
  //     picks several different filter types and exercises each unfilter path.
  const rgba = new Uint8Array(8 * 8 * 4)
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const o = (y * 8 + x) * 4
      rgba[o] = (x * 31) & 0xff
      rgba[o + 1] = (y * 17) & 0xff
      rgba[o + 2] = (x * y * 13) & 0xff
      rgba[o + 3] = x + y > 6 ? 0 : 255
    }
  }
  const png8 = await encodePngRaw({ width: 8, height: 8, channels: 4, bitDepth: 8, data: rgba })
  const back8 = await decodePngRaw(png8)
  for (let i = 0; i < rgba.length; i++) {
    if ((back8.data as Uint8Array)[i] !== rgba[i]) {
      throw new Error(`[png] 8-bit RGBA round trip mismatch at ${i}: ${(back8.data as Uint8Array)[i]} != ${rgba[i]}`)
    }
  }
}

// ---------------------------------------------------------------------------
// Fetching, with the SPA-fallback guard
// ---------------------------------------------------------------------------

const SPA_FALLBACK_HINT =
  'The dev server answers a MISSING file with index.html at HTTP 200, so a 200 response is not proof the file exists. ' +
  'Check the path and that the file is committed.'

/** Throws unless `bytes` really are a PNG, naming `url` and the likely cause. */
export function assertPngBytes(bytes: Uint8Array, url: string): void {
  if (hasPngMagic(bytes)) return
  const head = new TextDecoder().decode(bytes.subarray(0, 64))
  const looksHtml = /^\s*(?:<!doctype|<html|<!--)/i.test(head)
  throw new Error(
    `${url}: not a PNG (${bytes.length} bytes, starts with ${JSON.stringify(head.slice(0, 40))}). ` +
      (looksHtml ? `That is an HTML page — ${SPA_FALLBACK_HINT}` : 'Expected the magic 89 50 4E 47 0D 0A 1A 0A.'),
  )
}

/** Fetch PNG bytes, refusing HTML and anything without the PNG magic. */
export async function fetchPngBytes(url: string): Promise<Uint8Array> {
  const resolved = assetUrl(url)
  const res = await fetch(resolved).catch((err: unknown) => {
    throw new Error(`${resolved}: fetch failed (${String(err)})`)
  })
  if (!res.ok) throw new Error(`${resolved}: HTTP ${res.status} ${res.statusText}`)
  const type = res.headers.get('content-type') ?? ''
  if (type.includes('text/html')) {
    throw new Error(`${resolved}: served as ${type}, not an image. ${SPA_FALLBACK_HINT}`)
  }
  const bytes = new Uint8Array(await res.arrayBuffer())
  assertPngBytes(bytes, resolved)
  return bytes
}

// ---------------------------------------------------------------------------
// PNG -> GPUTexture
// ---------------------------------------------------------------------------

export interface LoadImageTextureOptions {
  /**
   * `rgba8unorm-srgb` instead of `rgba8unorm`. TRUE FOR ALBEDO ONLY — every
   * data map (normal, height, AO, mask, flow) is linear and must be false, or
   * the hardware will apply an sRGB decode to numbers that were never colours.
   */
  srgb: boolean
  label: string
  /** Build a mip chain with these semantics. Requires `shaders`. */
  mips?: MipPlan
  /** Needed only when `mips` is set — pass `ctx.shaders`. */
  shaders?: ShaderRegistry
  attr?: VramAttr
  flipY?: boolean
}

/**
 * Load an 8-bit PNG into a GPUTexture through `createImageBitmap`.
 *
 * THROWS on a 16-bit PNG rather than truncating it: `createImageBitmap`
 * silently drops the low byte of every sample, and a heightmap that quietly
 * lost half its precision is exactly the kind of bug that survives review.
 * Use `loadImage16Texture` for those.
 */
export async function loadImageTexture(
  res: VramScope,
  device: GPUDevice,
  url: string,
  opts: LoadImageTextureOptions,
): Promise<GPUTexture> {
  const bytes = await fetchPngBytes(url)
  const header = readPngHeader(bytes)
  if (header.bitDepth === 16) {
    throw new Error(
      `${url} is a 16-bit PNG. createImageBitmap SILENTLY TRUNCATES it to 8 bits — ` +
        'use loadImage16Texture() (or bakedImage(), which branches on IHDR for you).',
    )
  }
  const bitmap = await createImageBitmap(new Blob([blobPart(bytes)], { type: 'image/png' }), {
    // The defaults premultiply and apply the display profile; both silently
    // rewrite texel values. Never rely on them.
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
  })
  try {
    return uploadBitmap(res, device, bitmap, opts)
  } finally {
    bitmap.close()
  }
}

function uploadBitmap(
  res: VramScope,
  device: GPUDevice,
  bitmap: ImageBitmap,
  opts: LoadImageTextureOptions,
): GPUTexture {
  const levels = opts.mips ? (opts.mips.levels ?? mipLevels(bitmap.width, bitmap.height)) : 1
  if (opts.mips && !opts.shaders) {
    throw new Error(`loadImageTexture("${opts.label}"): opts.mips needs opts.shaders (pass ctx.shaders)`)
  }
  const tex = createTexture2D(res, {
    label: opts.label,
    size: [bitmap.width, bitmap.height],
    format: opts.srgb ? 'rgba8unorm-srgb' : 'rgba8unorm',
    mipLevelCount: levels,
    ...(opts.mips && { encoding: opts.mips.semantics }),
    ...(opts.attr && { attr: opts.attr }),
  })
  uploadImageBitmap(device, tex, bitmap, { flipY: opts.flipY ?? false })
  if (opts.mips && opts.shaders) generateMips(res, opts.shaders, tex, { ...opts.mips, levels })
  return tex
}

function mipLevels(width: number, height: number): number {
  return 1 + Math.floor(Math.log2(Math.max(1, Math.max(width, height))))
}

export interface LoadImage16Options {
  label: string
  /**
   * `'uint16'` (default) keeps all 16 bits exactly in r/rg/rgba16uint — read
   * it with `textureLoad`, it is unfilterable and cannot be mipped.
   * `'float32'` normalises to [0,1] in r/rg/rgba32float, which IS mippable
   * through the shared filter and exact for 16-bit integers; filtering it in
   * hardware additionally needs the `float32-filterable` feature.
   */
  precision?: 'uint16' | 'float32'
  mips?: MipPlan
  shaders?: ShaderRegistry
  attr?: VramAttr
}

const UINT16_FORMATS: Record<PngChannels, GPUTextureFormat> = {
  1: 'r16uint',
  2: 'rg16uint',
  3: 'rgba16uint',
  4: 'rgba16uint',
}
const FLOAT32_FORMATS: Record<PngChannels, GPUTextureFormat> = {
  1: 'r32float',
  2: 'rg32float',
  3: 'rgba32float',
  4: 'rgba32float',
}

/** Load a PNG of any supported bit depth through the hand-written codec. */
export async function loadImage16Texture(
  res: VramScope,
  device: GPUDevice,
  url: string,
  opts: LoadImage16Options,
): Promise<GPUTexture> {
  const bytes = await fetchPngBytes(url)
  return textureFromPng16(res, device, await decodePng(bytes), opts)
}

/** Upload an already-decoded PNG as a 16-bit-precision data texture. */
export function textureFromPng16(
  res: VramScope,
  device: GPUDevice,
  img: PngImage,
  opts: LoadImage16Options,
): GPUTexture {
  const precision = opts.precision ?? 'uint16'
  const { width, height, channels } = img
  // 3-channel PNGs are widened to 4: WebGPU has no 3-component texture format.
  const dstChannels = channels === 3 ? 4 : channels
  const format = (precision === 'uint16' ? UINT16_FORMATS : FLOAT32_FORMATS)[channels]

  if (opts.mips && precision === 'uint16') {
    throw new Error(
      `loadImage16Texture("${opts.label}"): mips are not available for precision 'uint16' (${format} is an integer format ` +
        `the shared float filter cannot write). Use precision: 'float32' if you need a mip chain, or textureLoad the uint texture.`,
    )
  }
  if (opts.mips && !opts.shaders) {
    throw new Error(`loadImage16Texture("${opts.label}"): opts.mips needs opts.shaders (pass ctx.shaders)`)
  }
  const levels = opts.mips ? (opts.mips.levels ?? mipLevels(width, height)) : 1

  const tex = createTexture2D(res, {
    label: opts.label,
    size: [width, height],
    format,
    mipLevelCount: levels,
    ...(opts.mips && { encoding: opts.mips.semantics }),
    ...(opts.attr && { attr: opts.attr }),
  })

  const n = width * height
  const src = img.data
  const scale = img.bitDepth === 16 ? 65535 : 255
  let bytesPerRow: number
  if (precision === 'uint16') {
    const packed = new Uint16Array(n * dstChannels)
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < dstChannels; c++) packed[i * dstChannels + c] = c < channels ? src[i * channels + c]! : 0
    }
    bytesPerRow = width * dstChannels * 2
    device.queue.writeTexture({ texture: tex }, packed, { bytesPerRow, rowsPerImage: height }, [width, height])
  } else {
    const packed = new Float32Array(n * dstChannels)
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < dstChannels; c++) {
        packed[i * dstChannels + c] = c < channels ? src[i * channels + c]! / scale : 0
      }
    }
    bytesPerRow = width * dstChannels * 4
    device.queue.writeTexture({ texture: tex }, packed, { bytesPerRow, rowsPerImage: height }, [width, height])
  }

  if (opts.mips && opts.shaders) generateMips(res, opts.shaders, tex, { ...opts.mips, levels })
  return tex
}

// ---------------------------------------------------------------------------
// bakedImage — the bake flow, returning a texture
// ---------------------------------------------------------------------------

export interface BakedImageContext extends BakeContext {
  res: VramScope
  device: GPUDevice
  label: string
  /** True for albedo only; every data map stays linear. Ignored at 16-bit. */
  srgb: boolean
  mips?: MipPlan
  shaders?: ShaderRegistry
  attr?: VramAttr
  /** For 16-bit sources — see LoadImage16Options.precision. */
  precision?: 'uint16' | 'float32'
  flipY?: boolean
}

/**
 * `bakedArtifact` for images: OPFS cache -> committed mesh/baked/<exp>/<key>.png
 * -> in-browser bake, then decoded into a GPUTexture. Guards the SPA-fallback
 * HTML, checks the PNG magic (an older poisoned cache entry would otherwise
 * survive), and branches on the IHDR bit depth so 16-bit data never quietly
 * goes through the truncating ImageBitmap path.
 */
export async function bakedImage(ctx: BakedImageContext, bake: () => Promise<Uint8Array>): Promise<GPUTexture> {
  const bakeCtx: BakeContext = {
    expId: ctx.expId,
    key: ctx.key,
    ext: 'png',
    ...(ctx.onProgress && { onProgress: ctx.onProgress }),
  }
  const buffer = await bakedArtifact(bakeCtx, async () => {
    const png = await bake()
    return png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength) as ArrayBuffer
  })
  const bytes = new Uint8Array(buffer)
  assertPngBytes(bytes, `mesh/baked/${ctx.expId}/${ctx.key}.png`)
  const header = readPngHeader(bytes)

  if (header.bitDepth === 16) {
    return textureFromPng16(ctx.res, ctx.device, await decodePng(bytes), {
      label: ctx.label,
      ...(ctx.precision && { precision: ctx.precision }),
      ...(ctx.mips && { mips: ctx.mips }),
      ...(ctx.shaders && { shaders: ctx.shaders }),
      ...(ctx.attr && { attr: ctx.attr }),
    })
  }
  const bitmap = await createImageBitmap(new Blob([blobPart(bytes)], { type: 'image/png' }), {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
  })
  try {
    return uploadBitmap(ctx.res, ctx.device, bitmap, {
      srgb: ctx.srgb,
      label: ctx.label,
      ...(ctx.mips && { mips: ctx.mips }),
      ...(ctx.shaders && { shaders: ctx.shaders }),
      ...(ctx.attr && { attr: ctx.attr }),
      ...(ctx.flipY !== undefined && { flipY: ctx.flipY }),
    })
  } finally {
    bitmap.close()
  }
}

// ---------------------------------------------------------------------------
// Texture -> CPU (shared with src/harness/capture.ts)
// ---------------------------------------------------------------------------

export interface ReadRgba8Options {
  mipLevel?: number
  /** Force alpha to 255 — what a screenshot wants, not what a bake wants. */
  forceOpaque?: boolean
  /** Suppress the VRAM tracker's untracked warning; pass `tracker.exempt`. */
  exempt?: <T>(fn: () => T) => T
}

/**
 * Read an 8-bit colour texture back as ImageData: 256-byte row alignment on
 * the copy, row unpadding afterwards, and the BGRA swizzle when the target is
 * a bgra format (which the preferred canvas format usually is).
 */
export async function readTextureRgba8(
  device: GPUDevice,
  tex: GPUTexture,
  opts: ReadRgba8Options = {},
): Promise<ImageData> {
  const level = opts.mipLevel ?? 0
  const width = Math.max(1, tex.width >> level)
  const height = Math.max(1, tex.height >> level)
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256
  const run = opts.exempt ?? (<T>(fn: () => T): T => fn())

  const readback = run(() =>
    device.createBuffer({
      label: `readTextureRgba8/${tex.label}`,
      size: bytesPerRow * height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  )
  const enc = device.createCommandEncoder({ label: `readTextureRgba8/${tex.label}` })
  enc.copyTextureToBuffer({ texture: tex, mipLevel: level }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [
    width,
    height,
  ])
  device.queue.submit([enc.finish()])
  await readback.mapAsync(GPUMapMode.READ)
  const src = new Uint8Array(readback.getMappedRange())

  const bgra = tex.format.startsWith('bgra')
  const image = new ImageData(width, height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = y * bytesPerRow + x * 4
      const d = (y * width + x) * 4
      image.data[d] = src[bgra ? s + 2 : s]!
      image.data[d + 1] = src[s + 1]!
      image.data[d + 2] = src[bgra ? s : s + 2]!
      image.data[d + 3] = opts.forceOpaque ? 255 : src[s + 3]!
    }
  }
  readback.unmap()
  readback.destroy()
  return image
}
