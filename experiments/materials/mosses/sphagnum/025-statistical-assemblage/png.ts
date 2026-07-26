function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1))
  }
  return (crc ^ 0xffffffff) >>> 0
}

function u32(value: number): Uint8Array {
  return new Uint8Array([(value >>> 24) & 255, (value >>> 16) & 255, (value >>> 8) & 255, value & 255])
}

function chunk(name: string, data: Uint8Array): Uint8Array {
  const type = new TextEncoder().encode(name)
  const body = new Uint8Array(type.length + data.length)
  body.set(type)
  body.set(data, type.length)
  const out = new Uint8Array(12 + data.length)
  out.set(u32(data.length))
  out.set(body, 4)
  out.set(u32(crc32(body)), 8 + data.length)
  return out
}

export async function encodeRgbaPng(width: number, height: number, rgba: Uint8Array): Promise<Uint8Array> {
  if (rgba.length !== width * height * 4) throw new Error(`RGBA size mismatch: ${rgba.length} for ${width}x${height}`)
  const scan = new Uint8Array(height * (width * 4 + 1))
  for (let y = 0; y < height; y++) {
    scan.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), y * (width * 4 + 1) + 1)
  }
  const compressed = new Uint8Array(
    await new Response(new Blob([scan]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer(),
  )
  const ihdr = new Uint8Array(13)
  ihdr.set(u32(width))
  ihdr.set(u32(height), 4)
  ihdr[8] = 8
  ihdr[9] = 6
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', compressed),
    chunk('IEND', new Uint8Array()),
  ]
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0))
  let offset = 0
  for (const part of parts) {
    out.set(part, offset)
    offset += part.length
  }
  return out
}
