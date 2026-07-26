/**
 * A ZIP writer, hand-written, in ~150 lines and with no dependency.
 *
 * The lab already hand-writes a PNG codec (src/gpu/image.ts) for the same
 * reason: a browser-first tool that reaches for a library to move bytes around
 * ends up with a build step nobody can inspect. ZIP is the older and simpler of
 * the two formats — a header per file, a central directory, an end record —
 * and `CompressionStream('deflate-raw')` supplies the only hard part.
 *
 * Two decisions worth knowing:
 *
 *  - **PNG entries are STORED, not deflated.** A PNG's IDAT is already a
 *    deflate stream; running it through deflate again costs seconds on a 16 MB
 *    map set and typically GROWS the file. `compress: false` says so per entry.
 *  - **The timestamp is fixed.** Two exports of the same material produce
 *    byte-identical archives, so a bundle can be diffed. A real mtime would
 *    make every export look changed.
 */

import { blobPart, crc32 } from './image.ts'

export interface ZipEntry {
  /** Path inside the archive, '/'-separated. */
  name: string
  bytes: Uint8Array
  /** Deflate this entry. Default true; pass false for already-compressed data. */
  compress?: boolean
}

/** 2020-01-01 00:00:00 in the DOS date/time encoding — see the module header. */
const DOS_TIME = 0
const DOS_DATE = ((2020 - 1980) << 9) | (1 << 5) | 1

const LOCAL_SIG = 0x04034b50
const CENTRAL_SIG = 0x02014b50
const EOCD_SIG = 0x06054b50
/** Bit 11: the name is UTF-8. */
const FLAG_UTF8 = 0x0800

async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  // 'deflate-raw' (RFC 1951) is what a ZIP entry stores. Note that PNG's IDAT
  // wants 'deflate' (RFC 1950, zlib-wrapped) instead — the two are one header
  // apart and silently produce a corrupt file if swapped.
  const stream = new Blob([blobPart(bytes)]).stream().pipeThrough(new CompressionStream('deflate-raw'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

interface Prepared {
  name: Uint8Array
  method: 0 | 8
  crc: number
  compressed: Uint8Array
  uncompressedSize: number
  offset: number
}

/** Build a ZIP archive in memory. Entries land in the order given. */
export async function makeZip(entries: readonly ZipEntry[]): Promise<Blob> {
  const encoder = new TextEncoder()
  const prepared: Prepared[] = []
  let offset = 0

  for (const entry of entries) {
    const wantsDeflate = entry.compress ?? true
    const compressed = wantsDeflate ? await deflateRaw(entry.bytes) : entry.bytes
    // Deflate can grow incompressible data; store it rather than pay for that.
    const stored = !wantsDeflate || compressed.length >= entry.bytes.length
    const payload = stored ? entry.bytes : compressed
    const name = encoder.encode(entry.name)
    prepared.push({
      name,
      method: stored ? 0 : 8,
      crc: crc32(entry.bytes),
      compressed: payload,
      uncompressedSize: entry.bytes.length,
      offset,
    })
    offset += 30 + name.length + payload.length
  }

  const centralSize = prepared.reduce((n, p) => n + 46 + p.name.length, 0)
  const total = offset + centralSize + 22
  const out = new Uint8Array(total)
  const dv = new DataView(out.buffer)
  let at = 0

  const u16 = (v: number): void => {
    dv.setUint16(at, v, true)
    at += 2
  }
  const u32 = (v: number): void => {
    dv.setUint32(at, v >>> 0, true)
    at += 4
  }
  const raw = (bytes: Uint8Array): void => {
    out.set(bytes, at)
    at += bytes.length
  }

  for (const p of prepared) {
    u32(LOCAL_SIG)
    u16(20) // version needed
    u16(FLAG_UTF8)
    u16(p.method)
    u16(DOS_TIME)
    u16(DOS_DATE)
    u32(p.crc)
    u32(p.compressed.length)
    u32(p.uncompressedSize)
    u16(p.name.length)
    u16(0) // extra field length
    raw(p.name)
    raw(p.compressed)
  }

  const centralOffset = at
  for (const p of prepared) {
    u32(CENTRAL_SIG)
    u16(20) // version made by
    u16(20) // version needed
    u16(FLAG_UTF8)
    u16(p.method)
    u16(DOS_TIME)
    u16(DOS_DATE)
    u32(p.crc)
    u32(p.compressed.length)
    u32(p.uncompressedSize)
    u16(p.name.length)
    u16(0) // extra
    u16(0) // comment
    u16(0) // disk number start
    u16(0) // internal attributes
    u32(0) // external attributes
    u32(p.offset)
    raw(p.name)
  }

  // Capture the size BEFORE writing the EOCD: `at` moves as the record is
  // written, and reading it inside the call put the central directory's own
  // length 12 bytes over (unzip: "reported length of central directory is 12
  // bytes too long").
  const centralSizeWritten = at - centralOffset
  u32(EOCD_SIG)
  u16(0) // this disk
  u16(0) // disk with the central directory
  u16(prepared.length)
  u16(prepared.length)
  u32(centralSizeWritten)
  u32(centralOffset)
  u16(0) // comment length

  return new Blob([out], { type: 'application/zip' })
}
