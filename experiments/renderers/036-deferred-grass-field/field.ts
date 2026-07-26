/**
 * The baked ray-answer table ("grass field table") — layout, indexing math and
 * artifact (de)serialization. Shared by bake.ts and main.ts so the CPU packer
 * and the WGSL sampler can never disagree about what a texel means.
 *
 * WHAT IS TABULATED
 * -----------------
 * One periodic canopy patch (TILE_M metres square, baked flat, populated by
 * the ACTIVE STAND's own scatter) is queried by eye rays. A ray is identified
 * by where it ENTERS the canopy — its crossing of the plane y = H — and by its
 * direction (azimuth, shear). The table stores the ANSWER of that query:
 *
 *   surf[e] : texture_3d rgba8unorm  (u, v, azimuth)  = (albedo*cov, cov)
 *   geom[e] : texture_3d rgba8snorm  (u, v, azimuth)  = (n.x*c, n.z*c, h01*c, c)
 *
 * `h01` is the HEIGHT of the first hit above the ground plane, normalized by
 * the canopy height H. Height (not distance) is stored because it makes the
 * table independent of the elevation quantisation: the runtime converts it to
 * a distance along the TRUE eye ray with the closed-form |OB| = |OA|/cos(a),
 * i.e. t = h / |d.y|.
 *
 * The elevation axis is the shear s = |d.xz| / |d.y| = cot(elevation), sampled
 * uniformly in q = s/(1+s) so the cells crowd toward the grazing end where the
 * answer changes fastest. Everything is premultiplied by coverage, which makes
 * a plain box filter the correct prefilter for mips (coverage-weighted).
 *
 * WHY THE ENTRY POINT AND NOT THE GROUND CROSSING
 * -----------------------------------------------
 * Both index the same content (they differ by a constant shift per cell), but
 * they extrapolate and interpolate completely differently. Indexed by the
 * ground crossing, the answer for a 2-degree ray is nothing like the answer for
 * a 9-degree ray through the same ground point (one grazes the tips, the other
 * dives to the soil), so clamping the elevation axis paints deep, dark hits all
 * over the mid-field — that was v2, and it looked like a dark plateau. Indexed
 * by the ENTRY point, every answer is "what is just below where the ray came
 * in", which converges as the ray flattens: clamping at s_max is then a
 * sub-metre error along the ray instead of a wrong surface, and neighbouring
 * elevation cells are similar enough that lerping them does not ghost.
 */

/** World period of the baked patch (m). */
export const TILE_M = 6.0
/** Texels per axis of the surface slab (mip 0) -> TILE_M/TILE_PX metres. */
export const TILE_PX = 256
/** Geometry slab resolution divisor (1 = full: blade-scale normals matter). */
export const GEOM_DIV = 1
export const GEOM_PX = TILE_PX / GEOM_DIV
/** Azimuth slices per elevation slab (the filtered 3rd texture axis). */
export const AZ = 16
/** Elevation cells (separate texture pair each). */
export const ELEV = 6
/** Elevation warp: q = s/(1+s), sampled at q0 + i*dq. */
export const ELEV_Q0 = 0.107
export const ELEV_DQ = 0.1604
/** Plant yaw quantisation for the stamp library (must divide 360/AZ). */
export const YAWS = 16
/** Stage-B supersample factor of the composite render (per field texel). */
export const SS = 4

export const MAGIC = 0x31464744 // 'DGF1'
export const VERSION = 4
const HEADER_BYTES = 64

/** Shear (cot of elevation angle) of elevation cell `e`. */
export function shearOfCell(e: number): number {
  const q = ELEV_Q0 + ELEV_DQ * e
  return q / (1 - q)
}

export interface FieldTable {
  tilePx: number
  geomPx: number
  az: number
  elev: number
  /** Canopy height (m) the h01 channel is normalized by. */
  canopyH: number
  tileM: number
  /** Per elevation, mip-0 bytes of the surface slab (rgba8unorm). */
  surf: Uint8Array<ArrayBuffer>[]
  /** Per elevation, mip-0 bytes of the geometry slab (rgba8snorm). */
  geom: Uint8Array<ArrayBuffer>[]
}

function surfBytes(): number {
  return TILE_PX * TILE_PX * AZ * 4
}
function geomBytes(): number {
  return GEOM_PX * GEOM_PX * AZ * 4
}

export function packField(t: FieldTable): ArrayBuffer {
  const per = surfBytes() + geomBytes()
  const total = HEADER_BYTES + per * ELEV
  const buf = new ArrayBuffer(total)
  const u = new Uint32Array(buf, 0, 8)
  u[0] = MAGIC
  u[1] = VERSION
  u[2] = t.tilePx
  u[3] = t.geomPx
  u[4] = t.az
  u[5] = t.elev
  const f = new Float32Array(buf, 32, 4)
  f[0] = t.canopyH
  f[1] = t.tileM
  let off = HEADER_BYTES
  for (let e = 0; e < ELEV; e++) {
    new Uint8Array(buf, off, surfBytes()).set(t.surf[e]!)
    off += surfBytes()
    new Uint8Array(buf, off, geomBytes()).set(t.geom[e]!)
    off += geomBytes()
  }
  return buf
}

export function unpackField(buf: ArrayBuffer): FieldTable | null {
  if (buf.byteLength < HEADER_BYTES) return null
  const u = new Uint32Array(buf, 0, 8)
  if (u[0] !== MAGIC || u[1] !== VERSION) return null
  if (u[2] !== TILE_PX || u[3] !== GEOM_PX || u[4] !== AZ || u[5] !== ELEV) return null
  const expected = HEADER_BYTES + (surfBytes() + geomBytes()) * ELEV
  if (buf.byteLength !== expected) return null
  const f = new Float32Array(buf, 32, 4)
  const surf: Uint8Array<ArrayBuffer>[] = []
  const geom: Uint8Array<ArrayBuffer>[] = []
  let off = HEADER_BYTES
  for (let e = 0; e < ELEV; e++) {
    surf.push(new Uint8Array(buf, off, surfBytes()) as Uint8Array<ArrayBuffer>)
    off += surfBytes()
    geom.push(new Uint8Array(buf, off, geomBytes()) as Uint8Array<ArrayBuffer>)
    off += geomBytes()
  }
  return {
    tilePx: TILE_PX,
    geomPx: GEOM_PX,
    az: AZ,
    elev: ELEV,
    canopyH: f[0]!,
    tileM: f[1]!,
    surf,
    geom,
  }
}

/** Mip level count for a 3D slab (all three axes halve, depth clamped at 1). */
export function mipCount(px: number): number {
  return Math.floor(Math.log2(px)) + 1
}
