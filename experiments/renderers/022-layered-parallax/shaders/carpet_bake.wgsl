// CARPET BAKE — height-banded top-down capture of a PERIODIC community tile.
//
// A carpet species (stand_table[i].carpet_div > 0, e.g. Sphagnum palustre) is
// a 0.18m periodic tile, not a plant: nothing about the 8 side azimuths of the
// slab bake is useful for it, and the one view that matters — straight down —
// is captured there over the mesh's full support radius at the SIDE view's
// aspect ratio, which wastes ~96% of the atlas on a low cushion.
//
// So carpets get their own capture: orthographic straight down over EXACTLY
// the tile square [0, tileM]^2 in the mesh frame (tile origin is (0,0) for
// every source mesh), square, at native resolution, split into N_BAND HEIGHT
// BANDS plus one merged tile. Each band records the mean height of the
// geometry inside it — the plane the runtime shader intersects the eye ray
// with, which is where the cushion parallax comes from.
//
// The tile is periodic, so the capture is WRAPPED: the mesh is drawn once per
// 3x3 neighbour offset and everything that overflows one edge re-enters on the
// opposite one. That makes the resulting image exactly one period, so it can
// be sampled with `repeat` addressing and mipped all the way down without ever
// growing a seam.
//
// Each instance = (neighbour offset, band). Fragments outside the tile square
// are discarded rather than scissored, because one draw covers several tiles.
//
// Outputs:
//   target 0: rgb = authored albedo, a = 1 (coverage after downsampling)
//   target 1: rgb = mesh-frame normal * 0.5 + 0.5, flipped toward +Y

#include "src/wgsl/gcmesh.wgsl"

struct BandRec {
  // xy: mesh-frame xz wrap offset; zw: band min (exclusive) / max (inclusive)
  wrap_band: vec4f,
  // xy: NDC offset of this band's slot, zw: NDC scale
  slot: vec4f,
}

struct CarpetInfo {
  tile: vec4f,    // x: tile origin x, y: tile origin z, z: tileM, w: unused
  height: vec4f,  // x: y_min, y: y_max
  b_min: vec4f,   // mesh bounds min (dequantization)
  b_range: vec4f, // mesh bounds max - min
}

@group(0) @binding(0) var<storage, read> band_recs: array<BandRec>;
@group(0) @binding(1) var<uniform> carpet_info: CarpetInfo;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
  @location(2) tile_uv: vec2f,
  @location(3) height_t: f32,
  @location(4) @interpolate(flat) band: vec2f,
}

@vertex
fn vs(
  @builtin(instance_index) ii: u32,
  @location(0) q_pos: vec4<u32>,
  @location(1) q_attr: vec4<u32>,
) -> VOut {
  let rec = band_recs[ii];

  // Vertex record: [x y z r] [g b octU octV], all u16 UNORM against bounds.
  let p = carpet_info.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * carpet_info.b_range.xyz;
  let color = vec3f(f32(q_pos.w), f32(q_attr.x), f32(q_attr.y)) / 65535.0;

  let tile_m = carpet_info.tile.z;
  let origin = carpet_info.tile.xy;
  // Wrapped copy: the neighbour's overflow becomes this tile's fringe.
  let shifted = p.xz + rec.wrap_band.xy;
  let uv = (shifted - origin) / tile_m;

  let y_min = carpet_info.height.x;
  let y_max = carpet_info.height.y;
  let depth01 = clamp((y_max - p.y) / max(y_max - y_min, 1e-5), 0.0, 1.0);

  var n = gcmesh_normal_decode_u16(q_attr.z, q_attr.w);
  if (n.y < 0.0) {
    n = -n; // two-sided foliage: both faces light like the one facing up
  }

  var out: VOut;
  out.pos = vec4f(vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * rec.slot.zw + rec.slot.xy, depth01, 1.0);
  out.color = color;
  out.nrm = n;
  out.tile_uv = uv;
  out.height_t = p.y;
  out.band = rec.wrap_band.zw;
  return out;
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  // Outside the tile square this fragment belongs to a neighbouring slot.
  if (in.tile_uv.x < 0.0 || in.tile_uv.x >= 1.0 || in.tile_uv.y < 0.0 || in.tile_uv.y >= 1.0) {
    discard;
  }
  if (in.height_t <= in.band.x || in.height_t > in.band.y) {
    discard;
  }
  var out: FOut;
  out.albedo = vec4f(in.color, 1.0);
  out.nrm = vec4f(normalize(in.nrm) * 0.5 + 0.5, 1.0);
  return out;
}
