#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"

// Deterministic plant placement — bit-identical twin of Scatter in
// src/scene/scatter.ts. Placement is fully determined by the active stand
// (stand_table) + seed: world is divided into SCATTER_CELL_SIZE cells; each
// cell holds SCATTER_MAX_PER_CELL candidate slots per stand entry, and a
// candidate exists iff its hash clears the entry's density threshold.
// Placement is a pure function: NO global array exists at any plant count.

const SCATTER_CELL_SIZE: f32 = 4.0;
const SCATTER_MAX_PER_CELL: u32 = 128u;
const SCATTER_MAX_DENSITY: f32 = 8.0;   // plants per m^2 when every slot exists

struct ScatterPoint {
  exists: bool,
  pos: vec3f,     // y = terrain height
  yaw: f32,       // [0, 2pi)
  scale: f32,     // within the stand entry's scale range
  phase: f32,     // wind phase offset [0, 2pi)
}

// --- Habitat modulation ------------------------------------------------------
// A shared "wetness" field in [0,1] that stands can zone species against, so a
// mixed community reads as ecology rather than as a random shuffle. Smooth
// value noise on a 12 m grid, damped on slopes because water collects on flat
// ground. Bit-identical twin: Scatter.wetness() in src/scene/scatter.ts.
const WET_CELL: f32 = 12.0;
const WET_SALT: u32 = 0x9e37u;
// 1 - 0.82^2: the slope^2 at which ground stops counting as flat.
const WET_SLOPE_SQ: f32 = 0.3276;

fn wet_corner(seed: u32, cx: i32, cz: i32) -> f32 {
  return hash_f32(hash4(seed, bitcast<u32>(cx), bitcast<u32>(cz), WET_SALT));
}

fn scatter_wetness(seed: u32, xz: vec2f) -> f32 {
  let u = xz / WET_CELL;
  let b = floor(u);
  let f = u - b;
  let s = f * f * (3.0 - 2.0 * f);
  let cx = i32(b.x);
  let cz = i32(b.y);
  let a = wet_corner(seed, cx, cz);
  let c = wet_corner(seed, cx + 1, cz);
  let d = wet_corner(seed, cx, cz + 1);
  let e = wet_corner(seed, cx + 1, cz + 1);
  let lo = a + (c - a) * s.x;
  let hi = d + (e - d) * s.x;
  let n = lo + (hi - lo) * s.y;
  // Slope from the shared heightmap — avoids the sqrt in terrain_normal so the
  // TypeScript mirror has fewer rounding steps to match.
  let g = terrain_sample(xz);
  let slope_sq = g.y * g.y + g.z * g.z;
  let flat = clamp((WET_SLOPE_SQ - slope_sq) / WET_SLOPE_SQ, 0.0, 1.0);
  return clamp(n * (0.30 + 0.70 * flat), 0.0, 1.0);
}

fn scatter_candidate(seed: u32, entry_index: u32, cell: vec2i, i: u32) -> ScatterPoint {
  var out: ScatterPoint;
  let entry = stand_table[entry_index];
  let h = hash4(seed, bitcast<u32>(cell.x), bitcast<u32>(cell.y), (entry_index << 16u) ^ i);
  let density = min(entry.density, SCATTER_MAX_DENSITY);
  out.exists = hash_f32(hash2(h, 0u)) < density / SCATTER_MAX_DENSITY;
  let ox = hash_f32(hash2(h, 1u));
  let oz = hash_f32(hash2(h, 2u));
  let xz = (vec2f(cell) + vec2f(ox, oz)) * SCATTER_CELL_SIZE;
  // Habitat band, if this entry declares one. Acceptance falls off linearly
  // from the band centre and is resolved stochastically per plant, so zones
  // intergrade naturally instead of ending on a hard line.
  if (entry.wet_width > 0.0) {
    let dist = abs(scatter_wetness(seed, xz) - entry.wet_center) / entry.wet_width;
    if (hash_f32(hash2(h, 6u)) >= clamp(1.0 - dist, 0.0, 1.0)) {
      out.exists = false;
    }
  }
  out.pos = vec3f(xz.x, terrain_height(xz), xz.y);
  out.yaw = hash_f32(hash2(h, 3u)) * 6.2831853;
  out.scale = mix(entry.scale_min, entry.scale_max, hash_f32(hash2(h, 4u)));
  out.phase = hash_f32(hash2(h, 5u)) * 6.2831853;
  return out;
}
