#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"

// Deterministic plant placement — bit-identical twin of Scatter in
// src/scene/scatter.ts. World is divided into SCATTER_CELL_SIZE cells; each
// cell holds SCATTER_MAX_PER_CELL candidate slots per species. Whether a
// candidate exists depends on species density × the experiment's density
// scale, so densities animate without any reallocation and placement is a
// pure function: NO global array exists at any plant count.

const SCATTER_CELL_SIZE: f32 = 4.0;
const SCATTER_MAX_PER_CELL: u32 = 128u;
const SCATTER_MAX_DENSITY: f32 = 8.0;   // plants per m^2 when every slot exists

struct ScatterPoint {
  exists: bool,
  pos: vec3f,     // y = terrain height
  yaw: f32,       // [0, 2pi)
  scale: f32,     // within species scale range
  phase: f32,     // wind phase offset [0, 2pi)
}

fn scatter_candidate(seed: u32, species_index: u32, cell: vec2i, i: u32, density_scale: f32) -> ScatterPoint {
  var out: ScatterPoint;
  let sp = species_table[species_index];
  let h = hash4(seed, bitcast<u32>(cell.x), bitcast<u32>(cell.y), (species_index << 16u) ^ i);
  let density = clamp(sp.density * density_scale, 0.0, SCATTER_MAX_DENSITY);
  out.exists = hash_f32(hash2(h, 0u)) < density / SCATTER_MAX_DENSITY;
  let ox = hash_f32(hash2(h, 1u));
  let oz = hash_f32(hash2(h, 2u));
  let xz = (vec2f(cell) + vec2f(ox, oz)) * SCATTER_CELL_SIZE;
  out.pos = vec3f(xz.x, terrain_height(xz), xz.y);
  out.yaw = hash_f32(hash2(h, 3u)) * 6.2831853;
  out.scale = mix(sp.scale_min, sp.scale_max, hash_f32(hash2(h, 4u)));
  out.phase = hash_f32(hash2(h, 5u)) * 6.2831853;
  return out;
}
