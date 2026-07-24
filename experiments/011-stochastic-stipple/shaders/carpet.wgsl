#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Mean-field carpet: beyond the point where clump amplification saturates,
// the ensemble collapses to its first-order statistics — a terrain-following
// canopy sheet at mean canopy height, species-mottled by the stand's density
// mix, dissolving in stochastically (1m hashed patches) under the card layer.
// One draw, ~12k triangles, camera-bounded: cost independent of plant count.

struct CarpetUni {
  colors: array<vec4f, 4>, // rgb = species mean albedo, w = cumulative weight
  n_entries: u32,
  seed: u32,
  _pa: u32,
  _pb: u32,
  inner_r: f32,
  outer_r: f32,
  canopy_h: f32,
  stand_radius: f32,
  fade_band: f32,
  _pc: f32,
  _pd: f32,
  _pe: f32,
}
@group(1) @binding(0) var<uniform> carpet_uni: CarpetUni;

const SEGS: u32 = 96u;
const ROWS: u32 = 20u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let quad = vi / 6u;
  let c = corners[vi % 6u];
  let seg = f32(quad % SEGS) + c.x;
  let row = f32(quad / SEGS) + c.y;
  let theta = seg * 6.2831853 / f32(SEGS);
  let rho = carpet_uni.inner_r * pow(carpet_uni.outer_r / carpet_uni.inner_r, row / f32(ROWS));
  var xz = frame.camera_pos.xz + rho * vec2f(cos(theta), sin(theta));
  // Clamp to the stand region — geometry outside the stand degenerates.
  xz = clamp(xz, vec2f(-carpet_uni.stand_radius), vec2f(carpet_uni.stand_radius));
  let world = vec3f(xz.x, terrain_height(xz) + carpet_uni.canopy_h, xz.y);
  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let dc = distance(in.world.xz, frame.camera_pos.xz);
  // Stochastic dissolve-in over the fade band, in 1m world-hash patches.
  let ramp = clamp((dc - carpet_uni.inner_r) / carpet_uni.fade_band, 0.0, 1.0);
  let dcell = vec2i(floor(in.world.xz));
  let xi = hash_f32(hash3(carpet_uni.seed ^ 0x77u, bitcast<u32>(dcell.x), bitcast<u32>(dcell.y)));
  if (xi > ramp) { discard; }

  // Species mottle: 2.5m patches pick a species by stand coverage weight.
  let mcell = vec2i(floor(in.world.xz / 2.5));
  let mh = hash3(carpet_uni.seed ^ 0x5bd1u, bitcast<u32>(mcell.x), bitcast<u32>(mcell.y));
  let sel = hash_f32(mh);
  var tint = carpet_uni.colors[0].rgb;
  var prev = 0.0;
  for (var i = 0u; i < 4u; i = i + 1u) {
    if (i < carpet_uni.n_entries) {
      let cw = carpet_uni.colors[i].w;
      if (sel >= prev && sel < cw) { tint = carpet_uni.colors[i].rgb; }
      prev = cw;
    }
  }
  tint = tint * (0.82 + 0.30 * hash_f32(hash2(mh, 7u)));

  // Canopy normal statistics: terrain normal pulled upward, hash-perturbed.
  let tn = terrain_normal(in.world.xz);
  let jx = hash_f32(hash2(mh, 11u)) - 0.5;
  let jz = hash_f32(hash2(mh, 13u)) - 0.5;
  let n = normalize(tn + vec3f(jx * 0.5, 0.6, jz * 0.5));

  // Statistical wind shimmer — gust waves sweep brightness across the field.
  let along = dot(in.world.xz, frame.wind_dir);
  let shimmer = 1.0 + 0.06 * frame.wind_strength * wind_gust(frame.time)
    * sin(along * 0.35 - frame.time * 1.9);

  var color = light_surface(tint * 0.9 * shimmer, n, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
