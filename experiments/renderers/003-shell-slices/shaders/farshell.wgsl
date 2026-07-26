#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/debug.wgsl"

// The far canopy shell: beyond the per-plant slice region the entire
// meadow collapses into ONE terrain-draped translucent layer at canopy
// height — the logical end state of "fewer and fewer slices". A polar
// grid around the camera (screen-bounded, plant-count-free) samples the
// baked top-down composites of each species, tiled in world space with
// per-tile random rotation, blended by stand density and a mottle noise.

@group(1) @binding(0) var top0: texture_2d<f32>;
@group(1) @binding(1) var top1: texture_2d<f32>;
@group(1) @binding(2) var top2: texture_2d<f32>;

struct ShellU {
  weights: vec3f,
  stand_r: f32,
  inv_tiles: vec3f,
  avg_h: f32,
  inner_start: f32,
  inner_full: f32,
  r0: f32,
  r1: f32,
  px_factor: f32,
  mip_max: f32,
  pad0: f32,
  pad1: f32,
}
@group(1) @binding(3) var<uniform> shell: ShellU;

const RINGS: u32 = 48u;
const SECTORS: u32 = 96u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
}

fn vn_hash(p: vec2i, salt: u32) -> f32 {
  return hash_f32(hash3(bitcast<u32>(p.x), bitcast<u32>(p.y), salt));
}

fn value_noise(p: vec2f, salt: u32) -> f32 {
  let b = floor(p);
  let f = p - b;
  let i = vec2i(b);
  let u = f * f * (3.0 - 2.0 * f);
  let s00 = vn_hash(i, salt);
  let s10 = vn_hash(i + vec2i(1, 0), salt);
  let s01 = vn_hash(i + vec2i(0, 1), salt);
  let s11 = vn_hash(i + vec2i(1, 1), salt);
  return mix(mix(s00, s10, u.x), mix(s01, s11, u.x), u.y);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  let quad = vi / 6u;
  let corner = vi % 6u;
  let ring = quad / SECTORS;
  let sec = quad % SECTORS;
  var d = vec2f(0.0, 0.0);
  switch corner {
    case 1u, 3u: { d = vec2f(1.0, 0.0); }
    case 2u, 5u: { d = vec2f(0.0, 1.0); }
    case 4u: { d = vec2f(1.0, 1.0); }
    default: {}
  }
  let t = (f32(ring) + d.y) / f32(RINGS);
  let ang = (f32(sec) + d.x) / f32(SECTORS) * 6.2831853;
  let r = shell.r0 * pow(shell.r1 / shell.r0, t);
  let xz = frame.camera_pos.xz + vec2f(cos(ang), sin(ang)) * r;
  let bump = value_noise(xz * 0.53, 3u);
  let y = terrain_height(xz) + shell.avg_h * (0.7 + 0.5 * bump);
  let swy = wind_sway(vec3f(xz.x, y, xz.y), frame.time, 0.6, 0.0);
  let world = vec3f(xz.x + swy.x * 0.5, y + swy.y * 0.4, xz.y + swy.z * 0.5);

  var out: VOut;
  out.world = world;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  return out;
}

fn sample_top(tex: texture_2d<f32>, world_xz: vec2f, inv_tile: f32, lod: f32) -> vec4f {
  // Straight periodic tiling — the textures are seamless by construction
  // (per-tile rotation would break continuity into a visible grid). The
  // species mottle noise hides the repetition.
  let uv = fract(world_xz * inv_tile);
  return textureSampleLevel(tex, linear_sampler, uv, lod);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let w = in.world;
  if (abs(w.x) > shell.stand_r || abs(w.z) > shell.stand_r) {
    discard;
  }
  let dist = distance(w, frame.camera_pos);

  // Dither the inner edge in from under the fading slice-stack plants,
  // keyed on world position so the pattern is camera-stable.
  let dfrac = clamp((dist - shell.inner_start) / (shell.inner_full - shell.inner_start), 0.0, 1.0);
  if (dfrac < 1.0) {
    let dh = hash_f32(hash3(bitcast<u32>(i32(floor(w.x * 29.0))), bitcast<u32>(i32(floor(w.z * 29.0))), 17u));
    if (dh > dfrac) {
      discard;
    }
  }

  var rgb = vec3f(0.0);
  var cov = 0.0;
  var wsum = 0.0;
  // Branch on the stand weight FIRST: a species the stand does not use has
  // weight 0, and then its mottle noise (4 hashes) is pure waste.
  if (shell.weights.x > 0.0) {
    let wt = shell.weights.x * (0.3 + 0.7 * value_noise(w.xz * 0.31, 11u));
    let lod = clamp(log2(max(dist * shell.px_factor * shell.inv_tiles.x, 1.0)), 0.0, shell.mip_max);
    let s = sample_top(top0, w.xz, shell.inv_tiles.x, lod);
    rgb += wt * s.rgb;
    cov += wt * s.a;
    wsum += wt;
  }
  if (shell.weights.y > 0.0) {
    let wt = shell.weights.y * (0.3 + 0.7 * value_noise(w.xz * 0.27, 12u));
    let lod = clamp(log2(max(dist * shell.px_factor * shell.inv_tiles.y, 1.0)), 0.0, shell.mip_max);
    let s = sample_top(top1, w.xz, shell.inv_tiles.y, lod);
    rgb += wt * s.rgb;
    cov += wt * s.a;
    wsum += wt;
  }
  if (shell.weights.z > 0.0) {
    let wt = shell.weights.z * (0.3 + 0.7 * value_noise(w.xz * 0.23, 13u));
    let lod = clamp(log2(max(dist * shell.px_factor * shell.inv_tiles.z, 1.0)), 0.0, shell.mip_max);
    let s = sample_top(top2, w.xz, shell.inv_tiles.z, lod);
    rgb += wt * s.rgb;
    cov += wt * s.a;
    wsum += wt;
  }
  if (wsum <= 0.0) {
    discard;
  }
  let coverage = cov / wsum;
  let albedo = rgb / max(cov, 1e-4);
  let soil = vec3f(0.16, 0.14, 0.09);
  // 0.88: tone toward the slice-stack field, which carries baked
  // self-occlusion — kills the brightness seam at the handover ring.
  let col = mix(soil, albedo, clamp(coverage * 1.4, 0.0, 1.0)) * 0.88;

  let tn = terrain_normal(w.xz);
  let n = normalize(mix(tn, vec3f(0.0, 1.0, 0.0), 0.5));
  var color = light_surface(col, n, w);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, w);
  }
  return vec4f(debug_shade(color, col, n, coverage, w), 1.0);
}
