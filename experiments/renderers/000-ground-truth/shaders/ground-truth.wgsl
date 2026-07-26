#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

struct Params {
  bounds_min: vec3f,
  tiles: f32,
  bounds_max: vec3f,
  sway: f32,
  tile: vec2f,
  top_h: f32,
  _pad: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) world: vec3f,
}

fn oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var n = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
  if (n.y < 0.0) {
    let sx = select(-1.0, 1.0, f.x >= 0.0);
    let sz = select(-1.0, 1.0, f.y >= 0.0);
    n = vec3f((1.0 - abs(f.y)) * sx, n.y, (1.0 - abs(f.x)) * sz);
  }
  return normalize(n);
}

@vertex
fn vs_main(
  @location(0) a0: vec4<u32>, // x, y, z, r
  @location(1) a1: vec4<u32>, // g, b, octU, octV
  @builtin(instance_index) ii: u32,
) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  var local = params.bounds_min + q * (params.bounds_max - params.bounds_min);

  // Center the tile grid on the origin.
  let side = u32(params.tiles);
  let gx = f32(i32(ii % side)) - (params.tiles - 1.0) * 0.5;
  let gz = f32(i32(ii / side)) - (params.tiles - 1.0) * 0.5;
  let tile_center = vec2f(gx * params.tile.x, gz * params.tile.y);
  let ground = terrain_height(tile_center);
  var world = vec3f(local.x + tile_center.x, local.y + ground, local.z + tile_center.y);

  // Wind: tips sway, roots stay — weight by normalized height in the tile.
  // Reference is stand-independent: fixed calamagrostis sway response.
  let weight = clamp(local.y / params.top_h, 0.0, 1.0);
  world += wind_sway(world, frame.time, 0.6 * params.sway, f32(ii) * 2.39996) * weight * weight;

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.color = vec3f(f32(a0.w), f32(a1.x), f32(a1.y)) / 65535.0;
  out.normal = oct_decode(vec2f(f32(a1.z), f32(a1.w)) / 65535.0);
  out.world = world;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Thin foliage is lit from both sides.
  var n = normalize(in.normal);
  let to_cam = frame.camera_pos - in.world;
  if (dot(n, to_cam) < 0.0) {
    n = -n;
  }
  var color = light_surface(in.color, n, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
