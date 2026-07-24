#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Instance layout: two vec4 per plant — see SCATTER_INSTANCE_FLOATS.
// [x, y, z, yaw] [scale, speciesIndex, phase, 0]
@group(1) @binding(0) var<storage, read> instances: array<vec4f>;

struct Params {
  height: f32,
  sway: f32,
  _pad0: f32,
  _pad1: f32,
}
@group(1) @binding(1) var<uniform> params: Params;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) world: vec3f,
  @location(2) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let d0 = instances[ii * 2u];
  let d1 = instances[ii * 2u + 1u];
  let base = d0.xyz;
  let yaw = d0.w;
  let scale = d1.x;
  let sp = species_table[u32(d1.y)];
  let phase = d1.z;

  var corners = array<vec2f, 6>(
    vec2f(-0.5, 0.0), vec2f(0.5, 0.0), vec2f(-0.5, 1.0),
    vec2f(0.5, 0.0), vec2f(0.5, 1.0), vec2f(-0.5, 1.0),
  );
  let c = corners[vi];

  // Cylindrical billboard facing the camera.
  let to_cam = frame.camera_pos - base;
  let dist = length(to_cam.xz);
  let fwd = normalize(to_cam.xz + vec2f(1e-5, 0.0));
  let right = vec2f(-fwd.y, fwd.x);

  let h = params.height * sp.height_scale * scale;
  let w = h * 0.22;
  // Camera-inside-plant rule: collapse the blade as the camera enters it.
  let fade = clamp((dist - 0.25) / 0.4, 0.0, 1.0);

  var world = base + vec3f(right.x * c.x * w, c.y * h, right.y * c.x * w) * fade;
  world += wind_sway(base, frame.time, sp.sway * params.sway, phase) * c.y;

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  let tint = 0.8 + 0.4 * fract(yaw * 0.63661977);
  out.color = vec3f(0.16, 0.31, 0.09) * tint;
  out.world = world;
  out.uv = c;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Taper the quad into a blade profile.
  if (abs(in.uv.x) > 0.5 * (1.0 - in.uv.y * 0.85)) {
    discard;
  }
  let ground = terrain_normal(in.world.xz);
  let n = normalize(ground + vec3f(0.0, 1.2, 0.0));
  var color = light_surface(in.color * (0.55 + 0.45 * in.uv.y), n, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
