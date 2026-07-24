// Shared @group(0) — bound by the harness for every pass of every experiment.
// Layout must match src/scene/frameUbo.ts exactly.

struct Frame {
  view: mat4x4f,
  proj: mat4x4f,
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  camera_pos: vec3f,
  time: f32,
  sun_dir: vec3f,
  dt: f32,
  sun_color: vec3f,
  wind_strength: f32,
  ambient: vec3f,
  wind_gust_freq: f32,
  wind_dir: vec2f,
  viewport: vec2f,
  terrain_size: f32,
  terrain_height_scale: f32,
  terrain_resolution: f32,
  frame_index: f32,
}

struct Species {
  density: f32,      // plants per m^2 at density scale 1 (max 8)
  scale_min: f32,
  scale_max: f32,
  sway: f32,         // 0 = rigid (moss), 1 = full wind response
  height_scale: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
// rgba16float: r = height (m), g = normal.x, b = normal.z, a unused.
@group(0) @binding(1) var terrain_heightmap: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<storage, read> species_table: array<Species>;
