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
  // Global debug view selector — see src/wgsl/debug.wgsl. EVERY renderer must
  // honour it (CLAUDE.md requirement), so albedo/normals/lighting/coverage/
  // depth can be inspected identically across methods.
  debug_mode: f32,
  _frame_pad0: f32,
  _frame_pad1: f32,
  _frame_pad2: f32,
}

// One row per species entry of the ACTIVE STAND (src/scene/stands.ts) —
// the standardized placement setup every renderer draws. Indexed by the
// stand entry index carried in scatter instance data.
struct StandEntry {
  density: f32,       // plants per m^2 (max 8)
  scale_min: f32,
  scale_max: f32,
  sway: f32,          // 0 = rigid (moss), 1 = full wind response
  height_scale: f32,  // nominal plant height (m) from the species catalog
  species_index: f32, // global species catalog index (mesh identity)
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
// rgba16float: r = height (m), g = normal.x, b = normal.z, a unused.
@group(0) @binding(1) var terrain_heightmap: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<storage, read> stand_table: array<StandEntry>;
