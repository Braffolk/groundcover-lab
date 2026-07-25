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
  // Optional habitat band: this entry only grows where the shared wetness
  // field is near wet_center, thinning out over wet_width. wet_width = 0
  // disables it entirely (uniform cover, the historical behaviour).
  wet_center: f32,
  wet_width: f32,
  // Carpet layout. >0 lays this species out as a SEAMLESS MAT: carpet_div x
  // carpet_div periodic tiles per scatter cell, grid-snapped, constant scale,
  // and rotated only in 90-degree steps — the three things a periodic square
  // tile needs to abut its neighbours invisibly. 0 = ordinary per-plant
  // scatter. In carpet mode wet_center/wet_width are read as a half-open
  // interval [c - w/2, c + w/2) and the carpet entries PARTITION the wetness
  // axis, so exactly one of them claims each grid node and the mat has no
  // holes.
  carpet_div: f32,
  /**
   * Horizontal footprint of the species at scale 1 (metres) — the periodic
   * tile size, or 0 if unknown. NEVER size a plant's width from height_scale:
   * that is only ~right for tall grasses, and it makes a ground carpet like
   * Sphagnum (0.07m tall, 0.24m wide) about 3.5x too small, leaving gaps in
   * what should be a closed mat. Use this, or the extents you baked yourself.
   */
  footprint_m: f32,
  /**
   * How much this species lies into the terrain plane instead of standing
   * upright: 1 = fully aligned to the terrain normal (correct for mats and
   * carpets, which must follow the ground or they bury one edge and float the
   * other), ~0.2-0.4 for tall plants (real grass grows upright regardless of
   * slope), 0 = always vertical. Apply it with plant_basis() in terrain.wgsl.
   */
  slope_align: f32,
  _pad0: f32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
// rgba16float: r = height (m), g = normal.x, b = normal.z, a unused.
@group(0) @binding(1) var terrain_heightmap: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<storage, read> stand_table: array<StandEntry>;
