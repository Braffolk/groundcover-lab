// Shared uniform blocks for 034-strand-shell-fidelity.
// TS mirror: main.ts writeFrameData() — keep the field order identical.

struct Globals {
  region_min: vec2i,      // scatter-cell coords of the culled region corner
  region_dims: vec2i,     // region size in cells (clamped to the stand)
  seed: u32,
  blades_param: f32,      // blade rows at full detail (render param)
  blades_baked: f32,      // rows present in the baked library (BLADES)
  stations_baked: f32,    // station columns baked (STATIONS)
  c_rows: f32,            // blades_param * r_full — the "rows = c_rows/d" law
  r_outer: f32,           // plant region edge; the canopy shell takes over
  width_scale: f32,
  cover_pow: f32,         // coverage-conservation exponent of the width boost
  keel: f32,              // cross-section bulge (rings that draw 3 verts wide)
  curl: f32,              // cross-blade shading-normal rotation
  orient_near: f32,       // below this the ribbon uses its BAKED plane
  orient_far: f32,        // above this it is camera-facing (sub-pixel anyway)
  min_px: f32,            // minimum projected blade width, pixels
  ao_min: f32,            // albedo floor of the baked occlusion
  up_bias: f32,           // gentle +Y nudge of blade normals (NOT a flatten)
  shell_in: f32,          // far-shell inner radius
  shell_h: f32,           // far-shell canopy height (m, stand-mixed)
  stand_radius: f32,
  shell_bob: f32,         // density-weighted mean sway for the shell wind bob
  ring_debug: f32,        // OUR debugRings param — never frame.debug_mode
  bend_amp: f32,          // per-blade static bend that breaks the clone look
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  // Per ring: x = outer radius, y = instance base, z = capacity, w = rows drawn
  rings: array<vec4f, 5>,
  // Per SPECIES-CATALOG index: rgb = baked canopy albedo, w = baked topH (m).
  canopy: array<vec4f, 3>,
  // Stand density per species-catalog index (for the far-shell mix).
  species_density: vec4f,
}

struct RingInfo {
  ring_index: u32,
  stations: u32,      // station columns drawn in this ring
  lat_count: u32,     // vertices across the ribbon: 3 = keeled, 2 = flat
  row_gain: f32,      // per-ring row-count multiplier (far rings shed rows)
}

fn yaw_rotate(p: vec3f, cy: f32, sy: f32) -> vec3f {
  return vec3f(p.x * cy - p.z * sy, p.y, p.x * sy + p.z * cy);
}
