// Shared uniform block for 013-displacement-shell. TS mirror: main.ts writeGlobals().

struct Globals {
  region_min: vec2i,      // scatter-cell coords of the culled region corner
  region_dims: vec2i,     // region size in cells (clamped to the stand)
  seed: u32,
  s0: f32,                // strand budget at zero distance (ring0 rows)
  r0: f32,                // full-detail radius; strand count falls off as r0/d
  r1: f32,                // ring1 -> ring2 boundary
  r_outer: f32,           // plant region edge; the shell takes over past this
  width_scale: f32,
  s_baked: f32,           // strand rows baked in the field texture
  t_baked: f32,           // station columns baked
  shell_in: f32,          // far-shell inner radius
  shell_h: f32,           // far-shell canopy height (m, stand-mixed)
  stand_radius: f32,
  ring_debug: f32,        // OUR `debugRings` param — not the global debug view
                          // selector (that one lives in frame.debug_mode).
  shell_bob: f32,         // density-weighted mean sway for the shell wind bob
  cap0: u32,
  cap1: u32,
  cap2: u32,
  // Per SPECIES-CATALOG index: rgb = baked canopy albedo, w = baked topH (m).
  canopy: array<vec4f, 3>,
  // Stand density per species-catalog index (for the far-shell mix).
  species_density: vec4f,
}

struct RingInfo {
  t_ring: u32,        // stations drawn in this ring
  ring_index: u32,
  _pad0: u32,
  _pad1: u32,
}

fn yaw_rotate(p: vec3f, cy: f32, sy: f32) -> vec3f {
  return vec3f(p.x * cy - p.z * sy, p.y, p.x * sy + p.z * cy);
}
