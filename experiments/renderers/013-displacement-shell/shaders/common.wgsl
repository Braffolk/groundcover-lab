// Shared uniform block for 013-displacement-shell. TS mirror: main.ts writeFrameData().

struct Globals {
  region_min: vec2i,      // scatter-cell coords of the culled region corner
  region_dims: vec2i,     // region size in cells (clamped to the stand)
  carpet_min: vec2i,      // same, for the (much smaller) carpet region
  carpet_dims: vec2i,
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
  shell_ramp: f32,        // metres over which the shell rises to full canopy
  carpet_outer: f32,      // carpet (mat) geometry radius; shell takes over past
  carpet_px: f32,         // target screen pixels per baked grid quad
  shell_3d: f32,          // 1 = ramp the shell by 3D distance (carpet stands)
  entry_count: u32,       // stand entries, for the shell's wetness zone mix
  shell_r0: f32,          // annulus innermost HORIZONTAL radius (m)
  shell_ratio: f32,       // geometric row growth of the annulus
  // Per-draw capacities: [0..2] strand rings, [3..8] carpet LOD buckets.
  caps: array<vec4u, 3>,
  // Record offset of each carpet LOD bucket inside the shared carpet buffer.
  carpet_base: array<vec4u, 2>,
  // Per SPECIES-CATALOG index: rgb = baked canopy albedo, w = baked topH (m).
  canopy: array<vec4f, 6>,
  // Stand coverage weight per species-catalog index (for the far-shell mix),
  // packed 4 per vec4.
  species_density: array<vec4f, 2>,
  // x = shell albedo trim (a mat aggregate is darker than a flat plane because
  // its own relief shades itself), y..w spare.
  shell_tune: vec4f,
}

struct RingInfo {
  t_ring: u32,        // strand rings: stations drawn
  ring_index: u32,
  grid_stride: u32,   // carpet buckets: texel stride (LOD); 0 for strand rings
  grid_bands: u32,    // carpet buckets: quads per axis drawn (GRID_N / stride)
}

fn yaw_rotate(p: vec3f, cy: f32, sy: f32) -> vec3f {
  return vec3f(p.x * cy - p.z * sy, p.y, p.x * sy + p.z * cy);
}

/// Rotate v by the minimal rotation taking +Y onto `up` (unit). Used to lay a
/// carpet's baked normals into the local ground frame; positions conform
/// per-vertex instead, so the mat keeps tiling in plan view.
fn tilt_from_up(v: vec3f, up: vec3f) -> vec3f {
  let c = up.y;
  if (c > 0.99999) { return v; }
  let w = vec3f(up.z, 0.0, -up.x);
  let wv = cross(w, v);
  return v + wv + cross(w, wv) / (1.0 + max(c, -0.99));
}
