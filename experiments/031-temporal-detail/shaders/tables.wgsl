// Uniform blocks shared by the three runtime shaders (cull / cards / canopy).
// Field order must match the writes in main.ts.

// Per-stand-entry, rewritten every frame (planes + params).
struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  origin_cell: vec2f,      // south-west scatter cell of the region (stand-clamped)
  side_x: f32,             // region width in cells
  side_z: f32,             // region depth in cells
  seed: f32,
  entry_index: f32,        // index into stand_table
  region_r: f32,           // region radius (m) around the camera
  near_r: f32,             // card-cloud radius (m); beyond it plants collapse
  cap_near: f32,           // instance capacity of the near list
  cap_far: f32,            // instance capacity of the far list
  alpha_ref: f32,
  bottom_shade: f32,
  tint_var: f32,
  ao_strength: f32,
  sun_shadow: f32,
  top_frac: f32,           // crown card height as a fraction of the part box
  cull_radius: f32,        // whole-plant bounding sphere radius at scale 1
  whole_y1: f32,           // whole-plant top (m at scale 1) — wind + volume ref
  _pad0: f32,
  _pad1: f32,
}

// Static per species: the baked part boxes and per-tile coverage boxes.
const N_PARTS: u32 = 5u;
const N_VIEWS: u32 = 9u;
const N_SIDE: u32 = 8u;
const ATLAS_GRID: f32 = 7.0;
const ATLAS_PX: f32 = 1792.0;

struct AtlasInfo {
  part_box: array<vec4f, 5>,   // cx, cz, y0, y1 (metres at scale 1)
  part_r: array<vec4f, 5>,     // x: horizontal support radius
  tiles: array<vec4f, 45>,     // coverage box a0, a1, b0, b1 (card-normalized)
}

// The world-anchored canopy cache (density + sun/sky transmittance).
// Grid dimensions live here so the reader and the writer cannot disagree.
const CANOPY_NX: f32 = 128.0;

const CANOPY_MAX_ENTRIES: u32 = 8u;

struct CanopyInfo {
  entry_geo: array<vec4f, 8>, // per stand entry: x = radius, y = top height (scale 1)
  win_cell: vec2f,            // world cell index of the window's min corner
  splat_origin: vec2f,        // scatter cell rect origin for the splat dispatch
  splat_side: vec2f,          // scatter cell rect size
  n_entries: f32,
  seed: f32,
  cell: f32,                  // xz cell size (m)
  y_top: f32,                 // vertical span (m above the plant base plane)
  density_scale: f32,
  groups: f32,                // slabs the light pass is split into
  group: f32,                 // slab refreshed by this dispatch
  sun_step: f32,
  sky_k: f32,
  _cpad0: f32,
}
