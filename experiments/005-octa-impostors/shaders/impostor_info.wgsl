#include "src/wgsl/frame.wgsl"

// Shared layout + math for the octahedral view-set impostors: the per-stand-
// entry uniform (rewritten every frame by main.ts), the compacted plant record
// the cull pass emits, and the hemi-octahedral direction <-> view-grid mapping,
// which must stay bit-identical to the TypeScript twin in bake.ts (the bake
// builds the view bases with exactly these formulas).

// GRID_N x GRID_N view nodes over the upper hemisphere, one texture-array
// layer each. Node (i, j) sits at oct-uv (i/(N-1), j/(N-1)) and its layer
// index is j * GRID_N + i. Keep in sync with GRID_N / TILE in bake.ts —
// main.ts validates the artifact header against these constants.
const GRID_N: i32 = 11;
const TILE_PX: f32 = 128.0;
// The card is the plant AABB projected along the ACTUAL view direction; the
// three blended views look from slightly different directions, so their
// content can reach a little past that. Pad the quad, not the tiles.
const CARD_PAD: f32 = 1.10;

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  region: vec4f,           // origin cell xz (already clamped to the stand), side_x, side_z
  setup: vec4f,            // seed, stand entry index, region radius (m), alpha ref
  center: vec4f,           // mesh-frame card centre xyz, plant height y1-y0
  half_ext: vec4f,         // mesh-frame AABB half extents xyz, mesh y0
  bucket_r: vec4f,         // outer radius (m) of the four front-to-back draw buckets
  bucket_base: vec4f,      // first instance slot of each bucket
  bucket_cap: vec4f,       // instance capacity of each bucket
  shape: vec4f,            // cull sphere radius, core radius (near fade), view tint, _pad
}

/** One surviving plant. 16 B — the cull pass writes it, the draw reads it. */
struct PlantInst {
  pos: vec3f,
  packed: u32,             // yaw 10b | scale 12b (0..4 m) | phase 10b
}

/**
 * One baked view: its orthographic basis in the plant's local (yaw-unrotated)
 * frame plus the half-extents that basis was captured with. Built once on the
 * CPU per species; the fragment shader reprojects into it to get atlas uv.
 */
struct ViewBasis {
  right: vec4f,            // xyz axis, w = ortho half extent along u (m, unit scale)
  up: vec4f,               // xyz axis, w = ortho half extent along v
}

fn rot_y_cs(v: vec3f, cs: vec2f) -> vec3f {
  return vec3f(cs.x * v.x + cs.y * v.z, v.y, -cs.y * v.x + cs.x * v.z);
}

/**
 * Card/view right axis: cross(+Y, d), with a fixed fallback where d is
 * vertical (the straight-down view node and plants directly below the
 * camera). bake.ts uses the identical rule so uv reprojection is exact.
 */
fn view_right(d: vec3f) -> vec3f {
  let h = vec2f(d.z, -d.x);
  let len = length(h);
  if (len < 1.0e-4) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return vec3f(h.x / len, 0.0, h.y / len);
}

/** Hemi-octahedral projection of a unit direction with d.y >= 0 into [0,1]^2. */
fn hemioct_uv(d: vec3f) -> vec2f {
  let n = d / max(abs(d.x) + abs(d.y) + abs(d.z), 1.0e-6);
  return vec2f(n.x + n.z, n.z - n.x) * 0.5 + vec2f(0.5);
}

/** Octahedral normal decode, y-primary — matches GcMesh.normalAt(). */
fn oct_decode_y(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}
