#include "src/wgsl/frame.wgsl"
#include "src/wgsl/hash.wgsl"

// Shared definitions for the dissolving canopy: the uniform blocks, the
// dissolve field that decides where the canopy is still individual plants and
// where it is one continuous surface, and the analytic canopy relief the shell
// is displaced by.

const AZ_COUNT: f32 = 8.0;
const NODE_COLS: f32 = 5.0; // atlas columns: node 0 = whole plant, 1..4 quadrants

// Per-stand-entry block, shared by cull.wgsl and tufts.wgsl. Field order must
// match the info[] writes in main.ts.
struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  nodes: array<vec4f, 10>, // per node: (cx, cy, cz, halfW), (y0, y1, azOffset, _)
  region: vec4f,           // origin cell xz, region width/depth in cells
  cfg0: vec4f,             // seed, entry index, region radius, tuft radius
  cfg1: vec4f,             // band-0 capacity, band-1 capacity, alpha ref, ground shade
  cfg2: vec4f,             // dissolve grazing, dissolve overhead, plant height, root half width
  cfg3: vec4f,             // cull radius, _, _, _
}

// Canopy shell block (shell.wgsl).
struct ShellInfo {
  planes: array<vec4f, 6>,
  levels: array<vec4f, 2>,      // per level: snapped origin xz, cell size, _
  holes: array<vec4f, 2>,       // per level: inner hole lo.xz, hi.xz (skip if inside)
  p0: vec4f,                    // stand half extent, canopy base height, canopy tile size (m), relief
  p1: vec4f,                    // shell floor threshold, dissolve grazing, dissolve overhead, ground shade
  p2: vec4f,                    // cells per level, grid side, mean sway, edge fade width
  p3: vec4f,                    // side-view flower-band colour (rgb), tile mean luminance
  p4: vec4f,                    // side-view lower-band colour (rgb), canopy top (m)
  p5: vec4f,                    // aux-height -> flower-band mix gain
}

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

/**
 * How continuous the canopy is at a world point: 0 = individual plants,
 * 1 = one surface. The transition radius follows the viewing elevation of that
 * point, which is the physical rule behind the whole method — looking along a
 * canopy you see between the plants far away, looking down at it you see one
 * closed surface almost immediately.
 */
fn dissolve_amount(p: vec3f, r_graze: f32, r_over: f32) -> f32 {
  let to_cam = frame.camera_pos - p;
  let dist = max(length(to_cam), 1.0e-3);
  let sin_elev = clamp(abs(to_cam.y) / dist, 0.0, 1.0);
  let r = mix(r_graze, r_over, smoothstep(0.12, 0.85, sin_elev));
  return smoothstep(r * 0.5, r, dist);
}

/** Coverage multiplier of the per-plant splats — they erode as the shell arrives. */
fn tuft_fade(d: f32) -> f32 {
  return 1.0 - smoothstep(0.62, 1.0, d);
}

/**
 * Coverage threshold of the shell. Above 1.0 nothing can pass, which is how the
 * near field ends up as plants only (and lets whole quads be skipped in the
 * vertex stage instead of rasterized and discarded).
 */
fn shell_threshold(d: f32, floor_thresh: f32) -> f32 {
  return mix(1.2, floor_thresh, smoothstep(0.15, 1.0, d));
}

/** The d below which shell_threshold() can never pass — nothing to rasterize. */
fn shell_dead_d(floor_thresh: f32) -> f32 {
  return 0.15 + 0.85 * (0.2 / max(1.2 - floor_thresh, 1.0e-3));
}

fn vhash(ix: i32, iz: i32) -> f32 {
  return hash_f32(hash2(bitcast<u32>(ix) ^ 0x9e3779b9u, bitcast<u32>(iz)));
}

/** C1-ish value noise in [0,1] — continuous in world space, so the shell's two
 *  grid levels agree on the surface and nothing swims when the camera moves. */
fn vnoise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let w = f * f * (3.0 - 2.0 * f);
  let ix = i32(i.x);
  let iz = i32(i.y);
  let a = vhash(ix, iz);
  let b = vhash(ix + 1, iz);
  let c = vhash(ix, iz + 1);
  let e = vhash(ix + 1, iz + 1);
  return mix(mix(a, b, w.x), mix(c, e, w.x), w.y);
}

/** Canopy top height (m above terrain) of the continuous surface. */
fn canopy_relief(xz: vec2f, base_h: f32, relief: f32) -> f32 {
  let n1 = vnoise2(xz * 0.1) - 0.5;
  let n2 = vnoise2(xz * 0.2778) - 0.5;
  return max(base_h * (1.0 + relief * 0.55 * (n1 + n2 * 0.5)), base_h * 0.25);
}

/** Octahedral decode, y-primary — matches the bake's encode. */
fn oct_decode_ds(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}
