// Offscreen depth-light-field bake: renders the source mesh orthographically
// from one hemi-octahedral view direction into one r8unorm atlas tile (set via
// viewport/scissor). One draw per tile; the per-tile basis comes from the
// BakeTile uniform (dynamic offset). Only DEPTH along the capture axis is
// stored — appearance is factored out into the 3D volumes baked on CPU.

struct BakeTile {
  center: vec3f, inv_r: f32,
  right: vec3f, _p0: f32,
  up: vec3f, _p1: f32,
  fwd: vec3f, _p2: f32,   // points from plant toward the capture camera
  bmin: vec3f, _p3: f32,
  bmax: vec3f, _p4: f32,
}
@group(0) @binding(0) var<uniform> bake_tile: BakeTile;

struct VOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vs(@location(0) a0: vec4<u32>) -> VOut {
  let q = vec3f(f32(a0.x), f32(a0.y), f32(a0.z)) / 65535.0;
  let p = bake_tile.bmin + q * (bake_tile.bmax - bake_tile.bmin);
  let rel = p - bake_tile.center;
  // Orthographic projection onto the tile basis. z mapped to [0,1]: points
  // nearer the capture camera (larger dot with fwd) get smaller z, so
  // depthCompare 'less' keeps the first hit of each ortho ray bundle.
  let cx = dot(rel, bake_tile.right) * bake_tile.inv_r;
  let cy = dot(rel, bake_tile.up) * bake_tile.inv_r;
  let cz = 0.5 - dot(rel, bake_tile.fwd) * bake_tile.inv_r * 0.5;
  var o: VOut;
  o.pos = vec4f(cx, cy, cz, 1.0);
  return o;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // Fragment z is the ortho ray depth in [0,1]; rescale so exactly 1.0 stays
  // reserved as the "miss" sentinel (the clear value).
  return vec4f(pos.z * (254.0 / 255.0), 0.0, 0.0, 1.0);
}

// --- downsample: 2x2 supersampled strip -> final rg8 atlas row --------------
// r = min depth of the hit subtexels (1.0 sentinel if none), g = fractional
// coverage. Fractional coverage is what keeps sub-texel fluff wispy at
// runtime instead of ballooning to its convex hull.

@group(0) @binding(0) var strip_tex: texture_2d<f32>;

@vertex
fn vs_down(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  // Fullscreen triangle; the caller's viewport maps it onto one atlas row.
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_down(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let ax = u32(pos.x);
  let ay = u32(pos.y) % 128u; // atlas rows are 128px; the strip is one row
  let base = vec2u(ax * 2u, ay * 2u);
  var min_d = 1.0;
  var hits = 0.0;
  for (var sy = 0u; sy < 2u; sy++) {
    for (var sx = 0u; sx < 2u; sx++) {
      let d = textureLoad(strip_tex, base + vec2u(sx, sy), 0).r;
      if (d < 0.998) {
        hits += 1.0;
        min_d = min(min_d, d);
      }
    }
  }
  let cov = hits * 0.25;
  var depth_out = 1.0;
  if (hits > 0.0) { depth_out = min_d; }
  return vec4f(depth_out, cov, 0.0, 1.0);
}
