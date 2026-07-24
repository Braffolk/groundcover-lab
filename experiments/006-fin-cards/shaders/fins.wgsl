#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Trig-weighted crossed fins. Every plant is 5 STATIC quads in world space:
// 3 vertical fins crossed at 60° through the plant axis + 2 horizontal slab
// cards. Nothing billboards — parallax, intersections and occlusion are real
// 3D, which is what makes the motion rock-solid. The classic crossed-card
// artifacts are dissolved trigonometrically:
//   * a fin fades (hashed-alpha) as the view goes edge-on to it — weight
//     smoothstep(|cos(view, fin normal)|); the two other fins are near their
//     best angle exactly then, so coverage holds;
//   * each fin shows a DIFFERENT baked azimuth view per side (6 views on 3
//     quads, back side U-mirrored), so orbiting changes real content instead
//     of sliding one poster around;
//   * looking down, the fins (and their star-shaped crossing) dissolve via a
//     vd.y smoothstep while two top-down slab cards (lower/upper half of the
//     plant, baked separately) take over with genuine inter-card parallax.
// Placement is the shared scatter twin evaluated over a bounded cell region
// around the camera — per-frame cost is independent of total plant count.

struct Meta {
  cx: f32, yc: f32, cz: f32, half_w: f32,
  half_h: f32, y_low: f32, y_high: f32, radius: f32,
  origin_cell_x: f32, origin_cell_z: f32, side: f32, seed: f32,
  max_dist: f32, entry_index: f32, edge_fade: f32, top_blend: f32,
  alpha_sharp: f32, _p0: f32, _p1: f32, _p2: f32,
}
@group(1) @binding(0) var<uniform> fin: Meta;
@group(1) @binding(1) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(2) var atlas_normal: texture_2d<f32>;
@group(1) @binding(3) var atlas_samp: sampler;

const TILE_COLS = 4.0;
const TILE_ROWS = 2.0;
const TILE_PX = 512.0;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
  @location(2) @interpolate(flat) tile: u32,
  @location(3) @interpolate(flat) yaw: f32,
  @location(4) @interpolate(flat) alpha: f32,
  @location(5) @interpolate(flat) dither_seed: f32,
}

fn degenerate() -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let side = u32(fin.side);
  let slot = ii % SCATTER_MAX_PER_CELL;
  let cell_lin = ii / SCATTER_MAX_PER_CELL;
  let cxi = i32(fin.origin_cell_x) + i32(cell_lin % side);
  let czi = i32(fin.origin_cell_z) + i32(cell_lin / side);
  let entry_index = u32(fin.entry_index);

  let sp = scatter_candidate(u32(fin.seed), entry_index, vec2i(cxi, czi), slot);
  if (!sp.exists) { return degenerate(); }

  let scale = sp.scale;
  let yaw = sp.yaw;
  let center = sp.pos + rot_y(vec3f(fin.cx, fin.yc, fin.cz) * scale, yaw);

  let to_cam = frame.camera_pos - center;
  let dcam = length(to_cam);
  let r_world = fin.radius * scale;
  // Fade far cards out toward the region boundary; fade out (collapse) when
  // the camera moves inside the plant.
  let far_fade = 1.0 - smoothstep(fin.max_dist * 0.72, fin.max_dist * 0.97, dcam);
  let near_fade = smoothstep(r_world * 0.35, r_world * 0.95, dcam);
  let fade = far_fade * near_fade;
  if (fade < 0.004) { return degenerate(); }

  let vd = to_cam / max(dcam, 1e-4);

  let quad = vi / 6u;
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];

  var world: vec3f;
  var uv: vec2f;
  var tile: u32;
  var w_card: f32;
  var height_norm: f32;

  if (quad < 3u) {
    // -------- vertical fin (crossed at 60°) --------
    let phi = f32(quad) * 1.0471975512;
    let n_w = rot_y(vec3f(sin(phi), 0.0, cos(phi)), yaw);
    let t_w = rot_y(vec3f(cos(phi), 0.0, -sin(phi)), yaw);
    world = center + t_w * (c.x * fin.half_w * scale) + vec3f(0.0, c.y * fin.half_h * scale, 0.0);

    // Per-side content: front side shows azimuth view `quad`; back side shows
    // the opposite capture (quad+3) with U mirrored (its right axis is -t).
    let facing = dot(vd, n_w);
    if (facing >= 0.0) {
      tile = quad;
      uv = vec2f(0.5 + 0.5 * c.x, 0.5 - 0.5 * c.y);
    } else {
      tile = quad + 3u;
      uv = vec2f(0.5 - 0.5 * c.x, 0.5 - 0.5 * c.y);
    }

    // Trig weights: dissolve edge-on fins; dissolve ALL fins as the view goes
    // overhead (that removes the star crossing — the slab cards take over).
    let w_edge = smoothstep(0.03, fin.edge_fade, abs(facing));
    let w_not_top = 1.0 - smoothstep(fin.top_blend, 0.98, vd.y);
    w_card = w_edge * w_not_top;
    height_norm = 0.5 + 0.5 * c.y;
  } else {
    // -------- horizontal slab card (lower / upper half of the plant) --------
    let low = quad == 3u;
    let y_card = select(fin.y_high, fin.y_low, low);
    tile = select(7u, 6u, low);
    let x_w = rot_y(vec3f(1.0, 0.0, 0.0), yaw);
    let z_w = rot_y(vec3f(0.0, 0.0, 1.0), yaw);
    world = sp.pos + rot_y(vec3f(fin.cx, 0.0, fin.cz) * scale, yaw)
      + vec3f(0.0, y_card * scale, 0.0)
      + x_w * (c.x * fin.half_w * scale)
      + z_w * (c.y * fin.half_w * scale);
    uv = vec2f(0.5 + 0.5 * c.x, 0.5 + 0.5 * c.y);

    // Fade in with view elevation; near-grazing they are near-invisible lines
    // anyway, the smoothstep just removes the last sliver artifact.
    w_card = smoothstep(0.05, 0.28, abs(vd.y));
    height_norm = clamp((y_card - (fin.yc - fin.half_h)) / max(2.0 * fin.half_h, 1e-4), 0.0, 1.0);
  }

  let alpha = w_card * fade;
  if (alpha < 0.004) { return degenerate(); }

  // Wind: shared sway, tips move, roots pinned.
  world += wind_sway(sp.pos, frame.time, stand_table[entry_index].sway, sp.phase) * height_norm;

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.uv = uv;
  out.tile = tile;
  out.yaw = yaw;
  out.alpha = alpha;
  // Decorrelate the dissolve pattern between overlapping cards and plants.
  out.dither_seed = f32(tile) * 23.7 + sp.phase * 41.3 + sp.yaw * 13.1;
  return out;
}

// Interleaved gradient noise — stable per-pixel dissolve, no sorting needed.
fn ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let col = f32(in.tile % 4u);
  let row = f32(in.tile / 4u);
  // Keep bilinear+mip taps inside the tile (mips are clamped to level 4).
  let inset = 8.0 / TILE_PX;
  let tuv = clamp(in.uv, vec2f(inset), vec2f(1.0 - inset));
  let auv = (vec2f(col, row) + tuv) * vec2f(1.0 / TILE_COLS, 1.0 / TILE_ROWS);

  // Atlas mips are premultiplied — un-premultiply after the (trilinear) fetch.
  let alb = textureSample(atlas_albedo, atlas_samp, auv);
  let nrm = textureSample(atlas_normal, atlas_samp, auv);

  let coverage = min(alb.a * fin.alpha_sharp, 1.0);
  let a = coverage * in.alpha;
  let d = ign(in.pos.xy + vec2f(in.dither_seed, in.dither_seed * 1.618));
  if (a < clamp(d, 0.002, 0.998)) { discard; }

  let albedo = alb.rgb / max(alb.a, 1e-3);
  var n_local = nrm.xyz / max(nrm.a, 1e-3) * 2.0 - 1.0;
  if (dot(n_local, n_local) < 1e-4) { n_local = vec3f(0.0, 1.0, 0.0); }
  let n_world = rot_y(normalize(n_local), in.yaw);

  var color = light_surface(albedo, n_world, in.world);
  color = apply_fog(color, in.world);
  return vec4f(color, 1.0);
}
