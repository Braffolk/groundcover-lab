#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// SLICE PRISMS.
//
// A plant is not a picture, it is three slabs of plant. Per azimuth the bake
// cut the mesh into three equal-mass depth slices and recorded, for each, the
// CENTROID DEPTH of the geometry inside it. Here each slice is drawn as its
// own camera-facing card placed at that centroid — offset along the BAKED
// AZIMUTH direction, not along the current view ray. That one detail is what
// makes it 3D rather than three copies of the same flat thing:
//
//   * the offset has a lateral component whenever the view is off the baked
//     azimuth, so the layers SLIDE against each other as the camera orbits;
//   * the offset has a depth component, so perspective magnifies the front
//     slab more than the rear one and the plant opens up as you approach;
//   * each card writes real depth at its own distance, so the slabs occlude
//     one another, other plants interleave between them, and the plant meets
//     the ground over a footprint instead of along a single line;
//   * the residual error of the impostor is (c - c_slab)*sin(delta_azimuth)
//     instead of c*sin(delta_azimuth) — a third of the billboard's — so the
//     45-degree view snap is correspondingly smaller.
//
// Three horizontal height slices do the same job for top-down views.
//
// Half of the view azimuths read their prisms MIRRORED (u flipped, slab order
// reversed, depth negated, mesh-frame normal horizontally negated), which is
// what buys 384px prism tiles inside the same VRAM as a 256px unmirrored set.
//
// Two fixed texture taps per fragment, hard alpha test, depth write, front
// slab first so early-z can reject what it covers. No loops, no marching, no
// dither. Beyond slab_dist the whole thing collapses (continuously) into the
// merged card, which is a plain billboard.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var prism_albedo: texture_2d<f32>;
@group(1) @binding(3) var prism_normal: texture_2d<f32>;
@group(1) @binding(4) var prism_sampler: sampler;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) @interpolate(flat) mirror: f32,   // 1 = tile read mirrored
  @location(5) shade: f32,                       // canopy + grounding occlusion
}

struct Plant {
  base: vec3f,
  axis: vec3f, // base + the baked clump-centre offset, yawed
  yaw: f32,
  scale: f32,
  r: f32,
  h0: f32,
  h1: f32,
  az: u32,     // nearest VIEW azimuth in the plant's own frame (0..7)
  fade: f32,
  d_cam: f32,
  elev: f32,   // |sin| of the view elevation onto the plant
  sway: vec3f,
  right: vec3f,
}

fn load_plant(ii: u32) -> Plant {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  var pl: Plant;
  pl.yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  pl.scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  pl.base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];

  pl.r = info.card_r * pl.scale;
  pl.h0 = info.card_y0 * pl.scale;
  pl.h1 = info.card_y1 * pl.scale;
  // The baked capture is centered on the clump, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  pl.axis = pl.base + rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * pl.scale, pl.yaw);

  let to_cam = frame.camera_pos - pl.axis;
  let len_xz = max(length(to_cam.xz), 1e-4);
  let fwd_xz = to_cam.xz / len_xz;
  pl.right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
  let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -pl.yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  pl.az = u32((i32(round(ang * (f32(N_AZ_VIEW) / 6.2831853))) + i32(N_AZ_VIEW)) % i32(N_AZ_VIEW));
  pl.sway = wind_sway(pl.base, frame.time, entry.sway, phase);

  // Camera-inside fade: 3D distance to the plant's core segment. Region rim:
  // erode to nothing exactly at region_r (the cull matches).
  let seg_y = clamp(frame.camera_pos.y, pl.base.y + pl.h0, pl.base.y + pl.h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(pl.r * 0.35, pl.r * 1.05, d_core);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, len_xz);
  pl.fade = near_fade * edge_fade;

  let centre = pl.base + vec3f(0.0, (pl.h0 + pl.h1) * 0.5, 0.0);
  pl.d_cam = max(distance(centre, frame.camera_pos), 1e-4);
  // Signed elevation above the CANOPY TOP, not above the plant's middle: a
  // horizontal slice through the plant only describes anything once you are
  // actually looking down onto the canopy. Measured at the middle instead, an
  // eye-level camera skimming the tips still counts as "above" every plant
  // within a metre and lays pale edge-on lids through the near field.
  pl.elev = (frame.camera_pos.y - (pl.base.y + pl.h1)) / pl.d_cam;
  return pl;
}

/**
 * A vertical prism card: camera-facing, standing on the ground, its quad
 * shrunk to the tile's tight coverage box, positioned at `centre_ws`.
 * `mirror` flips the tile's U axis (and, in the fragment stage, the normal).
 */
fn side_card(pl: Plant, tile: u32, centre_ws: vec3f, c: vec2f, shade_mul: f32, fade: f32, mirror: f32) -> VOut {
  let geo = info.geo_rect[tile];
  let uvr = info.uv_rect[tile];
  let s = c.x * 0.5 + 0.5;
  let gu = mix(geo.x, geo.z, s);
  let gv = mix(geo.w, geo.y, c.y);
  let hfrac = 1.0 - gv;
  // Stored lateral position of this column; a mirrored tile reads its U axis
  // backwards, so the same texels land on the other side of the plant.
  let lateral = (gu * 2.0 - 1.0) * pl.r * (1.0 - 2.0 * mirror);

  var world = centre_ws + pl.right * lateral;
  world.y = pl.base.y + mix(pl.h0, pl.h1, hfrac);
  world += pl.sway * hfrac;

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards — emit it behind the near plane
  // instead of rasterizing a guaranteed-empty quad.
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = vec2f(mix(uvr.x, uvr.z, s), mix(uvr.w, uvr.y, c.y));
  out.world = world;
  out.yaw_cs = vec2f(cos(pl.yaw), sin(pl.yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.mirror = mirror;
  out.shade = shade_mul * mix(1.0 - info.bottom_shade, 1.0, hfrac);
  return out;
}

/** A horizontal prism card: one height slice seen from straight above. */
fn top_card(pl: Plant, tile: u32, height: f32, c: vec2f, shade_mul: f32, fade: f32) -> VOut {
  let geo = info.geo_rect[tile];
  let uvr = info.uv_rect[tile];
  let s = c.x * 0.5 + 0.5;
  let gu = mix(geo.x, geo.z, s);
  let gv = mix(geo.y, geo.w, c.y);
  let hfrac = clamp((height - info.card_y0) / max(info.card_y1 - info.card_y0, 1e-4), 0.0, 1.0);

  let local = vec3f(
    info.card_cx + (gu * 2.0 - 1.0) * info.card_r,
    0.0,
    info.card_cz + (gv * 2.0 - 1.0) * info.card_r,
  ) * pl.scale;
  var world = pl.base + rot_y(local, pl.yaw);
  world.y = pl.base.y + height * pl.scale;
  world += pl.sway * hfrac;

  var out: VOut;
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = vec2f(mix(uvr.x, uvr.z, s), mix(uvr.y, uvr.w, c.y));
  out.world = world;
  out.yaw_cs = vec2f(cos(pl.yaw), sin(pl.yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.mirror = 0.0;
  out.shade = shade_mul * mix(1.0 - info.bottom_shade, 1.0, hfrac);
  return out;
}

fn quad_corner(vi: u32) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  return corners[vi % 6u];
}

/** Top cards only make sense from above; below that they read as pale lids. */
fn top_fade(pl: Plant) -> f32 {
  return pl.fade * smoothstep(info.top_elev_lo, info.top_elev_hi, pl.elev);
}

// --- near LOD: 3 vertical prisms + 3 horizontal prisms, 36 verts -----------
@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let pl = load_plant(ii);
  let card = vi / 6u;
  let c = quad_corner(vi);

  // Prism separation ramps to zero at the LOD edge, so the three prisms
  // become the merged card exactly where the far bucket takes over and the
  // switch has nothing left to pop.
  let sd = max(info.slab_dist * pl.scale, 1e-3);
  let sep = 1.0 - smoothstep(sd * 0.7, sd, pl.d_cam);
  // Canopy self-occlusion, centred on 1 so the prisms average out to the
  // merged card's brightness: the exposed front slab reads a little hotter
  // than the sky-starved interior, and overall luminance does not change as a
  // plant crosses the LOD boundary.
  let layer = f32(card % N_SLAB) / f32(N_SLAB - 1u);
  let shade = 1.0 + info.canopy_shade * (0.5 - layer) * sep;

  if (card < N_SLAB) {
    let mirrored = pl.az >= N_AZ_BAKED;
    let az_baked = pl.az % N_AZ_BAKED;
    // A mirrored view sees the baked slabs back to front.
    let stored = select(card, N_SLAB - 1u - card, mirrored);
    let tile = az_baked * N_SLAB + stored;
    let stored_depth = info.slab_depth[tile >> 2u][tile & 3u];
    let depth = select(stored_depth, -stored_depth, mirrored) * pl.scale * sep;
    let a = f32(pl.az) * (6.2831853 / f32(N_AZ_VIEW));
    let az_dir = rot_y(vec3f(sin(a), 0.0, cos(a)), pl.yaw);
    return side_card(pl, tile, pl.axis + az_dir * depth, c, shade, pl.fade, select(0.0, 1.0, mirrored));
  }

  let j = card - N_SLAB;
  let h = mix(info.top_height.w, info.top_height[j], sep);
  return top_card(pl, TILE_TOP_SLAB0 + j, h, c, shade, top_fade(pl));
}

// --- far LOD: the merged card + merged top card, 12 verts ------------------
@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let pl = load_plant(ii);
  let c = quad_corner(vi);
  if (vi < 6u) {
    // Merged tiles are stored per VIEW azimuth, already un-mirrored.
    return side_card(pl, TILE_MERGED0 + pl.az, pl.axis, c, 1.0, pl.fade, 0.0);
  }
  return top_card(pl, TILE_TOP_MERGED, info.top_height.w, c, 1.0, top_fade(pl));
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode_tile(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let alb = textureSample(prism_albedo, prism_sampler, in.uv);
  let enc = textureSample(prism_normal, prism_sampler, in.uv).xy;
  if (alb.a < in.erode) {
    discard;
  }
  var n_mesh = oct_decode_tile(enc * 2.0 - 1.0);
  // Mirrored tile: negate the horizontal components. That is the geometric
  // mirror about the tile's U axis composed with re-facing the normal toward
  // the opposite camera, and it deliberately leaves n.y untouched so the
  // hemisphere ambient does not step across the 180-degree seam.
  let ms = 1.0 - 2.0 * in.mirror;
  n_mesh = vec3f(n_mesh.x * ms, n_mesh.y, n_mesh.z * ms);
  // Rotate the mesh-frame normal by the plant's yaw.
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  // in.shade is canopy-depth + grounding occlusion, so it belongs to the
  // light term: the albedo view then shows the baked colour exactly as
  // captured and the lighting view shows sun+ambient x occlusion.
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
