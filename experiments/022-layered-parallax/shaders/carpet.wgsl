#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// LAYERED PARALLAX, CARPET FORM.
//
// A carpet species (stand_table[i].carpet_div > 0) is a periodic community
// tile, not a plant: 0.18m across, 0.07-0.09m tall, laid out grid-snapped at a
// constant scale with 90-degree-only rotations. The upright card pair in
// cards.wgsl is the wrong shape for it twice over — a vertical card slices
// through the ground and shows nothing edge-on, and a tile-sized quad is not
// what `card_r` (the plant's support radius) describes.
//
// So a carpet draws ONE ground-parallel quad of exactly `footprint_m * scale`,
// terrain-conformed PER VERTEX (ladder rung 3: neighbouring tiles share corner
// positions, so this is the only rung that keeps the mat C0-continuous — any
// per-tile plane fit cracks at every tile boundary), and the parallax that is
// the whole point of this experiment moves from depth slabs to HEIGHT BANDS:
//
//   the tile is baked into N_BAND top-down images, each holding the geometry
//   in one quartile of the cushion's height distribution and recording the
//   MEAN HEIGHT of that geometry. The quad is rasterized on the topmost band's
//   plane; the fragment shader walks the real eye ray DOWN through the other
//   planes and takes the first opaque hit. Capitula shift against the crevices
//   between them exactly as 3.3cm of real cushion relief would, and the deeper
//   a hit lies the less light it gets.
//
// The bands are one array texture with `repeat` addressing, which is legal
// precisely because the bake is wrapped: the image is exactly one period, so
// the ray may walk out of the tile and simply continue into the next copy —
// no clamping, no in-tile test, no seam at any mip level.
//
// Past the distance where a tile stops resolving, the band probes would only
// alias, so the effective alpha reference is raised with the mip level until
// every band fails and the MERGED tile (layer N_BAND, all heights) takes over.
// That is a per-texel dissolve ordered by coverage rather than a hard ring,
// and at the far end the shader is structurally the billboard baseline: one
// anisotropic tap, one normal tap.

const TWO_PI: f32 = 6.2831853;
const N_BAND: i32 = 4;

struct PlantInst {
  pos: vec3f,
  packed_bits: u32, // yaw 10b | scale 12b | phase 10b (carpet: yaw only)
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var band_albedo: texture_2d_array<f32>;
@group(1) @binding(3) var band_normal: texture_2d_array<f32>;
@group(1) @binding(4) var tile_sampler: sampler; // address mode: repeat

struct CarpetOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
  @location(2) up: vec3f,
  @location(3) @interpolate(flat) yaw_cs: vec2f,
  // x: 1 / (footprint * scale), y: alpha-test reference, z: canopy extinction
  // per metre of depth, w: coverage a band needs to claim a fragment
  @location(4) @interpolate(flat) tile: vec4f,
  // metres the eye ray must descend from the proxy plane to reach bands 1..3
  // and the merged tile. Band 0 IS the proxy plane, so its drop is 0.
  @location(5) @interpolate(flat) drops: vec4f,
}

fn quad_corner(vi: u32) -> vec2f {
  var c = array<vec2f, 6>(
    vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(-0.5, 0.5),
    vec2f(0.5, -0.5), vec2f(0.5, 0.5), vec2f(-0.5, 0.5),
  );
  return c[vi % 6u];
}

/// Rotate a tile-local xz offset into world by the tile's yaw.
fn to_world(l: vec2f, cs: vec2f) -> vec2f {
  return vec2f(cs.x * l.x + cs.y * l.y, -cs.y * l.x + cs.x * l.y);
}

/// Inverse of to_world: a world xz offset expressed in the tile's own frame.
fn to_local(w: vec2f, cs: vec2f) -> vec2f {
  return vec2f(cs.x * w.x - cs.y * w.y, cs.y * w.x + cs.x * w.y);
}

@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> CarpetOut {
  let inst = insts[ii];
  let entry = stand_table[u32(info.entry_index)];
  // A carpet's scale is CONSTANT by construction, so take it from the stand
  // rather than the 12-bit packed field: the tile must fill its grid step to
  // the last micrometre or the mat shows a seam.
  let scale = entry.scale_min;
  let step = entry.footprint_m * scale;
  let yaw = f32(inst.packed_bits & 1023u) * (TWO_PI / 1024.0);
  let cs = vec2f(cos(yaw), sin(yaw));

  let c = quad_corner(vi);
  let world_xz = inst.pos.xz + to_world(c * step, cs);
  // Rung 3, per vertex: height AND (nx, nz) out of one bilinear fetch. Corners
  // are shared with the neighbouring tiles, so the mat stays continuous.
  let g = terrain_sample(world_xz);
  let world = vec3f(world_xz.x, g.x + info.carpet_h0 * scale, world_xz.y);

  // No camera-inside fade: a mat you are standing on must not open a hole
  // under you. Only the region rim erodes, measured from the tile CENTRE so a
  // tile never gets some vertices clipped and others not.
  let d_xz = distance(inst.pos.xz, frame.camera_pos.xz);
  let fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);

  var out: CarpetOut;
  if (fade < info.carpet_alpha) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.world = world;
  out.uv = c + vec2f(0.5);
  out.up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  out.yaw_cs = cs;
  let drops = info.carpet_drops * scale;
  // Canopy extinction: light into a cushion falls off with depth, so the
  // shading is exp(-k * depth) with k fixed by "the bottom band keeps
  // (1 - carpet_depth_shade) of the light". Continuous in depth rather than
  // stepped per band, which is what lets the merged tile — whose plane is the
  // mean height of everything — dissolve in with no brightness step.
  let ao_k = -log(max(1.0 - info.carpet_depth_shade, 0.02)) / max(drops.z, 1.0e-4);
  out.tile = vec4f(1.0 / max(step, 1e-5), info.carpet_alpha / max(fade, 1e-3), ao_k, info.carpet_band_ref);
  out.drops = drops;
  return out;
}

@fragment
fn fs_carpet(in: CarpetOut) -> @location(0) vec4f {
  // Gradients first, in uniform control flow. Everything below samples with
  // explicit gradients or an explicit level, so the band probes may branch.
  let gx = dpdx(in.uv);
  let gy = dpdy(in.uv);
  let texels = info.carpet_tex;
  let ax = gx * texels;
  let ay = gy * texels;
  let lod = max(0.5 * log2(max(dot(ax, ax), dot(ay, ay)) + 1e-12), 0.0);

  // Eye ray, in metres of horizontal travel per metre of descent, expressed in
  // the tile's own frame and in uv units. Clamped: at a few degrees above the
  // mat a 5cm drop would land two tiles away and the bands would stop being
  // correlated with each other, which reads as noise rather than as relief.
  let dir = normalize(in.world - frame.camera_pos);
  var slope = dir.xz / max(-dir.y, 1.0e-3);
  let sl = length(slope);
  slope *= min(sl, info.carpet_max_slope) / max(sl, 1.0e-6);
  let duv = to_local(slope, in.yaw_cs) * in.tile.x;

  // A band CLAIMS a fragment only where it genuinely has geometry, which is a
  // much higher bar than "is there moss here at all" — that second question is
  // the merged tile's, and it is the one that must stay closed. Running the
  // band probes at the mat's own alpha reference (0.06) let the top band win
  // on 6%-covered fringe texels, which flattened the relief to nothing.
  //
  // On top of that, dissolve the bands into the merged tile as the tile stops
  // resolving: past that point a band probe is only aliasing.
  //
  // max() with the eroded reference, not just the band's own: at the region
  // rim the fade raises the reference past 1, and if the bands ignored that
  // they would keep claiming fragments the mat is supposed to have dissolved.
  let merge = smoothstep(info.carpet_merge_lod, info.carpet_merge_lod + info.carpet_merge_span, lod);
  let band_ref = mix(max(in.tile.w, in.tile.y), 2.0, merge);

  var hit = vec4f(0.0);
  var hit_uv = in.uv;
  var hit_layer = -1;
  var hit_drop = 0.0;

  if (merge < 0.999) {
    // Band 0 lives ON the proxy plane: no offset, and it owns most fragments,
    // so it is the one that gets the anisotropic tap.
    let s0 = textureSampleGrad(band_albedo, tile_sampler, in.uv, 0, gx, gy);
    if (s0.a >= band_ref) {
      hit = s0;
      hit_layer = 0;
    }
    if (hit_layer < 0) {
      let uv1 = in.uv + duv * in.drops.x;
      let s1 = textureSampleLevel(band_albedo, tile_sampler, uv1, 1, lod);
      if (s1.a >= band_ref) {
        hit = s1;
        hit_uv = uv1;
        hit_layer = 1;
        hit_drop = in.drops.x;
      }
    }
    if (hit_layer < 0) {
      let uv2 = in.uv + duv * in.drops.y;
      let s2 = textureSampleLevel(band_albedo, tile_sampler, uv2, 2, lod);
      if (s2.a >= band_ref) {
        hit = s2;
        hit_uv = uv2;
        hit_layer = 2;
        hit_drop = in.drops.y;
      }
    }
    if (hit_layer < 0) {
      let uv3 = in.uv + duv * in.drops.z;
      let s3 = textureSampleLevel(band_albedo, tile_sampler, uv3, 3, lod);
      if (s3.a >= band_ref) {
        hit = s3;
        hit_uv = uv3;
        hit_layer = 3;
        hit_drop = in.drops.z;
      }
    }
  }
  if (hit_layer < 0) {
    let uvm = in.uv + duv * in.drops.w;
    let sm = textureSampleGrad(band_albedo, tile_sampler, uvm, N_BAND, gx, gy);
    if (sm.a < in.tile.y) {
      discard; // a genuine gap down to the peat
    }
    hit = sm;
    hit_uv = uvm;
    hit_layer = N_BAND;
    hit_drop = in.drops.w;
  }

  // Normals live at half the albedo resolution. Sampled with the GRADIENTS,
  // not an explicit level, for two reasons: the anisotropic filter then holds
  // capitulum-scale shading at grazing where an isotropic level blurs it to a
  // flat mat, and the level is derived from this texture's own size, so the
  // half resolution needs no hand-applied -1.
  let nt = textureSampleGrad(band_normal, tile_sampler, hit_uv, hit_layer, gx, gy).xyz * 2.0 - 1.0;
  // Lift the mesh-frame normal into the GROUND frame: a mat on a slope must
  // light as a slope. plant_basis_from_up inlined — the tile already carries
  // cos/sin(yaw) and the helper wants the angle.
  let up = normalize(in.up);
  var tang = vec3f(in.yaw_cs.x, 0.0, -in.yaw_cs.y);
  tang = normalize(tang - up * dot(up, tang));
  let bitan = cross(tang, up);
  let n = normalize(tang * nt.x + up * nt.y + bitan * nt.z);

  // Canopy self-occlusion by DEPTH below the cushion top, not by band index:
  // continuous, and it gives the merged tile (whose plane is the mean height
  // of everything) exactly the average darkening the band probes produce, so
  // the dissolve into it has no brightness step.
  let shade = exp(-in.tile.z * hit_drop);

  var color = light_surface(hit.rgb * shade, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, hit.rgb, n, hit.a, in.world), 1.0);
}
