#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/fullscreen.wgsl"
#include "src/wgsl/debug.wgsl"

// PER-PIXEL STAMP RESOLVE. One fullscreen pass; each pixel walks its screen
// tile's front-to-back plant list (built by bin.wgsl), intersects the pixel
// ray with each plant's impostor card, samples the plant's nearest baked
// hemi-octa view (mipmapped), and composites over-operator front-to-back with
// early termination — overdraw is bounded by transmittance, not plant count.
// Terrain occlusion is a manual compare against the scene depth buffer (no
// depth attachment, so soft alpha never punches holes in early-z). Where the
// list ends (or the tile skipped individual plants entirely) a statistical
// meadow tint — species average albedos mixed by world-anchored value noise,
// lit by the terrain normal — takes over.
//
// Debug views (frame.debug_mode) describe what THIS pass contributed: albedo,
// world normal and the shaded colour are accumulated with the same
// front-to-back weights as the colour, so albedo/normals/lighting show the
// coverage-weighted average of every stamp (and the tint field) that landed on
// the pixel, and coverage is the accumulated alpha.

const TILE_W: u32 = 16u;
const TILE_H: u32 = 8u;
const MAX_LIST: u32 = 64u;
const HEADER_U32: u32 = 8u;
const ENTRY_U32: u32 = 8u;
const STRIDE_U32: u32 = HEADER_U32 + MAX_LIST * ENTRY_U32;
const GRID_F: f32 = 10.0;
const ATLAS_TILE_F: f32 = 128.0;
const ATLAS_F: f32 = 1280.0;
const MAX_LOD: f32 = 3.0;
const TWO_PI_S: f32 = 6.2831853;

struct StampParams {
  dims: vec4u,                 // x,y unused, z = seed, w = entry_count
  tuning: vec4f,               // x = max_dist, y = tint_strength, z = tile overlay, w unused
  entry_meta: array<vec4f, 4>,
  entry_info: array<vec4f, 4>, // species average albedo rgb, pad
}
@group(1) @binding(0) var<uniform> sp: StampParams;
@group(1) @binding(1) var<storage, read> tile_lists: array<u32>;
@group(1) @binding(2) var scene_depth: texture_depth_2d;
@group(1) @binding(3) var alb0: texture_2d<f32>;
@group(1) @binding(4) var alb1: texture_2d<f32>;
@group(1) @binding(5) var alb2: texture_2d<f32>;
@group(1) @binding(6) var nrm0: texture_2d<f32>;
@group(1) @binding(7) var nrm1: texture_2d<f32>;
@group(1) @binding(8) var nrm2: texture_2d<f32>;
@group(1) @binding(9) var atlas_samp: sampler;

fn rot_y_v(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn hemioct_decode(e: vec2f) -> vec3f {
  let px = (e.x + e.y) * 0.5;
  let pz = (e.x - e.y) * 0.5;
  let y = 1.0 - abs(px) - abs(pz);
  return normalize(vec3f(px, y, pz));
}

fn sample_albedo(entry_index: u32, uv: vec2f, lod: f32) -> vec4f {
  if (entry_index == 0u) { return textureSampleLevel(alb0, atlas_samp, uv, lod); }
  if (entry_index == 1u) { return textureSampleLevel(alb1, atlas_samp, uv, lod); }
  return textureSampleLevel(alb2, atlas_samp, uv, lod);
}

fn sample_normal(entry_index: u32, uv: vec2f, lod: f32) -> vec4f {
  if (entry_index == 0u) { return textureSampleLevel(nrm0, atlas_samp, uv, lod); }
  if (entry_index == 1u) { return textureSampleLevel(nrm1, atlas_samp, uv, lod); }
  return textureSampleLevel(nrm2, atlas_samp, uv, lod);
}

fn value_noise(p: vec2f) -> f32 {
  let ip = vec2i(floor(p));
  let fp = fract(p);
  let s = fp * fp * (3.0 - 2.0 * fp);
  let a = hash_f32(hash2(bitcast<u32>(ip.x), bitcast<u32>(ip.y)));
  let b = hash_f32(hash2(bitcast<u32>(ip.x + 1), bitcast<u32>(ip.y)));
  let c = hash_f32(hash2(bitcast<u32>(ip.x), bitcast<u32>(ip.y + 1)));
  let d = hash_f32(hash2(bitcast<u32>(ip.x + 1), bitcast<u32>(ip.y + 1)));
  return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4f {
  let dbg = debug_mode();
  let px = vec2u(in.pos.xy);
  let tiles_x = (u32(frame.viewport.x) + TILE_W - 1u) / TILE_W;
  let tiles_y = (u32(frame.viewport.y) + TILE_H - 1u) / TILE_H;
  let tile = px / vec2u(TILE_W, TILE_H);
  let tbase = (tile.y * tiles_x + tile.x) * STRIDE_U32;
  let count = min(tile_lists[tbase], MAX_LIST);

  // Tint horizon, bilinearly blended across the 2x2 nearest tiles so the
  // per-tile overflow horizon never shows as blocks.
  let fpos = in.pos.xy / vec2f(f32(TILE_W), f32(TILE_H)) - 0.5;
  let f0 = floor(fpos);
  let fw = fpos - f0;
  let ta = vec2i(f0);
  let tmax = vec2i(i32(tiles_x) - 1, i32(tiles_y) - 1);
  let t00 = clamp(ta, vec2i(0), tmax);
  let t11 = clamp(ta + vec2i(1), vec2i(0), tmax);
  let l00 = bitcast<f32>(tile_lists[(u32(t00.y) * tiles_x + u32(t00.x)) * STRIDE_U32 + 1u]);
  let l10 = bitcast<f32>(tile_lists[(u32(t00.y) * tiles_x + u32(t11.x)) * STRIDE_U32 + 1u]);
  let l01 = bitcast<f32>(tile_lists[(u32(t11.y) * tiles_x + u32(t00.x)) * STRIDE_U32 + 1u]);
  let l11 = bitcast<f32>(tile_lists[(u32(t11.y) * tiles_x + u32(t11.x)) * STRIDE_U32 + 1u]);
  let bin_limit = mix(mix(l00, l10, fw.x), mix(l01, l11, fw.x), fw.y);

  let cam = frame.camera_pos;
  let ndc = vec2f(
    (f32(px.x) + 0.5) / frame.viewport.x * 2.0 - 1.0,
    1.0 - (f32(px.y) + 0.5) / frame.viewport.y * 2.0,
  );
  let wp_far = frame.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  let ray = normalize(wp_far.xyz / wp_far.w - cam);

  let d = textureLoad(scene_depth, vec2i(px), 0);
  var scene_dist = 1e9;
  var scene_pos = cam + ray * 200.0;
  if (d < 0.99999) {
    let wp = frame.inv_view_proj * vec4f(ndc, d, 1.0);
    scene_pos = wp.xyz / wp.w;
    scene_dist = distance(scene_pos, cam);
  }
  // world size of one pixel at unit distance
  let pixel_scale = 2.0 / (frame.viewport.y * frame.proj[1][1]);

  var acc = vec3f(0.0);
  var trans = 1.0;
  // Debug accumulators — same front-to-back weights as `acc`, so the debug
  // views report exactly what this pass composited (see header).
  var alb_acc = vec3f(0.0);
  var nrm_acc = vec3f(0.0);
  var dbg_world = scene_pos;
  var have_world = false;

  for (var i = 0u; i < count; i++) {
    let ebase = tbase + HEADER_U32 + i * ENTRY_U32;
    let center = vec3f(
      bitcast<f32>(tile_lists[ebase]),
      bitcast<f32>(tile_lists[ebase + 1u]),
      bitcast<f32>(tile_lists[ebase + 2u]),
    );
    let r_world = bitcast<f32>(tile_lists[ebase + 3u]);
    let packed = tile_lists[ebase + 4u];

    // Card plane: through center, facing the camera (spherical billboard).
    let to_cam = cam - center;
    let dc_len = length(to_cam);
    // The list is sorted front-to-back by center distance: once an entry is
    // fully behind the visible surface, so is everything after it.
    if (dc_len - r_world * 1.05 - 0.6 > scene_dist) { break; }
    let n_plane = to_cam / max(dc_len, 1e-4);
    let denom = dot(ray, n_plane);
    if (denom > -1e-4) { continue; }
    let t = -dc_len / denom;
    if (t < 0.02 || t > scene_dist + 0.6) { continue; }
    let hit = cam + ray * t;

    // View-angle card clip — must mirror bin.wgsl's k_clip exactly.
    let k_clip = mix(1.0, 0.45, smoothstep(0.55, 0.9, (cam.y - center.y) / max(dc_len, 1e-4)));

    // Coarse circle reject before touching the rest of the record (sway
    // shear moves the sample by <0.4 m, so pad the radius and bail early).
    let coarse = hit - center;
    let pad = r_world * k_clip * 1.02 + 0.4;
    if (dot(coarse, coarse) > pad * pad) { continue; }

    // Wind shear: bin displaced the card by 0.6*sway; redistribute so roots
    // stay put and tips lead (sample position shifts opposite the residual).
    let swayv2 = unpack2x16float(tile_lists[ebase + 5u]);
    let bh = unpack2x16float(tile_lists[ebase + 7u]); // base y, plant height
    let hf = clamp((hit.y - bh.x) / max(bh.y, 0.1), 0.0, 1.0);
    let shear = vec3f(swayv2.x, 0.0, swayv2.y) * (hf - 0.6);

    let yaw = f32((packed >> 13u) & 255u) * (TWO_PI_S / 255.0);
    let local = rot_y_v(hit - shear - center, -yaw) / max(r_world, 1e-4);
    if (dot(local, local) > k_clip * k_clip * 1.02) { continue; }

    // Project into the baked view's orthographic basis (parallax mapping).
    let ni = f32((packed >> 3u) & 31u);
    let nj = f32((packed >> 8u) & 31u);
    let e2 = vec2f(ni, nj) / (GRID_F - 1.0) * 2.0 - 1.0;
    let node_dir = hemioct_decode(e2);
    var up_ref = vec3f(0.0, 1.0, 0.0);
    if (abs(node_dir.y) > 0.999) { up_ref = vec3f(0.0, 0.0, 1.0); }
    let bx = normalize(cross(up_ref, node_dir));
    let by = normalize(cross(node_dir, bx));
    var uvp = vec2f(dot(local, bx), dot(local, by)) * 0.5 + 0.5;
    // Texture row 0 is the capture's +up: flip v or plants render upside down.
    uvp.y = 1.0 - uvp.y;

    let lod = clamp(log2(max(t * pixel_scale * ATLAS_TILE_F / (2.0 * r_world), 1.0)), 0.0, MAX_LOD);
    let inset = (0.5 * exp2(lod) + 0.5) / ATLAS_TILE_F;
    // REJECT outside the safe interior instead of clamping: community-tile
    // meshes are cropped at their bbox, so atlas tile borders carry coverage
    // and clamping smears them into hollow-rectangle card outlines.
    if (uvp.x < inset || uvp.x > 1.0 - inset || uvp.y < inset || uvp.y > 1.0 - inset) { continue; }
    let auv = (vec2f(ni, nj) + uvp) * (ATLAS_TILE_F / ATLAS_F);

    let entry_index = packed & 7u;
    var fade = f32((packed >> 21u) & 255u) / 255.0;
    // Overflow tiles stop their lists short of max_dist; fade stamps out
    // toward the (bilinearly blended) horizon so the handoff to the tint
    // field is a crossfade, not a seam.
    if (bin_limit < sp.tuning.x * 0.99) {
      fade *= 1.0 - smoothstep(bin_limit * 0.7, bin_limit * 0.98, dc_len);
    }
    let alb = sample_albedo(entry_index, auv, lod);
    let a = alb.a * fade;
    if (a < 0.02) { continue; }
    // Atlas albedo is coverage-premultiplied (empty texels are black).
    let albedo = alb.rgb / max(alb.a, 0.05);

    var n_local = sample_normal(entry_index, auv, lod).xyz * 2.0 - 1.0;
    if (dot(n_local, n_local) < 1e-4) { n_local = vec3f(0.0, 1.0, 0.0); }
    let n_ws = rot_y_v(normalize(n_local), yaw);
    var col = light_surface(albedo, n_ws, hit);
    // Fog only in the normal view — debug views stay unfogged and honest.
    if (dbg == DEBUG_OFF) {
      col = apply_fog(col, hit);
    }

    let w = trans * a;
    acc += w * col;
    alb_acc += w * albedo;
    nrm_acc += w * n_ws;
    if (!have_world) {
      dbg_world = hit;
      have_world = true;
    }
    trans *= 1.0 - a;
    if (trans < 0.04) { break; }
  }

  // Aggregate meadow field beyond the individually-stamped range.
  if (trans > 0.02 && d < 0.99999) {
    // bin_limit is the per-tile stamp horizon: the global max_dist normally,
    // smaller if the tile's list overflowed, 0 for tint-only tiles. Match the
    // stamp far-fade window (0.72-1.0 x max_dist) when at the global horizon.
    var tstart = max(bin_limit * 0.7, 4.0);
    var tend = max(bin_limit * 0.98, tstart + 6.0);
    if (bin_limit >= sp.tuning.x * 0.999) {
      tstart = sp.tuning.x * 0.72;
      tend = sp.tuning.x * 0.95;
    }
    let tf = smoothstep(tstart, tend, scene_dist) * sp.tuning.y;
    if (tf > 0.002) {
      let wxz = scene_pos.xz;
      let ec = sp.dims.w;
      let n1 = value_noise(wxz * 0.9);
      let n2 = value_noise(wxz * 0.17);
      var tint = sp.entry_info[0].rgb;
      tint = mix(tint, sp.entry_info[min(1u, ec - 1u)].rgb, n1 * 0.7);
      tint = mix(tint, sp.entry_info[min(2u, ec - 1u)].rgb, n2 * 0.5);
      // Two extra octaves: per-plant-scale speckle + dark canopy gaps, so
      // the aggregate field carries the same busyness as the stamped zone.
      tint *= 0.72 + 0.56 * value_noise(wxz * 2.3);
      tint *= 0.78 + 0.34 * value_noise(wxz * 5.1);
      let tn = normalize(terrain_normal(wxz) + vec3f(0.0, 0.6, 0.0));
      var tc = light_surface(tint, tn, scene_pos);
      // Fog only in the normal view — debug views stay unfogged and honest.
      if (dbg == DEBUG_OFF) {
        tc = apply_fog(tc, scene_pos);
      }
      let wt = trans * tf;
      acc += wt * tc;
      alb_acc += wt * tint;
      nrm_acc += wt * tn;
      if (!have_world) {
        dbg_world = scene_pos;
        have_world = true;
      }
      trans *= 1.0 - tf;
    }
  }

  if (dbg != DEBUG_OFF) {
    // Un-premultiply the accumulators to get this pass's coverage-weighted
    // average surface, then let the shared debug_shade() do the rest.
    let cov = 1.0 - trans;
    let inv = 1.0 / max(cov, 1e-4);
    var n_avg = vec3f(0.0, 1.0, 0.0);
    if (dot(nrm_acc, nrm_acc) > 1e-8) { n_avg = normalize(nrm_acc); }
    let dcol = debug_shade(acc * inv, alb_acc * inv, n_avg, cov, dbg_world);
    if (dbg == DEBUG_COVERAGE) {
      // Coverage is drawn opaque — the terrain writes 1.0 everywhere, so a
      // blended coverage map would wash out to uniform white. Exception: the
      // near-field card pass already wrote opaque groundcover into this pixel
      // (its surface sits well above the terrain), and overwriting it with the
      // stamp pass's ~0 coverage would hide real coverage.
      if (d < 0.99999 && scene_pos.y - terrain_height(scene_pos.xz) > 0.15) { discard; }
      return vec4f(dcol, 1.0);
    }
    if (cov < 0.003) { discard; }
    return vec4f(dcol * cov, cov);
  }

  // Method-specific inspection overlay (own param — see manifest tileView).
  let overlay = u32(sp.tuning.z + 0.5);
  if (overlay == 1u) {
    let heat = f32(count) / f32(MAX_LIST);
    acc = mix(acc, vec3f(heat, 1.0 - heat, 0.15), 0.4);
    trans = min(trans, 0.5);
  } else if (overlay == 2u) {
    // Which binning strategy the tile picked: blue = tint-only,
    // green = enumerate, orange = column march.
    let tile_mode = tile_lists[tbase + 2u];
    var mc = vec3f(0.10, 0.16, 0.65);
    if (tile_mode == 1u) { mc = vec3f(0.12, 0.62, 0.18); }
    if (tile_mode == 2u) { mc = vec3f(0.75, 0.45, 0.08); }
    acc = mix(acc, mc, 0.45);
    trans = min(trans, 0.5);
  }

  let alpha = 1.0 - trans;
  if (alpha < 0.003) { discard; }
  return vec4f(acc, alpha);
}
