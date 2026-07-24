#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./impostor_info.wgsl"

// One quad per plant, textured from the baked hemi-octahedral view set.
//
// Vertex stage, per plant (the whole selection problem lives here, not in the
// fragment shader — the plant subtends a fraction of a degree, so its view
// direction is effectively constant over its own silhouette):
//   * direction plant -> camera, taken into the plant's yaw-unrotated frame;
//   * hemi-octahedral projection of that direction lands in one grid cell,
//     split into two triangles -> the 3 surrounding view nodes and their
//     barycentric weights;
//   * the quad is the plant's local AABB projected onto the card basis of the
//     ACTUAL direction, so it is tight from every angle (a bounding sphere
//     would waste ~2.5x the area on these plants) and never foreshortened.
//
// Fragment stage: the card position is turned back into a local-frame offset
// and REPROJECTED into each selected view's own orthographic basis. That
// reprojection is what keeps the blend sharp — the three images line up on the
// card plane instead of cross-fading as three misaligned pictures — and it is
// also the parallax that makes off-axis views usable at all.
//
// Coverage is hard alpha-tested with depth write: an impostor that writes
// solid depth is an occluder for everything behind it, which is worth far more
// at grazing angles than a soft dithered edge (see CLAUDE.md taste rules).

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var<storage, read> view_table: array<ViewBasis>;
@group(1) @binding(3) var atlas_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var atlas_normal: texture_2d_array<f32>;
@group(1) @binding(5) var atlas_samp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) card: vec2f,                        // card coords in [-1, 1]
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) weights: vec4f,  // w0, w1, w2, effective alpha ref
  @location(3) @interpolate(flat) sel: vec4u,      // layer0, layer1, layer2, dominant slot
  @location(4) @interpolate(flat) card_u: vec4f,   // card right axis (local), half extent u
  @location(5) @interpolate(flat) card_v: vec4f,   // card up axis (local), half extent v
  @location(6) @interpolate(flat) yaw_cs: vec2f,   // cos(yaw), sin(yaw)
  @location(7) @interpolate(flat) view_dir: vec3f, // world-space plant -> camera
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.packed;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let entry = stand_table[u32(info.setup.y)];
  let base = inst.pos;
  let cs = vec2f(cos(yaw), sin(yaw));

  // The capture was centred on the mesh's own bounds, which sit off its
  // origin — carry that offset (yawed) so imagery lands where geometry was.
  let centre = base + rot_y_cs(info.center.xyz * scale, cs);
  let to_cam = frame.camera_pos - centre;
  let dist = max(length(to_cam), 1.0e-4);
  let dir_w = to_cam / dist;
  let dir_l = rot_y_cs(dir_w, vec2f(cs.x, -cs.y));

  // --- card basis: face the camera exactly, size = projected AABB -----------
  let ru = view_right(dir_l);
  let rv = normalize(cross(dir_l, ru));
  let half_ext = info.half_ext.xyz * scale;
  let ext_u = dot(abs(ru), half_ext) * CARD_PAD;
  let ext_v = dot(abs(rv), half_ext) * CARD_PAD;

  // --- view selection: hemi-octahedral cell + barycentric weights -----------
  // Views below the horizon are not baked; clamp there and let the near fade
  // handle the only case that really produces them (camera inside the plant).
  let dir_up = normalize(vec3f(dir_l.x, max(dir_l.y, 0.0), dir_l.z));
  let g = f32(GRID_N - 1);
  let oct = clamp(hemioct_uv(dir_up), vec2f(0.0), vec2f(1.0)) * g;
  let cell = clamp(floor(oct), vec2f(0.0), vec2f(g - 1.0));
  let f = oct - cell;
  var n0 = cell;
  var n1 = cell + vec2f(1.0, 0.0);
  var n2 = cell + vec2f(0.0, 1.0);
  var w = vec3f(1.0 - f.x - f.y, f.x, f.y);
  if (f.x + f.y >= 1.0) {
    n0 = cell + vec2f(1.0, 1.0);
    w = vec3f(f.x + f.y - 1.0, 1.0 - f.y, 1.0 - f.x);
    // n1 stays (i+1, j) with weight 1 - f.y, n2 stays (i, j+1) with 1 - f.x.
  }
  let stride = f32(GRID_N);
  let layers = vec3u(
    u32(n0.y * stride + n0.x),
    u32(n1.y * stride + n1.x),
    u32(n2.y * stride + n2.x),
  );
  var dom = 0u;
  var wm = w.x;
  if (w.y > wm) { dom = 1u; wm = w.y; }
  if (w.z > wm) { dom = 2u; }

  // --- quad ------------------------------------------------------------------
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = quad[vi];
  let off_l = ru * (c.x * ext_u) + rv * (c.y * ext_v);
  var world = centre + rot_y_cs(off_l, cs);

  // Wind: tips lean, roots hold. Shears the card, which is why the fragment
  // shader reconstructs its local offset from card coords and not from the
  // interpolated world position.
  let foot = base.y + info.half_ext.w * scale;
  let height = max(info.center.w * scale, 1.0e-4);
  let hfrac = clamp((world.y - foot) / height, 0.0, 1.0);
  world += wind_sway(base, frame.time, entry.sway, phase) * hfrac * hfrac;

  // --- fades (eroded through the alpha reference, never dithered) -----------
  let core_r = info.shape.y * scale;
  let seg_y = clamp(frame.camera_pos.y, foot, foot + height);
  let d_core = length(vec3f(frame.camera_pos.x - centre.x, frame.camera_pos.y - seg_y, frame.camera_pos.z - centre.z));
  let near_fade = smoothstep(core_r * 0.45, core_r * 1.25, d_core);
  let edge_fade = 1.0 - smoothstep(info.setup.z * 0.88, info.setup.z, length(to_cam.xz));
  let fade = near_fade * edge_fade;

  // Minification thins an alpha-tested silhouette (the mip average of a spiky
  // coverage mask drops below the cutoff); relax the cutoff with the mip the
  // fragments will land on, estimated once here from the card's screen size.
  let px_per_m = frame.proj[1][1] * frame.viewport.y / (2.0 * dist);
  let texels_per_px = TILE_PX / (2.0 * ext_u * max(px_per_m, 1.0e-4));
  let lod = log2(max(texels_per_px, 1.0));
  let a_ref = info.setup.w * mix(1.0, 0.5, clamp(lod * 0.3, 0.0, 1.0));

  var out: VOut;
  // A card faded below the alpha reference discards EVERY fragment — emit it
  // behind the near plane instead of rasterizing a guaranteed-empty quad.
  if (fade <= a_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.card = c;
  out.world = world;
  out.weights = vec4f(w, a_ref / max(fade, 1.0e-3));
  out.sel = vec4u(layers, dom);
  out.card_u = vec4f(ru, ext_u);
  out.card_v = vec4f(rv, ext_v);
  out.yaw_cs = cs;
  out.view_dir = dir_w;
  return out;
}

/** Local-frame offset of this fragment on the card plane (wind-shear free). */
fn card_offset(in: VOut) -> vec3f {
  return in.card_u.xyz * (in.card.x * in.card_u.w) + in.card_v.xyz * (in.card.y * in.card_v.w);
}

/** Reproject a card-plane offset into one baked view's tile uv. */
fn view_uv(layer: u32, off_l: vec3f) -> vec2f {
  let b = view_table[layer];
  let u = dot(off_l, b.right.xyz) / b.right.w;
  let v = dot(off_l, b.up.xyz) / b.up.w;
  return vec2f(0.5 + 0.5 * u, 0.5 - 0.5 * v);
}

fn view_tint_color(layer: u32) -> vec3f {
  let h = f32(layer) * 0.61803399;
  return vec3f(
    0.35 + 0.5 * fract(h),
    0.35 + 0.5 * fract(h * 2.7 + 0.33),
    0.35 + 0.5 * fract(h * 5.3 + 0.66),
  );
}

/**
 * Shared tail: alpha test on blended coverage, real per-fragment normal from
 * the dominant view (decoded, yaw-rotated, faced toward the viewer), shared
 * lighting, fog only outside debug views.
 */
fn resolve(in: VOut, albedo_in: vec3f, coverage: f32, uv_dom: vec2f, layer_dom: u32) -> vec4f {
  if (coverage < in.weights.w) {
    discard;
  }
  let enc = textureSample(atlas_normal, atlas_samp, uv_dom, layer_dom).xy;
  let n_l = oct_decode_y(enc * 2.0 - 1.0);
  var n = rot_y_cs(n_l, in.yaw_cs);
  // Two-sided foliage: the baked normal faces its own view node, this faces
  // the actual one (and fixes the rare below-horizon clamp).
  if (dot(n, in.view_dir) < 0.0) {
    n = -n;
  }
  var albedo = albedo_in;
  if (info.shape.z > 0.5) {
    albedo = view_tint_color(layer_dom);
  }
  var color = light_surface(albedo, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, coverage, in.world), 1.0);
}

/**
 * Barycentric blend of the three surrounding views. Colour is accumulated
 * coverage-weighted so a blade's colour never gets diluted by the empty half
 * of a neighbouring view; the normal is taken from the dominant view only —
 * one tap instead of three, and normals differ far less between neighbouring
 * views than silhouettes do.
 */
@fragment
fn fs_blend3(in: VOut) -> @location(0) vec4f {
  let off_l = card_offset(in);
  let uv0 = view_uv(in.sel.x, off_l);
  let uv1 = view_uv(in.sel.y, off_l);
  let uv2 = view_uv(in.sel.z, off_l);
  let s0 = textureSample(atlas_albedo, atlas_samp, uv0, in.sel.x);
  let s1 = textureSample(atlas_albedo, atlas_samp, uv1, in.sel.y);
  let s2 = textureSample(atlas_albedo, atlas_samp, uv2, in.sel.z);
  var acc = vec4f(s0.rgb * s0.a, s0.a) * in.weights.x;
  acc += vec4f(s1.rgb * s1.a, s1.a) * in.weights.y;
  acc += vec4f(s2.rgb * s2.a, s2.a) * in.weights.z;

  let uv_dom = select(select(uv0, uv1, in.sel.w == 1u), uv2, in.sel.w == 2u);
  let layer_dom = select(select(in.sel.x, in.sel.y, in.sel.w == 1u), in.sel.z, in.sel.w == 2u);
  return resolve(in, acc.rgb / max(acc.a, 1.0e-4), acc.a, uv_dom, layer_dom);
}

/** Nearest view only — the A/B for what the blend buys (pop vs. ghost). */
@fragment
fn fs_nearest(in: VOut) -> @location(0) vec4f {
  let off_l = card_offset(in);
  let layer_dom = select(select(in.sel.x, in.sel.y, in.sel.w == 1u), in.sel.z, in.sel.w == 2u);
  let uv_dom = view_uv(layer_dom, off_l);
  let s = textureSample(atlas_albedo, atlas_samp, uv_dom, layer_dom);
  return resolve(in, s.rgb, s.a, uv_dom, layer_dom);
}
