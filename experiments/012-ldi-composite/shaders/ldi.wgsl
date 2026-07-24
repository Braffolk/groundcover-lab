#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/wind.wgsl"
#include "./ldi_common.wgsl"

// Runtime LDI compositing, two indirect draws fed by cull.wgsl:
//
// NEAR (vs_main/fs_main): 12 quads per plant — 3 stacks (the two side
// captures nearest the current azimuth + the top capture) x 4 depth-peeled
// layers. Each quad is a card FIXED to its capture basis (not camera-facing)
// at that layer's mean capture depth; fragments write the true per-texel
// baked depth via frag_depth, so the union of stacks composites itself
// through the ordinary depth test — no blending, no sorting, no cross-fade
// weights. A stack viewed off-axis foreshortens toward invisibility, which
// IS the view weighting: side cards vanish from above exactly as the top
// cards turn face-on, and vice versa.
//
// FAR (vs_far/fs_far): 2 quads per plant — front peel layer of the nearest
// side stack + of the top stack, rasterized at their card-plane depth with no
// frag_depth so early-z stays enabled where overdraw is worst.

@group(1) @binding(0) var<uniform> uni: SpeciesU;
@group(1) @binding(1) var<storage, read> instances: array<vec4f>;
@group(1) @binding(2) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(3) var atlas_aux: texture_2d<f32>;
@group(1) @binding(4) var atlas_samp: sampler;

const HALF_PI: f32 = 1.5707963;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
  @location(2) @interpolate(flat) fdat: vec4f,  // yaw, scale, fade, s_plane (m)
  @location(3) @interpolate(flat) idx: vec2u,   // dir index, layer
}

fn degenerate() -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  return out;
}

fn build_card(ii: u32, stack: u32, layer: u32, corner_i: u32) -> VOut {
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let root = i0.xyz;
  let yaw = i0.w;
  let scale = i1.x;
  let phase = i1.y;
  let entry = u32(i1.z);

  let center_w = root + rot_yaw(uni.center.xyz, yaw) * scale;
  let to_cam = frame.camera_pos - center_w;
  let dcam = length(to_cam);

  // Stack -> capture direction: two azimuth-nearest side captures + top.
  var dir_i: u32;
  if (stack == 2u) {
    dir_i = 4u;
  } else {
    let v_loc = rot_yaw(to_cam, -yaw);
    let a01 = atan2(v_loc.z, v_loc.x) / HALF_PI;  // [-2, 2), capture k at azimuth k*90deg
    let k = i32(floor(a01)) + i32(stack);
    dir_i = u32(((k % 4) + 4) % 4);
  }
  let d = uni.dirs[dir_i];
  let mean_d = d.mean_d[layer];
  if (mean_d < 0.0) { return degenerate(); }  // layer empty in the bake

  // Fades: camera-inside-plant and bounded-region edge.
  let bound_r = uni.center.w * scale;
  let near_fade = smoothstep(bound_r * 0.4, bound_r * 0.95, dcam);
  let d_xz = distance(center_w.xz, frame.camera_pos.xz);
  let reg = uni.fade.x;
  let far_fade = 1.0 - smoothstep(reg * 0.85, reg * 0.99, d_xz);
  // Foreshortening cull: beyond close range, a card nearly edge-on to the
  // view contributes almost nothing — top cards at grazing, side cards from
  // straight above (side captures are tilted 20 deg, hence the 0.45 knee).
  let fore = abs(dot(rot_yaw(d.fwd.xyz, yaw), to_cam)) / max(dcam, 1e-4);
  let fore_fade = select(1.0, smoothstep(0.25, 0.45, fore), dcam > 12.0);
  let fade = near_fade * far_fade * fore_fade;
  if (fade < 0.002) { return degenerate(); }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[corner_i];

  // Card plane at this layer's mean capture depth, in the capture basis.
  let s_plane = d.fwd.w - mean_d;
  let local = uni.center.xyz + d.fwd.xyz * s_plane
            + d.right.xyz * (c.x * d.right.w)
            + d.up.xyz * (c.y * d.up.w);
  var world = root + rot_yaw(local, yaw) * scale;

  // Wind: roots pinned, tips sway (shared model, per-species response).
  let se = stand_table[entry];
  let h01 = clamp((world.y - root.y) / max(se.height_scale * scale, 1e-3), 0.0, 1.0);
  world += wind_sway(root, frame.time, se.sway, phase) * h01;

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  // NDC +y is texture row 0, so v flips.
  out.uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  out.fdat = vec4f(yaw, scale, fade, s_plane);
  out.idx = vec2u(dir_i, layer);
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let quad = vi / 6u;
  return build_card(ii, quad / LDI_LAYERS, quad % LDI_LAYERS, vi % 6u);
}

// Far plants: quad 0 = nearest side stack, quad 1 = top stack, front layer only.
@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let quad = vi / 6u;
  return build_card(ii, quad * 2u, 0u, vi % 6u);
}

fn oct_decode_yup(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

fn atlas_uv_for(dir_i: u32, layer: u32, uv: vec2f) -> vec2f {
  return (vec2f(f32(dir_i), f32(layer)) + uv) * uni.atlas.x / uni.atlas.yz;
}

fn aux_depth(aux: vec4f) -> f32 {
  return (aux.b * 255.0 * 256.0 + aux.a * 255.0) / 65535.0;
}

fn shade(alb: vec3f, aux: vec4f, yaw: f32, world_texel: vec3f) -> vec3f {
  var n = rot_yaw(oct_decode_yup(aux.rg * 2.0 - 1.0), yaw);
  let vdir = normalize(frame.camera_pos - world_texel);
  if (dot(n, vdir) < 0.0) { n = -n; }  // thin foliage is two-sided
  var color = light_surface(alb, n, world_texel);
  return apply_fog(color, world_texel);
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let yaw = in.fdat.x;
  let scale = in.fdat.y;
  let fade = in.fdat.z;
  let s_plane = in.fdat.w;
  let dir_i = in.idx.x;
  let layer = in.idx.y;

  let d = uni.dirs[dir_i];
  let inset = 0.5 / uni.atlas.x;
  var uv = clamp(in.uv, vec2f(inset), vec2f(1.0 - inset));

  var aux = textureSampleLevel(atlas_aux, atlas_samp, atlas_uv_for(dir_i, layer, uv), 0.0);

  // One-tap parallax: reproject the sample by its own baked depth offset so
  // texels off the card plane land closer to their true screen position.
  // A single precomputed-depth lookup — no marching. Sub-pixel at range, so
  // skip it there and save the extra taps.
  if (uni.fade.z > 0.5 && distance(frame.camera_pos, in.world) < 30.0) {
    let s_texel = d.fwd.w - aux_depth(aux) * 2.0 * d.fwd.w;
    let ds = s_texel - s_plane;  // metres in front of the card plane (unit scale)
    let v_loc = rot_yaw(normalize(frame.camera_pos - in.world), -yaw);
    let vf = max(abs(dot(v_loc, d.fwd.xyz)), 0.2);
    let vr = dot(v_loc, d.right.xyz);
    let vu = dot(v_loc, d.up.xyz);
    var duv = vec2f(-(vr / vf) * ds / (2.0 * d.right.w), (vu / vf) * ds / (2.0 * d.up.w));
    duv = clamp(duv, vec2f(-0.06), vec2f(0.06)); // never smear further than ~15 texels
    uv = clamp(uv + duv, vec2f(inset), vec2f(1.0 - inset));
    aux = textureSampleLevel(atlas_aux, atlas_samp, atlas_uv_for(dir_i, layer, uv), 0.0);
  }

  let alb = textureSampleLevel(atlas_albedo, atlas_samp, atlas_uv_for(dir_i, layer, uv), 0.0);
  if (alb.a * fade < uni.misc.x) { discard; }

  // True per-texel depth: push the fragment from the card plane to the baked
  // capture depth along the capture axis, then reproject.
  let s_texel = d.fwd.w - aux_depth(aux) * 2.0 * d.fwd.w;
  let fwd_w = rot_yaw(d.fwd.xyz, yaw);
  let world_texel = in.world + fwd_w * ((s_texel - s_plane) * scale);
  let clip = frame.view_proj * vec4f(world_texel, 1.0);

  var out: FOut;
  out.color = vec4f(shade(alb.rgb, aux, yaw, world_texel), 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1e-4), 0.0, 1.0);
  return out;
}

// Far fragments: card-plane depth from the rasterizer (early-z stays on),
// no parallax, straight shade.
@fragment
fn fs_far(in: VOut) -> @location(0) vec4f {
  let inset = 0.5 / uni.atlas.x;
  let uv0 = atlas_uv_for(in.idx.x, in.idx.y, clamp(in.uv, vec2f(inset), vec2f(1.0 - inset)));
  let alb = textureSampleLevel(atlas_albedo, atlas_samp, uv0, 0.0);
  if (alb.a * in.fdat.z < uni.misc.x) { discard; }
  let aux = textureSampleLevel(atlas_aux, atlas_samp, uv0, 0.0);
  return vec4f(shade(alb.rgb, aux, in.fdat.x, in.world), 1.0);
}
