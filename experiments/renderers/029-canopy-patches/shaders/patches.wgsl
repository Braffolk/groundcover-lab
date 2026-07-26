#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./patch_info.wgsl"

// Canopy patches. Every plant is drawn as a small STACK of quads instead of one
// card, and every quad is a baked canopy patch (see bake.ts):
//
//   quad 0..1  depth slabs of the nearest baked azimuth, each placed at the
//              mean depth of the foliage it contains, PERPENDICULAR TO THE BAKE
//              AZIMUTH (not to the camera). That is what buys parallax: the
//              planes are fixed in the plant's frame, so as the camera moves
//              inside the +/-22.5 deg bin the front slab slides across the back
//              slab exactly as real geometry would, the union silhouette
//              changes shape, and the front slab depth-occludes the one behind
//              it through the ordinary depth test (no frag_depth anywhere).
//   quad 2..3  horizontal crown patches (upper/lower height band, top-down
//              imagery) at the mean height of their band — real parallax from
//              above, and they are what fills a top-down view.
//   quad 4..5  the distance LOD: one camera-facing composite card + one
//              composite crown card, from a much smaller atlas. Exactly a
//              billboard, exactly a billboard's cost.
//
// The collapse from stack to card is GEOMETRIC, not a crossfade: approaching
// patch_dist, `blend` slides the slab depths to zero and rotates the planes to
// face the camera, so at the switch distance the slabs are coincident
// camera-facing quads whose union is the composite image. No dither, no
// blending: hard alpha test with depth write everywhere.

struct PlantInst {
  xz: u32,
  y: f32,
  bits: u32,
}

@group(1) @binding(0) var<uniform> info: PatchInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var patch_albedo: texture_2d_array<f32>;
@group(1) @binding(3) var patch_normal: texture_2d_array<f32>;
@group(1) @binding(4) var patch_sampler: sampler;

const TAU: f32 = 6.2831853;
const BINS_F: f32 = 8.0;      // keep in sync with bake.ts BINS
const BINS_I: i32 = 8;
const SLABS: u32 = 2u;        // ... and SLABS
const CROWN_BASE: u32 = 16u;  // BINS * SLABS
const NEAR_SLICES: u32 = 18u; // BINS * SLABS + CROWNS
const BINS_U: u32 = 8u;
// Quad index (from firstVertex / 6): 0..1 depth slabs, 2..3 crown patches,
// 4 composite card, 5 composite crown.
const Q_CROWN0: u32 = 2u;
const Q_FAR: u32 = 4u;
const Q_FAR_CROWN: u32 = 5u;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) layer: i32,       // texture array layer
  @location(3) @interpolate(flat) rot_cs: vec2f,    // cos/sin of content rotation
  @location(4) @interpolate(flat) erode: f32,       // effective alpha reference
  @location(5) shade: f32,                          // canopy-depth + grounding
  @location(6) @interpolate(flat) quad: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  // Quad corners as (s, t) in [0,1]^2; t = 0 is the TOP of the capture rect,
  // which is where the bake put texture row 0.
  var corners = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(1.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0),
  );
  let q = vi / 6u;
  let c = corners[vi % 6u];
  let is_far = q >= Q_FAR;
  let is_crown = (q == 2u) || (q == 3u) || (q == Q_FAR_CROWN);

  let inst = insts[ii + select(0u, u32(info.far_base), is_far)];
  let xz = unpack2x16snorm(inst.xz) * info.pos_range + vec2f(info.pos_origin_x, info.pos_origin_z);
  let base = vec3f(xz.x, inst.y, xz.y);
  let bits = inst.bits;
  let yaw = f32(bits & 1023u) * (TAU / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (TAU / 1024.0);
  let entry = stand_table[u32(info.entry_index)];

  // The baked captures are centred on the clump, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  let axis = base + rot_y(vec3f(info.cx, 0.0, info.cz) * scale, yaw);
  let to_cam = frame.camera_pos - axis;
  let dist_xz = length(to_cam.xz);
  let fwd_xz = to_cam.xz / max(dist_xz, 1e-4);

  // Nearest baked azimuth in the plant's local frame, plus the residual angle
  // to the true view direction (which the slab parallax covers).
  let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  let kf = ang * (BINS_F / TAU);
  let kr = round(kf);
  let bin = u32((i32(kr) + 4 * BINS_I) % BINS_I);
  let d_ang = (kf - kr) * (TAU / BINS_F);

  // blend: 0 = bin-fixed slab stack, 1 = collapsed camera-facing card. The
  // distance metric must match the cull's LOD test exactly (3D, to the plant's
  // mid-height centre) so the two representations meet where they swap.
  let pd = max(info.patch_dist, 1e-3);
  let center_y = base.y + (info.y0 + info.y1) * 0.5 * scale;
  let dist_3d = length(vec3f(to_cam.x, frame.camera_pos.y - center_y, to_cam.z));
  let blend = select(clamp(smoothstep(pd * 0.5, pd, dist_3d), 0.0, 1.0), 1.0, is_far);
  // Distance to the plant's core segment — drives both the camera-inside fade
  // and the slab-spread collapse below.
  let r = info.r_xz * scale;
  let h0 = info.y0 * scale;
  let h1 = info.y1 * scale;
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));

  let a_eff = f32(bin) * (TAU / BINS_F) + blend * d_ang;
  let ce = cos(a_eff);
  let se = sin(a_eff);
  let right_l = vec3f(ce, 0.0, -se);
  let depth_l = vec3f(se, 0.0, ce); // toward the camera

  // --- slice selection ------------------------------------------------------
  var slice = 0u;
  if (is_far) {
    slice = NEAR_SLICES + select(bin, BINS_U, q == Q_FAR_CROWN);
  } else if (is_crown) {
    slice = CROWN_BASE + (q - Q_CROWN0);
  } else {
    slice = bin * SLABS + q;
  }
  let layer = select(slice, slice - NEAR_SLICES, is_far);
  let rect = info.slice_rect[slice];
  let dep = info.slice_depth[slice].x;

  // --- quad geometry in the plant's local (unyawed) frame -------------------
  let u = mix(rect.x, rect.y, c.x);
  let v = mix(rect.w, rect.z, c.y);
  var local_xz: vec2f;
  var height: f32;
  if (is_crown) {
    // Crown capture basis: U = +X, V = -Z, so v maps to -z.
    local_xz = vec2f(info.cx + u, info.cz - v);
    // Collapse the two crown heights toward the composite height so the LOD
    // switch has nothing left to pop.
    height = select(mix(dep, info.crown_h, blend), dep, is_far);
  } else {
    // Slab planes carry the parallax, but pushing the front slab toward the
    // camera is only valid while the camera is OUTSIDE the plant's volume:
    // closer than that it magnifies the front half into soft blobs. Collapse
    // the spread over the bounding sphere so the stack degrades into exactly
    // the flat card the baseline draws, and only then let the near fade erode.
    let vol = smoothstep(r * 1.2, r * 2.6, d_core);
    let t = dep * info.slab_spread * (1.0 - blend) * vol;
    local_xz = vec2f(info.cx, info.cz) + right_l.xz * u + depth_l.xz * t;
    height = v;
  }

  var world = base + rot_y(vec3f(local_xz.x, height, local_xz.y), yaw) * scale;
  let h_frac = clamp(height / max(info.y1, 1e-4), 0.0, 1.0);
  world += wind_sway(base, frame.time, entry.sway, phase) * h_frac;

  // --- coverage fades (erosion of the alpha reference, never dither) --------
  let near_fade = smoothstep(r * 0.35, r * 1.05, d_core);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, dist_xz);
  var fade = near_fade * edge_fade;
  if (is_crown) {
    // A horizontal patch only makes sense seen from ABOVE: it holds top-down
    // imagery, so from below it is a dark floating cutout and from a flat angle
    // a pale pancake lying through the plant. Gate it on real elevation over
    // its own height — and gate the LOWER crown much harder, because the only
    // way to see the bottom band from above is to be nearly overhead.
    var gate = vec2f(0.50, 0.72);          // upper crown
    if (q == 3u) { gate = vec2f(0.70, 0.88); }
    if (q == Q_FAR_CROWN) { gate = vec2f(0.40, 0.65); }
    let dy = frame.camera_pos.y - (base.y + height * scale);
    let elev = dy / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
    fade *= smoothstep(gate.x, gate.y, elev) * info.top_enable;
  }

  // Canopy-depth darkening: slab 0 is the lit outer shell, slab 1 sits one slab
  // deeper inside the canopy; the composite cards get the average of the two.
  var depth_idx = f32(q);
  if (q == 2u) { depth_idx = 0.0; }
  if (q == 3u) { depth_idx = 1.0; }
  if (q >= Q_FAR) { depth_idx = 0.5; }

  var out: VOut;
  // A quad whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards — emit it behind the near plane
  // instead of rasterizing a guaranteed-empty quad (that set is exactly the
  // screen-filling one: crown patches at eye level, plants the camera stands
  // inside, the region rim).
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = c;
  out.world = world;
  out.layer = i32(layer);
  // Content rotation = plant yaw plus however far the plane was turned away
  // from the bake azimuth, so baked mesh-frame normals land in world space.
  let rot = yaw + blend * d_ang;
  out.rot_cs = vec2f(cos(rot), sin(rot));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.shade = mix(1.0 - info.bottom_shade, 1.0, h_frac) * (1.0 - info.slab_shade * depth_idx);
  out.quad = q;
  return out;
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode_patch(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

fn quad_tint(q: u32) -> vec3f {
  if (q == 0u) { return vec3f(0.95, 0.25, 0.25); }  // front slab
  if (q == 1u) { return vec3f(0.30, 0.45, 0.95); }  // back slab
  if (q == 2u) { return vec3f(0.95, 0.85, 0.30); }  // upper crown
  if (q == 3u) { return vec3f(0.30, 0.85, 0.90); }  // lower crown
  if (q == Q_FAR) { return vec3f(0.80, 0.80, 0.80); }
  return vec3f(0.95, 0.55, 0.20);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // ONE tap decides coverage. Screen-space derivatives are taken here, in
  // uniform control flow, so the normal can be fetched with an explicit
  // gradient AFTER the alpha test — a slab quad discards most of its
  // fragments and those must not pay for a second fetch.
  let du = dpdx(in.uv);
  let dv = dpdy(in.uv);
  let alb = textureSample(patch_albedo, patch_sampler, in.uv, in.layer);
  if (alb.a < in.erode) {
    discard;
  }
  let enc = textureSampleGrad(patch_normal, patch_sampler, in.uv, in.layer, du, dv).xy;
  let n_mesh = oct_decode_patch(enc * 2.0 - 1.0);
  var n = vec3f(
    in.rot_cs.x * n_mesh.x + in.rot_cs.y * n_mesh.z,
    n_mesh.y,
    -in.rot_cs.y * n_mesh.x + in.rot_cs.x * n_mesh.z,
  );
  // Thin foliage is lit from both sides (the bake already flipped toward the
  // capture camera; this catches the off-axis remainder).
  if (dot(n, frame.camera_pos - in.world) < 0.0) {
    n = -n;
  }

  var albedo = alb.rgb;
  if (info.tint > 0.5) {
    albedo = mix(albedo, quad_tint(in.quad), 0.65);
  }
  // in.shade is fake occlusion (canopy depth + grounding), so it belongs to the
  // light term: the albedo view then shows the baked patch colour exactly as
  // captured, and the lighting view shows sun+ambient x occlusion.
  var color = light_surface(albedo * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the baked alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin.
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}
