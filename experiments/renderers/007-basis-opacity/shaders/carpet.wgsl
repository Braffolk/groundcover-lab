#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// CARPET path — the shape a tiled mat (stand_table[i].carpet_div > 0) gets
// instead of the camera-facing Fourier card.
//
// One ground-parallel quad per tile (4 verts, triangle-strip), exactly
// footprint_m * scale across, conformed to the terrain per vertex, textured with
// the tile's own periodic top-down capture. No camera-facing geometry at all: a
// 7 cm cushion has no silhouette worth a billboard, and a vertical card through
// a mat is pure artifact. The grid step, the 90-degree yaw and the constant
// scale come from the stand and are used exactly as given — that agreement
// between neighbours is what makes it a mat rather than confetti.
//
// The Fourier basis is still here, but applied to the tile's GEOMETRY rather
// than its appearance: per texel we carry a truncated series in azimuth of the
// HORIZON angle (see horizon.wgsl), evaluated in closed form per fragment
// against the sun elevation (self-shadowing between capitula) and against the
// view elevation (crevices close as the camera drops toward grazing). That is
// the part a flat textured plane cannot do, and it is what gives the mat its
// cushion reading.

struct CarpetMeta {
  origin_cell: vec2f, side: f32, seed: f32,
  max_dist: f32, entry_index: f32, slot_mask: f32, slot_shift: f32,
  // plane_h: height of the quad above the tile base at scale 1 (the mean
  // capitulum apex — the surface the top-down capture actually shows).
  plane_h: f32, alpha_ref: f32, sun_shadow: f32, view_occl: f32,
  ao_amount: f32, overscale: f32, top_h: f32, relief_tint: f32,
}
@group(1) @binding(0) var<uniform> cm: CarpetMeta;
@group(1) @binding(1) var carpet_albedo: texture_2d<f32>;
@group(1) @binding(2) var carpet_relief: texture_2d<f32>;
@group(1) @binding(3) var carpet_horizon: texture_2d<f32>;
@group(1) @binding(4) var carpet_samp: sampler;

// Softness of the horizon comparison, in units of sin(elevation). ~7 degrees:
// the fit is order 1, so a hard step would show the fit's own error as a hard
// edge. Wide enough to hide it, narrow enough to stay a shadow and not a haze.
const HZ_SOFT: f32 = 0.12;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) up_ws: vec3f,                       // ground up, per vertex
  @location(3) @interpolate(flat) yaw_cs: vec2f,   // cos(yaw), sin(yaw)
  @location(4) @interpolate(flat) erode: f32,      // effective alpha reference
}

fn degenerate() -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;

  // Slots per cell is carpet_div^2 (484 for the bog moss), NOT
  // SCATTER_MAX_PER_CELL — enumerating 128 of them renders a quarter of the mat.
  // Rounded up to a power of two on the CPU so this is a shift, not a divide;
  // the scatter rejects the surplus slots itself.
  let slot = ii & u32(cm.slot_mask);
  let cell_lin = ii >> u32(cm.slot_shift);
  let side = u32(cm.side);
  let cxi = i32(cm.origin_cell.x) + i32(cell_lin % side);
  let czi = i32(cm.origin_cell.y) + i32(cell_lin / side);
  let entry = u32(cm.entry_index);

  // Cell-level reject before any placement work (see card.wgsl).
  let cell_lo = vec2f(f32(cxi), f32(czi)) * SCATTER_CELL_SIZE;
  let near_xz = clamp(frame.camera_pos.xz, cell_lo, cell_lo + SCATTER_CELL_SIZE);
  if (distance(frame.camera_pos.xz, near_xz) > cm.max_dist) { return degenerate(); }

  let sp = scatter_candidate(u32(cm.seed), entry, vec2i(cxi, czi), slot);
  if (!sp.exists) { return degenerate(); }

  // Region-edge dissolve, measured from the tile CENTRE so all four corners
  // agree (a per-vertex fade emits some corners at the clip point and
  // rasterizes the tile as a metres-long sliver). Eroding through the alpha
  // reference keeps hard edges — no dither. There is deliberately NO
  // camera-inside fade: a mat you are standing on must not open a hole.
  let d_xz = distance(frame.camera_pos.xz, sp.pos.xz);
  let fade = 1.0 - smoothstep(cm.max_dist * 0.82, cm.max_dist, d_xz);
  let erode = cm.alpha_ref / max(fade, 1.0e-4);
  if (erode > 1.0) { return degenerate(); }

  var corners = array<vec2f, 4>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  let c = corners[vi];

  // Width from the species' periodic FOOTPRINT, never from height_scale: this
  // moss is 0.07 m tall and 0.24 m wide, and footprint_m * scale is exactly the
  // carpet's grid step, so tiles abut.
  let tile_m = stand_table[entry].footprint_m;
  let mesh_off = c * (0.5 * tile_m * cm.overscale);
  let cy = cos(sp.yaw);
  let sy = sin(sp.yaw);
  // rot_y(offset, yaw) for the 90-degree-stepped tile yaw.
  let off = vec2f(cy * mesh_off.x + sy * mesh_off.y, -sy * mesh_off.x + cy * mesh_off.y) * sp.scale;
  let xz = sp.pos.xz + off;

  // Rung 3 of the terrain-fitting ladder: the ground under every vertex.
  // Neighbouring tiles SHARE corner positions, so per-vertex is the only rung
  // that keeps the whole mat C0-continuous — a per-tile plane fit (rung 1/2)
  // cracks at every tile edge because two neighbours fit two different planes.
  // terrain_sample gives height and (nx, nz) from the same four texel loads, so
  // the shading basis costs nothing extra.
  let g = terrain_sample(xz);
  out.up_ws = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  let world = vec3f(xz.x, g.x + cm.plane_h * sp.scale, xz.y);

  out.pos = frame.view_proj * vec4f(world, 1.0);
  // Tile-local uv: the capture is the tile square [0, tile_m]^2 of the mesh
  // frame (origin (0,0) for every source mesh), so this is a straight 0..1 map.
  // The sampler wraps, and the capture is exactly periodic, so an overscale
  // simply reads into the next period instead of stretching an edge.
  out.uv = 0.5 + 0.5 * c * cm.overscale;
  out.world = world;
  out.yaw_cs = vec2f(cy, sy);
  out.erode = erode;
  return out;
}

/// Horizon elevation (radians) at local azimuth (ct, st) from the order-1 fit.
fn horizon_beta(hz: vec4f, ct: f32, st: f32) -> f32 {
  let dc = hz.r * 1.5707963;
  let c1 = (hz.g * 2.0 - 1.0) * 1.5707963;
  let s1 = (hz.b * 2.0 - 1.0) * 1.5707963;
  return max(dc + c1 * ct + s1 * st, 0.0);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Sample everything before any non-uniform discard (uniform control flow).
  let alb = textureSample(carpet_albedo, carpet_samp, in.uv);
  let relief = textureSample(carpet_relief, carpet_samp, in.uv);
  let hz = textureSample(carpet_horizon, carpet_samp, in.uv);
  // The same two fits four mip levels coarser, i.e. averaged over ~16x16 texels
  // (5.6 mm, about one capitulum): the LOCAL MEAN of the horizon and of the sky
  // visibility. Every geometric shading term below is the difference between the
  // texel and its neighbourhood, never the absolute value — see the shade block.
  // A relative bias rather than a fixed level so the window stays four levels
  // coarser at every distance, and both taps clamp to the same 1x1 level in the
  // far field, where relief is genuinely unresolvable and the terms vanish.
  let hz_mean = textureSampleBias(carpet_horizon, carpet_samp, in.uv, 4.0);
  let relief_mean = textureSampleBias(carpet_relief, carpet_samp, in.uv, 4.0);
  if (alb.a < in.erode) { discard; }

  // Tile basis: mesh +x -> t, mesh +y -> ground up, mesh +z -> b. This is
  // plant_basis_from_up(up, yaw) inlined (the vertex stage already carries
  // cos/sin(yaw) rather than the angle), so a mat on a slope both lights and
  // self-shadows as a slope.
  let up = normalize(in.up_ws);
  var t = vec3f(in.yaw_cs.x, 0.0, -in.yaw_cs.y);
  let proj = t - up * dot(up, t);
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  let b = cross(t, up);

  // Cushion normal: the height field's gradient, stored as the xz of a unit
  // vector so y is recovered here and a mip-filtered tap simply flattens (see
  // horizon.wgsl for why the mesh's own leaf normals are unusable). Lifted from
  // the mesh frame into the tile basis.
  let nxz = relief.rg * 2.0 - 1.0;
  let n_mesh = vec3f(nxz.x, sqrt(max(1.0 - dot(nxz, nxz), 0.0)), nxz.y);
  let normal = normalize(t * n_mesh.x + up * n_mesh.y + b * n_mesh.z);

  // --- closed-form horizon evaluation ---------------------------------------
  // Both directions are taken in the tile's own frame, so the fit is queried in
  // the frame it was baked in.
  let sun = frame.sun_dir;
  let ms = vec3f(dot(sun, t), dot(sun, up), dot(sun, b));
  let sl = max(length(ms.xz), 1.0e-5);
  let sct = ms.x / sl;
  let sst = ms.z / sl;
  let sun_vis = smoothstep(-HZ_SOFT, HZ_SOFT, ms.y - sin(horizon_beta(hz, sct, sst)));
  let sun_vis_mean = smoothstep(-HZ_SOFT, HZ_SOFT, ms.y - sin(horizon_beta(hz_mean, sct, sst)));

  let vdir = normalize(frame.camera_pos - in.world);
  let mv = vec3f(dot(vdir, t), dot(vdir, up), dot(vdir, b));
  let vl = max(length(mv.xz), 1.0e-5);
  let vct = mv.x / vl;
  let vst = mv.z / vl;
  let view_beta = horizon_beta(hz, vct, vst);
  let view_contrast = sin(horizon_beta(hz_mean, vct, vst)) - sin(view_beta);
  let obliquity = 1.0 - clamp(mv.y, 0.0, 1.0);

  // Fake occlusion belongs to the LIGHT term, not to the albedo: `albedo` stays
  // exactly the baked capture (so debug=albedo is honest) and debug=lighting
  // shows sun+ambient x self-shadow x sky visibility x view bias.
  //
  // All three terms are measured AGAINST THE LOCAL MEAN, so they shade the mat's
  // relief without moving its mean brightness. Absolute occlusion was tried first
  // and is worse on both counts: the fitted horizon over-estimates occlusion
  // (Sphagnum's top surface is a fuzz of leaf tips that light passes between, not
  // a solid height field), which cost the whole mat ~35% brightness against the
  // 001-billboard-smoke baseline, and it darkens the far field most, exactly where
  // no relief is resolvable and the answer should converge to a plain textured
  // plane. Relative terms vanish there on their own.
  //   * sun: darken where this texel is more shadowed than its capitulum. 0.62 is
  //     roughly the direct sun's share of the shared light model.
  //   * sky: same, for the ambient share (~0.22 here).
  //   * view: at grazing what you see is biased toward what sticks up — the tops
  //     of the capitula, not the hollows behind them. Absolute view occlusion is
  //     useless (at grazing EVERY texel is behind something, so it collapses to
  //     one uniform darkening of the whole field — measured, it turned the mat
  //     into a flat brown plane), so this is a signed contrast scaled by obliquity.
  // Signed, not one-sided: a term that only ever darkens is not mean-neutral (a
  // one-sided clamp still cost the mat 19% of its brightness), so a texel that is
  // LESS occluded than its capitulum brightens by the same law.
  let sun_term = clamp(1.0 - cm.sun_shadow * 0.62 * (sun_vis_mean - sun_vis), 0.3, 1.5);
  let ao_term = clamp(1.0 - cm.ao_amount * 0.22 * (relief_mean.b - relief.b), 0.6, 1.3);
  let view_term = clamp(1.0 + cm.view_occl * obliquity * 2.0 * view_contrast, 0.25, 1.75);
  let shade = ao_term * sun_term * view_term;

  var color = light_surface(alb.rgb * shade, normal, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
    if (cm.relief_tint > 0.0) {
      // Inspection aid: baked relief height (red -> green) and, in blue, whether
      // this texel is visible from here at all (1 = above the view horizon) —
      // method-specific state the global debug views cannot show.
      let view_vis = smoothstep(-HZ_SOFT, HZ_SOFT, mv.y - sin(view_beta));
      color = mix(color, vec3f(relief.a, 1.0 - relief.a, view_vis), cm.relief_tint);
    }
  }
  return vec4f(debug_shade(color, alb.rgb, normal, alb.a, in.world), 1.0);
}
