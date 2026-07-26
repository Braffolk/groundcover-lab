#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// The continuous end of the dissolve: ONE displaced canopy surface, textured
// with the tileable top-view composite of the active stand's species mix
// (built at init from the baked per-species top captures — see canopy.ts).
//
// Bufferless nested grid, two levels of 96x96 quads at 0.75 m and 3 m, snapped
// to world multiples of the cell size so vertices never swim. Cost is O(visible
// area), NOT O(plants): this is where thousands of distant plants collapse into
// a few thousand triangles and one alpha-tested surface.
//
// The surface only exists where the canopy has dissolved: the coverage
// threshold starts above 1.0 (nothing passes) near the camera and falls to the
// far-field floor, so the same texture reads as "islands of dense clumps"
// mid-way and as a closed canopy far away. Quads that cannot pass the threshold
// at all are collapsed in the vertex stage, so the near field — the part that
// would cover the most pixels — is never rasterized.

@group(1) @binding(0) var<uniform> shell: ShellInfo;
@group(1) @binding(1) var canopy_alb: texture_2d<f32>;
@group(1) @binding(2) var canopy_aux: texture_2d<f32>;
@group(1) @binding(3) var canopy_sampler: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) nrm: vec3f,
  @location(3) thresh: f32,
  @location(4) macro_var: f32,
  @location(5) sin_elev: f32,
}

fn terrain_normal_from(s: vec3f) -> vec3f {
  return vec3f(s.y, sqrt(max(1.0 - s.y * s.y - s.z * s.z, 0.0)), s.z);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let cells = u32(shell.p2.x);
  let g = u32(shell.p2.y);
  let level = min(ii / cells, 1u);
  let ci = ii % cells;
  let lv = shell.levels[level];
  let s = lv.z;
  let p0 = lv.xy + vec2f(f32(ci % g), f32(ci / g)) * s;
  let pc = p0 + vec2f(s * 0.5);

  let stand_half = shell.p0.x;
  let base_h = shell.p0.y;
  let relief = shell.p0.w;
  let center_t = terrain_sample(pc);
  let center_y = center_t.x + canopy_relief(pc, base_h, relief);
  let d_center = dissolve_amount(vec3f(pc.x, center_y, pc.y), shell.p1.y, shell.p1.z);

  // --- quad-uniform rejects -------------------------------------------------
  var skip = max(abs(pc.x), abs(pc.y)) > stand_half;
  let hole = shell.holes[level];
  if (p0.x >= hole.x && p0.y >= hole.y && p0.x + s <= hole.z && p0.y + s <= hole.w) {
    skip = true; // covered by the finer level
  }
  if (d_center < shell_dead_d(shell.p1.x)) {
    skip = true; // still individual plants here — nothing could pass the threshold
  }
  if (!skip) {
    let sph_r = s * 0.71 + 0.6;
    for (var i = 0; i < 6; i++) {
      let pl = shell.planes[i];
      if (dot(pl.xyz, vec3f(pc.x, center_y, pc.y)) + pl.w < -sph_r) {
        skip = true;
      }
    }
  }

  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let xz = p0 + corners[vi % 6u] * s;
  let t = terrain_sample(xz);
  let cr = canopy_relief(xz, base_h, relief);
  let y = t.x + cr;

  // Analytic macro normal: terrain slope plus the canopy relief's own slope.
  let e = 1.2;
  let dhx = (canopy_relief(xz + vec2f(e, 0.0), base_h, relief) - cr) / e;
  let dhz = (canopy_relief(xz + vec2f(0.0, e), base_h, relief) - cr) / e;
  let tn = terrain_normal_from(t);
  let macro_n = normalize(vec3f(tn.x - dhx, 1.0, tn.z - dhz));

  // Coherent wind ripple: a smooth function of world position, so the surface
  // stays watertight and the whole meadow waves at once.
  let sway = wind_sway(vec3f(xz.x, t.x, xz.y), frame.time, shell.p2.z, 0.0) * 0.8;
  let world = vec3f(xz.x + sway.x, y + sway.y, xz.y + sway.z);

  let d = dissolve_amount(vec3f(xz.x, y, xz.y), shell.p1.y, shell.p1.z);
  let e_out = max(abs(xz.x), abs(xz.y));
  let edge_k = 1.0 - smoothstep(stand_half - shell.p2.w, stand_half, e_out);

  var out: VOut;
  if (skip) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = xz / shell.p0.z;
  out.world = world;
  out.nrm = macro_n;
  out.thresh = mix(1.2, shell_threshold(d, shell.p1.x), edge_k);
  // Slow macro variation, so the 12 m texture repeat does not read as a grid.
  let to_cam = frame.camera_pos - vec3f(xz.x, y, xz.y);
  out.macro_var = 0.87 + 0.28 * vnoise2(xz * 0.055);
  out.sin_elev = clamp(abs(to_cam.y) / max(length(to_cam), 1.0e-3), 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let alb = textureSample(canopy_alb, canopy_sampler, in.uv);
  let aux = textureSample(canopy_aux, canopy_sampler, in.uv);
  // The baked coverage is what a canopy hides looking straight DOWN. A ray at
  // elevation θ travels 1/sin(θ) as far through the same layer, so its coverage
  // is 1-(1-c)^(1/sinθ) — the physical reason a meadow is see-through from above
  // and opaque edge-on. One pow(), and it is what closes the far field at
  // grazing without touching the near field (the dissolve threshold above 1.0
  // still kills that, since this can never exceed 1).
  let cov = 1.0 - pow(max(1.0 - alb.a, 0.0), 1.0 / max(in.sin_elev, 0.07));
  if (cov < in.thresh) {
    discard;
  }
  // Shading normal. Looking down, the visible surface IS the canopy top, so use
  // the macro surface normal perturbed by the baked detail normal. Looking along
  // the canopy you see the sides of blades, which statistically face the viewer —
  // the same "flip toward the camera" convention the splats' baked normals use.
  let n_detail = oct_decode_ds(aux.xy * 2.0 - 1.0);
  let n_top = normalize(normalize(in.nrm) + (n_detail - vec3f(0.0, 1.0, 0.0)));
  let to_cam_dir = normalize(frame.camera_pos - in.world);
  let n_side = normalize(vec3f(to_cam_dir.x, max(to_cam_dir.y, 0.2), to_cam_dir.z) + n_detail * 0.35);
  let view_k = smoothstep(0.08, 0.75, in.sin_elev);
  let n = normalize(mix(n_side, n_top, view_k));
  // aux.z is the baked canopy height fraction: troughs sit deeper in the
  // canopy, so they are occluded — cheap, correct-looking depth cue.
  let ao = mix(1.0 - shell.p1.w, 1.0, aux.z);
  // One extra tap along the sun's azimuth turns the same heightfield into real
  // clump shadows: if the canopy one step toward the sun is higher than the sun
  // ray at that step, this texel is in its shadow. This is what makes the
  // surface read as clumped 3D mass rather than a printed texture.
  let hmax = shell.p4.w;
  let sun_xz = normalize(frame.sun_dir.xz + vec2f(1.0e-5, 0.0));
  let tan_elev = frame.sun_dir.y / max(length(frame.sun_dir.xz), 1.0e-3);
  let step_m = 0.55;
  let h_here = aux.z * hmax;
  let h_sun = textureSample(canopy_aux, canopy_sampler, in.uv + sun_xz * (step_m / shell.p0.z)).z * hmax;
  let shadow = 1.0 - 0.5 * shell.p1.w * smoothstep(0.02, 0.3, h_sun - (h_here + step_m * tan_elev));
  // Albedo: the tile's own colour from above, the side capture's hue at grazing,
  // with the tile supplying the texture in both cases.
  let lum = (alb.r + alb.g + alb.b) * (1.0 / 3.0);
  let side_hue = mix(shell.p4.rgb, shell.p3.rgb, clamp(aux.z * shell.p5.x, 0.0, 1.0));
  let side_alb = side_hue * (lum / max(shell.p3.w, 1.0e-3));
  let albedo = mix(side_alb, alb.rgb, view_k) * in.macro_var;
  var color = light_surface(albedo * ao * shadow, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, cov, in.world), 1.0);
}
