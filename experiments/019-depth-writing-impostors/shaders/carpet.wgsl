#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// TRUE-DEPTH CARPET — the same idea as impostor.wgsl, applied to a mat.
// ====================================================================
// A carpet species (stand_table[i].carpet_div > 0, i.e. the bog's Sphagnum) is
// not an upright plant and must not be drawn like one. It is a periodic
// community tile 0.18 m across and 0.07-0.09 m tall, grid-snapped at a constant
// scale with 90-degree-only rotations. A screen-aligned card for that is
// indefensible: it stands the mat on edge, breaks the lattice the moment the
// camera moves, and shows nothing at grazing angles.
//
// So a carpet tile draws ONE ground-parallel quad — but the impostor's central
// claim still holds, and this is where it pays off against a plain textured
// card: a texel of the baked TOP VIEW is not paint, it is the 3D point
//
//     Q = P + N * (2*ew*scale) * (depth - 0.5)                          (1)
//
// where P is the point on the quad, N the ground normal and `depth` the signed
// depth channel the bake stored. The mesh's own capitulum relief (33 mm on the
// wet state) therefore survives, in two ways that a flat card cannot do:
//
//  1. PARALLAX. The ray that lands at P really sees the surface at P + W,
//     W = d * (V/(V.N) - N), i.e. the in-plane slide of a point lifted by d.
//     One dependent tap gives d, one first-order step gives W — no loop, no
//     marching. Bumps slide against the hollows between them as the camera
//     moves, which is the whole reason a cushion reads as a cushion.
//     Offset-limited (the 1/(V.N) is clamped) and faded out at grazing, where
//     a single-layer reprojection has no right to an answer. The tile image is
//     PERIODIC, so a step that leaves the tile square wraps rather than clamps
//     — the wrap is exactly the geometric continuation.
//
//  2. TRUE PER-PIXEL DEPTH. The near band writes (1) as frag_depth, so the mat
//     has thickness: neighbouring tiles interleave per pixel at grazing, the
//     grass stems growing through it are occluded at their real depth, and the
//     mat's edge against the sky is the ragged capitulum line instead of a
//     straight plane. It costs early-z, so only the near shell pays for it.
//
// THE LATTICE IS THE GEOMETRY. The quad is axis-aligned in xz and exactly one
// grid step across (stand_table.footprint_m * the carpet scale), so its corners
// coincide with its neighbours' corners. The tile's 90-degree yaw rotates the
// TEXTURE inside the tile square, never the quad. Nothing here billboards,
// jitters the scale, or shrinks a tile with distance.
//
// TERRAIN FITTING: rung 3 of the CLAUDE.md ladder (per-vertex conforming).
// Every corner gets its own terrain_sample() — height and (nx, nz) from one
// bilinear fetch — so the whole mat is C0-continuous. A per-tile plane fit
// (rungs 1-2) is not merely cheaper here, it is wrong: two neighbours fit two
// different planes and crack apart along their shared edge.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(3) var albedo_tex: texture_2d_array<f32>;
@group(1) @binding(4) var geo_tex: texture_2d_array<f32>;
@group(1) @binding(5) var atlas_sampler: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  /** texcoord of the quad point itself (before any relief step) */
  @location(0) uv: vec2f,
  /** world position on the quad plane */
  @location(1) world: vec3f,
  /** ground normal, sampled per vertex and interpolated */
  @location(2) up_ws: vec3f,
  /** cos(yaw), sin(yaw) — the tile's 90-degree texture rotation */
  @location(3) @interpolate(flat) yaw_cs: vec2f,
  /** effective alpha reference after the region-rim fade */
  @location(4) @interpolate(flat) erode: f32,
}

@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  // Only the yaw is read from the instance: a carpet's scale is constant by
  // definition and comes from the stand table, so it is never the quantised
  // per-plant value and every tile is guaranteed the same size.
  let yaw = f32(inst.packed_bits & 1023u) * (6.2831853 / 1024.0);
  let cs = cos(yaw);
  let sn = sin(yaw);

  var corners = array<vec2f, 6>(
    vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(-0.5, 0.5),
    vec2f(0.5, -0.5), vec2f(0.5, 0.5), vec2f(-0.5, 0.5),
  );
  let a = corners[vi];

  // Axis-aligned, exactly one grid step across: corners land on the lattice and
  // are shared with the four neighbours.
  let xz = inst.pos.xz + a * info.tile_world;
  let g = terrain_sample(xz);
  let up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  // Quad plane = the baked capture centre, i.e. mid-canopy, so the stored depth
  // reaches from the peat surface to the capitulum tops symmetrically. The
  // offset is vertical and constant, which keeps neighbours in agreement.
  let world = vec3f(xz.x, g.x + info.carpet_y, xz.y);

  // Texture rotation by -yaw about the tile centre, mapped into the tile's own
  // square of the baked top view.
  let m = vec2f(cs * a.x - sn * a.y, sn * a.x + cs * a.y) + vec2f(0.5);
  let uv = vec2f(info.t_lo_x, info.t_lo_y) + m * vec2f(info.t_span_x, info.t_span_y);

  // No camera-inside fade: a mat you are standing on must not open a hole
  // under you. Only the region rim erodes, measured from the tile CENTRE — a
  // per-vertex fade would emit part of a quad behind the near plane and
  // rasterize the rest as a metres-long sliver.
  let d_xz = length(frame.camera_pos.xz - inst.pos.xz);
  let fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);

  var out: VOut;
  if (fade < info.carpet_alpha) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = uv;
  out.world = world;
  out.up_ws = up;
  out.yaw_cs = vec2f(cs, sn);
  out.erode = info.carpet_alpha / max(fade, 1.0e-3);
  return out;
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

struct Hit {
  color: vec3f,
  /** reconstructed 3D surface point, equation (1) */
  world: vec3f,
}

fn shade_carpet(in: VOut, do_relief: bool) -> Hit {
  let layer = i32(info.carpet_layer);
  let t0 = in.uv;
  // Mip level from the SCREEN-SPACE footprint of the unwarped texcoord. A mat
  // is a ground plane, so its minification is strongly anisotropic and a
  // distance-derived level (right for an upright plant) either blurs it or
  // aliases it. Taking the derivatives of t0 rather than of the relief-stepped
  // coordinate also stops a depth discontinuity from spiking the level.
  let fw = max(length(dpdx(t0)), length(dpdy(t0)));
  let lod = clamp(log2(max(fw * info.tile_px, 1.0e-6)), 0.0, info.max_lod);

  let up = normalize(in.up_ws);
  var t = t0;
  var slide = vec3f(0.0);
  if (do_relief) {
    let to_cam = frame.camera_pos - in.world;
    let v = to_cam / max(length(to_cam), 1.0e-5);
    let vn = dot(v, up);
    // Tap 1: the canopy height under this pixel, in world metres about the
    // quad plane.
    let d0 = (textureSampleLevel(geo_tex, atlas_sampler, t0, layer, lod).b - 0.5) * info.relief_m;
    // W = d * (V/(V.N) - N): the in-plane slide of a point lifted by d, seen
    // along V. 1/(V.N) is clamped (offset limiting) and the whole step fades
    // out below ~25 degrees, where one Newton step off a single depth layer
    // would land anywhere and smear the mat.
    let f = info.warp_scale * smoothstep(0.12, 0.45, vn);
    slide = (d0 * f) * (v / max(vn, 0.30) - up);
    let w = slide.xz;
    // World xz -> the tile's own (yawed) texture axes.
    let mm = vec2f(in.yaw_cs.x * w.x - in.yaw_cs.y * w.y, in.yaw_cs.y * w.x + in.yaw_cs.x * w.y);
    t = t0 + mm * vec2f(info.t_per_m_x, info.t_per_m_y);
    // The tile image is periodic, so a step that leaves the square wraps into
    // the neighbouring period — which is the same content, continuously.
    let span = vec2f(info.t_span_x, info.t_span_y);
    let lo = vec2f(info.t_lo_x, info.t_lo_y);
    let rel = (t - lo) / span;
    t = lo + (rel - floor(rel)) * span;
  }

  let alb = textureSampleLevel(albedo_tex, atlas_sampler, t, layer, lod);
  if (alb.a < in.erode) {
    discard;
  }
  let geo = textureSampleLevel(geo_tex, atlas_sampler, t, layer, lod);

  // The baked normal is a mesh-frame normal captured over flat ground, so it
  // has to be lifted into the local GROUND frame, not merely yawed, or a mat
  // on a slope lights as if it were level. This is plant_basis_from_up(up, yaw)
  // from terrain.wgsl, inlined because the quad already carries cos/sin(yaw).
  let n_mesh = oct_decode(geo.rg * 2.0 - 1.0);
  var basis_t = vec3f(in.yaw_cs.x, 0.0, -in.yaw_cs.y);
  let proj = basis_t - up * dot(up, basis_t);
  basis_t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  let n = basis_t * n_mesh.x + up * n_mesh.y + cross(basis_t, up) * n_mesh.z;

  // Equation (1): the real surface point this pixel sees.
  let world = in.world + slide + up * ((geo.b - 0.5) * info.relief_m);

  // Occlusion, but NOT the baked volumetric grid. That grid is 24x40x24 over
  // the mesh AABB, which for a 0.21 x 0.09 x 0.23 m mat means 9.2 mm voxels
  // horizontally and 2.4 mm vertically — a 3.8:1 aspect the trace does not
  // correct for, since it steps in VOXELS: the up-rays leave the canopy after
  // 22 mm while the sideways rays run 83 mm, so the result is dominated by
  // horizontal transmittance and paints the source tile's ramet rows as broad
  // soft bands. Repeated by every tile and turned 90 degrees by the lattice,
  // that reads as basketweave — measured, not assumed (aoStrength 0 removes it
  // completely, 0.35 still shows it).
  //
  // The cushion cue a mat actually wants is "how deep in the cushion is this
  // texel", and the depth channel IS that, per texel and already sampled: the
  // capitulum tops stay lit, the gaps down toward the peat go dark, at full
  // atlas resolution instead of a blurred 9 mm grid.
  // Deliberately gentle: the depth channel also carries the tile's own
  // low-frequency hummock, and a steep ramp would repeat THAT in every tile —
  // the same basketweave by another route. At 0.45 of contrast the sharp
  // per-texel crevices dominate and the low-frequency part stays under ~7%.
  let cavity = 0.55 + 0.45 * geo.b;
  let ao = mix(1.0, cavity, info.ao_strength);
  var color = light_surface(alb.rgb * ao, n, world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, world);
  }

  var out: Hit;
  out.color = debug_shade(color, alb.rgb, n, alb.a, world);
  out.world = world;
  return out;
}

struct DepthOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

/// Near band: relief step + TRUE per-pixel depth. This is what gives the mat
/// thickness — tiles interleave per pixel across a slope, grass stems are
/// occluded at their real depth, and the silhouette is the capitulum line.
@fragment
fn fs_carpet_near(in: VOut) -> DepthOut {
  let hit = shade_carpet(in, true);
  // A hair toward the camera: the reconstructed peat surface sits exactly on
  // the terrain the base pass already rendered.
  let p = hit.world + normalize(frame.camera_pos - in.world) * info.carpet_push;
  let clip = frame.view_proj * vec4f(p, 1.0);
  var out: DepthOut;
  out.color = vec4f(hit.color, 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1.0e-5), 0.0, 1.0);
  return out;
}

/// Mid band: same relief step, planar depth from the conformed quad — which is
/// an honest ground surface, unlike a card, so early-z stays alive.
@fragment
fn fs_carpet_mid(in: VOut) -> @location(0) vec4f {
  return vec4f(shade_carpet(in, true).color, 1.0);
}

/// Far band: no relief step (one tap saved). Beyond ~30 m the 3 cm of
/// capitulum relief is well under a pixel.
@fragment
fn fs_carpet_far(in: VOut) -> @location(0) vec4f {
  return vec4f(shade_carpet(in, false).color, 1.0);
}
