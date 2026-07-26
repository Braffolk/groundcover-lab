#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./common.wgsl"

// The carpet end of the same stand: a periodic mat species (stand carpet_div >
// 0 — Sphagnum palustre here) is NOT a plant with a silhouette. It is a 0.18m
// square community tile, 0.09m tall, laid on a grid at constant scale and
// 90-degree rotations. Drawn as this experiment's sub-tuft splats it was
// indefensible: four camera-facing cards standing upright out of a cushion,
// sized from the plant's capture radius, ignoring the slope — vertical blades
// slicing through the ground with bare peat between them.
//
// So a carpet entry gets its own primitive: ONE ground-parallel quad per tile,
// exactly footprint_m * scale across, conformed to the terrain PER VERTEX
// (ladder rung 3 — neighbouring tiles share corner positions, so this is the
// only rung that keeps the whole mat C0-continuous; any per-tile plane fit
// cracks at every tile edge), textured with the seamless top-down capture of
// the tile square (bake.ts, 512px over 0.18m).
//
// What makes it more than a picture of moss on the ground: the same capture
// stores the HEIGHT of the topmost surface, so the 3.3cm of capitulum relief is
// available per texel. Three things use it, in rising order of what they buy:
//   * cavity occlusion — the deeper a texel sits between capitula, the less sky
//     it sees;
//   * a one-tap self-shadow along the sun azimuth, so capitula cast onto their
//     neighbours and the mat reads as a lumpy mass rather than a flat print;
//   * a single-step parallax offset of the sample point, so the cushion
//     actually shifts against itself as the camera moves. That is the one cue a
//     flat card cannot fake, and it is bounded (`cfg3.w`) so a grazing ray can
//     never smear the texture across the tile.
// The lattice is untouched: the yaw comes from the scatter and is snapped to an
// exact quarter turn, the scale is the stand's constant, and there is no
// overscale — neighbouring tiles still agree.

struct PlantInst {
  pos: vec3f,
  bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var tile_albedo: texture_2d<f32>;
@group(1) @binding(3) var tile_aux: texture_2d<f32>;
@group(1) @binding(4) var tile_sampler: sampler;

/** Distance (m) of the self-shadow probe along the sun azimuth. */
const SUN_STEP: f32 = 0.022;
/**
 * Mip the parallax height is read from: 512px/0.18m at level 5 = 11mm texels.
 * A one-step offset can only stay coherent while it is SMALLER than the lateral
 * correlation length of the height that drives it — offset it by more and
 * neighbouring fragments sample unrelated places, which at a low oblique angle
 * combs the mat into vertical streaks (measured at level 3 + a 20mm clamp).
 * Level 5 resolves the cushion mounds (~15mm) and nothing finer, so the ~8mm
 * clamp below stays well inside it.
 */
const PARALLAX_LOD: f32 = 5.0;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) up_ws: vec3f,
  @location(3) @interpolate(flat) tan_ws: vec3f,
  @location(4) @interpolate(flat) erode: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let entry = stand_table[u32(info.cfg0.y)];

  // The tile's yaw is one of four quarter turns. Snap the packed angle back to
  // the exact quarter and build cos/sin from a table: a 0.35-degree rounding
  // error would be enough to stop neighbouring tiles abutting.
  let yaw_q = f32(inst.bits & 1023u) * (6.2831853 / 1024.0);
  let quarter = u32(round(yaw_q * (2.0 / 3.14159265))) & 3u;
  var cs = vec2f(1.0, 0.0);
  if (quarter == 1u) {
    cs = vec2f(0.0, 1.0);
  } else if (quarter == 2u) {
    cs = vec2f(-1.0, 0.0);
  } else if (quarter == 3u) {
    cs = vec2f(0.0, -1.0);
  }
  // Constant scale, straight from the stand: a carpet's tiles must all be the
  // same size, so there is nothing to unpack (and no quantization error).
  let scale = entry.scale_min;
  // Width from the species' periodic FOOTPRINT, never from its height — moss is
  // 0.07m tall and 0.24m wide, and sizing it by height leaves a mat with gaps.
  let half_m = entry.footprint_m * 0.5 * scale;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];
  let local = c * half_m;
  let off = vec2f(cs.x * local.x + cs.y * local.y, -cs.y * local.x + cs.x * local.y);
  let xz = inst.pos.xz + off;

  // Rung 3: the ground under THIS corner. terrain_sample returns the height and
  // the (nx, nz) of the ground normal from the same four texel loads, so the
  // shading basis costs nothing extra.
  let g = terrain_sample(xz);
  let up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  // cfg4 is already in world metres (a carpet's scale is a constant).
  let plane_y = info.cfg4.z + info.cfg4.x * info.cfg4.y;
  let world = vec3f(xz.x, g.x + plane_y, xz.y);

  // Region hand-off, measured at the tile CENTRE — never per vertex. The fade
  // decides whether the quad is emitted at all, so two corners disagreeing
  // would stretch one tile into a screen-long sliver.
  let mid = inst.pos + vec3f(0.0, plane_y, 0.0);
  let fade = tuft_fade(dissolve_amount(mid, info.cfg2.x, info.cfg2.y));
  // No camera-inside fade: a mat you are standing on must not open a hole.

  var out: VOut;
  if (fade < info.cfg1.z) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0); // every fragment would discard
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = c * 0.5 + vec2f(0.5);
  out.world = world;
  out.up_ws = up;
  out.tan_ws = vec3f(cs.x, 0.0, -cs.y); // the tile's mesh +x axis in world xz
  out.erode = info.cfg1.z / max(fade, 1.0e-3);
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let up = normalize(in.up_ws);
  // Tile frame lifted into the local GROUND plane: mesh +x -> t, +y -> up,
  // +z -> b. The capture was taken over flat ground, so on a slope the baked
  // normals have to be rotated into this frame, not merely yawed.
  var t = in.tan_ws - up * dot(up, in.tan_ws);
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(t), dot(t, t) > 1.0e-6);
  let b = cross(t, up);

  let tile_m = info.cfg3.z;
  let span_m = info.cfg4.y;
  let dux = dpdx(in.uv);
  let duy = dpdy(in.uv);
  // Single-step parallax. The quad's plane sits at the mat's MEAN surface
  // (cfg4.x), so a texel above it comes toward the viewer and one below recedes,
  // which halves the offset a top-referenced plane would need. The offset is
  // clamped in metres, so at grazing it saturates into a fixed shift instead of
  // smearing the tile across the screen.
  //
  // The height that DRIVES the offset is read from a deliberately coarse mip.
  // A one-step offset warps the texture by whatever the height says, so reading
  // it per texel scrambles the fine detail into mush (measured: visibly softer
  // than parallax off). At ~3mm texels the cushion mounds survive and the fine
  // capitula ride along undistorted, which is the parallax you actually see —
  // the lumps slide, the fuzz does not. `lod_geom` keeps it no finer than the
  // fragment's own footprint, so the offset also fades out with distance
  // instead of turning into noise.
  let dim = f32(textureDimensions(tile_aux, 0).x);
  let lod_geom = 0.5 * log2(max(dot(dux * dim, dux * dim), dot(duy * dim, duy * dim)) + 1.0e-9);
  let eye = normalize(frame.camera_pos - in.world);
  let h_plane = textureSampleLevel(tile_aux, tile_sampler, in.uv, max(PARALLAX_LOD, lod_geom)).z;
  let depth_m = (info.cfg4.x - h_plane) * span_m * info.cfg4.w;
  var slide = -vec2f(dot(eye, t), dot(eye, b)) * (depth_m / max(dot(eye, up), 0.25));
  let lim = info.cfg3.w;
  let mag = length(slide);
  slide = select(slide, slide * (lim / max(mag, 1.0e-6)), mag > lim);
  let uv = in.uv + slide / max(tile_m, 1.0e-6);

  // Sample the OFFSET point with the UNOFFSET derivatives. The parallax slide
  // varies per fragment, so an implicit-derivative sample reads a noisy
  // footprint and lands one or two mip levels too coarse — the whole mat goes
  // soft the moment parallax is switched on, which is exactly the sharpness the
  // 512px tile was baked to buy. The quad's own uv gradient is the honest
  // footprint here.
  let alb = textureSampleGrad(tile_albedo, tile_sampler, uv, dux, duy);
  let aux = textureSampleGrad(tile_aux, tile_sampler, uv, dux, duy);
  // Self-shadow probe: if the cushion one step toward the sun stands higher
  // than the sun ray does at that step, this texel is in its shadow.
  let sun_h = vec2f(dot(frame.sun_dir, t), dot(frame.sun_dir, b));
  let sun_dir_t = normalize(sun_h + vec2f(1.0e-5, 0.0));
  let sun_up = dot(frame.sun_dir, up);
  let h_sun = textureSampleGrad(
    tile_aux, tile_sampler, uv + sun_dir_t * (SUN_STEP / max(tile_m, 1.0e-6)), dux, duy).z;
  if (alb.a < in.erode) {
    discard;
  }

  let n_mesh = oct_decode_ds(aux.xy * 2.0 - 1.0);
  let n = t * n_mesh.x + up * n_mesh.y + b * n_mesh.z;

  let shade_g = info.cfg1.w;
  let h = aux.z;
  // Cavity occlusion, BAKED per texel (aux.a) rather than derived from the
  // height here: a term derived from the height is non-linear in it and does
  // not survive mip filtering, so the mat would get brighter with distance as
  // the height chain flattens toward its mean. Stored occlusion mip-averages
  // linearly and keeps the far field at the mat's real mean darkness.
  let ao = mix(1.0 - shade_g, 1.0, aux.w);
  let rise = SUN_STEP * (sun_up / max(length(sun_h), 1.0e-3));
  let shadow = 1.0 - 0.55 * shade_g * smoothstep(0.0, 0.012, (h_sun - h) * span_m - rise);

  // ao/shadow are occlusion, so they belong to the light term: debug=albedo then
  // shows exactly the baked capture.
  var color = light_surface(alb.rgb * ao * shadow, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
