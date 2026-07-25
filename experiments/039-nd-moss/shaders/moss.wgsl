#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// The Uncharted 4 moss material (Maximov/Ganev, "Technical Art of Uncharted 4",
// slides 42-58 + 29-41), applied to a periodic Sphagnum carpet.
//
// What the talk actually prescribes is a SURFACE MATERIAL, not a geometry
// method: Colour + Normal + Heightmap + a very dark AO map, shaded with
// parallax + parallax shadows, a tinted wrapped diffuse ("light wrap", which
// deliberately ignores the normal map), a microfiber "Fuzz BRDF" for the
// subpixel regime, AO Fresnel (occlude the cracks at glancing angles, because
// the geometry that should occlude them does not exist), sun micro shadowing
// from the AO aperture, and AO saturation (bounce colour back into the dark).
// All of that is here, evaluated in O(1) per fragment: 2 height taps + 1 normal
// tap + 1 optional shadow tap near, 2 taps far.
//
// The one thing the paper does NOT solve is thickness — its moss sits on real
// game geometry that owns the silhouette, while a Sphagnum cushion IS the
// geometry. So the carpet has three distance bands:
//   vs_patch  near: the tile is a div x div displaced patch driven by the baked
//             heightmap, so the cushion has real geometric relief, a real
//             silhouette and a real depth profile. The patch's BOUNDARY ring is
//             pinned to the mean cushion plane: the heightfield of a tile
//             rotated by a quarter turn does not agree with its neighbour's
//             along their shared edge (the field is only periodic under
//             translation), so pinning the boundary is what keeps the lattice
//             crack-free under the stand's 90-degree rotations.
//   vs_quad   mid: one flat tile-sized quad; parallax does the relief.
//   vs_quad   far: one quad per 2x2 phase-locked block (merged == 1), which is
//             the same lattice at the same spacing with the same per-tile
//             quarter turns (four rotation codes ride in the instance bits) —
//             only the terrain fit and the quad count are coarser.
//
// Tiles are axis-aligned in world; a square periodic tile's quarter turn is a
// rotation of its TEXTURE, not of its geometry.

struct Inst {
  pos: vec3f,
  bits: u32, // rot0 | rot1<<2 | rot2<<4 | rot3<<6 | merged<<8
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<Inst>;
@group(1) @binding(2) var moss_a: texture_2d<f32>; // rgb albedo, a height
@group(1) @binding(3) var moss_b: texture_2d<f32>; // rgb normal, a cavity AO
@group(1) @binding(4) var moss_sampler: sampler;

const FLAG_MICRO: u32 = 1u;
const FLAG_PSHADOW: u32 = 2u;
const FLAG_REFINE: u32 = 4u;
/// Offset limiting: the smallest |view.y| the parallax divide is allowed to see.
const PARALLAX_LIMIT: f32 = 0.30;
/// Vertical rise (fraction of the capture range) of the one-tap sun shadow ray.
const SHADOW_RISE: f32 = 0.18;

/// k quarter turns, CCW.
fn rot2(v: vec2f, k: u32) -> vec2f {
  let m = k & 3u;
  if (m == 0u) { return v; }
  if (m == 1u) { return vec2f(-v.y, v.x); }
  if (m == 2u) { return -v; }
  return vec2f(v.y, -v.x);
}

/// Tile-local position (in [0,1] of the tile) -> texture uv for a tile placed
/// with `k` quarter turns. The inverse rotation, because the lookup goes from
/// world back into the mesh frame.
fn tile_uv(f: vec2f, k: u32) -> vec2f {
  return rot2(f - vec2f(0.5), 4u - k) + vec2f(0.5);
}

fn quad_corner(k: u32) -> vec2f {
  var c = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  return c[k];
}

fn ground_up(g: vec3f) -> vec3f {
  return vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
}

/// Height channel at a given mip, WRAPPED. The wrap matters: a tile boundary
/// sits at uv 0 for one neighbour and uv 1 for the other, and only a wrapping
/// fetch makes those two land on the same texel and displace to the same
/// height. A clamp would leave a sub-millimetre crack along every grid line.
fn height_load(uv: vec2f, lvl: i32, dim: vec2f) -> f32 {
  let d = vec2i(dim);
  let t = vec2i(floor(uv * dim));
  return textureLoad(moss_a, ((t % d) + d) % d, lvl).a;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) luv: vec2f,                     // tile-local, [0,1] or [0,2]
  @location(1) world: vec3f,
  @location(2) up: vec3f,                      // ground normal (the mat's macro normal)
  @location(3) @interpolate(flat) bits: u32,
  @location(4) ground_y: f32,                  // terrain height under the fragment
}

// --- mid / far: one flat quad per tile, or per 2x2 phase-locked block --------
@vertex
fn vs_quad(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let reps = f32(1u + ((inst.bits >> 8u) & 1u));
  let c = quad_corner(vi);
  let xz = inst.pos.xz + (c - vec2f(0.5)) * (info.tile_span * reps);
  // Rung 3 of the terrain ladder: the ground under every vertex. Neighbouring
  // quads share corner positions, so the mat stays C0-continuous; any rung that
  // fits one plane per tile cracks at every tile boundary instead.
  let g = terrain_sample(xz);

  var out: VOut;
  out.world = vec3f(xz.x, g.x + info.plane_h, xz.y);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  out.luv = c * reps;
  out.up = ground_up(g);
  out.bits = inst.bits;
  out.ground_y = g.x;
  return out;
}

// --- near: displaced patch --------------------------------------------------
@vertex
fn vs_patch(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let div = u32(info.patch_div);
  let cell = vi / 6u;
  let c = (vec2f(f32(cell % div), f32(cell / div)) + quad_corner(vi % 6u)) / f32(div);
  let xz = inst.pos.xz + (c - vec2f(0.5)) * info.tile_span;
  let g = terrain_sample(xz);

  // Band-limited cushion height at this vertex: a coarse mip of the baked
  // heightmap, so the displacement carries only what the patch can resolve.
  let rot = inst.bits & 3u;
  let lvl = i32(info.patch_mip);
  let dim = vec2f(textureDimensions(moss_a, lvl));
  let h = height_load(tile_uv(c, rot), lvl, dim);

  // The tile's boundary needs a height that every neighbour agrees on whatever
  // quarter turn it drew. Pinning it to the mean plane does that but leaves a
  // flat crease along every grid line — the "square block with a flat cutoff"
  // problem. Instead use the ROTATION-SYMMETRISED field: the mean of the
  // heightmap over all four quarter turns. That is invariant under the tile's
  // own rotation, and (because the tile is periodic and the fetch wraps) two
  // neighbours evaluate bit-identical values at a shared boundary point — so
  // the patch edge follows real relief and still joins seamlessly.
  let f = c - vec2f(0.5);
  let h_sym =
    0.25 * (height_load(rot2(f, 0u) + vec2f(0.5), lvl, dim) +
            height_load(rot2(f, 1u) + vec2f(0.5), lvl, dim) +
            height_load(rot2(f, 2u) + vec2f(0.5), lvl, dim) +
            height_load(rot2(f, 3u) + vec2f(0.5), lvl, dim));
  let edge = min(min(c.x, 1.0 - c.x), min(c.y, 1.0 - c.y));
  let w = smoothstep(0.0, 1.6 / f32(div), edge) * info.displace;
  let surf = info.base_y + mix(h_sym, h, w) * info.relief_h;

  var out: VOut;
  out.world = vec3f(xz.x, g.x + surf, xz.y);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  out.luv = c;
  out.up = ground_up(g);
  out.bits = inst.bits;
  out.ground_y = g.x;
  return out;
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

struct Surface {
  uv: vec2f,
  rot: u32,
  albedo: vec3f,
  height: f32,
}

/// Resolve which sub-tile of the instance this fragment is in, and that
/// sub-tile's quarter turn (packed in the instance bits, so a merged quad is
/// pixel-equivalent to four separate tiles).
fn sub_tile(in: VOut) -> Surface {
  let merged = (in.bits >> 8u) & 1u;
  let sub = min(vec2u(floor(max(in.luv, vec2f(0.0)))), vec2u(merged));
  var s: Surface;
  s.rot = (in.bits >> (2u * (sub.y * 2u + sub.x))) & 3u;
  s.uv = tile_uv(in.luv - vec2f(sub), s.rot);
  return s;
}

/// World y of the cushion surface for a stored height value.
fn surface_y(in: VOut, h: f32) -> f32 {
  return in.ground_y + info.base_y + h * info.relief_h;
}

/// One offset-limited parallax step (Kaneko/Welsh), optionally refined once by
/// re-solving at the new height. NOT a march: two dependent taps, fixed cost.
fn parallax_uv(in: VOut, base_uv: vec2f, rot: u32, h: f32) -> vec2f {
  let v = normalize(frame.camera_pos - in.world);
  let dh = in.world.y - surface_y(in, h);
  // Horizontal travel of the view ray to reach that depth, with the divide
  // capped (classic offset limiting) AND the result bounded by half the relief.
  // Without the bound a deep gap (dh up to the full 9cm) at a grazing angle
  // slides the lookup most of a tile away, and since the tile is periodic the
  // lookup lands in the NEXT copy of the same moss — which shows up as
  // concentric rings of phase-aligned repeats rather than as parallax.
  var travel = -v.xz * (dh / max(v.y, PARALLAX_LIMIT)) * info.parallax;
  let limit = info.relief_h * 0.5;
  let len = length(travel);
  travel *= select(1.0, limit / len, len > limit);
  return base_uv + rot2(travel / info.tile_span, 4u - rot);
}

struct Shaded {
  color: vec3f,
  albedo: vec3f,
  normal: vec3f,
}

/// The paper's fuzzy-moss stack. `sun_shadow` is the (optional) parallax
/// self-shadow term; everything else is derived here.
fn shade_moss(in: VOut, s: Surface, nao: vec4f, sun_shadow: f32) -> Shaded {
  let up = normalize(in.up);
  let v = normalize(frame.camera_pos - in.world);
  let sun = frame.sun_dir;
  let flags = u32(info.flags);

  // --- normal: real per-fragment mesh normal, quarter-turned into the tile's
  //     world orientation and then lifted into the local ground frame, so a mat
  //     on a slope lights as a slope.
  let n_mesh = nao.rgb * 2.0 - 1.0;
  let n_rot = rot2(n_mesh.xz, s.rot);
  var t = vec3f(1.0, 0.0, 0.0);
  let proj = t - up * dot(up, t);
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  var n = normalize(t * n_rot.x + up * n_mesh.y + cross(t, up) * n_rot.y);

  // --- unified wetness (slides 63-65). Each Sphagnum state occupies its own
  //     band of the shared wetness field, so the state's own band centre IS the
  //     wetness input; within a tile, water collects in the crevices. Porosity
  //     darkening is the paper's lerp toward the squared base colour, and a
  //     saturated surface has its normal pulled toward flat.
  let wet = clamp(info.wetness * (0.45 + 0.55 * (1.0 - s.height)), 0.0, 1.0);
  var alb = mix(s.albedo, s.albedo * s.albedo, wet * 0.55);
  n = normalize(mix(n, up, wet * 0.3));

  // --- AO, then AO Fresnel (slide 40): the cracks baked into the normal/AO
  //     maps SHOULD be occluded by geometry at a glancing angle, and that
  //     geometry does not exist, so fade the AO toward white there. ADAPTED: the
  //     paper fades all the way (`ao = lerp(1, ao, dot(N,V))`), which is right
  //     for a moss layer on a rock — but a CARPET is seen at a glancing angle
  //     almost everywhere, and fading the AO out there deletes the only thing
  //     that makes the mat read as a cushion (its albedo is nearly uniform, so
  //     all of moss's structure is shading). `ao_fresnel` caps how much of the
  //     AO the glancing angle is allowed to take: 1.0 is the paper exactly.
  var ao = pow(nao.a, max(info.ao_strength, 0.01));
  ao = mix(ao, 1.0, info.ao_fresnel * (1.0 - saturate(dot(up, v))));

  // --- AO saturation (slide 55): "add back a bit of colour where the ambient
  //     occlusion gets dark", simulating bounce light and SSS. It is additive,
  //     tinted by the measured deep-cushion colour and scaled by the ambient
  //     level — which is also what keeps a physically dark cavity AO from
  //     turning the whole cushion into a hole when seen from straight above.
  //     The tint is normalised to luma 1 so the term adds light of the cushion's
  //     hue rather than of its (dark) intensity.
  let base_hue = info.base_color.rgb / max(dot(info.base_color.rgb, vec3f(0.3333)), 1.0e-3);
  let bounce = base_hue * frame.ambient * ((1.0 - ao) * info.ao_sat);

  // --- sun micro shadowing (slide 37, verbatim) --------------------------
  var micro = 1.0;
  if ((flags & FLAG_MICRO) != 0u) {
    let aperture = 2.0 * ao * ao;
    micro = saturate(abs(dot(sun, n)) + aperture - 1.0);
  }

  // --- light wrap (slide 51): broaden the terminator using the GEOMETRY
  //     normal (the paper's wrap ignores the normal map so it fills the cracks),
  //     tinted by the measured deep-cushion colour. Converges to no-op as the
  //     strength goes to 0.
  let ndl_geo = dot(up, sun);
  let wrapped = saturate((ndl_geo + 0.35) / 1.35);
  let extra = max(wrapped - saturate(ndl_geo), 0.0) * info.light_wrap;
  let wrap_light = info.base_color.rgb * extra * frame.sun_color;

  // --- Fuzz BRDF (slide 52): the subpixel microfiber layer, tinted by the
  //     measured capitulum-tip colour ("lighter at the tips than at the base").
  //     This is also what carries the far field, where the relief is subpixel.
  let sheen = pow(1.0 - saturate(dot(up, v)), 4.0) * saturate(ndl_geo * 0.5 + 0.5) * info.fuzz;
  let fuzz_light = info.tip_color.rgb * sheen * frame.sun_color;

  // The shared lighting model is albedo * (sun + ambient * hemi), and occlusion
  // does not act equally on those halves: cavity AO occludes the INDIRECT term,
  // while the sun is occluded by the two shadow terms (the paper multiplies its
  // micro shadow into the sun's shadow, slide 37). So split the shared model
  // apart by subtraction — with ao = micro = shadow = 1 this is exactly
  // light_surface(), and the ambient formula is never duplicated.
  let lit = light_surface(alb, n, in.world);
  let ndl = dot(n, sun) * 0.5 + 0.5;
  let sun_part = alb * frame.sun_color * (ndl * ndl);
  let amb_part = max(lit - sun_part, vec3f(0.0));

  var out: Shaded;
  out.albedo = alb;
  out.normal = n;
  out.color = sun_part * (micro * sun_shadow) + amb_part * ao + alb * (wrap_light + fuzz_light + bounce);
  return out;
}

fn finish(in: VOut, sh: Shaded, coverage: f32) -> vec4f {
  var color = sh.color;
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, sh.albedo, sh.normal, coverage, in.world), 1.0);
}

/// Near + mid: parallax relief, self-shadow, the full stack. 4-5 taps.
@fragment
fn fs_full(in: VOut) -> @location(0) vec4f {
  var s = sub_tile(in);
  let flags = u32(info.flags);

  // Texture gradients come from the UNPERTURBED lookup, and are derived from
  // `luv` rather than from `uv` itself. Both parts matter:
  //  * the parallax offset is a per-fragment function of the sampled height, so
  //    its screen-space derivative is the HEIGHTMAP's gradient — enormous. Left
  //    to pick its own mip, the sampler drops to near the top of the chain at
  //    grazing angles and the whole mat turns into flat olive paint (measured:
  //    it did);
  //  * `uv` also jumps at the sub-tile seams of a merged quad (it is a rotated
  //    fract), while `luv` is continuous, and a rotation maps gradients exactly.
  let grad_x = rot2(dpdx(in.luv), 4u - s.rot);
  let grad_y = rot2(dpdy(in.luv), 4u - s.rot);

  let a0 = textureSampleGrad(moss_a, moss_sampler, s.uv, grad_x, grad_y);
  var uv = parallax_uv(in, s.uv, s.rot, a0.a);
  var a = textureSampleGrad(moss_a, moss_sampler, uv, grad_x, grad_y);
  if ((flags & FLAG_REFINE) != 0u) {
    uv = parallax_uv(in, s.uv, s.rot, a.a);
    a = textureSampleGrad(moss_a, moss_sampler, uv, grad_x, grad_y);
  }
  let nao = textureSampleGrad(moss_b, moss_sampler, uv, grad_x, grad_y);

  // --- parallax shadows (slide 53), one tap: step toward the sun by a fixed
  //     fraction of the relief and ask whether the cushion is higher there than
  //     the ray. Soft, because this represents micro features (slide 37).
  var sun_shadow = 1.0;
  if ((flags & FLAG_PSHADOW) != 0u) {
    let sun = frame.sun_dir;
    let step = sun.xz / max(sun.y, 0.25) * (SHADOW_RISE * info.relief_h / info.tile_span);
    let hs = textureSampleGrad(moss_a, moss_sampler, uv + rot2(step, 4u - s.rot), grad_x, grad_y).a;
    sun_shadow = 1.0 - saturate((hs - a.a - SHADOW_RISE) * 8.0) * 0.75;
  }

  if (a.a < info.gap_ref) {
    discard; // a real gap down to the peat — hard edge, no dither
  }
  s.albedo = a.rgb;
  s.height = a.a;
  return finish(in, shade_moss(in, s, nao, sun_shadow), a.a);
}

/// Far: no parallax, no self-shadow — 2 taps. The paper's own answer to "the
/// details become subpixel more than a few metres away" is the Fuzz BRDF, which
/// this path keeps; only the texture-space relief goes away.
@fragment
fn fs_cheap(in: VOut) -> @location(0) vec4f {
  var s = sub_tile(in);
  // Gradients from the continuous `luv`: `uv` is a rotated fract and jumps at
  // the sub-tile seams of a merged quad, which would blur a one-pixel line.
  let grad_x = rot2(dpdx(in.luv), 4u - s.rot);
  let grad_y = rot2(dpdy(in.luv), 4u - s.rot);
  let a = textureSampleGrad(moss_a, moss_sampler, s.uv, grad_x, grad_y);
  let nao = textureSampleGrad(moss_b, moss_sampler, s.uv, grad_x, grad_y);
  if (a.a < info.gap_ref) {
    discard;
  }
  s.albedo = a.rgb;
  s.height = a.a;
  return finish(in, shade_moss(in, s, nao, 1.0), a.a);
}
