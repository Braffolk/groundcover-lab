#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/terrain.wgsl"

// Crossed-card plant proxies, drawn from a fill.wgsl instance list, in two
// modes sharing all geometry math:
//  - NEAR (vs_near/fs_near): every frame, direct clusters only, current
//    camera, live per-plant wind, fog. Cost bounded by the direct radius.
//  - REFRESH (vs_refresh/fs_refresh): the amortized path — renders one
//    cluster's plants into its 176x176 cache slot with the slot's frozen
//    frustum, canonical pose (NO wind: sway is applied analytically at
//    composite time so cached content never ages), lit but unfogged, and
//    writes packed 16-bit slot depth for later re-projection. Distant levels
//    collapse to one camera-facing card with a minimum-footprint clamp so
//    sub-texel plants stay visible instead of alpha-testing away.
//
// Both paths shade identically: baked plant-local normal -> rotated by the
// CARD's yaw -> flipped toward the viewer -> light_surface() ONCE, with a
// grounding occlusion factor folded into the light term. The composite pass
// never lights again; it only re-projects and fogs what the cache holds.
//
// CARPET entries (stand_table[i].carpet_div > 0, i.e. the bog Sphagnum) take a
// third shape in both paths: ONE ground-parallel quad covering exactly the
// species' periodic tile, conformed to the terrain per vertex, textured with
// the tile's own sub-rectangle of the TOP capture. A 7cm-tall, 18cm-wide mat
// has no silhouette for a camera-facing card to show, and upright cards for it
// slice through the ground and through each other; ground-parallel geometry is
// the only honest shape. The crossed side cards and the far single card are
// skipped entirely for those entries.

struct CardsU {
  vp: mat4x4f,      // refresh frustum view-proj (identity/unused in near mode)
  cam_pos: vec4f,   // xyz = refresh cam pos, w = mode: 0 near, 1 refresh, 2 refresh-far
  // x = radians per target pixel, y = min card px, z = instance base,
  // w = base shade (grounding occlusion depth at the plant root)
  proj_info: vec4f,
}

struct PlantEntry {
  center: vec4f, // xyz = mesh bbox center (plant-local), w = species atlas layer
  dims: vec4f,   // sideHalfW, sideHalfH, topHalfW, topHalfH (m)
  yrange: vec4f, // x = yMin, y = yMax, z,w unused
}

@group(1) @binding(0) var<uniform> cards: CardsU;
@group(1) @binding(1) var<storage, read> instances: array<vec4f>;
@group(1) @binding(2) var<uniform> plant_table: array<PlantEntry, 8>;
@group(1) @binding(3) var card_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var card_normal: texture_2d_array<f32>;
@group(1) @binding(5) var card_sampler: sampler;
/// Anisotropic, for carpet tiles only. A ground-parallel quad at a grazing
/// angle is minified far harder along the view direction than across it, and a
/// single trilinear lod has to pick one: the sharp one aliases, the safe one
/// erased ALL texel detail — a 0.18m tile 2m from the camera landed on mip 2.5,
/// which averages the mat's per-facet normals into a flat up-vector and its
/// albedo into flat olive. Sampling with real gradients lets the hardware keep
/// the cross-axis sharpness, which is where the moss's intricacy lives.
@group(1) @binding(6) var carpet_sampler: sampler;

const QUAD_U = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const QUAD_V = array<f32, 6>(1.0, 1.0, 0.0, 1.0, 0.0, 0.0);

/// Alpha reference for carpet tiles, INSTEAD of the grass ramp. A mat is a
/// closed surface and must stay a solid depth-writing occluder: the top tile
/// is 92-97% covered at mip 0 (alpha is essentially binary there, so this
/// threshold changes nothing up close), but the mip chain pulls partly-covered
/// texels down and at the grass reference whole distant tiles fail the test
/// and punch tile-shaped holes in the carpet.
const CARPET_ALPHA_REF: f32 = 0.06;

/// Where the periodic tile sits inside a carpet species' rebuilt top capture
/// tile — must match CARPET_BORDER / CARPET_UV_SPAN in bake.ts (32 texels of
/// wrapped border on each side of a 256px tile).
const CARPET_UV0: f32 = 0.125;
const CARPET_UV_SPAN: f32 = 0.75;

fn rot_yaw(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

/// Albedo out of a card-atlas tap. `a` is COVERAGE, and it must not be divided
/// out here — `buildMips` in bake.ts already normalises by it:
///
///     rgb = sum(rgb_i * a_i) / sum(a_i)      a = sum(a_i) / 4
///
/// i.e. the stored rgb is the coverage-weighted mean colour of the *covered*
/// texels, un-premultiplied, at every mip level. Dividing by `a` again inflated
/// the albedo by 1/coverage, and coverage falls monotonically as the mip chain
/// deepens — so the field got brighter the further away it was, up to 2.42x at
/// mip 5 with 87% of texels clipping to white. Measured mean luma of the stored
/// rgb across mips 0..5 is 0.399 / 0.399 / 0.404 / 0.399 / 0.395 / 0.377: flat
/// by construction, which is exactly why no division belongs here.
///
/// (The CACHE texture in composite.wgsl is a different story and does need its
/// divide: refresh writes a = 1 into a zero-cleared target, so a bilinear tap at
/// a slot silhouette is genuinely premultiplied against black.)
fn card_albedo_of(tap: vec4f) -> vec3f {
  return tap.rgb;
}

/// The bake stores a plant-local unit normal as rgb = n * 0.5 + 0.5, already
/// flipped toward its capture axis, so mip levels stay meaningful: a mip texel
/// holds the aggregate normal of the blades under it, which may legitimately be
/// SHORT (opposing tilts partly cancelling) — that is information, not error, so
/// it is normalised rather than rejected. The only unusable case is an exact
/// cancellation to zero, which falls back to the capture axis (+Z plant-local,
/// i.e. the direction the card faces once rot_yaw is applied) — never +Y, which
/// is precisely the degenerate normal the v3 atlas produced everywhere.
fn decode_normal(t: vec3f) -> vec3f {
  let v = t * 2.0 - 1.0;
  if (dot(v, v) < 1.0e-8) {
    return vec3f(0.0, 0.0, 1.0);
  }
  return normalize(v);
}

/// World-space shading normal for a card fragment. `view_pos` is the camera
/// this fragment is being lit for (the live camera for the near ring, the
/// slot's frozen camera for a refresh). Thin foliage is lit from both sides —
/// the same rule 000-ground-truth applies — so a card seen from behind gets the
/// flipped normal instead of going black.
fn shading_normal(packed_n: vec3f, card_yaw: f32, world_pos: vec3f, view_pos: vec3f) -> vec3f {
  var n = rot_yaw(decode_normal(packed_n), card_yaw);
  if (dot(n, view_pos - world_pos) < 0.0) {
    n = -n;
  }
  return n;
}

/// World normal for a CARPET fragment. The top capture flipped every mesh
/// normal into the +Y hemisphere over flat ground, so the baked vector has to
/// be lifted into the local GROUND frame — a mat on a slope must light as a
/// slope, and yawing alone would light it as if it were level. This is
/// plant_basis_from_up(up, yaw) from terrain.wgsl, inlined. No two-sided flip:
/// a mat is a one-sided surface and a capitulum flank turned away from the
/// viewer is genuinely in shade.
/// Grounding occlusion for a carpet fragment. The upright cards fake
/// self-shadowing with a root-to-tip gradient, which a mat has no room for —
/// but a Sphagnum cushion occludes itself the same way, sideways: a capitulum
/// apex sees the whole sky, the crevice wall between two of them barely sees
/// any. The baked normal's tilt away from the tile's up axis is a free proxy
/// for that (mean n.y over the tile is 0.48, so this is real signal, not a
/// constant), and it goes into the LIGHT term so debug=albedo stays the baked
/// colour. Scaled by the same baseShade knob as the upright path.
fn carpet_shade(packed_n: vec3f) -> f32 {
  let ny = clamp(decode_normal(packed_n).y, 0.0, 1.0);
  return mix(1.0 - cards.proj_info.w, 1.0, ny);
}

fn carpet_normal(packed_n: vec3f, yaw: f32, up_in: vec3f) -> vec3f {
  let nm = decode_normal(packed_n);
  let up = normalize(up_in);
  var t = vec3f(cos(yaw), 0.0, -sin(yaw));
  let proj = t - up * dot(up, t);
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  return normalize(t * nm.x + up * nm.y + cross(t, up) * nm.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,           // atlas uv (tile-mapped)
  @location(1) world_pos: vec3f,
  // Yaw that maps the CAPTURE's frame onto this card's frame. The side capture
  // looks along +Z in plant-local space, so a side card facing direction d gets
  // atan2(d.x, d.z): instance yaw for quad 0, +90 deg for quad 1, and the
  // actual camera azimuth for the far single card. Without this the two quads
  // of every cross shaded identically and the far card's normals were unrelated
  // to the geometry it stands in for.
  @location(2) @interpolate(flat) card_yaw: f32,
  @location(3) @interpolate(flat) lod: f32,
  @location(4) @interpolate(flat) layer: i32,
  // Extra alpha-test threshold: fades the horizontal crown card when seen
  // edge-on (otherwise it smears into dark streaks at eye level).
  @location(5) @interpolate(flat) thr_bias: f32,
  // Grounding occlusion: dark at the plant root, 1 at the tip.
  @location(6) shade: f32,
  // Carpet tiles only: the ground normal under this vertex, so the baked
  // normal can be lifted into the local GROUND frame instead of a flat one.
  @location(7) up_ws: vec3f,
  @location(8) @interpolate(flat) carpet: u32,
}

fn crown_fade(vi: u32, crown_pos: vec3f, cam: vec3f) -> f32 {
  if (vi / 6u != 2u) { return 0.0; }
  let vdir = normalize(cam - crown_pos);
  // Fully faded below 25 deg elevation, fully present above ~33 deg: the
  // spidery top capture only ever reads well from steep angles.
  return 1.0 - clamp((abs(vdir.y) - 0.42) / 0.13, 0.0, 1.0);
}

struct CardVertex {
  world: vec3f,
  uv: vec2f,
  degenerate: bool,
  hf: f32,       // normalized height up the plant (0 root, 1 tip) for wind
  card_yaw: f32, // capture frame -> card frame rotation (see VOut.card_yaw)
  shade: f32,    // grounding occlusion at this vertex
  up_ws: vec3f,  // ground normal under this vertex (carpet tiles)
  carpet: bool,
}

fn card_vertex(vi: u32, ii: u32, mode: u32) -> CardVertex {
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let base = i0.xyz;
  let yaw = i0.w;
  let scale = i1.x;
  let entry = u32(i1.z);
  let pe = plant_table[entry];

  let quad = vi / 6u;
  let corner = vi % 6u;
  let u = QUAD_U[corner];
  let v = QUAD_V[corner];

  let base_shade = cards.proj_info.w;

  var out: CardVertex;
  out.degenerate = mode == 2u && quad > 0u;
  out.uv = vec2f(0.0);
  out.hf = 0.0;
  out.card_yaw = yaw;
  out.shade = 1.0;
  out.up_ws = vec3f(0.0, 1.0, 0.0);
  out.carpet = false;

  let tile_m = stand_table[entry].footprint_m;
  if (stand_table[entry].carpet_div > 0.0) {
    // --- carpet tile: one ground-parallel quad, conformed per vertex --------
    out.carpet = true;
    out.degenerate = quad > 0u; // no side cards, no crown card, no far card
    if (out.degenerate) { return out; }

    // Width from the species' periodic FOOTPRINT, never from its height:
    // footprint_m * scale is exactly the carpet's grid step, so tiles abut
    // instead of overlapping (this mesh is 0.245m of geometry inside a 0.18m
    // period, and drawing the whole capture would overscale every tile 1.36x).
    // The scatter's 90-degree yaw and constant scale are used as given — the
    // two things that keep a mat a mat rather than confetti.
    let corner_m = vec2f(2.0 * u - 1.0, 2.0 * v - 1.0) * (tile_m * 0.5);
    let off_w = rot_yaw(vec3f(corner_m.x, 0.0, corner_m.y), yaw) * scale;
    let xz = base.xz + off_w.xz;

    // Ladder rung 3: the ground under EVERY vertex. Neighbouring tiles share
    // corner positions, so this is the only rung that keeps the mat C0
    // continuous — a per-tile plane fit cracks at every tile edge. One
    // terrain_sample gives the height and (nx, nz) in the same four taps, so
    // the shading basis costs nothing extra.
    let g = terrain_sample(xz);
    out.up_ws = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
    // yrange.z is the carpet plane height in metres, SHARED by every carpet
    // entry of the stand (main.ts): a per-species height would step 1.6cm at
    // every zone boundary and show bare peat through the step.
    out.world = vec3f(xz.x, g.x + pe.yrange.z, xz.y);

    // A carpet species' TOP tile was rewritten at load time to be the periodic
    // tile square itself, inset by a wrapped border (rebuildCarpetTopTile in
    // bake.ts), so the quad maps straight onto that inset. Compared with
    // sampling the raw capture this both raises texel density (192 texels over
    // 0.18m instead of 256 over 0.245m) and keeps every mip level honest.
    out.uv = vec2f(
      0.5 + (CARPET_UV0 + (u * CARPET_UV_SPAN)) * 0.5,
      CARPET_UV0 + v * CARPET_UV_SPAN,
    );
    return out;
  }

  let center_w = base + rot_yaw(pe.center.xyz, yaw) * scale;

  if (quad == 2u) {
    // Horizontal card at crown height, textured with the top capture. Its
    // normals were flipped toward +Y at bake time, and rot_yaw leaves Y alone,
    // so the instance yaw is already the right card frame.
    let local = vec3f((2.0 * u - 1.0) * pe.dims.z, pe.center.y, -(1.0 - 2.0 * v) * pe.dims.w);
    out.world = base + rot_yaw(local + vec3f(pe.center.x, 0.0, pe.center.z), yaw) * scale;
    out.uv = vec2f(0.5 + u * 0.5, v);
    out.hf = clamp((pe.center.y - pe.yrange.x) / max(pe.yrange.y - pe.yrange.x, 1e-4), 0.0, 1.0);
    // Seen from above you look at the tips, so the crown barely occludes.
    out.shade = 1.0 - base_shade * 0.25;
  } else {
    var hw = pe.dims.x;
    let hh = pe.dims.y;
    var right: vec3f;
    if (mode == 2u) {
      // Far single card: face the (refresh) camera, clamp the footprint to a
      // minimum pixel size so distant plants do not alpha-test into dust.
      let to_cam = cards.cam_pos.xyz - center_w;
      right = normalize(vec3f(-to_cam.z, 0.0, to_cam.x));
      let dist = max(length(to_cam), 0.01);
      let px = (2.0 * hw * scale) / (dist * cards.proj_info.x);
      hw *= max(1.0, cards.proj_info.y / max(px, 1e-4));
      // The card faces the camera, so the capture axis maps onto that azimuth.
      out.card_yaw = atan2(to_cam.x, to_cam.z);
    } else {
      right = rot_yaw(vec3f(1.0, 0.0, 0.0), yaw + f32(quad) * 1.5707963);
      out.card_yaw = yaw + f32(quad) * 1.5707963;
    }
    let ly = pe.center.y + (1.0 - 2.0 * v) * hh;
    out.world = base
      + rot_yaw(vec3f(pe.center.x, 0.0, pe.center.z), yaw) * scale
      + right * ((2.0 * u - 1.0) * hw * scale)
      + vec3f(0.0, ly * scale, 0.0);
    out.uv = vec2f(u * 0.5, v);
    out.hf = clamp((ly - pe.yrange.x) / max(pe.yrange.y - pe.yrange.x, 1e-4), 0.0, 1.0);
    // Deep groundcover self-occludes: almost no sky reaches the root, and the
    // shared light model has no shadowing of its own. Without this the base of
    // every card is as bright as its tip and nothing reads as being in shadow.
    out.shade = mix(1.0 - base_shade, 1.0, out.hf);
  }
  return out;
}

fn card_lod(world: vec3f, cam: vec3f, hh: f32, scale: f32) -> f32 {
  let dist = max(distance(world, cam), 0.05);
  let px = (2.0 * hh * scale) / (dist * cards.proj_info.x);
  return clamp(log2(256.0 / max(px, 0.5)), 0.0, 5.0);
}

/// Albedo + normal tap for a fragment. Upright cards keep the explicit,
/// distance-derived lod (their card is camera-facing, so screen-space
/// derivatives buy nothing and the far refresh path renders sub-pixel cards
/// where derivatives are meaningless). Carpet tiles sample with real
/// gradients through the anisotropic sampler instead — see carpet_sampler.
struct CardTap {
  albedo: vec4f,
  normal: vec4f,
}

fn card_tap(uv: vec2f, layer: i32, lod: f32, carpet: bool, ddx: vec2f, ddy: vec2f) -> CardTap {
  var t: CardTap;
  if (carpet) {
    t.albedo = textureSampleGrad(card_albedo, carpet_sampler, uv, layer, ddx, ddy);
    t.normal = textureSampleGrad(card_normal, carpet_sampler, uv, layer, ddx, ddy);
  } else {
    t.albedo = textureSampleLevel(card_albedo, card_sampler, uv, layer, lod);
    t.normal = textureSampleLevel(card_normal, card_sampler, uv, layer, lod);
  }
  return t;
}

// ---------------------------------------------------------------------------

@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) raw_ii: u32) -> VOut {
  let ii = u32(cards.proj_info.z) + raw_ii;
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let cv = card_vertex(vi, ii, 0u);
  let entry = u32(i1.z);
  let sway = stand_table[entry].sway;
  let world = cv.world + wind_sway(i0.xyz, frame.time, sway * cv.hf, i1.y);

  var o: VOut;
  if (cv.degenerate) {
    o.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped away (carpet: quads 1 and 2)
  } else {
    o.pos = frame.view_proj * vec4f(world, 1.0);
  }
  o.uv = cv.uv;
  o.world_pos = world;
  o.card_yaw = cv.card_yaw;
  // Carpet tiles ignore this: they sample by gradient (see card_tap).
  o.lod = card_lod(i0.xyz, frame.camera_pos, plant_table[entry].dims.y, i1.x);
  o.layer = i32(plant_table[entry].center.w);
  o.thr_bias = crown_fade(vi, i0.xyz + vec3f(0.0, plant_table[entry].center.y * i1.x, 0.0), frame.camera_pos);
  o.shade = cv.shade;
  o.up_ws = cv.up_ws;
  o.carpet = select(0u, 1u, cv.carpet);
  return o;
}

@fragment
fn fs_near(i: VOut) -> @location(0) vec4f {
  // Derivatives must be taken in uniform control flow, so they are computed
  // for every fragment and only USED by the carpet branch inside card_tap.
  let ddx = dpdx(i.uv);
  let ddy = dpdy(i.uv);
  let tap = card_tap(i.uv, i.layer, i.lod, i.carpet == 1u, ddx, ddy);
  let tex = tap.albedo;
  let dcam = distance(i.world_pos, frame.camera_pos);
  // Base threshold relaxes with lod (mips dilute coverage), rises to 1 as the
  // camera enters the plant so it fades away instead of clipping. A carpet
  // gets neither: a mat must not dissolve with distance, and eroding the mat
  // you are standing on would open a hole under your feet.
  var thr = CARPET_ALPHA_REF;
  if (i.carpet == 0u) {
    thr = mix(0.5, 0.28, i.lod / 5.0) + smoothstep(0.9, 0.25, dcam) + i.thr_bias;
  }
  if (tex.a < thr) { discard; }
  // Per-fragment normal: baked plant-local unit normal, rotated into the world
  // by THIS card's yaw and flipped toward the viewer (thin foliage is
  // two-sided). See shading_normal().
  let ne = tap.normal;
  let albedo = card_albedo_of(tex);
  var n = shading_normal(ne.rgb, i.card_yaw, i.world_pos, frame.camera_pos);
  var shade = i.shade;
  if (i.carpet == 1u) {
    n = carpet_normal(ne.rgb, i.card_yaw, i.up_ws);
    shade = carpet_shade(ne.rgb);
  }
  // The grounding gradient is occlusion, so it multiplies into the LIGHT term,
  // not the albedo: debug=albedo stays the baked atlas colour exactly as
  // captured, and debug=lighting shows sun + ambient x grounding.
  var color = light_surface(albedo * shade, n, i.world_pos);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, i.world_pos);
  }
  return vec4f(debug_shade(color, albedo, n, tex.a, i.world_pos), 1.0);
}

// ---------------------------------------------------------------------------

@vertex
fn vs_refresh(@builtin(vertex_index) vi: u32, @builtin(instance_index) raw_ii: u32) -> VOut {
  let ii = u32(cards.proj_info.z) + raw_ii;
  let i0 = instances[ii * 2u];
  let i1 = instances[ii * 2u + 1u];
  let mode = u32(cards.cam_pos.w + 0.5);
  let cv = card_vertex(vi, ii, mode);

  var o: VOut;
  if (cv.degenerate) {
    o.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped away
  } else {
    o.pos = cards.vp * vec4f(cv.world, 1.0);
  }
  let entry = u32(i1.z);
  o.uv = cv.uv;
  o.world_pos = cv.world;
  o.card_yaw = cv.card_yaw;
  o.lod = card_lod(i0.xyz, cards.cam_pos.xyz, plant_table[entry].dims.y, i1.x);
  o.layer = i32(plant_table[entry].center.w);
  o.thr_bias = crown_fade(vi, i0.xyz + vec3f(0.0, plant_table[entry].center.y * i1.x, 0.0), cards.cam_pos.xyz);
  o.shade = cv.shade;
  o.up_ws = cv.up_ws;
  o.carpet = select(0u, 1u, cv.carpet);
  return o;
}

struct RefreshOut {
  @location(0) color: vec4f,
  @location(1) aux: vec4f, // rg = packed 16-bit slot depth
}

@fragment
fn fs_refresh(i: VOut) -> RefreshOut {
  let ddx = dpdx(i.uv);
  let ddy = dpdy(i.uv);
  let tap = card_tap(i.uv, i.layer, i.lod, i.carpet == 1u, ddx, ddy);
  let tex = tap.albedo;
  var thr = CARPET_ALPHA_REF;
  if (i.carpet == 0u) {
    thr = mix(0.5, 0.22, i.lod / 5.0) + i.thr_bias;
  }
  if (tex.a < thr) { discard; }
  let ne = tap.normal;
  let albedo = card_albedo_of(tex);
  // Lit for the SLOT's camera — the only view-dependent term is the two-sided
  // flip, which the parallax invalidation already bounds (a slot is refreshed
  // long before the viewer crosses a card's plane). A carpet has no flip at
  // all, so its cached shading is view-independent and never goes stale.
  var n = shading_normal(ne.rgb, i.card_yaw, i.world_pos, cards.cam_pos.xyz);
  var shade = i.shade;
  if (i.carpet == 1u) {
    n = carpet_normal(ne.rgb, i.card_yaw, i.up_ws);
    shade = carpet_shade(ne.rgb);
  }
  let lit = light_surface(albedo * shade, n, i.world_pos);

  let d16 = floor(clamp(i.pos.z, 0.0, 1.0) * 65535.0 + 0.5);
  let hi = floor(d16 / 256.0);
  let lo = d16 - hi * 256.0;

  var o: RefreshOut;
  // The cache holds SHADED imagery, so the debug view has to be baked in here
  // (main.ts re-reconstructs every slot when the selector changes). Composite
  // overrides the coverage/depth views with live values.
  o.color = vec4f(debug_shade(lit, albedo, n, tex.a, i.world_pos), 1.0);
  o.aux = vec4f(hi / 255.0, lo / 255.0, 0.0, 1.0);
  return o;
}
