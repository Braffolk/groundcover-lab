#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./fin_shared.wgsl"

// Carpet cards — the fin set re-oriented for a species that lies along the
// ground instead of standing off it (stand_table[i].carpet_div > 0).
//
// A Sphagnum tile is 0.18 m across and 0.07-0.09 m tall. Crossed vertical fins
// through it are the classic mat failure: they slice through the ground and
// each other, and edge-on they show a 7 cm line. So a carpet entry draws no
// fins at all. Instead it draws MOSS_LEVELS GROUND-PARALLEL cross-sections
// through the cushion: card k carries the top-down capture of everything above
// height y_k and sits at y_k, so for a ray from above the highest card that is
// opaque at that xz is where the cushion surface actually is. The stack is a
// terraced height field of the real capitulum relief — thickness, silhouette
// and inter-card parallax, which is exactly what a single flat quad cannot do.
//
// Card 0 sits just above the peat and carries the full top view, so the mat is
// a closed, hard-edged, depth-writing occluder (no dither anywhere in here).
// Cards are conformed to the terrain PER VERTEX (rung 3 of the fitting ladder):
// neighbouring tiles share corner positions, so only per-vertex conforming
// keeps the mat C0-continuous — any per-tile plane fit cracks at every seam.
// Width comes from stand_table[i].footprint_m (never height_scale), so a tile
// exactly fills its grid step and the lattice is untouched: no billboarding, no
// extra yaw, no per-tile scale, no distance shrink.

@group(1) @binding(0) var<uniform> fin: FinCfg;
@group(1) @binding(1) var<storage, read> plants: array<FinPlant>;
@group(1) @binding(2) var moss_albedo: texture_2d_array<f32>;
@group(1) @binding(3) var moss_normal: texture_2d<f32>;
@group(1) @binding(4) var moss_samp: sampler;

/// Alpha reference for the closure card. A mat must not dissolve with distance:
/// its level-0 alpha is ~0.8 up close, but the mip chain pulls it toward the
/// tile mean, and at a grass-like reference whole distant tiles fail the test
/// and punch tile-shaped holes down to bare peat. Low reference = MORE solid
/// coverage, and the genuinely empty texels (the gaps down to the peat) still
/// open. The cross-section cards above it use the `mossRelief` param instead —
/// their silhouette should stay honest.
const CLOSURE_REF: f32 = 0.06;

/**
 * Height of the ground as the harness actually DRAWS it.
 *
 * The base pass renders the terrain as a TERRAIN_QUADS^2 grid over
 * frame.terrain_size, evaluating terrain_height() only at the quad corners and
 * interpolating linearly across each triangle. That surface is NOT
 * terrain_height(xz): the heightmap is finer than the grid (0.5 m texels vs 1 m
 * quads), so inside a quad the two disagree by a mean 1.1 cm and by up to 7.2 cm
 * over the bog region — and a Sphagnum tile is only 7-9 cm TALL. Conforming the
 * mat to terrain_height() therefore buried it: whole patches of the carpet
 * vanished inside the terrain they were lying on, leaving faceted bare ground
 * (the terrain's own triangles) showing through. Tall grass never notices this;
 * a 7 cm mat cannot afford to ignore it.
 *
 * TERRAIN_QUADS mirrors QUADS in src/scene/terrain-draw.wgsl. It would be
 * better as a harness primitive — see NOTES.md, interface feedback.
 */
const TERRAIN_QUADS: f32 = 256.0;

fn drawn_ground(xz: vec2f) -> f32 {
  let step = frame.terrain_size / TERRAIN_QUADS;
  let g = (xz + frame.terrain_size * 0.5) / step;
  let b = floor(g);
  let f = g - b;
  let o = b * step - frame.terrain_size * 0.5;
  let hb = terrain_height(o + vec2f(0.0, step));
  let hc = terrain_height(o + vec2f(step, 0.0));
  // The grid's two triangles are (0,0),(0,1),(1,0) and (1,0),(0,1),(1,1).
  if (f.x + f.y <= 1.0) {
    let ha = terrain_height(o);
    return ha + (hc - ha) * f.x + (hb - ha) * f.y;
  }
  let hd = terrain_height(o + vec2f(step, step));
  return hd + (hb - hd) * (1.0 - f.x) + (hc - hd) * (1.0 - f.y);
}

struct MOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
  @location(2) up_ws: vec3f,
  @location(3) @interpolate(flat) layer: u32,
  @location(4) @interpolate(flat) yaw: f32,
  @location(5) @interpolate(flat) a_ref: f32,
}

fn degenerate() -> MOut {
  var out: MOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  return out;
}

/// Cards in the stack at a given viewing distance. A 0.18 m tile subtends ~55px
/// at 4 m and ~25px at 9 m, where a 7 mm terrace step is well under a pixel —
/// so the stack collapses with distance instead of paying 6x the vertex work
/// for sub-pixel relief. This is an LOD on HOW MANY cards, never on the tile's
/// size, spacing or rotation, so the lattice is identical at every distance.
fn moss_card_count(d: f32) -> u32 {
  if (d < 1.5) { return 6u; }
  if (d < 4.0) { return 3u; }
  if (d < MOSS_NEAR_M) { return 2u; }
  return 1u;
}

/// Which cross-section card `q` of `n` maps to. Card 0 (the closure card) is
/// always present; the rest spread evenly over the relief band.
fn moss_layer_of(q: u32, n: u32) -> u32 {
  if (n >= 6u) { return q; }
  if (n <= 1u) { return 0u; }
  // Two cards: the second one is level 3 (0.72 of tile height), which is where
  // the mean capitulum apex is — the same height the one-card LOD lerps its
  // closure card to, so the 1 <-> 2 transition moves the surface by ~1 mm.
  if (n == 2u) { return select(0u, 3u, q == 1u); }
  return u32(round(f32(q) * 4.0 / f32(n - 1u)));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let plant = plants[ii];
  let base = vec3f(plant.px, plant.py, plant.pz);
  let entry = stand_table[u32(fin.entry_index)];

  // Distance is measured from the TILE CENTRE (instance-uniform), so all six
  // vertices of a card agree on the LOD and neighbouring tiles cannot disagree
  // about how many cards they have mid-quad.
  let d_center = length(frame.camera_pos.xz - base.xz);
  let cards = min(moss_card_count(d_center), u32(fin.layer_count));
  let quad = vi / 6u;
  if (quad >= cards) { return degenerate(); }
  let layer = moss_layer_of(quad, cards);

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];

  // footprint_m * scale IS the carpet's grid step, so tiles abut exactly.
  let tile_m = entry.footprint_m;
  let mesh_off = c * tile_m * 0.5;
  let off = rot_y(vec3f(mesh_off.x, 0.0, mesh_off.y) * plant.scale, plant.yaw);
  let xz = base.xz + off.xz;

  // Rung 3: the ground under THIS vertex, so the mat follows every bump and
  // neighbouring tiles (which share corner positions) stay C0-continuous.
  // terrain_sample's (nx, nz) is the SMOOTH ground normal — the same one the
  // terrain fragment shader uses — so mat and ground light identically, even
  // though the height comes from the faceted surface the base pass draws.
  let g = terrain_sample(xz);
  let align = clamp(entry.slope_align, 0.0, 1.0);
  let up_g = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  let ground_y = drawn_ground(xz);

  var levels = array<f32, 6>(fin.lvl_a.x, fin.lvl_a.y, fin.lvl_a.z, fin.lvl_a.w, fin.lvl_b.x, fin.lvl_b.y);
  var y_local = levels[layer];
  if (layer == 0u) {
    // With the stack collapsed to one card, the best flat stand-in for the
    // cushion surface is its mean capitulum apex, not the cross-section height
    // 4 cm below it. Lerp between them by distance — a pure function of xz, so
    // tiles that share a corner still share its height exactly (no ledge at the
    // LOD boundary, and by construction the far height already equals the card
    // the 2-card LOD adds, so that transition is invisible too).
    let d_v = length(frame.camera_pos.xz - xz);
    y_local = mix(y_local, fin.lvl_b.z, smoothstep(4.0, 10.0, d_v));
  }

  var out: MOut;
  let world = vec3f(xz.x, mix(base.y, ground_y, align) + y_local * plant.scale, xz.y);
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  // The capture was framed to the tile's own square, sampled with repeat
  // addressing — the tile is periodic, so wrapping is the correct filter at its
  // boundary and no inset is needed.
  out.uv = 0.5 + mesh_off / tile_m;
  out.up_ws = mix(vec3f(0.0, 1.0, 0.0), up_g, align);
  out.layer = layer;
  out.yaw = plant.yaw;
  // Region edge erodes through the alpha reference (hard edges dissolve first),
  // never through a per-vertex clip: a card whose vertices disagreed about
  // being emitted rasterizes as a metre-long sliver. There is NO camera-inside
  // fade — a mat you are standing on must not open a hole under you.
  let fade = carpet_fade(d_center, fin.max_dist);
  let ref0 = select(fin.moss_ref, CLOSURE_REF, layer == 0u);
  out.a_ref = mix(1.001, ref0, fade);
  return out;
}

/// `cardTint`: one hue per cross-section, so the stack's terracing is visible.
fn layer_tint(l: u32) -> vec3f {
  return 0.35 + 0.45 * cos(6.2831853 * (f32(l) * 0.1667 + vec3f(0.0, 0.33, 0.67)));
}

@fragment
fn fs_main(in: MOut) -> @location(0) vec4f {
  // Both taps happen before the alpha test: derivatives (and therefore the
  // implicit mip level) are undefined after a discard. The cross-section layer
  // decides COVERAGE and colour; the shading comes from the one macro-surface
  // map, because wherever this card is opaque the surface under it is the same
  // one cross-section 0 saw (see carpet.ts).
  let alb = textureSample(moss_albedo, moss_samp, in.uv, in.layer);
  let nrm = textureSample(moss_normal, moss_samp, in.uv);
  if (alb.a < in.a_ref) { discard; }

  let unmul = 1.0 / max(nrm.a, 1.0e-3);
  // rg = the cushion's macro-surface gradient, b = baked crevice occlusion.
  // n.y is recovered rather than stored: a height-field normal never points
  // down, so two channels are exact and (unlike an octahedral pair) they
  // survive mip filtering. The mesh's own leaf normals are isotropic and
  // average to noise — see carpet.ts.
  let g2 = (nrm.rg * unmul) * 2.0 - 1.0;
  let n_local = vec3f(g2.x, sqrt(max(1.0 - dot(g2, g2), 1.0e-4)), g2.y);
  let occl = clamp(nrm.b * unmul, 0.0, 1.0);
  var albedo = alb.rgb / max(alb.a, 1.0e-3);
  if (fin.card_tint > 0.5) { albedo = layer_tint(in.layer); }
  // The capture is in the mesh frame over flat ground, so the baked normal has
  // to be lifted into the tile's own ground frame: a mat on a slope must light
  // as a slope, not as flat ground that happens to be tilted.
  let basis = plant_basis_from_up(in.up_ws, in.yaw);
  let n_world = normalize(basis * n_local);
  // NO two-sided flip here: this is an outward macro-surface normal, not a
  // thin-leaf normal, and flipping it toward the camera at grazing angles would
  // light the mat from the wrong side.

  // Occlusion multiplies the LIGHT, not the stored colour, so `albedo` stays the
  // raw baked colour and `lighting` shows the crevice shading it belongs to.
  var color = light_surface(albedo, n_world, in.world) * occl;
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n_world, alb.a, in.world), 1.0);
}
