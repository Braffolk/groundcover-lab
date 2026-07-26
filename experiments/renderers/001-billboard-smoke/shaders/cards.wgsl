#include "src/wgsl/wind.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// Billboard cards over baked imagery. Two shapes, chosen per stand entry:
//
// UPRIGHT PLANTS (carpet_div == 0) — 12 vertices per instance:
//   verts 0..5  — cylindrical camera-facing side card. The fragment samples
//                 the baked side view whose azimuth (in the plant's yawed
//                 frame) is nearest the actual viewing azimuth (8 views).
//   verts 6..11 — horizontal top card at top_frac height, textured with the
//                 baked straight-down view; edge-on (invisible) at grazing
//                 angles, it takes over naturally as the camera rises.
//
// CARPET TILES (carpet_div > 0, e.g. Sphagnum) — 6 vertices per instance:
//   ONE ground-parallel quad covering exactly the species' periodic tile
//   (stand_table.footprint_m, optionally uniformly overscaled), textured with
//   the tile's own sub-rectangle of the baked top view and conformed to the
//   ground per vertex. A mat has no meaningful silhouette from the side, so it
//   gets no camera-facing card at all: vertical cards would slice through the
//   ground and through each other. The tile's 90-degree yaw and its constant
//   scale come from the stand and are used AS GIVEN — they are what keeps
//   neighbouring tiles agreeing, i.e. a mat instead of confetti.
//
// Hard alpha-test edges with depth write; near/region fades erode coverage
// through the alpha reference instead of dithering or blending.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var card_albedo: texture_2d<f32>;
@group(1) @binding(3) var card_normal: texture_2d<f32>;
@group(1) @binding(4) var card_sampler: sampler;

const GRID: f32 = 3.0;       // 3x3 tiles in the atlas; tile 8 is the top view
const N_SIDE_VIEWS: f32 = 8.0;
const UV_INSET: f32 = 0.0078125; // 4 texels of a 512 tile

// Height of a carpet tile's ground plane, as a fraction of the baked capture
// box [y0, y1]. 0.74 is where the source meshes put their mean capitulum apex
// (canopy.capitulumApexMeanH / geometry.tile.topH is 0.739-0.754 for all three
// Sphagnum states), i.e. the surface the baked top view actually shows.
const CARPET_TOP: f32 = 0.74;
// Sub-millimetre height bias in FOUR phases over the grid parity. With an
// overscale > 1 a tile overlaps its 4 edge and 4 diagonal neighbours; two
// ground-parallel quads that are nearly coplanar would z-fight across the
// overlap. Four phases give every tile a different height from all 8
// neighbours (only tiles 2 steps apart share a phase, and those never
// overlap), so each overlap resolves to a stable hard edge instead of
// speckle. Purely depth ordering: it does not move a tile in xz, rotate it,
// or change its scale — the lattice is untouched.
const CARPET_BIAS: f32 = 0.0003;
// Alpha reference for carpet tiles, INSTEAD of the params' alphaRef. A mat is
// a closed surface, so its tiles must not dissolve with distance: the atlas
// alpha of a tile averages ~0.8 up close but the mip chain pulls it toward the
// whole top tile's mean, and at the 0.4 grass reference entire distant tiles
// then fail the test and punch holes in the carpet. A low reference keeps the
// mat solid (a depth-writing occluder, per the taste rules) while the genuinely
// empty texels — 16-20% of the tile, the gaps down to the peat — still open.
const CARPET_ALPHA_REF: f32 = 0.06;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn tile_uv(view_index: u32, local: vec2f) -> vec2f {
  let col = f32(view_index % 3u);
  let row = f32(view_index / 3u);
  let clamped = clamp(local, vec2f(UV_INSET), vec2f(1.0 - UV_INSET));
  return (vec2f(col, row) + clamped) / GRID;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) shade: f32,                       // grounding gradient
  @location(5) up_ws: vec3f,                     // ground up for carpet tiles
  @location(6) @interpolate(flat) carpet: u32,   // 1 = ground-parallel mat tile
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];

  let r = info.card_r * scale;
  let h0 = info.card_y0 * scale;
  let h1 = info.card_y1 * scale;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];

  var world: vec3f;
  var uv: vec2f;
  var shade: f32;
  var fade: f32;
  var up_ws = vec3f(0.0, 1.0, 0.0);
  let is_carpet = entry.carpet_div > 0.0;
  let a_ref = select(info.alpha_ref, CARPET_ALPHA_REF, is_carpet);

  if (is_carpet) {
    // --- carpet tile: one ground-parallel quad, conformed per vertex --------
    // Width comes from the species' periodic footprint, NEVER from its height:
    // footprint_m * scale is exactly the carpet's grid step, so tiles abut.
    let c2 = vec2f(c.x, c.y * 2.0 - 1.0);
    let tile_m = entry.footprint_m;
    let half_mesh = tile_m * 0.5 * info.carpet_over;
    let mesh_off = c2 * half_mesh;                    // from the TILE centre
    let off = rot_y(vec3f(mesh_off.x, 0.0, mesh_off.y) * scale, yaw);
    let xz = base.xz + off.xz;

    // Rung 3 of the fitting ladder: evaluate the ground under every vertex.
    // Neighbouring tiles share corner positions, so the whole mat stays C0
    // continuous — no rung that fits ONE plane per tile can promise that.
    // terrain_sample returns height and the (nx, nz) of the ground normal in
    // the same four texel loads, so the shading basis is free.
    let g = terrain_sample(xz);
    up_ws = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
    let step = tile_m * scale;
    let gi = i32(floor(base.x / step));
    let gj = i32(floor(base.z / step));
    // Only needed when tiles actually overlap: at overscale 1.0 they abut
    // exactly, and then a bias would only break the mat's C0 continuity.
    let bias = select(0.0, CARPET_BIAS * f32((gi & 1) + 2 * (gj & 1)), info.carpet_over > 1.0);
    world = vec3f(xz.x, g.x + mix(h0, h1, CARPET_TOP) + bias, xz.y);

    // The tile's own sub-rectangle of the baked top view: the capture square
    // is 2*card_r across and centred on the mesh bounds, the tile is
    // [0, tile_m]^2 in the mesh frame.
    let uv_off = vec2f(tile_m * 0.5) - vec2f(info.card_cx, info.card_cz) + mesh_off;
    uv = tile_uv(8u, 0.5 + 0.5 * uv_off / info.card_r);
    shade = 1.0;
    // No camera-inside fade — a mat you are standing on must not open a hole.
    // Region edge only, matching the cull boundary. Measured from the tile
    // CENTRE, never per vertex: the fade decides whether the whole quad is
    // emitted behind the near plane, so if two vertices of one tile disagreed
    // the quad would be stretched from its real position to the clip point and
    // rasterize as a long sliver (this is exactly what it did before).
    let d_xz = length(frame.camera_pos.xz - base.xz);
    fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  } else {
    // --- upright plant: camera-facing side card + horizontal top card ------
    // The baked capture is centered on the clump, which sits off the mesh
    // origin — reproduce that offset so imagery lands where the geometry was.
    let clump = rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * scale, yaw);
    let axis = base + clump;
    let is_top = vi >= 6u;

    let to_cam = frame.camera_pos - axis;
    let sway = wind_sway(base, frame.time, entry.sway, phase);

    // Camera-inside fade: 3D distance to the plant's core segment.
    let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
    let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
    let near_fade = smoothstep(r * 0.35, r * 1.05, d_core);
    // Region edge: erode to nothing exactly at region_r (cull matches).
    let d_xz = length(to_cam.xz);
    let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
    fade = near_fade * edge_fade;
    if (is_top) {
      // The top card only makes sense from well above — at flatter views it
      // reads as a pale floating cutout among the side cards, so erode it away
      // unless the card itself is seen steeply (elevation vs the card height).
      let dy = frame.camera_pos.y - (base.y + mix(h0, h1, info.top_frac));
      let elev = abs(dy) / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
      fade *= smoothstep(0.35, 0.6, elev);
    }

    if (is_top) {
      let c2 = vec2f(c.x, c.y * 2.0 - 1.0); // top card spans [-1,1] on both axes
      world = base + rot_y(vec3f(info.card_cx + c2.x * info.card_r, 0.0, info.card_cz + c2.y * info.card_r) * scale, yaw);
      world.y += mix(h0, h1, info.top_frac);
      world += sway * info.top_frac;
      uv = tile_uv(8u, c2 * 0.5 + 0.5);
      shade = 1.0 - info.bottom_shade * 0.3;
    } else {
      let fwd_xz = to_cam.xz / max(length(to_cam.xz), 1e-4);
      let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
      world = axis + right * (c.x * r);
      world.y = base.y + mix(h0, h1, c.y);
      world += sway * c.y;

      // Nearest baked azimuth in the plant's local frame.
      let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -yaw);
      let ang = atan2(local_dir.x, local_dir.z);
      let k = u32((i32(round(ang * (N_SIDE_VIEWS / 6.2831853))) + i32(N_SIDE_VIEWS)) % i32(N_SIDE_VIEWS));
      uv = tile_uv(k, vec2f(c.x * 0.5 + 0.5, 1.0 - c.y));
      shade = mix(1.0 - info.bottom_shade, 1.0, c.y);
    }
  }

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards — emit it behind the near plane
  // instead of rasterizing a guaranteed-empty quad. This is exactly the set
  // the fragment shader was already throwing away (fully faded top cards at
  // low elevations, cards the camera is standing inside, the region rim), and
  // those are precisely the ones that cover the most screen area.
  if (fade < a_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = uv;
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = a_ref / max(fade, 1e-3);
  out.shade = shade;
  out.up_ws = up_ws;
  out.carpet = select(0u, 1u, is_carpet);
  return out;
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode_card(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let alb = textureSample(card_albedo, card_sampler, in.uv);
  let enc = textureSample(card_normal, card_sampler, in.uv).xy;
  if (alb.a < in.erode) {
    discard;
  }
  let n_mesh = oct_decode_card(enc * 2.0 - 1.0);
  var n: vec3f;
  if (in.carpet == 1u) {
    // Carpet tile: the mesh frame was captured over flat ground, so the baked
    // normal has to be lifted into the local GROUND frame, not just yawed —
    // otherwise a mat on a slope lights as if it were level. This is
    // plant_basis_from_up(up, yaw) from terrain.wgsl, inlined because the card
    // already carries cos/sin(yaw) rather than the angle.
    let up = normalize(in.up_ws);
    var t = vec3f(in.yaw_cs.x, 0.0, -in.yaw_cs.y);
    let proj = t - up * dot(up, t);
    t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
    n = t * n_mesh.x + up * n_mesh.y + cross(t, up) * n_mesh.z;
  } else {
    // Rotate the mesh-frame normal by the plant's yaw.
    n = vec3f(
      in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
      n_mesh.y,
      -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
    );
  }
  // in.shade is a fake grounding occlusion, so it belongs to the light term:
  // the albedo view then shows the baked atlas colour exactly as captured and
  // the lighting view shows sun+ambient x grounding.
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the baked alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin (everything below `erode` is gone).
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
