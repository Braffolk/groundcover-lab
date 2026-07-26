#include "src/wgsl/wind.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./card_info.wgsl"

// CURVED CARDS — a card that is a piece of the plant's real front surface.
//
// A billboard is a flat quad that spins to face you, so every texel of a plant
// sits at one depth: no parallax inside the silhouette, no view-dependent
// outline, and neighbouring plants stack like paper cut-outs. This method
// keeps the billboard's entire cost structure and changes ONE thing: the card
// is a small lattice whose vertices are pushed along the view axis by the
// baked front-surface distance, and the card's orientation is SNAPPED to the
// baked azimuth instead of tracking the camera.
//
// Both halves of that matter:
//   * snapping the orientation makes the card a fixed 3-D object inside its
//     45deg sector, so orbiting the camera reveals genuine parallax and a
//     genuinely changing outline instead of a rotating cut-out;
//   * at the exact bake azimuth the displacement is parallel to the view ray,
//     so the render degenerates to the pixel-exact baked image — the technique
//     is never worse than the billboard it replaces, only more correct in
//     between (which is also why the 45deg view switch pops far less: both
//     neighbouring surfaces reproject to nearly the same picture).
//
// Vertex layout per instance (grid dims come from the per-LOD uniform):
//   0 .. sx*sy-1                       curved side patch, u in [-1,1], v in [0,1]
//   sx*sy .. sx*sy + tx*ty-1           canopy patch, displaced in Y by the
//                                      baked top-view height field
// The DISP atlas (32 texels/tile, hole-free, blurred) drives the vertices; the
// full-resolution ALBEDO atlas gives the hard alpha-tested silhouette and the
// SURF atlas gives normals, front distance and baked sky occlusion. The
// fragment shader is deliberately IDENTICAL IN SHAPE to the billboard
// baseline's: the atlas uv is finished in the vertex stage and both taps read
// the same coordinate, so they issue in parallel and share one cache line. No
// dependent reads, no loops, no frag_depth. All the 3-D lives in the vertices.
//
// (A per-pixel parallax residual was tried on top and thrown out: it cost a
// dependent tap and, at close range, smeared the silhouette into rubber. The
// lattice alone carries the relief.)
//
// CARPET SPECIES (stand carpet_div > 0, e.g. Sphagnum) take the same idea and
// turn it on its side. A mat has no silhouette to snap an azimuth to — seen
// edge-on it is almost nothing, and a vertical card would slice through the
// ground — so the card becomes a GROUND-PARALLEL lattice over the species'
// periodic tile, and the displacement runs along the ground normal instead of
// the view axis, driven by the baked TOP-VIEW front-surface distance, which
// for a straight-down capture is exactly the cushion's height field. That is
// the one thing a flat quad cannot express: real cushion relief, a bumpy
// silhouette at grazing, and per-pixel depth interleaving with its neighbours.
// The tile's 90-degree yaw and constant scale come from the stand and are used
// AS GIVEN — they are what keeps neighbouring tiles agreeing. Every lattice
// vertex resolves its own ground height (ladder rung 3), so tiles that share an
// edge share its vertices and the mat stays continuous over any relief.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: CardInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var tex_albedo: texture_2d<f32>;
@group(1) @binding(3) var tex_surf: texture_2d<f32>;
@group(1) @binding(4) var tex_disp: texture_2d<f32>;
@group(1) @binding(5) var card_sampler: sampler;

const TILE_GRID: f32 = 3.0;   // 3x3 tiles in every atlas; tile 8 is the top view
const N_SIDE_VIEWS: f32 = 8.0;
const VIEW_STEP: f32 = 6.2831853 / N_SIDE_VIEWS;
const INSET_FINE: f32 = 0.0078125; // 4 texels of a 512 albedo tile
const INSET_DISP: f32 = 0.046875;  // 1.5 texels of a 32 disp tile
// A carpet species binds the TOP TILE'S OWN SQUARE ALONE (see uploadAtlas in
// main.ts): the 8 side views describe nothing a mat ever shows, and dropping
// them frees the ~15 MB that a life-size tile count needs in the instance
// buffer instead. So a carpet's texture coordinate is the tile-local one, with
// no 3x3 atlas transform and no inset.

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn tile_uv(view_index: u32, local: vec2f, inset: f32) -> vec2f {
  let col = f32(view_index % 3u);
  let row = f32(view_index / 3u);
  let clamped = clamp(local, vec2f(inset), vec2f(1.0 - inset));
  return (vec2f(col, row) + clamped) / TILE_GRID;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,                        // finished atlas uv
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) up_ws: vec3f,                     // ground up (carpet tiles)
  @location(5) @interpolate(flat) carpet: u32,   // 1 = ground-parallel mat tile
}

/**
 * A tiled mat: one ground-parallel lattice over the species' periodic tile,
 * displaced along the ground by the baked top-view height field.
 */
fn vs_carpet(vi: u32, ii: u32, entry: StandEntry) -> VOut {
  let inst = insts[u32(info.ring.x) + ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let base = inst.pos;

  // Lattice over the tile's own square, in the mesh frame. Width comes from
  // the species' periodic FOOTPRINT, never from its height: footprint_m *
  // scale is exactly the carpet's grid step, so tiles abut with no overscale.
  let cols = u32(info.grid.x);
  let fu = f32(vi % cols) / (info.grid.x - 1.0);
  let fv = f32(vi / cols) / (info.grid.y - 1.0);
  let tile_m = entry.footprint_m;
  let mesh_off = (vec2f(fu, fv) - 0.5) * tile_m; // from the tile centre
  let off = rot_y(vec3f(mesh_off.x, 0.0, mesh_off.y) * scale, yaw);
  let xz = base.xz + off.xz;

  // Rung 3 of the terrain-fitting ladder: the ground under EVERY vertex.
  // Neighbouring tiles share their edge vertices, so the mat stays C0
  // continuous over any relief — no per-tile plane fit can promise that, and
  // at 0.18m a tile is far too small to fit a plane to anyway. terrain_sample
  // returns height and (nx, nz) in the same four texel loads, so the shading
  // basis is free.
  let g = terrain_sample(xz);
  let up_ws = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);

  // A mat's three textures were cropped at load time to the tile's own square
  // in the mesh frame — [0, tile_m]^2, whose origin is (0,0) for every current
  // source tile — so the lattice fraction IS the texture coordinate, with no
  // atlas transform anywhere in the carpet path. The crop is worth ~3.4x the
  // on-screen texel density of the whole capture, and it is what keeps the mip
  // chain averaging moss instead of empty background (see resampleTileSquare).
  let local = vec2f(fu, fv);

  // THE relief: the baked top-view front-surface distance IS the cushion's
  // height field, so the lattice rides the capitula instead of hovering over
  // them as a plane. The far ring is flat by construction (shape.w = 0) and its
  // whole draw skips the fetch; the test is draw-uniform. top_frac is the
  // species' mean canopy height, so a flat tile and a displaced tile sit at the
  // same average height and the ring change is invisible.
  var hfrac = info.top_frac;
  if (info.shape.w > 0.0) {
    let q = textureSampleLevel(tex_disp, card_sampler, local, 0.0).r;
    hfrac = mix(info.top_frac, info.ring.w + q * info.shape.x, info.shape.w);
  }
  let world = vec3f(xz.x, g.x + mix(info.card_y0, info.card_y1, hfrac) * scale, xz.y);

  // No camera-inside fade: a mat you are standing on must not open a hole
  // under you. Region edge only, and measured from the tile CENTRE — the fade
  // decides whether the whole tile is emitted behind the near plane, so two
  // vertices of one tile must never disagree about it.
  let fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, length(frame.camera_pos.xz - base.xz));

  var out: VOut;
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = local;
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.up_ws = up_ws;
  out.carpet = 1u;
  return out;
}

fn vs_plant(vi: u32, ii: u32, entry: StandEntry) -> VOut {
  let inst = insts[u32(info.ring.x) + ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let base = inst.pos;

  let r = info.card_r * scale;
  let h0 = info.card_y0 * scale;
  let h1 = info.card_y1 * scale;
  // The baked capture is centered on the clump, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  let clump = rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * scale, yaw);
  let axis = base + clump;

  let to_cam = frame.camera_pos - axis;
  let sway = wind_sway(base, frame.time, entry.sway, phase);

  // --- which baked azimuth, and therefore which card orientation ------------
  let fwd_cam = to_cam.xz / max(length(to_cam.xz), 1e-4);
  let local_dir = rot_y(vec3f(fwd_cam.x, 0.0, fwd_cam.y), -yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  let k = u32((i32(round(ang / VIEW_STEP)) + i32(N_SIDE_VIEWS)) % i32(N_SIDE_VIEWS));
  // Bake view k has fwd = (sin a, 0, cos a) in the mesh frame; yawing it gives
  // the card's world normal, and right = (fwd.z, 0, -fwd.x).
  let card_a = f32(k) * VIEW_STEP + yaw;
  let card_fwd = vec3f(sin(card_a), 0.0, cos(card_a));
  let card_right = vec3f(card_fwd.z, 0.0, -card_fwd.x);

  let sx = u32(info.grid.x);
  let sy = u32(info.grid.y);
  let tx = u32(info.grid.z);
  let n_side_v = sx * sy;
  let is_top = vi >= n_side_v;

  var local: vec2f;
  var world: vec3f;
  var tile: u32;
  var lid_fade = 1.0;

  if (is_top) {
    let j = vi - n_side_v;
    let gx = f32(j % tx);
    let gy = f32(j / tx);
    let cu = gx / (info.grid.z - 1.0) * 2.0 - 1.0;
    let cv = gy / (info.grid.w - 1.0) * 2.0 - 1.0;
    local = vec2f(cu * 0.5 + 0.5, cv * 0.5 + 0.5);
    tile = 8u;
    // How steeply is the canopy patch seen? A height-field lid is only
    // meaningful from above: edge-on it degenerates into a long smeared sheet,
    // so the relief is folded away on exactly the same ramp that erodes the
    // patch's coverage. Measured against the FLAT plane so it is independent
    // of the displacement it controls.
    let flat_y = base.y + mix(h0, h1, info.top_frac);
    let dy = frame.camera_pos.y - flat_y;
    // Signed, not absolute: a canopy HEIGHT field only describes the surface
    // seen from above. From underneath, the patch would be a bumpy lid hanging
    // in the air, so it erodes to nothing and the side patch carries the plant.
    let elev = max(dy, 0.0) / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
    lid_fade = smoothstep(0.40, 0.65, elev);
    // Baked canopy height at this (x,z). The far ring is flat by construction,
    // so its whole draw skips the fetch (the test is draw-uniform).
    var hfrac = info.top_frac;
    if (info.shape.w > 0.0) {
      let canopy = textureSampleLevel(tex_disp, card_sampler, tile_uv(8u, local, INSET_DISP), 0.0).r;
      hfrac = mix(info.top_frac, canopy, info.shape.w * lid_fade);
    }
    world = base + rot_y(vec3f(info.card_cx + cu * info.card_r, 0.0, info.card_cz + cv * info.card_r) * scale, yaw);
    world.y = base.y + mix(h0, h1, hfrac);
    world += sway * hfrac;
  } else {
    let gx = f32(vi % sx);
    let gy = f32(vi / sx);
    let u = gx / (info.grid.x - 1.0) * 2.0 - 1.0;
    let v = gy / (info.grid.y - 1.0);
    local = vec2f(u * 0.5 + 0.5, 1.0 - v);
    tile = k;
    // THE curve: push the lattice out to the baked front surface. Same
    // draw-uniform skip — the far ring is a plain quad and fetches nothing.
    var t = 0.0;
    if (info.shape.w > 0.0) {
      let d01 = textureSampleLevel(tex_disp, card_sampler, tile_uv(k, local, INSET_DISP), 0.0).r;
      t = (d01 * 2.0 - 1.0) * info.shape.w;
    }
    world = axis + card_right * (u * r) + card_fwd * (t * r);
    world.y = base.y + mix(h0, h1, v);
    world += sway * v;
  }

  // --- coverage fades (eroded through the alpha test, never blended) --------
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(r * 0.35, r * 1.05, d_core);
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);
  // The canopy patch only makes sense from well above — at flatter views it
  // reads as a pale floating lid among the side cards (lid_fade is 1 on the
  // side patch).
  var fade = near_fade * edge_fade * lid_fade;

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards — emit it behind the near plane
  // instead of rasterizing a guaranteed-empty patch.
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  // Finish the atlas coordinate here, exactly like the baseline does: the
  // fragment shader must not repeat the tile transform (two integer divides
  // per pixel) and must not make its second tap wait for its first.
  out.uv = tile_uv(tile, local, INSET_FINE);
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.up_ws = vec3f(0.0, 1.0, 0.0);
  out.carpet = 0u;
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let entry = stand_table[u32(info.entry_index)];
  // Draw-uniform: every instance of a draw belongs to exactly one stand entry,
  // so the branch is free and the two paths never diverge within a draw.
  if (entry.carpet_div > 0.0) {
    return vs_carpet(vi, ii, entry);
  }
  return vs_plant(vi, ii, entry);
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
  // Two independent taps at one coordinate — sampled before any non-uniform
  // discard (uniform-control-flow rule).
  let alb = textureSample(tex_albedo, card_sampler, in.uv);
  let surf = textureSample(tex_surf, card_sampler, in.uv);
  if (alb.a < in.erode) {
    discard;
  }

  let n_mesh = oct_decode_card(surf.rg * 2.0 - 1.0);
  var n: vec3f;
  if (in.carpet == 1u) {
    // A mat tile was captured over level ground, so its baked normal has to be
    // lifted into the LOCAL GROUND frame, not merely yawed — otherwise moss on
    // a slope lights as if it were lying flat. This is plant_basis_from_up(up,
    // yaw) from terrain.wgsl, inlined because the vertex stage already carries
    // cos/sin(yaw) rather than the angle.
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

  // Occlusion is fake shadowing, so it belongs to the LIGHT term: the albedo
  // debug view then shows the baked atlas colour exactly as captured.
  //   surf.a  baked sky occlusion (canopy integrated above the texel)
  //   surf.b  front distance — texels deep inside the clump see less sky
  // shape.y is 1 / (the field's own coverage-weighted mean), computed on the
  // CPU: the occlusion then REDISTRIBUTES light (deep texels darker, exposed
  // tips brighter) at the same average exposure as an unshaded card, instead of
  // dimming the whole meadow — a global dimmer reads as "worse", not "deeper".
  let occ = mix(1.0, surf.a, info.ao_strength) * mix(1.0 - info.shape.z, 1.0, surf.b) * info.shape.y;
  var color = light_surface(alb.rgb * min(occ, 1.4), n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the baked alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin (everything below `erode` is gone).
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
