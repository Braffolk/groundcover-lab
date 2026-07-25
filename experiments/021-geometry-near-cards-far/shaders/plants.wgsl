#include "src/wgsl/wind.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/hash.wgsl"
#include "./entry_info.wgsl"

// Two LODs over one capture box.
//
// vs_cloud / fs_cloud (near): N_CARDS vertical planes FIXED IN THE PLANT'S
//   OWN FRAME, each textured with the wedge of source geometry that lies in
//   it, plus a horizontal top card. The planes do not face the camera, so the
//   silhouette is the union of three differently-foreshortened images: it
//   genuinely changes shape with view direction, near blades slide across far
//   ones as the camera moves, and the planes depth-test against each other so
//   the crossing points read as real self-occlusion.
//
// vs_far / fs_far (far): one camera-facing card from the 8-azimuth impostor
//   sheet plus the top card — i.e. exactly a billboard, at billboard cost.
//
// vs_carpet / vs_carpet_far / fs_carpet (mat species, carpet_div > 0): a
//   Sphagnum tile is a 0.18m periodic cushion 0.09m tall that LIES on the
//   ground; azimuth cards through it would slice the terrain and each other
//   and show nothing edge-on. It is drawn instead as a stack of
//   ground-parallel SHELLS over the tile's own square, cut from one top-down
//   capture that carries a height channel: shell k keeps only the texels whose
//   surface reaches its plane, so the stack is a terraced reconstruction of
//   the real 3.3cm of capitulum relief. Far away the stack collapses to the
//   single quad a flat card would draw. Same grid, same 90-degree yaw, same
//   constant scale as the stand gave — the lattice is never touched.
//
// Both: hard alpha test with depth write (no blending, no dithering, no
// frag_depth), 2 texture taps per fragment, card bottoms sunk into the ground
// so the terrain depth buffer clips the cut edge away. The interpolant set is
// kept to 11 components in 4 slots — a card is mostly alpha-test rejects, so
// per-fragment attribute traffic is a first-order cost here.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var near_albedo: texture_2d_array<f32>;
@group(1) @binding(2) var near_normal: texture_2d_array<f32>;
@group(1) @binding(3) var far_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var far_normal: texture_2d_array<f32>;
@group(1) @binding(5) var card_sampler: sampler;
@group(2) @binding(0) var<storage, read> insts: array<PlantInst>;

const TOP_LAYER_NEAR: u32 = 3u;              // = N_CARDS
const TOP_LAYER_FAR: u32 = 12u;              // = NEAR_LAYERS + N_AZI

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv_shade: vec3f,                 // tile uv, grounding gradient
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) card: vec4f,  // cos yaw, sin yaw, erode, tint
  @location(3) @interpolate(flat) layer: u32,
}

struct Plant {
  base: vec3f,
  axis: vec3f,
  yaw: f32,
  scale: f32,
  h0: f32,
  h1: f32,
  r: f32,
  sway: vec3f,
  to_cam: vec3f,
  fade: f32,
  tint: f32,
}

fn load_plant(ii: u32) -> Plant {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  var p: Plant;
  p.yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  p.scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  p.base = inst.pos;
  p.r = info.card_r * p.scale;
  p.h0 = info.card_y0 * p.scale;
  p.h1 = info.card_y1 * p.scale;
  // The baked capture is centered on the clump, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  p.axis = p.base + rot_y(vec3f(info.card_cx, 0.0, info.card_cz) * p.scale, p.yaw);
  p.to_cam = frame.camera_pos - p.axis;

  let entry = stand_table[u32(info.entry_index)];
  p.sway = wind_sway(p.base, frame.time, entry.sway, phase);

  // Camera-inside fade: 3D distance to the plant's core segment.
  let seg_y = clamp(frame.camera_pos.y, p.base.y + p.h0, p.base.y + p.h1);
  let d_core = length(vec3f(p.to_cam.x, frame.camera_pos.y - seg_y, p.to_cam.z));
  let near_fade = smoothstep(p.r * 0.35, p.r * 1.05, d_core);
  // Region edge: erode to nothing exactly at region_r (cull matches).
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, length(p.to_cam.xz));
  p.fade = near_fade * edge_fade;
  p.tint = hash_f32(pcg(bits)) * 2.0 - 1.0;
  return p;
}

/// The top card only makes sense from well above — at flatter views it reads
/// as a pale floating cutout, so erode it away unless it is seen steeply.
fn top_fade(p: Plant) -> f32 {
  let dy = frame.camera_pos.y - (p.base.y + mix(p.h0, p.h1, info.top_frac));
  let elev = abs(dy) / max(length(vec3f(p.to_cam.x, dy, p.to_cam.z)), 1e-4);
  return smoothstep(0.35, 0.6, elev);
}

fn quad_corner(ci: u32, b: vec4f) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let c = corners[ci];
  return vec2f(mix(b.x, b.z, c.x), mix(b.y, b.w, c.y));
}

/// World position of a vertical card corner. `dir_ws` is the card's in-plane
/// horizontal axis; the bottom edge is pushed `sink` below the capture box so
/// the terrain (already in the depth buffer) clips the cut line away.
fn side_vertex(p: Plant, dir_ws: vec3f, t: vec2f) -> vec3f {
  let hf = 1.0 - t.y;
  var w = p.axis + dir_ws * ((t.x * 2.0 - 1.0) * p.r);
  w.y = p.base.y + mix(p.h0, p.h1, hf) - info.sink * p.scale * (1.0 - hf);
  return w + p.sway * hf;
}

fn top_vertex(p: Plant, t: vec2f) -> vec3f {
  let off = vec3f(
    info.card_cx + (t.x * 2.0 - 1.0) * info.card_r,
    0.0,
    info.card_cz + (t.y * 2.0 - 1.0) * info.card_r,
  ) * p.scale;
  var w = p.base + rot_y(off, p.yaw);
  w.y += mix(p.h0, p.h1, info.top_frac);
  return w + p.sway * info.top_frac;
}

/// A card whose fade dropped below the alpha reference has erode > 1, so EVERY
/// one of its fragments discards — emit it behind the near plane instead of
/// rasterizing a guaranteed-empty quad. Those are precisely the cards that
/// would cover the most screen area (fully faded top cards at low elevations,
/// cards the camera stands inside, the region rim).
fn finish(p: Plant, world: vec3f, uv: vec2f, layer: u32, fade: f32, shade: f32) -> VOut {
  var out: VOut;
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv_shade = vec3f(uv, shade);
  out.world = world;
  out.card = vec4f(cos(p.yaw), sin(p.yaw), info.alpha_ref / max(fade, 1e-3), p.tint);
  out.layer = layer;
  return out;
}

// --- shared shading ---------------------------------------------------------

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

fn shade_card(in: VOut, alb: vec4f, enc: vec2f) -> vec4f {
  let n_mesh = oct_decode_card(enc * 2.0 - 1.0);
  // Rotate the mesh-frame normal by the plant's yaw.
  var n = vec3f(
    in.card.x * n_mesh.x + in.card.y * n_mesh.z,
    n_mesh.y,
    -in.card.y * n_mesh.x + in.card.x * n_mesh.z,
  );
  // Blades are thin two-sided sheets and the bake stored the normal facing the
  // bake camera, so on a vertical card seen from its far side, flip it toward
  // the viewer. (Top cards are only ever seen from above — leave them alone.)
  let two_sided = in.layer != TOP_LAYER_NEAR && in.layer != TOP_LAYER_FAR;
  if (two_sided && dot(n, frame.camera_pos - in.world) < 0.0) {
    n = -n;
  }
  // The grounding gradient and the per-plant tint belong to the albedo so the
  // lighting debug view stays exact.
  let j = info.tint_jitter * in.card.w;
  let tinted = alb.rgb * vec3f(1.0 + j, 1.0 + j * 0.45, 1.0 - j * 0.25);
  let albedo = clamp(tinted * in.uv_shade.z, vec3f(0.0), vec3f(1.0));
  // Shared model + a little sun transmission (grass is translucent). Still
  // multiplicative in albedo, so DEBUG_LIGHTING divides out exactly.
  let back = max(0.0, -dot(n, frame.sun_dir));
  var color = light_surface(albedo, n, in.world) + albedo * frame.sun_color * (info.translucency * back * back);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the baked alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin (everything below `erode` is gone).
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}

// --- near LOD: the card cloud ----------------------------------------------

@vertex
fn vs_cloud(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = load_plant(ii);
  let ci = vi % 6u;

  if (vi >= CLOUD_VERTS) {
    let t = quad_corner(ci, info.tile_bounds[TOP_LAYER_NEAR]);
    return finish(p, top_vertex(p, t), t, TOP_LAYER_NEAR, p.fade * top_fade(p), 1.0 - info.bottom_shade * 0.3);
  }

  // One quad per (card, height band): same mapping as one tall quad — both the
  // height lerp and the sway are linear in height — but only as wide as the
  // ink actually reaches at that height, which drops ~35% of the empty area
  // a card of 4-10% coverage would otherwise rasterize and reject.
  let card = vi / (BANDS * 6u);
  let band = (vi / 6u) % BANDS;
  let i = card * BANDS + band;
  let e = info.card_bands[i >> 1u];
  let u = select(e.xy, e.zw, (i & 1u) == 1u);
  let v0 = f32(band) / f32(BANDS);
  let t = quad_corner(ci, vec4f(u.x, v0, u.y, v0 + 1.0 / f32(BANDS)));

  // Card k's plane contains the horizontal direction a_k = k*pi/N_CARDS in the
  // plant's own frame — the same axis the wedge was baked onto.
  let a = f32(card) * (3.14159265 / f32(N_CARDS));
  let dir_ws = rot_y(vec3f(cos(a), 0.0, -sin(a)), p.yaw);
  let shade = mix(1.0 - info.bottom_shade, 1.0, 1.0 - t.y);
  return finish(p, side_vertex(p, dir_ws, t), t, card, p.fade, shade);
}

@fragment
fn fs_cloud(in: VOut) -> @location(0) vec4f {
  let alb = textureSample(near_albedo, card_sampler, in.uv_shade.xy, in.layer);
  let enc = textureSample(near_normal, card_sampler, in.uv_shade.xy, in.layer).xy;
  if (alb.a < in.card.z) {
    discard;
  }
  return shade_card(in, alb, enc);
}

// --- far LOD: the impostor --------------------------------------------------

@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let p = load_plant(ii);
  let ci = vi % 6u;

  if (vi >= 6u) {
    let t = quad_corner(ci, info.tile_bounds[TOP_LAYER_FAR]);
    let fade = p.fade * top_fade(p);
    return finish(p, top_vertex(p, t), t, TOP_LAYER_FAR, fade, 1.0 - info.bottom_shade * 0.3);
  }

  // Nearest baked azimuth in the plant's local frame.
  let fwd_xz = p.to_cam.xz / max(length(p.to_cam.xz), 1e-4);
  let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -p.yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  let k = u32((i32(round(ang * (f32(N_AZI) / 6.2831853))) + i32(N_AZI)) % i32(N_AZI));
  let layer = NEAR_LAYERS + k;
  let t = quad_corner(ci, info.tile_bounds[layer]);
  let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
  let shade = mix(1.0 - info.bottom_shade, 1.0, 1.0 - t.y);
  return finish(p, side_vertex(p, right, t), t, layer, p.fade, shade);
}

@fragment
fn fs_far(in: VOut) -> @location(0) vec4f {
  let l = in.layer - NEAR_LAYERS;
  let alb = textureSample(far_albedo, card_sampler, in.uv_shade.xy, l);
  let enc = textureSample(far_normal, card_sampler, in.uv_shade.xy, l).xy;
  if (alb.a < in.card.z) {
    discard;
  }
  return shade_card(in, alb, enc);
}

// --- carpet: ground-parallel relief shells -----------------------------------
// Own vertex/fragment pair (and own interpolants) rather than a branch in the
// card path: a mat needs a ground basis per vertex and a height gate, the card
// path needs neither, and the card path's fragment cost is dominated by
// per-fragment attribute traffic.

struct COut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) up_ws: vec3f,                    // ground normal, per vertex
  @location(3) @interpolate(flat) tile: vec4f,  // cos yaw, sin yaw, erode, height gate
  @location(4) @interpolate(flat) shade: f32,
}

/**
 * One shell of a mat tile (or the single far quad). `shell_y` is the plane's
 * height in the mesh frame at scale 1; `gate` is the normalized surface height
 * a texel must reach to exist on this shell.
 */
fn carpet_vertex(ii: u32, ci: u32, shell_y: f32, gate: f32, shade: f32) -> COut {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let entry = stand_table[u32(info.entry_index)];
  // The stand's exact carpet scale, NOT the 12-bit instance value: every tile
  // of a mat has the same scale by construction, and the 0.08% quantization
  // error would leave the tiles not quite abutting.
  let scale = entry.scale_min;

  // Exact quarter turn from the quantized yaw: a mat only ever rotates in
  // 90-degree steps (that is what keeps a square periodic tile matching its
  // neighbours), and rounding to the nearest quarter avoids the ~0.35-degree
  // error the 10-bit yaw quantization would otherwise leave in it.
  let qt = (((bits & 1023u) + 128u) / 256u) & 3u;
  var cs = array<vec2f, 4>(vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(-1.0, 0.0), vec2f(0.0, -1.0));
  let yc = cs[qt].x;
  let ys = cs[qt].y;

  // Footprint from the species' periodic tile size — NEVER from its height,
  // which would make a 0.24m-wide 0.07m-tall cushion ~3.5x too small.
  let t = quad_corner(ci, vec4f(0.0, 0.0, 1.0, 1.0));
  let local = (t * 2.0 - 1.0) * (entry.footprint_m * 0.5 * scale);
  let off = vec2f(yc * local.x + ys * local.y, -ys * local.x + yc * local.y);
  let xz = inst.pos.xz + off;

  // Ladder rung 3 — the ground under EVERY vertex. Neighbouring tiles share
  // their corner positions exactly (the grid step IS the tile size), so
  // per-vertex conforming is the only rung that keeps the whole mat
  // C0-continuous; any per-tile plane fit cracks along every tile boundary.
  // terrain_sample gives height and the ground normal's (nx, nz) in one
  // bilinear fetch, so the shading basis costs nothing extra.
  let g = terrain_sample(xz);
  let up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  let world = vec3f(xz.x, g.x + shell_y * scale, xz.y);

  // Region edge only: NO camera-inside fade — a mat you are standing on must
  // not open a hole under you. Measured from the tile centre, never per
  // vertex: the fade decides whether the quad is emitted at all, so vertices
  // that disagreed would stretch it into a sliver metres long.
  let fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, length(frame.camera_pos.xz - inst.pos.xz));
  let a_ref = info.shell_more.y;

  var out: COut;
  if (fade < a_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = t;
  out.world = world;
  out.up_ws = up;
  out.tile = vec4f(yc, ys, a_ref / max(fade, 1e-3), gate);
  out.shade = shade;
  return out;
}

@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> COut {
  let n = max(u32(info.shell_span.w), 2u);
  // Top shell first: shell coverage is nested, so from above the higher (and
  // nearer) shells give early-z something to reject the base shell against.
  let k = (n - 1u) - min(vi / 6u, n - 1u);
  let f = f32(k) / f32(n - 1u);
  let shell_y = mix(info.shell_span.x, info.shell_span.z, f);
  // Shell 0 carries no height gate at all: it is the closed base of the mat,
  // and a mat that dissolves with distance is worse than one with no relief.
  let gate = select(0.0, (shell_y - info.card_y0) / max(info.card_y1 - info.card_y0, 1e-6), k > 0u);
  // Cushion occlusion — the deeper a shell sits between the capitula, the less
  // sky reaches it. This is what makes the terraces read as depth.
  let shade = mix(1.0 - info.shell_more.z, 1.0, f);
  return carpet_vertex(ii, vi % 6u, shell_y, gate, shade);
}

@vertex
fn vs_carpet_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> COut {
  // Past the shell radius: one quad at the surface's mean height with full
  // coverage — exactly what a flat card draws, which is all a tile of ~17px
  // can show anyway. Same texture, same box, so nothing moves at the handover.
  // Its shade is the AREA-WEIGHTED MEAN of the shells it replaces (measured
  // from the tile's own height histogram at load), so the two LODs match in
  // brightness and no dark ring sweeps the field at the handover radius.
  let shade = 1.0 - info.shell_more.z * (1.0 - info.shell_more.w);
  return carpet_vertex(ii, vi % 6u, info.shell_span.y, 0.0, shade);
}

@fragment
fn fs_carpet(in: COut) -> @location(0) vec4f {
  let alb = textureSample(near_albedo, card_sampler, in.uv, 0u);
  let nh = textureSample(near_normal, card_sampler, in.uv, 0u);
  // Two hard tests, no dithering: the baked coverage, and this shell's height
  // gate (the surface at this texel must reach the shell's own plane).
  if (alb.a < in.tile.z || nh.a < in.tile.w) {
    discard;
  }
  // The tile was captured over flat ground, so the baked normal must be lifted
  // into the local GROUND frame or a mat on a slope lights as if it were
  // level. plant_basis_from_up(up, yaw), inlined — the vertex already carries
  // cos/sin(yaw) rather than the angle. Normals are stored as plain unit
  // vectors (not octahedral) precisely so this stays valid through the mips.
  let n_mesh = normalize(nh.xyz * 2.0 - 1.0);
  let up = normalize(in.up_ws);
  var tang = vec3f(in.tile.x, 0.0, -in.tile.y);
  let proj = tang - up * dot(up, tang);
  tang = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  let n = tang * n_mesh.x + up * n_mesh.y + cross(tang, up) * n_mesh.z;

  let albedo = clamp(alb.rgb * in.shade, vec3f(0.0), vec3f(1.0));
  var color = light_surface(albedo, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}
