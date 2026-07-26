#include "src/wgsl/wind.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./info.wgsl"

// Clump-impostor cards. Two shapes, chosen per stand entry.
//
// UPRIGHT PLANTS (carpet_div == 0)
//   NEAR list (one instance = one plant, 12 * K_SUB vertices):
//     verts 0 .. 6*K_SUB-1   — one camera-facing card per SUB-CLUMP, standing
//                              at that sub-clump's real offset inside the
//                              plant, so the cards sit at genuinely different
//                              depths: they parallax against each other,
//                              occlude each other through the depth buffer, and
//                              interleave with the neighbouring plants' cards.
//     verts 6*K_SUB .. 12K-1 — one horizontal top card per sub-clump.
//   FAR list (12 vertices): the merged whole-plant card + its top card, i.e.
//   exactly a billboard — the distance collapse.
//
// CARPET TILES (carpet_div > 0, e.g. Sphagnum) — the same idea turned on its
// side. A mat has no silhouette from the side and no camera-facing anything;
// what it has is 3.3cm of capitulum relief over a 0.18m periodic tile. So the
// near LOD is SHELLS ground-parallel quads at the height quantiles of the
// baked cushion height field, each keeping only the texels whose cushion
// reaches its level (shell 0, the top, first, so it writes depth and early-z
// rejects the copies beneath it). The stack is a stepped, self-occluding,
// depth-writing approximation of the real relief; the fragments drawn well
// below their own apex are darkened, which is what turns four planes into a
// cushion. Past carpetShellDist it collapses to ONE quad at the height the
// shells converge to.
//   * the tile's 90-degree yaw and its constant scale come from the stand and
//     are used AS GIVEN (snapped back to exactly 90 degrees and exactly
//     scale_min — the packed quantization would otherwise open a ~1mm crack at
//     every tile edge), which is what keeps neighbouring tiles agreeing;
//   * width comes from footprint_m, NEVER from height_scale;
//   * every vertex is conformed to the ground individually (rung 3), so the
//     mat stays C0 continuous across tile borders on any slope.
//
// Hard alpha test with depth write throughout; near/region fades erode coverage
// through the alpha reference instead of dithering or blending.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: ClumpDyn;
@group(1) @binding(1) var<uniform> atlas_info: ClumpAtlas;
@group(1) @binding(2) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(3) var card_albedo: texture_2d<f32>;
// Upright plants: rg8 octahedral mesh normals. Carpet tiles: rgba8 with
// r,g = mesh-frame normal xz (ny reconstructed), b = cushion apex height,
// a = coverage.
@group(1) @binding(4) var card_normal: texture_2d<f32>;
@group(1) @binding(5) var card_sampler: sampler;

// Alpha reference for carpet tiles, INSTEAD of the params' alphaRef. A mat is a
// closed surface, so its tiles must not dissolve with distance: tile alpha is
// ~80% up close but the mip chain pulls it toward the tile mean, and at the 0.4
// grass reference whole distant tiles then fail the test and punch holes in the
// carpet. A low reference keeps the mat a solid depth-writing occluder while
// the genuinely empty texels — the gaps down to the peat — still open.
const CARPET_ALPHA_REF: f32 = 0.06;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) shade: f32,                       // grounding gradient
  @location(5) up_ws: vec3f,                     // ground up (carpet tiles)
  // x: 1 = carpet tile; y: this shell's lower height threshold; z: this shell's
  // height as a fraction of the capture box.
  @location(6) @interpolate(flat) shell: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 2047u) * (4.0 / 2047.0);
  let phase = f32((bits >> 21u) & 1023u) * (6.2831853 / 1024.0);
  let is_near = (bits >> 31u) == 1u;
  let base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];

  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let q = corners[vi % 6u];

  var out: VOut;
  if (entry.carpet_div > 0.0) {
    // ----------------------------------------------------------------- carpet
    let shell = select(0u, vi / 6u, is_near);
    var shell_y = array<f32, 4>(atlas_info.shell_y.x, atlas_info.shell_y.y, atlas_info.shell_y.z, atlas_info.shell_y.w);
    var shell_t = array<f32, 4>(atlas_info.shell_t.x, atlas_info.shell_t.y, atlas_info.shell_t.z, atlas_info.shell_t.w);
    let hf = select(atlas_info.carpet_box.z, shell_y[shell], is_near);
    let t_lo = select(-1.0, shell_t[shell], is_near);

    // The scatter's yaw is one of four quarter turns and its scale is constant;
    // both arrive quantized through the instance record (yaw to 1/1024 turn,
    // scale to 1/2047 of 4m), and at tile scale those errors are a 0.4-degree
    // rotation and a 0.2% shrink — i.e. a ~1mm crack at every tile edge, drawn
    // as a chicken-wire lattice over the whole mat. Snap both back to what the
    // stand actually specified.
    let quarter = u32(round(yaw * (2.0 / 3.14159265))) & 3u;
    var qcs = array<vec2f, 4>(vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(-1.0, 0.0), vec2f(0.0, -1.0));
    let cs = qcs[quarter];
    let tile_m = entry.footprint_m * entry.scale_min;

    // Tile-local corner offsets in the MESH frame (uv follows them directly,
    // so the texture turns with the tile), rotated into the world by the
    // quarter turn.
    let mesh_off = (q * 2.0 - 1.0) * (tile_m * 0.5);
    let off = vec2f(cs.x * mesh_off.x + cs.y * mesh_off.y, -cs.y * mesh_off.x + cs.x * mesh_off.y);
    let xz = base.xz + off;

    // Rung 3 of the fitting ladder: the ground under EVERY vertex. Neighbouring
    // tiles share corner positions, so the mat stays C0 continuous — a per-tile
    // plane fit cannot promise that, it cracks along every shared edge.
    // terrain_sample returns height and (nx, nz) in one bilinear fetch, so the
    // shading basis costs nothing extra.
    let g = terrain_sample(xz);
    let up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
    let y_off = mix(atlas_info.carpet_box.x, atlas_info.carpet_box.y, hf) * entry.scale_min;
    let world = vec3f(xz.x, g.x + y_off, xz.y);

    // No camera-inside fade — a mat you are standing on must not open a hole.
    // Region edge only, measured from the tile CENTRE so the whole quad agrees:
    // if two vertices disagreed the quad would stretch to the clip point and
    // rasterize as a sliver.
    let fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, length(frame.camera_pos.xz - base.xz));
    if (fade < CARPET_ALPHA_REF) {
      out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
    } else {
      out.pos = frame.view_proj * vec4f(world, 1.0);
    }
    out.uv = q;
    out.world = world;
    out.yaw_cs = cs;
    out.erode = CARPET_ALPHA_REF / max(fade, 1e-3);
    out.shade = 1.0;
    out.up_ws = up;
    out.shell = vec3f(1.0, t_lo, hf);
    return out;
  }

  // ------------------------------------------------------------ upright plant
  // Vertex layout: `n_side` side cards, then the same count of top cards.
  let n_side = select(1u, K_SUB, is_near);
  let card = vi / 6u;
  let is_top = card >= n_side;
  let unit = select(MERGED, card % K_SUB, is_near);

  let box = atlas_info.unit_box[unit];      // cx, cz, rXZ, y0
  let y1_unit = atlas_info.unit_ext[unit].x;
  let r = box.z * scale;
  let h0 = box.w * scale;
  let h1 = y1_unit * scale;
  // The capture was centred on this unit's own bbox, which sits off the mesh
  // origin — reproduce that offset so imagery lands where the geometry was.
  let axis = base + rot_y(vec3f(box.x, 0.0, box.y) * scale, yaw);
  let to_cam = frame.camera_pos - axis;

  // --- pick the baked view ---------------------------------------------------
  let fwd_xz = to_cam.xz / max(length(to_cam.xz), 1e-4);
  let local_dir = rot_y(vec3f(fwd_xz.x, 0.0, fwd_xz.y), -yaw);
  let ang = atan2(local_dir.x, local_dir.z);
  let k_side = u32((i32(round(ang * (N_SIDE / 6.2831853))) + i32(N_SIDE)) % i32(N_SIDE));
  let view = select(k_side, u32(N_SIDE), is_top);
  let tile = (unit * N_VIEWS + view) * 2u;
  let uv_rect = atlas_info.tiles[tile];     // u0, v0, du, dv (already tight)
  let loc = atlas_info.tiles[tile + 1u];    // lx0, lt0, lw, lh in card space

  // Card-space coordinates: s across (0 = -r), tt down (0 = the card's top).
  let s = loc.x + q.x * loc.z;
  let tt = loc.y + q.y * loc.w;

  // --- fades -----------------------------------------------------------------
  // Region edge is per plant; the camera-inside fade is per CARD, measured to
  // that card's own core. A sub-clump card stands up to ~0.25m off the plant
  // axis, so a per-plant test would happily leave a card 0.1m from the eye:
  // one enormous smeared quad across the screen. The erosion band is still
  // scaled by the whole PLANT's radius, so it only ever triggers for cards the
  // camera is practically touching.
  let plant_h1 = info.merged_y1 * scale;
  let h_ref = max(plant_h1, 1e-3);
  let to_plant = frame.camera_pos - base;
  let fade_r = info.merged_r * scale;
  let d_xz = length(to_plant.xz);
  let edge_fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);

  // Sub-clumps sway slightly out of step with each other — the strongest cue
  // that they are separate masses rather than one flat card.
  let spread = select(0.0, (f32(unit) - 0.5 * (f32(K_SUB) - 1.0)) * info.sway_spread, is_near);
  let sway = wind_sway(base, frame.time, entry.sway, phase + spread);

  var world: vec3f;
  var ly: f32;
  var d_card: f32;
  var fade = edge_fade;
  if (is_top) {
    ly = mix(h0, h1, info.top_frac);
    let off = vec3f(box.x + (2.0 * s - 1.0) * box.z, 0.0, box.y + (2.0 * tt - 1.0) * box.z) * scale;
    world = base + rot_y(off, yaw);
    world.y += ly;
    // The top card only makes sense from well above: at flatter views it reads
    // as a pale floating cutout among the side cards, and from BELOW it is the
    // classic "pale pancake overhead" artifact (the straight-down capture seen
    // from underneath). The elevation is therefore signed — negative dy erodes
    // the card away completely.
    let dy = frame.camera_pos.y - (base.y + ly);
    d_card = length(vec3f(to_cam.x, dy, to_cam.z));
    fade *= smoothstep(0.35, 0.6, dy / max(d_card, 1e-4));
  } else {
    ly = mix(h0, h1, 1.0 - tt);
    let right = vec3f(fwd_xz.y, 0.0, -fwd_xz.x);
    world = axis + right * (r * (2.0 * s - 1.0));
    world.y = base.y + ly;
    let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
    d_card = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  }
  // Side cards may come close — grass in your face is the point of the inside
  // view — but a horizontal top card that close is always a smeared disc, so
  // it erodes from twice the distance.
  let fade_lo = select(fade_r * 0.30, fade_r * 0.50, is_top);
  let fade_hi = select(fade_r * 0.80, fade_r * 1.30, is_top);
  fade *= smoothstep(fade_lo, fade_hi, d_card);
  let hf_plant = clamp(ly / h_ref, 0.0, 1.0);
  world += sway * hf_plant;

  // A card whose fade dropped below the alpha reference has erode > 1, so
  // EVERY one of its fragments discards; an empty baked tile has a zero-area
  // tight rect. Emit both behind the near plane instead of rasterizing quads
  // that are guaranteed to produce nothing.
  let a_ref = info.alpha_ref * select(1.0, info.near_alpha_bias, is_near);
  if (fade < a_ref || loc.z <= 0.0) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = vec2f(uv_rect.x + q.x * uv_rect.z, uv_rect.y + q.y * uv_rect.w);
  out.world = world;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = a_ref / max(fade, 1e-3);
  out.shade = mix(1.0 - info.bottom_shade, 1.0, hf_plant);
  out.up_ws = vec3f(0.0, 1.0, 0.0);
  out.shell = vec3f(0.0, -1.0, 0.0);
  return out;
}

// Octahedral decode, y-primary — same convention the clump bake stored.
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
  let nrm_tex = textureSample(card_normal, card_sampler, in.uv);
  let is_carpet = in.shell.x > 0.5;
  // A carpet fragment survives on this shell only if the cushion it sits under
  // actually reaches this height; the bottom shell's threshold is -1, so the
  // mat is closed from below and never opens a hole.
  if (alb.a < in.erode || (is_carpet && nrm_tex.z < in.shell.y)) {
    discard;
  }

  var n: vec3f;
  var shade = in.shade;
  if (is_carpet) {
    // Hemisphere pair, not octahedral: the top-down capture flipped every
    // normal into ny >= 0, so ny reconstructs exactly and the stored channels
    // survive mip filtering (an octahedral pair does not — it averages toward
    // straight up and lights the far field flat).
    let nx = nrm_tex.x * 2.0 - 1.0;
    let nz = nrm_tex.y * 2.0 - 1.0;
    let n_mesh = vec3f(nx, sqrt(max(1.0 - nx * nx - nz * nz, 0.0)), nz);
    // Lift the mesh-frame normal into the local GROUND frame, not just yaw it,
    // or a mat on a slope lights as if it were level. This is
    // plant_basis_from_up(up, yaw) inlined — the tile carries cos/sin of its
    // quarter turn rather than the angle.
    let up = normalize(in.up_ws);
    var t = vec3f(in.yaw_cs.x, 0.0, -in.yaw_cs.y);
    let proj = t - up * dot(up, t);
    t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
    n = t * n_mesh.x + up * n_mesh.y + cross(t, up) * n_mesh.z;
    // Cavity term: how far this texel sits BELOW its own neighbourhood — the
    // gaps between capitula are where light does not reach, and this is what
    // stops a closed 98%-cover mat from reading as a painted plane. Biased so
    // the dome tops clamp at 1 and the mean darkening stays around 12%. Per
    // TEXEL, not per shell, so it never bands at a shell boundary. Measuring
    // depth against an ABSOLUTE height instead (the obvious thing) put the mean
    // shade at 0.65 and made the whole bog 1.5x darker than 001's carpet.
    //
    // The local mean is mip 4 of the 0.18m aux tile, i.e. ~6mm across. An
    // explicit LOD needs no uniform control flow, so this tap stays inside the
    // carpet branch and never costs the grass path anything; it also fades out
    // by itself with distance, as both taps converge on the same mip.
    let h_local = textureSampleLevel(card_normal, card_sampler, in.uv, 4.0).z;
    let cav = clamp(0.75 + (nrm_tex.z - h_local) / atlas_info.carpet_box.w, 0.0, 1.0);
    shade = 1.0 - info.carpet_shade * (1.0 - cav);
  } else {
    // Rotate the mesh-frame normal by the plant's yaw.
    let n_mesh = oct_decode_card(nrm_tex.xy * 2.0 - 1.0);
    n = vec3f(
      in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
      n_mesh.y,
      -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
    );
  }
  // `shade` is a fake occlusion term, so it belongs to the light term: the
  // albedo view then shows the baked colour exactly as captured and the
  // lighting view shows sun+ambient x occlusion.
  var color = light_surface(alb.rgb * shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
