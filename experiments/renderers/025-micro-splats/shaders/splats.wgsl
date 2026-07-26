#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "./entry_info.wgsl"

// One micro-splat = one hard-edged elliptical micro-card standing exactly
// where its patch of the source mesh stood, in that patch's own PCA frame.
// Four vertices per splat, indexed; the whole level of one plant is a single
// instance. Nothing here is camera-facing at close range — the parallax,
// the view-dependent silhouette and the self-occlusion are all just real 3D
// geometry going through the normal depth buffer.

// 32 B, produced by cull.wgsl. Everything that is per-PLANT (yaw sin/cos, the
// wind displacement, the coverage fade) is resolved once there instead of
// being recomputed by every one of the plant's ~8192x4 vertex invocations.
struct PlantInst {
  pos_scale: vec4f,  // xyz = base, w = scale
  packed: vec4<u32>, // sway.xy | sway.z, fade | cos(yaw), sin(yaw) | unused
}

// 16 B: pos u16x3 | oct normal u8x2 | tangent angle u8 | half-extents u8x2 |
//       stamp 4b + coverage 4b | rgb u8x3 | canopy occlusion u8
struct Splat {
  w0: u32,
  w1: u32,
  w2: u32,
  w3: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<uniform> lodi: LodInfo;
@group(1) @binding(2) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(3) var<storage, read> splats: array<Splat>;
@group(1) @binding(4) var stamp_tex: texture_2d<f32>;
@group(1) @binding(5) var stamp_smp: sampler;

const STAMP_GRID: f32 = 4.0;
const STAMP_INSET: f32 = 0.0625; // half a texel of the smallest mip's tile
const MAX_SPLAT_PX: f32 = 170.0; // projected half-extent cap (camera-inside guard)

fn rot_y(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// Octahedral decode, y-primary — mirrors octDecode() in bake.ts.
fn oct_decode_splat(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

// Branch-free orthonormal basis (Duff et al. 2017) — the exact twin of onb()
// in bake.ts, evaluated on the same dequantized normal, so the baked tangent
// angle reconstructs the PCA major axis.
fn duff_basis(n: vec3f) -> mat2x3f {
  let sgn = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sgn + n.z);
  let b = n.x * n.y * a;
  return mat2x3f(
    vec3f(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x),
    vec3f(b, sgn + n.y * n.y * a, -n.y),
  );
}

// --- the ground the base pass actually DRAWS --------------------------------
// `terrain_height()` is a bilinear sample of a 0.5 m heightmap, and that is
// where the scatter puts every plant. It is NOT the surface you can see: the
// harness base pass (src/scene/terrain-draw.wgsl) rasterizes the ground as a
// TERRAIN_QUADS x TERRAIN_QUADS grid — the integer world lattice, 1 m quads,
// two triangles each — so between the lattice nodes the visible ground is a
// LINEAR triangle that runs up to 66 mm ABOVE the bilinear height (measured
// over the bog stand; mean overshoot 5 mm, >2 cm over ~11% of the area).
//
// A 1.2 m grass never notices. A Sphagnum mat is 70-90 mm tall and sits ON the
// sampled height, so most of it ends up INSIDE the drawn ground and is depth
// -rejected: before this, ~60% of the carpet was invisible in tile-shaped
// patches with 45-degree edges — the triangulation of the terrain, showing
// through the mat.
//
// So a carpet is conformed to the drawn triangle, evaluated PER SPLAT (rung 3).
// A per-tile plane fit of the same surface is not good enough here: a tile's
// geometry reaches 0.155 m from its centre and straddles the lattice, and
// extrapolating one triangle's plane that far mis-places the edge by up to
// 73 mm — as much as the mat is tall.
const TERRAIN_QUADS: f32 = 256.0;

struct Ground {
  h: f32,
  up: vec3f,
}

/// Height and facet up-vector of the drawn terrain triangle under `xz`.
/// Three lattice taps; the normal comes out of the same three heights, so this
/// costs 3 bilinear fetches, not 4.
fn ground_drawn(xz: vec2f) -> Ground {
  let step = frame.terrain_size / TERRAIN_QUADS;
  let half = frame.terrain_size * 0.5;
  let g = floor((xz + half) / step);
  let node = g * step - half;
  let f = (xz - node) / step;
  // Quad corners split (0,0)-(0,1)-(1,0) / (1,0)-(0,1)-(1,1), matching the
  // base pass's vertex order exactly, so the mat lands on the same plane the
  // rasterizer produced rather than near it.
  let h01 = terrain_height(node + vec2f(0.0, step));
  let h10 = terrain_height(node + vec2f(step, 0.0));
  var out: Ground;
  var dh: vec2f;
  if (f.x + f.y <= 1.0) {
    let h00 = terrain_height(node);
    dh = vec2f(h10 - h00, h01 - h00);
    out.h = h00 + dot(dh, f);
  } else {
    let h11 = terrain_height(node + vec2f(step));
    dh = vec2f(h11 - h01, h11 - h10);
    out.h = h11 + dot(dh, f - vec2f(1.0));
  }
  out.up = normalize(vec3f(-dh.x, step, -dh.y));
  return out;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) luv: vec2f,
  @location(2) world: vec3f,
  @location(3) @interpolate(flat) nrm: vec3f,
  @location(4) @interpolate(flat) minor: vec3f,
  @location(5) @interpolate(flat) albedo: vec3f,
  @location(6) @interpolate(flat) occ: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[u32(lodi.inst_base) + ii];
  let sp = splats[u32(lodi.splat_base) + (vi >> 2u)];
  let corner = vi & 3u;
  let c = vec2f(f32(corner & 1u) * 2.0 - 1.0, f32(corner >> 1u) * 2.0 - 1.0);

  let base = inst.pos_scale.xyz;
  let scale = inst.pos_scale.w;
  let sway_xy = unpack2x16float(inst.packed.x);
  let sway_z_fade = unpack2x16float(inst.packed.y);
  let yaw_cs = unpack2x16float(inst.packed.z);
  let sway = vec3f(sway_xy, sway_z_fade.x);
  let fade = sway_z_fade.y;

  // --- unpack the splat ------------------------------------------------------
  let q = vec3f(f32(sp.w0 & 0xffffu), f32(sp.w0 >> 16u), f32(sp.w1 & 0xffffu)) * (1.0 / 65535.0);
  let local = info.bmin.xyz + q * info.bext.xyz;
  let oct = vec2f(f32((sp.w1 >> 16u) & 0xffu), f32((sp.w1 >> 24u) & 0xffu)) * (2.0 / 255.0) - 1.0;
  let n_local = oct_decode_splat(oct);
  let ang = f32(sp.w2 & 0xffu) * (6.2831853 / 256.0);
  let qa = f32((sp.w2 >> 8u) & 0xffu) * (1.0 / 255.0);
  let qb = f32((sp.w2 >> 16u) & 0xffu) * (1.0 / 255.0);
  let stamp_i = (sp.w2 >> 24u) & 15u;
  let rho = max(f32((sp.w2 >> 28u) & 15u) * (1.0 / 15.0), 0.04);
  let albedo = vec3f(f32(sp.w3 & 0xffu), f32((sp.w3 >> 8u) & 0xffu), f32((sp.w3 >> 16u) & 0xffu)) * (1.0 / 255.0);
  let occ = f32((sp.w3 >> 24u) & 0xffu) * (1.0 / 255.0);

  let onb = duff_basis(n_local);
  let ca = cos(ang);
  let sa = sin(ang);
  let t1_local = ca * onb[0] + sa * onb[1];
  let t2_local = -sa * onb[0] + ca * onb[1];

  // --- place it on the plant -------------------------------------------------
  let cy = yaw_cs.x;
  let sy = yaw_cs.y;
  let hf = clamp((local.y - info.splat_y0) / max(info.splat_y1 - info.splat_y0, 1e-4), 0.0, 1.0);
  let off = rot_y(local * scale, cy, sy);
  var centre = base + off + sway * (hf * hf);
  var t1 = rot_y(t1_local, cy, sy);
  var t2 = rot_y(t2_local, cy, sy);
  var nrm = rot_y(n_local, cy, sy);

  // --- carpet: conform to the ground, PER SPLAT ------------------------------
  // Rung 3 of the ladder in CLAUDE.md. A mat must follow the terrain, and for a
  // TILED species a per-tile fit is not merely coarse but wrong: neighbouring
  // tiles fit different planes and crack apart at their shared edge. Here the
  // ground is re-queried under every splat, so the height is a pure function of
  // world xz and the surface stays continuous straight across tile borders —
  // 8192 conform points on the finest level, collapsing to one per tile (a
  // rigid tilt, rung 1) exactly where a tile is a handful of pixels anyway.
  // One terrain_sample = one bilinear fetch for BOTH the height and the normal.
  if (info.carpet_info.x > 0.5) {
    let pxz = base.xz + off.xz;
    let g = terrain_sample(pxz);
    let up = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
    let ground = plant_basis_from_up(up, 0.0);
    // Height comes from the ground under this splat; orientation is rotated
    // into the ground frame, so a sloped mat is lit as a sloped mat.
    centre = vec3f(pxz.x, g.x + off.y, pxz.y);
    t1 = ground * t1;
    t2 = ground * t2;
    nrm = ground * nrm;
  }

  let to_cam = frame.camera_pos - centre;
  let dist = max(length(to_cam), 0.02);

  var ha = qa * qa * lodi.size_a;
  var hb = qb * qb * lodi.size_b;
  // Far levels drop the stamp cut-out (its mips would just dissolve at 2px) and
  // become solid ellipses of the same covered AREA instead.
  // A little over-coverage (1.7x) because a real canopy at that range is opaque
  // by layering, and a solid ellipse of exactly rho x area leaves it speckled.
  let shrink = mix(1.0, sqrt(min(rho * 1.7, 1.0)), lodi.shrink_mix);
  ha = ha * scale * info.splat_scale * shrink;
  hb = hb * scale * info.splat_scale * shrink;

  // Far levels also roll the minor axis toward the eye around the (baked) major
  // axis, so a coarse patch can never turn exactly edge-on and vanish. The
  // major axis stays baked, which is what keeps the silhouette view-dependent.
  if (lodi.face_cam > 0.0) {
    let v_dir = to_cam / dist;
    let cr = cross(t1, v_dir);
    let l2 = length(cr);
    if (l2 > 1e-4) {
      let dir = select(-1.0, 1.0, dot(cr, t2) >= 0.0);
      t2 = normalize(mix(t2, cr * (dir / l2), lodi.face_cam));
    }
  }

  // Screen-space extent guards. Below ~1px a splat stops being geometry and
  // starts being noise, so grow it; above MAX_SPLAT_PX a single blade segment
  // the camera is standing inside would streak right across the frame, so cap
  // it (this only ever binds within ~0.5 m of the eye, where the technique is
  // allowed to break down anyway).
  let px_to_m = dist / max(info.px_scale, 1e-4);
  let min_extent = info.min_px * px_to_m;
  let max_extent = MAX_SPLAT_PX * px_to_m;
  ha = min(max(ha, min_extent), max_extent) * fade;
  hb = min(max(hb, min_extent), max_extent) * fade;

  let world = centre + t1 * (ha * c.x) + t2 * (hb * c.y);

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  let tile = vec2f(f32(stamp_i & 3u), f32(stamp_i >> 2u));
  let luv01 = clamp(c * 0.5 + 0.5, vec2f(STAMP_INSET), vec2f(1.0 - STAMP_INSET));
  out.uv = (tile + luv01) * (1.0 / STAMP_GRID);
  out.luv = c;
  out.world = world;
  out.nrm = nrm;
  out.minor = t2;
  out.albedo = albedo;
  out.occ = occ;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Exactly one texture tap, at every level. The stamp's mips are
  // coverage-corrected, so keeping the hard alpha cut all the way out is what
  // makes distant silhouettes feathery instead of blobby — the far field lives
  // or dies on that, and it is the cheapest fragment shader in the experiment.
  let tex = textureSample(stamp_tex, stamp_smp, in.uv);
  let a_ref = info.alpha_ref * (1.0 - lodi.solid);
  if (dot(in.luv, in.luv) > 1.0 || tex.a < a_ref) {
    discard;
  }

  // Thin foliage: one baked normal, mirrored to whichever face we are looking
  // at, bent across the minor axis so the micro-card reads as a curved blade
  // instead of a flat chip.
  let bent = normalize(in.nrm + in.minor * (in.luv.y * info.bulge));
  let v_dir = normalize(in.world - frame.camera_pos);
  let n = select(bent, -bent, dot(bent, v_dir) > 0.0);

  // Stamp colour is a modulation around a neutral 0.5, dilated only 4 texels
  // into empty space — where `neutral` says the ellipse went solid, fall back
  // to the splat's own baked mean instead of sampling that empty space.
  let alb = in.albedo * mix(tex.rgb * 2.0, vec3f(1.0), lodi.neutral);
  // Distance lifts the occlusion: a 3px plant shows only its sunlit shell.
  let occ = mix(in.occ, 1.0, lodi.ao_lift * 0.7);
  let ao = mix(1.0, occ, info.ao_strength);
  var color = light_surface(alb * ao, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  // Coverage = the stamp alpha this fragment resolved to; with a hard alpha
  // test that is also the alpha-test margin (below a_ref nothing survives).
  return vec4f(debug_shade(color, alb, n, tex.a, in.world), 1.0);
}
