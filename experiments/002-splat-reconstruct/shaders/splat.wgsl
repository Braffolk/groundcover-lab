#include "src/wgsl/wind.wgsl"
#include "src/wgsl/hash.wgsl"
#include "./common.wgsl"

// Pass 2: half-res splat scatter. One indexed indirect draw per (stand entry,
// LOD); instances are the culled plant records, vertices enumerate that LOD's
// splats (4 unique corners per splat, 6 shared indices — a quad only has four
// corners, so shading six would waste a third of this pass's vertex work
// across the ~1M splats a frame draws). Each splat is an anisotropic quad spanned by
// its baked elongation axis and the view-perpendicular, dither-discarded to
// its elliptical footprint so depth stays exact. Outputs an albedo+height
// target and a normal+viewdepth target for the reconstruction pass.

struct DrawInfo {
  bounds_min: vec4f, // xyz = splat bounds min (plant-local), w = topH
  bounds_ext: vec4f, // xyz = splat bounds extent, w = max ra (m)
  misc: vec4f,       // x = max rb (m)
  idx: vec4u,        // splat offset, splat count, bucket base, entry index
}

@group(1) @binding(0) var<uniform> di: DrawInfo;
@group(1) @binding(1) var<storage, read> records: array<PlantRec>;
@group(1) @binding(2) var<storage, read> splats: array<vec4u>;
@group(1) @binding(3) var<uniform> params: SrParams;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) uv_hf: vec3f,      // ellipse uv, height fraction
  @location(2) normal_vz: vec4f,  // world normal, linear view depth
  @location(3) misc: vec2f,       // colorVar, per-splat rnd
}

fn rot_y(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let rec = records[di.idx.z + ii];
  // Indexed quads: 4 corners per splat, the index buffer spans the two
  // triangles (0,1,2, 1,3,2) — same triangles as an unindexed 6-vertex quad.
  let s = vi / 4u;
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let c = corners[vi % 4u];

  let w = splats[di.idx.x + s];
  let local = di.bounds_min.xyz + vec3f(
    f32(w.x & 0xffffu), f32(w.x >> 16u), f32(w.y & 0xffffu),
  ) / 65535.0 * di.bounds_ext.xyz;
  let albedo = vec3f(
    f32((w.y >> 16u) & 0xffu), f32(w.y >> 24u), f32(w.z & 0xffu),
  ) / 255.0;
  let cvar = f32((w.z >> 8u) & 0xffu) / 255.0;
  let n_local = sr_oct_decode(f32((w.z >> 16u) & 0xffu) / 255.0, f32(w.z >> 24u) / 255.0);
  let a_local = sr_oct_decode(f32(w.w & 0xffu) / 255.0, f32((w.w >> 8u) & 0xffu) / 255.0);
  let ra_q = f32((w.w >> 16u) & 0xffu) / 255.0;
  let rb_q = f32(w.w >> 24u) / 255.0;

  let ys = unpack2x16float(rec.ys);
  let yaw = ys.x;
  let scale = ys.y;
  let base_y = unpack2x16float(rec.pfy & 0xffffu).x;
  let phase = f32((rec.pfy >> 16u) & 0xffu) / 255.0 * 6.2831853;
  let fade = f32(rec.pfy >> 24u) / 255.0;

  let cy = cos(yaw);
  let sy = sin(yaw);
  let base = vec3f(rec.xz.x, base_y, rec.xz.y);
  let entry = stand_table[di.idx.w];

  // Plant fade dissolves whole splats (not pixels): a partially-faded plant
  // must never leave sparse dithered depth samples that occlude the field
  // behind it — that punches sky-colored holes into the reconstruction.
  let pid = hash2(bitcast<u32>(rec.xz.x), bitcast<u32>(rec.xz.y));
  let rnd = hash_f32(hash2(pid, s));
  if (rnd >= fade) {
    var dead: VOut;
    dead.pos = vec4f(2.0, 2.0, 2.0, 1.0); // zero-area, off-screen
    return dead;
  }

  let hf = clamp(local.y / max(di.bounds_min.w, 1e-3), 0.0, 1.0);
  let sway = wind_sway(base, frame.time, entry.sway, phase) * (hf * (0.55 + 0.45 * hf));
  let center = base + rot_y(local, cy, sy) * scale + sway;

  // Radii are sqrt-encoded against the per-species maxima.
  let ra = ra_q * ra_q * di.bounds_ext.w * scale * params.splat_scale;
  let rb = rb_q * rb_q * di.misc.x * scale * params.splat_scale;

  // Quad basis: baked elongation axis projected off the view direction
  // (min width rb when a blade is seen end-on), view-perpendicular second axis.
  let vdir = normalize(center - frame.camera_pos);
  let a_ws = rot_y(a_local, cy, sy);
  var a_p = a_ws - vdir * dot(a_ws, vdir);
  let la = length(a_p);
  var a_e: vec3f;
  if (la > 0.1) {
    a_e = a_p / la;
  } else {
    a_e = normalize(cross(vdir, vec3f(0.017, 1.0, 0.013)));
  }
  let b_e = cross(vdir, a_e);
  var ra_eff = max(ra * max(la, 0.35), rb);
  var rb_eff = rb;
  // Screen-space radius clamp: very close splats would otherwise cover
  // hundreds of pixels as one featureless blur-bomb. ~48px cap keeps the
  // camera-adjacent stems made of many small splats instead.
  let vz_c = -(frame.view * vec4f(center, 1.0)).z;
  let px_limit = 48.0 * vz_c / (frame.proj[1][1] * frame.viewport.y * 0.5);
  ra_eff = min(ra_eff, px_limit);
  rb_eff = min(rb_eff, px_limit);
  let world = center + a_e * (ra_eff * c.x) + b_e * (rb_eff * c.y);

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  // Per-splat brightness jitter: survives the reconstruction averaging that
  // would wash out pure per-pixel noise, and breaks up flat color masses.
  let jitter = 0.82 + 0.36 * hash_f32(hash2(pid, s ^ 0xa511e9b3u));
  out.color = albedo * jitter;
  out.uv_hf = vec3f(c, hf);
  let vz = -(frame.view * vec4f(world, 1.0)).z;
  out.normal_vz = vec4f(rot_y(n_local, cy, sy), vz);
  out.misc = vec2f(cvar, rnd);
  return out;
}

struct FragOut {
  @location(0) color: vec4f, // albedo rgb, height fraction
  @location(1) aux: vec4f,   // world normal, linear view depth
}

@fragment
fn fs_splat(in: VOut) -> FragOut {
  let r2 = dot(in.uv_hf.xy, in.uv_hf.xy);
  if (r2 > 1.0) { discard; }
  // Dithered coverage: soft elliptical falloff x plant fade, resolved by the
  // reconstruction pass into smooth alpha. Screen-hash keeps it static.
  let p = vec2u(in.pos.xy);
  let noise = hash_f32(hash2(p.x, p.y ^ 0x517cc1b7u));
  let cover = clamp((1.0 - r2) * 2.4, 0.0, 1.0);
  if (cover <= fract(noise + in.misc.y)) { discard; }

  var albedo = in.color;
  let vn = hash_f32(hash2(p.x * 3u + 1u, p.y ^ 0x9e3779b9u)) - 0.5;
  albedo = albedo * (1.0 + vn * in.misc.x * params.var_amt * 2.0);

  var out: FragOut;
  out.color = vec4f(albedo, in.uv_hf.z);
  out.aux = vec4f(in.normal_vz.xyz, in.normal_vz.w);
  return out;
}
