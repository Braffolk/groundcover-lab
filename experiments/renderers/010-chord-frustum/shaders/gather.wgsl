#include "./frustum_math.wgsl"
#include "src/wgsl/gcmesh.wgsl"

// Bake stage 2: resample the ortho view G-buffers into the chord field.
// One thread per chord texel: build the bin's chord (entry/exit points on the
// frustum surface), pick the nearest baked view direction, and resolve the
// chord's first hit by projecting into that view with parallax-correction
// iterations (a fixed-point solve against the view's depth image — offline,
// so iteration count is irrelevant to runtime). 8 jittered parallel chords
// prefilter each bin: coverage becomes fractional (sub-mm fluff stays wispy
// instead of collapsing to solid), colors/normals/hits become means.
//
// Views are rendered in row-batches (memory); a thread only resolves its
// texel when its chord's view row is resident, so the two dispatches write
// disjoint texel sets that union to the full atlas.

const GV: f32 = 20.0;    // view grid (full-sphere octahedral)
const GT: f32 = 256.0;   // view tile px
const GROWS: f32 = 10.0; // resident view-grid rows per batch

struct GatherCfg {
  r0: f32, r1: f32, h: f32, side_len: f32,
  cap_b: f32, cap_t: f32, rb: f32, jr: f32,
  batch_row0: f32, _p0: f32, _p1: f32, _p2: f32,
}
@group(0) @binding(0) var<uniform> cfg: GatherCfg;
@group(0) @binding(1) var view_albedo: texture_2d<f32>;
@group(0) @binding(2) var view_oct: texture_2d<f32>;
@group(0) @binding(3) var view_depth: texture_2d<f32>;
@group(0) @binding(4) var chord_surf: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var chord_geom: texture_storage_2d<rgba8unorm, write>;

fn view_basis(fwd: vec3f) -> mat3x3f {
  var ref_up = vec3f(0.0, 1.0, 0.0);
  if (abs(fwd.y) > 0.99) { ref_up = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(ref_up, fwd));
  let up = cross(fwd, right);
  return mat3x3f(right, up, fwd);
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(CF_W) || gid.y >= u32(CF_H)) { return; }
  let px = i32(gid.x);
  let py = i32(gid.y);

  var fd: FrustumDims;
  fd.r0 = cfg.r0; fd.r1 = cfg.r1; fd.h = cfg.h;
  fd.side_len = cfg.side_len; fd.cap_b = cfg.cap_b; fd.cap_t = cfg.cap_t;
  let m_lo = chart_m_lo(fd);
  let m_range = chart_m_range(fd);

  let tile_w = i32(CF_A_X) + 1;
  let bi = px / tile_w;
  let ix = px % tile_w;
  let bj = py / i32(CF_M_X);
  let iy = py % i32(CF_M_X);

  let phi_e = (f32(bi) + 0.5) / CF_A_E * CF_TWO_PI;
  let m_e = m_lo + (f32(bj) + 0.5) / CF_M_E * m_range;
  let dphi = f32(ix) / CF_A_X * CF_TWO_PI; // ix == A_X duplicates ix == 0 (gutter)
  let m_x = m_lo + f32(iy) / (CF_M_X - 1.0) * m_range;

  let entry_p = point_of(phi_e, m_e, fd);
  let exit_p = point_of(phi_e + dphi, m_x, fd);
  let chord = exit_p - entry_p;
  let len = length(chord);

  if (len <= 1e-3) {
    // Degenerate zero-length chord: empty, owned by the first batch.
    if (cfg.batch_row0 < 0.5) {
      textureStore(chord_surf, vec2i(px, py), vec4f(0.0));
      textureStore(chord_geom, vec2i(px, py), vec4f(0.5, 0.5, 0.5, 0.0));
    }
    return;
  }

  let dir = chord / len;

  // Nearest baked view for this chord direction; skip if not in this batch.
  let e = oct_nrm_encode(dir);
  let cell = clamp(vec2f(floor((e * 0.5 + 0.5) * GV)), vec2f(0.0), vec2f(GV - 1.0));
  if (cell.y < cfg.batch_row0 || cell.y >= cfg.batch_row0 + GROWS) { return; }
  let fwd = oct_nrm_decode((cell + 0.5) / GV * 2.0 - 1.0);
  let basis = view_basis(fwd);
  let tile_org = vec2f(cell.x, cell.y - cfg.batch_row0) * GT;
  let cb = vec3f(0.0, cfg.h * 0.5, 0.0);

  // Perpendicular jitter basis for the 8-chord prefilter bundle.
  var jref = vec3f(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.9) { jref = vec3f(1.0, 0.0, 0.0); }
  let j1 = normalize(cross(dir, jref));
  let j2 = cross(dir, j1);

  var acc_rgb = vec3f(0.0);
  var acc_cov = 0.0;
  var acc_t = 0.0;
  var acc_n = vec3f(0.0);
  const N_SUB: u32 = 8u;
  for (var sc = 0u; sc < N_SUB; sc++) {
    // Deterministic disc-ish pattern of 8 offsets within the bin footprint.
    let ja = (f32(sc) + 0.5) / f32(N_SUB) * CF_TWO_PI;
    let jrad = select(0.3, 0.75, (sc & 1u) != 0u) * cfg.jr;
    let e0 = entry_p + (j1 * cos(ja) + j2 * sin(ja)) * jrad;

    // March the chord (2-texel steps, BAKE TIME ONLY) against the view's
    // first-hit depth image: a hit is the first step whose pixel's surface
    // lies inside a ~2.5-texel tube around the chord. This is robust where a
    // fixed-point parallax solve is not (fluffy discontinuous depth, long
    // near-vertical chords with a few degrees of view mismatch).
    let texel_w = 2.0 * cfg.rb / GT;
    let ray_thick = 2.5 * texel_w;
    let step_len = 2.0 * texel_w;
    let n_steps = min(u32(len / step_len) + 1u, 256u);
    for (var k = 0u; k <= n_steps; k++) {
      let s = min(f32(k) * step_len, len);
      let p = e0 + dir * s;
      let rel = p - cb;
      let u = dot(rel, basis[0]) / (2.0 * cfg.rb) + 0.5;
      let v = 0.5 - dot(rel, basis[1]) / (2.0 * cfg.rb);
      let texel = vec2i(clamp(vec2f(u, v) * GT, vec2f(0.0), vec2f(GT - 1.0)) + tile_org);
      let d01 = textureLoad(view_depth, texel, 0).x;
      if (d01 > 0.999) { continue; } // background pixel
      // Reconstruct the view ray's first hit at this pixel (texel center).
      let lx = ((f32(texel.x) - tile_org.x + 0.5) / GT - 0.5) * 2.0 * cfg.rb;
      let ly = (0.5 - (f32(texel.y) - tile_org.y + 0.5) / GT) * 2.0 * cfg.rb;
      let zf = (d01 - 0.5) * 2.0 * cfg.rb;
      let hit = cb + basis[0] * lx + basis[1] * ly + basis[2] * zf;
      if (distance(hit, p) < ray_thick) {
        let alb = textureLoad(view_albedo, texel, 0);
        if (alb.a > 0.5) {
          acc_rgb += alb.rgb;
          acc_cov += 1.0;
          acc_t += clamp(dot(hit - e0, dir) / len, 0.0, 1.0);
          // view_oct passes the SOURCE vertex pair through unchanged (bake_views
          // only requantizes it), so this is where the GCMESH1 convention applies.
          let src_oct = textureLoad(view_oct, texel, 0).xy;
          acc_n += gcmesh_normal_decode(src_oct.x, src_oct.y);
        }
        break;
      }
    }
  }

  var out_surf = vec4f(0.0);
  var out_geom = vec4f(0.5, 0.5, 0.5, 0.0);
  if (acc_cov > 0.0) {
    let inv = 1.0 / f32(N_SUB);
    out_surf = vec4f(acc_rgb * inv, acc_cov * inv); // premultiplied by coverage
    var n = acc_n / acc_cov;
    if (length(n) < 1e-3) { n = vec3f(0.0, 1.0, 0.0); }
    let oe = oct_nrm_encode(normalize(n)) * 0.5 + 0.5;
    out_geom = vec4f(acc_t / acc_cov, oe, 0.0);
  }
  textureStore(chord_surf, vec2i(px, py), out_surf);
  textureStore(chord_geom, vec2i(px, py), out_geom);
}
