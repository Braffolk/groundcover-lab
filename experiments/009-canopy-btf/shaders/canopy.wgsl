#include "src/wgsl/terrain.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"

// Canopy BTF runtime. The scene's groundcover is drawn as ONE plant-count-free
// proxy: a camera-centred terrain-following grid rasterised twice (ground
// layer + canopy-top layer). Every fragment finds its view ray's ground hit G
// and reconstructs the aggregate stand appearance there from the per-species
// baked BTF: 4 hemi-oct view bins are sampled (each with its own parallax
// reprojection to the ground plane) and blended; species are composited
// front-to-back by their reconstructed hit distances. Coverage is resolved by
// dithered discard so depth stays writeable and order-independent.

const BINS: f32 = 5.0;
const ATLAS_PX: f32 = 1360.0;
const CELL_PX: f32 = 272.0;
const CONTENT_PX: f32 = 256.0;
const BORDER_PX: f32 = 8.0;
const INNER_CELLS: f32 = 32.0; // uniform-spacing half-width of the grid

struct EntryU {
  tile: f32,        // world metres per BTF tile (already scale-adjusted)
  top_h: f32,       // world canopy top of this entry (m)
  s_mean: f32,      // mean plant scale (info)
  sway: f32,        // stand wind response
  alpha_scale: f32, // density modulation vs the baked reference
  species: f32,     // catalog species index -> texture slot
  avg_r: f32,
  avg_g: f32,
  avg_b: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
}

struct U {
  snap: vec2f,
  spacing: f32,
  cells: f32,
  region: f32,
  shell_top: f32,
  entry_count: f32,
  taps: f32,
  wind_amount: f32,
  coverage: f32,
  macro_tint: f32,
  _pad0: f32,
  entries: array<EntryU, 4>,
  bin_dirs: array<vec4f, 32>,
}

@group(1) @binding(0) var<uniform> u: U;
@group(1) @binding(1) var btf0: texture_2d_array<f32>;
@group(1) @binding(2) var btf1: texture_2d_array<f32>;
@group(1) @binding(3) var btf2: texture_2d_array<f32>;
@group(1) @binding(4) var samp: sampler;

// --- helpers ---------------------------------------------------------------

fn oct_decode8(e: vec2f) -> vec3f {
  let s = e * 2.0 - vec2f(1.0);
  var x = s.x;
  var z = s.y;
  let y = 1.0 - abs(s.x) - abs(s.y);
  if (y < 0.0) {
    x = (1.0 - abs(s.y)) * sign(s.x);
    z = (1.0 - abs(s.x)) * sign(s.y);
  }
  return normalize(vec3f(x, y, z));
}

fn hemioct_encode(d: vec3f) -> vec2f {
  let s = 1.0 / (abs(d.x) + d.y + abs(d.z));
  let a = d.x * s;
  let b = d.z * s;
  return vec2f(a + b, a - b);
}

fn ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn sample_a(si: u32, uv: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  switch si {
    case 1u: { return textureSampleGrad(btf1, samp, uv, 0u, gx, gy); }
    case 2u: { return textureSampleGrad(btf2, samp, uv, 0u, gx, gy); }
    default: { return textureSampleGrad(btf0, samp, uv, 0u, gx, gy); }
  }
}

fn sample_b(si: u32, uv: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  switch si {
    case 1u: { return textureSampleGrad(btf1, samp, uv, 1u, gx, gy); }
    case 2u: { return textureSampleGrad(btf2, samp, uv, 1u, gx, gy); }
    default: { return textureSampleGrad(btf0, samp, uv, 1u, gx, gy); }
  }
}

// Non-linear grid spacing: 1:1 for the inner INNER_CELLS, exponential beyond,
// reaching ~1.3 km at the grid edge (past fog saturation, inside the far
// plane). Per-frame vertex count is constant regardless of plant count.
fn grid_offset(i: f32) -> f32 {
  let x = i - u.cells * 0.5;
  let a = abs(x);
  let inner = min(a, INNER_CELLS);
  let outer = max(a - INNER_CELLS, 0.0);
  return sign(x) * u.spacing * inner * pow(1.115, outer);
}

// --- vertex ----------------------------------------------------------------

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) layer: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) layer: u32) -> VOut {
  let quad = vi / 6u;
  let corner = vi % 6u;
  let n = u32(u.cells);
  let cx = quad % n;
  let cz = quad / n;
  var co = array<vec2u, 6>(
    vec2u(0u, 0u), vec2u(1u, 0u), vec2u(0u, 1u),
    vec2u(1u, 0u), vec2u(1u, 1u), vec2u(0u, 1u),
  );
  let c = co[corner];
  let x = u.snap.x + grid_offset(f32(cx + c.x));
  let z = u.snap.y + grid_offset(f32(cz + c.y));
  let y = terrain_height(vec2f(x, z)) + select(0.02, u.shell_top, layer == 1u);

  var out: VOut;
  out.world = vec3f(x, y, z);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  out.layer = f32(layer);
  return out;
}

// --- fragment --------------------------------------------------------------

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let cam = frame.camera_pos;
  var v = in.world - cam;
  v = v / max(length(v), 1e-5);

  // Ground hit along the ray: two fixed refinement steps against the
  // heightfield (a lookup, not a march). Bottom-layer fragments ARE the
  // ground; top-layer fragments project forward along the ray.
  var G = in.world;
  var hg = terrain_height(G.xz);
  if (in.layer > 0.5) {
    for (var it = 0; it < 2; it++) {
      let t = clamp((G.y - hg) / max(-v.y, 1e-4), 0.0, 4000.0);
      G += v * t;
      hg = terrain_height(G.xz);
    }
  }
  G.y = hg;
  let tG = distance(cam, G);

  // Screen-space footprint of the ground point (for mip selection) — computed
  // before any divergence.
  let dGx = dpdx(G.xz);
  let dGy = dpdy(G.xz);

  // A top-layer fragment whose ray lands almost straight below is redundant:
  // the ground layer produces the identical result at the identical depth, so
  // skip the expensive lookups (big win at steep view angles).
  if (in.layer > 0.5 && distance(G.xz, in.world.xz) < 2.5) { discard; }

  // View direction for bin selection (surface -> camera), elevation-clamped.
  var dv = cam - G;
  dv = dv / max(length(dv), 1e-5);
  dv = normalize(vec3f(dv.x, max(dv.y, 0.06), dv.z));
  let p = hemioct_encode(dv);
  let gc = clamp((p * 0.5 + vec2f(0.5)) * (BINS - 1.0), vec2f(0.0), vec2f(BINS - 1.0));

  let region_d = max(abs(G.x), abs(G.z));
  let edge = 1.0 - smoothstep(u.region - 1.5, u.region, region_d);

  var cols: array<vec3f, 4>;
  var alphas: array<f32, 4>;
  var ts: array<f32, 4>;
  let ec = min(u32(u.entry_count), 4u);

  for (var e = 0u; e < 4u; e++) {
    alphas[e] = 0.0;
    ts[e] = 1e9;
    if (e >= ec || edge <= 0.001) { continue; }
    let ent = u.entries[e];
    let si = u32(ent.species);
    let sc = ent.tile;

    // Aggregate wind: the whole texel column is advected upstream by the
    // shared sway field at ~mid-canopy weight.
    let sw = wind_sway(G, frame.time, ent.sway, 0.0) * (0.62 * u.wind_amount);

    let gscale = CONTENT_PX / ATLAS_PX / sc;
    let gx = dGx * gscale;
    let gy = dGy * gscale;
    // Footprint in atlas texels; past the last mip we fade detail out, and we
    // also use it to damp parallax (a mip-averaged tile has no height left).
    let foot = max(length(gx), length(gy)) * ATLAS_PX;
    let lodX = clamp((log2(max(foot, 1e-4)) - 3.0) * 0.5, 0.0, 1.0);

    // Tap set. Bilinear blending of view bins ghosts the PERIODIC tile into
    // moiré fringes, so the default resolves the blend stochastically: one tap
    // whose bin is picked per-pixel with probability = the bilinear weight.
    var i0 = clamp(floor(gc), vec2f(0.0), vec2f(BINS - 2.0));
    var f = gc - i0;
    f = f * f * (3.0 - 2.0 * f); // C1 blend kills mach-band rings between bins
    if (u.taps < 2.0) {
      // nearest ('1') or stochastic ('dither', taps==0)
      if (u.taps < 0.5) {
        let n1 = ign(in.pos.xy + vec2f(13.0, 41.0));
        let n2 = ign(in.pos.xy + vec2f(59.0, 17.0));
        i0 = clamp(i0 + vec2f(select(0.0, 1.0, n1 < f.x), select(0.0, 1.0, n2 < f.y)),
          vec2f(0.0), vec2f(BINS - 1.0));
      } else {
        i0 = clamp(round(gc), vec2f(0.0), vec2f(BINS - 1.0));
      }
      f = vec2f(0.0);
    }
    let wts = vec4f(
      (1.0 - f.x) * (1.0 - f.y),
      f.x * (1.0 - f.y),
      (1.0 - f.x) * f.y,
      f.x * f.y,
    );

    // Parallax anchor at the RECONSTRUCTED content height: a cheap pre-tap at
    // the ground anchor estimates the local hit height, and the real taps then
    // reproject about that height. Anchoring all bins at the height the
    // content actually sits at makes their periodic copies line up — a fixed
    // mid-canopy anchor produces moiré fringes between neighbouring bins.
    let ni = clamp(round(gc), vec2f(0.0), vec2f(BINS - 1.0));
    let nidx = u32(ni.y) * u32(BINS) + u32(ni.x);
    let fr0 = fract((G.xz - sw.xz) / sc);
    let auv0 = (ni * CELL_PX + vec2f(BORDER_PX) + fr0 * CONTENT_PX) / ATLAS_PX;
    let h_est = sample_b(si, auv0, gx, gy).r * (1.0 - lodX);

    let h_anchor = hg + h_est * ent.top_h;
    var X = G;
    if (cam.y > h_anchor && v.y < -0.02) {
      X = cam + v * ((cam.y - h_anchor) / (-v.y));
    }
    let Xw = X.xz - sw.xz;

    var accC = vec3f(0.0);
    var accA = 0.0;
    var accH = 0.0;
    var accN = vec3f(0.0);
    var accSig = 0.0;

    let ntaps = select(1u, 4u, u.taps >= 2.0);
    for (var t = 0u; t < ntaps; t++) {
      let w = select(1.0, wts[t], ntaps == 4u);
      if (w < 1e-4) { continue; }
      let bi = i0 + vec2f(f32(t & 1u), f32(t >> 1u));
      let idx = u32(bi.y) * u32(BINS) + u32(bi.x);
      let d = u.bin_dirs[idx].xyz;
      // Reproject the anchor to the ground plane along this bin's direction.
      let q = Xw - ((X.y - hg) / d.y) * d.xz;
      let uvt = q / sc;
      let fr = fract(uvt);
      let auv = (bi * CELL_PX + vec2f(BORDER_PX) + fr * CONTENT_PX) / ATLAS_PX;
      let a = sample_a(si, auv, gx, gy);
      let b = sample_b(si, auv, gx, gy);
      let wa = w * a.a;
      accC += a.rgb * wa;
      accA += wa;
      accH += b.r * wa;
      accN += oct_decode8(b.gb) * wa;
      accSig += b.a * wa;
    }

    if (accA < 1e-3) { continue; }
    var col = accC / accA;
    var alpha = accA;
    let hN = accH / accA;
    var nrm = normalize(accN);
    let sigma = accSig / accA;

    // Distant detail fade: past the last mip, converge to the baked mean.
    col = mix(col, vec3f(ent.avg_r, ent.avg_g, ent.avg_b), lodX);
    nrm = normalize(mix(nrm, vec3f(0.0, 1.0, 0.0), lodX * 0.7));

    // Reconstructed hit along the actual view ray.
    let hitY = hg + hN * ent.top_h;
    let tH = clamp((cam.y - hitY) / max(-v.y, 1e-4), 0.1, tG * 1.5 + 60.0);
    let H = cam + v * tH;

    // Camera-inside-canopy fade (the sanctioned break-down case).
    alpha *= smoothstep(0.12, 0.55, tH);
    alpha *= ent.alpha_scale * u.coverage * edge;

    // Macro variation: per 2-tile cell brightness jitter breaks periodicity.
    var tint = 1.0;
    if (u.macro_tint > 0.5) {
      let ci = vec2i(floor(Xw / (sc * 2.0)));
      let hsh = hash3(bitcast<u32>(ci.x), bitcast<u32>(ci.y), 77u + e);
      tint = 0.86 + 0.28 * hash_f32(hsh);
    }
    // Statistical sparkle: baked luminance sigma re-injected as spatial noise,
    // faded out with distance (footprint) so it never shimmers.
    let cellN = vec2i(floor(Xw * 96.0));
    let spark = (hash_f32(hash2(bitcast<u32>(cellN.x), bitcast<u32>(cellN.y))) - 0.5)
      * sigma * 1.6 * (1.0 - lodX) * clamp(1.0 - foot * 0.25, 0.0, 1.0);

    var lit = light_surface(col * tint * (1.0 + spark), nrm, H);
    // Gust shimmer: cheap brightness ripple advected along the wind.
    let shim = 1.0 + 0.05 * u.wind_amount * ent.sway
      * sin(frame.time * 3.1 + dot(G.xz, frame.wind_dir) * 1.4);
    cols[e] = apply_fog(lit * shim, H);
    alphas[e] = clamp(alpha, 0.0, 1.0);
    ts[e] = tH;
  }

  // Front-to-back composite of the (up to 4) species layers.
  var order = array<u32, 4>(0u, 1u, 2u, 3u);
  for (var i = 0u; i < 3u; i++) {
    for (var j = i + 1u; j < 4u; j++) {
      if (ts[order[j]] < ts[order[i]]) {
        let tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
    }
  }
  var C = vec3f(0.0);
  var A = 0.0;
  var depth_t = -1.0;
  var first_t = -1.0;
  for (var i = 0u; i < 4u; i++) {
    let e = order[i];
    let a = alphas[e];
    if (a <= 0.002) { continue; }
    C += (1.0 - A) * a * cols[e];
    if (first_t < 0.0) { first_t = ts[e]; }
    if (depth_t < 0.0 && a > 0.25) { depth_t = ts[e]; }
    A += (1.0 - A) * a;
  }

  if (A < 0.004) { discard; }
  // A dense stand reads near-opaque: lift mid coverage before the dither so
  // the stipple only appears at genuine edges/thin spots.
  A = 1.0 - pow(1.0 - A, 1.6);
  // Dithered coverage: keeps the pass opaque (depth-writing, order-free).
  if (A < ign(in.pos.xy)) { discard; }

  if (depth_t < 0.0) { depth_t = first_t; }
  let H = cam + v * depth_t;
  let hp = frame.view_proj * vec4f(H, 1.0);

  var out: FOut;
  out.color = vec4f(C / A, 1.0);
  out.depth = clamp(hp.z / max(hp.w, 1e-5), 0.0, 0.999999);
  return out;
}
