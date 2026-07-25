// Carpet relief + horizon fit — the Fourier basis applied to the tile's
// GEOMETRY instead of to its appearance.
//
// The upright path fits a plant's *appearance* as a truncated Fourier series in
// view azimuth. That is wrong for a mat: a 0.18 m periodic cushion seen from a
// near-grazing angle parallax-shifts by more than a whole tile period, i.e. the
// appearance is high-frequency in azimuth and an order-3 fit collapses it into
// rotational mush (exactly the featureless blob the camera-facing card produced).
//
// What IS smooth in azimuth is the tile's HORIZON: the elevation angle of the
// highest obstruction seen from a point on the cushion looking out along azimuth
// theta. So this pass sweeps the captured height field and fits
//
//     beta(theta) ~ dc + c1*cos(theta) + s1*sin(theta)      [radians]
//
// per texel, plus the cosine-weighted sky visibility. One closed-form evaluation
// per fragment at runtime then gives (a) sun self-shadowing in the crevices
// between capitula and (b) view-dependent occlusion — which is what makes the
// mat read as a field of separate cushions instead of a photograph of one.
//
// It also replaces the SHADING NORMAL. The source mesh's own normals are leaf
// normals, and at 0.35 mm/texel a top view of Sphagnum resolves individual leaf
// tips pointing every which way: their local mean is ~zero, so any filtering (a
// bilinear tap, let alone a mip) leaves a tiny residual vector that normalize()
// blows up into an arbitrary near-horizontal direction — measured, that lit whole
// tiles almost black and made the mat flicker between four colours with the
// 90-degree tile yaw. The normal that actually means something for a cushion is
// the gradient of the CUSHION, so the shading normal here is the smoothed
// gradient of the height field, taken at the capitulum scale (a few mm).
//
// Wrap addressing everywhere: the tile is periodic and the capture is exactly
// periodic (3x3 lattice replication), so relief and horizon are seamless across
// tile boundaries by construction.

// Must match bake.ts (CARPET_TEX) and the loop constants below.
const TEX: i32 = 512;
const MASK: i32 = 511;      // TEX - 1, TEX is a power of two
const AZIMUTHS: u32 = 16u;
const STEPS: u32 = 20u;
const HALF_PI: f32 = 1.5707963;
// Central-difference radii for the cushion normal, in texels. 4 texels = 1.4 mm
// and 11 = 3.9 mm, i.e. below and around the capitulum radius: averaging the two
// keeps the dome shape and drops the leaf-tip noise.
const GRAD_R0: i32 = 4;
const GRAD_R1: i32 = 11;
// Box footprint (in texels) of every height tap. Sphagnum's top surface is a
// fuzz of 0.3-0.5 mm leaf tips over 5-15 mm capitula; the tips are below any
// scale this shading can resolve, and a raw single-texel tap lets one tip
// dominate a horizon (a 3 mm tip 1 texel away reads as an 83-degree wall).
// Averaging 3x3 taps 2 texels apart smooths at 0.7 mm and leaves the capitulum
// dome intact.
const TAP_SPREAD: i32 = 2;

struct HzInfo {
  // texel_m: metres per texel (tile_m / TEX); top_h: height normalisation.
  texel_m: f32, top_h: f32, _p0: f32, _p1: f32,
}
@group(0) @binding(0) var<uniform> hz: HzInfo;
@group(0) @binding(1) var capture_in: texture_2d<f32>;
@group(0) @binding(2) var relief_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var horizon_out: texture_storage_2d<rgba8unorm, write>;

/// Wrapped height (m) above the tile base. The capture stores it normalised by
/// top_h in alpha; empty texels are cleared to 0 = bare peat. The capture target
/// is rgba16float, NOT rgba8: at 8 bits one step is 0.36 mm, which over a
/// 0.35 mm texel step is a slope of 1.0 — the horizon sweep would have been
/// reading pure quantisation noise and saturating everywhere (measured: it cost
/// the whole mat ~20% brightness in spurious self-shadow).
fn height_at(t: vec2i) -> f32 {
  return textureLoad(capture_in, vec2u(t & vec2i(MASK)), 0).a * hz.top_h;
}

/// 3x3 box-averaged height at TAP_SPREAD spacing — see TAP_SPREAD.
fn height_box(t: vec2i) -> f32 {
  var s = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      s += height_at(t + vec2i(dx, dy) * TAP_SPREAD);
    }
  }
  return s * (1.0 / 9.0);
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let t0 = vec2i(gid.xy);
  if (t0.x >= TEX || t0.y >= TEX) { return; }
  let h0 = height_box(t0);

  // --- cushion normal from the height gradient at two scales -----------------
  var slope = vec2f(0.0);
  for (var k = 0; k < 2; k++) {
    let r = select(GRAD_R1, GRAD_R0, k == 0);
    let dx = height_box(t0 + vec2i(r, 0)) - height_box(t0 - vec2i(r, 0));
    let dz = height_box(t0 + vec2i(0, r)) - height_box(t0 - vec2i(0, r));
    slope += vec2f(dx, dz) / (2.0 * f32(r) * hz.texel_m);
  }
  slope = slope * 0.5;
  // Mesh-frame normal of the height field, upper hemisphere by construction.
  let n = normalize(vec3f(-slope.x, 1.0, -slope.y));

  // --- horizon sweep ---------------------------------------------------------
  var dc = 0.0;
  var c1 = 0.0;
  var s1 = 0.0;
  var sky = 0.0;
  for (var k = 0u; k < AZIMUTHS; k++) {
    let th = 6.2831853 * f32(k) / f32(AZIMUTHS);
    let ct = cos(th);
    let st = sin(th);
    // Log-spaced radii: the horizon of a bumpy surface is dominated by the
    // nearest few millimetres, so geometric spacing covers the whole capitulum
    // scale (~2 cm) in 20 taps where linear spacing would need 60.
    var r = 2.0;
    var tan_max = 0.0;
    for (var j = 0u; j < STEPS; j++) {
      let t = t0 + vec2i(i32(round(ct * r)), i32(round(st * r)));
      let dh = height_box(t) - h0;
      tan_max = max(tan_max, dh / (r * hz.texel_m));
      r = r * 1.25;
    }
    let beta = atan(tan_max);
    dc += beta;
    c1 += beta * ct;
    s1 += beta * st;
    let sb = sin(beta);
    sky += sb * sb;
  }
  let inv = 1.0 / f32(AZIMUTHS);
  dc = dc * inv;
  c1 = c1 * inv * 2.0;
  s1 = s1 * inv * 2.0;
  // Cosine-weighted sky visibility of the fitted horizon ring.
  let ao = clamp(1.0 - sky * inv, 0.0, 1.0);

  // relief: xz of the unit cushion normal (y is recovered as sqrt(1 - x^2 - z^2),
  // which is why a box-filtered mip of it correctly FLATTENS with distance
  // instead of drifting like an octahedral encoding would), sky visibility, and
  // the height itself.
  textureStore(
    relief_out,
    vec2u(gid.xy),
    vec4f(n.x * 0.5 + 0.5, n.z * 0.5 + 0.5, ao, clamp(h0 / hz.top_h, 0.0, 1.0)),
  );
  // horizon: angles normalised by pi/2; the two harmonics are signed, hence
  // biased. All channels are LINEAR in the fitted quantity, so mip-averaging
  // them is valid. Alpha is reserved (kept at 1 so a weighted mip is a no-op).
  textureStore(
    horizon_out,
    vec2u(gid.xy),
    vec4f(
      clamp(dc / HALF_PI, 0.0, 1.0),
      clamp(c1 / HALF_PI * 0.5 + 0.5, 0.0, 1.0),
      clamp(s1 / HALF_PI * 0.5 + 0.5, 0.0, 1.0),
      1.0,
    ),
  );
}
