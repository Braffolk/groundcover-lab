// Shared math of the baked ray-answer field ("ray LUT"). The atlas is a
// RF_VIEWS x RF_VIEWS grid of orthographic ray-bundle answers: cell (i,j)
// holds every precomputed ray with direction oct_decode((ij+0.5)/RF_VIEWS),
// resolved over a RF_SLAB x RF_SLAB grid of ray offsets across the plant's
// bounding sphere. TS mirror of the direction/basis math lives in bake.ts —
// keep them in sync.

const RF_VIEWS: f32 = 24.0;
const RF_SLAB: f32 = 64.0;

fn rf_sgn(x: f32) -> f32 {
  return select(-1.0, 1.0, x >= 0.0);
}

// Full-sphere octahedral direction <-> [0,1]^2. Same fold as the GCMESH1
// normal encoding, so one decoder serves both directions and baked normals.
fn rf_oct_encode(d: vec3f) -> vec2f {
  let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
  var e = n.xz;
  if (n.y < 0.0) {
    e = (vec2f(1.0) - abs(vec2f(n.z, n.x))) * vec2f(rf_sgn(n.x), rf_sgn(n.z));
  }
  return e * 0.5 + 0.5;
}

fn rf_oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var n = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
  if (n.y < 0.0) {
    n = vec3f((1.0 - abs(f.y)) * rf_sgn(f.x), n.y, (1.0 - abs(f.x)) * rf_sgn(f.y));
  }
  return normalize(n);
}

// Deterministic per-view slab basis. Must match viewBasis() in bake.ts.
fn rf_view_right(dv: vec3f) -> vec3f {
  if (abs(dv.y) > 0.999) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return normalize(cross(vec3f(0.0, 1.0, 0.0), dv));
}

// Atlas uv for slab coordinates sp (metres along r_ax/u_ax from the sphere
// center) inside view cell `cellf`. Half-texel inset stops bilinear bleeding
// across neighbouring view cells.
fn rf_atlas_uv(cellf: vec2f, sp: vec2f, radius: f32) -> vec2f {
  let hp = 0.5 / RF_SLAB;
  let suv = clamp(sp / (2.0 * radius) + vec2f(0.5), vec2f(hp), vec2f(1.0 - hp));
  return (cellf + suv) / RF_VIEWS;
}
