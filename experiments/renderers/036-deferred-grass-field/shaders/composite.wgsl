// Stage B of the bake: compose the periodic canopy patch for ONE direction.
//
// The patch's plants come from the ACTIVE STAND's own scatter (positions,
// scales, yaws, species mix — nothing invented here). Each plant contributes
// its pre-baked sheared-ortho stamp, placed by an exact affine rule:
//
//   image(plant at p, yaw psi, scale o, direction az) =
//       p.xz + o * R(az) * stamp_{yaw = psi - az}(local)
//
// so ONE stamp library serves every azimuth. The depth test (frag_depth = the
// hit height) resolves which plant the ray meets first, and copies at integer
// tile offsets make the answer periodic — which is what lets a ray travel
// arbitrarily far horizontally inside a finite table.

struct DirCfg {
  az: vec4f,        // cos az, sin az, shear s, canopy H
  tiling: vec4f,    // tile metres, copies per axis (2W+1), W, unused
}
struct EntryCfg {
  rect: vec4f,      // stamp rect u0, u1, v0, v1 (metres, scale 1)
  atlas: vec4f,     // tile w, tile h, atlas w, atlas h (texels)
  info: vec4f,      // plant count, species topH, yaw steps, unused
  grid: vec4f,      // atlas columns, gutter, unused, unused
}
@group(0) @binding(0) var<uniform> dir: DirCfg;
@group(0) @binding(1) var<uniform> entry_cfg: EntryCfg;
@group(0) @binding(2) var<storage, read> plants: array<vec4f>; // x, z, scale, yaw
@group(0) @binding(3) var stamp_surf: texture_2d<f32>;
@group(0) @binding(4) var stamp_geom: texture_2d<f32>;
@group(0) @binding(5) var stamp_samp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) hscale: f32, // metres of hit height per unit of stamp h01
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let count = u32(entry_cfg.info.x);
  let per_axis = u32(dir.tiling.y);
  let plant_i = ii % count;
  let copy = ii / count;
  let w = i32(dir.tiling.z);
  let ox = f32(i32(copy % per_axis) - w) * dir.tiling.x;
  let oz = f32(i32(copy / per_axis) - w) * dir.tiling.x;

  let pl = plants[plant_i];
  let scale = pl.z;
  let yaw = pl.w;

  // Quantised local yaw: psi - az, wrapped into the stamp library's grid.
  let steps = entry_cfg.info.z;
  let az_ang = atan2(dir.az.y, dir.az.x);
  var local_yaw = (yaw - az_ang) / 6.2831853 * steps;
  local_yaw = local_yaw - floor(local_yaw / steps) * steps;
  let yaw_idx = u32(floor(local_yaw + 0.5)) % u32(steps);

  // Stamp rect corner -> local sheared image coords -> field metres.
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
  );
  let c = corners[vi];
  let r = entry_cfg.rect;
  // The stamp library is baked against the GROUND plane; the field table is
  // indexed by the ray's ENTRY into the canopy top (y = H). For a fixed
  // direction that is a pure translation of -s*H along +u, so the stamp images
  // are reused unchanged and only their placement shifts. Note it does NOT
  // scale with the plant: H is the table's global canopy height.
  let entry_shift = dir.az.z * dir.az.w;
  let su = mix(r.x, r.y, c.x) * scale - entry_shift;
  let sv = mix(r.z, r.w, c.y) * scale;
  // Rotate the stamp frame by the azimuth: R(az) * (su, sv).
  let fx = pl.x + ox + su * dir.az.x - sv * dir.az.y;
  let fz = pl.y + oz + su * dir.az.y + sv * dir.az.x;

  let tile = dir.tiling.x;
  var out: VOut;
  out.pos = vec4f(fx / tile * 2.0 - 1.0, -(fz / tile * 2.0 - 1.0), 0.0, 1.0);

  // Atlas coords for this corner (tiles laid out in a square grid, gutter 2).
  let gut = entry_cfg.grid.y;
  let tw = entry_cfg.atlas.x;
  let th = entry_cfg.atlas.y;
  let cols = u32(entry_cfg.grid.x);
  let col = f32(yaw_idx % cols);
  let row = f32(yaw_idx / cols);
  let origin = vec2f(col * (tw + gut * 2.0) + gut, row * (th + gut * 2.0) + gut);
  out.uv = (origin + c * vec2f(tw, th)) / entry_cfg.atlas.zw;
  out.hscale = scale * entry_cfg.info.y / dir.az.w;
  return out;
}

struct FOut {
  @location(0) surf: vec4f,
  @location(1) geom: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VOut) -> FOut {
  let s0 = textureSampleLevel(stamp_surf, stamp_samp, in.uv, 0.0);
  if (s0.a < 0.5) { discard; }
  let s1 = textureSampleLevel(stamp_geom, stamp_samp, in.uv, 0.0);
  let inv = 1.0 / max(s0.a, 1e-3);
  let albedo = s0.rgb * inv;
  let nx = (s1.r * inv - 0.5) * 2.0;
  let nz = (s1.g * inv - 0.5) * 2.0;
  let h01_local = s1.b * inv;
  // Rotate the stamp's normal into the field frame by the azimuth.
  let rnx = nx * dir.az.x - nz * dir.az.y;
  let rnz = nx * dir.az.y + nz * dir.az.x;
  let h01 = clamp(h01_local * in.hscale, 0.0, 1.0);

  var out: FOut;
  out.surf = vec4f(albedo, 1.0);
  out.geom = vec4f(rnx * 0.5 + 0.5, rnz * 0.5 + 0.5, h01, 1.0);
  out.depth = 1.0 - h01;
  return out;
}
