// Visibility probe bake: one orthographic depth map per probe direction,
// packed into a tile atlas. Each texel stores the largest `dot(p - centre, d)`
// of any surface along that ray, i.e. the surface closest to a light placed at
// infinity along +d. The AO/bent-normal resolve then answers "is this point
// the first thing a ray from +d hits?" with a single texel fetch per direction
// — 32 lookups, no marching, entirely at bake time.

struct VisUni {
  r_axis: vec4f,  // xyz = probe right axis (plant-local)
  u_axis: vec4f,  // xyz = probe up axis
  d_axis: vec4f,  // xyz = toward the light
  sphere: vec4f,  // xyz = bounding-sphere centre (plant-local), w = radius
  anchor: vec4f,  // xyz = mesh-space origin of the plant-local frame
  b_min: vec4f,
  b_range: vec4f,
  spacing: vec4f, // xy = canopy spacing (m) for the 3x3 neighbour ring
}
@group(0) @binding(0) var<uniform> vis_uni: VisUni;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) along: f32,
}

// Drawn with 9 instances: the plant plus its 8 canopy neighbours, offset by the
// mesh's own periodic tile size (or, for a finite specimen, by its footprint).
// An occluder of P along the probe direction projects to the SAME (u,v) as P,
// so the neighbours need no extra ortho box — only the centre plant's box.
@vertex
fn vs(
  @location(0) q_pos: vec4<u32>,
  @location(1) q_attr: vec4<u32>,
  @builtin(instance_index) inst: u32,
) -> VOut {
  let neighbour = vec3f(
    (f32(inst % 3u) - 1.0) * vis_uni.spacing.x,
    0.0,
    (f32(inst / 3u) - 1.0) * vis_uni.spacing.y,
  );
  let p = vis_uni.b_min.xyz + (vec3f(vec3<u32>(q_pos.xyz)) / 65535.0) * vis_uni.b_range.xyz + neighbour;
  let o = (p - vis_uni.anchor.xyz) - vis_uni.sphere.xyz;
  let inv_r = 1.0 / vis_uni.sphere.w;
  let along = dot(o, vis_uni.d_axis.xyz);

  var out: VOut;
  // Closest to the light = largest `along` = smallest clip z (depthCompare less).
  out.pos = vec4f(
    dot(o, vis_uni.r_axis.xyz) * inv_r,
    dot(o, vis_uni.u_axis.xyz) * inv_r,
    clamp(0.5 - 0.5 * along * inv_r, 0.0, 1.0),
    1.0,
  );
  out.along = along;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) f32 {
  return in.along;
}
