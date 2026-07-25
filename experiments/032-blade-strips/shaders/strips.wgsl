#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./strip_info.wgsl"

// Blade-strip draw. One instance = one plant; the bucket's index count decides
// how many baked ribbons it unrolls (16/8/4/2/1) plus an optional canopy card.
//
// A ribbon is a 5-node curved strip: node positions come from the baked table
// (the bundle's real mean blade arc in the clump frame), the width axis is the
// baked lateral axis twisted `face_cam` of the way toward the camera, and the
// tile holds that bundle's real geometry unrolled along the same arc. The
// ribbons are world-anchored, so near ribbons genuinely shift against far ones
// as the camera moves and the silhouette is the union of real 3D arcs. Hard
// alpha test + depth write, no blending and no dither.

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
// 6 vec4 per baked strip: 5 x (node xyz, half-width) + (latX, latZ, depth, _).
@group(1) @binding(2) var<storage, read> strip_table: array<vec4f>;
@group(1) @binding(3) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(4) var atlas_normal: texture_2d<f32>;
@group(1) @binding(5) var atlas_sampler: sampler;

const NODES: u32 = 5u;
const VERTS_PER_RIBBON: u32 = 10u;
const TOP_SLOT: u32 = 31u;
const GRID_U: u32 = 16u; // atlas is 16 x 2 tiles of 128 x 512
const GRID_V: u32 = 2u;
const INSET_U: f32 = 0.015625; // 2 texels of a 128-wide tile
const INSET_V: f32 = 0.00390625; // 2 texels of a 512-tall tile

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,                         // tile-local
  @location(1) world: vec3f,
  @location(2) lat: vec3f,                        // ribbon width axis (world)
  @location(3) tan: vec3f,                        // ribbon arc tangent (world)
  @location(4) @interpolate(flat) slot: u32,      // atlas tile
  @location(5) @interpolate(flat) erode: f32,     // effective alpha reference
  @location(6) @interpolate(flat) flip_base: u32, // per-plant imagery mirror
  @location(7) @interpolate(flat) two_sided: f32, // 1 = ribbon, 0 = canopy card
  @location(8) shade: f32,                        // grounding gradient
  @location(9) @interpolate(flat) tint: f32,      // per-plant albedo jitter
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[u32(info.bucket.z) + ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (6.2831853 / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (6.2831853 / 1024.0);
  let base = inst.pos;
  let entry = stand_table[u32(info.ids.y)];

  let h0 = info.box.x * scale;
  let h1 = info.box.y * scale;
  let r = info.box.z * scale;
  // The baked capture is centred on the clump, which sits off the mesh origin —
  // reproduce that offset so imagery lands where the geometry was.
  let axis = base + rot_y(vec3f(info.clump.x, 0.0, info.clump.y) * scale, yaw);
  let to_cam = frame.camera_pos - axis;
  let sway = wind_sway(base, frame.time, entry.sway, phase);

  // Camera-inside fade: 3D distance to the plant's core segment. Ribbons
  // collapse in WIDTH (no dither, no alpha ramp) as the camera enters.
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let near_fade = smoothstep(r * 0.30, r * 1.05, d_core);
  // Region edge: erode to nothing exactly at region_r (the cull matches).
  let d_xz = length(to_cam.xz);
  let edge_fade = 1.0 - smoothstep(info.ids.z * 0.85, info.ids.z, d_xz);

  let hb = bits * 2654435761u;
  var out: VOut;
  out.tint = 1.0 + (f32((hb >> 8u) & 255u) / 255.0 - 0.5) * 2.0 * info.shade.w;
  out.flip_base = (hb >> 20u) & 1u;
  out.erode = info.clump.z / max(edge_fade, 1e-3);

  var world: vec3f;
  var alive = near_fade > 0.02 && edge_fade > info.clump.z;

  if (vi < 6u) {
    // --- horizontal canopy card (baked straight-down view) ------------------
    var corners = array<vec2f, 6>(
      vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
      vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    let c = corners[vi];
    let top_frac = info.shade.z;
    world = axis + rot_y(vec3f(c.x * r, 0.0, c.y * r), yaw);
    world.y = base.y + mix(h0, h1, top_frac) + sway.y * top_frac;
    world += vec3f(sway.x, 0.0, sway.z) * top_frac;
    // Only meaningful from well above: at flatter views a pale floating cutout.
    let dy = frame.camera_pos.y - world.y;
    let elev = abs(dy) / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);
    alive = alive && info.shade.y > 0.5 && elev > 0.5;
    out.uv = c * 0.5 + 0.5;
    out.slot = TOP_SLOT;
    out.lat = rot_y(vec3f(0.0, 0.0, 1.0), yaw);
    out.tan = rot_y(vec3f(1.0, 0.0, 0.0), yaw);
    out.two_sided = 0.0;
    out.shade = 1.0 - info.shade.x * 0.3;
    out.erode = max(out.erode, info.clump.z / max(smoothstep(0.5, 0.78, elev), 1e-3));
  } else {
    // --- curved blade-bundle ribbon ---------------------------------------
    let li = vi - 6u;
    let ribbon = li / VERTS_PER_RIBBON;
    let local = li % VERTS_PER_RIBBON;
    let node = local / 2u;
    let side = f32(local % 2u) * 2.0 - 1.0;
    let o = (u32(info.bucket.x) + ribbon) * 6u;
    let nd = strip_table[o + node];
    let prev = strip_table[o + max(node, 1u) - 1u];
    let next = strip_table[o + min(node + 1u, NODES - 1u)];
    let lat_local = strip_table[o + 5u].xy;

    // Node -> world, wind-bent by height fraction (tips move, roots do not).
    let hf = clamp(nd.y / max(info.box.y, 1e-3), 0.0, 1.0);
    let bend = pow(hf, 1.5);
    let node_ws = axis + rot_y(nd.xyz * scale, yaw) + sway * bend;

    // Arc tangent, tilted by the wind gradient d(sway)/dy.
    var tan_ws = rot_y(normalize(next.xyz - prev.xyz + vec3f(0.0, 1e-5, 0.0)), yaw);
    tan_ws = normalize(tan_ws + sway * (1.5 * sqrt(hf) / max(info.box.y * scale, 0.05)));

    // Width axis: baked lateral axis twisted toward the camera by face_cam.
    // Fully world-anchored ribbons vanish when seen edge-on; the twist keeps
    // coverage stable while the ARC stays anchored, so parallax survives.
    var lat_ws = rot_y(vec3f(lat_local.x, 0.0, lat_local.y), yaw);
    let view_dir = normalize(frame.camera_pos - node_ws);
    var lat_face = cross(tan_ws, view_dir);
    let flen = length(lat_face);
    if (flen > 1e-4) {
      lat_face = lat_face / flen;
      if (dot(lat_face, lat_ws) < 0.0) {
        lat_face = -lat_face;
      }
      lat_ws = normalize(mix(lat_ws, lat_face, info.clump.w * smoothstep(0.08, 0.35, flen)));
    }
    lat_ws = normalize(lat_ws - tan_ws * dot(lat_ws, tan_ws));

    world = node_ws + lat_ws * (side * nd.w * scale * near_fade);
    out.uv = vec2f(f32(local % 2u), 1.0 - f32(node) / f32(NODES - 1u));
    out.slot = u32(info.bucket.x) + ribbon;
    out.lat = lat_ws;
    out.tan = tan_ws;
    out.two_sided = 1.0;
    out.shade = mix(1.0 - info.shade.x, 1.0, hf);
  }

  // A strip whose fade dropped out has every fragment discarded, so emit it
  // behind the near plane instead of rasterizing a guaranteed-empty quad.
  if (alive) {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  } else {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  }
  out.world = world;
  return out;
}

@fragment
fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  // Back faces show the mirrored capture: u flips, and with it the lateral
  // component of the stored normal; the face component flips with the side.
  let back = select(1.0, 0.0, front) * in.two_sided;
  let flip = abs(f32(in.flip_base) - back);
  let u = mix(in.uv.x, 1.0 - in.uv.x, flip);
  let tile = vec2f(f32(in.slot % GRID_U), f32(in.slot / GRID_U));
  let uv = (tile + vec2f(clamp(u, INSET_U, 1.0 - INSET_U), clamp(in.uv.y, INSET_V, 1.0 - INSET_V)))
    / vec2f(f32(GRID_U), f32(GRID_V));

  // Sample before any non-uniform discard (uniform-control-flow rule). The
  // normal tap is skipped for the flat far bucket — the branch is on a uniform
  // value, so it stays uniform control flow.
  let alb = textureSample(atlas_albedo, atlas_sampler, uv);
  var enc = vec2f(0.0);
  if (info.bucket.w < 0.5) {
    enc = textureSample(atlas_normal, atlas_sampler, uv).xy * 2.0 - 1.0;
  }
  if (alb.a < in.erode) {
    discard;
  }

  // Rebuild the ribbon tangent frame and lift the stored tangent-frame normal
  // into world space. The face component is implied: the bake flipped every
  // normal toward the capture side, so it is never negative.
  let lat = normalize(in.lat);
  let tangent = normalize(in.tan);
  let face = normalize(cross(lat, tangent));
  let t2 = cross(face, lat);
  let n_face = sqrt(max(0.0, 1.0 - dot(enc, enc)));
  let n = normalize(t2 * enc.x + lat * (enc.y * (1.0 - 2.0 * flip)) + face * (n_face * (1.0 - 2.0 * back)));

  let albedo = alb.rgb * in.tint;
  // in.shade is a fake grounding occlusion, so it belongs to the light term:
  // the albedo view then shows the baked capture exactly as it was taken.
  var color = light_surface(albedo * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, alb.a, in.world), 1.0);
}
