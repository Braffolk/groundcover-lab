#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./entry_info.wgsl"

// Part-assembled plants. A plant inside the LOD radius is not one card: it is
// its OWN parts, each drawn as a small camera-facing card standing at the part's
// true 3-D position inside the plant, textured with that part's own baked view.
// The parts partition the source mesh, so the union of their alpha-tested
// cutouts is the plant's real silhouette from any direction — and because the
// positions are real 3-D and only the images are quantized, the silhouette
// reshapes and the interior parallaxes as the camera moves instead of rotating
// rigidly.
//
// Near draw (draw_level = 1), 15 cards = 90 vertices per plant:
//   cards 0..11  — upright part cards (3 bands x 4 sectors), emitted NEAREST
//                  SECTOR FIRST so the far side of the clump is early-z
//                  rejected instead of shaded; in the outer half of the LOD
//                  ring the far sector is dropped entirely.
//   cards 12..14 — one horizontal fill card per height band, at the band's mean
//                  height; edge-on (free) at grazing angles, it takes over as
//                  the camera rises and gives top-down views three real layers.
// Far draw (draw_level = 0), 2 cards = 12 vertices per plant: the classic
// whole-plant billboard pair (nearest of 8 azimuths + top card).
//
// Hard alpha test with depth write, no dither, no fragment loops: 2 texture
// taps per fragment (albedo+coverage, oct normal), both from texture ARRAYS so
// tiles cannot bleed into each other under minification.

struct PlantInst {
  pos: vec3f,
  packed_bits: u32,
}

@group(1) @binding(0) var<uniform> info: EntryInfo;
@group(1) @binding(1) var<storage, read> insts: array<PlantInst>;
@group(1) @binding(2) var tile_albedo: texture_2d_array<f32>;
@group(1) @binding(3) var tile_normal: texture_2d_array<f32>;
@group(1) @binding(4) var tile_sampler: sampler;

const BANDS: u32 = 3u;
const SECTORS: u32 = 4u;
const PARTS: u32 = 12u;
const PART_AZ: u32 = 4u;
const FAR_AZ: u32 = 8u;
const TAU: f32 = 6.2831853;

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

/** Nearest baked index for an azimuth, with a per-part offset. */
fn quantize_az(theta: f32, offset: f32, n: u32) -> u32 {
  let step = TAU / f32(n);
  let q = i32(round((theta - offset) / step));
  return u32(((q % i32(n)) + i32(n)) % i32(n));
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) layer: u32,
  @location(3) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(4) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(5) shade: f32,                       // canopy-depth * sun-side term
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let inst = insts[ii];
  let bits = inst.packed_bits;
  let yaw = f32(bits & 1023u) * (TAU / 1024.0);
  let scale = f32((bits >> 10u) & 4095u) * (4.0 / 4095.0);
  let phase = f32((bits >> 22u) & 1023u) * (TAU / 1024.0);
  let base = inst.pos;
  let entry = stand_table[u32(info.entry_index)];

  // Clump axis: the baked capture is centered on the clump, which sits off the
  // mesh origin — reproduce that offset so imagery lands where geometry was.
  let clump_xz = vec3f(info.clump_cx, 0.0, info.clump_cz);
  let axis = base + rot_y(clump_xz * scale, yaw);
  let to_cam = frame.camera_pos - axis;
  let d_xz = max(length(to_cam.xz), 1e-4);
  let dir_xz = to_cam.xz / d_xz;
  let right_ws = vec3f(dir_xz.y, 0.0, -dir_xz.x);
  // Viewing azimuth in the plant's own (yawed) frame.
  let local_dir = rot_y(vec3f(dir_xz.x, 0.0, dir_xz.y), -yaw);
  let theta = atan2(local_dir.x, local_dir.z);

  let h0 = info.plant_y0 * scale;
  let h1 = info.plant_y1 * scale;
  let mid_y = base.y + (h0 + h1) * 0.5;
  let dy = frame.camera_pos.y - mid_y;
  let elev = abs(dy) / max(length(vec3f(to_cam.x, dy, to_cam.z)), 1e-4);

  // Camera-inside fade: 3D distance to the plant's core segment. Region edge:
  // erode to nothing exactly at region_r (the cull matches).
  let seg_y = clamp(frame.camera_pos.y, base.y + h0, base.y + h1);
  let d_core = length(vec3f(to_cam.x, frame.camera_pos.y - seg_y, to_cam.z));
  let r_plant = info.plant_rxz * scale;
  var fade = smoothstep(r_plant * 0.35, r_plant * 1.05, d_core);
  fade *= 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);

  var corners = array<vec2f, 6>(
    vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(-1.0, 1.0),
    vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi % 6u];
  let card = vi / 6u;
  let is_near = info.draw_level > 0.5;

  var world: vec3f;
  var uv: vec2f;
  var layer: u32 = 0u;
  var y_local: f32 = 0.0;
  var occl: f32 = 1.0;

  if (is_near && card < PARTS) {
    // ---- upright part card ------------------------------------------------
    let band = card / SECTORS;
    let rank = card % SECTORS;
    // Nearest sector first, then the two flanks, then the far side: with depth
    // write on, the hidden half of the clump dies in early-z.
    var order = array<u32, 4>(0u, 1u, 3u, 2u);
    let s_near = quantize_az(theta, 0.0, SECTORS);
    let sector = (s_near + order[rank]) % SECTORS;
    let p = band * SECTORS + sector;
    let ctr = info.part_box[p * 2u].xyz;
    let ext = info.part_box[p * 2u + 1u].xyz;

    // Per-sector azimuth stagger: the 4 sectors switch baked view at different
    // camera angles, so an orbit never pops the whole plant at once.
    let stagger = f32(sector) * (TAU / f32(PART_AZ * SECTORS));
    let k = quantize_az(theta, stagger, PART_AZ);
    let az = f32(k) * (TAU / f32(PART_AZ)) + stagger;
    let eu = abs(cos(az)) * ext.x + abs(sin(az)) * ext.z;

    let ctr_ws = base + rot_y((vec3f(ctr.x, 0.0, ctr.z) + clump_xz) * scale, yaw);
    y_local = ctr.y + (c.y * 2.0 - 1.0) * ext.y;
    world = ctr_ws + right_ws * (c.x * eu * scale);
    world.y = base.y + y_local * scale;
    uv = vec2f(c.x * 0.5 + 0.5, 1.0 - c.y);
    layer = p * PART_AZ + k;

    // Sun-side modelling across the clump: which side of the plant axis a part
    // sits on decides whether it reads as lit or self-shadowed. One dot product
    // per card, and it is what makes 12 cutouts read as one solid volume.
    let sun_xz = frame.sun_dir.xz;
    let sun_len = length(sun_xz);
    if (sun_len > 1e-3) {
      let pdir = rot_y(vec3f(ctr.x, 0.0, ctr.z), yaw).xz;
      let pl = length(pdir);
      if (pl > 1e-4) {
        // Mean-preserving: the sunward parts brighten by as much as the shaded
        // ones darken, so the assembly gains modelling without going muddy.
        let t = dot(pdir / pl, sun_xz / sun_len);
        occl = 1.0 + info.self_shade * clamp(t, -1.0, 1.0);
      }
    }
    // Straight down, an upright card is edge-on anyway — stop rasterizing it.
    fade *= 1.0 - smoothstep(0.92, 0.995, elev);
    // Distance LOD inside the ring: the back sector only shows through gaps, so
    // past half the LOD radius (where a plant is under ~90 px) drop it and save
    // a quarter of the near-ring quad area.
    if (rank == 3u && length(to_cam) > info.part_r * 0.5) {
      fade = 0.0;
    }
  } else if (is_near) {
    // ---- horizontal band fill card ---------------------------------------
    let band = card - PARTS;
    let bb = info.band_box[band * 2u];      // center x, center z, half x, half z
    let bm = info.band_box[band * 2u + 1u]; // card height (mesh frame)
    let q = vec2f(c.x, c.y * 2.0 - 1.0);
    world = base + rot_y((vec3f(bb.x + q.x * bb.z, 0.0, bb.y + q.y * bb.w) + clump_xz) * scale, yaw);
    y_local = bm.x;
    world.y = base.y + y_local * scale;
    uv = q * 0.5 + vec2f(0.5);
    layer = PARTS * PART_AZ + band;
    // Only meaningful seen from well above: below ~33deg a horizontal card is a
    // pale slab lying across the upright cards (the classic billboard "floating
    // pancake"), and the upright cards already cover that range.
    fade *= smoothstep(0.55, 0.78, elev);
  } else if (card == 0u) {
    // ---- far: whole-plant side card --------------------------------------
    let k = quantize_az(theta, 0.0, FAR_AZ);
    y_local = mix(info.plant_y0, info.plant_y1, c.y);
    world = axis + right_ws * (c.x * info.plant_rxz * scale);
    world.y = base.y + y_local * scale;
    uv = vec2f(c.x * 0.5 + 0.5, 1.0 - c.y);
    layer = k;
  } else {
    // ---- far: whole-plant top card ---------------------------------------
    let q = vec2f(c.x, c.y * 2.0 - 1.0);
    world = base + rot_y((vec3f(q.x * info.plant_rxz, 0.0, q.y * info.plant_rxz) + clump_xz) * scale, yaw);
    y_local = mix(info.plant_y0, info.plant_y1, info.top_frac);
    world.y = base.y + y_local * scale;
    uv = q * 0.5 + vec2f(0.5);
    layer = FAR_AZ;
    fade *= smoothstep(0.35, 0.6, elev);
  }

  // Wind: displacement scaled by height up the plant (roots do not move).
  let hfrac = clamp(y_local / max(info.plant_y1, 1e-3), 0.0, 1.0);
  world += wind_sway(base, frame.time, entry.sway, phase) * hfrac;

  var out: VOut;
  // A card whose fade dropped below the alpha reference has erode > 1, so EVERY
  // one of its fragments would discard — emit it behind the near plane instead
  // of rasterizing a guaranteed-empty quad (that is the whole set of band cards
  // at eye level, upright cards seen from straight above, cards the camera
  // stands inside, and the region rim).
  if (fade < info.alpha_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = uv;
  out.world = world;
  out.layer = layer;
  out.yaw_cs = vec2f(cos(yaw), sin(yaw));
  out.erode = info.alpha_ref / max(fade, 1e-3);
  out.shade = occl * mix(1.0 - info.bottom_shade, 1.0, hfrac);
  return out;
}

// Octahedral decode, y-primary — same convention the bake stored.
fn oct_decode_tile(e: vec2f) -> vec3f {
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
  let alb = textureSample(tile_albedo, tile_sampler, in.uv, in.layer);
  let enc = textureSample(tile_normal, tile_sampler, in.uv, in.layer).xy;
  if (alb.a < in.erode) {
    discard;
  }
  let n_mesh = oct_decode_tile(enc * 2.0 - 1.0);
  // Rotate the mesh-frame normal by the plant's yaw.
  let n = vec3f(
    in.yaw_cs.x * n_mesh.x + in.yaw_cs.y * n_mesh.z,
    n_mesh.y,
    -in.yaw_cs.y * n_mesh.x + in.yaw_cs.x * n_mesh.z,
  );
  // in.shade is fake occlusion, so it belongs to the light term: the albedo
  // view then shows the baked tile colour exactly as captured, and the lighting
  // view shows sun+ambient x (canopy depth * sun-side).
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
