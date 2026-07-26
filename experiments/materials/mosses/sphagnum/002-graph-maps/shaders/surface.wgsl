// Stage 4 — assemble the Surface from the graph's channels.
//
// Do NOT #include anything here: the generator pastes material.wgsl (which
// already brings in frame / lighting / debug) around this file, and the
// validator rejects a stage source that redefines a shared symbol.
//
// In scope: Geo, Surface, Shaded, ViewUv, EvalCtx, the generated `Channels`
// struct, and the generated uniform `u` (u.p.<param> in snake_case).

fn material_surface(g: Geo, ch: Channels, base: Surface) -> Surface {
  var s = base;
  // Straight LINEAR colour. The map is measured, not authored: its bytes are
  // the linear albedo (mean 0.304, 0.586, 0.147 == measured.json meanColor),
  // so the image node declares srgb: false and there is no decode anywhere.
  s.albedo = ch.albedo;

  // The measured normal map is MESH-FRAME, not the usual tangent-space
  // convention: rgb = mesh (x, y, z), upper hemisphere, and the bake laid the
  // tile out with +u along mesh +x and +v along mesh +z. So the axis order is
  // (T, N, B) — the map's GREEN channel is the surface normal, not its blue.
  // Reading it as (T, B, N) tilts every fragment ~90 degrees and looks, at a
  // glance, merely "wrong-ish".
  let m = ch.normal;
  let n_map = normalize(g.tangent * m.x + g.geo_n * m.y + g.bitangent * m.z);
  s.normal = normalize(mix(g.geo_n, n_map, u.p.normal_strength));
  s.geo_normal = g.geo_n;

  // The `ao` channel already IS the weighted occlusion — the aoStrength curve
  // is a graph node (`occlusion`), not a line here, because it is the one node
  // in this material that can honestly be evaluated live OR materialized.
  s.ao = ch.ao;
  // Flat maps on an opaque surface cover every fragment they are asked about.
  // A silhouette material is where that stops being true — and there it is the
  // view-uv stage's clip, not this, that says so.
  s.opacity = 1.0;
  return s;
}
