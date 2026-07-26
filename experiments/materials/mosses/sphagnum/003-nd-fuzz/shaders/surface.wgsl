// Stage 4 — assemble the Surface from the graph's channels.
//
// Do NOT #include anything here: the generator pastes material.wgsl (which
// already brings in frame / lighting / debug) around this file, and the
// validator rejects a stage source that redefines a shared symbol.
//
// In scope: Geo, Surface, Shaded, ViewUv, EvalCtx, the generated `Channels`
// struct, and the generated uniform `u` (u.p.<param> in snake_case).
//
// WHAT IS HERE AND NOT IN THE BRDF: the two terms of 039's `shade_moss` that
// are not BRDF at all — building the shading normal, and the paper's unified
// wetness. Wetness changes what the surface IS (its albedo and its normal),
// not how light bounces off it, and both of its inputs are view-independent.
// 039 has no surface/shade seam, so it did them inside the BRDF; here they
// belong on this side of it, and moving them is the only reordering the port
// makes.

fn material_surface(g: Geo, ch: Channels, base: Surface) -> Surface {
  var s = base;

  // Straight LINEAR colour. The map is measured, not authored: its bytes ARE
  // the linear albedo (mean 0.304, 0.586, 0.147 == measured.json meanColor),
  // so the image node declares srgb: false and nothing decodes anything.
  var alb = ch.albedo;

  // The measured normal map is MESH-FRAME, not the usual tangent-space
  // convention: rgb = mesh (x, y, z), upper hemisphere, and the bake laid the
  // tile out with +u along mesh +x and +v along mesh +z. So the axis order is
  // (T, N, B) — the map's GREEN channel is the surface normal, not its blue.
  // Reading it as (T, B, N) tilts every fragment ~90 degrees and looks, at a
  // glance, merely "wrong-ish". The convention is declared on the node (the
  // `unit-vector` hemisphere is +Y); this is where it is consumed.
  let m = ch.normal;
  let n_map = normalize(g.tangent * m.x + g.geo_n * m.y + g.bitangent * m.z);
  var n = normalize(mix(g.geo_n, n_map, u.p.normal_strength));

  // --- unified wetness (slides 63-65) --------------------------------------
  // Porous surfaces darken more than smooth ones, so the paper lerps the base
  // colour toward its own SQUARE, and pulls the normal flat as the surface
  // saturates. 039: `wet = wetness * (0.45 + 0.55 * (1 - height))`, i.e. water
  // collects below the midpoint of the capture range. Here the pivot is
  // MEASURED: `plane_norm` is the mean cushion plane, and water pools below
  // THAT — above it (the capitula themselves) the wetness stays at its dry
  // floor instead of ramping the moment a fragment is off the very top.
  let below = saturate((ch.plane_norm - ch.height) / max(ch.plane_norm, 1.0e-3));
  let wet = clamp(u.p.wetness * (0.45 + 0.55 * below), 0.0, 1.0);
  alb = mix(alb, alb * alb, wet * 0.55);
  n = normalize(mix(n, g.geo_n, wet * 0.3));

  s.albedo = alb;
  s.normal = n;
  s.geo_normal = g.geo_n;
  // The `ao` channel already IS pow(ao, aoStrength) — that curve is a graph
  // node, because a curve over measured data is data. What is NOT here is AO
  // Fresnel: it is keyed off the view direction and belongs to the BRDF.
  s.ao = ch.ao;
  s.height = ch.height;

  // The two measured tints the BRDF needs, riding on Surface.
  //
  // `sheen` is the right slot and means what it says: the Fuzz BRDF is a
  // microfibre layer with its OWN colour, and that colour is the measured
  // capitulum-tip albedo ("lighter at the tips than at the base", slide 44).
  s.sheen = ch.tip_color;
  // `subsurface` is the measured deep-cushion albedo — the colour of light that
  // came back out of the cushion's interior. It tints the light wrap (fake SSS)
  // and the AO-saturation bounce. This slot exists BECAUSE of this material: the
  // first port had to smuggle it through `emissive`, and a fuzzy surface plainly
  // needs two tints, one for the fibre tips and one for the interior.
  s.subsurface = ch.base_color;

  // A moss layer on opaque geometry covers every fragment it is asked about.
  // The measured tile is 93-98% closed and 039 alpha-tested the rest away down
  // to peat; a material on a preview object has nothing behind it, so a
  // discard would punch holes in the OBJECT rather than reveal a substrate.
  s.opacity = 1.0;
  return s;
}
