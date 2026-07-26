// Stage 5 — the BRDF: Uncharted 4's fuzzy-moss stack.
//
// This is 039-nd-moss' `shade_moss` (experiments/renderers/039-nd-moss/shaders/
// moss.wgsl, lines 226-311) with the plumbing deleted. What was deleted, and
// nothing else: the tile's quarter-turn rotation of a mesh-frame normal into
// world (a carpet-lattice concern, gone with the lattice), the `up`/`v`
// reconstruction from a hand-rolled vertex output (Geo carries both), and the
// wetness function (a surface property — it is in surface.wgsl now).
//
// Do NOT #include anything: everything below is already in scope.
//
// The observation the whole thing is built on (slide 44): moss is micro-FIBRE,
// not micro-facet — porous, hugely self-shadowing, transmitting light, lighter
// at the tips than at the base, and strongly view-dependent. Every term below
// is one sentence of that.

fn material_shade(g: Geo, s: Surface, vu: ViewUv) -> Shaded {
  // The GEOMETRY normal, not the shading normal. Three of the terms below are
  // *defined* by that choice — this is why the ABI carries both.
  let up = g.geo_n;
  let v = g.view_ws;
  let sun = frame.sun_dir;
  let base_tint = s.subsurface; // measured deep-cushion albedo (see surface.wgsl)

  // --- AO Fresnel (slides 38-40) -------------------------------------------
  // The cracks baked into the normal/AO maps SHOULD be occluded by geometry at
  // a glancing angle; that geometry does not exist, so fade the AO toward
  // white there instead. Keyed off the GEOMETRY normal, per the slide.
  // ADAPTED: the paper fades all the way (`ao = lerp(1, ao, dot(N, V))`). That
  // is right for moss coating a rock, whose silhouette belongs to the rock —
  // but a mat, or a sphere seen anywhere near its limb, is glancing almost
  // everywhere, and moss's albedo is nearly uniform (tip vs base differ by
  // ~7%), so ALL of its structure is shading. Fading it out there measured as
  // flat olive paint in 039. `ao_fresnel` caps how much the angle may take;
  // 1.0 is the paper exactly.
  var ao = mix(s.ao, 1.0, u.p.ao_fresnel * (1.0 - saturate(dot(up, v))));

  // --- AO saturation (slide 55) --------------------------------------------
  // "Add back a bit of colour where the ambient occlusion gets dark" — bounce
  // light and SSS; "our code was a simple hack". ADDITIVE, tinted by the
  // measured deep-cushion colour normalised to luma 1 (so it adds light of the
  // cushion's HUE, not of its dark intensity) and scaled by the ambient level.
  let base_hue = base_tint / max(dot(base_tint, vec3f(0.3333)), 1.0e-3);
  let bounce = base_hue * frame.ambient * ((1.0 - ao) * u.p.ao_saturation);

  // --- sun micro shadowing (slide 37, verbatim) ----------------------------
  var micro = 1.0;
  if (u.p.micro_shadow > 0.5) {
    let aperture = 2.0 * ao * ao;
    micro = saturate(abs(dot(sun, s.normal)) + aperture - 1.0);
  }

  // --- light wrap (slide 51) -----------------------------------------------
  // Broaden NdotL past the terminator to fake SSS. Deliberately uses the
  // GEOMETRY normal and ignores the normal map, so it fills the cracks rather
  // than following them, and it is TINTED (the paper's artist picks green; the
  // tint here was measured off the cushion). Converges to plain NdotL as the
  // strength goes to 0: `extra` is exactly the amount by which the wrapped
  // term exceeds the unwrapped one, so at light_wrap = 0 nothing is added.
  let ndl_geo = dot(up, sun);
  let wrapped = saturate((ndl_geo + 0.35) / 1.35);
  let extra = max(wrapped - saturate(ndl_geo), 0.0) * u.p.light_wrap;
  let wrap_light = base_tint * extra * frame.sun_color;

  // --- Fuzz BRDF (slide 52) ------------------------------------------------
  // A microfibre layer over the microfacet one, with its own colour — and
  // explicitly the SUBPIXEL half of the answer: "moss details are large enough
  // to be seen in the texture when the camera is close, but become subpixel
  // when more than a few meters away. We need to solve both of these cases."
  // The parallax/normal/AO half solves close; this solves far, and it is the
  // term that survives when the relief is smaller than a pixel.
  let fuzz_term = pow(1.0 - saturate(dot(up, v)), 4.0) * saturate(ndl_geo * 0.5 + 0.5) * u.p.fuzz;
  let fuzz_light = s.sheen * fuzz_term * frame.sun_color;

  // --- AO drives the indirect half, the micro shadow drives the sun --------
  // The shared lighting model is `albedo * (sun + ambient * hemi)`, and
  // occlusion does not act equally on those halves: cavity AO occludes the
  // INDIRECT term, while the sun is occluded by the shadow terms (the paper
  // multiplies its micro shadow into the sun's shadow, slide 37). So split the
  // shared model apart by SUBTRACTION rather than reimplementing it — with
  // ao = micro = shadow = 1 this is exactly `light_surface()`, and the ambient
  // formula exists in exactly one place in the repo.
  let lit = light_surface(s.albedo, s.normal, g.world);
  let ndl = dot(s.normal, sun) * 0.5 + 0.5;
  let sun_part = s.albedo * frame.sun_color * (ndl * ndl);
  let amb_part = max(lit - sun_part, vec3f(0.0));

  var sh = mat_shaded_default(s);
  // `vu.shadow` is the view-uv stage's self-shadow along the sun (slide 53),
  // and it is 1.0 unless a stage actually marched one.
  sh.color =
    sun_part * (micro * vu.shadow) + amb_part * ao + s.albedo * (wrap_light + fuzz_light + bounce);
  sh.albedo = s.albedo;
  sh.normal = s.normal;
  sh.coverage = s.opacity;
  return sh;
}
