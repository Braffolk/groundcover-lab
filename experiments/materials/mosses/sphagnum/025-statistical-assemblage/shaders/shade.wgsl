fn material_shade(g: Geo, s: Surface, vu: ViewUv) -> Shaded {
  let sun = frame.sun_dir;
  var micro = 1.0;
  if (u.p.micro_shadow > 0.5) {
    micro = 0.32 + 0.68 * saturate(abs(dot(sun, s.normal)) + 1.72 * s.ao * s.ao - 1.0);
  }
  let lit = light_surface(s.albedo, s.normal, g.world);
  let ndl = dot(s.normal, sun) * 0.5 + 0.5;
  let direct = s.albedo * frame.sun_color * ndl * ndl;
  let ambient = max(lit - direct, vec3f(0.0));
  let wrapped = saturate((dot(g.geo_n, sun) + 0.38) / 1.38);
  let wrap_light = s.subsurface * wrapped * u.p.light_wrap * frame.sun_color;
  let fuzz_light = s.sheen * pow(1.0 - saturate(dot(g.geo_n, g.view_ws)), 4.0) * u.p.fuzz;
  var sh = mat_shaded_default(s);
  sh.color = direct * micro * vu.shadow + ambient * s.ao + s.albedo * (wrap_light + fuzz_light);
  sh.albedo = s.albedo;
  sh.normal = s.normal;
  sh.coverage = 1.0;
  return sh;
}
