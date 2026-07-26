import {
  createMaterialExperiment,
  tilesPerMetre,
  type ChannelSpec,
  type Experiment,
  type ExperimentContext,
  type MaterialContextExt,
  type MaterialDef,
  type MaterialGraph,
  type MaterialNode,
} from '@harness'
import { PARAMS } from './manifest.ts'
import shadeSrc from './shaders/shade.wgsl'
import surfaceSrc from './shaders/surface.wgsl'

const TILE_M = 0.42
const RELIEF_M = 0.028
const TEX_PX = 2048
const DESCRIPTOR_URL = new URL('./assets/descriptor.png', import.meta.url).href

const PACKED: ChannelSpec = {
  type: 'vec4f',
  semantics: { kind: 'scalar-field', empty: 'zero' },
  srgb: false,
  doc: 'causal structural projection: R tissue age, G height, B aperture/AO, A colony pigmentation',
}
const SCALAR: ChannelSpec = { type: 'f32', semantics: { kind: 'scalar-field', empty: 'zero' } }
const COLOR: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'color-with-coverage', weight: 'a' },
  srgb: false,
}
const NORMAL: ChannelSpec = {
  type: 'vec3f',
  semantics: { kind: 'unit-vector', hemisphere: [0, 1, 0] },
  doc: 'mesh-frame (T,N,B) normal derived from the exact height used by POM',
}

export async function create(ctx: ExperimentContext<typeof PARAMS, MaterialContextExt>): Promise<Experiment> {
  ctx.progress(0.02, 'statistical assemblage: loading 2048 causal descriptor')
  const def: MaterialDef<typeof PARAMS> = {
    id: ctx.id,
    params: PARAMS,
    surface: surfaceSrc,
    shade: shadeSrc,
    uvTransform: (mesh, params) => ({ scale: tilesPerMetre(mesh, TILE_M, params.tileScale) }),
    structuralParams: ['pomSteps', 'shadowSteps'],
    graph: (params): MaterialGraph => {
      const nodes: MaterialNode[] = [
        {
          kind: 'image',
          id: 'descriptor',
          url: DESCRIPTOR_URL,
          spec: PACKED,
          doc: 'deterministic event raster; no photographed pixels and no independent noise height',
        },
        {
          kind: 'procedural',
          id: 'descriptor-varied',
          inputs: { source: 'descriptor' },
          spec: PACKED,
          materialize: 'never',
          doc: 'small C1 interior deformation decorrelates repeated tiles while preserving every periodic boundary exactly',
          body: `let tile = floor(c.uv);
let local = fract(c.uv);
let h0 = fract(sin(dot(tile + vec2f(17.0, 43.0), vec2f(127.1, 311.7))) * 43758.5453);
let h1 = fract(sin(dot(tile + vec2f(71.0, 19.0), vec2f(269.5, 183.3))) * 24634.6345);
let sx = sin(3.14159265 * local.x);
let sy = sin(3.14159265 * local.y);
let interior = sx * sx * sy * sy;
if (interior < 0.00001) { return in_source; }
let angle = h0 * 6.2831853;
let magnitude = (0.025 + 0.050 * h1) * interior;
var warped = c;
warped.uv = local + vec2f(cos(angle), sin(angle)) * magnitude;
return n_descriptor_eval(warped);`,
        },
        {
          kind: 'procedural', id: 'height', inputs: { d: 'descriptor-varied' }, spec: SCALAR,
          body: 'return clamp(in_d.g, 0.0, 1.0);',
        },
        {
          kind: 'procedural', id: 'albedo', inputs: { d: 'descriptor-varied' }, spec: COLOR,
          body: `let tissue = in_d.r;
let height = in_d.g;
let aperture = in_d.b;
let pigment = clamp(0.5 + (in_d.a - 0.5) * (0.25 + 1.75 * u.p.state_variation), 0.0, 1.0);
var cavity = vec3f(0.010, 0.026, 0.006);
var body = vec3f(0.070, 0.295, 0.018);
var tip = vec3f(0.50, 0.72, 0.080);
var ochre = vec3f(0.48, 0.285, 0.035);
var rose = vec3f(0.43, 0.075, 0.052);
if (u.p.habitat > 0.5 && u.p.habitat < 1.5) {
  cavity = vec3f(0.006, 0.026, 0.005);
  body = vec3f(0.045, 0.30, 0.014);
  tip = vec3f(0.43, 0.72, 0.070);
  ochre = vec3f(0.28, 0.34, 0.025);
  rose = vec3f(0.24, 0.18, 0.030);
}
if (u.p.habitat >= 1.5 && u.p.habitat < 2.5) {
  cavity = vec3f(0.025, 0.024, 0.006);
  body = vec3f(0.235, 0.255, 0.022);
  tip = vec3f(0.66, 0.58, 0.075);
  ochre = vec3f(0.52, 0.31, 0.035);
  rose = vec3f(0.46, 0.12, 0.045);
}
if (u.p.habitat >= 2.5) {
  cavity = vec3f(0.026, 0.010, 0.009);
  body = vec3f(0.27, 0.060, 0.040);
  tip = vec3f(0.62, 0.205, 0.085);
  ochre = vec3f(0.48, 0.22, 0.035);
  rose = vec3f(0.45, 0.038, 0.060);
}
let live = mix(body, tip, smoothstep(0.50, 0.94, tissue));
let warm = mix(ochre, rose, smoothstep(0.52, 0.90, pigment));
let age_mix = smoothstep(0.46, 0.82, pigment) * u.p.state_variation;
var color = mix(live, warm, age_mix);
let biomass = smoothstep(0.095, 0.22, height);
color = mix(cavity, color, biomass);
let height_exposure = mix(0.72, 1.12, smoothstep(0.16, 0.62, height));
let local_aperture = mix(0.74, 1.04, aperture);
return clamp(color * height_exposure * local_aperture * u.p.albedo_gain, vec3f(0.0), vec3f(1.0));`,
        },
        {
          kind: 'procedural', id: 'normal', inputs: { center: 'height' }, spec: NORMAL,
          body: `let du = 1.0 / ${TEX_PX}.0;
var l = c; var r = c; var d = c; var up = c;
l.uv -= vec2f(du, 0.0); r.uv += vec2f(du, 0.0);
d.uv -= vec2f(0.0, du); up.uv += vec2f(0.0, du);
let sx = (n_height_eval(r) - n_height_eval(l)) / (2.0 * du);
let sy = (n_height_eval(up) - n_height_eval(d)) / (2.0 * du);
let metres_per_tile = ${TILE_M} * max(u.p.tile_scale, 1.0);
let physical = vec2f(sx, sy) * (${RELIEF_M} / metres_per_tile) * u.p.relief_gain;
return normalize(vec3f(-physical.x, 1.0, -physical.y));`,
        },
        {
          kind: 'procedural', id: 'ao', inputs: { d: 'descriptor-varied' }, spec: SCALAR,
          body: 'return mix(1.0, clamp(in_d.b, 0.0, 1.0), clamp(u.p.ao_strength, 0.0, 1.5));',
        },
        {
          kind: 'procedural', id: 'relief-m', spec: SCALAR,
          body: `return ${RELIEF_M} * u.p.relief_gain;`,
          doc: 'physical POM thickness; tileScale changes feature size, not relief depth',
        },
      ]
      return {
        nodes,
        channels: { albedo: 'albedo', normal: 'normal', ao: 'ao', height: 'height' },
        viewUv: {
          kind: 'pom',
          height: 'height',
          steps: Math.round(params.pomSteps),
          scale: { node: 'relief-m' },
          pdo: true,
          shadowSteps: Math.round(params.shadowSteps),
        },
      }
    },
  }
  return createMaterialExperiment(ctx, def)
}
