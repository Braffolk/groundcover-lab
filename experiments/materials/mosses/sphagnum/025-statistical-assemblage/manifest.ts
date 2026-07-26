import { HARNESS_API, defineMaterial, p } from '@harness'

export const PARAMS = {
  habitat: p.enum('field-mosaic', ['field-mosaic', 'wet-green', 'sun-gold', 'late-red'] as const),
  stateVariation: p.num(0.72, { min: 0, max: 1, step: 0.02 }),
  tileScale: p.num(1, { min: 1, max: 8, step: 0.25 }),
  texPx: p.enum('2048', ['2048'] as const),
  reliefGain: p.num(1.2, { min: 0.25, max: 2.5, step: 0.05 }),
  normalStrength: p.num(0.92, { min: 0, max: 2, step: 0.05 }),
  pomSteps: p.num(36, { min: 8, max: 56, step: 1 }),
  shadowSteps: p.num(8, { min: 0, max: 12, step: 1 }),
  aoStrength: p.num(0.9, { min: 0, max: 1.5, step: 0.05 }),
  microShadow: p.bool(true),
  lightWrap: p.num(0.34, { min: 0, max: 1, step: 0.05 }),
  fuzz: p.num(0.38, { min: 0, max: 1.5, step: 0.05 }),
  albedoGain: p.num(1, { min: 0.5, max: 1.5, step: 0.02 }),
}

export default defineMaterial({
  id: '025-statistical-assemblage',
  title: 'Sphagnum — multiscale statistical assemblage',
  description: 'A Cox-distributed population of varied capitula and fascicle packets forms one depth-ranked biomass canopy; shared structural events drive height, colour, AO, normals, and ordinary POM.',
  status: 'working',
  harnessApi: HARNESS_API,
  species: ['025-statistical-assemblage'],
  previewObject: 'sphere',
  params: PARAMS,
  load: () => import('./main.ts'),
})
