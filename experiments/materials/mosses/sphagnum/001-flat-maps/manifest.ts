import { defineMaterial, HARNESS_API, p } from '@harness'

/**
 * Shading params only. What is drawn (the preview object) belongs to the
 * MaterialStage, exactly as placement belongs to the stand for a renderer.
 */
export const PARAMS = {
  /** Which measured micro-habitat state of the Sphagnum tile to render. */
  habitat: p.enum('wet-vigorous', ['wet-vigorous', 'late-season', 'sun-exposed'] as const),
  /** 1.0 = the measured 0.18 m tile at life size on the preview object. */
  tileScale: p.num(1, { min: 0.25, max: 4, step: 0.05 }),
  /** 0 = geometric normal only, 1 = the measured map, >1 exaggerated. */
  normalStrength: p.num(1, { min: 0, max: 2, step: 0.05 }),
  /** Weight of the measured cavity AO. */
  aoStrength: p.num(1, { min: 0, max: 1, step: 0.05 }),
}

export default defineMaterial({
  id: '001-flat-maps',
  title: 'Sphagnum — flat measured maps',
  description:
    'The measured Sphagnum map set (albedo, mesh-frame normal, cavity AO) sampled straight onto the preview ' +
    'geometry: no parallax, no fuzz BRDF, shared light_surface(). The honest floor every later Sphagnum ' +
    'material is measured against.',
  status: 'working',
  harnessApi: HARNESS_API,
  // A material has no species. It tags its VRAM against its own id so the
  // existing per-species budget bar has one meaningful row: "these maps".
  species: ['001-flat-maps'],
  previewObject: 'sphere',
  params: PARAMS,
  load: () => import('./main.ts'),
})
