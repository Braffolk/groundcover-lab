export const TILE = 2048
export const DEPTH_LAYERS = 4

const GRID = 96
const CANDIDATES_PER_CELL = 3
const EMPTY_SURFACE = 0xffffffff
const TAU = Math.PI * 2

interface Raster {
  width: number
  height: number
  periodic: boolean
  depths: Float32Array
  surfaces: Uint32Array
  tissue: Uint8Array
  pigment: Uint8Array
}

interface StructuralEvent {
  id: number
  family: 'capitulum' | 'fascicle'
  x: number
  y: number
  baseZ: number
  phase: number
  size: number
  openness: number
  density: number
  verticality: number
  asymmetry: number
  curvature: number
  vitality: number
  pigment: number
}

interface Counters {
  nextSurface: number
  branchPackets: number
  packetSamples: number
  capitula: number
  fascicles: number
}

export interface AssemblageMetrics {
  seed: number
  events: number
  capitula: number
  fascicles: number
  branchPackets: number
  packetSamples: number
  eventsPerSquareMetre: number
  meanNominalCapitulumMm: number
  coverage: number
  meanDepthLayers: number
  depthLayerFractions: [number, number, number, number]
  meanHeight: number
  heightStdDev: number
  meanGradient: number
  p95Gradient: number
  highGradientFraction: number
  generationMs: number
}

export interface AssemblageResult {
  descriptor: Uint8Array
  height: Float32Array
  ao: Float32Array
  topSurface: Uint32Array
  depthCount: Uint8Array
  events: StructuralEvent[]
  metrics: AssemblageMetrics
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function hash32(value: number): number {
  let x = value >>> 0
  x ^= x >>> 16
  x = Math.imul(x, 0x7feb352d)
  x ^= x >>> 15
  x = Math.imul(x, 0x846ca68b)
  return (x ^ (x >>> 16)) >>> 0
}

function unit(a: number, b: number, seed: number): number {
  return (
    hash32(Math.imul(a + 31, 0x9e3779b9) ^ Math.imul(b + 79, 0x85ebca6b) ^ Math.imul(seed + 131, 0xc2b2ae35)) &
    0x00ffffff
  ) / 0x01000000
}

function gaussianish(index: number, channel: number, seed: number): number {
  return (
    unit(index, channel, seed) +
    unit(index, channel + 101, seed) +
    unit(index, channel + 211, seed) +
    unit(index, channel + 337, seed) -
    2
  ) * 0.82
}

function smooth(value: number): number {
  return value * value * (3 - 2 * value)
}

function wrap(value: number, extent: number): number {
  value %= extent
  return value < 0 ? value + extent : value
}

function periodicField(x: number, y: number, cells: number, channel: number, seed: number): number {
  const fx = (x / TILE) * cells
  const fy = (y / TILE) * cells
  const ix = Math.floor(fx)
  const iy = Math.floor(fy)
  const tx = smooth(fx - ix)
  const ty = smooth(fy - iy)
  const sample = (ox: number, oy: number): number => {
    const gx = (ix + ox + cells) % cells
    const gy = (iy + oy + cells) % cells
    return unit(gx + gy * cells, channel, seed) * 2 - 1
  }
  const a = sample(0, 0) * (1 - tx) + sample(1, 0) * tx
  const b = sample(0, 1) * (1 - tx) + sample(1, 1) * tx
  return a * (1 - ty) + b * ty
}

function macroElevation(x: number, y: number, seed: number): number {
  return (
    periodicField(x + 137, y - 73, 4, 1, seed) * 0.50 +
    periodicField(x - 311, y + 191, 9, 2, seed) * 0.31 +
    periodicField(x + 79, y + 367, 19, 3, seed) * 0.19
  )
}

function macroForm(x: number, y: number, seed: number): number {
  return periodicField(x - 239, y + 113, 6, 5, seed) * 0.66 + periodicField(x + 401, y - 277, 15, 6, seed) * 0.34
}

function macroPigment(x: number, y: number, seed: number): number {
  return periodicField(x + 521, y + 61, 5, 8, seed) * 0.58 + periodicField(x - 173, y - 431, 12, 9, seed) * 0.42
}

function makeRaster(width: number, height: number, periodic: boolean): Raster {
  const pixels = width * height
  const depths = new Float32Array(pixels * DEPTH_LAYERS)
  const surfaces = new Uint32Array(pixels * DEPTH_LAYERS)
  const tissue = new Uint8Array(pixels * DEPTH_LAYERS)
  const pigment = new Uint8Array(pixels * DEPTH_LAYERS)
  surfaces.fill(EMPTY_SURFACE)
  return { width, height, periodic, depths, surfaces, tissue, pigment }
}

function insertSurface(
  raster: Raster,
  pixel: number,
  z: number,
  surface: number,
  tissue: number,
  pigment: number,
): void {
  const base = pixel * DEPTH_LAYERS
  for (let layer = 0; layer < DEPTH_LAYERS; layer++) {
    if (raster.surfaces[base + layer] !== surface) continue
    if (z <= raster.depths[base + layer]!) return
    for (let move = layer; move < DEPTH_LAYERS - 1; move++) {
      raster.depths[base + move] = raster.depths[base + move + 1]!
      raster.surfaces[base + move] = raster.surfaces[base + move + 1]!
      raster.tissue[base + move] = raster.tissue[base + move + 1]!
      raster.pigment[base + move] = raster.pigment[base + move + 1]!
    }
    raster.depths[base + DEPTH_LAYERS - 1] = 0
    raster.surfaces[base + DEPTH_LAYERS - 1] = EMPTY_SURFACE
    raster.tissue[base + DEPTH_LAYERS - 1] = 0
    raster.pigment[base + DEPTH_LAYERS - 1] = 0
    break
  }

  let layer = 0
  while (layer < DEPTH_LAYERS && raster.surfaces[base + layer] !== EMPTY_SURFACE && raster.depths[base + layer]! >= z) layer++
  if (layer >= DEPTH_LAYERS) return
  for (let move = DEPTH_LAYERS - 1; move > layer; move--) {
    raster.depths[base + move] = raster.depths[base + move - 1]!
    raster.surfaces[base + move] = raster.surfaces[base + move - 1]!
    raster.tissue[base + move] = raster.tissue[base + move - 1]!
    raster.pigment[base + move] = raster.pigment[base + move - 1]!
  }
  raster.depths[base + layer] = z
  raster.surfaces[base + layer] = surface
  raster.tissue[base + layer] = Math.round(clamp01(tissue) * 255)
  raster.pigment[base + layer] = Math.round(clamp01(pigment) * 255)
}

function splatEllipse(
  raster: Raster,
  cx: number,
  cy: number,
  alongRadius: number,
  acrossRadius: number,
  angle: number,
  floorZ: number,
  crownHeight: number,
  surface: number,
  tissue: number,
  pigment: number,
): void {
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  const radius = Math.ceil(Math.max(alongRadius, acrossRadius) + 1)
  const ix = Math.floor(cx)
  const iy = Math.floor(cy)
  for (let oy = -radius; oy <= radius; oy++) {
    const sourceY = iy + oy
    if (!raster.periodic && (sourceY < 0 || sourceY >= raster.height)) continue
    const py = sourceY + 0.5 - cy
    const y = raster.periodic ? wrap(sourceY, raster.height) : sourceY
    for (let ox = -radius; ox <= radius; ox++) {
      const sourceX = ix + ox
      if (!raster.periodic && (sourceX < 0 || sourceX >= raster.width)) continue
      const px = sourceX + 0.5 - cx
      const along = (px * cosine + py * sine) / alongRadius
      const across = (-px * sine + py * cosine) / acrossRadius
      const r2 = along * along + across * across
      if (r2 >= 1) continue
      const profile = Math.pow(1 - r2, 0.62)
      const x = raster.periodic ? wrap(sourceX, raster.width) : sourceX
      insertSurface(raster, y * raster.width + x, floorZ + crownHeight * profile, surface, tissue, pigment)
    }
  }
}

function packetPoint(
  event: StructuralEvent,
  angle: number,
  startRadius: number,
  length: number,
  bend: number,
  t: number,
): { x: number; y: number; tangent: number } {
  const localX = startRadius + length * t
  const localY = bend * 4 * t * (1 - t) + event.asymmetry * length * 0.025 * t
  const derivativeY = bend * 4 * (1 - 2 * t) + event.asymmetry * length * 0.025
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  return {
    x: event.x + localX * cosine - localY * sine,
    y: event.y + localX * sine + localY * cosine,
    tangent: angle + Math.atan2(derivativeY, length),
  }
}

function renderPacket(
  raster: Raster,
  event: StructuralEvent,
  angle: number,
  length: number,
  halfWidth: number,
  startRadius: number,
  bend: number,
  zOffset: number,
  seed: number,
  packetIndex: number,
  counters: Counters,
): void {
  const surface = counters.nextSurface++
  counters.branchPackets++
  const samples = Math.max(4, Math.ceil(length / 4.2))
  const step = length / samples
  const bands = Math.max(3, Math.round(length / (4.4 + 0.8 * event.openness)))
  const phase = unit(event.id * 43 + packetIndex, 201, seed)
  for (let sample = 0; sample <= samples; sample++) {
    counters.packetSamples++
    const t = sample / samples
    const point = packetPoint(event, angle, startRadius, length, bend, t)
    const leafPulse = 0.5 + 0.5 * Math.cos((t * bands + phase) * TAU)
    const taper = Math.sqrt(Math.max(0.08, 1 - Math.pow(t, 2.7)))
    const localHalfWidth = halfWidth * taper * (0.87 + leafPulse * 0.13)
    const axialArch = (0.010 + event.verticality * 0.010) * Math.sin(t * Math.PI) -
      (0.008 + event.verticality * 0.025) * t
    const microRise = leafPulse * 0.011
    const crown = (0.055 + event.density * 0.022 + event.verticality * 0.035) * (0.97 - 0.15 * t)
    const peakZ = event.baseZ + zOffset + axialArch + microRise
    const floorZ = peakZ - crown
    const buriedAge = Math.max(0, -zOffset)
    const freshness = clamp01(event.vitality * 0.61 + (1 - t) * 0.21 + 0.11 - buriedAge * 1.75)
    const localPigment = clamp01(event.pigment + buriedAge * 1.55)
    splatEllipse(
      raster,
      point.x,
      point.y,
      Math.max(1.25, step * 0.74),
      Math.max(0.92, localHalfWidth),
      point.tangent,
      floorZ,
      crown,
      surface,
      freshness,
      localPigment,
    )
  }
}

function renderCapitulum(raster: Raster, event: StructuralEvent, seed: number, counters: Counters): void {
  counters.capitula++
  const branchCount = Math.max(10, Math.min(17, Math.round(11.2 + event.density * 4.2 - event.openness * 1.0)))
  const weights = new Float32Array(branchCount)
  let weightSum = 0
  for (let branch = 0; branch < branchCount; branch++) {
    const weight = 0.68 + unit(event.id * 19 + branch, 301, seed) * 0.72
    weights[branch] = weight
    weightSum += weight
  }

  let cumulative = 0
  for (let branch = 0; branch < branchCount; branch++) {
    const angleFraction = (cumulative + weights[branch]! * 0.5) / weightSum
    cumulative += weights[branch]!
    const directionalBias = event.asymmetry * Math.cos(angleFraction * TAU - event.phase) * 0.13
    const angle = event.phase + angleFraction * TAU + directionalBias
    const randomLength = 0.66 + unit(event.id * 23 + branch, 331, seed) * 0.52
    const length = event.size * randomLength * (0.62 + event.openness * 0.45)
    const halfWidth = event.size * (0.140 + event.density * 0.048) *
      (0.84 + unit(event.id * 29 + branch, 351, seed) * 0.28)
    const bend = (event.curvature * 0.08 + unit(event.id * 31 + branch, 371, seed) * 0.18 - 0.09) * length
    const startRadius = event.size * (0.055 + event.density * 0.040) + unit(event.id + branch, 391, seed) * 0.55
    const zOffset = 0.008 + event.verticality * 0.050 +
      (unit(event.id * 37 + branch, 411, seed) - 0.5) * 0.046
    renderPacket(raster, event, angle, length, halfWidth, startRadius, bend, zOffset, seed, branch, counters)
  }

  // Young compact packets fill the capitulum without introducing a disc or a
  // hidden substrate. They are branches from the same continuously ranked event.
  const innerCount = 6 + Math.round(event.density * 3)
  for (let inner = 0; inner < innerCount; inner++) {
    const angle = event.phase + (inner / innerCount) * TAU + (unit(event.id * 41 + inner, 431, seed) - 0.5) * 0.65
    const length = event.size * (0.22 + unit(event.id + inner, 451, seed) * 0.22)
    const halfWidth = event.size * (0.150 + event.density * 0.052)
    const bend = (unit(event.id * 43 + inner, 471, seed) - 0.5) * length * 0.12
    const innerRise = 0.034 + event.verticality * 0.060 +
      (unit(event.id * 47 + inner, 477, seed) - 0.5) * 0.018
    renderPacket(raster, event, angle, length, halfWidth, 0.3, bend, innerRise, seed, 32 + inner, counters)
  }

  // A Sphagnum shoot does not end at the visible capitulum. Spreading and
  // pendent branches descend through a continuous range of lower heights.
  // They are distinct occluding packets, not a synthetic under-layer, and are
  // seen only through openings left by the crowded upper canopy.
  const lowerCount = 2 + (unit(event.id, 481, seed) > 0.56 ? 1 : 0)
  for (let lower = 0; lower < lowerCount; lower++) {
    const angle = event.phase + (lower + 0.35) / lowerCount * TAU +
      (unit(event.id * 67 + lower, 483, seed) - 0.5) * 0.62
    const length = event.size * (1.02 + unit(event.id * 71 + lower, 485, seed) * 0.72)
    const halfWidth = event.size * (0.070 + event.density * 0.030) *
      (0.86 + unit(event.id + lower, 487, seed) * 0.26)
    const bend = (event.curvature * 0.15 + unit(event.id * 73 + lower, 489, seed) * 0.28 - 0.14) * length
    const zOffset = -0.060 - event.verticality * 0.040 - unit(event.id * 79 + lower, 491, seed) * 0.090
    renderPacket(raster, event, angle, length, halfWidth, 0.4, bend, zOffset, seed, 48 + lower, counters)
  }
}

function renderFascicle(raster: Raster, event: StructuralEvent, seed: number, counters: Counters): void {
  counters.fascicles++
  const count = 2 + Math.round(event.density * 2 + unit(event.id, 501, seed))
  for (let packet = 0; packet < count; packet++) {
    const fan = (packet - (count - 1) * 0.5) * (0.16 + event.openness * 0.09)
    const angle = event.phase + fan + (unit(event.id * 47 + packet, 521, seed) - 0.5) * 0.18
    const length = event.size * (1.28 + unit(event.id * 53 + packet, 541, seed) * 0.75)
    const halfWidth = event.size * (0.068 + event.density * 0.032) *
      (0.88 + unit(event.id + packet, 561, seed) * 0.22)
    const bend = (event.curvature * 0.16 + unit(event.id * 59 + packet, 581, seed) * 0.26 - 0.13) * length
    const zOffset = (unit(event.id * 61 + packet, 601, seed) - 0.5) * 0.030
    renderPacket(raster, event, angle, length, halfWidth, 0.2, bend, zOffset, seed, packet, counters)
  }
}

function makeEvents(seed: number): StructuralEvent[] {
  const events: StructuralEvent[] = []
  const spacing = TILE / GRID
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const cell = gy * GRID + gx
      const cellX = (gx + 0.5) * spacing
      const cellY = (gy + 0.5) * spacing
      const cellForm = macroForm(cellX, cellY, seed)
      const cellElevation = macroElevation(cellX, cellY, seed)
      const acceptance = clamp01(0.28 + cellForm * 0.060 + cellElevation * 0.018)
      for (let candidate = 0; candidate < CANDIDATES_PER_CELL; candidate++) {
        if (unit(cell * CANDIDATES_PER_CELL + candidate, 701, seed) > acceptance) continue
        const id = events.length
        const x = wrap((gx + unit(cell * 7 + candidate, 711, seed)) * spacing, TILE)
        const y = wrap((gy + unit(cell * 11 + candidate, 721, seed)) * spacing, TILE)
        const elevation = macroElevation(x, y, seed)
        const form = macroForm(x, y, seed)
        const age = gaussianish(id, 731, seed)
        const density = clamp01(0.55 - form * 0.10 + gaussianish(id, 741, seed) * 0.18)
        const openness = clamp01(0.30 + form * 0.20 + gaussianish(id, 751, seed) * 0.21 - density * 0.05)
        const verticality = clamp01(
          0.43 + density * 0.30 - openness * 0.26 + elevation * 0.10 + gaussianish(id, 756, seed) * 0.18,
        )
        const family = unit(id, 761, seed) < 0.90 + form * 0.04 ? 'capitulum' : 'fascicle'
        events.push({
          id,
          family,
          x,
          y,
          baseZ: clamp01(
            0.395 + elevation * 0.145 + gaussianish(id, 771, seed) * 0.065 +
              (verticality - 0.5) * 0.030 - openness * 0.012,
          ),
          phase: unit(id, 781, seed) * TAU,
          size: Math.max(22, 25.0 + unit(id, 791, seed) * 23.0 + gaussianish(id, 795, seed) * 4.2 + density * 3.0),
          openness,
          density,
          verticality,
          asymmetry: Math.max(-1, Math.min(1, gaussianish(id, 801, seed))),
          curvature: Math.max(-1, Math.min(1, gaussianish(id, 811, seed))),
          vitality: clamp01(0.61 + elevation * 0.12 - age * 0.13 + gaussianish(id, 821, seed) * 0.05),
          pigment: clamp01(0.47 + macroPigment(x, y, seed) * 0.30 + age * 0.14),
        })
      }
    }
  }
  return events
}

function periodicBoxBlur(input: Float32Array, width: number, radius: number): Float32Array {
  const pixels = width * width
  const horizontal = new Float32Array(pixels)
  const output = new Float32Array(pixels)
  const diameter = radius * 2 + 1
  for (let y = 0; y < width; y++) {
    let sum = 0
    for (let x = -radius; x <= radius; x++) sum += input[y * width + wrap(x, width)]!
    for (let x = 0; x < width; x++) {
      horizontal[y * width + x] = sum / diameter
      sum += input[y * width + wrap(x + radius + 1, width)]! - input[y * width + wrap(x - radius, width)]!
    }
  }
  for (let x = 0; x < width; x++) {
    let sum = 0
    for (let y = -radius; y <= radius; y++) sum += horizontal[wrap(y, width) * width + x]!
    for (let y = 0; y < width; y++) {
      output[y * width + x] = sum / diameter
      sum += horizontal[wrap(y + radius + 1, width) * width + x]! - horizontal[wrap(y - radius, width) * width + x]!
    }
  }
  return output
}

function histogramQuantile(histogram: Uint32Array, maxValue: number, q: number, count: number): number {
  const target = Math.floor(count * q)
  let accumulated = 0
  for (let bin = 0; bin < histogram.length; bin++) {
    accumulated += histogram[bin]!
    if (accumulated >= target) return (bin / (histogram.length - 1)) * maxValue
  }
  return maxValue
}

export function generateAssemblage(seed: number): AssemblageResult {
  const started = performance.now()
  const raster = makeRaster(TILE, TILE, true)
  const events = makeEvents(seed)
  const counters: Counters = { nextSurface: 0, branchPackets: 0, packetSamples: 0, capitula: 0, fascicles: 0 }
  for (const event of events) {
    if (event.family === 'capitulum') renderCapitulum(raster, event, seed, counters)
    else renderFascicle(raster, event, seed, counters)
  }

  const pixels = TILE * TILE
  const height = new Float32Array(pixels)
  const topSurface = new Uint32Array(pixels)
  const depthCount = new Uint8Array(pixels)
  const topTissue = new Uint8Array(pixels)
  const topPigment = new Uint8Array(pixels)
  let covered = 0
  let layerSum = 0
  const layerCounts = [0, 0, 0, 0]
  for (let pixel = 0; pixel < pixels; pixel++) {
    const base = pixel * DEPTH_LAYERS
    const surface = raster.surfaces[base]!
    topSurface[pixel] = surface
    if (surface === EMPTY_SURFACE) {
      height[pixel] = 0.075
      continue
    }
    covered++
    height[pixel] = clamp01(raster.depths[base]!)
    topTissue[pixel] = raster.tissue[base]!
    topPigment[pixel] = raster.pigment[base]!
    let count = 0
    for (let layer = 0; layer < DEPTH_LAYERS; layer++) {
      if (raster.surfaces[base + layer] === EMPTY_SURFACE) break
      count++
      layerCounts[layer] = layerCounts[layer]! + 1
    }
    depthCount[pixel] = count
    layerSum += count
  }

  const blurNear = periodicBoxBlur(height, TILE, 3)
  const blurFar = periodicBoxBlur(height, TILE, 13)
  const ao = new Float32Array(pixels)
  const descriptor = new Uint8Array(pixels * 4)
  const gradientHistogram = new Uint32Array(512)
  const histogramMax = 0.32
  let heightSum = 0
  let heightSquareSum = 0
  let gradientSum = 0
  let highGradients = 0
  for (let y = 0; y < TILE; y++) {
    for (let x = 0; x < TILE; x++) {
      const pixel = y * TILE + x
      const h = height[pixel]!
      heightSum += h
      heightSquareSum += h * h
      const dx = (height[y * TILE + wrap(x + 1, TILE)]! - height[y * TILE + wrap(x - 1, TILE)]!) * 0.5
      const dy = (height[wrap(y + 1, TILE) * TILE + x]! - height[wrap(y - 1, TILE) * TILE + x]!) * 0.5
      const gradient = Math.hypot(dx, dy)
      gradientSum += gradient
      if (gradient > 0.085) highGradients++
      const bin = Math.min(gradientHistogram.length - 1, Math.floor((gradient / histogramMax) * gradientHistogram.length))
      gradientHistogram[bin] = gradientHistogram[bin]! + 1

      const valleyNear = Math.max(0, blurNear[pixel]! - h)
      const valleyFar = Math.max(0, blurFar[pixel]! - h)
      const count = depthCount[pixel]!
      const base = pixel * DEPTH_LAYERS
      const gap = count >= 2 ? Math.max(0, raster.depths[base]! - raster.depths[base + 1]!) : 0.20
      const congestion = count >= 2 ? Math.exp(-gap * 21) * (count / DEPTH_LAYERS) : 0
      const aperture = clamp01(0.96 - valleyNear * 4.6 - valleyFar * 2.3 - congestion * 0.17)
      ao[pixel] = Math.max(0.13, aperture)

      descriptor[pixel * 4] = topTissue[pixel]!
      descriptor[pixel * 4 + 1] = Math.round(h * 255)
      descriptor[pixel * 4 + 2] = Math.round(ao[pixel]! * 255)
      descriptor[pixel * 4 + 3] = topPigment[pixel]!
    }
  }

  const meanHeight = heightSum / pixels
  let capitulumSizeSum = 0
  let capitulumCount = 0
  for (const event of events) {
    if (event.family !== 'capitulum') continue
    capitulumSizeSum += event.size
    capitulumCount++
  }
  const metrics: AssemblageMetrics = {
    seed,
    events: events.length,
    capitula: counters.capitula,
    fascicles: counters.fascicles,
    branchPackets: counters.branchPackets,
    packetSamples: counters.packetSamples,
    eventsPerSquareMetre: events.length / (0.42 * 0.42),
    meanNominalCapitulumMm: (capitulumSizeSum / Math.max(1, capitulumCount)) * 1.55 * (420 / TILE),
    coverage: covered / pixels,
    meanDepthLayers: layerSum / pixels,
    depthLayerFractions: layerCounts.map((count) => count / pixels) as [number, number, number, number],
    meanHeight,
    heightStdDev: Math.sqrt(Math.max(0, heightSquareSum / pixels - meanHeight * meanHeight)),
    meanGradient: gradientSum / pixels,
    p95Gradient: histogramQuantile(gradientHistogram, histogramMax, 0.95, pixels),
    highGradientFraction: highGradients / pixels,
    generationMs: performance.now() - started,
  }
  return { descriptor, height, ao, topSurface, depthCount, events, metrics }
}

export function generateShapeAtlas(seed: number): { width: number; height: Float32Array; neutral: Uint8Array } {
  const width = 1024
  const cells = 4
  const cell = width / cells
  const raster = makeRaster(width, width, false)
  const counters: Counters = { nextSurface: 0, branchPackets: 0, packetSamples: 0, capitula: 0, fascicles: 0 }
  for (let gy = 0; gy < cells; gy++) {
    for (let gx = 0; gx < cells; gx++) {
      const id = gy * cells + gx
      const event: StructuralEvent = {
        id,
        family: 'capitulum',
        x: gx * cell + cell * 0.5,
        y: gy * cell + cell * 0.5,
        baseZ: 0.30,
        phase: unit(id, 901, seed) * TAU,
        size: 24.0 + gx * 7.0,
        openness: clamp01(0.16 + gx * 0.23 + gaussianish(id, 911, seed) * 0.06),
        density: clamp01(0.30 + gy * 0.19 + gaussianish(id, 921, seed) * 0.06),
        verticality: clamp01(0.25 + gy * 0.21 + gaussianish(id, 926, seed) * 0.06),
        asymmetry: gaussianish(id, 931, seed),
        curvature: gaussianish(id, 941, seed),
        vitality: 0.65,
        pigment: gy / (cells - 1),
      }
      renderCapitulum(raster, event, seed, counters)
    }
  }
  const height = new Float32Array(width * width)
  for (let pixel = 0; pixel < height.length; pixel++) {
    height[pixel] = raster.surfaces[pixel * DEPTH_LAYERS] === EMPTY_SURFACE ? 0 : raster.depths[pixel * DEPTH_LAYERS]!
  }
  return { width, height, neutral: neutralRgba(height, width, 35) }
}

export function scalarRgba(input: Float32Array): Uint8Array {
  const output = new Uint8Array(input.length * 4)
  for (let pixel = 0; pixel < input.length; pixel++) {
    const value = Math.round(clamp01(input[pixel]!) * 255)
    output[pixel * 4] = value
    output[pixel * 4 + 1] = value
    output[pixel * 4 + 2] = value
    output[pixel * 4 + 3] = 255
  }
  return output
}

export function depthCountRgba(input: Uint8Array): Uint8Array {
  const output = new Uint8Array(input.length * 4)
  const colors = [
    [8, 8, 12],
    [52, 88, 154],
    [47, 153, 125],
    [213, 170, 62],
    [226, 77, 67],
  ]
  for (let pixel = 0; pixel < input.length; pixel++) {
    const color = colors[Math.min(4, input[pixel]!)]!
    output[pixel * 4] = color[0]!
    output[pixel * 4 + 1] = color[1]!
    output[pixel * 4 + 2] = color[2]!
    output[pixel * 4 + 3] = 255
  }
  return output
}

export function neutralRgba(height: Float32Array, width: number, azimuthDegrees: number): Uint8Array {
  const output = new Uint8Array(height.length * 4)
  const angle = (azimuthDegrees / 180) * Math.PI
  const lx = Math.cos(angle) * 0.58
  const ly = Math.sin(angle) * 0.58
  const lz = 0.57
  for (let y = 0; y < width; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x
      const left = height[y * width + wrap(x - 1, width)]!
      const right = height[y * width + wrap(x + 1, width)]!
      const down = height[wrap(y - 1, width) * width + x]!
      const up = height[wrap(y + 1, width) * width + x]!
      const gx = (right - left) * 7.5
      const gy = (up - down) * 7.5
      const inv = 1 / Math.hypot(gx, gy, 1)
      const light = clamp01((-gx * lx - gy * ly + lz) * inv * 0.72 + 0.28)
      const value = Math.round(light * 255)
      output[pixel * 4] = value
      output[pixel * 4 + 1] = value
      output[pixel * 4 + 2] = value
      output[pixel * 4 + 3] = 255
    }
  }
  return output
}

export function albedoRgba(descriptor: Uint8Array): Uint8Array {
  const output = new Uint8Array(descriptor.length)
  for (let pixel = 0; pixel < descriptor.length / 4; pixel++) {
    const fresh = descriptor[pixel * 4]! / 255
    const aperture = descriptor[pixel * 4 + 2]! / 255
    const pigment = descriptor[pixel * 4 + 3]! / 255
    const green = [0.16, 0.43, 0.035]
    const lime = [0.49, 0.68, 0.075]
    const ochre = [0.50, 0.31, 0.045]
    const rose = [0.48, 0.11, 0.065]
    const young = green.map((value, channel) => value * (1 - fresh) + lime[channel]! * fresh)
    const aged = ochre.map((value, channel) => value * (1 - pigment) + rose[channel]! * pigment)
    const ageMix = smooth(clamp01((pigment - 0.38) * 1.55))
    const color = young.map((value, channel) => value * (1 - ageMix) + aged[channel]! * ageMix)
    const cavity = 0.76 + aperture * 0.24
    output[pixel * 4] = Math.round(clamp01(color[0]! * cavity) * 255)
    output[pixel * 4 + 1] = Math.round(clamp01(color[1]! * cavity) * 255)
    output[pixel * 4 + 2] = Math.round(clamp01(color[2]! * cavity) * 255)
    output[pixel * 4 + 3] = 255
  }
  return output
}

export function shadedRgba(albedo: Uint8Array, neutral: Uint8Array, ao: Float32Array): Uint8Array {
  const output = new Uint8Array(albedo.length)
  for (let pixel = 0; pixel < albedo.length / 4; pixel++) {
    const light = 0.42 + neutral[pixel * 4]! / 255 * 0.72
    const aperture = 0.58 + ao[pixel]! * 0.42
    output[pixel * 4] = Math.min(255, Math.round(albedo[pixel * 4]! * light * aperture))
    output[pixel * 4 + 1] = Math.min(255, Math.round(albedo[pixel * 4 + 1]! * light * aperture))
    output[pixel * 4 + 2] = Math.min(255, Math.round(albedo[pixel * 4 + 2]! * light * aperture))
    output[pixel * 4 + 3] = 255
  }
  return output
}

export function crop(input: Uint8Array, sourceWidth: number, x0: number, y0: number, size: number): Uint8Array {
  const output = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y++) {
    const start = ((y0 + y) * sourceWidth + x0) * 4
    output.set(input.subarray(start, start + size * 4), y * size * 4)
  }
  return output
}

export function downsample(input: Float32Array, width: number): Float32Array {
  const next = width >> 1
  const output = new Float32Array(next * next)
  for (let y = 0; y < next; y++) {
    for (let x = 0; x < next; x++) {
      const source = y * 2 * width + x * 2
      output[y * next + x] =
        (input[source]! + input[source + 1]! + input[source + width]! + input[source + width + 1]!) * 0.25
    }
  }
  return output
}
