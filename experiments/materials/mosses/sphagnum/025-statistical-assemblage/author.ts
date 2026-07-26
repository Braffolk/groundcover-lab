import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import {
  TILE,
  albedoRgba,
  crop,
  depthCountRgba,
  downsample,
  generateAssemblage,
  generateShapeAtlas,
  neutralRgba,
  scalarRgba,
  shadedRgba,
} from './generator.ts'
import { encodeRgbaPng } from './png.ts'

const output = path.join(import.meta.dirname, 'captures')
const assets = path.join(import.meta.dirname, 'assets')
await Promise.all([mkdir(output, { recursive: true }), mkdir(assets, { recursive: true })])

const result = generateAssemblage(2507)
const neutral = neutralRgba(result.height, TILE, 35)
const albedo = albedoRgba(result.descriptor)
const shaded = shadedRgba(albedo, neutral, result.ao)
const centre = (TILE - 768) >> 1
const atlas = generateShapeAtlas(2507)

let mip = result.height
let mipWidth = TILE
for (let level = 1; level <= 3; level++) {
  mip = downsample(mip, mipWidth)
  mipWidth >>= 1
  await writeFile(path.join(output, `height-plus${level}.png`), await encodeRgbaPng(mipWidth, mipWidth, scalarRgba(mip)))
}

await Promise.all([
  writeFile(path.join(assets, 'descriptor.png'), await encodeRgbaPng(TILE, TILE, result.descriptor)),
  writeFile(path.join(output, 'shape-atlas-height.png'), await encodeRgbaPng(atlas.width, atlas.width, scalarRgba(atlas.height))),
  writeFile(path.join(output, 'shape-atlas-neutral.png'), await encodeRgbaPng(atlas.width, atlas.width, atlas.neutral)),
  writeFile(path.join(output, 'raw-height.png'), await encodeRgbaPng(TILE, TILE, scalarRgba(result.height))),
  writeFile(path.join(output, 'raw-ao.png'), await encodeRgbaPng(TILE, TILE, scalarRgba(result.ao))),
  writeFile(path.join(output, 'depth-count.png'), await encodeRgbaPng(TILE, TILE, depthCountRgba(result.depthCount))),
  writeFile(path.join(output, 'raw-albedo.png'), await encodeRgbaPng(TILE, TILE, albedo)),
  writeFile(path.join(output, 'neutral.png'), await encodeRgbaPng(TILE, TILE, neutral)),
  writeFile(path.join(output, 'shaded.png'), await encodeRgbaPng(TILE, TILE, shaded)),
  writeFile(path.join(output, 'centre-height.png'), await encodeRgbaPng(768, 768, crop(scalarRgba(result.height), TILE, centre, centre, 768))),
  writeFile(path.join(output, 'centre-neutral.png'), await encodeRgbaPng(768, 768, crop(neutral, TILE, centre, centre, 768))),
  writeFile(path.join(output, 'centre-shaded.png'), await encodeRgbaPng(768, 768, crop(shaded, TILE, centre, centre, 768))),
  writeFile(path.join(import.meta.dirname, 'thumbnail.png'), await encodeRgbaPng(768, 768, crop(shaded, TILE, centre, centre, 768))),
  writeFile(path.join(output, 'metrics.json'), `${JSON.stringify(result.metrics, null, 2)}\n`),
])

console.log(JSON.stringify(result.metrics, null, 2))
