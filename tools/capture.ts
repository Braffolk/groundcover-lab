/**
 * Playwright batch runner over all experiments. Thin by design.
 *
 *   npm run capture:thumbs    thumbnails for every experiment (headless ok)
 *   npm run capture:goldens   goldens at the 4 standard cams (headless ok)
 *   npm run bench:all         bench every experiment (HEADED — headless GPU
 *                             scheduling distorts frame times; never trust
 *                             headless numbers)
 *
 * Requires the dev server to be running (npm run dev) — results/thumbnails
 * land in the repo via its sink endpoints.
 */
import { existsSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const BASE = process.env['GC_BASE_URL'] ?? 'http://localhost:5173'
const STANDARD_CAMS = ['grazing', 'topdown', 'inside-plant', 'far-horizon']
const mode = process.argv[2]

if (mode !== 'thumbs' && mode !== 'goldens' && mode !== 'bench') {
  console.error('usage: tsx tools/capture.ts <thumbs|goldens|bench>')
  process.exit(1)
}

const root = path.resolve(import.meta.dirname, '..')

/**
 * Experiment ids, found by walking for `manifest.ts` at any depth — experiments
 * are a tree now (materials/<class>/<subject>/<attempt>), and an id is the last
 * path segment rather than a child of experiments/.
 */
function findExperimentIds(dir: string): string[] {
  const out: string[] = []
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (!e.isDirectory() || e.name.startsWith('_') || e.name.startsWith('.')) continue
    const child = path.join(dir, e.name)
    if (existsSync(path.join(child, 'manifest.ts'))) out.push(e.name)
    else out.push(...findExperimentIds(child))
  }
  return out
}

const experiments = findExperimentIds(path.join(root, 'experiments')).sort()

if (experiments.length === 0) {
  console.error('no experiments found')
  process.exit(1)
}

try {
  // Must be OUR dev server (its sink returns a JSON array) — another Vite app
  // on the same port would return index.html with a 200.
  const body = await (await fetch(`${BASE}/__bench/list`)).text()
  if (!Array.isArray(JSON.parse(body))) throw new Error('not the groundcover sink')
} catch {
  console.error(`${BASE} is not the groundcover dev server — run \`npm run dev\` and check the port (GC_BASE_URL=... to override)`)
  process.exit(1)
}

const headless = mode !== 'bench'
const browser = await chromium.launch({
  headless,
  channel: 'chrome', // system Chrome — no playwright browser download needed
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--hide-scrollbars'],
})
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } })
page.on('console', (msg) => {
  if (msg.type() === 'error') console.log(`  [console.error] ${msg.text()}`)
})

async function settleAndClick(url: string, buttonText: string, settleMs: number): Promise<boolean> {
  await page.goto(url)
  const button = page.getByRole('button', { name: buttonText })
  try {
    await button.waitFor({ state: 'visible', timeout: 30_000 })
  } catch {
    console.log(`  ✗ no "${buttonText}" button (experiment failed to start?) — ${url}`)
    return false
  }
  await page.waitForTimeout(settleMs) // let bakes/warmup settle
  await button.click()
  try {
    await page.locator('.toast', { hasText: 'saved' }).first().waitFor({ timeout: 60_000 })
    return true
  } catch {
    console.log(`  ✗ save toast never appeared — ${url}`)
    return false
  }
}

for (const id of experiments) {
  console.log(id)
  if (mode === 'thumbs') {
    const url = `${BASE}/#/run/${id}?det=1&t=2&cam=grazing`
    if (await settleAndClick(url, 'thumbnail', 2500)) console.log('  ✓ thumbnail')
  } else if (mode === 'goldens') {
    for (const cam of STANDARD_CAMS) {
      const url = `${BASE}/#/run/${id}?det=1&t=2&cam=${cam}`
      if (await settleAndClick(url, 'golden @cam', 2500)) console.log(`  ✓ golden ${cam}`)
    }
  } else {
    await page.goto(`${BASE}/#/bench/${id}`)
    try {
      await page.locator('.toast', { hasText: 'saved' }).first().waitFor({ timeout: 300_000 })
      console.log('  ✓ bench saved')
    } catch {
      console.log('  ✗ bench did not finish within 5min')
    }
  }
}

await browser.close()
