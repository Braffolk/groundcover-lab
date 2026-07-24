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
import { readdirSync } from 'node:fs'
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
const experiments = readdirSync(path.join(root, 'experiments'), { withFileTypes: true })
  .filter((e) => e.isDirectory() && !e.name.startsWith('_'))
  .map((e) => e.name)
  .sort()

if (experiments.length === 0) {
  console.error('no experiments found')
  process.exit(1)
}

try {
  await fetch(`${BASE}/__bench/list`)
} catch {
  console.error(`dev server not reachable at ${BASE} — run \`npm run dev\` first`)
  process.exit(1)
}

const headless = mode !== 'bench'
const browser = await chromium.launch({
  headless,
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
    await page.locator('.toast', { hasText: 'saved' }).waitFor({ timeout: 60_000 })
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
    if (await settleAndClick(url, '📷 thumbnail', 2500)) console.log('  ✓ thumbnail')
  } else if (mode === 'goldens') {
    for (const cam of STANDARD_CAMS) {
      const url = `${BASE}/#/run/${id}?det=1&t=2&cam=${cam}`
      if (await settleAndClick(url, 'golden @cam', 2500)) console.log(`  ✓ golden ${cam}`)
    }
  } else {
    await page.goto(`${BASE}/#/bench/${id}`)
    try {
      await page.locator('.toast', { hasText: 'saved' }).waitFor({ timeout: 300_000 })
      console.log('  ✓ bench saved')
    } catch {
      console.log('  ✗ bench did not finish within 5min')
    }
  }
}

await browser.close()
