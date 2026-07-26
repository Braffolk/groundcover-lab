import { mkdirSync } from 'node:fs'
import path from 'node:path'
import { chromium, type Page } from 'playwright'

const id = '025-statistical-assemblage'
const base = process.env['GC_BASE_URL'] ?? 'http://127.0.0.1:5175'
const outDir = path.join(import.meta.dirname, 'captures')
mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch({
  headless: true,
  channel: 'chrome',
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--hide-scrollbars'],
})
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors: string[] = []
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text())
})
page.on('pageerror', (error) => errors.push(error.message))

async function waitForReport(p: Page): Promise<number> {
  const started = Date.now()
  await p.waitForFunction(
    (experimentId) =>
      (globalThis as typeof globalThis & { __materialReports?: Record<string, unknown> }).__materialReports?.[
        experimentId as string
      ] !== undefined,
    id,
    { timeout: 300_000 },
  )
  await p.waitForTimeout(600)
  return Date.now() - started
}

async function runner(
  name: string,
  obj: 'sphere' | 'cube-edge' | 'plane',
  tileScale: 1 | 8,
  reliefGain: 1.2 | 2,
  cam: 'three-quarter' | 'macro' = 'three-quarter',
) {
  const url = `${base}/#/run/${id}?det=1&t=2&cam=${cam}&obj=${obj}` +
    `&p.texPx=2048&p.tileScale=${tileScale}&p.reliefGain=${reliefGain}` +
    '&p.normalStrength=0.92&p.microShadow=1&p.stateVariation=0.72'
  await page.goto(url, { waitUntil: 'domcontentloaded' })
  const elapsedMs = await waitForReport(page)
  const toasts = await page.locator('.toast').allTextContents()
  await page.evaluate(() => {
    for (const selector of ['.hud', '.pane-holder', '.topbar', '.toolbar', '.pips', '.run-bar', '.run-controls', '.tp-dfwv']) {
      for (const node of Array.from(document.querySelectorAll(selector))) (node as HTMLElement).style.display = 'none'
    }
  })
  const output = path.join(outDir, `${name}.png`)
  await page.locator('canvas').first().screenshot({ path: output })
  return { name, url: page.url(), output, elapsedMs, toasts }
}

const runCaptures = [
  await runner('qa-plane-scale1-macro', 'plane', 1, 1.2, 'macro'),
  await runner('qa-sphere-scale1', 'sphere', 1, 1.2),
  await runner('qa-sphere-scale8-gain2', 'sphere', 8, 2, 'macro'),
  await runner('qa-cube-scale1', 'cube-edge', 1, 1.2),
  await runner('qa-cube-scale8-gain2', 'cube-edge', 8, 2, 'macro'),
]

await page.goto(
  `${base}/#/material/${id}?p.texPx=2048&p.tileScale=1&p.reliefGain=2&p.normalStrength=0.92&p.microShadow=1`,
  { waitUntil: 'domcontentloaded' },
)
const inspectorElapsedMs = await waitForReport(page)
const channelCaptures: Array<{ node: string; output: string }> = []
for (const nodeId of ['height', 'albedo', 'normal', 'ao']) {
  const node = page.locator('.mat-node').filter({ hasText: nodeId }).first()
  await node.click()
  await page.waitForTimeout(250)
  const output = path.join(outDir, `qa-raw-${nodeId}.png`)
  await page.locator('canvas.mat-canvas').screenshot({ path: output })
  channelCaptures.push({ node: nodeId, output })
}
const report = await page.evaluate((experimentId) => {
  const value = (globalThis as typeof globalThis & {
    __materialReports?: Record<string, { ok: boolean; textureBytes: number; issues: unknown[] }>
  }).__materialReports?.[experimentId as string]
  return value && { ok: value.ok, textureBytes: value.textureBytes, issues: value.issues }
}, id)
const inspectorToasts = await page.locator('.toast').allTextContents()

console.log(JSON.stringify({ runCaptures, channelCaptures, inspectorElapsedMs, errors, inspectorToasts, report }, null, 2))
await browser.close()
