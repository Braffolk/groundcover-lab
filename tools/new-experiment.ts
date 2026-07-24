/**
 * Scaffold a new experiment from experiments/_template.
 * Usage: npm run new -- <slug>       (slug: lowercase, digits, dashes)
 *
 * This is the ENTIRE registration process — discovery picks the folder up
 * automatically; no shared file is touched, so parallel agents never conflict.
 */
import { cpSync, existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const slug = process.argv[2]

if (!slug || !/^[a-z0-9][a-z0-9-]*$/.test(slug)) {
  console.error('usage: npm run new -- <slug>   (lowercase letters, digits, dashes)')
  process.exit(1)
}

const experimentsDir = path.join(root, 'experiments')
const numbers = readdirSync(experimentsDir, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => /^(\d{3})-/.exec(e.name)?.[1])
  .filter((n): n is string => n !== undefined)
  .map(Number)
const next = String(numbers.length > 0 ? Math.max(...numbers) + 1 : 1).padStart(3, '0')
const id = `${next}-${slug}`
const dir = path.join(experimentsDir, id)

if (existsSync(dir)) {
  console.error(`${dir} already exists`)
  process.exit(1)
}

cpSync(path.join(experimentsDir, '_template'), dir, { recursive: true })

const title = slug.replace(/-/g, ' ')
for (const file of ['manifest.ts', 'NOTES.md']) {
  const p = path.join(dir, file)
  writeFileSync(p, readFileSync(p, 'utf8').replaceAll('__SLUG__', id).replaceAll('__NAME__', title))
}

console.log(`created experiments/${id}/`)
console.log(`run it:  http://localhost:5173/#/run/${id}`)
console.log(`rules:   see CLAUDE.md — edit ONLY files inside experiments/${id}/`)
