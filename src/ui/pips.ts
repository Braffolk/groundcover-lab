/**
 * 1-5 rating as five square pips — filled white = rating, hover previews,
 * click sets, clicking the current value clears. Monochrome by design.
 *
 * `readOnly` (static deployments, where nothing can be written back) keeps
 * the display and drops all interaction: no hover preview, no click, no
 * pointer cursor.
 */
export function pipsRow(opts: {
  value: number | null
  onSet: (value: number | null) => void
  title?: string
  readOnly?: boolean
}): { el: HTMLElement; set: (value: number | null) => void } {
  const row = document.createElement('div')
  row.className = opts.readOnly ? 'pips read-only' : 'pips'
  row.title = opts.title ?? (opts.readOnly ? "owner's visual rating (read-only)" : 'visual rating')
  let current = opts.value

  const pips: HTMLButtonElement[] = []
  const paint = (shown: number | null): void => {
    pips.forEach((pip, i) => pip.classList.toggle('filled', shown !== null && i < shown))
  }
  for (let i = 1; i <= 5; i++) {
    const pip = document.createElement('button')
    pip.className = 'pip'
    if (opts.readOnly) {
      pip.disabled = true
    } else {
      pip.addEventListener('mouseenter', () => paint(i))
      pip.addEventListener('click', (ev) => {
        ev.stopPropagation()
        current = current === i ? null : i
        paint(current)
        opts.onSet(current)
      })
    }
    pips.push(pip)
    row.appendChild(pip)
  }
  if (!opts.readOnly) row.addEventListener('mouseleave', () => paint(current))
  paint(current)

  return {
    el: row,
    set: (value) => {
      current = value
      paint(current)
    },
  }
}
