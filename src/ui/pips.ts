/**
 * 1-5 rating as five square pips — filled white = rating, hover previews,
 * click sets, clicking the current value clears. Monochrome by design.
 */
export function pipsRow(opts: {
  value: number | null
  onSet: (value: number | null) => void
  title?: string
}): { el: HTMLElement; set: (value: number | null) => void } {
  const row = document.createElement('div')
  row.className = 'pips'
  row.title = opts.title ?? 'visual rating'
  let current = opts.value

  const pips: HTMLButtonElement[] = []
  const paint = (shown: number | null): void => {
    pips.forEach((pip, i) => pip.classList.toggle('filled', shown !== null && i < shown))
  }
  for (let i = 1; i <= 5; i++) {
    const pip = document.createElement('button')
    pip.className = 'pip'
    pip.addEventListener('mouseenter', () => paint(i))
    pip.addEventListener('click', (ev) => {
      ev.stopPropagation()
      current = current === i ? null : i
      paint(current)
      opts.onSet(current)
    })
    pips.push(pip)
    row.appendChild(pip)
  }
  row.addEventListener('mouseleave', () => paint(current))
  paint(current)

  return {
    el: row,
    set: (value) => {
      current = value
      paint(current)
    },
  }
}
