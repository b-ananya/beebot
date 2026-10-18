import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/** A label with an explanation on hover or keyboard focus. The bubble is portaled to <body> with fixed
 *  positioning, so scrolling panels (overflow: auto) never clip it. */
export function Hint({ tip, children, block = false }: { tip: ReactNode; children: ReactNode; block?: boolean }) {
  const id = useId()
  const anchor = useRef<HTMLSpanElement>(null)
  const bubble = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  useLayoutEffect(() => {
    if (!open || !anchor.current || !bubble.current) return
    const a = anchor.current.getBoundingClientRect()
    const b = bubble.current.getBoundingClientRect()
    const margin = 8
    let left = a.left
    if (left + b.width > window.innerWidth - margin) left = window.innerWidth - margin - b.width
    left = Math.max(margin, left)
    let top = a.bottom + 6
    if (top + b.height > window.innerHeight - margin) top = Math.max(margin, a.top - 6 - b.height)
    setPos({ left, top })
  }, [open])

  const show = () => setOpen(true)
  const hide = () => {
    setOpen(false)
    setPos(null)
  }

  return (
    <span
      ref={anchor}
      className={block ? 'hint hint-block' : 'hint'}
      tabIndex={0}
      aria-describedby={open ? id : undefined}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onKeyDown={(event) => {
        if (event.key === 'Escape') hide()
      }}
    >
      {children}
      <span className="hint-mark" aria-hidden="true">?</span>
      {open &&
        createPortal(
          <div
            ref={bubble}
            id={id}
            role="tooltip"
            className="hint-bubble"
            style={pos ? { left: pos.left, top: pos.top } : { left: -9999, top: -9999 }}
          >
            {tip}
          </div>,
          document.body,
        )}
    </span>
  )
}

/** A short "what am I looking at" box at the top of a panel. */
export function Intro({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="intro" aria-label={title}>
      <div className="intro-title">{title}</div>
      <div className="intro-body">{children}</div>
    </section>
  )
}
