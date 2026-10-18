import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { HELP } from './help'
import { Hint } from './hint'
import { query, tokens, type Moment } from './search'

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0
  const total = Math.round(seconds * 100)
  const hundredths = total % 100
  const wholeSeconds = Math.floor(total / 100)
  const minutes = Math.floor(wholeSeconds / 60)
  const whole = wholeSeconds % 60
  return `${minutes}:${whole.toString().padStart(2, '0')}.${hundredths.toString().padStart(2, '0')}`
}

function asMoment(value: unknown): Moment | null {
  if (!value || typeof value !== 'object') return null
  const row = value as Record<string, unknown>
  if (typeof row.clip_id !== 'string' || typeof row.text !== 'string') return null
  if (typeof row.t_start !== 'number' || typeof row.t_end !== 'number') return null
  if (!Number.isFinite(row.t_start) || !Number.isFinite(row.t_end)) return null
  const moment: Moment = {
    clip_id: row.clip_id,
    t_start: row.t_start,
    t_end: row.t_end,
    text: row.text,
  }
  if (typeof row.kind === 'string') moment.kind = row.kind
  if (typeof row.source === 'string') moment.source = row.source
  return moment
}

export function MomentSearch({
  index,
  clipId,
  onOpen,
}: {
  index: Moment[] | null
  clipId: string
  onOpen: (moment: Moment) => void
}) {
  const [text, setText] = useState('')
  const [scope, setScope] = useState<'all' | 'clip'>('all')
  const [live, setLive] = useState(false)
  const [liveHits, setLiveHits] = useState<Moment[] | null>(null)
  const [liveNote, setLiveNote] = useState<string | null>(null)
  const liveGen = useRef(0)

  const offlineHits = useMemo(() => {
    if (!index || tokens(text).length === 0) return []
    return query(text, index, scope === 'clip' ? clipId : null)
  }, [index, text, scope, clipId])

  const hits = live ? (liveHits ?? []) : offlineHits
  const asked = tokens(text).length > 0

  const openHit = (hit: Moment) => {
    onOpen({ ...hit, t_start: Math.max(hit.t_start - 0.5, 0) })
  }

  const runLive = async (q: string) => {
    const gen = ++liveGen.current
    setLiveNote(null)
    setLiveHits(null)
    try {
      const res = await fetch('/api/search/live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clipId, q }),
      })
      if (gen !== liveGen.current) return
      if (!res.ok) {
        setLiveNote('Live search is unavailable.')
        return
      }
      const body: unknown = await res.json()
      if (gen !== liveGen.current) return
      if (!Array.isArray(body)) {
        setLiveNote('Live search is unavailable.')
        return
      }
      setLiveHits(body.flatMap((row) => {
        const moment = asMoment(row)
        return moment ? [moment] : []
      }))
    } catch {
      if (gen !== liveGen.current) return
      setLiveNote('Live search is unavailable.')
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter' || event.nativeEvent.isComposing || !live) return
    event.preventDefault()
    void runLive(text)
  }

  const toggleLive = () => {
    liveGen.current += 1
    setLive((on) => !on)
    setLiveHits(null)
    setLiveNote(null)
  }

  return (
    <div className="search">
      <input
        type="search"
        aria-label="Search"
        placeholder="Search moments"
        value={text}
        onChange={(event) => {
          setText(event.target.value)
          if (live) {
            liveGen.current += 1
            setLiveHits(null)
            setLiveNote(null)
          }
        }}
        onKeyDown={onKeyDown}
      />
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 4, marginTop: 4 }}>
        <div role="group" aria-label="Search scope" style={{ display: 'flex', gap: 4 }}>
          <button
            type="button"
            className={scope === 'all' ? 'toggle on' : 'toggle'}
            aria-pressed={scope === 'all'}
            onClick={() => setScope('all')}
          >
            All clips
          </button>
          <button
            type="button"
            className={scope === 'clip' ? 'toggle on' : 'toggle'}
            aria-pressed={scope === 'clip'}
            onClick={() => setScope('clip')}
          >
            This clip
          </button>
        </div>
        <Hint tip={HELP.liveSearch}>
          <button type="button" className={live ? 'toggle on' : 'toggle'} aria-pressed={live} onClick={toggleLive}>
            Live
          </button>
        </Hint>
      </div>
      {hits.map((hit, index) => (
        <button
          key={`${hit.clip_id}-${hit.t_start}-${hit.t_end}-${index}`}
          type="button"
          className="search-hit"
          onClick={() => openHit(hit)}
        >
          <span className="meta" style={{ flexWrap: 'wrap' }}>
            <span>{hit.clip_id}</span>
            <span>{formatTime(hit.t_start)}–{formatTime(hit.t_end)}</span>
            <span style={{ flex: '1 0 100%' }}>{[hit.kind, hit.source].filter((part) => part).join(' · ')}</span>
          </span>
          <span className="text">{hit.text}</span>
        </button>
      ))}
      {live && liveNote && <div className="search-empty">{liveNote}</div>}
      {!live && asked && index && hits.length === 0 && <div className="search-empty">No matches</div>}
      {live && !liveNote && liveHits && liveHits.length === 0 && <div className="search-empty">No matches</div>}
    </div>
  )
}
