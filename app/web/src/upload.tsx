import { useEffect, useRef, useState } from 'react'
import type { TrackDoc } from './types'

const SPECIES = ['unknown', 'honeybee', 'vespidae', 'syrphidae', 'lepidoptera'] as const
const PANEL_W = 320

type Species = (typeof SPECIES)[number]

type AnalyzeResult = {
  clipId: string
  summary: Record<string, unknown>
  videoUrl?: string
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function trim(value: number): string {
  return String(Math.round(value * 1000) / 1000)
}

function countText(value: unknown): string {
  const n = asNumber(value)
  return n == null ? '—' : String(Math.round(n))
}

function dwellText(value: unknown): string {
  const n = asNumber(value)
  return n == null ? '—' : `${trim(n)} s`
}

function speedText(summary: Record<string, unknown>): string {
  const cm = asNumber(summary.median_flight_speed_cm_s)
  if (cm != null) return `${trim(cm)} cm/s`
  const units = asNumber(summary.median_flight_speed_units_s)
  if (units != null) return `${trim(units)} units/s`
  return '—'
}

function errorText(value: unknown): string | null {
  const error = asRecord(value)?.error
  return typeof error === 'string' && error.trim() ? error : null
}

function isResult(value: unknown): value is AnalyzeResult {
  const row = asRecord(value)
  if (!row || typeof row.clipId !== 'string' || !row.clipId) return false
  if (!asRecord(row.summary)) return false
  return row.videoUrl === undefined || typeof row.videoUrl === 'string'
}

function isForecast(value: unknown): value is Record<string, unknown> {
  const row = asRecord(value)
  if (!row) return false
  return typeof row.track_id === 'string' || typeof row.t === 'number'
}

function lastTarget(doc: TrackDoc): { trackId: string; t: number } | null {
  const track = doc.tracks?.[doc.tracks.length - 1]
  if (!track?.track_id) return null
  let last = Number.NEGATIVE_INFINITY
  for (const point of track.points ?? []) {
    if (typeof point.t === 'number' && Number.isFinite(point.t) && point.t >= last) last = point.t
  }
  if (last === Number.NEGATIVE_INFINITY) return null
  return { trackId: track.track_id, t: Math.round(last * 1000) / 1000 }
}

function forecastSentence(data: Record<string, unknown>): string {
  const rows = Array.isArray(data.next_flower) ? data.next_flower : []
  const parts: string[] = []
  for (const item of rows) {
    const row = asRecord(item)
    const p = asNumber(row?.p)
    if (!row || typeof row.flower_id !== 'string' || !row.flower_id || p == null) continue
    parts.push(`${row.flower_id} ${(p * 100).toFixed(1)}%`)
  }
  const another = asNumber(data.p_another_visit)
  return `Next flower: ${parts.length ? parts.join(', ') : '—'}. p_another_visit ${another == null ? '—' : trim(another)}.`
}

export function UploadButton() {
  const [open, setOpen] = useState(false)
  const [api, setApi] = useState<boolean | null>(null)
  const [useApi, setUseApi] = useState(false)
  const [species, setSpecies] = useState<Species>('unknown')
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [forecastLine, setForecastLine] = useState<string | null>(null)
  const [pos, setPos] = useState({ top: 32, left: 8 })
  const buttonRef = useRef<HTMLButtonElement>(null)
  const gen = useRef(0)

  useEffect(() => {
    const ac = new AbortController()
    fetch('/api/health', { signal: ac.signal })
      .then(async (res) => {
        if (!res.ok) return false
        const body: unknown = await res.json()
        return asRecord(body)?.api === true
      })
      .then((on) => {
        if (!ac.signal.aborted) setApi(on)
      })
      .catch(() => {
        if (!ac.signal.aborted) setApi(false)
      })
    return () => ac.abort()
  }, [])

  useEffect(() => {
    if (api === false) setUseApi(false)
  }, [api])

  useEffect(() => {
    if (!open) return
    const place = () => {
      const rect = buttonRef.current?.getBoundingClientRect()
      if (!rect) return
      const left = Math.max(8, Math.min(rect.right - PANEL_W, window.innerWidth - PANEL_W - 8))
      setPos({ top: rect.bottom + 4, left })
    }
    place()
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const analyze = async () => {
    if (!file || busy) return
    const ticket = ++gen.current
    setBusy(true)
    setError(null)
    setResult(null)
    setForecastLine(null)
    const body = new FormData()
    body.append('video', file)
    body.append('useApi', useApi ? '1' : '0')
    body.append('species', species)
    try {
      const res = await fetch('/api/analyze', { method: 'POST', body })
      const payload: unknown = await res.json().catch(() => null)
      if (ticket !== gen.current) return
      if (!res.ok || !isResult(payload)) {
        setError(errorText(payload) ?? 'Analyze failed.')
        return
      }
      setResult(payload)
      void loadForecast(payload.clipId, ticket)
    } catch {
      if (ticket === gen.current) setError('Analyze failed.')
    } finally {
      if (ticket === gen.current) setBusy(false)
    }
  }

  return (
    <div className="up-anchor">
      <style>{CSS}</style>
      <button
        ref={buttonRef}
        type="button"
        className={open ? 'toggle on' : 'toggle'}
        aria-expanded={open}
        aria-controls="upload-panel"
        onClick={() => setOpen((on) => !on)}
      >
        Upload
      </button>
      {open && (
        <div
          id="upload-panel"
          className="panel up-pop"
          role="dialog"
          aria-label="Upload a clip"
          style={{ top: pos.top, left: pos.left }}
        >
          <div className="region-header">
            <span>Upload</span>
            <span className="header-spacer" />
            <button type="button" className="toggle" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
          <div className="region-body up-body">
            <p className="up-note">Fixed camera, insects on flowers. The first 60 seconds are used.</p>
            <div className="up-row">
              <button
                type="button"
                className={useApi ? 'toggle on' : 'toggle'}
                aria-pressed={useApi}
                disabled={api !== true}
                onClick={() => setUseApi((on) => !on)}
              >
                Mk1.5
              </button>
              {api === false && (
                <span className="up-hint" id="upload-api-hint">
                  Set PERCEPTRON_API_KEY in .env
                </span>
              )}
            </div>
            <label className="up-field">
              <span>Species</span>
              <select
                className="clip-select"
                aria-label="Species"
                value={species}
                onChange={(event) => setSpecies(event.target.value as Species)}
              >
                {SPECIES.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>
            <label className="up-field">
              <span>Video</span>
              <input
                type="file"
                aria-label="Video"
                accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
                onChange={(event) => {
                  setFile(event.target.files?.[0] ?? null)
                  setError(null)
                }}
              />
            </label>
            <button type="button" className="toggle" disabled={!file || busy} onClick={() => void analyze()}>
              {busy ? 'Analyzing…' : 'Analyze'}
            </button>
            {error && <p className="up-error">{error}</p>}
            {result && (
              <div className="up-result" aria-live="polite">
                {result.videoUrl && (
                  <video className="up-video" src={result.videoUrl} controls playsInline preload="metadata" aria-label={result.clipId} />
                )}
                <div className="props">
                  <div className="prop">
                    <span className="prop-key">insects</span>
                    <span className="prop-value">{countText(result.summary.insects)}</span>
                  </div>
                  <div className="prop">
                    <span className="prop-key">visits</span>
                    <span className="prop-value">{countText(result.summary.visits)}</span>
                  </div>
                  <div className="prop">
                    <span className="prop-key">flowers</span>
                    <span className="prop-value">{countText(result.summary.flowers_detected)}</span>
                  </div>
                  <div className="prop">
                    <span className="prop-key">median dwell</span>
                    <span className="prop-value">{dwellText(result.summary.median_dwell_s)}</span>
                  </div>
                  <div className="prop">
                    <span className="prop-key">flight speed</span>
                    <span className="prop-value">{speedText(result.summary)}</span>
                  </div>
                </div>
                <a className="up-download" href={`/tracks/uploads/${encodeURIComponent(result.clipId)}.json`} download={`${result.clipId}.json`}>
                  Download tracks JSON
                </a>
                {forecastLine && <p className="up-forecast">{forecastLine}</p>}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )

  async function loadForecast(clipId: string, ticket: number) {
    try {
      const docRes = await fetch(`/tracks/uploads/${encodeURIComponent(clipId)}.json`)
      if (!docRes.ok || ticket !== gen.current) return
      const doc = (await docRes.json()) as TrackDoc
      const target = lastTarget(doc)
      if (!target || ticket !== gen.current) return
      const res = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clipId,
          source: 'tracks/uploads',
          trackId: target.trackId,
          t: target.t,
          horizonS: 10,
        }),
      })
      if (!res.ok || ticket !== gen.current) return
      const data: unknown = await res.json()
      if (ticket !== gen.current || !isForecast(data)) return
      setForecastLine(forecastSentence(data))
    } catch {
      // The analyze result stands on its own when a forecast is not available.
    }
  }
}

const CSS = `
.up-anchor {
  display: flex;
  align-items: center;
}
.up-pop {
  position: fixed;
  z-index: 40;
  width: 320px;
  max-height: min(70vh, 560px);
  border: 1px solid #191919;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
}
.up-body {
  padding: 8px;
  overflow: auto;
}
.up-note,
.up-hint,
.up-forecast {
  margin: 0;
  color: #8d8d8d;
  line-height: 1.35;
}
.up-note {
  margin-bottom: 8px;
}
.up-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.up-field {
  display: block;
  margin: 0 0 8px;
}
.up-field > span {
  display: block;
  margin-bottom: 2px;
  color: #bdbdbd;
}
.up-field .clip-select,
.up-field input[type='file'] {
  width: 100%;
  max-width: none;
}
.up-field input[type='file'] {
  color: inherit;
  font: inherit;
}
.up-pop .toggle:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.up-pop .toggle:disabled:hover {
  background: transparent;
}
.up-error {
  margin: 8px 0 0;
  color: #f3d2c4;
  line-height: 1.35;
}
.up-video {
  display: block;
  width: 100%;
  max-height: 160px;
  margin-top: 8px;
  background: #111;
}
.up-body .prop {
  grid-template-columns: 96px minmax(0, 1fr);
}
.up-download {
  display: inline-block;
  margin: 4px 8px 0;
  color: inherit;
}
.up-forecast {
  margin: 8px 8px 0;
}
`
