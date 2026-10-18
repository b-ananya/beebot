import { useEffect, useMemo, useRef, useState } from 'react'
import { HELP } from './help'
import { Hint, InfoMark } from './hint'
import type { Flower, TrackPoint } from './overlay'
import type { ClipTrack, PanelProps } from './types'

const OBSERVED = '#2a78d6'
const FORECAST = '#eb6834'
const ACTUAL = '#8a8985'
const FLOWER = '#eda100'
const FLOWER_STOP = '#aaaaaa'
const PRESENCE_GAP_S = 0.6
const ACTUAL_HORIZON_S = 2
const SLIDER_DEBOUNCE_MS = 200
const FAN_ALPHA = 0.45

type Xy = [number, number]
type TimedPoint = { t: number; x: number; y: number }
type DrawnFlower = { box: [number, number, number, number]; source?: string }
type Phase = 'idle' | 'loading' | 'ready' | 'unavailable'

type ForecastResponse = {
  track_id?: string
  t?: number
  path?: { median?: unknown; fan?: unknown; history?: unknown; t?: unknown }
  dwell_model_s?: unknown
  dwell_remaining_s?: unknown
  next_flower?: unknown
  p_another_visit?: unknown
  rollout?: unknown
}

type DrawModel = {
  t: number
  flowers: DrawnFlower[]
  past: TimedPoint[]
  future: TimedPoint[]
  fan: Xy[][]
  median: Xy[]
}

type Bag = { key: string; phase: Phase; data: ForecastResponse | null }

export function ForecastPanel({ clipId, source, doc, manifestEntry, time, onSeek }: PanelProps) {
  const tracks = useMemo(() => listTracks(doc?.tracks), [doc])
  const [picked, setPicked] = useState<string | null>(null)
  const active = tracks.find((track) => track.track_id === picked) ?? tracks[0]
  const activeId = active?.track_id
  const range = useMemo(() => timeRange(active), [active])
  const key = activeId ? `${clipId}:${activeId}` : ''
  const [override, setOverride] = useState<{ key: string; t: number } | null>(null)
  const [horizon, setHorizon] = useState(10)
  const [bag, setBag] = useState<Bag>({ key: '', phase: 'idle', data: null })
  const [report, setReport] = useState<{ status: 'loading' | 'ready' | 'error'; data: unknown }>({
    status: 'loading',
    data: null,
  })
  const timeRef = useRef(time)
  const onSeekRef = useRef(onSeek)
  timeRef.current = time
  onSeekRef.current = onSeek

  const forecastT =
    range == null ? null : clamp(override?.key === key ? override.t : finite(time, range.lo), range.lo, range.hi)

  useEffect(() => {
    if (!activeId || range == null) return
    const nextKey = `${clipId}:${activeId}`
    setOverride((prev) => {
      if (prev?.key === nextKey) {
        const t = clamp(prev.t, range.lo, range.hi)
        return t === prev.t ? prev : { key: nextKey, t }
      }
      return { key: nextKey, t: clamp(finite(timeRef.current, range.lo), range.lo, range.hi) }
    })
  }, [clipId, activeId, range])

  const aligned = useRef('')
  useEffect(() => {
    if (!activeId || forecastT == null) return
    const alignKey = `${clipId}:${activeId}`
    if (aligned.current === alignKey) return
    aligned.current = alignKey
    if (Math.abs(timeRef.current - forecastT) > 0.08) onSeekRef.current(forecastT)
  }, [clipId, activeId, forecastT])

  const requestKey = activeId && forecastT != null ? `${clipId}|${activeId}|${round3(forecastT)}|${horizon}` : ''

  useEffect(() => {
    if (!activeId || forecastT == null || !requestKey) return
    const ac = new AbortController()
    const sentKey = requestKey
    const body = {
      clipId,
      source,
      trackId: activeId,
      t: round3(forecastT),
      horizonS: horizon,
    }
    const timer = window.setTimeout(() => {
      fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: ac.signal,
      })
        .then(async (res) => {
          if (!res.ok) throw new Error('unavailable')
          const type = res.headers.get('content-type') ?? ''
          if (!type.includes('json')) throw new Error('unavailable')
          const data: unknown = await res.json()
          if (!isForecast(data)) throw new Error('unavailable')
          return data
        })
        .then((data) => {
          if (ac.signal.aborted) return
          setBag({ key: sentKey, phase: 'ready', data })
        })
        .catch(() => {
          if (ac.signal.aborted) return
          setBag({ key: sentKey, phase: 'unavailable', data: null })
        })
    }, SLIDER_DEBOUNCE_MS)
    return () => {
      window.clearTimeout(timer)
      ac.abort()
    }
  }, [requestKey, clipId, source, activeId, forecastT, horizon])

  useEffect(() => {
    const ac = new AbortController()
    fetch('/forecast/report.json', { signal: ac.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error('report')
        const type = res.headers.get('content-type') ?? ''
        if (!type.includes('json')) throw new Error('report')
        return res.json() as Promise<unknown>
      })
      .then((data) => {
        if (!ac.signal.aborted) setReport({ status: 'ready', data })
      })
      .catch(() => {
        if (!ac.signal.aborted) setReport({ status: 'error', data: null })
      })
    return () => ac.abort()
  }, [])

  const settled = requestKey !== '' && bag.key === requestKey
  const phase: Phase = !requestKey ? 'idle' : settled ? bag.phase : 'loading'
  const data = settled && bag.phase === 'ready' ? bag.data : null

  const draw = useMemo(() => {
    if (!active || forecastT == null) return null
    const points = timedPoints(active)
    return {
      t: forecastT,
      flowers: flowerBoxes(doc?.flowers),
      past: points.filter((point) => point.t <= forecastT + 1e-6),
      future: points.filter((point) => point.t > forecastT + 1e-6 && point.t <= forecastT + ACTUAL_HORIZON_S + 1e-6),
      fan: data ? fanOf(data.path?.fan) : [],
      median: data ? lineOf(data.path?.median) : [],
    }
  }, [active, forecastT, doc, data])

  if (!doc) {
    return (
      <div className="region-body">
        <div className="empty-note">Loading track…</div>
      </div>
    )
  }
  if (tracks.length === 0) {
    return (
      <div className="region-body">
        <div className="empty-note">No insects tracked in this clip.</div>
      </div>
    )
  }

  const dwellCaption = data ? dwellHelp(data.dwell_model_s) : undefined

  return (
    <div className="region-body fc-shell">
      <style>{CSS}</style>
      {draw && activeId && (
        <ForecastFrame
          clipId={clipId}
          trackId={activeId}
          fallbackWidth={manifestEntry?.width ?? 0}
          fallbackHeight={manifestEntry?.height ?? 0}
          draw={draw}
          pathNote={phase === 'ready' && draw.median.length === 0}
        />
      )}
      <aside className="fc-side">
        <p className="fc-caption">
          Predicts where the selected insect goes next, using models trained on 2,335 recorded insect flights. The
          picture shows its path so far, the forecast for the next 2 s with a spread of possible futures, and (dotted)
          what really happened. Hover any label with an <InfoMark inline /> for details.
        </p>
        <label className="fc-field">
          <span className="fc-field-label">
            <Hint tip={HELP.insect}>Insect</Hint>
          </span>
          <select
            aria-label="Insect"
            value={activeId ?? ''}
            onChange={(event) => setPicked(event.target.value)}
          >
            {tracks.map((track) => (
              <option key={track.track_id} value={track.track_id}>
                {track.species ? `${track.track_id} (${track.species})` : track.track_id}
              </option>
            ))}
          </select>
        </label>
        {range && forecastT != null ? (
          <label className="fc-field">
            <span className="fc-field-label">
              <Hint tip={HELP.forecastFrom}>Forecast from</Hint>
              <span className="fc-field-value">{formatSeconds(forecastT)}</span>
            </span>
            <input
              type="range"
              aria-label="Forecast from"
              min={range.lo}
              max={range.hi > range.lo ? range.hi : range.lo + 0.1}
              step={0.1}
              value={forecastT}
              disabled={range.hi <= range.lo}
              onChange={(event) => {
                const next = clamp(Number(event.target.value), range.lo, range.hi)
                setOverride({ key, t: next })
                onSeek(next)
              }}
            />
          </label>
        ) : (
          <p className="fc-caption">This track has no observed times.</p>
        )}
        <label className="fc-field">
          <span className="fc-field-label">
            <Hint tip={HELP.horizon}>Horizon</Hint>
            <span className="fc-field-value">{horizon}s</span>
          </span>
          <input
            type="range"
            aria-label="Horizon"
            min={5}
            max={30}
            step={1}
            value={horizon}
            onChange={(event) => setHorizon(clamp(Math.round(Number(event.target.value)), 5, 30))}
          />
        </label>
        {clipId.startsWith('hb') && (
          <p className="fc-caption">
            Models were trained on the farm cameras, so honeybee forecasts are qualitative.
          </p>
        )}
        <div aria-live="polite">
          {phase === 'unavailable' && <p className="fc-unavailable">Forecast endpoint is unavailable</p>}
          {phase === 'loading' && <p className="fc-status">Forecasting…</p>}
        </div>
        <div className="fc-metrics">
          <Metric label="Dwell remaining" tip={HELP.dwellRemaining} value={data ? dwellText(data) : '—'}
            help={dwellCaption} />
          <Metric label="Another visit" tip={HELP.anotherVisit} value={data ? formatRate(num(data.p_another_visit)) : '—'} />
          <Metric label={`Leaves within ${horizon}s`} tip={HELP.leavesWithin} value={data ? leavesText(data) : '—'} />
          <Metric
            label="Most likely sequence"
            tip={HELP.likelySequence}
            value={data ? sequenceText(data) : '—'}
            help={data ? 'Expected visits below show the full picture.' : undefined}
          />
        </div>
        {data && <FlowerBars data={data} horizon={horizon} />}
        <ReportExpander report={report} />
      </aside>
    </div>
  )
}

function ForecastFrame({
  clipId,
  trackId,
  fallbackWidth,
  fallbackHeight,
  draw,
  pathNote,
}: {
  clipId: string
  trackId: string
  fallbackWidth: number
  fallbackHeight: number
  draw: DrawModel
  pathNote: boolean
}) {
  const frameRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawRef = useRef(draw)
  const paintRef = useRef<() => void>(() => {})
  drawRef.current = draw

  useEffect(() => {
    const frame = frameRef.current
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!frame || !video || !canvas) return

    const paint = () => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const vw = video.videoWidth || fallbackWidth
      const vh = video.videoHeight || fallbackHeight
      const rect = contentRect(frame.clientWidth, frame.clientHeight, vw, vh)
      if (!rect) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        return
      }
      const dpr = window.devicePixelRatio || 1
      canvas.style.left = `${rect.x}px`
      canvas.style.top = `${rect.y}px`
      canvas.style.width = `${rect.w}px`
      canvas.style.height = `${rect.h}px`
      const bw = Math.max(1, Math.round(rect.w * dpr))
      const bh = Math.max(1, Math.round(rect.h * dpr))
      if (canvas.width !== bw) canvas.width = bw
      if (canvas.height !== bh) canvas.height = bh
      paintForecast(ctx, drawRef.current, dpr)
    }
    paintRef.current = paint
    const observer = new ResizeObserver(paint)
    observer.observe(frame)
    video.addEventListener('loadedmetadata', paint)
    video.addEventListener('seeked', paint)
    paint()
    return () => {
      observer.disconnect()
      video.removeEventListener('loadedmetadata', paint)
      video.removeEventListener('seeked', paint)
      paintRef.current = () => {}
    }
  }, [clipId, fallbackWidth, fallbackHeight])

  useEffect(() => {
    paintRef.current()
  }, [draw])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let cancelled = false
    const seek = () => {
      if (cancelled) return
      video.pause()
      const limit = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : draw.t
      const next = clamp(draw.t, 0, Math.max(limit, 0))
      if (Math.abs(video.currentTime - next) > 0.03) video.currentTime = next
    }
    if (video.readyState >= 1) seek()
    else video.addEventListener('loadedmetadata', seek)
    return () => {
      cancelled = true
      video.removeEventListener('loadedmetadata', seek)
    }
  }, [clipId, draw.t])

  return (
    <section className="fc-frame" ref={frameRef} aria-label={`Path forecast for ${trackId}`}>
      <video
        key={clipId}
        ref={videoRef}
        src={`/clips/${clipId}.mp4`}
        muted
        playsInline
        preload="auto"
        aria-label={`${clipId} at ${formatSeconds(draw.t)}`}
      />
      <canvas ref={canvasRef} aria-hidden="true" />
      <div className="fc-title">
        <div>
          {trackId} at {draw.t.toFixed(1)}s: path forecast
        </div>
        {pathNote && <div className="fc-note">Not enough history here for a path forecast.</div>}
      </div>
      <div className="fc-legend">
        <div className="fc-key">
          <span className="fc-swatch observed" />
          observed path
        </div>
        <div className="fc-key">
          <span className="fc-swatch actual" />
          what actually happened
        </div>
        {draw.fan.length > 0 && (
          <div className="fc-key">
            <span className="fc-swatch fan" />
            possible futures ({draw.fan.length})
          </div>
        )}
        {draw.median.length > 0 && (
          <div className="fc-key">
            <span className="fc-swatch median" />
            forecast (next 2 s)
          </div>
        )}
      </div>
    </section>
  )
}

function Metric({ label, value, help, tip }: { label: string; value: string; help?: string; tip?: string }) {
  return (
    <div>
      <div className="fc-metric-label">{tip ? <Hint tip={tip}>{label}</Hint> : label}</div>
      <div className="fc-metric-value" title={help}>
        {value}
      </div>
      {help && <span className="fc-help">{help}</span>}
    </div>
  )
}

function FlowerBars({ data, horizon }: { data: ForecastResponse; horizon: number }) {
  const next = nextRows(data.next_flower)
  const rollout = rec(data.rollout)
  const visits = visitRows(rollout?.expected_visits)
  const n = num(rollout?.n)
  const span = num(rollout?.horizon_s) ?? horizon
  return (
    <>
      {next.length > 0 && <HBars title="Next flower probability" rows={next} asPercent />}
      {visits.length > 0 && (
        <HBars
          title={`Expected visits in the next ${trim(span)}s${n == null ? '' : ` (${Math.round(n)} rollouts)`}`}
          rows={visits}
          asPercent={false}
        />
      )}
    </>
  )
}

function HBars({
  title,
  rows,
  asPercent,
}: {
  title: string
  rows: Array<{ label: string; value: number }>
  asPercent: boolean
}) {
  const max = asPercent ? 1 : Math.max(...rows.map((row) => row.value), 1e-9)
  return (
    <div className="fc-bars">
      <div className="fc-bar-title">{title}</div>
      {rows.map((row, index) => (
        <div className="fc-bar-row" key={`${row.label}-${index}`}>
          <span className="fc-bar-label" title={row.label}>
            {row.label}
          </span>
          <span className="fc-bar-track">
            <span
              className="fc-bar-fill"
              style={{ width: `${Math.max(0, Math.min(1, row.value / max)) * 100}%` }}
            />
          </span>
          <span className="fc-bar-value">{asPercent ? formatRate(row.value) : trim(row.value)}</span>
        </div>
      ))}
    </div>
  )
}

function ReportExpander({ report }: { report: { status: 'loading' | 'ready' | 'error'; data: unknown } }) {
  return (
    <details className="fc-report">
      <summary>How good are these forecasts?</summary>
      {report.status === 'loading' && <p className="fc-caption">Loading report…</p>}
      {report.status === 'error' && <p className="fc-caption">Could not load the forecast report.</p>}
      {report.status === 'ready' && <ReportBody data={report.data} />}
    </details>
  )
}

function ReportBody({ data }: { data: unknown }) {
  const top3Model = dig(data, ['next_flower', 'model', 'top3'])
  const top3Near = dig(data, ['next_flower', 'nearest_other_flower', 'top3'])
  const top3Random = dig(data, ['next_flower', 'random', 'top3'])
  const decisions = dig(data, ['next_flower', 'model', 'decisions'])
  const moving = pathErrors(data, 'moving')
  const all = pathErrors(data, 'all')
  const fan = dig(data, ['path', 'knn_fan_minFDE_2s_moving'])
  const accuracy = dig(data, ['continue', 'accuracy'])
  const majority = dig(data, ['continue', 'majority_baseline'])
  const units = rec(data)?.units
  const stations = rec(data)?.held_out_stations
  const stationText = Array.isArray(stations) ? stations.map((item) => String(item)).join(', ') : ''

  const flower = flowerSentence(top3Model, top3Near, top3Random, decisions)
  const lines = [
    flower,
    moving && `2-second path error, moving insects (fde_2s): ${moving}.`,
    all && `2-second path error, all insects (fde_2s): ${all}.`,
    fan != null && `Closest of the 25 futures (knn_fan_minFDE_2s_moving): ${trim(fan)} units.`,
    accuracy != null && majority != null
      ? `Continue: accuracy ${formatRate(accuracy)} versus majority baseline ${formatRate(majority)}.`
      : accuracy != null
        ? `Continue accuracy: ${formatRate(accuracy)}.`
        : majority != null
          ? `Continue majority baseline: ${formatRate(majority)}.`
          : null,
  ].filter((line): line is string => Boolean(line))

  if (!lines.length) return <p className="fc-caption">The report has none of these scores.</p>

  return (
    <>
      <ul>
        {lines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {typeof units === 'string' && units && <p className="fc-caption">Units: {units}</p>}
      {stationText && <p className="fc-caption">Held-out stations: {stationText}</p>}
    </>
  )
}

function flowerSentence(
  model: number | null,
  nearest: number | null,
  random: number | null,
  decisions: number | null,
): string | null {
  if (model == null && nearest == null && random == null) return null
  const parts = [
    model != null ? `model ${formatRate(model)}` : null,
    nearest != null ? `nearest flower ${formatRate(nearest)}` : null,
    random != null ? `random ${formatRate(random)}` : null,
  ].filter((part): part is string => Boolean(part))
  const count = decisions == null ? '' : ` (${Math.round(decisions)} decisions)`
  if (parts.length === 3) {
    return `Next-flower top-3: ${parts[0]} versus ${parts[1]} and ${parts[2]}${count}.`
  }
  return `Next-flower top-3: ${parts.join(' versus ')}${count}.`
}

function pathErrors(data: unknown, slice: 'moving' | 'all'): string | null {
  const knn = dig(data, ['path', slice, 'knn', 'fde_2s'])
  const stay = dig(data, ['path', slice, 'stay', 'fde_2s'])
  const velocity = dig(data, ['path', slice, 'const_vel', 'fde_2s'])
  const parts = [
    knn != null ? `knn ${trim(knn)}` : null,
    stay != null ? `stay ${trim(stay)}` : null,
    velocity != null ? `constant velocity ${trim(velocity)}` : null,
  ].filter((part): part is string => Boolean(part))
  if (!parts.length) return null
  return parts.join(' versus ')
}

function paintForecast(ctx: CanvasRenderingContext2D, draw: DrawModel, dpr: number) {
  const { width, height } = ctx.canvas
  ctx.clearRect(0, 0, width, height)
  if (width < 2 || height < 2) return
  const sx = width / 1000
  const sy = height / 1000
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'

  ctx.lineWidth = Math.max(1, dpr)
  for (const flower of draw.flowers) {
    ctx.strokeStyle = flower.source === 'stop' ? FLOWER_STOP : FLOWER
    strokeBox(ctx, flower.box, sx, sy)
  }

  ctx.strokeStyle = OBSERVED
  ctx.lineWidth = 2 * dpr
  ctx.setLineDash([])
  strokeTimed(ctx, draw.past, sx, sy, PRESENCE_GAP_S)

  ctx.save()
  ctx.strokeStyle = ACTUAL
  ctx.lineWidth = 1.75 * dpr
  ctx.setLineDash([2.5 * dpr, 4 * dpr])
  strokeTimed(ctx, draw.future, sx, sy, PRESENCE_GAP_S)
  ctx.restore()

  const anchor = draw.past.length ? ([draw.past[draw.past.length - 1].x, draw.past[draw.past.length - 1].y] as Xy) : null

  ctx.save()
  ctx.globalAlpha = FAN_ALPHA
  ctx.strokeStyle = FORECAST
  ctx.lineWidth = 1.5 * dpr
  ctx.setLineDash([])
  for (const line of draw.fan) strokeLine(ctx, prepend(anchor, line), sx, sy)
  ctx.restore()

  const median = prepend(anchor, draw.median)
  ctx.strokeStyle = FORECAST
  ctx.lineWidth = 2.6 * dpr
  ctx.setLineDash([])
  strokeLine(ctx, median, sx, sy)
  ctx.fillStyle = FORECAST
  const radius = 3.4 * dpr
  for (const point of draw.median) {
    const at = toPx(point, sx, sy)
    if (!at) continue
    ctx.beginPath()
    ctx.arc(at[0], at[1], radius, 0, Math.PI * 2)
    ctx.fill()
  }

  if (anchor) {
    const at = toPx(anchor, sx, sy)
    if (at) {
      ctx.fillStyle = OBSERVED
      ctx.beginPath()
      ctx.arc(at[0], at[1], 3.2 * dpr, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

function contentRect(sw: number, sh: number, vw: number, vh: number) {
  if (vw <= 0 || vh <= 0 || sw <= 0 || sh <= 0) return null
  const scale = Math.min(sw / vw, sh / vh)
  const w = vw * scale
  const h = vh * scale
  return { x: (sw - w) / 2, y: (sh - h) / 2, w, h }
}

function strokeBox(ctx: CanvasRenderingContext2D, box: [number, number, number, number], sx: number, sy: number) {
  const x1 = Math.min(box[0], box[2]) * sx
  const y1 = Math.min(box[1], box[3]) * sy
  const x2 = Math.max(box[0], box[2]) * sx
  const y2 = Math.max(box[1], box[3]) * sy
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
}

function strokeTimed(
  ctx: CanvasRenderingContext2D,
  points: TimedPoint[],
  sx: number,
  sy: number,
  maxGap: number,
) {
  let open = false
  let prev = 0
  for (const point of points) {
    const x = point.x * sx
    const y = point.y * sy
    if (!open || point.t - prev > maxGap) {
      if (open) ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(x, y)
      open = true
    } else {
      ctx.lineTo(x, y)
    }
    prev = point.t
  }
  if (open) ctx.stroke()
}

function strokeLine(ctx: CanvasRenderingContext2D, points: Xy[], sx: number, sy: number) {
  let open = false
  for (const point of points) {
    const at = toPx(point, sx, sy)
    if (!at) {
      if (open) ctx.stroke()
      open = false
      continue
    }
    if (!open) {
      ctx.beginPath()
      ctx.moveTo(at[0], at[1])
      open = true
    } else {
      ctx.lineTo(at[0], at[1])
    }
  }
  if (open) ctx.stroke()
}

function toPx(point: Xy, sx: number, sy: number): Xy | null {
  if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) return null
  return [point[0] * sx, point[1] * sy]
}

function prepend(anchor: Xy | null, line: Xy[]): Xy[] {
  if (!anchor || line.length === 0) return line
  const [x, y] = line[0]
  if (Math.hypot(x - anchor[0], y - anchor[1]) < 0.8) return line
  return [anchor, ...line]
}

function listTracks(tracks: ClipTrack[] | undefined): ClipTrack[] {
  return (tracks ?? []).filter((track) => typeof track.track_id === 'string' && track.track_id.length > 0)
}

function timeRange(track: ClipTrack | undefined): { lo: number; hi: number } | null {
  const points = track ? timedPoints(track) : []
  if (!points.length) return null
  let lo = points[0].t
  let hi = points[0].t
  for (const point of points) {
    if (point.t < lo) lo = point.t
    if (point.t > hi) hi = point.t
  }
  return { lo, hi }
}

function timedPoints(track: ClipTrack): TimedPoint[] {
  const out: TimedPoint[] = []
  for (const point of track.points ?? []) {
    if (point.interpolated === true || !Number.isFinite(point.t)) continue
    const xy = centerOf(point)
    if (!xy) continue
    out.push({ t: point.t, x: xy[0], y: xy[1] })
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

function centerOf(point: TrackPoint): Xy | null {
  const center = point.center
  if (center && center.length >= 2 && Number.isFinite(center[0]) && Number.isFinite(center[1])) {
    return [center[0], center[1]]
  }
  const box = point.box
  if (!box || box.length < 4 || !box.every((value) => Number.isFinite(value))) return null
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
}

function flowerBoxes(flowers: Flower[] | undefined): DrawnFlower[] {
  const out: DrawnFlower[] = []
  for (const flower of flowers ?? []) {
    const box = flower.box
    if (!box || box.length < 4 || !box.every((value) => Number.isFinite(value))) continue
    out.push({ box: [box[0], box[1], box[2], box[3]], source: flower.source })
  }
  return out
}

function lineOf(value: unknown): Xy[] {
  if (!Array.isArray(value)) return []
  const out: Xy[] = []
  for (const point of value) {
    if (!Array.isArray(point) || point.length < 2) continue
    const x = point[0]
    const y = point[1]
    if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) continue
    out.push([x, y])
  }
  return out
}

function fanOf(value: unknown): Xy[][] {
  if (!Array.isArray(value)) return []
  return value.map((line) => lineOf(line)).filter((line) => line.length >= 2)
}

function isForecast(value: unknown): value is ForecastResponse {
  const row = rec(value)
  if (!row) return false
  return typeof row.track_id === 'string' || typeof row.t === 'number'
}

function rec(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function dig(root: unknown, path: string[]): number | null {
  let cur: unknown = root
  for (const key of path) {
    const row = rec(cur)
    if (!row || !(key in row)) return null
    cur = row[key]
  }
  return num(cur)
}

function dwellHelp(value: unknown): string | undefined {
  const row = rec(value)
  if (!row) return undefined
  const q50 = num(row.q50)
  const q10 = num(row.q10)
  const q90 = num(row.q90)
  if (q50 == null) return undefined
  if (q10 != null && q90 != null) {
    return `Typical dwell for this species: median ${trim(q50)} s (10–90%: ${trim(q10)}–${trim(q90)} s).`
  }
  return `Typical dwell for this species: median ${trim(q50)} s.`
}

function dwellText(data: ForecastResponse): string {
  const remaining = num(data.dwell_remaining_s)
  return remaining == null ? 'not on a flower' : `${trim(remaining)} s`
}

function leavesText(data: ForecastResponse): string {
  return formatRate(num(rec(data.rollout)?.p_leaves_patch))
}

function sequenceText(data: ForecastResponse): string {
  const rollout = rec(data.rollout)
  if (!rollout) return '—'
  const seq = rollout.most_likely_sequence
  if (!Array.isArray(seq)) return 'leaves the patch'
  const names = seq.filter((item): item is string => typeof item === 'string' && item.length > 0)
  if (!names.length) return 'leaves the patch'
  const shown = names.slice(0, 4).join(' → ')
  return names.length > 4 ? `${shown}…` : shown
}

function nextRows(value: unknown): Array<{ label: string; value: number }> {
  if (!Array.isArray(value)) return []
  const rows: Array<{ label: string; value: number }> = []
  for (const item of value) {
    const row = rec(item)
    const p = num(row?.p)
    const label = row?.flower_id
    if (!row || p == null || typeof label !== 'string' || !label) continue
    rows.push({ label, value: p })
  }
  rows.sort((a, b) => b.value - a.value)
  return rows
}

function visitRows(value: unknown): Array<{ label: string; value: number }> {
  const row = rec(value)
  if (!row) return []
  const rows: Array<{ label: string; value: number }> = []
  for (const [label, raw] of Object.entries(row)) {
    const visit = num(raw)
    if (visit == null || visit <= 0) continue
    rows.push({ label, value: visit })
  }
  rows.sort((a, b) => b.value - a.value)
  return rows.slice(0, 8)
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value))
}

function finite(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000
}

function trim(value: number): string {
  return String(Math.round(value * 1000) / 1000)
}

function formatRate(value: number | null): string {
  if (value == null) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function formatSeconds(value: number): string {
  return `${value.toFixed(1)}s`
}

const CSS = `
.stage-panel > .region-body.fc-shell {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  text-align: left;
  container-type: size;
  container-name: forecast;
}
.stage.field-mode .fc-frame video,
.stage.field-mode .fc-frame canvas {
  visibility: visible;
}
.fc-frame {
  position: relative;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 120px;
  background: #1d1d1d;
}
.fc-frame video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #111;
}
.fc-frame canvas {
  position: absolute;
  pointer-events: none;
}
.fc-side {
  flex: 0 1 auto;
  width: auto;
  min-width: 0;
  max-width: none;
  max-height: 52%;
  overflow: auto;
  padding: 8px 8px 12px;
  background: #2c2c2c;
  border-top: 1px solid #191919;
}
@container forecast (min-width: 560px) {
  .stage-panel > .region-body.fc-shell {
    flex-direction: row;
  }
  .fc-frame {
    min-height: 0;
  }
  .fc-side {
    flex: 0 0 280px;
    width: 280px;
    max-height: none;
    border-top: 0;
    border-left: 1px solid #191919;
  }
}
.fc-field {
  display: block;
  margin: 0 0 10px;
}
.fc-field-label {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: #bdbdbd;
}
.fc-field-value,
.fc-metric-value,
.fc-bar-label,
.fc-bar-value {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
.fc-side select {
  width: 100%;
  height: 22px;
  margin-top: 2px;
  padding: 0 4px;
  border: 1px solid #191919;
  border-radius: 2px;
  background: #1d1d1d;
  color: inherit;
  font: inherit;
}
.fc-side input[type='range'] {
  width: 100%;
  margin: 4px 0 0;
  accent-color: #d0d0d0;
}
.fc-caption,
.fc-status,
.fc-help,
.fc-note,
.fc-metric-label,
.fc-bar-title {
  color: #8d8d8d;
}
.fc-caption,
.fc-status,
.fc-unavailable {
  margin: 0 0 10px;
  line-height: 1.35;
}
.fc-unavailable {
  padding: 8px;
  border: 1px solid #8a4b32;
  background: #3a261f;
  color: #f3d2c4;
}
.fc-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 12px;
  margin: 0 0 12px;
}
.fc-metric-value {
  margin-top: 2px;
  font-size: 13px;
  overflow-wrap: anywhere;
}
.fc-help,
.fc-note {
  display: block;
  margin-top: 2px;
  font-size: 11px;
  line-height: 1.3;
}
.fc-bars {
  margin: 0 0 12px;
}
.fc-bar-title {
  margin-bottom: 4px;
}
.fc-bar-row {
  display: grid;
  grid-template-columns: 76px minmax(0, 1fr) 52px;
  gap: 6px;
  align-items: center;
  margin: 3px 0;
}
.fc-bar-label,
.fc-bar-value {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fc-bar-value {
  text-align: right;
}
.fc-bar-track {
  height: 8px;
  background: #1a1a1a;
  overflow: hidden;
}
.fc-bar-fill {
  display: block;
  height: 100%;
  background: #2a78d6;
}
.fc-title,
.fc-legend {
  position: absolute;
  z-index: 2;
  max-width: calc(100% - 16px);
  padding: 6px 8px;
  background: rgba(20, 20, 20, 0.82);
  pointer-events: none;
  line-height: 1.45;
}
.fc-title {
  top: 8px;
  left: 8px;
  max-width: calc(100% - 16px);
}
.fc-legend {
  left: 8px;
  right: 8px;
  bottom: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 2px 10px;
}
.fc-key {
  display: flex;
  align-items: center;
  gap: 6px;
}
.fc-swatch {
  width: 16px;
  flex: 0 0 16px;
  border-top: 2px solid transparent;
}
.fc-swatch.observed { border-top-color: #2a78d6; }
.fc-swatch.actual { border-top-style: dotted; border-top-color: #8a8985; }
.fc-swatch.fan { border-top-color: rgba(235, 104, 52, 0.75); }
.fc-swatch.median { border-top-width: 3px; border-top-color: #eb6834; }
.fc-report {
  margin-top: 4px;
  padding-top: 8px;
  border-top: 1px solid #191919;
}
.fc-report summary {
  cursor: pointer;
}
.fc-report ul {
  margin: 8px 0 0;
  padding-left: 16px;
}
.fc-report li {
  margin: 5px 0;
  line-height: 1.35;
}
`
