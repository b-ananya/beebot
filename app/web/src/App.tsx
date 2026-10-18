import { useEffect, useRef, useState } from 'react'
import { AnalyticsPanel } from './analytics'
import { DatasetPanel } from './dataset'
import { FieldDiagram, type FieldFile, type PlanFile } from './field'
import { ForecastPanel } from './forecast'
import { drawOverlay, GAP_S, usablePoints, type Box, type Flower, type TrackPoint } from './overlay'
import { type Moment } from './search'
import { MomentSearch } from './search-ui'
import { trackUrl, type ClipTrack, type ManifestEntry, type PanelProps, type Source, type TrackDoc, type VisitEvent } from './types'
import { UploadButton } from './upload'

const DEFAULT_CLIP = 'hb6_w00'
const TOP_H = 28
/** Usable points farther apart than this are separate times the insect is in frame. */
const VISIBLE_GAP_S = 0.6

type Mode = 'video' | 'analytics' | 'forecast' | 'field' | 'dataset'
type ClipSet = 'honeybee' | 'farm'
type DocStatus = 'loading' | 'missing' | 'ready'

const SOURCES: Array<{ id: Source; label: string }> = [
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'heuristic', label: 'Heuristic' },
  { id: 'truth', label: 'Truth' },
  { id: 'mock', label: 'Mock' },
]

const CLIP_SETS: Array<{ id: ClipSet; label: string }> = [
  { id: 'honeybee', label: 'Honeybee' },
  { id: 'farm', label: 'Farm' },
]

const MODES: Array<{ id: Mode; label: string }> = [
  { id: 'video', label: 'Video' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'forecast', label: 'Forecast' },
  { id: 'field', label: 'Field' },
  { id: 'dataset', label: 'Dataset' },
]

function clipPrefix(set: ClipSet): 'hb' | 'sm' {
  return set === 'farm' ? 'sm' : 'hb'
}

function clipSetFor(id: string): ClipSet {
  return id.startsWith('sm') ? 'farm' : 'honeybee'
}

function parseManifest(body: unknown): ManifestEntry[] {
  if (!Array.isArray(body)) return []
  const entries: ManifestEntry[] = []
  for (const row of body) {
    if (!row || typeof row !== 'object') continue
    const item = row as Partial<ManifestEntry>
    if (typeof item.clip_id !== 'string' || item.clip_id.length === 0) continue
    if (typeof item.path !== 'string') continue
    if (typeof item.fps !== 'number' || typeof item.width !== 'number' || typeof item.height !== 'number') continue
    if (typeof item.duration_s !== 'number' || typeof item.source !== 'string') continue
    if (typeof item.has_truth !== 'boolean' || typeof item.truth_kind !== 'string') continue
    entries.push({
      clip_id: item.clip_id,
      path: item.path,
      fps: item.fps,
      width: item.width,
      height: item.height,
      duration_s: item.duration_s,
      source: item.source,
      has_truth: item.has_truth,
      truth_kind: item.truth_kind,
    })
  }
  return entries
}

const SPLIT = 5
const MIN_SIDE = 120
const MIN_CENTER = 220
const MIN_BOTTOM = 48
const MIN_VIEW = 96

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max))
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0
  const total = Math.round(seconds * 100)
  const hundredths = total % 100
  const wholeSeconds = Math.floor(total / 100)
  const minutes = Math.floor(wholeSeconds / 60)
  const whole = wholeSeconds % 60
  return `${minutes}:${whole.toString().padStart(2, '0')}.${hundredths.toString().padStart(2, '0')}`
}

function beginDrag(
  event: React.PointerEvent<HTMLElement>,
  axis: 'x' | 'y',
  onMove: (client: number) => void,
): void {
  event.preventDefault()
  const handle = event.currentTarget
  handle.setPointerCapture(event.pointerId)
  handle.classList.add('active')

  const move = (e: PointerEvent) => {
    onMove(axis === 'x' ? e.clientX : e.clientY)
  }
  const end = (e: PointerEvent) => {
    handle.classList.remove('active')
    handle.removeEventListener('pointermove', move)
    handle.removeEventListener('pointerup', end)
    handle.removeEventListener('pointercancel', end)
    if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId)
  }

  handle.addEventListener('pointermove', move)
  handle.addEventListener('pointerup', end)
  handle.addEventListener('pointercancel', end)
}

function PlayIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
      <path d="M2 1.2 L8.5 5 L2 8.8 Z" fill="currentColor" />
    </svg>
  )
}

function PauseIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
      <rect x="1.6" y="1.2" width="2.2" height="7.6" fill="currentColor" />
      <rect x="6.2" y="1.2" width="2.2" height="7.6" fill="currentColor" />
    </svg>
  )
}

/** Same swatches the overlay uses, so a row matches the box it selects. */
const INSECT_SWATCH = ['#2a78d6', '#eb6834', '#1baf7a']

type RowSelection = { kind: 'track' | 'flower'; id: string }

function trackKey(track: ClipTrack, index: number): string {
  return track.track_id ? `track:${track.track_id}` : `track:#${index}`
}

function flowerKey(flower: Flower, index: number): string {
  return flower.flower_id ? `flower:${flower.flower_id}` : `flower:#${index}`
}

function flowerSwatch(flower: Flower): string {
  return flower.source === 'stop' ? '#aaaaaa' : '#ffd200'
}

function shown(value: string | undefined): string {
  return value && value.length > 0 ? value : '—'
}

function formatBox(box: Box | undefined): string {
  if (!box || box.length !== 4 || box.some((n) => !Number.isFinite(n))) return '—'
  const nums = box.map((n) => {
    const rounded = Math.round(n * 10) / 10
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)
  })
  return `[${nums.join(', ')}]`
}

function endpointTimes(track: ClipTrack): { first: number; last: number } | null {
  const points = usablePoints(track.points)
  if (points.length === 0) return null
  return { first: points[0].t, last: points[points.length - 1].t }
}

function visitsFor(doc: TrackDoc, flowerId: string | undefined): Array<VisitEvent & { t_start: number; t_end: number }> {
  if (!flowerId) return []
  const visits: Array<VisitEvent & { t_start: number; t_end: number }> = []
  for (const event of doc.events ?? []) {
    if (event.type !== 'visit' || event.flower_id !== flowerId) continue
    if (!Number.isFinite(event.t_start) || !Number.isFinite(event.t_end)) continue
    visits.push({ ...event, t_start: event.t_start as number, t_end: event.t_end as number })
  }
  return visits
}

function isBox(box: Box | undefined): box is Box {
  return !!box && box.length === 4 && box.every((n) => Number.isFinite(n))
}

function boxAtTime(points: TrackPoint[], time: number): Box | null {
  if (!Number.isFinite(time) || points.length === 0) return null
  let lo = 0
  let hi = points.length - 1
  let ans = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (points[mid].t <= time) {
      ans = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  if (ans < 0) return null
  const cur = points[ans]
  if (time - cur.t > GAP_S) return null
  return isBox(cur.box) ? cur.box : null
}

function selectionBox(doc: TrackDoc | null, selection: RowSelection | null, time: number): Box | null {
  if (!doc || !selection) return null
  if (selection.kind === 'flower') {
    const flower = (doc.flowers ?? []).find((item, index) => flowerKey(item, index) === selection.id)
    return flower && isBox(flower.box) ? flower.box : null
  }
  const track = (doc.tracks ?? []).find((item, index) => trackKey(item, index) === selection.id)
  if (!track) return null
  return boxAtTime(usablePoints(track.points), time)
}

function strokeHighlight(ctx: CanvasRenderingContext2D, box: Box): void {
  const { width, height } = ctx.canvas
  if (width < 2 || height < 2) return
  const sx = width / 1000
  const sy = height / 1000
  const x1 = Math.min(box[0], box[2]) * sx
  const y1 = Math.min(box[1], box[3]) * sy
  const x2 = Math.max(box[0], box[2]) * sx
  const y2 = Math.max(box[1], box[3]) * sy
  const pad = Math.max(2, width / 500)
  ctx.save()
  ctx.globalAlpha = 1
  ctx.lineJoin = 'miter'
  ctx.strokeStyle = '#5ee0ff'
  ctx.lineWidth = Math.max(2.5, width / 280)
  ctx.strokeRect(x1 - pad, y1 - pad, x2 - x1 + pad * 2, y2 - y1 + pad * 2)
  ctx.restore()
}

function Outliner({
  doc,
  selection,
  onSelect,
}: {
  doc: TrackDoc | null
  selection: RowSelection | null
  onSelect: (row: RowSelection) => void
}) {
  const tracks = doc?.tracks ?? []
  const flowers = doc?.flowers ?? []
  return (
    <div className="tree" role="listbox" aria-label="Tracks and flowers">
      <div className="tree-group">Tracks</div>
      {tracks.map((track, index) => {
        const id = trackKey(track, index)
        const on = selection?.kind === 'track' && selection.id === id
        return (
          <button
            key={id}
            type="button"
            role="option"
            className={on ? 'tree-row on' : 'tree-row'}
            aria-selected={on}
            onClick={() => onSelect({ kind: 'track', id })}
          >
            <span className="swatch" style={{ background: INSECT_SWATCH[index % INSECT_SWATCH.length] }} />
            <span className="label">{shown(track.track_id)}</span>
          </button>
        )
      })}
      <div className="tree-group">Flowers</div>
      {flowers.map((flower, index) => {
        const id = flowerKey(flower, index)
        const on = selection?.kind === 'flower' && selection.id === id
        return (
          <button
            key={id}
            type="button"
            role="option"
            className={on ? 'tree-row on' : 'tree-row'}
            aria-selected={on}
            onClick={() => onSelect({ kind: 'flower', id })}
          >
            <span className="swatch" style={{ background: flowerSwatch(flower) }} />
            <span className="label">{shown(flower.flower_id)}</span>
          </button>
        )
      })}
    </div>
  )
}

type VisitSpan = { t_start: number; t_end: number; flower_id?: string; track_id?: string }

function visibleSegments(track: ClipTrack): Array<{ t_start: number; t_end: number }> {
  const times = usablePoints(track.points)
    .map((point) => point.t)
    .sort((a, b) => a - b)
  if (times.length === 0) return []
  const segments: Array<{ t_start: number; t_end: number }> = []
  let start = times[0]
  let end = times[0]
  for (let i = 1; i < times.length; i++) {
    const t = times[i]
    if (t - end > VISIBLE_GAP_S) {
      segments.push({ t_start: start, t_end: end })
      start = t
    }
    end = t
  }
  segments.push({ t_start: start, t_end: end })
  return segments
}

function flowersWithVisits(doc: TrackDoc | null): Array<{ id: string; swatch: string; visits: VisitSpan[] }> {
  if (!doc) return []
  const grouped = new Map<string, VisitSpan[]>()
  for (const event of doc.events ?? []) {
    if (event.type !== 'visit' || !event.flower_id) continue
    if (typeof event.t_start !== 'number' || typeof event.t_end !== 'number') continue
    if (!Number.isFinite(event.t_start) || !Number.isFinite(event.t_end)) continue
    const span: VisitSpan = {
      t_start: event.t_start,
      t_end: event.t_end,
      flower_id: event.flower_id,
      track_id: event.track_id,
    }
    const visits = grouped.get(event.flower_id)
    if (visits) visits.push(span)
    else grouped.set(event.flower_id, [span])
  }
  const ordered: string[] = []
  for (const flower of doc.flowers ?? []) {
    if (!flower.flower_id || !grouped.has(flower.flower_id) || ordered.includes(flower.flower_id)) continue
    ordered.push(flower.flower_id)
  }
  for (const id of grouped.keys()) {
    if (!ordered.includes(id)) ordered.push(id)
  }
  return ordered.map((id) => {
    const flower = (doc.flowers ?? []).find((item) => item.flower_id === id)
    return { id, swatch: flower ? flowerSwatch(flower) : '#ffd200', visits: grouped.get(id) ?? [] }
  })
}

function trackVisits(doc: TrackDoc | null, trackId: string | undefined): VisitSpan[] {
  if (!doc || !trackId) return []
  const spans: VisitSpan[] = []
  for (const event of doc.events ?? []) {
    if (event.type !== 'visit' || event.track_id !== trackId) continue
    if (typeof event.t_start !== 'number' || typeof event.t_end !== 'number') continue
    if (!Number.isFinite(event.t_start) || !Number.isFinite(event.t_end)) continue
    spans.push({
      t_start: event.t_start,
      t_end: event.t_end,
      flower_id: event.flower_id,
      track_id: event.track_id,
    })
  }
  return spans
}

/** Stack overlapping visits so each event keeps its own t_start–t_end. */
function visitRows(spans: VisitSpan[]): number[] {
  const order = spans
    .map((span, index) => ({ span, index }))
    .sort((a, b) => a.span.t_start - b.span.t_start || a.index - b.index)
  const rowEnds: number[] = []
  const rows = new Array<number>(spans.length).fill(0)
  for (const item of order) {
    let row = rowEnds.findIndex((end) => item.span.t_start >= end)
    if (row < 0) {
      row = rowEnds.length
      rowEnds.push(item.span.t_end)
    } else {
      rowEnds[row] = item.span.t_end
    }
    rows[item.index] = row
  }
  return rows
}

const LANE_ROW = 14

function laneHeight(rowCount: number): number {
  return Math.max(18, rowCount * LANE_ROW + 4)
}

function tickStep(duration: number): number {
  if (duration <= 20) return 1
  if (duration <= 60) return 5
  if (duration <= 180) return 15
  return 30
}

type TimelineLane = {
  key: string
  label: string
  swatch: string
  bar: string
  kind: 'track' | 'flower'
  visits: VisitSpan[]
  segments: Array<{ t_start: number; t_end: number }>
}

function timelineLanes(doc: TrackDoc | null): TimelineLane[] {
  const lanes: TimelineLane[] = []
  ;(doc?.tracks ?? []).forEach((track, index) => {
    const color = INSECT_SWATCH[index % INSECT_SWATCH.length]
    lanes.push({
      key: trackKey(track, index),
      label: shown(track.track_id),
      swatch: color,
      bar: color,
      kind: 'track',
      visits: trackVisits(doc, track.track_id),
      segments: visibleSegments(track),
    })
  })
  for (const flower of flowersWithVisits(doc)) {
    lanes.push({
      key: `flower:${flower.id}`,
      label: flower.id,
      swatch: flower.swatch,
      bar: flower.swatch,
      kind: 'flower',
      visits: flower.visits,
      segments: [],
    })
  }
  return lanes
}

function Timeline({
  doc,
  time,
  duration,
  onSeek,
}: {
  doc: TrackDoc | null
  time: number
  duration: number
  onSeek: (seconds: number) => void
}) {
  const lanes = timelineLanes(doc)
  const span = duration > 0 ? duration : 0
  const playhead = span > 0 ? clamp(time, 0, span) / span : 0
  const ticks: number[] = []
  if (span > 0) {
    const step = tickStep(span)
    for (let t = 0; t <= span + 1e-6; t += step) ticks.push(Math.min(t, span))
  }

  const scrub = (event: React.PointerEvent<HTMLDivElement>) => {
    if (span <= 0) return
    const sheet = event.currentTarget
    const seekAt = (clientX: number) => {
      const rect = sheet.getBoundingClientRect()
      if (rect.width <= 0) return
      const ratio = clamp((clientX - rect.left) / rect.width, 0, 1)
      onSeek(ratio * span)
    }
    seekAt(event.clientX)
    event.preventDefault()
    sheet.setPointerCapture(event.pointerId)
    const move = (e: PointerEvent) => seekAt(e.clientX)
    const end = (e: PointerEvent) => {
      sheet.removeEventListener('pointermove', move)
      sheet.removeEventListener('pointerup', end)
      sheet.removeEventListener('pointercancel', end)
      if (sheet.hasPointerCapture(e.pointerId)) sheet.releasePointerCapture(e.pointerId)
    }
    sheet.addEventListener('pointermove', move)
    sheet.addEventListener('pointerup', end)
    sheet.addEventListener('pointercancel', end)
  }

  return (
    <div className="tl">
      <div className="tl-labels">
        <div className="tl-ruler-pad" />
        {lanes.map((lane) => {
          const rows = visitRows(lane.visits)
          const count = rows.length > 0 ? Math.max(...rows) + 1 : 1
          return (
            <div key={lane.key} className="tl-label" style={{ height: laneHeight(count) }}>
              <span className="swatch" style={{ background: lane.swatch }} />
              <span className="name">{lane.label}</span>
            </div>
          )
        })}
      </div>
      <div
        className="tl-sheet"
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={span}
        aria-valuenow={span > 0 ? clamp(time, 0, span) : 0}
        aria-valuetext={formatTime(time)}
        tabIndex={0}
        onPointerDown={scrub}
        onKeyDown={(event) => {
          if (span <= 0) return
          if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
          event.preventDefault()
          const step = event.shiftKey ? 1 : 0.1
          const delta = event.key === 'ArrowLeft' ? -step : step
          onSeek(clamp(time + delta, 0, span))
        }}
      >
        <div className="tl-ruler">
          {ticks.map((t) => (
            <span key={t} className="tl-tick" style={{ left: `${(t / span) * 100}%` }} />
          ))}
          <span className="tl-cap start">{formatTime(0)}</span>
          {span > 0 && <span className="tl-cap end">{formatTime(span)}</span>}
        </div>
        {lanes.map((lane) => {
          const rows = visitRows(lane.visits)
          const count = rows.length > 0 ? Math.max(...rows) + 1 : 1
          return (
            <div key={lane.key} className="tl-lane" data-lane={lane.key} style={{ height: laneHeight(count) }}>
              {span > 0 &&
                lane.segments.map((segment) => (
                  <span
                    key={`vis-${segment.t_start}-${segment.t_end}`}
                    className="tl-vis"
                    data-t-start={segment.t_start}
                    data-t-end={segment.t_end}
                    style={{
                      left: `${(segment.t_start / span) * 100}%`,
                      width: `${((segment.t_end - segment.t_start) / span) * 100}%`,
                      top: 2,
                      bottom: 2,
                      background: lane.swatch,
                    }}
                  />
                ))}
              {span > 0 &&
                lane.visits.map((visit, visitIndex) => (
                  <span
                    key={`${visit.t_start}-${visit.t_end}-${visitIndex}`}
                    className="tl-bar"
                    data-t-start={visit.t_start}
                    data-t-end={visit.t_end}
                    title={
                      lane.kind === 'flower'
                        ? `${shown(visit.track_id)} ${formatTime(visit.t_start)}–${formatTime(visit.t_end)}`
                        : `${shown(visit.flower_id)} ${formatTime(visit.t_start)}–${formatTime(visit.t_end)}`
                    }
                    style={{
                      left: `${(visit.t_start / span) * 100}%`,
                      width: `${((visit.t_end - visit.t_start) / span) * 100}%`,
                      top: 2 + rows[visitIndex] * LANE_ROW,
                      height: LANE_ROW - 4,
                      background: lane.bar,
                    }}
                  />
                ))}
            </div>
          )
        })}
        {span > 0 && <div className="tl-playhead" style={{ left: `${playhead * 100}%` }} />}
      </div>
    </div>
  )
}

function formatDwell(seconds: number): string {
  return `${(Math.round(seconds * 1000) / 1000).toFixed(3)} s`
}

function clipVisits(doc: TrackDoc | null): Array<{ flowerId: string; dwells: number[] }> {
  const order: string[] = []
  const grouped = new Map<string, number[]>()
  for (const event of doc?.events ?? []) {
    if (event.type !== 'visit' || !event.flower_id) continue
    if (typeof event.t_start !== 'number' || typeof event.t_end !== 'number') continue
    if (!Number.isFinite(event.t_start) || !Number.isFinite(event.t_end)) continue
    let dwells = grouped.get(event.flower_id)
    if (!dwells) {
      dwells = []
      grouped.set(event.flower_id, dwells)
      order.push(event.flower_id)
    }
    dwells.push(event.t_end - event.t_start)
  }
  return order.map((flowerId) => ({ flowerId, dwells: grouped.get(flowerId) ?? [] }))
}

function ClipVisits({ doc }: { doc: TrackDoc | null }) {
  const rows = clipVisits(doc)
  return (
    <div className="props">
      <div className="prop-head">Visits</div>
      {rows.length === 0 && <div className="empty-note">No visit events</div>}
      {rows.map((row) => (
        <div key={row.flowerId}>
          <div className="prop">
            <span className="prop-key">flower_id</span>
            <span className="prop-value">{row.flowerId}</span>
          </div>
          <div className="prop">
            <span className="prop-key">count</span>
            <span className="prop-value">{row.dwells.length}</span>
          </div>
          {row.dwells.map((dwell, index) => (
            <div key={index} className="prop">
              <span className="prop-key">dwell</span>
              <span className="prop-value">{formatDwell(dwell)}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

function Properties({
  doc,
  selection,
  onSeek,
}: {
  doc: TrackDoc | null
  selection: RowSelection | null
  onSeek: (seconds: number) => void
}) {
  if (!doc || !selection) return <div className="empty-note">Select a track or flower</div>

  if (selection.kind === 'track') {
    const track = (doc.tracks ?? []).find((item, index) => trackKey(item, index) === selection.id)
    if (!track) return <div className="empty-note">Select a track or flower</div>
    const span = endpointTimes(track)
    return (
      <div className="props">
        <div className="prop">
          <span className="prop-key">track_id</span>
          <span className="prop-value">{shown(track.track_id)}</span>
        </div>
        <div className="prop">
          <span className="prop-key">species</span>
          <span className="prop-value">{shown(track.species)}</span>
        </div>
        <div className="prop">
          <span className="prop-key">first</span>
          <span className="prop-value">{span ? formatTime(span.first) : '—'}</span>
        </div>
        <div className="prop">
          <span className="prop-key">last</span>
          <span className="prop-value">{span ? formatTime(span.last) : '—'}</span>
        </div>
      </div>
    )
  }

  const flower = (doc.flowers ?? []).find((item, index) => flowerKey(item, index) === selection.id)
  if (!flower) return <div className="empty-note">Select a track or flower</div>
  const visits = visitsFor(doc, flower.flower_id)
  return (
    <div className="props">
      <div className="prop">
        <span className="prop-key">flower_id</span>
        <span className="prop-value">{shown(flower.flower_id)}</span>
      </div>
      <div className="prop">
        <span className="prop-key">box</span>
        <span className="prop-value">{formatBox(flower.box)}</span>
      </div>
      {visits.length > 0 && (
        <>
          <div className="prop-head">Visits</div>
          {visits.map((visit) => (
            <button
              key={`${visit.t_start}-${visit.t_end}`}
              type="button"
              className="visit-row"
              onClick={() => onSeek(visit.t_start)}
            >
              {formatTime(visit.t_start)}–{formatTime(visit.t_end)}
            </button>
          ))}
        </>
      )}
    </div>
  )
}

export function App() {
  const [leftW, setLeftW] = useState(260)
  const [rightW, setRightW] = useState(280)
  const [bottomH, setBottomH] = useState(140)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [manifest, setManifest] = useState<ManifestEntry[]>([])
  const [clipSet, setClipSet] = useState<ClipSet>('honeybee')
  const [source, setSource] = useState<Source>('pipeline')
  const [mode, setMode] = useState<Mode>('video')
  const [clipId, setClipId] = useState(DEFAULT_CLIP)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showTrail, setShowTrail] = useState(true)
  const [doc, setDoc] = useState<TrackDoc | null>(null)
  const [docStatus, setDocStatus] = useState<DocStatus>('loading')
  const [selection, setSelection] = useState<RowSelection | null>(null)
  const [moments, setMoments] = useState<Moment[] | null>(null)
  const [fieldDoc, setFieldDoc] = useState<FieldFile | null>(null)
  const [planDoc, setPlanDoc] = useState<PlanFile | null>(null)
  const [fieldStatus, setFieldStatus] = useState<'loading' | 'missing' | 'ready'>('loading')

  const shellRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<TrackDoc | null>(null)
  const selectionRef = useRef<RowSelection | null>(null)
  const pendingSeek = useRef<number | null>(null)
  const showBoxesRef = useRef(showBoxes)
  const showTrailRef = useRef(showTrail)
  const paintRef = useRef<() => void>(() => {})
  showBoxesRef.current = showBoxes
  showTrailRef.current = showTrail
  selectionRef.current = selection

  paintRef.current = () => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!video || video.videoWidth === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    drawOverlay(ctx, trackRef.current, video.currentTime, {
      boxes: showBoxesRef.current,
      trail: showTrailRef.current,
    })
    const box = selectionBox(trackRef.current, selectionRef.current, video.currentTime)
    if (box) strokeHighlight(ctx, box)
  }
  const leftRef = useRef(leftW)
  const rightRef = useRef(rightW)
  const bottomRef = useRef(bottomH)
  leftRef.current = leftW
  rightRef.current = rightW
  bottomRef.current = bottomH

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let frame = 0
    const stopTick = () => cancelAnimationFrame(frame)
    const tick = () => {
      setTime(video.currentTime)
      paintRef.current()
      frame = requestAnimationFrame(tick)
    }
    const onPlay = () => {
      setPlaying(true)
      stopTick()
      frame = requestAnimationFrame(tick)
    }
    const onPause = () => {
      setPlaying(false)
      stopTick()
      setTime(video.currentTime)
      paintRef.current()
    }
    const onMeta = () => {
      const length = Number.isFinite(video.duration) ? video.duration : 0
      setDuration(length)
      if (pendingSeek.current != null && length > 0) {
        const next = clamp(pendingSeek.current, 0, length)
        if (Math.abs(video.currentTime - next) > 0.04) {
          video.currentTime = next
          setTime(next)
          paintRef.current()
          return
        }
        pendingSeek.current = null
      }
      setTime(video.currentTime)
      paintRef.current()
    }

    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('ended', onPause)
    video.addEventListener('loadedmetadata', onMeta)
    video.addEventListener('durationchange', onMeta)
    video.addEventListener('timeupdate', onMeta)
    video.addEventListener('seeked', onMeta)
    return () => {
      stopTick()
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('ended', onPause)
      video.removeEventListener('loadedmetadata', onMeta)
      video.removeEventListener('durationchange', onMeta)
      video.removeEventListener('timeupdate', onMeta)
      video.removeEventListener('seeked', onMeta)
    }
  }, [])

  useEffect(() => {
    const stage = stageRef.current
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!stage || !video || !canvas) return

    const sync = () => {
      const sw = stage.clientWidth
      const sh = stage.clientHeight
      const vw = video.videoWidth
      const vh = video.videoHeight
      let x = 0
      let y = 0
      let w = sw
      let h = sh
      if (vw > 0 && vh > 0 && sw > 0 && sh > 0) {
        const scale = Math.min(sw / vw, sh / vh)
        w = vw * scale
        h = vh * scale
        x = (sw - w) / 2
        y = (sh - h) / 2
      }
      const dpr = window.devicePixelRatio || 1
      canvas.style.left = `${x}px`
      canvas.style.top = `${y}px`
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const bw = Math.max(1, Math.round(w * dpr))
      const bh = Math.max(1, Math.round(h * dpr))
      if (canvas.width !== bw) canvas.width = bw
      if (canvas.height !== bh) canvas.height = bh
      paintRef.current()
    }

    const observer = new ResizeObserver(sync)
    observer.observe(stage)
    video.addEventListener('loadedmetadata', sync)
    sync()
    return () => {
      observer.disconnect()
      video.removeEventListener('loadedmetadata', sync)
    }
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      const target = event.target
      if (target instanceof HTMLElement && (target.tagName === 'BUTTON' || target.tagName === 'INPUT' || target.tagName === 'SELECT')) return
      event.preventDefault()
      const video = videoRef.current
      if (!video) return
      if (video.paused) void video.play()
      else video.pause()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (mode !== 'field') return
    const ac = new AbortController()
    setFieldDoc(null)
    setPlanDoc(null)
    setFieldStatus('loading')
    const readJson = (res: Response) => {
      if (!res.ok) return null
      const type = res.headers.get('content-type') ?? ''
      if (!type.includes('application/json')) return null
      return res.json()
    }
    Promise.all([
      fetch(`/field/${clipId}.json`, { signal: ac.signal }).then(readJson),
      fetch(`/plan/${clipId}.json`, { signal: ac.signal }).then(readJson),
    ])
      .then(([field, plan]: [FieldFile | null, PlanFile | null]) => {
        if (ac.signal.aborted) return
        if (!field || !Array.isArray(field.flowers)) {
          setFieldStatus('missing')
          return
        }
        setFieldDoc(field)
        setPlanDoc(plan)
        setFieldStatus('ready')
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
        setFieldStatus('missing')
      })
    return () => ac.abort()
  }, [clipId, mode])

  useEffect(() => {
    let cancel = false
    fetch('/search/index.json')
      .then((res) => (res.ok ? res.json() : []))
      .then((rows: Moment[]) => {
        if (!cancel && Array.isArray(rows)) setMoments(rows)
      })
      .catch(() => {
        if (!cancel) setMoments([])
      })
    return () => {
      cancel = true
    }
  }, [])

  useEffect(() => {
    let cancel = false
    fetch('/clips/manifest.json')
      .then(async (res) => {
        if (!res.ok) return []
        const type = res.headers.get('content-type') ?? ''
        if (!type.includes('json')) return []
        return parseManifest(await res.json())
      })
      .then((entries) => {
        if (!cancel) setManifest(entries)
      })
      .catch(() => {})
    return () => {
      cancel = true
    }
  }, [])

  useEffect(() => {
    if (manifest.length === 0) return
    const prefix = clipPrefix(clipSet)
    if (clipId.startsWith(prefix)) return
    const first = manifest.find((entry) => entry.clip_id.startsWith(prefix))
    if (!first) return
    setPlaying(false)
    videoRef.current?.pause()
    setSelection(null)
    setTime(0)
    setClipId(first.clip_id)
  }, [clipSet, clipId, manifest])

  useEffect(() => {
    const ac = new AbortController()
    trackRef.current = null
    selectionRef.current = null
    setDoc(null)
    setSelection(null)
    setDocStatus('loading')
    paintRef.current()
    fetch(trackUrl(source, clipId), { signal: ac.signal })
      .then(async (res) => {
        if (!res.ok) return null
        const type = res.headers.get('content-type') ?? ''
        if (!type.includes('json')) return null
        const body: unknown = await res.json()
        if (!body || typeof body !== 'object' || Array.isArray(body)) return null
        return body as TrackDoc
      })
      .then((loaded) => {
        if (ac.signal.aborted) return
        trackRef.current = loaded
        setDoc(loaded)
        setDocStatus(loaded ? 'ready' : 'missing')
        paintRef.current()
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
        trackRef.current = null
        setDoc(null)
        setDocStatus('missing')
        paintRef.current()
      })
    return () => ac.abort()
  }, [clipId, source])

  useEffect(() => {
    paintRef.current()
  }, [showBoxes, showTrail, doc, selection])

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) void video.play().catch(() => setPlaying(false))
    else video.pause()
  }

  const seekTo = (seconds: number) => {
    const video = videoRef.current
    if (!video || !Number.isFinite(seconds)) return
    const limit = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : seconds
    const next = clamp(seconds, 0, limit)
    video.currentTime = next
    setTime(next)
  }

  const openClip = (nextId: string, seconds: number) => {
    setPlaying(false)
    videoRef.current?.pause()
    setClipSet(clipSetFor(nextId))
    if (nextId === clipId) {
      pendingSeek.current = null
      seekTo(seconds)
      return
    }
    pendingSeek.current = seconds
    setSelection(null)
    setTime(seconds)
    setClipId(nextId)
  }

  const openMoment = (moment: Moment) => {
    openClip(moment.clip_id, moment.t_start)
  }

  const onLeftResize = (event: React.PointerEvent<HTMLDivElement>) => {
    const origin = event.clientX
    const start = leftRef.current
    const width = shellRef.current?.clientWidth ?? window.innerWidth
    beginDrag(event, 'x', (x) => {
      const max = width - rightRef.current - MIN_CENTER - SPLIT * 2
      setLeftW(clamp(start + (x - origin), MIN_SIDE, max))
    })
  }

  const onRightResize = (event: React.PointerEvent<HTMLDivElement>) => {
    const origin = event.clientX
    const start = rightRef.current
    const width = shellRef.current?.clientWidth ?? window.innerWidth
    beginDrag(event, 'x', (x) => {
      const max = width - leftRef.current - MIN_CENTER - SPLIT * 2
      setRightW(clamp(start - (x - origin), MIN_SIDE, max))
    })
  }

  const onBottomResize = (event: React.PointerEvent<HTMLDivElement>) => {
    const origin = event.clientY
    const start = bottomRef.current
    const height = shellRef.current?.clientHeight ?? window.innerHeight
    beginDrag(event, 'y', (y) => {
      const max = height - TOP_H - MIN_VIEW - SPLIT
      setBottomH(clamp(start - (y - origin), MIN_BOTTOM, max))
    })
  }

  const prefix = clipPrefix(clipSet)
  const clipIds = manifest.flatMap((entry) => (entry.clip_id.startsWith(prefix) ? [entry.clip_id] : []))
  const clipOptions = clipIds.length > 0 ? clipIds : [clipId]
  const clipValue = clipOptions.includes(clipId) ? clipId : clipOptions[0]
  const manifestEntry = manifest.find((entry) => entry.clip_id === clipId) ?? null
  const caption =
    source === 'truth'
      ? 'Agreement with the dataset tracker (HyDaT/Polytrack), not hand labels.'
      : (manifestEntry?.source ?? '')
  const panelProps: PanelProps = {
    clipId,
    source,
    doc,
    manifestEntry,
    time,
    onSeek: seekTo,
    onOpenClip: openClip,
  }

  return (
    <div className="shell" ref={shellRef}>
      <header className="topbar">
        <span className="brand">BeeBot</span>
        <select
          className="clip-select"
          aria-label="Source"
          value={source}
          onChange={(event) => setSource(event.target.value as Source)}
        >
          {SOURCES.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </select>
        <select
          className="clip-select"
          aria-label="Clip set"
          value={clipSet}
          onChange={(event) => {
            const next = event.target.value as ClipSet
            const nextPrefix = clipPrefix(next)
            setClipSet(next)
            if (clipId.startsWith(nextPrefix)) return
            const first = manifest.find((entry) => entry.clip_id.startsWith(nextPrefix))
            if (!first) return
            setPlaying(false)
            videoRef.current?.pause()
            setSelection(null)
            setTime(0)
            setClipId(first.clip_id)
          }}
        >
          {CLIP_SETS.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label}
            </option>
          ))}
        </select>
        <select
          className="clip-select"
          aria-label="Clip"
          value={clipValue}
          onChange={(event) => {
            setClipId(event.target.value)
            setTime(0)
            setPlaying(false)
            setSelection(null)
          }}
        >
          {clipOptions.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <span className="modes">
          {MODES.map((item) => (
            <button
              key={item.id}
              type="button"
              className={mode === item.id ? 'toggle on' : 'toggle'}
              aria-pressed={mode === item.id}
              onClick={() => {
                if (mode === item.id) return
                if (item.id !== 'video') videoRef.current?.pause()
                setMode(item.id)
              }}
            >
              {item.label}
            </button>
          ))}
        </span>
        <UploadButton />
      </header>
      <div className="row">
        <section className="panel outliner" style={{ width: leftW }} aria-label="Outliner">
          <div className="region-header">Outliner</div>
          <div className="region-body">
            {docStatus === 'ready' ? (
              <Outliner doc={doc} selection={selection} onSelect={setSelection} />
            ) : (
              <div className="empty-note">
                {docStatus === 'missing' ? `No track file for ${clipId}.` : `Loading ${clipId}…`}
              </div>
            )}
          </div>
        </section>
        <div
          className="splitter v"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize outliner"
          onPointerDown={onLeftResize}
        />
        <section className="panel view" aria-label="Clip">
          <div className="region-header">
            <button
              type="button"
              className="transport"
              onClick={togglePlay}
              aria-label={playing ? 'Pause' : 'Play'}
              aria-pressed={playing}
            >
              {playing ? <PauseIcon /> : <PlayIcon />}
            </button>
            <span className="time">
              {formatTime(time)} / {formatTime(duration)}
            </span>
            <span className="header-spacer" />
            <button
              type="button"
              className={showBoxes ? 'toggle on' : 'toggle'}
              aria-pressed={showBoxes}
              onClick={() => setShowBoxes((on) => !on)}
            >
              Boxes
            </button>
            <button
              type="button"
              className={showTrail ? 'toggle on' : 'toggle'}
              aria-pressed={showTrail}
              onClick={() => setShowTrail((on) => !on)}
            >
              Trail
            </button>
          </div>
          <div className={mode === 'video' ? 'stage' : 'stage field-mode'} ref={stageRef}>
            <video
              ref={videoRef}
              src={`/clips/${clipId}.mp4`}
              preload="auto"
              playsInline
              aria-label={clipId}
            />
            <canvas ref={canvasRef} aria-hidden="true" />
            {mode === 'field' && fieldStatus === 'loading' && <div className="field-note">Loading field…</div>}
            {mode === 'field' && fieldStatus === 'missing' && <div className="field-note">No field file for {clipId}.</div>}
            {mode === 'field' && fieldStatus === 'ready' && fieldDoc && (
              <FieldDiagram clipId={clipId} field={fieldDoc} plan={planDoc} />
            )}
            {mode === 'analytics' && (
              <div className="stage-panel">
                <AnalyticsPanel {...panelProps} />
              </div>
            )}
            {mode === 'forecast' && (
              <div className="stage-panel">
                <ForecastPanel {...panelProps} />
              </div>
            )}
            {mode === 'dataset' && (
              <div className="stage-panel">
                <DatasetPanel source={source} />
              </div>
            )}
          </div>
        </section>
        <div
          className="splitter v"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize properties"
          onPointerDown={onRightResize}
        />
        <section className="panel properties" style={{ width: rightW }} aria-label="Properties">
          <div className="region-header">Properties</div>
          <div className="region-body">
            {caption && <div className="empty-note">{caption}</div>}
            <MomentSearch index={moments} clipId={clipId} onOpen={openMoment} />
            {docStatus === 'missing' && <div className="empty-note">No track file for {clipId}.</div>}
            {docStatus === 'ready' && <Properties doc={doc} selection={selection} onSeek={seekTo} />}
            {docStatus === 'ready' && <ClipVisits doc={doc} />}
          </div>
        </section>
      </div>
      <div
        className="splitter h"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize timeline"
        onPointerDown={onBottomResize}
      />
      <section className="panel timeline" style={{ height: bottomH }} aria-label="Timeline">
        <div className="region-header">Timeline</div>
        <div className="region-body">
          <Timeline doc={doc} time={time} duration={duration} onSeek={seekTo} />
        </div>
      </section>
    </div>
  )
}
