import { useEffect, useMemo, useRef } from 'react'
import { clip_metrics, heatmap, type FlowerMetric, type Summary, type TransitionRow, type VisitRow } from './analytics'
import type { PanelProps } from './types'

const RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
const VISIT_HELP = 'insect inside a flower zone for at least 0.3 s'
const PROP = { gridTemplateColumns: '168px minmax(0, 1fr)' }
const VISIT_GRID =
  'minmax(88px, 1.2fr) minmax(88px, 1.2fr) minmax(64px, 0.8fr) minmax(64px, 0.8fr) minmax(56px, 0.7fr) minmax(52px, 0.6fr)'
const TRANS_GRID =
  'minmax(88px, 1.1fr) minmax(80px, 1fr) minmax(80px, 1fr) minmax(72px, 0.8fr) minmax(96px, 0.9fr) minmax(96px, 0.9fr)'

function fmt(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return String(value)
}

function rampColor(value: number, max: number): string | null {
  if (value <= 0 || max <= 0) return null
  const index = Math.min(RAMP.length - 1, Math.max(0, Math.ceil((value / max) * RAMP.length) - 1))
  return RAMP[index]
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="prop" style={PROP}>
      <span className="prop-key">{label}</span>
      <span className="prop-value">
        {value}
        {note ? <span className="prop-key"> {note}</span> : null}
      </span>
    </div>
  )
}

function DwellBars({ flowers }: { flowers: FlowerMetric[] }) {
  const rows = flowers.filter((flower) => flower.visits > 0)
  if (rows.length === 0) return <div className="empty-note">No flower visits</div>
  const max = Math.max(...rows.map((row) => row.dwell_s))
  return (
    <div>
      {rows.map((row) => {
        const width = max > 0 ? (row.dwell_s / max) * 100 : 0
        return (
          <div key={row.flower_id} className="prop" style={{ gridTemplateColumns: '96px minmax(0, 1fr) 56px' }}>
            <span className="prop-key">{row.flower_id}</span>
            <span style={{ display: 'block', height: 8, background: '#242424', alignSelf: 'center' }}>
              <span style={{ display: 'block', width: `${width}%`, height: '100%', background: '#3987e5' }} />
            </span>
            <span className="prop-value">{fmt(row.dwell_s)} s</span>
          </div>
        )
      })}
    </div>
  )
}

function SpeedHistogram({ speeds }: { speeds: number[] }) {
  if (speeds.length === 0) return <div className="empty-note">No speed samples</div>
  const bins = 12
  const peakSpeed = Math.max(...speeds)
  const span = peakSpeed > 0 ? peakSpeed : 1
  const counts = Array(bins).fill(0) as number[]
  for (const speed of speeds) {
    let index = Math.floor((speed / span) * bins)
    if (index >= bins) index = bins - 1
    if (index < 0) index = 0
    counts[index] += 1
  }
  const peak = Math.max(...counts)
  const width = 360
  const height = 96
  const bar = width / bins
  return (
    <svg
      viewBox={`0 0 ${width} ${height + 18}`}
      role="img"
      aria-label="Speed histogram, units per second"
      style={{ display: 'block', width: '100%', minWidth: 0, height: 'auto', margin: '4px 8px 8px' }}
    >
      {counts.map((count, index) => {
        const barHeight = peak > 0 ? (count / peak) * height : 0
        return (
          <rect
            key={index}
            x={index * bar + 1}
            y={height - barHeight}
            width={Math.max(bar - 2, 1)}
            height={barHeight}
            fill="#3987e5"
          >
            <title>{count}</title>
          </rect>
        )
      })}
      <text x="0" y={height + 14} fill="#8d8d8d" fontSize="11">
        0
      </text>
      <text x={width - 2} y={height + 14} fill="#8d8d8d" fontSize="11" textAnchor="end">
        {peakSpeed.toFixed(0)}
      </text>
    </svg>
  )
}

function HeatmapFrame({
  clipId,
  time,
  width,
  height,
  flowers,
  grid,
}: {
  clipId: string
  time: number
  width: number
  height: number
  flowers: PanelProps['doc']
  grid: number[][]
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const seek = () => {
      if (!Number.isFinite(time)) return
      const limit = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : time
      const next = Math.min(Math.max(time, 0), limit)
      if (Math.abs(video.currentTime - next) > 0.03) video.currentTime = next
    }
    if (video.readyState >= 1) seek()
    video.addEventListener('loadedmetadata', seek)
    return () => video.removeEventListener('loadedmetadata', seek)
  }, [clipId, time])

  let max = 0
  for (const row of grid) {
    for (const value of row) if (value > max) max = value
  }
  const bins = grid.length || 1
  const cell = 1000 / bins
  const boxes = flowers?.flowers ?? []

  return (
    <div style={{ padding: '4px 8px 8px' }}>
      <div style={{ position: 'relative', width: '100%', maxWidth: 720, aspectRatio: `${width} / ${height}` }}>
        <video
          ref={videoRef}
          src={`/clips/${clipId}.mp4`}
          muted
          playsInline
          preload="auto"
          aria-label={`${clipId} frame`}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'fill',
            display: 'block',
            background: '#111',
            visibility: 'visible',
          }}
        />
        <svg
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          aria-label="Time spent heatmap"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
        >
          {grid.map((row, y) =>
            row.map((value, x) => {
              const fill = rampColor(value, max)
              if (!fill) return null
              return (
                <rect
                  key={`${y}-${x}`}
                  x={x * cell}
                  y={y * cell}
                  width={cell}
                  height={cell}
                  fill={fill}
                  fillOpacity={0.75}
                />
              )
            }),
          )}
          {boxes.map((flower, index) => {
            const box = flower.box
            if (!box || box.length < 4) return null
            const x = Math.min(box[0], box[2])
            const y = Math.min(box[1], box[3])
            return (
              <rect
                key={flower.flower_id ?? index}
                x={x}
                y={y}
                width={Math.abs(box[2] - box[0])}
                height={Math.abs(box[3] - box[1])}
                fill="none"
                stroke="#ffd200"
                strokeWidth={4}
              />
            )
          })}
        </svg>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 6 }}>
        <span className="prop-key">low</span>
        {RAMP.map((color) => (
          <span key={color} style={{ width: 16, height: 8, background: color, opacity: 0.75 }} />
        ))}
        <span className="prop-key">high</span>
      </div>
    </div>
  )
}

function TransitionTable({ rows }: { rows: TransitionRow[] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <div
        className="prop-key"
        style={{ display: 'grid', gridTemplateColumns: TRANS_GRID, gap: 8, padding: '3px 8px' }}
      >
        <span>track_id</span>
        <span>from</span>
        <span>to</span>
        <span>transit_s</span>
        <span>distance_units</span>
        <span>speed_units_s</span>
      </div>
      {rows.length === 0 && <div className="empty-note">No transitions</div>}
      {rows.map((row, index) => (
        <div
          key={`${row.track_id}-${row.from}-${row.to}-${index}`}
          className="prop-value"
          style={{ display: 'grid', gridTemplateColumns: TRANS_GRID, gap: 8, padding: '3px 8px' }}
        >
          <span>{row.track_id}</span>
          <span>{row.from}</span>
          <span>{row.to}</span>
          <span>{fmt(row.transit_s)}</span>
          <span>{fmt(row.distance_units)}</span>
          <span>{fmt(row.speed_units_s)}</span>
        </div>
      ))}
    </div>
  )
}

function VisitTable({ rows, onSeek }: { rows: VisitRow[]; onSeek: (seconds: number) => void }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <div
        className="prop-key"
        style={{ display: 'grid', gridTemplateColumns: VISIT_GRID, gap: 8, padding: '3px 8px' }}
      >
        <span>track_id</span>
        <span>flower_id</span>
        <span>t_start</span>
        <span>t_end</span>
        <span>dwell_s</span>
        <span>revisit</span>
      </div>
      {rows.length === 0 && <div className="empty-note">No visits</div>}
      {rows.map((row) => (
        <button
          key={`${row.track_id}-${row.flower_id}-${row.t_start}`}
          type="button"
          className="visit-row"
          style={{ display: 'grid', gridTemplateColumns: VISIT_GRID, gap: 8 }}
          onClick={() => onSeek(row.t_start)}
        >
          <span>{row.track_id}</span>
          <span>{row.flower_id}</span>
          <span>{fmt(row.t_start)}</span>
          <span>{fmt(row.t_end)}</span>
          <span>{fmt(row.dwell_s)}</span>
          <span>{row.revisit ? 'true' : 'false'}</span>
        </button>
      ))}
    </div>
  )
}

function SummaryStats({ summary }: { summary: Summary }) {
  return (
    <>
      <div className="props">
        <div className="prop-head">Visits</div>
        <Stat label="visits" value={fmt(summary.visits)} note={VISIT_HELP} />
        <Stat label="flowers visited" value={fmt(summary.flowers_visited)} />
        <Stat label="flowers never visited" value={fmt(summary.flowers_never_visited)} />
        <Stat label="visits per minute" value={fmt(summary.visits_per_min)} />
        <Stat label="revisit rate" value={fmt(summary.revisit_rate)} />
        <div className="prop-head">Robot parameters</div>
        <Stat label="median dwell" value={summary.median_dwell_s == null ? '—' : `${fmt(summary.median_dwell_s)} s`} />
        <Stat label="median transit" value={summary.median_transit_s == null ? '—' : `${fmt(summary.median_transit_s)} s`} />
        <Stat
          label="flight speed"
          value={summary.median_flight_speed_cm_s == null ? '—' : `${fmt(summary.median_flight_speed_cm_s)} cm/s`}
          note={summary.scale_note}
        />
        <Stat label="approach time" value={summary.median_approach_s == null ? '—' : `${fmt(summary.median_approach_s)} s`} />
        <Stat label="nearest-flower rate" value={fmt(summary.nearest_flower_rate)} />
        <Stat
          label="median turn"
          value={summary.median_abs_turn_deg == null ? '—' : `${fmt(summary.median_abs_turn_deg)}°`}
        />
        <Stat label="moving share" value={fmt(summary.flying_share_by_speed)} />
        <Stat label="tracked share" value={fmt(summary.tracked_share)} />
        <Stat label="interpolated share" value={fmt(summary.interpolated_share)} />
        <Stat
          label="scale"
          value={summary.scale_units_per_cm == null ? '—' : `${fmt(summary.scale_units_per_cm)} units/cm`}
          note={summary.scale_note}
        />
      </div>
    </>
  )
}

export function AnalyticsPanel({ clipId, source, doc, manifestEntry, time, onSeek }: PanelProps) {
  const duration = manifestEntry?.duration_s
  const aspect =
    manifestEntry && manifestEntry.width > 0 ? manifestEntry.height / manifestEntry.width : 720 / 1280
  const metrics = useMemo(
    () => (doc ? clip_metrics(doc, duration ?? null, aspect) : null),
    [doc, duration, aspect],
  )
  const grid = useMemo(() => (doc ? heatmap(doc) : null), [doc])
  const frameW = manifestEntry?.width || 1280
  const frameH = manifestEntry?.height || 720

  return (
    <div
      className="region-body"
      style={{ alignSelf: 'stretch', minHeight: 0, overflow: 'auto', display: 'block' }}
    >
      {!doc || !metrics || !grid ? (
        <div className="empty-note">
          No track document for {clipId} ({source}).
        </div>
      ) : (
        <>
          <SummaryStats summary={metrics.summary} />
          <div className="prop-head">Dwell per flower</div>
          <DwellBars flowers={metrics.flowers} />
          <div className="prop-head">Speed</div>
          <div className="empty-note">units per second, 1 unit = 1/1000 of frame width</div>
          <SpeedHistogram speeds={metrics.speeds} />
          <div className="prop-head">Heatmap</div>
          <HeatmapFrame
            clipId={clipId}
            time={time}
            width={frameW}
            height={frameH}
            flowers={doc}
            grid={grid}
          />
          <div className="prop-head">Transitions</div>
          <TransitionTable rows={metrics.transitions} />
          <div className="prop-head">Visits</div>
          <VisitTable rows={metrics.visits} onSeek={onSeek} />
        </>
      )}
    </div>
  )
}
