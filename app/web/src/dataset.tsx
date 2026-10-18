import { useEffect, useState, type CSSProperties } from 'react'
import { HELP, IMAGE_NOTES, SOURCE_INFO } from './help'
import { Hint, InfoMark, Intro } from './hint'
import { clip_metrics, pyRound, type MetricDoc, type Summary } from './analytics'
import { trackUrl, type ManifestEntry, type PanelProps } from './types'

const FLOWER_ROWS: Array<{ key: 'ref_iou' | 'ref_center' | 'audit'; label: string }> = [
  { key: 'ref_iou', label: 'vs dataset flower list, box overlap at least 0.3' },
  { key: 'ref_center', label: 'vs dataset flower list, centers within one flower width' },
  { key: 'audit', label: 'vs visual audit of every open flower' },
]

const EVAL_REPORTS: Array<{ url: string; method: string }> = [
  { url: '/eval/report.json', method: SOURCE_INFO.pipeline.label },
  { url: '/eval/report_heuristic.json', method: SOURCE_INFO.heuristic.label },
  { url: '/eval/report_mk15_v1.json', method: 'Mk1.5 alone, whole clip' },
]

const DISPUTED_IMAGE = '/eval/audit/disputed_points_v0.7.jpg'
const FLOWER_IMAGES = [
  '/eval/audit/flowers_sm2_08_boxes.jpg',
  '/eval/audit/flowers_sm3_12_boxes.jpg',
  '/eval/audit/flowers_unboxed_candidates.jpg',
]

const VIEW_FIELDS = ['distinct_boxes', 'boxes_on_open_flowers', 'open_flowers', 'precision', 'recall'] as const

const COVERAGE_CAPTION =
  'Coverage is the share of reference points with a prediction within 50 units at the same time; false alarms are predicted points not within 50 units; visits are scored on farm clips only.'

const FALSE_ALARM_NOTE =
  'Most false alarms are real bees because the reference follows one bee and many honeybee videos have several.'

const FLOWER_WHY =
  'Three numbers exist because the dataset list only contains flowers its tracker used, as radius-estimated squares, and on camera 3 they sit about 13 px low, so overlap misses real flowers.'

type ClipRow = {
  clipId: string
  summary: Summary
}

type EvalRow = {
  method: string
  clips: number | null
  coverage: number | null
  medianCenterError: number | null
  falseAlarms: number | null
  visitRecall: number | null
  visitPrecision: number | null
}

type FlowerScore = {
  key: string
  label: string
  precision: number | null
  recall: number | null
  pred: number | null
  ref: number | null
}

type FlowersView = {
  scores: FlowerScore[]
  views: string[]
}

type EvalState = {
  reports: EvalRow[]
  flowers: FlowersView | null
  disputed: boolean
  flowerImages: string[]
}

const CELL: CSSProperties = {
  textAlign: 'left',
  padding: '3px 8px',
  borderBottom: '1px solid #333',
  whiteSpace: 'nowrap',
  fontWeight: 400,
}

function finite(value: number | null): value is number {
  return value != null && Number.isFinite(value)
}

function numOrNull(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  return null
}

function fmt(value: number | null): string {
  if (!finite(value)) return '—'
  return String(value)
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = values.slice().sort((a, b) => a - b)
  const mid = sorted.length >> 1
  const raw = sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
  return pyRound(raw, 2)
}

function withUnit(value: number | null, unit: string): string {
  if (!finite(value)) return '—'
  return `${value} ${unit}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function parseManifest(body: unknown): ManifestEntry[] {
  if (!Array.isArray(body)) return []
  const entries: ManifestEntry[] = []
  for (const row of body) {
    if (!isRecord(row)) continue
    if (typeof row.clip_id !== 'string' || row.clip_id.length === 0) continue
    if (typeof row.path !== 'string') continue
    if (typeof row.fps !== 'number' || typeof row.width !== 'number' || typeof row.height !== 'number') continue
    if (typeof row.duration_s !== 'number' || typeof row.source !== 'string') continue
    if (typeof row.has_truth !== 'boolean' || typeof row.truth_kind !== 'string') continue
    entries.push({
      clip_id: row.clip_id,
      path: row.path,
      fps: row.fps,
      width: row.width,
      height: row.height,
      duration_s: row.duration_s,
      source: row.source,
      has_truth: row.has_truth,
      truth_kind: row.truth_kind,
    })
  }
  return entries
}

async function loadJson(url: string, signal: AbortSignal): Promise<unknown | null> {
  const res = await fetch(url, { signal })
  if (!res.ok) return null
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('json')) return null
  return res.json()
}

async function imageExists(url: string, signal: AbortSignal): Promise<boolean> {
  const res = await fetch(url, { method: 'HEAD', signal })
  if (!res.ok) return false
  return (res.headers.get('content-type') ?? '').toLowerCase().startsWith('image/')
}

function asDoc(body: unknown, clipId: string): MetricDoc | null {
  if (!isRecord(body)) return null
  const doc = body as MetricDoc
  if (!doc.clip_id) doc.clip_id = clipId
  return doc
}

function parseEval(method: string, body: unknown): EvalRow | null {
  if (!isRecord(body)) return null
  return {
    method,
    clips: numOrNull(body.clips),
    coverage: numOrNull(body.track_coverage),
    medianCenterError: numOrNull(body.median_center_error),
    falseAlarms: numOrNull(body.false_alarm_rate),
    visitRecall: numOrNull(body.visit_recall),
    visitPrecision: numOrNull(body.visit_precision),
  }
}

function parseFlowers(body: unknown): FlowersView | null {
  if (!isRecord(body) || !isRecord(body.overall_per_clip)) return null
  const overall = body.overall_per_clip
  const scores: FlowerScore[] = []
  for (const spec of FLOWER_ROWS) {
    const row = overall[spec.key]
    if (!isRecord(row)) continue
    scores.push({
      key: spec.key,
      label: spec.label,
      precision: numOrNull(row.precision),
      recall: numOrNull(row.recall),
      pred: numOrNull(row.pred),
      ref: numOrNull(row.ref),
    })
  }
  const views: string[] = []
  if (isRecord(body.per_view_distinct_flowers)) {
    for (const [id, row] of Object.entries(body.per_view_distinct_flowers)) {
      if (!isRecord(row)) continue
      const n = (key: (typeof VIEW_FIELDS)[number]) =>
        typeof row[key] === 'number' && Number.isFinite(row[key]) ? (row[key] as number) : null
      const boxes = n('distinct_boxes')
      const onFlowers = n('boxes_on_open_flowers')
      const open = n('open_flowers')
      if (boxes == null || onFlowers == null || open == null) continue
      const camera = /^sm(\d)/.exec(id)?.[1]
      views.push(
        `${camera ? `Camera ${camera}` : id} (${id}): ${onFlowers} of ${boxes} flower boxes are real open flowers, ` +
          `and ${onFlowers} of the ${open} open flowers in view were found.`,
      )
    }
  }
  return { scores, views }
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="prop" style={{ gridTemplateColumns: 'minmax(0, 1fr) auto' }}>
      <span className="prop-key">{label}</span>
      <span className="prop-value">{value}</span>
    </div>
  )
}

function Table({
  label,
  columns,
  rows,
  wrapFirst = false,
  columnHelp,
}: {
  label: string
  columns: string[]
  rows: string[][]
  wrapFirst?: boolean
  columnHelp?: Record<string, string>
}) {
  const head: CSSProperties = { ...CELL, color: '#8d8d8d', fontWeight: 500 }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table
        aria-label={label}
        style={{
          borderCollapse: 'collapse',
          width: 'max-content',
          minWidth: '100%',
          fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
          fontSize: 11,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column || label} scope="col" style={head}>
                {columnHelp?.[column] ? <Hint tip={columnHelp[column]}>{column}</Hint> : column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row[0]}>
              {row.map((cell, index) => (
                <td
                  key={`${row[0]}-${index}`}
                  style={
                    wrapFirst && index === 0
                      ? { ...CELL, whiteSpace: 'normal', maxWidth: 280, verticalAlign: 'top' }
                      : CELL
                  }
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AuditImage({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) return null
  return (
    <figure style={{ margin: '4px 8px 12px' }}>
      <img
        src={src}
        alt={alt}
        onError={() => setFailed(true)}
        style={{ display: 'block', maxWidth: '100%', height: 'auto', background: '#111' }}
      />
      <figcaption className="empty-note" style={{ padding: '6px 0 0', lineHeight: 1.5 }}>
        {(() => {
          const note = IMAGE_NOTES[src.split('/').pop() ?? '']
          return note ? (
            <>
              <b style={{ color: '#e6e6e6' }}>{/[?!.]$/.test(note.title) ? note.title : `${note.title}.`}</b> {note.body}
            </>
          ) : (
            src
          )
        })()}
      </figcaption>
    </figure>
  )
}

function ClipTable({ rows }: { rows: ClipRow[] }) {
  return (
    <Table
      label="Clip metrics"
      columns={[
        'clip',
        'insects',
        'visits',
        'flowers_detected',
        'median_dwell_s',
        'median_transit_s',
        'median_flight_speed_cm_s',
        'revisit_rate',
        'tracked_share',
      ]}
      rows={rows.map((row) => {
        const summary = row.summary
        return [
          row.clipId,
          fmt(summary.insects),
          fmt(summary.visits),
          fmt(summary.flowers_detected),
          fmt(summary.median_dwell_s),
          fmt(summary.median_transit_s),
          fmt(summary.median_flight_speed_cm_s),
          fmt(summary.revisit_rate),
          fmt(summary.tracked_share),
        ]
      })}
    />
  )
}

function headerStats(rows: ClipRow[]) {
  const farm = rows.filter((row) => row.clipId.startsWith('sm'))
  const totalVisits = rows.reduce((sum, row) => sum + row.summary.visits, 0)
  return {
    totalVisits,
    dwell: median(farm.map((row) => row.summary.median_dwell_s).filter(finite)),
    transit: median(farm.map((row) => row.summary.median_transit_s).filter(finite)),
    flight: median(rows.map((row) => row.summary.median_flight_speed_cm_s).filter(finite)),
    tracked: median(rows.map((row) => row.summary.tracked_share).filter(finite)),
  }
}

export function DatasetPanel(props: { source: PanelProps['source'] }) {
  const { source } = props
  const [clips, setClips] = useState<{ status: 'loading' | 'ready' | 'error'; rows: ClipRow[] }>({
    status: 'loading',
    rows: [],
  })
  const [evalState, setEvalState] = useState<{ status: 'loading' | 'ready'; data: EvalState }>({
    status: 'loading',
    data: { reports: [], flowers: null, disputed: false, flowerImages: [] },
  })

  useEffect(() => {
    const ac = new AbortController()
    setClips({ status: 'loading', rows: [] })
    ;(async () => {
      const manifest = parseManifest(await loadJson('/clips/manifest.json', ac.signal))
      const rows: ClipRow[] = []
      await Promise.all(
        manifest.map(async (entry) => {
          try {
            const body = await loadJson(trackUrl(source, entry.clip_id), ac.signal)
            const doc = body ? asDoc(body, entry.clip_id) : null
            if (!doc) return
            const aspect = entry.width > 0 ? entry.height / entry.width : 720 / 1280
            const metrics = clip_metrics(doc, entry.duration_s, aspect)
            rows.push({ clipId: entry.clip_id, summary: metrics.summary })
          } catch (err: unknown) {
            if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) throw err
          }
        }),
      )
      if (ac.signal.aborted) return
      rows.sort((a, b) => manifest.findIndex((entry) => entry.clip_id === a.clipId) - manifest.findIndex((entry) => entry.clip_id === b.clipId))
      setClips({ status: 'ready', rows })
    })().catch((err: unknown) => {
      if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
      setClips({ status: 'error', rows: [] })
    })
    return () => ac.abort()
  }, [source])

  useEffect(() => {
    const ac = new AbortController()
    setEvalState({ status: 'loading', data: { reports: [], flowers: null, disputed: false, flowerImages: [] } })
    ;(async () => {
      const [reports, flowersBody, disputed, flowerImages] = await Promise.all([
        Promise.all(
          EVAL_REPORTS.map(async (spec) => {
            try {
              const body = await loadJson(spec.url, ac.signal)
              return body ? parseEval(spec.method, body) : null
            } catch (err: unknown) {
              if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) throw err
              return null
            }
          }),
        ),
        loadJson('/eval/flowers_report.json', ac.signal).catch((err: unknown) => {
          if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) throw err
          return null
        }),
        imageExists(DISPUTED_IMAGE, ac.signal).catch(() => false),
        Promise.all(FLOWER_IMAGES.map(async (url) => ((await imageExists(url, ac.signal).catch(() => false)) ? url : null))),
      ])
      if (ac.signal.aborted) return
      setEvalState({
        status: 'ready',
        data: {
          reports: reports.filter((row): row is EvalRow => row != null),
          flowers: parseFlowers(flowersBody),
          disputed,
          flowerImages: flowerImages.filter((url): url is string => url != null),
        },
      })
    })().catch((err: unknown) => {
      if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
      setEvalState({ status: 'ready', data: { reports: [], flowers: null, disputed: false, flowerImages: [] } })
    })
    return () => ac.abort()
  }, [])

  const stats = headerStats(clips.rows)
  const flowers = evalState.data.flowers

  return (
    <div
      className="region-body"
      style={{ alignSelf: 'stretch', minHeight: 0, overflow: 'auto', display: 'block' }}
    >
      <Intro title="What this view shows">
        <p>
          How well the system works across all 48 clips. <b>Clips</b> summarizes what was measured in each clip.
          <b> Tracking accuracy</b> compares our insect tracks with the dataset authors&rsquo; own tracker (a reference,
          not hand labels). <b>Flower detection</b> scores the flower boxes three ways. The images are the visual
          checks behind the numbers. Hover any <InfoMark /> for details.
        </p>
      </Intro>
      <div className="prop-head">Clips</div>
      <div className="empty-note">One row per clip for the selected source. “sm*” = the strawberry-farm clips.</div>
      {clips.status === 'loading' && <div className="empty-note">Loading clip metrics…</div>}
      {clips.status === 'error' && <div className="empty-note">Could not load the clip manifest.</div>}
      {clips.status === 'ready' && clips.rows.length === 0 && (
        <div className="empty-note">No track files for this source.</div>
      )}
      {clips.status === 'ready' && clips.rows.length > 0 && (
        <>
          <div className="props">
            <Stat label="total visits" value={fmt(stats.totalVisits)} />
            <Stat label="median dwell, sm* clips" value={withUnit(stats.dwell, 's')} />
            <Stat label="median transit, sm* clips" value={withUnit(stats.transit, 's')} />
            <Stat label="median flight speed" value={withUnit(stats.flight, 'cm/s')} />
            <Stat label="median tracked share" value={fmt(stats.tracked)} />
          </div>
          <ClipTable rows={clips.rows} />
        </>
      )}

      <div className="prop-head">Tracking accuracy</div>
      {evalState.status === 'loading' && <div className="empty-note">Loading eval reports…</div>}
      {evalState.status === 'ready' && evalState.data.reports.length === 0 && (
        <div className="empty-note">No eval reports.</div>
      )}
      {evalState.status === 'ready' && evalState.data.reports.length > 0 && (
        <Table
          label="Tracking accuracy"
          columns={['method', 'clips', 'coverage', 'median center error', 'false alarms', 'visit recall', 'visit precision']}
          columnHelp={{
            method:
              `${SOURCE_INFO.pipeline.label} = our system (motion tracking + Mk1.5). ${SOURCE_INFO.heuristic.label} = ` +
              'the same tracking without Mk1.5. Mk1.5 alone, whole clip = asking Mk1.5 to track a whole clip by itself ' +
              '(tried on 1 clip; it invented a straight-line path, which is why the system uses motion tracking).',
            coverage: HELP.coverage,
            'median center error': HELP.medianError,
            'false alarms': HELP.falseAlarms,
            'visit recall': HELP.visitRecall,
            'visit precision': HELP.visitPrecision,
          }}
          rows={evalState.data.reports.map((row) => [
            row.method,
            fmt(row.clips),
            fmt(row.coverage),
            fmt(row.medianCenterError),
            fmt(row.falseAlarms),
            fmt(row.visitRecall),
            fmt(row.visitPrecision),
          ])}
        />
      )}
      <div className="empty-note">{COVERAGE_CAPTION}</div>
      <div className="empty-note">{FALSE_ALARM_NOTE}</div>
      {evalState.data.disputed && <AuditImage src={DISPUTED_IMAGE} alt="Disputed tracking points" />}

      <div className="prop-head">Flower detection</div>
      <div className="empty-note">
        Precision = share of our flower boxes that are real flowers. Recall = share of real flowers we boxed. The same
        boxes are scored against the dataset&rsquo;s flower list (two ways) and against a visual check of every open
        flower; the note below explains why they differ.
      </div>
      {evalState.status === 'loading' && <div className="empty-note">Loading flower report…</div>}
      {evalState.status === 'ready' && (!flowers || flowers.scores.length === 0) && (
        <div className="empty-note">No flower report.</div>
      )}
      {flowers && flowers.scores.length > 0 && (
        <Table
          label="Flower detection"
          wrapFirst
          columns={['', 'precision', 'recall', 'pred', 'ref']}
          rows={flowers.scores.map((row) => [row.label, fmt(row.precision), fmt(row.recall), fmt(row.pred), fmt(row.ref)])}
        />
      )}
      {flowers &&
        flowers.views.map((line) => (
          <div key={line} className="empty-note">
            {line}
          </div>
        ))}
      <div className="empty-note">{FLOWER_WHY}</div>
      {evalState.data.flowerImages.map((src) => (
        <AuditImage key={src} src={src} alt={src} />
      ))}
    </div>
  )
}
