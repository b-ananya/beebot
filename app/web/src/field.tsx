import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { TrackPoint } from './overlay'
import type { ManifestEntry, Source, TrackDoc } from './types'

export type FieldFlower = {
  flower_id?: string
  center?: number[]
  radius?: number
  deficit?: number
  state?: string
  visits?: number
  dwell_s?: number
  seen_in_clips?: number
  station?: number | string
}

export type FieldFile = {
  field_id?: string
  source?: string
  units?: string
  target_visits?: number
  visits_scope?: string
  flowers?: FieldFlower[]
  stations?: unknown[]
}

export type PlanFile = {
  start?: number[]
  route?: string[]
  deficit_covered?: number
  baselines?: Record<string, number>
  planner?: string
  path_length?: number
  budget_s?: number
  field_id?: string
  total_deficit?: number
  video?: string
  source?: string
  robot?: {
    speed?: number
    dwell_s?: number
    units?: string
    time_scale?: number
  }
}

type Point = [number, number]

type FlowerRow = {
  id: string
  state: string
  visits: number
  dwellS: number
  deficit: number
  seen: number | undefined
  center: Point | null
  radius: number | null
  open: boolean
}

type LivePlan = {
  planner: string
  route: string[]
  pathLength: number
  deficitCovered: number
  totalDeficit: number
  baselines: { nearest_all: number; random: number; follow_bee: number }
  budgetS: number
  speed: number
  dwellS: number
}

type FarmStation = {
  id: string
  x: number
  y: number
  visits?: number
  deficit?: number
  flowers?: number
}

type Frame = { left: number; top: number; width: number; height: number; boxH: number }

const START: Point = [0, 1000]
const BEE_GAP_S = 0.6
const EMPTY_FRAME: Frame = { left: 0, top: 0, width: 0, height: 0, boxH: 0 }

const FIELD_CSS = `
.stage.field-mode > video,
.stage.field-mode .field-view video {
  visibility: visible;
}
.field-view {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  pointer-events: none;
  background: transparent;
  z-index: 1;
}
.field-hud,
.field-caption,
.field-detail {
  pointer-events: auto;
  position: relative;
  z-index: 2;
}
.field-hud,
.field-detail {
  background: rgba(20, 20, 20, 0.92);
}
.field-hud {
  flex: 0 0 auto;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  align-items: center;
  padding: 6px 8px;
}
.field-slider {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #e8e8e8;
  white-space: nowrap;
}
.field-slider input[type='range'] {
  width: 92px;
  accent-color: #ff7a18;
}
.field-slider b {
  min-width: 2.4em;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-weight: 500;
}
.field-caption {
  flex: 0 0 auto;
  margin: 0;
  padding: 5px 8px 6px;
  background: rgba(20, 20, 20, 0.94);
  color: #f0f0f0;
  line-height: 1.35;
}
.field-gap {
  flex: 1 1 auto;
  min-height: 72px;
}
.field-detail {
  flex: 0 1 auto;
  max-height: 46%;
  min-height: 0;
  overflow: auto;
}
.field-stats,
.field-scroll {
  flex: 0 0 auto;
}
.field-readouts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px 8px;
  padding: 4px 8px 2px;
  font-size: 11px;
}
.field-wide {
  grid-column: 1 / -1;
}
.field-readouts span,
.field-route,
.field-plan,
.field-missing,
.field-clear {
  line-height: 1.3;
}
.field-clear {
  margin: 0;
  padding: 0 8px 4px;
}
.field-route,
.field-plan,
.field-missing {
  margin: 0;
  padding: 1px 8px;
  font-size: 11px;
}
.field-plan {
  flex: 0 0 auto;
  pointer-events: auto;
  position: relative;
  z-index: 2;
  background: rgba(20, 20, 20, 0.94);
}
.field-route,
.field-plan,
.field-table {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
.field-route {
  overflow-wrap: anywhere;
}
.field-bars {
  display: grid;
  gap: 2px;
  padding: 3px 8px 4px;
}
.field-bar-row {
  display: grid;
  grid-template-columns: 108px minmax(0, 1fr) 42px;
  gap: 6px;
  align-items: center;
}
.field-bar-label,
.field-bar-value {
  font-variant-numeric: tabular-nums;
}
.field-bar-value {
  text-align: right;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.field-bar-track {
  height: 7px;
  background: #3a3a3a;
  border-radius: 1px;
  overflow: hidden;
}
.field-bar-fill {
  display: block;
  height: 100%;
  min-width: 0;
}
.field-bar-fill.ours {
  background: #ff7a18;
}
.field-bar-fill.base {
  background: #9a9a9a;
}
.field-view > svg.field-map {
  position: absolute;
  z-index: 1;
  overflow: visible;
  pointer-events: none;
}
.field-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}
.field-table th,
.field-table td {
  text-align: left;
  padding: 2px 6px;
  border-bottom: 1px solid #333;
  white-space: nowrap;
}
.field-table th {
  color: #8d8d8d;
  font-weight: 500;
  position: sticky;
  top: 0;
  background: #242424;
}
.field-rollout {
  display: block;
  width: 100%;
  max-height: 180px;
  background: #111;
}
.field-farm {
  margin: 0;
  padding: 6px 8px 8px;
}
.field-farm figcaption {
  margin-bottom: 4px;
  color: #8d8d8d;
}
.field-view .field-farm svg {
  width: 100%;
  height: 140px;
  display: block;
  background: #161616;
}
`

function point(value: number[] | undefined): Point | null {
  if (!value || value.length < 2) return null
  const x = value[0]
  const y = value[1]
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  return [x, y]
}

function finite(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function roundTo(value: number, digits: number): number {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

function cell(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  if (Number.isInteger(value)) return String(value)
  return String(roundTo(value, 2))
}

function shade(deficit: number, min: number, max: number): string {
  const t = max > min ? (deficit - min) / (max - min) : 0
  const light = [255, 214, 166]
  const dark = [122, 42, 0]
  const rgb = light.map((channel, index) => Math.round(channel + (dark[index] - channel) * t))
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

function flowerRows(flowers: FieldFlower[] | undefined, target: number): FlowerRow[] {
  return (flowers ?? []).map((flower, index) => {
    const state = flower.state ?? 'open'
    const open = state === 'open'
    const visits = finite(flower.visits) ?? 0
    return {
      id: flower.flower_id ?? `unnamed-${index + 1}`,
      state,
      visits,
      dwellS: finite(flower.dwell_s) ?? 0,
      deficit: open ? Math.max(target - visits, 0) : 0,
      seen: finite(flower.seen_in_clips),
      center: point(flower.center),
      radius: finite(flower.radius) ?? null,
      open,
    }
  })
}

function flowerTitle(row: FlowerRow): string {
  return [
    row.id,
    `state ${row.state}`,
    `visits ${row.visits}`,
    `dwell_s ${cell(row.dwellS)}`,
    `deficit ${cell(row.deficit)}`,
    `seen_in_clips ${row.seen == null ? '—' : row.seen}`,
  ].join(', ')
}

function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1])
}

function routeTime(start: Point, stops: Point[], speed: number, dwellS: number): number {
  let t = 0
  let here = start
  for (const stop of stops) {
    t += dist(here, stop) / speed + dwellS
    here = stop
  }
  return t
}

function simulate(
  start: Point,
  order: number[],
  pts: Point[],
  prize: number[],
  speed: number,
  dwellS: number,
  budgetS: number,
): { done: number[]; cleared: number; length: number } {
  let t = 0
  let here = start
  let cleared = 0
  let length = 0
  const done: number[] = []
  for (const index of order) {
    const leg = dist(here, pts[index])
    if (t + leg / speed + dwellS > budgetS) break
    t += leg / speed + dwellS
    length += leg
    cleared += prize[index]
    here = pts[index]
    done.push(index)
  }
  return { done, cleared, length }
}

/** Shorten an open route (fixed start, free end) by reversing segments. Port of sim/planner.py. */
function twoOpt(start: Point, order: number[], pts: Point[]): number[] {
  let best = order.slice()
  let improved = true
  while (improved) {
    improved = false
    for (let i = 0; i < best.length - 1; i++) {
      for (let j = i + 1; j < best.length; j++) {
        const cand = best.slice(0, i).concat(best.slice(i, j + 1).reverse(), best.slice(j + 1))
        const candTime = routeTime(start, cand.map((k) => pts[k]), 1, 0)
        const bestTime = routeTime(start, best.map((k) => pts[k]), 1, 0)
        if (candTime < bestTime - 1e-9) {
          best = cand
          improved = true
        }
      }
    }
  }
  return best
}

function greedy2opt(start: Point, pts: Point[], prize: number[], speed: number, dwellS: number, budgetS: number): number[] {
  let order: number[] = []
  const left = new Set<number>()
  prize.forEach((value, index) => {
    if (value > 0) left.add(index)
  })
  while (left.size > 0) {
    let best: { score: number; cand: number[]; index: number } | null = null
    for (const index of left) {
      for (let pos = 0; pos <= order.length; pos++) {
        const cand = order.slice(0, pos).concat(index, order.slice(pos))
        const t = routeTime(start, cand.map((k) => pts[k]), speed, dwellS)
        if (t <= budgetS) {
          const added = t - routeTime(start, order.map((k) => pts[k]), speed, dwellS)
          const score = prize[index] / Math.max(added, 1e-6)
          if (best === null || score > best.score) best = { score, cand, index }
        }
      }
    }
    if (best === null) break
    order = twoOpt(start, best.cand, pts)
    left.delete(best.index)
  }
  return order
}

function nearestAll(start: Point, pts: Point[]): number[] {
  const order: number[] = []
  let here = start
  const left = new Set(pts.map((_, index) => index))
  while (left.size > 0) {
    let pick = -1
    let bestD = Infinity
    for (const index of left) {
      const d = dist(here, pts[index])
      if (pick < 0 || d < bestD) {
        pick = index
        bestD = d
      }
    }
    order.push(pick)
    here = pts[pick]
    left.delete(pick)
  }
  return order
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function randomOrder(count: number, seed: number): number[] {
  const order = Array.from({ length: count }, (_, index) => index)
  const rand = mulberry32(seed + 1)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    const swap = order[i]
    order[i] = order[j]
    order[j] = swap
  }
  return order
}

function planRoute(
  flowers: FlowerRow[],
  aspect: number,
  speed: number,
  dwellS: number,
  budgetS: number,
  beeOrder: string[],
): LivePlan {
  const placed = flowers.filter(
    (flower): flower is FlowerRow & { center: Point; radius: number } => flower.center != null && flower.radius != null,
  )
  const pts = placed.map((flower) => [flower.center[0] * aspect, flower.center[1]] as Point)
  const prize = placed.map((flower) => (flower.open ? flower.deficit : 0))
  const total = prize.reduce((sum, value) => sum + value, 0)
  const denom = total || 1
  const start: Point = [START[0] * aspect, START[1]]
  const score = (order: number[]) => {
    const result = simulate(start, order, pts, prize, speed, dwellS, budgetS)
    return { done: result.done, covered: result.cleared / denom, length: result.length }
  }
  const ours = score(greedy2opt(start, pts, prize, speed, dwellS, budgetS))
  const nearest = score(nearestAll(start, pts))
  let randomSum = 0
  for (let seed = 0; seed < 20; seed++) randomSum += score(randomOrder(placed.length, seed)).covered
  const ids = new Map(placed.map((flower, index) => [flower.id, index]))
  const follow = beeOrder.flatMap((id) => {
    const index = ids.get(id)
    return index == null ? [] : [index]
  })
  return {
    planner: 'greedy_2opt',
    route: ours.done.map((index) => placed[index].id),
    pathLength: roundTo(ours.length, 1),
    deficitCovered: roundTo(ours.covered, 3),
    totalDeficit: total,
    baselines: {
      nearest_all: roundTo(nearest.covered, 3),
      random: roundTo(randomSum / 20, 3),
      follow_bee: roundTo(score(follow).covered, 3),
    },
    budgetS,
    speed,
    dwellS,
  }
}

function visitEvents(doc: TrackDoc | null) {
  return (doc?.events ?? []).filter((event) => event.flower_id && (event.type == null || event.type === 'visit'))
}

function medianVisitDwell(doc: TrackDoc | null): number {
  const dwells = visitEvents(doc)
    .flatMap((event) => {
      if (event.t_start == null || event.t_end == null || !(event.t_end > event.t_start)) return []
      return [event.t_end - event.t_start]
    })
    .sort((a, b) => a - b)
  if (dwells.length === 0) return 3
  return roundTo(dwells[Math.floor(dwells.length / 2)], 2)
}

function beeVisitOrder(doc: TrackDoc | null): string[] {
  return visitEvents(doc)
    .slice()
    .sort((a, b) => (a.t_start ?? 0) - (b.t_start ?? 0))
    .flatMap((event) => (event.flower_id ? [event.flower_id] : []))
}

function trackCenter(p: TrackPoint): Point | null {
  const center = point(p.center)
  if (center) return center
  const box = p.box
  if (!box || box.length < 4 || !box.every((value) => Number.isFinite(value))) return null
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
}

function longestTrack(doc: TrackDoc | null): TrackPoint[] {
  let best: TrackPoint[] = []
  for (const track of doc?.tracks ?? []) {
    const points = track.points ?? []
    if (points.length > best.length) best = points
  }
  return best
}

function beePaths(points: TrackPoint[]): Point[][] {
  const sorted = points.filter((p) => Number.isFinite(p.t)).slice().sort((a, b) => a.t - b.t)
  const segments: Point[][] = []
  let current: Point[] = []
  let prevT: number | null = null
  for (const p of sorted) {
    const center = trackCenter(p)
    if (prevT != null && p.t - prevT > BEE_GAP_S) {
      if (current.length > 1) segments.push(current)
      current = []
    }
    if (center) current.push(center)
    else if (current.length > 1) {
      segments.push(current)
      current = []
    } else {
      current = []
    }
    prevT = p.t
  }
  if (current.length > 1) segments.push(current)
  return segments
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function coordPair(value: unknown): Point | null {
  if (!Array.isArray(value)) return null
  const x = finite(value[0])
  const y = finite(value[1])
  if (x == null || y == null) return null
  return [x, y]
}

function stationPoint(item: Record<string, unknown>): Point | null {
  return (
    coordPair(item.center) ??
    coordPair(item.position) ??
    coordPair(item.xy) ??
    (finite(item.x) != null && finite(item.y) != null ? [finite(item.x) as number, finite(item.y) as number] : null)
  )
}

/** Station marks only where the farm file already has coordinates. */
function parseFarm(body: unknown): FarmStation[] | null {
  const doc = asRecord(body)
  const listed = doc && Array.isArray(doc.stations) ? doc.stations : Array.isArray(body) ? body : null
  const stations: FarmStation[] = []
  if (listed) {
    for (const raw of listed) {
      const item = asRecord(raw)
      if (!item) continue
      const center = stationPoint(item)
      if (!center) continue
      const id = String(item.station ?? item.station_id ?? item.id ?? stations.length + 1)
      stations.push({
        id,
        x: center[0],
        y: center[1],
        visits: finite(item.visits),
        deficit: finite(item.deficit),
        flowers: finite(item.flowers) ?? finite(item.n_flowers),
      })
    }
  } else if (doc && Array.isArray(doc.flowers)) {
    const groups = new Map<string, { x: number; y: number; n: number; visits: number; deficit: number }>()
    for (const raw of doc.flowers) {
      const flower = asRecord(raw)
      if (!flower || flower.station == null) continue
      const center = stationPoint(flower)
      if (!center) continue
      const id = String(flower.station)
      const group = groups.get(id) ?? { x: 0, y: 0, n: 0, visits: 0, deficit: 0 }
      group.x += center[0]
      group.y += center[1]
      group.n += 1
      group.visits += finite(flower.visits) ?? 0
      group.deficit += finite(flower.deficit) ?? 0
      groups.set(id, group)
    }
    for (const [id, group] of groups) {
      stations.push({
        id,
        x: group.x / group.n,
        y: group.y / group.n,
        visits: group.visits,
        deficit: group.deficit,
        flowers: group.n,
      })
    }
  }
  if (stations.length === 0) return null
  stations.sort((a, b) => {
    const na = Number(a.id)
    const nb = Number(b.id)
    if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb
    return a.id.localeCompare(b.id)
  })
  return stations
}

function containedFrame(boxW: number, boxH: number, mediaW: number, mediaH: number): Frame {
  if (boxW <= 0 || boxH <= 0 || mediaW <= 0 || mediaH <= 0) return { ...EMPTY_FRAME, boxH }
  const scale = Math.min(boxW / mediaW, boxH / mediaH)
  const width = mediaW * scale
  const height = mediaH * scale
  return { left: (boxW - width) / 2, top: (boxH - height) / 2, width, height, boxH }
}

function formatShare(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

const BARS: Array<{ key: 'ours' | 'nearest_all' | 'random' | 'follow_bee'; label: string; ours: boolean }> = [
  { key: 'ours', label: 'Our route', ours: true },
  { key: 'nearest_all', label: 'Nearest neighbor', ours: false },
  { key: 'random', label: 'Random', ours: false },
  { key: 'follow_bee', label: 'Follow the bee', ours: false },
]

export function FieldDiagram({
  clipId,
  field,
  plan,
  source,
  doc,
  manifestEntry,
}: {
  clipId: string
  field: FieldFile
  plan: PlanFile | null
  source?: Source
  doc?: TrackDoc | null
  manifestEntry?: ManifestEntry | null
}) {
  const rootRef = useRef<HTMLDivElement>(null)
  const [target, setTarget] = useState(4)
  const [budget, setBudget] = useState(30)
  const [speed, setSpeed] = useState(100)
  const [fetched, setFetched] = useState<TrackDoc | null>(null)
  const [frame, setFrame] = useState<Frame>(EMPTY_FRAME)
  const [rollout, setRollout] = useState<'pending' | 'ready' | 'missing'>('pending')
  const [stations, setStations] = useState<FarmStation[] | null>(null)

  const track = doc === undefined ? fetched : doc
  const aspect =
    manifestEntry && manifestEntry.width > 0 && manifestEntry.height > 0
      ? manifestEntry.width / manifestEntry.height
      : 16 / 9
  const truthRollout = source === 'truth' || field.source === 'truth'
  const rolloutUrl = `/rollouts/${clipId}_plan${truthRollout ? '_truth' : ''}.mp4`
  const timeScale = finite(plan?.robot?.time_scale)

  useEffect(() => {
    setTarget(4)
    setBudget(30)
    setSpeed(100)
  }, [clipId])

  useEffect(() => {
    if (doc !== undefined) return
    const ac = new AbortController()
    setFetched(null)
    fetch(`/tracks/${clipId}.json`, { signal: ac.signal })
      .then(async (res) => {
        if (!res.ok) return null
        const type = res.headers.get('content-type') ?? ''
        if (!type.includes('json')) return null
        const body: unknown = await res.json()
        if (!body || typeof body !== 'object' || Array.isArray(body)) return null
        return body as TrackDoc
      })
      .then((loaded) => {
        if (!ac.signal.aborted) setFetched(loaded)
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
        setFetched(null)
      })
    return () => ac.abort()
  }, [clipId, doc])

  useEffect(() => {
    const ac = new AbortController()
    setRollout('pending')
    fetch(rolloutUrl, { method: 'HEAD', signal: ac.signal })
      .then((res) => {
        const type = res.headers.get('content-type') ?? ''
        if (!ac.signal.aborted) setRollout(res.ok && type.startsWith('video/') ? 'ready' : 'missing')
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
        setRollout('missing')
      })
    return () => ac.abort()
  }, [rolloutUrl])

  useEffect(() => {
    const ac = new AbortController()
    fetch('/field/farm_day01.json', { signal: ac.signal })
      .then(async (res) => {
        if (!res.ok) return null
        const type = res.headers.get('content-type') ?? ''
        if (!type.includes('json')) return null
        return res.json() as Promise<unknown>
      })
      .then((body) => {
        if (!ac.signal.aborted) setStations(parseFarm(body))
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return
        setStations(null)
      })
    return () => ac.abort()
  }, [])

  useLayoutEffect(() => {
    const root = rootRef.current
    const stage = root?.parentElement
    if (!root || !stage) return
    const video = stage.querySelector(':scope > video')
    const fit = () => {
      const media =
        video instanceof HTMLVideoElement && video.videoWidth > 0 && video.videoHeight > 0
          ? [video.videoWidth, video.videoHeight]
          : manifestEntry && manifestEntry.width > 0 && manifestEntry.height > 0
            ? [manifestEntry.width, manifestEntry.height]
            : [16, 9]
      const next = containedFrame(root.clientWidth, root.clientHeight, media[0], media[1])
      setFrame((prev) =>
        prev.left === next.left &&
        prev.top === next.top &&
        prev.width === next.width &&
        prev.height === next.height &&
        prev.boxH === next.boxH
          ? prev
          : next,
      )
    }
    fit()
    const observer = new ResizeObserver(fit)
    observer.observe(root)
    if (video instanceof HTMLVideoElement) video.addEventListener('loadedmetadata', fit)
    return () => {
      observer.disconnect()
      if (video instanceof HTMLVideoElement) video.removeEventListener('loadedmetadata', fit)
    }
  }, [clipId, manifestEntry])

  const rows = useMemo(() => flowerRows(field.flowers, target), [field.flowers, target])
  const dwell = useMemo(() => medianVisitDwell(track), [track])
  const beeOrder = useMemo(() => beeVisitOrder(track), [track])
  const bee = useMemo(() => beePaths(longestTrack(track)), [track])
  const live = useMemo(
    () => planRoute(rows, aspect, speed, dwell, budget, beeOrder),
    [rows, aspect, speed, dwell, budget, beeOrder],
  )

  const placed = rows.filter((row): row is FlowerRow & { center: Point; radius: number } => row.center != null && row.radius != null)
  const byId = new Map(placed.map((row) => [row.id, row]))
  const openDeficits = placed.filter((row) => row.open).map((row) => row.deficit)
  const minDeficit = openDeficits.length ? Math.min(...openDeficits) : 0
  const maxDeficit = openDeficits.length ? Math.max(...openDeficits) : 0
  const neverVisited = placed.filter((row) => row.visits === 0).length
  const routePoints: Point[] = [START]
  for (const id of live.route) {
    const flower = byId.get(id)
    if (!flower) continue
    routePoints.push(flower.center)
  }
  const scores = BARS.map((bar) => ({
    ...bar,
    value: bar.key === 'ours' ? live.deficitCovered : live.baselines[bar.key],
  }))
  const sourceLabel = field.source || source || '—'
  const bottomBand = frame.boxH - frame.top - frame.height
  const detailMax = bottomBand >= 120 ? bottomBand : undefined

  return (
    <div className="field-view" ref={rootRef} aria-label={`Field ${clipId}`}>
      <style>{FIELD_CSS}</style>
      <div className="field-hud">
        <label className="field-slider">
          Target visits
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={target}
            aria-label="Target visits"
            onChange={(event) => setTarget(Number(event.target.value))}
          />
          <b>{target}</b>
        </label>
        <label className="field-slider">
          Budget
          <input
            type="range"
            min={10}
            max={120}
            step={5}
            value={budget}
            aria-label="Budget"
            onChange={(event) => setBudget(Number(event.target.value))}
          />
          <b>{budget} s</b>
        </label>
        <label className="field-slider">
          Robot speed
          <input
            type="range"
            min={25}
            max={300}
            step={25}
            value={speed}
            aria-label="Robot speed"
            onChange={(event) => setSpeed(Number(event.target.value))}
          />
          <b>{speed}</b>
        </label>
      </div>
      <p className="field-caption">
        A visit is not pollination. Target visits are a setting. The layout is approximate. This is a planning prototype, not flight physics.
      </p>
      <p className="field-plan">
        Planner {live.planner}. Path length {live.pathLength.toFixed(1)}. Budget {live.budgetS} s. Speed {live.speed}. Dwell{' '}
        {live.dwellS.toFixed(2)} s.{timeScale != null ? ` Time scale ${timeScale}.` : ''}
      </p>
      <div className="field-gap" />
      {frame.width > 0 && (
        <svg
          className="field-map"
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
          role="img"
          aria-label={`Pollination map for ${clipId}`}
          style={{ left: frame.left, top: frame.top, width: frame.width, height: frame.height }}
        >
          {bee.map((segment, index) => (
            <polyline
              key={`bee-${index}`}
              points={segment.map((p) => p.join(',')).join(' ')}
              fill="none"
              stroke="#d2d2d2"
              strokeWidth="2.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {placed.map((flower) => (
            <circle
              key={flower.id}
              cx={flower.center[0]}
              cy={flower.center[1]}
              r={flower.radius}
              fill={flower.open ? shade(flower.deficit, minDeficit, maxDeficit) : '#d9d9d9'}
              stroke={flower.open ? '#3d2208' : '#555'}
              strokeWidth="1.5"
              vectorEffect="non-scaling-stroke"
              pointerEvents="auto"
            >
              <title>{flowerTitle(flower)}</title>
            </circle>
          ))}
          {routePoints.length > 1 && (
            <polyline
              points={routePoints.map((p) => p.join(',')).join(' ')}
              fill="none"
              stroke="#111"
              strokeWidth="5"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {routePoints.length > 1 && (
            <polyline
              points={routePoints.map((p) => p.join(',')).join(' ')}
              fill="none"
              stroke="#ff7a18"
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          )}
          <circle cx={START[0]} cy={START[1]} r={Math.max(8, (frame.height > 0 ? 5000 / frame.height : 8))} fill="#ff7a18" />
          {live.route.map((id, index) => {
            const flower = byId.get(id)
            if (!flower) return null
            return (
              <text
                key={`${id}-${index}`}
                x={flower.center[0]}
                y={flower.center[1] + 6}
                textAnchor="middle"
                fill="#f4f4f4"
                stroke="#111"
                strokeWidth="4"
                paintOrder="stroke"
                fontSize="22"
              >
                {index + 1}
              </text>
            )
          })}
        </svg>
      )}
      <div className="field-detail" style={detailMax != null ? { maxHeight: detailMax } : undefined}>
        <div className="field-stats">
          <div className="field-readouts">
            <span>Flowers mapped {placed.length}</span>
            <span>Never visited {neverVisited}</span>
            <span>Visits still needed {live.totalDeficit}</span>
            <span>Share cleared {live.totalDeficit === 0 ? '—' : formatShare(live.deficitCovered)}</span>
            <span className="field-wide">Visits scope {field.visits_scope ?? '—'}</span>
            <span>Source {sourceLabel}</span>
          </div>
          {live.totalDeficit === 0 && <p className="field-clear">Every mapped flower is already at the target.</p>}
          <div className="field-bars" role="list" aria-label="Deficit covered">
            {scores.map((bar) => (
              <div className="field-bar-row" role="listitem" key={bar.key}>
                <span className="field-bar-label">{bar.label}</span>
                <span className="field-bar-track">
                  <span
                    className={bar.ours ? 'field-bar-fill ours' : 'field-bar-fill base'}
                    style={{ width: `${Math.max(0, Math.min(1, bar.value)) * 100}%` }}
                  />
                </span>
                <span className="field-bar-value">{bar.value.toFixed(3)}</span>
              </div>
            ))}
          </div>
          <p className="field-route">Route {live.route.length > 0 ? live.route.join(' → ') : '—'}</p>
        </div>
        <div className="field-scroll">
          <table className="field-table" aria-label="Flowers">
            <thead>
              <tr>
                <th>id</th>
                <th>state</th>
                <th>visits</th>
                <th>dwell_s</th>
                <th>deficit</th>
                <th>seen_in_clips</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>{row.id}</td>
                  <td>{row.state}</td>
                  <td>{row.visits}</td>
                  <td>{cell(row.dwellS)}</td>
                  <td>{cell(row.deficit)}</td>
                  <td>{row.seen == null ? '—' : row.seen}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rollout === 'ready' && <video className="field-rollout" src={rolloutUrl} controls playsInline preload="metadata" />}
          {rollout === 'missing' && <p className="field-missing">Run python -m sim.run_plan {clipId}</p>}
          {stations && stations.length > 0 && <FarmStations stations={stations} />}
        </div>
      </div>
    </div>
  )
}

function FarmStations({ stations }: { stations: FarmStation[] }) {
  const xs = stations.map((station) => station.x)
  const ys = stations.map((station) => station.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const span = Math.max(maxX - minX, maxY - minY, 1)
  const pad = span * 0.2
  const radius = Math.max(span * 0.04, span * 0.02)
  return (
    <figure className="field-farm">
      <figcaption>Stations</figcaption>
      <svg
        viewBox={`${minX - pad} ${minY - pad} ${Math.max(maxX - minX, 0) + pad * 2} ${Math.max(maxY - minY, 0) + pad * 2}`}
        role="img"
        aria-label="Farm stations"
      >
        {stations.map((station) => (
          <g key={station.id}>
            <circle cx={station.x} cy={station.y} r={radius} fill="#ff7a18" />
            <text x={station.x} y={station.y + radius * 0.35} textAnchor="middle" fill="#111" fontSize={radius}>
              {station.id}
            </text>
            <title>
              {`station ${station.id}${station.flowers != null ? `, flowers ${station.flowers}` : ''}${station.visits != null ? `, visits ${station.visits}` : ''}${station.deficit != null ? `, deficit ${station.deficit}` : ''}`}
            </title>
          </g>
        ))}
      </svg>
    </figure>
  )
}
