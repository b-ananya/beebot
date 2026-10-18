/** Pollinator metrics ported from analytics/metrics.py. Observed points only; y is scaled by aspect (H/W). */

export { AnalyticsPanel } from './analytics.tsx'

export const SCALE_HB = 18
export const STRAWBERRY_FLOWER_CM = 2.5
export const FLIGHT_SPEED = 40

export type MetricPoint = {
  t: number
  box?: number[]
  center?: number[]
  interpolated?: boolean
  label?: string
}

export type MetricFlower = {
  flower_id?: string
  box?: number[]
  source?: string
}

export type MetricEvent = {
  type?: string
  track_id?: string
  flower_id?: string
  t_start?: number
  t_end?: number
}

export type MetricDoc = {
  clip_id?: string
  tracks?: Array<{
    track_id?: string
    species?: string
    points?: MetricPoint[]
  }>
  flowers?: MetricFlower[]
  events?: MetricEvent[]
}

export type VisitRow = {
  track_id: string
  flower_id: string
  t_start: number
  t_end: number
  dwell_s: number
  revisit: boolean
}

export type TransitionRow = {
  track_id: string
  from: string
  to: string
  transit_s: number
  distance_units: number
  speed_units_s: number | null
}

export type FlowerMetric = {
  flower_id: string
  source: string
  visits: number
  dwell_s: number
  first_visit_s: number | null
  visitors: string[]
  center: [number, number]
}

export type TrackMetric = {
  species: string
  observed_s: number
  path_units: number
  tortuosity: number | null
}

export type Summary = {
  insects: number
  species: string[]
  visits: number
  flowers_detected: number
  flowers_visited: number
  flowers_never_visited: number
  visits_per_min: number | null
  median_dwell_s: number | null
  mean_dwell_s: number | null
  median_transit_s: number | null
  median_transit_units: number | null
  median_speed_units_s: number | null
  p90_speed_units_s: number | null
  median_speed_cm_s: number | null
  median_flight_speed_units_s: number | null
  median_flight_speed_cm_s: number | null
  flying_share_by_speed: number | null
  nearest_flower_rate: number | null
  revisit_rate: number | null
  median_revisit_interval_s: number | null
  median_approach_s: number | null
  median_abs_turn_deg: number | null
  foraging_share: number | null
  tracked_s: number | null
  tracked_share: number | null
  interpolated_share: number | null
  scale_units_per_cm: number | null
  scale_note: string
}

export type ClipMetrics = {
  summary: Summary
  visits: VisitRow[]
  transitions: TransitionRow[]
  flowers: FlowerMetric[]
  tracks: Record<string, TrackMetric>
  speeds: number[]
  turns_deg: number[]
  behavior_s: Record<string, number>
  aspect: number
}

type FlowerRec = {
  flower_id: string
  source: string
  box: [number, number, number, number]
}

type Motion = {
  observed_s: number
  path_units: number
  speeds: number[]
  turns_deg: number[]
  tortuosity: number | null
}

/** Python 3 round: exact binary value, half to even. */
export function pyRound(x: number, ndigits: number): number {
  if (!Number.isFinite(x)) return x
  if (x === 0 || ndigits < 0) return x === 0 ? 0 : x
  const sign = x < 0 ? -1 : 1
  const buf = new ArrayBuffer(8)
  const f64 = new Float64Array(buf)
  const u64 = new BigUint64Array(buf)
  f64[0] = Math.abs(x)
  const bits = u64[0]
  const expBits = Number((bits >> 52n) & 0x7ffn)
  const frac = bits & ((1n << 52n) - 1n)
  let mantissa: bigint
  let exp2: number
  if (expBits === 0) {
    if (frac === 0n) return 0
    mantissa = frac
    exp2 = -1022 - 52
  } else if (expBits === 0x7ff) {
    return x
  } else {
    mantissa = (1n << 52n) + frac
    exp2 = expBits - 1023 - 52
  }
  const numer = mantissa * 5n ** BigInt(ndigits)
  const shift = exp2 + ndigits
  let rounded: bigint
  if (shift >= 0) {
    rounded = numer << BigInt(shift)
  } else {
    const denom = 1n << BigInt(-shift)
    const q = numer / denom
    const r = numer % denom
    const twice = r * 2n
    rounded = twice > denom || (twice === denom && (q & 1n) === 1n) ? q + 1n : q
  }
  const scale = 10n ** BigInt(ndigits)
  const whole = rounded / scale
  const rem = rounded % scale
  const text = `${whole.toString()}.${rem.toString().padStart(ndigits, '0')}`
  const value = Number(text)
  return sign < 0 ? -value : value
}

function medianRaw(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function med(values: number[]): number | null {
  if (values.length === 0) return null
  return pyRound(medianRaw(values), 2)
}

function percentileLinear(values: number[], q: number): number {
  const sorted = values.slice().sort((a, b) => a - b)
  if (sorted.length === 1) return sorted[0]
  const pos = (sorted.length - 1) * (q / 100)
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, sorted.length - 1)
  const g = pos - lo
  return sorted[lo] * (1 - g) + sorted[hi] * g
}

function trunc(n: number): number {
  return n < 0 ? Math.ceil(n) : Math.floor(n)
}

function center(p: MetricPoint, aspect: number): [number, number] {
  if (p.center && p.center.length >= 2) return [Number(p.center[0]), Number(p.center[1]) * aspect]
  const b = p.box ?? [0, 0, 0, 0]
  return [(b[0] + b[2]) / 2, ((b[1] + b[3]) / 2) * aspect]
}

function fcenter(box: [number, number, number, number]): [number, number] {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
}

function scaleFor(clipId: string, flowers: MetricFlower[], aspect: number): [number | null, string] {
  if (clipId.startsWith('hb')) return [SCALE_HB, 'folding ruler in frame (approx.)']
  const sizes: number[] = []
  for (const flower of flowers) {
    if ((flower.source ?? 'mk1.5') === 'stop') continue
    const box = flower.box
    if (!box || box.length < 4) continue
    sizes.push(Math.max(box[2] - box[0], (box[3] - box[1]) * aspect))
  }
  if (sizes.length === 0) return [null, 'no scale']
  return [medianRaw(sizes) / STRAWBERRY_FLOWER_CM, `median flower = ${STRAWBERRY_FLOWER_CM} cm (approx.)`]
}

function trackMotion(points: MetricPoint[], aspect: number): Motion {
  const obs = points.filter((p) => !p.interpolated)
  if (obs.length < 2) return { observed_s: 0, path_units: 0, speeds: [], turns_deg: [], tortuosity: null }
  const keep = [0]
  for (let i = 1; i < obs.length; i++) {
    if (obs[i].t - obs[keep[keep.length - 1]].t >= 0.2) keep.push(i)
  }
  const ts = keep.map((i) => obs[i].t)
  const cs = keep.map((i) => center(obs[i], aspect))
  if (ts.length < 2) return { observed_s: 0, path_units: 0, speeds: [], turns_deg: [], tortuosity: null }
  const seg: Array<[number, number]> = []
  const dt: number[] = []
  const step: number[] = []
  for (let i = 1; i < ts.length; i++) {
    const dx = cs[i][0] - cs[i - 1][0]
    const dy = cs[i][1] - cs[i - 1][1]
    seg.push([dx, dy])
    dt.push(ts[i] - ts[i - 1])
    step.push(Math.hypot(dx, dy))
  }
  const speeds: number[] = []
  let observed = 0
  for (let i = 0; i < dt.length; i++) {
    if (dt[i] < 0.6) {
      speeds.push(step[i] / dt[i])
      observed += dt[i]
    }
  }
  const turns: number[] = []
  for (let i = 0; i < seg.length - 1; i++) {
    const a = seg[i]
    const b = seg[i + 1]
    if (dt[i] < 0.6 && dt[i + 1] < 0.6 && Math.hypot(a[0], a[1]) > 3 && Math.hypot(b[0], b[1]) > 3) {
      const cross = a[0] * b[1] - a[1] * b[0]
      const dot = a[0] * b[0] + a[1] * b[1]
      turns.push((Math.atan2(cross, dot) * 180) / Math.PI)
    }
  }
  let path = 0
  for (const length of step) path += length
  const net = Math.hypot(cs[cs.length - 1][0] - cs[0][0], cs[cs.length - 1][1] - cs[0][1])
  return {
    observed_s: observed,
    path_units: path,
    speeds,
    turns_deg: turns,
    tortuosity: net > 5 ? pyRound(path / net, 2) : null,
  }
}

function cm(x: number | null, scale: number | null): number | null {
  if (x == null || !scale) return null
  return pyRound(x / scale, 2)
}

export function clip_metrics(doc: MetricDoc, duration_s: number | null = null, aspect = 720 / 1280): ClipMetrics {
  const clipId = doc.clip_id ?? ''
  const flowers = new Map<string, FlowerRec>()
  for (const flower of doc.flowers ?? []) {
    if (!flower.flower_id || !flower.box || flower.box.length < 4) continue
    const box = flower.box
    flowers.set(flower.flower_id, {
      flower_id: flower.flower_id,
      source: flower.source ?? 'mk1.5',
      box: [box[0], box[1] * aspect, box[2], box[3] * aspect],
    })
  }
  const [scale, scaleNote] = scaleFor(clipId, doc.flowers ?? [], aspect)

  const motions = new Map<string, Motion & { species: string }>()
  const labels: Record<string, number> = {}
  let nObs = 0
  let nInterp = 0
  for (const track of doc.tracks ?? []) {
    const points = track.points ?? []
    const motion = trackMotion(points, aspect)
    motions.set(track.track_id ?? '', { species: track.species ?? 'unknown', ...motion })
    for (const point of points) {
      if (point.interpolated) nInterp += 1
      else nObs += 1
    }
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1]
      const b = points[i]
      if (b.t - a.t < 0.6 && a.label) labels[a.label] = (labels[a.label] ?? 0) + (b.t - a.t)
    }
  }

  const events: Array<{ track_id: string; flower_id: string; t_start: number; t_end: number }> = []
  for (const event of doc.events ?? []) {
    if (event.type !== 'visit' || !event.flower_id || !flowers.has(event.flower_id)) continue
    if (typeof event.t_start !== 'number' || typeof event.t_end !== 'number') continue
    events.push({
      track_id: event.track_id ?? '',
      flower_id: event.flower_id,
      t_start: event.t_start,
      t_end: event.t_end,
    })
  }
  events.sort((a, b) => (a.track_id < b.track_id ? -1 : a.track_id > b.track_id ? 1 : a.t_start - b.t_start))

  const byTrack = new Map<string, typeof events>()
  for (const event of events) {
    const list = byTrack.get(event.track_id)
    if (list) list.push(event)
    else byTrack.set(event.track_id, [event])
  }

  const trackPts = new Map<string, MetricPoint[]>()
  for (const track of doc.tracks ?? []) {
    trackPts.set(track.track_id ?? '', (track.points ?? []).filter((point) => !point.interpolated))
  }

  const visitRows: VisitRow[] = []
  const transitions: TransitionRow[] = []
  const seen = new Map<string, Map<string, number>>()
  const revisitIntervals: number[] = []
  const approach: number[] = []
  let nearestHits = 0
  let nearestTotal = 0

  for (const [tid, vs] of byTrack) {
    const seenFlowers = new Map<string, number>()
    seen.set(tid, seenFlowers)
    const pts = trackPts.get(tid) ?? []
    for (let i = 0; i < vs.length; i++) {
      const visit = vs[i]
      const dwell = visit.t_end - visit.t_start
      const revisit = seenFlowers.has(visit.flower_id)
      if (revisit) revisitIntervals.push(visit.t_start - (seenFlowers.get(visit.flower_id) as number))
      visitRows.push({
        track_id: tid,
        flower_id: visit.flower_id,
        t_start: visit.t_start,
        t_end: visit.t_end,
        dwell_s: pyRound(dwell, 2),
        revisit,
      })
      const flower = flowers.get(visit.flower_id) as FlowerRec
      const fc = fcenter(flower.box)
      const radius = Math.max(flower.box[2] - flower.box[0], flower.box[3] - flower.box[1]) / 2
      const before = pts.filter((point) => visit.t_start - 2 <= point.t && point.t < visit.t_start)
      const near = before.filter((point) => {
        const c = center(point, aspect)
        return Math.hypot(c[0] - fc[0], c[1] - fc[1]) <= 3 * radius + 15
      })
      if (before.length) approach.push(near.length ? pyRound(visit.t_start - near[0].t, 2) : 0)
      if (i + 1 < vs.length) {
        const next = vs[i + 1]
        const there = fcenter((flowers.get(next.flower_id) as FlowerRec).box)
        const dist = Math.hypot(fc[0] - there[0], fc[1] - there[1])
        const dt = next.t_start - visit.t_end
        transitions.push({
          track_id: tid,
          from: visit.flower_id,
          to: next.flower_id,
          transit_s: pyRound(dt, 2),
          distance_units: pyRound(dist, 1),
          speed_units_s: dt > 0.05 ? pyRound(dist / dt, 1) : null,
        })
        const others: FlowerRec[] = []
        for (const [fid, item] of flowers) {
          if (fid !== visit.flower_id) others.push(item)
        }
        if (others.length >= 2 && next.flower_id !== visit.flower_id) {
          let nearest = others[0]
          let best = Math.hypot(fcenter(nearest.box)[0] - fc[0], fcenter(nearest.box)[1] - fc[1])
          for (let k = 1; k < others.length; k++) {
            const c = fcenter(others[k].box)
            const d = Math.hypot(c[0] - fc[0], c[1] - fc[1])
            if (d < best) {
              best = d
              nearest = others[k]
            }
          }
          nearestTotal += 1
          if (nearest.flower_id === next.flower_id) nearestHits += 1
        }
      }
      seenFlowers.set(visit.flower_id, visit.t_end)
    }
  }

  const perFlower: FlowerMetric[] = []
  for (const [fid, flower] of flowers) {
    const rows = visitRows.filter((row) => row.flower_id === fid)
    let dwellSum = 0
    let first: number | null = null
    const visitors = new Set<string>()
    for (const row of rows) {
      dwellSum += row.dwell_s
      if (first == null || row.t_start < first) first = row.t_start
      visitors.add(row.track_id)
    }
    perFlower.push({
      flower_id: fid,
      source: flower.source,
      visits: rows.length,
      dwell_s: pyRound(dwellSum, 2),
      first_visit_s: first,
      visitors: [...visitors].sort(),
      center: fcenter(flower.box),
    })
  }

  const speeds: number[] = []
  const turns: number[] = []
  let observed = 0
  const tracks: Record<string, TrackMetric> = {}
  const species = new Set<string>()
  for (const [id, motion] of motions) {
    species.add(motion.species)
    observed += motion.observed_s
    for (const speed of motion.speeds) speeds.push(speed)
    for (const turn of motion.turns_deg) turns.push(turn)
    tracks[id] = {
      species: motion.species,
      observed_s: motion.observed_s,
      path_units: motion.path_units,
      tortuosity: motion.tortuosity,
    }
  }

  const spans: Array<[number, number]> = []
  for (const track of doc.tracks ?? []) {
    const obs = (track.points ?? []).filter((point) => !point.interpolated)
    for (let i = 1; i < obs.length; i++) {
      const a = obs[i - 1].t
      const b = obs[i].t
      if (b - a < 0.6) spans.push([a, b])
    }
  }
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  let covered = 0
  let cur: [number, number] | null = null
  for (const [a, b] of spans) {
    if (cur && a <= cur[1]) cur[1] = Math.max(cur[1], b)
    else {
      if (cur) covered += cur[1] - cur[0]
      cur = [a, b]
    }
  }
  if (cur) covered += cur[1] - cur[0]

  const dwells = visitRows.map((row) => row.dwell_s)
  const flightSpeeds = speeds.filter((speed) => speed >= FLIGHT_SPEED)
  let labelSum = 0
  for (const value of Object.values(labels)) labelSum += value
  let detected = 0
  for (const flower of flowers.values()) {
    if (flower.source !== 'stop') detected += 1
  }
  const visitedDetected = new Set<string>()
  for (const row of visitRows) {
    const flower = flowers.get(row.flower_id)
    if (flower && flower.source !== 'stop') visitedDetected.add(row.flower_id)
  }
  const hasDuration = duration_s != null && duration_s !== 0 && Number.isFinite(duration_s)
  const meanDwell = dwells.length ? dwells.reduce((sum, value) => sum + value, 0) / dwells.length : null

  const summary: Summary = {
    insects: motions.size,
    species: [...species].sort(),
    visits: visitRows.length,
    flowers_detected: detected,
    flowers_visited: new Set(visitRows.map((row) => row.flower_id)).size,
    flowers_never_visited: detected - visitedDetected.size,
    visits_per_min: hasDuration ? pyRound(visitRows.length / ((duration_s as number) / 60), 2) : null,
    median_dwell_s: med(dwells),
    mean_dwell_s: meanDwell == null ? null : pyRound(meanDwell, 2),
    median_transit_s: med(transitions.map((row) => row.transit_s)),
    median_transit_units: med(transitions.map((row) => row.distance_units)),
    median_speed_units_s: med(speeds),
    p90_speed_units_s: speeds.length ? pyRound(percentileLinear(speeds, 90), 1) : null,
    median_speed_cm_s: cm(med(speeds), scale),
    median_flight_speed_units_s: med(flightSpeeds),
    median_flight_speed_cm_s: cm(med(flightSpeeds), scale),
    flying_share_by_speed: speeds.length ? pyRound(flightSpeeds.length / speeds.length, 2) : null,
    nearest_flower_rate: nearestTotal ? pyRound(nearestHits / nearestTotal, 2) : null,
    revisit_rate: visitRows.length ? pyRound(visitRows.filter((row) => row.revisit).length / visitRows.length, 2) : null,
    median_revisit_interval_s: med(revisitIntervals),
    median_approach_s: med(approach),
    median_abs_turn_deg: med(turns.map((turn) => Math.abs(turn))),
    foraging_share: labelSum ? pyRound((labels.foraging ?? 0) / labelSum, 2) : null,
    tracked_s: pyRound(observed, 2),
    tracked_share: hasDuration ? pyRound(Math.min(covered / (duration_s as number), 1), 2) : null,
    interpolated_share: nObs + nInterp ? pyRound(nInterp / (nObs + nInterp), 3) : null,
    scale_units_per_cm: scale ? pyRound(scale, 2) : null,
    scale_note: scaleNote,
  }

  return {
    summary,
    visits: visitRows,
    transitions,
    flowers: perFlower,
    tracks,
    speeds,
    turns_deg: turns,
    behavior_s: labels,
    aspect,
  }
}

/** Observed-point counts per bin. Rows are y. Matches analytics/metrics.py heatmap. */
export function heatmap(doc: MetricDoc, bins = 40): number[][] {
  const grid: number[][] = Array.from({ length: bins }, () => Array(bins).fill(0))
  for (const track of doc.tracks ?? []) {
    for (const point of track.points ?? []) {
      if (point.interpolated) continue
      const [x, y] = center(point, 1)
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      let row = Math.min(trunc((y / 1000) * bins), bins - 1)
      let col = Math.min(trunc((x / 1000) * bins), bins - 1)
      row = ((row % bins) + bins) % bins
      col = ((col % bins) + bins) % bins
      grid[row][col] += 1
    }
  }
  return grid
}
