/** One video frame at 30fps, plus a little slack. Farther apart is a gap: do not join or hold across it. */
export const GAP_S = 0.05
export const TRAIL_S = 2

export type Box = [number, number, number, number]

export type TrackPoint = {
  t: number
  box: Box
  center?: [number, number]
  interpolated?: boolean
  /** foraging, flying, hovering, walking */
  label?: string
}

export type InsectTrack = {
  track_id?: string
  species?: string
  points?: TrackPoint[]
}

export type Flower = {
  flower_id?: string
  box?: Box
  source?: string
}

export type VisitEvent = {
  type?: string
  flower_id?: string
  track_id?: string
  t_start?: number
  t_end?: number
}

export type TrackFile = {
  tracks?: InsectTrack[]
  flowers?: Flower[]
  events?: VisitEvent[]
}

const INSECT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
/** Visit fill. RGB of the green app/render.py burns in for a visit in progress. */
const VISIT_FILL = '#78ff00'
const VISIT_ALPHA = 0.35
/** Label height in the same 0–1000 frame the boxes use, then scaled by canvas width. */
const LABEL_UNITS = 16

function isBox(box: Box | undefined): box is Box {
  return !!box && box.length === 4 && box.every((n) => Number.isFinite(n))
}

export function usablePoints(points: TrackPoint[] | undefined): TrackPoint[] {
  if (!points) return []
  return points.filter((p) => p.interpolated !== true && Number.isFinite(p.t) && isBox(p.box))
}

function centerOf(p: TrackPoint): [number, number] {
  const c = p.center
  if (c && c.length === 2 && c.every((n) => Number.isFinite(n))) return [c[0], c[1]]
  return [(p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2]
}

function lastIndexAtOrBefore(points: TrackPoint[], t: number): number {
  let lo = 0
  let hi = points.length - 1
  let ans = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (points[mid].t <= t) {
      ans = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return ans
}

function firstIndexAtOrAfter(points: TrackPoint[], t: number): number {
  let lo = 0
  let hi = points.length - 1
  let ans = points.length
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (points[mid].t >= t) {
      ans = mid
      hi = mid - 1
    } else {
      lo = mid + 1
    }
  }
  return ans
}

function boxRect(box: Box, sx: number, sy: number): { x: number; y: number; w: number; h: number } {
  const x1 = Math.min(box[0], box[2]) * sx
  const y1 = Math.min(box[1], box[3]) * sy
  const x2 = Math.max(box[0], box[2]) * sx
  const y2 = Math.max(box[1], box[3]) * sy
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
}

function strokeBox(ctx: CanvasRenderingContext2D, box: Box, sx: number, sy: number): void {
  const rect = boxRect(box, sx, sy)
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h)
}

function labelPx(canvasWidth: number): number {
  return Math.max(12, (LABEL_UNITS * canvasWidth) / 1000)
}

function textOf(value: string | undefined): string {
  return typeof value === 'string' ? value.trim() : ''
}

function activeFlowerIds(events: VisitEvent[] | undefined, time: number): Set<string> {
  const ids = new Set<string>()
  for (const event of events ?? []) {
    if (event.type !== 'visit' || !event.flower_id) continue
    const start = event.t_start
    const end = event.t_end
    if (typeof start !== 'number' || typeof end !== 'number') continue
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue
    if (start <= time && time <= end) ids.add(event.flower_id)
  }
  return ids
}

function visitedFlowerIds(events: VisitEvent[] | undefined): Set<string> {
  const ids = new Set<string>()
  for (const event of events ?? []) {
    if (event.type === 'visit' && event.flower_id) ids.add(event.flower_id)
  }
  return ids
}

/** Species, or track_id when species is missing or "unknown", plus the point's behavior. */
function insectCaption(track: InsectTrack, point: TrackPoint): string {
  const species = textOf(track.species)
  const name = species && species !== 'unknown' ? species : textOf(track.track_id)
  const behavior = textOf(point.label)
  if (name && behavior) return `${name} · ${behavior}`
  return name || behavior
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  baseline: 'top' | 'bottom',
): void {
  if (!text) return
  const { width, height } = ctx.canvas
  const size = labelPx(width)
  const pad = 2
  ctx.font = `${size}px system-ui, sans-serif`
  ctx.textAlign = 'left'
  ctx.textBaseline = baseline
  const textW = ctx.measureText(text).width
  let px = x
  if (px + textW > width - pad) px = width - pad - textW
  if (px < pad) px = pad
  let py = y
  if (baseline === 'bottom') {
    if (py < size + pad) py = size + pad
    if (py > height - pad) py = height - pad
  } else if (py + size > height - pad) {
    py = height - pad - size
  }
  if (py < pad) py = pad
  ctx.lineJoin = 'round'
  ctx.lineWidth = Math.max(2, size / 5)
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)'
  ctx.strokeText(text, px, py)
  ctx.fillStyle = color
  ctx.fillText(text, px, py)
}

export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  doc: TrackFile | null,
  time: number,
  opts: { boxes: boolean; trail: boolean },
): void {
  const { width, height } = ctx.canvas
  ctx.clearRect(0, 0, width, height)
  ctx.globalAlpha = 1
  if (!doc || width < 2 || height < 2 || !Number.isFinite(time)) return

  const sx = width / 1000
  const sy = height / 1000
  const tracks = doc.tracks ?? []

  const activeFlowers = activeFlowerIds(doc.events, time)
  const visitedFlowers = visitedFlowerIds(doc.events)
  const labelGap = (6 * height) / 1000

  if (opts.boxes) {
    ctx.lineJoin = 'miter'
    ctx.lineWidth = Math.max(1, width / 640)
    for (const flower of doc.flowers ?? []) {
      if (!isBox(flower.box)) continue
      const visiting = !!flower.flower_id && activeFlowers.has(flower.flower_id)
      if (visiting) {
        ctx.save()
        ctx.globalAlpha = VISIT_ALPHA
        ctx.fillStyle = VISIT_FILL
        const rect = boxRect(flower.box, sx, sy)
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h)
        ctx.restore()
      }
      ctx.strokeStyle = flower.source === 'stop' ? '#aaaaaa' : '#ffd200'
      strokeBox(ctx, flower.box, sx, sy)
    }
  }

  if (opts.trail) {
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.lineWidth = Math.max(1.25, width / 500)
    tracks.forEach((tr, i) => {
      const points = usablePoints(tr.points)
      const from = firstIndexAtOrAfter(points, time - TRAIL_S)
      const to = lastIndexAtOrBefore(points, time)
      if (to < from) return
      ctx.strokeStyle = INSECT_COLORS[i % INSECT_COLORS.length]
      for (let k = from; k < to; k++) {
        const a = points[k]
        const b = points[k + 1]
        if (b.t - a.t > GAP_S) continue
        const age = (time - b.t) / TRAIL_S
        ctx.globalAlpha = Math.max(0.28, 1 - age * 0.72)
        const ca = centerOf(a)
        const cb = centerOf(b)
        ctx.beginPath()
        ctx.moveTo(ca[0] * sx, ca[1] * sy)
        ctx.lineTo(cb[0] * sx, cb[1] * sy)
        ctx.stroke()
      }
    })
    ctx.globalAlpha = 1
  }

  if (opts.boxes) {
    ctx.globalAlpha = 1
    ctx.lineJoin = 'miter'
    ctx.lineWidth = Math.max(1.5, width / 420)
    const captions: Array<{ text: string; box: Box; color: string }> = []
    tracks.forEach((tr, i) => {
      const points = usablePoints(tr.points)
      const idx = lastIndexAtOrBefore(points, time)
      if (idx < 0) return
      const cur = points[idx]
      if (time - cur.t > GAP_S) return
      const color = INSECT_COLORS[i % INSECT_COLORS.length]
      ctx.strokeStyle = color
      strokeBox(ctx, cur.box, sx, sy)
      const text = insectCaption(tr, cur)
      if (text) captions.push({ text, box: cur.box, color })
    })
    for (const caption of captions) {
      const rect = boxRect(caption.box, sx, sy)
      drawLabel(ctx, caption.text, rect.x, rect.y + rect.h + labelGap, caption.color, 'top')
    }
    for (const flower of doc.flowers ?? []) {
      if (!isBox(flower.box) || !flower.flower_id || !visitedFlowers.has(flower.flower_id)) continue
      const rect = boxRect(flower.box, sx, sy)
      const color = flower.source === 'stop' ? '#aaaaaa' : '#ffd200'
      drawLabel(ctx, flower.flower_id, rect.x, rect.y - labelGap, color, 'bottom')
    }
  }
}
