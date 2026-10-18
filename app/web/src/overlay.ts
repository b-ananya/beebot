/** One video frame at 30fps, plus a little slack. Farther apart is a gap: do not join or hold across it. */
export const GAP_S = 0.05
export const TRAIL_S = 2

export type Box = [number, number, number, number]

export type TrackPoint = {
  t: number
  box: Box
  center?: [number, number]
  interpolated?: boolean
}

export type InsectTrack = {
  track_id?: string
  points?: TrackPoint[]
}

export type Flower = {
  flower_id?: string
  box?: Box
  source?: string
}

export type TrackFile = {
  tracks?: InsectTrack[]
  flowers?: Flower[]
}

const INSECT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']

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

function strokeBox(ctx: CanvasRenderingContext2D, box: Box, sx: number, sy: number): void {
  const x1 = Math.min(box[0], box[2]) * sx
  const y1 = Math.min(box[1], box[3]) * sy
  const x2 = Math.max(box[0], box[2]) * sx
  const y2 = Math.max(box[1], box[3]) * sy
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
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

  if (opts.boxes) {
    ctx.lineJoin = 'miter'
    ctx.lineWidth = Math.max(1, width / 640)
    for (const flower of doc.flowers ?? []) {
      if (!isBox(flower.box)) continue
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
    ctx.lineJoin = 'miter'
    ctx.lineWidth = Math.max(1.5, width / 420)
    tracks.forEach((tr, i) => {
      const points = usablePoints(tr.points)
      const idx = lastIndexAtOrBefore(points, time)
      if (idx < 0) return
      const cur = points[idx]
      if (time - cur.t > GAP_S) return
      ctx.strokeStyle = INSECT_COLORS[i % INSECT_COLORS.length]
      strokeBox(ctx, cur.box, sx, sy)
    })
  }
}
