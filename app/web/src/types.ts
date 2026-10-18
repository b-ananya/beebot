import type { Flower, TrackPoint } from './overlay'

/** Where the open clip's track document is loaded from. */
export type Source = 'pipeline' | 'heuristic' | 'truth' | 'mock'

export type ManifestEntry = {
  clip_id: string
  path: string
  fps: number
  width: number
  height: number
  duration_s: number
  source: string
  has_truth: boolean
  truth_kind: string
}

export type VisitEvent = {
  track_id?: string
  flower_id?: string
  type?: string
  t_start?: number
  t_end?: number
}

export type ClipTrack = {
  track_id?: string
  species?: string
  complete?: boolean
  points?: TrackPoint[]
}

export type TrackDoc = {
  clip_id?: string
  model?: string
  prompt_version?: string
  tracks?: ClipTrack[]
  flowers?: Flower[]
  events?: VisitEvent[]
}

export type PanelProps = {
  clipId: string
  source: Source
  doc: TrackDoc | null
  manifestEntry: ManifestEntry | null
  time: number
  onSeek: (seconds: number) => void
  onOpenClip: (clipId: string, seconds: number) => void
}

export function trackUrl(source: Source, clipId: string): string {
  switch (source) {
    case 'pipeline':
      return `/tracks/${clipId}.json`
    case 'heuristic':
      return `/tracks/heuristic/${clipId}.json`
    case 'truth':
      return `/truth/${clipId}.json`
    case 'mock':
      return `/tracks/mock/${clipId}.json`
  }
}
