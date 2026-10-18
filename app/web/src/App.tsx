import { useEffect, useRef, useState } from 'react'

const CLIP_ID = 'hb6_w00'
const TOP_H = 28
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
  const minutes = Math.floor(seconds / 60)
  const whole = Math.floor(seconds % 60)
  const hundredths = Math.floor((seconds - Math.floor(seconds)) * 100)
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

export function App() {
  const [leftW, setLeftW] = useState(260)
  const [rightW, setRightW] = useState(280)
  const [bottomH, setBottomH] = useState(140)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)

  const shellRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
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
    }
    const onMeta = () => {
      setDuration(Number.isFinite(video.duration) ? video.duration : 0)
      setTime(video.currentTime)
    }

    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('ended', onPause)
    video.addEventListener('loadedmetadata', onMeta)
    video.addEventListener('durationchange', onMeta)
    video.addEventListener('timeupdate', onMeta)
    return () => {
      stopTick()
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('ended', onPause)
      video.removeEventListener('loadedmetadata', onMeta)
      video.removeEventListener('durationchange', onMeta)
      video.removeEventListener('timeupdate', onMeta)
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
      if (target instanceof HTMLElement && (target.tagName === 'BUTTON' || target.tagName === 'INPUT')) return
      event.preventDefault()
      const video = videoRef.current
      if (!video) return
      if (video.paused) void video.play()
      else video.pause()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) void video.play().catch(() => setPlaying(false))
    else video.pause()
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

  return (
    <div className="shell" ref={shellRef}>
      <header className="topbar">
        <span className="brand">BeeBot</span>
        <span className="clip-id">{CLIP_ID}</span>
      </header>
      <div className="row">
        <section className="panel outliner" style={{ width: leftW }} aria-label="Outliner">
          <div className="region-header">Outliner</div>
          <div className="region-body" />
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
          </div>
          <div className="stage" ref={stageRef}>
            <video
              ref={videoRef}
              src={`/clips/${CLIP_ID}.mp4`}
              preload="auto"
              playsInline
              aria-label={CLIP_ID}
            />
            <canvas ref={canvasRef} aria-hidden="true" />
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
          <div className="region-body" />
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
        <div className="region-body" />
      </section>
    </div>
  )
}
