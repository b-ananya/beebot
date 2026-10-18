import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Connect } from 'vite'

const MISSING_MODELS = 'Forecast models not trained. Run python -m forecast.train.'
const KEY_ERROR = 'PERCEPTRON_API_KEY is not set. Add it to .env.'
const VIDEO_ERROR = 'Upload an mp4, mov, avi, or mkv video.'
const JSON_LIMIT = 1_000_000
const UPLOAD_LIMIT = 512 * 1024 * 1024
const VIDEO_EXT = new Set(['.mp4', '.mov', '.avi', '.mkv'])
const SPECIES = new Set(['unknown', 'honeybee', 'vespidae', 'syrphidae', 'lepidoptera'])

/** Frontend source ids, plus the track directories the forecast route loads. */
const SOURCE_DIRS: Record<string, string> = {
  tracks: 'tracks',
  'tracks/heuristic': 'tracks/heuristic',
  truth: 'truth',
  'tracks/mock': 'tracks/mock',
  'tracks/uploads': 'tracks/uploads',
  pipeline: 'tracks',
  heuristic: 'tracks/heuristic',
  mock: 'tracks/mock',
}

const PY = `
import json, math, os, shutil, sys
from pathlib import Path

ROOT = Path(os.environ["BEEBOT_ROOT"])
MISSING = "Forecast models not trained. Run python -m forecast.train."
SPECIES = {"unknown", "honeybee", "vespidae", "syrphidae", "lepidoptera"}

def inside(root, file):
    root_s = os.path.abspath(root)
    file_s = os.path.abspath(file)
    try:
        return os.path.commonpath([root_s, file_s]) == root_s
    except ValueError:
        return False

def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    mod = type(value).__module__
    if mod == "numpy" or mod.startswith("numpy."):
        import numpy as np
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return to_jsonable(value.item())
    raise TypeError("Cannot encode " + type(value).__name__)

def reply(ok, status=200, error=None, data=None):
    body = {"ok": bool(ok), "status": int(status)}
    if error is not None:
        body["error"] = str(error)
    if data is not None:
        body["data"] = to_jsonable(data)
    sys.stdout.write(json.dumps(body, allow_nan=False))
    sys.stdout.write("\\n")
    sys.stdout.flush()

def do_search(req):
    video = Path(req["video"])
    if not video.is_file() or not inside(ROOT, video):
        reply(False, 501, error="No video for clip " + req["clipId"] + ".")
        return
    from search.search import live
    reply(True, 200, data=live(req["clipId"], req["q"], video))

def do_forecast(req):
    allowed = {"tracks", "tracks/heuristic", "truth", "tracks/mock", "tracks/uploads"}
    rel = req["dir"]
    if rel not in allowed:
        reply(False, 400, error="Unknown track source.")
        return
    models = ROOT / "forecast" / "models"
    if not (models / "path.joblib").is_file() or not (models / "flower.joblib").is_file():
        reply(False, 501, error=MISSING)
        return
    doc_path = ROOT / rel / (req["clipId"] + ".json")
    if not inside(ROOT, doc_path) or not doc_path.is_file():
        reply(False, 501, error="No track file for " + req["clipId"] + ".")
        return
    doc = json.loads(doc_path.read_text())
    from forecast.predict import forecast
    try:
        out = forecast(doc, req["trackId"], float(req["t"]), horizon_s=float(req["horizonS"]))
    except StopIteration:
        reply(False, 501, error="No track " + req["trackId"] + ".")
        return
    except FileNotFoundError:
        reply(False, 501, error=MISSING)
        return
    reply(True, 200, data=out)

def probe_duration(mp4):
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            return float(frames) / float(fps)
    except Exception:
        return None
    return None

def do_analyze(req):
    use_api = bool(req.get("useApi"))
    if use_api and not os.environ.get("PERCEPTRON_API_KEY", "").strip():
        reply(False, 400, error="PERCEPTRON_API_KEY is not set. Add it to .env.")
        return
    species = req.get("species") or "unknown"
    if species not in SPECIES:
        reply(False, 400, error="Unknown species.")
        return
    video = Path(req["video"])
    if not video.is_file():
        reply(False, 400, error="Upload a video file.")
        return
    from perception.pipeline import analyze_video
    from analytics.metrics import clip_metrics
    clip_id, mp4, _doc = analyze_video(video, use_api=use_api, species=species)
    written = ROOT / "tracks" / "uploads" / (clip_id + ".json")
    if not written.is_file():
        reply(False, 501, error="Pipeline did not write a track file.")
        return
    doc = json.loads(written.read_text())
    summary = clip_metrics(doc, probe_duration(mp4))["summary"]
    clips = ROOT / "clips"
    playable = Path(mp4)
    if not playable.is_file() or not inside(clips, playable):
        playable = clips / "uploads" / (clip_id + ".mp4")
        playable.parent.mkdir(parents=True, exist_ok=True)
        if not playable.is_file():
            shutil.copyfile(video, playable)
    rel = os.path.relpath(playable, clips).replace(os.sep, "/")
    video_url = "/clips/" + rel
    overlay = clips / "uploads" / (clip_id + "_overlay.mp4")
    try:
        from app.render import render
        render(playable, doc, overlay)
        if overlay.is_file() and overlay.stat().st_size > 0:
            video_url = "/clips/uploads/" + clip_id + "_overlay.mp4"
    except Exception:
        pass
    reply(True, 200, data={"clipId": clip_id, "summary": summary, "videoUrl": video_url})

def main():
    req = json.load(sys.stdin)
    op = req.get("op")
    if op == "search":
        do_search(req)
    elif op == "forecast":
        do_forecast(req)
    elif op == "analyze":
        do_analyze(req)
    else:
        reply(False, 501, error="Unknown operation.")

try:
    main()
except Exception as exc:
    msg = str(exc).strip() or exc.__class__.__name__
    if "path.joblib" in msg or "flower.joblib" in msg:
        msg = MISSING
    else:
        msg = msg.splitlines()[-1][:500]
    sys.stdout.write(json.dumps({"ok": False, "status": 501, "error": msg}))
    sys.stdout.write("\\n")
`

type PyResult = {
  ok?: boolean
  status?: number
  error?: string
  data?: unknown
}

type UploadFile = {
  filename: string
  data: Buffer
}

type HttpError = Error & { status: number }

let repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const successCache = new Map<string, unknown>()
const inflight = new Map<string, Promise<unknown>>()

function httpError(status: number, error: string): HttpError {
  return Object.assign(new Error(error), { status })
}

function statusOf(err: unknown): number {
  if (typeof err === 'object' && err && 'status' in err && typeof err.status === 'number') return err.status
  return 501
}

function resolveRepo(explicit?: string): string {
  const seeds = [explicit, process.cwd(), path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')]
  for (const seed of seeds) {
    if (!seed) continue
    let dir = path.resolve(seed)
    for (let i = 0; i < 5; i++) {
      if (fs.existsSync(path.join(dir, 'clips', 'manifest.json'))) return dir
      const parent = path.dirname(dir)
      if (parent === dir) break
      dir = parent
    }
  }
  return explicit ? path.resolve(explicit) : path.resolve(process.cwd(), '../..')
}

function readDotenv(): Record<string, string> {
  const file = path.join(repoRoot, '.env')
  const out: Record<string, string> = {}
  if (!fs.existsSync(file)) return out
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const eq = trimmed.indexOf('=')
    if (eq <= 0) continue
    const key = trimmed.slice(0, eq).trim()
    let value = trimmed.slice(eq + 1).trim()
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    }
    out[key] = value
  }
  return out
}

function apiKey(): string {
  if (process.env.PERCEPTRON_API_KEY !== undefined) return process.env.PERCEPTRON_API_KEY.trim()
  return (readDotenv().PERCEPTRON_API_KEY ?? '').trim()
}

function pythonBin(): string {
  // The project venv: .venv/bin on macOS and Linux, .venv/Scripts on Windows. Without one, fall back to the
  // interpreter on PATH ("python3" does not exist on Windows outside the Store stub, so try "python" there).
  for (const candidate of [
    path.join(repoRoot, '.venv', 'bin', 'python'),
    path.join(repoRoot, '.venv', 'Scripts', 'python.exe'),
  ]) {
    if (fs.existsSync(candidate)) return candidate
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

function modelsTrained(): boolean {
  const dir = path.join(repoRoot, 'forecast', 'models')
  return fs.existsSync(path.join(dir, 'path.joblib')) && fs.existsSync(path.join(dir, 'flower.joblib'))
}

function safeId(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const id = value.trim()
  if (!id || id.length > 200) return null
  if (id.includes('/') || id.includes('\\') || id.includes('..') || id.includes('\0')) return null
  return id
}

function redact(text: string): string {
  const key = apiKey()
  return key && text.includes(key) ? text.split(key).join('[redacted]') : text
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  if (res.destroyed || res.writableEnded) return
  if (res.headersSent) {
    res.end()
    return
  }
  const raw = JSON.stringify(body)
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.setHeader('Cache-Control', 'no-store')
  res.end(raw)
}

function pathnameOf(req: IncomingMessage): string {
  const raw = req.url ?? '/'
  const query = raw.indexOf('?')
  return query === -1 ? raw : raw.slice(0, query)
}

function readBody(req: IncomingMessage, limit: number): Promise<Buffer> {
  const declared = Number(req.headers['content-length'] ?? 0)
  if (Number.isFinite(declared) && declared > limit) return Promise.reject(httpError(400, 'Upload is too large.'))
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    let settled = false
    const done = (fn: () => void) => {
      if (settled) return
      settled = true
      fn()
    }
    req.on('data', (chunk: Buffer) => {
      if (settled) return
      size += chunk.length
      if (size > limit) {
        req.destroy()
        done(() => reject(httpError(400, 'Upload is too large.')))
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => done(() => resolve(Buffer.concat(chunks))))
    req.on('error', (err) => done(() => reject(err)))
    req.on('aborted', () => done(() => reject(httpError(400, 'Upload aborted.'))))
  })
}

function headerParam(headers: string, key: 'name' | 'filename'): string | undefined {
  if (key === 'filename') {
    const star = /filename\*=(?:UTF-8''|utf-8'')([^;\r\n]+)/i.exec(headers)
    if (star) {
      try {
        return decodeURIComponent(star[1])
      } catch {
        return star[1]
      }
    }
  }
  const quoted = new RegExp(`${key}="([^"]*)"`, 'i').exec(headers)
  if (quoted) return quoted[1]
  const bare = new RegExp(`${key}=([^;\\s]+)`, 'i').exec(headers)
  return bare?.[1]
}

function parseMultipart(body: Buffer, contentType: string): { fields: Record<string, string>; file: UploadFile | null } {
  const match = /boundary=(?:"([^"]+)"|([^;\s]+))/i.exec(contentType)
  if (!match) throw httpError(400, 'Expected multipart form data.')
  const boundary = (match[1] ?? match[2]).trim()
  if (!boundary) throw httpError(400, 'Expected multipart form data.')
  const delim = Buffer.from(`--${boundary}`)
  const fields: Record<string, string> = {}
  let file: UploadFile | null = null
  let cursor = body.indexOf(delim)
  if (cursor < 0) throw httpError(400, 'Malformed upload.')

  while (cursor >= 0 && cursor < body.length) {
    const after = cursor + delim.length
    if (body[after] === 45 && body[after + 1] === 45) break
    let start = after
    if (body[start] === 13 && body[start + 1] === 10) start += 2
    else if (body[start] === 10) start += 1
    const next = body.indexOf(delim, start)
    if (next < 0) break
    let end = next
    if (end >= 2 && body[end - 2] === 13 && body[end - 1] === 10) end -= 2
    else if (end >= 1 && body[end - 1] === 10) end -= 1
    const part = body.subarray(start, end)
    const crlf = part.indexOf('\r\n\r\n')
    const lf = crlf >= 0 ? -1 : part.indexOf('\n\n')
    const splitAt = crlf >= 0 ? crlf : lf
    const sepLen = crlf >= 0 ? 4 : 2
    if (splitAt >= 0) {
      const headerText = part.subarray(0, splitAt).toString('utf8')
      const data = part.subarray(splitAt + sepLen)
      const name = headerParam(headerText, 'name')
      if (name) {
        const filename = headerParam(headerText, 'filename')
        if (filename !== undefined && name === 'video') file = { filename, data }
        else if (filename === undefined) fields[name] = data.toString('utf8')
      }
    }
    cursor = next
  }
  return { fields, file }
}

function looksLikeVideo(filename: string, data: Buffer): boolean {
  const ext = path.extname(filename).toLowerCase()
  if (!VIDEO_EXT.has(ext) || data.length < 12) return false
  if (ext === '.avi') {
    return data.subarray(0, 4).toString('latin1') === 'RIFF' && data.subarray(8, 12).toString('latin1') === 'AVI '
  }
  if (ext === '.mkv') return data[0] === 0x1a && data[1] === 0x45 && data[2] === 0xdf && data[3] === 0xa3
  return data.subarray(0, Math.min(data.length, 64)).toString('latin1').includes('ftyp')
}

function videoForClip(clipId: string): string | null {
  const manifestPath = path.join(repoRoot, 'clips', 'manifest.json')
  if (!fs.existsSync(manifestPath)) return null
  let manifest: unknown
  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as unknown
  } catch {
    return null
  }
  if (!Array.isArray(manifest)) return null
  const entry = manifest.find((row) => {
    return !!row && typeof row === 'object' && (row as { clip_id?: unknown }).clip_id === clipId
  }) as { path?: unknown } | undefined
  if (!entry || typeof entry.path !== 'string' || !entry.path) return null
  const file = path.resolve(repoRoot, entry.path)
  const rel = path.relative(repoRoot, file)
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) return null
  return fs.existsSync(file) ? file : null
}

function bodyKey(pathname: string, raw: Buffer): string {
  return createHash('sha256').update(pathname).update('\0').update(raw).digest('hex')
}

function withCache(key: string, produce: () => Promise<unknown>): Promise<unknown> {
  const hit = successCache.get(key)
  if (hit !== undefined) return Promise.resolve(hit)
  const pending = inflight.get(key)
  if (pending) return pending
  const job = produce()
    .then((data) => {
      successCache.set(key, data)
      return data
    })
    .finally(() => {
      inflight.delete(key)
    })
  inflight.set(key, job)
  return job
}

function parsePyJson(text: string): PyResult | null {
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const value = JSON.parse(lines[i]) as unknown
      if (value && typeof value === 'object' && !Array.isArray(value) && 'ok' in value) return value as PyResult
    } catch {
      continue
    }
  }
  return null
}

function runPy(payload: unknown, timeoutMs: number): Promise<PyResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin(), ['-c', PY], {
      cwd: repoRoot,
      env: childEnv(),
    })
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      fn()
    }
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      finish(() => reject(httpError(501, 'Python timed out.')))
    }, timeoutMs)
    child.stdout.on('data', (chunk: Buffer) => stdout.push(chunk))
    child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk))
    child.on('error', (err: NodeJS.ErrnoException) => {
      const message = err.code === 'ENOENT' ? 'Python is not available.' : err.message
      finish(() => reject(httpError(501, message)))
    })
    child.on('close', () => {
      finish(() => {
        const parsed = parsePyJson(Buffer.concat(stdout).toString('utf8'))
        if (parsed) {
          resolve(parsed)
          return
        }
        const tail = redact(Buffer.concat(stderr).toString('utf8').trim()).split('\n').slice(-12).join('\n').slice(-1500)
        resolve({ ok: false, status: 501, error: tail || 'Python failed.' })
      })
    })
    child.stdin.on('error', () => {})
    child.stdin.end(JSON.stringify(payload))
  })
}

function childEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    BEEBOT_ROOT: repoRoot,
    PYTHONUNBUFFERED: '1',
    PYTHONWARNINGS: 'ignore',
    PYTHONIOENCODING: 'utf-8',
    LOKY_MAX_CPU_COUNT: process.env.LOKY_MAX_CPU_COUNT || '4',
  }
  const key = apiKey()
  if (key) env.PERCEPTRON_API_KEY = key
  env.PYTHONPATH = env.PYTHONPATH ? `${repoRoot}${path.delimiter}${env.PYTHONPATH}` : repoRoot
  return env
}

async function pyData(payload: unknown, timeoutMs: number): Promise<unknown> {
  const result = await runPy(payload, timeoutMs)
  if (!result.ok) throw httpError(result.status === 400 ? 400 : 501, result.error || 'Request failed.')
  return result.data
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

async function postSearch(raw: Buffer): Promise<unknown> {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw.toString('utf8')) as unknown
  } catch {
    throw httpError(400, 'Expected JSON.')
  }
  const body = asRecord(parsed)
  const clipId = safeId(body?.clipId)
  if (!body || typeof body.q !== 'string' || !clipId) throw httpError(400, 'Expected clipId and q.')
  const q = body.q.trim()
  if (!q) return []
  const video = videoForClip(clipId)
  if (!video) throw httpError(501, `No video for clip ${clipId}.`)
  const data = await pyData({ op: 'search', clipId, q, video }, 180_000)
  if (!Array.isArray(data)) throw httpError(501, 'Live search did not return a list.')
  return data
}

async function postForecast(raw: Buffer): Promise<unknown> {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw.toString('utf8')) as unknown
  } catch {
    throw httpError(400, 'Expected JSON.')
  }
  const body = asRecord(parsed)
  const clipId = safeId(body?.clipId)
  const trackId = safeId(body?.trackId)
  const source = typeof body?.source === 'string' ? SOURCE_DIRS[body.source] : undefined
  const t = typeof body?.t === 'number' ? body.t : Number.NaN
  const horizonS = typeof body?.horizonS === 'number' ? body.horizonS : Number.NaN
  if (!body || !clipId || !trackId || !source) throw httpError(400, 'Expected clipId, source, and trackId.')
  if (!Number.isFinite(t) || !Number.isFinite(horizonS) || horizonS <= 0 || horizonS > 120) {
    throw httpError(400, 'Expected a finite time and a horizon up to 120 seconds.')
  }
  if (!modelsTrained()) throw httpError(501, MISSING_MODELS)
  return pyData({ op: 'forecast', dir: source, clipId, trackId, t, horizonS }, 120_000)
}

async function postAnalyze(req: IncomingMessage, raw: Buffer): Promise<unknown> {
  const contentType = req.headers['content-type'] ?? ''
  if (!contentType.toLowerCase().includes('multipart/form-data')) throw httpError(400, 'Expected multipart form data.')
  const form = parseMultipart(raw, contentType)
  const useApi = form.fields.useApi ?? '0'
  if (useApi !== '0' && useApi !== '1') throw httpError(400, 'useApi must be 0 or 1.')
  if (useApi === '1' && !apiKey()) throw httpError(400, KEY_ERROR)
  const species = form.fields.species ?? 'unknown'
  if (!SPECIES.has(species)) throw httpError(400, 'Unknown species.')
  const file = form.file
  if (!file || !file.filename || file.data.length === 0 || !looksLikeVideo(file.filename, file.data)) {
    throw httpError(400, file ? VIDEO_ERROR : 'Choose a video file.')
  }
  const ext = path.extname(file.filename).toLowerCase()
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'beebot-'))
  const dest = path.join(dir, `upload${ext}`)
  fs.writeFileSync(dest, file.data)
  try {
    return await pyData({ op: 'analyze', video: dest, useApi: useApi === '1', species }, 10 * 60_000)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
}

async function dispatch(req: IncomingMessage, res: ServerResponse, pathname: string): Promise<void> {
  if (pathname === '/api/health' && req.method === 'GET') {
    sendJson(res, 200, { api: apiKey().length > 0 })
    return
  }
  if (req.method !== 'POST') throw httpError(405, 'Method not allowed.')
  const limit = pathname === '/api/analyze' ? UPLOAD_LIMIT : JSON_LIMIT
  const raw = await readBody(req, limit)
  const key = bodyKey(pathname, raw)
  if (pathname === '/api/search/live') {
    sendJson(res, 200, await withCache(key, () => postSearch(raw)))
    return
  }
  if (pathname === '/api/forecast') {
    sendJson(res, 200, await withCache(key, () => postForecast(raw)))
    return
  }
  if (pathname === '/api/analyze') {
    sendJson(res, 200, await withCache(key, () => postAnalyze(req, raw)))
    return
  }
  throw httpError(404, 'Not found.')
}

export function attachApis(middlewares: Connect.Server, root?: string): void {
  repoRoot = resolveRepo(root)
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const pathname = pathnameOf(req)
    if (!pathname.startsWith('/api/')) {
      next()
      return
    }
    void dispatch(req, res, pathname).catch((err: unknown) => {
      const message = err instanceof Error && err.message ? err.message : 'API failed.'
      sendJson(res, statusOf(err), { error: message })
    })
  }
  middlewares.use(handler)
}
