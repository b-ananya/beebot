/** Simulations page API: random field -> RL bee rollout (sim.mj_random), Crazyflie meshes, MuJoCo video.
 *
 *  GET /api/sim/run?seed=3&flowers=8&budget=30      -> rollout JSON (flowers + waypoints + visits)
 *  GET /api/sim/video?seed=3&flowers=8&budget=30&camera=orbit -> { video: "/rollouts/random_3_mujoco.mp4" }
 *  GET /api/sim/cf2/<file>.obj                      -> mesh from MuJoCo Menagerie bitcraze_crazyflie_2/assets
 *
 *  Menagerie is looked up at MUJOCO_MENAGERIE, else next to the repo (../mujoco_menagerie). */
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import type { ServerResponse } from 'node:http'
import path from 'node:path'
import type { Connect } from 'vite'
import { readDotenv } from './apis.ts'

let repo = ''

function python(): string {
  // like apis.ts: the environment first, then the repo-root .env (where a venv outside the repo is configured)
  const configured = (process.env.BEEBOT_PYTHON ?? readDotenv().BEEBOT_PYTHON ?? '').trim()
  if (configured) return configured
  for (const venv of [path.join(repo, '.venv', 'bin', 'python'), path.join(repo, '.venv', 'Scripts', 'python.exe')]) {
    if (fs.existsSync(venv)) return venv
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

function menagerie(): string {
  return (
    process.env.MUJOCO_MENAGERIE?.trim() ||
    readDotenv().MUJOCO_MENAGERIE?.trim() ||
    path.resolve(repo, '..', 'mujoco_menagerie')
  )
}

function send(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.setHeader('Cache-Control', 'no-store')
  res.end(JSON.stringify(body))
}

function intParam(q: URLSearchParams, key: string, lo: number, hi: number): number | null {
  const raw = q.get(key)
  if (raw === null || raw === '') return null
  const n = Number(raw)
  if (!Number.isInteger(n) || n < lo || n > hi) throw new Error(`${key} must be a whole number from ${lo} to ${hi}`)
  return n
}

/** Run python -m sim.mj_random with these args from the repo root; resolve when it exits cleanly. */
function runSim(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(python(), ['-m', 'sim.mj_random', ...args], { cwd: repo, windowsHide: true })
    let err = ''
    child.stderr.on('data', (d: Buffer) => {
      err = (err + d.toString()).slice(-4000)
    })
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(err.trim() || `python exited ${code}`))))
  })
}

function simArgs(q: URLSearchParams): { seed: number; args: string[] } {
  const seed = intParam(q, 'seed', 0, 1_000_000) ?? 0
  const flowers = intParam(q, 'flowers', 1, 20)
  const budget = intParam(q, 'budget', 5, 120) ?? 30
  const args = ['--seed', String(seed), '--budget', String(budget)]
  if (flowers !== null) args.push('--flowers', String(flowers))
  return { seed, args }
}

async function handle(res: ServerResponse, url: URL): Promise<void> {
  const q = url.searchParams
  if (url.pathname === '/api/sim/run') {
    const { seed, args } = simArgs(q)
    await runSim([...args, '--no-video'])
    const file = path.join(repo, 'rollouts', `random_${seed}_rl.json`)
    send(res, 200, JSON.parse(fs.readFileSync(file, 'utf8')))
    return
  }
  if (url.pathname === '/api/sim/video') {
    const { seed, args } = simArgs(q)
    const camera = q.get('camera') ?? 'orbit'
    if (!['orbit', 'chase', 'fixed'].includes(camera)) throw new Error('camera must be orbit, chase or fixed')
    // web preview: reuse the path the page already has, 2x speed, 1024x576, lighter shadows (about 4x faster)
    await runSim([...args, '--reuse', '--camera', camera, '--speedup', '2', '--size', '1024x576', '--shadow', '2048',
                  '--menagerie', menagerie()])
    send(res, 200, { video: `/rollouts/random_${seed}_mujoco.mp4?v=${Date.now()}` })
    return
  }
  const mesh = /^\/api\/sim\/cf2\/(cf2_\d+\.obj)$/.exec(url.pathname)
  if (mesh) {
    const file = path.join(menagerie(), 'bitcraze_crazyflie_2', 'assets', mesh[1])
    if (!fs.existsSync(file)) {
      send(res, 404, { error: `Crazyflie mesh not found at ${file}. Clone MuJoCo Menagerie next to the repo.` })
      return
    }
    res.statusCode = 200
    res.setHeader('Content-Type', 'text/plain; charset=utf-8')
    res.setHeader('Cache-Control', 'max-age=3600')
    fs.createReadStream(file).pipe(res)
    return
  }
  send(res, 404, { error: 'Not found.' })
}

export function attachSimApi(middlewares: Connect.Server, root: string): void {
  repo = root
  middlewares.use((req, res, next) => {
    const url = new URL(req.url ?? '/', 'http://localhost')
    if (!url.pathname.startsWith('/api/sim/')) {
      next()
      return
    }
    if (req.method !== 'GET') {
      send(res, 405, { error: 'Method not allowed.' })
      return
    }
    handle(res, url).catch((err: unknown) => {
      send(res, 500, { error: err instanceof Error ? err.message : 'Simulation failed.' })
    })
  })
}
