/** Simulations page: a random flower field, the RL bee policy's flight over it, in interactive 3D.
 *
 *  Python does the work (sim.mj_random: random_layout + PPO policy, via /api/sim/run); this page redraws the
 *  same flowers and path with three.js so it can be rotated, zoomed and scrubbed. The MuJoCo mp4 of the same
 *  field is one button away (/api/sim/video). Coordinates match sim/mj_render.py: the 0-1000 grid maps to a
 *  square field of FIELD_M metres, the drone is a Crazyflie 2 from MuJoCo Menagerie scaled down. */
import { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'

type Flower = { flower_id: string; box: [number, number, number, number] }
type Waypoint = { t: number; x: number; y: number; flower_id: string | null }
type Visit = { flower_id: string; t: number }
type Rollout = {
  clip_id: string
  flowers: Flower[]
  waypoints: Waypoint[]
  visits: Visit[]
  budget_s: number
  dwell_s: number
  n_open: number
}

const FPS = 30
const FIELD_M = 0.42 // field side in metres, as in sim/mj_render.py (flower radius ~ 1.25 cm)
const S = 10 // field side in scene units
const M = S / FIELD_M // scene units per metre
const CRUISE = 0.06 * M
const LAND = 0.03 * M
const HEAD = 0.02 * M // flower head height
const DRONE_SCALE = 0.3 // 1 = real Crazyflie (9 cm across)
const TRAIL = 150 // frames of trail kept behind the drone
const SPARKS = 14

// Crazyflie 2 visual meshes and their materials, from Menagerie cf2.xml
const CF2_PARTS: Array<[string, number]> = [
  ['cf2_0.obj', 0xcad1ee], // propeller_plastic
  ['cf2_1.obj', 0x1c2f00], // medium_gloss_plastic
  ['cf2_2.obj', 0xf7e099], // polished_gold
  ['cf2_3.obj', 0xa1a8ad], // polished_plastic
  ['cf2_4.obj', 0xe5e5e5], // burnished_chrome
  ['cf2_5.obj', 0x1a1a1a], // body_frame_plastic
  ['cf2_6.obj', 0xffffff], // white
]

// ---------------------------------------------------------------------------------------------- track

type Track = { t: Float32Array; pos: THREE.Vector3[]; quat: THREE.Quaternion[]; end: number }

function interp(t: number, ts: number[], vs: number[]): number {
  if (t <= ts[0]) return vs[0]
  const n = ts.length - 1
  if (t >= ts[n]) return vs[n]
  let lo = 0
  let hi = n
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (ts[mid] <= t) lo = mid
    else hi = mid
  }
  const u = (t - ts[lo]) / (ts[hi] - ts[lo] || 1)
  return vs[lo] + (vs[hi] - vs[lo]) * u
}

function gaussian(a: number[], sigma: number): number[] {
  const r = Math.ceil(sigma * 3)
  const k: number[] = []
  for (let i = -r; i <= r; i++) k.push(Math.exp(-(i * i) / (2 * sigma * sigma)))
  const sum = k.reduce((p, q) => p + q, 0)
  return a.map((_, i) => {
    let acc = 0
    for (let j = -r; j <= r; j++) acc += a[Math.min(a.length - 1, Math.max(0, i + j))] * k[j + r]
    return acc / sum
  })
}

const toX = (x: number) => (x / 1000 - 0.5) * S
const toZ = (y: number) => (y / 1000 - 0.5) * S

/** Same smoothing as sim/mj_render.py rl_trajectory: rounded corners, eased landings, gradual turns, lean. */
function buildTrack(r: Rollout): Track {
  const wt = r.waypoints.map((w) => w.t)
  const wx = r.waypoints.map((w) => toX(w.x))
  const wz = r.waypoints.map((w) => toZ(w.y))
  const won = r.waypoints.map((w) => (w.flower_id ? 1 : 0))
  const end = wt[wt.length - 1]
  const n = Math.max(2, Math.floor(end * FPS) + 1)
  const ts = Array.from({ length: n }, (_, i) => i / FPS)
  const xs = gaussian(ts.map((t) => interp(t, wt, wx)), 0.12 * FPS)
  const zs = gaussian(ts.map((t) => interp(t, wt, wz)), 0.12 * FPS)
  const k = gaussian(ts.map((t) => interp(t, wt, won)), 0.25 * FPS)
  const vx = xs.map((_, i) => (xs[Math.min(n - 1, i + 1)] - xs[Math.max(0, i - 1)]) * FPS / 2)
  const vz = zs.map((_, i) => (zs[Math.min(n - 1, i + 1)] - zs[Math.max(0, i - 1)]) * FPS / 2)
  const ax = gaussian(vx.map((_, i) => (vx[Math.min(n - 1, i + 1)] - vx[Math.max(0, i - 1)]) * FPS / 2), 0.1 * FPS)
  const az = gaussian(vz.map((_, i) => (vz[Math.min(n - 1, i + 1)] - vz[Math.max(0, i - 1)]) * FPS / 2), 0.1 * FPS)
  const pos: THREE.Vector3[] = []
  const quat: THREE.Quaternion[] = []
  const up = new THREE.Vector3(0, 1, 0)
  let yaw = 0
  for (let i = 0; i < n; i++) {
    const t = ts[i]
    const y = CRUISE - (CRUISE - LAND - 0.005 * M) * k[i] + 0.002 * M * Math.sin(9 * t) * k[i]
    pos.push(new THREE.Vector3(xs[i], y, zs[i]))
    if (Math.hypot(vx[i], vz[i]) > 0.01 * M) {
      const target = Math.atan2(-vz[i], vx[i]) // MuJoCo yaw: world y is -z here
      let d = target - yaw
      d = Math.atan2(Math.sin(d), Math.cos(d))
      yaw += d * 0.15
    }
    const q = new THREE.Quaternion().setFromAxisAngle(up, yaw)
    const am = Math.hypot(ax[i], az[i]) / M // m/s^2
    if (am > 1e-4) {
      const axis = new THREE.Vector3(az[i], 0, -ax[i]).normalize() // lean toward the acceleration
      q.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, Math.min(am * 0.8, 0.3)))
    }
    quat.push(q)
  }
  return { t: Float32Array.from(ts), pos, quat, end }
}

// ---------------------------------------------------------------------------------------------- scene

function grassTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas')
  c.width = c.height = 1024
  const g = c.getContext('2d')!
  g.fillStyle = '#4c9a3c'
  g.fillRect(0, 0, 1024, 1024)
  let seed = 7
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  for (let i = 0; i < 260; i++) {
    // soft light and dark patches
    const x = rnd() * 1024
    const y = rnd() * 1024
    const r = 40 + rnd() * 140
    const grad = g.createRadialGradient(x, y, 0, x, y, r)
    const light = rnd() > 0.5
    grad.addColorStop(0, light ? 'rgba(120,190,80,0.35)' : 'rgba(30,80,30,0.35)')
    grad.addColorStop(1, 'rgba(0,0,0,0)')
    g.fillStyle = grad
    g.fillRect(x - r, y - r, 2 * r, 2 * r)
  }
  for (let i = 0; i < 26000; i++) {
    // blades
    const x = rnd() * 1024
    const y = rnd() * 1024
    const l = 3 + rnd() * 7
    const a = -Math.PI / 2 + (rnd() - 0.5) * 0.9
    g.strokeStyle = `hsla(${95 + rnd() * 30}, ${45 + rnd() * 25}%, ${22 + rnd() * 28}%, 0.55)`
    g.lineWidth = 1
    g.beginPath()
    g.moveTo(x, y)
    g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l)
    g.stroke()
  }
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  return tex
}

function skyTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas')
  c.width = 2
  c.height = 256
  const g = c.getContext('2d')!
  const grad = g.createLinearGradient(0, 0, 0, 256)
  grad.addColorStop(0, '#6fa8e8')
  grad.addColorStop(0.55, '#b9d7f3')
  grad.addColorStop(1, '#eef5fc')
  g.fillStyle = grad
  g.fillRect(0, 0, 2, 256)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

function fallbackDrone(): THREE.Group {
  const g = new THREE.Group()
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.008, 0.03), new THREE.MeshStandardMaterial({ color: 0x222222 }))
  g.add(body)
  for (const [x, z] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.017, 0.002, 24),
      new THREE.MeshStandardMaterial({ color: 0xcad1ee, transparent: true, opacity: 0.8 }))
    arm.position.set(x * 0.032, 0.006, z * 0.032)
    g.add(arm)
  }
  return g
}

type Flowers = { group: THREE.Group; rings: Map<string, THREE.Mesh>; centers: Map<string, THREE.Vector3> }

function buildFlowers(flowers: Flower[]): Flowers {
  const group = new THREE.Group()
  const rings = new Map<string, THREE.Mesh>()
  const centers = new Map<string, THREE.Vector3>()
  const stemMat = new THREE.MeshStandardMaterial({ color: 0x2f7a22, roughness: 0.8 })
  const petalMat = new THREE.MeshStandardMaterial({ color: 0xf8f8f2, roughness: 0.55 })
  const coreMat = new THREE.MeshStandardMaterial({ color: 0xf6c51a, roughness: 0.5, emissive: 0x3a2a00 })
  for (const f of flowers) {
    const [x1, y1, x2, y2] = f.box
    const cx = toX((x1 + x2) / 2)
    const cz = toZ((y1 + y2) / 2)
    const r = (Math.max(x2 - x1, y2 - y1) / 2 / 1000) * S
    centers.set(f.flower_id, new THREE.Vector3(cx, HEAD, cz))
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.07, r * 0.09, HEAD, 8), stemMat)
    stem.position.set(cx, HEAD / 2, cz)
    stem.castShadow = true
    group.add(stem)
    for (let i = 0; i < 5; i++) {
      const a = (2 * Math.PI * i) / 5
      const petal = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), petalMat)
      petal.scale.set(r * 0.5, r * 0.07, r * 0.27)
      petal.position.set(cx + Math.cos(a) * r * 0.5, HEAD, cz - Math.sin(a) * r * 0.5)
      petal.rotation.y = a
      petal.castShadow = true
      group.add(petal)
    }
    const core = new THREE.Mesh(new THREE.SphereGeometry(r * 0.3, 16, 12), coreMat)
    core.position.set(cx, HEAD + r * 0.05, cz)
    core.castShadow = true
    group.add(core)
    const ring = new THREE.Mesh(new THREE.CircleGeometry(r * 1.3, 40),
      new THREE.MeshBasicMaterial({ color: 0x4dff6a, transparent: true, opacity: 0, depthWrite: false }))
    ring.rotation.x = -Math.PI / 2
    ring.position.set(cx, 0.01, cz)
    group.add(ring)
    rings.set(f.flower_id, ring)
  }
  return { group, rings, centers }
}

class World {
  renderer: THREE.WebGLRenderer
  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(42, 1, 0.05, 400)
  controls: OrbitControls
  drone = new THREE.Group()
  flowers: Flowers | null = null
  trail: Line2
  trailGeo = new LineGeometry()
  trailMat: LineMaterial
  pollen: THREE.InstancedMesh
  sparkDirs: THREE.Vector3[] = []
  rollout: Rollout | null = null
  track: Track | null = null
  follow = false
  meshNote = ''
  private dummy = new THREE.Object3D()

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.scene.background = skyTexture()
    this.scene.fog = new THREE.Fog(0xcfe3f6, 30, 90)

    this.camera.position.set(0, 8.5, 11)
    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.maxPolarAngle = Math.PI / 2.08
    this.controls.minDistance = 0.6
    this.controls.maxDistance = 40
    this.controls.autoRotateSpeed = 0.6

    this.scene.add(new THREE.HemisphereLight(0xdcecff, 0x3b5a2a, 1.1))
    const sun = new THREE.DirectionalLight(0xfff4e0, 2.2)
    sun.position.set(6, 14, 4)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    const sc = sun.shadow.camera
    sc.left = sc.bottom = -S * 0.75
    sc.right = sc.top = S * 0.75
    sc.near = 1
    sc.far = 40
    sun.shadow.bias = -0.0005
    sun.shadow.normalBias = 0.02
    this.scene.add(sun)

    const field = new THREE.Mesh(new THREE.PlaneGeometry(S, S),
      new THREE.MeshStandardMaterial({ map: grassTexture(), roughness: 0.95 }))
    field.rotation.x = -Math.PI / 2
    field.receiveShadow = true
    this.scene.add(field)
    const meadow = new THREE.Mesh(new THREE.CircleGeometry(80, 64),
      new THREE.MeshStandardMaterial({ color: 0x3f7f33, roughness: 1 }))
    meadow.rotation.x = -Math.PI / 2
    meadow.position.y = -0.02
    meadow.receiveShadow = true
    this.scene.add(meadow)
    const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S, 0.001, S)),
      new THREE.LineBasicMaterial({ color: 0xe8f5c8, transparent: true, opacity: 0.35 }))
    border.position.y = 0.005
    this.scene.add(border)

    this.trailMat = new LineMaterial({ linewidth: 3, vertexColors: true, worldUnits: false })
    this.trail = new Line2(this.trailGeo, this.trailMat)
    this.trail.visible = false
    this.scene.add(this.trail)

    this.pollen = new THREE.InstancedMesh(new THREE.SphereGeometry(0.035, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xffd23a }), 20 * SPARKS)
    this.pollen.frustumCulled = false
    this.scene.add(this.pollen)
    let seed = 3
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2
    for (let i = 0; i < SPARKS; i++) this.sparkDirs.push(new THREE.Vector3(rnd(), 1.2 + rnd() * 0.4, rnd()))

    this.scene.add(this.drone)
    void this.loadDrone()
  }

  async loadDrone(): Promise<void> {
    const loader = new OBJLoader()
    const inner = new THREE.Group()
    inner.rotation.x = -Math.PI / 2 // MuJoCo is z-up, three.js is y-up
    try {
      for (const [file, color] of CF2_PARTS) {
        const obj = await loader.loadAsync(`/api/sim/cf2/${file}`)
        const mat = new THREE.MeshStandardMaterial({ color, metalness: color === 0xf7e099 ? 0.6 : 0.1, roughness: 0.45 })
        obj.traverse((o) => {
          if (o instanceof THREE.Mesh) {
            o.material = mat
            o.castShadow = true
          }
        })
        inner.add(obj)
      }
    } catch {
      inner.clear()
      inner.rotation.x = 0
      inner.add(fallbackDrone())
      this.meshNote = 'Crazyflie meshes not found (clone MuJoCo Menagerie next to the repo); showing a simple drone.'
    }
    inner.scale.setScalar(M * DRONE_SCALE)
    this.drone.add(inner)
  }

  setRollout(r: Rollout): void {
    if (this.flowers) {
      this.scene.remove(this.flowers.group)
      this.flowers.group.traverse((o) => {
        if (o instanceof THREE.Mesh) o.geometry.dispose()
      })
    }
    this.rollout = r
    this.track = buildTrack(r)
    this.flowers = buildFlowers(r.flowers)
    this.scene.add(this.flowers.group)
  }

  resize(w: number, h: number): void {
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / Math.max(h, 1)
    this.camera.updateProjectionMatrix()
    this.trailMat.resolution.set(w, h)
  }

  /** Pose everything for time t (seconds) and draw. */
  draw(t: number): void {
    const tr = this.track
    const r = this.rollout
    if (tr && r) {
      const f = Math.min(tr.pos.length - 1, Math.max(0, t * FPS))
      const i = Math.floor(f)
      const j = Math.min(tr.pos.length - 1, i + 1)
      const u = f - i
      this.drone.position.lerpVectors(tr.pos[i], tr.pos[j], u)
      this.drone.quaternion.slerpQuaternions(tr.quat[i], tr.quat[j], u)

      const from = Math.max(0, i - TRAIL)
      const pts: number[] = []
      const cols: number[] = []
      const orange = new THREE.Color(0xff8c1a)
      const pale = new THREE.Color(0xfff1d6)
      for (let k = from; k <= i; k += 2) {
        const p = tr.pos[k]
        pts.push(p.x, p.y, p.z)
        const c = pale.clone().lerp(orange, (k - from) / Math.max(1, i - from))
        cols.push(c.r, c.g, c.b)
      }
      pts.push(this.drone.position.x, this.drone.position.y, this.drone.position.z)
      cols.push(orange.r, orange.g, orange.b)
      if (pts.length >= 6) {
        this.trailGeo.dispose()
        this.trailGeo = new LineGeometry()
        this.trailGeo.setPositions(pts)
        this.trailGeo.setColors(cols)
        this.trail.geometry = this.trailGeo
        this.trail.computeLineDistances()
        this.trail.visible = true
      } else {
        this.trail.visible = false
      }

      let n = 0
      for (const v of r.visits) {
        const ring = this.flowers?.rings.get(v.flower_id)
        const done = v.t <= t
        if (ring) (ring.material as THREE.MeshBasicMaterial).opacity = done ? 0.55 : 0
        const age = t - v.t
        const c = this.flowers?.centers.get(v.flower_id)
        if (c && age >= 0 && age < 0.9) {
          for (const d of this.sparkDirs) {
            if (n >= this.pollen.count) break
            this.dummy.position.copy(c).addScaledVector(d, age * 0.9)
            this.dummy.scale.setScalar(1 - age / 0.9)
            this.dummy.updateMatrix()
            this.pollen.setMatrixAt(n++, this.dummy.matrix)
          }
        }
      }
      this.dummy.scale.setScalar(0)
      this.dummy.updateMatrix()
      for (let k = n; k < this.pollen.count; k++) this.pollen.setMatrixAt(k, this.dummy.matrix)
      this.pollen.instanceMatrix.needsUpdate = true

      if (this.follow) this.controls.target.lerp(this.drone.position, 0.12)
    }
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  dispose(): void {
    this.controls.dispose()
    this.renderer.dispose()
  }
}

// ---------------------------------------------------------------------------------------------- page

const SPEEDS = [0.5, 1, 2, 4]

/** Spinner text plus seconds elapsed, so a long render visibly keeps going. */
function BusyOverlay({ text }: { text: string }) {
  const [secs, setSecs] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const id = window.setInterval(() => setSecs(Math.floor((performance.now() - start) / 1000)), 500)
    return () => window.clearInterval(id)
  }, [text])
  return (
    <div className="sim-busy">
      {text} {secs}s
    </div>
  )
}

export function SimulatePage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewRef = useRef<HTMLDivElement>(null)
  const worldRef = useRef<World | null>(null)
  const timeRef = useRef(0)
  const playingRef = useRef(true)
  const speedRef = useRef(1)

  const [seedText, setSeedText] = useState('3')
  const seed = Math.min(1_000_000, Math.max(0, Math.round(Number(seedText)) || 0)) // clamp only when used
  const lastQuery = useRef('') // settings of the field on screen, so its video matches it
  const [flowers, setFlowers] = useState('auto')
  const [budgetText, setBudgetText] = useState('30')
  const budget = Math.min(120, Math.max(5, Math.round(Number(budgetText)) || 30)) // clamp only when used
  const [rollout, setRollout] = useState<Rollout | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [orbit, setOrbit] = useState(false)
  const [follow, setFollow] = useState(false)
  const [video, setVideo] = useState<string | null>(null)
  const [meshNote, setMeshNote] = useState('')

  useEffect(() => {
    const canvas = canvasRef.current
    const view = viewRef.current
    if (!canvas || !view) return
    const world = new World(canvas)
    worldRef.current = world
    const fit = () => world.resize(view.clientWidth, view.clientHeight)
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(view)
    let raf = 0
    let last = performance.now()
    let lastUi = 0
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      const end = world.track?.end ?? 0
      if (playingRef.current && world.track) {
        timeRef.current += dt * speedRef.current
        if (timeRef.current >= end) {
          timeRef.current = end
          playingRef.current = false
          setPlaying(false)
        }
      }
      world.draw(timeRef.current)
      if (now - lastUi > 80) {
        lastUi = now
        setTime(timeRef.current)
        if (world.meshNote) setMeshNote(world.meshNote)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      world.dispose()
      worldRef.current = null
    }
  }, [])

  const query = useCallback(
    (s: number) => `seed=${s}&budget=${budget}${flowers === 'auto' ? '' : `&flowers=${flowers}`}`,
    [budget, flowers],
  )

  const generate = useCallback(
    async (s: number) => {
      setBusy('Running the RL bee policy on a new field…')
      setError(null)
      setVideo(null)
      try {
        const q = query(s)
        const res = await fetch(`/api/sim/run?${q}`)
        const body = await res.json()
        if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`)
        const r = body as Rollout
        worldRef.current?.setRollout(r)
        setRollout(r)
        lastQuery.current = q
        timeRef.current = 0
        playingRef.current = true
        setPlaying(true)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(null)
      }
    },
    [query],
  )


  const renderVideo = async () => {
    setBusy('Rendering the MuJoCo video (1–2 min)…')
    setError(null)
    try {
      const res = await fetch(`/api/sim/video?${lastQuery.current}&camera=orbit`)
      const body = await res.json()
      if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`)
      setVideo(body.video)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const togglePlay = () => {
    const end = worldRef.current?.track?.end ?? 0
    if (!playingRef.current && timeRef.current >= end) timeRef.current = 0
    playingRef.current = !playingRef.current
    setPlaying(playingRef.current)
  }

  const end = rollout ? worldRef.current?.track?.end ?? rollout.budget_s : 0
  const visitsSoFar = rollout ? rollout.visits.filter((v) => v.t <= time).length : 0

  return (
    <div className="sim">
      <header className="sim-top">
        <a href="/">← Pollensight</a>
        <span className="title">Simulations</span>
        <span style={{ opacity: 0.6 }}>RL bee policy on a random field it has never seen</span>
      </header>
      <div className="sim-row">
        <aside className="sim-side">
          <h3>Field</h3>
          <label className="sim-field">
            Seed
            <span className="sim-inline">
              <input
                type="number"
                min={0}
                value={seedText}
                onChange={(e) => setSeedText(e.target.value)}
                onBlur={() => setSeedText(String(seed))}
              />
              <button type="button" className="sim-btn icon" title="Random seed" onClick={() => setSeedText(String(Math.floor(Math.random() * 10000)))}>
                🎲
              </button>
            </span>
          </label>
          <label className="sim-field">
            Flowers
            <select value={flowers} onChange={(e) => setFlowers(e.target.value)}>
              <option value="auto">Random 4–10 (as in training)</option>
              {[4, 5, 6, 7, 8, 9, 10, 12, 15].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="sim-field">
            Time budget (s)
            <input
              type="number"
              min={5}
              max={120}
              value={budgetText}
              onChange={(e) => setBudgetText(e.target.value)}
              onBlur={() => setBudgetText(String(budget))}
            />
          </label>
          <button type="button" className="sim-btn primary" disabled={!!busy} onClick={() => void generate(seed)}>
            Generate field
          </button>

          <h3>View</h3>
          <div className="sim-inline">
            <button type="button" className={orbit ? 'sim-btn on' : 'sim-btn'} onClick={() => {
              const w = worldRef.current
              if (w) w.controls.autoRotate = !orbit
              setOrbit(!orbit)
            }}>
              Orbit
            </button>
            <button type="button" className={follow ? 'sim-btn on' : 'sim-btn'} onClick={() => {
              const w = worldRef.current
              if (w) {
                w.follow = !follow
                if (follow) w.controls.target.set(0, 0, 0)
              }
              setFollow(!follow)
            }}>
              Follow bee
            </button>
          </div>

          {rollout && (
            <>
              <h3>Run</h3>
              <div className="sim-stats">
                <span>Flowers</span>
                <span>{rollout.n_open}</span>
                <span>Visited so far</span>
                <span>{visitsSoFar}</span>
                <span>Visited by the end</span>
                <span>{rollout.visits.length}</span>
                <span>Dwell per flower</span>
                <span>{rollout.dwell_s.toFixed(2)} s</span>
                <span>Budget</span>
                <span>{rollout.budget_s} s</span>
                {end < rollout.budget_s - 0.05 && (
                  <>
                    <span>Finished early</span>
                    <span>{end.toFixed(1)} s</span>
                  </>
                )}
              </div>
              <button type="button" className="sim-btn" disabled={!!busy} onClick={() => void renderVideo()}>
                Render MuJoCo video of this field
              </button>
              {video && <video className="sim-video" src={video} controls autoPlay muted loop />}
            </>
          )}
          <p className="sim-note">
            The policy (sim/policy.zip, PPO) decides every 0.1 s from its position, the time left and the 6 nearest
            unvisited flowers. Flight is smoothed for display; visits and timing are the policy's own. Not flight
            physics.
          </p>
          {meshNote && <p className="sim-note">{meshNote}</p>}
          {error && <div className="sim-error">{error}</div>}
        </aside>
        <main className="sim-main">
          <div className="sim-view" ref={viewRef}>
            <canvas ref={canvasRef} />
            {rollout && (
              <div className="sim-hud">
                t {time.toFixed(1)} s · visits {visitsSoFar}/{rollout.n_open} · seed {rollout.clip_id.replace('random_', '')}
              </div>
            )}
            <div className="sim-hint">Drag to rotate · right-drag to pan · scroll to zoom</div>
            {!rollout && !busy && (
              <div className="sim-busy">Pick a seed and settings, then click Generate field.</div>
            )}
            {busy && <BusyOverlay text={busy} />}
          </div>
          <div className="sim-timeline">
            <button type="button" className="sim-btn" onClick={togglePlay} disabled={!rollout} style={{ width: 70 }}>
              {playing ? 'Pause' : 'Play'}
            </button>
            <input
              type="range"
              min={0}
              max={end || 1}
              step={1 / FPS}
              value={Math.min(time, end || 1)}
              disabled={!rollout}
              onChange={(e) => {
                timeRef.current = Number(e.target.value)
                setTime(timeRef.current)
              }}
            />
            <span className="clock">
              {time.toFixed(1)} / {end.toFixed(1)} s
            </span>
            <select value={speed} onChange={(e) => {
              speedRef.current = Number(e.target.value)
              setSpeed(speedRef.current)
            }}>
              {SPEEDS.map((s) => (
                <option key={s} value={s}>
                  {s}×
                </option>
              ))}
            </select>
          </div>
        </main>
      </div>
    </div>
  )
}
