"""Random field -> RL bee policy -> MuJoCo 3D replay. No clip, no API calls.

    python -m sim.mj_random                 # seed 0
    python -m sim.mj_random --seed 7 --flowers 8 --budget 30 --speedup 2 --camera chase
    python -m sim.mj_random --seed 7 --live         # interactive 3D window

Flowers come from sim.field_env.random_layout (the same generator the policy was trained on), so the
policy has never seen this exact field. Writes rollouts/random_<seed>_rl.json (rollout format) and
rollouts/random_<seed>_mujoco.mp4.
"""
import argparse, json, math
from pathlib import Path

import numpy as np

from sim.field_env import FieldEnv, random_layout
from sim.rollout import POLICY_PATH, make_actor, run_episode

PETALS = 5


def grass_png(path, seed, n=1024):
    import cv2
    """Procedural grass: blurred green noise with darker blades, so the ground isn't a flat colour."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 24)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    blades = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 1.2)
    k = 0.65 * base + 0.35 * blades
    img = np.stack([40 + 30 * k, 95 + 70 * k, 35 + 25 * k], axis=-1).astype(np.uint8)  # BGR
    cv2.imwrite(str(path), img)


def flower_xml(f, S):
    """Stem, petals, yellow centre, and a hidden ring that turns on when the bee finishes a visit."""
    x1, y1, x2, y2 = f["box"]
    cx, cy = ((x1 + x2) / 2 / 1000 - 0.5) * S, (0.5 - (y1 + y2) / 2 / 1000) * S
    r = max(x2 - x1, y2 - y1) / 2 / 1000 * S
    h = 0.02                                   # flower head height above ground
    g = [f'<geom type="cylinder" size="{r*0.08:.4f} {h/2:.4f}" pos="{cx:.4f} {cy:.4f} {h/2:.4f}" rgba="0.2 0.5 0.15 1"/>']
    for i in range(PETALS):
        a = 2 * math.pi * i / PETALS
        g.append(f'<geom type="ellipsoid" size="{r*0.5:.4f} {r*0.28:.4f} {r*0.06:.4f}" '
                 f'pos="{cx + r*0.5*math.cos(a):.4f} {cy + r*0.5*math.sin(a):.4f} {h:.4f}" '
                 f'euler="0 0 {math.degrees(a):.1f}" rgba="0.97 0.97 0.95 1"/>')
    g.append(f'<geom type="sphere" size="{r*0.3:.4f}" pos="{cx:.4f} {cy:.4f} {h + r*0.05:.4f}" rgba="0.98 0.8 0.1 1"/>')
    g.append(f'<geom name="ring_{f["flower_id"]}" type="cylinder" size="{r*1.25:.4f} 0.0008" '
             f'pos="{cx:.4f} {cy:.4f} 0.001" rgba="0.2 0.9 0.3 0"/>')
    return "\n    ".join(g)


def build_xml(flowers, S, grass, shadow=4096):
    return f"""<mujoco model="beebot-random">
  <include file="cf2_beebot.xml"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="{shadow}"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.4 0.4 0.4"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.75 0.95" rgb2="0.95 0.97 1" width="512" height="3072"/>
    <texture name="grass" type="2d" file="{Path(grass).resolve().as_posix()}"/>
    <material name="grass" texture="grass" texrepeat="1 1"/>
  </asset>
  <worldbody>
    <light pos="{S*0.3:.3f} {-S*0.3:.3f} {S*2:.3f}" dir="-0.2 0.2 -1" directional="true" castshadow="true" diffuse="0.7 0.7 0.65"/>
    <geom type="plane" size="{S/2:.4f} {S/2:.4f} 0.01" material="grass"/>
    {"".join(flower_xml(f, S) + chr(10) + "    " for f in flowers)}
    <camera name="overview" pos="0 {-0.95*S:.3f} {0.85*S:.3f}" xyaxes="1 0 0 0 0.67 0.74"/>
  </worldbody>
</mujoco>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flowers", type=int, default=None, help="exact count (default: random 4-10, as in training)")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds the bee gets")
    ap.add_argument("--policy", default=str(POLICY_PATH), help="PPO zip, or 'baseline' (nearest unvisited)")
    ap.add_argument("--menagerie", default="../mujoco_menagerie")
    ap.add_argument("--size-m", type=float, default=0.42, help="field side in metres (0.42 = real flower scale)")
    ap.add_argument("--drone-scale", type=float, default=0.3)
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--camera", choices=["orbit", "chase", "fixed"], default="orbit")
    ap.add_argument("--live", action="store_true", help="open an interactive 3D window instead of writing an mp4")
    ap.add_argument("--no-video", action="store_true", help="only write the rollout JSON (the web Simulations page draws it)")
    ap.add_argument("--reuse", action="store_true", help="reuse rollouts/random_<seed>_rl.json if made with the same settings")
    ap.add_argument("--size", default="1280x720", help="video size WxH (max 1280x720)")
    ap.add_argument("--shadow", type=int, default=4096, help="shadow map size; 1024 renders faster")
    a = ap.parse_args()

    name = f"random_{a.seed}"
    out_dir = Path("rollouts"); out_dir.mkdir(exist_ok=True)
    js = out_dir / f"{name}_rl.json"
    settings = {"flowers": a.flowers, "budget": a.budget, "policy": a.policy}
    rl = json.loads(js.read_text()) if a.reuse and js.exists() else None
    if rl is None or rl.get("settings") != settings:   # run the policy unless this exact field is already done
        rng = np.random.default_rng(a.seed)
        n = a.flowers
        flowers = random_layout(rng, n or 4, n or 10)
        start = rng.uniform(50, 950, size=2)
        env = FieldEnv(flowers=flowers, budget_s=a.budget, start=start, shaping=False)
        way, visits = run_episode(env, make_actor(a.policy))
        rl = {"clip_id": name, "planner": "rl-ppo" if a.policy != "baseline" else "nearest_unvisited",
              "policy": a.policy, "flowers_from": f"random_layout(seed={a.seed})", "flowers": flowers,
              "units": "norm1000", "budget_s": a.budget, "dt": env.dt, "start": [way[0]["x"], way[0]["y"]],
              "speed": env.speed, "dwell_s": env.dwell_s, "visits": visits, "n_visits": len(visits),
              "n_open": int(env.open.sum()), "waypoints": way, "settings": settings}
        js.write_text(json.dumps(rl, indent=1))
    flowers, visits = rl["flowers"], rl["visits"]
    print(f"{name}: {len(flowers)} flowers, visited {len(visits)} in {a.budget:g}s")
    if a.no_video:
        return

    # MuJoCo/OpenCV load only after the policy has run: importing them before torch can crash torch's import
    global mujoco, pose_quat
    import cv2, imageio.v2 as imageio, mujoco
    from sim.mj_render import FPS, pose_quat, rl_trajectory, scaled_cf2
    S = a.size_m
    cf2_dir = Path(a.menagerie) / "bitcraze_crazyflie_2"
    scaled_cf2(cf2_dir, a.drone_scale)
    grass = out_dir / f"{name}_grass.png"
    grass_png(grass, a.seed)
    scene = cf2_dir / "beebot_random.xml"
    scene.write_text(build_xml(flowers, S, grass, a.shadow))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ring = {f["flower_id"]: model.geom(f"ring_{f['flower_id']}").id for f in flowers}
    done_at = {v["flower_id"]: v["t"] for v in visits}

    rl["_W"] = rl["_H"] = S
    traj = rl_trajectory(rl)
    centers = {f["flower_id"]: (((f["box"][0] + f["box"][2]) / 2 / 1000 - 0.5) * S,
                                (0.5 - (f["box"][1] + f["box"][3]) / 2 / 1000) * S) for f in flowers}
    fx = Effects(model, ring, centers, visits, S)
    if a.live:
        play_live(model, data, traj, fx, a.speedup)
        return
    step = max(int(round(a.speedup)), 1)
    out = out_dir / f"{name}_mujoco.mp4"
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", macro_block_size=8)
    cam = mujoco.MjvCamera()
    vw, vh = (min(int(v), lim) for v, lim in zip(a.size.lower().split("x"), (1280, 720)))
    with mujoco.Renderer(model, vh, vw) as r:
        for t, x, y, z, vel, tilt in traj[::step]:
            fx.pose(data, t, x, y, z, vel, tilt)
            mujoco.mj_forward(model, data)
            if a.camera == "fixed":
                r.update_scene(data, camera="overview")
            else:
                fx.aim(cam, t, (x, y, z), a.camera)
                r.update_scene(data, camera=cam)
            fx.decorate(r.scene, t)
            img = hud(np.ascontiguousarray(r.render()), t, sum(1 for v in visits if v["t"] <= t), len(flowers),
                      step, a.seed)
            writer.append_data(img)
    writer.close()
    print("wrote", out)



class Effects:
    """Everything drawn on top of the model each frame: visit rings, a fading trail, pollen bursts."""

    def __init__(self, model, ring, centers, visits, S):
        self.model, self.ring, self.centers, self.S = model, ring, centers, S
        self.done_at = {v["flower_id"]: v["t"] for v in visits}
        self.trail, self.yaw = [], 0.0
        rng = np.random.default_rng(0)
        self.sparks = rng.normal(size=(14, 3)) * [1, 1, 0.4] + [0, 0, 1.2]   # burst directions

    def pose(self, data, t, x, y, z, vel, tilt):
        for fid, gid in self.ring.items():
            self.model.geom_rgba[gid, 3] = 0.85 if self.done_at.get(fid, 1e9) <= t else 0.0
        if vel is not None:
            self.yaw += math.remainder(math.atan2(vel[1], vel[0]) - self.yaw, 2 * math.pi) * 0.15
        data.qpos[:3] = (x, y, z)
        data.qpos[3:7] = pose_quat(self.yaw, tilt)
        if not self.trail or np.linalg.norm(np.subtract(self.trail[-1], (x, y, z))) > self.S * 0.006:
            self.trail = (self.trail + [(x, y, z)])[-120:]

    def aim(self, cam, t, pos, mode):
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if mode == "chase":                       # follows the drone from behind and above
            cam.lookat[:] = pos
            cam.distance, cam.elevation = self.S * 0.45, -28
            cam.azimuth = math.degrees(self.yaw) + 180 + 15 * math.sin(t * 0.3)
        else:                                     # slow orbit around the whole field
            cam.lookat[:] = (0, 0, 0)
            cam.distance, cam.elevation = self.S * 1.55, -38
            cam.azimuth = 90 + 35 * math.sin(t * 0.12)

    def decorate(self, scn, t):
        n = len(self.trail)
        for i in range(1, n):                     # orange trail, older segments fade out
            add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [1.0, 0.55, 0.1, 0.9 * i / n], self.S * 0.0025,
                     self.trail[i - 1], self.trail[i])
        for fid, tv in self.done_at.items():      # pollen burst for 0.8 s after each visit
            age = t - tv
            if 0 <= age < 0.8:
                cx, cy = self.centers[fid]
                for d in self.sparks:
                    p = np.array([cx, cy, 0.025]) + d * self.S * 0.04 * age
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [1.0, 0.85, 0.2, 1 - age / 0.8],
                             self.S * 0.003, p)


def add_geom(scn, kind, rgba, size, a, b=None):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array([size, size, size]), np.asarray(a, float), np.eye(3).flatten(),
                        np.array(rgba, dtype=np.float32))
    if b is not None:
        mujoco.mjv_connector(g, kind, size, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def hud(img, t, n_done, n_total, step, seed):
    """Dark bar along the top: what this is, time, visits. Scales with the frame width."""
    import cv2
    k = img.shape[1] / 1280
    hb, base = int(56 * k), int(36 * k)
    img[:hb] = (img[:hb].astype(np.float32) * 0.45).astype(np.uint8)
    speed = "real time" if step == 1 else f"{step}x speed"
    cv2.putText(img, "RL bee policy  |  field it has never seen", (int(18 * k), base), cv2.FONT_HERSHEY_SIMPLEX,
                0.75 * k, (255, 255, 255), 2, cv2.LINE_AA)
    right = f"t {t:4.1f}s   visits {n_done}/{n_total}   seed {seed}   {speed}"
    (w, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, 2)
    cv2.putText(img, right, (img.shape[1] - w - int(18 * k), base), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k,
                (255, 210, 120), 2, cv2.LINE_AA)
    return img


def play_live(model, data, traj, fx, speedup):
    """Interactive MuJoCo window: drag to rotate, right-drag to pan, scroll to zoom. Loops until closed."""
    import time
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = (0, 0, 0)
        v.cam.distance, v.cam.elevation, v.cam.azimuth = fx.S * 1.55, -38, 90
        while v.is_running():
            fx.trail = []
            t0 = time.perf_counter()
            for t, x, y, z, vel, tilt in traj:
                if not v.is_running():
                    break
                with v.lock():
                    fx.pose(data, t, x, y, z, vel, tilt)
                    mujoco.mj_forward(model, data)
                    v.user_scn.ngeom = 0
                    fx.decorate(v.user_scn, t)
                v.sync()
                wait = t / speedup - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
