"""MuJoCo replay of a BeeBot plan: Crazyflie 2 flies plan route over the clip frame (kinematic, no physics)."""
import argparse, json, math, os, re
from pathlib import Path
import cv2, imageio.v2 as imageio, mujoco, numpy as np
from scipy.interpolate import CubicSpline

FPS, CRUISE, LAND = 30, 0.06, 0.03   # metres above ground

def deficit_rgba(d, dmax):
    k = 0 if dmax == 0 else d / dmax
    return f"{k:.2f} {1-k:.2f} 0.1 1"

def build_xml(cf2_dir, tex_png, field, W, H):
    to_w = lambda c: ((c[0]/1000-0.5)*W, (0.5-c[1]/1000)*H)
    dmax = max((f["deficit"] for f in field["flowers"]), default=0)
    flowers = "\n".join(
        f'<geom type="cylinder" size="{max(f["radius"]/1000*H,0.01):.4f} 0.002" pos="{to_w(f["center"])[0]:.4f} {to_w(f["center"])[1]:.4f} 0.002" '
        f'rgba="{deficit_rgba(f["deficit"], dmax)}" contype="0" conaffinity="0"/>'
        for f in field["flowers"] if f.get("state", "open") == "open")
    return f"""<mujoco model="beebot">
  <include file="cf2_beebot.xml"/>
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="clip" type="2d" file="{Path(tex_png).resolve().as_posix()}"/>
    <material name="clip" texture="clip" texrepeat="1 1" texuniform="false"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="{W/2} {H/2} 0.01" material="clip" contype="0" conaffinity="0"/>
    {flowers}
    <camera name="overview" pos="0 {-0.8*H:.3f} {1.3*H:.3f}" xyaxes="1 0 0 0 0.85 0.53"/>
  </worldbody>
</mujoco>"""

def scaled_cf2(cf2_dir, scale):
    """Copy of cf2.xml with every mesh scaled, so a small pollinator robot can stand in for the Crazyflie."""
    xml = (cf2_dir / "cf2.xml").read_text()
    xml = re.sub(r'<mesh file="([^"]+)"/>', rf'<mesh file="\1" scale="{scale} {scale} {scale}"/>', xml)
    # written to a private name, then swapped in, so a render running at the same time never reads a half file
    tmp = cf2_dir / f"cf2_beebot.{os.getpid()}.tmp"
    tmp.write_text(xml)
    os.replace(tmp, cf2_dir / "cf2_beebot.xml")


def load_scene(cf2_dir, xml, stem):
    """Compile a scene that must sit next to cf2.xml (so mesh paths resolve) under a per-process name, and remove
    the file afterwards: two renders at once (two browser tabs) must not load each other's flowers."""
    import mujoco

    scene = cf2_dir / f"{stem}_{os.getpid()}.xml"
    scene.write_text(xml)
    try:
        return mujoco.MjModel.from_xml_path(str(scene))
    finally:
        scene.unlink(missing_ok=True)

def pose_quat(yaw, tilt=None):
    """Quaternion for heading yaw plus a lean toward world-frame acceleration tilt=(ax, ay) radians."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.0 if tilt is None else -tilt[1], 0.0 if tilt is None else tilt[0], yaw]), "xyz")
    return q

def _smooth(a, sigma):
    """Gaussian smoothing along time (sigma in frames), edges held."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")

def rl_trajectory(rl):
    """(t, x, y, z, vel, tilt) at FPS from an RL rollout, smoothed so flight looks continuous.

    Waypoints come every dt; flower_id set means the bee is on a flower. Position, height and heading
    are smoothed (display only: the visits and timing stay the policy's own)."""
    W_, H_ = rl["_W"], rl["_H"]
    wp = rl["waypoints"]
    t = np.array([p["t"] for p in wp])
    P = np.array([((p["x"]/1000-0.5)*W_, (0.5-p["y"]/1000)*H_) for p in wp])
    on = np.array([1.0 if p["flower_id"] else 0.0 for p in wp])
    ts = np.arange(0, t[-1], 1/FPS)
    xy = np.stack([np.interp(ts, t, P[:, 0]), np.interp(ts, t, P[:, 1])], axis=1)
    xy = _smooth(xy, 0.12 * FPS)                          # round off corners
    k = _smooth(np.interp(ts, t, on), 0.25 * FPS)         # ease down onto / up off a flower
    z = CRUISE - (CRUISE - LAND - 0.005) * k + 0.002 * np.sin(9 * ts) * k
    vel = np.gradient(xy, axis=0) * FPS
    acc = _smooth(np.gradient(vel, axis=0) * FPS, 0.1 * FPS)
    out = []
    for i, s in enumerate(ts):
        v = vel[i] if np.hypot(*vel[i]) > 0.01 else None
        tilt = np.clip(acc[i] * 0.8, -0.3, 0.3)        # lean into accelerations (radians, world x/y)
        out.append((s, xy[i, 0], xy[i, 1], z[i], v, tilt))
    return out

def trajectory(field, plan, W, H, aspect):
    """(t, xyz) samples: spline through route stops at plan speed, dwell + hover wobble at each flower."""
    by_id = {f["flower_id"]: f for f in field["flowers"]}
    stops = [plan["start"]] + [by_id[f]["center"] for f in plan["route"]]
    speed_m = plan["robot"]["speed"] * H / 1000          # y-norm units/s -> m/s
    dwell = plan["robot"]["dwell_s"]
    P = np.array([((x/1000-0.5)*W, (0.5-y/1000)*H) for x, y in stops])
    t_knots, t = [0.0], 0.0
    for a, b in zip(P, P[1:]):
        t += np.linalg.norm(b - a) / speed_m; t_knots.append(t)
    spl = CubicSpline(t_knots, P, bc_type="clamped")       # smooth path through every flower
    out, clock = [], 0.0
    for i in range(1, len(P)):
        t0, t1 = t_knots[i-1], t_knots[i]
        for s in np.arange(t0, t1, 1/FPS):
            u = (s - t0) / (t1 - t0)
            z = LAND + (CRUISE - LAND) * math.sin(math.pi * u)   # climb off one flower, drop onto the next
            xy = spl(s); out.append((clock, xy[0], xy[1], z, spl(s, 1), None)); clock += 1/FPS
        for s in np.arange(0, dwell, 1/FPS):                 # hover wobble on the flower
            w = 0.004*np.array([math.sin(7*s), math.cos(5*s), math.sin(11*s)])
            out.append((clock, P[i][0]+w[0], P[i][1]+w[1], LAND+0.005+w[2], None, None)); clock += 1/FPS
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id"); ap.add_argument("--menagerie", default="../mujoco_menagerie")
    ap.add_argument("--frame", help="png to use as ground (default: frame 0 of clips/<clip>.mp4)")
    ap.add_argument("--height-m", type=float, default=0.42, help="world height H of the camera view, metres")
    ap.add_argument("--path", choices=["plan", "rl"], default="plan",
                    help="plan: planner route in plan/<clip>.json; rl: PPO rollout in rollouts/<clip>_rl.json")
    ap.add_argument("--drone-scale", type=float, default=0.3, help="1 = real Crazyflie (9 cm); 0.3 is about bee-robot size")
    ap.add_argument("--speedup", type=float, default=1.0, help="playback speed, shown on screen")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.path == "rl":   # the RL rollout uses truth flowers, so draw the truth map (same flower ids)
        rl = json.loads(Path(f"rollouts/{a.clip_id}_rl.json").read_text())
        fp = Path(f"field/truth/{a.clip_id}.json")
        field = json.loads((fp if fp.exists() else Path(f"field/{a.clip_id}.json")).read_text())
    else:
        field = json.loads(Path(f"field/{a.clip_id}.json").read_text())
        plan = json.loads(Path(f"plan/{a.clip_id}.json").read_text())
        if not plan["route"]:
            print(f"{a.clip_id}: empty route, skipped"); return
    frame = a.frame
    if not frame:
        cap = cv2.VideoCapture(f"clips/{a.clip_id}.mp4"); ok, img = cap.read(); cap.release()
        if not ok: raise SystemExit("no frame; pass --frame")
        frame = f"rollouts/{a.clip_id}_ground.png"; Path("rollouts").mkdir(exist_ok=True); cv2.imwrite(frame, img)
    h_px, w_px = cv2.imread(frame).shape[:2]; aspect = w_px / h_px
    H = a.height_m; W = H * aspect
    cf2_dir = Path(a.menagerie) / "bitcraze_crazyflie_2"
    scaled_cf2(cf2_dir, a.drone_scale)
    model = load_scene(cf2_dir, build_xml(cf2_dir, frame, field, W, H), "beebot_scene")
    data = mujoco.MjData(model)
    out = a.out or f"rollouts/{a.clip_id}_mujoco{'_rl' if a.path == 'rl' else ''}.mp4"; Path(out).parent.mkdir(exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", macro_block_size=8)
    yaw = 0.0
    if a.path == "rl":
        rl["_W"], rl["_H"] = W, H
        traj = rl_trajectory(rl)
    else:
        traj = trajectory(field, plan, W, H, aspect)
    step = max(int(round(a.speedup)), 1)
    label = f"{'RL policy' if a.path == 'rl' else 'planner route'}  |  " + ("real time" if step == 1 else f"{step}x speed")
    with mujoco.Renderer(model, 720, 1280) as r:
        for t, x, y, z, vel, tilt in traj[::step]:
            if vel is not None and np.hypot(*vel) > 1e-6:
                target = math.atan2(vel[1], vel[0])
                yaw += (math.remainder(target - yaw, 2 * math.pi)) * 0.15   # turn gradually, no snapping
            data.qpos[:3] = (x, y, z)
            data.qpos[3:7] = pose_quat(yaw, tilt)
            mujoco.mj_forward(model, data)                    # kinematics only, no mj_step
            r.update_scene(data, camera="overview")
            img = np.ascontiguousarray(r.render())
            cv2.putText(img, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.append_data(img)
    writer.close(); print("wrote", out)

if __name__ == "__main__":
    main()
