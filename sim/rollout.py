"""Render a clip with the RL agent's predicted bee path drawn over it.

    python -m sim.rollout sm3_12_123903 [--policy sim/policy.zip | --policy baseline]
                                        [--video clips/<clip>.mp4] [--no-video]

Flowers come from tracks/parts/<clip>.flowers.json when it exists, else truth/<clip>.json.
Writes rollouts/<clip>_rl.mp4 (orange dot + trail, visited flowers outlined) and
rollouts/<clip>_rl.json with the waypoint list [{t, x, y, flower_id or null}].
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sim.field_env import ROOT, FieldEnv, nearest_unvisited_action

POLICY_PATH = ROOT / "sim" / "policy.zip"
ORANGE = (255, 140, 0)  # RGB (imageio frames are RGB)
GREEN = (60, 220, 90)
WHITE = (240, 240, 240)


def load_flowers(clip_id):
    """Prefer the Mk1.5 flower file once it exists; fall back to the truth file."""
    parts = ROOT / "tracks" / "parts" / f"{clip_id}.flowers.json"
    if parts.exists():
        d = json.load(open(parts))
        return (d["flowers"] if isinstance(d, dict) else d), "mk1.5"
    return json.load(open(ROOT / "truth" / f"{clip_id}.json"))["flowers"], "truth"


def clip_meta(clip_id):
    for c in json.load(open(ROOT / "clips" / "manifest.json")):
        if c["clip_id"] == clip_id:
            return c
    return {"clip_id": clip_id, "fps": 30, "width": 1280, "height": 720, "duration_s": 15.0}


def start_point(clip_id):
    """Where the real bee entered the frame, if truth has a track; else the bottom-right corner."""
    p = ROOT / "truth" / f"{clip_id}.json"
    if p.exists():
        for tr in json.load(open(p)).get("tracks", []):
            if tr["points"]:
                q = tr["points"][0]
                return q.get("center") or [(q["box"][0] + q["box"][2]) / 2, (q["box"][1] + q["box"][3]) / 2]
    return [1000, 1000]


def make_actor(policy):
    """policy: 'baseline', a path to an SB3 PPO zip, or a loaded model. Returns f(env, obs) -> action."""
    if policy == "baseline":
        return lambda env, obs: nearest_unvisited_action(env)
    if isinstance(policy, (str, Path)):
        from stable_baselines3 import PPO
        policy = PPO.load(str(policy), device="cpu")
    return lambda env, obs: policy.predict(obs, deterministic=True)[0]


def run_episode(env, actor):
    """Run one episode; return waypoints [{t, x, y, flower_id}] (flower_id = flower being dwelt on) and visits."""
    obs, _ = env.reset()
    way = [{"t": 0.0, "x": round(float(env.pos[0]), 1), "y": round(float(env.pos[1]), 1), "flower_id": None}]
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(actor(env, obs))
        way.append({"t": round(info["t"], 3), "x": round(float(env.pos[0]), 1), "y": round(float(env.pos[1]), 1),
                    "flower_id": info["on_flower"] or info["visited_now"]})
        done = term or trunc
    visits = [{"flower_id": fid, "t": round(step * env.dt, 3)} for step, fid in env.visits]
    return way, visits


def render(video_path, out_path, way, flowers, visits, fps, width, height):
    import cv2
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7, macro_block_size=1)
    ts = np.array([w["t"] for w in way])
    xs = np.array([w["x"] for w in way]) * width / 1000.0
    ys = np.array([w["y"] for w in way]) * height / 1000.0
    visited_at = {v["flower_id"]: v["t"] for v in visits}
    sx, sy = width / 1000.0, height / 1000.0
    n = 0
    for i, frame in enumerate(reader):
        t = i / fps
        if t > ts[-1] + 1.0 / fps:
            break
        frame = np.ascontiguousarray(frame[:, :, :3])
        for f in flowers:
            x1, y1, x2, y2 = f["box"]
            done = f["flower_id"] in visited_at and visited_at[f["flower_id"]] <= t
            cv2.rectangle(frame, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                          GREEN if done else WHITE, 3 if done else 1)
        k = int(np.searchsorted(ts, t, side="right"))
        if k >= 2:
            pts = np.stack([xs[:k], ys[:k]], axis=1).astype(np.int32)
            cv2.polylines(frame, [pts], False, ORANGE, 2, cv2.LINE_AA)
        x = float(np.interp(t, ts, xs))
        y = float(np.interp(t, ts, ys))
        cv2.circle(frame, (int(x), int(y)), 9, ORANGE, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 9, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(frame, f"RL bee  t={t:4.1f}s  visits={sum(1 for v in visits if v['t'] <= t)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
        writer.append_data(frame)
        n += 1
    writer.close()
    reader.close()
    return n


def rollout(clip_id, policy=POLICY_PATH, video=None, write_video=True, out_dir=ROOT / "rollouts",
            budget_s=None, dt=0.1):
    flowers, source = load_flowers(clip_id)
    meta = clip_meta(clip_id)
    budget_s = budget_s or float(meta["duration_s"])
    env = FieldEnv(flowers=flowers, budget_s=budget_s, dt=dt, start=start_point(clip_id), shaping=False)
    way, visits = run_episode(env, make_actor(policy))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"clip_id": clip_id, "planner": "rl-ppo" if policy != "baseline" else "nearest_unvisited",
              "policy": str(policy), "flowers_from": source, "units": "norm1000",
              "budget_s": budget_s, "dt": dt, "start": way[0] and [way[0]["x"], way[0]["y"]],
              "speed": env.speed, "dwell_s": env.dwell_s,
              "visits": visits, "n_visits": len(visits), "n_open": int(env.open.sum()),
              "waypoints": way, "video": None}
    if write_video:
        video = Path(video) if video else ROOT / "clips" / f"{clip_id}.mp4"
        if video.exists():
            mp4 = out_dir / f"{clip_id}_rl.mp4"
            frames = render(video, mp4, way, flowers, visits, meta["fps"], meta["width"], meta["height"])
            result["video"] = str(mp4.relative_to(ROOT)) if mp4.is_relative_to(ROOT) else str(mp4)
            result["frames"] = frames
        else:
            print(f"no video at {video}; wrote JSON only")
    js = out_dir / f"{clip_id}_rl.json"
    js.write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_id")
    ap.add_argument("--policy", default=str(POLICY_PATH), help="path to PPO zip, or 'baseline'")
    ap.add_argument("--video", default=None, help="source mp4 (default clips/<clip>.mp4)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="seconds (default: clip duration)")
    a = ap.parse_args()
    r = rollout(a.clip_id, policy=a.policy, video=a.video, write_video=not a.no_video, budget_s=a.budget)
    print(f"{a.clip_id}: {r['n_visits']}/{r['n_open']} flowers visited in {r['budget_s']:.1f}s "
          f"({r['planner']}); video={r['video']}; json=rollouts/{a.clip_id}_rl.json")
    for v in r["visits"]:
        print(f"  t={v['t']:5.1f}  {v['flower_id']}")


if __name__ == "__main__":
    main()
