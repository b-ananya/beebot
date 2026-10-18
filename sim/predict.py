"""Any picture of flowers -> predicted bee path, with how long the bee waits at each flower.

    python -m sim.predict path/to/photo.jpg [--budget 30] [--policy sim/policy.zip | baseline]
                                            [--prompt any|strawberry] [--no-video] [--start x,y]

Steps: Mk1.5 boxes the open flowers in the image (one cached call through perception.client, so a
repeat on the same bytes is free; farm reference frames already in tracks/parts/ cost nothing);
the PPO policy flies the field and parks the median bee dwell time on each flower it visits.

Writes rollouts/<name>_rl.json (flowers, per-visit arrive/leave/dwell, waypoints [{t, x, y, flower_id}]),
rollouts/<name>_rl.png (whole path with visit order and dwell labels) and
rollouts/<name>_rl.mp4 (the dot animated over the still image).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from sim.field_env import ROOT, FieldEnv
from sim.rollout import GREEN, ORANGE, POLICY_PATH, WHITE, Overlay, make_actor, run_episode

PROMPTS = {"any": ROOT / "sim" / "prompts" / "flowers_any-v2.txt",
           "strawberry": ROOT / "prompts" / "flowers" / "v1.txt"}
PARTS_DIR = ROOT / "tracks" / "parts"


def image_name(image_path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_path).stem)


def detect_flowers(image_path, prompt="any", client=None):
    """Mk1.5 flower boxes for one image, via the shared cached client. Returns (flowers, info)."""
    from perception.client import PerceptionClient
    from perception.flowers import parse_flowers

    name = image_name(image_path)
    ref = re.sub(r"_ref$", "", name)
    part = PARTS_DIR / f"{ref}.flowers.json"
    if part.exists():  # a farm clip's reference frame: the team's detection already on disk, no call at all
        d = json.load(open(part))
        return d["flowers"], {"source": "tracks/parts", "part": part.name, "cost_usd": 0.0, "cache_hit": True}
    client = client or PerceptionClient("p2")
    prompt_path = PROMPTS[prompt]
    version = prompt_path.stem.split("-")[-1] if prompt == "any" else prompt_path.stem
    resp = client.call(clip_id=name, prompt_name=f"flowers-{prompt}", prompt_version=version,
                       prompt_text=prompt_path.read_text().strip(), image_paths=[Path(image_path)],
                       annotation_format="box", max_tokens=4096)
    flowers, dropped = sane_boxes(parse_flowers(resp.text))
    info = {"source": "mk1.5", "model": resp.model, "prompt": f"flowers-{prompt}-{version}",
            "cost_usd": resp.cost_usd, "cache_hit": resp.cache_hit, "complete": resp.complete,
            "dropped_boxes": dropped, "cache_file": resp.cache_path.name}
    if resp.complete and dropped <= len(flowers):
        return flowers, info
    # Dense scenes make Mk1.5 emit rows of sliver boxes until the token limit: ask for centre points instead
    # (far fewer tokens) and give each the median farm flower radius.
    from sim.bee_stats import load as load_stats

    pts_prompt = ROOT / "sim" / "prompts" / "flowers_points-v1.txt"
    resp2 = client.call(clip_id=name, prompt_name="flowers-points", prompt_version="v1",
                        prompt_text=pts_prompt.read_text().strip(), image_paths=[Path(image_path)],
                        annotation_format="point", max_tokens=2048)
    r = load_stats()["median_flower_radius"]
    pts = []
    for pt in resp2.annotations().points:
        x, y = float(pt.x), float(pt.y)
        if 0 <= x <= 1000 and 0 <= y <= 1000 and all(abs(x - a) + abs(y - b) > r for a, b in pts):
            pts.append((x, y))
    pts.sort(key=lambda q: (q[1], q[0]))
    flowers2 = [{"flower_id": f"flower_{i}", "box": [max(0, x - r), max(0, y - r), min(1000, x + r), min(1000, y + r)]}
                for i, (x, y) in enumerate(pts, 1)]
    info.update({"source": "mk1.5-points", "fallback": "box answer was cut off or degenerate; used centre points "
                 "with the median farm flower radius", "box_attempt": info["cache_file"],
                 "cost_usd": info["cost_usd"] + resp2.cost_usd, "cache_hit": info["cache_hit"] and resp2.cache_hit,
                 "complete": resp2.complete, "cache_file": resp2.cache_path.name})
    if not resp2.complete or not flowers2:
        info["warning"] = "Mk1.5 could not list the flowers in this image (too dense?); the path below is unreliable"
    return flowers2, info


def sane_boxes(flowers, min_side=5.0):
    """Drop sliver boxes (a side under 5 grid units): Mk1.5 emits rows of them when a scene is too dense."""
    kept = [f for f in flowers if (f["box"][2] - f["box"][0]) >= min_side and (f["box"][3] - f["box"][1]) >= min_side]
    return kept, len(flowers) - len(kept)


def visit_table(way, visits, dt):
    """Per visit: when the bee arrived on the flower, when it left, and how long it stayed (ticks inclusive)."""
    rows = []
    for v in visits:
        fid, t_done = v["flower_id"], v["t"]
        k = max(i for i, w in enumerate(way) if w["t"] <= t_done)
        while k > 0 and way[k - 1]["flower_id"] == fid:
            k -= 1
        t_arrive = way[k]["t"]
        rows.append({"order": len(rows) + 1, "flower_id": fid, "t_arrive": round(t_arrive, 2),
                     "t_leave": round(t_done, 2), "dwell_s": round(t_done - t_arrive + dt, 2)})
    return rows


def draw_summary(image_rgb, way, flowers, table):
    """One picture: whole trail, visit order numbers and dwell labels on the visited flowers."""
    import cv2

    h, w = image_rgb.shape[:2]
    ov = Overlay(way, flowers, [{"flower_id": r["flower_id"], "t": r["t_leave"]} for r in table], w, h,
                 label="RL bee path")
    img = ov.draw(image_rgb.copy(), ov.t_end)
    centers = {f["flower_id"]: ((f["box"][0] + f["box"][2]) / 2 * w / 1000, f["box"][1] * h / 1000) for f in flowers}
    for r in table:
        cx, y1 = centers[r["flower_id"]]
        text = f"{r['order']}: {r['t_arrive']:.0f}s +{r['dwell_s']:.1f}s"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = int(cx - tw / 2), int(max(y1 - 6, th + 4))
        cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + 3), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    x0, y0 = int(ov.xs[0]), int(ov.ys[0])
    cv2.putText(img, "start", (x0 + 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)
    return img


def write_video(image_rgb, way, flowers, table, out_path, fps=30):
    import imageio.v2 as imageio

    h, w = image_rgb.shape[:2]
    ov = Overlay(way, flowers, [{"flower_id": r["flower_id"], "t": r["t_leave"]} for r in table], w, h)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7, macro_block_size=1)
    n = int(np.ceil(ov.t_end * fps)) + 1
    for i in range(n):
        writer.append_data(ov.draw(image_rgb.copy(), i / fps))
    writer.close()
    return n


def predict(image_path, policy=POLICY_PATH, budget_s=30.0, prompt="any", start=None, write_media=True,
            write_video_file=True, out_dir=ROOT / "rollouts", dt=0.1, detector=None):
    """The whole pipeline for one image; `detector` (image_path -> (flowers, info)) is swappable for tests."""
    import cv2

    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    flowers, info = (detector or (lambda p: detect_flowers(p, prompt=prompt)))(image_path)
    start = [1000.0, 1000.0] if start is None else list(start)
    env = FieldEnv(flowers=flowers, budget_s=budget_s, dt=dt, start=start, shaping=False)
    way, visits = run_episode(env, make_actor(policy))
    table = visit_table(way, visits, dt)
    name = image_name(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"image": str(image_path), "name": name, "width": w, "height": h, "units": "norm1000",
              "planner": "rl-ppo" if policy != "baseline" else "nearest_unvisited", "policy": str(policy),
              "flowers": flowers, "detection": info, "n_open": int(env.open.sum()),
              "budget_s": budget_s, "dt": dt, "start": start, "speed": env.speed, "dwell_s": env.dwell_s,
              "visits": table, "n_visits": len(table), "waypoints": way, "png": None, "video": None}
    if write_media:
        png = out_dir / f"{name}_rl.png"
        cv2.imwrite(str(png), cv2.cvtColor(draw_summary(rgb, way, flowers, table), cv2.COLOR_RGB2BGR))
        result["png"] = _rel(png)
        if write_video_file:
            mp4 = out_dir / f"{name}_rl.mp4"
            result["frames"] = write_video(rgb, way, flowers, table, mp4)
            result["video"] = _rel(mp4)
    (out_dir / f"{name}_rl.json").write_text(json.dumps(result, indent=1))
    return result


def _rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--policy", default=str(POLICY_PATH), help="PPO zip or 'baseline'")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds of bee time to simulate")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="any",
                    help="'any' for a general photo, 'strawberry' for the farm camera prompt")
    ap.add_argument("--start", default=None, help="x,y on the 0-1000 grid (default 1000,1000)")
    ap.add_argument("--no-video", action="store_true", help="JSON and PNG only")
    a = ap.parse_args()
    start = [float(v) for v in a.start.split(",")] if a.start else None
    r = predict(a.image, policy=a.policy, budget_s=a.budget, prompt=a.prompt, start=start,
                write_video_file=not a.no_video)
    d = r["detection"]
    print(f"{r['name']}: {len(r['flowers'])} flowers from {d['source']}"
          f" (cost ${d.get('cost_usd', 0):.3f}, cache_hit={d.get('cache_hit')})")
    if d.get("warning"):
        print("WARNING:", d["warning"])
    print(f"path: {r['n_visits']}/{r['n_open']} flowers in {r['budget_s']:.0f}s, dwell {r['dwell_s']} s each")
    for v in r["visits"]:
        print(f"  {v['order']}. {v['flower_id']:10s} arrive {v['t_arrive']:5.1f}s  leave {v['t_leave']:5.1f}s")
    print(f"wrote rollouts/{r['name']}_rl.json, {r['png']}, {r['video']}")


if __name__ == "__main__":
    main()
