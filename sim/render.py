"""Render a plan over a real frame: flowers colored by deficit, bee path in grey, robot flying the route.

Headless (OpenCV + imageio-ffmpeg, H.264), so it runs on any laptop and plays in the app.
"""
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio

FPS = 30
GREY, ROBOT, ROUTE = (170, 170, 170), (255, 120, 0), (255, 200, 80)  # BGR


def _deficit_color(d, dmax):
    k = 0 if dmax == 0 else d / dmax  # green (no deficit) -> red (largest deficit)
    return (40, int(200 * (1 - k)) + 40, int(220 * k) + 30)


def background(clip_path, t):
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {clip_path}")
    return cv2.addWeighted(img, 0.75, img * 0, 0.25, 0)  # dim so overlays read


def render_plan(clip_path, field, plan, out_path, bee_points=(), time_scale=1.0):
    """Animate the route. time_scale > 1 shows the robot slower than real time (shown on screen)."""
    base = background(clip_path, 0.0)
    h, w = base.shape[:2]
    px = lambda c: (int(c[0] / 1000 * w), int(c[1] / 1000 * h))
    by_id = {f["flower_id"]: f for f in field["flowers"]}
    dmax = max((f["deficit"] for f in field["flowers"]), default=0)

    for a, b in zip(bee_points, bee_points[1:]):  # where the bee actually went
        if b["t"] - a["t"] <= 0.5:
            cv2.line(base, px(a["center"]), px(b["center"]), GREY, 1, cv2.LINE_AA)
    for f in field["flowers"]:
        r = max(int(f["radius"] / 1000 * w), 8)
        cv2.circle(base, px(f["center"]), r, _deficit_color(f["deficit"], dmax), 2, cv2.LINE_AA)
        cv2.putText(base, f"{f['visits']}v", (px(f["center"])[0] + r + 2, px(f["center"])[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # waypoints in pixels; legs timed with the planner's speed (y-norm units/s) and dwell
    speed, dwell = plan["robot"]["speed"], plan["robot"]["dwell_s"]
    aspect = w / h
    stops = [plan["start"]] + [by_id[f]["center"] for f in plan["route"]]
    frames, t_total = [], 0.0
    for i in range(1, len(stops)):
        a, b = stops[i - 1], stops[i]
        leg = math.hypot((b[0] - a[0]) * aspect, b[1] - a[1]) / speed
        frames.append(("fly", a, b, leg))
        frames.append(("dwell", b, b, dwell))
        t_total += leg + dwell

    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    total = max(plan.get("total_deficit") or 1, 1)
    speed_note = "real time" if time_scale == 1 else f"shown {time_scale:g}x slower than real time"
    state = {"cleared": 0.0, "visited": [], "trail": [px(plan["start"])], "pos": plan["start"]}

    def draw():
        img = base.copy()
        for p, q in zip(state["trail"], state["trail"][1:]):
            cv2.line(img, p, q, ROUTE, 2, cv2.LINE_AA)
        for c in state["visited"]:
            cv2.circle(img, px(c), 6, (80, 230, 80), -1)
        cv2.circle(img, px(state["pos"]), 9, ROBOT, -1, cv2.LINE_AA)
        cv2.putText(img, f"deficit cleared {state['cleared']:.0f}/{total:.0f}   {speed_note}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    stop_ids = iter(plan["route"])
    for kind, a, b, dur in frames:
        n = max(int(dur * time_scale * FPS), 1)
        for k in range(n):
            u = (k + 1) / n
            state["pos"] = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u) if kind == "fly" else b
            if kind == "fly":
                state["trail"].append(px(state["pos"]))
            draw()
        if kind == "dwell":  # credit the flower once the robot has finished dwelling on it
            state["cleared"] += by_id[next(stop_ids)]["deficit"]
            state["visited"].append(b)
    for _ in range(FPS):  # hold the final state for a second
        draw()
    writer.close()
    return t_total
