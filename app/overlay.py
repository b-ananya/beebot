"""Draw a tracks file onto its clip: bee box + trail, flower boxes, visit flashes.

Writes H.264 via imageio-ffmpeg so browsers (and st.video) can play it. Output is cached in
app/cache/ keyed by clip + tracks file contents, so the app only renders once per change.
"""
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

CACHE = Path("app/cache")
TRAIL_S = 1.5
BEE, TRAIL, FLOWER, VISIT = (0, 220, 255), (255, 200, 0), (80, 220, 80), (255, 80, 255)  # BGR


def _px(box, w, h):
    return int(box[0] / 1000 * w), int(box[1] / 1000 * h), int(box[2] / 1000 * w), int(box[3] / 1000 * h)


def _center(p):
    c = p.get("center") or [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]
    return c


def render(clip_path, doc, out_path):
    reader = imageio.get_reader(clip_path)
    fps = reader.get_meta_data().get("fps", 30)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for i, frame in enumerate(reader):
        t = i / fps
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        visiting = {e["flower_id"] for e in doc.get("events", []) if e["t_start"] <= t <= e["t_end"]}
        for f in doc.get("flowers", []):
            x1, y1, x2, y2 = _px(f["box"], w, h)
            on = f["flower_id"] in visiting
            cv2.rectangle(img, (x1, y1), (x2, y2), VISIT if on else FLOWER, 3 if on else 1)
        for tr in doc.get("tracks", []):
            pts = tr["points"]
            trail = [p for p in pts if t - TRAIL_S <= p["t"] <= t]
            for a, b in zip(trail, trail[1:]):
                if b["t"] - a["t"] > 0.5:  # a gap stays a gap
                    continue
                ca, cb = _center(a), _center(b)
                cv2.line(img, (int(ca[0] / 1000 * w), int(ca[1] / 1000 * h)),
                         (int(cb[0] / 1000 * w), int(cb[1] / 1000 * h)), TRAIL, 2)
            # current box: nearest observation within 1/4 s, otherwise nothing (occluded)
            near = min(pts, key=lambda p: abs(p["t"] - t), default=None)
            if near and abs(near["t"] - t) <= 0.25:
                x1, y1, x2, y2 = _px(near["box"], w, h)
                cv2.rectangle(img, (x1, y1), (x2, y2), BEE, 2)
                label = near.get("label") or tr.get("species") or tr["track_id"]
                cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BEE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{t:5.2f}s", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    writer.close()
    reader.close()


def annotated(clip_path, tracks_path):
    """Path to the rendered overlay video, rendering it if the inputs changed."""
    key = hashlib.sha256(Path(tracks_path).read_bytes() + str(clip_path).encode()).hexdigest()[:12]
    out = CACHE / f"{Path(clip_path).stem}_{key}.mp4"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        render(clip_path, json.loads(Path(tracks_path).read_text()), str(tmp))
        tmp.replace(out)
    return out


def frame_with_overlay(clip_path, doc, t):
    """One annotated still (RGB array) for thumbnails or the field view."""
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return np.zeros((720, 1280, 3), np.uint8)
    h, w = img.shape[:2]
    for f in doc.get("flowers", []):
        x1, y1, x2, y2 = _px(f["box"], w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), FLOWER, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
