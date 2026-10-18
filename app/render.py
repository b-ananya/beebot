"""Burn tracks, trails, flower boxes and visit flashes into an mp4 the browser can play (H.264, yuv420p).

    python -m app.render --clip sm3_12_123903 --source tracks          # -> app/cache/tracks/sm3_12_123903.mp4
    python -m app.render --all --source truth

Writes a standalone mp4 with overlays burned in. The panel UI draws boxes on a canvas instead of using this.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "app" / "cache"

# BGR. Insects take the app's categorical slots in fixed order (blue, orange, aqua; the three that stay
# distinguishable for color-blind viewers), so a track has the same color in the video and in the charts.
# Flowers: detected = warm yellow, stop-derived = grey, visit in progress = filled green.
TRACK_COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27)]
FLOWER = (0, 210, 255)
FLOWER_STOP = (170, 170, 170)
TRAIL_S = 2.0


def source_dir(source: str) -> Path:
    return {"tracks": ROOT / "tracks", "truth": ROOT / "truth", "mock": ROOT / "tracks" / "mock",
            "cv": ROOT / "tracks" / "cv", "verified": ROOT / "tracks" / "verified"}.get(source, ROOT / source)


def load_doc(source: str, clip_id: str) -> dict | None:
    p = source_dir(source) / f"{clip_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _px(box, W, H):
    return int(box[0] / 1000 * W), int(box[1] / 1000 * H), int(box[2] / 1000 * W), int(box[3] / 1000 * H)


def _center(p):
    return p.get("center") or [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]


def render(video: Path, doc: dict, out: Path, *, show_all_flowers: bool = True) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracks = doc.get("tracks", [])
    tarrs = [(np.array([p["t"] for p in tr["points"]]), tr) for tr in tracks]
    visits = [e for e in doc.get("events", []) if e["type"] == "visit"]
    visited_ids = {e["flower_id"] for e in visits}
    flowers = [f for f in doc.get("flowers", []) if show_all_flowers or f["flower_id"] in visited_ids]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, pixelformat="yuv420p",
                                macro_block_size=8, ffmpeg_params=["-movflags", "+faststart"])
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        overlay = frame.copy()
        active = {e["flower_id"] for e in visits if e["t_start"] <= t <= e["t_end"]}
        for f in flowers:
            x1, y1, x2, y2 = _px(f["box"], W, H)
            color = FLOWER_STOP if f.get("source") == "stop" else FLOWER
            if f["flower_id"] in active:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 120), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if f["flower_id"] not in visited_ids else 2)
            if f["flower_id"] in visited_ids:
                cv2.putText(frame, f["flower_id"], (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                            cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        for k, (ts, tr) in enumerate(tarrs):
            if not len(ts) or t < ts[0] - 0.05 or t > ts[-1] + 0.05:
                continue
            color = TRACK_COLORS[k % len(TRACK_COLORS)]
            lo, hi = np.searchsorted(ts, t - TRAIL_S), np.searchsorted(ts, t, side="right")
            pts = tr["points"][lo:hi]
            for a, b in zip(pts[:-1], pts[1:]):
                if b["t"] - a["t"] > 0.6:
                    continue
                age = (t - b["t"]) / TRAIL_S
                ca, cb = _center(a), _center(b)
                c = tuple(int(v * (1 - age) + 255 * age * 0.3) for v in color)
                cv2.line(frame, (int(ca[0] / 1000 * W), int(ca[1] / 1000 * H)),
                         (int(cb[0] / 1000 * W), int(cb[1] / 1000 * H)), c, 2, cv2.LINE_AA)
            if hi == 0:
                continue
            cur = tr["points"][hi - 1]
            if t - cur["t"] > 0.2:
                continue
            x1, y1, x2, y2 = _px(cur["box"], W, H)
            dashed = cur.get("interpolated")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if dashed else 2)
            label = tr.get("species", "") if tr.get("species") not in (None, "unknown") else tr["track_id"]
            if cur.get("label"):
                label += f" · {cur['label']}"
            cv2.putText(frame, label.replace(" · ", " - "), (x1, min(y2 + 14, H - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{t:5.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fi += 1
    cap.release()
    writer.close()
    return out


def cached_render(source: str, clip_id: str, video: Path, *, force: bool = False) -> Path | None:
    doc = load_doc(source, clip_id)
    if doc is None:
        return None
    src = source_dir(source) / f"{clip_id}.json"
    out = CACHE / source.replace("/", "_") / f"{clip_id}.mp4"
    if force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        render(video, doc, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--source", default="tracks")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    manifest = {e["clip_id"]: e for e in json.loads((ROOT / "clips" / "manifest.json").read_text())}
    clips = list(manifest) if args.all else args.clip
    for cid in clips:
        out = cached_render(args.source, cid, ROOT / manifest[cid]["path"], force=args.force)
        print(f"{cid}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
