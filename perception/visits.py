"""Flower boxes (Mk1.5 on an insect-free reference frame) and visit events (insect inside a flower).

    python -m perception.visits --clips sm3_12_123903       # tracks/verified/<clip>.json -> tracks/<clip>.json
    python -m perception.visits --all
    python -m perception.visits --all --no-api              # stop-based flowers only

Reference frame: the per-pixel median of the clip, so moving insects vanish and flowers stay. Mk1.5 boxes every
open flower on it (tiled 2x2 with overlap so small flowers are big enough to see; duplicates merged by IoU).
Dense ground cover is detected from the first tile (cut-off answer or > DENSE_BOXES boxes) and skipped: on the
honeybee clips 43 of 76 answers hit the 6,000-token limit at ~1,000 blossoms per clip.

Visits follow the Polytrack definition the farm truth uses: the insect's center is inside a flower's box (grown by
``GROW`` on each side) for at least ``MIN_VISIT_S``; gaps shorter than ``MERGE_GAP_S`` are bridged. In dense ground
cover (the honeybee clips have hundreds of flowers) a bee that stops for ``MIN_STOP_S`` outside every detected box
still counts: the stop itself becomes a flower (``"source": "stop"``), so analytics never lose a visit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from perception.client import ROOT, BudgetExceeded, PerceptionClient
from perception.track import manifest

IN_DIR = ROOT / "tracks" / "verified"
OUT_DIR = ROOT / "tracks"
REF_DIR = ROOT / "tracks" / "raw" / "frames" / "reference"
PIECE = "flowers"     # cache dir tracks/raw/flowers (kept so cached answers stay valid)
PROMPT_VERSION = "v1"

GROW = 0.25          # visit zone = flower box grown by this share of its size on each side
MIN_VISIT_S = 0.3
MERGE_GAP_S = 0.5
MIN_STOP_S = 0.6     # a bee this still for this long is on something
STOP_SPEED = 25.0    # units (0-1000) per second, smoothed
STOP_BOX = 30        # side of a stop-flower box in 0-1000 units

DENSE_BOXES = 150    # first tile with more boxes than this (or a cut-off answer) = dense ground cover

PROMPT = ("Your goal is to segment out the following categories: open flower. Box every individual open flower "
          "head you can see, including small and partly hidden ones. Do not box leaves, buds or fruit.")


def reference_frame(video: Path, clip_id: str, samples: int = 41) -> Path:
    path = REF_DIR / f"{clip_id}.png"
    if path.exists():
        return path
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, min(samples, n)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    ref = np.median(np.stack(frames), axis=0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), ref)
    return path


def _tiles(ref: Path, clip_id: str) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """2x2 overlapping tiles; each with its (x0, y0, w, h) in 0-1000 units of the full frame."""
    img = cv2.imread(str(ref))
    H, W = img.shape[:2]
    tw, th = int(W * 0.6), int(H * 0.6)
    out = []
    for i, x0 in enumerate([0, W - tw]):
        for j, y0 in enumerate([0, H - th]):
            p = ref.with_name(f"{clip_id}_tile{i}{j}.png")
            if not p.exists():
                cv2.imwrite(str(p), img[y0:y0 + th, x0:x0 + tw])
            out.append((p, (x0 / W * 1000, y0 / H * 1000, tw / W * 1000, th / H * 1000)))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def detect_flowers(client: PerceptionClient, clip_id: str, video: Path) -> list[dict]:
    from perceptron import BoundingBox

    boxes = []
    for k, (tile, (x0, y0, w, h)) in enumerate(_tiles(reference_frame(video, clip_id), clip_id)):
        r = client.call(clip_id=clip_id, prompt_name="flowers", prompt_version=PROMPT_VERSION, prompt_text=PROMPT,
                        image_paths=[tile], annotation_format="box", max_tokens=6000)
        tile_boxes = r.annotations().boxes
        if k == 0 and (r.finish_reason == "length" or len(tile_boxes) > DENSE_BOXES):
            # dense ground cover (the honeybee clips: ~1,000 blossoms, answers cut off at the token limit):
            # a full inventory is neither complete nor useful, so visits fall back to stop-flowers
            return []
        for b in tile_boxes:
            if not isinstance(b, BoundingBox):
                continue
            bx = [x0 + b.top_left.x / 1000 * w, y0 + b.top_left.y / 1000 * h,
                  x0 + b.bottom_right.x / 1000 * w, y0 + b.bottom_right.y / 1000 * h]
            if bx[2] - bx[0] < 3 or bx[3] - bx[1] < 3 or (bx[2] - bx[0]) * (bx[3] - bx[1]) > 250 * 250:
                continue
            boxes.append([round(min(max(v, 0), 1000), 1) for v in bx])
    merged: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(b, m) < 0.4 for m in merged):
            merged.append(b)
    merged.sort(key=lambda b: (b[1], b[0]))
    return [{"flower_id": f"flower_{i}", "box": b, "source": "mk1.5"} for i, b in enumerate(merged)]


def _center(p):
    return p.get("center") or [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]


def _runs(flags: list[bool], ts: list[float], min_s: float, gap_s: float) -> list[tuple[float, float]]:
    runs, start, last = [], None, None
    for f, t in zip(flags, ts):
        if f:
            if start is None:
                start = t
            elif t - last > gap_s:
                runs.append((start, last))
                start = t
            last = t
        elif start is not None and t - last > gap_s:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return [(a, b) for a, b in runs if b - a >= min_s - 1e-6]


def stops(points: list[dict]) -> list[tuple[float, float, list[float]]]:
    """Stretches where the smoothed speed stays under STOP_SPEED; (t_start, t_end, mean center)."""
    obs = [p for p in points if not p.get("interpolated")]
    if len(obs) < 3:
        return []
    ts = np.array([p["t"] for p in obs])
    cs = np.array([_center(p) for p in obs])
    k = 5
    sm = np.stack([np.convolve(cs[:, d], np.ones(k) / k, mode="same") for d in range(2)], axis=1)
    sm[: k // 2], sm[-(k // 2):] = cs[: k // 2], cs[-(k // 2):]
    sp = np.zeros(len(ts))
    sp[1:] = np.linalg.norm(np.diff(sm, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3)
    out = []
    for a, b in _runs(list(sp < STOP_SPEED), list(ts), MIN_STOP_S, 0.2):
        m = (ts >= a) & (ts <= b)
        out.append((float(a), float(b), cs[m].mean(axis=0).tolist()))
    return out


def visit_events(tracks: list[dict], flowers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Visit events, plus stop-flowers for stops that are not inside any detected flower."""
    flowers = list(flowers)
    events = []
    n_stop = 0
    for tr in tracks:
        obs = [p for p in tr["points"] if not p.get("interpolated")]
        ts = [p["t"] for p in obs]
        cs = [_center(p) for p in obs]
        for st in stops(tr["points"]):
            a, b, c = st
            inside = False
            for f in flowers:
                x1, y1, x2, y2 = f["box"]
                gx, gy = (x2 - x1) * GROW, (y2 - y1) * GROW
                inside |= x1 - gx <= c[0] <= x2 + gx and y1 - gy <= c[1] <= y2 + gy
            if not inside:
                h = STOP_BOX / 2
                flowers.append({"flower_id": f"stop_{n_stop}", "source": "stop",
                                "box": [round(max(c[0] - h, 0), 1), round(max(c[1] - h, 0), 1),
                                        round(min(c[0] + h, 1000), 1), round(min(c[1] + h, 1000), 1)]})
                n_stop += 1
        for f in flowers:
            x1, y1, x2, y2 = f["box"]
            gx, gy = (x2 - x1) * GROW, (y2 - y1) * GROW
            flags = [x1 - gx <= c[0] <= x2 + gx and y1 - gy <= c[1] <= y2 + gy for c in cs]
            for a, b in _runs(flags, ts, MIN_VISIT_S, MERGE_GAP_S):
                events.append({"track_id": tr["track_id"], "flower_id": f["flower_id"], "type": "visit",
                               "t_start": round(a, 3), "t_end": round(b, 3)})
    events.sort(key=lambda e: e["t_start"])
    return events, flowers


FLY_SPEED = 40.0      # units/s; faster than this between observed points = flying


def label_behavior(tracks: list[dict], events: list[dict]) -> None:
    """Per-point label from motion and visits: inside a visit = foraging, fast = flying, otherwise walking.
    (Mk1.5's own behavior answers on 20 px insects were unreliable, see perception/verify.py.)"""
    for tr in tracks:
        vis = [(e["t_start"], e["t_end"]) for e in events if e["track_id"] == tr["track_id"]]
        pts = tr["points"]
        for i, p in enumerate(pts):
            if any(a <= p["t"] <= b for a, b in vis):
                p["label"] = "foraging"
                continue
            j = i + 1 if i + 1 < len(pts) else i - 1
            if j < 0 or pts[j]["t"] == p["t"]:
                continue
            a, b = _center(p), _center(pts[j])
            speed = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / abs(pts[j]["t"] - p["t"])
            p["label"] = "flying" if speed >= FLY_SPEED else "walking"


def finish_clip(clip_id: str, *, client: PerceptionClient | None, video: Path | None = None,
                in_dir: Path = IN_DIR, out_dir: Path = OUT_DIR) -> dict:
    doc = json.loads((in_dir / f"{clip_id}.json").read_text())
    video = video or (ROOT / manifest()[clip_id]["path"])
    flowers = detect_flowers(client, clip_id, video) if client else []
    events, flowers = visit_events(doc["tracks"], flowers)
    label_behavior(doc["tracks"], events)
    for tr in doc["tracks"]:
        for k in ("source_track", "confirmed_samples", "kind_votes", "hit_rate"):
            tr.pop(k, None)
    out = {"clip_id": clip_id, "model": doc["model"], "prompt_version": doc["prompt_version"],
           "tracks": doc["tracks"], "flowers": flowers, "events": events}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip_id}.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--in", dest="in_dir", default=str(IN_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    in_dir = Path(args.in_dir)
    clips = [c for c in manifest() if (in_dir / f"{c}.json").exists()] if args.all else args.clips
    client = None if args.no_api else PerceptionClient(PIECE)
    for cid in clips:
        try:
            doc = finish_clip(cid, client=client, in_dir=in_dir, out_dir=Path(args.out))
        except BudgetExceeded as e:
            print(f"stopping: {e}")
            break
        nf = sum(f.get("source") == "mk1.5" for f in doc["flowers"])
        print(f"{cid:16s} {nf:3d} flowers, {len(doc['flowers']) - nf} stop-flowers, {len(doc['events'])} visits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
