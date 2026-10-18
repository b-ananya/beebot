"""Classical motion tracker: static-camera background subtraction + best-path linking. No API calls.

    python -m perception.cv_track --clips hb1_w00            # -> tracks/cv/<clip>.json
    python -m perception.cv_track --all

Why: asking Mk1.5 to track a whole 10 s clip returned a smooth invented line (3.9% coverage on hb1_w00). The
cameras are fixed, so moving insects stand out against a per-clip median background. Frame-to-frame linking breaks
the bee into dozens of pieces among hundreds of swaying-flower blobs, so instead we find the best-scoring path
through all detections at once (dynamic programming over a DAG whose edges may skip up to ``max_skip`` frames, so a
short occlusion doesn't end the track). Paths are extracted greedily: best path, remove its detections, repeat.

``perception.verify`` then asks Mk1.5 which paths are really insects, and ``perception.visits`` adds flower boxes
and visit events. Output follows CONTRACT.md: tracks ``cand_<n>`` with per-track stats under ``"stats"``.
Coordinates are normalized to 0-1000 like Mk1.5.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from perception.client import ROOT
from perception.track import manifest

OUT_DIR = ROOT / "tracks" / "cv"


@dataclass
class Params:
    bg_samples: int = 41          # frames sampled over the clip for the median background
    diff_thresh: int = 28         # gray-level difference that counts as motion
    min_area: int = 12            # blob area limits in pixels at 720p (bees are ~15-30 px across)
    max_area: int = 2500
    max_skip: int = 15            # frames a path may jump over (occlusion, missed detection)
    speed_px: float = 45.0        # max plausible movement per frame at 720p
    move_cost: float = 0.02       # path cost per pixel moved
    skip_cost: float = 0.6        # path cost per skipped frame
    node_bias: float = 1.2        # subtracted from every detection's score; junk blobs go negative
    hot_weight: float = 4.0       # penalty for spots that are in motion most of the clip (swaying flowers)
    min_len: int = 10             # detections in a path to keep it
    min_score: float = 8.0        # path score to keep it
    max_paths: int = 6


def read_gray(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames), fps


def detect(frames: np.ndarray, p: Params) -> list[np.ndarray]:
    """Per frame, an array of blobs that differ from the clip's median background.

    Columns: x, y, w, h, area, contrast (mean |diff| / threshold), hot (share of the clip that spot is in motion).
    """
    n = len(frames)
    idx = np.linspace(0, n - 1, min(p.bg_samples, n)).astype(int)
    bg = np.median(frames[idx], axis=0).astype(np.int16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masks, diffs = [], []
    heat = np.zeros(frames.shape[1:], np.float32)
    for i in range(n):
        diff = np.abs(cv2.GaussianBlur(frames[i], (3, 3), 0).astype(np.int16) - bg).astype(np.uint8)
        mask = cv2.morphologyEx((diff > p.diff_thresh).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        heat += mask
        masks.append(mask)
        diffs.append(diff)
    heat = cv2.blur(heat / n, (9, 9))
    dets = []
    for mask, diff in zip(masks, diffs):
        k, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rows = []
        for j in range(1, k):
            x, y, w, h, area = stats[j]
            if not (p.min_area <= area <= p.max_area and max(w, h) <= 120):
                continue
            sl = (slice(y, y + h), slice(x, x + w))
            blob = labels[sl] == j
            contrast = float(diff[sl][blob].mean()) / p.diff_thresh
            hot = float(heat[sl][blob].mean())
            rows.append((cents[j][0], cents[j][1], w, h, area, contrast, hot))
        dets.append(np.array(rows, np.float32).reshape(-1, 7))
    return dets


def node_scores(d: np.ndarray, p: Params) -> np.ndarray:
    if not len(d):
        return np.zeros(0, np.float32)
    area, contrast, hot = d[:, 4], d[:, 5], d[:, 6]
    return np.log1p(area / p.min_area) * 0.5 + np.minimum(contrast, 3.0) - p.hot_weight * hot - p.node_bias


def best_path(dets: list[np.ndarray], used: list[np.ndarray], p: Params) -> tuple[float, list[tuple[int, int]]]:
    """Highest-scoring chain of detections (frame, index) that skips at most ``max_skip`` frames between links."""
    n = len(dets)
    scores = [np.where(u, -np.inf, node_scores(d, p)) for d, u in zip(dets, used)]
    best = [s.copy() for s in scores]                    # best path score ending at each detection
    back = [np.full((len(d), 2), -1, np.int64) for d in dets]
    for f in range(n):
        if not len(dets[f]):
            continue
        cur = dets[f][:, :2]
        for g in range(max(0, f - p.max_skip - 1), f):
            if not len(dets[g]):
                continue
            gap = f - g
            prev = dets[g][:, :2]
            dist = np.linalg.norm(cur[:, None, :] - prev[None, :, :], axis=2)       # (cur, prev)
            ok = dist <= p.speed_px * gap
            cand = best[g][None, :] - p.move_cost * dist - p.skip_cost * (gap - 1) + scores[f][:, None]
            cand = np.where(ok, cand, -np.inf)
            j = cand.argmax(axis=1)
            val = cand[np.arange(len(cur)), j]
            better = val > best[f]
            best[f] = np.where(better, val, best[f])
            back[f][better] = np.stack([np.full(better.sum(), g), j[better]], axis=1)
    top, end = -np.inf, None
    for f, b in enumerate(best):
        if len(b) and b.max() > top:
            top, end = float(b.max()), (f, int(b.argmax()))
    path = []
    while end is not None and end[0] >= 0:
        path.append(end)
        g, j = back[end[0]][end[1]]
        end = (int(g), int(j)) if g >= 0 else None
    return top, path[::-1]


def extract_paths(dets: list[np.ndarray], p: Params) -> list[tuple[float, list[tuple[int, int]]]]:
    used = [np.zeros(len(d), bool) for d in dets]
    paths = []
    for _ in range(p.max_paths):
        score, path = best_path(dets, used, p)
        if not path or score < p.min_score:
            break
        for f, j in path:
            used[f][j] = True
            # also retire blobs overlapping the chosen one (fragments of the same insect)
            d = dets[f]
            near = np.linalg.norm(d[:, :2] - d[j, :2], axis=1) < max(d[j, 2], d[j, 3])
            used[f] |= near
        if len(path) >= p.min_len:
            paths.append((score, path))
    return paths


def path_stats(dets, path, fps: float, score: float) -> dict:
    pts = np.array([dets[f][j, :2] for f, j in path])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.zeros(1)
    span = path[-1][0] - path[0][0] + 1
    return {"score": round(score, 1), "hits": len(path), "duration_s": round(span / fps, 2),
            "fill": round(len(path) / span, 2), "path_px": round(float(step.sum()), 1),
            "mean_area": round(float(np.mean([dets[f][j, 4] for f, j in path])), 1),
            "mean_hot": round(float(np.mean([dets[f][j, 6] for f, j in path])), 3)}


def to_contract(clip_id: str, dets, paths, fps: float, W: int, H: int, p: Params) -> dict:
    out_tracks = []
    for n, (score, path) in enumerate(paths):
        pts = []
        for f, j in path:
            x, y, w, h = dets[f][j, :4]
            half = max(float(w), float(h), 16.0) / 2 * 1.2
            box = [max(0.0, (x - half) / W * 1000), max(0.0, (y - half) / H * 1000),
                   min(1000.0, (x + half) / W * 1000), min(1000.0, (y + half) / H * 1000)]
            pts.append({"t": round(f / fps, 3), "box": [round(float(v), 1) for v in box],
                        "center": [round(float(x) / W * 1000, 1), round(float(y) / H * 1000, 1)],
                        "interpolated": False})
        out_tracks.append({"track_id": f"cand_{n}", "species": "unknown", "complete": True, "points": pts,
                           "stats": path_stats(dets, path, fps, score)})
    return {"clip_id": clip_id, "model": "cv-bgsub-dp", "prompt_version": None, "tracks": out_tracks,
            "flowers": [], "events": [], "params": asdict(p)}


def run_clip(clip_id: str, p: Params | None = None, out_dir: Path = OUT_DIR, video: Path | None = None) -> dict:
    p = p or Params()
    frames, fps = read_gray(video or (ROOT / manifest()[clip_id]["path"]))
    H, W = frames.shape[1:]
    dets = detect(frames, p)
    doc = to_contract(clip_id, dets, extract_paths(dets, p), fps, W, H, p)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip_id}.json").write_text(json.dumps(doc))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    clips = list(manifest()) if args.all else args.clips
    for cid in clips:
        doc = run_clip(cid, out_dir=Path(args.out))
        print(f"{cid:16s} {len(doc['tracks'])} paths  " +
              "  ".join(f"[{t['stats']['score']} {t['stats']['hits']}]" for t in doc["tracks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
