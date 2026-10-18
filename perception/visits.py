"""Visit events (insect inside a flower), stop-flowers and behavior labels; flower boxes come from perception.flowers.

    python -m perception.visits --clips sm3_12_123903       # tracks/verified/<clip>.json -> tracks/<clip>.json
    python -m perception.visits --all
    python -m perception.visits --all --no-api              # stop-based flowers only

Flower boxes: ``perception.flowers.detect_tiled`` (the project's one flower detector: insect-free median frame,
2x2 tiles, dense ground cover skipped). Ids are made stable per camera view by ``relabel_views``.

Visits follow the Polytrack definition the farm truth uses: the insect's center is inside a flower's box (grown by
``GROW`` on each side) for at least ``MIN_VISIT_S``; gaps shorter than ``MERGE_GAP_S`` are bridged. In dense ground
cover (the honeybee clips have hundreds of flowers) a bee that stops for ``MIN_STOP_S`` outside every detected box
still counts: the stop itself becomes a flower (``"source": "stop"``), so analytics never lose a visit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perception.client import ROOT, BudgetExceeded, PerceptionClient
from perception.flowers import detect_tiled
from perception.track import manifest

IN_DIR = ROOT / "tracks" / "verified"
OUT_DIR = ROOT / "tracks"
PIECE = "flowers"     # Mk1.5 cache dir for flower calls (tracks/raw/flowers/), see perception.flowers.detect_tiled

GROW = 0.25          # visit zone = flower box grown by this share of its size on each side
MIN_VISIT_S = 0.3
MERGE_GAP_S = 0.5
MIN_STOP_S = 0.6     # a bee this still for this long is on something
STOP_SPEED = 25.0    # units (0-1000) per second, smoothed
STOP_BOX = 30        # side of a stop-flower box in 0-1000 units


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
    flowers = detect_tiled(clip_id, client=client, video=video)["flowers"] if client else []
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


def stable_ids(docs: list[dict]) -> list[dict]:
    """Rename flowers so one physical flower has one flower_id in every clip of a camera view.

    Matching is Falisha's ``perception.flowers.assign_view_ids`` (one-to-one box IoU >= 0.4 against the view's
    flowers so far, ids ordered top-to-bottom). Extra fields (``source``) are kept and visit events are renamed
    to match. All docs must come from one view; boxes are unchanged.
    """
    from perception.flowers import assign_view_ids

    parts = [{"clip_id": d["clip_id"], "flowers": [{"flower_id": f"f{i}", "box": f["box"]}
                                                   for i, f in enumerate(d["flowers"])]} for d in docs]
    relabeled = assign_view_ids(parts)
    out = []
    for d in docs:
        new_of_old = {}
        new_flowers = []
        for f_new in relabeled[d["clip_id"]]["flowers"]:
            i = next(k for k, f in enumerate(d["flowers"]) if f["box"] == f_new["box"]
                     and d["flowers"][k]["flower_id"] not in new_of_old)
            old = d["flowers"][i]
            new_of_old[old["flower_id"]] = f_new["flower_id"]
            new_flowers.append({**old, "flower_id": f_new["flower_id"]})
        events = [{**e, "flower_id": new_of_old.get(e["flower_id"], e["flower_id"])} for e in d["events"]]
        out.append({**d, "flowers": new_flowers, "events": events, "view_id": relabeled[d["clip_id"]]["view_id"]})
    return out


def relabel_views(tracks_dir: Path = OUT_DIR, clip_ids: list[str] | None = None) -> dict[str, int]:
    """Apply ``stable_ids`` to every camera view under tracks_dir; returns flowers per view."""
    from perception.flowers import view_id

    clip_ids = clip_ids or [c for c in manifest() if (tracks_dir / f"{c}.json").exists()]
    views: dict[str, list[str]] = {}
    for c in clip_ids:
        views.setdefault(view_id(c), []).append(c)
    summary = {}
    for view, cids in views.items():
        docs = stable_ids([json.loads((tracks_dir / f"{c}.json").read_text()) for c in cids])
        for d in docs:
            (tracks_dir / f"{d['clip_id']}.json").write_text(json.dumps(d, indent=1))
        summary[view] = len({f["flower_id"] for d in docs for f in d["flowers"]})
    return summary


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
