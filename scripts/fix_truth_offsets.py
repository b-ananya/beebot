"""Re-time farm truth (sm*) by one constant offset per source video.

    python -m scripts.fix_truth_offsets            # dry run: print per-clip and per-video offsets
    python -m scripts.fix_truth_offsets --apply    # rewrite truth/sm*.json, record "time_offset_s"

Why: farm truth times come from Polytrack's ``absframe`` (DATA_NOTES: +-1.3 s). Against the motion tracker
(``tracks/cv``, which never sees truth), agreement jumps from ~30% to >90% when truth is shifted by a constant per
source video (cam 2 day 08: about -0.3 s; cam 3 day 12: about +1.4 s). Zoomed crops confirm it: after the shift the
bee sits in the truth box; before it, the box is on empty leaves (eval/checks/offset_*.jpg).

The offset is the median over that video's clips of the shift (-3..+3 s, 0.1 s steps) that maximizes coverage,
so one odd clip can't move it. Points and events are both shifted; points that land outside the clip are dropped.
Re-running with --apply is a no-op once ``time_offset_s`` is present.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.run_eval import score_clip

ROOT = Path(__file__).resolve().parents[1]


def shifted(doc: dict, off: float, duration: float | None = None) -> dict:
    d = copy.deepcopy(doc)
    for tr in d["tracks"]:
        pts = []
        for p in tr["points"]:
            p["t"] = round(p["t"] + off, 3)
            if duration is None or 0 <= p["t"] <= duration:
                pts.append(p)
        tr["points"] = pts
    for e in d["events"]:
        e["t_start"] = round(e["t_start"] + off, 3)
        e["t_end"] = round(e["t_end"] + off, 3)
        if duration is not None:
            e["t_start"], e["t_end"] = max(e["t_start"], 0.0), min(e["t_end"], duration)
    if duration is not None:
        d["events"] = [e for e in d["events"] if e["t_end"] > e["t_start"]]
    return d


def best_offset(truth: dict, pred: dict) -> tuple[float, float]:
    res = []
    for off in np.arange(-3.0, 3.001, 0.1):
        s = score_clip(shifted(truth, float(off)), pred)
        res.append(((s["coverage"] or 0), -abs(off), round(float(off), 1)))
    cov, _, off = max(res)
    return off, cov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    manifest = {e["clip_id"]: e for e in json.loads((ROOT / "clips" / "manifest.json").read_text())}
    per_video = defaultdict(list)
    for cid, e in manifest.items():
        tp, pp = ROOT / "truth" / f"{cid}.json", ROOT / "tracks" / "cv" / f"{cid}.json"
        if not cid.startswith("sm") or not tp.exists() or not pp.exists():
            continue
        truth = json.loads(tp.read_text())
        if "time_offset_s" in truth:
            print(f"{cid}: already shifted by {truth['time_offset_s']} s")
            continue
        off, cov = best_offset(truth, json.loads(pp.read_text()))
        per_video[e["source_video"]].append((cid, off, cov))
        print(f"{cid:16s} best {off:+.1f} s (coverage {cov:.2f})")
    for video, rows in per_video.items():
        off = round(statistics.median(o for _, o, _ in rows), 1)
        print(f"{video}: offset {off:+.1f} s from {len(rows)} clips")
        if not args.apply:
            continue
        for cid, _, _ in rows:
            tp = ROOT / "truth" / f"{cid}.json"
            doc = shifted(json.loads(tp.read_text()), off, manifest[cid]["duration_s"])
            doc["time_offset_s"] = off
            doc["time_offset_note"] = "per-video constant shift, see scripts/fix_truth_offsets.py"
            tp.write_text(json.dumps(doc, indent=1))
    if not args.apply:
        print("dry run; add --apply to rewrite truth/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
