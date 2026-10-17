"""A2: score tracking prompt versions against truth/ and pick the best one.

    python -m perception.tune                       # the standard 5 clips, every config in CONFIGS
    python -m perception.tune --configs v2 v2-think --clips hb1_w02 sm3_12_123903

Each (config, clip) is one cached Perceptron call (about $0.002, more with reasoning). Parts go to
tracks/parts/tune/<config>/ so they never overwrite the real tracks/parts/; results to tracks/parts/tune/results.json.

Scores (0-1000 grid, 40 units is about 50 px at 1280 wide, twice a bee):
  n        model points inside the clip
  ghost    model points at times where truth has no bee (model saw something that was not there)
  median   median center error of matched points
  hit      share of model points within 40 units of the truth center (ghosts count as misses)
  cover    share of truth time covered by a hit within 0.3 s
Truth is a reference tracker, not hand labels (DATA_NOTES.md), so treat scores as agreement, not accuracy.
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
from pathlib import Path

from perception.client import ROOT, PerceptionClient
from perception.track import PIECE, track_clip

TUNE_DIR = ROOT / "tracks" / "parts" / "tune"

# Five clips that differ in what makes tracking hard: full-length clear track, occlusion gaps, a second full
# clip from another video, a farm honeybee (smaller, flowers, visits) and a wasp (other species).
CLIPS = ["hb1_w02", "hb3_w04", "hb6_w02", "sm3_12_123903", "sm2_08_112733"]

# name -> track_clip settings. A prompt version is prompts/track/<version>.txt; the rest are request settings.
CONFIGS = {
    "v1": {"version": "v1", "step_s": 0.5},
    "v2": {"version": "v2", "step_s": 0.5},
    "v2-fine": {"version": "v2", "step_s": 0.25},
    "v2-think": {"version": "v2", "step_s": 0.5, "reasoning_effort": "medium"},
    "v3-point": {"version": "v3", "step_s": 0.25, "annotation_format": "point"},
    "v3-point-think": {"version": "v3", "step_s": 0.25, "annotation_format": "point", "reasoning_effort": "medium"},
    # per-frame point detection: the model sees full-resolution frames instead of the server's downscaled video
    "f1-2fps": {"version": "f1", "mode": "frames", "fps": 2.0},
    "f2-2fps-t2": {"version": "f2", "mode": "frames", "fps": 2.0, "tiles": 2},
    # propose (tiles + motion) then verify each candidate on a 192 px zoomed crop
    "h1-2fps": {"version": "f3", "mode": "hybrid", "fps": 2.0, "tiles": 2},
}

HIT_UNITS = 40
MATCH_TOL_S = 0.1
COVER_TOL_S = 0.3


def score(points: list[dict], truth: list[dict]) -> dict:
    """Compare one model track with one truth track (both CONTRACT.md point lists)."""
    observed = [p for p in points if not p.get("interpolated")]
    tt = [q["t"] for q in truth]
    errors, ghosts, hits = [], 0, 0
    hit_times = []
    for p in observed:
        i = bisect.bisect_left(tt, p["t"])
        cands = [truth[j] for j in (i - 1, i) if 0 <= j < len(truth)]
        q = min(cands, key=lambda q: abs(q["t"] - p["t"])) if cands else None
        if q is None or abs(q["t"] - p["t"]) > MATCH_TOL_S:
            ghosts += 1
            continue
        e = ((p["center"][0] - q["center"][0]) ** 2 + (p["center"][1] - q["center"][1]) ** 2) ** 0.5
        errors.append(e)
        if e <= HIT_UNITS:
            hits += 1
            hit_times.append(p["t"])
    covered = sum(1 for t in tt if hit_times and min(abs(t - h) for h in hit_times) <= COVER_TOL_S)
    return {"n": len(observed), "ghost": ghosts, "matched": len(errors),
            "median": round(statistics.median(errors), 1) if errors else None,
            "hit": round(hits / len(observed), 2) if observed else 0.0,
            "cover": round(covered / len(tt), 2) if tt else None}


def load_truth(clip_id: str, root: Path = ROOT) -> list[dict]:
    tracks = json.loads((root / "truth" / f"{clip_id}.json").read_text())["tracks"]
    return max((t["points"] for t in tracks), key=len) if tracks else []


def run(configs: list[str], clips: list[str], client: PerceptionClient | None = None) -> dict:
    client = client or PerceptionClient(PIECE)
    results = json.loads((TUNE_DIR / "results.json").read_text()) if (TUNE_DIR / "results.json").exists() else {}
    print(f"{'config':16s} {'clip':14s} {'n':>3s} {'ghost':>5s} {'median':>6s} {'hit':>5s} {'cover':>5s}  "
          f"{'tokens':>11s} {'cost':>7s}  state")
    for name in configs:
        cfg = CONFIGS[name]
        for clip_id in clips:
            part = track_clip(clip_id, client=client, parts_dir=TUNE_DIR / name, **cfg)
            tr = part["tracks"][0]
            s = score(tr["points"], load_truth(clip_id))
            u = part.get("usage") or {}
            s.update(cost_usd=part["cost_usd"], finish=part["finish_reason"], complete=tr["complete"],
                     prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))
            results.setdefault(name, {})[clip_id] = s
            state = ("" if tr["complete"] else f"INCOMPLETE({part['finish_reason']}) ") + ("cache" if part["cache_hit"] else "")
            med = "-" if s["median"] is None else f"{s['median']:.0f}"
            print(f"{name:16s} {clip_id:14s} {s['n']:3d} {s['ghost']:5d} {med:>6s} {s['hit']:5.2f} "
                  f"{(s['cover'] or 0):5.2f}  {u.get('prompt_tokens', '?'):>5}+{u.get('completion_tokens', '?'):<5} "
                  f"{part['cost_usd']:7.4f}  {state}")
        rows = [results[name][c] for c in clips if c in results[name]]
        print(f"{name:16s} {'MEAN':14s} {sum(r['n'] for r in rows) / len(rows):3.0f} "
              f"{sum(r['ghost'] for r in rows) / len(rows):5.1f} {'':>6s} "
              f"{sum(r['hit'] for r in rows) / len(rows):5.2f} {sum(r['cover'] or 0 for r in rows) / len(rows):5.2f}")
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    (TUNE_DIR / "results.json").write_text(json.dumps(results, indent=1), newline="\n")
    print(f"{client.api_calls} API calls, {client.cache_hits} cache hits, ${client.spent_usd():.3f} total on "
          f"{client.cost_log.name}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--clips", nargs="*", default=CLIPS)
    args = ap.parse_args(argv)
    run(args.configs, args.clips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
