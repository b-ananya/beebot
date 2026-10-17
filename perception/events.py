"""Visit events from geometry: bee track points inside flower boxes -> tracks/parts/<clip>.events.json.

    python -m perception.events --clips sm3_12_123903
    python -m perception.events --all          # every clip that has both a track part and a flowers part

No API call. A visit is a run of track points whose centre lies inside one flower's box (grown by ``margin``
units) for at least ``min_dwell_s``, allowing gaps up to ``max_gap_s`` inside the same flower. It is a stopgap
until P2 produces events from Mk1.5 itself; ``source: "geometry"`` in the part says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from perception.client import ROOT
from perception.track import PARTS_DIR

MARGIN = 15        # grid units (~19 px): bees perch on the edge of a flower
MIN_DWELL_S = 0.4  # shorter than this is a fly-over, not a visit
MAX_GAP_S = 1.0    # unobserved stretch allowed inside one visit


def inside(center: list[float], box: list[float], margin: float = MARGIN) -> bool:
    return box[0] - margin <= center[0] <= box[2] + margin and box[1] - margin <= center[1] <= box[3] + margin


def visits(points: list[dict], flowers: list[dict], track_id: str, *, margin: float = MARGIN,
           min_dwell_s: float = MIN_DWELL_S, max_gap_s: float = MAX_GAP_S) -> list[dict]:
    """Visit events for one track. Each point is assigned to the smallest flower box that contains it."""
    events = []
    current = None  # (flower_id, t_start, t_last)

    def close():
        nonlocal current
        if current and current[2] - current[1] >= min_dwell_s:
            events.append({"track_id": track_id, "flower_id": current[0], "type": "visit",
                           "t_start": round(current[1], 3), "t_end": round(current[2], 3)})
        current = None

    for p in sorted(points, key=lambda p: p["t"]):
        hits = [f for f in flowers if inside(p["center"], f["box"], margin)]
        fid = min(hits, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))["flower_id"] if hits else None
        if current and (fid != current[0] or p["t"] - current[2] > max_gap_s):
            close()
        if fid is None:
            continue
        current = (fid, current[1], p["t"]) if current else (fid, p["t"], p["t"])
    close()
    return events


def events_for_clip(clip_id: str, *, parts_dir: Path | None = None, **kw) -> Path:
    parts_dir = parts_dir or PARTS_DIR
    track_part = json.loads((parts_dir / f"{clip_id}.track.json").read_text())
    flowers_part = json.loads((parts_dir / f"{clip_id}.flowers.json").read_text())
    events = []
    for tr in track_part["tracks"]:
        events += visits(tr["points"], flowers_part["flowers"], tr["track_id"], **kw)
    doc = {"clip_id": clip_id, "source": "geometry", "margin": kw.get("margin", MARGIN),
           "min_dwell_s": kw.get("min_dwell_s", MIN_DWELL_S), "events": events}
    out = parts_dir / f"{clip_id}.events.json"
    out.write_text(json.dumps(doc, indent=1), newline="\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if args.all:
        clips = sorted(p.name[: -len(".flowers.json")] for p in PARTS_DIR.glob("*.flowers.json")
                       if (PARTS_DIR / f"{p.name[: -len('.flowers.json')]}.track.json").exists())
    else:
        clips = args.clips
    if not clips:
        ap.error("give --clips ... or --all")
    for clip_id in clips:
        out = events_for_clip(clip_id)
        ev = json.loads(out.read_text())["events"]
        print(f"{clip_id:16s} {len(ev)} visits" + (": " + ", ".join(
            f"{e['flower_id']} {e['t_start']}-{e['t_end']}s" for e in ev) if ev else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
