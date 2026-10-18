"""Measure how a real bee moves, from the farm truth files (truth/sm*.json).

Writes sim/bee_stats.json with the median flight speed (grid units per second), the median
visit dwell time (seconds) and the median flower radius (grid units). The env reads that file.
Run: python -m sim.bee_stats
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = ROOT / "sim" / "bee_stats.json"


def compute(truth_dir=ROOT / "truth", pattern="sm*.json"):
    speeds, flight, dwells, radii, files = [], [], [], [], []
    for f in sorted(glob.glob(str(truth_dir / pattern))):
        d = json.load(open(f))
        files.append(Path(f).name)
        visits = [(ev["track_id"], ev["t_start"], ev["t_end"]) for ev in d.get("events", []) if ev.get("type") == "visit"]
        for tr in d.get("tracks", []):
            pts = [p for p in tr["points"] if not p.get("interpolated", False)]
            for a, b in zip(pts, pts[1:]):
                dt = b["t"] - a["t"]
                if dt <= 0 or dt > 0.2:  # a gap stays a gap: do not measure speed across it
                    continue
                ca, cb = _center(a), _center(b)
                v = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]) / dt)
                speeds.append(v)
                on_flower = any(tid == tr["track_id"] and t0 <= a["t"] <= t1 for tid, t0, t1 in visits)
                if not on_flower:
                    flight.append(v)
        for ev in d.get("events", []):
            if ev.get("type") == "visit":
                dwells.append(float(ev["t_end"] - ev["t_start"]))
        for fl in d.get("flowers", []):
            x1, y1, x2, y2 = fl["box"]
            radii.append(float(max(x2 - x1, y2 - y1) / 2))
    return {
        "source": "truth/sm*.json (Polytrack reference tracks, 0-1000 grid)",
        "files": files,
        "n_speed_samples": len(speeds),
        "n_visits": len(dwells),
        "n_flowers": len(radii),
        "median_speed": round(float(np.median(speeds)), 1),
        "median_flight_speed": round(float(np.median(flight)), 1),
        "p90_flight_speed": round(float(np.percentile(flight, 90)), 1),
        "note": "median_speed counts every frame, most of them a bee sitting on a flower; "
                "median_flight_speed drops frames inside visit events and is the env speed cap",
        "median_dwell_s": round(float(np.median(dwells)), 2),
        "median_flower_radius": round(float(np.median(radii)), 1),
    }


def _center(p):
    if p.get("center"):
        return p["center"]
    x1, y1, x2, y2 = p["box"]
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def load():
    if not STATS_PATH.exists():
        STATS_PATH.write_text(json.dumps(compute(), indent=1))
    return json.load(open(STATS_PATH))


if __name__ == "__main__":
    stats = compute()
    STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
