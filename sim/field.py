"""Build field/*.json (pollination maps) from the dataset, so C can plan before A and B ship.

Patch field for a farm clip: flower boxes from truth/<clip>.json, visit counts for that whole
station-day from data/spatial/Flower_Positions_CSV/flowers_<station><day>.csv (Total_visits).
B's builder replaces this with Mk1.5-derived maps (source "mk1.5") in step B5.
"""
import csv
import json
from pathlib import Path

SPATIAL = Path("data/spatial/Flower_Positions_CSV")
DEFAULT_TARGET = 4  # visits per open flower; a setting, not a biological claim (cite one before quoting)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def patch_from_truth(clip_id, target_visits=DEFAULT_TARGET):
    """sm<station>_<day>_<HHMMSS> -> field doc with day-level visits per flower."""
    truth = json.loads(Path(f"truth/{clip_id}.json").read_text())
    station, day = clip_id[2:].split("_")[:2]
    visits = {}
    csv_path = SPATIAL / f"flowers_{station}{day}.csv"
    for r in csv.DictReader(open(csv_path)):
        n, v, t = _num(r["flower_num"]), _num(r["Total_visits"]), _num(r["Total_time"])
        if n is not None:
            visits[f"flower_{int(n)}"] = (int(v or 0), round((t or 0) / 30, 1))  # Total_time is in frames at 30 fps
    flowers = []
    for f in truth["flowers"]:
        b = f["box"]
        v, dwell = visits.get(f["flower_id"], (0, 0.0))
        flowers.append({"flower_id": f["flower_id"], "center": [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)],
                        "radius": round((b[2] - b[0]) / 2, 1), "state": "open", "visits": v, "dwell_s": dwell,
                        "deficit": max(target_visits - v, 0)})
    return {"field_id": clip_id, "source": "truth", "units": "norm1000", "target_visits": target_visits,
            "visits_scope": f"station {station}, day {day} (whole day, from dataset)",
            "station": int(station), "flowers": flowers}


def bee_dwell_s(clip_ids):
    """Median bee dwell per visit across truth events: the robot's time on each flower."""
    dwells = sorted(e["t_end"] - e["t_start"] for c in clip_ids
                    for e in json.loads(Path(f"truth/{c}.json").read_text())["events"] if e["t_end"] > e["t_start"])
    return round(dwells[len(dwells) // 2], 2) if dwells else 3.0
