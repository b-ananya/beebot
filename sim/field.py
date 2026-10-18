"""Build field/*.json (pollination maps) from the dataset, so C can plan before A and B ship.

Two builders, never mixed in one file (CONTRACT.md):
- patch_from_pipeline (source "mk1.5"): flowers and visits from the pipeline's tracks/<clip>.json. Flower ids are
  stable per camera view, so the map for a clip holds every flower that camera has seen and the visits counted
  over all of its clips up to and including this one: replaying the clips in order shows the deficit filling in.
- patch_from_truth (source "truth"): flower boxes from truth/<clip>.json, visit counts for the whole station-day
  from data/spatial/Flower_Positions_CSV/flowers_<station><day>.csv (Total_visits).
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


def _view_clips(clip_id, tracks_dir):
    view = clip_id.rsplit("_", 1)[0]
    return sorted(p.stem for p in Path(tracks_dir).glob(f"{view}_*.json") if p.stem.rsplit("_", 1)[0] == view)


def patch_from_pipeline(clip_id, target_visits=DEFAULT_TARGET, tracks_dir="tracks"):
    """Field for a clip from pipeline output: the view's flowers so far, visits accumulated through this clip."""
    manifest = {c["clip_id"]: c for c in json.loads(Path("clips/manifest.json").read_text())}
    clips = [c for c in _view_clips(clip_id, tracks_dir) if c <= clip_id]
    boxes, visits, dwell = {}, {}, {}
    observed = 0.0
    for c in clips:
        doc = json.loads((Path(tracks_dir) / f"{c}.json").read_text())
        observed += manifest[c]["duration_s"]
        for f in doc["flowers"]:
            if f.get("source", "mk1.5") == "stop":
                continue
            boxes.setdefault(f["flower_id"], []).append(f["box"])
        for e in doc["events"]:
            if e["type"] == "visit":
                visits[e["flower_id"]] = visits.get(e["flower_id"], 0) + 1
                dwell[e["flower_id"]] = dwell.get(e["flower_id"], 0.0) + e["t_end"] - e["t_start"]
    flowers = []
    for fid in sorted(boxes, key=lambda k: int(k.split("_")[1])):
        b = [sum(v) / len(v) for v in zip(*boxes[fid])]
        v = visits.get(fid, 0)
        flowers.append({"flower_id": fid, "center": [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)],
                        "radius": round(max(b[2] - b[0], b[3] - b[1]) / 2, 1), "state": "open", "visits": v,
                        "dwell_s": round(dwell.get(fid, 0.0), 2), "deficit": max(target_visits - v, 0),
                        "seen_in_clips": len(boxes[fid])})
    view = clip_id.rsplit("_", 1)[0]
    return {"field_id": clip_id, "source": "mk1.5", "units": "norm1000", "target_visits": target_visits,
            "visits_scope": f"{len(clips)} clip(s) of camera view {view} through {clip_id} ({observed:.0f} s observed)",
            "clips": clips, "flowers": flowers}


def pipeline_bee(clip_id, tracks_dir="tracks"):
    """(bee points of the longest track, visit order) from pipeline output, for rendering and follow_bee."""
    doc = json.loads((Path(tracks_dir) / f"{clip_id}.json").read_text())
    points = max((t["points"] for t in doc["tracks"]), key=len, default=[])
    order = [e["flower_id"] for e in sorted(doc["events"], key=lambda e: e["t_start"]) if e["type"] == "visit"]
    return points, order


def pipeline_dwell_s(clip_ids, tracks_dir="tracks"):
    """Median visit dwell in the pipeline output: the robot's time on each flower."""
    dwells = sorted(e["t_end"] - e["t_start"] for c in clip_ids if (Path(tracks_dir) / f"{c}.json").exists()
                    for e in json.loads((Path(tracks_dir) / f"{c}.json").read_text())["events"]
                    if e["type"] == "visit" and e["t_end"] > e["t_start"])
    return round(dwells[len(dwells) // 2], 2) if dwells else 3.0


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
