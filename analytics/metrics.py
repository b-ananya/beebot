"""Pollinator analytics from one CONTRACT.md tracks document. Pure functions, no I/O: the app and the summary
script both call ``clip_metrics``.

Units: the contract's 0-1000 coordinates stretch y (1000 units = 1280 px across but 720 px down), so every
distance here is measured after scaling y by the frame aspect (H/W): one unit = 1/1000 of the frame width in both
directions. ``SCALE`` (units per cm) turns those into cm.
Every number is computed from observed points only (interpolated points are skipped).

What a robot pollinator / simulator needs, and where it comes from:
- dwell per visit             -> how long the robot should hold at a flower
- transit time, distance, speed between flowers -> motion budget between stops
- nearest-flower rate          -> does the insect choose greedily? (baseline policy for the planner)
- revisit rate and interval    -> how soon a flower is worth visiting again
- approach time                -> slow-down / hover before landing
- turn angles, tortuosity      -> flight-path shape for trajectory generation
- foraging vs flying share     -> duty cycle
- coverage and interpolation share -> how much to trust all of the above
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np

# 0-1000 units per cm, approximate. Honeybee clips: the folding ruler in frame (marks 62-81 cm span ~440 px at
# 1280 px). Farm clips: median detected strawberry flower ~ 2.5 cm across. Both are labelled approximate in the UI.
SCALE = {"hb": 18.0, "sm": None}
STRAWBERRY_FLOWER_CM = 2.5


def center(p: dict) -> tuple[float, float]:
    if p.get("center"):
        return float(p["center"][0]), float(p["center"][1])
    b = p["box"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def fcenter(f: dict) -> tuple[float, float]:
    b = f["box"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def scale_for(doc: dict) -> tuple[float | None, str]:
    """(units per cm, how it was estimated)."""
    cid = doc.get("clip_id", "")
    if cid.startswith("hb"):
        return SCALE["hb"], "folding ruler in frame (approx.)"
    sizes = [max(f["box"][2] - f["box"][0], f["box"][3] - f["box"][1])
             for f in doc.get("flowers", []) if f.get("source", "mk1.5") != "stop"]
    if sizes:
        return float(np.median(sizes)) / STRAWBERRY_FLOWER_CM, f"median flower = {STRAWBERRY_FLOWER_CM} cm (approx.)"
    return None, "no scale"


def track_motion(points: list[dict]) -> dict:
    obs = [p for p in points if not p.get("interpolated")]
    if len(obs) < 2:
        return {"observed_s": 0.0, "path_units": 0.0, "speeds": [], "turns_deg": [], "tortuosity": None}
    ts = np.array([p["t"] for p in obs])
    cs = np.array([center(p) for p in obs])
    # subsample to ~5 Hz so tracker jitter doesn't inflate path length and turn angles
    keep = [0]
    for i in range(1, len(ts)):
        if ts[i] - ts[keep[-1]] >= 0.2:
            keep.append(i)
    ts, cs = ts[keep], cs[keep]
    seg = np.diff(cs, axis=0)
    dt = np.diff(ts)
    step = np.linalg.norm(seg, axis=1)
    ok = dt < 0.6                                  # don't measure speed across tracking gaps
    speeds = (step[ok] / dt[ok]).tolist()
    turns = []
    for a, b, oa, ob in zip(seg[:-1], seg[1:], ok[:-1], ok[1:]):
        if oa and ob and np.linalg.norm(a) > 3 and np.linalg.norm(b) > 3:
            ang = math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b))
            turns.append(ang)
    net = float(np.linalg.norm(cs[-1] - cs[0]))
    path = float(step.sum())
    return {"observed_s": float(sum(dt[ok])), "path_units": path, "speeds": speeds, "turns_deg": turns,
            "tortuosity": round(path / net, 2) if net > 5 else None}


def isotropic(doc: dict, aspect: float) -> dict:
    """Copy of doc with every y coordinate multiplied by aspect (H/W)."""
    doc = copy.deepcopy(doc)
    for tr in doc.get("tracks", []):
        for p in tr["points"]:
            p["box"] = [p["box"][0], p["box"][1] * aspect, p["box"][2], p["box"][3] * aspect]
            if p.get("center"):
                p["center"] = [p["center"][0], p["center"][1] * aspect]
    for f in doc.get("flowers", []):
        f["box"] = [f["box"][0], f["box"][1] * aspect, f["box"][2], f["box"][3] * aspect]
    return doc


FLIGHT_SPEED = 40.0     # units/s; slower than this is crawling or sitting


def clip_metrics(doc: dict, duration_s: float | None = None, aspect: float = 720 / 1280) -> dict:
    doc = isotropic(doc, aspect)
    flowers = {f["flower_id"]: f for f in doc.get("flowers", [])}
    visits = sorted([e for e in doc.get("events", []) if e["type"] == "visit" and e["flower_id"] in flowers],
                    key=lambda e: (e["track_id"], e["t_start"]))
    scale, scale_note = scale_for(doc)

    # ---- per track motion
    tracks = {}
    labels = defaultdict(float)
    n_obs = n_interp = 0
    for tr in doc.get("tracks", []):
        m = track_motion(tr["points"])
        tracks[tr["track_id"]] = {"species": tr.get("species", "unknown"), **m}
        pts = tr["points"]
        n_obs += sum(not p.get("interpolated") for p in pts)
        n_interp += sum(bool(p.get("interpolated")) for p in pts)
        for a, b in zip(pts[:-1], pts[1:]):
            if b["t"] - a["t"] < 0.6 and a.get("label"):
                labels[a["label"]] += b["t"] - a["t"]

    # ---- per visit, transitions, revisits, nearest-flower choice
    visit_rows, transitions = [], []
    seen: dict[str, dict[str, float]] = defaultdict(dict)        # track -> flower -> last visit end
    revisit_intervals = []
    nearest_hits = nearest_total = 0
    approach = []
    by_track = defaultdict(list)
    for v in visits:
        by_track[v["track_id"]].append(v)
    track_pts = {tr["track_id"]: [p for p in tr["points"] if not p.get("interpolated")] for tr in doc.get("tracks", [])}
    for tid, vs in by_track.items():
        for i, v in enumerate(vs):
            dwell = v["t_end"] - v["t_start"]
            revisit = v["flower_id"] in seen[tid]
            if revisit:
                revisit_intervals.append(v["t_start"] - seen[tid][v["flower_id"]])
            visit_rows.append({"track_id": tid, "flower_id": v["flower_id"], "t_start": v["t_start"],
                               "t_end": v["t_end"], "dwell_s": round(dwell, 2), "revisit": revisit})
            # approach: time spent within 3 flower-radii before entering, moving slower than half median speed
            fc = fcenter(flowers[v["flower_id"]])
            fb = flowers[v["flower_id"]]["box"]
            r = max(fb[2] - fb[0], fb[3] - fb[1]) / 2
            before = [p for p in track_pts.get(tid, []) if v["t_start"] - 2.0 <= p["t"] < v["t_start"]]
            near = [p for p in before if math.dist(center(p), fc) <= 3 * r + 15]
            if before:
                approach.append(round(v["t_start"] - near[0]["t"], 2) if near else 0.0)
            if i + 1 < len(vs):
                nxt = vs[i + 1]
                a, b = fcenter(flowers[v["flower_id"]]), fcenter(flowers[nxt["flower_id"]])
                dist = math.dist(a, b)
                dt = nxt["t_start"] - v["t_end"]
                transitions.append({"track_id": tid, "from": v["flower_id"], "to": nxt["flower_id"],
                                    "transit_s": round(dt, 2), "distance_units": round(dist, 1),
                                    "speed_units_s": round(dist / dt, 1) if dt > 0.05 else None})
                others = [f for fid, f in flowers.items() if fid != v["flower_id"]]
                if len(others) >= 2 and nxt["flower_id"] != v["flower_id"]:
                    nearest = min(others, key=lambda f: math.dist(fcenter(f), a))
                    nearest_total += 1
                    nearest_hits += nearest["flower_id"] == nxt["flower_id"]
            seen[tid][v["flower_id"]] = v["t_end"]

    # ---- per flower
    per_flower = []
    for fid, f in flowers.items():
        vs = [r for r in visit_rows if r["flower_id"] == fid]
        per_flower.append({"flower_id": fid, "source": f.get("source", "mk1.5"), "visits": len(vs),
                           "dwell_s": round(sum(r["dwell_s"] for r in vs), 2),
                           "first_visit_s": min((r["t_start"] for r in vs), default=None),
                           "visitors": sorted({r["track_id"] for r in vs}), "center": fcenter(f)})

    dwell = [r["dwell_s"] for r in visit_rows]
    speeds = [s for t in tracks.values() for s in t["speeds"]]
    turns = [a for t in tracks.values() for a in t["turns_deg"]]
    observed = sum(t["observed_s"] for t in tracks.values())          # insect-seconds (adds up across insects)
    # time with at least one insect tracked: union of observed stretches (gaps < 0.6 s) over all tracks
    spans = sorted((a["t"], b["t"]) for tr in doc.get("tracks", [])
                   for a, b in zip([p for p in tr["points"] if not p.get("interpolated")][:-1],
                                   [p for p in tr["points"] if not p.get("interpolated")][1:])
                   if b["t"] - a["t"] < 0.6)
    covered, cur = 0.0, None
    for a, b in spans:
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            covered += (cur[1] - cur[0]) if cur else 0.0
            cur = [a, b]
    covered += (cur[1] - cur[0]) if cur else 0.0

    def med(x):
        return round(float(np.median(x)), 2) if len(x) else None

    def cm(x):
        return round(x / scale, 2) if (x is not None and scale) else None

    detected = [f for f in flowers.values() if f.get("source", "mk1.5") != "stop"]
    visited_detected = {r["flower_id"] for r in visit_rows if flowers[r["flower_id"]].get("source", "mk1.5") != "stop"}
    summary = {
        "insects": len(tracks),
        "species": sorted({t["species"] for t in tracks.values()}),
        "visits": len(visit_rows),
        "flowers_detected": len(detected),
        "flowers_visited": len({r["flower_id"] for r in visit_rows}),
        "flowers_never_visited": len(detected) - len(visited_detected),
        "visits_per_min": round(len(visit_rows) / (duration_s / 60), 2) if duration_s else None,
        "median_dwell_s": med(dwell),
        "mean_dwell_s": round(float(np.mean(dwell)), 2) if dwell else None,
        "median_transit_s": med([t["transit_s"] for t in transitions]),
        "median_transit_units": med([t["distance_units"] for t in transitions]),
        "median_speed_units_s": med(speeds),
        "p90_speed_units_s": round(float(np.percentile(speeds, 90)), 1) if speeds else None,
        "median_speed_cm_s": cm(med(speeds)),
        "median_flight_speed_units_s": med([v for v in speeds if v >= FLIGHT_SPEED]),
        "median_flight_speed_cm_s": cm(med([v for v in speeds if v >= FLIGHT_SPEED])),
        "flying_share_by_speed": round(sum(v >= FLIGHT_SPEED for v in speeds) / len(speeds), 2) if speeds else None,
        "nearest_flower_rate": round(nearest_hits / nearest_total, 2) if nearest_total else None,
        "revisit_rate": round(sum(r["revisit"] for r in visit_rows) / len(visit_rows), 2) if visit_rows else None,
        "median_revisit_interval_s": med(revisit_intervals),
        "median_approach_s": med(approach),
        "median_abs_turn_deg": med([abs(a) for a in turns]),
        "foraging_share": round(labels.get("foraging", 0) / sum(labels.values()), 2) if sum(labels.values()) > 0 else None,
        "tracked_s": round(observed, 2),
        "tracked_share": round(min(covered / duration_s, 1.0), 2) if duration_s else None,
        "interpolated_share": round(n_interp / (n_obs + n_interp), 3) if (n_obs + n_interp) else None,
        "scale_units_per_cm": round(scale, 2) if scale else None,
        "scale_note": scale_note,
    }
    return {"summary": summary, "visits": visit_rows, "transitions": transitions, "flowers": per_flower,
            "tracks": {k: {kk: vv for kk, vv in v.items() if kk not in ("speeds", "turns_deg")} for k, v in tracks.items()},
            "speeds": speeds, "turns_deg": turns, "behavior_s": dict(labels), "aspect": aspect}


def heatmap(doc: dict, bins: int = 40) -> np.ndarray:
    """Time-weighted occupancy grid (rows = y), observed points only."""
    grid = np.zeros((bins, bins))
    for tr in doc.get("tracks", []):
        for p in tr["points"]:
            if p.get("interpolated"):
                continue
            x, y = center(p)
            grid[min(int(y / 1000 * bins), bins - 1), min(int(x / 1000 * bins), bins - 1)] += 1
    return grid
