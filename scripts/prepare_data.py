"""Turn raw Monash downloads into the shared working set:
clips/*.mp4 + clips/manifest.json, truth/*.json, tracks/mock/*.json.

Stdlib + ffmpeg only, so it runs before anyone has pip-installed anything.
Run from the repo root:  python -I scripts/prepare_data.py [--skip-video]

Sources (both CC BY 4.0, cite on the attribution slide):
  hb_*  Honeybee video tracking data, Experiment_1 (DOI 10.26180/5f4c8d5815940).
        Truth = HyDaT tracker output (a reference tracker, not hand labels).
  sm_*  Spatial Monitoring and Insect Behavioural Analysis (DOI 10.26180/21533760).
        Truth = Polytrack output, plus flower positions and flower visits.
"""
import argparse
import csv
import json
import math
import random
import re
import shutil
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

RAW = Path("raw")
CLIPS, TRUTH, MOCK = Path("clips"), Path("truth"), Path("tracks/mock")
OUT_H, OUT_FPS = 720, 30
WINDOW_S = 10.0
MIN_WINDOW_S = 5.0

HB_VIDEOS = RAW / "honeybee/Experiment_1-Test_Videos/Experiment_1-Test_Videos"
HB_TRACKS = RAW / "honeybee/Experiment_1-CSV_HyDaT/Experiment_1-CSV_HyDaT"
SM_VIDEOS = RAW / "spatial_monitoring/videos"
SM_TRACKS = RAW / "spatial_monitoring/trajectories/Extracted_Tracks"
SM_VIDEO_FRAMES = 18000  # each Spatial video is 10 min at 30 fps; absframe counts across the day
SPECIES = {0: "honeybee", 1: "syrphidae", 2: "lepidoptera", 3: "vespidae"}

LICENSE = "CC BY 4.0, Monash University (cite dataset DOI)"


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"],
            "fps": float(num) / float(den), "duration_s": float(d["format"]["duration"])}


def cut_clip(src, start, dur, dest, skip):
    if skip and dest.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
         "-vf", f"scale=-2:{OUT_H}", "-r", str(OUT_FPS),
         "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p", "-an",
         str(dest)], check=True)


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def norm_box(cx, cy, side, w, h):
    """Center + square side in source pixels -> [x1,y1,x2,y2] in 0-1000, clamped."""
    x1, x2 = (cx - side / 2) / w * 1000, (cx + side / 2) / w * 1000
    y1, y2 = (cy - side / 2) / h * 1000, (cy + side / 2) / h * 1000
    c = lambda v: round(min(max(v, 0), 1000), 1)
    return [c(x1), c(y1), c(x2), c(y2)]


def side_from_area(area, fallback):
    """Trackers give blob area, not a box. Approximate a square box around the blob."""
    a = num(area)
    side = math.sqrt(a) * 1.6 if a and a > 0 else fallback
    return min(max(side, 16), 200)


def point(t, cx, cy, side, w, h):
    return {"t": round(t, 3),
            "center": [round(cx / w * 1000, 1), round(cy / h * 1000, 1)],
            "box": norm_box(cx, cy, side, w, h),
            "interpolated": False}


def truth_doc(clip_id, tracks, flowers=(), events=()):
    return {"clip_id": clip_id, "model": "ground_truth", "prompt_version": None,
            "tracks": tracks, "flowers": list(flowers), "events": list(events)}


# ---------- Honeybee Experiment_1: one bee per video, 60 fps, HyDaT reference tracks

def honeybee(skip_video):
    entries = []
    for video in sorted(HB_VIDEOS.glob("bee_test_*.mp4")):
        n = int(re.search(r"(\d+)", video.stem).group(1))
        meta = ffprobe(video)
        w, h, fps = meta["width"], meta["height"], meta["fps"]
        rows = list(csv.DictReader(open(HB_TRACKS / f"bee_test_HyDaT_{n}.csv")))
        areas = [num(r["area"]) for r in rows if num(r["area"])]
        fallback = math.sqrt(statistics.median(areas)) * 1.6 if areas else 40
        # Only frames the tracker actually recorded; "Non" rows are occlusions and stay gaps.
        obs = [(int(float(r["nframe"])), num(r["x0"]), num(r["y0"]), r["area"])
               for r in rows if r["method"].startswith("Recorded") and num(r["x0"]) is not None]

        start = 0.0
        k = 0
        while meta["duration_s"] - start >= MIN_WINDOW_S:
            dur = min(WINDOW_S, meta["duration_s"] - start)
            clip_id = f"hb{n}_w{k:02d}"
            pts = []
            for fr, x, y, area in obs:
                t = (fr - 1) / fps - start  # nframe is 1-indexed
                if 0 <= t < dur:
                    pts.append(point(t, x, y, side_from_area(area, fallback), w, h))
            track = {"track_id": "bee_1", "species": "honeybee", "complete": True,
                     "box_from": "blob_area", "points": pts}
            write_json(TRUTH / f"{clip_id}.json", truth_doc(clip_id, [track]))
            cut_clip(video, start, dur, CLIPS / f"{clip_id}.mp4", skip_video)
            entries.append(manifest_entry(clip_id, dur, f"Monash honeybee Experiment_1 {video.name}",
                                          video, start, len(pts) > 0, "HyDaT tracker output"))
            start += WINDOW_S
            k += 1
    return entries


# ---------- Spatial Monitoring: strawberry farm, 30 fps, Polytrack tracks + flowers + visits

def video_start(name):
    m = re.match(r"cam_(\d+)_[NS]_video_(\d{8})_(\d{6})", name)
    return int(m.group(1)), datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")


def load_insect_track(path):
    rows = list(csv.DictReader(open(path)))
    key = lambda prefix: next(c for c in rows[0] if c.startswith(prefix))
    k = {p: key(p) for p in ("nframe", "absframe", "x0_", "y0_", "area", "flower", "visit_num")}
    return [{"absframe": int(float(r[k["absframe"]])), "x": num(r[k["x0_"]]), "y": num(r[k["y0_"]]),
             "area": r[k["area"]], "flower": num(r[k["flower"]])} for r in rows]


def visits_from(rows, fps, start_s, dur, track_id):
    """Contiguous runs where the track is on a flower become visit events."""
    events, run = [], None
    for r in rows + [{"flower": None, "t": None}]:
        f = r["flower"]
        if run and f != run["f"]:
            t0, t1 = max(run["t0"], 0), min(run["t1"], dur)
            if t1 > t0:
                events.append({"track_id": track_id, "flower_id": f"flower_{int(run['f'])}",
                               "type": "visit", "t_start": round(t0, 3), "t_end": round(t1, 3)})
            run = None
        if f is not None and r["t"] is not None:
            if run is None:
                run = {"f": f, "t0": r["t"], "t1": r["t"]}
            run["t1"] = r["t"] + 1 / fps
    return events


def spatial(skip_video):
    entries = []
    for video in sorted(SM_VIDEOS.glob("*.avi")):
        cam, t_video = video_start(video.name)
        meta = ffprobe(video)
        w, h, fps = meta["width"], meta["height"], meta["fps"]
        day = t_video.strftime("%d")
        prefix = f"{cam}{day}"
        flowers_csv = SM_TRACKS / "Flower_Positions_CSV" / f"flowers_{prefix}.csv"
        flowers = [{"flower_id": f"flower_{int(float(r['flower_num']))}",
                    "box": norm_box(float(r["x0"]), float(r["y0"]), 2 * float(r["radius"]), w, h)}
                   for r in csv.DictReader(open(flowers_csv))
                   if num(r["flower_num"]) is not None and num(r["x0"]) is not None
                   ] if flowers_csv.exists() else []

        for path in sorted((SM_TRACKS / "Insect_Tracks_CSV").glob(f"{prefix}*.csv")):
            stem = path.stem  # PDDHHMMSS0I
            t_track = datetime.strptime(t_video.strftime("%Y%m%d") + stem[3:9], "%Y%m%d%H%M%S")
            coarse = (t_track - t_video).total_seconds()
            if not 0 <= coarse < meta["duration_s"]:
                continue
            rows = load_insect_track(path)
            # absframe counts frames across the day's 10-min videos; its remainder gives a
            # sub-second offset. Fall back to the filename time if the two disagree.
            fine = (rows[0]["absframe"] % SM_VIDEO_FRAMES) / fps
            offset = fine if abs(fine - coarse) < 2 else coarse
            for r in rows:
                r["t"] = offset + (r["absframe"] - rows[0]["absframe"]) / fps

            track_len = rows[-1]["t"] - rows[0]["t"]
            start = max(offset - 1.0, 0)
            dur = min(max(track_len + 2.0, MIN_WINDOW_S), 15.0, meta["duration_s"] - start)
            clip_id = f"sm{cam}_{day}_{stem[3:9]}"
            species = SPECIES.get(int(stem[-1]), "unknown")
            areas = [num(r["area"]) for r in rows if num(r["area"])]
            fallback = math.sqrt(statistics.median(areas)) * 1.6 if areas else 30
            pts = [point(r["t"] - start, r["x"], r["y"], side_from_area(r["area"], fallback), w, h)
                   for r in rows if r["x"] is not None and 0 <= r["t"] - start < dur]
            for r in rows:
                r["t"] -= start
            track = {"track_id": f"{species}_1", "species": species,
                     "complete": rows[-1]["t"] <= dur, "box_from": "blob_area", "points": pts}
            events = visits_from(rows, fps, start, dur, track["track_id"])
            write_json(TRUTH / f"{clip_id}.json", truth_doc(clip_id, [track], flowers, events))
            cut_clip(video, start, dur, CLIPS / f"{clip_id}.mp4", skip_video)
            e = manifest_entry(clip_id, dur, f"Monash Spatial Monitoring {video.name}", video, start,
                               True, "Polytrack output + flower positions/visits")
            e["offset_check"] = {"filename_s": coarse, "absframe_s": round(fine, 2)}
            entries.append(e)
    return entries


# ---------- shared

def manifest_entry(clip_id, dur, source, src_video, start, has_truth, truth_kind):
    return {"clip_id": clip_id, "path": f"clips/{clip_id}.mp4", "fps": OUT_FPS,
            "width": round(OUT_H * 16 / 9), "height": OUT_H, "duration_s": round(dur, 3),
            "source": source, "source_video": src_video.name, "source_start_s": round(start, 3),
            "license": LICENSE, "has_truth": has_truth, "truth_kind": truth_kind}


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


def make_mock(truth, rng):
    """Fake model output for C and D: sparse (~5 fps), jittered, with dropouts."""
    doc = json.loads(json.dumps(truth))
    doc["model"], doc["prompt_version"] = "mock", "mock"
    for tr in doc["tracks"]:
        pts, last = [], -1.0
        for p in tr["points"]:
            if p["t"] - last < 0.2 or rng.random() < 0.15:
                continue
            last = p["t"]
            dx, dy = rng.gauss(0, 8), rng.gauss(0, 8)
            p["center"] = [round(min(max(p["center"][0] + dx, 0), 1000), 1),
                           round(min(max(p["center"][1] + dy, 0), 1000), 1)]
            p["box"] = [round(min(max(v + (dx if i % 2 == 0 else dy), 0), 1000), 1)
                        for i, v in enumerate(p["box"])]
            pts.append(p)
        tr["points"] = pts
    for e in doc["events"]:
        # jitter inside the clip, never past the truth end (which is clamped to duration)
        start = max(e["t_start"] + rng.uniform(-0.3, 0.3), 0)
        end = min(e["t_end"] + rng.uniform(-0.3, 0.3), e["t_end"])
        e["t_start"], e["t_end"] = round(min(start, end), 3), round(max(start, end), 3)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video", action="store_true", help="reuse existing clips, rebuild json only")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH (winget install Gyan.FFmpeg, then open a new shell)")
    for d in (CLIPS, TRUTH, MOCK):
        d.mkdir(parents=True, exist_ok=True)

    entries = honeybee(args.skip_video) + spatial(args.skip_video)
    write_json(CLIPS / "manifest.json", entries)

    rng = random.Random(0)
    for e in entries:
        truth = json.loads((TRUTH / f"{e['clip_id']}.json").read_text())
        write_json(MOCK / f"{e['clip_id']}.json", make_mock(truth, rng))

    n_events = sum(len(json.loads((TRUTH / f"{e['clip_id']}.json").read_text())["events"]) for e in entries)
    print(f"{len(entries)} clips, {sum(e['has_truth'] for e in entries)} with truth, {n_events} visit events")


if __name__ == "__main__":
    main()
