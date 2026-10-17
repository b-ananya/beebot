"""Check contract files before anyone else loads them. Stdlib only.

    python -I scripts/validate.py                    # truth/ and tracks/ (incl. tracks/mock/)
    python -I scripts/validate.py tracks/hb1_w00.json
Exit code 1 if anything fails, so it can run as a pre-commit hook.
"""
import json
import sys
from pathlib import Path

EVENT_TYPES = {"visit", "landing", "hover", "takeoff"}


def check_tracks_doc(doc, duration):
    errs = []
    for key in ("clip_id", "model", "tracks", "flowers", "events"):
        if key not in doc:
            errs.append(f"missing '{key}'")
    if errs:
        return errs

    def box_ok(b, where):
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            errs.append(f"{where}: box must be 4 numbers")
        elif not all(0 <= v <= 1000 for v in b):
            errs.append(f"{where}: box outside 0-1000 {b}")
        elif b[0] > b[2] or b[1] > b[3]:
            errs.append(f"{where}: box not ordered [x1,y1,x2,y2] {b}")

    track_ids = set()
    for tr in doc["tracks"]:
        tid = tr.get("track_id")
        track_ids.add(tid)
        last = -1.0
        for i, p in enumerate(tr.get("points", [])):
            where = f"{tid} point {i}"
            t = p.get("t")
            if not isinstance(t, (int, float)):
                errs.append(f"{where}: t missing")
                continue
            if duration is not None and not 0 <= t <= duration + 1e-3:
                errs.append(f"{where}: t={t} outside clip duration {duration}")
            if t < last:
                errs.append(f"{where}: points not sorted by t")
            last = t
            box_ok(p.get("box"), where)
            if "center" in p and not all(0 <= v <= 1000 for v in p["center"]):
                errs.append(f"{where}: center outside 0-1000")
    flower_ids = set()
    for f in doc["flowers"]:
        flower_ids.add(f.get("flower_id"))
        box_ok(f.get("box"), f"flower {f.get('flower_id')}")
    for i, e in enumerate(doc["events"]):
        where = f"event {i}"
        if e.get("type") not in EVENT_TYPES:
            errs.append(f"{where}: type '{e.get('type')}' not in {sorted(EVENT_TYPES)}")
        if e.get("track_id") not in track_ids:
            errs.append(f"{where}: unknown track_id {e.get('track_id')}")
        if e.get("flower_id") not in flower_ids:
            errs.append(f"{where}: unknown flower_id {e.get('flower_id')}")
        if not e.get("t_start", 0) <= e.get("t_end", -1):
            errs.append(f"{where}: t_start > t_end")
    return errs


def main():
    manifest = {}
    if Path("clips/manifest.json").exists():
        manifest = {e["clip_id"]: e for e in json.loads(Path("clips/manifest.json").read_text())}
    paths = [Path(p) for p in sys.argv[1:]] or [
        p for d in ("truth", "tracks", "tracks/mock") for p in sorted(Path(d).glob("*.json"))]
    bad = 0
    for path in paths:
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"FAIL {path}: invalid json ({e})")
            bad += 1
            continue
        entry = manifest.get(doc.get("clip_id"))
        errs = check_tracks_doc(doc, entry["duration_s"] if entry else None)
        if not entry:
            errs.append(f"clip_id '{doc.get('clip_id')}' not in clips/manifest.json")
        if errs:
            bad += 1
            print(f"FAIL {path}")
            for e in errs[:10]:
                print(f"   {e}")
    print(f"{len(paths) - bad}/{len(paths)} files valid")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
