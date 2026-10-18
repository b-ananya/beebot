"""Combine perception parts into tracks/<clip>.json, the file CONTRACT.md and validate.py expect.

    python -m perception.merge --all                 # every clip with a track part
    python -m perception.merge --clips hb1_w00 --interpolate 1.0

Parts live in tracks/parts/ (a contract addition; each owner writes only their own):

    <clip>.track.json    P1 (Raika)    {"clip_id", "model", "prompt_version", "tracks": [...]}
    <clip>.flowers.json  P2 (Falisha)  {"clip_id", "flowers": [{"flower_id", "box"}]}
    <clip>.events.json   P2 (Falisha)  {"clip_id", "events": [{"track_id", "flower_id", "type", "t_start", "t_end"}]}

Events must name a track_id from the track part (``<species>_1``, e.g. ``honeybee_1``) and a flower_id from the
flowers part; others are dropped with a warning. Missing flowers or events parts mean empty lists.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from perception.client import ROOT
from perception.track import PARTS_DIR, interpolate

TRACKS_DIR = ROOT / "tracks"


def _validator(root: Path):
    spec = importlib.util.spec_from_file_location("beebot_validate", root / "scripts" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_tracks_doc


def load_part(clip_id: str, kind: str, parts_dir: Path = PARTS_DIR) -> dict | None:
    path = parts_dir / f"{clip_id}.{kind}.json"
    return json.loads(path.read_text()) if path.exists() else None


def merge_clip(clip_id: str, *, interpolate_gap_s: float | None = None, root: Path = ROOT,
               parts_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    parts_dir = parts_dir or (root / "tracks" / "parts")
    out_dir = out_dir or (root / "tracks")
    track_part = load_part(clip_id, "track", parts_dir)
    if track_part is None:
        raise FileNotFoundError(f"{parts_dir / (clip_id + '.track.json')} missing: run perception.track first")
    flowers = (load_part(clip_id, "flowers", parts_dir) or {}).get("flowers") or []
    events = (load_part(clip_id, "events", parts_dir) or {}).get("events") or []

    tracks = []
    for tr in track_part["tracks"]:
        tr = dict(tr)
        if interpolate_gap_s:
            tr["points"] = interpolate(tr["points"], max_gap_s=interpolate_gap_s)
        tracks.append(tr)
    track_ids = {t["track_id"] for t in tracks}
    flower_ids = {f["flower_id"] for f in flowers}
    kept = []
    for e in events:
        if e.get("track_id") in track_ids and e.get("flower_id") in flower_ids:
            kept.append(e)
        else:
            print(f"warning: {clip_id}: dropping event {e} (unknown track_id or flower_id)")

    doc = {"clip_id": clip_id, "model": track_part["model"], "prompt_version": track_part.get("prompt_version"),
           "tracks": tracks, "flowers": flowers, "events": kept}
    entry = {e["clip_id"]: e for e in json.loads((root / "clips" / "manifest.json").read_text())}.get(clip_id)
    errs = _validator(root)(doc, entry["duration_s"] if entry else None)
    if errs:
        raise ValueError(f"{clip_id}: merged doc fails the contract: {errs[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{clip_id}.json"
    out.write_text(json.dumps(doc, indent=1), newline="\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every clip that has a track part")
    ap.add_argument("--interpolate", type=float, default=None, metavar="GAP_S",
                    help="fill gaps up to GAP_S seconds with interpolated boxes (flagged, eval skips them)")
    args = ap.parse_args(argv)
    clips = sorted(p.name[: -len(".track.json")] for p in PARTS_DIR.glob("*.track.json")) if args.all else args.clips
    if not clips:
        ap.error("give --clips ... or --all")
    for clip_id in clips:
        out = merge_clip(clip_id, interpolate_gap_s=args.interpolate)
        doc = json.loads(out.read_text())
        print(f"{out.relative_to(ROOT)}: {sum(len(t['points']) for t in doc['tracks'])} points, "
              f"{len(doc['flowers'])} flowers, {len(doc['events'])} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
