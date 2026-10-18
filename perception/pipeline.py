"""Whole pipeline: video -> motion candidates -> Mk1.5 verification -> flowers + visits -> tracks/<clip>.json.

    python -m perception.pipeline --all                    # every clip in the manifest (API if a key is set)
    python -m perception.pipeline --all --no-api           # motion + heuristics only, writes tracks/heuristic/
    python -m perception.pipeline --video my_bees.mp4      # any video: transcoded to 720p30, written to tracks/uploads/

``analyze_video`` is what the app's upload button calls. Every Mk1.5 answer is cached, so a rerun is free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from perception import cv_track, verify, visits
from perception.client import ROOT, PerceptionClient

UPLOADS = ROOT / "clips" / "uploads"


def api_ready() -> bool:
    """Mk1.5 stages run when a key is set. BEEBOT_OFFLINE=1 keeps them on but cache-only (the client raises
    CacheMiss instead of calling), so a demo replays Mk1.5 results with the network off."""
    return bool(os.getenv("PERCEPTRON_API_KEY")) or bool(int(os.getenv("BEEBOT_OFFLINE", "0")))


def run(clip_id: str, *, use_api: bool, video: Path | None = None, out_dir: Path | None = None,
        workers: int = 4, species: str | None = None) -> dict:
    out_dir = out_dir or (ROOT / "tracks" if use_api else ROOT / "tracks" / "heuristic")
    stage = out_dir / "_stages"
    cv_track.run_clip(clip_id, out_dir=stage / "cv", video=video)
    client_v = PerceptionClient(verify.PIECE) if use_api else None
    verify.verify_clip(clip_id, client=client_v, workers=workers, cv_dir=stage / "cv", out_dir=stage / "verified",
                       video=video, species=species)
    client_f = PerceptionClient(visits.PIECE) if use_api else None
    return visits.finish_clip(clip_id, client=client_f, video=video, in_dir=stage / "verified", out_dir=out_dir)


def analyze_video(path: Path, *, use_api: bool | None = None, species: str = "unknown") -> tuple[str, Path, dict]:
    """Transcode an uploaded video to the working format and run the pipeline. Returns (clip_id, mp4, doc)."""
    use_api = api_ready() if use_api is None else use_api
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]
    clip_id = f"up_{h}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    mp4 = UPLOADS / f"{clip_id}.mp4"
    if not mp4.exists():
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        # 720p, 30 fps, at most 60 s: the motion tracker holds the clip in memory, Mk1.5 sees crops only
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(path), "-t", "60", "-vf", "scale=-2:720,fps=30",
                        "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", "-an",
                        "-movflags", "+faststart", str(mp4)], check=True)
    out_dir = ROOT / "tracks" / "uploads"
    doc = run(clip_id, use_api=use_api, video=mp4, out_dir=out_dir, species=species)
    return clip_id, mp4, doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--species", default="unknown", help="for --video: honeybee, vespidae, syrphidae, lepidoptera")
    args = ap.parse_args(argv)
    use_api = api_ready() and not args.no_api
    if args.video:
        cid, mp4, doc = analyze_video(Path(args.video), use_api=use_api, species=args.species)
        print(f"{cid}: {len(doc['tracks'])} insects, {len(doc['events'])} visits -> tracks/uploads/{cid}.json")
        return 0
    manifest = json.loads((ROOT / "clips" / "manifest.json").read_text())
    clips = [e["clip_id"] for e in manifest] if args.all else args.clips
    print(f"Mk1.5 {'ON' if use_api else 'OFF (heuristic)'}")
    for cid in clips:
        doc = run(cid, use_api=use_api, workers=args.workers)
        print(f"{cid:16s} {len(doc['tracks'])} insects, {len(doc['flowers'])} flowers, {len(doc['events'])} visits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
