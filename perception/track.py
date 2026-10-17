"""P1: one-bee tracking with Perceptron -> tracks/parts/<clip>.track.json.

    python -m perception.track --clips hb1_w00 sm3_12_123903     # a few clips
    python -m perception.track --all --merge                     # 48-clip batch, then tracks/*.json
    python -m perception.track --all --resolution 1080p          # needs raw/ sources (A2 test)

Reruns are free: the shared client serves every repeated request from tracks/raw/. The part file holds one
CONTRACT.md track (``<species>_1``) plus run metadata; ``merge.py`` turns parts into ``tracks/<clip>.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from perception.client import ROOT, PerceptionClient

PARTS_DIR = ROOT / "tracks" / "parts"
PROMPTS_DIR = ROOT / "prompts" / "track"
PIECE = "p1"

# how the prompt names each species (truth/*.json says which one is in the clip)
SPECIES_WORDS = {"honeybee": "honeybee", "vespidae": "wasp", "lepidoptera": "butterfly or moth",
                 "syrphidae": "hoverfly", "unknown": "insect"}


def manifest(root: Path = ROOT) -> dict[str, dict]:
    return {e["clip_id"]: e for e in json.loads((root / "clips" / "manifest.json").read_text())}


def species_for(clip_id: str, root: Path = ROOT) -> str:
    """Species from the truth file when there is one (it only tells the prompt what to look for)."""
    truth = root / "truth" / f"{clip_id}.json"
    if truth.exists():
        tracks = json.loads(truth.read_text()).get("tracks") or []
        if tracks and tracks[0].get("species"):
            return tracks[0]["species"]
    return "honeybee" if clip_id.startswith("hb") else "unknown"


def clip_path(clip_id: str, resolution: str = "720p", root: Path = ROOT) -> Path:
    entry = manifest(root)[clip_id]
    if resolution == "720p":
        return root / entry["path"]
    if resolution == "1080p":
        path = root / "clips" / "1080p" / f"{clip_id}.mp4"
        if not path.exists():
            cut_native(clip_id, path, root)
        return path
    raise ValueError(f"resolution must be 720p or 1080p, not {resolution}")


def cut_native(clip_id: str, dest: Path, root: Path = ROOT) -> Path:
    """Cut the clip's window from the native source under raw/ (needs the dataset download and ffmpeg)."""
    entry = manifest(root)[clip_id]
    sources = list((root / "raw").rglob(entry["source_video"])) if (root / "raw").exists() else []
    if not sources:
        raise FileNotFoundError(f"{entry['source_video']} not found under raw/; run scripts/fetch_data.py first")
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not on PATH (winget install Gyan.FFmpeg, then open a new shell)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{entry['source_start_s']:.3f}", "-i", str(sources[0]),
                    "-t", f"{entry['duration_s']:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
                    "-movflags", "+faststart", str(dest)], check=True)
    return dest


def load_prompt(version: str, **fields) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text().format(**fields).strip()


def parse_track(text: str, duration_s: float, species: str, *, complete: bool = True) -> dict:
    """The answer's markup as one CONTRACT.md track. Waypoints outside the clip or malformed are dropped, never
    fixed up; ``complete`` is False when the markup or the answer was cut off."""
    from perceptron import BoundingBox, collect_annotations

    tracks = [t for t in collect_annotations(text or "").tracks if t.points]
    track = max(tracks, key=lambda t: len(t.points)) if tracks else None
    points = []
    if track is not None:
        for wp in track.points:
            if not isinstance(wp, BoundingBox) or wp.t is None:
                continue
            t = float(wp.t)
            box = [wp.top_left.x, wp.top_left.y, wp.bottom_right.x, wp.bottom_right.y]
            if not (0 <= t <= duration_s + 1e-3) or not all(0 <= v <= 1000 for v in box):
                continue
            if box[0] > box[2] or box[1] > box[3]:
                continue
            points.append({"t": round(t, 3), "box": box,
                           "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                           "interpolated": False})
    points.sort(key=lambda p: p["t"])
    deduped = [p for i, p in enumerate(points) if i == 0 or p["t"] > points[i - 1]["t"]]
    return {"track_id": f"{species}_1", "species": species,
            "complete": bool(complete and (track is None or track.complete)), "points": deduped}


def interpolate(points: list[dict], max_gap_s: float = 1.0, step_s: float = 0.1) -> list[dict]:
    """Fill gaps up to ``max_gap_s`` between observed points with linear boxes flagged ``interpolated`` (eval skips
    them). Longer gaps stay gaps: the bee was probably occluded or off screen."""
    out = []
    for i, p in enumerate(points):
        out.append(p)
        if i + 1 >= len(points):
            break
        q = points[i + 1]
        gap = q["t"] - p["t"]
        if gap <= step_s + 1e-9 or gap > max_gap_s:
            continue
        n = int(gap / step_s + 1e-9)
        for k in range(1, n + (0 if abs(n * step_s - gap) < 1e-9 else 1)):
            t = p["t"] + k * step_s
            if t >= q["t"] - 1e-9:
                break
            a = (t - p["t"]) / gap
            box = [round(p["box"][j] + a * (q["box"][j] - p["box"][j]), 1) for j in range(4)]
            out.append({"t": round(t, 3), "box": box,
                        "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                        "interpolated": True})
    return out


def track_clip(clip_id: str, *, version: str = "v1", resolution: str = "720p", client: PerceptionClient | None = None,
               max_tokens: int = 4096, reasoning_effort: str | None = None, step_s: float = 0.5,
               root: Path = ROOT, parts_dir: Path | None = None) -> dict:
    client = client or PerceptionClient(PIECE)
    entry = manifest(root)[clip_id]
    species = species_for(clip_id, root)
    prompt = load_prompt(version, species=SPECIES_WORDS.get(species, species), duration_s=entry["duration_s"],
                         step_s=step_s)
    resp = client.call(clip_id=clip_id, prompt_name="track", prompt_version=version, prompt_text=prompt,
                       video_path=clip_path(clip_id, resolution, root), annotation_format="box",
                       max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    track = parse_track(resp.text, entry["duration_s"], species, complete=resp.complete)
    part = {"clip_id": clip_id, "model": resp.model or client.model, "prompt_version": version,
            "resolution": resolution, "finish_reason": resp.finish_reason, "cache_hit": resp.cache_hit,
            "cost_usd": resp.cost_usd, "cache_file": str(resp.cache_path.relative_to(client.cache_dir)).replace("\\", "/"),
            "parse_errors": resp.annotations().errors, "tracks": [track]}
    parts_dir = parts_dir or PARTS_DIR
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / f"{clip_id}.track.json").write_text(json.dumps(part, indent=1))
    return part


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[], help="clip ids (default: none)")
    ap.add_argument("--all", action="store_true", help="every clip in clips/manifest.json")
    ap.add_argument("--version", default="v1", help="prompt version: prompts/track/<version>.txt")
    ap.add_argument("--resolution", default="720p", choices=["720p", "1080p"])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reasoning-effort", default=None, help="e.g. low, medium, high (costs output tokens)")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between boxes the prompt asks for")
    ap.add_argument("--merge", action="store_true", help="also write tracks/<clip>.json for these clips")
    ap.add_argument("--model-info", action="store_true", help="refresh tracks/raw/model_info.json (pricing, limits)")
    args = ap.parse_args(argv)

    client = PerceptionClient(PIECE)
    if args.model_info:
        info = client.model_info()
        print(f"{info['id']}: context {info.get('max_context_tokens')}, max output {info.get('max_output_tokens')}, "
              f"prices/M in={client.prices()['input']} out={client.prices()['output']}")
    clips = list(manifest()) if args.all else args.clips
    if not clips:
        ap.error("give --clips ... or --all")
    print(f"model {client.model} · user {client.user} · spent ${client.spent_usd():.3f} of ${client.cost_cap_usd:.0f}")
    total = 0.0
    for clip_id in clips:
        part = track_clip(clip_id, version=args.version, resolution=args.resolution, client=client,
                          max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort, step_s=args.step)
        tr = part["tracks"][0]
        total += part["cost_usd"]
        flag = "cache" if part["cache_hit"] else f"${part['cost_usd']:.4f}"
        state = "" if tr["complete"] else f"  INCOMPLETE ({part['finish_reason']})"
        print(f"{clip_id:16s} {len(tr['points']):3d} boxes  {flag:>8s}{state}")
    print(f"{client.api_calls} API calls, {client.cache_hits} cache hits, ${total:.4f} this run, "
          f"${client.spent_usd():.3f} total on {client.cost_log.name}")
    if args.merge:
        from perception.merge import merge_clip

        for clip_id in clips:
            merge_clip(clip_id)
        print(f"merged {len(clips)} clips into tracks/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
