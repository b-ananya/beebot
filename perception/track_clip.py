"""Run Mk1.5 on clips and write contract-format tracks/<clip_id>.json.

    python -m perception.track_clip hb1_w00              # one clip, bee track
    python -m perception.track_clip hb1_w00 --flowers    # also detect flowers on a reference frame
    python -m perception.track_clip --all                # every clip in the manifest (cached reruns are free)

Prompts live in prompts/<name>_<version>.txt; --version picks them and is recorded in the output.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from perception.mk15 import BudgetExceeded, Mk15

MANIFEST = Path("clips/manifest.json")
OUT = Path("tracks")


def load_manifest():
    return {e["clip_id"]: e for e in json.loads(MANIFEST.read_text())}


def reference_frame(clip, t):
    """Extract one frame (for flower detection) into tracks/raw/frames/."""
    out = Path("tracks/raw/frames") / f"{clip['clip_id']}_{t:.1f}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", clip["path"],
                        "-frames:v", "1", "-q:v", "2", str(out)], check=True)
    return out


def clamp_box(b):
    x1, y1, x2, y2 = (min(max(float(v), 0), 1000) for v in b)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def to_contract(clip, res, prompt_version, flowers_res=None):
    """Mk15.ask() result -> tracks file. Drops waypoints without a time or outside the clip."""
    dur = clip["duration_s"]
    tracks = []
    for i, tr in enumerate(res["tracks"], 1):
        pts = []
        for p in tr["points"]:
            if p.get("t") is None or not 0 <= p["t"] <= dur:
                continue
            if "box" in p:
                box = clamp_box(p["box"])
            elif "point" in p:  # a point track still gets a small box so overlays work
                x, y = p["point"]
                box = clamp_box([x - 10, y - 10, x + 10, y + 10])
            else:
                continue
            pts.append({"t": round(float(p["t"]), 3), "box": box,
                        "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                        "interpolated": False})
        pts.sort(key=lambda p: p["t"])
        tracks.append({"track_id": f"bee_{i}", "species": "unknown", "complete": tr["complete"],
                       "mention": tr.get("mention"), "points": pts})

    flowers = []
    if flowers_res:
        for j, b in enumerate(flowers_res["boxes"]):
            if "box" in b:
                flowers.append({"flower_id": f"flower_{j}", "box": clamp_box(b["box"])})

    return {"clip_id": clip["clip_id"], "model": "perceptron-mk1.5", "prompt_version": prompt_version,
            "tracks": tracks, "flowers": flowers, "events": [],
            "finish_reason": res.get("finish_reason"), "cost_usd": res.get("cost_usd"),
            "raw_response": res.get("cache_path")}


def run(mk, clip, version, flowers):
    prompt = Path(f"prompts/track_{version}.txt").read_text().strip()
    res = mk.ask([clip["path"]], prompt, expects="box", tag=clip["clip_id"])
    fres = None
    if flowers:
        frame = reference_frame(clip, clip["duration_s"] / 2)
        fprompt = Path(f"prompts/flowers_{version}.txt").read_text().strip() \
            if Path(f"prompts/flowers_{version}.txt").exists() else Path("prompts/flowers_v1.txt").read_text().strip()
        fres = mk.ask([frame], fprompt, expects="box", tag=clip["clip_id"])
    doc = to_contract(clip, res, version, fres)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{clip['clip_id']}.json").write_text(json.dumps(doc, indent=1))
    n = sum(len(t["points"]) for t in doc["tracks"])
    print(f"{clip['clip_id']}: {len(doc['tracks'])} track(s), {n} points, {len(doc['flowers'])} flowers, "
          f"finish={doc['finish_reason']}, ${res['cost_usd']:.4f}{' (cached)' if res['cached'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--version", default="v1", help="prompt version, prompts/track_<version>.txt")
    ap.add_argument("--flowers", action="store_true", help="also detect flowers on the middle frame")
    args = ap.parse_args()

    manifest = load_manifest()
    ids = list(manifest) if args.all else args.clips
    if not ids:
        ap.error("give clip ids or --all")
    mk = Mk15()
    print(f"user {mk.user}: ${mk.spent():.2f} spent of ${mk.budget_usd:.0f} cap")
    for cid in ids:
        try:
            run(mk, manifest[cid], args.version, args.flowers)
        except BudgetExceeded as e:
            sys.exit(str(e))
        except Exception as e:  # AuthError and friends: one clear line instead of a traceback
            if type(e).__name__ == "AuthError":
                sys.exit("No Perceptron key: copy .env.example to .env and set PERCEPTRON_API_KEY")
            raise
    print(f"total spent by {mk.user}: ${mk.spent():.2f}")


if __name__ == "__main__":
    main()
