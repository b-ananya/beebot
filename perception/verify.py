"""Mk1.5 checks motion-tracker candidates: is there an insect at the center of this crop, what is it, what is it doing?

    python -m perception.verify --clips hb1_w00              # tracks/cv/<clip>.json -> tracks/verified/<clip>.json
    python -m perception.verify --all --workers 8
    python -m perception.verify --all --heuristic            # no API: keep paths by motion score only

Each candidate path is sampled every ``--step`` seconds; the crop around the path point goes to Mk1.5 with a JSON
schema. A path is cut wherever two samples in a row say "no insect" and trimmed at both ends to the first and last
"yes", so a track that drifts onto swaying flowers after the bee leaves loses its tail. Kept pieces get the
majority species and a per-point behavior label (``foraging``, ``walking``, ``flying``) from the nearest sample.

Cost: one small image per sample, about $0.0002 each; every answer is cached in tracks/raw/verify/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from perception.client import ROOT, BudgetExceeded, PerceptionClient
from perception.track import manifest

CV_DIR = ROOT / "tracks" / "cv"
OUT_DIR = ROOT / "tracks" / "verified"
CROP_DIR = ROOT / "tracks" / "raw" / "frames" / "verify"
PIECE = "verify"
PROMPT_VERSION = "v1"
CROP_PX = 160          # crop side at 720p; bees are 15-30 px, so the crop shows the insect plus its surroundings
UPSCALE = 3

KINDS = ["honeybee", "bumblebee", "other_bee", "wasp", "hoverfly", "butterfly_or_moth", "other_insect", "none"]
BEHAVIORS = ["on_flower", "on_leaf_or_ground", "flying", "none"]
SPECIES_OF_KIND = {"honeybee": "honeybee", "bumblebee": "honeybee", "other_bee": "honeybee", "wasp": "vespidae",
                   "hoverfly": "syrphidae", "butterfly_or_moth": "lepidoptera", "other_insect": "unknown"}
LABEL_OF_BEHAVIOR = {"on_flower": "foraging", "on_leaf_or_ground": "walking", "flying": "flying"}

PROMPT = (
    "This is a zoomed-in crop from a fixed camera filming flowering plants outdoors. Look at the CENTER of the "
    "image. Is there an insect (a bee, wasp, hoverfly, moth, butterfly or other insect) at or very near the "
    "center? Petals, flower centers, leaves, buds, shadows and debris are not insects. If there is one, say what "
    "kind it is and whether it is sitting on a flower, on a leaf or the ground, or flying. Answer in JSON."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "insect_at_center": {"type": "boolean"},
        "kind": {"type": "string", "enum": KINDS},
        "behavior": {"type": "string", "enum": BEHAVIORS},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["insect_at_center", "kind", "behavior", "confidence"],
}


def sample_points(points: list[dict], step_s: float) -> list[dict]:
    """Path points about ``step_s`` apart, always including the first and last."""
    out, last_t = [], -1e9
    for p in points:
        if p["t"] - last_t >= step_s - 1e-6:
            out.append(p)
            last_t = p["t"]
    if points and out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def write_crops(clip_id: str, video: Path, requests: list[tuple[str, dict]]) -> dict[tuple[str, float], Path]:
    """Save one PNG crop per (track_id, point); read the video once in frame order."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    by_frame: dict[int, list[tuple[str, dict]]] = {}
    for tid, p in requests:
        by_frame.setdefault(int(round(p["t"] * fps)), []).append((tid, p))
    out_dir = CROP_DIR / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    fi = 0
    for target in sorted(by_frame):
        while fi <= target:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
        if not ok:
            break
        for tid, p in by_frame[target]:
            cx, cy = p["center"][0] / 1000 * W, p["center"][1] / 1000 * H
            x0 = int(min(max(cx - CROP_PX / 2, 0), W - CROP_PX))
            y0 = int(min(max(cy - CROP_PX / 2, 0), H - CROP_PX))
            crop = cv2.resize(frame[y0:y0 + CROP_PX, x0:x0 + CROP_PX], None, fx=UPSCALE, fy=UPSCALE,
                              interpolation=cv2.INTER_CUBIC)
            path = out_dir / f"{tid}_{p['t']:.3f}.png"
            cv2.imwrite(str(path), crop)
            paths[(tid, p["t"])] = path
    cap.release()
    return paths


def ask(client: PerceptionClient, clip_id: str, crop: Path) -> dict:
    from perceptron import json_schema_format

    r = client.call(clip_id=clip_id, prompt_name="verify", prompt_version=PROMPT_VERSION, prompt_text=PROMPT,
                    image_paths=[crop], annotation_format=None, max_tokens=200,
                    response_format=json_schema_format(SCHEMA, name="insect_check"))
    try:
        ans = json.loads(r.text)
    except (json.JSONDecodeError, TypeError):
        ans = {"insect_at_center": False, "kind": "none", "behavior": "none", "confidence": "low", "parse_error": True}
    return ans


def split_by_answers(points: list[dict], samples: list[dict], answers: list[dict]) -> list[dict]:
    """Pieces of one path that Mk1.5 confirms, each with species and per-point behavior labels."""
    yes = [bool(a.get("insect_at_center")) and a.get("kind") != "none" for a in answers]
    pieces, cur = [], []
    misses = 0
    for s, a, y in zip(samples, answers, yes):
        if y:
            cur.append((s, a))
            misses = 0
        else:
            misses += 1
            if misses >= 2 and cur:
                pieces.append(cur)
                cur = []
    if cur:
        pieces.append(cur)

    out = []
    for piece in pieces:
        t0, t1 = piece[0][0]["t"], piece[-1][0]["t"]
        # extend a little past the last confirmed sample, but never beyond the next "no" sample
        later_no = [s["t"] for s, y in zip(samples, yes) if not y and s["t"] > t1]
        t1_ext = min(later_no[0] - 1e-3, t1 + 0.25) if later_no else points[-1]["t"]
        earlier_no = [s["t"] for s, y in zip(samples, yes) if not y and s["t"] < t0]
        t0_ext = max(earlier_no[-1] + 1e-3, t0 - 0.25) if earlier_no else points[0]["t"]
        kinds = Counter(a["kind"] for _, a in piece)
        species = SPECIES_OF_KIND.get(kinds.most_common(1)[0][0], "unknown")
        keep = []
        for p in points:
            if t0_ext <= p["t"] <= t1_ext:
                nearest = min(piece, key=lambda sa: abs(sa[0]["t"] - p["t"]))[1]
                q = dict(p)
                label = LABEL_OF_BEHAVIOR.get(nearest.get("behavior"))
                if label:
                    q["label"] = label
                keep.append(q)
        if keep:
            out.append({"species": species, "points": keep, "confirmed_samples": len(piece),
                        "kind_votes": dict(kinds)})
    return out


def heuristic(doc: dict) -> list[dict]:
    """No API: keep paths whose motion score is at least a third of the best one."""
    tracks = doc["tracks"]
    if not tracks:
        return []
    top = max(t["stats"]["score"] for t in tracks)
    return [{"species": "unknown", "points": t["points"], "confirmed_samples": None, "source_track": t["track_id"]}
            for t in tracks if t["stats"]["score"] >= top / 3]


def verify_clip(clip_id: str, *, client: PerceptionClient | None, step_s: float = 0.5, workers: int = 8,
                cv_dir: Path = CV_DIR, out_dir: Path = OUT_DIR, video: Path | None = None) -> dict:
    doc = json.loads((cv_dir / f"{clip_id}.json").read_text())
    log = {}
    if client is None:
        pieces = heuristic(doc)
        method = "heuristic"
    else:
        video = video or (ROOT / manifest()[clip_id]["path"])
        plan = {t["track_id"]: sample_points(t["points"], step_s) for t in doc["tracks"]}
        crops = write_crops(clip_id, video, [(tid, p) for tid, ps in plan.items() for p in ps])
        jobs = [(tid, p) for tid, ps in plan.items() for p in ps if (tid, p["t"]) in crops]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(lambda j: ask(client, clip_id, crops[(j[0], j[1]["t"])]), jobs))
        by_track: dict[str, list] = {}
        for (tid, p), a in zip(jobs, answers):
            by_track.setdefault(tid, []).append((p, a))
        pieces = []
        for t in doc["tracks"]:
            sa = by_track.get(t["track_id"], [])
            log[t["track_id"]] = [{"t": p["t"], **a} for p, a in sa]
            for piece in split_by_answers(t["points"], [p for p, _ in sa], [a for _, a in sa]):
                piece["source_track"] = t["track_id"]
                pieces.append(piece)
        method = f"mk1.5-verify-{PROMPT_VERSION}"

    counts: Counter = Counter()
    tracks = []
    for piece in sorted(pieces, key=lambda pc: pc["points"][0]["t"]):
        counts[piece["species"]] += 1
        tracks.append({"track_id": f"{piece['species']}_{counts[piece['species']]}", "species": piece["species"],
                       "complete": True, "points": piece["points"], "source_track": piece["source_track"],
                       "confirmed_samples": piece.get("confirmed_samples"), "kind_votes": piece.get("kind_votes")})
    out = {"clip_id": clip_id, "model": "cv-bgsub-dp+" + ("perceptron-mk1.5" if client else "heuristic"),
           "prompt_version": method, "tracks": tracks, "flowers": [], "events": [], "verify_log": log}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip_id}.json").write_text(json.dumps(out))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--heuristic", action="store_true", help="no API calls")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    clips = [c for c in manifest() if (CV_DIR / f"{c}.json").exists()] if args.all else args.clips
    client = None if args.heuristic else PerceptionClient(PIECE)
    if client:
        print(f"user {client.user} | spent ${client.spent_usd():.3f} of ${client.cost_cap_usd:.0f}")
    for cid in clips:
        try:
            doc = verify_clip(cid, client=client, step_s=args.step, workers=args.workers, out_dir=Path(args.out))
        except BudgetExceeded as e:
            print(f"stopping: {e}")
            break
        print(f"{cid:16s} kept {len(doc['tracks'])} tracks: " +
              ", ".join(f"{t['track_id']} {t['points'][0]['t']:.1f}-{t['points'][-1]['t']:.1f}s" for t in doc["tracks"]))
    if client:
        print(f"{client.api_calls} API calls, {client.cache_hits} cache hits, ${client.spent_usd():.3f} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
