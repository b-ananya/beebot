"""Mk1.5 checks motion-tracker candidates: does it see an insect where the path says one is?

    python -m perception.verify --clips hb1_w00              # tracks/cv/<clip>.json -> tracks/verified/<clip>.json
    python -m perception.verify --all --workers 4
    python -m perception.verify --all --heuristic            # no API: keep paths by motion score only

Each candidate path is sampled every ``--step`` seconds. The 160 px crop around the point (gray border where it
runs off the frame, so the point stays centered) goes to Mk1.5 as a detection task, "segment out: insect, bee,
wasp, fly". A sample is a *hit* when a returned box is centered in the middle half of the crop.

Why detection and per-path votes (measured on 64 truth-labelled crops, scratch/calib.py, Oct 2026):
- yes/no JSON questions ("is there an insect at the center?") found only 30-57% of real bees;
- detection finds 93%, but also boxes something in about half the bee-free crops;
- species answers were "hoverfly" for every clip, honeybee and wasp alike, so species are not taken from Mk1.5.
One crop is therefore weak evidence, while 10-20 crops along a path are strong evidence. A path is kept when its
hit rate is at least ``MIN_HIT_RATE``; it is cut where ``CUT_MISSES`` samples in a row miss (2 s), and its ends are
trimmed to the first and last hit.

Cost: one small image per sample (~$0.0001 measured); every answer is cached in tracks/raw/verify/.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from perception.client import ROOT, BudgetExceeded, PerceptionClient
from perception.track import manifest

CV_DIR = ROOT / "tracks" / "cv"
OUT_DIR = ROOT / "tracks" / "verified"
CROP_DIR = ROOT / "tracks" / "raw" / "frames" / "verify"
PIECE = "verify"
PROMPT_VERSION = "v2-detect"
CROP_PX = 160          # crop side at 720p; bees are 15-30 px, so the crop shows the insect plus its surroundings
OUT_PX = 480
PROMPT = "Your goal is to segment out the following categories: insect, bee, wasp, fly"
CENTER_HALF = 250      # hit if a box center is within this many 0-1000 units of the crop center
MAX_BOX = 600          # boxes bigger than this (0-1000) are "the whole picture", not an insect
MIN_HIT_RATE = 0.7      # swept 0.5-0.95 on all 48 clips: 0.7 keeps 90% coverage with the fewest disputed points
CUT_MISSES = 4

# species from dataset metadata, not from Mk1.5 (see above); None = unknown
SPECIES_BY_PREFIX = {"hb": "honeybee"}     # the Monash honeybee dataset films only honeybees


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
    """Save one PNG crop per (track_id, point), centered on the point; read the video once in frame order."""
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
    ok = True
    half = CROP_PX // 2
    for target in sorted(by_frame):
        while fi <= target:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
        if not ok:
            break
        padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        for tid, p in by_frame[target]:
            cx, cy = int(p["center"][0] / 1000 * W), int(p["center"][1] / 1000 * H)
            crop = cv2.resize(padded[cy:cy + CROP_PX, cx:cx + CROP_PX], (OUT_PX, OUT_PX), interpolation=cv2.INTER_CUBIC)
            path = out_dir / f"{tid}_{p['t']:.3f}.png"
            cv2.imwrite(str(path), crop)
            paths[(tid, p["t"])] = path
    cap.release()
    return paths


def ask(client: PerceptionClient, clip_id: str, crop: Path) -> dict:
    """Mk1.5 boxes on one crop -> {"hit": bool, "boxes": [...]}."""
    from perceptron import BoundingBox

    r = client.call(clip_id=clip_id, prompt_name="verify", prompt_version=PROMPT_VERSION, prompt_text=PROMPT,
                    image_paths=[crop], annotation_format="box", max_tokens=400)
    boxes = []
    hit = False
    for b in r.annotations().boxes:
        if not isinstance(b, BoundingBox):
            continue
        x1, y1, x2, y2 = b.top_left.x, b.top_left.y, b.bottom_right.x, b.bottom_right.y
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1)
        boxes.append([x1, y1, x2, y2, b.mention])
        hit |= abs(cx - 500) <= CENTER_HALF and abs(cy - 500) <= CENTER_HALF and size <= MAX_BOX
    return {"hit": hit, "boxes": boxes[:5]}


def split_by_hits(points: list[dict], samples: list[dict], hits: list[bool], *,
                  min_hit_rate: float = MIN_HIT_RATE, cut_misses: int = CUT_MISSES) -> list[list[dict]]:
    """Pieces of one path that Mk1.5 supports: cut at ``cut_misses`` misses in a row, trimmed to first/last hit,
    kept when at least ``min_hit_rate`` of their samples are hits and at least two are."""
    pieces, cur, misses = [], [], 0
    for s, h in zip(samples, hits):
        cur.append((s, h))
        misses = 0 if h else misses + 1
        if misses >= cut_misses:
            pieces.append(cur[:-misses])
            cur, misses = [], 0
    pieces.append(cur)
    out = []
    for piece in pieces:
        idx = [i for i, (_, h) in enumerate(piece) if h]
        if len(idx) < 2:
            continue
        piece = piece[idx[0]:idx[-1] + 1]
        if sum(h for _, h in piece) / len(piece) < min_hit_rate:
            continue
        t0, t1 = piece[0][0]["t"], piece[-1][0]["t"]
        # extend up to 0.25 s past the end hits, but never into a sampled miss
        sample_ts = [s["t"] for s, _ in zip(samples, hits)]
        miss_ts = [s["t"] for s, h in zip(samples, hits) if not h]
        before = [t for t in miss_ts if t < t0]
        after = [t for t in miss_ts if t > t1]
        lo = max(before[-1] + 1e-3, t0 - 0.25) if before else min(sample_ts[0], t0 - 0.25)
        hi = min(after[0] - 1e-3, t1 + 0.25) if after else max(sample_ts[-1], t1 + 0.25)
        keep = [p for p in points if lo <= p["t"] <= hi]
        if keep:
            out.append(keep)
    return out


def heuristic(doc: dict) -> list[dict]:
    """No API: keep paths whose motion score is at least a third of the best one."""
    tracks = doc["tracks"]
    if not tracks:
        return []
    top = max(t["stats"]["score"] for t in tracks)
    return [{"points": t["points"], "source_track": t["track_id"], "hit_rate": None}
            for t in tracks if t["stats"]["score"] >= top / 3]


def verify_clip(clip_id: str, *, client: PerceptionClient | None, step_s: float = 0.5, workers: int = 4,
                cv_dir: Path = CV_DIR, out_dir: Path = OUT_DIR, video: Path | None = None,
                species: str | None = None, min_hit_rate: float = MIN_HIT_RATE) -> dict:
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
            hits = [a["hit"] for _, a in sa]
            for pts in split_by_hits(t["points"], [p for p, _ in sa], hits, min_hit_rate=min_hit_rate):
                inside = [h for (p, _), h in zip(sa, hits) if pts[0]["t"] <= p["t"] <= pts[-1]["t"]]
                pieces.append({"points": pts, "source_track": t["track_id"],
                               "hit_rate": round(sum(inside) / max(len(inside), 1), 2)})
        method = f"mk1.5-{PROMPT_VERSION}"

    if species is None:
        species = SPECIES_BY_PREFIX.get(clip_id[:2], "unknown")
    tracks = []
    for n, piece in enumerate(sorted(pieces, key=lambda pc: pc["points"][0]["t"]), start=1):
        tracks.append({"track_id": f"{'insect' if species == 'unknown' else species}_{n}", "species": species,
                       "complete": True, "points": piece["points"], "source_track": piece["source_track"],
                       "hit_rate": piece["hit_rate"]})
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-hit-rate", type=float, default=MIN_HIT_RATE)
    ap.add_argument("--heuristic", action="store_true", help="no API calls")
    ap.add_argument("--cv", default=str(CV_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    cv_dir = Path(args.cv)
    clips = [c for c in manifest() if (cv_dir / f"{c}.json").exists()] if args.all else args.clips
    client = None if args.heuristic else PerceptionClient(PIECE)
    if client:
        print(f"user {client.user} | spent ${client.spent_usd():.3f} of ${client.cost_cap_usd:.0f}")
    for cid in clips:
        try:
            doc = verify_clip(cid, client=client, step_s=args.step, workers=args.workers, cv_dir=cv_dir,
                              out_dir=Path(args.out), min_hit_rate=args.min_hit_rate)
        except BudgetExceeded as e:
            print(f"stopping: {e}")
            break
        print(f"{cid:16s} kept {len(doc['tracks'])} tracks: " +
              ", ".join(f"{t['points'][0]['t']:.1f}-{t['points'][-1]['t']:.1f}s ({t['hit_rate']})"
                        for t in doc["tracks"]))
    if client:
        print(f"{client.api_calls} API calls, {client.cache_hits} cache hits, ${client.spent_usd():.3f} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
