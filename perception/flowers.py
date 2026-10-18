"""P2: flower boxes -> tracks/parts/<clip>.flowers.json. The project's one flower detector (the pipeline uses it too).

    python -m perception.flowers --clips sm3_12_124130              # one clip
    python -m perception.flowers --sm --eval --overlay              # the 10 farm clips, scored against truth
    python -m perception.flowers --sm --eval --method middle        # the original middle-frame detector, to compare
    python -m perception.flowers --sm --merge                       # also tracks/<clip>.json (needs a track part)

Default method "tiled" (``detect_tiled``): Mk1.5 on the clip's per-pixel median frame (insects and shadows that
pass through vanish, so a flower briefly hidden in one frame is still found), split into 2x2 overlapping tiles so
small flowers are big enough to see, boxes merged across tiles by IoU. Dense ground cover (the honeybee clips,
~1,000 blossoms) is detected from the first tile and skipped. Method "middle" is the original single-call path:

The reference frame is the clip's middle frame (tracks/raw/frames/<clip>_ref.png, committed so the cache key, which
hashes the PNG bytes, is the same for everyone). Flowers do not move, so one frame per clip is enough and far cheaper
than sending the video. Reruns are free: the shared client serves repeated requests from tracks/raw/p2/.

The part file is the shape merge.py expects ({"clip_id", "flowers": [{"flower_id", "box"}]}) plus run metadata,
like P1's track part. Flower ids are stable per camera view: clips cut from one recording (sm3_12_*, hb1_*) share a
fixed camera, so the same physical flower gets the same flower_<n> in every clip of that view, numbered top-to-bottom
then left-to-right. A clip may lack some ids (a flower not detected in that frame).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from perception.client import ROOT, PerceptionClient, Response
from perception.track import PARTS_DIR, clip_path, manifest

PROMPTS_DIR = ROOT / "prompts" / "flowers"
PIECE = "p2"
DUP_IOU = 0.7
# same flower across clips of one view overlaps at IoU >= 0.7 (median ~0.9); distinct flowers in a clip at <= 0.09
VIEW_IOU = 0.4


def reference_frame(clip_id: str, *, root: Path = ROOT, frames_dir: Path | None = None) -> Path:
    """The clip's middle frame as a PNG, extracted once and reused."""
    out = (frames_dir or root / "tracks" / "raw" / "frames") / f"{clip_id}_ref.png"
    if out.exists():
        return out
    cap = cv2.VideoCapture(str(clip_path(clip_id, root=root)))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read the middle frame of {clip_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text().strip()


def detect_flowers(clip_id: str, version: str = "v1", *, client: PerceptionClient | None = None,
                   max_tokens: int = 4096, root: Path = ROOT, frames_dir: Path | None = None) -> Response:
    """One Mk1.5 box call on the reference frame; parsing is parse_flowers()."""
    client = client or PerceptionClient(PIECE)
    png = reference_frame(clip_id, root=root, frames_dir=frames_dir)
    return client.call(clip_id=clip_id, prompt_name="flowers", prompt_version=version,
                       prompt_text=load_prompt(version), image_paths=[png], annotation_format="box",
                       max_tokens=max_tokens)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_flowers(text: str) -> list[dict]:
    """The answer's boxes as CONTRACT.md flowers. Malformed, out-of-range, unordered or zero-area boxes are dropped,
    never fixed up; a box overlapping an earlier one by IoU > 0.7 is a duplicate and dropped."""
    from perceptron import collect_annotations

    kept = []
    for b in collect_annotations(text or "").boxes:
        box = [b.top_left.x, b.top_left.y, b.bottom_right.x, b.bottom_right.y]
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in box):
            continue
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        if any(iou(box, k) > DUP_IOU for k in kept):
            continue
        kept.append(box)
    kept.sort(key=lambda bx: (bx[1], bx[0]))
    return [{"flower_id": f"flower_{i}", "box": bx} for i, bx in enumerate(kept, 1)]


# ---- the default detector (method "tiled"): median frame, 2x2 tiles, dense rule ------------------------------
# These settings are part of the Mk1.5 cache key (tracks/raw/flowers/): change them and every clip is re-asked.
TILED_PIECE = "flowers"
TILED_VERSION = "v1"
TILED_PROMPT = ("Your goal is to segment out the following categories: open flower. Box every individual open flower "
                "head you can see, including small and partly hidden ones. Do not box leaves, buds or fruit.")
DENSE_BOXES = 150      # first tile with more boxes than this (or a cut-off answer) = dense ground cover
TILE_MERGE_IOU = 0.4   # boxes from overlapping tiles with IoU >= this are one flower (largest kept)


def median_frame(clip_id: str, video: Path | None = None, *, root: Path = ROOT, samples: int = 41) -> Path:
    """Per-pixel median of ``samples`` frames: moving insects and passing shadows vanish, flowers stay."""
    path = root / "tracks" / "raw" / "frames" / "reference" / f"{clip_id}.png"
    if path.exists():
        return path
    cap = cv2.VideoCapture(str(video or clip_path(clip_id, root=root)))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, min(samples, n)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    ref = np.median(np.stack(frames), axis=0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), ref)
    return path


def _tiles(ref: Path, clip_id: str) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """2x2 overlapping tiles; each with its (x0, y0, w, h) in 0-1000 units of the full frame."""
    img = cv2.imread(str(ref))
    H, W = img.shape[:2]
    tw, th = int(W * 0.6), int(H * 0.6)
    out = []
    for i, x0 in enumerate([0, W - tw]):
        for j, y0 in enumerate([0, H - th]):
            p = ref.with_name(f"{clip_id}_tile{i}{j}.png")
            if not p.exists():
                cv2.imwrite(str(p), img[y0:y0 + th, x0:x0 + tw])
            out.append((p, (x0 / W * 1000, y0 / H * 1000, tw / W * 1000, th / H * 1000)))
    return out


def detect_tiled(clip_id: str, *, client: PerceptionClient | None = None, video: Path | None = None,
                 root: Path = ROOT) -> dict:
    """The project's flower detector: Mk1.5 boxes on the median frame, 2x2 overlapping tiles, merged by IoU.

    Returns {"flowers": [{"flower_id", "box", "source": "mk1.5"}], "dense": bool, "frame": png}. Dense ground cover
    (first tile cut off at the token limit or > DENSE_BOXES boxes) returns no flowers: an inventory of ~1,000
    blossoms is neither complete nor useful, and visits fall back to stop-flowers (perception.visits).
    """
    from perceptron import BoundingBox

    boxes = []
    client = client or PerceptionClient(TILED_PIECE)
    frame = median_frame(clip_id, video, root=root)
    for k, (tile, (x0, y0, w, h)) in enumerate(_tiles(frame, clip_id)):
        r = client.call(clip_id=clip_id, prompt_name="flowers", prompt_version=TILED_VERSION, prompt_text=TILED_PROMPT,
                        image_paths=[tile], annotation_format="box", max_tokens=6000)
        tile_boxes = r.annotations().boxes
        if k == 0 and (r.finish_reason == "length" or len(tile_boxes) > DENSE_BOXES):
            # dense ground cover (the honeybee clips: ~1,000 blossoms, answers cut off at the token limit):
            # a full inventory is neither complete nor useful, so visits fall back to stop-flowers
            return {"flowers": [], "dense": True, "frame": frame}
        for b in tile_boxes:
            if not isinstance(b, BoundingBox):
                continue
            bx = [x0 + b.top_left.x / 1000 * w, y0 + b.top_left.y / 1000 * h,
                  x0 + b.bottom_right.x / 1000 * w, y0 + b.bottom_right.y / 1000 * h]
            if bx[2] - bx[0] < 3 or bx[3] - bx[1] < 3 or (bx[2] - bx[0]) * (bx[3] - bx[1]) > 250 * 250:
                continue
            boxes.append([round(min(max(v, 0), 1000), 1) for v in bx])
    merged: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(iou(b, m) < TILE_MERGE_IOU for m in merged):
            merged.append(b)
    merged.sort(key=lambda b: (b[1], b[0]))
    return {"flowers": [{"flower_id": f"flower_{i}", "box": b, "source": "mk1.5"} for i, b in enumerate(merged)],
            "dense": False, "frame": frame}


def flowers_clip(clip_id: str, *, method: str = "tiled", version: str = "v1", client: PerceptionClient | None = None,
                 max_tokens: int = 4096, root: Path = ROOT, parts_dir: Path | None = None,
                 frames_dir: Path | None = None) -> dict:
    """Write tracks/parts/<clip>.flowers.json with ``method`` "tiled" (default, what the pipeline uses) or "middle"
    (one middle-frame call with prompts/flowers/<version>.txt, kept for comparison), then relabel the view's ids."""
    parts_dir = parts_dir or PARTS_DIR
    if method == "tiled":
        client = client or PerceptionClient(TILED_PIECE)
        calls_before, hits_before = client.api_calls, client.cache_hits
        res = detect_tiled(clip_id, client=client, root=root)
        part = {"clip_id": clip_id, "model": client.model, "method": "tiled", "prompt_version": TILED_VERSION,
                "dense": res["dense"], "complete": True, "cache_hit": client.api_calls == calls_before,
                "api_calls": client.api_calls - calls_before, "cache_hits": client.cache_hits - hits_before,
                "flowers": [{"flower_id": fl["flower_id"], "box": fl["box"]} for fl in res["flowers"]]}
        parts_dir.mkdir(parents=True, exist_ok=True)
        (parts_dir / f"{clip_id}.flowers.json").write_text(json.dumps(part, indent=1))
        return label_view(view_id(clip_id), parts_dir)[clip_id]
    if method != "middle":
        raise ValueError(f"method must be tiled or middle, not {method}")
    client = client or PerceptionClient(PIECE)
    resp = detect_flowers(clip_id, version, client=client, max_tokens=max_tokens, root=root, frames_dir=frames_dir)
    part = {"clip_id": clip_id, "model": resp.model or client.model, "method": "middle", "prompt_version": version,
            "finish_reason": resp.finish_reason, "complete": resp.complete, "cache_hit": resp.cache_hit,
            "cost_usd": resp.cost_usd, "cache_file": str(resp.cache_path.relative_to(client.cache_dir)).replace("\\", "/"),
            "parse_errors": resp.annotations().errors, "flowers": parse_flowers(resp.text)}
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / f"{clip_id}.flowers.json").write_text(json.dumps(part, indent=1))
    return label_view(view_id(clip_id), parts_dir)[clip_id]


def view_id(clip_id: str) -> str:
    """The camera recording a clip was cut from: sm3_12_124130 -> sm3_12, hb1_w00 -> hb1."""
    return clip_id.rsplit("_", 1)[0]


def assign_view_ids(parts: list[dict]) -> dict[str, dict]:
    """Relabel the flowers of one view's parts so a physical flower has one flower_id in every clip.

    Boxes are matched one-to-one per clip against the view's flowers so far (IoU >= VIEW_IOU, best first); an unmatched
    box is a new flower. Ids follow each flower's mean box, top-to-bottom then left-to-right. Boxes themselves are kept
    as detected. The result does not depend on the parts' current ids, so relabeling twice changes nothing."""
    flowers, members = [], []  # per view flower: mean box, number of clips it was seen in
    seen = []
    for part in sorted(parts, key=lambda p: p["clip_id"]):
        boxes = [f["box"] for f in part["flowers"]]
        pairs = sorted(((iou(b, fl), i, j) for i, b in enumerate(boxes) for j, fl in enumerate(flowers)), reverse=True)
        which, used = {}, set()
        for v, i, j in pairs:
            if v < VIEW_IOU:
                break
            if i in which or j in used:
                continue
            which[i] = j
            used.add(j)
        for i, b in enumerate(boxes):
            if i not in which:
                flowers.append(list(b))
                members.append(0)
                which[i] = len(flowers) - 1
            j, n = which[i], members[which[i]]
            flowers[j] = [(m * n + v) / (n + 1) for m, v in zip(flowers[j], b)]
            members[j] = n + 1
        seen.append((part, which))
    order = sorted(range(len(flowers)), key=lambda j: (flowers[j][1], flowers[j][0]))
    rank = {j: k for k, j in enumerate(order, 1)}
    out = {}
    for part, which in seen:
        fl = sorted(({"flower_id": f"flower_{rank[which[i]]}", "box": f["box"]} for i, f in enumerate(part["flowers"])),
                    key=lambda f: int(f["flower_id"].split("_")[1]))
        out[part["clip_id"]] = {**part, "view_id": view_id(part["clip_id"]), "flowers": fl}
    return out


def label_view(view: str, parts_dir: Path | None = None) -> dict[str, dict]:
    """Relabel every flowers part of one view on disk with stable ids (free: no API calls)."""
    parts_dir = parts_dir or PARTS_DIR
    paths = [p for p in sorted(parts_dir.glob(f"{view}_*.flowers.json")) if view_id(p.name.split(".")[0]) == view]
    out = assign_view_ids([json.loads(p.read_text()) for p in paths])
    for clip_id, part in out.items():
        (parts_dir / f"{clip_id}.flowers.json").write_text(json.dumps(part, indent=1))
    return out


# ------------------------------------------------------------------ validation (A2, step 5)
def match(pred: list[dict], truth: list[dict], iou_thr: float = 0.3) -> dict:
    """Greedy one-to-one IoU matching. Truth flower boxes come from the dataset's annotations, so they are
    approximate; the threshold is looser than the usual 0.5 for that reason."""
    pairs = sorted(((iou(p["box"], t["box"]), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)),
                   reverse=True)
    used_p, used_t, matched = set(), set(), []
    for v, i, j in pairs:
        if v < iou_thr:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matched.append({"pred": pred[i]["flower_id"], "truth": truth[j]["flower_id"], "iou": round(v, 3)})
    tp = len(matched)
    return {"pred": len(pred), "truth": len(truth), "tp": tp,
            "precision": round(tp / len(pred), 3) if pred else None,
            "recall": round(tp / len(truth), 3) if truth else None,
            "mean_iou": round(sum(m["iou"] for m in matched) / tp, 3) if tp else None, "matches": matched}


def overlay(clip_id: str, pred: list[dict], truth: list[dict], *, root: Path = ROOT,
            out_dir: Path | None = None, frame: Path | None = None) -> Path:
    """Truth boxes in green and Mk1.5 boxes in magenta, on ``frame`` (default: the middle frame)."""
    img = cv2.imread(str(frame or reference_frame(clip_id, root=root)))
    h, w = img.shape[:2]
    for flowers, color in ((truth, (0, 200, 0)), (pred, (255, 0, 255))):
        for f in flowers:
            x1, y1, x2, y2 = (int(round(v / 1000 * s)) for v, s in zip(f["box"], (w, h, w, h)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f["flower_id"].split("_")[-1], (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
    out = (out_dir or root / "eval" / "checks") / f"{clip_id}_flowers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def truth_flowers(clip_id: str, root: Path = ROOT) -> list[dict] | None:
    path = root / "truth" / f"{clip_id}.json"
    return json.loads(path.read_text()).get("flowers") if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[], help="clip ids (default: none)")
    ap.add_argument("--sm", action="store_true", help="the farm clips (sm*), the ones with truth flowers")
    ap.add_argument("--all", action="store_true", help="every clip in clips/manifest.json")
    ap.add_argument("--method", default="tiled", choices=["tiled", "middle"],
                    help="tiled: median frame, 2x2 tiles (what the pipeline uses); middle: one middle-frame call")
    ap.add_argument("--version", default="v1", help="--method middle only: prompts/flowers/<version>.txt")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--eval", action="store_true", help="score against truth flower boxes (IoU match)")
    ap.add_argument("--overlay", action="store_true", help="draw eval/checks/<clip>_flowers.png")
    ap.add_argument("--merge", action="store_true", help="also write tracks/<clip>.json (clips with a track part)")
    args = ap.parse_args(argv)

    ids = list(manifest())
    clips = ids if args.all else [c for c in ids if c.startswith("sm")] if args.sm else args.clips
    if not clips:
        ap.error("give --clips ..., --sm or --all")
    client = PerceptionClient(TILED_PIECE if args.method == "tiled" else PIECE)
    print(f"model {client.model} | user {client.user} | spent ${client.spent_usd():.3f} of ${client.cost_cap_usd:.0f}")
    spent_before = client.spent_usd()
    runs = {c: flowers_clip(c, method=args.method, version=args.version, client=client, max_tokens=args.max_tokens)
            for c in clips}
    scores = []
    for clip_id in clips:
        # ids are final only once every clip of the view is labeled, so read the part back
        part = json.loads((PARTS_DIR / f"{clip_id}.flowers.json").read_text())
        flag = "cache" if part["cache_hit"] else ("new" if "cost_usd" not in part else f"${part['cost_usd']:.4f}")
        state = "  dense: stop-flowers" if part.get("dense") else (
            "" if part["complete"] else f"  INCOMPLETE ({part['finish_reason']})")
        line = f"{clip_id:16s} {len(part['flowers']):3d} flowers  {flag:>8s}{state}"
        truth = truth_flowers(clip_id)
        if args.eval and truth:          # only the farm clips have truth flowers
            s = match(part["flowers"], truth)
            scores.append(s)
            line += f"  P={s['precision']} R={s['recall']} mIoU={s['mean_iou']} ({s['tp']}/{s['truth']} truth)"
        if args.overlay:
            frame = median_frame(clip_id) if args.method == "tiled" else None
            overlay(clip_id, part["flowers"], truth or [], frame=frame)
        print(line)
    print(f"{client.api_calls} API calls, {client.cache_hits} cache hits, ${client.spent_usd() - spent_before:.4f} "
          f"this run, ${client.spent_usd():.3f} total on {client.cost_log.name}")
    if scores:
        tp, n_pred, n_truth = (sum(s[k] for s in scores) for k in ("tp", "pred", "truth"))
        ious = [m["iou"] for s in scores for m in s["matches"]]
        print(f"overall: precision {tp / n_pred if n_pred else 0:.3f}, recall {tp / n_truth if n_truth else 0:.3f}, "
              f"mean IoU {sum(ious) / len(ious) if ious else 0:.3f} over {len(scores)} clips")
    if args.merge:
        from perception.merge import merge_clip

        merged = [c for c in clips if (PARTS_DIR / f"{c}.track.json").exists()]
        for clip_id in merged:
            merge_clip(clip_id)
        print(f"merged {len(merged)} clips into tracks/ ({len(clips) - len(merged)} have no track part yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
