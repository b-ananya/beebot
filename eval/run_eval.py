"""Score tracks against truth: center error, coverage, false-alarm rate, visit recall/precision.

    python -m eval.run_eval                          # tracks/*.json vs truth/*.json -> eval/report.json
    python -m eval.run_eval --pred tracks/cv --out eval/report_cv.json
    python -m eval.run_eval --clips hb1_w00 hb1_w01

Truth is the dataset authors' tracker output (HyDaT / Polytrack), so center error is *agreement with a reference
tracker*, not error against hand labels. Interpolated predicted points are excluded from every metric.

Matching: for each truth point, the nearest observed predicted point (over all predicted tracks) within ``--tol``
seconds. Coverage counts a truth point as tracked only when that match is within ``HIT_DIST`` (50 units, about two
bee lengths at 720p). A predicted point is a false alarm when no truth point within ``--tol`` seconds is within ``HIT_DIST`` of it.
Visits match when they name flowers whose boxes overlap (IoU >= 0.1 or centers within 40 units) and their time
ranges overlap.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIT_DIST = 50


def center(p: dict) -> tuple[float, float]:
    if p.get("center"):
        return tuple(p["center"])
    b = p["box"]
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def observed(doc: dict) -> list[dict]:
    """All non-interpolated points of all tracks, sorted by time."""
    pts = [p for tr in doc.get("tracks", []) for p in tr.get("points", []) if not p.get("interpolated")]
    return sorted(pts, key=lambda p: p["t"])


def _nearest(pts: list[dict], times: list[float], t: float, tol: float, ref: tuple | None = None) -> dict | None:
    """Predicted point closest in time to t (within tol); with several at that time, the one closest to ref."""
    import bisect

    i = bisect.bisect_left(times, t - tol)
    best, best_key = None, None
    while i < len(pts) and times[i] <= t + tol:
        p = pts[i]
        dist = 0.0
        if ref is not None:
            c = center(p)
            dist = ((c[0] - ref[0]) ** 2 + (c[1] - ref[1]) ** 2) ** 0.5
        key = (round(abs(times[i] - t), 3), dist)
        if best_key is None or key < best_key:
            best, best_key = p, key
        i += 1
    return best


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def iou(a: list, b: list) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_flower(a: list, b: list) -> bool:
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return iou(a, b) >= 0.1 or ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 <= 40


def score_visits(truth: dict, pred: dict) -> tuple[int, int, int, int]:
    """(truth visits found, truth visits, predicted visits that are correct, predicted visits)."""
    tf = {f["flower_id"]: f["box"] for f in truth.get("flowers", [])}
    pf = {f["flower_id"]: f["box"] for f in pred.get("flowers", [])}
    tv = [e for e in truth.get("events", []) if e["type"] == "visit" and e["flower_id"] in tf]
    pv = [e for e in pred.get("events", []) if e["type"] == "visit" and e["flower_id"] in pf]

    def match(a, af, b, bf):
        return (a["t_start"] <= b["t_end"] and b["t_start"] <= a["t_end"]
                and same_flower(af[a["flower_id"]], bf[b["flower_id"]]))

    tp_t = sum(any(match(t, tf, p, pf) for p in pv) for t in tv)
    tp_p = sum(any(match(p, pf, t, tf) for t in tv) for p in pv)
    return tp_t, len(tv), tp_p, len(pv)


def score_clip(truth: dict, pred: dict, tol: float = 0.2) -> dict:
    tpts = observed(truth)
    ppts = observed(pred)
    ptimes = [p["t"] for p in ppts]
    errs = []
    for tp in tpts:
        c = center(tp)
        m = _nearest(ppts, ptimes, tp["t"], tol, ref=c)
        if m is not None:
            pc = center(m)
            errs.append(((pc[0] - c[0]) ** 2 + (pc[1] - c[1]) ** 2) ** 0.5)
    ttimes = [p["t"] for p in tpts]
    false_alarms = 0
    for p in ppts:
        m = _nearest(tpts, ttimes, p["t"], tol, ref=center(p))
        if m is None or _dist(center(m), center(p)) > HIT_DIST:
            false_alarms += 1
    # visits are only scored where the truth has flowers (the farm clips); the honeybee truth has none
    tp_t, n_t, tp_p, n_p = score_visits(truth, pred) if truth.get("flowers") else (0, 0, 0, 0)
    return {
        "truth_points": len(tpts), "pred_points": len(ppts),
        "matched": len(errs),
        "coverage": round(sum(e <= HIT_DIST for e in errs) / len(tpts), 3) if tpts else None,
        "median_center_error": round(statistics.median(errs), 1) if errs else None,
        "within_25": round(sum(e <= 25 for e in errs) / len(errs), 3) if errs else None,
        "false_alarm_rate": round(false_alarms / len(ppts), 3) if ppts else None,
        "visits_truth": n_t, "visits_truth_found": tp_t, "visits_pred": n_p, "visits_pred_correct": tp_p,
        "_errs": errs,
    }


def evaluate(pred_dir: Path, truth_dir: Path, clips: list[str] | None = None, tol: float = 0.2) -> dict:
    per_clip = {}
    for tpath in sorted(truth_dir.glob("*.json")):
        cid = tpath.stem
        if clips and cid not in clips:
            continue
        ppath = pred_dir / f"{cid}.json"
        if not ppath.exists():
            continue
        per_clip[cid] = score_clip(json.loads(tpath.read_text()), json.loads(ppath.read_text()), tol)

    all_errs = [e for s in per_clip.values() for e in s.pop("_errs")]
    n_truth = sum(s["truth_points"] for s in per_clip.values())
    n_pred = sum(s["pred_points"] for s in per_clip.values())
    fa = sum(round((s["false_alarm_rate"] or 0) * s["pred_points"]) for s in per_clip.values())
    vt = sum(s["visits_truth"] for s in per_clip.values())
    vtf = sum(s["visits_truth_found"] for s in per_clip.values())
    vp = sum(s["visits_pred"] for s in per_clip.values())
    vpc = sum(s["visits_pred_correct"] for s in per_clip.values())
    return {
        "pred_dir": str(pred_dir.relative_to(ROOT) if pred_dir.is_relative_to(ROOT) else pred_dir).replace("\\", "/"),
        "clips": len(per_clip),
        "median_center_error": round(statistics.median(all_errs), 1) if all_errs else None,
        "within_25": round(sum(e <= 25 for e in all_errs) / len(all_errs), 3) if all_errs else None,
        "track_coverage": round(sum(e <= HIT_DIST for e in all_errs) / n_truth, 3) if n_truth else None,
        "false_alarm_rate": round(fa / n_pred, 3) if n_pred else None,
        "visit_recall": round(vtf / vt, 3) if vt else None,
        "visit_precision": round(vpc / vp, 3) if vp else None,
        "notes": "center error in 0-1000 units vs reference tracker (HyDaT/Polytrack); interpolated points excluded",
        "per_clip": per_clip,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="tracks", help="directory of <clip>.json predictions")
    ap.add_argument("--truth", default="truth")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--tol", type=float, default=0.2, help="seconds for time matching")
    ap.add_argument("--out", default="eval/report.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rep = evaluate(ROOT / args.pred, ROOT / args.truth, args.clips, args.tol)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    if not args.quiet:
        for cid, s in rep["per_clip"].items():
            print(f"{cid:16s} cov {s['coverage']!s:>5}  err {s['median_center_error']!s:>6}  "
                  f"fa {s['false_alarm_rate']!s:>5}  visits {s['visits_truth_found']}/{s['visits_truth']}"
                  f" ({s['visits_pred_correct']}/{s['visits_pred']} pred)")
    summary = {k: v for k, v in rep.items() if k != "per_clip"}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
