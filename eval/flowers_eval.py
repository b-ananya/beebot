"""Flower detection scores on the farm clips, three ways side by side -> eval/flowers_report.json.

    python -m eval.flowers_eval

1. vs reference, IoU >= 0.3: the original score (perception.flowers --eval).
2. vs reference, center distance: a detection matches a reference flower if their centers are within one reference
   box width. The reference boxes are squares estimated from Polytrack's flower radius, positioned once per
   station-day; on these ~45 px flowers a 15-30 px offset (camera 3's reference sits ~13 px low on every flower)
   drops IoU under 0.3 even when both boxes are on the same flower.
3. vs audit labels (eval/audit/flower_labels.json): every open flower in each camera view, judged by eye on the
   median frame, including ones Mk1.5 missed. The reference lists only the flowers Polytrack used, so it can't
   measure precision: most "extra" boxes are real flowers (see the audit images).

Each camera view is one scene (its clips share it), so the 10 clips are 2 independent scenes; per-view numbers
count each physical flower once.
"""
from __future__ import annotations

import json
from pathlib import Path

from perception.flowers import iou, view_id

ROOT = Path(__file__).resolve().parents[1]


def _c(b):
    return ((b[0] + b[2]) / 2 * 1.28, (b[1] + b[3]) / 2 * 0.72)       # pixels at 1280x720


def _w(b):
    return max((b[2] - b[0]) * 1.28, (b[3] - b[1]) * 0.72)


def match_iou(pred, ref, thr=0.3):
    pairs = sorted(((iou(p, r), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref)), reverse=True)
    up, ur = set(), set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in up and j not in ur:
            up.add(i)
            ur.add(j)
    return len(up)


def match_center(pred, ref):
    """One-to-one, nearest first; a pair matches when centers are within one reference box width."""
    pairs = []
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            d = ((_c(p)[0] - _c(r)[0]) ** 2 + (_c(p)[1] - _c(r)[1]) ** 2) ** 0.5
            if d <= _w(r):
                pairs.append((d, i, j))
    up, ur = set(), set()
    for d, i, j in sorted(pairs):
        if i not in up and j not in ur:
            up.add(i)
            ur.add(j)
    return len(up)


def pr(tp, n_pred, n_ref):
    return {"precision": round(tp / n_pred, 3) if n_pred else None, "recall": round(tp / n_ref, 3) if n_ref else None,
            "tp": tp, "pred": n_pred, "ref": n_ref}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="tracks", help="directory of <clip>.json predictions")
    ap.add_argument("--out", default="eval/flowers_report.json")
    args = ap.parse_args(argv)
    manifest = [e["clip_id"] for e in json.loads((ROOT / "clips" / "manifest.json").read_text())]
    labels = json.loads((ROOT / "eval" / "audit" / "flower_labels.json").read_text())["views"]
    per_clip, totals = {}, {k: [0, 0, 0] for k in ("ref_iou", "ref_center", "audit")}
    for cid in manifest:
        truth = json.loads((ROOT / "truth" / f"{cid}.json").read_text()) if (ROOT / "truth" / f"{cid}.json").exists() else {}
        ref = [f["box"] for f in truth.get("flowers", [])]
        if not ref:
            continue
        pred = [f["box"] for f in json.loads((ROOT / args.pred / f"{cid}.json").read_text())["flowers"]
                if f.get("source", "mk1.5") != "stop"]
        aud = [f["box"] for f in labels[view_id(cid)]["open_flowers"]]
        row = {"ref_iou": pr(match_iou(pred, ref), len(pred), len(ref)),
               "ref_center": pr(match_center(pred, ref), len(pred), len(ref)),
               "audit": pr(match_center(pred, aud), len(pred), len(aud))}
        per_clip[cid] = row
        for k in totals:
            totals[k][0] += row[k]["tp"]
            totals[k][1] += row[k]["pred"]
            totals[k][2] += row[k]["ref"]
    views = {}
    for v, d in labels.items():
        n_found = sum(f["found_by_mk15"] for f in d["open_flowers"])
        n_boxed = n_found + len(d["not_flowers_boxed"])
        views[v] = {"distinct_boxes": n_boxed, "boxes_on_open_flowers": n_found, "open_flowers": len(d["open_flowers"]),
                    "precision": round(n_found / n_boxed, 3), "recall": round(n_found / len(d["open_flowers"]), 3)}
    rep = {"clips": len(per_clip), "scenes": len(views),
           "overall_per_clip": {k: pr(*v) for k, v in totals.items()},
           "per_view_distinct_flowers": views,
           "notes": "ref = dataset flower list (Polytrack), audit = visual labels (eval/audit/flower_labels.json, "
                    "needs a human spot-check). See module docstring for why three scores.",
           "per_clip": per_clip}
    out = ROOT / args.out
    out.write_text(json.dumps(rep, indent=1))
    for k, v in rep["overall_per_clip"].items():
        print(f"{k:11s} precision {v['precision']}  recall {v['recall']}  ({v['tp']} matched, {v['pred']} boxes, {v['ref']} labels)")
    for v, s in views.items():
        print(f"{v}: {s['boxes_on_open_flowers']}/{s['distinct_boxes']} boxes on open flowers, "
              f"{s['boxes_on_open_flowers']}/{s['open_flowers']} open flowers found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
