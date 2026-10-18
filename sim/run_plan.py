"""Field -> plan -> rollout video for farm clips.

    python -m sim.run_plan sm3_12_123903            # one clip
    python -m sim.run_plan --all                    # every sm* clip
    python -m sim.run_plan sm3_12_123903 --budget 20 --target 6
    python -m sim.run_plan --all --source truth      # dataset-based maps -> field/truth/, plan/truth/

Default source is the pipeline (mk1.5): field/<clip>.json, plan/<clip>.json, rollouts/<clip>_plan.mp4.
"""
import argparse
import json
from pathlib import Path

from sim.field import (DEFAULT_TARGET, bee_dwell_s, patch_from_pipeline, patch_from_truth, pipeline_bee,
                       pipeline_dwell_s)
from sim.planner import plan_route
from sim.render import render_plan


def run(clip_id, args, dwell):
    manifest = {c["clip_id"]: c for c in json.loads(Path("clips/manifest.json").read_text())}
    clip = manifest[clip_id]
    if args.source == "truth":
        field = patch_from_truth(clip_id, args.target)
        truth = json.loads(Path(f"truth/{clip_id}.json").read_text())
        bee_points, bee_order = truth["tracks"][0]["points"], [e["flower_id"] for e in truth["events"]]
        sub, suffix = "truth", "_truth"
    else:
        field = patch_from_pipeline(clip_id, args.target)
        bee_points, bee_order = pipeline_bee(clip_id)
        sub, suffix = "", ""
    plan = plan_route(field, start=(args.start_x, args.start_y), speed=args.speed, dwell_s=dwell,
                      budget_s=args.budget, aspect=clip["width"] / clip["height"], bee_order=bee_order)
    video = Path("rollouts") / f"{clip_id}_plan{suffix}.mp4"
    video.parent.mkdir(exist_ok=True)
    if not args.no_video:
        render_plan(clip["path"], field, plan, str(video), bee_points, args.time_scale)
    plan["video"] = video.as_posix()
    plan["robot"]["time_scale"] = args.time_scale
    plan["source"] = field["source"]
    for folder, doc in (("field", field), ("plan", plan)):
        out = Path(folder) / sub
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{clip_id}.json").write_text(json.dumps(doc, indent=1))
    b = plan["baselines"]
    print(f"{clip_id}: route {plan['route']} covers {plan['deficit_covered']:.0%} of deficit "
          f"{plan['total_deficit']:.0f} | nearest_all {b['nearest_all']:.0%} random {b['random']:.0%}"
          + (f" follow_bee {b['follow_bee']:.0%}" if "follow_bee" in b else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget", type=float, default=30.0, help="robot time budget, seconds")
    ap.add_argument("--speed", type=float, default=100.0, help="y-norm1000 units per second")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="target visits per flower")
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-y", type=float, default=1000.0)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--source", choices=["mk1.5", "truth"], default="mk1.5")
    args = ap.parse_args()
    manifest = json.loads(Path("clips/manifest.json").read_text())
    farm = [c["clip_id"] for c in manifest if c["clip_id"].startswith("sm")]
    ids = farm if args.all else args.clips
    if not ids:
        ap.error("give clip ids or --all")
    dwell = bee_dwell_s(farm) if args.source == "truth" else pipeline_dwell_s(farm)
    print(f"robot dwell per flower = median bee dwell ({args.source}) = {dwell}s")
    for cid in ids:
        run(cid, args, dwell)


if __name__ == "__main__":
    main()
