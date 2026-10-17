"""Train PPO on random flower fields, then score it against the nearest-unvisited baseline
on the 10 real farm layouts (truth/sm*.json).

    python -m sim.train [--minutes 4] [--steps 600000] [--eval-only]

Saves sim/policy.zip and sim/eval.json. Score = flowers visited per budget (mean over layouts).
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from sim.field_env import ROOT, FieldEnv, flowers_from_truth
from sim.rollout import POLICY_PATH, make_actor, run_episode, start_point

EVAL_PATH = ROOT / "sim" / "eval.json"


def train(minutes=5.0, steps=1_500_000, n_envs=8, budget_s=30.0, seed=0):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    class StopAfter(BaseCallback):
        def __init__(self, seconds):
            super().__init__()
            self.deadline = time.time() + seconds

        def _on_step(self):
            return time.time() < self.deadline

    venv = make_vec_env(lambda: FieldEnv(budget_s=budget_s), n_envs=n_envs, seed=seed)
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=6, learning_rate=3e-4,
                gamma=0.995, gae_lambda=0.95, ent_coef=0.003, clip_range=0.2,
                policy_kwargs={"net_arch": [128, 128], "log_std_init": -1.0}, seed=seed, device="cpu", verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=StopAfter(minutes * 60), progress_bar=False)
    model.save(str(POLICY_PATH))
    return model, {"timesteps": int(model.num_timesteps), "train_seconds": round(time.time() - t0, 1)}


def evaluate(policy=POLICY_PATH, budget_s=30.0, dt=0.1):
    """Visits per budget on each real sm* layout, starting where the real bee entered."""
    actors = {"ppo": make_actor(policy), "nearest_unvisited": make_actor("baseline")}
    rows = []
    for f in sorted(glob.glob(str(ROOT / "truth" / "sm*.json"))):
        clip = Path(f).stem
        flowers = flowers_from_truth(clip)
        row = {"clip_id": clip, "n_open": sum(1 for x in flowers if x.get("state", "open") == "open")}
        for name, actor in actors.items():
            env = FieldEnv(flowers=flowers, budget_s=budget_s, dt=dt, start=start_point(clip), shaping=False)
            way, visits = run_episode(env, actor)
            row[name] = len(visits)
            row[name + "_first_visit_s"] = visits[0]["t"] if visits else None
        rows.append(row)
    summary = {"budget_s": budget_s, "layouts": len(rows),
               "ppo_visits_mean": float(np.mean([r["ppo"] for r in rows])),
               "baseline_visits_mean": float(np.mean([r["nearest_unvisited"] for r in rows])),
               "ppo_first_visit_s_mean": float(np.mean([r["ppo_first_visit_s"] or budget_s for r in rows])),
               "baseline_first_visit_s_mean": float(np.mean([r["nearest_unvisited_first_visit_s"] or budget_s
                                                             for r in rows])),
               "per_layout": rows}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=5.0, help="wall-clock cap for training")
    ap.add_argument("--steps", type=int, default=1_500_000, help="timestep cap for training")
    ap.add_argument("--budget", type=float, default=30.0, help="episode budget in seconds")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    info = {}
    if not a.eval_only:
        print(f"training PPO on random fields for up to {a.minutes} min / {a.steps} steps (CPU)...")
        _, info = train(minutes=a.minutes, steps=a.steps, budget_s=a.budget)
        print(f"saved {POLICY_PATH.relative_to(ROOT)} after {info['timesteps']} steps, {info['train_seconds']} s")
    for budget in sorted({a.budget, 15.0}):
        s = evaluate(budget_s=budget)
        s.update(info)
        if budget == a.budget:
            EVAL_PATH.write_text(json.dumps(s, indent=1))
        print(f"\nbudget {budget:.0f} s, {s['layouts']} real layouts (visits per budget):")
        print(f"  {'clip':18s} {'open':>4s} {'ppo':>4s} {'base':>5s}")
        for r in s["per_layout"]:
            print(f"  {r['clip_id']:18s} {r['n_open']:4d} {r['ppo']:4d} {r['nearest_unvisited']:5d}")
        print(f"  mean visits: ppo {s['ppo_visits_mean']:.2f}  baseline {s['baseline_visits_mean']:.2f}"
              f"   mean first visit: ppo {s['ppo_first_visit_s_mean']:.1f}s  "
              f"baseline {s['baseline_first_visit_s_mean']:.1f}s")


if __name__ == "__main__":
    main()
