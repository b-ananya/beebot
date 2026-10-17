# sim/ — RL prototype for the bee path

A proposal, not part of PLAN.md yet: instead of (or next to) the OR-Tools route planner, train a
small policy that *behaves like a bee* on the 0–1000 field grid and draw its predicted path over a
flower-only clip. Everything here lives under `sim/` and `rollouts/`.

```bash
python -m sim.bee_stats                 # measure speed / dwell from truth/sm*.json -> sim/bee_stats.json
python -m sim.train                     # ~5 min CPU PPO -> sim/policy.zip, sim/eval.json
python -m sim.rollout sm3_12_123903     # -> rollouts/sm3_12_123903_rl.mp4 + .json
python -m pytest tests/test_sim.py
```

Needs `gymnasium` and `stable-baselines3` in `.venv` (`pip install gymnasium stable-baselines3`).
`sim.rollout` reads `clips/<clip>.mp4`, so run it from the main checkout (or pass `--video`).

## What the agent learned from

- **Real bee numbers, not real bee paths.** From the 10 farm truth files (`truth/sm*.json`,
  Polytrack reference tracks) we take the median flight speed outside visit events
  (284 grid units/s), the median visit dwell (3.33 s) and the median flower radius (30 units).
  These are the env's speed cap, the time the agent must hold on a flower to score, and the
  size of the flower targets. See `sim/bee_stats.json`.
- **Random fields.** Training episodes drop 4–10 flowers of that size at random positions and
  start the agent at a random point. The real farm layouts are only used for evaluation and
  rendering, so the policy has never seen them.
- **The reward.** +1 for reaching an unvisited open flower and dwelling the median dwell time,
  −0.01 per 0.1 s step, episode over after the time budget (30 s in training). A potential-based
  shaping term (distance to the nearest unvisited flower) speeds up learning without changing
  which policy is optimal.
- **What it sees.** Its own position, time left, dwell progress, and the relative position of
  the 6 nearest unvisited open flowers. Flowers with `state` `bud`/`fruit`/`unknown` are drawn
  but never rewarded, matching the CONTRACT.md rule that only open flowers get a deficit.

## What it did not learn from

- **No pixels.** The policy never sees video. Flowers arrive as boxes (today from `truth/`,
  later from `tracks/parts/<clip>.flowers.json` when Mk1.5 produces them); the video is only
  the backdrop for the rendered dot.
- **No flight physics.** The agent is a point with a velocity cap. No inertia, wind, wings,
  hovering, occlusion by leaves, or the real bee's habit of circling before landing.
- **No bee preferences.** It does not know species, flower quality, scent, or which flowers the
  real bee actually chose. "Looks like a bee" means "moves at bee speed and stops for bee dwell
  time", nothing more.
- **No learning from the real tracks.** The truth tracks set three constants; they are not
  imitation data. A next step would be behaviour cloning on the 48 clips' tracks.

## How it compares to the baseline

The baseline is *nearest-unvisited*: fly at the speed cap straight to the closest unvisited
open flower, hold still for the dwell time, repeat. On the farm layouts this is close to
optimal, because the flowers are few and the dwell time dominates travel time.

Trained 5 min on CPU (1.1 M steps, `sim/eval.json`). Flowers visited per budget on the 10 real
farm layouts, both starting where the real bee entered the frame:

| Budget | Layouts | PPO visits (mean) | Baseline visits (mean) | Layouts where PPO < baseline |
|---|---|---|---|---|
| 15 s (clip length) | 10 | 3.3 | 3.4 | 1 of 10 (`sm3_12_123810`: 3 vs 4) |
| 30 s (training budget) | 10 | 6.5 (all open flowers) | 6.5 (all open flowers) | 0 of 10 |

Mean time to the first visit is 4.1 s for PPO and 4.0 s for the baseline. A 30-second training
run reached only 1.3 visits, so the policy really is learned, not a lucky initialisation.

Read the table as: the baseline is the ceiling for this toy problem, and PPO's job was to
rediscover it from reward alone. Where PPO ties it, the RL plumbing works; where it falls short,
it is a training-budget gap, not a modelling one. The RL framing earns its keep once the env
gets things a greedy rule cannot handle: flower deficits as reward weights, several bees or
robots sharing a field, flowers that close, or motion costs that are not straight lines.

## Files

| Path | What |
|---|---|
| `sim/bee_stats.py` → `sim/bee_stats.json` | median flight speed, dwell, flower radius from `truth/sm*.json` |
| `sim/field_env.py` | `FieldEnv` (Gymnasium), `random_layout`, `nearest_unvisited_action` baseline |
| `sim/train.py` → `sim/policy.zip`, `sim/eval.json` | PPO training and the per-layout comparison |
| `sim/rollout.py` → `rollouts/<clip>_rl.mp4`, `rollouts/<clip>_rl.json` | render + waypoints `[{t, x, y, flower_id}]` |
| `tests/test_sim.py` | env reset/step, reward fires on a visit, rollout JSON shape |

`rollouts/*.mp4` is gitignored; the JSON is committed. The rollout starts the dot where the
real bee entered the frame (first truth point) so the overlay is comparable with the truth
track; without a track it starts at the bottom-right corner.
