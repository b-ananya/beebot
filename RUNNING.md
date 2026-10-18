# Running BeeBot

All commands run from the repo folder with the project's Python environment. On this machine that is:

```powershell
cd C:\Users\panan\OneDrive\Desktop\beebot
C:\Users\panan\.venvs\beebot\Scripts\Activate.ps1      # once per terminal; then "python" means the project Python
```

(The environment lives outside OneDrive on purpose, so OneDrive doesn't sync thousands of package files.
On another machine: `python -m venv .venv`, activate it, `pip install -r requirements.txt`.)

The API key goes in `.env` as `PERCEPTRON_API_KEY=...`. Every Mk1.5 answer is cached under `tracks/raw/`, so
running anything twice costs nothing the second time. `costs/<you>.csv` logs every paid call; calls stop at $40.

## 1. The UI

The Streamlit app has been removed. The replacement is the panel UI in `app/web` (Blender-style shell: video in the center, lists and properties on the sides, timeline along the bottom).

Setup and start: [app/web/SETUP.md](app/web/SETUP.md) (Node, `npm install`, `npm run dev`, then http://localhost:5173). With the Python environment outside the repo, set `BEEBOT_PYTHON` in `.env`.

Overlays can still be burned into an mp4 without a UI:

```powershell
python -m app.render --clip sm3_12_123903 --source tracks
```

## 2. The pipeline (video -> tracks, flowers, visits)

| Command | What it does | Cost |
|---|---|---|
| `python -m perception.pipeline --all` | all 48 clips: motion tracker -> Mk1.5 crop checks -> Mk1.5 flower boxes -> `tracks/<clip>.json` | ~$1-2 first time, then free |
| `python -m perception.pipeline --clips sm3_12_123903` | one clip | cents |
| `python -m perception.pipeline --all --no-api` | same without Mk1.5 -> `tracks/heuristic/` | free |
| `python -m perception.pipeline --video my.mp4 --species honeybee` | any video (first 60 s) -> `tracks/uploads/` | cents |

The stages can also run alone: `perception.cv_track`, `perception.verify`, `perception.visits` (each has `--help`).
Flower boxes alone: `python -m perception.flowers --sm --eval --overlay` (same detector as the pipeline; add
`--method middle` for the original single middle-frame call). After changing the flower detector, rerun the
pipeline, `sim.run_plan --all` and `search.search build`, and commit tracks/, field/, plan/ together so flower ids
agree across files.

## 3. Accuracy

```powershell
python -m eval.run_eval                                   # tracks/ vs truth/ -> eval/report.json
python -m eval.run_eval --pred tracks/heuristic --out eval/report_heuristic.json
```

Flower boxes: `python -m eval.flowers_eval` scores the farm clips three ways (dataset list by overlap, dataset list
by center distance, visual audit) -> `eval/flowers_report.json`.

Coverage = share of reference points we match within 50 units; false alarms = our points not near the reference.
The honeybee videos often contain more than one bee and the reference follows only one, so some "false alarms"
are real bees.

## 4. Search

```powershell
python -m search.search build                 # search/index.json from tracks/ (add --captions for Mk1.5 captions)
python -m search.search query "wasp leaves the flower"
python -m search.search live sm3_12_123903 "bee lands on a flower"     # asks Mk1.5 directly (paid, cached)
```

## 5. Forecasting

```powershell
python -m forecast.train                             # downloads trajectories (100 MB) once; ~1 min; forecast/report.json
python -m forecast.predict sm3_12_123903 --t 9.0     # path fan, next flower, dwell, rollouts at t = 9 s
```

`forecast/models/` is gitignored, so a fresh clone has no trained models. The panel Forecast tab says "Forecast endpoint is unavailable" until you run `python -m forecast.train` once (about 2 minutes, no API cost).

## 6. Pollination deficit map and robot route

```powershell
python -m sim.run_plan --all                         # field/<clip>.json + plan/<clip>.json + rollouts/<clip>_plan.mp4
python -m sim.run_plan sm3_12_124130 --budget 60 --target 6
python -m sim.run_plan --all --source truth          # dataset whole-day maps -> field/truth/, plan/truth/
```

The field map for a farm clip holds every flower its camera has seen (stable flower ids across clips) and the
visits counted over all of that camera's clips up to this one.

## 7. Tests

```powershell
python -m pytest -q                                  # offline, no API calls
python -I scripts/validate.py                        # contract check for truth/ and tracks/
```

## 8. Simulations (MuJoCo)

```powershell
pip install -r requirements-stretch.txt                     # mujoco, gymnasium, stable-baselines3
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie ..\mujoco_menagerie
git -C ..\mujoco_menagerie sparse-checkout set bitcraze_crazyflie_2
python -m sim.mj_random --seed 7 --live                     # random field, RL bee, interactive MuJoCo window
python -m sim.mj_random --seed 7 --camera chase             # same, as rollouts/random_7_mujoco.mp4
python -m sim.mj_render sm3_12_123903                       # replay a planned robot route over a real clip
```

In the app, the **Simulations** link in the top bar opens the 3D page. Menagerie elsewhere: set `MUJOCO_MENAGERIE` in `.env`.
