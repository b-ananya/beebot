# BeeBot

Bee-pollination video → Mk1.5 tracking, analytics and search → a pollination map → the optimal route for a robot pollinator.
See [PLAN.md](PLAN.md) for the plan and roles, [CONTRACT.md](CONTRACT.md) for the file formats every workstream shares, and
[DATA_NOTES.md](DATA_NOTES.md) for where the data comes from.

## Setup (once)

Install Python 3.10+, Git, the GitHub CLI and ffmpeg:

| Windows | Mac |
|---|---|
| `winget install Python.Python.3.11 Git.Git GitHub.cli Gyan.FFmpeg` | `brew install python@3.11 git gh ffmpeg` |

Open a new terminal afterwards so PATH updates. Then:

```bash
gh auth login
gh repo clone <owner>/beebot && cd beebot
python -m venv .venv
.venv\Scripts\activate            # Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add PERCEPTRON_API_KEY
```

## Get the data (2 minutes, no Monash download)

```bash
gh release download data-v1 --pattern "beebot-data-v1.zip"
python -I -c "import zipfile; zipfile.ZipFile('beebot-data-v1.zip').extractall('.')"
python -I scripts/validate.py     # should print 96/96 files valid
```

You now have `clips/*.mp4` (48 clips) next to `clips/manifest.json`, `truth/` and `tracks/mock/` from git.

Rebuilding from scratch (only B needs this): `python -I scripts/fetch_data.py` then `python -I scripts/prepare_data.py`.

## Pipeline

```bash
python -m perception.pipeline --all            # motion tracker -> Mk1.5 crop checks -> Mk1.5 flower boxes -> tracks/*.json
python -m perception.pipeline --all --no-api   # same without the API (score heuristic, stop-based flowers) -> tracks/heuristic/
python -m eval.run_eval                        # vs truth -> eval/report.json  (--pred tracks/heuristic for the no-API run)
python -m search.search build                  # search/index.json (add --captions for Mk1.5 moment captions)
python -m forecast.train                       # downloads the Polytrack trajectories (100 MB) once, trains in ~1 min
```

Why not let Mk1.5 track the whole clip: on `hb1_w00` it returned an evenly spaced straight line across the frame
(3.9% coverage, `eval/report_mk15_v1.json`); the bee is ~20 px and the clip is ~9k tokens. The cameras are fixed, so
`perception/cv_track.py` finds moving blobs against the clip's median background and links them with a global
best-path search (it bridges occlusions of up to 0.5 s). Its top path is the bee in every honeybee clip we checked,
but it also proposes swaying flowers. `perception/verify.py` therefore asks Mk1.5 to detect insects in a 160 px
crop every 0.5 s along each path and keeps paths where at least 70% of crops contain one (single crops are weak
evidence: 93% recall but many false boxes; a path of 10-20 is strong). Species come from dataset metadata or the
user, because Mk1.5 called every insect a hoverfly at this resolution. Flowers come from the project's one flower
detector, `perception/flowers.py` (`detect_tiled`: Mk1.5 on the clip's insect-free median frame in 2x2 tiles, dense
ground cover skipped so stops become the flowers; ids stable per camera view); `perception/visits.py` adds visits:
a visit is the insect inside a flower box for ≥ 0.3 s (Polytrack's rule).

**Flower boxes** (`python -m eval.flowers_eval`, 10 farm clips = 2 fixed-camera scenes): vs a visual audit of every
open flower (`eval/audit/flower_labels.json`, needs a human spot-check) precision 0.98, recall 0.87 per clip; per
scene 9/9 and 11/13 boxes are open flowers and 9/9 and 11/13 open flowers are found. Against the dataset's flower
list the same boxes score 0.36 / 0.54 by box overlap and 0.57 / 0.86 by center distance: that list holds only the
flowers Polytrack used (most "extras" are real flowers), as radius-estimated squares that sit ~13 px off on camera 3.
Tried and not adopted (scratch study, 2 clips vs the audit): strawberry-specific prompt 0.91/0.91, 3x3 tiles
0.46/0.95, 3x3 proposals confirmed on crops 0.95/0.91, vs current 1.00/0.86; adopted: a flower found in >= 2 clips
of a camera is added to the clips that missed it (`visits.fill_view`).

`tracks/*.json` is written by `perception.pipeline`. The earlier hybrid tracker's tracks (Raika, 2 fps) are kept in
`tracks/hybrid/` for comparison: on the same eval they cover 38% of the reference path vs 90% for the pipeline
(0.25 s window; pipeline better on 45 of 46 clips). Note `python -m perception.merge` rebuilds `tracks/*.json` from
`tracks/parts/` and would overwrite the pipeline output; use `merge_clip(..., out_dir=...)` when experimenting.

Forecasting (`forecast/`) trains on the 2,335 Polytrack farm trajectories, scored on 11 camera stations held out of
training (`forecast/report.json`): next-flower top-3 accuracy 79% (nearest-flower baseline 68%); 2 s path forecast as
a fan of 25 sampled futures, whose closest member ends a median 8 units (~0.3 cm) from the real position.

## Layout

```
clips/        720p working clips (mp4 via release) + manifest.json (git)
truth/        reference tracks in contract format
tracks/       model output per clip; tracks/raw/ keeps every API response; tracks/mock/ for C and D
rollouts/     sim and plan videos (mp4 out of git)
search/       natural-language search index
field/        pollination maps per clip and for the farm
plan/         robot routes and baseline scores
eval/         run_eval.py, report.json
analytics/    metrics.py: visits, dwell, transit, speed, revisits, approach, turn angles (robot parameters)
forecast/     Polytrack data, path + next-flower models, rollouts
sim/  app/  prompts/  scripts/
```

Until A ships real `tracks/*.json` (gate G2, hour 8), C and D build against `tracks/mock/`.
Run `python -I scripts/validate.py` before pushing any tracks or truth file.
