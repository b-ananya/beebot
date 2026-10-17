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
sim/  app/  prompts/  scripts/
```

Until A ships real `tracks/*.json` (gate G2, hour 8), C and D build against `tracks/mock/`.
Run `python -I scripts/validate.py` before pushing any tracks or truth file.
