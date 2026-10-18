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

## 1. The app (start here)

```powershell
streamlit run app/app.py
```

Opens http://localhost:8501. Sidebar: pick the data source (Mk1.5 pipeline, motion-only, reference tracker, mock)
and a clip. Tabs: video + search, analytics, forecast, dataset & accuracy, analyze your own video.
Shareable link to a view: `http://localhost:8501/?source=truth&clip=sm3_12_123903`.

## 2. The pipeline (video -> tracks, flowers, visits)

| Command | What it does | Cost |
|---|---|---|
| `python -m perception.pipeline --all` | all 48 clips: motion tracker -> Mk1.5 crop checks -> Mk1.5 flower boxes -> `tracks/<clip>.json` | ~$1-2 first time, then free |
| `python -m perception.pipeline --clips sm3_12_123903` | one clip | cents |
| `python -m perception.pipeline --all --no-api` | same without Mk1.5 -> `tracks/heuristic/` | free |
| `python -m perception.pipeline --video my.mp4 --species honeybee` | any video (first 60 s) -> `tracks/uploads/` | cents |

The stages can also run alone: `perception.cv_track`, `perception.verify`, `perception.visits` (each has `--help`).

## 3. Accuracy

```powershell
python -m eval.run_eval                                   # tracks/ vs truth/ -> eval/report.json
python -m eval.run_eval --pred tracks/heuristic --out eval/report_heuristic.json
```

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

## 6. Tests

```powershell
python -m pytest -q                                  # offline, no API calls
python -I scripts/validate.py                        # contract check for truth/ and tracks/
```
