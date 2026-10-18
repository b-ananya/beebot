# Set up the BeeBot frontend

The panel UI lives in `app/web` (React, TypeScript, Vite). Video, tracks, and the other JSON files are served from the repo root. Live search, forecast, and upload call the project's Python from `.venv`, so you do not need the virtualenv activated to run the UI.

Repo-wide Python setup, the data release, and the pipeline are in [README.md](../../README.md) and [RUNNING.md](../../RUNNING.md). This page is only what you need to run the UI.

## What to install

| Tool | Version | Why |
|---|---|---|
| Git | current | clone the repo |
| GitHub CLI (`gh`) | current | download the clip release |
| Node.js | 20.19+ or 22.12+ | Vite 8 will not start on older Node |
| Python | 3.10+ | the dev server shells out for search, forecast, and upload |
| ffmpeg | current | overlays and rendered plan videos |

**Mac**

```bash
brew install git gh node python@3.11 ffmpeg
```

**Windows** (PowerShell, then open a new terminal so PATH updates):

```powershell
winget install Git.Git GitHub.cli OpenJS.NodeJS.LTS Python.Python.3.11 Gyan.FFmpeg
```

The Windows Python package includes the `py` launcher. Use that in the commands below. `python3` is often missing on Windows even when Python is installed.

Check Node before continuing. `node -v` must be `v20.19` or newer, or `v22.12` or newer.

## Mac

From the repo root:

```bash
gh auth login
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Clips are not in git. `clips/manifest.json` is.

```bash
gh release download data-v1 --pattern "beebot-data-v1.zip"
.venv/bin/python -c "import zipfile; zipfile.ZipFile('beebot-data-v1.zip').extractall('.')"
.venv/bin/python -I scripts/validate.py
```

`validate.py` should report the contract files valid. You want `clips/*.mp4` next to the manifest.

```bash
cd app/web
npm install
npm run dev
```

## Windows

From the repo root, in PowerShell:

```powershell
gh auth login
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If `py` is not recognized, close the terminal, open a new one, and try again. The winget install does not update PATH in the terminal that ran it.

Clips are not in git. `clips/manifest.json` is.

```powershell
gh release download data-v1 --pattern "beebot-data-v1.zip"
.\.venv\Scripts\python.exe -c "import zipfile; zipfile.ZipFile('beebot-data-v1.zip').extractall('.')"
.\.venv\Scripts\python.exe -I scripts/validate.py
```

`validate.py` should report the contract files valid. You want `clips\*.mp4` next to the manifest.

```powershell
cd app\web
npm install
npm run dev
```

You do not need to activate `.venv`. The dev server finds `.venv\Scripts\python.exe` on its own. Activating in PowerShell can fail with an execution-policy error; skip activation unless you want a bare `python` command for the optional steps below.

## API key

Put your Perceptron key in the repo-root `.env`:

```
PERCEPTRON_API_KEY=...
```

`.env` stays uncommitted. Video, analytics, and the offline search index work without a key. Live search and "analyze with the API" on an upload do not.

## Open the app

Open [http://localhost:5173](http://localhost:5173). The dev server uses port 5173 and will pick the next free port if that one is taken; use the URL Vite prints.

Leave that terminal running. The app reads files from the repo root (`clips/`, `tracks/`, `search/`, `field/`, `plan/`, `truth/`, `eval/`, `forecast/`, `rollouts/`), so start it from `app/web` inside a normal checkout, not from a copy of this folder on its own.

## What you should see

Pick **Mock** in the source list if pipeline tracks are not generated yet. The default clip is `hb6_w00`. Tracks for Mock are already in `tracks/mock/`.

| Panel | Works when |
|---|---|
| Video | `clips/<id>.mp4` and a track JSON for the selected source exist |
| Analytics | that same track JSON exists |
| Dataset | `clips/manifest.json` loads |
| Field | `field/<clip>.json` exists (farm clips). Routes need `plan/<clip>.json` |
| Forecast | `forecast/models/path.joblib` and `flower.joblib` exist |

Offline search uses `search/index.json`. If that file is missing, the box stays empty until someone builds the index (command below).

## Optional, when you need those panels

From the repo root. On Windows, `python` in these commands means `.\.venv\Scripts\python.exe`.

```bash
python -m forecast.train          # once; downloads ~100 MB, then Forecast works
python -m sim.run_plan --all      # field maps, routes, rollout mp4s
python -m search.search build     # offline search index
```

Upload and live search are buttons in the app. They call that same virtualenv through the dev server. The first forecast or analyze call can take a while; the failure text in the panel is the Python error.

## If it does not come up

- **`npm run dev` exits immediately.** Node is too old. Install 20.19+ or 22.12+ and open a new terminal. On Windows, `node -v` in a terminal opened before the winget install still shows the old version.
- **`py` or `npm` is not recognized.** Open a new PowerShell window. winget does not update PATH in the terminal that ran the install.
- **The page loads and every clip is missing.** The data zip was not extracted into the repo root. `clips/manifest.json` should sit beside real mp4 files.
- **Video plays, overlays do not.** Switch the source to Mock, or generate tracks with `python -m perception.pipeline` (see [RUNNING.md](../../RUNNING.md)).
- **"Python is not available."** Create `.venv` in the repo root with the commands above, then restart `npm run dev`. The server looks for `BEEBOT_PYTHON` first (environment or repo-root `.env`), then `.venv/bin/python` (Mac) and `.venv\Scripts\python.exe` (Windows), then `python` / `python3`.
- **Your virtualenv lives outside the repo** (for example to keep a OneDrive-synced folder light). Point the server at it in the repo-root `.env`: `BEEBOT_PYTHON=C:\Users\<you>\.venvs\beebot\Scripts\python.exe`, then restart `npm run dev`.
- **Forecast says models are not trained.** Run `python -m forecast.train` once.
- **Live search or upload says the API key is missing.** Fill in `PERCEPTRON_API_KEY` in the repo-root `.env` and restart `npm run dev`.
