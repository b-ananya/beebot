"""Polytrack trajectories (Spatial Monitoring dataset, 2,335 insect tracks over a strawberry farm) as arrays.

Source: raw/trajectories/ (``Insect_Trajectory_Dataset.zip``, figshare file 38169609, CC BY 4.0, see DATA_NOTES).
Cameras are 1920x1080 at 30 fps, looking down like our ``sm*`` clips.

Units match ``analytics.metrics`` after ``isotropic``: 1 unit = 1/1000 of the frame width on both axes, so a model
trained here runs on our pipeline output for the farm clips without rescaling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# scripts/fetch_data.py extracts to raw/spatial_monitoring/trajectories; ensure_downloaded() uses raw/trajectories
RAW = next((p for p in (ROOT / "raw" / "trajectories", ROOT / "raw" / "spatial_monitoring" / "trajectories")
            if (p / "Extracted_Tracks").exists()), ROOT / "raw" / "trajectories")
ZIP_URL = "https://ndownloader.figshare.com/files/38169609"     # Insect_Trajectory_Dataset.zip, 100 MB
TRACKS = RAW / "Extracted_Tracks" / "Insect_Tracks_CSV"
FLOWERS = RAW / "Extracted_Tracks" / "Flower_Positions_CSV"
SUMMARY = RAW / "Pollination_Analysis_Summary" / "track_summary.csv"
FPS = 30.0
SRC_W = 1920.0
SPECIES = {"honeybee": "honeybee", "wasp": "vespidae", "hoverfly": "syrphidae", "moth": "lepidoptera"}


@dataclass
class Track:
    track_id: str
    station: str                 # Polytrack data point (camera + day), e.g. "109"
    species: str
    t: np.ndarray                # seconds from track start
    xy: np.ndarray               # (n, 2) units, NaN when not observed
    flower: np.ndarray           # (n,) flower index the insect is on, -1 when none


@dataclass
class Visit:
    flower: int
    t_start: float
    t_end: float
    i_start: int
    i_end: int


def to_units(x, y):
    return np.asarray(x, float) / SRC_W * 1000, np.asarray(y, float) / SRC_W * 1000


@lru_cache(maxsize=None)
def flowers(station: str) -> np.ndarray:
    """(n, 3) array of x, y, radius in units; row index = Polytrack flower number."""
    p = FLOWERS / f"flowers_{station}.csv"
    if not p.exists():
        return np.zeros((0, 3))
    d = pd.read_csv(p)
    d = d.dropna(subset=["flower_num", "x0", "y0"]).sort_values("flower_num")
    if d.empty:
        return np.zeros((0, 3))
    x, y = to_units(d["x0"], d["y0"])
    r = d["radius"].to_numpy(float) / SRC_W * 1000
    out = np.full((int(d["flower_num"].max()) + 1, 3), np.nan)
    out[d["flower_num"].astype(int).to_numpy()] = np.stack([x, y, r], axis=1)
    return out


def ensure_downloaded() -> None:
    """Fetch and unzip the trajectory dataset into raw/trajectories/ if it isn't there (CSV and PNG only)."""
    if TRACKS.exists():
        return
    import urllib.request
    import zipfile

    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "Insect_Trajectory_Dataset.zip"
    if not zpath.exists():
        print(f"downloading {ZIP_URL} (100 MB) -> {zpath}")
        urllib.request.urlretrieve(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW, members=[m for m in z.namelist() if m.endswith((".csv", ".png", "/"))])


def load_tracks(limit: int | None = None) -> list[Track]:
    ensure_downloaded()
    summary = pd.read_csv(SUMMARY)
    species = {str(r.track): SPECIES.get(r.species, "unknown") for r in summary.itertuples()}
    out = []
    for p in sorted(TRACKS.glob("*.csv"))[:limit]:
        tid = p.stem
        d = pd.read_csv(p)
        d.columns = [re.sub(r"_\d+$", "", c) for c in d.columns]
        n0 = d["nframe" if "nframe" in d else "absframe"].to_numpy()     # one file has a broken nframe header
        x, y = to_units(d["x0"], d["y0"])
        fl = d["flower"].to_numpy(float)
        out.append(Track(tid, tid[:3], species.get(tid, "unknown"), (n0 - n0[0]) / FPS, np.stack([x, y], axis=1),
                         np.where(np.isnan(fl), -1, fl).astype(int)))
    return out


def visits(tr: Track, min_frames: int = 3, merge_gap_s: float = 0.5) -> list[Visit]:
    """Runs of the per-frame flower id; short gaps on the same flower are bridged."""
    runs = []
    cur, start = -1, 0
    for i, f in enumerate(list(tr.flower) + [-1]):
        if f != cur:
            if cur >= 0 and i - start >= min_frames:
                runs.append(Visit(cur, float(tr.t[start]), float(tr.t[i - 1]), start, i - 1))
            cur, start = f, i
    merged: list[Visit] = []
    for v in runs:
        if merged and merged[-1].flower == v.flower and v.t_start - merged[-1].t_end <= merge_gap_s:
            merged[-1].t_end, merged[-1].i_end = v.t_end, v.i_end
        else:
            merged.append(v)
    return merged


def resample(tr: Track, hz: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (FPS / hz)-th frame: t, xy, flower."""
    step = int(round(FPS / hz))
    return tr.t[::step], tr.xy[::step], tr.flower[::step]
