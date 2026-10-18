"""Apply the trained forecasters to a CONTRACT.md tracks document (pipeline output or truth) at time t.

    python -m forecast.predict sm3_12_123903 --t 5.0                 # tracks/<clip>.json
    python -m forecast.predict sm3_12_123903 --t 5.0 --source truth

Everything is computed in isotropic units (y scaled by H/W, see analytics.metrics) and converted back to the
contract's 0-1000 coordinates for drawing. Models were trained on the farm cameras (sm* clips come from the same
farm); on the honeybee close-ups (hb*) the scale differs, so treat those forecasts as qualitative.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from analytics.metrics import center, isotropic
from forecast.models import FUT, HZ, PAST, candidate_features, path_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "forecast" / "models"
ASPECT = 720 / 1280


@lru_cache(maxsize=1)
def models():
    return joblib.load(MODELS / "path.joblib"), joblib.load(MODELS / "flower.joblib")


def available() -> bool:
    return (MODELS / "path.joblib").exists() and (MODELS / "flower.joblib").exists()


def _flowers_xyr(doc_iso: dict) -> tuple[np.ndarray, list[str]]:
    rows, ids = [], []
    for f in doc_iso.get("flowers", []):
        b = f["box"]
        rows.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) / 2])
        ids.append(f["flower_id"])
    return np.array(rows).reshape(-1, 3), ids


def _history(track: dict, t: float) -> np.ndarray | None:
    """(PAST+1, 2) positions at 10 Hz ending at t, linearly filled across short gaps; None if not enough data."""
    obs = [p for p in track["points"] if not p.get("interpolated") and p["t"] <= t + 1e-6]
    if len(obs) < 3 or t - obs[-1]["t"] > 0.3 or obs[-1]["t"] - obs[0]["t"] < 0.5:
        return None
    ts = np.array([p["t"] for p in obs])
    cs = np.array([center(p) for p in obs])
    grid = obs[-1]["t"] - np.arange(PAST, -1, -1) / HZ
    grid = np.maximum(grid, ts[0])
    return np.stack([np.interp(grid, ts, cs[:, 0]), np.interp(grid, ts, cs[:, 1])], axis=1)


def _to_contract(xy_iso: np.ndarray, aspect: float) -> np.ndarray:
    out = np.array(xy_iso, float).copy()
    out[..., 1] = out[..., 1] / aspect
    return out


def current_visit(doc: dict, track_id: str, t: float) -> dict | None:
    vs = [e for e in doc.get("events", []) if e["type"] == "visit" and e["track_id"] == track_id
          and e["t_start"] <= t + 1e-6]
    return max(vs, key=lambda e: e["t_end"]) if vs else None


def forecast(doc: dict, track_id: str, t: float, *, aspect: float = ASPECT, horizon_s: float = 10.0,
             rollouts: int = 300, seed: int = 0) -> dict:
    pm, fm = models()
    iso = isotropic(doc, aspect)
    track = next(tr for tr in iso["tracks"] if tr["track_id"] == track_id)
    fxyr, fids = _flowers_xyr(iso)
    out: dict = {"track_id": track_id, "t": t}

    hist = _history(track, t)
    visit = current_visit(doc, track_id, t)
    on_flower = bool(visit and visit["t_end"] >= t - 0.1)
    if hist is not None:
        x = path_features(hist, fxyr, on_flower)
        med = pm.predict(x[None])[0].reshape(FUT, 2) + hist[-1]          # the model train.py picked (pm.best)
        fan = pm.fan(x, k=25) + hist[-1]
        out["path"] = {"t": (t + np.arange(1, FUT + 1) / HZ).round(2).tolist(),
                       "median": _to_contract(med, aspect).round(1).tolist(),
                       "fan": _to_contract(fan, aspect).round(1).tolist(),
                       "history": _to_contract(hist, aspect).round(1).tolist()}

    species = next((tr.get("species", "unknown") for tr in doc["tracks"] if tr["track_id"] == track_id), "unknown")
    dq = fm.dwell(species)
    out["dwell_model_s"] = dq
    if not len(fids):
        return out

    # decision state: the flower it's on (or last visited) and the visits so far
    past_visits = [e for e in doc.get("events", []) if e["type"] == "visit" and e["track_id"] == track_id
                   and e["t_start"] <= t]
    counts: dict[int, int] = {}
    for e in past_visits:
        if e["flower_id"] in fids:
            j = fids.index(e["flower_id"])
            counts[j] = counts.get(j, 0) + 1
    if visit and visit["flower_id"] in fids:
        cur = fids.index(visit["flower_id"])
        cur_xy = fxyr[cur, :2]
    elif hist is not None:
        cur = -1
        cur_xy = hist[-1]
    else:
        return out
    heading = (hist[-1] - hist[0]) if hist is not None else np.zeros(2)
    rows, idx = candidate_features(cur_xy, heading, fxyr, cur, counts)
    probs = fm.next_probs(rows)
    order = np.argsort(-probs)
    out["next_flower"] = [{"flower_id": fids[idx[k]], "p": round(float(probs[k]), 3)} for k in order[:5]]
    if on_flower:
        out["dwell_remaining_s"] = round(fm.remaining(species, t - visit["t_start"]), 2)
    unvisited = [j for j in range(len(fids)) if j not in counts]
    d_unv = min((float(np.linalg.norm(fxyr[j, :2] - cur_xy)) for j in unvisited), default=500.0)
    p_cont = fm.p_continue(len(past_visits) or 1, (visit["t_end"] - visit["t_start"]) if visit else 0.0,
                           len(fids), len(unvisited), d_unv, species)
    out["p_another_visit"] = round(p_cont, 3)

    # Monte Carlo rollouts over the next horizon_s: sample dwell, next flower, transit, repeat
    rng = np.random.default_rng(seed)
    expected = np.zeros(len(fids))
    sequences = []
    leave = 0
    heading0 = heading
    for r in range(rollouts):
        heading = heading0                 # every rollout starts from the insect's real heading
        tt = t + (out["dwell_remaining_s"] if on_flower else 0.0)
        c, c_xy, cnt, seq = cur, cur_xy, dict(counts), []
        n_done = len(past_visits)
        while tt < t + horizon_s:
            dwell_k = float(max(fm.sample_dwell(species, rng, 1)[0], 0.1))
            unv = [j for j in range(len(fids)) if j not in cnt]
            du = min((float(np.linalg.norm(fxyr[j, :2] - c_xy)) for j in unv), default=500.0)
            if n_done and rng.random() > fm.p_continue(n_done, dwell_k, len(fids), len(unv), du,
                                                       species):
                leave += 1
                break
            rows, idx = candidate_features(c_xy, heading, fxyr, c, cnt)
            pr = fm.next_probs(rows)
            j = idx[rng.choice(len(idx), p=pr)]
            d = float(np.linalg.norm(fxyr[j, :2] - c_xy))
            tt += d / fm.transit_speed
            if tt >= t + horizon_s:
                break
            expected[j] += 1
            seq.append((fids[j], round(tt - t, 2)))
            cnt[j] = cnt.get(j, 0) + 1
            heading = fxyr[j, :2] - c_xy
            c, c_xy = j, fxyr[j, :2]
            tt += dwell_k
            n_done += 1
        sequences.append(tuple(f for f, _ in seq))
    expected /= rollouts
    out["rollout"] = {
        "horizon_s": horizon_s, "n": rollouts,
        "expected_visits": {fids[j]: round(float(expected[j]), 3) for j in np.argsort(-expected) if expected[j] > 0},
        "p_leaves_patch": round(leave / rollouts, 3),
        "most_likely_sequence": list(max(set(sequences), key=sequences.count)) if sequences else [],
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--source", default="tracks")
    ap.add_argument("--track")
    args = ap.parse_args(argv)
    d = ROOT / ("truth" if args.source == "truth" else args.source)
    doc = json.loads((d / f"{args.clip}.json").read_text())
    tid = args.track or doc["tracks"][0]["track_id"]
    out = forecast(doc, tid, args.t)
    if "path" in out:
        out["path"] = {k: (v if k != "fan" else f"{len(v)} sampled futures") for k, v in out["path"].items()
                       if k in ("median", "fan")}
        out["path"]["median"] = out["path"]["median"][::5]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
