"""Train and evaluate the forecasters on Polytrack; held-out stations (camera + day) are never seen in training.

    python -m forecast.train                  # -> forecast/models/*.joblib, forecast/report.json

Splitting by station, not by track, means the test scores say how well the model does on a patch of flowers it
has never seen, which is the situation for a new video.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from forecast import data
from forecast.models import (CAND_FEATURES, FlowerModel, PathModel, ade_fde, choice_samples, const_vel,
                             path_samples, topk_accuracy)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "forecast" / "models"
SEED = 0


def split(groups: np.ndarray, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def main() -> int:
    t0 = time.time()
    tracks = data.load_tracks()
    stations = np.array(sorted({t.station for t in tracks}))
    tr_idx, te_idx = split(stations)
    test_st = set(stations[te_idx])
    print(f"{len(tracks)} tracks, {len(stations)} stations, {len(test_st)} held out ({time.time() - t0:.0f}s)")

    report = {"held_out_stations": sorted(test_st), "units": "1/1000 frame width (~0.04 cm at farm scale)"}

    # ---------------- path forecast
    X, Y, G, S = path_samples(tracks, data.flowers)
    te = np.isin(G, list(test_st))
    Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
    print(f"path samples: {len(Xtr)} train, {len(Xte)} test")
    scaler = StandardScaler().fit(Xtr)
    knn = NearestNeighbors(n_neighbors=25).fit(scaler.transform(Xtr))
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=60, early_stopping=True, random_state=SEED)
    mlp.fit(scaler.transform(Xtr), Ytr)
    pm = PathModel(scaler, knn, Ytr, mlp)
    res = {name: ade_fde(pm.predict(Xte, name), Yte) for name in ("stay", "const_vel", "knn", "mlp")}
    # moving moments only (speed > 40 units/s): where forecasting is hard and matters
    moving = Xte[:, 2 * 10 + 2] > 40
    res_moving = {name: ade_fde(pm.predict(Xte[moving], name), Yte[moving]) for name in ("stay", "const_vel", "knn", "mlp")}
    # fan coverage: does any of the 25 neighbor futures end within 30 units of the truth (minFDE@25)?
    sub = np.where(moving)[0][:2000]
    _, idx = knn.kneighbors(scaler.transform(Xte[sub]))
    fut = Ytr[idx].reshape(len(sub), 25, -1, 2)[:, :, -1, :]
    min_fde = np.linalg.norm(fut - Yte[sub].reshape(len(sub), -1, 2)[:, None, -1, :], axis=2).min(axis=1)
    pm.best = min(("knn", "mlp"), key=lambda n: res[n]["ade"])
    report["path"] = {"all": res, "moving": res_moving, "moving_share": round(float(moving.mean()), 3),
                      "knn_fan_minFDE_2s_moving": round(float(np.median(min_fde)), 1), "best": pm.best,
                      "train_samples": int(len(Xtr)), "test_samples": int(len(Xte))}
    print(json.dumps(report["path"], indent=1))

    # ---------------- next flower, continue, dwell
    Xc, yc, dec, Gc, CX, cy, CG, dwell = choice_samples(tracks, data.flowers)
    te = np.isin(Gc, list(test_st))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED).fit(Xc[~te], yc[~te])
    scores = clf.predict_proba(Xc[te])[:, 1]
    nearest = -Xc[te][:, CAND_FEATURES.index("dist")] - 1e6 * Xc[te][:, CAND_FEATURES.index("is_current")]
    rng = np.random.default_rng(SEED)
    report["next_flower"] = {
        "model": topk_accuracy(scores, yc[te], dec[te]),
        "nearest_other_flower": topk_accuracy(nearest, yc[te], dec[te]),
        "random": topk_accuracy(rng.random(te.sum()), yc[te], dec[te]),
        "same_flower_share": round(float(yc[Xc[:, CAND_FEATURES.index("is_current")] == 1].sum() /
                                         len(np.unique(dec))), 3),
        "train_decisions": int(len(np.unique(dec[~te]))),
    }
    tec = np.isin(CG, list(test_st))
    cont = LogisticRegression(max_iter=1000).fit(CX[~tec], cy[~tec])
    acc = float((cont.predict(CX[tec]) == cy[tec]).mean())
    report["continue"] = {"accuracy": round(acc, 3), "base_rate": round(float(cy[tec].mean()), 3),
                          "majority_baseline": round(float(max(cy[tec].mean(), 1 - cy[tec].mean())), 3)}

    dwell_q = {}
    for sp, ds in list(dwell.items()) + [("all", [d for v in dwell.values() for d in v])]:
        ds = np.array(ds)
        dwell_q[sp] = {"n": int(len(ds)), "q10": round(float(np.percentile(ds, 10)), 2),
                       "q50": round(float(np.median(ds)), 2), "q90": round(float(np.percentile(ds, 90)), 2),
                       "mean": round(float(ds.mean()), 2)}
    # transit speed between consecutive flowers
    speeds = []
    for tr in tracks:
        vs = data.visits(tr)
        f = data.flowers(tr.station)
        for a, b in zip(vs[:-1], vs[1:]):
            if a.flower != b.flower and max(a.flower, b.flower) < len(f) and b.t_start - a.t_end > 0.1:
                speeds.append(np.linalg.norm(f[a.flower, :2] - f[b.flower, :2]) / (b.t_start - a.t_end))
    samples = {sp: np.sort(np.array(ds, np.float32)) for sp, ds in dwell.items()}
    samples["all"] = np.sort(np.concatenate(list(samples.values())))
    fm = FlowerModel(clf, cont, dwell_q, float(np.median(speeds)) if speeds else 100.0, samples)
    report["dwell_s"] = dwell_q
    report["transit_speed_units_s"] = round(fm.transit_speed, 1)
    print(json.dumps({k: report[k] for k in ("next_flower", "continue", "transit_speed_units_s")}, indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(pm, OUT / "path.joblib", compress=3)
    joblib.dump(fm, OUT / "flower.joblib", compress=3)
    (ROOT / "forecast" / "report.json").write_text(json.dumps(report, indent=1))
    print(f"saved to {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
