"""Forecasting models: where the insect goes in the next seconds, which flower it visits next, how long it stays.

Path forecast (10 Hz, 1 s history -> 2 s future, positions relative to the current one):
- ``stay``        it doesn't move
- ``const_vel``   keeps its last 0.3 s velocity
- ``knn``         median of the futures of the 50 most similar moments in the training set; the 25 nearest
                  futures are the forecast fan (multi-modal: a bee near a flower may land or fly past)
- ``mlp``         small neural net on the same features
Features: past displacements, speed, whether it's on a flower, and the 3 nearest flowers relative to it.

Next flower (decision at take-off): every flower at the station is a candidate; a gradient-boosted classifier
scores candidates (distance, distance rank, heading toward it, visited before, is the current flower, size) and the
scores are normalized per decision. Baselines: nearest other flower, uniform random.

Continue: probability the insect visits another flower at all (vs leaving the patch).
Dwell: empirical quantiles per species.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HZ = 10
PAST = 10            # 1.0 s
FUT = 20             # 2.0 s
N_NEAR = 3


# ------------------------------------------------------------------ path features
def path_features(past_xy: np.ndarray, flowers_xyr: np.ndarray, on_flower: bool) -> np.ndarray:
    """past_xy: (PAST+1, 2) oldest..current, units. flowers_xyr: (n, 3)."""
    cur = past_xy[-1]
    rel = (past_xy[:-1] - cur).ravel()                                  # 2*PAST
    v = (past_xy[-1] - past_xy[-4]) / 0.3                               # recent velocity
    speed = np.linalg.norm(np.diff(past_xy, axis=0), axis=1) * HZ
    fl = flowers_xyr[~np.isnan(flowers_xyr[:, 0])] if len(flowers_xyr) else np.zeros((0, 3))
    near = np.zeros((N_NEAR, 4))
    near[:, :2] = 300.0
    near[:, 2] = 424.0
    if len(fl):
        d = np.linalg.norm(fl[:, :2] - cur, axis=1)
        for k, j in enumerate(np.argsort(d)[:N_NEAR]):
            near[k] = [fl[j, 0] - cur[0], fl[j, 1] - cur[1], d[j], fl[j, 2]]
    return np.concatenate([rel, v, [speed.mean(), speed.max(), float(on_flower)], near.ravel()]).astype(np.float32)


def path_samples(tracks, flowers_of, stride: int = 3):
    """X, Y (FUT*2 future displacements), groups (station) from Polytrack tracks."""
    from forecast.data import resample

    X, Y, G, S = [], [], [], []
    for tr in tracks:
        t, xy, fl = resample(tr, HZ)
        ok = ~np.isnan(xy[:, 0])
        fxyr = flowers_of(tr.station)
        for i in range(PAST, len(t) - FUT, stride):
            if not ok[i - PAST:i + FUT + 1].all():
                continue
            X.append(path_features(xy[i - PAST:i + 1], fxyr, fl[i] >= 0))
            Y.append((xy[i + 1:i + FUT + 1] - xy[i]).ravel())
            G.append(tr.station)
            S.append(tr.species)
    return np.array(X), np.array(Y, dtype=np.float32), np.array(G), np.array(S)


def const_vel(X: np.ndarray) -> np.ndarray:
    v = X[:, 2 * PAST:2 * PAST + 2]                                    # units/s
    steps = np.arange(1, FUT + 1) / HZ
    return (v[:, None, :] * steps[None, :, None]).reshape(len(X), -1)


def ade_fde(pred: np.ndarray, Y: np.ndarray) -> dict:
    p, y = pred.reshape(len(pred), FUT, 2), Y.reshape(len(Y), FUT, 2)
    err = np.linalg.norm(p - y, axis=2)
    return {"ade": round(float(err.mean()), 1), "fde_1s": round(float(err[:, HZ - 1].mean()), 1),
            "fde_2s": round(float(err[:, -1].mean()), 1), "median_fde_2s": round(float(np.median(err[:, -1])), 1)}


@dataclass
class PathModel:
    scaler: object
    knn: object
    Y_train: np.ndarray
    mlp: object | None = None
    best: str = "knn"

    def fan(self, x: np.ndarray, k: int = 25) -> np.ndarray:
        """(k, FUT, 2) futures of the k most similar training moments."""
        _, idx = self.knn.kneighbors(self.scaler.transform(x[None]), n_neighbors=k)
        return self.Y_train[idx[0]].reshape(k, FUT, 2)

    def predict(self, X: np.ndarray, which: str | None = None) -> np.ndarray:
        which = which or self.best
        if which == "stay":
            return np.zeros((len(X), FUT * 2))
        if which == "const_vel":
            return const_vel(X)
        Xs = self.scaler.transform(X)
        if which == "mlp" and self.mlp is not None:
            return self.mlp.predict(Xs)
        # median of the neighbors' futures: the mean blurs "lands here" and "flies past" into a path nobody takes
        _, idx = self.knn.kneighbors(Xs, n_neighbors=50)
        return np.median(self.Y_train[idx], axis=1)


# ------------------------------------------------------------------ next flower
CAND_FEATURES = ["dist", "dist_rank", "dist_rel", "cos_heading", "visited", "is_current", "radius", "n_flowers"]


def candidate_features(cur_xy, heading, flowers_xyr, current: int, visited_counts: dict) -> tuple[np.ndarray, list]:
    """One row per candidate flower (CAND_FEATURES), and the flower indices."""
    idx = [j for j in range(len(flowers_xyr)) if not np.isnan(flowers_xyr[j, 0])]
    if not idx:
        return np.zeros((0, len(CAND_FEATURES))), []
    F = flowers_xyr[idx]
    d = np.linalg.norm(F[:, :2] - cur_xy, axis=1)
    order = np.argsort(np.argsort(d))
    others = d[np.array(idx) != current]
    med = float(np.median(others)) if len(others) else 1.0
    rows = []
    hn = np.linalg.norm(heading)
    for k, j in enumerate(idx):
        vec = F[k, :2] - cur_xy
        cos = float(vec @ heading / (np.linalg.norm(vec) * hn)) if hn > 1e-6 and np.linalg.norm(vec) > 1e-6 else 0.0
        rows.append([d[k], order[k], d[k] / max(med, 1e-6), cos, visited_counts.get(j, 0), float(j == current),
                     F[k, 2], len(idx)])
    return np.array(rows, np.float32), idx


def choice_samples(tracks, flowers_of):
    """Rows for every candidate at every take-off that is followed by another visit."""
    from forecast.data import visits

    X, y, dec, G = [], [], [], []
    cont_X, cont_y, cont_G = [], [], []
    dwell = {}
    did = 0
    for tr in tracks:
        vs = visits(tr)
        fxyr = flowers_of(tr.station)
        if not len(fxyr):
            continue
        counts: dict[int, int] = {}
        for k, v in enumerate(vs):
            dwell.setdefault(tr.species, []).append(v.t_end - v.t_start)
            counts[v.flower] = counts.get(v.flower, 0) + 1
            if v.flower >= len(fxyr) or np.isnan(fxyr[v.flower, 0]):
                continue
            cur = fxyr[v.flower, :2]
            # heading: from where the insect was 0.5 s before landing, toward this flower
            j0 = max(v.i_start - 15, 0)
            before = tr.xy[j0:v.i_start + 1]
            before = before[~np.isnan(before[:, 0])]
            heading = (cur - before[0]) if len(before) else np.zeros(2)
            unvisited = [j for j in range(len(fxyr)) if j not in counts and not np.isnan(fxyr[j, 0])]
            d_unv = min((np.linalg.norm(fxyr[j, :2] - cur) for j in unvisited), default=500.0)
            cont_X.append([k + 1, v.t_end - v.t_start, len(fxyr), len(unvisited), d_unv,
                           {"honeybee": 0, "vespidae": 1, "syrphidae": 2, "lepidoptera": 3}.get(tr.species, 4)])
            cont_y.append(int(k + 1 < len(vs)))
            cont_G.append(tr.station)
            if k + 1 >= len(vs):
                continue
            nxt = vs[k + 1].flower
            rows, idx = candidate_features(cur, heading, fxyr, v.flower, counts)
            if nxt not in idx:
                continue
            for r, j in zip(rows, idx):
                X.append(r)
                y.append(int(j == nxt))
                dec.append(did)
                G.append(tr.station)
            did += 1
    return (np.array(X), np.array(y), np.array(dec), np.array(G),
            np.array(cont_X, np.float32), np.array(cont_y), np.array(cont_G), dwell)


def topk_accuracy(scores: np.ndarray, y: np.ndarray, dec: np.ndarray, ks=(1, 3)) -> dict:
    out = {f"top{k}": 0 for k in ks}
    n = 0
    for d in np.unique(dec):
        m = dec == d
        order = np.argsort(-scores[m])
        rank = int(np.where(y[m][order] == 1)[0][0])
        for k in ks:
            out[f"top{k}"] += rank < k
        n += 1
    return {k: round(v / n, 3) for k, v in out.items()} | {"decisions": n}


@dataclass
class FlowerModel:
    clf: object
    cont: object
    dwell_q: dict          # species -> {q10, q50, q90, mean}
    transit_speed: float   # median units/s between flowers
    dwell_samples: dict | None = None    # species -> sorted observed dwell times (s)

    def next_probs(self, rows: np.ndarray) -> np.ndarray:
        if not len(rows):
            return rows
        s = self.clf.predict_proba(rows)[:, 1]
        s = np.clip(s, 1e-6, None)
        return s / s.sum()

    def p_continue(self, visit_num: int, dwell_s: float, n_flowers: int, n_unvisited: int, d_unvisited: float,
                   species: str) -> float:
        sp = {"honeybee": 0, "vespidae": 1, "syrphidae": 2, "lepidoptera": 3}.get(species, 4)
        return float(self.cont.predict_proba([[visit_num, dwell_s, n_flowers, n_unvisited, d_unvisited, sp]])[0, 1])

    def dwell(self, species: str) -> dict:
        return self.dwell_q.get(species) or self.dwell_q.get("all")

    def _samples(self, species: str) -> np.ndarray:
        s = self.dwell_samples or {}
        return np.asarray(s.get(species) if s.get(species) is not None and len(s.get(species)) >= 30 else s.get("all"))

    def remaining(self, species: str, elapsed: float) -> float:
        """Median remaining dwell given the insect has already stayed ``elapsed`` s: median of D - e over D > e."""
        d = self._samples(species)
        rest = d[d > elapsed] - elapsed
        return float(np.median(rest)) if len(rest) >= 5 else 0.0

    def sample_dwell(self, species: str, rng, size) -> np.ndarray:
        return rng.choice(self._samples(species), size=size)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
