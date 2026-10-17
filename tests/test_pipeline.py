"""Offline tests for eval, verification logic, visits, analytics, search and forecasting features. No network."""
import json

import numpy as np

from analytics.metrics import clip_metrics, isotropic
from eval.run_eval import score_clip
from forecast.models import FUT, PAST, candidate_features, path_features
from perception.flowers import stops, visit_events
from perception.verify import sample_points, split_by_answers
from search.search import derived_moments, query, tokens


def pts(seq, step=0.1, size=10):
    return [{"t": round(i * step, 3), "box": [x - size, y - size, x + size, y + size], "center": [x, y]}
            for i, (x, y) in enumerate(seq)]


def doc(points, flowers=(), events=(), cid="sm_test"):
    return {"clip_id": cid, "model": "test", "tracks": [{"track_id": "honeybee_1", "species": "honeybee",
                                                         "points": points}],
            "flowers": list(flowers), "events": list(events)}


# ------------------------------------------------------------------ eval
def test_score_clip_counts_far_points_as_false_alarms():
    truth = doc(pts([(100, 100)] * 10))
    near = doc(pts([(105, 100)] * 10))
    far = doc(pts([(400, 400)] * 10))
    s = score_clip(truth, near)
    assert s["coverage"] == 1.0 and s["false_alarm_rate"] == 0.0 and s["median_center_error"] == 5.0
    s = score_clip(truth, far)
    assert s["coverage"] == 0.0 and s["false_alarm_rate"] == 1.0


def test_interpolated_points_are_ignored():
    truth = doc(pts([(100, 100)] * 5))
    p = pts([(100, 100)] * 5)
    for q in p:
        q["interpolated"] = True
    assert score_clip(truth, doc(p))["pred_points"] == 0


# ------------------------------------------------------------------ verify
def test_split_trims_tail_after_two_noes():
    points = pts([(100 + i, 100) for i in range(50)])          # 0.0 .. 4.9 s
    samples = sample_points(points, 0.5)
    yes = {"insect_at_center": True, "kind": "honeybee", "behavior": "on_flower", "confidence": "high"}
    no = {"insect_at_center": False, "kind": "none", "behavior": "none", "confidence": "high"}
    answers = [yes if s["t"] < 2.0 else no for s in samples]
    pieces = split_by_answers(points, samples, answers)
    assert len(pieces) == 1
    assert pieces[0]["species"] == "honeybee"
    assert max(p["t"] for p in pieces[0]["points"]) < 2.0
    assert all(p["label"] == "foraging" for p in pieces[0]["points"])


def test_split_drops_unconfirmed_path():
    points = pts([(100, 100)] * 30)
    samples = sample_points(points, 0.5)
    no = {"insect_at_center": False, "kind": "none", "behavior": "none", "confidence": "high"}
    assert split_by_answers(points, samples, [no] * len(samples)) == []


# ------------------------------------------------------------------ visits
def test_visit_inside_flower_and_stop_flower():
    # 1 s inside flower_0, fly away, then sit still 1 s somewhere with no flower
    seq = [(200, 200)] * 10 + [(200 + 60 * i, 200) for i in range(1, 6)] + [(600, 600)] * 10
    events, flowers = visit_events(doc(pts(seq))["tracks"],
                                   [{"flower_id": "flower_0", "box": [180, 180, 220, 220]}])
    ids = [e["flower_id"] for e in events]
    assert "flower_0" in ids
    assert any(f.get("source") == "stop" for f in flowers)
    assert any(i.startswith("stop_") for i in ids)


def test_stops_need_stillness():
    moving = pts([(10 * i, 0) for i in range(40)])
    assert stops(moving) == []


# ------------------------------------------------------------------ analytics
def test_metrics_dwell_transit_revisit():
    fl = [{"flower_id": "a", "box": [90, 90, 110, 110]}, {"flower_id": "b", "box": [490, 90, 510, 110]}]
    ev = [{"track_id": "honeybee_1", "flower_id": "a", "type": "visit", "t_start": 0.0, "t_end": 2.0},
          {"track_id": "honeybee_1", "flower_id": "b", "type": "visit", "t_start": 3.0, "t_end": 4.0},
          {"track_id": "honeybee_1", "flower_id": "a", "type": "visit", "t_start": 6.0, "t_end": 7.0}]
    m = clip_metrics(doc(pts([(100, 100)] * 80), fl, ev), duration_s=8.0)
    s = m["summary"]
    assert s["visits"] == 3 and s["flowers_visited"] == 2
    assert s["median_dwell_s"] == 1.0
    assert s["revisit_rate"] == round(1 / 3, 2)
    assert m["transitions"][0]["transit_s"] == 1.0


def test_isotropic_scales_y_only():
    d = isotropic(doc(pts([(100, 1000)])), 0.5)
    assert d["tracks"][0]["points"][0]["center"] == [100, 500]


# ------------------------------------------------------------------ search
def test_search_finds_landing():
    fl = [{"flower_id": "flower_1", "box": [90, 90, 110, 110]}]
    ev = [{"track_id": "honeybee_1", "flower_id": "flower_1", "type": "visit", "t_start": 2.0, "t_end": 5.0}]
    idx = derived_moments(doc(pts([(100, 100)] * 60), fl, ev), 6.0)
    top = query("when does the bee land on a flower?", idx)[0]
    assert "lands" in top["text"] and abs(top["t_start"] - 1.5) < 1e-6
    assert tokens("takes off") == ["takeoff"]


# ------------------------------------------------------------------ forecast features
def test_path_features_shape_and_candidates():
    hist = np.cumsum(np.ones((PAST + 1, 2)), axis=0)
    fl = np.array([[0, 0, 10], [50, 50, 10], [500, 500, 10]], float)
    x = path_features(hist, fl, False)
    assert x.shape == (2 * PAST + 2 + 3 + 12,)
    rows, idx = candidate_features(np.array([0.0, 0.0]), np.array([1.0, 1.0]), fl, 0, {0: 1})
    assert idx == [0, 1, 2] and rows[0, 5] == 1.0 and rows[1, 1] == 1     # current flag, distance rank
    assert FUT == 20
