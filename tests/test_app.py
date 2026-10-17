"""App logic tests (no browser). Full UI smoke test: python -c "from streamlit.testing.v1 import AppTest; ..."."""
from app import analytics, search

DOC = {"clip_id": "c1", "model": "mock",
       "tracks": [{"track_id": "bee_1", "species": "honeybee",
                   "points": [{"t": 0.0, "box": [0, 0, 10, 10]}, {"t": 0.4, "box": [0, 0, 10, 10]},
                              {"t": 3.0, "box": [0, 0, 10, 10]}, {"t": 3.2, "box": [0, 0, 10, 10]}]}],
       "flowers": [{"flower_id": "flower_1", "box": [100, 100, 200, 200]}],
       "events": [{"track_id": "bee_1", "flower_id": "flower_1", "type": "visit", "t_start": 3.0, "t_end": 5.0}]}


def test_gaps_split_visible_segments():
    assert analytics.segments(DOC["tracks"][0]) == [[0.0, 0.4], [3.0, 3.2]]
    assert abs(analytics.visible_seconds(DOC) - 0.6) < 1e-9


def test_summary_counts_visits_and_dwell():
    s = analytics.summary(DOC, 10.0)
    assert s["flower visits"] == 1 and s["dwell on flowers (s)"] == 2.0


def test_search_ranks_landings_first():
    hits = search.search(search.build_fallback([DOC]), "when does the bee land on a flower?")
    assert hits[0]["kind"] == "event" and hits[0]["t_start"] == 3.0
