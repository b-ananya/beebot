"""Tests for the RL field env and the rollout writer. No training, no video: fast and offline."""
import json

import numpy as np
import pytest

from sim import rollout as ro
from sim.field_env import FieldEnv, flowers_from_truth, nearest_unvisited_action, random_layout

FLOWERS = [{"flower_id": "flower_0", "box": [100, 100, 160, 160]},
           {"flower_id": "flower_1", "box": [800, 800, 860, 860], "state": "open"},
           {"flower_id": "flower_2", "box": [500, 500, 560, 560], "state": "bud"}]


def test_reset_and_step_shapes():
    env = FieldEnv(seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(r, float) and not term and not trunc
    assert 4 <= len(env.flowers) <= 10


def test_speed_cap_from_bee_stats():
    env = FieldEnv(flowers=FLOWERS, start=[500, 100], seed=0)
    env.reset()
    env.step(np.array([10.0, 0.0]))  # over-long action is normalised to the cap
    assert env.pos[0] == pytest.approx(500 + env.speed * env.dt)
    assert env.speed == env.stats["median_flight_speed"] > 0
    assert env.dwell_s == env.stats["median_dwell_s"] > 0


def test_visit_reward_fires_after_dwell():
    env = FieldEnv(flowers=FLOWERS, start=[130, 130], budget_s=20, shaping=False)
    env.reset()
    rewards, fired = [], None
    for i in range(env.dwell_steps + 2):
        _, r, term, trunc, info = env.step(np.zeros(2))
        rewards.append(r)
        if info["visited_now"]:
            fired = (i + 1, info["visited_now"])
            break
    assert fired == (env.dwell_steps, "flower_0")
    assert rewards[-1] == pytest.approx(0.99)  # +1 visit, -0.01 step
    assert all(r == pytest.approx(-0.01) for r in rewards[:-1])
    assert env.visited[0] and not env.visited[1]
    # staying put earns nothing more: the flower is spent
    _, r, *_ = env.step(np.zeros(2))
    assert r == pytest.approx(-0.01)


def test_bud_flower_never_rewards_and_episode_ends_on_budget():
    env = FieldEnv(flowers=FLOWERS, start=[530, 530], budget_s=5, dt=0.1, shaping=False)
    env.reset()
    for _ in range(49):
        _, r, term, trunc, _ = env.step(np.zeros(2))
        assert r == pytest.approx(-0.01) and not term and not trunc
    _, r, term, trunc, _ = env.step(np.zeros(2))
    assert trunc and not term and env.visits == []


def test_baseline_clears_real_layout():
    flowers = flowers_from_truth("sm3_12_123903")
    env = FieldEnv(flowers=flowers, start=[986, 738], budget_s=60, shaping=False)
    way, visits = ro.run_episode(env, lambda e, o: nearest_unvisited_action(e))
    assert len(visits) == len(flowers)
    assert [v["flower_id"] for v in visits] == [fid for _, fid in env.visits]
    assert way[0]["t"] == 0.0 and way[-1]["t"] <= 60.0


def test_random_layout_matches_contract():
    rng = np.random.default_rng(3)
    for f in random_layout(rng):
        x1, y1, x2, y2 = f["box"]
        assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
        assert f["flower_id"].startswith("flower_")


def test_rollout_json_is_valid(tmp_path):
    r = ro.rollout("sm3_12_123903", policy="baseline", write_video=False, out_dir=tmp_path, budget_s=15)
    path = tmp_path / "sm3_12_123903_rl.json"
    d = json.loads(path.read_text())
    assert d == r
    assert d["clip_id"] == "sm3_12_123903" and d["units"] == "norm1000" and d["video"] is None
    way = d["waypoints"]
    assert way and set(way[0]) == {"t", "x", "y", "flower_id"}
    ts = [w["t"] for w in way]
    assert ts == sorted(ts) and ts[0] == 0.0 and ts[-1] == pytest.approx(15.0)
    assert all(0 <= w["x"] <= 1000 and 0 <= w["y"] <= 1000 for w in way)
    flowers, source = ro.load_flowers("sm3_12_123903")
    assert d["flowers_from"] == source in ("truth", "mk1.5")
    ids = {f["flower_id"] for f in flowers}
    assert all(w["flower_id"] is None or w["flower_id"] in ids for w in way)
    assert d["n_visits"] == len(d["visits"]) >= 1
    assert all(v["flower_id"] in ids and 0 < v["t"] <= 15 for v in d["visits"])
    # every visit is marked on a waypoint at that time
    marked = {w["flower_id"] for w in way if w["flower_id"]}
    assert {v["flower_id"] for v in d["visits"]} <= marked


# ---------------------------------------------------------------- sim.predict (image -> path), no API, no video
def _fake_detector(flowers):
    return lambda path: (flowers, {"source": "fake", "cost_usd": 0.0, "cache_hit": True})


def test_predict_writes_json_and_png(tmp_path):
    import cv2
    from sim import predict as pr

    img = tmp_path / "my photo.jpg"
    cv2.imwrite(str(img), np.full((300, 400, 3), 90, dtype=np.uint8))
    r = pr.predict(img, policy="baseline", budget_s=20, write_video_file=False, out_dir=tmp_path,
                   detector=_fake_detector(FLOWERS))
    d = json.loads((tmp_path / "my_photo_rl.json").read_text())
    assert d == r and d["width"] == 400 and d["height"] == 300 and d["video"] is None
    assert (tmp_path / "my_photo_rl.png").exists()
    assert d["n_open"] == 2 and d["n_visits"] == 2  # the bud is drawn but never visited
    for v in d["visits"]:
        assert v["dwell_s"] == pytest.approx(d["dwell_s"], abs=0.05)
        assert v["t_arrive"] < v["t_leave"] <= 20
    assert [v["order"] for v in d["visits"]] == [1, 2]
    ids = {f["flower_id"] for f in FLOWERS}
    assert all(w["flower_id"] is None or w["flower_id"] in ids for w in d["waypoints"])


def test_visit_table_uses_last_arrival():
    from sim.predict import visit_table

    way = [{"t": 0.0, "flower_id": None}, {"t": 0.1, "flower_id": "flower_0"}, {"t": 0.2, "flower_id": None},
           {"t": 0.3, "flower_id": "flower_0"}, {"t": 0.4, "flower_id": "flower_0"}]
    rows = visit_table(way, [{"flower_id": "flower_0", "t": 0.4}], dt=0.1)
    assert rows == [{"order": 1, "flower_id": "flower_0", "t_arrive": 0.3, "t_leave": 0.4, "dwell_s": 0.2}]


def test_sane_boxes_drops_slivers():
    from sim.predict import sane_boxes

    kept, dropped = sane_boxes([{"flower_id": "a", "box": [0, 0, 9, 1]}, {"flower_id": "b", "box": [10, 10, 40, 40]}])
    assert [f["flower_id"] for f in kept] == ["b"] and dropped == 1
