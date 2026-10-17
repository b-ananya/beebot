"""Offline tests for the shared client, the tracking parser and the merge. No network: ``_request`` is replaced."""
import csv
import json
import shutil
from pathlib import Path

import pytest

from perception import merge, track
from perception.client import ROOT, BudgetExceeded, CacheMiss, PerceptionClient

MARKUP = """<track mention="honeybee" asset_idx="0">
<point_box t="0.5 seconds"> (410,520) (450,570) </point_box>
<point_box t="1.0 seconds"> (470,300) (510,350) </point_box>
<point_box t="1.0 seconds"> (471,301) (511,351) </point_box>
<point_box t="1.5 seconds"> (620,230) (580,180) </point_box>
<point_box t="12.0 seconds"> (10,10) (20,20) </point_box>
<point_box t="2.0 seconds"> (600,200) (640,240) </point_box>
</track>"""


class FakeClient(PerceptionClient):
    """Serves canned markup; counts network calls."""

    def __init__(self, tmp, **kw):
        super().__init__("p1", user="test", cache_dir=tmp / "raw", costs_dir=tmp / "costs", api_key="k", **kw)
        self.requests = 0
        self.finish = "stop"

    def _media_parts(self, video_path, image_paths):
        return ["<video>"]

    def _request(self, parts, prompt_text, **kw):
        self.requests += 1
        return {"text": MARKUP, "finish_reason": self.finish, "model": "perceptron-mk1.5", "request_id": "req-1",
                "usage": {"prompt_tokens": 20000, "completion_tokens": 400,
                          "prompt_tokens_details": {"cached_tokens": 10000}}, "raw": {"ok": True}}


@pytest.fixture
def repo(tmp_path):
    """A tiny repo: one clip in the manifest, a fake mp4, the real validator and prompt."""
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "hb1_w00.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\0" * 64)
    (tmp_path / "clips" / "manifest.json").write_text(json.dumps([
        {"clip_id": "hb1_w00", "path": "clips/hb1_w00.mp4", "fps": 30, "width": 1280, "height": 720,
         "duration_s": 10.0}]))
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "validate.py", tmp_path / "scripts" / "validate.py")
    return tmp_path


def call(client, clip="hb1_w00", prompt="track the bee", **kw):
    return client.call(clip_id=clip, prompt_name="track", prompt_version="v1", prompt_text=prompt,
                       video_path=ROOT / "clips" / "manifest.json", **kw)


def test_cache_and_cost_log(tmp_path):
    c = FakeClient(tmp_path)
    r1 = call(c)
    assert not r1.cache_hit and c.requests == 1
    # (20000-10000)*0.15 + 10000*0.0375 + 400*1.50 per million
    assert r1.cost_usd == pytest.approx((10000 * 0.15 + 10000 * 0.0375 + 400 * 1.5) / 1e6)
    assert r1.cache_path.exists() and r1.cache_path.parent.name == "p1"
    r2 = call(c)
    assert r2.cache_hit and r2.cost_usd == 0 and c.requests == 1 and r2.text == MARKUP
    rows = list(csv.DictReader(open(c.cost_log, newline="")))
    assert len(rows) == 1 and rows[0]["clip_id"] == "hb1_w00" and rows[0]["cached_tokens"] == "10000"
    assert c.spent_usd() == pytest.approx(r1.cost_usd)
    # a different prompt or setting is a different request
    call(c, prompt="track the wasp")
    call(c, max_tokens=2048)
    assert c.requests == 3


def test_budget_cap(tmp_path):
    c = FakeClient(tmp_path, cost_cap_usd=0.001)
    call(c)
    with pytest.raises(BudgetExceeded):
        call(c, prompt="another")
    assert call(c).cache_hit  # cached answers are still served past the cap


def test_offline_mode(tmp_path):
    c = FakeClient(tmp_path, offline=True)
    with pytest.raises(CacheMiss):
        call(c)
    assert c.requests == 0


def test_parse_track_is_contract_clean():
    tr = track.parse_track(MARKUP, 10.0, "honeybee")
    ts = [p["t"] for p in tr["points"]]
    assert ts == [0.5, 1.0, 2.0]          # sorted, duplicate t dropped, unordered box and t>duration dropped
    assert tr["track_id"] == "honeybee_1" and tr["complete"] is True
    assert tr["points"][0]["center"] == [430.0, 545.0] and tr["points"][0]["interpolated"] is False
    cut = track.parse_track(MARKUP.replace("</track>", ""), 10.0, "honeybee")
    assert cut["complete"] is False and len(cut["points"]) == 3
    assert track.parse_track("no markup here", 10.0, "vespidae") == {
        "track_id": "vespidae_1", "species": "vespidae", "complete": True, "points": []}


def test_interpolate():
    pts = [{"t": 0.0, "box": [0, 0, 10, 10], "center": [5, 5], "interpolated": False},
           {"t": 0.5, "box": [50, 0, 60, 10], "center": [55, 5], "interpolated": False},
           {"t": 3.0, "box": [90, 0, 100, 10], "center": [95, 5], "interpolated": False}]
    out = track.interpolate(pts, max_gap_s=1.0, step_s=0.1)
    filled = [p for p in out if p["interpolated"]]
    assert [p["t"] for p in filled] == [0.1, 0.2, 0.3, 0.4]
    assert filled[1]["box"][0] == pytest.approx(20.0)
    assert len(out) == 7  # the 2.5 s gap stays a gap


def test_track_clip_and_merge(repo):
    c = FakeClient(repo)
    parts = repo / "tracks" / "parts"
    part = track.track_clip("hb1_w00", client=c, root=repo, parts_dir=parts)
    assert part["tracks"][0]["track_id"] == "honeybee_1" and len(part["tracks"][0]["points"]) == 3
    assert part["model"] == "perceptron-mk1.5" and not part["cache_hit"]
    assert track.track_clip("hb1_w00", client=c, root=repo, parts_dir=parts)["cache_hit"]

    (parts / "hb1_w00.flowers.json").write_text(json.dumps(
        {"clip_id": "hb1_w00", "flowers": [{"flower_id": "flower_1", "box": [400, 500, 460, 580]}]}))
    (parts / "hb1_w00.events.json").write_text(json.dumps({"clip_id": "hb1_w00", "events": [
        {"track_id": "honeybee_1", "flower_id": "flower_1", "type": "visit", "t_start": 0.5, "t_end": 1.0},
        {"track_id": "bee_9", "flower_id": "flower_1", "type": "visit", "t_start": 0.5, "t_end": 1.0}]}))
    out = merge.merge_clip("hb1_w00", interpolate_gap_s=1.0, root=repo, parts_dir=parts, out_dir=repo / "tracks")
    doc = json.loads(out.read_text())
    assert doc["model"] == "perceptron-mk1.5" and doc["prompt_version"] == "v1"
    assert len(doc["events"]) == 1 and len(doc["flowers"]) == 1
    pts = doc["tracks"][0]["points"]
    assert sum(p["interpolated"] for p in pts) == 4 + 9 and len(pts) == 3 + 13
    assert merge._validator(repo)(doc, 10.0) == []


def test_truncated_answer_marks_incomplete(repo):
    c = FakeClient(repo)
    c.finish = "length"
    part = track.track_clip("hb1_w00", client=c, root=repo, parts_dir=repo / "parts")
    assert part["tracks"][0]["complete"] is False and part["finish_reason"] == "length"


def test_real_prompt_renders():
    text = track.load_prompt("v1", species="honeybee", duration_s=10.0, step_s=0.5)
    assert "10.0 seconds" in text and "point_box" in text and "{" not in text
