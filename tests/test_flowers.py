"""Offline tests for flower boxes (P2). No network: ``_request`` is replaced."""
import json
import shutil

import cv2
import numpy as np
import pytest

from perception import flowers, merge, track
from perception.client import ROOT, PerceptionClient

MARKUP = """<point_box mention="flower"> (560,300) (700,440) </point_box>
<point_box mention="flower"> (100,100) (160,160) </point_box>
<point_box mention="flower"> (562,302) (702,442) </point_box>
<point_box mention="flower"> (300,100) (240,160) </point_box>
<point_box mention="flower"> (900,50) (960,110) </point_box>
<point_box mention="flower"> (10,10) (10,40) </point_box>"""


class FakeClient(PerceptionClient):
    """Serves canned flower markup; counts network calls."""

    def __init__(self, tmp, **kw):
        super().__init__("p2", user="test", cache_dir=tmp / "raw", costs_dir=tmp / "costs", api_key="k", **kw)
        self.requests = 0

    def _media_parts(self, video_path, image_paths):
        return ["<image>"]

    def _request(self, parts, prompt_text, **kw):
        self.requests += 1
        return {"text": MARKUP, "finish_reason": "stop", "model": "perceptron-mk1.5", "request_id": "req-1",
                "usage": {"prompt_tokens": 1500, "completion_tokens": 120}, "raw": None}


@pytest.fixture
def repo(tmp_path):
    """A tiny repo: one clip (a real 20-frame video), the real validator and prompt."""
    (tmp_path / "clips").mkdir()
    vw = cv2.VideoWriter(str(tmp_path / "clips" / "sm1_01_000000.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 36))
    for i in range(20):
        vw.write(np.full((36, 64, 3), i * 10, np.uint8))
    vw.release()
    (tmp_path / "clips" / "manifest.json").write_text(json.dumps([
        {"clip_id": "sm1_01_000000", "path": "clips/sm1_01_000000.avi", "fps": 10, "width": 64, "height": 36,
         "duration_s": 2.0}]))
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "validate.py", tmp_path / "scripts" / "validate.py")
    return tmp_path


def test_reference_frame_is_middle_and_reused(repo):
    png = flowers.reference_frame("sm1_01_000000", root=repo)
    assert png == repo / "tracks" / "raw" / "frames" / "sm1_01_000000_ref.png"
    img = cv2.imread(str(png))
    assert img.shape == (36, 64, 3) and abs(int(img.mean()) - 100) <= 12   # frame 10 of 20 (MJPG is lossy)
    mtime = png.stat().st_mtime_ns
    assert flowers.reference_frame("sm1_01_000000", root=repo).stat().st_mtime_ns == mtime


def test_parse_flowers_is_contract_clean():
    fl = flowers.parse_flowers(MARKUP)
    # duplicate (IoU > 0.7), unordered and zero-width boxes dropped; sorted top-to-bottom, left-to-right
    assert [f["box"] for f in fl] == [[900, 50, 960, 110], [100, 100, 160, 160], [560, 300, 700, 440]]
    assert [f["flower_id"] for f in fl] == ["flower_1", "flower_2", "flower_3"]
    assert flowers.parse_flowers("no markup") == []


def test_flowers_clip_caches_and_merges_with_track(repo):
    c = FakeClient(repo)
    parts = repo / "tracks" / "parts"
    part = flowers.flowers_clip("sm1_01_000000", method="middle", client=c, root=repo, parts_dir=parts)
    assert len(part["flowers"]) == 3 and not part["cache_hit"] and part["complete"]
    assert part["cache_file"].startswith("p2/sm1_01_000000__flowers-v1__")
    assert flowers.flowers_clip("sm1_01_000000", method="middle", client=c, root=repo, parts_dir=parts)["cache_hit"]
    assert c.requests == 1

    # P1's track part and a P2 events part referencing a flower id from this module merge into a valid doc
    (parts / "sm1_01_000000.track.json").write_text(json.dumps(
        {"clip_id": "sm1_01_000000", "model": "perceptron-mk1.5", "prompt_version": "v1", "tracks": [
            track.parse_track('<track mention="honeybee"><point_box t="0.5"> (570,310) (590,330) </point_box></track>',
                              2.0, "honeybee")]}))
    (parts / "sm1_01_000000.events.json").write_text(json.dumps({"clip_id": "sm1_01_000000", "events": [
        {"track_id": "honeybee_1", "flower_id": "flower_3", "type": "visit", "t_start": 0.5, "t_end": 1.0}]}))
    doc = json.loads(merge.merge_clip("sm1_01_000000", root=repo, parts_dir=parts, out_dir=repo / "tracks").read_text())
    assert [f["flower_id"] for f in doc["flowers"]] == ["flower_1", "flower_2", "flower_3"]
    assert len(doc["events"]) == 1 and merge._validator(repo)(doc, 2.0) == []


def test_view_ids_are_stable_across_clips():
    assert flowers.view_id("sm3_12_124130") == "sm3_12" and flowers.view_id("hb1_w00") == "hb1"
    a = {"clip_id": "sm1_01_000000", "flowers": [{"flower_id": "flower_1", "box": [500, 100, 540, 140]},
                                                 {"flower_id": "flower_2", "box": [100, 300, 140, 340]}]}
    # same two flowers detected slightly differently, listed in another order, plus one flower only this clip saw
    b = {"clip_id": "sm1_01_000010", "flowers": [{"flower_id": "flower_1", "box": [50, 50, 90, 90]},
                                                 {"flower_id": "flower_2", "box": [102, 302, 142, 342]},
                                                 {"flower_id": "flower_3", "box": [498, 98, 538, 138]}]}
    out = flowers.assign_view_ids([b, a])
    ids = {c: {f["flower_id"]: f["box"] for f in p["flowers"]} for c, p in out.items()}
    assert ids["sm1_01_000000"] == {"flower_2": [500, 100, 540, 140], "flower_3": [100, 300, 140, 340]}
    assert ids["sm1_01_000010"] == {"flower_1": [50, 50, 90, 90], "flower_2": [498, 98, 538, 138],
                                    "flower_3": [102, 302, 142, 342]}
    assert out["sm1_01_000000"]["view_id"] == "sm1_01"
    assert flowers.assign_view_ids(list(out.values())) == out     # relabeling twice changes nothing


def test_match_scores():
    pred = [{"flower_id": "flower_1", "box": [0, 0, 100, 100]}, {"flower_id": "flower_2", "box": [500, 500, 600, 600]}]
    truth = [{"flower_id": "flower_0", "box": [10, 10, 100, 100]}, {"flower_id": "flower_1", "box": [800, 800, 900, 900]}]
    s = flowers.match(pred, truth)
    assert (s["tp"], s["precision"], s["recall"], s["mean_iou"]) == (1, 0.5, 0.5, 0.81)


def test_real_prompt_loads():
    text = flowers.load_prompt("v1")
    assert "point_box" in text and "{" not in text


class DenseClient(FakeClient):
    """First tile answer is cut off at the token limit, like the honeybee ground cover."""

    def _request(self, parts, prompt_text, **kw):
        self.requests += 1
        return {"text": MARKUP, "finish_reason": "length", "model": "perceptron-mk1.5", "request_id": "req-2",
                "usage": {"prompt_tokens": 1500, "completion_tokens": 6000}, "raw": None}


def test_tiled_is_default_merges_tiles_and_caches(repo):
    c = FakeClient(repo)
    parts = repo / "tracks" / "parts"
    part = flowers.flowers_clip("sm1_01_000000", client=c, root=repo, parts_dir=parts)
    # 2x2 tiles; the test video is flat gray, so identical tiles are answered from the cache after the first
    assert part["method"] == "tiled" and not part["dense"] and c.api_calls + c.cache_hits == 4
    assert (repo / "tracks" / "raw" / "frames" / "reference" / "sm1_01_000000.png").exists()   # median frame
    # every tile returned the same markup: boxes land in different places of the full frame, none outside 0-1000
    assert part["flowers"] and all(0 <= v <= 1000 for f in part["flowers"] for v in f["box"])
    again = flowers.flowers_clip("sm1_01_000000", client=c, root=repo, parts_dir=parts)
    assert again["cache_hit"] and c.requests == part["api_calls"] and again["flowers"] == part["flowers"]


def test_tiled_skips_dense_ground_cover(repo):
    c = DenseClient(repo)
    res = flowers.detect_tiled("sm1_01_000000", client=c, root=repo)
    assert res["dense"] and res["flowers"] == [] and c.requests == 1                  # stops after the first tile
