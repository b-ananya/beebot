"""Offline tests for the Mk1.5 wrapper: no API key or network needed."""
import json

from perception import mk15
from perception.track_clip import to_contract
from scripts.validate import check_tracks_doc

CLIP = {"clip_id": "hb1_w00", "path": "clips/hb1_w00.mp4", "duration_s": 10.0}

MARKUP = ('<track mention="honeybee">'
          '<point_box t="0.5 seconds"> (410,520) (450,570) </point_box>'
          '<point_box t="1.0 seconds"> (420,525) (460,575) </point_box>'
          '<point_box t="12.0 seconds"> (430,530) (470,580) </point_box>'  # past clip end: dropped
          '</track>')


def test_parse_track_markup():
    parsed = mk15._parse(MARKUP)
    assert len(parsed["tracks"]) == 1
    track = parsed["tracks"][0]
    assert track["mention"] == "honeybee"  # mention lives on the track, not each waypoint
    assert track["points"][0]["t"] == 0.5 and track["points"][0]["box"] == [410, 520, 450, 570]


def test_to_contract_is_valid():
    res = {**mk15._parse(MARKUP), "finish_reason": "stop", "cost_usd": 0.01, "cache_path": "x"}
    flowers = mk15._parse('<point_box> (560,300) (700,440) </point_box>')
    doc = to_contract(CLIP, res, "v1", flowers)
    assert [p["t"] for p in doc["tracks"][0]["points"]] == [0.5, 1.0]
    assert doc["flowers"][0]["box"] == [560, 300, 700, 440]
    assert check_tracks_doc(doc, CLIP["duration_s"]) == []


def test_cost_and_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert abs(mk15.cost_of({"prompt_tokens": 1_000_000, "completion_tokens": 0}) - mk15.PRICES["input"]) < 1e-9
    mk = mk15.Mk15(budget_usd=0.05)
    mk.ledger.parent.mkdir(parents=True)
    mk.ledger.write_text(json.dumps({"cost_usd": 0.06}) + "\n")
    try:
        mk.ask([__file__], "hi")
        raise AssertionError("expected BudgetExceeded")
    except mk15.BudgetExceeded:
        pass
