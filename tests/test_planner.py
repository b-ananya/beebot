"""Planner behaviour on small synthetic fields."""
from sim.planner import plan_route


def field(flowers):
    return {"field_id": "t", "flowers": [{"flower_id": f"f{i}", "center": c, "radius": 10, "state": s,
                                          "visits": 0, "deficit": d} for i, (c, d, s) in enumerate(flowers)]}


def test_skips_nearby_zero_deficit_for_far_deficit():
    # many covered flowers near the start, the deficit is far away: nearest-first wastes the budget
    near = [([50 * i, 950], 0, "open") for i in range(1, 8)]
    f = field(near + [([900, 100], 4, "open")])
    p = plan_route(f, start=(0, 1000), speed=200, dwell_s=1, budget_s=8, aspect=1.0)
    assert p["route"] == ["f7"] and p["deficit_covered"] == 1.0
    assert p["baselines"]["nearest_all"] == 0.0


def test_respects_budget_and_closed_flowers():
    f = field([([100, 900], 4, "bud"), ([200, 900], 4, "open"), ([900, 100], 4, "open")])
    p = plan_route(f, start=(0, 1000), speed=100, dwell_s=1, budget_s=4, aspect=1.0)
    assert p["route"] == ["f1"]  # bud ignored, far flower out of budget
    assert p["total_deficit"] == 8


def test_no_deficit_no_route():
    p = plan_route(field([([500, 500], 0, "open")]), budget_s=30)
    assert p["route"] == [] and p["deficit_covered"] == 0
