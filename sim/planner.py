"""Robot route planning over a pollination map (orienteering: max deficit cleared within a time budget).

v1 is greedy insertion + 2-opt, pure Python. C3 adds an OR-Tools solver behind the same
plan_route() signature, and keeps this as the fallback and as a baseline to beat.

Coordinates are aspect-corrected before measuring distance (a norm1000 frame is wider than tall),
so "units" below are y-norm units: 1000 = frame height.
"""
import math
import random


def _xy(f, aspect):
    return (f["center"][0] * aspect, f["center"][1])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_time(start, stops, speed, dwell_s):
    """Seconds to fly start -> stops in order, dwelling at each. stops: list of (x, y)."""
    t, here = 0.0, start
    for p in stops:
        t += _dist(here, p) / speed + dwell_s
        here = p
    return t


def simulate(start, order, pts, prize, speed, dwell_s, budget_s):
    """Walk a route; returns (stops reached in time, deficit cleared, path length)."""
    t, here, cleared, length, done = 0.0, start, 0.0, 0.0, []
    for i in order:
        leg = _dist(here, pts[i])
        if t + leg / speed + dwell_s > budget_s:
            break
        t += leg / speed + dwell_s
        length += leg
        cleared += prize[i]
        here = pts[i]
        done.append(i)
    return done, cleared, length


def two_opt(start, order, pts):
    """Shorten an open route (fixed start, free end) by reversing segments."""
    best = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_time(start, [pts[k] for k in cand], 1, 0) < route_time(start, [pts[k] for k in best], 1, 0) - 1e-9:
                    best, improved = cand, True
    return best


def greedy_2opt(start, pts, prize, speed, dwell_s, budget_s):
    """Insert the flower with the best deficit per added second, re-optimize order, repeat."""
    order, left = [], {i for i, p in enumerate(prize) if p > 0}
    while left:
        best = None
        for i in left:
            for pos in range(len(order) + 1):
                cand = order[:pos] + [i] + order[pos:]
                t = route_time(start, [pts[k] for k in cand], speed, dwell_s)
                if t <= budget_s:
                    added = t - route_time(start, [pts[k] for k in order], speed, dwell_s)
                    score = prize[i] / max(added, 1e-6)
                    if best is None or score > best[0]:
                        best = (score, cand, i)
        if best is None:
            break
        order = two_opt(start, best[1], pts)
        left.discard(best[2])
    return order


def nearest_all(start, pts, prize, *_):
    """Baseline: visit every flower nearest-first, ignoring deficits."""
    order, here, left = [], start, set(range(len(pts)))
    while left:
        i = min(left, key=lambda k: _dist(here, pts[k]))
        order.append(i)
        here = pts[i]
        left.discard(i)
    return order


def random_order(start, pts, prize, *_, seed=0):
    order = list(range(len(pts)))
    random.Random(seed).shuffle(order)
    return order


def plan_route(field, *, start=(0, 1000), speed=100.0, dwell_s=3.0, budget_s=30.0, aspect=16 / 9,
               bee_order=None):
    """field: contract field doc. Returns a plan dict (contract plan/ shape, minus video)."""
    flowers = field["flowers"]
    pts = [_xy(f, aspect) for f in flowers]
    prize = [float(f["deficit"]) if f.get("state", "open") == "open" else 0.0 for f in flowers]
    total = sum(prize) or 1.0
    s = (start[0] * aspect, start[1])

    def score(order):
        done, cleared, length = simulate(s, order, pts, prize, speed, dwell_s, budget_s)
        return done, cleared / total, length

    ours, covered, length = score(greedy_2opt(s, pts, prize, speed, dwell_s, budget_s))
    baselines = {"nearest_all": round(score(nearest_all(s, pts, prize))[1], 3),
                 "random": round(sum(score(random_order(s, pts, prize, seed=k))[1] for k in range(20)) / 20, 3)}
    if bee_order:
        ids = {f["flower_id"]: i for i, f in enumerate(flowers)}
        baselines["follow_bee"] = round(score([ids[b] for b in bee_order if b in ids])[1], 3)
    return {"field_id": field["field_id"], "planner": "greedy_2opt", "start": list(start), "budget_s": budget_s,
            "route": [flowers[i]["flower_id"] for i in ours], "path_length": round(length, 1),
            "deficit_covered": round(covered, 3), "total_deficit": sum(prize),
            "robot": {"speed": speed, "dwell_s": dwell_s, "units": "y-norm1000 per s"}, "baselines": baselines}
