"""Gymnasium env: a point agent visiting flowers on the 0-1000 field grid.

The agent is a bee-like point with a speed cap taken from real bee tracks (sim/bee_stats.json).
It earns +1 for reaching an unvisited open flower and staying there for the median bee dwell
time, pays -0.01 per step, and the episode ends when the time budget runs out.

Observation (float32, length 4 + 3 * K):
  [x, y] / 1000, time left fraction, dwell progress fraction, then for each of the K nearest
  unvisited open flowers: (dx, dy) / 1000 and a flag (1 = real flower, 0 = padding).
Action: Box(-1, 1, (2,)) velocity; |v| <= 1 maps to the speed cap.

Flowers use the CONTRACT.md format: {"flower_id", "box": [x1, y1, x2, y2], "state"?}.
Only "open" (or unstated) flowers count; "bud", "fruit" and "unknown" are drawn but never rewarded.
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.bee_stats import load as load_stats

ROOT = Path(__file__).resolve().parents[1]
GRID = 1000.0
K = 6


def random_layout(rng, n_min=4, n_max=10, radius=None):
    """A random field: n flower boxes with a real-looking radius, kept apart from each other."""
    radius = radius or load_stats()["median_flower_radius"]
    n = int(rng.integers(n_min, n_max + 1))
    centers = []
    while len(centers) < n:
        c = rng.uniform(60, GRID - 60, size=2)
        if all(np.linalg.norm(c - o) > 3 * radius for o in centers):
            centers.append(c)
    flowers = []
    for i, (cx, cy) in enumerate(centers):
        r = radius * rng.uniform(0.7, 1.3)
        flowers.append({"flower_id": f"flower_{i}", "box": [cx - r, cy - r, cx + r, cy + r], "state": "open"})
    return flowers


def flowers_from_truth(clip_id, truth_dir=ROOT / "truth"):
    return json.load(open(truth_dir / f"{clip_id}.json"))["flowers"]


class FieldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, flowers=None, budget_s=60.0, dt=0.1, start=None, k=K, stats=None,
                 shaping=True, min_radius=20.0, seed=None):
        super().__init__()
        self.stats = stats or load_stats()
        self.speed = float(self.stats["median_flight_speed"])  # grid units / s
        self.dwell_s = float(self.stats["median_dwell_s"])
        self.dt = float(dt)
        self.budget_s = float(budget_s)
        self.max_steps = int(round(self.budget_s / self.dt))
        self.dwell_steps = max(1, int(round(self.dwell_s / self.dt)))
        self.fixed_flowers = flowers
        self.fixed_start = None if start is None else np.asarray(start, dtype=np.float64)
        self.k = k
        self.shaping = shaping
        self.min_radius = min_radius
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4 + 3 * k,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.flowers = []

    # -- helpers ---------------------------------------------------------------------------
    def _set_flowers(self, flowers):
        self.flowers = [dict(f) for f in flowers]
        self.centers = np.array([[(f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2]
                                 for f in self.flowers], dtype=np.float64).reshape(-1, 2)
        self.radii = np.array([max(self.min_radius, max(f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) / 2)
                               for f in self.flowers], dtype=np.float64)
        self.open = np.array([f.get("state", "open") == "open" for f in self.flowers], dtype=bool)
        self.visited = np.zeros(len(self.flowers), dtype=bool)

    def _targets(self):
        return np.where(self.open & ~self.visited)[0]

    def _nearest_unvisited(self):
        idx = self._targets()
        if len(idx) == 0:
            return None
        d = np.linalg.norm(self.centers[idx] - self.pos, axis=1)
        return int(idx[np.argmin(d)])

    def _obs(self):
        obs = np.zeros(4 + 3 * self.k, dtype=np.float32)
        obs[0:2] = self.pos / GRID
        obs[2] = 1.0 - self.t / self.max_steps
        obs[3] = self.dwell_count / self.dwell_steps
        idx = self._targets()
        if len(idx):
            rel = self.centers[idx] - self.pos
            order = np.argsort(np.linalg.norm(rel, axis=1))[: self.k]
            for j, o in enumerate(order):
                obs[4 + 3 * j: 6 + 3 * j] = rel[o] / GRID
                obs[6 + 3 * j] = 1.0
        return np.clip(obs, -1.0, 1.0)

    def _potential(self):
        n = self._nearest_unvisited()
        if n is None:
            return 0.0
        return -float(np.linalg.norm(self.centers[n] - self.pos)) / GRID

    # -- gym api ---------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._set_flowers(self.fixed_flowers if self.fixed_flowers is not None else random_layout(self._rng))
        self.pos = (self.fixed_start.copy() if self.fixed_start is not None
                    else self._rng.uniform(0, GRID, size=2))
        self.t = 0
        self.dwell_count = 0
        self.dwell_on = None
        self.visits = []  # (step, flower_id)
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        norm = np.linalg.norm(a)
        if norm > 1.0:
            a = a / norm
        prev_phi = self._potential() if self.shaping else 0.0
        self.pos = np.clip(self.pos + a * self.speed * self.dt, 0.0, GRID)
        self.t += 1
        reward = -0.01
        on = None
        if len(self.flowers):
            dist = np.linalg.norm(self.centers - self.pos, axis=1)
            inside = np.where((dist <= self.radii) & self.open & ~self.visited)[0]
            if len(inside):
                on = int(inside[np.argmin(dist[inside])])
        if on is not None and on == self.dwell_on:
            self.dwell_count += 1
        else:
            self.dwell_on = on
            self.dwell_count = 1 if on is not None else 0
        visited_now = None
        if on is not None and self.dwell_count >= self.dwell_steps:
            self.visited[on] = True
            self.visits.append((self.t, self.flowers[on]["flower_id"]))
            visited_now = self.flowers[on]["flower_id"]
            reward += 1.0
            self.dwell_on, self.dwell_count = None, 0
        if self.shaping:
            reward += (self._potential() - prev_phi) * 2.0  # potential-based: optimal policy unchanged
        truncated = self.t >= self.max_steps
        terminated = bool(len(self._targets()) == 0)
        info = {"t": self.t * self.dt, "on_flower": None if on is None else self.flowers[on]["flower_id"],
                "visited_now": visited_now, "visits": len(self.visits)}
        return self._obs(), float(reward), terminated, truncated, info


def nearest_unvisited_action(env):
    """Baseline controller: fly at full speed to the nearest unvisited open flower, then hold still."""
    n = env._nearest_unvisited()
    if n is None:
        return np.zeros(2, dtype=np.float32)
    rel = env.centers[n] - env.pos
    d = np.linalg.norm(rel)
    if d <= env.radii[n] * 0.5:
        return np.zeros(2, dtype=np.float32)
    step = env.speed * env.dt
    return (rel / max(d, 1e-6) * min(1.0, d / step)).astype(np.float32)
