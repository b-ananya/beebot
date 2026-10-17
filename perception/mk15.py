"""Shared Mk1.5 client: every call is cached, costed, and capped.

    from perception.mk15 import Mk15
    mk = Mk15()                                   # reads PERCEPTRON_API_KEY from .env
    res = mk.ask(["clips/hb1_w00.mp4"], "Track the bee.", expects="box", tag="hb1_w00")
    res["tracks"], res["usage"], res["cost_usd"], res["cached"]

Cache: tracks/raw/<tag>/<hash>.json, keyed by file contents + prompt + settings, so a rerun with
the same inputs costs nothing and the demo works offline. Commit these files.

Spend: each person's calls are logged to tracks/raw/spend_<git user>.jsonl. The client refuses
new calls once that person's logged total reaches BEEBOT_BUDGET_USD (default 40, of the $50).
Prices are estimates until A1 measures them; override with env vars, see PRICES.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

RAW = Path("tracks/raw")
MODEL = "perceptron-mk1.5"

# USD per million tokens. Third-party listings, not confirmed by Perceptron: input $0.15,
# output $1.50, video possibly $1.00. We charge all input at the video rate to stay conservative.
PRICES = {
    "input": float(os.environ.get("BEEBOT_PRICE_INPUT", "1.00")),
    "output": float(os.environ.get("BEEBOT_PRICE_OUTPUT", "1.50")),
}
BUDGET_USD = float(os.environ.get("BEEBOT_BUDGET_USD", "40"))


class BudgetExceeded(RuntimeError):
    pass


def _user():
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
    except OSError:
        name = ""
    return "".join(c for c in (name or os.environ.get("USERNAME") or os.environ.get("USER") or "unknown")
                   if c.isalnum() or c in "-_") or "unknown"


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cost_of(usage):
    usage = usage or {}
    return (usage.get("prompt_tokens") or 0) / 1e6 * PRICES["input"] + \
           (usage.get("completion_tokens") or 0) / 1e6 * PRICES["output"]


def _parse(content):
    """Markup -> plain dicts (tracks, boxes, points, clips). Same code path for live and cached."""
    from perceptron import collect_annotations

    found = collect_annotations(content or "")

    def shape(p):
        if hasattr(p, "top_left"):
            return {"t": p.t, "box": [p.top_left.x, p.top_left.y, p.bottom_right.x, p.bottom_right.y],
                    "mention": p.mention}
        if hasattr(p, "x"):
            return {"t": p.t, "point": [p.x, p.y], "mention": p.mention}
        return {"t": getattr(p, "t", None), "kind": type(p).__name__, "mention": getattr(p, "mention", None)}

    return {
        "tracks": [{"mention": t.mention, "complete": t.complete, "points": [shape(p) for p in t.points]}
                   for t in found.tracks],
        "boxes": [shape(b) for b in found.boxes],
        "points": [shape(p) for p in found.points],
        "clips": [{"at": c.timestamp.at, "until": c.timestamp.until, "mention": c.mention} for c in found.clips],
        "parse_errors": [str(e) for e in found.errors],
    }


class Mk15:
    def __init__(self, model=MODEL, budget_usd=BUDGET_USD):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.model = model
        self.budget_usd = budget_usd
        self.user = _user()
        self.ledger = RAW / f"spend_{self.user}.jsonl"

    # ---- spend

    def spent(self):
        if not self.ledger.exists():
            return 0.0
        return sum(json.loads(line)["cost_usd"] for line in self.ledger.read_text().splitlines() if line.strip())

    def _log(self, entry):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ---- calls

    def ask(self, media, prompt, *, expects=None, tag="misc", system=None, refresh=False, **gen):
        """media: list of paths (.mp4/.mov -> video, else image). Returns a dict with text, tracks,
        boxes, points, clips, usage, cost_usd, finish_reason, cached, cache_path."""
        from perceptron import image, perceive, system as system_node, text, video

        media = [Path(m) for m in media]
        key_src = json.dumps({"model": self.model, "media": [_file_hash(m) for m in media], "prompt": prompt,
                              "system": system, "expects": expects, "gen": gen}, sort_keys=True)
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        cache = RAW / tag / f"{key}.json"

        if cache.exists() and not refresh:
            rec = json.loads(cache.read_text())
            return {**rec, **_parse(rec["content"]), "cached": True, "cache_path": str(cache)}

        if self.spent() >= self.budget_usd:
            raise BudgetExceeded(f"{self.user} has spent ${self.spent():.2f} of the ${self.budget_usd:.0f} cap "
                                 f"({self.ledger}). Raise BEEBOT_BUDGET_USD only on purpose.")

        nodes = ([system_node(system)] if system else []) + \
                [video(str(m)) if m.suffix.lower() in (".mp4", ".mov", ".avi", ".webm") else image(str(m))
                 for m in media] + [text(prompt)]
        t0 = time.time()
        res = perceive(nodes, expects=expects, model=self.model, **gen)
        latency = time.time() - t0

        content = res.text
        raw = res.raw if isinstance(res.raw, dict) else None
        try:  # prefer the untouched message content, so tags survive for re-parsing
            content = raw["choices"][0]["message"]["content"] or content
        except (TypeError, KeyError, IndexError):
            pass
        usage = res.usage if isinstance(res.usage, dict) else (res.usage.to_dict() if res.usage else {})
        rec = {"model": self.model, "media": [m.as_posix() for m in media], "prompt": prompt, "system": system,
               "expects": expects, "gen": gen, "content": content, "text": res.text,
               "finish_reason": res.finish_reason, "usage": usage, "cost_usd": round(cost_of(usage), 6),
               "latency_s": round(latency, 2), "request_id": res.request_id, "user": self.user,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec, indent=1))
        self._log({"time": rec["created"], "tag": tag, "key": key, "usage": usage,
                   "cost_usd": rec["cost_usd"], "latency_s": rec["latency_s"]})
        if res.finish_reason == "length":
            print(f"warning: {tag} hit the output limit; shorten the window instead of patching the markup")
        return {**rec, **_parse(content), "cached": False, "cache_path": str(cache)}
