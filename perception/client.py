"""Shared Perceptron client: cache first, cost log, spending cap.

Every Perceptron call in this repo goes through ``PerceptionClient.call()``, so no clip and prompt is paid for twice:

- cache: ``tracks/raw/<piece>/<clip>__<prompt>-<version>__<hash>.json``, keyed by model, prompt text, media bytes and
  settings. It is committed to git, so teammates pull it instead of calling.
- cost log: ``costs/<user>.csv``, one row per real call (cache hits are never logged).
- cap: a real call is refused once the user's log passes ``BEEBOT_COST_CAP`` (default $40).
- ``BEEBOT_OFFLINE=1``: cache only; a miss raises ``CacheMiss`` (demo mode, A7).

    from perception.client import PerceptionClient
    client = PerceptionClient(piece="p1")
    r = client.call(clip_id="hb1_w00", prompt_name="track", prompt_version="v1",
                    prompt_text=prompt, video_path=Path("clips/hb1_w00.mp4"))
    r.annotations().tracks       # perceptron Track objects
    r.cost_usd, r.cache_hit, r.finish_reason

Model id comes from ``BEEBOT_MODEL`` (default ``perceptron-mk1.5``), so switching to Isaac is a one-line change.
Videos are uploaded once per API key to the Files API and reused by id (``tracks/raw/file_ids.json``).
"""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = os.getenv("BEEBOT_MODEL", "perceptron-mk1.5")
CACHE_DIR = ROOT / "tracks" / "raw"
COSTS_DIR = ROOT / "costs"
COST_CAP_USD = float(os.getenv("BEEBOT_COST_CAP", "40"))

# USD per million tokens from the Mk1.5 model card (docs.perceptron.inc). ``model_info()`` saves the API's own
# numbers to tracks/raw/model_info.json, which take precedence when present.
PRICES = {"perceptron-mk1.5": {"input": 0.15, "output": 1.50, "cached": 0.0375}}

COST_COLUMNS = ["time", "user", "piece", "clip_id", "prompt", "version", "model", "prompt_tokens", "cached_tokens",
                "completion_tokens", "cost_usd", "finish_reason", "request_id", "cache_file"]

RETRY_DELAYS_S = (2, 5, 15)


class BudgetExceeded(RuntimeError):
    """The user's cost log is at or past the cap; no real call was made."""


class CacheMiss(RuntimeError):
    """Offline mode and the response is not cached."""


@dataclass
class Response:
    text: str
    finish_reason: str | None
    usage: dict
    cost_usd: float
    model: str | None
    request_id: str | None
    cache_hit: bool
    cache_path: Path
    raw: dict | None = None

    @property
    def complete(self) -> bool:
        """True when the answer finished; ``length`` means the output budget ran out and the markup is cut."""
        return self.finish_reason == "stop"

    def annotations(self, *, strict: bool = False):
        """The answer's markup parsed by the SDK (``.tracks``, ``.boxes``, ``.clips``, ``.errors``)."""
        from perceptron import collect_annotations

        return collect_annotations(self.text or "", strict=strict)


_sha_cache: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def git_user() -> str:
    """First name from git config, lowercased, else the OS user: the cost log is costs/<user>.csv."""
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, check=True).stdout
        if name.strip():
            return name.strip().split()[0].lower()
    except (OSError, subprocess.CalledProcessError):
        pass
    return getpass.getuser().lower()


class PerceptionClient:
    def __init__(self, piece: str, *, user: str | None = None, model: str | None = None,
                 cache_dir: Path = CACHE_DIR, costs_dir: Path = COSTS_DIR, cost_cap_usd: float = COST_CAP_USD,
                 offline: bool | None = None, api_key: str | None = None):
        self.piece = piece
        self.user = user or os.getenv("BEEBOT_USER") or git_user()
        self.model = model or DEFAULT_MODEL
        self.cache_dir = Path(cache_dir)
        self.costs_dir = Path(costs_dir)
        self.cost_cap_usd = cost_cap_usd
        self.offline = bool(int(os.getenv("BEEBOT_OFFLINE", "0"))) if offline is None else offline
        self.api_key = api_key or os.getenv("PERCEPTRON_API_KEY")
        self._sdk = None
        self._lock = threading.Lock()  # frames mode calls from several threads; the logs and counters are shared
        self.api_calls = 0
        self.cache_hits = 0

    # ----------------------------------------------------------------- spend
    @property
    def cost_log(self) -> Path:
        return self.costs_dir / f"{self.user}.csv"

    def spent_usd(self) -> float:
        if not self.cost_log.exists():
            return 0.0
        with open(self.cost_log, newline="") as f:
            return round(sum(float(row["cost_usd"] or 0) for row in csv.DictReader(f)), 6)

    def _log(self, row: dict) -> None:
        with self._lock:
            self.costs_dir.mkdir(parents=True, exist_ok=True)
            new = not self.cost_log.exists()
            with open(self.cost_log, "a", newline="\n") as f:
                w = csv.DictWriter(f, fieldnames=COST_COLUMNS)
                if new:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in COST_COLUMNS})

    def prices(self, model: str | None = None) -> dict:
        model = model or self.model
        info_path = self.cache_dir / "model_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            if info.get("id") == model and info.get("credits_per_million_input_tokens") is not None:
                return {"input": info["credits_per_million_input_tokens"] / 1e6,
                        "output": (info.get("credits_per_million_output_tokens") or 0) / 1e6,
                        "cached": (info.get("credits_per_million_cache_read_tokens") or 0) / 1e6}
        if model not in PRICES:
            print(f"warning: no price table for {model}; costing it like perceptron-mk1.5")
        return PRICES.get(model, PRICES["perceptron-mk1.5"])

    def cost_of(self, usage: dict, model: str | None = None) -> float:
        p = self.prices(model)
        prompt = usage.get("prompt_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        return round(((prompt - cached) * p["input"] + cached * p["cached"] + completion * p["output"]) / 1e6, 6)

    def model_info(self) -> dict:
        """Fetch the model's limits and pricing once (GET /models is rate limited) into tracks/raw/model_info.json."""
        info = self._sdk_client().models.retrieve(self.model, extended=True).to_dict()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "model_info.json").write_text(json.dumps(info, indent=1), newline="\n")
        return info

    # ----------------------------------------------------------------- calls
    def call(self, *, clip_id: str, prompt_name: str, prompt_version: str, prompt_text: str,
             video_path: Path | str | None = None, image_paths=(), annotation_format: str = "box",
             max_tokens: int = 4096, reasoning_effort: str | None = None,
             temperature: float | None = None) -> Response:
        """One Perceptron request, served from the cache when the same request was made before.

        ``annotation_format`` is ``box``, ``point``, ``polygon`` or ``clip`` (the server's ``vision_config``).
        """
        video_path = Path(video_path) if video_path else None
        image_paths = [Path(p) for p in image_paths]
        media = ([video_path] if video_path else []) + image_paths
        settings = {"model": self.model, "prompt_name": prompt_name, "prompt_version": prompt_version,
                    "prompt_text": prompt_text, "media": [sha256_file(p) for p in media],
                    "annotation_format": annotation_format, "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort, "temperature": temperature}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / self.piece / f"{clip_id}__{prompt_name}-{prompt_version}__{key[:10]}.json"

        if path.exists():
            doc = json.loads(path.read_text())
            with self._lock:
                self.cache_hits += 1
            r = doc["response"]
            return Response(text=r["text"], finish_reason=r.get("finish_reason"), usage=r.get("usage") or {},
                            cost_usd=0.0, model=r.get("model"), request_id=r.get("request_id"),
                            cache_hit=True, cache_path=path, raw=doc.get("raw"))
        if self.offline:
            raise CacheMiss(f"{clip_id} {prompt_name}-{prompt_version} is not cached and BEEBOT_OFFLINE is set")
        spent = self.spent_usd()
        if spent >= self.cost_cap_usd:
            raise BudgetExceeded(f"{self.cost_log.name} shows ${spent:.2f} spent, cap is ${self.cost_cap_usd:.2f}")
        if not self.api_key:
            raise RuntimeError("PERCEPTRON_API_KEY is not set: copy .env.example to .env and fill it in")

        parts = self._media_parts(video_path, image_paths)
        result = self._request(parts, prompt_text, annotation_format=annotation_format, max_tokens=max_tokens,
                               reasoning_effort=reasoning_effort, temperature=temperature)
        with self._lock:
            self.api_calls += 1
        cost = self.cost_of(result["usage"], result.get("model"))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        doc = {
            "request": {"clip_id": clip_id, "piece": self.piece, "user": self.user, "created_at": now,
                        **{k: v for k, v in settings.items() if k != "media"},
                        "media": [p.name for p in media], "media_sha256": settings["media"]},
            "response": {"text": result["text"], "finish_reason": result["finish_reason"], "usage": result["usage"],
                         "model": result.get("model"), "request_id": result.get("request_id"), "cost_usd": cost},
            "raw": result.get("raw"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1), newline="\n")
        usage = result["usage"]
        self._log({"time": now, "user": self.user, "piece": self.piece, "clip_id": clip_id, "prompt": prompt_name,
                   "version": prompt_version, "model": result.get("model") or self.model,
                   "prompt_tokens": usage.get("prompt_tokens", ""),
                   "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", ""),
                   "completion_tokens": usage.get("completion_tokens", ""), "cost_usd": cost,
                   "finish_reason": result["finish_reason"], "request_id": result.get("request_id") or "",
                   "cache_file": str(path.relative_to(self.cache_dir)).replace("\\", "/")})
        return Response(text=result["text"], finish_reason=result["finish_reason"], usage=usage, cost_usd=cost,
                        model=result.get("model"), request_id=result.get("request_id"), cache_hit=False,
                        cache_path=path, raw=result.get("raw"))

    # ------------------------------------------------------------ SDK layer
    def _sdk_client(self):
        if self._sdk is None:
            from perceptron import Client

            self._sdk = Client(api_key=self.api_key)
        return self._sdk

    def _file_id(self, path: Path) -> str | None:
        """Upload a video once per API key and remember its id. None if the upload fails (then send bytes inline)."""
        from perceptron import SDKError

        ids_path = self.cache_dir / "file_ids.json"
        key = f"{hashlib.sha256(self.api_key.encode()).hexdigest()[:8]}:{sha256_file(path)}"
        with self._lock:
            ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
            if key in ids:
                return ids[key]["file_id"]
            try:
                uploaded = self._sdk_client().files.upload(str(path), purpose="vision")
            except SDKError as e:
                print(f"warning: upload of {path.name} failed ({e}); sending it inline")
                return None
            ids[key] = {"file_id": uploaded.id, "file": path.name,
                        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            ids_path.write_text(json.dumps(ids, indent=1, sort_keys=True), newline="\n")
            return uploaded.id

    def _media_parts(self, video_path: Path | None, image_paths: list[Path]) -> list:
        from perceptron import image, video

        parts = []
        if video_path is not None:
            fid = self._file_id(video_path)
            parts.append(video(file_id=fid) if fid else video(str(video_path)))
        parts.extend(image(str(p)) for p in image_paths)
        return parts

    def _request(self, parts: list, prompt_text: str, *, annotation_format: str, max_tokens: int,
                 reasoning_effort: str | None, temperature: float | None) -> dict:
        """The network call. Tests replace this method; nothing else in the class touches the API."""
        from perceptron import RateLimitError, ServerError, TimeoutError, TransportError

        kwargs = {"model": self.model,
                  "messages": [{"role": "user", "content": [*parts, {"type": "text", "text": prompt_text}]}],
                  "max_completion_tokens": max_tokens, "vision_config": {"annotation_format": annotation_format}}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            kwargs["temperature"] = temperature
        sdk = self._sdk_client()
        for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
            try:
                resp = sdk.chat.completions.create(**kwargs)
                break
            except (RateLimitError, ServerError, TimeoutError, TransportError) as e:
                if delay is None:
                    raise
                print(f"warning: {type(e).__name__} on attempt {attempt + 1}, retrying in {delay}s")
                time.sleep(delay)
        choice = resp.choices[0] if resp.choices else None
        u = resp.usage
        usage = {} if u is None else {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                                      "total_tokens": u.total_tokens,
                                      "prompt_tokens_details": u.prompt_tokens_details}
        return {"text": resp.text or "", "finish_reason": choice.finish_reason if choice else None, "usage": usage,
                "raw": resp.raw, "model": resp.model, "request_id": resp.request_id}
