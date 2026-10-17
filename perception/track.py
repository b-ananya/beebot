"""P1: one-bee tracking with Perceptron -> tracks/parts/<clip>.track.json.

    python -m perception.track --clips hb1_w00 sm3_12_123903     # a few clips
    python -m perception.track --all --merge                     # 48-clip batch, then tracks/*.json
    python -m perception.track --all --resolution 1080p          # needs raw/ sources (A2 test)

Reruns are free: the shared client serves every repeated request from tracks/raw/. The part file holds one
CONTRACT.md track (``<species>_1``) plus run metadata; ``merge.py`` turns parts into ``tracks/<clip>.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from perception.client import ROOT, PerceptionClient

PARTS_DIR = ROOT / "tracks" / "parts"
PROMPTS_DIR = ROOT / "prompts" / "track"
PIECE = "p1"

# how the prompt names each species (truth/*.json says which one is in the clip)
SPECIES_WORDS = {"honeybee": "honeybee", "vespidae": "wasp", "lepidoptera": "butterfly or moth",
                 "syrphidae": "hoverfly", "unknown": "insect"}


def manifest(root: Path = ROOT) -> dict[str, dict]:
    return {e["clip_id"]: e for e in json.loads((root / "clips" / "manifest.json").read_text())}


def species_for(clip_id: str, root: Path = ROOT) -> str:
    """Species from the truth file when there is one (it only tells the prompt what to look for)."""
    truth = root / "truth" / f"{clip_id}.json"
    if truth.exists():
        tracks = json.loads(truth.read_text()).get("tracks") or []
        if tracks and tracks[0].get("species"):
            return tracks[0]["species"]
    return "honeybee" if clip_id.startswith("hb") else "unknown"


def clip_path(clip_id: str, resolution: str = "720p", root: Path = ROOT) -> Path:
    entry = manifest(root)[clip_id]
    if resolution == "720p":
        return root / entry["path"]
    if resolution == "1080p":
        path = root / "clips" / "1080p" / f"{clip_id}.mp4"
        if not path.exists():
            cut_native(clip_id, path, root)
        return path
    raise ValueError(f"resolution must be 720p or 1080p, not {resolution}")


def cut_native(clip_id: str, dest: Path, root: Path = ROOT) -> Path:
    """Cut the clip's window from the native source under raw/ (needs the dataset download and ffmpeg)."""
    entry = manifest(root)[clip_id]
    sources = list((root / "raw").rglob(entry["source_video"])) if (root / "raw").exists() else []
    if not sources:
        raise FileNotFoundError(f"{entry['source_video']} not found under raw/; run scripts/fetch_data.py first")
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not on PATH (winget install Gyan.FFmpeg, then open a new shell)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{entry['source_start_s']:.3f}", "-i", str(sources[0]),
                    "-t", f"{entry['duration_s']:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
                    "-movflags", "+faststart", str(dest)], check=True)
    return dest


def load_prompt(version: str, **fields) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text().format(**fields).strip()


POINT_BOX_SIDE = 24  # a bee is ~25 px wide at 1280 px, about 20-30 units on the 0-1000 grid (DATA_NOTES.md)


def parse_track(text: str, duration_s: float, species: str, *, complete: bool = True,
                point_box_side: int = POINT_BOX_SIDE) -> dict:
    """The answer's markup as one CONTRACT.md track. Waypoints outside the clip or malformed are dropped, never
    fixed up; ``complete`` is False when the markup or the answer was cut off. Point waypoints become a box of
    ``point_box_side`` units around the point (``box_from: "point"``)."""
    from perceptron import BoundingBox, SinglePoint, collect_annotations

    tracks = [t for t in collect_annotations(text or "").tracks if t.points]
    track = max(tracks, key=lambda t: len(t.points)) if tracks else None
    points = []
    from_point = False
    if track is not None:
        for wp in track.points:
            if wp.t is None:
                continue
            t = float(wp.t)
            if isinstance(wp, BoundingBox):
                box = [wp.top_left.x, wp.top_left.y, wp.bottom_right.x, wp.bottom_right.y]
            elif isinstance(wp, SinglePoint):
                if not (0 <= wp.x <= 1000 and 0 <= wp.y <= 1000):
                    continue
                h = point_box_side / 2
                box = [max(0, wp.x - h), max(0, wp.y - h), min(1000, wp.x + h), min(1000, wp.y + h)]
                from_point = True
            else:
                continue
            if not (0 <= t <= duration_s + 1e-3) or not all(0 <= v <= 1000 for v in box):
                continue
            if box[0] > box[2] or box[1] > box[3]:
                continue
            points.append({"t": round(t, 3), "box": box,
                           "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                           "interpolated": False})
    points.sort(key=lambda p: p["t"])
    deduped = [p for i, p in enumerate(points) if i == 0 or p["t"] > points[i - 1]["t"]]
    out = {"track_id": f"{species}_1", "species": species,
           "complete": bool(complete and (track is None or track.complete)), "points": deduped}
    if from_point:
        out["box_from"] = "point"
    return out


def interpolate(points: list[dict], max_gap_s: float = 1.0, step_s: float = 0.1) -> list[dict]:
    """Fill gaps up to ``max_gap_s`` between observed points with linear boxes flagged ``interpolated`` (eval skips
    them). Longer gaps stay gaps: the bee was probably occluded or off screen."""
    out = []
    for i, p in enumerate(points):
        out.append(p)
        if i + 1 >= len(points):
            break
        q = points[i + 1]
        gap = q["t"] - p["t"]
        if gap <= step_s + 1e-9 or gap > max_gap_s:
            continue
        n = int(gap / step_s + 1e-9)
        for k in range(1, n + (0 if abs(n * step_s - gap) < 1e-9 else 1)):
            t = p["t"] + k * step_s
            if t >= q["t"] - 1e-9:
                break
            a = (t - p["t"]) / gap
            box = [round(p["box"][j] + a * (q["box"][j] - p["box"][j]), 1) for j in range(4)]
            out.append({"t": round(t, 3), "box": box,
                        "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                        "interpolated": True})
    return out


FRAMES_DIR = ROOT / "tracks" / "raw" / "frames"  # gitignored: frames are rebuilt from clips/
FRAME_WORKERS = 4


def extract_frames(clip_id: str, fps: float, resolution: str = "720p", root: Path = ROOT,
                   frames_dir: Path | None = None) -> list[tuple[float, Path]]:
    """Sample the clip every 1/fps seconds into JPEG frames: ``[(t_seconds, path), ...]``."""
    import cv2

    src = clip_path(clip_id, resolution, root)
    out_dir = (frames_dir or FRAMES_DIR) / resolution / clip_id / f"{fps:g}fps"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = {}
    k = 0
    while True:
        t = k / fps
        idx = round(t * video_fps)
        if idx >= n_frames:
            break
        wanted.setdefault(idx, round(idx / video_fps, 3))
        k += 1
    frames = []
    missing = [idx for idx in wanted if not (out_dir / f"{int(wanted[idx] * 1000):06d}.jpg").exists()]
    if missing:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx in wanted:
                cv2.imwrite(str(out_dir / f"{int(wanted[idx] * 1000):06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            idx += 1
    cap.release()
    for idx in sorted(wanted):
        path = out_dir / f"{int(wanted[idx] * 1000):06d}.jpg"
        if path.exists():
            frames.append((wanted[idx], path))
    return frames


def tile_frame(frame: Path, tiles: int, overlap: float = 0.1) -> list[tuple[Path, int, int, int, int, int, int]]:
    """Cut a frame into ``tiles`` x ``tiles`` overlapping crops: ``[(path, x0, y0, w, h, W, H), ...]`` in pixels.
    A 2x2 split makes a 25 px bee look twice as big to the model."""
    import cv2

    img = cv2.imread(str(frame))
    H, W = img.shape[:2]
    out = []
    w, h = W / tiles, H / tiles
    for r in range(tiles):
        for c in range(tiles):
            x0 = int(max(0, c * w - overlap * w))
            y0 = int(max(0, r * h - overlap * h))
            x1 = int(min(W, (c + 1) * w + overlap * w))
            y1 = int(min(H, (r + 1) * h + overlap * h))
            path = frame.with_name(f"{frame.stem}_t{tiles}_{r}{c}.jpg")
            if not path.exists():
                cv2.imwrite(str(path), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            out.append((path, x0, y0, x1 - x0, y1 - y0, W, H))
    return out


def parse_frame_point(text: str) -> tuple[int, int] | None:
    """The first point in a per-frame answer, or None when the model saw no insect."""
    from perceptron import BoundingBox, SinglePoint, collect_annotations

    ann = collect_annotations(text or "")
    for item in ann.points + ann.boxes:
        if isinstance(item, SinglePoint) and 0 <= item.x <= 1000 and 0 <= item.y <= 1000:
            return item.x, item.y
        if isinstance(item, BoundingBox):
            return (item.top_left.x + item.bottom_right.x) // 2, (item.top_left.y + item.bottom_right.y) // 2
    return None


MAX_JUMP_UNITS = 150  # with several tile candidates, prefer the one near the last accepted position


def track_frames(clip_id: str, *, version: str = "f1", fps: float = 2.0, resolution: str = "720p",
                 client: PerceptionClient, max_tokens: int = 256, reasoning_effort: str | None = None,
                 tiles: int = 1, root: Path = ROOT, point_box_side: int = POINT_BOX_SIDE) -> tuple[dict, dict]:
    """Per-frame point detection, one image call per sampled frame (or per tile of it), assembled into one track.
    The model sees each frame at full clip resolution, unlike video input, which the server samples and downscales.
    With ``tiles`` > 1 each frame is cut into an overlapping grid and every tile is asked separately."""
    from concurrent.futures import ThreadPoolExecutor

    entry = manifest(root)[clip_id]
    species = species_for(clip_id, root)
    prompt = load_prompt(version, species=SPECIES_WORDS.get(species, species), duration_s=entry["duration_s"],
                         step_s=round(1 / fps, 3))
    frames = extract_frames(clip_id, fps, resolution, root)
    jobs = []  # (t, image path, x0, y0, w, h, W, H): how to map a tile point back to the full frame
    for t, path in frames:
        if tiles > 1:
            jobs.extend((t, *tile) for tile in tile_frame(path, tiles))
        else:
            jobs.append((t, path, 0, 0, 1, 1, 1, 1))
    prompt_name = f"frame{fps:g}" + (f"t{tiles}" if tiles > 1 else "")

    def one(job):
        r = client.call(clip_id=clip_id, prompt_name=prompt_name, prompt_version=version, prompt_text=prompt,
                        image_paths=[job[1]], annotation_format="point", max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort)
        return job, r

    with ThreadPoolExecutor(FRAME_WORKERS) as pool:
        answers = list(pool.map(one, jobs))
    usage, cost, calls = {"prompt_tokens": 0, "completion_tokens": 0}, 0.0, 0
    candidates: dict[float, list[tuple[float, float]]] = {}
    for (t, _, x0, y0, w, h, W, H), r in answers:
        cost += r.cost_usd
        calls += not r.cache_hit
        for k in usage:
            usage[k] += (r.usage or {}).get(k) or 0
        xy = parse_frame_point(r.text)
        if xy is None:
            continue
        # tile coordinates (0-1000 of the tile) -> full-frame coordinates (0-1000 of the frame)
        x = (x0 + xy[0] / 1000 * w) / W * 1000
        y = (y0 + xy[1] / 1000 * h) / H * 1000
        candidates.setdefault(t, []).append((round(x, 1), round(y, 1)))
    points, last = [], None
    for t in sorted(candidates):
        cands = candidates[t]
        if last is not None:
            near = [c for c in cands if ((c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2) ** 0.5 <= MAX_JUMP_UNITS]
            xy = min(near, key=lambda c: (c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2) if near else cands[0]
        else:
            xy = cands[0]
        last = xy
        hs = point_box_side / 2
        box = [max(0, xy[0] - hs), max(0, xy[1] - hs), min(1000, xy[0] + hs), min(1000, xy[1] + hs)]
        points.append({"t": t, "box": box, "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                       "interpolated": False})
    track = {"track_id": f"{species}_1", "species": species, "complete": True, "points": points, "box_from": "point"}
    meta = {"model": client.model, "finish_reason": "stop", "usage": usage, "cost_usd": round(cost, 6),
            "cache_hit": calls == 0, "frames": len(frames), "fps": fps, "tiles": tiles,
            "frames_with_candidates": len(candidates),
            "cache_file": f"p1/{clip_id}__{prompt_name}-{version}__*.json", "parse_errors": []}
    return track, meta


def motion_candidates(video: Path, times: list[float], dt_s: float = 0.1, top: int = 3,
                      min_area: int = 60, max_area: int = 6000) -> dict[float, list[tuple[float, float]]]:
    """Moving blobs at each sample time, as 0-1000 centres: the difference between the frame at ``t`` and the one
    ``dt_s`` later, on a fixed camera. Free, no API call; the model verifies what is actually an insect."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    pairs = {round(t * fps): round((t + dt_s) * fps) for t in times}
    wanted = set(pairs) | set(pairs.values())
    grey = {}
    idx = 0
    while wanted and idx <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            grey[idx] = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        idx += 1
    cap.release()
    out = {}
    for t, a in zip(times, pairs):
        b = pairs[a]
        if a not in grey or b not in grey:
            continue
        diff = cv2.absdiff(grey[a], grey[b])
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = sorted([cv2.boundingRect(c) for c in cnts if min_area < cv2.contourArea(c) < max_area],
                       key=lambda r: -r[2] * r[3])[:top]
        out[t] = [(round((x + w / 2) / W * 1000, 1), round((y + h / 2) / H * 1000, 1)) for x, y, w, h in blobs]
    return out


def crop_around(frame: Path, x: float, y: float, size_px: int = 192) -> tuple[Path, int, int, int, int, int, int]:
    """A square crop centred on a 0-1000 point, clamped to the frame: ``(path, x0, y0, w, h, W, H)`` in pixels."""
    import cv2

    img = cv2.imread(str(frame))
    H, W = img.shape[:2]
    cx, cy = x / 1000 * W, y / 1000 * H
    x0 = int(min(max(0, cx - size_px / 2), W - size_px))
    y0 = int(min(max(0, cy - size_px / 2), H - size_px))
    path = frame.with_name(f"{frame.stem}_c{x0}_{y0}_{size_px}.jpg")
    if not path.exists():
        cv2.imwrite(str(path), img[y0:y0 + size_px, x0:x0 + size_px], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path, x0, y0, size_px, size_px, W, H


def _dedupe(points: list[tuple[float, float]], radius: float = 30) -> list[tuple[float, float]]:
    kept = []
    for p in points:
        if all(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 > radius for q in kept):
            kept.append(p)
    return kept


def track_hybrid(clip_id: str, *, version: str = "f3", tile_version: str = "f2", fps: float = 2.0, tiles: int = 2,
                 resolution: str = "720p", client: PerceptionClient, crop_px: int = 192, use_motion: bool = True,
                 root: Path = ROOT, point_box_side: int = POINT_BOX_SIDE) -> tuple[dict, dict]:
    """Propose, then verify. Candidates per frame come from the tiled point prompt (``tile_version``) and from
    motion; each candidate is re-asked on a zoomed crop (``version``), where a 25 px bee is large and obvious.
    Only verified candidates become track points."""
    from concurrent.futures import ThreadPoolExecutor

    entry = manifest(root)[clip_id]
    species = species_for(clip_id, root)
    word = SPECIES_WORDS.get(species, species)
    frames = extract_frames(clip_id, fps, resolution, root)
    times = [t for t, _ in frames]
    tile_prompt = load_prompt(tile_version, species=word, duration_s=entry["duration_s"], step_s=round(1 / fps, 3))
    verify_prompt = load_prompt(version, species=word, duration_s=entry["duration_s"], step_s=round(1 / fps, 3))
    usage, cost = {"prompt_tokens": 0, "completion_tokens": 0}, [0.0]
    calls = [0]

    def ask(prompt_name, prompt_version, prompt, path):
        r = client.call(clip_id=clip_id, prompt_name=prompt_name, prompt_version=prompt_version, prompt_text=prompt,
                        image_paths=[path], annotation_format="point", max_tokens=128)
        with client._lock:
            cost[0] += r.cost_usd
            calls[0] += not r.cache_hit
            for k in usage:
                usage[k] += (r.usage or {}).get(k) or 0
        return r

    def to_frame(xy, x0, y0, w, h, W, H):
        return round((x0 + xy[0] / 1000 * w) / W * 1000, 1), round((y0 + xy[1] / 1000 * h) / H * 1000, 1)

    # 1. propose
    tile_jobs = [(t, *tile) for t, path in frames for tile in tile_frame(path, tiles)] if tiles > 1 else []
    tile_name = f"frame{fps:g}t{tiles}"
    with ThreadPoolExecutor(FRAME_WORKERS) as pool:
        tile_answers = list(pool.map(lambda j: (j, ask(tile_name, tile_version, tile_prompt, j[1])), tile_jobs))
    candidates: dict[float, dict[tuple[float, float], set]] = {t: {} for t in times}
    for (t, _, *geom), r in tile_answers:
        xy = parse_frame_point(r.text)
        if xy is not None:
            candidates[t].setdefault(to_frame(xy, *geom), set()).add("tile")
    if use_motion:
        for t, pts in motion_candidates(clip_path(clip_id, resolution, root), times).items():
            for p in pts:
                candidates[t].setdefault(p, set()).add("motion")
    # merge near-duplicates, keeping every source that proposed them
    merged: dict[float, dict[tuple[float, float], set]] = {}
    for t, cands in candidates.items():
        keep = {}
        for p in _dedupe(sorted(cands, key=lambda p: -len(cands[p]))):
            keep[p] = set().union(*(s for q, s in cands.items() if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 <= 30))
        merged[t] = keep

    # 2. verify each candidate on a zoomed crop
    frame_by_t = dict(frames)
    verify_jobs = [(t, p, crop_around(frame_by_t[t], *p, size_px=crop_px)) for t, cands in merged.items() for p in cands]
    verify_name = f"verify{crop_px}"
    with ThreadPoolExecutor(FRAME_WORKERS) as pool:
        verified = list(pool.map(lambda j: (j, ask(verify_name, version, verify_prompt, j[2][0])), verify_jobs))
    confirmed: dict[float, list[tuple[tuple[float, float], int]]] = {}
    for (t, p, geom), r in verified:
        xy = parse_frame_point(r.text)
        if xy is None:
            continue
        confirmed.setdefault(t, []).append((to_frame(xy, *geom[1:]), len(merged[t][p])))

    # 3. one point per frame: most support, then closest to the last accepted point
    points, last = [], None
    for t in times:
        if t not in confirmed:
            continue
        opts = confirmed[t]
        if last is not None:
            opts = sorted(opts, key=lambda o: (-o[1], (o[0][0] - last[0]) ** 2 + (o[0][1] - last[1]) ** 2))
        else:
            opts = sorted(opts, key=lambda o: -o[1])
        xy = opts[0][0]
        last = xy
        hs = point_box_side / 2
        box = [max(0, xy[0] - hs), max(0, xy[1] - hs), min(1000, xy[0] + hs), min(1000, xy[1] + hs)]
        points.append({"t": t, "box": box, "center": [round((box[0] + box[2]) / 2, 1), round((box[1] + box[3]) / 2, 1)],
                       "interpolated": False})
    track = {"track_id": f"{species}_1", "species": species, "complete": True, "points": points, "box_from": "point"}
    meta = {"model": client.model, "finish_reason": "stop", "usage": usage, "cost_usd": round(cost[0], 6),
            "cache_hit": calls[0] == 0, "frames": len(frames), "fps": fps, "tiles": tiles, "crop_px": crop_px,
            "tile_version": tile_version, "candidates": sum(len(c) for c in merged.values()),
            "verified": sum(len(c) for c in confirmed.values()),
            "cache_file": f"p1/{clip_id}__{{{tile_name},{verify_name}}}-*.json", "parse_errors": []}
    return track, meta


def track_clip(clip_id: str, *, version: str = "v1", resolution: str = "720p", client: PerceptionClient | None = None,
               max_tokens: int = 4096, reasoning_effort: str | None = None, step_s: float = 0.5,
               annotation_format: str = "box", mode: str = "video", fps: float = 2.0, tiles: int = 1,
               root: Path = ROOT, parts_dir: Path | None = None) -> dict:
    """One clip -> tracks/parts/<clip>.track.json. ``mode`` is ``video`` (one call on the whole clip, the prompt asks
    for a track) or ``frames`` (one point call per frame sampled at ``fps``, or per tile with ``tiles`` > 1)."""
    client = client or PerceptionClient(PIECE)
    entry = manifest(root)[clip_id]
    species = species_for(clip_id, root)
    if mode == "hybrid":
        track, meta = track_hybrid(clip_id, version=version, fps=fps, tiles=tiles, resolution=resolution,
                                   client=client, root=root)
    elif mode == "frames":
        track, meta = track_frames(clip_id, version=version, fps=fps, resolution=resolution, client=client,
                                   max_tokens=max_tokens if max_tokens < 1024 else 256,
                                   reasoning_effort=reasoning_effort, tiles=tiles, root=root)
    else:
        prompt = load_prompt(version, species=SPECIES_WORDS.get(species, species), duration_s=entry["duration_s"],
                             step_s=step_s)
        resp = client.call(clip_id=clip_id, prompt_name="track", prompt_version=version, prompt_text=prompt,
                           video_path=clip_path(clip_id, resolution, root), annotation_format=annotation_format,
                           max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        track = parse_track(resp.text, entry["duration_s"], species, complete=resp.complete)
        meta = {"model": resp.model or client.model, "finish_reason": resp.finish_reason, "usage": resp.usage,
                "cost_usd": resp.cost_usd, "cache_hit": resp.cache_hit,
                "cache_file": str(resp.cache_path.relative_to(client.cache_dir)).replace("\\", "/"),
                "parse_errors": resp.annotations().errors}
    part = {"clip_id": clip_id, "prompt_version": version, "mode": mode, "resolution": resolution,
            "annotation_format": annotation_format, "step_s": step_s if mode == "video" else round(1 / fps, 3),
            "reasoning_effort": reasoning_effort, **meta, "tracks": [track]}
    parts_dir = parts_dir or PARTS_DIR
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / f"{clip_id}.track.json").write_text(json.dumps(part, indent=1), newline="\n")
    return part


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[], help="clip ids (default: none)")
    ap.add_argument("--all", action="store_true", help="every clip in clips/manifest.json")
    ap.add_argument("--version", default="v1", help="prompt version: prompts/track/<version>.txt")
    ap.add_argument("--resolution", default="720p", choices=["720p", "1080p"])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reasoning-effort", default=None, help="e.g. low, medium, high (costs output tokens)")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between boxes the prompt asks for")
    ap.add_argument("--format", default="box", choices=["box", "point"], dest="annotation_format",
                    help="what the prompt asks for; points become fixed-size boxes (box_from: point)")
    ap.add_argument("--mode", default="video", choices=["video", "frames", "hybrid"],
                    help="video: one call per clip; frames: one point call per sampled frame (see --fps); "
                         "hybrid: tile + motion candidates verified on zoomed crops (--version is the verify prompt)")
    ap.add_argument("--fps", type=float, default=2.0, help="frames mode: frames sampled per second")
    ap.add_argument("--tiles", type=int, default=1, help="frames mode: cut each frame into an N x N grid")
    ap.add_argument("--merge", action="store_true", help="also write tracks/<clip>.json for these clips")
    ap.add_argument("--model-info", action="store_true", help="refresh tracks/raw/model_info.json (pricing, limits)")
    args = ap.parse_args(argv)

    client = PerceptionClient(PIECE)
    if args.model_info:
        info = client.model_info()
        print(f"{info['id']}: context {info.get('max_context_tokens')}, max output {info.get('max_output_tokens')}, "
              f"prices/M in={client.prices()['input']} out={client.prices()['output']}")
    clips = list(manifest()) if args.all else args.clips
    if not clips:
        ap.error("give --clips ... or --all")
    print(f"model {client.model} | user {client.user} | spent ${client.spent_usd():.3f} of ${client.cost_cap_usd:.0f}")
    total = 0.0
    for clip_id in clips:
        part = track_clip(clip_id, version=args.version, resolution=args.resolution, client=client,
                          max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort, step_s=args.step,
                          annotation_format=args.annotation_format, mode=args.mode, fps=args.fps, tiles=args.tiles)
        tr = part["tracks"][0]
        total += part["cost_usd"]
        flag = "cache" if part["cache_hit"] else f"${part['cost_usd']:.4f}"
        state = "" if tr["complete"] else f"  INCOMPLETE ({part['finish_reason']})"
        print(f"{clip_id:16s} {len(tr['points']):3d} boxes  {flag:>8s}{state}")
    print(f"{client.api_calls} API calls, {client.cache_hits} cache hits, ${total:.4f} this run, "
          f"${client.spent_usd():.3f} total on {client.cost_log.name}")
    if args.merge:
        from perception.merge import merge_clip

        for clip_id in clips:
            merge_clip(clip_id)
        print(f"merged {len(clips)} clips into tracks/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
