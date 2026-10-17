"""Natural-language search over clips: "when does the bee land on a flower?" -> clip and timestamps.

    python -m search.search build                    # tracks/*.json (+ Mk1.5 captions if cached / key set) -> search/index.json
    python -m search.search build --captions         # also ask Mk1.5 to describe each clip's moments
    python -m search.search query "wasp leaves the flower"
    python -m search.search live sm3_12_123903 "bee lands on a flower"     # ask Mk1.5 directly

Index (CONTRACT.md): one entry per moment. ``kind`` is ``event`` (derived from tracks, flowers and events, so its
timestamps are as good as the tracking) or ``caption`` (Mk1.5's own description of a moment in the video).
Queries are matched offline with synonym expansion and IDF weighting, so search needs no network at demo time.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "search" / "index.json"

SPECIES_WORDS = {"honeybee": "honeybee bee", "vespidae": "wasp", "syrphidae": "hoverfly fly",
                 "lepidoptera": "butterfly moth", "unknown": "insect"}

# query word -> words it should also match (applied after stemming)
SYNONYMS = {
    "land": ["visit", "arrive", "touch", "alight", "settle"], "visit": ["land", "forage", "feed"],
    "forage": ["visit", "feed", "pollinate"], "feed": ["forage", "visit"], "pollinate": ["visit", "forage"],
    "leave": ["takeoff", "depart", "exit"], "takeoff": ["leave", "depart"], "depart": ["leave", "takeoff"],
    "fly": ["flight", "dart"], "flight": ["fly"], "fast": ["dart", "quick"], "quick": ["fast"],
    "enter": ["arrive", "appear"], "appear": ["enter", "arrive"], "arrive": ["enter", "land"],
    "exit": ["leave"], "disappear": ["leave", "exit"],
    "bee": ["honeybee", "insect"], "honeybee": ["bee"], "insect": ["bee", "wasp", "fly", "honeybee"],
    "pollinator": ["insect", "bee"], "wasp": ["vespid"], "flower": ["bloom", "blossom"], "bloom": ["flower"],
    "long": ["linger", "stay"], "linger": ["long", "stay"], "stay": ["linger", "long"],
    "return": ["revisit"], "revisit": ["return"], "walk": ["crawl"], "crawl": ["walk"],
    "two": ["multiple", "several"], "multiple": ["two", "several"], "together": ["multiple", "same"],
}
STOP = {"the", "a", "an", "on", "of", "to", "in", "at", "is", "does", "do", "when", "where", "what", "show", "me",
        "find", "moment", "moments", "time", "it", "its", "and", "or", "with", "from", "for", "any", "this", "that"}


def stem(w: str) -> str:
    w = w.lower()
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + rep
            break
    return {"landing": "land", "lands": "land", "flie": "fly", "flying": "fly", "takeoff": "takeoff",
            "taking": "take", "visitor": "visit", "foraging": "forage", "forag": "forage", "leav": "leave",
            "arriv": "arrive", "lingr": "linger", "linger": "linger", "pollinat": "pollinate"}.get(w, w)


def tokens(text: str) -> list[str]:
    text = text.lower().replace("take off", "takeoff").replace("takes off", "takeoff").replace("took off", "takeoff")
    return [stem(w) for w in re.findall(r"[a-z]+", text) if w not in STOP]


# ------------------------------------------------------------------ build
def derived_moments(doc: dict, duration_s: float) -> list[dict]:
    from analytics.metrics import center, clip_metrics

    cid = doc["clip_id"]
    out = []
    flowers = {f["flower_id"]: f for f in doc.get("flowers", [])}
    sp = {t["track_id"]: SPECIES_WORDS.get(t.get("species", "unknown"), "insect") for t in doc.get("tracks", [])}

    def add(t0, t1, text):
        out.append({"clip_id": cid, "t_start": round(max(t0, 0), 2), "t_end": round(min(t1, duration_s), 2),
                    "kind": "event", "text": text, "source": doc.get("model", "pipeline")})

    m = clip_metrics(doc, duration_s)
    for v in m["visits"]:
        who = sp.get(v["track_id"], "insect")
        fl = v["flower_id"]
        kind = "flower" if flowers.get(fl, {}).get("source", "mk1.5") != "stop" else "flower (stop)"
        add(v["t_start"], v["t_end"], f"{who} visits {kind} {fl}, forages for {v['dwell_s']:.1f} s")
        add(v["t_start"] - 0.5, v["t_start"] + 0.5, f"{who} lands on {kind} {fl}")
        add(v["t_end"] - 0.5, v["t_end"] + 0.5, f"{who} takes off and leaves {kind} {fl}")
        if v["dwell_s"] >= 4:
            add(v["t_start"], v["t_end"], f"{who} lingers, long visit at {fl} ({v['dwell_s']:.1f} s)")
        if v["revisit"]:
            add(v["t_start"], v["t_end"], f"{who} returns to {fl}, revisits a flower")
    for tr in doc.get("tracks", []):
        who = sp.get(tr["track_id"], "insect")
        obs = [p for p in tr["points"] if not p.get("interpolated")]
        if not obs:
            continue
        c0, c1 = center(obs[0]), center(obs[-1])
        edge = lambda c: min(c[0], c[1], 1000 - c[0], 1000 - c[1]) < 80
        if obs[0]["t"] > 0.3 or edge(c0):
            add(obs[0]["t"] - 0.3, obs[0]["t"] + 1.0, f"{who} enters the frame, appears, arrives")
        if obs[-1]["t"] < duration_s - 0.3 or edge(c1):
            add(obs[-1]["t"] - 1.0, obs[-1]["t"] + 0.3, f"{who} leaves the frame, flies away, disappears")
        # fast flight: 0.5 s windows where the bee covers > 120 units
        i = 0
        while i < len(obs):
            j = i
            while j + 1 < len(obs) and obs[j + 1]["t"] - obs[i]["t"] <= 0.5:
                j += 1
            if j > i and math.dist(center(obs[i]), center(obs[j])) > 120:
                add(obs[i]["t"], obs[j]["t"], f"{who} flies fast, quick flight, darts across")
                i = j
            i += 1
        # behavior label runs (from Mk1.5 crop checks)
        run = None
        for p in obs + [{"t": 1e9, "label": None}]:
            lab = p.get("label")
            if run and (lab != run[0] or p["t"] - run[2] > 0.6):
                if run[0] in ("walking", "flying") and run[2] - run[1] >= 0.5:
                    add(run[1], run[2], f"{who} {'walks, crawls on a leaf' if run[0] == 'walking' else 'flies, in flight'}")
                run = None
            if lab and run is None:
                run = [lab, p["t"], p["t"]]
            elif run:
                run[2] = p["t"]
    # several insects at once
    spans = [(min(p["t"] for p in t["points"]), max(p["t"] for p in t["points"])) for t in doc.get("tracks", []) if t["points"]]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = max(spans[i][0], spans[j][0]), min(spans[i][1], spans[j][1])
            if b - a > 0.5:
                add(a, b, "two insects at the same time, multiple pollinators together")
    return out


def caption_moments(clip_id: str, video: Path, client) -> list[dict]:
    prompt = ("Watch this video of insects on flowers. List every distinct thing an insect does (arrives, lands on "
              "a flower, feeds, walks, flies off, leaves the frame). Mark each moment with a clip tag whose mention is "
              "a short description, e.g. a bee lands on a white flower. Only describe what you can actually see.")
    r = client.call(clip_id=clip_id, prompt_name="captions", prompt_version="v1", prompt_text=prompt,
                    video_path=video, annotation_format="clip", max_tokens=1500)
    out = []
    for c in r.annotations().clips:
        at, until = c.timestamp.at, c.timestamp.until
        if c.mention:
            out.append({"clip_id": clip_id, "t_start": round(at, 2), "t_end": round(until if until else at + 1, 2),
                        "kind": "caption", "text": c.mention, "source": "mk1.5"})
    return out


def build(tracks_dir: Path, *, captions: bool) -> list[dict]:
    manifest = {e["clip_id"]: e for e in json.loads((ROOT / "clips" / "manifest.json").read_text())}
    client = None
    if captions:
        from perception.client import PerceptionClient
        client = PerceptionClient("search")
    index = []
    for cid, e in manifest.items():
        p = tracks_dir / f"{cid}.json"
        if p.exists():
            index += derived_moments(json.loads(p.read_text()), e["duration_s"])
        if client:
            index += caption_moments(cid, ROOT / e["path"], client)
    return index


# ------------------------------------------------------------------ query
def query(q: str, index: list[dict], *, clip_id: str | None = None, k: int = 10) -> list[dict]:
    docs = [e for e in index if clip_id in (None, e["clip_id"])]
    if not docs:
        return []
    dtoks = [set(tokens(e["text"])) for e in docs]
    df = Counter(t for ts in dtoks for t in ts)
    n = len(docs)
    idf = lambda t: math.log(1 + n / (1 + df.get(t, 0)))
    qt = tokens(q)
    if not qt:
        return []
    scored = []
    for e, ts in zip(docs, dtoks):
        s = 0.0
        for t in qt:
            if t in ts:
                s += idf(t)
            elif any(sy in ts for sy in SYNONYMS.get(t, [])):
                s += 0.6 * max(idf(sy) for sy in SYNONYMS[t] if sy in ts)
        if s > 0:
            # prefer short, specific moments over whole visits when scores tie
            scored.append((s / len(qt), -(e["t_end"] - e["t_start"]), e))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out, seen = [], set()
    for s, _, e in scored:
        key = (e["clip_id"], round(e["t_start"]), e["text"])
        if key not in seen:
            seen.add(key)
            out.append({**e, "score": round(s, 3)})
        if len(out) >= k:
            break
    return out


def live(clip_id: str, q: str, video: Path) -> list[dict]:
    """Ask Mk1.5 directly (costs a call; cached per query)."""
    from perception.client import PerceptionClient

    client = PerceptionClient("search")
    prompt = (f"Find every moment in this video where the following happens: {q}. Mark each one with a clip tag "
              "covering when it happens, with a short description as the mention. If it never happens, say so.")
    r = client.call(clip_id=clip_id, prompt_name="live", prompt_version="v1", prompt_text=prompt, video_path=video,
                    annotation_format="clip", max_tokens=800)
    return [{"clip_id": clip_id, "t_start": round(c.timestamp.at, 2),
             "t_end": round(c.timestamp.until if c.timestamp.until else c.timestamp.at + 1, 2), "kind": "caption",
             "text": c.mention or q, "source": "mk1.5-live"} for c in r.annotations().clips]


def load_index() -> list[dict]:
    return json.loads(INDEX.read_text()) if INDEX.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tracks", default="tracks")
    b.add_argument("--captions", action="store_true")
    qp = sub.add_parser("query")
    qp.add_argument("q")
    qp.add_argument("--clip")
    lp = sub.add_parser("live")
    lp.add_argument("clip")
    lp.add_argument("q")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        idx = build(ROOT / args.tracks, captions=args.captions)
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(json.dumps(idx, indent=1))
        print(f"{len(idx)} moments from {len({e['clip_id'] for e in idx})} clips -> {INDEX.relative_to(ROOT)}")
    elif args.cmd == "query":
        for r in query(args.q, load_index(), clip_id=args.clip):
            print(f"{r['score']:.2f}  {r['clip_id']:16s} {r['t_start']:6.2f}-{r['t_end']:6.2f}  [{r['kind']}] {r['text']}")
    else:
        manifest = {e["clip_id"]: e for e in json.loads((ROOT / "clips" / "manifest.json").read_text())}
        for r in live(args.clip, args.q, ROOT / manifest[args.clip]["path"]):
            print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
