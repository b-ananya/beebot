"""Natural-language search over clips.

Offline layer: search/index.json (built by B from Mk1.5 output, see CONTRACT.md). Until it
exists, we index the events and track presence in whatever tracks files the app is showing, so
the search box works from hour 0. Matching is simple word overlap; B can swap in better scoring.
Live layer (optional): ask Mk1.5 directly about one clip, with expects="clip".
"""
import json
import re
from pathlib import Path

INDEX = Path("search/index.json")
STOP = {"the", "a", "an", "does", "do", "is", "when", "where", "what", "on", "to", "of", "in", "show", "me"}
SYNONYMS = {"lands": "visit", "land": "visit", "landing": "visit", "visits": "visit", "forages": "visit",
            "feeding": "visit", "flies": "visible", "flying": "visible", "appears": "visible", "seen": "visible"}


def _words(s):
    return {SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", s.lower())} - STOP


def build_fallback(docs):
    """Index entries from tracks docs: one per event, one per visible stretch."""
    entries = []
    for doc in docs:
        for e in doc.get("events", []):
            entries.append({"clip_id": doc["clip_id"], "t_start": e["t_start"], "t_end": e["t_end"], "kind": "event",
                            "text": f"bee {e['track_id']} lands on {e['flower_id']} ({e['type']})",
                            "source": doc.get("model")})
        for tr in doc.get("tracks", []):
            if tr["points"]:
                entries.append({"clip_id": doc["clip_id"], "t_start": tr["points"][0]["t"],
                                "t_end": tr["points"][-1]["t"], "kind": "caption",
                                "text": f"{tr.get('species') or 'bee'} visible flying bee appears",
                                "source": doc.get("model")})
    return entries


def load_index(fallback_docs):
    if INDEX.exists():
        return json.loads(INDEX.read_text()), "search/index.json"
    return build_fallback(fallback_docs), "fallback (events in loaded tracks)"


def search(entries, query, clip_id=None, k=8):
    q = _words(query)
    if not q:
        return []
    scored = []
    for e in entries:
        if clip_id and e["clip_id"] != clip_id:
            continue
        score = len(q & _words(e["text"])) / len(q)
        if score > 0:
            scored.append((score, e))
    # ties: events before generic captions, then clip order
    scored.sort(key=lambda s: (-s[0], s[1]["kind"] != "event", s[1]["clip_id"], s[1]["t_start"]))
    return [e for _, e in scored[:k]]


def ask_live(clip_path, query):
    """One Mk1.5 call (cached by perception.mk15) returning time ranges for the query."""
    from perception.mk15 import Mk15

    res = Mk15().ask([clip_path], f"{query} Answer with the time ranges in the video where this happens.",
                     expects="clip", tag=f"search_{Path(clip_path).stem}")
    return [{"clip_id": Path(clip_path).stem, "t_start": c["at"], "t_end": c["until"] or c["at"],
             "kind": "live", "text": c.get("mention") or query, "source": "mk1.5-live"} for c in res["clips"]], res
