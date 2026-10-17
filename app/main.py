"""BeeBot demo app.  Run from the repo root:  streamlit run app/main.py

Panel 1: clip with bee box, trail, flower boxes, visit flashes.
Panel 2: analytics (summary, visits per flower, heatmap, timeline).
Search: natural-language search that jumps the video; optional live Mk1.5 question.
Phase 2 (field view, robot route) gets its own tab once field/ and plan/ files exist.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `app.*` and `perception.*` import when run via streamlit

import os  # noqa: E402

os.chdir(ROOT)

import streamlit as st  # noqa: E402

from app import analytics, overlay, search  # noqa: E402

st.set_page_config(page_title="BeeBot", page_icon="🐝", layout="wide")

SOURCES = {"Mk1.5 tracks": Path("tracks"), "Mock (for building)": Path("tracks/mock"), "Reference truth": Path("truth")}


@st.cache_data
def manifest():
    return json.loads(Path("clips/manifest.json").read_text())


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


clips = manifest()
by_id = {c["clip_id"]: c for c in clips}

with st.sidebar:
    st.title("🐝 BeeBot")
    available = {name: d for name, d in SOURCES.items() if any(d.glob("*.json"))}
    source = st.radio("Tracks source", list(available), index=0)
    src_dir = available[source]
    with_tracks = [c["clip_id"] for c in clips if (src_dir / f"{c['clip_id']}.json").exists()]
    group = st.radio("Clips", ["Farm (flowers + visits)", "Honeybee close-ups", "All"], index=0)
    shown = [c for c in with_tracks if group == "All" or c.startswith("sm" if group.startswith("Farm") else "hb")]
    clip_id = st.selectbox("Clip", shown or with_tracks)
    st.caption(f"{len(with_tracks)} clips have {source.lower()}")

clip = by_id[clip_id]
doc = load(src_dir / f"{clip_id}.json")
if not Path(clip["path"]).exists():
    st.error(f"{clip['path']} is missing. Download the data release (see README).")
    st.stop()

tab_video, tab_field = st.tabs(["Video analysis", "Field planner (Phase 2)"])

with tab_video:
    # ---- search (sets the start time for the video)
    docs = [d for d in (load(p) for p in sorted(src_dir.glob("*.json"))) if d]
    entries, index_name = search.load_index(docs)
    q_col, live_col = st.columns([4, 1])
    query = q_col.text_input("Search your footage", placeholder="when does the bee land on a flower?")
    live = live_col.toggle("Ask Mk1.5 live", help="Uses your API credits; answers are cached.")
    start = st.session_state.get("seek", {}).get(clip_id, 0.0)
    if query:
        if live:
            try:
                hits, res = search.ask_live(clip["path"], query)
                st.caption(f"Mk1.5 answer {'(cached)' if res['cached'] else ''} · ${res['cost_usd']:.4f}")
            except Exception as e:  # no key, budget cap, network: say so, don't crash the demo
                st.warning(f"Live search unavailable ({type(e).__name__}); showing the offline index.")
                hits = search.search(entries, query)
        else:
            hits = search.search(entries, query)
        if not hits:
            st.info(f"No matches in {index_name}.")
        for i, h in enumerate(hits):
            label = f"{h['clip_id']} · {h['t_start']:.1f}–{h['t_end']:.1f}s · {h['text']}"
            if st.button(label, key=f"hit{i}"):
                st.session_state.setdefault("seek", {})[h["clip_id"]] = h["t_start"]
                if h["clip_id"] != clip_id:
                    st.info(f"Pick {h['clip_id']} in the sidebar to jump there.")
                st.rerun()
        st.caption(f"Searching {index_name}")

    # ---- panel 1 + panel 2
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Tracking")
        with st.spinner("Rendering overlay (first time only)..."):
            video = overlay.annotated(clip["path"], src_dir / f"{clip_id}.json")
        st.video(str(video), start_time=int(start))
        st.caption(f"{clip['source']} · {clip['duration_s']:.1f}s · {doc['model']}"
                   + (f" · prompt {doc['prompt_version']}" if doc.get("prompt_version") else ""))
    with right:
        st.subheader("Analytics")
        stats = analytics.summary(doc, clip["duration_s"])
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)
        if doc.get("flowers") or doc.get("events"):
            st.plotly_chart(analytics.visits_per_flower(doc), width="stretch")
        if any(tr["points"] for tr in doc["tracks"]):
            st.plotly_chart(analytics.heatmap(doc), width="stretch")
    st.plotly_chart(analytics.timeline(doc, clip["duration_s"]), width="stretch")

with tab_field:
    plan = load(Path("plan") / f"{clip_id}.json")
    field = load(Path("field") / f"{clip_id}.json")
    if not (plan or field):
        st.info("Phase 2: the pollination map (field/) and robot route (plan/) appear here once B and C write them.")
    else:
        if field:
            st.json(field, expanded=False)
        if plan:
            if plan.get("video") and Path(plan["video"]).exists():
                st.video(plan["video"])
            st.metric("deficit covered", f"{plan['deficit_covered']:.0%}")
            st.bar_chart({"BeeBot": plan["deficit_covered"], **plan.get("baselines", {})})
