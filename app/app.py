"""BeeBot app: tracked video + search, analytics, forecasting, dataset view, analyze-your-own-video.

    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics.metrics import clip_metrics, heatmap  # noqa: E402
from app.render import cached_render, render  # noqa: E402
from search.search import derived_moments, load_index, query  # noqa: E402

# Validated reference palette (dataviz skill): first three categorical slots pass all-pairs CVD checks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED = "#8a8985"
GRID = "rgba(128,128,128,0.18)"

SOURCES = {
    "Mk1.5 pipeline": "tracks",
    "Motion only (no API)": "tracks/heuristic",
    "Reference tracker (truth)": "truth",
    "Mock": "tracks/mock",
}

st.set_page_config(page_title="BeeBot", page_icon="🐝", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def manifest() -> dict:
    return {e["clip_id"]: e for e in json.loads((ROOT / "clips" / "manifest.json").read_text())}


def doc_path(src: str, clip_id: str) -> Path:
    return ROOT / src / f"{clip_id}.json"


@st.cache_data
def load_doc(src: str, clip_id: str, mtime: float) -> dict | None:
    p = doc_path(src, clip_id)
    return json.loads(p.read_text()) if p.exists() else None


def get_doc(src: str, clip_id: str) -> dict | None:
    p = doc_path(src, clip_id)
    return load_doc(src, clip_id, p.stat().st_mtime) if p.exists() else None


def available_clips(src: str) -> list[str]:
    return [c for c in manifest() if doc_path(src, c).exists()]


@st.cache_data
def reference_rgb(video: str) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, 15).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return cv2.cvtColor(np.median(np.stack(frames), axis=0).astype(np.uint8), cv2.COLOR_BGR2RGB)


@st.cache_data
def search_index(src: str, stamp: tuple) -> list[dict]:
    idx = []
    for cid, e in manifest().items():
        d = get_doc(src, cid)
        if d:
            idx += derived_moments(d, e["duration_s"])
    if src == "tracks":
        idx += [m for m in load_index() if m["kind"] == "caption"]
    return idx


def style(fig: go.Figure, height: int = 300, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12), hoverlabel=dict(font_size=12), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def image_fig(img: np.ndarray, height: int = 420) -> go.Figure:
    """Plot in contract units (0-1000 on both axes, y down) over the frame."""
    fig = go.Figure()
    fig.add_layout_image(dict(source=_to_pil(img), xref="x", yref="y", x=0, y=0, sizex=1000, sizey=1000,
                              sizing="stretch", layer="below"))
    fig.update_xaxes(range=[0, 1000], visible=False)
    fig.update_yaxes(range=[1000, 0], visible=False)
    return style(fig, height, showlegend=True, legend=dict(orientation="h", y=-0.02))


def _to_pil(img):
    from PIL import Image
    return Image.fromarray(img)


def track_color(doc: dict, track_id: str) -> str:
    ids = [t["track_id"] for t in doc["tracks"]]
    return SERIES[ids.index(track_id) % len(SERIES)] if track_id in ids else MUTED


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🐝 BeeBot")
st.sidebar.caption("Pollinator video → tracks, visits, analytics and forecasts for robot pollinators.")
qp = st.query_params      # ?source=truth&clip=sm3_12_123903 for shareable views
_src_from_url = next((i for i, v in enumerate(SOURCES.values()) if v.split("/")[-1] == qp.get("source")), None)
src_label = st.sidebar.radio("Data source", list(SOURCES),
                             index=_src_from_url if _src_from_url is not None else (0 if available_clips("tracks") else 1),
                             help="Mk1.5 pipeline = motion tracker + Mk1.5 crop verification + Mk1.5 flower "
                                  "boxes. Reference tracker = the dataset authors' HyDaT/Polytrack output.")
SRC = SOURCES[src_label]
clips = available_clips(SRC)
if not clips:
    st.sidebar.warning(f"No clips for this source yet. Run `python -m perception.pipeline --all"
                       f"{' --no-api' if SRC.endswith('heuristic') else ''}`.")
    clip_id = None
else:
    groups = {"Farm (strawberry)": [c for c in clips if c.startswith("sm")],
              "Honeybee close-up": [c for c in clips if c.startswith("hb")]}
    want = st.session_state.get("jump_clip") or qp.get("clip")
    names = [g for g, cs in groups.items() if cs]
    group = st.sidebar.selectbox("Clip set", names,
                                 index=next((i for i, g in enumerate(names) if want in groups[g]), 0))
    clip_id = st.sidebar.selectbox("Clip", groups[group],
                                   index=groups[group].index(want) if want in groups[group] else 0)
st.sidebar.divider()
st.sidebar.caption("Truth is the dataset authors' tracker, not hand labels: errors are *agreement with a "
                   "reference tracker*. A visit is not proof of pollination.")

tab_video, tab_stats, tab_forecast, tab_data, tab_upload = st.tabs(
    ["🎥 Video & search", "📊 Analytics", "🔮 Forecast", "🗂 Dataset & accuracy", "⬆️ Analyze a video"])

entry = manifest().get(clip_id) if clip_id else None
doc = get_doc(SRC, clip_id) if clip_id else None
video = ROOT / entry["path"] if entry else None
metrics = clip_metrics(doc, entry["duration_s"]) if doc else None


# ------------------------------------------------------------------ tab: video & search
with tab_video:
    if not doc:
        st.info("Pick a source and clip in the sidebar.")
    else:
        q_col, opt_col = st.columns([4, 1])
        with q_col:
            q = st.text_input("Search moments", placeholder="when does the bee land on a flower?",
                              key="q")
        with opt_col:
            scope = st.radio("Scope", ["This clip", "All clips"], horizontal=False, label_visibility="collapsed")
        stamp = tuple(sorted((c, doc_path(SRC, c).stat().st_mtime) for c in clips))
        results = query(q, search_index(SRC, stamp), clip_id=clip_id if scope == "This clip" else None, k=8) if q else []
        if q and not results:
            st.caption("No matching moments.")
        start = st.session_state.get("start", 0.0) if st.session_state.get("start_clip") == clip_id else 0.0
        if results:
            cols = st.columns(min(len(results), 4))
            for i, r in enumerate(results):
                with cols[i % len(cols)]:
                    label = f"{r['clip_id']} · {r['t_start']:.1f}s" if scope == "All clips" else f"{r['t_start']:.1f}s"
                    if st.button(label, key=f"res{i}", help=f"{r['text']}  [{r['kind']}, {r['source']}]",
                                 width="stretch"):
                        st.session_state.update(start=max(r["t_start"] - 0.5, 0), start_clip=r["clip_id"],
                                                jump_clip=r["clip_id"])
                        st.rerun()
                    st.caption(r["text"])
        v_col, side = st.columns([3, 1])
        with v_col:
            with st.spinner("Rendering overlay (first time only)…"):
                out = cached_render(SRC, clip_id, video)
            st.video(str(out), start_time=int(start))
        with side:
            s = metrics["summary"]
            st.metric("Insects", s["insects"], help=", ".join(s["species"]))
            st.metric("Flower visits", s["visits"])
            st.metric("Median dwell", f"{s['median_dwell_s']} s" if s["median_dwell_s"] is not None else "–")
            st.metric("Tracked share of clip", f"{s['tracked_share']:.0%}" if s["tracked_share"] is not None else "–")
            st.caption(entry["source"])

        # timeline: insect presence and visits, one row per flower
        fig = go.Figure()
        rows = []
        for tr in doc["tracks"]:
            obs = [p["t"] for p in tr["points"] if not p.get("interpolated")]
            if obs:
                rows.append(tr["track_id"])
                segs, a, b = [], obs[0], obs[0]
                for t in obs[1:]:
                    if t - b > 0.6:
                        segs.append((a, b))
                        a = t
                    b = t
                segs.append((a, b))
                for a, b in segs:
                    fig.add_trace(go.Bar(x=[max(b - a, 0.05)], base=[a], y=[tr["track_id"]], orientation="h",
                                         marker=dict(color=track_color(doc, tr["track_id"]), cornerradius=4),
                                         width=0.5, showlegend=False,
                                         hovertemplate=f"{tr['track_id']} visible %{{base:.1f}}–{b:.1f}s<extra></extra>"))
        for v in metrics["visits"]:
            fig.add_trace(go.Bar(x=[max(v["t_end"] - v["t_start"], 0.05)], base=[v["t_start"]], y=[v["flower_id"]],
                                 orientation="h", width=0.5, showlegend=False,
                                 marker=dict(color=track_color(doc, v["track_id"]), cornerradius=4),
                                 hovertemplate=(f"{v['track_id']} on {v['flower_id']}<br>{v['t_start']:.1f}–"
                                                f"{v['t_end']:.1f}s · dwell {v['dwell_s']}s<extra></extra>")))
        fig.add_vline(x=start, line=dict(color=MUTED, dash="dot", width=1))
        style(fig, 90 + 26 * (len(rows) + len({v['flower_id'] for v in metrics['visits']})),
              title=dict(text="Timeline: when each insect is visible and which flower it's on", font_size=13),
              barmode="overlay", xaxis_title="seconds")
        fig.update_xaxes(range=[0, entry["duration_s"]])
        st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------ tab: analytics
with tab_stats:
    if not doc:
        st.info("Pick a source and clip in the sidebar.")
    else:
        s = metrics["summary"]

        def fmt(v, unit="", pct=False):
            if v is None:
                return "–"
            return f"{v:.0%}" if pct else f"{v}{unit}"

        st.subheader("Visits")
        c = st.columns(5)
        c[0].metric("Visits", s["visits"], help="insect inside a flower's zone for ≥ 0.3 s")
        c[1].metric("Flowers visited", s["flowers_visited"])
        c[2].metric("Flowers never visited", s["flowers_never_visited"], help="detected flowers with zero visits")
        c[3].metric("Visits / min", fmt(s["visits_per_min"]))
        c[4].metric("Revisit rate", fmt(s["revisit_rate"], pct=True), help="visits to a flower the same insect already visited")
        st.subheader("Robot parameters")
        c = st.columns(5)
        c[0].metric("Median dwell", fmt(s["median_dwell_s"], " s"), help="how long to hold at a flower")
        c[1].metric("Median transit", fmt(s["median_transit_s"], " s"), help="take-off to next landing")
        c[2].metric("Flight speed", fmt(s["median_flight_speed_cm_s"], " cm/s"),
                    help=f"median speed while moving faster than 40 units/s. Scale: {s['scale_note']}")
        c[3].metric("Approach time", fmt(s["median_approach_s"], " s"), help="time near the flower before landing")
        c[4].metric("Picks nearest flower", fmt(s["nearest_flower_rate"], pct=True),
                    help="share of moves to the nearest other flower — how greedy the insect's route is")
        c = st.columns(5)
        c[0].metric("Median turn", fmt(s["median_abs_turn_deg"], "°"), help="per 0.2 s step; path shape for simulation")
        c[1].metric("Moving share", fmt(s["flying_share_by_speed"], pct=True))
        c[2].metric("Tracked share", fmt(s["tracked_share"], pct=True))
        c[3].metric("Interpolated points", fmt(s["interpolated_share"], pct=True), help="lower is better")
        c[4].metric("Scale", fmt(s["scale_units_per_cm"], " u/cm"), help=s["scale_note"])

        left, right = st.columns(2)
        with left:
            fl = pd.DataFrame(metrics["flowers"])
            if len(fl) and fl["visits"].sum() > 0:
                fl = fl[fl["visits"] > 0].sort_values("dwell_s")
                fig = go.Figure(go.Bar(x=fl["dwell_s"], y=fl["flower_id"], orientation="h",
                                       marker=dict(color=SERIES[0], cornerradius=4),
                                       customdata=fl[["visits"]].to_numpy(),
                                       hovertemplate="%{y}<br>%{x:.1f} s total · %{customdata[0]} visit(s)<extra></extra>"))
                st.plotly_chart(style(fig, 60 + 28 * len(fl), title=dict(text="Time at each flower (s)", font_size=13)),
                                width="stretch")
            else:
                st.caption("No visits in this clip.")
            if metrics["speeds"]:
                fig = go.Figure(go.Histogram(x=metrics["speeds"], nbinsx=40, marker=dict(color=SERIES[0]),
                                             hovertemplate="%{x} units/s · %{y} steps<extra></extra>"))
                fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
                st.plotly_chart(style(fig, 260, title=dict(text="Speed distribution (units/s, 1 unit = 1/1000 frame width)",
                                                            font_size=13), bargap=0.05), width="stretch")
        with right:
            img = reference_rgb(str(video))
            hm = heatmap(doc, bins=40)
            fig = image_fig(img, 380)
            if hm.sum() > 0:
                z = np.where(hm > 0, hm / hm.max(), np.nan)
                fig.add_trace(go.Heatmap(z=z, x=np.linspace(12.5, 987.5, 40), y=np.linspace(12.5, 987.5, 40),
                                         colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)], opacity=0.75,
                                         showscale=True, colorbar=dict(title="time", tickvals=[]),
                                         hovertemplate="relative time here: %{z:.2f}<extra></extra>"))
            for f in doc["flowers"]:
                b = f["box"]
                fig.add_shape(type="rect", x0=b[0], y0=b[1], x1=b[2], y1=b[3], line=dict(color="#eda100", width=1))
            fig.update_layout(title=dict(text="Where the insects spend time", font_size=13), showlegend=False)
            st.plotly_chart(fig, width="stretch")
            if metrics["transitions"]:
                st.caption("Flower-to-flower moves")
                st.dataframe(pd.DataFrame(metrics["transitions"]), hide_index=True, width="stretch")
        with st.expander("Visit table"):
            st.dataframe(pd.DataFrame(metrics["visits"]), hide_index=True, width="stretch")
        with st.expander("All metrics (JSON)"):
            st.json(s)


# ------------------------------------------------------------------ tab: forecast
with tab_forecast:
    from forecast import predict as fc

    if not doc:
        st.info("Pick a source and clip in the sidebar.")
    elif not fc.available():
        st.warning("Forecast models not trained yet: run `python -m forecast.train`.")
    elif not doc["tracks"]:
        st.caption("No insects tracked in this clip.")
    else:
        if clip_id.startswith("hb"):
            st.caption("Models were trained on the strawberry-farm cameras; on honeybee close-ups the scale "
                       "differs, so read these forecasts as qualitative.")
        c1, c2 = st.columns([1, 3])
        with c1:
            tid = st.selectbox("Insect", [t["track_id"] for t in doc["tracks"]])
            tr = next(t for t in doc["tracks"] if t["track_id"] == tid)
            t_lo, t_hi = tr["points"][0]["t"] + 1.0, tr["points"][-1]["t"]
            t_now = st.slider("Forecast from (s)", float(round(min(t_lo, t_hi), 1)), float(round(t_hi, 1)),
                              float(round((t_lo + t_hi) / 2, 1)), 0.1)
            horizon = st.slider("Rollout horizon (s)", 5, 30, 10)
        res = fc.forecast(doc, tid, t_now, horizon_s=horizon)
        with c2:
            fig = image_fig(reference_rgb(str(video)), 460)
            for f in doc["flowers"]:
                b = f["box"]
                fig.add_shape(type="rect", x0=b[0], y0=b[1], x1=b[2], y1=b[3], line=dict(color="#eda100", width=1))
            past = [p for p in tr["points"] if not p.get("interpolated") and p["t"] <= t_now]
            if past:
                cs = np.array([p.get("center") or [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]
                               for p in past])
                fig.add_trace(go.Scatter(x=cs[:, 0], y=cs[:, 1], mode="lines", name="observed path",
                                         line=dict(color=SERIES[0], width=2), hoverinfo="skip"))
            fut = [p for p in tr["points"] if not p.get("interpolated") and t_now < p["t"] <= t_now + 2.0]
            if fut:
                cs = np.array([p.get("center") or [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]
                               for p in fut])
                fig.add_trace(go.Scatter(x=cs[:, 0], y=cs[:, 1], mode="lines", name="what actually happened",
                                         line=dict(color=MUTED, width=2, dash="dot"), hoverinfo="skip"))
            if "path" in res:
                for k, f in enumerate(res["path"]["fan"]):
                    f = np.array(f)
                    fig.add_trace(go.Scatter(x=f[:, 0], y=f[:, 1], mode="lines", showlegend=k == 0,
                                             name="possible futures (25)", line=dict(color=SERIES[1], width=1.5),
                                             opacity=0.45, hoverinfo="skip"))
                m = np.array(res["path"]["median"])
                fig.add_trace(go.Scatter(x=m[:, 0], y=m[:, 1], mode="lines+markers", name="forecast (next 2 s)",
                                         line=dict(color=SERIES[1], width=2), marker=dict(size=8),
                                         customdata=res["path"]["t"],
                                         hovertemplate="t = %{customdata:.1f}s<extra></extra>"))
            fig.update_layout(title=dict(text=f"{tid} at {t_now:.1f}s: path forecast", font_size=13))
            st.plotly_chart(fig, width="stretch")
        c = st.columns(4)
        dq = res.get("dwell_model_s") or {}
        c[0].metric("Dwell remaining", f"{res['dwell_remaining_s']} s" if "dwell_remaining_s" in res else "not on a flower",
                    help=f"typical dwell for this species: median {dq.get('q50')} s (10–90%: {dq.get('q10')}–{dq.get('q90')} s)")
        c[1].metric("Visits another flower", f"{res['p_another_visit']:.0%}" if "p_another_visit" in res else "–")
        ro = res.get("rollout") or {}
        c[2].metric(f"Leaves within {horizon}s", f"{ro['p_leaves_patch']:.0%}" if ro else "–")
        seq = ro.get("most_likely_sequence", [])
        c[3].metric("Most likely sequence", " → ".join(seq[:4]) if seq else ("leaves the patch" if ro else "–"),
                    help="the single most common rollout; see expected visits for the full picture")
        l, r = st.columns(2)
        if res.get("next_flower"):
            nf = pd.DataFrame(res["next_flower"])[::-1]
            fig = go.Figure(go.Bar(x=nf["p"], y=nf["flower_id"], orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                                   hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
            fig.update_xaxes(tickformat=".0%")
            l.plotly_chart(style(fig, 230, title=dict(text="Next flower probability", font_size=13)), width="stretch")
        if ro.get("expected_visits"):
            ev = pd.DataFrame(list(ro["expected_visits"].items())[:8], columns=["flower", "visits"])[::-1]
            fig = go.Figure(go.Bar(x=ev["visits"], y=ev["flower"], orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                                   hovertemplate="%{y}: %{x:.2f} expected visits<extra></extra>"))
            r.plotly_chart(style(fig, 230, title=dict(text=f"Expected visits in the next {horizon}s ({ro['n']} rollouts)",
                                                      font_size=13)), width="stretch")
        rep = ROOT / "forecast" / "report.json"
        if rep.exists():
            with st.expander("How good are these forecasts? (held-out farm stations)"):
                R = json.loads(rep.read_text())
                st.markdown(
                    f"- **Next flower**: top-3 accuracy **{R['next_flower']['model']['top3']:.0%}** vs "
                    f"{R['next_flower']['nearest_other_flower']['top3']:.0%} for 'nearest flower' and "
                    f"{R['next_flower']['random']['top3']:.0%} random ({R['next_flower']['model']['decisions']} decisions).\n"
                    f"- **2 s path, moving insects**: median-of-neighbours error {R['path']['moving']['knn']['fde_2s']} units "
                    f"vs {R['path']['moving']['stay']['fde_2s']} for 'stays put' and {R['path']['moving']['const_vel']['fde_2s']} "
                    f"for constant velocity. The closest of the 25 sampled futures lands within "
                    f"**{R['path']['knn_fan_minFDE_2s_moving']} units** (median): the future is multi-modal, so the fan matters "
                    f"more than the single line.\n"
                    f"- **Another visit?** {R['continue']['accuracy']:.0%} accuracy vs {R['continue']['majority_baseline']:.0%} "
                    f"always-guess-yes: weak signal, shown as a probability.")


# ------------------------------------------------------------------ tab: dataset
with tab_data:
    rows = []
    for cid in clips:
        d = get_doc(SRC, cid)
        m = clip_metrics(d, manifest()[cid]["duration_s"])["summary"]
        rows.append({"clip": cid, **{k: m[k] for k in ("insects", "visits", "flowers_detected", "median_dwell_s",
                                                       "median_transit_s", "median_flight_speed_cm_s",
                                                       "revisit_rate", "tracked_share")}})
    if rows:
        df = pd.DataFrame(rows)
        st.subheader(f"{src_label}: {len(df)} clips")
        c = st.columns(5)
        farm = df[df["clip"].str.startswith("sm")]
        c[0].metric("Total visits", int(df.visits.sum()))
        c[1].metric("Median dwell (farm)", f"{farm.median_dwell_s.median():.1f} s" if farm.median_dwell_s.notna().any() else "–")
        c[2].metric("Median transit (farm)", f"{farm.median_transit_s.median():.1f} s" if farm.median_transit_s.notna().any() else "–")
        c[3].metric("Flight speed (all)", f"{df.median_flight_speed_cm_s.median():.0f} cm/s" if df.median_flight_speed_cm_s.notna().any() else "–")
        c[4].metric("Tracked share", f"{df.tracked_share.median():.0%}" if df.tracked_share.notna().any() else "–")
        st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Tracking accuracy vs reference tracker")
    reps = {"Mk1.5 pipeline": "eval/report.json", "Motion only (no API)": "eval/report_heuristic.json",
            "Mk1.5 whole-video tracking (prompt v1, 1 clip)": "eval/report_mk15_v1.json"}
    arows = []
    for name, p in reps.items():
        if (ROOT / p).exists():
            R = json.loads((ROOT / p).read_text())
            if R.get("pred_dir") in ("tracks", "tracks/heuristic") or "mk15" in p:
                arows.append({"method": name, "clips": R["clips"], "coverage": R["track_coverage"],
                              "median error (units)": R["median_center_error"], "false alarms": R["false_alarm_rate"],
                              "visit recall": R["visit_recall"], "visit precision": R["visit_precision"]})
    if arows:
        st.dataframe(pd.DataFrame(arows), hide_index=True, width="stretch")
        st.caption("Coverage: share of reference points with a prediction within 50 units at the same time. "
                   "False alarms: predicted points not within 50 units of the reference. Visits scored on farm clips only.")


# ------------------------------------------------------------------ tab: upload
with tab_upload:
    from perception.pipeline import analyze_video, api_ready

    st.markdown("Upload a short video from a **fixed camera** (tripod or mounted) of insects on flowers. "
                "BeeBot tracks every insect, checks it with Mk1.5, boxes the flowers, and computes the analytics "
                "and forecasts above. First 60 s are used.")
    use_api = st.toggle("Use Mk1.5 (verification + flower boxes)", value=api_ready(), disabled=not api_ready(),
                        help=None if api_ready() else "Set PERCEPTRON_API_KEY in .env to enable")
    up = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
    if up is not None and st.button("Analyze", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
            tmp.write(up.getbuffer())
        with st.spinner("Tracking, verifying, finding flowers…"):
            cid, mp4, udoc = analyze_video(Path(tmp.name), use_api=use_api)
            out = ROOT / "app" / "cache" / "uploads" / f"{cid}.mp4"
            render(mp4, udoc, out)
        st.session_state["upload"] = (cid, str(mp4), str(out))
    if st.session_state.get("upload"):
        cid, mp4, out = st.session_state["upload"]
        udoc = json.loads((ROOT / "tracks" / "uploads" / f"{cid}.json").read_text())
        dur = cv2.VideoCapture(mp4).get(cv2.CAP_PROP_FRAME_COUNT) / 30.0
        um = clip_metrics(udoc, dur)["summary"]
        st.video(out)
        c = st.columns(5)
        c[0].metric("Insects", um["insects"])
        c[1].metric("Visits", um["visits"])
        c[2].metric("Flowers", um["flowers_detected"])
        c[3].metric("Median dwell", f"{um['median_dwell_s']} s" if um["median_dwell_s"] is not None else "–")
        c[4].metric("Flight speed", f"{um['median_flight_speed_units_s']} u/s" if um["median_flight_speed_units_s"] else "–")
        if udoc["tracks"]:
            from forecast import predict as fc
            if fc.available():
                tr = udoc["tracks"][0]
                res = fc.forecast(udoc, tr["track_id"], tr["points"][-1]["t"])
                nxt = ", ".join("{} ({:.0%})".format(n["flower_id"], n["p"]) for n in res.get("next_flower", [])[:3])
                st.markdown(f"**Forecast for {tr['track_id']} at the end of the video:** "
                            f"next flower {nxt or '–'}; "
                            f"probability of another visit {res.get('p_another_visit', 0):.0%}.")
        st.download_button("Download tracks JSON", json.dumps(udoc, indent=1), file_name=f"{cid}.json")
