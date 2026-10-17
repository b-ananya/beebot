"""Numbers and charts from one tracks file. Pure functions so the app and the deck share them."""
import plotly.express as px
import plotly.graph_objects as go


def segments(track, max_gap=0.5):
    """Merge observations into visible stretches; a gap longer than max_gap splits them."""
    segs = []
    for p in track["points"]:
        if segs and p["t"] - segs[-1][1] <= max_gap:
            segs[-1][1] = p["t"]
        else:
            segs.append([p["t"], p["t"]])
    return segs


def visible_seconds(doc):
    return sum(b - a for tr in doc.get("tracks", []) for a, b in segments(tr))


def summary(doc, duration):
    events = doc.get("events", [])
    return {
        "bees tracked": len(doc.get("tracks", [])),
        "flowers": len(doc.get("flowers", [])),
        "flower visits": len(events),
        "dwell on flowers (s)": round(sum(e["t_end"] - e["t_start"] for e in events), 1),
        "bee visible (%)": round(100 * visible_seconds(doc) / duration) if duration else 0,
    }


def visits_per_flower(doc):
    counts, dwell = {}, {}
    for e in doc.get("events", []):
        counts[e["flower_id"]] = counts.get(e["flower_id"], 0) + 1
        dwell[e["flower_id"]] = dwell.get(e["flower_id"], 0) + e["t_end"] - e["t_start"]
    flowers = [f["flower_id"] for f in doc.get("flowers", [])] or sorted(counts)
    fig = go.Figure([go.Bar(name="visits", x=flowers, y=[counts.get(f, 0) for f in flowers]),
                     go.Bar(name="dwell (s)", x=flowers, y=[round(dwell.get(f, 0), 1) for f in flowers])])
    fig.update_layout(barmode="group", height=260, margin=dict(l=10, r=10, t=30, b=10), title="Visits per flower")
    return fig


def heatmap(doc):
    xs = [(p.get("center") or [(p["box"][0] + p["box"][2]) / 2])[0] for tr in doc["tracks"] for p in tr["points"]]
    ys = [(p.get("center") or [0, (p["box"][1] + p["box"][3]) / 2])[1] for tr in doc["tracks"] for p in tr["points"]]
    fig = px.density_heatmap(x=xs, y=ys, nbinsx=24, nbinsy=14, range_x=[0, 1000], range_y=[1000, 0],
                             labels={"x": "x (0-1000)", "y": "y (0-1000)"}, title="Where the bee flew")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
    return fig


def timeline(doc, duration):
    rows = [(tr["track_id"] + " visible", a, b) for tr in doc.get("tracks", []) for a, b in segments(tr)]
    rows += [(f"visit {e['flower_id']}", e["t_start"], e["t_end"]) for e in doc.get("events", [])]
    fig = go.Figure()
    for name, a, b in rows:
        fig.add_trace(go.Bar(y=[name], x=[max(b - a, 0.05)], base=[a], orientation="h", showlegend=False,
                             marker_color="#e0a800" if "visible" in name else "#c040c0"))
    fig.update_layout(height=60 + 28 * len({r[0] for r in rows}), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(range=[0, duration], title="seconds"), title="Timeline")
    return fig
