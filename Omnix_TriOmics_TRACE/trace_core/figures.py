"""figures.py - interactive Plotly figures for the TRACE app (download as PNG/SVG from each figure)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

C_UP, C_DOWN, C_GREY = "#C0392B", "#2C6FB7", "#9AA5A2"
TEAL, ORANGE = "#00695C", "#E65100"
FONT = dict(family="Arial, Helvetica, sans-serif", size=13, color="#1F2A28")
PATTERN_COLORS = {
    "Throughput-consistent (up)": "#C0392B", "Throughput-consistent (down)": "#2C6FB7",
    "Candidate bottleneck": "#E65100", "Constrained step": "#8E44AD", "Substrate-limited": "#B9770E",
    "Non-abundance regulation (lower flux)": "#00897B", "Non-abundance regulation (higher flux)": "#26A69A",
    "Compensated": "#5D6D7E", "Product drain": "#D35400", "Metabolite-only change": "#7FB3D5",
    "Enzyme-only change": "#F1948A", "No clear pattern": "#D5DBDB",
}
CLASS_COLORS = {"Coupled": "#00695C", "Opposing": "#8E44AD", "Protein-only (post-transcriptional)": "#E65100",
                "mRNA-only (buffered)": "#2C6FB7", "Unchanged": "#D0D5D4"}


def _repel_annotations(fig, x, y, labels, all_x, all_y, xr, yr, W_px=1000, H_px=560, margins=(70, 260, 70, 60),
                       fs=11):
    """Collision-free labels for a Plotly scatter: positions computed in inches on the expected canvas,
    drawn as annotations with leader arrows (no arrow head) when a label had to move away."""
    from .layout import place_labels, text_size
    l, r, t, b = margins
    W = max(1.0, (W_px - l - r) / 96)
    H = max(1.0, (H_px - t - b) / 96)
    tx = lambda v: (np.asarray(v, float) - xr[0]) / (xr[1] - xr[0]) * W
    ty = lambda v: (np.asarray(v, float) - yr[0]) / (yr[1] - yr[0]) * H
    A = np.c_[tx(x), ty(y)]
    O = np.c_[tx(all_x), ty(all_y)]
    O = O[np.isfinite(O).all(1)]
    pt = fs * 0.75
    sizes = np.array([text_size(s_, pt) for s_ in labels]) if len(labels) else np.zeros((0, 2))
    C, placed, lead = place_labels(A, sizes, O, np.full(len(O), 0.045), np.zeros((0, 2)), W, H,
                                   anchor_r=np.full(len(A), 0.05), margin=0.0)
    for i, lab in enumerate(labels):
        if not placed[i]:
            continue
        cx = xr[0] + C[i, 0] / W * (xr[1] - xr[0])
        cy = yr[0] + C[i, 1] / H * (yr[1] - yr[0])
        if lead[i]:
            fig.add_annotation(x=x[i], y=y[i], ax=cx, ay=cy, axref="x", ayref="y", text=lab, showarrow=True,
                               arrowhead=0, arrowwidth=0.8, arrowcolor="#7F8C8D", font=dict(size=fs),
                               bgcolor="rgba(255,255,255,0.8)", standoff=4)
        else:
            fig.add_annotation(x=cx, y=cy, text=lab, showarrow=False, font=dict(size=fs),
                               bgcolor="rgba(255,255,255,0.8)")


def _layout(fig, title, h=520, **kw):
    """Shared layout: responsive width (autosize), axes that grow their margins to fit their labels, and long
    titles wrapped onto two lines so nothing spills over on half-width windows."""
    kw.setdefault("margin", dict(l=60, r=30, t=60, b=55))
    if title and len(title) > 48 and "<br>" not in title:
        cut = title.rfind(":", 0, 60)
        cut = cut + 1 if cut > 10 else title.rfind(" ", 0, 50)
        title = title[:cut].strip() + "<br><sup>" + title[cut:].strip() + "</sup>"
    fig.update_layout(title=dict(text=title, x=0.01, y=1, yref="container", yanchor="top", pad=dict(t=12),
                                 font=dict(size=16)), font=FONT, height=h, autosize=True,
                      template="simple_white", **kw)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def concordance(conc: pd.DataFrame, beta: float, label_top: int = 12):
    fig = go.Figure()
    for cls, col in CLASS_COLORS.items():
        d = conc[conc["class"] == cls]
        if d.empty:
            continue
        fig.add_trace(go.Scattergl(x=d["z_mRNA"], y=d["z_protein"], mode="markers", name=f"{cls} ({len(d)})",
                                   marker=dict(color=col, size=6 if cls == "Unchanged" else 8,
                                               opacity=0.45 if cls == "Unchanged" else 0.9,
                                               line=dict(width=0.4, color="white")),
                                   text=d["gene"], hovertemplate="%{text}<br>z mRNA %{x:.2f}<br>z protein %{y:.2f}"))
    lim = float(np.nanmax(np.abs(conc[["z_mRNA", "z_protein"]].values))) * 1.05 if len(conc) else 5
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-beta * lim, beta * lim], mode="lines", name=f"Deming fit (slope {beta:.2f})",
                             line=dict(color="#1F2A28", dash="dash", width=1.2)))
    top = conc[conc["class"] != "Unchanged"].nsmallest(label_top, "p_discordance")
    _repel_annotations(fig, top["z_mRNA"].values, top["z_protein"].values, list(top["gene"]),
                       conc["z_mRNA"].values, conc["z_protein"].values, (-lim, lim), (-lim, lim), 600, 560,
                       (60, 30, 60, 150))
    fig.add_hline(y=0, line_color="#BBB", line_width=0.8)
    fig.add_vline(x=0, line_color="#BBB", line_width=0.8)
    fig.update_xaxes(title="mRNA z-score", range=[-lim, lim])
    fig.update_yaxes(title="Protein z-score", range=[-lim, lim])
    return _layout(fig, "Gene-protein concordance", 560, legend=dict(orientation="h", y=-0.18))


def pattern_grid(rx: pd.DataFrame, n_label: int = 10):
    d = rx.copy()
    q = pd.to_numeric(d["q_label"], errors="coerce").fillna(1.0).clip(1e-12, 1.0)
    d["size"] = (6 + 3 * np.clip(-np.log10(q), 0, 6)).astype(float)
    d = d[np.isfinite(d["E"].astype(float)) & np.isfinite(d["M"].astype(float))]
    fig = go.Figure()
    for pat, col in PATTERN_COLORS.items():
        s = d[d["pattern"] == pat]
        if s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s["E"], y=s["M"], mode="markers", name=f"{pat} ({len(s)})",
            marker=dict(color=col, size=s["size"], opacity=0.85, line=dict(width=0.5, color="white")),
            text=s["name"], customdata=np.c_[s["enzymes"], s["q_label"], s["q_topology"]],
            hovertemplate="%{text}<br>Enzymes: %{customdata[0]}<br>E %{x:.2f}  M %{y:.2f}"
                          "<br>q label %{customdata[1]:.2g}, q topology %{customdata[2]:.2g}<extra></extra>"))
    sig = d[d["significant"]].copy()
    sig["lab"] = [str(e).split(",")[0] for e in sig["enzymes"]]
    sig = sig.drop_duplicates("lab").head(n_label)
    xr = (float(np.nanmin(d["E"])) - 0.6, float(np.nanmax(d["E"])) + 0.6)
    yr = (float(np.nanmin(d["M"])) - 0.6, float(np.nanmax(d["M"])) + 0.6)
    _repel_annotations(fig, sig["E"].values, sig["M"].values, list(sig["lab"]), d["E"].values, d["M"].values,
                       xr, yr, 600, 760, (60, 20, 60, 190))   # placed for a half-screen width: stays clear when wider
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    fig.add_vline(x=0, line_color="#999", line_width=0.8)
    for x, y, t in ((0.98, 0.98, "coherent: enzyme up,<br>products up / substrates down"),
                    (0.02, 0.02, "coherent: enzyme down,<br>substrates accumulate"),
                    (0.98, 0.02, "enzyme up, balance down"), (0.02, 0.98, "enzyme down, balance up")):
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=t, showarrow=False,
                           xanchor="right" if x > 0.5 else "left", yanchor="top" if y > 0.5 else "bottom",
                           font=dict(size=10, color="#6B7775"))
    fig.update_xaxes(title="Enzyme evidence E (z)", range=list(xr))
    fig.update_yaxes(title="Metabolite balance M = (products - substrates)/sqrt(2)", range=list(yr))
    return _layout(fig, "Reaction pattern map (each dot = one reaction)", 760,
                   legend=dict(font=dict(size=11), orientation="h", x=0, y=-0.12, yanchor="top"),
                   margin=dict(l=60, r=20, t=60, b=40))


CLASS_PATHWAY_COLORS = {"Shared-concordant": "#00695C", "Shared-discordant": "#8E44AD", "mRNA-specific": "#2C6FB7",
                        "Protein-specific": "#E65100", "Metabolite-specific": "#B9770E", "Not significant": "#D5DBDB",
                        "Not assessable": "#F4F6F6"}


def _wrap(text: str, width: int = 34, lines: int = 2) -> str:
    """Word-wrap a label into at most `lines` lines (Plotly <br>), ellipsis if it still does not fit."""
    words, out, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = f"{cur} {w}" if cur else w
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    out = [o for o in out if o]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: width - 1].rstrip() + "…"
    return "<br>".join(o if len(o) <= width else o[: width - 1] + "…" for o in out)


def pathway_dotplot(pw: pd.DataFrame, top: int = 25):
    """Dot plot of pathway activity per layer. Dots are drawn as circles in data units on a 1:1 grid (radius at
    most 0.42 of a cell), so they can never touch whatever the window size: the whole plot, gaps included,
    scales with the browser width. Class is a coloured tile in its own column with a legend."""
    from plotly.colors import sample_colorscale
    d = pw.dropna(subset=["z_combined"]).head(top).iloc[::-1].reset_index(drop=True)
    cols = [c for c in ("mRNA", "protein", "metabolite") if f"z_{c}" in d.columns] + ["combined", "coherence"]
    ncol = len(cols)
    hx, hy, htxt, shapes = [], [], [], []
    for i, r in d.iterrows():
        for j, c in enumerate(cols):
            if c == "coherence":
                z, q, n = r["coherence"], r["q_coherence"], r["n_reactions_observed"]
                zc = np.clip(z * 2, -5, 5) if np.isfinite(z) else np.nan
            elif c == "combined":
                z, q, n = r["z_combined"], r["q_combined"], r.get("layers_assessable", np.nan)
                zc = np.clip(z, -5, 5)
            else:
                z, q, n = r[f"z_{c}"], r[f"q_{c}"], r[f"n_{c}"]
                zc = np.clip(z, -5, 5) if np.isfinite(z) else np.nan
            if not np.isfinite(z):
                continue
            sig = min(1.0, -np.log10(max(q, 1e-12)) / 6) if np.isfinite(q) else 0.0
            rad = 0.13 + 0.29 * sig                      # <= 0.42: at least 0.16 cell of clear space
            col = sample_colorscale("RdBu_r", [(zc + 5) / 10])[0]
            shapes.append(dict(type="circle", xref="x", yref="y", x0=j - rad, x1=j + rad, y0=i - rad, y1=i + rad,
                               fillcolor=col, line=dict(color="#555", width=0.7), layer="above"))
            hx.append(j)
            hy.append(i)
            htxt.append(f"<b>{r['pathway']}</b><br>{c}: z = {z:.2f}<br>q = {q:.2g}<br>n = {n}<br>{r['class']}")
        cc = CLASS_PATHWAY_COLORS.get(r["class"], "#BBBBBB")
        shapes.append(dict(type="rect", xref="x", yref="y", x0=ncol + 0.15 - 0.33, x1=ncol + 0.15 + 0.33,
                           y0=i - 0.33, y1=i + 0.33, fillcolor=cc, line=dict(color="#666", width=0.6), layer="above"))
        hx.append(ncol + 0.15)
        hy.append(i)
        htxt.append(f"<b>{r['pathway']}</b><br>class: {r['class']}")
    shapes.append(dict(type="line", xref="x", yref="paper", x0=ncol - 0.42, x1=ncol - 0.42, y0=0, y1=1,
                       line=dict(color="#C9D3D1", width=1, dash="dot")))
    fig = go.Figure(go.Scatter(x=hx, y=hy, mode="markers", text=htxt, hovertemplate="%{text}<extra></extra>",
                               marker=dict(size=18, opacity=0), showlegend=False))
    # colour bar drawn on the same grid (right of the class column), so it scales and never overlaps
    nb = 24
    span = max(2.0, min(4.0, len(d) - 1.0))          # short colour scale (about 4 rows)
    y_top = len(d) - 1 + 0.3
    y_bot = y_top - span
    xb0, xb1 = ncol + 0.95, ncol + 1.15
    for k in range(nb):
        t0 = k / nb
        shapes.append(dict(type="rect", xref="x", yref="y", x0=xb0, x1=xb1, y0=y_bot + span * t0,
                           y1=y_bot + span * (t0 + 1 / nb) + 1e-3, line=dict(width=0),
                           fillcolor=sample_colorscale("RdBu_r", [(k + 0.5) / nb])[0], layer="above"))
    shapes.append(dict(type="rect", xref="x", yref="y", x0=xb0, x1=xb1, y0=y_bot, y1=y_top,
                       fillcolor="rgba(0,0,0,0)", line=dict(color="#888", width=0.6), layer="above"))
    ann = [dict(x=xb1 + 0.08, y=y_bot + span * (v + 5) / 10, xref="x", yref="y", text=f"{v:+d}" if v else "0",
                showarrow=False, xanchor="left", font=dict(size=10)) for v in (-5, 0, 5)]
    ann.append(dict(x=(xb0 + xb1) / 2, y=y_top + 0.15, xref="x", yref="y", text="z", showarrow=False,
                    yanchor="bottom", font=dict(size=11)))
    for cl in [c for c in CLASS_PATHWAY_COLORS if c in set(d["class"])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=cl,
                                 marker=dict(symbol="square", size=12, color=CLASS_PATHWAY_COLORS[cl],
                                             line=dict(color="#666", width=0.6))))
    for sz_q, lab in ((0.05, "q = 0.05"), (1e-3, "q = 0.001"), (1e-6, "q ≤ 1e-6")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"size: {lab}",
                                 marker=dict(size=4 + 22 * (0.13 + 0.29 * min(1, -np.log10(sz_q) / 6)) / 0.42 * 0.5,
                                             color="#BBBBBB", line=dict(color="#555", width=0.7))))
    cell = 34                                            # px per row (and per column, 1:1)
    fig.update_xaxes(tickvals=list(range(ncol)) + [ncol + 0.15], ticktext=cols + ["class"], side="top",
                     tickangle=-45, range=[-0.6, ncol + 1.75], showgrid=False, zeroline=False, ticks="",
                     showline=False, constrain="domain", constraintoward="center", automargin=True, fixedrange=True)
    fig.update_yaxes(tickvals=list(range(len(d))), ticktext=[_wrap(p) for p in d["pathway"]], range=[-0.6, len(d) - 0.4],
                     scaleanchor="x", scaleratio=1, constrain="domain", showgrid=True, gridcolor="#F0F2F2",
                     zeroline=False, ticks="", showline=False, automargin=True, fixedrange=True,
                     tickfont=dict(size=11))
    for sh in shapes:
        sh.setdefault("opacity", 1)
    fig.update_layout(shapes=shapes, annotations=ann, hovermode="closest",
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top", yref="paper",
                                  font=dict(size=11), itemsizing="trace", entrywidth=0, traceorder="normal"),
                      )
    fig = _layout(fig, "Pathway activity across layers", cell * len(d) + 300,
                  margin=dict(l=10, r=10, t=115, b=40))
    fig.update_layout(title_x=0.5, title_xanchor="center")
    return fig


def cascade_sankey(rx: pd.DataFrame, title: str = "Cascade"):
    """Gene -> reaction -> metabolite ribbons for a set of reactions (rows of the reaction table)."""
    labels, idx, src, dst, val, col = [], {}, [], [], [], []

    def node(key, lab):
        if key not in idx:
            idx[key] = len(labels)
            labels.append(lab)
        return idx[key]

    rgba = lambda v: "rgba(192,57,43,0.55)" if v > 0 else "rgba(44,111,183,0.55)"
    for _, r in rx.iterrows():
        rn = node("r:" + r["reaction"], (r["name"][:40] + "...") if len(r["name"]) > 43 else r["name"])
        for g in [x.strip() for x in str(r["enzymes"]).split(",") if x.strip()][:6]:
            src.append(node("g:" + g, g))
            dst.append(rn)
            val.append(max(abs(r["E"]), 0.3))
            col.append(rgba(r["E"]))
        for m in [x.strip() for x in str(r["products"]).split(";") if x.strip()][:4]:
            src.append(rn)
            dst.append(node("m:" + m, m))
            val.append(max(abs(r["P"]) if np.isfinite(r["P"]) else 0.3, 0.3))
            col.append(rgba(r["P"]) if np.isfinite(r["P"]) else "rgba(150,150,150,0.4)")
        for m in [x.strip() for x in str(r["substrates"]).split(";") if x.strip()][:4]:
            src.append(rn)
            dst.append(node("m:" + m, m + " (substrate)"))
            val.append(max(abs(r["S"]) if np.isfinite(r["S"]) else 0.3, 0.3))
            col.append(rgba(r["S"]).replace("0.55", "0.3") if np.isfinite(r["S"]) else "rgba(150,150,150,0.3)")
    fig = go.Figure(go.Sankey(arrangement="snap", textfont=dict(size=10),
                              node=dict(label=labels, pad=16, thickness=14,
                                                             color="#E3F2EF", line=dict(color=TEAL, width=0.6)),
                              link=dict(source=src, target=dst, value=val, color=col)))
    return _layout(fig, title + "  (enzyme -> reaction -> metabolite; red = up, blue = down)",
                   max(460, 30 * len(labels) // 2 + 220))


def reaction_map(rx: pd.DataFrame, met_z: dict, title: str, seed: int = 1):
    """Bipartite reaction-metabolite map of one pathway; squares = reactions (colour = coherence),
    circles = metabolites (colour = z), enzyme names under each reaction."""
    import networkx as nx
    G = nx.Graph()
    for _, r in rx.iterrows():
        R = "r:" + r["reaction"]
        G.add_node(R, kind="r")
        for m in [x.strip() for x in str(r["substrates"]).split(";") if x.strip()] + \
                 [x.strip() for x in str(r["products"]).split(";") if x.strip()]:
            G.add_node("m:" + m, kind="m")
            G.add_edge(R, "m:" + m)
    if G.number_of_nodes() == 0:
        return _layout(go.Figure(), title + " (no observable reactions)", 300)
    pos = nx.kamada_kawai_layout(G) if G.number_of_nodes() < 200 else nx.spring_layout(G, seed=seed)
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#B8C2C0", width=1), hoverinfo="skip",
                               showlegend=False))
    rmap = rx.set_index("reaction")
    rn = [n for n in G if n.startswith("r:")]
    cz = [float(rmap.loc[n[2:], "C_std"]) if np.isfinite(rmap.loc[n[2:], "C_std"]) else 0 for n in rn]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in rn], y=[pos[n][1] for n in rn], mode="markers+text", name="reactions",
        marker=dict(symbol="square", size=14, color=cz, colorscale="PuOr_r", cmin=-4, cmax=4,
                    line=dict(color="#333", width=0.8), colorbar=dict(title="reaction<br>coherence z", x=1.02, y=0.77, len=0.42, thickness=10)),
        text=[str(rmap.loc[n[2:], "enzymes"]).split(",")[0] for n in rn], textposition="bottom center",
        textfont=dict(size=10, color="#333"),
        hovertext=[f"{rmap.loc[n[2:], 'name']}<br>{rmap.loc[n[2:], 'enzymes']}<br>{rmap.loc[n[2:], 'pattern']}"
                   for n in rn], hoverinfo="text"))
    mn = [n for n in G if n.startswith("m:")]
    mz = [met_z.get(n[2:], np.nan) for n in mn]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in mn], y=[pos[n][1] for n in mn], mode="markers+text", name="metabolites",
        marker=dict(size=16, color=[z if np.isfinite(z) else 0 for z in mz], colorscale="RdBu_r", cmin=-4,
                    cmax=4, line=dict(color=["#333" if np.isfinite(z) else "#CCC" for z in mz], width=1),
                    colorbar=dict(title="metabolite z", x=1.02, y=0.23, len=0.42, thickness=10)),
        text=[n[2:] for n in mn], textposition="top center", textfont=dict(size=10),
        hovertext=[f"{n[2:]}<br>z = {z:.2f}" if np.isfinite(z) else f"{n[2:]} (not measured)" for n, z in zip(mn, mz)],
        hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, 680, showlegend=False, margin=dict(l=20, r=120, t=60, b=20))


def network_plot(nodes: pd.DataFrame, edges: list, title: str, seed: int = 2):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes["node"])
    G.add_edges_from([(a, b) for a, b, _ in edges if a in G and b in G])
    try:
        pos = nx.forceatlas2_layout(G, max_iter=300, gravity=2.0, strong_gravity=True, seed=seed)
    except Exception:
        pos = nx.spring_layout(G, seed=seed, k=0.6 / np.sqrt(max(G.number_of_nodes(), 1)))
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#C9D1CF", width=0.8), hoverinfo="skip",
                               showlegend=False))
    sym = {"enzyme": "diamond", "reaction": "square", "metabolite": "circle"}
    for t, s in sym.items():
        d = nodes[nodes["type"] == t]
        if d.empty:
            continue
        z = d["z"].fillna(0).clip(-5, 5)
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in d["node"]], y=[pos[n][1] for n in d["node"]], mode="markers+text" if t != "reaction" else "markers",
            name=t, marker=dict(symbol=s, size=11 if t != "reaction" else 8, color=z, colorscale="RdBu_r", cmin=-4,
                                cmax=4, line=dict(width=0.6, color="#333")),
            text=[str(x)[:18] for x in d["label"]] if t != "reaction" else None, textposition="top center",
            textfont=dict(size=9), hovertext=[f"{t}: {l}<br>z = {v:.2f}" for l, v in zip(d["label"], d["z"].fillna(0))],
            hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title + "  (diamond = enzyme, square = reaction, circle = metabolite)", 700,
                   legend=dict(orientation="h", y=-0.05))


def variance_heatmap(r2: pd.DataFrame):
    fig = go.Figure(go.Heatmap(z=r2.values, x=r2.columns, y=r2.index, colorscale="Teal", zmin=0,
                               text=np.round(r2.values, 1), texttemplate="%{text}",
                               colorbar=dict(title="% variance")))
    return _layout(fig, "Variance explained per factor and layer (MOFA+)", 320)


def factor_scatter(scores: pd.DataFrame, groups: pd.Series, fx: str, fy: str):
    fig = go.Figure()
    for g, col in zip(sorted(groups.dropna().unique()), [TEAL, ORANGE, "#2C6FB7", "#8E44AD"]):
        s = scores[groups.reindex(scores.index) == g]
        fig.add_trace(go.Scatter(x=s[fx], y=s[fy], mode="markers", name=g, text=s.index,
                                 marker=dict(size=10, color=col, line=dict(width=0.6, color="white"))))
    fig.update_xaxes(title=fx)
    fig.update_yaxes(title=fy)
    return _layout(fig, "Sample scores", 460)


def roc_plot(sig: dict):
    fpr, tpr = sig["roc"]
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=TEAL, width=3),
                               name=f"Integrated (AUC {sig['auc_mean']:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="chance"))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _layout(fig, "Out-of-fold ROC (nested CV, mean prediction over repeats)", 440,
                   legend=dict(x=0.45, y=0.08))


def perm_hist(sig: dict):
    fig = go.Figure(go.Histogram(x=sig["perm_aucs"], nbinsx=20, marker_color="#B8C2C0", name="permuted labels"))
    fig.add_vline(x=sig["auc_mean"], line_color=ORANGE, line_width=3,
                  annotation_text=f"observed {sig['auc_mean']:.2f} (p = {sig['p_perm']:.3f})")
    fig.update_xaxes(title="Cross-validated AUC")
    return _layout(fig, "Label-permutation null", 340, showlegend=False)


def panel_bars(panel: pd.DataFrame):
    d = panel.sort_values("selection_frequency")
    cmap = {"mRNA": "#2C6FB7", "protein": TEAL, "metabolite": ORANGE}
    fig = go.Figure(go.Bar(x=d["selection_frequency"], y=[f"{a} ({b})" for a, b in zip(d["label"], d["layer"])],
                           orientation="h", marker_color=[cmap.get(l, "#888") for l in d["layer"]]))
    fig.add_vline(x=0.7, line_dash="dot", line_color="#555")
    fig.update_xaxes(title="Selection frequency across outer folds", range=[0, 1.02])
    return _layout(fig, "Stability-selected integrated panel", max(320, 24 * len(d) + 120))


def targets_bar(tg: pd.DataFrame, top: int = 20):
    d = tg.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=np.nan_to_num(-np.log10(d["rra_score"].astype(float).clip(1e-30, 1.0))), y=d["gene"], orientation="h",
                           marker_color=[ORANGE if c else TEAL for c in d["choke_point"]],
                           text=[e if len(e) <= 34 else e[:33] + "…" for e in d["evidence"].astype(str)],
                           customdata=d["evidence"], textposition="inside", insidetextanchor="start",
                           textfont=dict(color="white", size=11), cliponaxis=True,
                           hovertemplate="%{y}: %{customdata}<extra></extra>"))
    fig.update_xaxes(title="-log10 rank-aggregation score")
    return _layout(fig, "Prioritised targets (orange = choke point)", max(380, 24 * len(d) + 120))


def coverage_bars(layers: dict):
    rows = []
    for k, L in layers.items():
        if k == "met":
            mapped = (L.feat["map_rule"] != "unmapped").sum()
        else:
            mapped = (L.feat["ensg"] != "").sum()
        rows.append((L.name, len(L.feat), mapped))
    d = pd.DataFrame(rows, columns=["layer", "features", "mapped"])
    fig = go.Figure([go.Bar(x=d["layer"], y=d["features"], name="features", marker_color="#C9D1CF"),
                     go.Bar(x=d["layer"], y=d["mapped"], name="mapped to metabolic KB", marker_color=TEAL)])
    fig.update_layout(barmode="overlay")
    return _layout(fig, "Identifier mapping coverage", 360)


def config(fmt: str = "png", scale: int = 6, name: str = "trace_figure") -> dict:
    """Plotly figures are drawn as vectors on screen; the camera button saves PNG at `scale` x (6 x by default,
    i.e. ~600 DPI for a typical figure) or SVG."""
    return {"displaylogo": False, "responsive": True,
            "toImageButtonOptions": {"format": fmt, "scale": scale, "filename": name}}


def canonical_bars(cp: pd.DataFrame, top: int = 25, thr: float = 0.05):
    """IPA-style canonical pathway chart: bars = -log10 p coloured by directional z, line = ratio."""
    from .ipa import z_color
    d = cp.head(top).iloc[::-1]
    lab = [_wrap(p, 40) for p in d["pathway"]]
    row = 40 if any("<br>" in x for x in lab) else 26
    hov = [f"<b>{p}</b><br>p = {pv:.2g}, q = {qv:.2g}<br>z = {z:.2f}<br>{k}/{n} measured significant<br>"
           f"up: {u[:120]}<br>down: {dn[:120]}" for p, pv, qv, z, k, n, u, dn in
           zip(d["pathway"], d["p"], d["q"], d["z_directional"].fillna(np.nan), d["n_significant"], d["n_measured"],
               d["molecules_up"], d["molecules_down"])]
    fig = go.Figure(go.Bar(x=d["minus_log10_p"], y=lab, orientation="h", marker=dict(
        color=[z_color(z) for z in d["z_directional"]], line=dict(color="#555", width=0.6)),
        hovertext=hov, hoverinfo="text", name="-log10 p"))
    fig.add_trace(go.Scatter(x=d["ratio"], y=lab, xaxis="x2", mode="lines+markers", name="ratio",
                             line=dict(color="#E67E22", width=1.2), marker=dict(size=5)))
    fig.add_vline(x=-np.log10(thr), line_dash="dash", line_color=ORANGE)
    fig.update_layout(xaxis2=dict(overlaying="x", side="top", title="Ratio", range=[0, max(0.05, d["ratio"].max() * 1.15)], tickformat=".2f",
                                  showgrid=False, color="#E67E22"), showlegend=False)
    fig.update_xaxes(title="-log10(p), right-tailed Fisher's exact", selector=dict(overlaying=None))
    fig.update_yaxes(automargin=True)
    return _layout(fig, "Canonical pathways (IPA-style): orange = increased, blue = decreased, grey = no pattern",
                   max(420, row * len(d) + 150), margin=dict(l=20, r=30, t=95, b=55))


def layer_bars(pw: pd.DataFrame, top: int = 6):
    """Overview: per-layer pathway activity z (permutation-standardised Stouffer) for the top pathways by the
    combined test; grouped bars per layer, dashed lines at |z| = 2."""
    d = pw.dropna(subset=["z_combined"]).head(top)
    colors = {"mRNA": "#6366F1", "protein": "#10B981", "metabolite": "#F59E0B"}
    names = {"mRNA": "Transcriptomics (z)", "protein": "Proteomics (z)", "metabolite": "Metabolomics (z)"}
    fig = go.Figure()
    xl = [p if len(p) <= 26 else p[:25].rstrip() + "…" for p in d["pathway"]]   # full name on hover
    for k in ("mRNA", "protein", "metabolite"):
        if f"z_{k}" not in d.columns:
            continue
        fig.add_trace(go.Bar(x=xl, y=d[f"z_{k}"], name=names[k], marker=dict(color=colors[k], cornerradius=4),
                             customdata=np.c_[d["pathway"], d[f"q_{k}"], d[f"n_{k}"]],
                             hovertemplate="<b>%{customdata[0]}</b><br>" + names[k] +
                                           " = %{y:.2f}<br>q = %{customdata[1]:.2g}<br>n = %{customdata[2]}<extra></extra>"))
    for yv in (2, -2):
        fig.add_hline(y=yv, line=dict(color="#475569", dash="dash", width=1))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06,
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, yanchor="bottom"))
    fig.update_yaxes(title="pathway activity z", zeroline=True)
    fig.update_xaxes(tickfont=dict(size=11), tickangle=-30, automargin=True)  # angled single lines never collide
    return _layout(fig, "", 430, margin=dict(l=55, r=15, t=40, b=20))


def reaction_donut(rx: pd.DataFrame):
    """Overview: classification of all observable reactions."""
    sig = rx["significant"]
    md = rx["metabolite_driven"] & ~sig
    pat = (rx["pattern"] != "No clear pattern") & ~sig & ~md
    rest = ~(sig | md | pat)
    labs = ["Coherent (FDR)", "Metabolite-driven", "Directional pattern (n.s.)", "No clear pattern"]
    vals = [int(sig.sum()), int(md.sum()), int(pat.sum()), int(rest.sum())]
    cols = ["#10B981", "#F59E0B", "#6366F1", "#9AAEAA"]
    fig = go.Figure(go.Pie(labels=labs, values=vals, hole=0.66, sort=False, direction="clockwise",
                           marker=dict(colors=cols, line=dict(color="white", width=2)), textinfo="none",
                           hovertemplate="%{label}: %{value} reactions (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(rx):,}</b><br><span style='font-size:12px'>reactions</span>", showarrow=False,
                       font=dict(size=22))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05, yanchor="top",
                                                   font=dict(size=11)))
    return _layout(fig, "", 430, margin=dict(l=10, r=10, t=10, b=70))
