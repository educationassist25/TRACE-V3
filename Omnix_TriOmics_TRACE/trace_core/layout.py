"""
layout.py - graph layouts, node-overlap removal and collision-free label placement.

Everything works in physical units (inches) of the final canvas, so the same geometry is used by the
interactive (Plotly) and the publication (matplotlib) renderers and labels never sit on nodes, edges or
other labels.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------- layouts
LAYOUTS = ["Subcellular (IPA-style)", "Hierarchical flow (top-down)", "Hierarchical flow (left-right)",
           "Force-directed", "Kamada-Kawai", "Circular", "Shell by molecule type", "Grid by molecule type",
           "Spectral"]


def _norm(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    if len(P) == 0:
        return P
    lo, hi = P.min(0), P.max(0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (P - lo) / span


def force(G, seed=1, iters=400):
    import networkx as nx
    if G.number_of_nodes() <= 2:
        return {n: np.array([i, 0.5]) for i, n in enumerate(G)}
    try:
        pos = nx.forceatlas2_layout(G, max_iter=iters, gravity=1.5, strong_gravity=True, seed=seed,
                                    scaling_ratio=2.0)
    except Exception:
        pos = nx.spring_layout(G, seed=seed, iterations=iters, k=1.2 / np.sqrt(G.number_of_nodes()))
    return pos


def hierarchical(G, orient="TB", satellites: set | None = None):
    """Sugiyama-style layered layout: break cycles (DFS back edges), longest-path layering, barycentre
    ordering sweeps. Direction follows substrate -> enzyme/reaction -> product."""
    import networkx as nx
    D0 = nx.DiGraph(G) if G.is_directed() else nx.DiGraph(G.to_directed())
    satellites = {n for n in (satellites or set()) if n in D0}
    D = D0.subgraph([n for n in D0 if n not in satellites]).copy()
    if D.number_of_nodes() == 0:
        D, satellites = D0, set()
    # remove back edges to obtain a DAG
    dag = nx.DiGraph()
    dag.add_nodes_from(D)
    state = {}

    def dfs(u):
        state[u] = 1
        for v in D.successors(u):
            if state.get(v) == 1:
                continue  # back edge
            dag.add_edge(u, v)
            if v not in state:
                dfs(v)
        state[u] = 2

    import sys
    sys.setrecursionlimit(max(10000, 4 * D.number_of_nodes()))
    for n in sorted(D, key=lambda x: D.in_degree(x)):
        if n not in state:
            dfs(n)
    layer = {}
    for n in nx.topological_sort(dag):
        preds = list(dag.predecessors(n))
        layer[n] = 1 + max((layer[p] for p in preds), default=-1)
    L = max(layer.values()) + 1 if layer else 1
    rows = [[n for n in layer if layer[n] == i] for i in range(L)]
    order = {n: i for r in rows for i, n in enumerate(r)}
    U = dag.to_undirected()
    for sweep in range(8):
        rng = range(1, L) if sweep % 2 == 0 else range(L - 2, -1, -1)
        for i in rng:
            ref = i - 1 if sweep % 2 == 0 else i + 1
            def bary(n):
                nb = [order[m] for m in U.neighbors(n) if layer.get(m) == ref]
                return np.mean(nb) if nb else order[n]
            rows[i].sort(key=bary)
            for j, n in enumerate(rows[i]):
                order[n] = j
    # satellites (enzymes in the reaction view) sit beside the reaction they catalyse
    for sat in satellites:
        nb = [m for m in D0.successors(sat) if m in layer] + [m for m in D0.predecessors(sat) if m in layer]
        if nb:
            host = nb[0]
            i = layer[host]
            rows[i].insert(rows[i].index(host), sat)
            layer[sat] = i
        else:
            rows[0].append(sat)
            layer[sat] = 0
    pos = {}
    for i, r in enumerate(rows):
        k = len(r)
        for j, n in enumerate(r):
            x = (j + 0.5) / max(k, 1)
            y = 1 - (i + 0.5) / L
            pos[n] = np.array([x, y]) if orient == "TB" else np.array([1 - y, 1 - x])
    return pos


def subcellular(G, band_of: dict, bands: list, seed=1):
    """IPA-like subcellular layout: horizontal bands (top = extracellular, bottom = nucleus);
    x from a force layout, y inside each node's band."""
    fp = force(G, seed=seed)
    nodes = list(G)
    P = _norm(np.array([fp[n] for n in nodes])) if nodes else np.zeros((0, 2))
    present = [b for b in bands if any(band_of.get(n) == b for n in nodes)]
    if not present:
        present = ["Other"]
    counts = np.array([max(1, sum(band_of.get(n) == b for n in nodes)) for b in present], float)
    h = np.sqrt(counts)
    h = np.maximum(h / h.sum(), 0.10)
    h = h / h.sum()
    tops = 1 - np.concatenate([[0], np.cumsum(h)[:-1]])
    geom = {b: (t - hh, t) for b, t, hh in zip(present, tops, h)}
    pos = {}
    for b in present:
        mem = [i for i, n in enumerate(nodes) if band_of.get(n, "Other") == b]
        if not mem:
            continue
        lo, hi = geom[b]
        ys = P[mem, 1]
        ys = (ys - ys.min()) / (np.ptp(ys) or 1.0)
        for k, i in enumerate(mem):
            pos[nodes[i]] = np.array([0.10 + 0.88 * P[i, 0], lo + (0.18 + 0.64 * ys[k]) * (hi - lo)])
    for n in nodes:
        pos.setdefault(n, np.array([0.5, 0.5]))
    return pos, geom


def compute_layout(G, name, kinds: dict, band_of: dict | None = None, bands: list | None = None, seed=1,
                   focus: list | None = None, module: dict | None = None):
    import networkx as nx
    geom = None
    U = G.to_undirected() if G.is_directed() else G
    if G.number_of_nodes() == 0:
        return {}, None
    if name.startswith("Subcellular"):
        pos, geom = subcellular(U, band_of or {}, bands or ["Other"], seed)
    elif name.startswith("Hierarchical"):
        has_rx = any(k == "reaction" for k in kinds.values())
        sats = {n for n, k in kinds.items() if has_rx and k not in ("reaction", "metabolite")}
        pos = hierarchical(G, "TB" if "top-down" in name else "LR", sats)
    elif name == "Kamada-Kawai":
        pos = {}
        for i, comp in enumerate(nx.connected_components(U)):
            sub = U.subgraph(comp)
            p = nx.kamada_kawai_layout(sub) if len(comp) > 2 else {n: np.array([j, 0.0]) for j, n in enumerate(comp)}
            pos.update(p)
        pos = _pack_components(U, pos)
    elif name == "Circular":
        order = sorted(G, key=lambda n: (kinds.get(n, ""), str(n)))
        pos = {n: np.array([np.cos(2 * np.pi * i / len(order)), np.sin(2 * np.pi * i / len(order))])
               for i, n in enumerate(order)}
    elif name == "Shell by molecule type":
        groups = {}
        for n in G:
            groups.setdefault(kinds.get(n, "other"), []).append(n)
        shells = [groups[k] for k in ("metabolite", "reaction", "enzyme", "kinase", "phosphatase", "peptidase",
                                      "transporter", "other") if k in groups]
        pos = nx.shell_layout(U, nlist=shells)
    elif name == "Grid by molecule type":
        groups = {}
        for n in G:
            groups.setdefault(kinds.get(n, "other"), []).append(n)
        cols = list(groups)
        pos = {}
        for ci, k in enumerate(cols):
            mem = sorted(groups[k], key=lambda n: -U.degree(n))
            nc = int(np.ceil(np.sqrt(len(mem))))
            for j, n in enumerate(mem):
                pos[n] = np.array([ci + (j % nc) / (nc + 1), -(j // nc)])
    elif name == "Radial (focus-centred)":
        pos = radial(G, focus or [])
    elif name.startswith("Bipartite columns"):
        pos = bipartite_columns(G, kinds)
    elif name == "Circular (edge-bundled chord)":
        comm = module or {}
        pos = circular_sorted(G, lambda n: (comm.get(n, 0), kinds.get(n, ""), str(n)))
    elif name.startswith("Hive"):
        pos, _ = hive(G, kinds)
    elif name == "Arc diagram":
        comm = module or {}
        pos = arc(G, lambda n: (comm.get(n, 0), kinds.get(n, ""), str(n)))
    elif name == "Spectral":
        try:
            pos = nx.spectral_layout(U)
        except Exception:
            pos = force(U, seed)
    else:
        pos = force(U, seed)
    nodes = list(G)
    P = np.array([pos[n] for n in nodes], float)
    if name in SPECIAL or name.startswith("Radial"):
        # keep the aspect ratio of circular / radial geometry
        c = (P.max(0) + P.min(0)) / 2 if len(P) else np.zeros(2)
        span = np.abs(P - c).max() or 1.0
        P = 0.5 + 0.46 * (P - c) / span
        if name == "Arc diagram":
            P[:, 1] = 0.30
    elif not name.startswith("Subcellular"):
        P = 0.04 + 0.92 * _norm(P)
    return {n: P[i] for i, n in enumerate(nodes)}, geom


def _pack_components(U, pos):
    import networkx as nx
    comps = sorted(nx.connected_components(U), key=len, reverse=True)
    out, x0 = {}, 0.0
    for c in comps:
        P = np.array([pos[n] for n in c])
        P = _norm(P) * np.sqrt(len(c))
        for n, p in zip(c, P):
            out[n] = p + np.array([x0, 0])
        x0 += P[:, 0].max() + 1.0
    return out


# ------------------------------------------------------------------------------------------- overlap removal
def remove_overlaps(P: np.ndarray, r: np.ndarray, W: float, H: float, iters: int = 300,
                    fixed_y: np.ndarray | None = None, ybounds: np.ndarray | None = None) -> np.ndarray:
    """P in [0,1]^2, r = node radius in inches, canvas W x H inches. Pushes nodes apart until the gap
    between any two nodes is at least max(r_i, r_j) * 0.6. ybounds: per-node (lo, hi) clamp (bands)."""
    X = P.copy() * [W, H]
    n = len(X)
    if n < 2:
        return P
    X += np.random.default_rng(0).normal(scale=1e-3, size=X.shape)  # separate coincident nodes
    gap = 0.6
    for _ in range(max(iters, 600)):
        d = X[:, None, :] - X[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
        need = (r[:, None] + r[None, :]) + gap * np.maximum(r[:, None], r[None, :])
        ov = np.clip(need - dist, 0, None)
        np.fill_diagonal(ov, 0)
        if ov.max() < 1e-4:
            break
        push = (d / dist[..., None]) * (ov[..., None] / 2)
        X += push.sum(1) * 0.5
        X[:, 0] = np.clip(X[:, 0], r + 0.02, W - r - 0.02)
        if ybounds is not None:
            X[:, 1] = np.clip(X[:, 1], ybounds[:, 0] * H + r, ybounds[:, 1] * H - r)
        else:
            X[:, 1] = np.clip(X[:, 1], r + 0.02, H - r - 0.02)
    return X / [W, H]


# ------------------------------------------------------------------------------------------- labels
def text_size(text: str, fontsize_pt: float) -> tuple[float, float]:
    """Approximate rendered size (inches) of a single-line label."""
    w = 0.56 * fontsize_pt / 72 * max(1, len(text)) + 0.06
    h = 1.30 * fontsize_pt / 72 + 0.04
    return w, h


def place_labels(anchors: np.ndarray, sizes: np.ndarray, node_xy: np.ndarray, node_r: np.ndarray,
                 seg_pts: np.ndarray, W: float, H: float, order: np.ndarray | None = None,
                 hide_unplaced: bool = False, margin: float = 0.03, anchor_r: np.ndarray | None = None):
    """Greedy collision-free label placement in inches.

    anchors: k x 2 points (inches) the labels belong to; sizes: k x 2 (w, h) inches; node_xy / node_r:
    all node discs (obstacles); seg_pts: points sampled along edges (obstacles); anchor_r: radius of the
    node each label belongs to. Candidates are tried in 8 directions at growing distances; the first
    candidate with no collision wins, otherwise the least colliding one (or the label is hidden when
    hide_unplaced). Returns (centers k x 2, placed mask, needs-leader-line mask)."""
    k = len(anchors)
    centers = np.zeros((k, 2))
    placed = np.zeros(k, bool)
    leader = np.zeros(k, bool)
    anchor_r = np.full(k, 0.05) if anchor_r is None else anchor_r
    boxes = []
    order = np.arange(k) if order is None else order
    dirs = np.array([[0, -1], [0, 1], [1, 0], [-1, 0], [1, -1], [-1, -1], [1, 1], [-1, 1],
                     [0.5, -1], [-0.5, -1], [0.5, 1], [-0.5, 1], [1, -0.5], [-1, -0.5], [1, 0.5], [-1, 0.5]], float)
    for i in order:
        a = anchors[i]
        w, h = sizes[i]
        rr = anchor_r[i]
        best, best_cost = None, np.inf
        for step in (1.0, 1.3, 1.6, 2.0, 2.4, 3.4, 4.8, 6.5, 8.5, 11.0):
            for d in dirs:
                off = np.array([d[0] * (rr + 0.03 + w / 2), d[1] * (rr + 0.02 + h / 2)])
                c = a + off * step
                x0, x1, y0, y1 = c[0] - w / 2, c[0] + w / 2, c[1] - h / 2, c[1] + h / 2
                cost = 0.0
                if x0 < margin or x1 > W - margin or y0 < margin or y1 > H - margin:
                    cost += 50
                if boxes:
                    B = np.array(boxes)
                    ox = np.clip(np.minimum(x1, B[:, 1]) - np.maximum(x0, B[:, 0]) + 0.02, 0, None)
                    oy = np.clip(np.minimum(y1, B[:, 3]) - np.maximum(y0, B[:, 2]) + 0.01, 0, None)
                    cost += 400 * (ox * oy).sum() + 30 * ((ox > 0) & (oy > 0)).sum()
                if len(node_xy):
                    cx = np.clip(node_xy[:, 0], x0, x1)
                    cy = np.clip(node_xy[:, 1], y0, y1)
                    dd = np.sqrt((node_xy[:, 0] - cx) ** 2 + (node_xy[:, 1] - cy) ** 2)
                    pen = np.clip(node_r + 0.015 - dd, 0, None)
                    cost += 300 * pen.sum() + 20 * (pen > 0).sum()
                if len(seg_pts):
                    inside = ((seg_pts[:, 0] > x0) & (seg_pts[:, 0] < x1) & (seg_pts[:, 1] > y0)
                              & (seg_pts[:, 1] < y1))
                    cost += 0.8 * min(inside.sum(), 3)
                cost += 1.1 * (step - 1) + 0.02
                if cost < best_cost:
                    best, best_cost = c, cost
            if best_cost < 0.4:
                break
        if hide_unplaced and best_cost >= 2.0:
            continue
        centers[i] = best
        placed[i] = True
        nx_ = np.clip(a[0], best[0] - w / 2, best[0] + w / 2)
        ny_ = np.clip(a[1], best[1] - h / 2, best[1] + h / 2)
        leader[i] = np.hypot(a[0] - nx_, a[1] - ny_) > rr + 0.035
        boxes.append([best[0] - w / 2, best[0] + w / 2, best[1] - h / 2, best[1] + h / 2])
    return centers, placed, leader


def sample_segments(segs: list, step: float = 0.06) -> np.ndarray:
    """segs: list of (x0, y0, x1, y1) in inches -> points along them (excluding the ends)."""
    pts = []
    for x0, y0, x1, y1 in segs:
        L = np.hypot(x1 - x0, y1 - y0)
        m = int(L / step)
        if m < 3:
            continue
        t = np.linspace(0, 1, m)[1:-1]
        pts.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0)])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def repel_point_labels(xy_in: np.ndarray, texts: list, fontsize: float, W: float, H: float,
                       point_r: float = 0.04, others_xy: np.ndarray | None = None,
                       others_r: float = 0.03):
    """Convenience wrapper for scatter plots: labels for selected points, avoiding all points."""
    sizes = np.array([text_size(t, fontsize) for t in texts]) if texts else np.zeros((0, 2))
    obst = xy_in if others_xy is None else np.vstack([xy_in, others_xy])
    rr = np.r_[np.full(len(xy_in), point_r), np.full(0 if others_xy is None else len(others_xy), others_r)]
    return place_labels(xy_in, sizes, obst, rr, np.zeros((0, 2)), W, H, anchor_r=np.full(len(xy_in), point_r))


def remove_overlaps_rect(P: np.ndarray, w: np.ndarray, h: np.ndarray, off: np.ndarray, W: float, H: float,
                         iters: int = 400, pad: float = 0.06, ybounds: np.ndarray | None = None) -> np.ndarray:
    """Label-aware overlap removal. Each node owns a footprint rectangle (node + its label underneath):
    width w, height h, centred at node + (0, off). Overlapping footprints are pushed apart along the axis
    of least overlap until no two footprints intersect (or iters is reached)."""
    X = P.copy() * [W, H]
    n = len(X)
    if n < 2:
        return P
    w = w + pad
    h = h + pad
    for _ in range(iters):
        C = X + np.c_[np.zeros(n), off]
        dx = C[:, None, 0] - C[None, :, 0]
        dy = C[:, None, 1] - C[None, :, 1]
        ox = (w[:, None] + w[None, :]) / 2 - np.abs(dx)
        oy = (h[:, None] + h[None, :]) / 2 - np.abs(dy)
        ov = (ox > 0) & (oy > 0)
        np.fill_diagonal(ov, False)
        if not ov.any():
            break
        movex = np.where(ov & (ox <= oy), np.sign(dx + 1e-9) * ox / 2, 0.0).sum(1)
        movey = np.where(ov & (ox > oy), np.sign(dy + 1e-9) * oy / 2, 0.0).sum(1)
        X[:, 0] += 0.6 * movex
        X[:, 1] += 0.6 * movey
        X[:, 0] = np.clip(X[:, 0], w / 2, W - w / 2)
        if ybounds is not None:
            lo = ybounds[:, 0] * H + h / 2 - off
            hi = ybounds[:, 1] * H - h / 2 - off
            X[:, 1] = np.clip(X[:, 1], np.minimum(lo, hi), np.maximum(lo, hi))
        else:
            X[:, 1] = np.clip(X[:, 1], h / 2 - off, H - h / 2 - off)
    return X / [W, H]


# ------------------------------------------------------------------------------------------- special layouts
SPECIAL = {"Circular (edge-bundled chord)", "Hive plot (axes by molecule type)", "Arc diagram"}
LAYOUTS += ["Radial (focus-centred)", "Bipartite columns (substrates | enzymes | products)",
            "Circular (edge-bundled chord)", "Hive plot (axes by molecule type)", "Arc diagram"]


def radial(G, focus: list):
    """Concentric rings by graph distance from the focus molecules."""
    import networkx as nx
    U = G.to_undirected() if G.is_directed() else G
    focus = [f for f in focus if f in U] or [max(U.degree, key=lambda t: t[1])[0]]
    dist = {}
    for comp in nx.connected_components(U):
        src = [f for f in focus if f in comp] or [max(comp, key=U.degree)]
        dist.update(nx.multi_source_dijkstra_path_length(U.subgraph(comp), src))
    rings = {}
    for n, d in dist.items():
        rings.setdefault(d, []).append(n)
    pos = {}
    angle = {}
    for d in sorted(rings):
        mem = rings[d]
        if d == 0:
            for i, n in enumerate(mem):
                a = 2 * np.pi * i / len(mem)
                pos[n] = np.array([0.15 * np.cos(a), 0.15 * np.sin(a)]) if len(mem) > 1 else np.zeros(2)
                angle[n] = a
            continue
        def par(n):
            ps = [angle[m] for m in U.neighbors(n) if m in angle]
            return np.angle(np.mean(np.exp(1j * np.array(ps)))) if ps else 0.0
        mem.sort(key=par)
        for i, n in enumerate(mem):
            a = 2 * np.pi * i / len(mem) + 0.3 * d
            angle[n] = a
            pos[n] = d * np.array([np.cos(a), np.sin(a)])
    return pos


def bipartite_columns(G, kinds: dict):
    """Three columns: metabolites used only as substrates | enzymes/reactions | the other metabolites."""
    import networkx as nx
    D = nx.DiGraph(G)
    cols = {}
    for n in D:
        if kinds.get(n) != "metabolite":
            cols[n] = 1
        elif D.in_degree(n) == 0:
            cols[n] = 0
        else:
            cols[n] = 2
    groups = {c: [n for n in D if cols[n] == c] for c in (0, 1, 2)}
    order = {n: i for c in groups for i, n in enumerate(sorted(groups[c], key=str))}
    U = D.to_undirected()
    for _ in range(6):
        for c in (1, 0, 2, 1):
            def bary(n):
                nb = [order[m] / max(1, len(groups[cols[m]])) for m in U.neighbors(n) if cols[m] != c]
                return np.mean(nb) if nb else order[n] / max(1, len(groups[c]))
            groups[c].sort(key=bary)
            for i, n in enumerate(groups[c]):
                order[n] = i
    pos = {}
    for c, mem in groups.items():
        for i, n in enumerate(mem):
            pos[n] = np.array([c / 2, 1 - (i + 0.5) / max(1, len(mem))])
    return pos


def circular_sorted(G, key):
    order = sorted(G, key=key)
    n = len(order)
    return {v: np.array([np.cos(2 * np.pi * i / n + np.pi / 2), np.sin(2 * np.pi * i / n + np.pi / 2)])
            for i, v in enumerate(order)}


def hive(G, kinds: dict):
    """Hive plot: one radial axis per molecule class, position along the axis = degree rank."""
    U = G.to_undirected() if G.is_directed() else G
    cls = lambda k: "metabolite" if k == "metabolite" else ("reaction" if k == "reaction" else "enzyme")
    axes = [a for a in ("enzyme", "metabolite", "reaction") if any(cls(kinds.get(n)) == a for n in G)]
    fixed = {"enzyme": np.pi / 2, "metabolite": np.pi / 2 + 2 * np.pi / 3, "reaction": np.pi / 2 + 4 * np.pi / 3}
    ang = {a: fixed[a] for a in axes}
    pos = {}
    for a in axes:
        mem = sorted([n for n in G if cls(kinds.get(n)) == a], key=lambda n: U.degree(n))
        for i, n in enumerate(mem):
            r = 0.18 + 0.82 * (i + 0.5) / len(mem)
            pos[n] = r * np.array([np.cos(ang[a]), np.sin(ang[a])])
    return pos, ang


def arc(G, key):
    order = sorted(G, key=key)
    n = len(order)
    return {v: np.array([(i + 0.5) / n, 0.0]) for i, v in enumerate(order)}


def label_collisions(centers: np.ndarray, sizes: np.ndarray, placed: np.ndarray, node_xy: np.ndarray,
                     node_r: np.ndarray, seg_pts: np.ndarray, pad: float = 0.01) -> dict:
    """Count real collisions after placement (inches): label-label overlaps, labels covering a node, labels
    lying on an edge. Returns counts and a per-label 'clear' mask."""
    k = len(centers)
    clear = placed.copy()
    ll = ln = le = 0
    if k == 0:
        return dict(label_label=0, label_node=0, label_edge=0, clear=clear)
    w, h = sizes[:, 0] / 2, sizes[:, 1] / 2
    x0, x1, y0, y1 = centers[:, 0] - w, centers[:, 0] + w, centers[:, 1] - h, centers[:, 1] + h
    idx = np.where(placed)[0]
    for a in idx:
        ox = np.minimum(x1[a], x1[idx]) - np.maximum(x0[a], x0[idx])
        oy = np.minimum(y1[a], y1[idx]) - np.maximum(y0[a], y0[idx])
        hit = (ox > pad) & (oy > pad) & (idx != a)
        if hit.any():
            ll += int(hit.sum())
            clear[a] = False
        if len(node_xy):
            cx = np.clip(node_xy[:, 0], x0[a], x1[a])
            cy = np.clip(node_xy[:, 1], y0[a], y1[a])
            if (np.hypot(node_xy[:, 0] - cx, node_xy[:, 1] - cy) < node_r - pad).any():
                ln += 1
                clear[a] = False
        if len(seg_pts):
            ins = ((seg_pts[:, 0] > x0[a] + pad) & (seg_pts[:, 0] < x1[a] - pad) &
                   (seg_pts[:, 1] > y0[a] + pad) & (seg_pts[:, 1] < y1[a] - pad))
            if ins.any():
                le += 1
                clear[a] = False
    return dict(label_label=ll // 2, label_node=ln, label_edge=le, clear=clear)
