"""
netview.py - Network studio: IPA-inspired, fully customisable molecular network and pathway maps.

Conventions (inspired by Ingenuity Pathway Analysis, IPA):
  * Shapes by molecule type: enzyme = tall diamond, kinase = inverted triangle, phosphatase = triangle,
    peptidase = wide diamond, transporter = trapezoid, metabolite / chemical = circle, reaction = small
    square (reaction view), transcription regulator = wide ellipse.
  * Measured molecules: red = up, green = down (colour intensity = |z|); alternative palettes available.
  * Molecule Activity Predictor (MAP)-like overlay: unmeasured metabolites are predicted from the enzyme
    evidence of the reactions that produce / consume them: orange = predicted increase, blue = decrease.
  * Edges: orange = leads to increase, blue = leads to decrease, yellow = observation inconsistent with
    the prediction, grey = no prediction; solid = catalytic (direct), dashed = protein-protein (indirect).
  * Subcellular layout: Extracellular space / Plasma membrane / Cytoplasm / Nucleus / Other bands, or an
    organelle-resolved version, from Human-GEM localisation.
This is an IPA-style visual grammar computed from TRACE statistics; it is not IPA's proprietary
knowledge base or its activation z-score algorithm.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .layout import compute_layout, place_labels, remove_overlaps, remove_overlaps_rect, sample_segments, text_size

IPA_BANDS = ["Extracellular Space", "Plasma Membrane", "Cytoplasm", "Nucleus", "Other"]
ORG_BANDS = ["Extracellular", "Plasma membrane", "Cytosol", "Mitochondria", "Endoplasmic reticulum", "Golgi",
             "Lysosome", "Peroxisome", "Nucleus", "Other"]
LETTER = {"e": "Extracellular", "c": "Cytosol", "m": "Mitochondria", "i": "Mitochondria", "r": "Endoplasmic reticulum",
          "g": "Golgi", "l": "Lysosome", "x": "Peroxisome", "n": "Nucleus"}
ORG2IPA = {"Extracellular": "Extracellular Space", "Plasma membrane": "Plasma Membrane", "Cell membrane": "Plasma Membrane",
           "Cytosol": "Cytoplasm", "Mitochondria": "Cytoplasm", "Inner mitochondria": "Cytoplasm",
           "Endoplasmic reticulum": "Cytoplasm", "Golgi": "Cytoplasm", "Golgi apparatus": "Cytoplasm",
           "Lysosome": "Cytoplasm", "Peroxisome": "Cytoplasm", "Nucleus": "Nucleus"}
GENE_ORG = {"Cytosol": "Cytosol", "Nucleus": "Nucleus", "Endoplasmic reticulum": "Endoplasmic reticulum",
            "Golgi apparatus": "Golgi", "Mitochondria": "Mitochondria", "Inner mitochondria": "Mitochondria",
            "Cell membrane": "Plasma membrane", "Lysosome": "Lysosome", "Peroxisome": "Peroxisome",
            "Extracellular": "Extracellular"}

PALETTES = {
    "IPA (red up / green down)": ("#1A9850", "#F7F7F7", "#D73027"),
    "Red / blue": ("#2166AC", "#F7F7F7", "#B2182B"),
    "Orange / purple (colour-blind safe)": ("#5E3C99", "#F7F7F7", "#E66101"),
    "Teal / brown": ("#01665E", "#F5F5F5", "#8C510A"),
}
PRED_UP, PRED_DOWN, INCONS, NOPRED = "#F28C28", "#2E86C1", "#E8C547", "#9AA5A2"
MPL_MARK = {"enzyme": "d", "kinase": "v", "phosphatase": "^", "peptidase": "D", "transporter": "trap",
            "metabolite": "o", "reaction": "s", "tf": "ellipse", "other": "o"}
PLY_MARK = {"enzyme": "diamond-tall", "kinase": "triangle-down", "phosphatase": "triangle-up",
            "peptidase": "diamond-wide", "transporter": "pentagon", "metabolite": "circle", "reaction": "square",
            "tf": "circle", "other": "circle"}
TYPE_LABEL = {"enzyme": "Enzyme", "kinase": "Kinase", "phosphatase": "Phosphatase", "peptidase": "Peptidase",
              "transporter": "Transporter", "metabolite": "Metabolite / chemical", "reaction": "Reaction",
              "tf": "Transcription regulator"}


@dataclass
class NetOptions:
    scope: str = "Coherent reactions"           # Coherent reactions | Pathway | Enzyme neighbourhood | Molecule list | Top targets
    pathway: str = ""
    seeds: list = field(default_factory=list)
    hops: int = 1
    representation: str = "Molecule view (IPA-style)"   # or "Reaction view (bipartite)"
    max_reactions: int = 40
    isozymes_per_reaction: int = 3
    include_unmeasured: bool = True
    include_ppi: bool = False
    hide_isolated: bool = True
    layout: str = "Subcellular (IPA-style)"
    band_detail: str = "IPA (5 bands)"
    palette: str = "IPA (red up / green down)"
    zmax: float = 4.0
    value: str = "Best available (protein > mRNA)"   # or "Protein", "mRNA"
    prediction: bool = True
    pred_thr: float = 1.5
    size_mode: str = "Constant"                 # Constant | |z| | Degree | Target score
    node_size: float = 16.0                     # points
    size_range: tuple = (10.0, 28.0)
    label_mode: str = "All"                     # All | Measured | Significant | Top N by |z| | None
    label_top_n: int = 30
    font_size: float = 8.0
    font_family: str = "DejaVu Sans"
    label_box: bool = True
    hide_unplaced: bool = False
    edge_color: str = "IPA relationship"        # IPA relationship | By relation type | Single colour
    edge_single: str = "#7F8C8D"
    edge_width: float = 1.0
    edge_alpha: float = 0.85
    curved: bool = True
    arrows: bool = True
    show_legend: bool = True
    show_bands: bool = True
    title: str = ""
    width_in: float = 11.0
    height_in: float = 8.0
    background: str = "white"
    highlight: list = field(default_factory=list)
    highlight_color: str = "#111111"
    seed: int = 1
    spacing: float = 1.0
    auto_size: bool = True
    color_by: str = "z-score (measured / predicted)"
    shape_scheme: str = "IPA shapes"
    edge_route: str = "Curved"                 # Curved | Straight | Orthogonal (elbow)
    edge_width_by: str = "Constant"            # Constant | |enzyme evidence|
    effect_linestyle: bool = False             # print-friendly: dashed = decrease, dotted = inconsistent
    hulls: bool = False                        # shade network modules (communities)
    theme: str = "Light"                       # Light | Dark
    label_inside: bool = True                  # KEGG scheme: gene names inside their boxes
    focus: list = field(default_factory=list)  # radial layout centre


# ---------------------------------------------------------------------------------------------- graph
def _gene_info(kb):
    g = kb.genes.drop_duplicates("symbol").set_index("symbol")
    return g["mol_type"].to_dict(), g["compartments"].fillna("").to_dict()


def build_view(res, kb, o: NetOptions, targets: pd.DataFrame | None = None):
    """Returns (nodes DataFrame, edges DataFrame) for the chosen scope and representation."""
    cx = res.cascade
    rx = res.reactions.set_index("reaction")
    mname = dict(zip(kb.metabolites.met_base, kb.metabolites.name))
    mcomp = dict(zip(kb.metabolites.met_base, kb.metabolites.primary_compartment))
    rcomp = dict(zip(kb.reactions.rxn, kb.reactions.compartment))
    rname = dict(zip(kb.reactions.rxn, kb.reactions.name))
    cur = set(kb.metabolites.loc[kb.metabolites["currency"], "met_base"])
    mtype, gcomp = _gene_info(kb)
    # ---------------- reactions in scope
    if o.scope == "Pathway" and o.pathway:
        pr = kb.pathway_reactions
        cand = set(pr.loc[pr["pathway"] == o.pathway, "rxn"])
        bad = set(kb.reactions.loc[kb.reactions["is_transport"] | kb.reactions["is_exchange"] | kb.reactions["is_pool"], "rxn"])
        cand -= bad
        obs = [r for r in rx.index if r in cand]
        obs = sorted(obs, key=lambda r: rx.at[r, "p_label"])
        rest = sorted(cand - set(obs))
        rxns = (obs + (rest if o.include_unmeasured else []))[: o.max_reactions]
    elif o.scope == "Enzyme neighbourhood" and o.seeds:
        seeds = {s.strip().upper() for s in o.seeds}
        rxns = [r for r in rx.index if any(g.upper() in seeds for g in cx.rx_genes.get(r, []))]
        if o.hops > 1:
            mets = {m for r in rxns for m in cx.sub_ids.get(r, []) + cx.pro_ids.get(r, [])}
            rxns += [r for r in rx.index if set(cx.sub_ids.get(r, []) + cx.pro_ids.get(r, [])) & mets]
        rxns = sorted(set(rxns), key=lambda r: rx.at[r, "p_label"])[: o.max_reactions]
    elif o.scope == "Molecule list" and o.seeds:
        seeds = {s.strip().lower() for s in o.seeds}
        rxns = [r for r in rx.index if any(g.lower() in seeds for g in cx.rx_genes.get(r, []))
                or any(mname.get(m, m).lower() in seeds for m in cx.sub_ids.get(r, []) + cx.pro_ids.get(r, []))]
        rxns = sorted(set(rxns), key=lambda r: rx.at[r, "p_label"])[: o.max_reactions]
    elif o.scope == "Top targets" and targets is not None:
        top = set(targets.head(15)["gene"])
        rxns = [r for r in rx.index if set(cx.rx_genes.get(r, [])) & top]
        rxns = sorted(rxns, key=lambda r: rx.at[r, "p_label"])[: o.max_reactions]
    else:
        sel = rx[rx["significant"] | rx["metabolite_driven"]]
        if len(sel) < 5:
            sel = rx.nsmallest(o.max_reactions, "p_label")
        rxns = list(sel.sort_values("p_label").index[: o.max_reactions])
    # ---------------- values
    ge = res.gene_evidence.set_index("gene")
    feats = res.features
    pz = feats["prot"].sort_values("mean_log2", ascending=False).drop_duplicates("symbol").set_index("symbol")[["z", "q"]] \
        if "prot" in feats else pd.DataFrame(columns=["z", "q"])
    rz = feats["rna"].sort_values("mean_log2", ascending=False).drop_duplicates("symbol").set_index("symbol")[["z", "q"]] \
        if "rna" in feats else pd.DataFrame(columns=["z", "q"])
    mz = feats["met"].sort_values("mean_log2", ascending=False).drop_duplicates("met_base").set_index("met_base")[["z", "q"]]

    def gval(g):
        if o.value == "Protein":
            return (pz.at[g, "z"], pz.at[g, "q"], "protein") if g in pz.index else (np.nan, np.nan, "")
        if o.value == "mRNA":
            return (rz.at[g, "z"], rz.at[g, "q"], "mRNA") if g in rz.index else (np.nan, np.nan, "")
        if g in pz.index:
            return pz.at[g, "z"], pz.at[g, "q"], "protein"
        if g in rz.index:
            return rz.at[g, "z"], rz.at[g, "q"], "mRNA"
        return np.nan, np.nan, ""

    gpr = kb.gpr[kb.gpr["rxn"].isin(rxns)]
    st = kb.stoich[kb.stoich["rxn"].isin(rxns) & ~kb.stoich["met_base"].isin(cur)]
    nodes, edges = {}, []

    def add_gene(g):
        if f"g:{g}" in nodes:
            return
        z, q, src = gval(g)
        comps = [c for c in (gcomp.get(g, "") or "").split(";") if c]
        prio = ["Cytosol", "Mitochondria", "Inner mitochondria", "Endoplasmic reticulum", "Golgi apparatus",
                "Lysosome", "Peroxisome", "Cell membrane", "Extracellular", "Nucleus"]
        comp = next((c for c in prio if c in comps), comps[0] if comps else "")
        nodes[f"g:{g}"] = dict(id=f"g:{g}", label=g, kind=mtype.get(g, "enzyme"), z=z, q=q, source=src,
                               measured=np.isfinite(z), organelle=GENE_ORG.get(comp, "Other"))

    def add_met(m):
        if f"m:{m}" in nodes:
            return
        z, q = (mz.at[m, "z"], mz.at[m, "q"]) if m in mz.index else (np.nan, np.nan)
        nodes[f"m:{m}"] = dict(id=f"m:{m}", label=mname.get(m, m), kind="metabolite", z=z, q=q, source="metabolite",
                               measured=np.isfinite(z), organelle=LETTER.get(mcomp.get(m, "c"), "Other"))

    reaction_E = {}
    for r in rxns:
        genes = sorted(set(gpr.loc[gpr["rxn"] == r, "symbol"]))
        vals = [(g, gval(g)[0]) for g in genes]
        meas = [(g, v) for g, v in vals if np.isfinite(v)]
        if r in rx.index and np.isfinite(rx.at[r, "E"]):
            E = rx.at[r, "E"]
        else:
            E = float(np.mean([v for _, v in meas])) if meas else np.nan
        reaction_E[r] = E
        if o.representation.startswith("Molecule"):
            keep = sorted(meas, key=lambda t: -abs(t[1]))[: o.isozymes_per_reaction]
            if not keep and o.include_unmeasured:
                keep = [(g, np.nan) for g in genes[: max(1, o.isozymes_per_reaction // 2)]]
            show_genes = [g for g, _ in keep]
        else:
            show_genes = [g for g, _ in sorted(meas, key=lambda t: -abs(t[1]))][: o.isozymes_per_reaction] or \
                (genes[:1] if o.include_unmeasured else [])
        sub = st.loc[(st["rxn"] == r) & (st["coef"] < 0), "met_base"].tolist()
        pro = st.loc[(st["rxn"] == r) & (st["coef"] > 0), "met_base"].tolist()
        if not o.include_unmeasured:
            sub = [m for m in sub if m in mz.index]
            pro = [m for m in pro if m in mz.index]
        if not show_genes and o.representation.startswith("Molecule"):
            continue
        for g in show_genes:
            add_gene(g)
        for m in sub + pro:
            add_met(m)
        if o.representation.startswith("Molecule"):
            for g in show_genes:
                for m in sub:
                    edges.append(dict(source=f"m:{m}", target=f"g:{g}", relation="substrate of", rxn=r))
                for m in pro:
                    edges.append(dict(source=f"g:{g}", target=f"m:{m}", relation="produces", rxn=r))
        else:
            R = f"r:{r}"
            nodes[R] = dict(id=R, label=rname.get(r, r), kind="reaction",
                            z=rx.at[r, "C_std"] if r in rx.index else np.nan,
                            q=rx.at[r, "q_label"] if r in rx.index else np.nan, source="reaction",
                            measured=r in rx.index, organelle=LETTER.get(rcomp.get(r, "c"), "Other"))
            for g in show_genes:
                edges.append(dict(source=f"g:{g}", target=R, relation="catalyses", rxn=r))
            for m in sub:
                edges.append(dict(source=f"m:{m}", target=R, relation="substrate of", rxn=r))
            for m in pro:
                edges.append(dict(source=R, target=f"m:{m}", relation="produces", rxn=r))
    if o.include_ppi and "ppi_string" in kb.extra:
        gs = {n[2:] for n in nodes if n.startswith("g:")}
        ppi = kb.extra["ppi_string"]
        ppi = ppi[ppi["a"].isin(gs) & ppi["b"].isin(gs)]
        for a, b in zip(ppi["a"], ppi["b"]):
            if a < b:
                edges.append(dict(source=f"g:{a}", target=f"g:{b}", relation="PPI", rxn=""))
    N = pd.DataFrame(list(nodes.values()))
    E = pd.DataFrame(edges, columns=["source", "target", "relation", "rxn"]).drop_duplicates(["source", "target", "relation"])
    if N.empty:
        return N, E
    # ---------------- MAP-like prediction for unmeasured metabolites, IPA edge colours
    thr = o.pred_thr
    pred = {}
    for m in N.loc[N["kind"] == "metabolite", "id"]:
        sc = 0.0
        for _, e in E[(E["target"] == m) & (E["relation"] == "produces")].iterrows():
            v = reaction_E.get(e["rxn"], np.nan)
            sc += v if np.isfinite(v) and abs(v) >= thr else 0
        for _, e in E[(E["source"] == m) & (E["relation"] == "substrate of")].iterrows():
            v = reaction_E.get(e["rxn"], np.nan)
            sc -= v if np.isfinite(v) and abs(v) >= thr else 0
        pred[m] = np.sign(sc)
    N["predicted"] = N["id"].map(pred).fillna(0.0)
    zd = N.set_index("id")["z"].to_dict()
    cols = []
    for _, e in E.iterrows():
        if e["relation"] == "PPI":
            cols.append("ppi")
            continue
        Er = reaction_E.get(e["rxn"], np.nan)
        if not np.isfinite(Er) or abs(Er) < thr:
            cols.append("none")
            continue
        if e["relation"] == "produces":
            p, tgt = np.sign(Er), e["target"]
        elif e["relation"] == "substrate of":
            p, tgt = -np.sign(Er), e["source"]
        else:
            cols.append("up" if Er > 0 else "down")
            continue
        obs = zd.get(tgt, np.nan)
        if np.isfinite(obs) and abs(obs) >= thr and np.sign(obs) != p:
            cols.append("inconsistent")
        else:
            cols.append("up" if p > 0 else "down")
    E["ipa"] = cols
    E["evidence"] = [reaction_E.get(r, np.nan) for r in E["rxn"]]
    deg = pd.concat([E["source"], E["target"]]).value_counts()
    N["degree"] = N["id"].map(deg).fillna(0).astype(int)
    if o.hide_isolated:
        N = N[N["degree"] > 0]
    if targets is not None:
        ts = dict(zip("g:" + targets["gene"], -np.log10(targets["rra_score"].astype(float).clip(1e-30, 1))))
        N["target_score"] = N["id"].map(ts)
    N["band_ipa"] = N["organelle"].map(lambda x: ORG2IPA.get(x, "Other"))
    return N.reset_index(drop=True), E.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- styles
COLOR_BY = ["z-score (measured / predicted)", "Molecule type", "Compartment", "Data layer", "Module (community)",
            "Betweenness centrality", "Target score"]
SHAPES = ["IPA shapes", "KEGG (boxes for genes)", "All circles"]
ROUTES = ["Curved", "Straight", "Orthogonal (elbow)"]
CAT = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F",
       "#BAB0AC", "#1F77B4", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#17BECF"]
PALETTES["Greyscale (print)"] = ("#3A3A3A", "#F2F2F2", "#000000")

PRESETS = {
    "IPA classic": dict(layout="Subcellular (IPA-style)", shape_scheme="IPA shapes", palette="IPA (red up / green down)",
                        color_by=COLOR_BY[0], edge_color="IPA relationship", edge_route="Curved", show_bands=True,
                        theme="Light", hulls=False, effect_linestyle=False, prediction=True, label_box=True),
    "KEGG map": dict(layout="Hierarchical flow (left-right)", shape_scheme="KEGG (boxes for genes)", palette="Red / blue",
                     color_by=COLOR_BY[0], edge_color="Single colour", edge_single="#333333", edge_route="Orthogonal (elbow)",
                     show_bands=False, theme="Light", hulls=False, label_inside=True, prediction=False),
    "Cytoscape default": dict(layout="Force-directed", shape_scheme="All circles", palette="Red / blue",
                              color_by=COLOR_BY[0], edge_color="Single colour", edge_single="#9AA5A2",
                              edge_route="Straight", show_bands=False, theme="Light", hulls=False),
    "Journal minimal": dict(layout="Force-directed", shape_scheme="All circles", palette="Red / blue", node_size=11.0,
                            edge_color="Single colour", edge_single="#C0C6C5", edge_width=0.6, edge_route="Straight",
                            show_bands=False, label_mode="Measured", font_size=7.0, label_box=False, theme="Light"),
    "Dark presentation": dict(layout="Force-directed", shape_scheme="IPA shapes", palette="Orange / purple (colour-blind safe)",
                              edge_color="IPA relationship", edge_route="Curved", show_bands=False, theme="Dark",
                              label_box=False, font_size=9.0),
    "Colour-blind safe": dict(palette="Orange / purple (colour-blind safe)", edge_color="By relation type",
                              effect_linestyle=True, theme="Light"),
    "Greyscale print": dict(palette="Greyscale (print)", edge_color="Single colour", edge_single="#444444",
                            effect_linestyle=True, show_bands=True, theme="Light", hulls=False),
    "Module map": dict(layout="Force-directed", color_by="Module (community)", hulls=True, edge_color="Single colour",
                       edge_single="#B8C2C0", edge_route="Straight", show_bands=False, theme="Light"),
    "Hub focus": dict(layout="Radial (focus-centred)", color_by="Betweenness centrality", size_mode="Degree",
                      edge_color="Single colour", edge_single="#B8C2C0", edge_route="Straight", show_bands=False),
    "Chord / circos": dict(layout="Circular (edge-bundled chord)", color_by="Molecule type", edge_color="IPA relationship",
                           show_bands=False, hulls=False, arrows=False),
    "Hive plot": dict(layout="Hive plot (axes by molecule type)", color_by="z-score (measured / predicted)",
                      edge_color="IPA relationship", show_bands=False, arrows=False),
    "Arc diagram": dict(layout="Arc diagram", color_by="Module (community)", edge_color="IPA relationship",
                        show_bands=False, arrows=False),
    "Pathway flow": dict(layout="Hierarchical flow (top-down)", shape_scheme="IPA shapes", edge_color="IPA relationship",
                         edge_route="Curved", show_bands=False),
    "Bipartite columns": dict(layout="Bipartite columns (substrates | enzymes | products)", edge_route="Curved",
                              show_bands=False, edge_color="IPA relationship"),
}


def apply_preset(o: "NetOptions", name: str) -> "NetOptions":
    import dataclasses
    return dataclasses.replace(o, **PRESETS.get(name, {}))


# ---------------------------------------------------------------------------------------------- geometry
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255


def _blend(z, palette, zmax):
    lo, mid, hi = (_hex_to_rgb(c) for c in PALETTES[palette])
    t = np.clip(abs(z) / zmax, 0, 1) if np.isfinite(z) else 0
    c = mid + (hi - mid) * t if z >= 0 else mid + (lo - mid) * t
    return "#%02x%02x%02x" % tuple((c * 255).round().astype(int))


def _seq(v, vmax):
    import matplotlib
    c = matplotlib.colormaps["viridis"](np.clip(v / (vmax or 1), 0, 1) * 0.92 + 0.04)
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in c[:3])


def _modules(G):
    import networkx as nx
    U = nx.Graph(G.to_undirected())
    try:
        comms = nx.community.louvain_communities(U, seed=1, resolution=1.0)
    except Exception:
        comms = list(nx.connected_components(U))
    comms = sorted(comms, key=len, reverse=True)
    return {n: i for i, c in enumerate(comms) for n in c}


def _route(p0, p1, mode, center):
    """Polyline for one edge."""
    t = np.linspace(0, 1, 24)[:, None]
    if mode == "Straight":
        return np.vstack([p0, p1])
    if mode == "Orthogonal (elbow)":
        xm = (p0[0] + p1[0]) / 2
        return np.array([p0, [xm, p0[1]], [xm, p1[1]], p1])
    if mode == "bundle":
        ctrl = center + 0.15 * ((p0 + p1) / 2 - center)
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    if mode == "arc":
        d = abs(p1[0] - p0[0])
        ctrl = np.array([(p0[0] + p1[0]) / 2, p0[1] + d * 0.62])
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    d = p1 - p0
    L = np.hypot(*d) or 1
    ctrl = (p0 + p1) / 2 + np.array([-d[1], d[0]]) / L * 0.13 * L
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def _curve(p0, p1, k, n=24):
    """Quadratic Bezier from p0 to p1 bent sideways by k x length (k = 0: straight)."""
    t = np.linspace(0, 1, n)[:, None]
    d = p1 - p0
    L = np.hypot(*d) or 1
    ctrl = (p0 + p1) / 2 + np.array([-d[1], d[0]]) / L * k * L
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def _route_avoid(p0, p1, mode, center, obs_xy, obs_r, skip):
    """Edge route that steers around node discs: candidate bends are tried in order of preference and the first
    route that clears every other node (or the least-penetrating one) is kept."""
    if mode in ("bundle", "arc") or not len(obs_xy):
        return _route(p0, p1, mode, center)
    keep = np.ones(len(obs_xy), bool)
    keep[list(skip)] = False
    O, R = obs_xy[keep], obs_r[keep]
    if mode == "Orthogonal (elbow)":
        xm, ym = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        cands = [np.array([p0, [xm + f * dx, p0[1]], [xm + f * dx, p1[1]], p1]) for f in (0, -0.25, 0.25, -0.45, 0.45)]
        cands += [np.array([p0, [p0[0], ym + f * dy], [p1[0], ym + f * dy], p1]) for f in (0, -0.25, 0.25)]
        dense = []
        for c in cands:
            seg = [np.linspace(c[i], c[i + 1], 10) for i in range(len(c) - 1)]
            dense.append(np.vstack(seg))
    else:
        ks = (0.13, -0.13, 0.0, 0.25, -0.25, 0.4, -0.4, 0.6, -0.6, 0.85, -0.85) if mode == "Curved" else \
             (0.0, 0.12, -0.12, 0.25, -0.25, 0.4, -0.4, 0.6, -0.6)
        cands = [(_curve(p0, p1, k) if (k or mode == "Curved") else np.vstack([p0, p1])) for k in ks]
        dense = [_curve(p0, p1, k) for k in ks]
    best, best_c = cands[0], np.inf
    for j, (c, dp) in enumerate(zip(cands, dense)):
        if not len(O):
            return c
        dd = np.sqrt(((dp[:, None, :] - O[None, :, :]) ** 2).sum(-1))
        pen = np.clip(R[None, :] - dd, 0, None).max(0)
        cost = 50 * pen.sum() + (pen > 0).sum() + 0.02 * j
        if pen.sum() == 0:
            return c
        if cost < best_c:
            best, best_c = c, cost
    return best


def _trim(pts, r0, r1):
    """Shorten a polyline by r0 at the start and r1 at the end (arrow heads stop at node borders)."""
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    L = seg[-1]
    if L <= r0 + r1 + 1e-6:
        return pts
    s = np.linspace(r0, L - r1, max(2, len(pts)))
    return np.c_[np.interp(s, seg, pts[:, 0]), np.interp(s, seg, pts[:, 1])]


def geometry(N: pd.DataFrame, E: pd.DataFrame, o: NetOptions):
    """Layout + overlap removal + label placement + colours + edge routes, all in inches."""
    import networkx as nx
    W, H = o.width_in, o.height_in
    legend_w = 2.4 if o.show_legend else 0.0
    title_h = 0.45 if o.title else 0.15
    G = nx.DiGraph()
    G.add_nodes_from(N["id"])
    G.add_edges_from(zip(E["source"], E["target"]))
    kinds = dict(zip(N["id"], N["kind"]))
    module = _modules(G) if len(N) else {}
    bands = IPA_BANDS if o.band_detail.startswith("IPA") else ORG_BANDS
    band_col = "band_ipa" if o.band_detail.startswith("IPA") else "organelle"
    band_of = dict(zip(N["id"], N[band_col]))
    focus = [i for i, l in zip(N["id"], N["label"]) if l.lower() in {f.strip().lower() for f in (o.focus or o.highlight)}]
    if not focus and len(N):
        focus = [N.loc[N["z"].abs().fillna(0).idxmax(), "id"]]
    special = o.layout in ("Circular (edge-bundled chord)", "Hive plot (axes by molecule type)", "Arc diagram", "Circular")
    use_bands = o.layout.startswith("Subcellular")
    pos, geom = compute_layout(G, o.layout, kinds, band_of, bands, seed=o.seed, focus=focus, module=module)
    ids = list(N["id"])
    P = np.array([pos[i] for i in ids]) if ids else np.zeros((0, 2))
    # node sizes
    if o.size_mode == "|z|":
        v = N["z"].abs().fillna(0).values
        sz = o.size_range[0] + (o.size_range[1] - o.size_range[0]) * np.clip(v / o.zmax, 0, 1)
    elif o.size_mode == "Degree":
        v = N["degree"].values.astype(float)
        sz = o.size_range[0] + (o.size_range[1] - o.size_range[0]) * (v - v.min()) / (np.ptp(v) or 1)
    elif o.size_mode == "Target score" and "target_score" in N:
        v = N["target_score"].fillna(0).values
        sz = o.size_range[0] + (o.size_range[1] - o.size_range[0]) * (v / (np.nanmax(v) or 1))
    else:
        sz = np.full(len(N), o.node_size)
    kind_arr = N["kind"].values
    sz = np.where(kind_arr == "reaction", sz * 0.55, sz)
    kegg = o.shape_scheme.startswith("KEGG")
    if kegg:
        sz = np.where(kind_arr == "metabolite", sz * 0.7, sz)
    r_in = sz / 72 / 2 * o.spacing
    # labels
    lab_mask = np.ones(len(N), bool)
    if o.label_mode == "None":
        lab_mask[:] = False
    elif o.label_mode == "Measured":
        lab_mask = N["measured"].values.astype(bool)
    elif o.label_mode == "Significant":
        lab_mask = (N["q"].fillna(1) < 0.05).values
    elif o.label_mode.startswith("Top"):
        lab_mask = (N["z"].abs().fillna(-1).rank(ascending=False, method="first") <= o.label_top_n).values
    if o.representation.startswith("Reaction") and o.label_mode != "All":
        lab_mask &= kind_arr != "reaction"
    hl = {h.strip().lower() for h in o.highlight if h.strip()}
    lab_mask |= N["label"].str.lower().isin(hl).values
    alltexts = []
    for i in range(len(N)):
        t = str(N["label"].iat[i])
        t = t[:26] + "…" if kind_arr[i] == "reaction" and len(t) > 27 else t
        alltexts.append(t if len(t) <= 30 else t[:28] + "…")
    lw_ = np.array([text_size(t, o.font_size)[0] if m else 0 for t, m in zip(alltexts, lab_mask)])
    lh_ = np.array([text_size(t, o.font_size)[1] if m else 0 for t, m in zip(alltexts, lab_mask)])
    inside = kegg and o.label_inside
    box_wh = np.zeros((len(N), 2))
    if inside:   # gene boxes sized to their name
        g_m = np.isin(kind_arr, ["enzyme", "kinase", "phosphatase", "peptidase", "transporter", "tf"])
        for i in np.where(g_m)[0]:
            w_, h_ = text_size(alltexts[i], o.font_size)
            box_wh[i] = (w_ + 0.06, h_ + 0.04)
            r_in[i] = max(box_wh[i]) / 2
        lab_mask_out = lab_mask & ~g_m
    else:
        lab_mask_out = lab_mask
    # canvas size
    n = max(len(N), 1)
    maxlab = float(lw_.max()) if len(lw_) else 0.5
    if special:
        if o.layout == "Arc diagram":
            step = max(2 * r_in.max() * 1.5, float(lh_.max() if len(lh_) else 0.15) * 1.25)
            PW = max(W - legend_w, n * step + 0.6)
            PH = max(H - title_h, PW * 0.42 + maxlab + 0.3)
        else:
            R = max(n * (2 * r_in.mean() * 1.6) / (2 * np.pi), 1.2)
            side = 2 * (R + maxlab + 0.35)
            PW = PH = max(min(W - legend_w, H - title_h), side) if o.auto_size else min(W - legend_w, H - title_h)
    else:
        PW, PH = W - legend_w, H - title_h
    fw = np.maximum(2 * r_in, np.where(lab_mask_out, lw_, 0)) * o.spacing
    fh = (2 * r_in + np.where(lab_mask_out, lh_, 0) + 0.02) * o.spacing
    off = -np.where(lab_mask_out, lh_, 0) / 2
    if o.auto_size and len(N) and not special:
        need = float((fw * fh).sum()) * 3.2
        scale = np.sqrt(need / (PW * PH))
        if o.layout.startswith("Hierarchical") or o.layout.startswith("Bipartite"):
            vertical = "top-down" in o.layout
            keys = np.round(P[:, 1 if vertical else 0], 3)
            widest = max((fw[keys == k].sum() if vertical else fh[keys == k].sum()) * 1.25 for k in np.unique(keys))
            scale = max(scale, widest / (PW if vertical else PH))
        if scale > 1:
            scale = min(scale, 2.6)
            PW, PH = PW * scale, PH * scale
    W, H = PW + legend_w, PH + title_h
    ybounds = np.array([geom.get(band_of.get(i, "Other"), (0, 1)) for i in ids]) if geom else None
    if not special:
        P = remove_overlaps(P, r_in * 1.3, PW, PH, ybounds=ybounds)
        P = remove_overlaps_rect(P, fw, fh, off, PW, PH, ybounds=ybounds)
    idx = {nid: k for k, nid in enumerate(ids)}
    mode = "bundle" if o.layout.startswith("Circular (edge") else ("arc" if o.layout == "Arc diagram" else
           ("bundle" if o.layout.startswith("Hive") else o.edge_route))
    from .layout import label_collisions, sample_segments
    li = np.where(lab_mask_out)[0]
    texts = [alltexts[i] for i in li]
    sizes = np.c_[lw_[li], lh_[li]] if len(li) else np.zeros((0, 2))
    rounds = 6 if (o.auto_size and not special) else 1
    PW0 = PW
    for attempt in range(rounds):
        XY = np.c_[P[:, 0] * PW, P[:, 1] * PH]
        center = np.array([PW / 2, PH / 2])
        # edge routes: steer around every node that is not an endpoint
        obs_r = r_in + 0.04
        if inside:
            obs_r = np.maximum(obs_r, np.hypot(*box_wh.T) / 2 + 0.02)
        routes, n_through = [], 0
        for a_, b_ in zip(E["source"], E["target"]):
            if a_ not in idx or b_ not in idx:
                routes.append(None)
                continue
            ia, ib = idx[a_], idx[b_]
            p0, p1 = XY[ia], XY[ib]
            if mode == "arc" and p1[0] < p0[0]:
                pts = _route(p1, p0, mode, center)[::-1]
            else:
                pts = _route_avoid(p0, p1, mode, center, XY, obs_r, (ia, ib))
            if len(XY) > 2 and mode not in ("bundle", "arc"):
                dp = pts if len(pts) > 2 else np.linspace(pts[0], pts[-1], 24)
                m = np.ones(len(XY), bool)
                m[[ia, ib]] = False
                dd = np.sqrt(((dp[:, None, :] - XY[m][None, :, :]) ** 2).sum(-1))
                n_through += int(((r_in[m] - dd) > 0.005).any(0).sum())
            routes.append(_trim(pts, r_in[ia] * 1.05, r_in[ib] * 1.2))
        seg_pts = np.zeros((0, 2))
        if any(r is not None for r in routes):
            segs = [(r[k, 0], r[k, 1], r[k + 1, 0], r[k + 1, 1]) for r in routes if r is not None for k in range(len(r) - 1)]
            seg_pts = sample_segments(segs, 0.04)
        # label placement
        rot = np.zeros(len(li))
        if special and len(li):
            centers = np.zeros((len(li), 2))
            for j, i in enumerate(li):
                if o.layout == "Arc diagram":
                    centers[j] = XY[i] + np.array([0, -(r_in[i] + 0.04 + lw_[i] / 2)])
                    rot[j] = 90
                else:
                    v = XY[i] - center
                    a = np.arctan2(v[1], v[0])
                    dist = r_in[i] + 0.05 + lw_[i] / 2
                    centers[j] = XY[i] + dist * np.array([np.cos(a), np.sin(a)])
                    deg = np.degrees(a)
                    rot[j] = deg if -90 <= deg <= 90 else deg + 180
            placed = np.ones(len(li), bool)
            leader = np.zeros(len(li), bool)
            # rotated radial labels: placement is geometric (outside the ring), not collision-checked
            coll = dict(label_label=0, label_node=0, label_edge=0, clear=placed.copy(), checked=False)
            break
        order = np.argsort(-(N["z"].abs().fillna(0).values[li] + 5 * N["measured"].values[li]))
        centers, placed, leader = place_labels(XY[li], sizes, XY, obs_r if inside else r_in, seg_pts, PW, PH,
                                               order=order, hide_unplaced=o.hide_unplaced, anchor_r=r_in[li])
        coll = label_collisions(centers, sizes, placed, XY, r_in, seg_pts)
        coll["checked"] = True
        # hard: labels on labels or nodes, edges through nodes; soft: a few labels resting on an edge (their
        # opaque background keeps them legible), tolerated up to 8 % of the labels
        bad = coll["label_label"] + coll["label_node"] + n_through
        if bad == 0 or attempt == rounds - 1 or PW * 1.1 > PW0 * 1.8:
            break
        grow = 1.12 if bad > 3 else 1.06
        PW, PH = PW * grow, PH * grow
    W, H = PW + legend_w, PH + title_h
    coll["edge_node"] = n_through if not special else 0
    # colours
    fill, edgec, legend_items, cbar = [], [], [], None
    cb = o.color_by
    if cb.startswith("z-score"):
        for _, r in N.iterrows():
            if r["measured"]:
                fill.append(_blend(r["z"], o.palette, o.zmax))
                edgec.append("#333333")
            elif o.prediction and r["predicted"] != 0:
                fill.append("#FBD9B8" if r["predicted"] > 0 else "#C6DDF0")
                edgec.append(PRED_UP if r["predicted"] > 0 else PRED_DOWN)
            else:
                fill.append("#FFFFFF")
                edgec.append("#8A9593")
        cbar = ("z-score", PALETTES[o.palette], -o.zmax, o.zmax)
    else:
        if cb == "Molecule type":
            cats = [TYPE_LABEL.get(k, k) for k in kind_arr]
        elif cb == "Compartment":
            cats = list(N[band_col].values)
        elif cb == "Data layer":
            cats = [{"protein": "Protein", "mRNA": "mRNA", "metabolite": "Metabolite", "reaction": "Reaction"}.get(s_, "Not measured")
                    if m_ else "Not measured" for s_, m_ in zip(N["source"], N["measured"])]
        elif cb == "Module (community)":
            cats = [f"Module {module.get(i, 0) + 1}" for i in ids]
        else:
            cats = None
        if cats is not None:
            uniq = list(dict.fromkeys(sorted(cats, key=lambda c: (c == "Not measured", c))))
            cmap = {c: ("#FFFFFF" if c == "Not measured" else CAT[k % len(CAT)]) for k, c in enumerate(uniq)}
            fill = [cmap[c] for c in cats]
            edgec = ["#333333"] * len(N)
            legend_items = [(c, cmap[c]) for c in uniq][:16]
        else:
            if cb.startswith("Betweenness"):
                U = nx.Graph(G.to_undirected())
                bc = nx.betweenness_centrality(U)
                v = np.array([bc.get(i, 0) for i in ids])
                lab = "betweenness"
            else:
                v = N["target_score"].fillna(0).values if "target_score" in N else np.zeros(len(N))
                lab = "-log10 target score"
            vmax = float(np.nanmax(v)) if len(v) else 1
            fill = [_seq(x, vmax) for x in v]
            edgec = ["#333333"] * len(N)
            cbar = (lab, "viridis", 0, vmax)
    hlm = N["label"].str.lower().isin(hl).values
    return dict(ids=ids, XY=XY, r=r_in, size_pt=sz, fill=fill, edgec=edgec, hl=hlm, li=li, texts=texts,
                centers=centers, placed=placed, leader=leader, rot=rot, routes=routes, geom=geom if use_bands else None,
                bands=bands, PW=PW, PH=PH, W=W, H=H, title_h=title_h, legend_w=legend_w, idx=idx, module=module,
                legend_items=legend_items, cbar=cbar, box_wh=box_wh, inside=inside, alltexts=alltexts,
                lab_mask=lab_mask, kegg=kegg, center=center, collisions=coll)


def _edge_color(kind, o):
    if o.edge_color == "Single colour":
        return o.edge_single
    if o.edge_color == "By relation type":
        return {"produces": "#C0392B", "substrate of": "#2C6FB7", "catalyses": "#555555", "PPI": "#999999"}.get(kind, "#777")
    return {"up": PRED_UP, "down": PRED_DOWN, "inconsistent": INCONS, "none": NOPRED, "ppi": "#B0B0B0"}.get(kind, NOPRED)


def _edge_style(e, o):
    col = _edge_color(e["ipa"] if o.edge_color == "IPA relationship" else e["relation"], o)
    ls = "--" if e["relation"] == "PPI" else "-"
    if o.effect_linestyle:
        ls = {"down": "--", "inconsistent": ":", "none": (0, (1, 2))}.get(e["ipa"], ls)
    w = o.edge_width
    if o.edge_width_by.startswith("|enzyme") and np.isfinite(e.get("evidence", np.nan)):
        w = o.edge_width * (0.5 + 1.5 * min(abs(e["evidence"]) / o.zmax, 1))
    return col, ls, w


# ---------------------------------------------------------------------------------------------- matplotlib
def render_mpl(N: pd.DataFrame, E: pd.DataFrame, o: NetOptions, g: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
    from matplotlib.path import Path

    g = g or geometry(N, E, o)
    dark = o.theme == "Dark"
    ink = "#F2F4F3" if dark else "#1F2A28"
    sub_ink = "#AEB8B6" if dark else "#5D6D6A"
    bg = "#15191C" if dark else "white"
    plt.rcParams["font.family"] = o.font_family
    fig = plt.figure(figsize=(g["W"], g["H"]), facecolor=bg)
    if o.background == "transparent":
        fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, g["PW"] / g["W"], g["PH"] / g["H"]], facecolor=bg)
    ax.set_xlim(0, g["PW"])
    ax.set_ylim(0, g["PH"])
    ax.axis("off")
    XY, idx = g["XY"], g["idx"]
    if g["geom"] and o.show_bands:
        cols = ["#1E2428", "#181D20"] if dark else ["#F4F8FB", "#FFFFFF"]
        for k, (b, (lo, hi)) in enumerate(g["geom"].items()):
            ax.add_patch(Rectangle((0, lo * g["PH"]), g["PW"], (hi - lo) * g["PH"], color=cols[k % 2], zorder=0, lw=0))
            ax.axhline(lo * g["PH"], color="#3A4448" if dark else "#C9D3D1", lw=0.8, ls=(0, (4, 3)), zorder=0.5)
            ax.text(0.08, hi * g["PH"] - 0.08, b, fontsize=o.font_size + 1, fontweight="bold", color=sub_ink,
                    va="top", ha="left", zorder=0.6)
    if o.layout.startswith("Hive") and len(XY):
        from .layout import hive
        import networkx as nx
        _, angs = hive(nx.Graph([(a, b) for a, b in zip(E["source"], E["target"])]) if len(E) else nx.Graph(),
                       dict(zip(N["id"], N["kind"])))
        R = 0.48 * min(g["PW"], g["PH"])
        for a_name, a in angs.items():
            p = g["center"] + R * np.array([np.cos(a), np.sin(a)])
            ax.plot([g["center"][0], p[0]], [g["center"][1], p[1]], color=sub_ink, lw=1.2, zorder=0.5)
            ax.text(*(g["center"] + (R + 0.25) * np.array([np.cos(a), np.sin(a)])), a_name.capitalize() + "s",
                    color=sub_ink, fontsize=o.font_size + 1, ha="center", va="center", fontweight="bold")
    if o.layout == "Arc diagram" and len(XY):
        ax.axhline(XY[0, 1], color=sub_ink, lw=0.6, zorder=0.5)
    if o.hulls and len(XY):
        from scipy.spatial import ConvexHull
        mods = {}
        for k, nid in enumerate(g["ids"]):
            mods.setdefault(g["module"].get(nid, 0), []).append(k)
        for m, mem in mods.items():
            if len(mem) < 3:
                continue
            pts = XY[mem]
            pad = []
            for p in pts:
                for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                    pad.append(p + (g["r"].max() + 0.12) * np.array([np.cos(a), np.sin(a)]))
            pad = np.array(pad)
            try:
                hull = ConvexHull(pad)
            except Exception:
                continue
            col = CAT[m % len(CAT)]
            ax.add_patch(Polygon(pad[hull.vertices], closed=True, fc=col, ec=col, alpha=0.10, lw=1.0, zorder=0.4))
            top = pad[hull.vertices][np.argmax(pad[hull.vertices][:, 1])]
            ax.text(top[0], top[1] + 0.05, f"M{m + 1}", color=col, fontsize=o.font_size, fontweight="bold",
                    ha="center", va="bottom", zorder=0.45)
    for k_e, (_, e) in enumerate(E.iterrows()):
        pts = g["routes"][k_e]
        if pts is None:
            continue
        col, ls, w = _edge_style(e, o)
        arrow = o.arrows and e["relation"] != "PPI"
        ap = FancyArrowPatch(path=Path(pts), arrowstyle="-|>" if arrow else "-", mutation_scale=6 + 2 * w,
                             color=col, lw=w, alpha=o.edge_alpha, linestyle=ls, zorder=1, shrinkA=0, shrinkB=0)
        ax.add_patch(ap)
    trap = Path([(-1, -0.6), (1, -0.6), (0.6, 0.6), (-0.6, 0.6), (-1, -0.6)])
    for k, kind in enumerate(N["kind"]):
        x, y = XY[k]
        s = g["size_pt"][k]
        hl = g["hl"][k]
        lw = 2.2 if hl else (1.3 if not N["measured"].iat[k] else 0.8)
        ec = o.highlight_color if hl else g["edgec"][k]
        if dark and ec in ("#333333",):
            ec = "#DDDDDD"
        is_gene = kind in ("enzyme", "kinase", "phosphatase", "peptidase", "transporter", "tf")
        if g["kegg"] and is_gene:
            bw, bh = g["box_wh"][k] if g["inside"] else (s / 72 * 1.6, s / 72 * 0.8)
            ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=g["fill"][k], ec=ec, lw=lw, zorder=3))
            if g["inside"]:
                r_, gg, b_ = _hex_to_rgb(g["fill"][k]) if g["fill"][k].startswith("#") else (1, 1, 1)
                txtc = "white" if (0.299 * r_ + 0.587 * gg + 0.114 * b_) < 0.5 else "#111111"
                ax.text(x, y, g["alltexts"][k], fontsize=o.font_size, ha="center", va="center", color=txtc, zorder=4,
                        fontweight="bold" if hl else "normal")
            continue
        mk = "o" if (o.shape_scheme == "All circles" or g["kegg"]) else MPL_MARK.get(kind, "o")
        if kind == "reaction":
            mk = "s"
        if mk == "trap":
            ax.scatter([x], [y], s=s ** 2, marker=trap, c=[g["fill"][k]], edgecolors=ec, linewidths=lw, zorder=3)
        elif mk == "ellipse":
            ax.add_patch(Ellipse((x, y), s / 72 * 1.5, s / 72 * 0.8, fc=g["fill"][k], ec=ec, lw=lw, zorder=3))
        else:
            ax.scatter([x], [y], s=s ** 2 * (0.8 if mk in ("d", "v", "^") else 0.65), marker=mk, c=[g["fill"][k]],
                       edgecolors=ec, linewidths=lw, zorder=3)
    for j, i in enumerate(g["li"]):
        if not g["placed"][j]:
            continue
        cxy = g["centers"][j]
        if g["leader"][j]:
            # leader: from the node border to the label border, dotted so it never reads as an edge
            v = cxy - XY[i]
            L = np.hypot(*v) or 1
            p0 = XY[i] + v / L * g["r"][i] * 1.1
            ax.plot([p0[0], cxy[0]], [p0[1], cxy[1]], color="#9AA5A2", lw=0.6, ls=(0, (1, 1.6)), zorder=2)
        bb = dict(boxstyle="round,pad=0.1", fc=bg, ec="none", alpha=1.0) if o.label_box else None
        ax.text(cxy[0], cxy[1], g["texts"][j], fontsize=o.font_size, ha="center", va="center", zorder=4, bbox=bb,
                rotation=g["rot"][j], rotation_mode="anchor" if g["rot"][j] else "default",
                fontweight="bold" if g["hl"][i] else "normal", color=ink)
    if o.title:
        fig.text(0.01, 1 - 0.08 / g["H"], o.title, fontsize=o.font_size + 5, fontweight="bold", va="top", color=ink)
    if o.show_legend:
        lax = fig.add_axes([g["PW"] / g["W"] + 0.004, 0.02, g["legend_w"] / g["W"] - 0.008, 0.96], facecolor=bg)
        lax.axis("off")
        h = []
        present = [k for k in ("enzyme", "kinase", "phosphatase", "peptidase", "transporter", "metabolite", "reaction")
                   if k in set(N["kind"])]
        for k in present:
            if g["kegg"]:
                mk = "s" if k not in ("metabolite", "reaction") else ("o" if k == "metabolite" else "s")
            elif o.shape_scheme == "All circles":
                mk = "o" if k != "reaction" else "s"
            else:
                mk = MPL_MARK[k] if MPL_MARK[k] not in ("trap", "ellipse") else ("p" if MPL_MARK[k] == "trap" else "o")
            h.append(Line2D([], [], marker=mk, ls="", mfc=bg, mec=ink, ms=8, label=TYPE_LABEL[k]))
        h.append(Line2D([], [], ls="", label=" "))
        if o.color_by.startswith("z-score"):
            lo, _, hi = PALETTES[o.palette]
            h += [Line2D([], [], marker="o", ls="", mfc=hi, mec="#333", ms=8, label="Measured: up"),
                  Line2D([], [], marker="o", ls="", mfc=lo, mec="#333", ms=8, label="Measured: down")]
            if o.prediction:
                h += [Line2D([], [], marker="o", ls="", mfc="#FBD9B8", mec=PRED_UP, mew=1.5, ms=8, label="Predicted increase"),
                      Line2D([], [], marker="o", ls="", mfc="#C6DDF0", mec=PRED_DOWN, mew=1.5, ms=8, label="Predicted decrease")]
            h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="#8A9593", ms=8, label="Not measured"))
        elif g["legend_items"]:
            h += [Patch(fc=c, ec="#333", label=t) for t, c in g["legend_items"]]
        if o.edge_color == "IPA relationship":
            st_ = (lambda k_: {"down": "--", "inconsistent": ":", "none": (0, (1, 2))}.get(k_, "-")) if o.effect_linestyle else (lambda k_: "-")
            h += [Line2D([], [], ls="", label=" "),
                  Line2D([], [], color=PRED_UP, lw=2, ls=st_("up"), label="Leads to increase"),
                  Line2D([], [], color=PRED_DOWN, lw=2, ls=st_("down"), label="Leads to decrease"),
                  Line2D([], [], color=INCONS, lw=2, ls=st_("inconsistent"), label="Inconsistent with data"),
                  Line2D([], [], color=NOPRED, lw=2, ls=st_("none"), label="Effect not predicted")]
        elif o.edge_color == "By relation type":
            h += [Line2D([], [], ls="", label=" "), Line2D([], [], color="#C0392B", lw=2, label="produces"),
                  Line2D([], [], color="#2C6FB7", lw=2, label="substrate of")]
        if (E["relation"] == "PPI").any():
            h.append(Line2D([], [], color="#B0B0B0", lw=1.5, ls="--", label="Protein-protein"))
        leg = lax.legend(handles=h, loc="upper left", frameon=False, fontsize=o.font_size, handlelength=1.8)
        for t in leg.get_texts():
            t.set_color(ink)
        if g["cbar"]:
            lab, pal, vmin, vmax = g["cbar"]
            cax = fig.add_axes([g["PW"] / g["W"] + 0.02, 0.05, 0.09 * g["legend_w"] / g["W"], 0.17])
            cmap = mc.LinearSegmentedColormap.from_list("p", pal) if isinstance(pal, tuple) else plt.get_cmap(pal)
            cbr = fig.colorbar(plt.cm.ScalarMappable(norm=mc.Normalize(vmin, vmax), cmap=cmap), cax=cax)
            cbr.set_label(lab, fontsize=o.font_size, color=ink)
            cbr.ax.tick_params(labelsize=o.font_size - 1, colors=ink)
    return fig


# ---------------------------------------------------------------------------------------------- plotly
def render_plotly(N: pd.DataFrame, E: pd.DataFrame, o: NetOptions, g: dict | None = None):
    import plotly.graph_objects as go
    g = g or geometry(N, E, o)
    XY, idx = g["XY"], g["idx"]
    dark = o.theme == "Dark"
    ink = "#F2F4F3" if dark else "#1F2A28"
    bg = "#15191C" if dark else "white"
    fig = go.Figure()
    shapes, ann = [], []
    if g["geom"] and o.show_bands:
        for k, (b, (lo, hi)) in enumerate(g["geom"].items()):
            shapes.append(dict(type="rect", x0=0, x1=g["PW"], y0=lo * g["PH"], y1=hi * g["PH"], line=dict(width=0),
                               fillcolor=("#1E2428" if dark else "#F4F8FB") if k % 2 == 0 else bg, layer="below"))
            ann.append(dict(x=0.08, y=hi * g["PH"] - 0.08, text=f"<b>{b}</b>", showarrow=False, xanchor="left",
                            yanchor="top", font=dict(size=o.font_size * 1.6 + 1, color="#7F8C8D")))
    groups = {}
    for k_e, (_, e) in enumerate(E.iterrows()):
        pts = g["routes"][k_e]
        if pts is None:
            continue
        col, ls, w = _edge_style(e, o)
        key = (col, ls if isinstance(ls, str) else "dot", round(w, 2))
        xs, ys = groups.setdefault(key, ([], []))
        xs += list(pts[:, 0]) + [None]
        ys += list(pts[:, 1]) + [None]
        if o.arrows and e["relation"] != "PPI" and len(pts) >= 2:
            ann.append(dict(x=pts[-1, 0], y=pts[-1, 1], ax=pts[-2, 0], ay=pts[-2, 1], xref="x", yref="y", axref="x",
                            ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=w, arrowcolor=col, text=""))
    dash = {"-": "solid", "--": "dash", ":": "dot", "dot": "dot"}
    for (col, ls, w), (xs, ys) in groups.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=w, dash=dash.get(ls, "solid")),
                                 opacity=o.edge_alpha, hoverinfo="skip", showlegend=False))
    for kind in N["kind"].unique():
        sel = np.where(N["kind"].values == kind)[0]
        sym = "square" if (g["kegg"] and kind != "metabolite") else ("circle" if o.shape_scheme == "All circles" or g["kegg"]
                                                                      else PLY_MARK.get(kind, "circle"))
        hov = []
        for k in sel:
            r = N.iloc[k]
            z = f"{r['z']:.2f}" if np.isfinite(r["z"]) else "not measured"
            hov.append(f"<b>{r['label']}</b><br>{TYPE_LABEL.get(kind, kind)} · {r['organelle']}<br>z = {z} {r['source']}")
        fig.add_trace(go.Scatter(x=XY[sel, 0], y=XY[sel, 1], mode="markers", name=TYPE_LABEL.get(kind, kind),
                                 marker=dict(symbol=sym, size=g["size_pt"][sel] * 1.33, color=[g["fill"][k] for k in sel],
                                             line=dict(color=[o.highlight_color if g["hl"][k] else g["edgec"][k] for k in sel],
                                                       width=[2.4 if g["hl"][k] else 1 for k in sel])),
                                 hovertext=hov, hoverinfo="text"))
    for j, i in enumerate(g["li"]):
        if g["placed"][j]:
            ann.append(dict(x=g["centers"][j][0], y=g["centers"][j][1], text=g["texts"][j], showarrow=False,
                            textangle=-g["rot"][j], font=dict(size=o.font_size * 1.33, color=ink)))
    if g["inside"]:
        for k, kind in enumerate(N["kind"]):
            if kind not in ("metabolite", "reaction"):
                ann.append(dict(x=XY[k, 0], y=XY[k, 1], text=g["alltexts"][k], showarrow=False,
                                font=dict(size=o.font_size * 1.2, color="#111")))
    # true size (1 in = 96 px, fonts in px = pt x 1.33): the geometry, and so the collision-free label
    # placement, is identical to the preview and the exports
    fig.update_layout(autosize=False, width=int(g["PW"] * 96) + (230 if o.show_legend else 20), height=int(g["H"] * 96),
                      shapes=shapes, annotations=ann, template="plotly_dark" if dark else "simple_white",
                      title=dict(text=o.title, x=0.01) if o.title else None, plot_bgcolor=bg, paper_bgcolor=bg,
                      margin=dict(l=8, r=8, t=40 if o.title else 8, b=8), showlegend=o.show_legend,
                      legend=dict(title="Molecule type", x=1.01, y=1))
    fig.update_xaxes(visible=False, range=[0, g["PW"]])
    fig.update_yaxes(visible=False, range=[0, g["PH"]], scaleanchor="x", scaleratio=1)
    return fig


# ---------------------------------------------------------------------------------------------- exports
def export_tables(N, E):
    return N.to_csv(index=False).encode(), E.to_csv(index=False).encode()


def to_cyjs(N, E, g=None) -> bytes:
    els = []
    for k, r in N.reset_index(drop=True).iterrows():
        d = {"id": r["id"], "label": r["label"], "type": r["kind"], "z": None if not np.isfinite(r["z"]) else float(r["z"]),
             "compartment": r["organelle"], "predicted": float(r["predicted"])}
        if g is not None:
            d["module"] = int(g["module"].get(r["id"], 0)) + 1
            d["color"] = g["fill"][k]
        el = {"data": d}
        if g is not None:
            el["position"] = {"x": float(g["XY"][k, 0] * 100), "y": float((g["PH"] - g["XY"][k, 1]) * 100)}
        els.append(el)
    for k, e in E.iterrows():
        els.append({"data": {"id": f"e{k}", "source": e["source"], "target": e["target"], "relation": e["relation"],
                             "ipa_effect": e["ipa"], "reaction": e["rxn"]}})
    return json.dumps({"elements": els}, indent=1).encode()


def to_graphml(N, E) -> bytes:
    import networkx as nx
    G = nx.DiGraph()
    for _, r in N.iterrows():
        G.add_node(r["id"], label=str(r["label"]), type=r["kind"], compartment=r["organelle"],
                   z=float(r["z"]) if np.isfinite(r["z"]) else 0.0, measured=bool(r["measured"]),
                   predicted=float(r["predicted"]))
    for _, e in E.iterrows():
        G.add_edge(e["source"], e["target"], relation=e["relation"], ipa_effect=e["ipa"], reaction=e["rxn"])
    b = io.BytesIO()
    nx.write_graphml(G, b)
    return b.getvalue()
