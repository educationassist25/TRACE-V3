"""TRACE validation tests: parsing, mapping, recovery of planted biology, null calibration, no leakage."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trace_core.data import build_design, load_demo, make_layer, norm_hmdb  # noqa: E402
from trace_core.demo import make_demo  # noqa: E402
from trace_core.engine import run_trace  # noqa: E402
from trace_core.kb import load_kb  # noqa: E402
from trace_core.kb_build import parse_gpr  # noqa: E402
from trace_core.stats import product_normal_p  # noqa: E402


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def _layers(kb, folder, drop=None):
    dfs, meta = load_demo(folder)
    for k, cols in (drop or {}).items():
        dfs[k] = dfs[k].drop(columns=cols)
    layers = {k: make_layer(k, dfs[k], meta, kb, batch_correct=(k == "met"), batch_col="Batch") for k in dfs}
    return layers, build_design(meta, layers, "Tumor", "Control")


@pytest.fixture(scope="module")
def demo_result(kb):
    layers, des = _layers(kb, os.path.join(ROOT, "demo_data"))
    return run_trace(layers, des, kb, {"B": 400, "B_topo": 400}), layers, des


def test_gpr_parser():
    assert parse_gpr("A and (B or C)") == [["A", "B"], ["A", "C"]]
    assert parse_gpr("A or B") == [["A"], ["B"]]
    assert parse_gpr("") == []


def test_identifier_normalisation():
    assert norm_hmdb("HMDB00122") == "HMDB0000122"
    assert norm_hmdb("hmdb0000122") == "HMDB0000122"


def test_product_normal_matches_simulation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400000)
    y = 0.3 * x + np.sqrt(1 - 0.09) * rng.normal(size=400000)
    emp = (np.abs(x * y) >= 3).mean()
    assert abs(product_normal_p(np.array([3.0]), np.array([0.3]))[0] - emp) < 0.002


def test_tier_and_mapping(demo_result):
    res, layers, des = demo_result
    assert des.tier == 1
    assert (layers["met"].feat["map_rule"] != "unmapped").mean() > 0.95


def test_recovers_bottleneck(demo_result):
    rx = demo_result[0].reactions
    r = rx[rx["name"] == "L-Citrulline:L-aspartate ligase (AMP-forming)"].iloc[0]
    assert r["significant"] and r["pattern"] == "Candidate bottleneck" and r["E"] < 0


def test_recovers_glycolysis_throughput(demo_result):
    rx = demo_result[0].reactions
    g = rx[rx["significant"] & rx["subsystem"].str.contains("Glycolysis")]
    assert (g["pattern"] == "Throughput-consistent (up)").any()


def test_pathway_classes(demo_result):
    pw = demo_result[0].pathways.set_index("pathway")
    assert pw.loc["Glycolysis / Gluconeogenesis", "class"] == "Shared-concordant"
    assert pw.loc["Cholesterol biosynthesis 1 (Bloch pathway)", "class"] == "mRNA-specific"
    fao = pw[pw.index.str.contains("Beta oxidation")]
    assert (fao["class"] == "Protein-specific").any()


def test_concordance_classes(demo_result):
    c = demo_result[0].concordance.set_index("gene")["class"]
    assert c.get("ASS1") == "Coupled"
    assert sum(c.get(g) == "Protein-only (post-transcriptional)" for g in ["CPT1A", "ACADM", "HADHA", "HADHB", "ACAA2"]) >= 3
    assert sum(c.get(g) == "mRNA-only (buffered)" for g in ["HMGCR", "HMGCS1", "SQLE", "FDPS", "LSS"]) >= 3


def test_metabolite_driven_step(demo_result):
    rx = demo_result[0].reactions
    r = rx[rx["name"] == "L-glutamine amidohydrolase"]
    assert r["metabolite_driven"].any()


def test_partial_overlap_runs(kb):
    layers, des = _layers(kb, os.path.join(ROOT, "demo_data"),
                          drop={"prot": ["S03", "S08", "S15", "S20"], "met": ["S05", "S11", "S17", "S23"]})
    assert des.tier == 2
    res = run_trace(layers, des, kb, {"B": 200, "B_topo": 200})
    assert res.reactions["significant"].sum() > 0


def test_null_calibration(kb, tmp_path):
    """Global null: no planted effects -> few nominal and essentially no FDR-significant reactions."""
    make_demo(kb, str(tmp_path), seed=99, effects=False)
    layers, des = _layers(kb, str(tmp_path))
    res = run_trace(layers, des, kb, {"B": 400, "B_topo": 400})
    rx = res.reactions
    assert (rx["p_label"] < 0.05).mean() < 0.10
    assert rx["significant"].sum() <= 3
    assert (res.pathways["q_combined"] < 0.05).sum() <= 2


def test_signature_no_leakage(kb, tmp_path):
    from trace_core.signature import run_signature
    make_demo(kb, str(tmp_path), seed=5, effects=False)
    layers, des = _layers(kb, str(tmp_path))
    s = run_signature(layers, des, n_repeats=3, n_perm=10)
    assert s["status"] == "ok" and s["auc_mean"] < 0.8


def test_exports(demo_result, kb):
    from trace_core import report as R
    res = demo_result[0]
    assert len(R.excel(res, {"kb_version": kb.version})) > 10000
    assert "Human-GEM" in R.methods_text(res, kb, {})
    m = R.manifest(res, {"x": b"1"}, kb, {})
    assert m["design"]["tier"] == 1


def test_ipa_canonical(demo_result, kb):
    from trace_core.ipa import canonical_pathways
    cp = canonical_pathways(demo_result[0], kb)
    g = cp.set_index("pathway").loc["Glycolysis / Gluconeogenesis"]
    assert g["p"] < 1e-6 and g["prediction"] == "Increased activity"


@pytest.mark.parametrize("layout", ["Subcellular (IPA-style)", "Hierarchical flow (top-down)", "Force-directed",
                                    "Shell by molecule type", "Grid by molecule type", "Kamada-Kawai",
                                    "Radial (focus-centred)", "Bipartite columns (substrates | enzymes | products)"])
def test_network_layouts_have_no_overlaps(demo_result, kb, layout):
    from trace_core import netview as NV
    o = NV.NetOptions(scope="Pathway", pathway="Glycolysis / Gluconeogenesis", layout=layout, max_reactions=25)
    N, E = NV.build_view(demo_result[0], kb, o)
    g = NV.geometry(N, E, o)
    XY, r = g["XY"], g["r"]
    d = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1)) + np.eye(len(XY)) * 99
    assert (d >= (r[:, None] + r[None, :]) * 0.98).all()          # no node overlaps
    C, li, placed = g["centers"], g["li"], g["placed"]
    from trace_core.layout import text_size
    boxes = [(c[0], c[1], *text_size(g["texts"][j], o.font_size)) for j, c in enumerate(C) if placed[j]]
    n_ov = 0
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            xa, ya, wa, ha = boxes[a]
            xb, yb, wb, hb = boxes[b]
            if abs(xa - xb) < (wa + wb) / 2 - 0.01 and abs(ya - yb) < (ha + hb) / 2 - 0.01:
                n_ov += 1
    assert n_ov <= max(1, len(boxes) // 50)                      # essentially no label collisions
    fig = NV.render_mpl(N, E, o, g)
    from trace_core.pubfig import export
    assert len(export(fig, "png", 150)) > 10000
    NV.render_plotly(N, E, o, g)


def test_all_export_formats(demo_result):
    from trace_core import pubfig as PF
    for f in PF.FORMATS.values():
        assert len(PF.export(PF.pattern_grid(demo_result[0].reactions), f, 300)) > 1000


def test_all_style_presets_render(demo_result, kb):
    from trace_core import netview as NV
    from trace_core.pubfig import export
    base = NV.NetOptions(scope="Pathway", pathway="Glycolysis / Gluconeogenesis", max_reactions=20)
    for name in NV.PRESETS:
        o = NV.apply_preset(base, name)
        N, E = NV.build_view(demo_result[0], kb, o)
        g = NV.geometry(N, E, o)
        assert len(export(NV.render_mpl(N, E, o, g), "png", 60)) > 5000, name
        NV.render_plotly(N, E, o, g)


@pytest.mark.parametrize("scope", ["Pathway", "Coherent reactions"])
def test_network_studio_nothing_overlaps(demo_result, kb, scope):
    """v1.1.2: labels never sit on labels or nodes, and edges are routed around every node they do not join."""
    from trace_core import netview as NV
    res = demo_result[0]
    o = NV.NetOptions(scope=scope, pathway="Glycolysis / Gluconeogenesis")
    o.seeds = ["ASS1"]
    N, E = NV.build_view(res, kb, o)
    g = NV.geometry(N, E, o)
    c = g["collisions"]
    assert c["checked"]
    assert c["label_label"] == 0 and c["label_node"] == 0 and c["edge_node"] == 0, c


def test_pathway_dotplot_dots_never_touch(demo_result):
    """Interactive dots are circles in grid units with radius <= 0.42 on a 1:1 grid, so neighbours keep a gap
    at any window width; the publication version caps diameters at 80 % of the row / column pitch."""
    from trace_core import figures as F
    from trace_core import pubfig as PF
    res = demo_result[0]
    fig = F.pathway_dotplot(res.pathways, 60)
    circles = [s for s in fig.layout.shapes if s.type == "circle"]
    assert circles
    assert max((s.x1 - s.x0) / 2 for s in circles) <= 0.42 + 1e-9
    assert fig.layout.yaxis.scaleanchor == "x"
    for w, h in ((7.2, None), (4.5, 4.0)):
        f = PF.pathway_dotplot(res.pathways, 40, w, h, 7)
        ax = f.axes[0]
        pos = ax.get_position()
        n = len(ax.get_yticks())
        pitch_pt = min(pos.height * f.get_figheight() / (n + 0.4), pos.width * f.get_figwidth() / 4) * 72
        sizes = np.concatenate([c.get_sizes() for c in ax.collections if len(c.get_sizes())])
        assert np.sqrt(sizes.max()) <= 0.8 * pitch_pt + 1e-6


def _text_boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts + ax.get_yticklabels():
            if t.get_text().strip():
                out.append((t.get_text(), t.get_window_extent(r)))
    return out


@pytest.mark.parametrize("width", [None, 6.0])
def test_cascade_and_targets_full_text_no_overlap(demo_result, kb, width):
    """v1.1.3: full reaction names (never truncated) and no two text boxes overlap, at natural and narrow widths."""
    from trace_core import network as NW
    from trace_core import pubfig as PF
    res = demo_result[0]
    rx = res.reactions
    sel = rx[rx["subsystem"].str.contains("Glycolysis", regex=False)].sort_values("p_label").head(12)
    nodes, edges = NW.build_graph(res, kb)
    tg = NW.rank_targets(res, kb, NW.propagate(nodes, edges, B=50, seed=1))
    for fig, names in ((PF.cascade_alluvial(sel, w=width), sel["name"]),
                       (PF.targets_bar(tg, 15, w=width), tg.head(15)["best_reaction"])):
        boxes = _text_boxes(fig)
        joined = " ".join(t for t, _ in boxes).replace("- ", "-")
        assert "…" not in joined
        for n in names:   # every word of every full name is drawn
            assert all(wd in joined for wd in str(n).split()), n
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                assert not (ox > 1 and oy > 1), (boxes[i][0], boxes[j][0])
