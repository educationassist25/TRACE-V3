"""
Omnix Integrix - tri-omics integration (transcriptomics, proteomics, metabolomics) (standalone Streamlit app, v1).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

# make the app's own folder importable however it is launched (streamlit run from another folder, an Omnix
# multipage portal, st.navigation, Render/Cloud working directories)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from trace_core.compat import patch_streamlit  # noqa: E402

ST_HAS = patch_streamlit(st)      # older Streamlit: unsupported options are dropped instead of crashing
from trace_core import __version__
from trace_core import figures as F
from trace_core import report as R
from trace_core.cascade import PATTERN_HELP
from trace_core.data import LAYER_NAMES, build_design, cramers_v, make_layer, read_table
from trace_core.engine import run_trace
from trace_core.kb import load_kb
from trace_core import ui as UI

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_data")
LOGO = os.path.join(HERE, "assets", "omnix_logo_web.png")

st.set_page_config(page_title="Omnix™ Integrix", page_icon=LOGO if os.path.exists(LOGO) else None,
                   layout="wide")
try:
    MODE = "dark" if (st.context.theme.type or "light") == "dark" else "light"
except Exception:  # older Streamlit
    MODE = "light"
st.markdown(UI.css(MODE), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _demo_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DEMO, name))


@st.cache_data(show_spinner=False)
def _demo_bytes(name: str) -> bytes:
    return open(os.path.join(DEMO, name), "rb").read()


@st.cache_resource(show_spinner="Loading knowledge base")
def get_kb():
    return load_kb()


def show(fig, name="trace_figure", notitle=False):
    cfg = F.config(st.session_state.get("fig_fmt", "png"), st.session_state.get("fig_scale", 6), name)
    fig = UI.theme_figure(fig, MODE)
    if notitle:   # the surrounding card already carries the title
        fig.update_layout(title_text="", margin_t=min(40, fig.layout.margin.t or 40))
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, theme=None)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)


def show_mpl(fig, max_scale: float = 1.15, cache_key=None):
    """Show a matplotlib figure as a crisp vector image (SVG, text converted to outlines so it renders exactly as
    laid out, at any zoom or screen density). It scales down with the window and never beyond 115 % of its
    natural size. `fig` may be a figure or a zero-argument builder; with `cache_key` the rendered SVG is kept for
    this session, so a rerun (any click) does not redraw it. Returns the natural size in inches."""
    import base64
    import matplotlib.pyplot as plt
    cache = st.session_state.setdefault("_svg_cache", {})
    key = (cache_key, MODE) if cache_key is not None else None
    if key is not None and key in cache:
        svg, w_in, h_in = cache[key]
    else:
        fig = fig() if callable(fig) else fig
        w_in, h_in = fig.get_size_inches()
        buf = io.BytesIO()
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        svg = UI.fluid_svg(buf.getvalue())
        if MODE == "dark":
            svg = UI.darken_svg(svg)
        if key is not None:
            if len(cache) > 24:
                cache.pop(next(iter(cache)))
            cache[key] = (svg, float(w_in), float(h_in))
    b64 = base64.b64encode(svg).decode()
    st.markdown(f'<div style="width:100%;text-align:center"><img src="data:image/svg+xml;base64,{b64}" '
                f'style="width:100%;max-width:{int(w_in * 96 * max_scale)}px;height:auto;display:inline-block"/></div>',
                unsafe_allow_html=True)
    return float(w_in), float(h_in)


def table(df, height=380, key=None):
    st.dataframe(df, height=height, hide_index=True, key=key)


def csv_button(df, name, label="Download table (CSV)", key=None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key)


def _fonts():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    pref = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "FreeSans", "Inter", "Carlito",
            "Times New Roman", "Liberation Serif", "DejaVu Serif", "FreeSerif"]
    return [f for f in pref if f in have] or ["DejaVu Sans"]


def download_panel(key: str, builder, w: float, h: float | None, fs: float = 7.0, name: str = "figure"):
    """Publication export for one figure: any format, 300-1200 DPI, custom size / font, or all formats
    at once. `builder(w, h, fs)` returns a matplotlib figure (rendered only when requested)."""
    from trace_core import pubfig as PF
    with st.expander("⬇ Download publication-quality figure (PNG · TIFF · JPEG · SVG · PDF · EPS)"):
        c = st.columns(3) + st.columns(3)          # two rows of three: readable at half-screen width
        fmt = c[0].selectbox("Format", list(PF.FORMATS) + ["All formats (ZIP)"], key=f"{key}_fmt")
        dpi = c[1].selectbox("DPI", [300, 600, 1200], index=1, key=f"{key}_dpi")
        font = c[2].selectbox("Font family", _fonts(), key=f"{key}_font")
        ww = c[3].number_input("Width (in)", 2.0, 30.0, float(w), 0.5, key=f"{key}_w")
        hh = c[4].number_input("Height (in, 0 = auto)", 0.0, 30.0, float(h or 0), 0.5, key=f"{key}_h")
        ff = c[5].number_input("Font (pt)", 4.0, 20.0, float(fs), 0.5, key=f"{key}_fs")
        t1, t2 = st.columns([1, 1])
        transp = t1.checkbox("Transparent background", key=f"{key}_tr")
        if t2.button("Render figure", key=f"{key}_go"):
            with st.spinner("Rendering"):
                PF.style(font, ff)
                fmts = list(PF.FORMATS.values()) if fmt.startswith("All") else [PF.FORMATS[fmt]]
                out = {}
                for f in fmts:
                    fig = builder(ww, hh or None, ff)
                    out[f"{name}.{f}"] = PF.export(fig, f, dpi, transp)
                if len(out) == 1:
                    (fn, data), = out.items()
                    st.session_state[f"_pub_{key}"] = (fn, data, PF.MIME[fn.rsplit('.', 1)[1]])
                else:
                    st.session_state[f"_pub_{key}"] = (f"{name}_all_formats.zip", R.bundle(out), "application/zip")
        if f"_pub_{key}" in st.session_state:
            fn, data, mime = st.session_state[f"_pub_{key}"]
            st.download_button(f"Download {fn} ({len(data) / 1e6:.1f} MB)", data, fn, mime=mime, key=f"{key}_btn",
                               type="primary")


kb = get_kb()

# ===================================================================================== setup (sidebar)
SRC_OPTS = ["Demo: paired tri-omics", "Demo: partially overlapping", "Upload my data"]
SRC_CAPS = ["3 layers, same 24 samples", "3 layers, some samples missing", "CSV / TSV / XLSX matrices"]
SECTION_NAMES = ["Results", "Networks & targets", "Advanced", "Data & export"]
SECTION_ICONS = [":material/insights:", ":material/hub:", ":material/science:", ":material/folder_open:"]
STEP_KEYS = ["step_data", "step_design", "step_stats", "step_engines"]


def _accordion(opened: str):
    """Only one setup step open at a time."""
    if st.session_state.get(opened):
        for k in STEP_KEYS:
            if k != opened:
                st.session_state[k] = False


source = st.session_state.get("src", SRC_OPTS[0])
up = {k: st.session_state.get(f"up_{k}") for k in ("rna", "prot", "met", "meta", "map")} \
    if source == "Upload my data" else {}

# ---------------------------------------------------------------- load raw tables
raw, meta, inputs = {}, None, {}
if source.startswith("Demo"):
    for k, f in (("rna", "demo_transcriptomics.csv"), ("prot", "demo_proteomics.csv"),
                 ("met", "demo_metabolomics.csv")):
        raw[k] = _demo_csv(f).copy()
        inputs[f] = _demo_bytes(f)
    meta = _demo_csv("demo_metadata.csv").copy()
    if source == "Demo: partially overlapping":
        drop_p = ["S03", "S08", "S15", "S20"]
        drop_m = ["S05", "S11", "S17", "S23"]
        raw["prot"] = raw["prot"].drop(columns=drop_p)
        raw["met"] = raw["met"].drop(columns=drop_m)
else:
    for k in ("rna", "prot", "met"):
        if up.get(k) is not None:
            up[k].seek(0)
            raw[k] = read_table(up[k])
            up[k].seek(0)
            inputs[up[k].name] = up[k].read()
    if up.get("meta") is not None:
        up["meta"].seek(0)
        meta = read_table(up["meta"])
    if up.get("map") is not None and "met" in raw:
        up["map"].seek(0)
        mp = read_table(up["map"])
        key = mp.columns[0]
        raw["met"] = raw["met"].merge(mp.rename(columns={key: raw["met"].columns[0]}), on=raw["met"].columns[0],
                                      how="left", suffixes=("", "_map"))

data_ok = "met" in raw and len(raw) >= 2 and meta is not None
n_samples = len(meta) if meta is not None else 0
layer_lbl = {"rna": "mRNA", "prot": "Protein", "met": "Metabolite"}
data_summary = (f"{source.split(': ')[-1] if source.startswith('Demo') else 'Uploaded'} · "
                f"{len(raw)} layers · {n_samples} samples") if data_ok else "Load the data tables"

with st.sidebar:
    _setup_head = st.empty()

    # ---- step 1: data
    with st.expander(f"**Data** · {data_summary}", key="step_data", expanded=not data_ok,
                     icon=":material/check_circle:" if data_ok else ":material/counter_1:",
                     on_change=_accordion, args=("step_data",)):
        source = st.radio("Dataset", SRC_OPTS, key="src", captions=SRC_CAPS, label_visibility="collapsed")
        id_cols = {}
        if source == "Upload my data":
            st.caption("Wide tables: one row per feature, one column per sample. The first column is the "
                       "identifier unless you choose another. Exports from the Omnix modules work directly.")
            st.file_uploader("Transcriptomics (genes)", type=["csv", "tsv", "txt", "xlsx"], key="up_rna")
            st.file_uploader("Proteomics (proteins)", type=["csv", "tsv", "txt", "xlsx"], key="up_prot")
            st.file_uploader("Metabolomics / Lipidomics (required)", type=["csv", "tsv", "txt", "xlsx"], key="up_met")
            st.file_uploader("Sample metadata (Sample, Group, optional Batch)", type=["csv", "tsv", "xlsx"],
                             key="up_meta")
            st.file_uploader("Optional: metabolite ID table (feature, HMDB / KEGG / ChEBI)",
                             type=["csv", "tsv", "xlsx"], key="up_map")
            for k, df in raw.items():
                id_cols[k] = st.selectbox(f"{LAYER_NAMES[k]} identifier column", list(df.columns), key=f"id_{k}")
        if raw:
            st.markdown(" ".join(f"`{layer_lbl[k]}`" for k in raw) + (f" `{n_samples} samples`" if n_samples else ""))

    # ---- step 2: study design
    sample_col, group_col, batch_col, case, ref = "Sample", "Group", None, None, None
    if meta is not None:
        mc = list(meta.columns)
        _gc = st.session_state.get("des_group", "Group" if "Group" in mc else mc[min(1, len(mc) - 1)])
        _groups = sorted(meta[_gc].astype(str).unique()) if _gc in mc else []
        _case = st.session_state.get("des_case", "Tumor" if "Tumor" in _groups else (_groups[-1] if _groups else ""))
        _ref = st.session_state.get("des_ref", next((g for g in _groups if g != _case), ""))
        design_summary = f"{_case} vs {_ref}"
    else:
        design_summary = "Needs sample metadata"
    with st.expander(f"**Study design** · {design_summary}", key="step_design", expanded=False,
                     icon=":material/check_circle:" if meta is not None else ":material/counter_2:",
                     on_change=_accordion, args=("step_design",)):
        if meta is not None:
            mc = list(meta.columns)
            sample_col = st.selectbox("Sample column", mc, index=mc.index("Sample") if "Sample" in mc else 0,
                                      key="des_sample")
            group_col = st.selectbox("Group column", mc,
                                     index=mc.index("Group") if "Group" in mc else min(1, len(mc) - 1),
                                     key="des_group")
            groups = sorted(meta[group_col].astype(str).unique())
            cg1, cg2 = st.columns(2)
            case = cg1.selectbox("Case", groups, key="des_case",
                                 index=groups.index("Tumor") if "Tumor" in groups else len(groups) - 1)
            ref = cg2.selectbox("Reference", [g for g in groups if g != case], key="des_ref")
            bopts = ["(none)"] + [c for c in mc if c not in (sample_col, group_col)]
            batch_col = st.selectbox("Batch column", bopts, key="des_batch",
                                     index=bopts.index("Batch") if "Batch" in bopts else 0)
            batch_col = None if batch_col == "(none)" else batch_col
            bc = {k: st.checkbox(f"Batch-correct {LAYER_NAMES[k]}", value=(k == "met" and batch_col is not None),
                                 key=f"bc_{k}", disabled=batch_col is None) for k in ("rna", "prot", "met")}
        else:
            st.caption("Upload a sample metadata table in step 1.")
            bc = {k: False for k in ("rna", "prot", "met")}

    # ---- step 3: statistics
    _B = st.session_state.get("st_B", 1000)
    _a = st.session_state.get("st_alpha", 0.05)
    _pt = st.session_state.get("st_pthr", 1.5)
    with st.expander(f"**Statistics** · {_B:,} perms · q {_a:g}", key="step_stats", expanded=False,
                     icon=":material/check_circle:", on_change=_accordion, args=("step_stats",)):
        B = st.select_slider("Label permutations", [200, 500, 1000, 2000, 5000], value=1000, key="st_B")
        c1_, c2_ = st.columns(2)
        alpha = c1_.selectbox("FDR (q)", [0.01, 0.05, 0.1], index=1, key="st_alpha")
        pthr = c2_.selectbox("Pattern |z|", [1.0, 1.5, 2.0, 2.5, 3.0], index=1, key="st_pthr")
        with st.popover("More options", icon=":material/tune:", width="stretch"):
            Bt = st.select_slider("Topology permutations", [200, 500, 1000, 2000], value=1000, key="st_Bt")
            moderated = st.checkbox("Moderated t (empirical Bayes)", True, key="st_mod")
            seed = st.number_input("Random seed", value=20261030, step=1, key="st_seed")
            transform = st.selectbox("Log transform", ["auto", "log2", "none"], key="st_tf",
                                     help="auto: log2 if values look like raw intensities")
            max_missing = st.slider("Max missing per group", 0.1, 0.9, 0.5, 0.05, key="st_miss")

    # ---- step 4: engines
    _eng = [n for n, k in (("Network", "en_net"), ("MOFA+", "en_fac"), ("Signature", "en_sig"))
            if st.session_state.get(k, True)]
    with st.expander(f"**Engines** · {', '.join(_eng) or 'cascade only'}", key="step_engines", expanded=False,
                     icon=":material/check_circle:", on_change=_accordion, args=("step_engines",)):
        do_net = st.toggle("Network, propagation and targets", True, key="en_net")
        do_fac = st.toggle("Latent factors (MOFA+)", True, key="en_fac")
        n_fac = st.slider("Maximum factors", 3, 15, 8, key="en_nfac", disabled=not do_fac)
        do_sig = st.toggle("Integrated biomarker signature", True, key="en_sig")
        sig_rep = st.slider("Outer CV repeats", 3, 20, 10, key="en_rep", disabled=not do_sig)
        sig_perm = st.select_slider("Signature permutations", [20, 50, 100, 200], value=50, key="en_perm",
                                    disabled=not do_sig)
        drug = st.file_uploader("Optional: druggable genes (one symbol per line)", type=["txt", "csv"], key="up_drug")
        st.session_state["fig_fmt"] = st.selectbox("Figure download format (camera icon)", ["png", "svg"], key="fg_fmt")
        st.session_state["fig_scale"] = st.select_slider("PNG scale (× screen size)", [4, 6, 8, 10], value=6,
                                                         key="fg_scale")

    ok = data_ok and case is not None and ref is not None
    n_ready = int(data_ok) + int(ok) + 2
    _setup_head.markdown(UI.setup_head(n_ready, 4), unsafe_allow_html=True)
    st.write("")
    run = st.button("Run analysis", type="primary", icon=":material/play_arrow:", disabled=not ok,
                    width="stretch", key="run_btn")
    st.caption(f"Knowledge base {kb.version} · Human-GEM {kb.manifest['sources']['Human-GEM'].get('version', '')}  \n"
               "Light / dark: ⋮ menu → Settings")

# ===================================================================================== top bar
_top = st.container(key="topbar")


def _go(section: int, tab: int | None = None):
    st.session_state["nav_sec"] = section
    if tab is not None:
        st.session_state[f"tabs_{section}"] = TAB_NAMES[tab]


TAB_NAMES = [":material/space_dashboard: Overview", ":material/table_chart: Data & mapping",
             ":material/compare_arrows: Concordance", ":material/bolt: Reactions", ":material/linear_scale: Cascade",
             ":material/share: Pathways", ":material/hub: Pathway maps & network", ":material/my_location: Targets",
             ":material/tune: Factors", ":material/fingerprint: Signature", ":material/download: Export",
             ":material/menu_book: Methods"]
SECTIONS = [[0, 3, 4, 5], [6, 7], [2, 8, 9], [1, 10, 11]]


SECTION_PAGES = ["Overview · Reactions · Cascade · Pathways", "Pathway maps & network · Targets",
                 "Concordance · Factors · Signature", "Data & mapping · Export · Methods"]


def draw_topbar(done: bool):
    """Option 1: brand header, then (after a run) four section cards and a row with the study chips and Export.
    The current section's pages follow as pill tabs. HTML + st.columns only, so it renders the same on every
    supported Streamlit version; each card is a real button stretched invisibly over the card."""
    tr = st.session_state.get("trace")
    sec = st.session_state.get("nav_sec", 0)
    kb_txt = f"Human-GEM {kb.manifest['sources']['Human-GEM'].get('version', '')}"
    with _top:
        with st.container(key="brandbar"):
            st.markdown(UI.topbar(__version__, LOGO, UI.status_pill(tr is not None, tr.get("seconds") if tr else None),
                                  kb_txt, with_nav=False), unsafe_allow_html=True)
        if not done:
            return
        with st.container(key="seccards"):
            cols = st.columns(4, gap="small")
            for i, nm in enumerate(SECTION_NAMES):
                with cols[i].container(key=f"seccard_{i}"):
                    st.markdown(UI.section_card(nm, SECTION_PAGES[i], i, i == sec), unsafe_allow_html=True)
                    st.button(f"Open {nm}", key=f"secbtn_{i}", on_click=_go, args=(i,), width="stretch")


if not ok:
    draw_topbar(False)
    st.markdown('<div class="trace-warn">Load a metabolomics layer, at least one of transcriptomics or proteomics, '
                'and a sample metadata table (step 1 in the setup panel), or choose a demo dataset.</div>',
                unsafe_allow_html=True)
    st.stop()

# ===================================================================================== run
@st.cache_resource(show_spinner=False)
def _result_store() -> dict:
    """One store for the whole server: finished analyses keyed by a hash of the input data and every setting."""
    return {}


def _run_key(raw: dict, meta, *settings) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(raw):
        df = raw[k]
        h.update(k.encode())
        h.update(repr((list(map(str, df.columns)), df.shape)).encode())
        h.update(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    if meta is not None:
        h.update(pd.util.hash_pandas_object(meta.astype(str), index=True).values.tobytes())
    h.update(repr(settings).encode())
    h.update(str(kb.version).encode())
    return h.hexdigest()


def _analyse(raw: dict, meta: pd.DataFrame, id_cols: dict, transform: str, max_missing: float, bc: dict,
             sample_col: str, group_col: str, batch_col, case: str, ref: str, params: dict, druggable: tuple,
             do_net: bool, do_fac: bool, n_fac: int, do_sig: bool, sig_rep: int, sig_perm: int, _prog=None):
    """The full Integrix analysis. Results are kept for the whole server (all sessions) under a hash of the input data
    and every setting, so a repeated run - the demo, a page reload, a second user - returns at once.
    (Not an st.cache_* function: those replay the progress-bar calls and fail across sessions.)"""
    import threading
    store = _result_store()
    key = _run_key(raw, meta, id_cols, transform, max_missing, bc, sample_col, group_col, batch_col, case, ref,
                   params, druggable, do_net, do_fac, n_fac, do_sig, sig_rep, sig_perm)
    if key in store:
        return store[key]
    pending = store.setdefault("_pending", {})
    with store.setdefault("_lock", threading.Lock()):
        ev = pending.get(key)
        mine = ev is None
        if mine:
            ev = pending[key] = threading.Event()
    if not mine:                             # the same analysis is already running (e.g. the background demo)
        t_w = time.time()
        while not ev.wait(0.5):
            if _prog is not None:
                _prog.progress(min(0.95, 0.1 + (time.time() - t_w) / 60), text="Finishing the analysis already in progress")
        if key in store:
            return store[key]
        with store["_lock"]:                 # it failed: compute here
            pending[key] = ev = threading.Event()
    try:
        out = _compute(raw, meta, id_cols, transform, max_missing, bc, sample_col, group_col, batch_col, case, ref,
                       params, druggable, do_net, do_fac, n_fac, do_sig, sig_rep, sig_perm, _prog)
        results = [k for k in store if not k.startswith("_")]
        if len(results) >= 8:                # keep memory bounded: drop the oldest analysis
            store.pop(results[0])
        store[key] = out
        return out
    finally:
        ev.set()
        pending.pop(key, None)


def _compute(raw, meta, id_cols, transform, max_missing, bc, sample_col, group_col, batch_col, case, ref, params,
             druggable, do_net, do_fac, n_fac, do_sig, sig_rep, sig_perm, _prog=None):
    say = (lambda f, m: _prog.progress(f, text=m)) if _prog is not None else (lambda f, m: None)
    say(0.02, "Preparing layers")
    layers = {}
    for k, df in raw.items():
        layers[k] = make_layer(k, df, meta, kb, id_col=id_cols.get(k), transform=transform,
                               max_missing=max_missing, batch_correct=bc[k], sample_col=sample_col,
                               group_col=group_col, batch_col=batch_col)
    design = build_design(meta, layers, case, ref, sample_col, group_col, batch_col)
    res = run_trace(layers, design, kb, params, progress=lambda f, m: say(min(0.6, 0.6 * f), f"Engine A · {m}"))
    extras = {"kb_version": kb.version}
    seed = params["seed"]
    if do_net:
        from trace_core import network as N
        say(0.65, "Engine C · network propagation and targets")
        drug_set = set(druggable) if druggable else None
        if drug_set:
            drug_set = {g for g in {x for gs in res.cascade.rx_genes.values() for x in gs} if g.upper() in drug_set}
        nodes, edges = N.build_graph(res, kb)
        prop = N.propagate(nodes, edges, B=200, seed=int(seed))
        extras["targets"] = N.rank_targets(res, kb, prop, drug_set)
        extras["subnet"] = N.subnetwork(res, prop, edges)
        extras["prop"] = prop
        extras["network_params"] = {"restart": 0.3, "null": "degree-binned seed permutation, B=200"}
    if do_fac:
        from trace_core import factors as FA
        say(0.75, "Engine B · MOFA+ latent factors")
        try:
            extras["factors"] = FA.run_factors(layers, design, n_factors=n_fac, seed=int(seed) % 10000)
            extras["factor_enrichment"] = FA.factor_enrichment(extras["factors"], layers, kb)
            extras["factor_params"] = {"max_factors": n_fac, "top_var": 1500, "restarts": 3}
        except Exception as e:  # keep the rest of the results
            extras["factors_error"] = str(e)
    if do_sig:
        from trace_core import signature as SG
        say(0.85, "Engine D · nested cross-validated signature")
        extras["signature"] = SG.run_signature(layers, design, n_repeats=sig_rep, n_perm=sig_perm,
                                               seed=int(seed) % 10000)
        extras["signature_params"] = {"keepX_grid": [5, 10, 20], "repeats": sig_rep, "permutations": sig_perm}
    return res, layers, extras


def _run_args():
    druggable_ = ()
    if drug is not None:
        druggable_ = tuple(sorted({x.strip().upper() for x in drug.getvalue().decode(errors="ignore")
                                   .replace(",", "\n").split("\n") if x.strip()}))
    params_ = dict(B=int(B), B_topo=int(Bt), alpha=float(alpha), pattern_thr=float(pthr), moderated=moderated,
                   seed=int(seed))
    return (raw, meta, dict(id_cols), transform, float(max_missing), dict(bc), sample_col, group_col, batch_col,
            case, ref, params_, druggable_, bool(do_net), bool(do_fac), int(n_fac), bool(do_sig), int(sig_rep),
            int(sig_perm))


if run:
    t0 = time.time()
    prog = st.progress(0.0, text="Starting the analysis")
    res, layers, extras = _analyse(*_run_args(), _prog=prog)
    st.session_state["trace"] = {"res": res, "layers": layers, "extras": extras, "inputs": inputs,
                                 "run_id": time.time(), "seconds": time.time() - t0}
    prog.empty()

if "trace" not in st.session_state:
    draw_topbar(False)
    st.markdown('<div class="trace-note">Check the four setup steps in the left panel, then press '
                '<b>Run analysis</b>. The demo takes about 20-40 seconds.</div>', unsafe_allow_html=True)
    st.stop()

draw_topbar(True)
S = st.session_state["trace"]
res, layers, extras = S["res"], S["layers"], S["extras"]
rx, pw = res.reactions, res.pathways
lname = {"rna": "mRNA", "prot": "protein", "met": "metabolite"}
from trace_core import ipa as IPA  # noqa: E402
from trace_core import netview as NV  # noqa: E402
from trace_core import pubfig as PF  # noqa: E402
from trace_core.layout import LAYOUTS  # noqa: E402

if "canonical" not in S:
    S["canonical"] = IPA.canonical_pathways(res, kb, res.params["alpha"])
cp_tbl = S["canonical"]

SEC = st.session_state.get("nav_sec", 0)
SEC_IDX = SECTIONS[SEC]
_tkey = f"tabs_{SEC}"
with st.container(key="pagebar"):
    _pb = st.columns([5, 1], vertical_alignment="center")
    _pb[0].markdown(UI.study_line(res.design.tier_text, res.design.case, res.design.ref, res.params["alpha"]),
                    unsafe_allow_html=True)
    _pb[1].button("Export", icon=":material/download:", key="top_export", on_click=_go, args=(3, 10),
                  width="stretch")
try:  # Streamlit >= 1.50: remember the active tab and render only that tab (much faster)
    _sub = st.tabs([TAB_NAMES[i] for i in SEC_IDX], key=_tkey, on_change="rerun")
    LAZY = True
except TypeError:
    _sub = st.tabs([TAB_NAMES[i] for i in SEC_IDX])
    LAZY = False
_hidden = st.container()          # tabs of other sections render nothing
tabs = [(_sub[SEC_IDX.index(i)] if i in SEC_IDX else _hidden) for i in range(len(TAB_NAMES))]


def on(i: int) -> bool:
    if i not in SEC_IDX:
        return False
    if not LAZY:
        return True
    return st.session_state.get(_tkey, TAB_NAMES[SEC_IDX[0]]) in (TAB_NAMES[i], None)


# ------------------------------------------------------------------------------------------- network UI
def _preset_cb(prefix):
    name = st.session_state.get(f"{prefix}_preset")
    for k, v in NV.PRESETS.get(name, {}).items():
        st.session_state[f"{prefix}_{k}"] = v


SCOPE_LABEL = {"Pathway": "Pathway map (IPA-style)", "Coherent reactions": "Coherent reactions",
               "Enzyme neighbourhood": "Enzyme / metabolite neighbourhood", "Molecule list": "My molecule list",
               "Top targets": "Top prioritised targets"}


def _img(data):
    try:
        st.image(data, width="stretch")
    except TypeError:  # older Streamlit
        st.image(data, use_container_width=True)


def network_ui(prefix: str):
    """One studio for IPA-style pathway maps and every other network view. Scope controls sit on top (they
    apply immediately); style settings live in a collapsible, tabbed panel whose two-column grids stack on
    narrow screens; the figure always spans the full width and scales with the window."""
    d = NV.NetOptions()
    K = lambda f: f"{prefix}_{f}"
    pchoices = pw.loc[pw["n_reactions_observed"] > 0].sort_values("p_combined")["pathway"].tolist()
    if K("scope") not in st.session_state:
        st.session_state[K("scope")] = "Pathway"
    if st.session_state.get(K("pathway")) not in pchoices:
        st.session_state[K("pathway")] = pchoices[0] if pchoices else ""
    for f, v in vars(d).items():   # initialise widget state once
        if K(f) not in st.session_state and not isinstance(v, (list, tuple)):
            st.session_state[K(f)] = v
    c1, c2 = st.columns([1, 1.6])
    c1.selectbox("Draw", list(SCOPE_LABEL), format_func=SCOPE_LABEL.get, key=K("scope"))
    scope = st.session_state[K("scope")]
    with c2:
        if scope == "Pathway":
            st.selectbox("Pathway (most significant first)", pchoices, key=K("pathway"))
        elif scope in ("Enzyme neighbourhood", "Molecule list"):
            cc = st.columns([3, 1])
            cc[0].text_input("Genes / metabolites (comma-separated)", key=K("seeds_txt"))
            if scope == "Enzyme neighbourhood":
                cc[1].number_input("Hops", 1, 2, key=K("hops"))
        elif scope == "Coherent reactions":
            st.caption("Reactions whose enzyme and metabolite changes agree (label and topology q below the FDR).")
        else:
            st.caption("The highest-ranked targets from Engine C and the reactions around them.")
    st.selectbox("Style preset (sets layout, shapes, colours and edges at once; fine-tune below)",
                 ["(custom)"] + list(NV.PRESETS), key=K("preset"), on_change=_preset_cb, args=(prefix,))
    with st.expander("Layout, colours, labels and edges", expanded=False):
        with st.form(f"{prefix}_form", border=False):
            t = st.tabs(["Content", "Layout", "Nodes & colours", "Labels", "Edges", "Canvas"])
            with t[0]:
                c = st.columns(2)
                c[0].slider("Maximum reactions", 5, 120, key=K("max_reactions"))
                c[1].slider("Isozymes shown per reaction", 1, 8, key=K("isozymes_per_reaction"))
                c[0].selectbox("Representation", ["Molecule view (IPA-style)", "Reaction view (bipartite)"], key=K("representation"))
                c[1].selectbox("Gene value", ["Best available (protein > mRNA)", "Protein", "mRNA"], key=K("value"))
                c[0].checkbox("Show unmeasured molecules", key=K("include_unmeasured"))
                c[1].checkbox("Hide isolated molecules", key=K("hide_isolated"))
                c[0].checkbox("Add protein-protein edges (STRING add-on)", key=K("include_ppi"),
                              disabled="ppi_string" not in kb.extra)
            with t[1]:
                c = st.columns(2)
                c[0].selectbox("Layout", LAYOUTS, key=K("layout"))
                c[1].selectbox("Subcellular bands", ["IPA (5 bands)", "Organelles (10 bands)"], key=K("band_detail"))
                c[0].slider("Spacing", 0.8, 2.5, key=K("spacing"), step=0.1)
                c[1].text_input("Focus molecules (radial layout)", "", key=K("focus_txt"))
                c[0].number_input("Layout seed", 1, 9999, key=K("seed"))
                c[1].checkbox("Auto-enlarge canvas so nothing overlaps", key=K("auto_size"))
                c[1].checkbox("Shade network modules (communities)", key=K("hulls"))
            with t[2]:
                c = st.columns(2)
                c[0].selectbox("Colour by", NV.COLOR_BY, key=K("color_by"))
                c[1].selectbox("Shape scheme", NV.SHAPES, key=K("shape_scheme"))
                c[0].selectbox("Palette", list(NV.PALETTES), key=K("palette"))
                c[1].slider("Colour scale |z| max", 1.0, 8.0, key=K("zmax"), step=0.5)
                c[0].selectbox("Node size by", ["Constant", "|z|", "Degree", "Target score"], key=K("size_mode"))
                c[1].slider("Node size (pt)", 6.0, 36.0, key=K("node_size"), step=1.0)
                c[0].checkbox("Predict unmeasured metabolites (MAP-like)", key=K("prediction"))
                c[1].slider("Prediction threshold |E|", 0.5, 3.0, key=K("pred_thr"), step=0.1)
                c[0].text_input("Highlight (comma-separated)", "", key=K("highlight_txt"))
                c[1].color_picker("Highlight outline", key=K("highlight_color"))
            with t[3]:
                c = st.columns(2)
                c[0].selectbox("Label", ["All", "Measured", "Significant", "Top N by |z|", "None"], key=K("label_mode"))
                c[1].slider("Top N", 5, 100, key=K("label_top_n"))
                c[0].slider("Label size (pt)", 5.0, 16.0, key=K("font_size"), step=0.5)
                c[1].selectbox("Font family", _fonts(), key=K("font_family"))
                c[0].checkbox("Label background", key=K("label_box"))
                c[1].checkbox("Gene names inside boxes (KEGG scheme)", key=K("label_inside"))
                c[0].checkbox("Hide labels that cannot be placed cleanly", key=K("hide_unplaced"))
            with t[4]:
                c = st.columns(2)
                c[0].selectbox("Edge colour", ["IPA relationship", "By relation type", "Single colour"], key=K("edge_color"))
                c[1].color_picker("Single colour", key=K("edge_single"))
                c[0].selectbox("Routing", NV.ROUTES, key=K("edge_route"))
                c[1].selectbox("Width by", ["Constant", "|enzyme evidence|"], key=K("edge_width_by"))
                c[0].slider("Edge width", 0.3, 4.0, key=K("edge_width"), step=0.1)
                c[1].slider("Edge opacity", 0.1, 1.0, key=K("edge_alpha"), step=0.05)
                c[0].checkbox("Arrow heads", key=K("arrows"))
                c[1].checkbox("Line style encodes effect (print)", key=K("effect_linestyle"))
            with t[5]:
                c = st.columns(2)
                c[0].text_input("Title", key=K("title"))
                c[1].selectbox("Theme", ["Light", "Dark"], key=K("theme"))
                c[0].slider("Width (in)", 5.0, 30.0, key=K("width_in"), step=0.5)
                c[1].slider("Height (in)", 4.0, 30.0, key=K("height_in"), step=0.5)
                c[0].checkbox("Legend", key=K("show_legend"))
                c[1].checkbox("Draw compartment bands", key=K("show_bands"))
                c[0].selectbox("Background", ["white", "transparent"], key=K("background"))
            st.form_submit_button("Apply", type="primary")
    vals = {f: st.session_state.get(K(f), v) for f, v in vars(d).items() if not isinstance(v, (list, tuple))}
    split = lambda t_: [x.strip() for x in str(st.session_state.get(K(t_), "")).split(",") if x.strip()]
    o = NV.NetOptions(**vals)
    o.seeds = split("seeds_txt") or ["ASS1"]
    o.highlight = split("highlight_txt")
    o.focus = split("focus_txt")
    o.size_range = d.size_range
    if o.scope == "Pathway" and not o.title:
        o.title = o.pathway
    ck = (S.get("run_id"), repr(o))
    cache = S.setdefault("net_cache", {})
    if ck not in cache:
        with st.spinner("Laying out the network"):
            N, E = NV.build_view(res, kb, o, extras.get("targets"))
            if N.empty:
                cache[ck] = (N, E, None, None)
            else:
                g = NV.geometry(N, E, o)
                import matplotlib.pyplot as plt
                buf = io.BytesIO()
                with plt.rc_context({"svg.fonttype": "path"}):
                    NV.render_mpl(N, E, o, g).savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
                plt.close("all")
                cache[ck] = (N, E, g, buf.getvalue())
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    N, E, g, png = cache[ck]
    if g is None:
        st.info("Nothing to draw for this selection. Widen the scope or show unmeasured molecules.")
        return
    cl = g.get("collisions", {})
    hard = cl.get("label_label", 0) + cl.get("label_node", 0) + cl.get("edge_node", 0)
    if not cl.get("checked", True):
        status = "labels placed radially outside the layout"
    elif hard == 0:
        status = "no overlapping labels or nodes, no edges through nodes"
    else:
        status = f"{hard} overlap(s) remain: raise Spacing or lower Maximum reactions"
    st.caption(f"{len(N)} molecules, {len(E)} relationships · canvas {g['W']:.1f} × {g['H']:.1f} in · {status}")
    view = st.radio("View", ["Exact preview (identical to the export)", "Interactive (hover and zoom)"],
                    horizontal=True, key=f"{prefix}_view", label_visibility="collapsed")
    if view.startswith("Exact"):
        import base64
        svg_ = UI.fluid_svg(UI.darken_svg(png) if MODE == "dark" else png)
        st.markdown(f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg_).decode()}" '
                    f'style="width:100%;height:auto;display:block"/>', unsafe_allow_html=True)
        st.caption("Vector preview, sharp at any zoom (Ctrl/Cmd + to enlarge). Same layout as the export; "
                   "downloads use the white publication style.")
    else:
        import streamlit.components.v1 as components
        fig = NV.render_plotly(N, E, o, g)
        cfg = F.config(st.session_state.get("fig_fmt", "png"), st.session_state.get("fig_scale", 6), f"{prefix}_network")
        html = fig.to_html(include_plotlyjs=True, full_html=False, config=cfg)
        components.html(html, height=min(int(fig.layout.height) + 30, 900), scrolling=True)
        st.caption("Shown at true size so no label overlaps: scroll inside the frame to pan, drag to zoom, "
                   "double-click to reset. The exact preview above scales to your window instead.")

    def build(w, h, fs_):
        import dataclasses
        o2 = dataclasses.replace(o, width_in=w, height_in=h or o.height_in * w / o.width_in, font_size=fs_)
        return NV.render_mpl(N, E, o2)
    download_panel(f"{prefix}_pub", build, o.width_in, o.height_in, o.font_size, f"{prefix}_network")
    c = st.columns(2)
    c[0].download_button("Cytoscape (.cyjs)", NV.to_cyjs(N, E, g), f"{prefix}_network.cyjs", key=f"{prefix}_cy",
                         width="stretch")
    c[1].download_button("GraphML", NV.to_graphml(N, E), f"{prefix}_network.graphml", key=f"{prefix}_gml",
                         width="stretch")
    nt, et = NV.export_tables(N, E)
    c[0].download_button("Nodes (CSV)", nt, f"{prefix}_nodes.csv", key=f"{prefix}_nodes", width="stretch")
    c[1].download_button("Edges (CSV)", et, f"{prefix}_edges.csv", key=f"{prefix}_edges", width="stretch")


# ===================================================================================== Overview
def _findings():
    """(kind, html) rows for the Key findings card, from the actual results."""
    E = UI._esc
    out = []
    sig = rx[rx["significant"]].drop_duplicates(subset=["enzymes", "E", "M"])
    for _, r in sig.head(8).iterrows():
        kind = ("Bottleneck" if r["pattern"] == "Candidate bottleneck" else
                "Throughput" if r["pattern"].startswith("Throughput") else "Reaction")
        out.append((kind, f"<b>{E(r['enzymes'].split(',')[0])}</b> &middot; {E(r['name'])} &mdash; <i>{E(r['pattern'])}</i>. "
                          f"{E(PATTERN_HELP.get(r['pattern'], ''))} <span class='stat'>E = {r['E']:.1f}, M = {r['M']:.1f}, "
                          f"q = {r['q_label']:.1e}</span>"))
    md = rx[rx["metabolite_driven"] & ~rx["significant"]].sort_values("p_balance")
    for _, r in md.head(3).iterrows():
        out.append(("Metabolite-driven", f"<b>{E(r['name'])}</b> &mdash; metabolite balance changes while the enzyme "
                    f"({E(r['enzymes'].split(',')[0])}) is unchanged. <span class='stat'>M = {r['M']:.1f}, "
                    f"q = {r['q_balance']:.1e}</span>"))
    for _, r in pw[pw["class"].str.startswith("Shared") | pw["class"].str.endswith("-specific")].head(5).iterrows():
        out.append(("Pathway", f"<b>{E(r['pathway'])}</b> &mdash; {E(r['class'])}. <span class='stat'>combined z = "
                    f"{r['z_combined']:.1f}, q = {r['q_combined']:.1e}</span>"))
    if len(cp_tbl):
        for _, r in cp_tbl[cp_tbl["prediction"].isin(["Increased activity", "Decreased activity"])].head(3).iterrows():
            out.append(("Canonical", f"<b>{E(r['pathway'])}</b>: {E(r['prediction'].lower())}. <span class='stat'>"
                        f"directional z = {r['z_directional']:.2f}, p = {r['p']:.1e}</span>"))
    if extras.get("targets") is not None:
        for _, r in extras["targets"].head(3).iterrows():
            out.append(("Target", f"<b>{E(r['gene'])}</b> &mdash; {E(r['best_reaction'])}. "
                        f"{E(r['evidence'] or 'ranked by rank aggregation')}. <span class='stat'>RRA = {r['rra_score']:.1e}</span>"))
    if res.concordance is not None:
        vc = res.concordance["class"].value_counts()
        out.append(("Concordance", f"Gene&ndash;protein concordance: <b>{vc.get('Coupled', 0)}</b> coupled, "
                    f"<b>{vc.get('Protein-only (post-transcriptional)', 0)}</b> protein-only, "
                    f"<b>{vc.get('mRNA-only (buffered)', 0)}</b> mRNA-only, <b>{vc.get('Opposing', 0)}</b> opposing genes."))
    return out


with tabs[0]:
  if on(0):
      shared = pw[pw["class"].str.startswith("Shared")]
      spec = pw[pw["class"].str.endswith("-specific")]
      cpz = res.coupling.get("pearson_z", np.nan) if res.concordance is not None else np.nan
      n_md = int((rx["metabolite_driven"]).sum())
      st.markdown(UI.section("Primary tri-omics metrics", "chart"), unsafe_allow_html=True)
      st.markdown(UI.metric_cards([
          dict(label="Observable reactions", value=f"{len(rx):,}", icon="db", accent="var(--faint)",
               sub="measured enzyme + metabolite"),
          dict(label="Coherent reactions", value=int(rx["significant"].sum()), icon="check", accent="var(--green)",
               highlight=True, sub=f"<b>label &amp; topology</b> q &lt; {res.params['alpha']:g}"),
          dict(label="Metabolite-driven", value=n_md, icon="flask", accent="var(--amber)", highlight=True,
               sub="<b>balance changes</b>, enzyme unchanged"),
          dict(label="Shared pathways", value=len(shared), icon="branch", accent="var(--purple)",
               sub=UI._esc(shared["pathway"].iloc[0][:34]) if len(shared) else "none at this FDR"),
          dict(label="Omics-specific", value=len(spec), icon="stack", accent="var(--sky)", sub="single-layer pathways"),
          dict(label="mRNA–protein r", value=f"{cpz:.2f}" if np.isfinite(cpz) else "n/a", icon="trend",
               accent="var(--brand)", sub=f"{res.coupling.get('n_genes', 0):,} genes in both layers"
               if res.concordance is not None else "needs mRNA and protein"),
      ]), unsafe_allow_html=True)
      g1, g2 = st.columns([1.85, 1])
      with g1.container(border=True):
          st.markdown(UI.card_head("Layer concordance across top pathways",
                                   "Per-layer pathway activity z (permutation-standardised Stouffer) for the six "
                                   "most significant pathways; dashed lines at |z| = 2.", "trend"), unsafe_allow_html=True)
          show(F.layer_bars(pw, 6), "layer_concordance")
      with g2.container(border=True):
          st.markdown(UI.card_head("Reaction classification", "All observable reactions by evidence category.",
                                   "target"), unsafe_allow_html=True)
          show(F.reaction_donut(rx), "reaction_classification")
      with st.container(border=True):
          h1, h2 = st.columns([1.5, 1.6])
          h1.markdown(UI.card_head("Key findings & biological interpretations",
                                   "Top-scoring reactions, pathways and targets, categorised by Integrix.", "spark"),
                      unsafe_allow_html=True)
          q_ = h2.text_input("Search", placeholder="Search gene, reaction or pathway", label_visibility="collapsed",
                             key="kf_q", icon=":material/search:")
          _seg = st.segmented_control if ST_HAS["segmented_control"] else \
              (lambda label, options, **kw: st.radio(label, options, horizontal=True, key=kw.get("key"),
                                                     label_visibility="collapsed"))
          kind_ = _seg("Filter", ["All", "Bottleneck", "Throughput", "Metabolite-driven", "Pathway",
                                                  "Target"], default="All", key="kf_kind", label_visibility="collapsed")
          rows_ = _findings()
          if kind_ and kind_ != "All":
              rows_ = [r for r in rows_ if r[0] == kind_]
          if q_:
              rows_ = [r for r in rows_ if q_.lower() in r[1].lower()]
          n_all = len(rows_)
          show_all = st.session_state.get("kf_all", False) or bool(q_) or (kind_ not in (None, "All"))
          st.markdown(UI.findings(rows_ if show_all else rows_[:6]), unsafe_allow_html=True)
          if n_all > 6 and not (q_ or (kind_ not in (None, "All"))):
              st.toggle(f"Show all {n_all} findings", key="kf_all")
          st.caption("Results describe evidence consistent with a mechanism, not proof of causation.")
      p1, p2 = st.columns(2)
      with p1.container(border=True):
          st.markdown(UI.card_head("Reaction pattern map", "Enzyme evidence E vs metabolite balance M.", "bolt"),
                      unsafe_allow_html=True)
          show(F.pattern_grid(rx), "reaction_patterns", notitle=True)
          download_panel("ov_pat", lambda w, h, fs: PF.pattern_grid(rx, 12, w, h or 4.2, fs), 5.6, 4.2, 7, "reaction_patterns")
      if len(cp_tbl):
          with p2.container(border=True):
              st.markdown(UI.card_head("Canonical pathways", "Right-tailed Fisher's exact test; colour = directional z.",
                                       "branch"), unsafe_allow_html=True)
              show(F.canonical_bars(cp_tbl, 15), "canonical_pathways", notitle=True)
              download_panel("ov_can", lambda w, h, fs: PF.canonical(cp_tbl, 15, w, h, fs), 7.6, None, 7, "canonical_pathways")

# ===================================================================================== Data & mapping
with tabs[1]:
  if on(1):
      st.markdown("### Samples and evidence tier")
      ov = res.design.overlap
      st.write(pd.DataFrame({"layer": [LAYER_NAMES[k] for k in ov["per_layer"]],
                             "samples in design": list(ov["per_layer"].values())}).assign(
          **{"samples in all layers": ov["all_layers"]}))
      st.caption(res.design.tier_text)
      if "Batch" in res.design.meta.columns:
          v = cramers_v(res.design.meta["Batch"], res.design.meta["Group"])
          msg = "warning: batch and group are confounded" if v > 0.5 else "batch is not confounded with group"
          st.caption(f"Batch vs group Cramér's V = {v:.2f} ({msg}).")
      show(F.coverage_bars(layers), "mapping_coverage")
      download_panel("cov", lambda w, h, fs: PF.coverage(layers, w, h or 2.6, fs), 3.6, 2.6, 7, "mapping_coverage")
      for k, L in layers.items():
          with st.expander(f"{L.name}: processing log and identifier mapping"):
              for line in L.log:
                  st.write("• " + line)
              f = L.feat.reset_index().rename(columns={"index": "feature"})
              table(f.head(500), 260, key=f"map_{k}")
              un = f[f["map_rule"].isin(["unmapped", "not in metabolic KB"])]
              csv_button(un, f"unmapped_{k}.csv", f"Download {len(un)} unmapped features", key=f"un_{k}")

# ===================================================================================== Concordance
with tabs[2]:
  if on(2):
      if res.concordance is None:
          st.info("Concordance needs both transcriptomics and proteomics.")
      else:
          cp = res.coupling
          c = st.columns(4)
          c[0].metric("Genes in both layers", f"{cp['n_genes']:,}")
          c[1].metric("Deming slope (protein on mRNA)", f"{cp['beta']:.2f}")
          c[2].metric("Correlation of z-scores", f"{cp['pearson_z']:.2f}")
          c[3].metric("Median within-sample rho", f"{cp.get('median_within_rho', np.nan):.2f}" if "median_within_rho" in cp else "n/a (unpaired)")
          nl = st.slider("Genes labelled", 0, 40, 15, key="conc_nl")
          show(F.concordance(res.concordance, cp["beta"], nl), "concordance")
          download_panel("conc", lambda w, h, fs: PF.concordance(res.concordance, cp["beta"], nl, w, h or 4.0, fs),
                         5.4, 4.0, 7, "concordance")
          cls = st.multiselect("Show classes", sorted(res.concordance["class"].unique()),
                               default=[c for c in res.concordance["class"].unique() if c != "Unchanged"])
          d = res.concordance[res.concordance["class"].isin(cls)]
          table(d, 360, key="conc_tbl")
          csv_button(d, "gene_protein_concordance.csv", key="conc_dl")

# ===================================================================================== Reactions
with tabs[3]:
  if on(3):
      st.markdown("Each reaction combines **enzyme evidence E** (protein z through the GPR rule) with the "
                  "**metabolite balance M** (products minus substrates). **Coherence C = E × M.** A reaction is "
                  f"called coherent when both label-permutation and topology q < {res.params['alpha']}.")
      nl = st.slider("Enzymes labelled", 0, 40, 14, key="rx_nl")
      show(F.pattern_grid(rx, nl), "reaction_patterns")
      download_panel("rxpat", lambda w, h, fs: PF.pattern_grid(rx, nl, w, h or 4.2, fs), 6.0, 4.4, 7, "reaction_patterns")
      c1, c2, c3 = st.columns([1.2, 2, 2])
      only = c1.radio("Rows", ["Coherent", "Metabolite-driven", "All"], horizontal=False)
      pats = c2.multiselect("Patterns", list(PATTERN_HELP), default=[])
      q = c3.text_input("Search enzyme, reaction or metabolite")
      d = rx
      if only == "Coherent":
          d = d[d["significant"]]
      elif only == "Metabolite-driven":
          d = d[d["metabolite_driven"]]
      if pats:
          d = d[d["pattern"].isin(pats)]
      if q:
          m = d[["name", "enzymes", "substrates", "products", "subsystem"]].apply(
              lambda s_: s_.astype(str).str.contains(q, case=False)).any(axis=1)
          d = d[m]
      cols = ["name", "enzymes", "subsystem", "pattern", "E", "S", "P", "M", "C", "q_label", "q_topology",
              "q_balance", "enzyme_evidence", "substrates", "products", "ec", "kegg_rxn", "reaction"]
      table(d[cols].round(4), 420, key="rx_tbl")
      csv_button(d, "reactions.csv", key="rx_dl")
      with st.expander("What the patterns mean"):
          for k, v in PATTERN_HELP.items():
              st.markdown(f"- **{k}**: {v}")

# ===================================================================================== Cascade
with tabs[4]:
  if on(4):
      st.markdown("Follow the chain **enzyme → reaction → metabolite**. Ribbon width = |z|, red = up, blue = down; "
                  "pale ribbons and italic names are substrates. Full reaction names are shown, each in its own box.")
      mode = st.radio("Select reactions by", ["Pathway", "Top coherent reactions", "Enzyme"], horizontal=True)
      if mode == "Pathway":
          choices = pw.loc[pw["n_reactions_observed"] > 0, "pathway"].tolist()
          p = st.selectbox("Pathway", choices, key="casc_pw")
          sel = rx[rx["subsystem"].str.split("; ").apply(lambda x: p in x)]
          sel = sel.sort_values("p_label").head(st.slider("Reactions shown", 3, 30, 12, key="casc_n"))
      elif mode == "Top coherent reactions":
          sel = rx.sort_values("p_label").head(st.slider("Reactions shown", 3, 30, 12, key="casc_n2"))
      else:
          genes = sorted({g.strip() for e in rx["enzymes"] for g in e.split(",")})
          g = st.selectbox("Enzyme", genes, index=genes.index("ASS1") if "ASS1" in genes else 0)
          sel = rx[rx["enzymes"].str.split(", ").apply(lambda x: g in x)].head(15)
      if len(sel):
          PF.style("DejaVu Sans", 8)
          with st.container(border=True):
              st.markdown(UI.card_head("Enzyme → reaction → metabolite cascade", "Ribbon width = |z|; red = up, "
                                       "blue = down; full reaction names.", "flow"), unsafe_allow_html=True)
              cw_, _ = show_mpl(lambda: PF.cascade_alluvial(sel, w=7.5, fs=8), max_scale=1.3,
                                cache_key=("cascade", S.get("run_id"), tuple(sel["reaction"])))
          download_panel("casc", lambda w, h, fs: PF.cascade_alluvial(sel, w, h, fs), round(cw_ * 2) / 2, None, 8, "cascade")
          table(sel[["name", "enzymes", "pattern", "E", "S", "P", "C", "q_label"]].round(4), 260, key="casc_tbl")

# ===================================================================================== Pathways
with tabs[5]:
  if on(5):
      t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
      t1.markdown(UI.page_title("Pathways", "Which pathways change, in which omics layer, and whether the layers "
                                "agree."), unsafe_allow_html=True)
      ntop = t2.selectbox("Pathways shown", [10, 15, 25, 40, 60], index=2, key="pw_n",
                          format_func=lambda n: f"Top {n}")
      c1, c2 = st.columns(2, gap="medium")
      with c1.container(border=True):
          st.markdown(UI.card_head("Canonical pathways (IPA-style)",
                                   "Bar = −log10 p (right-tailed Fisher's exact) · colour = directional z "
                                   "(orange up, blue down, grey &lt; 4 directional molecules) · line = ratio",
                                   "branch"), unsafe_allow_html=True)
          if len(cp_tbl):
              show(F.canonical_bars(cp_tbl, ntop), "canonical_pathways", notitle=True)
              download_panel("can", lambda w, h, fs: PF.canonical(cp_tbl, ntop, w, h, fs), 7.6, None, 7,
                             "canonical_pathways")
          else:
              st.info("No pathway has enough measured molecules.")
      with c2.container(border=True):
          st.markdown(UI.card_head("Pathway activity across layers",
                                   "Colour = permutation-standardised z · size = −log10 q · tile = class "
                                   "(shared-concordant, shared-discordant or omics-specific)", "chart"),
                      unsafe_allow_html=True)
          show(F.pathway_dotplot(pw, ntop), "pathways", notitle=True)
          download_panel("pwdot", lambda w, h, fs: PF.pathway_dotplot(pw, ntop, w, h, fs), 7.2, None, 7,
                         "pathway_activity")
      with st.container(border=True):
          h1, h2 = st.columns([2, 1.4], vertical_alignment="center")
          h1.markdown(UI.card_head("Pathway table", "Per-layer z and q, combined z, heterogeneity (Cochran's Q, I²) "
                                   "and class.", "grid"), unsafe_allow_html=True)
          cls = h2.multiselect("Classes", sorted(pw["class"].unique()), default=[], key="pw_cls",
                               placeholder="Filter by class", label_visibility="collapsed")
          tabA, tabB = st.tabs(["Cross-layer activity", "Canonical pathways"])
          with tabA:
              d = pw[pw["class"].isin(cls)] if cls else pw
              table(d.round(4), 360, key="pw_tbl")
              csv_button(d, "pathways.csv", key="pw_dl")
          with tabB:
              table(cp_tbl.round(5), 360, key="can_tbl")
              csv_button(cp_tbl, "canonical_pathways.csv", key="can_dl")

# ===================================================================================== Pathway maps & network studio
with tabs[6]:
  if on(6):
      st.markdown("**Pathway maps and network studio.** Start from an IPA-style pathway map or any other view. "
                  "**Shape** = molecule type; **red / green** = measured up / down; **orange / blue outline** = "
                  "predicted increase / decrease of unmeasured metabolites; **edge colour** = leads to increase "
                  "(orange), decrease (blue), inconsistent with the data (yellow) or not predicted (grey). Edges are "
                  "routed around nodes and every label is placed clear of other labels and nodes.")
      network_ui("net")

# ===================================================================================== Targets
with tabs[7]:
  if on(7):
      if extras.get("targets") is None:
          st.info("Enable 'Network, propagation and targets' in the sidebar and run again.")
      else:
          tg = extras["targets"]
          st.markdown("Targets are ranked by robust rank aggregation of reaction coherence, mRNA-protein agreement, "
                      "choke-point status and network propagation (plus druggability when a list is uploaded). "
                      "**They are hypotheses for experimental follow-up.**")
          nt = st.slider("Targets shown", 5, 50, 20, key="tg_n")
          PF.style("DejaVu Sans", 8)
          with st.container(border=True):
              st.markdown(UI.card_head("Prioritised targets", "Robust rank aggregation; orange = choke point.",
                                       "target"), unsafe_allow_html=True)
              tw_, _ = show_mpl(lambda: PF.targets_bar(tg, nt, fs=8), cache_key=("targets", S.get("run_id"), nt))
          download_panel("tg", lambda w, h, fs: PF.targets_bar(tg, nt, w, h, fs), round(tw_ * 2) / 2, None, 8, "targets")
          table(tg.round(4), 320, key="tg_tbl")
          csv_button(tg, "targets.csv", key="tg_dl")
      st.markdown("### Reporter metabolites")
      st.caption("Metabolites around which the surrounding reactions are jointly coherent (Patil & Nielsen 2005).")
      table(res.reporter_mets.head(200).round(5), 300, key="rep_tbl")
      if res.regulators is not None:
          st.markdown("### Transcription-factor regulators")
          table(res.regulators.round(5), 300, key="tf_tbl")
      else:
          st.caption("TF regulator inference becomes available when CollecTRI regulons are added to the knowledge "
                     "base (python -m trace_core.kb_build --skip-core --collectri).")

# ===================================================================================== Factors
with tabs[8]:
  if on(8):
      f = extras.get("factors")
      if not f:
          st.info(extras.get("factors_error", "Enable 'Latent factors (MOFA+)' in the sidebar and run again."))
      else:
          st.caption(f"Method: {f['method']}. Factors explaining ≥ 2% variance in at least one layer are shown.")
          show(F.variance_heatmap(f["r2"]), "factor_variance")
          download_panel("fvar", lambda w, h, fs: PF.variance_heatmap(f["r2"], w, h or 1.8, fs), 0.6 * f["r2"].shape[1] + 1.8,
                         1.8, 7, "factor_variance")
          cols = list(f["scores"].columns)
          if len(cols) >= 2:
              a, b = st.columns(2)
              fx = a.selectbox("x", cols, 0)
              fy = b.selectbox("y", cols, 1)
              lbl = st.checkbox("Label samples", False, key="fac_lbl")
              show(F.factor_scatter(f["scores"], res.design.meta["Group"], fx, fy), "factor_scores")
              download_panel("fsc", lambda w, h, fs: PF.factor_scatter(f["scores"], res.design.meta["Group"], fx, fy, w,
                                                                     h or 3.2, fs, label_samples=lbl), 4.2, 3.4, 7, "factor_scores")
          table(f["summary"].round(3), 220, key="fac_sum")
          st.markdown("**Factor associations with group and covariates**")
          table(f["assoc"].round(4), 240, key="fac_assoc")
          if extras.get("factor_enrichment") is not None and len(extras["factor_enrichment"]):
              st.markdown("**Pathways enriched in factor loadings**")
              table(extras["factor_enrichment"].head(300).round(5), 300, key="fac_enr")

# ===================================================================================== Signature
with tabs[9]:
  if on(9):
      s = extras.get("signature")
      if not s:
          st.info("Enable 'Integrated biomarker signature' in the sidebar and run again.")
      else:
          if s.get("status") != "ok":
              st.markdown(f'<div class="trace-warn">{s["status"]}</div>', unsafe_allow_html=True)
          else:
              c = st.columns(4)
              c[0].metric("AUC (nested CV)", f"{s['auc_mean']:.2f}", f"95% CI {s['auc_ci'][0]:.2f}-{s['auc_ci'][1]:.2f}",
                          delta_color="off")
              c[1].metric("Balanced accuracy", f"{s['balanced_accuracy']:.2f}")
              c[2].metric("Permutation p", f"{s['p_perm']:.3f}")
              c[3].metric("Features per block", s["keepX"])
              st.markdown('<div class="trace-warn">Cross-validated performance on one cohort is optimistic for '
                          'clinical use. Validate the panel in an independent cohort.</div>', unsafe_allow_html=True)
              a, b = st.columns(2)
              with a:
                  show(F.roc_plot(s), "signature_roc")
                  download_panel("roc", lambda w, h, fs: PF.roc(s, w, h or w, fs), 3.4, 3.4, 7, "signature_roc")
              with b:
                  show(F.perm_hist(s), "signature_permutation")
                  download_panel("perm", lambda w, h, fs: PF.perm_hist(s, w, h or 2.6, fs), 3.6, 2.6, 7, "signature_permutation")
              show(F.panel_bars(s["panel"]), "signature_panel")
              download_panel("panel", lambda w, h, fs: PF.panel_bars(s["panel"], w, h, fs), 5.0, None, 7, "signature_panel")
              st.write(pd.DataFrame({"model": ["Integrated (all layers)"] + [f"{k} only" for k in s["single_layer_auc"]],
                                     "AUC": [s["auc_mean"]] + list(s["single_layer_auc"].values())}).round(3))
              csv_button(s["panel"], "signature_panel.csv", key="sig_dl")
          lf = s.get("late_fusion", {})
          if lf.get("status") == "ok":
              st.markdown("**Late fusion (uses every sample available in each layer)**")
              st.write(pd.DataFrame({"model": ["Fused"] + list(lf["auc_per_layer"]),
                                     "AUC (CV)": [lf["auc_fused"]] + list(lf["auc_per_layer"].values())}).round(3))

# ===================================================================================== Export
with tabs[10]:
  if on(10):
      st.markdown("### Download results")
      xls = R.excel(res, extras)
      man = R.manifest(res, S["inputs"], kb, extras)
      meth = R.methods_text(res, kb, extras)
      c = st.columns(3)
      c[0].download_button("Excel workbook (all tables)", xls, "Integrix_results.xlsx", type="primary")
      c[1].download_button("Run manifest (JSON)", json.dumps(man, indent=2, default=str), "Integrix_manifest.json")
      c[2].download_button("Methods text (Markdown)", meth, "Integrix_methods.md")
      st.markdown("### All publication figures at once")
      c = st.columns(3)
      fmts = c[0].multiselect("Formats", list(PF.FORMATS), default=["PNG", "PDF", "SVG"], key="all_fmts")
      dpi = c[1].selectbox("DPI", [300, 600, 1200], index=1, key="all_dpi")
      font = c[2].selectbox("Font family", _fonts(), key="all_font")
      if st.button("Build figure set (ZIP)"):
          with st.spinner("Rendering all figures"):
              PF.style(font, 7)
              builders = {"reaction_patterns": lambda: PF.pattern_grid(rx),
                          "pathway_activity": lambda: PF.pathway_dotplot(pw),
                          "mapping_coverage": lambda: PF.coverage(layers)}
              if len(cp_tbl):
                  builders["canonical_pathways"] = lambda: PF.canonical(cp_tbl)
              if res.concordance is not None:
                  builders["concordance"] = lambda: PF.concordance(res.concordance, res.coupling["beta"])
              if extras.get("targets") is not None:
                  builders["targets"] = lambda: PF.targets_bar(extras["targets"])
              if extras.get("factors"):
                  builders["factor_variance"] = lambda: PF.variance_heatmap(extras["factors"]["r2"])
              sg = extras.get("signature")
              if sg and sg.get("status") == "ok":
                  builders["signature_roc"] = lambda: PF.roc(sg)
                  builders["signature_panel"] = lambda: PF.panel_bars(sg["panel"])
              top_pw = pw.loc[pw["n_reactions_observed"] >= 2, "pathway"].head(3).tolist()
              for k_, p_ in enumerate(top_pw):
                  o_ = NV.NetOptions(scope="Pathway", pathway=p_, max_reactions=30, title=p_)
                  builders[f"pathway_map_{k_ + 1}"] = (lambda o_=o_: NV.render_mpl(*NV.build_view(res, kb, o_, extras.get("targets")), o_))
              out = {}
              for nm, b in builders.items():
                  for fl in fmts:
                      ext = PF.FORMATS[fl]
                      out[f"{nm}.{ext}"] = PF.export(b(), ext, dpi)
              st.session_state["figzip"] = R.bundle(out)
      if "figzip" in st.session_state:
          st.download_button("Download figure set (ZIP)", st.session_state["figzip"], "Integrix_figures.zip", type="primary")
      st.markdown("### Everything in one archive")
      if st.button("Build full bundle"):
          files = {"Integrix_results.xlsx": xls, "Integrix_manifest.json": man, "Integrix_methods.md": meth}
          files.update({f"figures/{k}": v for k, v in R.pub_figures(res, extras, "png", 600).items()})
          if len(cp_tbl):
              files["canonical_pathways.csv"] = cp_tbl.to_csv(index=False)
          if extras.get("subnet"):
              from trace_core.network import to_graphml
              files["Integrix_subnetwork.graphml"] = to_graphml(*extras["subnet"])
          st.session_state["fullzip"] = R.bundle(files)
      if "fullzip" in st.session_state:
          st.download_button("Download full bundle (ZIP)", st.session_state["fullzip"], "Integrix_bundle.zip")

# ===================================================================================== Methods
with tabs[11]:
  if on(11):
      st.markdown(R.methods_text(res, kb, extras))
      st.markdown("**IPA-style displays.** Canonical-pathway bars, molecule shapes, measured (red/green) and predicted "
                  "(orange/blue) colours, relationship-coloured edges and subcellular layouts follow the visual "
                  "conventions of Ingenuity Pathway Analysis. They are computed from Integrix statistics and the "
                  "Human-GEM knowledge base; they do not use IPA's proprietary knowledge base or its activation "
                  "z-score algorithm.")
      with st.expander("Run log and timings"):
          for l in res.log:
              st.write("• " + l)
          st.write({k: f"{v:.1f} s" for k, v in res.timings.items()})
      with st.expander("Knowledge base"):
          st.json(kb.manifest)


# ===================================================================================== footer
st.markdown(UI.footer(__version__, kb.version), unsafe_allow_html=True)
