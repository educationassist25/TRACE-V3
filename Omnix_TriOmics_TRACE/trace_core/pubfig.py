"""
pubfig.py - publication renderer: every Integrix figure as a matplotlib figure with collision-free labels,
exported as PNG / TIFF / JPEG / SVG / PDF / EPS at any resolution.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .figures import CLASS_COLORS, PATTERN_COLORS  # noqa: E402
from .ipa import z_color  # noqa: E402
from .layout import place_labels, text_size  # noqa: E402

FORMATS = {"PNG": "png", "TIFF": "tiff", "JPEG": "jpg", "SVG": "svg", "PDF": "pdf", "EPS": "eps"}
MIME = {"png": "image/png", "tiff": "image/tiff", "jpg": "image/jpeg", "svg": "image/svg+xml",
        "pdf": "application/pdf", "eps": "application/postscript"}


# ------------------------------------------------------------------------------------------ exact text metrics
_RENDERER = None


def measure(text: str, fs: float, weight: str = "normal") -> tuple[float, float]:
    """Exact rendered size (inches) of one line of text in the current font (matplotlib Agg metrics)."""
    global _RENDERER
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties
    if _RENDERER is None:
        _RENDERER = RendererAgg(10, 10, 72)
    fp = FontProperties(family=plt.rcParams["font.family"], size=fs, weight=weight)
    w, h, _ = _RENDERER.get_text_width_height_descent(str(text) or " ", fp, ismath=False)
    return w / 72, h / 72


def wrap_text(text: str, fs: float, max_in: float, weight: str = "normal") -> list[str]:
    """Word-wrap to a width in inches using exact metrics; long chemical names break after - , : ; / )."""
    import re
    tokens = []
    for k, word in enumerate(str(text).split()):
        parts = re.findall(r"[^\-,:;/)]+[\-,:;/)]*|[\-,:;/)]+", word) or [word]
        for j, ptok in enumerate(parts):
            tokens.append((ptok, k > 0 and j == 0))
    lines, cur = [], ""
    for tok, space in tokens:
        cand = cur + (" " if space and cur else "") + tok
        if not cur or measure(cand, fs, weight)[0] <= max_in:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return lines or [""]


def style(font_family="DejaVu Sans", font_size=8):
    plt.rcParams.update({"font.family": font_family, "font.size": font_size, "axes.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "legend.frameon": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def export(fig, fmt: str = "png", dpi: int = 600, transparent: bool = False) -> bytes:
    b = io.BytesIO()
    kw = {}
    if fmt == "tiff":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    if fmt == "jpg":
        kw["pil_kwargs"] = {"quality": 95}
        transparent = False
    fig.savefig(b, format="jpeg" if fmt == "jpg" else fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                transparent=transparent, **kw)
    plt.close(fig)
    return b.getvalue()


def _repel(fig, ax, x, y, labels, all_x, all_y, fs, point_r=0.035):
    """Place labels for points (x, y) avoiding all points and each other; draws text + leader lines."""
    if len(labels) == 0:
        return
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    W, H = bb.width, bb.height
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tx = lambda v: (np.asarray(v, float) - x0) / (x1 - x0) * W
    ty = lambda v: (np.asarray(v, float) - y0) / (y1 - y0) * H
    A = np.c_[tx(x), ty(y)]
    O = np.c_[tx(all_x), ty(all_y)]
    ok = np.isfinite(O).all(1)
    O = O[ok]
    sizes = np.array([text_size(t, fs) for t in labels])
    C, placed, lead = place_labels(A, sizes, O, np.full(len(O), point_r), np.zeros((0, 2)), W, H,
                                   anchor_r=np.full(len(A), point_r), margin=0.0)
    for i, t in enumerate(labels):
        if not placed[i]:
            continue
        cx = x0 + C[i, 0] / W * (x1 - x0)
        cy = y0 + C[i, 1] / H * (y1 - y0)
        if lead[i]:
            ax.plot([x[i], cx], [y[i], cy], color="#7F8C8D", lw=0.5, zorder=4)
        ax.text(cx, cy, t, fontsize=fs, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


# --------------------------------------------------------------------------------------------- figures
def concordance(conc, beta, n_label=15, w=4.2, h=4.0, fs=7, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    for cls, col in CLASS_COLORS.items():
        d = conc[conc["class"] == cls]
        ax.scatter(d["z_mRNA"], d["z_protein"], s=5 if cls == "Unchanged" else 14, c=col, lw=0.3,
                   edgecolors="white", alpha=0.45 if cls == "Unchanged" else 0.95, label=f"{cls} ({len(d)})",
                   zorder=2 if cls == "Unchanged" else 3)
    lim = np.nanmax(np.abs(conc[["z_mRNA", "z_protein"]].values)) * 1.08
    ax.plot([-lim, lim], [-beta * lim, beta * lim], "--", c="#333", lw=0.8, zorder=1, label=f"Deming slope {beta:.2f}")
    ax.axhline(0, c="#ccc", lw=0.5, zorder=0)
    ax.axvline(0, c="#ccc", lw=0.5, zorder=0)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="mRNA z-score", ylabel="Protein z-score")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1))
    top = conc[conc["class"] != "Unchanged"].nsmallest(n_label, "p_discordance")
    _repel(fig, ax, top["z_mRNA"].values, top["z_protein"].values, list(top["gene"]),
           conc["z_mRNA"].values, conc["z_protein"].values, fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def pattern_grid(rx, n_label=12, w=5.2, h=4.2, fs=7, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    q = rx["q_label"].fillna(1).clip(1e-12, 1)
    for pat, col in PATTERN_COLORS.items():
        d = rx[rx["pattern"] == pat]
        if len(d):
            qq = d["q_label"].fillna(1).clip(1e-12, 1)
            ax.scatter(d["E"], d["M"], s=6 + 6 * np.clip(-np.log10(qq), 0, 6), c=col, lw=0.3, edgecolors="white",
                       alpha=0.9, label=f"{pat} ({len(d)})", zorder=3 if pat != "No clear pattern" else 2)
    ax.axhline(0, c="#999", lw=0.5, zorder=1)
    ax.axvline(0, c="#999", lw=0.5, zorder=1)
    ax.set(xlabel="Enzyme evidence E (z)", ylabel="Metabolite balance M")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=0.8)
    sig = rx[rx["significant"]].copy()
    sig["lab"] = [str(e).split(",")[0] for e in sig["enzymes"]]
    sig = sig.drop_duplicates("lab").head(n_label)
    labs = list(sig["lab"])
    _repel(fig, ax, sig["E"].values, sig["M"].values, labs, rx["E"].values, rx["M"].values, fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def pathway_dotplot(pw, top=25, w=6.4, h=None, fs=7, title=""):
    """Publication dot plot laid out in inches: pathway labels | dot columns | class text | colour bar + size key.
    Dot diameters are capped at 80 % of the smaller of the row and column pitch, so neighbouring dots never touch;
    when the chosen size is tight, fonts shrink (to >= 4 pt) and labels are shortened so text never collides."""
    d = pw.dropna(subset=["z_combined"]).head(top).iloc[::-1]
    cols = [c for c in ("mRNA", "protein", "metabolite") if f"z_{c}" in d.columns] + ["combined"]
    n = max(len(d), 1)
    h = h or max(2.2, 0.26 * n + 1.2)
    top_in, bot_in = (0.75 if title else 0.55), 0.2
    plot_h = max(0.5, h - top_in - bot_in)
    row_in = plot_h / (n + 0.4)
    f = max(4.0, min(fs, row_in * 72 / 1.3))                 # label font fits the row pitch
    cw = lambda chars, pt: 0.55 * pt / 72 * chars           # approx. text width in inches
    side_in = 1.0
    cls_chars = max(len(c) for c in d["class"]) if len(d) else 10
    cls_in = cw(cls_chars, f - 0.5) + 0.1
    max_chars = min(52, max(len(p) for p in d["pathway"])) if len(d) else 10
    min_cols_in = 0.32 * len(cols)
    lab_in = cw(max_chars, f) + 0.12
    spare = w - side_in - cls_in - min_cols_in - lab_in
    if spare < 0:                                            # shorten labels, then shrink fonts
        max_chars = max(14, int(max_chars + spare / cw(1, f)))
        lab_in = cw(max_chars, f) + 0.12
        spare = w - side_in - cls_in - min_cols_in - lab_in
        if spare < 0:
            k = max(0.55, (w - side_in - min_cols_in) / (lab_in + cls_in))
            f = max(4.0, f * k)
            lab_in, cls_in = cw(max_chars, f) + 0.12, cw(cls_chars, f - 0.5) + 0.1
    cols_in = max(min_cols_in, w - side_in - cls_in - lab_in)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([lab_in / w, bot_in / h, cols_in / w, plot_h / h])
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(-0.7, n - 0.3)
    dmax = 0.8 * min(row_in, cols_in / len(cols)) * 72
    size_of = lambda q: (dmax * (0.32 + 0.68 * min(1.0, -np.log10(max(q, 1e-12)) / 6))) ** 2
    cmap = plt.get_cmap("RdBu_r")
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    for i, (_, r) in enumerate(d.iterrows()):
        for j, c in enumerate(cols):
            z, q = r[f"z_{c}"], r[f"q_{c}"]
            if np.isfinite(z):
                ax.scatter(j, i, s=size_of(q if np.isfinite(q) else 1.0), c=[cmap((np.clip(z, -5, 5) + 5) / 10)],
                           edgecolors="#444", linewidths=0.4, zorder=3, clip_on=False)
        ax.text(1.0, i, "  " + r["class"], transform=tr, fontsize=f - 0.5, va="center", ha="left", color="#00695C")
    ax.set_yticks(range(n))
    ax.set_yticklabels([p if len(p) <= max_chars else p[:max_chars - 1] + "…" for p in d["pathway"]], fontsize=f)
    ax.set_xticks(range(len(cols)))
    rot = 0 if cols_in / len(cols) > cw(10, f) + 0.05 else 45
    ax.set_xticklabels(cols, fontsize=f, rotation=rot, ha="left" if rot else "center",
                       rotation_mode="anchor" if rot else "default")
    ax.xaxis.tick_top()
    ax.spines[["bottom", "right"]].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.grid(axis="y", color="#EEE", lw=0.5, zorder=0)
    x_side = (lab_in + cols_in + cls_in + 0.1) / w
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(-5, 5))
    cb_h = min(1.4, plot_h * 0.28)                           # short colour scale
    cax = fig.add_axes([x_side, (bot_in + plot_h - cb_h) / h, 0.1 / w, cb_h / h])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("z", fontsize=f)
    cb.ax.tick_params(labelsize=max(4, f - 1))
    lax = fig.add_axes([x_side - 0.04 / w, bot_in / h, (side_in - 0.05) / w, plot_h * 0.45 / h])
    lax.axis("off")
    for qv, lab in ((0.05, "q 0.05"), (1e-3, "q 0.001"), (1e-6, "q 1e-6")):
        lax.scatter([], [], s=size_of(qv), c="#BBB", edgecolors="#444", lw=0.4, label=lab)
    lax.legend(fontsize=max(4, f - 1), loc="lower left", title="size", title_fontsize=max(4, f - 1), frameon=False,
               labelspacing=1.0, borderaxespad=0)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=22)
    return fig


def canonical(cp, top=25, w=7.6, h=None, fs=7, title="", thr=0.05):
    d = cp.head(top).iloc[::-1]
    h = h or max(2.4, 0.24 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(d))
    ax.barh(y, d["minus_log10_p"], color=[z_color(z) for z in d["z_directional"]], edgecolor="#555", lw=0.4,
            height=0.72, zorder=2)
    ax.axvline(-np.log10(thr), color="#E65100", ls="--", lw=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([p if len(p) <= 66 else p[:64] + "…" for p in d["pathway"]], fontsize=fs)
    ax.set_xlabel("-log10(p)  (right-tailed Fisher's exact)")
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax2 = ax.twiny()
    ax2.plot(d["ratio"], y, "-o", color="#E67E22", ms=2.5, lw=0.8, zorder=4)
    ax2.set_xlabel("Ratio (significant / pathway molecules)", color="#E67E22", fontsize=fs)
    ax2.tick_params(axis="x", colors="#E67E22", labelsize=fs - 1)
    ax2.set_xlim(0, max(0.05, float(d["ratio"].max()) * 1.15))
    ax2.spines["top"].set_visible(True)
    import matplotlib.patches as mp
    hs = [mp.Patch(fc=z_color(3), ec="#555", label="z > 0 (increased)"), mp.Patch(fc=z_color(-3), ec="#555", label="z < 0 (decreased)"),
          mp.Patch(fc="#FFFFFF", ec="#555", label="z = 0"), mp.Patch(fc="#BDBDBD", ec="#555", label="no activity pattern")]
    ax.legend(handles=hs, fontsize=fs - 1, loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


def targets_bar(tg, top=20, w=None, h=None, fs=7, title="", text_in=3.6):
    """Prioritised targets: bar = -log10 rank-aggregation score (orange = choke point). In a separate column to
    the right: the FULL best reaction (bold), then the evidence and the reaction pattern, wrapped with exact text
    metrics. Every row is as tall as its text, so nothing is truncated or overlapping."""
    d = tg.head(top).reset_index(drop=True)
    v = np.nan_to_num(-np.log10(d["rra_score"].astype(float).clip(1e-30, 1)))
    bar_in = 1.9

    def build(fs_, text_in_):
        lh = fs_ * 1.3 / 72
        blocks = []
        for _, r in d.iterrows():
            rx = wrap_text(str(r.get("best_reaction", "") or ""), fs_, text_in_, "bold")
            pat = str(r.get("best_reaction_pattern", "") or "")
            ev_txt = str(r.get("evidence", "") or "") + (f"  ·  {pat}" if pat and pat != "nan" else "")
            ev = wrap_text(ev_txt, fs_ - 0.5, text_in_)
            blocks.append((rx, ev))
        rows_h = [max(0.26, (len(a) + len(b)) * lh + 0.12) for a, b in blocks]
        g_w = max([measure(g, fs_, "bold")[0] for g in d["gene"]] or [0.4])
        return lh, blocks, rows_h, g_w

    lh, blocks, rows_h, g_w = build(fs, text_in)
    if w is not None:
        f_, t_ = fs, text_in
        for _ in range(14):
            if 0.1 + g_w + 0.1 + 1.0 + 0.2 + t_ + 0.1 <= w:
                break
            if t_ > 2.0:
                t_ *= 0.9
            else:
                f_ = max(4.5, f_ * 0.9)
            lh, blocks, rows_h, g_w = build(f_, t_)
        fs, text_in = f_, t_
        bar_in = max(1.0, w - (0.1 + g_w + 0.1 + 0.2 + text_in + 0.1))
    W = 0.1 + g_w + 0.1 + bar_in + 0.2 + text_in + 0.1
    Hc = sum(rows_h)
    top_pad = 0.3 + (0.3 if title else 0)
    H = max(h or 0, Hc + top_pad + 0.5)
    fig = plt.figure(figsize=(W, H))
    x_bar = 0.1 + g_w + 0.1
    ax = fig.add_axes([x_bar / W, 0.5 / H, bar_in / W, Hc / H])
    ys, y = [], Hc
    for rh_ in rows_h:
        ys.append(y - rh_ / 2)
        y -= rh_
    vmax = float(v.max()) if len(v) and v.max() > 0 else 1.0
    ax.set_xlim(0, vmax * 1.05)
    ax.set_ylim(0, Hc)
    for i, (val, ch) in enumerate(zip(v, d["choke_point"])):
        ax.barh(ys[i], val, height=min(0.2, rows_h[i] * 0.6), color="#E65100" if ch else "#00695C", zorder=2)
        if i:
            ax.axhline(ys[i] + rows_h[i] / 2, color="#E4E9E8", lw=0.6, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(d["gene"], fontsize=fs, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("-log10 rank-aggregation score", fontsize=fs)
    ax.tick_params(axis="x", labelsize=fs - 0.5)
    tx = fig.add_axes([(x_bar + bar_in + 0.2) / W, 0.5 / H, text_in / W, Hc / H])
    tx.set_xlim(0, text_in)
    tx.set_ylim(0, Hc)
    tx.axis("off")
    for i, (rx, ev) in enumerate(blocks):
        lines = [(t, "bold", "#1F2A28", fs) for t in rx] + [(t, "normal", "#5D6D6A", fs - 0.5) for t in ev]
        n = len(lines)
        for q, (t, wt, c, f_) in enumerate(lines):
            tx.text(0, ys[i] + (n - 1) / 2 * lh - q * lh, t, fontsize=f_, fontweight=wt, color=c, va="center",
                    ha="left")
        if i:
            tx.axhline(ys[i] + rows_h[i] / 2, color="#E4E9E8", lw=0.6, zorder=0)
    tx.text(0, Hc + 0.07, "Best reaction  ·  evidence  ·  pattern", fontsize=fs, fontweight="bold", color="#00574D",
            va="bottom", ha="left")
    ax.text(0, Hc + 0.07, "orange = choke point", fontsize=fs - 0.5, color="#E65100", va="bottom", ha="left",
            transform=ax.transData)
    if title:
        fig.text(0.1 / W, 1 - 0.06 / H, title, fontsize=fs + 3, fontweight="bold", ha="left", va="top")
    return fig


def panel_bars(panel, w=4.6, h=None, fs=7, title=""):
    d = panel.sort_values("selection_frequency")
    h = h or max(2.0, 0.22 * len(d) + 0.8)
    cmap = {"mRNA": "#2C6FB7", "protein": "#00695C", "metabolite": "#E65100"}
    fig, ax = plt.subplots(figsize=(w, h))
    ax.barh(range(len(d)), d["selection_frequency"], color=[cmap.get(l, "#888") for l in d["layer"]], height=0.7)
    ax.axvline(0.7, ls=":", c="#555", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{a} ({b})" for a, b in zip(d["label"], d["layer"])], fontsize=fs)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Selection frequency (outer folds)")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def roc(sig, w=3.2, h=3.2, fs=7, title=""):
    fpr, tpr = sig["roc"]
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(fpr, tpr, c="#00695C", lw=1.6, label=f"Integrated AUC {sig['auc_mean']:.2f}\n95% CI {sig['auc_ci'][0]:.2f}-{sig['auc_ci'][1]:.2f}")
    ax.plot([0, 1], [0, 1], ":", c="#999")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def perm_hist(sig, w=3.4, h=2.6, fs=7, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.hist(sig["perm_aucs"], bins=20, color="#B8C2C0", edgecolor="white")
    ax.axvline(sig["auc_mean"], color="#E65100", lw=1.6)
    ax.text(sig["auc_mean"], ax.get_ylim()[1] * 0.95, f" observed {sig['auc_mean']:.2f}\n p = {sig['p_perm']:.3f}",
            fontsize=fs, va="top", ha="right" if sig["auc_mean"] > 0.8 else "left")
    ax.set(xlabel="Cross-validated AUC (permuted labels)", ylabel="Count")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def variance_heatmap(r2, w=None, h=1.8, fs=7, title=""):
    w = w or 0.6 * r2.shape[1] + 1.6
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(r2.values, cmap="BuGn", vmin=0, aspect="auto")
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            v = r2.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs - 0.5,
                    color="white" if v > r2.values.max() * 0.6 else "#1F2A28")
    ax.set_xticks(range(r2.shape[1]))
    ax.set_xticklabels(r2.columns, rotation=40, ha="right", fontsize=fs)
    ax.set_yticks(range(r2.shape[0]))
    ax.set_yticklabels(r2.index, fontsize=fs)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("% variance", fontsize=fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def factor_scatter(scores, groups, fx, fy, w=3.6, h=3.2, fs=7, title="", label_samples=False):
    fig, ax = plt.subplots(figsize=(w, h))
    cols = ["#00695C", "#E65100", "#2C6FB7", "#8E44AD"]
    g = groups.reindex(scores.index)
    for c, lv in zip(cols, sorted(g.dropna().unique())):
        s = scores[g == lv]
        ax.scatter(s[fx], s[fy], s=26, c=c, edgecolors="white", lw=0.5, label=lv)
    ax.set(xlabel=fx, ylabel=fy)
    ax.legend(fontsize=fs, loc="upper left", bbox_to_anchor=(1.01, 1))
    if label_samples:
        _repel(fig, ax, scores[fx].values, scores[fy].values, list(scores.index), scores[fx].values, scores[fy].values, fs - 1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def coverage(layers, w=3.6, h=2.6, fs=7, title=""):
    names, tot, mp = [], [], []
    for k, L in layers.items():
        names.append(L.name)
        tot.append(len(L.feat))
        mp.append(int((L.feat["map_rule"] != "unmapped").sum() if k == "met" else (L.feat["ensg"] != "").sum()))
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(names))
    ax.bar(x, tot, color="#C9D1CF", width=0.6, label="features")
    ax.bar(x, mp, color="#00695C", width=0.6, label="mapped to KB")
    for i, (t, m) in enumerate(zip(tot, mp)):
        ax.text(i, t, f"{m}/{t}", ha="center", va="bottom", fontsize=fs - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=fs)
    ax.set_ylabel("Features")
    ax.legend(fontsize=fs - 1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def cascade_alluvial(sel, w=None, h=None, fs=7, title="", wrap_rxn_in=2.6, wrap_met_in=1.9):
    """Enzyme -> reaction -> metabolite alluvial diagram with FULL reaction and metabolite names.

    Laid out in inches with exact text metrics: every reaction name sits inside its own box (wrapped, never
    truncated), enzyme labels sit left of their bars and metabolite labels right of theirs, so no text ever
    touches a ribbon or another label. Node heights grow with their text and with the ribbons they carry;
    ribbons are stacked along each node edge (ordered by partner position) so they never overlap at a node.
    Ribbon width = |z|; red = up, blue = down; pale ribbons = substrates."""
    from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
    from matplotlib.path import Path
    rows = sel.reset_index(drop=True)
    genes, mets, links = [], [], []      # links: (kind, source index, target index, z, is_substrate)
    for i, r in rows.iterrows():
        for g in [x.strip() for x in str(r["enzymes"]).split(",") if x.strip()][:6]:
            if g not in genes:
                genes.append(g)
            links.append(("g", genes.index(g), i, r["E"], False))
        for m in [x.strip() for x in str(r["products"]).split(";") if x.strip()][:4]:
            if m not in mets:
                mets.append(m)
            links.append(("m", i, mets.index(m), r["P"], False))
        for m in [x.strip() for x in str(r["substrates"]).split(";") if x.strip()][:4]:
            key = m + " (substrate)"
            if key not in mets:
                mets.append(key)
            links.append(("m", i, mets.index(key), r["S"], True))
    nR = len(rows)
    g_ord = sorted(range(len(genes)), key=lambda k: np.mean([l[2] for l in links if l[0] == "g" and l[1] == k]))
    m_ord = sorted(range(len(mets)), key=lambda k: np.mean([l[1] for l in links if l[0] == "m" and l[2] == k]))
    thick = lambda v: 0.035 + 0.11 * min(abs(v) if np.isfinite(v) else 0.3, 4) / 4      # inches

    def layout(fs_, rx_in, mt_in):
        lh = fs_ * 1.32 / 72
        r_lines = [wrap_text(n, fs_, rx_in) for n in rows["name"]]
        m_lines = [wrap_text(m, fs_, mt_in) for m in mets]
        g_w = max([measure(g, fs_, "bold")[0] for g in genes] or [0.3])
        r_w = max([measure(t, fs_)[0] for ls in r_lines for t in ls] or [0.5]) + 0.16
        m_w = max([measure(t, fs_)[0] for ls in m_lines for t in ls] or [0.5])
        g_in = [sum(thick(l[3]) for l in links if l[0] == "g" and l[1] == k) for k in range(len(genes))]
        r_left = [sum(thick(l[3]) for l in links if l[0] == "g" and l[2] == i) for i in range(nR)]
        r_right = [sum(thick(l[3]) for l in links if l[0] == "m" and l[1] == i) for i in range(nR)]
        m_in = [sum(thick(l[3]) for l in links if l[0] == "m" and l[2] == k) for k in range(len(mets))]
        gh = [max(lh + 0.06, g_in[k] + 0.04) for k in range(len(genes))]
        rh = [max(len(r_lines[i]) * lh + 0.12, r_left[i] + 0.06, r_right[i] + 0.06) for i in range(nR)]
        mh = [max(len(m_lines[k]) * lh + 0.06, m_in[k] + 0.04) for k in range(len(mets))]
        return lh, r_lines, m_lines, g_w, r_w, m_w, gh, rh, mh

    bw, gap, rib, pad = 0.07, 0.10, 1.25, 0.08
    lh, r_lines, m_lines, g_w, r_w, m_w, gh, rh, mh = layout(fs, wrap_rxn_in, wrap_met_in)
    fixed = lambda: pad + g_w + 0.06 + bw + r_w + bw + 0.06 + m_w + pad
    if w is not None:          # fit a chosen width: shorter ribbons first, then narrower wrapping, then smaller font
        f_, rx_, mt_ = fs, wrap_rxn_in, wrap_met_in
        for _ in range(14):
            if fixed() + 2 * 0.55 <= w:
                break
            if rx_ > 1.5:
                rx_, mt_ = rx_ * 0.88, max(1.0, mt_ * 0.9)
            else:
                f_ = max(4.5, f_ * 0.9)
            lh, r_lines, m_lines, g_w, r_w, m_w, gh, rh, mh = layout(f_, rx_, mt_)
        fs = f_
        rib = max(0.55, (w - fixed()) / 2)
    W = fixed() + 2 * rib
    col_h = [sum(c) + gap * (len(c) - 1) for c in (gh, rh, mh) if len(c)]
    Hc = max(col_h + [0.6])
    top_pad = 0.42 + (0.3 if title else 0)
    if h is not None and h - top_pad - 0.32 > Hc:
        Hc = h - top_pad - 0.32
    H = Hc + top_pad + 0.32

    def stack(heights, order):
        """y-centres (inches) of one column; gaps spread so every column spans the same height."""
        k = len(heights)
        tot = sum(heights)
        g_ = gap if k < 2 else max(gap, min((Hc - tot) / (k - 1), 0.5))
        y = Hc - (Hc - (tot + g_ * (k - 1))) / 2
        ys = {}
        for idx in order:
            ys[idx] = y - heights[idx] / 2
            y -= heights[idx] + g_
        return ys

    gy = stack(gh, g_ord)
    ry = stack(rh, list(range(nR)))
    my = stack(mh, m_ord)
    xg = pad + g_w + 0.06
    xr = xg + bw + rib
    xm = xr + r_w + rib
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0.32 / H, 1, Hc / H])
    ax.set_xlim(0, W)
    ax.set_ylim(0, Hc)
    ax.axis("off")

    def slots(node_links, centre):
        used = sum(t for _, t in node_links)
        out, y = {}, centre + used / 2
        for key, t in node_links:
            out[key] = (y - t, y)
            y -= t
        return out

    g_out, r_in_, r_out, m_in_ = {}, {}, {}, {}
    for k in range(len(genes)):
        L = sorted([(j, thick(l[3])) for j, l in enumerate(links) if l[0] == "g" and l[1] == k],
                   key=lambda x: -ry[links[x[0]][2]])
        g_out.update(slots(L, gy[k]))
    for i in range(nR):
        L = sorted([(j, thick(l[3])) for j, l in enumerate(links) if l[0] == "g" and l[2] == i],
                   key=lambda x: -gy[links[x[0]][1]])
        r_in_.update(slots(L, ry[i]))
        L = sorted([(j, thick(l[3])) for j, l in enumerate(links) if l[0] == "m" and l[1] == i],
                   key=lambda x: -my[links[x[0]][2]])
        r_out.update(slots(L, ry[i]))
    for k in range(len(mets)):
        L = sorted([(j, thick(l[3])) for j, l in enumerate(links) if l[0] == "m" and l[2] == k],
                   key=lambda x: -ry[links[x[0]][1]])
        m_in_.update(slots(L, my[k]))

    def ribbon(x0, a, x1, b, col, alpha):
        xm_ = (x0 + x1) / 2
        verts = [(x0, a[1]), (xm_, a[1]), (xm_, b[1]), (x1, b[1]), (x1, b[0]), (xm_, b[0]), (xm_, a[0]),
                 (x0, a[0]), (x0, a[1])]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc=col, ec="none", alpha=alpha, zorder=1))

    colr = lambda v: "#C0392B" if (np.isfinite(v) and v > 0) else ("#2C6FB7" if np.isfinite(v) else "#9AA5A2")
    for j, l in enumerate(links):
        if l[0] == "g":
            ribbon(xg + bw, g_out[j], xr, r_in_[j], colr(l[3]), 0.42)
        else:
            ribbon(xr + r_w, r_out[j], xm, m_in_[j], colr(l[3]), 0.2 if l[4] else 0.45)
    ink, sub = "#1F2A28", "#5D6D6A"
    for k, g in enumerate(genes):
        hgt = gh[k] - 0.04
        ax.add_patch(Rectangle((xg, gy[k] - hgt / 2), bw, hgt, fc="#00695C", ec="none", zorder=3))
        ax.text(xg - 0.06, gy[k], g, fontsize=fs, fontweight="bold", ha="right", va="center", color=ink, zorder=4)
    for i in range(nR):
        hgt = rh[i] - 0.04
        ax.add_patch(FancyBboxPatch((xr, ry[i] - hgt / 2), r_w, hgt, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="#E3F2EF", ec="#00695C", lw=0.7, zorder=3))
        n = len(r_lines[i])
        for q, t in enumerate(r_lines[i]):
            ax.text(xr + 0.08, ry[i] + (n - 1) / 2 * lh - q * lh, t, fontsize=fs, ha="left", va="center",
                    color=ink, zorder=4)
    for k in range(len(mets)):
        hgt = mh[k] - 0.04
        is_sub = mets[k].endswith("(substrate)")
        ax.add_patch(Rectangle((xm, my[k] - hgt / 2), bw, hgt, fc="#B9C4C2" if is_sub else "#00695C", ec="none",
                               zorder=3))
        n = len(m_lines[k])
        for q, t in enumerate(m_lines[k]):
            ax.text(xm + bw + 0.06, my[k] + (n - 1) / 2 * lh - q * lh, t, fontsize=fs, ha="left", va="center",
                    color=sub if is_sub else ink, style="italic" if is_sub else "normal", zorder=4)
    for x, t, ha in ((xg + bw, "Enzymes", "right"), (xr, "Reactions", "left"), (xm, "Metabolites", "left")):
        ax.text(x, Hc + 0.12, t, fontsize=fs + 1.5, fontweight="bold", ha=ha, va="bottom", color=ink, clip_on=False)
    fig.text(pad / W, 0.1 / H, "Ribbon width = |z|;  red = up, blue = down;  pale ribbons and grey bars = substrates "
             "(italic names).", fontsize=fs - 0.5, color=sub, ha="left", va="bottom")
    if title:
        fig.text(pad / W, 1 - 0.08 / H, title, fontsize=fs + 3, fontweight="bold", ha="left", va="top", color=ink)
    return fig
