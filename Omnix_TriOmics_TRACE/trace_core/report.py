"""report.py - exports: Excel workbook, run manifest, auto-generated methods, 600-DPI figures, ZIP bundle."""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import platform
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import __version__  # noqa: E402
from .figures import CLASS_COLORS, PATTERN_COLORS  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})


def excel(res, extras: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"item": ["Contrast", "Evidence tier", "Knowledge base", "Permutations (label)",
                               "Permutations (topology)", "FDR alpha", "Pattern threshold (z)"],
                      "value": [f"{res.design.case} vs {res.design.ref}", res.design.tier_text,
                                extras.get("kb_version", ""), res.params["B"], res.params["B_topo"],
                                res.params["alpha"], res.params["pattern_thr"]]}).to_excel(xw, sheet_name="Summary", index=False)
        res.reactions.to_excel(xw, sheet_name="Reactions", index=False)
        res.pathways.to_excel(xw, sheet_name="Pathways", index=False)
        if res.concordance is not None:
            res.concordance.to_excel(xw, sheet_name="Gene-protein concordance", index=False)
        res.reporter_mets.to_excel(xw, sheet_name="Reporter metabolites", index=False)
        if res.regulators is not None:
            res.regulators.to_excel(xw, sheet_name="Regulators (TF)", index=False)
        for k, t in res.features.items():
            t.reset_index().rename(columns={"index": "feature"}).to_excel(xw, sheet_name=f"Stats {k}"[:31], index=False)
        if extras.get("targets") is not None:
            extras["targets"].to_excel(xw, sheet_name="Targets", index=False)
        if extras.get("factors"):
            f = extras["factors"]
            f["summary"].to_excel(xw, sheet_name="Factors", index=False)
            f["assoc"].to_excel(xw, sheet_name="Factor associations", index=False)
            if extras.get("factor_enrichment") is not None and len(extras["factor_enrichment"]):
                extras["factor_enrichment"].to_excel(xw, sheet_name="Factor enrichment", index=False)
        if extras.get("signature") and extras["signature"].get("status") == "ok":
            s = extras["signature"]
            s["panel"].to_excel(xw, sheet_name="Signature panel", index=False)
            pd.DataFrame({"metric": ["AUC (mean over repeats)", "AUC SD", "AUC 95% CI low", "AUC 95% CI high",
                                     "Balanced accuracy", "Permutation p", "keepX per block"],
                          "value": [s["auc_mean"], s["auc_sd"], s["auc_ci"][0], s["auc_ci"][1],
                                    s["balanced_accuracy"], s["p_perm"], s["keepX"]]}).to_excel(
                xw, sheet_name="Signature performance", index=False)
        pd.DataFrame({"log": res.log}).to_excel(xw, sheet_name="Processing log", index=False)
    return buf.getvalue()


def manifest(res, inputs: dict, kb, extras: dict) -> dict:
    return {
        "software": f"Omnix Integrix {__version__}",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "knowledge_base": kb.manifest,
        "inputs_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in inputs.items()},
        "design": {"case": res.design.case, "reference": res.design.ref, "tier": res.design.tier,
                   "overlap": res.design.overlap},
        "parameters": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in res.params.items()},
        "engines": {"factors": extras.get("factor_params"), "signature": extras.get("signature_params"),
                    "network": extras.get("network_params")},
        "timings_s": {k: round(v, 2) for k, v in res.timings.items()},
    }


def methods_text(res, kb, extras: dict) -> str:
    P = res.params
    t = [f"## Methods: tri-omics integration (Omnix Integrix {__version__})", ""]
    t.append(
        f"Transcriptomics, proteomics and metabolomics data were integrated with Omnix Integrix. The contrast was {res.design.case} versus {res.design.ref}; "
        f"{res.design.tier_text.lower()}. Data were log2-transformed where needed, features missing in more "
        f"than 50% of samples in every group were removed and remaining missing values were imputed with the "
        f"feature minimum minus one log2 unit. Batch effects were corrected within each layer only, with the "
        f"group effect protected.")
    t.append("")
    t.append(
        "Per-layer differential statistics were moderated t-statistics with empirical-Bayes variance "
        "shrinkage (Smyth 2004), converted to signed z-scores so that layers on different scales are "
        "integrated on a common scale. Gene-protein concordance was assessed by orthogonal (Deming) regression "
        "of protein on mRNA z-scores and a per-gene discordance statistic; within-sample coupling was the "
        "Spearman correlation of group-centred mRNA and protein levels.")
    t.append("")
    t.append(
        f"Genes, proteins and metabolites were mapped to the Human-GEM genome-scale metabolic model "
        f"(knowledge base {kb.version}) by Ensembl, UniProt, symbol, HMDB, KEGG, ChEBI or name. For every "
        f"observable reaction (at least one measured enzyme and one measured non-currency metabolite; transport, "
        f"exchange and pool reactions excluded) enzyme evidence E was obtained from protein z-scores through "
        f"the gene-protein-reaction rule (AND = complex, minimum absolute evidence with sign agreement; OR = "
        f"isozymes, abundance-weighted mean), using mRNA z-scores scaled by the observed mRNA-protein coupling "
        f"when the protein was not measured. The metabolite balance M was the difference between the mean "
        f"z-score of products and of substrates divided by sqrt(2), and reaction coherence was C = E x M.")
    t.append("")
    t.append(
        f"Statistical significance was calibrated by {P['B']} sample-label permutations applied jointly to all "
        f"layers (preserving cross-omics dependence of matched samples) and by {P['B_topo']} topology "
        f"permutations that shuffle measured values among nodes of similar network degree. P-values for C were "
        f"computed from the product-normal distribution with location, scale and correlation estimated from the "
        f"permutations; q-values used the Benjamini-Hochberg procedure. A reaction was called coherent when "
        f"both label and topology q < {P['alpha']}. Interpretive patterns (e.g. candidate bottleneck, "
        f"non-abundance regulation) were assigned with a |z| threshold of {P['pattern_thr']} and describe "
        f"associations, not causal mechanisms.")
    t.append("")
    t.append(
        "Pathway activity per layer was the Stouffer sum of member z-scores standardised against the label "
        "permutations; layers were combined by a weighted Stouffer test, heterogeneity was assessed by "
        "Cochran's Q and I2, and pathways were classified as shared-concordant, shared-discordant or "
        "omics-specific. Pathways with fewer than "
        f"{P['min_set']} measured members in a layer were labelled not assessable in that layer. Reporter "
        "metabolites (Patil & Nielsen 2005) summarised the coherence of reactions around each metabolite.")
    if extras.get("targets") is not None:
        t += ["", "Targets were prioritised by robust rank aggregation (Kolde et al. 2012) of reaction coherence, "
              "mRNA-protein consistency, choke-point status (Yeh et al. 2004) and network propagation (random "
              "walk with restart, restart probability 0.3, degree-binned seed permutation null)."]
    if extras.get("factors"):
        t += ["", f"Latent factors were estimated with {extras['factors']['method']} (Argelaguet et al. 2020) "
              "on the most variable features per layer; factors explaining at least 2% of variance in one layer "
              "were retained and their stability was assessed across random seeds."]
    s = extras.get("signature")
    if s and s.get("status") == "ok":
        t += ["", f"An integrated signature was derived with multiblock sparse PLS-DA in a nested cross-validation "
              f"({s['n_splits']}-fold, {s['n_repeats']} repeats; inner CV tuned features per block); stability "
              f"selection retained features selected in at least 70% of outer folds. Performance: AUC "
              f"{s['auc_mean']:.2f} (95% bootstrap CI {s['auc_ci'][0]:.2f}-{s['auc_ci'][1]:.2f}), label-"
              f"permutation p = {s['p_perm']:.3f}. External validation is required before clinical use."]
    t += ["", "References: Robinson et al., Sci Signal 2020 (Human-GEM); Smyth, Stat Appl Genet Mol Biol 2004; "
          "Patil & Nielsen, PNAS 2005; Yeh et al., Genome Res 2004; Kolde et al., Bioinformatics 2012; "
          "Argelaguet et al., Genome Biol 2020; Singh et al., Bioinformatics 2019; Benjamini & Hochberg, JRSS-B 1995."]
    return "\n".join(t)


# --------------------------------------------------------------------------- 600-DPI publication figures
def _save(fig, fmt: str, dpi: int) -> bytes:
    b = io.BytesIO()
    fig.savefig(b, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return b.getvalue()


def pub_figures(res, extras: dict, fmt: str = "png", dpi: int = 600) -> dict:
    out = {}
    if res.concordance is not None:
        c = res.concordance
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        for cls, col in CLASS_COLORS.items():
            d = c[c["class"] == cls]
            ax.scatter(d["z_mRNA"], d["z_protein"], s=5 if cls == "Unchanged" else 10, c=col, lw=0,
                       alpha=0.5 if cls == "Unchanged" else 0.9, label=f"{cls} ({len(d)})")
        lim = np.nanmax(np.abs(c[["z_mRNA", "z_protein"]].values)) * 1.05
        b = res.coupling["beta"]
        ax.plot([-lim, lim], [-b * lim, b * lim], "k--", lw=0.8)
        for _, r in c[c["class"] != "Unchanged"].nsmallest(8, "p_discordance").iterrows():
            ax.annotate(r["gene"], (r["z_mRNA"], r["z_protein"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, c="#bbb", lw=0.5)
        ax.axvline(0, c="#bbb", lw=0.5)
        ax.set(xlabel="mRNA z-score", ylabel="Protein z-score", xlim=(-lim, lim), ylim=(-lim, lim))
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        out[f"concordance.{fmt}"] = _save(fig, fmt, dpi)
    rx = res.reactions
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for pat, col in PATTERN_COLORS.items():
        d = rx[rx["pattern"] == pat]
        if len(d):
            ax.scatter(d["E"], d["M"], s=4 + 4 * np.clip(-np.log10(d["q_label"].fillna(1.0).clip(1e-12, 1.0)), 0, 6), c=col, lw=0,
                       alpha=0.85, label=pat)
    for _, r in rx[rx["significant"]].head(8).iterrows():
        ax.annotate(str(r["enzymes"]).split(",")[0], (r["E"], r["M"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, c="#999", lw=0.5)
    ax.axvline(0, c="#999", lw=0.5)
    ax.set(xlabel="Enzyme evidence E (z)", ylabel="Metabolite balance M")
    ax.legend(fontsize=5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    out[f"reaction_patterns.{fmt}"] = _save(fig, fmt, dpi)
    pw = res.pathways.dropna(subset=["z_combined"]).head(20).iloc[::-1]
    cols = [c for c in ("mRNA", "protein", "metabolite") if f"z_{c}" in pw.columns] + ["combined"]
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(pw) + 1))
    for i, (_, r) in enumerate(pw.iterrows()):
        for j, c in enumerate(cols):
            z = r[f"z_{c}"]
            q = r[f"q_{c}"]
            if np.isfinite(z):
                ax.scatter(j, i, s=10 + 12 * min(6, -np.log10(max(q, 1e-12))), c=[z],
                           cmap="RdBu_r", vmin=-5, vmax=5, edgecolors="#444", linewidths=0.4)
        ax.text(len(cols) - 0.4, i, r["class"], fontsize=5.5, va="center", color="#00695C")
    ax.set_yticks(range(len(pw)))
    ax.set_yticklabels([p[:55] for p in pw["pathway"]], fontsize=6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_xlim(-0.5, len(cols) + 1.6)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-5, 5))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01).set_label("z", fontsize=7)
    out[f"pathways.{fmt}"] = _save(fig, fmt, dpi)
    s = extras.get("signature")
    if s and s.get("status") == "ok":
        fpr, tpr = s["roc"]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(fpr, tpr, c="#00695C", lw=1.6, label=f"AUC {s['auc_mean']:.2f}")
        ax.plot([0, 1], [0, 1], ":", c="#999")
        ax.set(xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(frameon=False, loc="lower right")
        out[f"signature_roc.{fmt}"] = _save(fig, fmt, dpi)
    f = extras.get("factors")
    if f:
        r2 = f["r2"]
        fig, ax = plt.subplots(figsize=(0.55 * r2.shape[1] + 1.4, 1.6))
        im = ax.imshow(r2.values, cmap="BuGn", vmin=0)
        for i in range(r2.shape[0]):
            for j in range(r2.shape[1]):
                ax.text(j, i, f"{r2.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(r2.shape[1]))
        ax.set_xticklabels(r2.columns, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(r2.shape[0]))
        ax.set_yticklabels(r2.index, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04).set_label("% variance", fontsize=6)
        out[f"factor_variance.{fmt}"] = _save(fig, fmt, dpi)
    return out


def bundle(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data if isinstance(data, (bytes, str)) else json.dumps(data, indent=2, default=str))
    return b.getvalue()
