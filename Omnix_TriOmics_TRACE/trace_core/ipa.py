"""
ipa.py - IPA-inspired canonical pathway analysis on Integrix results.

For each pathway (Human-GEM subsystem, plus KEGG sets when present):
  * Overlap: significant molecules (genes/proteins and metabolites, any layer, q < alpha) among the
    measured molecules of the pathway; ratio = significant / all pathway molecules in the knowledge base.
  * p-value: right-tailed Fisher's exact test against all measured, mapped molecules (as IPA does).
  * Directional z-score: (n_up - n_down) / sqrt(n_up + n_down) over significant molecules; IPA-like
    convention z >= 2 'predicted increased activity', z <= -2 'predicted decreased activity'.
    Requires at least 4 directional molecules (otherwise NaN), as in IPA.
This mirrors IPA's display logic but uses molecule direction only (no curated activation relationships),
so it is labelled a directional z-score, not IPA's activation z-score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import bh

SKIP = {"Transport reactions", "Exchange/demand reactions", "Isolated", "Miscellaneous", "Pool reactions",
        "Artificial reactions", "Drug metabolism", "Xenobiotics metabolism"}


def molecule_calls(res, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    f = res.features
    prot = f.get("prot")
    rna = f.get("rna")
    done = set()
    if prot is not None:
        p = prot[prot["ensg"].ne("")].sort_values("q").drop_duplicates("symbol")
        for s, z, q in zip(p["symbol"], p["z"], p["q"]):
            rows.append(("gene", s, z, q, "protein"))
            done.add(s)
    if rna is not None:
        r = rna[rna["ensg"].ne("")].sort_values("q").drop_duplicates("symbol")
        for s, z, q in zip(r["symbol"], r["z"], r["q"]):
            if s in done:
                # protein present: use mRNA only if protein not significant and mRNA is
                continue
            rows.append(("gene", s, z, q, "mRNA"))
    m = f["met"][f["met"]["met_base"].ne("")].sort_values("q").drop_duplicates("met_base")
    for b, z, q, n in zip(m["met_base"], m["z"], m["q"], m["kb_name"]):
        rows.append(("metabolite", b, z, q, "metabolite"))
    d = pd.DataFrame(rows, columns=["kind", "id", "z", "q", "layer"])
    d["significant"] = d["q"] < alpha
    return d


def canonical_pathways(res, kb, alpha: float = 0.05, min_size: int = 3) -> pd.DataFrame:
    calls = molecule_calls(res, alpha)
    cur = set(kb.metabolites.loc[kb.metabolites["currency"], "met_base"])
    g_of = kb.gpr.groupby("rxn")["symbol"].apply(set).to_dict()
    m_of = kb.stoich[~kb.stoich["met_base"].isin(cur)].groupby("rxn")["met_base"].apply(set).to_dict()
    mname = dict(zip(kb.metabolites.met_base, kb.metabolites.name))
    sets = {}
    pr = kb.pathway_reactions[~kb.pathway_reactions["pathway"].isin(SKIP)]
    for pw, grp in pr.groupby("pathway"):
        gs, ms = set(), set()
        for r in grp["rxn"]:
            gs |= g_of.get(r, set())
            ms |= m_of.get(r, set())
        sets[pw] = gs | ms
    if "kegg_gene_sets" in kb.extra:
        ent2sym = {}
        for s, e in zip(kb.genes.symbol, kb.genes.entrez):
            for x in str(e).split(";"):
                ent2sym[x] = s
        k2b = dict(zip(kb.metabolites.kegg, kb.metabolites.met_base))
        kc = kb.extra.get("kegg_compound_sets", pd.DataFrame(columns=["pathway", "kegg"]))
        for pw, grp in kb.extra["kegg_gene_sets"].groupby("pathway"):
            sets[f"KEGG: {pw}"] = {ent2sym[e] for e in grp["entrez"] if e in ent2sym} | \
                {k2b[c] for c in kc.loc[kc["pathway"] == pw, "kegg"] if c in k2b}
    universe = set(calls["id"])
    sig = set(calls.loc[calls["significant"], "id"])
    zmap = dict(zip(calls["id"], calls["z"]))
    N, K = len(universe), len(sig)
    rows = []
    for pw, mem in sets.items():
        meas = mem & universe
        if len(meas) < min_size:
            continue
        hit = meas & sig
        k, n = len(hit), len(meas)
        p = stats.hypergeom.sf(k - 1, N, K, n) if k else 1.0
        up = sorted([h for h in hit if zmap[h] > 0])
        dn = sorted([h for h in hit if zmap[h] < 0])
        z = (len(up) - len(dn)) / np.sqrt(len(up) + len(dn)) if len(up) + len(dn) >= 4 else np.nan
        lab = lambda x: mname.get(x, x)
        rows.append({"pathway": pw, "n_pathway": len(mem), "n_measured": n, "n_significant": k,
                     "ratio": k / len(mem) if mem else np.nan, "p": p, "minus_log10_p": -np.log10(max(p, 1e-300)),
                     "z_directional": z, "n_up": len(up), "n_down": len(dn),
                     "molecules_up": ", ".join(lab(x) for x in up), "molecules_down": ", ".join(lab(x) for x in dn)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q"] = bh(df["p"].values)
    df["prediction"] = np.where(df["z_directional"] >= 2, "Increased activity",
                                np.where(df["z_directional"] <= -2, "Decreased activity",
                                         np.where(df["z_directional"].isna(), "No activity pattern", "Not decisive")))
    return df.sort_values("p").reset_index(drop=True)


def z_color(z: float, zmax: float = 4.0) -> str:
    """IPA-like bar colour: orange = positive, blue = negative, white ~ 0, grey = no pattern."""
    if not np.isfinite(z):
        return "#BDBDBD"
    t = min(abs(z) / zmax, 1.0)
    hi = np.array([230, 97, 1]) if z > 0 else np.array([33, 102, 172])
    c = np.array([255, 255, 255]) + (hi - 255) * t
    return "#%02x%02x%02x" % tuple(c.round().astype(int))
