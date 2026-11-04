"""
factors.py - Engine B: latent factors shared across, or specific to, omics layers (MOFA+).

Uses mofapy2 (Argelaguet et al. 2020). Samples missing from a layer are entered as missing values, so
partially overlapping designs are supported. Factors are kept when they explain >= `min_r2` of variance in
at least one layer; stability is checked by re-fitting with different seeds and matching factors by
absolute correlation of their scores. If mofapy2 is not installed, a block-scaled PCA on complete samples
is used instead (clearly labelled).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from scipy import stats

from .stats import bh

LNAME = {"rna": "mRNA", "prot": "protein", "met": "metabolite"}


def _views(layers: dict, design, top_var: int):
    samples = sorted(set().union(*[set(L.X.columns) for L in layers.values()]) & set(design.meta.index))
    views, names, feats = [], [], []
    for k, L in layers.items():
        X = L.X[[c for c in L.X.columns if c in samples]]
        X = X.loc[X.var(axis=1).sort_values(ascending=False).index[:top_var]]
        X = X.sub(X.mean(axis=1), axis=0)
        M = X.T.reindex(samples)
        views.append(M.values.astype(float))
        names.append(k)
        feats.append(list(X.index))
    return samples, views, names, feats


def _fit(views, names, feats, samples, K, seed, iters):
    with contextlib.redirect_stdout(io.StringIO()):
        from mofapy2.run.entry_point import entry_point
        ent = entry_point()
        ent.set_data_options(scale_views=True)
        ent.set_data_matrix([[v] for v in views], likelihoods=["gaussian"] * len(views), views_names=names,
                            groups_names=["all"], samples_names=[samples], features_names=feats)
        ent.set_model_options(factors=K, spikeslab_weights=True, ard_weights=True)
        ent.set_train_options(iter=iters, convergence_mode="fast", seed=seed, verbose=False, quiet=True)
        ent.build()
        ent.run()
    Z = ent.model.nodes["Z"].getExpectation()
    W = ent.model.nodes["W"].getExpectation()
    return Z, W


def _r2(views, Z, W):
    out = np.zeros((len(views), Z.shape[1]))
    for v, (Y, Wv) in enumerate(zip(views, W)):
        Ys = Y / np.nanstd(Y)
        ok = np.isfinite(Ys)
        tot = np.nansum(Ys ** 2)
        for k in range(Z.shape[1]):
            pred = np.outer(Z[:, k], Wv[:, k]) / np.nanstd(Y)
            out[v, k] = 1 - np.nansum(np.where(ok, (Ys - pred) ** 2, 0)) / tot
    return np.clip(out, 0, 1)


def run_factors(layers: dict, design, n_factors: int = 8, top_var: int = 1500, seed: int = 1,
                iters: int = 1000, n_restarts: int = 3, min_r2: float = 0.02) -> dict:
    samples, views, names, feats = _views(layers, design, top_var)
    method = "MOFA+ (mofapy2)"
    try:
        Z, W = _fit(views, names, feats, samples, n_factors, seed, iters)
    except Exception as e:  # pragma: no cover - fallback path
        method = f"Block-scaled PCA (MOFA+ unavailable: {e.__class__.__name__})"
        complete = np.all([np.isfinite(v).all(1) for v in views], axis=0)
        blocks = [v[complete] / np.sqrt(np.nansum(v[complete] ** 2) / v.shape[1]) / np.sqrt(v.shape[1]) for v in views]
        Xc = np.hstack(blocks)
        U, s, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
        Zc = U[:, :n_factors] * s[:n_factors]
        Z = np.full((len(samples), Zc.shape[1]), np.nan)
        Z[complete] = Zc
        W, start = [], 0
        for v in views:
            W.append(Vt[:n_factors, start:start + v.shape[1]].T)
            start += v.shape[1]
        n_restarts = 1
    R2 = _r2(views, np.nan_to_num(Z), W)
    keep = np.where(R2.max(0) >= min_r2)[0]
    order = keep[np.argsort(-R2[:, keep].sum(0))]
    Z, W, R2 = Z[:, order], [w[:, order] for w in W], R2[:, order]
    fnames = [f"Factor {i + 1}" for i in range(len(order))]
    # stability across seeds
    stab = np.full(len(order), np.nan)
    if method.startswith("MOFA") and n_restarts > 1 and len(order):
        mins = np.ones(len(order))
        for s in range(1, n_restarts):
            Z2, _ = _fit(views, names, feats, samples, n_factors, seed + 100 * s, iters)
            C = np.abs(np.corrcoef(np.nan_to_num(Z).T, np.nan_to_num(Z2).T)[: len(order), len(order):])
            mins = np.minimum(mins, np.nanmax(C, 1))
        stab = mins
    scores = pd.DataFrame(Z, index=samples, columns=fnames)
    r2 = pd.DataFrame(R2 * 100, index=[LNAME.get(n, n) for n in names], columns=fnames)
    # association with group (+ numeric/categorical covariates)
    meta = design.meta.reindex(samples)
    assoc = []
    for f in fnames:
        y = meta["Group"] == design.case
        a, b = scores.loc[y, f].dropna(), scores.loc[~y, f].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else (np.nan, np.nan)
        assoc.append({"factor": f, "covariate": f"Group ({design.case} vs {design.ref})", "statistic": t, "p": p})
        for c in meta.columns:
            if c in ("Group",):
                continue
            col = meta[c]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > 3:
                r, p = stats.spearmanr(scores[f], col, nan_policy="omit")
                assoc.append({"factor": f, "covariate": c, "statistic": r, "p": p})
            elif col.nunique() >= 2 and col.nunique() <= 6:
                grp = [scores.loc[col == lv, f].dropna() for lv in col.dropna().unique()]
                grp = [g for g in grp if len(g) > 1]
                if len(grp) >= 2:
                    h, p = stats.kruskal(*grp)
                    assoc.append({"factor": f, "covariate": c, "statistic": h, "p": p})
    assoc = pd.DataFrame(assoc)
    if len(assoc):
        assoc["q"] = bh(assoc["p"].values)
    loadings = {LNAME.get(n, n): pd.DataFrame(w, index=fe, columns=fnames) for n, w, fe in zip(names, W, feats)}
    summary = pd.DataFrame({"factor": fnames, **{f"R2_{k} (%)": r2.loc[k].values for k in r2.index},
                            "stability (min |r| across seeds)": stab})
    return {"method": method, "scores": scores, "r2": r2, "loadings": loadings, "assoc": assoc,
            "summary": summary, "samples": samples}


def factor_enrichment(fac: dict, layers: dict, kb, n_perm: int = 500, seed: int = 3, min_set: int = 5) -> pd.DataFrame:
    """Competitive enrichment of pathways in each factor's loadings, per layer (feature-label permutation)."""
    rng = np.random.default_rng(seed)
    cur = set(kb.metabolites.loc[kb.metabolites["currency"], "met_base"])
    g_of = kb.gpr.groupby("rxn")["symbol"].apply(set).to_dict()
    m_of = kb.stoich[~kb.stoich["met_base"].isin(cur)].groupby("rxn")["met_base"].apply(set).to_dict()
    pr = kb.pathway_reactions[~kb.pathway_reactions["pathway"].isin({"Transport reactions", "Isolated",
                                                                    "Miscellaneous", "Exchange/demand reactions"})]
    rows = []
    rev = {"mRNA": "rna", "protein": "prot", "metabolite": "met"}
    for lname, Wdf in fac["loadings"].items():
        L = layers[rev[lname]]
        key = L.feat["met_base"] if rev[lname] == "met" else L.feat["symbol"]
        ids = key.reindex(Wdf.index).fillna("").values
        sets = {}
        for pw, grp in pr.groupby("pathway"):
            mem = set()
            for r in grp["rxn"]:
                mem |= (m_of if rev[lname] == "met" else g_of).get(r, set())
            idx = np.where(np.isin(ids, list(mem)))[0]
            if len(idx) >= min_set:
                sets[pw] = idx
        if not sets:
            continue
        Wm = Wdf.values
        Wm = (Wm - Wm.mean(0)) / (Wm.std(0) + 1e-12)
        nf = Wm.shape[0]
        permall = np.argsort(rng.random((n_perm, nf)), axis=1)
        for pw, idx in sets.items():
            obs = Wm[idx].mean(0)
            perm = permall[:, : len(idx)]
            null = Wm[perm].mean(1)
            z = (obs - null.mean(0)) / (null.std(0) + 1e-12)
            for j, f in enumerate(Wdf.columns):
                rows.append({"factor": f, "layer": lname, "pathway": pw, "n": len(idx), "z": z[j],
                             "p": 2 * stats.norm.sf(abs(z[j]))})
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = bh(df["p"].values)
        df = df.sort_values("p")
    return df
