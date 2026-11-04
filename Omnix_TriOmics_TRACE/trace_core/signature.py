"""
signature.py - Engine D: cross-validated integrated biomarker signature.

Early integration (paired samples): multiblock sparse PLS-DA in the spirit of DIABLO (Singh et al. 2019):
each block is standardised on the training fold, the top `keepX` features per block are kept by |t| on
the training fold only, blocks are weighted equally (1/sqrt(keepX)), and a PLS-DA model is fitted.
Late integration (partially overlapping samples): per-layer L2 logistic models, out-of-fold probabilities
averaged over the layers available for each sample.

Validation: nested cross-validation (inner CV tunes keepX; outer repeated stratified K-fold estimates
performance), stability selection across outer folds, bootstrap 95% CI of the AUC and a label-permutation
p-value for the whole procedure. Every feature selection step is inside the folds (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

LNAME = {"rna": "mRNA", "prot": "protein", "met": "metabolite"}


def _t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = X[y == 1], X[y == 0]
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb + 1e-12)


def _fit_predict(blocks, y, tr, te, keepX, ncomp=2):
    Xs_tr, Xs_te, sel = [], [], []
    for B in blocks:
        mu, sd = B[tr].mean(0), B[tr].std(0) + 1e-9
        Ztr, Zte = (B[tr] - mu) / sd, (B[te] - mu) / sd
        k = min(keepX, B.shape[1])
        idx = np.argsort(-np.abs(_t(Ztr, y[tr])))[:k]
        sel.append(idx)
        Xs_tr.append(Ztr[:, idx] / np.sqrt(k))
        Xs_te.append(Zte[:, idx] / np.sqrt(k))
    Xtr, Xte = np.hstack(Xs_tr), np.hstack(Xs_te)
    nc = max(1, min(ncomp, Xtr.shape[1], len(tr) - 2))
    m = PLSRegression(n_components=nc, scale=False).fit(Xtr, y[tr].astype(float))
    return m.predict(Xte).ravel(), sel


def _cv_auc(blocks, y, idx, keepX, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(idx))
    for a, b in skf.split(idx, y[idx]):
        pred[b], _ = _fit_predict(blocks, y, idx[a], idx[b], keepX)
    return roc_auc_score(y[idx], pred) if len(np.unique(y[idx])) == 2 else np.nan


def _outer(blocks, y, grid, n_splits, n_repeats, seed, tune=True, fixed=None):
    n = len(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.full((n_repeats, n), np.nan)
    freq = [np.zeros(B.shape[1]) for B in blocks]
    chosen = []
    for f, (tr, te) in enumerate(rskf.split(np.zeros(n), y)):
        rep = f // n_splits
        if tune and len(grid) > 1:
            inner = max(2, min(3, np.bincount(y[tr]).min()))
            scores = [_cv_auc(blocks, y, tr, k, inner, seed + f) for k in grid]
            kx = grid[int(np.nanargmax(scores))]
        else:
            kx = fixed or grid[0]
        chosen.append(kx)
        oof[rep, te], sel = _fit_predict(blocks, y, tr, te, kx)
        for j, s in enumerate(sel):
            freq[j][s] += 1
    nfold = n_splits * n_repeats
    aucs = np.array([roc_auc_score(y, o) for o in oof])
    return oof, aucs, [f / nfold for f in freq], chosen


def run_signature(layers: dict, design, keepX_grid=(5, 10, 20), n_splits: int = 5, n_repeats: int = 10,
                  n_perm: int = 50, top_var: int = 500, stability_thr: float = 0.7, seed: int = 11,
                  min_per_class: int = 10, force: bool = False) -> dict:
    meta = design.meta
    common = sorted(set.intersection(*[set(L.X.columns) for L in layers.values()]) & set(meta.index))
    y_all = (meta["Group"] == design.case).astype(int)
    y = y_all.reindex(common).values
    counts = np.bincount(y, minlength=2) if len(y) else np.array([0, 0])
    out = {"n_samples": len(common), "class_counts": {design.case: int(counts[1]), design.ref: int(counts[0])}}
    # late fusion is computed whenever possible (uses all samples available in each layer)
    out["late_fusion"] = _late_fusion(layers, design, seed, top_var)
    if counts.min() < 3 or (counts.min() < min_per_class and not force):
        out["status"] = (f"Early-integration signature disabled: {counts.min()} samples in the smallest class "
                         f"with all layers measured (minimum {min_per_class}).")
        return out
    n_splits = int(min(n_splits, counts.min()))
    blocks, names, featnames = [], [], []
    for k, L in layers.items():
        X = L.X[common]
        X = X.loc[X.var(axis=1).sort_values(ascending=False).index[:top_var]]
        blocks.append(X.T.values)
        names.append(k)
        featnames.append(list(X.index))
    oof, aucs, freq, chosen = _outer(blocks, y, list(keepX_grid), n_splits, n_repeats, seed)
    kx = max(set(chosen), key=chosen.count)
    mean_pred = np.nanmean(oof, 0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(1000):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            boots.append(roc_auc_score(y[b], mean_pred[b]))
    perm_aucs = []
    for i in range(n_perm):
        yp = rng.permutation(y)
        _, a, _, _ = _outer(blocks, yp, [kx], n_splits, 1, seed + 1000 + i, tune=False, fixed=kx)
        perm_aucs.append(a.mean())
    perm_aucs = np.array(perm_aucs)
    p_perm = (np.sum(perm_aucs >= aucs.mean()) + 1) / (len(perm_aucs) + 1)
    # per-layer single-block comparison
    single = {}
    for j, k in enumerate(names):
        _, a, _, _ = _outer([blocks[j]], y, [kx], n_splits, max(3, n_repeats // 3), seed + 7, tune=False, fixed=kx)
        single[LNAME[k]] = float(a.mean())
    # stability-selected panel
    panel = []
    for j, k in enumerate(names):
        L = layers[k]
        fr = pd.Series(freq[j], index=featnames[j])
        top = fr[fr >= stability_thr]
        if top.empty:
            top = fr.sort_values(ascending=False).head(kx)
        for f, v in top.sort_values(ascending=False).items():
            lab = L.feat["label"].get(f, f)
            panel.append({"layer": LNAME[k], "feature": f, "label": lab, "selection_frequency": v,
                          "log2FC": float(L.X.loc[f, [c for c in common if y_all[c] == 1]].mean()
                                          - L.X.loc[f, [c for c in common if y_all[c] == 0]].mean())})
    fpr, tpr, _ = roc_curve(y, mean_pred)
    out.update({
        "status": "ok", "keepX": kx, "n_splits": n_splits, "n_repeats": n_repeats,
        "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()),
        "auc_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan),
        "balanced_accuracy": float(balanced_accuracy_score(y, (mean_pred > 0.5).astype(int))),
        "perm_aucs": perm_aucs, "p_perm": float(p_perm), "single_layer_auc": single,
        "panel": pd.DataFrame(panel), "roc": (fpr, tpr), "oof_mean": pd.Series(mean_pred, index=common),
        "y": pd.Series(y, index=common),
    })
    return out


def _late_fusion(layers: dict, design, seed: int, top_var: int) -> dict:
    meta = design.meta
    probs = {}
    for k, L in layers.items():
        cols = [c for c in L.X.columns if c in meta.index]
        y = (meta.loc[cols, "Group"] == design.case).astype(int).values
        if len(cols) < 8 or np.bincount(y, minlength=2).min() < 3:
            continue
        X = L.X[cols]
        X = X.loc[X.var(axis=1).sort_values(ascending=False).index[:top_var]].T.values
        ns = int(min(5, np.bincount(y).min()))
        skf = StratifiedKFold(ns, shuffle=True, random_state=seed)
        pr = np.zeros(len(y))
        for a, b in skf.split(X, y):
            mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
            m = LogisticRegression(C=0.1, max_iter=2000).fit((X[a] - mu) / sd, y[a])
            pr[b] = m.predict_proba((X[b] - mu) / sd)[:, 1]
        probs[LNAME[k]] = pd.Series(pr, index=cols)
    if not probs:
        return {"status": "not enough samples"}
    P = pd.DataFrame(probs)
    fused = P.mean(axis=1)
    y = (meta.loc[fused.index, "Group"] == design.case).astype(int)
    res = {"status": "ok", "auc_fused": float(roc_auc_score(y, fused)),
           "auc_per_layer": {k: float(roc_auc_score((meta.loc[v.index, "Group"] == design.case).astype(int), v))
                             for k, v in probs.items()},
           "n_samples": int(len(fused)), "layers_per_sample": P.notna().sum(axis=1).value_counts().to_dict()}
    return res
