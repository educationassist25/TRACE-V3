"""
engine.py - runs Engine A (cascade), the cross-layer pathway analysis, reporter metabolites and
regulator inference, with label-permutation and topology nulls and Benjamini-Hochberg FDR.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sst

from .cascade import PATTERN_HELP, CascadeIndex, pattern
from .stats import TwoGroup, bh, calibrated_product_p, empirical_p, normal_p_from_null, permutation_labels

DEFAULTS = dict(B=1000, B_topo=1000, seed=20261030, alpha=0.05, pattern_thr=1.5, min_set=3,
                moderated=True, layer_q=0.05)


@dataclass
class TraceResult:
    params: dict
    design: object
    features: dict                    # layer -> feature statistics table
    concordance: pd.DataFrame | None
    coupling: dict
    reactions: pd.DataFrame
    pathways: pd.DataFrame
    reporter_mets: pd.DataFrame
    regulators: pd.DataFrame | None
    gene_evidence: pd.DataFrame
    log: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    cascade: object = None
    null_C: np.ndarray | None = None


def _rep(feat: pd.DataFrame, key: str) -> pd.Series:
    """Representative feature per identifier (highest mean intensity)."""
    f = feat[feat[key].astype(str).ne("")]
    f = f.sort_values("mean_log2", ascending=False)
    f = f[~f[key].duplicated()]
    return pd.Series(f.index, index=f[key].values)


def _deming(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return 1.0
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y)[0, 1]
    if abs(sxy) < 1e-12:
        return 0.0
    return float((syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy))


def _bins(deg: np.ndarray, nb: int = 5) -> np.ndarray:
    if len(deg) == 0:
        return deg
    qs = np.unique(np.quantile(deg, np.linspace(0, 1, nb + 1)))
    return np.digitize(deg, qs[1:-1])


def _shuffle_within(v: np.ndarray, bins: np.ndarray, B: int, rng) -> np.ndarray:
    out = np.empty((len(v), B))
    for b in range(B):
        w = v.copy()
        for lv in np.unique(bins):
            idx = np.where(bins == lv)[0]
            w[idx] = v[rng.permutation(idx)]
        out[:, b] = w
    return out


def run_trace(layers: dict, design, kb, params: dict | None = None, progress=None) -> TraceResult:
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run_trace(layers, design, kb, params, progress)


def _run_trace(layers: dict, design, kb, params: dict | None = None, progress=None) -> TraceResult:
    P = {**DEFAULTS, **(params or {})}
    T0 = time.time()
    timings, log = {}, []
    say = progress or (lambda *a, **k: None)
    rng = np.random.default_rng(P["seed"])
    groups = design.meta["Group"].to_dict()
    union = sorted(set().union(*[set(L.X.columns) for L in layers.values()]) & set(groups))
    Lab, samples = permutation_labels(union, groups, design.case, P["B"], P["seed"])
    pos = {s: i for i, s in enumerate(samples)}
    log.append(f"{design.tier_text}. {len(union)} samples in total; {Lab.shape[0] - 1} label permutations.")

    # ------------------------------------------------------------------ 1. per-layer statistics
    say(0.05, "Per-layer statistics")
    feats, Zperm = {}, {}
    need = {}
    for k, L in layers.items():
        cols = [c for c in L.X.columns if c in groups]
        X = L.X[cols]
        idx = np.array([pos[c] for c in cols])
        Ll = Lab[:, idx]
        tg = TwoGroup(X.values, Ll[0], moderated=P["moderated"])
        obs = tg.run(Ll[:1])
        t = L.feat.copy()
        for kk in ("logfc", "se", "t", "p", "z"):
            t[kk] = obs[kk][:, 0]
        t["q"] = bh(t["p"].values)
        t["n_case"] = int(Ll[0].sum())
        t["n_ref"] = int((1 - Ll[0]).sum())
        feats[k] = t
        # features carried through the permutations: KB-mapped ones only
        rep = _rep(t, "met_base") if k == "met" else _rep(t[t["ensg"].ne("")], "symbol")
        need[k] = rep
        rows = [X.index.get_loc(f) for f in rep.values]
        sub = TwoGroup.__new__(TwoGroup)
        sub.X = X.values[rows]
        sub.X2 = sub.X ** 2
        sub.d, sub.d0, sub.s0 = tg.d, tg.d0, tg.s0
        Z = np.empty((len(rows), Ll.shape[0]))
        ok = (Ll.sum(1) >= 2) & ((1 - Ll).sum(1) >= 2)
        for a in range(0, Ll.shape[0], 250):
            blk = Ll[a:a + 250]
            Z[:, a:a + 250] = sub.run(np.where(ok[a:a + 250, None], blk, Ll[:1]))["z"]
        Z[:, ~ok] = np.nan
        Zperm[k] = pd.DataFrame(Z, index=rep.index)
        log += [f"[{L.name}] " + x for x in L.log]
    timings["layer statistics"] = time.time() - T0

    # ------------------------------------------------------------------ 2. translation concordance
    say(0.25, "Translation concordance")
    conc, coupling = None, {"rho_hat": 0.5, "beta": 1.0, "n_genes": 0}
    if "rna" in feats and "prot" in feats:
        r = feats["rna"]
        p = feats["prot"]
        rr = r.sort_values("mean_log2", ascending=False).drop_duplicates("symbol").set_index("symbol")
        pp = p.sort_values("mean_log2", ascending=False).drop_duplicates("symbol").set_index("symbol")
        common = sorted(set(rr.index) & set(pp.index) - {""})
        zr, zp = rr.loc[common, "z"].values, pp.loc[common, "z"].values
        beta = _deming(zr, zp)
        rho_hat = float(np.clip(np.corrcoef(zr, zp)[0, 1], 0.2, 1.0)) if len(common) > 5 else 0.5
        dlt = (zp - beta * zr) / np.sqrt(1 + beta ** 2)
        pd_ = 2 * sst.norm.sf(np.abs(dlt))
        qr, qp = rr.loc[common, "q"].values, pp.loc[common, "q"].values
        a = P["layer_q"]
        cls = np.where((qr < a) & (qp < a) & (np.sign(zr) == np.sign(zp)), "Coupled",
              np.where((qr < a) & (qp < a), "Opposing",
              np.where(qp < a, "Protein-only (post-transcriptional)",
              np.where(qr < a, "mRNA-only (buffered)", "Unchanged"))))
        conc = pd.DataFrame({"gene": common, "z_mRNA": zr, "z_protein": zp,
                             "log2FC_mRNA": rr.loc[common, "logfc"].values,
                             "log2FC_protein": pp.loc[common, "logfc"].values,
                             "q_mRNA": qr, "q_protein": qp, "discordance": dlt, "p_discordance": pd_,
                             "q_discordance": bh(pd_), "class": cls})
        conc["metabolic_enzyme"] = conc["gene"].isin(set(kb.gpr["symbol"]))
        # within-sample coupling (paired samples only)
        shared = [s for s in layers["rna"].X.columns if s in layers["prot"].X.columns and s in groups]
        if len(shared) >= 8:
            g = pd.Series({s: groups[s] for s in shared})
            rid = _rep(feats["rna"], "symbol")
            pid = _rep(feats["prot"], "symbol")
            xr = layers["rna"].X.loc[rid.loc[common].values, shared].values
            xp = layers["prot"].X.loc[pid.loc[common].values, shared].values

            def resid(M):
                M = M.copy()
                for lv in g.unique():
                    idx = np.where(g.values == lv)[0]
                    M[:, idx] -= M[:, idx].mean(1, keepdims=True)
                return M
            rr_ = sst.rankdata(resid(xr), axis=1)
            rp_ = sst.rankdata(resid(xp), axis=1)
            rr_ -= rr_.mean(1, keepdims=True)
            rp_ -= rp_.mean(1, keepdims=True)
            rho = (rr_ * rp_).sum(1) / np.sqrt((rr_ ** 2).sum(1) * (rp_ ** 2).sum(1) + 1e-12)
            conc["within_sample_rho"] = rho
            n = len(shared)
            tt = rho * np.sqrt((n - 2) / np.maximum(1 - rho ** 2, 1e-12))
            conc["p_rho"] = 2 * sst.t.sf(np.abs(tt), n - 2)
            conc["q_rho"] = bh(conc["p_rho"].values)
            coupling["median_within_rho"] = float(np.median(rho))
            coupling["n_paired"] = n
        coupling.update(rho_hat=rho_hat, beta=beta, n_genes=len(common),
                        pearson_z=float(np.corrcoef(zr, zp)[0, 1]) if len(common) > 5 else np.nan)
        conc = conc.sort_values("p_discordance")
    timings["concordance"] = time.time() - T0

    # ------------------------------------------------------------------ 3. gene-level enzyme evidence
    say(0.35, "Reaction cascade")
    kb_genes = set(kb.gpr["symbol"])
    Zr = Zperm.get("rna", pd.DataFrame())
    Zp = Zperm.get("prot", pd.DataFrame())
    genes = sorted((set(Zr.index) | set(Zp.index)) & kb_genes)
    nB = Lab.shape[0]
    Eg = np.full((len(genes), nB), np.nan)
    src = []
    rho_hat = coupling["rho_hat"]
    for i, gname in enumerate(genes):
        if gname in Zp.index:
            Eg[i] = Zp.loc[gname].values
            src.append("protein")
        else:
            Eg[i] = rho_hat * Zr.loc[gname].values
            src.append("mRNA proxy")
    gene_ev = pd.DataFrame({"gene": genes, "evidence_source": src, "E_obs": Eg[:, 0]})
    if "met" not in Zperm:
        raise ValueError("Engine A needs a metabolomics layer")
    Zm = Zperm["met"]
    met_bases = list(Zm.index)
    pw_int = {}
    if "prot" in feats:
        pf = feats["prot"].sort_values("mean_log2", ascending=False).drop_duplicates("symbol")
        pw_int = dict(zip(pf["symbol"], 2 ** pf["mean_log2"]))
    cx = CascadeIndex(kb, genes, met_bases, set(Zp.index), pw_int)
    if not cx.rxns:
        raise ValueError("No observable reactions: check gene and metabolite identifier mapping")
    E, S, Pp, M, C = cx.score(np.where(np.isfinite(Eg), Eg, 0.0), np.nan_to_num(Zm.values))
    bad = ~np.isfinite(Eg).all(0) | ~np.isfinite(Zm.values).all(0)
    C[:, bad] = np.nan
    E[:, bad] = np.nan
    M[:, bad] = np.nan
    pl_emp = empirical_p(C[:, 0], C[:, 1:])
    pl, rho_null = calibrated_product_p(E, M)
    # topology null
    Bt = int(P["B_topo"])
    gb, mb = _bins(cx.gene_degree), _bins(cx.met_degree)
    Egn = _shuffle_within(np.nan_to_num(Eg[:, 0]), gb, Bt, rng)
    Zmn = _shuffle_within(np.nan_to_num(Zm.values[:, 0]), mb, Bt, rng)
    Et, _, _, Mt, Ct = cx.score(Egn, Zmn)
    pt, _ = calibrated_product_p(np.column_stack([E[:, 0], Et]), np.column_stack([M[:, 0], Mt]))
    timings["cascade"] = time.time() - T0

    mname = dict(zip(kb.metabolites.met_base, kb.metabolites.name))
    rx = cx.info.copy()
    rx["enzymes"] = [", ".join(cx.rx_genes.get(r, [])) for r in cx.rxns]
    rx["enzyme_evidence"] = ["protein" if cx.prot_measured.get(r) else "mRNA proxy" for r in cx.rxns]
    rx["substrates"] = ["; ".join(mname.get(m, m) for m in cx.sub_ids.get(r, [])) for r in cx.rxns]
    rx["products"] = ["; ".join(mname.get(m, m) for m in cx.pro_ids.get(r, [])) for r in cx.rxns]
    rx["E"], rx["S"], rx["P"], rx["M"], rx["C"] = E[:, 0], S[:, 0], Pp[:, 0], M[:, 0], C[:, 0]
    nullsd = np.nanstd(C[:, 1:], axis=1)
    rx["C_std"] = (C[:, 0] - np.nanmean(C[:, 1:], axis=1)) / np.where(nullsd > 0, nullsd, np.nan)
    rx["p_label"], rx["p_topology"] = pl, pt
    rx["p_label_empirical"] = pl_emp
    zE, pE = normal_p_from_null(E[:, 0], E[:, 1:])
    zM, pM = normal_p_from_null(M[:, 0], M[:, 1:])
    rx["z_enzyme"], rx["p_enzyme"], rx["q_enzyme"] = zE, pE, bh(pE)
    rx["z_balance"], rx["p_balance"], rx["q_balance"] = zM, pM, bh(pM)
    rx["null_EM_correlation"] = rho_null
    rx["q_label"], rx["q_topology"] = bh(pl), bh(pt)
    rx["significant"] = (rx["q_label"] < P["alpha"]) & (rx["q_topology"] < P["alpha"])
    # metabolite-driven steps: the metabolite balance changes while enzyme abundance does not
    rx["metabolite_driven"] = (rx["q_balance"] < P["alpha"]) & (rx["E"].abs() < P["pattern_thr"])
    rx["pattern"] = [pattern(e, s, p_, P["pattern_thr"]) for e, s, p_ in zip(rx["E"], rx["S"], rx["P"])]
    rx["interpretation"] = rx["pattern"].map(PATTERN_HELP)
    rx["call"] = np.where(rx["significant"], "Coherence (enzyme + metabolites)",
                          np.where(rx["metabolite_driven"], "Metabolite-driven (enzyme unchanged)", ""))
    rx = rx.sort_values(["significant", "p_label", "C"], ascending=[False, True, False]).reset_index(drop=True)

    # ------------------------------------------------------------------ 4. pathways
    say(0.6, "Pathways: shared vs omics-specific")
    pwy = _pathways(kb, Zperm, cx, C, P)
    timings["pathways"] = time.time() - T0

    # ------------------------------------------------------------------ 5. reporter metabolites
    say(0.75, "Reporter metabolites and regulators")
    rep = _reporter_metabolites(cx, C, mname, P)
    reg = _regulators(kb, feats, rx, P)
    timings["reporters/regulators"] = time.time() - T0

    say(1.0, "Done")
    log.append(f"{len(cx.rxns)} observable reactions; {int(rx['significant'].sum())} significant "
               f"(label and topology q < {P['alpha']}).")
    return TraceResult(params=P, design=design, features=feats, concordance=conc, coupling=coupling,
                       reactions=rx, pathways=pwy, reporter_mets=rep, regulators=reg, gene_evidence=gene_ev,
                       log=log, timings=timings, cascade=cx, null_C=None)


def _pathways(kb, Zperm: dict, cx: CascadeIndex, C: np.ndarray, P: dict) -> pd.DataFrame:
    pr = kb.pathway_reactions
    cur = set(kb.metabolites.loc[kb.metabolites["currency"], "met_base"])
    g_of = kb.gpr.groupby("rxn")["symbol"].apply(set).to_dict()
    st = kb.stoich[~kb.stoich["met_base"].isin(cur)]
    m_of = st.groupby("rxn")["met_base"].apply(set).to_dict()
    sets = {}
    skip = {"Transport reactions", "Exchange/demand reactions", "Isolated", "Miscellaneous", "Pool reactions",
            "Artificial reactions", "Drug metabolism", "Xenobiotics metabolism"}
    for pw, grp in pr[~pr["pathway"].isin(skip)].groupby("pathway"):
        gs, ms = set(), set()
        for r in grp["rxn"]:
            gs |= g_of.get(r, set())
            ms |= m_of.get(r, set())
        sets[pw] = {"rna": gs, "prot": gs, "met": ms, "rxn": set(grp["rxn"])}
    # user / KEGG gene + compound sets
    if "kegg_gene_sets" in kb.extra:
        ent2sym = {}
        for s, e in zip(kb.genes.symbol, kb.genes.entrez):
            for x in str(e).split(";"):
                ent2sym[x] = s
        k2b = dict(zip(kb.metabolites.kegg, kb.metabolites.met_base))
        kg = kb.extra["kegg_gene_sets"]
        kc = kb.extra.get("kegg_compound_sets", pd.DataFrame(columns=["pathway", "kegg"]))
        for pw, grp in kg.groupby("pathway"):
            gs = {ent2sym[e] for e in grp["entrez"] if e in ent2sym}
            ms = {k2b[c] for c in kc.loc[kc["pathway"] == pw, "kegg"] if c in k2b} - cur
            sets[f"KEGG: {pw}"] = {"rna": gs, "prot": gs, "met": ms, "rxn": set()}
    lay = [k for k in ("rna", "prot", "met") if k in Zperm]
    rows, zst_all = [], {}
    names = list(sets)
    for k in lay:
        Z = Zperm[k]
        fidx = {f: i for i, f in enumerate(Z.index)}
        Zv = np.nan_to_num(Z.values)
        S = np.full((len(names), Z.shape[1]), np.nan)
        n = np.zeros(len(names), int)
        for i, pw in enumerate(names):
            mem = [fidx[f] for f in sets[pw][k] if f in fidx]
            n[i] = len(mem)
            if len(mem) >= P["min_set"]:
                S[i] = Zv[mem].sum(0) / np.sqrt(len(mem))
        bad = ~np.isfinite(Z.values).all(0)
        S[:, bad] = np.nan
        mu = np.nanmean(S[:, 1:], 1, keepdims=True)
        sd = np.nanstd(S[:, 1:], 1, keepdims=True)
        zst = (S - mu) / np.where(sd > 0, sd, np.nan)
        zst_all[k] = (zst, n)
    out = pd.DataFrame({"pathway": names})
    lname = {"rna": "mRNA", "prot": "protein", "met": "metabolite"}
    comb_num = 0
    comb_den = 0
    for k in lay:
        zst, n = zst_all[k]
        out[f"n_{lname[k]}"] = n
        out[f"z_{lname[k]}"] = zst[:, 0]
        pk = 2 * sst.norm.sf(np.abs(zst[:, 0]))
        out[f"p_{lname[k]}"] = pk
        out[f"q_{lname[k]}"] = bh(pk)
        w = np.where(n >= P["min_set"], np.sqrt(n), 0.0)[:, None]
        comb_num = comb_num + w * np.nan_to_num(zst)
        comb_den = comb_den + w ** 2
    comb = comb_num / np.sqrt(np.where(comb_den > 0, comb_den, np.nan))
    zc, pc = normal_p_from_null(comb[:, 0], comb[:, 1:])
    out["z_combined"] = zc
    out["p_combined"], out["q_combined"] = pc, bh(pc)
    zs = np.column_stack([np.where(out[f"n_{lname[k]}"] >= P["min_set"], out[f"z_{lname[k]}"], np.nan) for k in lay])
    nl = np.isfinite(zs).sum(1)
    zbar = np.nanmean(np.where(np.isfinite(zs), zs, np.nan), 1)
    Q = np.nansum((zs - zbar[:, None]) ** 2, 1)
    out["layers_assessable"] = nl
    out["Q"] = np.where(nl >= 2, Q, np.nan)
    out["I2"] = np.where((nl >= 2) & (Q > 0), np.clip((Q - (nl - 1)) / np.where(Q > 0, Q, 1), 0, 1), np.nan)
    # pathway coherence from reaction scores
    ridx = {r: i for i, r in enumerate(cx.rxns)}
    coh, cohp, nr = [], [], []
    for pw in names:
        rs = [ridx[r] for r in sets[pw]["rxn"] if r in ridx]
        nr.append(len(rs))
        if len(rs) >= P["min_set"]:
            v = np.nanmean(C[rs], 0)
            coh.append(v[0])
            cohp.append(normal_p_from_null(np.array([v[0]]), v[None, 1:])[1][0])
        else:
            coh.append(np.nan)
            cohp.append(np.nan)
    out["n_reactions_observed"] = nr
    out["coherence"] = coh
    out["p_coherence"] = cohp
    out["q_coherence"] = bh(np.array(cohp, float))
    a = P["alpha"]
    cls = []
    for _, r in out.iterrows():
        sig = {k: (r.get(f"q_{lname[k]}", 1) < a) and (r[f"n_{lname[k]}"] >= P["min_set"]) for k in lay}
        signs = {k: np.sign(r[f"z_{lname[k]}"]) for k in lay if sig[k]}
        assess = [k for k in lay if r[f"n_{lname[k]}"] >= P["min_set"]]
        if not assess:
            cls.append("Not assessable")
        elif len(signs) >= 2 and len(set(signs.values())) > 1:
            cls.append("Shared-discordant")
        elif len(signs) >= 2 or (len(signs) == 1 and r["q_combined"] < a and len(assess) >= 2
                                 and all(np.sign(r[f"z_{lname[k]}"]) == list(signs.values())[0] for k in assess)
                                 and (np.isnan(r["I2"]) or r["I2"] < 0.5)):
            cls.append("Shared-concordant")
        elif len(signs) == 1 and len(assess) >= 2:
            k = list(signs)[0]
            cls.append(f"{ {'rna': 'mRNA', 'prot': 'Protein', 'met': 'Metabolite'}[k]}-specific")
        elif len(signs) == 1:
            cls.append("Significant (single layer assessable)")
        elif r["q_combined"] < a:
            cls.append("Shared (weak per layer)")
        else:
            cls.append("Not significant")
    out["class"] = cls
    return out.sort_values("p_combined").reset_index(drop=True)


def _reporter_metabolites(cx: CascadeIndex, C: np.ndarray, mname: dict, P: dict) -> pd.DataFrame:
    """Reporter-metabolite style score (Patil & Nielsen 2005) on cascade coherence: metabolites around
    which the surrounding reactions are jointly coherent."""
    mu = np.nanmean(C[:, 1:], 1, keepdims=True)
    sd = np.nanstd(C[:, 1:], 1, keepdims=True)
    Zc = np.nan_to_num((C - mu) / np.where(sd > 0, sd, np.nan))
    A = ((cx.Sm > 0).astype(float) + (cx.Pm > 0).astype(float))
    A.data = np.ones_like(A.data)
    k = np.asarray(A.sum(0)).ravel()
    Sc = np.asarray(A.T @ Zc) / np.sqrt(np.maximum(k, 1))[:, None]
    zr, p = normal_p_from_null(Sc[:, 0], Sc[:, 1:])
    df = pd.DataFrame({"met_base": cx.met_bases, "metabolite": [mname.get(m, m) for m in cx.met_bases],
                       "n_reactions": k.astype(int), "reporter_score": Sc[:, 0], "reporter_z": zr, "p": p})
    df = df[df["n_reactions"] >= 2].copy()
    df["q"] = bh(df["p"].values)
    return df.sort_values("p").reset_index(drop=True)


def _regulators(kb, feats: dict, rx: pd.DataFrame, P: dict) -> pd.DataFrame | None:
    """TF activity (univariate linear model on mRNA z, as decoupler ULM) for regulons in the KB or
    uploaded by the user; linked to enzymes of significant reactions."""
    reg = kb.extra.get("tf_regulons")
    if reg is None or "rna" not in feats:
        return None
    r = feats["rna"].sort_values("mean_log2", ascending=False).drop_duplicates("symbol").set_index("symbol")["z"]
    r = r[r.index != ""]
    sig_enz = set()
    for e in rx.loc[rx["significant"], "enzymes"]:
        sig_enz |= {x.strip() for x in str(e).split(",") if x.strip()}
    rows = []
    zv = r.values
    gi = {g: i for i, g in enumerate(r.index)}
    for tf, grp in reg.groupby("tf"):
        idx = [gi[t] for t in grp["target"] if t in gi]
        if len(idx) < 5:
            continue
        w = np.zeros(len(zv))
        w[idx] = grp.set_index("target").loc[[r.index[i] for i in idx], "sign"].values
        X = np.column_stack([np.ones(len(zv)), w])
        beta, res, *_ = np.linalg.lstsq(X, zv, rcond=None)
        dof = len(zv) - 2
        s2 = ((zv - X @ beta) ** 2).sum() / dof
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        t = beta[1] / se
        tg = set(grp["target"])
        rows.append({"TF": tf, "activity_t": t, "p": 2 * sst.t.sf(abs(t), dof), "n_targets": len(idx),
                     "TF_mRNA_z": r.get(tf, np.nan), "enzyme_targets_in_significant_reactions":
                     ", ".join(sorted(tg & sig_enz)), "n_metabolic_targets": len(tg & sig_enz)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["q"] = bh(df["p"].values)
    return df.sort_values("p").reset_index(drop=True)
