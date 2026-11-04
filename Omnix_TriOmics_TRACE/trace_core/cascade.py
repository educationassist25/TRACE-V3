"""
cascade.py - Engine A: the reaction-anchored directional cascade.

gene -> protein (translation concordance) -> reaction (GPR: AND = complex, OR = isozymes)
-> metabolites (substrate consumption / product formation).

For every observable reaction r:
    E_r  enzyme evidence   = GPR-combined protein z (mRNA z x coupling when the protein is not measured)
    S_r  substrate shift   = mean z of measured, non-currency substrates
    P_r  product shift     = mean z of measured, non-currency products
    M_r  metabolite balance = (P_r - S_r) / sqrt(2)   (one-sided reactions use P_r or -S_r)
    C_r  coherence          = E_r * M_r   (> 0: enzyme change and metabolite balance point the same way)

All statements are associations ("consistent with"), never causal claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

UP, DOWN, FLAT, NA = "up", "down", "flat", "n/a"


class CascadeIndex:
    """Pre-computed structure linking measured genes and metabolites to observable reactions."""

    def __init__(self, kb, gene_symbols: list, met_bases: list, prot_symbols: set, gene_weight: dict | None = None):
        rx = kb.reactions.set_index("rxn")
        ok = ~(rx["is_transport"] | rx["is_exchange"] | rx["is_pool"])
        cur = set(kb.metabolites.loc[kb.metabolites["currency"], "met_base"])
        gidx = {g: i for i, g in enumerate(gene_symbols)}
        midx = {m: i for i, m in enumerate(met_bases)}
        gpr = kb.gpr[kb.gpr["rxn"].isin(rx.index[ok]) & kb.gpr["symbol"].isin(gidx)]
        st = kb.stoich[kb.stoich["rxn"].isin(rx.index[ok]) & kb.stoich["met_base"].isin(midx)
                       & ~kb.stoich["met_base"].isin(cur)]
        rx_g = set(gpr["rxn"])
        rx_m = set(st["rxn"])
        self.rxns = sorted(rx_g & rx_m)
        ridx = {r: i for i, r in enumerate(self.rxns)}
        # complexes (measured subunits only)
        self.complexes, cpx_rx, cpx_w = [], [], []
        gw = gene_weight or {}
        for (r, c), grp in gpr[gpr["rxn"].isin(ridx)].groupby(["rxn", "complex"]):
            members = sorted({gidx[s] for s in grp["symbol"]})
            self.complexes.append(np.array(members))
            cpx_rx.append(ridx[r])
            w = [gw.get(gene_symbols[i]) for i in members]
            cpx_w.append(min(w) if all(v is not None for v in w) else np.nan)
        cpx_rx = np.array(cpx_rx)
        cpx_w = np.array(cpx_w, float)
        # isozyme weights: abundance-weighted within a reaction when all complexes have protein intensity
        W = np.zeros(len(self.complexes))
        for r in range(len(self.rxns)):
            sel = np.where(cpx_rx == r)[0]
            w = cpx_w[sel]
            W[sel] = (w / w.sum()) if np.all(np.isfinite(w)) and w.sum() > 0 else 1.0 / len(sel)
        self.Wc = sparse.csr_matrix((W, (cpx_rx, np.arange(len(self.complexes)))),
                                    shape=(len(self.rxns), len(self.complexes)))
        # metabolite incidence
        st = st[st["rxn"].isin(ridx)]
        sub = st[st["coef"] < 0]
        pro = st[st["coef"] > 0]

        def inc(df):
            rows = df["rxn"].map(ridx).values
            cols = df["met_base"].map(midx).values
            M = sparse.csr_matrix((np.ones(len(df)), (rows, cols)), shape=(len(self.rxns), len(met_bases)))
            n = np.asarray(M.sum(1)).ravel()
            Dinv = sparse.diags(np.where(n > 0, 1 / np.maximum(n, 1), 0))
            return Dinv @ M, n > 0

        self.Sm, self.has_S = inc(sub)
        self.Pm, self.has_P = inc(pro)
        self.sub_ids = sub.groupby("rxn")["met_base"].apply(list).to_dict()
        self.pro_ids = pro.groupby("rxn")["met_base"].apply(list).to_dict()
        self.gene_symbols = gene_symbols
        self.met_bases = met_bases
        self.cpx_rx = cpx_rx
        self.rx_genes = gpr[gpr["rxn"].isin(ridx)].groupby("rxn")["symbol"].apply(lambda s: sorted(set(s))).to_dict()
        self.prot_measured = {r: any(g in prot_symbols for g in gs) for r, gs in self.rx_genes.items()}
        info = rx.loc[self.rxns]
        self.info = pd.DataFrame({"reaction": self.rxns, "name": info["name"].values,
                                  "subsystem": info["subsystem"].values, "ec": info["ec"].values,
                                  "kegg_rxn": info["kegg_rxn"].values, "reversible": info["reversible"].values})
        # degrees for the topology null
        self.gene_degree = np.bincount(np.concatenate(self.complexes) if self.complexes else np.array([], int),
                                       minlength=len(gene_symbols))
        self.met_degree = np.asarray(((self.Sm > 0).astype(int) + (self.Pm > 0).astype(int)).sum(0)).ravel()

    # ------------------------------------------------------------------ scoring (vectorised over columns)
    def enzyme(self, Eg: np.ndarray) -> np.ndarray:
        """Eg: genes x B gene-level enzyme evidence -> reactions x B."""
        Ec = np.zeros((len(self.complexes), Eg.shape[1]))
        for k, mem in enumerate(self.complexes):
            v = Eg[mem]
            if len(mem) == 1:
                Ec[k] = v[0]
                continue
            fin = np.isfinite(v)
            vz = np.where(fin, v, np.nan)
            s = np.sign(vz)
            allpos = np.nansum(s > 0, 0) == fin.sum(0)
            allneg = np.nansum(s < 0, 0) == fin.sum(0)
            m = np.nanmin(np.abs(vz), 0)
            Ec[k] = np.where(allpos, m, np.where(allneg, -m, 0.0))
        Ec = np.nan_to_num(Ec)
        return np.asarray(self.Wc @ Ec)

    def balance(self, Zm: np.ndarray):
        S = np.asarray(self.Sm @ Zm)
        P = np.asarray(self.Pm @ Zm)
        hs, hp = self.has_S[:, None], self.has_P[:, None]
        M = np.where(hs & hp, (P - S) / np.sqrt(2), np.where(hp, P, -S))
        S = np.where(hs, S, np.nan)
        P = np.where(hp, P, np.nan)
        return S, P, M

    def score(self, Eg: np.ndarray, Zm: np.ndarray):
        E = self.enzyme(Eg)
        S, P, M = self.balance(Zm)
        return E, S, P, M, E * M


def _cat(x: float, thr: float) -> str:
    if not np.isfinite(x):
        return NA
    return UP if x >= thr else DOWN if x <= -thr else FLAT


def pattern(e: float, s: float, p: float, thr: float) -> str:
    """Interpretive pattern for one reaction (association language only)."""
    E, S, P = _cat(e, thr), _cat(s, thr), _cat(p, thr)
    s_up, s_dn = S == UP, S == DOWN
    p_up, p_dn = P == UP, P == DOWN
    if E == UP and (p_up or s_dn) and not (p_dn or s_up):
        return "Throughput-consistent (up)"
    if E == DOWN and s_up and (p_dn or P in (FLAT, NA)):
        return "Candidate bottleneck"
    if E == DOWN and (p_dn or s_up) and not (p_up or s_dn):
        return "Throughput-consistent (down)"
    if E == UP and s_up and not p_up:
        return "Constrained step"
    if E == UP and p_dn and s_dn:
        return "Substrate-limited"
    if E == FLAT and s_up and p_dn:
        return "Non-abundance regulation (lower flux)"
    if E == FLAT and s_dn and p_up:
        return "Non-abundance regulation (higher flux)"
    if E == DOWN and p_up:
        return "Compensated"
    if E == UP and p_dn:
        return "Product drain"
    if E == FLAT and (S in (UP, DOWN) or P in (UP, DOWN)):
        return "Metabolite-only change"
    if E in (UP, DOWN) and S in (FLAT, NA) and P in (FLAT, NA):
        return "Enzyme-only change"
    return "No clear pattern"


PATTERN_HELP = {
    "Throughput-consistent (up)": "Enzyme higher, product higher and/or substrate lower: consistent with increased flux through this step.",
    "Throughput-consistent (down)": "Enzyme lower, product lower and/or substrate higher: consistent with reduced flux.",
    "Candidate bottleneck": "Enzyme lower while its substrate accumulates: a possible metabolic bottleneck.",
    "Constrained step": "Enzyme higher but substrate accumulates and product does not follow: possible inhibition, cofactor limitation or fast downstream use.",
    "Substrate-limited": "Enzyme higher but both substrate and product are lower: possibly limited by substrate supply.",
    "Non-abundance regulation (lower flux)": "Enzyme level unchanged, substrate up and product down: consistent with allosteric, PTM or cofactor control.",
    "Non-abundance regulation (higher flux)": "Enzyme level unchanged, substrate down and product up: consistent with activity regulation independent of abundance.",
    "Product drain": "Enzyme higher but product lower: consistent with rapid downstream consumption of the product, or inhibition of this step.",
    "Compensated": "Enzyme lower but product maintained or higher: alternative route or slower consumption.",
    "Metabolite-only change": "Metabolites change without a change in enzyme abundance.",
    "Enzyme-only change": "Enzyme changes without measurable metabolite change.",
    "No clear pattern": "No consistent pattern at the chosen threshold.",
}
