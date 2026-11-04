"""
network.py - Engine C: integrated network, propagation, choke points and target prioritisation.

* Integrated network: genes (enzymes) - reactions - metabolites from the knowledge base, restricted to
  observable reactions; optional STRING protein-protein edges.
* Propagation: random walk with restart (RWR) seeded by |z| of every measured node, vectorised over a
  degree-binned seed-shuffling null (network-propagation review: Cowen et al. 2017).
* Choke points: reactions that are the only producer or the only consumer of a non-currency metabolite
  in the knowledge-base network (Yeh et al. 2004).
* Targets: robust rank aggregation (Kolde et al. 2012) of coherence, cross-layer consistency, choke-point
  status, propagation and optional druggability. Targets are hypotheses with an evidence trail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .stats import bh


def choke_points(kb) -> set:
    rx = kb.reactions
    ok = set(rx.loc[~(rx["is_transport"] | rx["is_exchange"] | rx["is_pool"]), "rxn"])
    rev = set(rx.loc[rx["reversible"], "rxn"])
    cur = set(kb.metabolites.loc[kb.metabolites["currency"], "met_base"])
    st = kb.stoich[kb.stoich["rxn"].isin(ok) & ~kb.stoich["met_base"].isin(cur)]
    prod, cons = {}, {}
    for r, m, c in zip(st["rxn"], st["met_base"], st["coef"]):
        if c > 0 or r in rev:
            prod.setdefault(m, set()).add(r)
        if c < 0 or r in rev:
            cons.setdefault(m, set()).add(r)
    out = set()
    for d in (prod, cons):
        for m, rs in d.items():
            if len(rs) == 1:
                out |= rs
    return out


def build_graph(res, kb, use_ppi: bool = True):
    """Nodes: 'g:SYMBOL', 'r:RXN', 'm:MET'. Returns (nodes DataFrame, edge list)."""
    cx = res.cascade
    mname = dict(zip(kb.metabolites.met_base, kb.metabolites.name))
    rx = res.reactions.set_index("reaction")
    edges = []
    for r in cx.rxns:
        for g in cx.rx_genes.get(r, []):
            edges.append((f"g:{g}", f"r:{r}", "catalyses"))
        for m in cx.sub_ids.get(r, []):
            edges.append((f"m:{m}", f"r:{r}", "substrate"))
        for m in cx.pro_ids.get(r, []):
            edges.append((f"r:{r}", f"m:{m}", "product"))
    if use_ppi and "ppi_string" in kb.extra:
        genes = {g for gs in cx.rx_genes.values() for g in gs}
        ppi = kb.extra["ppi_string"]
        ppi = ppi[ppi["a"].isin(genes) & ppi["b"].isin(genes)]
        edges += [(f"g:{a}", f"g:{b}", "PPI") for a, b in zip(ppi["a"], ppi["b"]) if a < b]
    nodes = sorted({e[0] for e in edges} | {e[1] for e in edges})
    ge = res.gene_evidence.set_index("gene")
    zm = res.features["met"].sort_values("mean_log2", ascending=False).drop_duplicates("met_base") \
        .set_index("met_base")["z"] if "met" in res.features else pd.Series(dtype=float)
    rows = []
    for n in nodes:
        t, k = n.split(":", 1)
        if t == "g":
            z = ge["E_obs"].get(k, np.nan)
            rows.append({"node": n, "type": "enzyme", "label": k, "z": z,
                         "source": ge["evidence_source"].get(k, "")})
        elif t == "m":
            rows.append({"node": n, "type": "metabolite", "label": mname.get(k, k), "z": zm.get(k, np.nan),
                         "source": "metabolite"})
        else:
            rows.append({"node": n, "type": "reaction", "label": rx["name"].get(k, k),
                         "z": rx["C_std"].get(k, np.nan), "source": "reaction"})
    return pd.DataFrame(rows), edges


def propagate(nodes: pd.DataFrame, edges: list, restart: float = 0.3, B: int = 200, seed: int = 1):
    idx = {n: i for i, n in enumerate(nodes["node"])}
    n = len(idx)
    a = np.array([idx[e[0]] for e in edges] + [idx[e[1]] for e in edges])
    b = np.array([idx[e[1]] for e in edges] + [idx[e[0]] for e in edges])
    A = sparse.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    deg = np.asarray(A.sum(0)).ravel()
    W = A @ sparse.diags(1 / np.maximum(deg, 1))          # column-stochastic
    seedv = np.nan_to_num(np.abs(nodes["z"].values))
    seedv[nodes["type"].values == "reaction"] = 0.0         # seeds = measured molecules only
    rng = np.random.default_rng(seed)
    measured = np.where(seedv > 0)[0]
    qs = np.unique(np.quantile(deg[measured], [0, .2, .4, .6, .8, 1])) if len(measured) else np.array([0])
    bins = np.digitize(deg[measured], qs[1:-1])
    S = np.zeros((n, B + 1))
    S[:, 0] = seedv
    for j in range(1, B + 1):
        v = np.zeros(n)
        for lv in np.unique(bins):
            ids = measured[bins == lv]
            v[ids] = seedv[rng.permutation(ids)]
        S[:, j] = v
    S = S / np.maximum(S.sum(0, keepdims=True), 1e-12)
    Pm = S.copy()
    for _ in range(100):
        new = (1 - restart) * (W @ Pm) + restart * S
        if np.abs(new - Pm).max() < 1e-10:
            Pm = new
            break
        Pm = new
    mu, sd = Pm[:, 1:].mean(1), Pm[:, 1:].std(1)
    z = (Pm[:, 0] - mu) / np.where(sd > 0, sd, np.nan)
    out = nodes.copy()
    out["degree"] = deg
    out["rwr_score"] = Pm[:, 0]
    out["propagation_z"] = z
    out["p_propagation"] = stats.norm.sf(z)
    out["q_propagation"] = bh(out["p_propagation"].values)
    return out


def rra(rank_matrix: np.ndarray) -> np.ndarray:
    """Robust rank aggregation rho score (Kolde 2012). rank_matrix: items x lists of normalised ranks
    in (0, 1]; NaN = absent from a list. Returns a p-value-like score (Bonferroni over lists)."""
    out = np.full(rank_matrix.shape[0], np.nan)
    for i, r in enumerate(rank_matrix):
        r = np.sort(r[np.isfinite(r)])
        k = len(r)
        if k == 0:
            continue
        p = stats.beta.cdf(r, np.arange(1, k + 1), k - np.arange(1, k + 1) + 1)
        out[i] = min(1.0, p.min() * k)
    return out


def rank_targets(res, kb, prop: pd.DataFrame, druggable: set | None = None, chokes: set | None = None) -> pd.DataFrame:
    cx = res.cascade
    rx = res.reactions.set_index("reaction")
    chokes = chokes if chokes is not None else choke_points(kb)
    genes = sorted({g for gs in cx.rx_genes.values() for g in gs})
    g2r = {}
    for r, gs in cx.rx_genes.items():
        for g in gs:
            g2r.setdefault(g, []).append(r)
    conc = res.concordance.set_index("gene") if res.concordance is not None else pd.DataFrame()
    pz = prop.set_index("node")["propagation_z"]
    ge = res.gene_evidence.set_index("gene")
    rows = []
    for g in genes:
        rs = g2r.get(g, [])
        sub = rx.loc[rs]
        pcomb = np.maximum(sub["p_label"].values, sub["p_topology"].values)
        best = sub.index[np.nanargmin(pcomb)] if np.isfinite(pcomb).any() else rs[0]
        coh = -np.log10(np.nanmin(pcomb)) if np.isfinite(pcomb).any() else 0.0
        if g in conc.index:
            zr, zp = conc.loc[g, "z_mRNA"], conc.loc[g, "z_protein"]
            cons = min(abs(zr), abs(zp)) if np.sign(zr) == np.sign(zp) else 0.0
        else:
            cons = 0.5 * abs(ge["E_obs"].get(g, 0.0))
        choke = any(r in chokes for r in rs)
        rows.append({"gene": g, "evidence_source": ge["evidence_source"].get(g, ""),
                     "enzyme_z": ge["E_obs"].get(g, np.nan), "n_reactions": len(rs),
                     "best_reaction": rx.loc[best, "name"], "best_reaction_pattern": rx.loc[best, "pattern"],
                     "coherence_score": coh, "cross_layer_consistency": cons, "choke_point": choke,
                     "propagation_z": pz.get(f"g:{g}", np.nan),
                     "druggable": (g in druggable) if druggable else np.nan,
                     "significant_reactions": int(rx.loc[rs, "significant"].sum())})
    df = pd.DataFrame(rows)
    n = len(df)
    ranks = []
    for col in ("coherence_score", "cross_layer_consistency", "propagation_z"):
        ranks.append(df[col].rank(ascending=False, method="average").values / n)
    ranks.append(np.where(df["choke_point"], 0.5 * df["choke_point"].sum() / n, np.nan))
    if druggable:
        ranks.append(np.where(df["druggable"] == True, 0.5 * (df["druggable"] == True).sum() / n, np.nan))  # noqa: E712
    df["rra_score"] = rra(np.column_stack(ranks))
    df["q_rra"] = bh(df["rra_score"].values)
    trail = []
    for _, r in df.iterrows():
        t = []
        if r["significant_reactions"]:
            t.append(f"{r['significant_reactions']} significant reaction(s)")
        if r["cross_layer_consistency"] > 1.5:
            t.append("mRNA and protein agree")
        if r["choke_point"]:
            t.append("choke point")
        if np.isfinite(r["propagation_z"]) and r["propagation_z"] > 2:
            t.append("network hub of the signal")
        if r["druggable"] is True:
            t.append("known ligand/drug")
        trail.append("; ".join(t))
    df["evidence"] = trail
    return df.sort_values("rra_score").reset_index(drop=True)


def subnetwork(res, prop: pd.DataFrame, edges: list, top: int = 30):
    rx = res.reactions
    keep = rx[rx["significant"] | rx["metabolite_driven"]]
    if len(keep) < 5:
        keep = rx.nsmallest(top, "p_label")
    keep = keep.head(top)
    rset = {f"r:{r}" for r in keep["reaction"]}
    E = [e for e in edges if e[0] in rset or e[1] in rset]
    nodes = {e[0] for e in E} | {e[1] for e in E}
    N = prop[prop["node"].isin(nodes)].copy()
    return N, E


def to_graphml(nodes: pd.DataFrame, edges: list) -> bytes:
    import io

    import networkx as nx
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(r["node"], label=str(r["label"]), type=r["type"],
                   z=float(r["z"]) if np.isfinite(r["z"]) else 0.0)
    for a, b, t in edges:
        G.add_edge(a, b, relation=t)
    buf = io.BytesIO()
    nx.write_graphml(G, buf)
    return buf.getvalue()
