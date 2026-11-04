"""
demo.py - synthetic paired tri-omics dataset with planted, known biology (ground truth for validation).

24 samples (12 Control, 12 Tumor), two processing batches. Planted effects (log2 units):
  1. Glycolysis up: enzymes up at mRNA and protein; glucose down, pyruvate and lactate up
     -> expected "Throughput-consistent (up)" reactions, Glycolysis shared-concordant.
  2. ASS1 loss (urea cycle): ASS1 down at mRNA and protein; citrulline up, argininosuccinate down
     -> expected "Candidate bottleneck" at L-citrulline:L-aspartate ligase.
  3. Fatty acid oxidation: proteins up without mRNA change -> "Protein-only (post-transcriptional)".
  4. Glutaminase step: GLS unchanged, glutamine up, glutamate down -> "Non-abundance regulation".
  5. Cholesterol biosynthesis: mRNA up, protein unchanged -> "mRNA-only (buffered)".
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

PLANT = {
    "glycolysis_genes": ["HK1", "HK2", "GPI", "PFKP", "PFKL", "ALDOA", "TPI1", "GAPDH", "PGK1", "PGAM1",
                         "ENO1", "PKM", "LDHA"],
    "glycolysis_mets": {"glucose": -0.7, "pyruvate": 0.8, "L-lactate": 1.1, "PEP": 0.6,
                        "fructose-1,6-bisphosphate": 0.6, "glucose-6-phosphate": 0.5},
    "bottleneck_gene": "ASS1",
    "bottleneck_mets": {"citrulline": 1.1, "argininosuccinate": -0.9},
    "fao_genes": ["CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA", "HADHB", "ECHS1", "HADH", "ACAA2", "ACADS"],
    "nonabundance_gene": "GLS",
    "nonabundance_mets": {"glutamine": 0.9, "glutamate": -0.9},
    "buffered_genes": ["HMGCR", "HMGCS1", "MVK", "PMVK", "MVD", "FDPS", "FDFT1", "SQLE", "LSS", "CYP51A1",
                       "DHCR7", "IDI1"],
}


def make_demo(kb, out_dir: str, seed: int = 7, n_per_group: int = 12, effects: bool = True) -> dict:
    """effects=False writes a global-null dataset (no planted biology) for calibration tests."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    n = 2 * n_per_group
    samples = [f"S{i + 1:02d}" for i in range(n)]
    group = np.array(["Control"] * n_per_group + ["Tumor"] * n_per_group)
    batch = np.array(["A", "B"] * (n // 2))
    tumor = (group == "Tumor").astype(float) * (1.0 if effects else 0.0)
    meta = pd.DataFrame({"Sample": samples, "Group": group, "Batch": batch,
                         "Age": rng.integers(45, 80, n), "Sex": rng.choice(["F", "M"], n)})

    genes = kb.genes[kb.genes["symbol"].ne("")].drop_duplicates("symbol").reset_index(drop=True)
    G = len(genes)
    sym = genes["symbol"].values
    latent = rng.normal(size=(3, n))
    load = rng.normal(scale=0.25, size=(G, 3))
    mu = rng.normal(8.0, 1.6, G)
    rna = mu[:, None] + load @ latent + rng.normal(scale=0.35, size=(G, n))
    eff_r = np.zeros(G)
    eff_p = np.zeros(G)
    gi = {s: i for i, s in enumerate(sym)}
    for s in PLANT["glycolysis_genes"]:
        if s in gi:
            eff_r[gi[s]] += 0.9
            eff_p[gi[s]] += 0.9
    if PLANT["bottleneck_gene"] in gi:
        eff_r[gi[PLANT["bottleneck_gene"]]] -= 1.3
        eff_p[gi[PLANT["bottleneck_gene"]]] -= 1.3
    for s in PLANT["fao_genes"]:
        if s in gi:
            eff_p[gi[s]] += 0.9
    for s in PLANT["buffered_genes"]:
        if s in gi:
            eff_r[gi[s]] += 0.9
            eff_p[gi[s]] -= 0.0
    rna_eff = rna + eff_r[:, None] * tumor[None, :]
    # protein: coupled to the mRNA deviation, with its own noise and planted effects
    mu_p = rng.normal(20.0, 2.0, G)
    prot = (mu_p[:, None] + 0.55 * (rna - rna.mean(1, keepdims=True)) + rng.normal(scale=0.3, size=(G, n))
            + eff_p[:, None] * tumor[None, :])
    # buffered genes: protein does not follow their mRNA effect (only baseline coupling)
    # proteome coverage: ~65% of metabolic genes + every planted gene
    planted = set(PLANT["glycolysis_genes"]) | set(PLANT["fao_genes"]) | set(PLANT["buffered_genes"]) | \
        {PLANT["bottleneck_gene"], PLANT["nonabundance_gene"]}
    keep_p = rng.random(G) < 0.65
    keep_p |= np.isin(sym, list(planted))

    # metabolites: non-currency, with reactions, prefer those with HMDB IDs
    m = kb.metabolites
    cur = set(m.loc[m["currency"], "met_base"])
    rx_ok = kb.reactions.loc[~(kb.reactions["is_transport"] | kb.reactions["is_exchange"]), "rxn"]
    deg = kb.stoich[kb.stoich["rxn"].isin(rx_ok) & ~kb.stoich["met_base"].isin(cur)].groupby("met_base").size()
    cand = m[m["met_base"].isin(deg.index) & ~m["currency"]].copy()
    cand["deg"] = cand["met_base"].map(deg)
    cand = cand[(cand["hmdb"] != "") | (cand["kegg"] != "")]
    named = set(PLANT["glycolysis_mets"]) | set(PLANT["bottleneck_mets"]) | set(PLANT["nonabundance_mets"])
    chosen = pd.concat([cand[cand["name"].isin(named)], cand[~cand["name"].isin(named)]
                        .sort_values("deg", ascending=False).head(320)]).drop_duplicates("met_base")
    M = len(chosen)
    mu_m = rng.normal(16.0, 2.0, M)
    met = mu_m[:, None] + (rng.normal(scale=0.2, size=(M, 3)) @ latent) + rng.normal(scale=0.35, size=(M, n))
    eff_m = np.zeros(M)
    nm = chosen["name"].values
    for d in (PLANT["glycolysis_mets"], PLANT["bottleneck_mets"], PLANT["nonabundance_mets"]):
        for k, v in d.items():
            eff_m[nm == k] += v
    met += eff_m[:, None] * tumor[None, :]
    # batch effect in the metabolomics layer (location shift per feature)
    met += rng.normal(0.35, 0.15, M)[:, None] * (batch == "B")[None, :]

    def to_raw(X, miss=0.03):
        R = 2 ** X
        low = X < np.quantile(X, 0.08)
        R[low & (rng.random(X.shape) < 0.3)] = np.nan
        R[rng.random(X.shape) < miss * 0.2] = np.nan
        return R

    t_rna = pd.DataFrame(to_raw(rna_eff), columns=samples)
    t_rna.insert(0, "Gene", sym)
    pr = genes[keep_p]
    t_prot = pd.DataFrame(to_raw(prot[keep_p]), columns=samples)
    t_prot.insert(0, "Gene", pr["symbol"].values)
    t_prot.insert(0, "Protein", [u.split(";")[0] if u else s for u, s in zip(pr["uniprot"], pr["symbol"])])
    # metabolites: identify ~70% by HMDB, the rest by name only (tests name matching)
    by_id = rng.random(M) < 0.7
    t_met = pd.DataFrame(to_raw(met), columns=samples)
    t_met.insert(0, "HMDB", np.where(by_id, chosen["hmdb"].values, ""))
    t_met.insert(0, "Metabolite", nm)
    t_rna.to_csv(os.path.join(out_dir, "demo_transcriptomics.csv"), index=False)
    t_prot.to_csv(os.path.join(out_dir, "demo_proteomics.csv"), index=False)
    t_met.to_csv(os.path.join(out_dir, "demo_metabolomics.csv"), index=False)
    meta.to_csv(os.path.join(out_dir, "demo_metadata.csv"), index=False)
    truth = {k: (v if not isinstance(v, dict) else v) for k, v in PLANT.items()}
    truth["expected"] = {
        "throughput_up_pathway": "Glycolysis / Gluconeogenesis",
        "bottleneck_reaction_name": "L-Citrulline:L-aspartate ligase (AMP-forming)",
        "nonabundance_reaction_name": "L-glutamine amidohydrolase",
    }
    json.dump(truth, open(os.path.join(out_dir, "demo_ground_truth.json"), "w"), indent=2)
    return {"rna": t_rna.shape, "prot": t_prot.shape, "met": t_met.shape}


if __name__ == "__main__":
    from .kb import load_kb
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(make_demo(load_kb(), os.path.join(here, "demo_data")))
