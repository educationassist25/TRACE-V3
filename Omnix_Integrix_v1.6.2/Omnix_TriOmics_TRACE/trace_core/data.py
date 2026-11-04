"""
data.py - loading, cleaning and harmonising the three omics layers.

A layer is held as a `Layer`: a features x samples matrix on log2 scale plus a feature annotation table.
Layers: "rna" (transcriptomics), "prot" (proteomics), "met" (metabolomics / lipidomics).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAYER_NAMES = {"rna": "Transcriptomics", "prot": "Proteomics", "met": "Metabolomics"}


@dataclass
class Layer:
    key: str
    X: pd.DataFrame                      # features x samples, log2 scale, no NaN after clean()
    feat: pd.DataFrame                   # index = feature id; columns: label, ids, mapping results
    log: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return LAYER_NAMES.get(self.key, self.key)


# ------------------------------------------------------------------------------------------ reading
def read_table(src) -> pd.DataFrame:
    """Read CSV/TSV/XLSX from a path or an uploaded file object."""
    name = getattr(src, "name", str(src)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(src)
    if name.endswith((".tsv", ".txt")):
        return pd.read_csv(src, sep="\t")
    return pd.read_csv(src)


def split_matrix(df: pd.DataFrame, id_col: str | None = None, sample_cols: list | None = None):
    """Split a wide table into (annotation, numeric matrix). Samples = numeric columns."""
    df = df.copy()
    if id_col is None:
        id_col = df.columns[0]
    if sample_cols is None:
        sample_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    ann_cols = [c for c in df.columns if c not in sample_cols]
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df[df[id_col].ne("") & df[id_col].ne("nan")]
    num = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    num.index = df[id_col].values
    ann = df[ann_cols].copy()
    ann.index = df[id_col].values
    # duplicated identifiers: average the measurements, keep first annotation
    if num.index.duplicated().any():
        num = num.groupby(level=0).mean()
        ann = ann[~ann.index.duplicated()].reindex(num.index)
    num.columns = [str(c).strip() for c in num.columns]
    return ann, num


def to_log2(X: pd.DataFrame, mode: str = "auto") -> tuple[pd.DataFrame, str]:
    """mode: auto | log2 | none. 'auto' log-transforms when the data look like raw intensities."""
    vals = X.values[np.isfinite(X.values)]
    if mode == "none" or (mode == "auto" and (vals.size == 0 or np.nanmax(vals) < 100)):
        return X.astype(float), "used as provided (already log scale)"
    Y = X.astype(float).where(X > 0)
    return np.log2(Y), "log2-transformed (non-positive values treated as missing)"


def clean(X: pd.DataFrame, groups: pd.Series, max_missing: float = 0.5) -> tuple[pd.DataFrame, list]:
    """Remove features missing in more than `max_missing` of samples in every group; impute the rest
    with the feature minimum minus 1 (half-minimum on log2 scale). Constant features are removed."""
    log = []
    g = groups.reindex(X.columns)
    keep = pd.Series(False, index=X.index)
    for lv in g.dropna().unique():
        cols = g.index[g == lv]
        keep |= X[cols].isna().mean(axis=1) <= max_missing
    n0 = len(X)
    X = X[keep]
    log.append(f"{n0 - len(X)} features removed (> {int(max_missing * 100)}% missing in every group)")
    nmiss = int(X.isna().sum().sum())
    if nmiss:
        mins = X.min(axis=1) - 1.0
        X = X.T.fillna(mins).T
        log.append(f"{nmiss} missing values imputed with feature minimum - 1 (log2 half-minimum)")
    sd = X.std(axis=1)
    X = X[sd > 1e-9]
    return X, log


def combat(X: pd.DataFrame, batch: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Location/scale batch correction (parametric ComBat without empirical-Bayes shrinkage),
    protecting the group effect. Applied within one layer only."""
    b = batch.reindex(X.columns).astype(str)
    if b.nunique() < 2:
        return X
    g = pd.get_dummies(groups.reindex(X.columns).astype(str), drop_first=True).astype(float)
    B = pd.get_dummies(b).astype(float)
    D = np.column_stack([np.ones(len(b)), g.values, B.values[:, 1:]])
    beta, *_ = np.linalg.lstsq(D, X.values.T, rcond=None)
    fit_group = (np.column_stack([np.ones(len(b)), g.values]) @ beta[: 1 + g.shape[1]]).T
    resid = X.values - (D @ beta).T
    batch_eff = X.values - fit_group - resid
    Y = X.values - batch_eff
    # scale: equalise residual variance per batch
    out = Y.copy()
    pooled = resid.std(axis=1, ddof=1, keepdims=True) + 1e-12
    for lv in B.columns:
        idx = np.where(b.values == lv)[0]
        if len(idx) > 2:
            sd = resid[:, idx].std(axis=1, ddof=1, keepdims=True) + 1e-12
            out[:, idx] = fit_group[:, idx] + resid[:, idx] / sd * pooled
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a.astype(str), b.astype(str))
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


# ------------------------------------------------------------------------------------------ design
@dataclass
class Design:
    meta: pd.DataFrame          # index = sample; columns Group (+ Batch)
    case: str
    ref: str
    tier: int
    tier_text: str
    overlap: dict


def build_design(meta: pd.DataFrame, layers: dict, case: str, ref: str, sample_col="Sample", group_col="Group",
                 batch_col: str | None = None) -> Design:
    m = meta.copy()
    m[sample_col] = m[sample_col].astype(str).str.strip()
    m = m.drop_duplicates(sample_col).set_index(sample_col)
    m = m.rename(columns={group_col: "Group"})
    if batch_col and batch_col in m.columns:
        m = m.rename(columns={batch_col: "Batch"})
    m = m[m["Group"].astype(str).isin([case, ref])]
    m["Group"] = m["Group"].astype(str)
    sets = {k: set(L.X.columns) & set(m.index) for k, L in layers.items()}
    allk = set.intersection(*sets.values()) if sets else set()
    smallest = min(len(s) for s in sets.values()) if sets else 0
    overlap = {"per_layer": {k: len(v) for k, v in sets.items()}, "all_layers": len(allk)}
    union = set().union(*sets.values()) if sets else set()
    if len(allk) >= 8 and len(allk) == len(union):
        tier, text = 1, "Tier 1 - paired: the same samples are measured in every layer"
    elif len(allk) >= 3:
        tier, text = 2, "Tier 2 - partially overlapping: some samples are shared across layers"
    else:
        tier, text = 3, "Tier 3 - unpaired: layers come from different samples; integration uses summary statistics"
    return Design(meta=m, case=case, ref=ref, tier=tier, tier_text=text, overlap=overlap)


# ------------------------------------------------------------------------------------------ ID mapping
def _norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\s_\-,'()\[\]]+", "", s)
    return s


def _strip_stereo(s: str) -> str:
    return re.sub(r"^(l|d|dl|alpha|beta|\(r\)|\(s\))[-\s]", "", str(s).lower().strip())


def norm_hmdb(x: str) -> str:
    m = re.match(r"^\s*HMDB0*(\d+)\s*$", str(x), re.I)
    return f"HMDB{int(m.group(1)):07d}" if m else ""


def norm_kegg(x: str) -> str:
    m = re.match(r"^\s*(?:cpd:)?(C\d{5})\s*$", str(x), re.I)
    return m.group(1).upper() if m else ""


def norm_chebi(x: str) -> str:
    m = re.match(r"^\s*(?:CHEBI:)?(\d+)\s*$", str(x), re.I)
    return f"CHEBI:{m.group(1)}" if m else ""


def map_metabolites(ann: pd.DataFrame, ids: pd.Index, kb) -> pd.DataFrame:
    """Map metabolites to Human-GEM base metabolites. Priority: HMDB > KEGG > ChEBI > exact name >
    name without stereo prefix. Returns a table with met_base and the rule that matched."""
    mets = kb.metabolites
    by_hmdb = {norm_hmdb(h): b for b, h in zip(mets.met_base, mets.hmdb) if norm_hmdb(h)}
    by_kegg = {k: b for b, k in zip(mets.met_base, mets.kegg) if k}
    by_chebi = {norm_chebi(c): b for b, c in zip(mets.met_base, mets.chebi) if norm_chebi(c)}
    by_name, by_name2 = {}, {}
    for b, n in zip(mets.met_base, mets.name):
        by_name.setdefault(_norm_name(n), b)
        by_name2.setdefault(_norm_name(_strip_stereo(n)), b)
    cols = {c.lower(): c for c in ann.columns}
    out = []
    for fid in ids:
        row = ann.loc[fid] if fid in ann.index else pd.Series(dtype=object)
        cand = []
        for key, fn, table, rule in (("hmdb", norm_hmdb, by_hmdb, "HMDB"), ("hmdb_id", norm_hmdb, by_hmdb, "HMDB"),
                                     ("kegg", norm_kegg, by_kegg, "KEGG"), ("kegg_id", norm_kegg, by_kegg, "KEGG"),
                                     ("chebi", norm_chebi, by_chebi, "ChEBI"), ("chebi_id", norm_chebi, by_chebi, "ChEBI")):
            if key in cols:
                v = fn(row.get(cols[key], ""))
                if v and v in table:
                    cand.append((table[v], rule))
        for v, fn, table, rule in ((fid, norm_hmdb, by_hmdb, "HMDB"), (fid, norm_kegg, by_kegg, "KEGG"),
                                   (fid, norm_chebi, by_chebi, "ChEBI")):
            v2 = fn(v)
            if v2 and v2 in table:
                cand.append((table[v2], rule))
        if not cand:
            n = _norm_name(fid)
            if n in by_name:
                cand.append((by_name[n], "name"))
            else:
                n2 = _norm_name(_strip_stereo(fid))
                if n2 in by_name2:
                    cand.append((by_name2[n2], "name (stereo-insensitive)"))
                elif n2 in by_name:
                    cand.append((by_name[n2], "name (stereo-insensitive)"))
        b, rule = cand[0] if cand else ("", "unmapped")
        out.append({"feature": fid, "met_base": b, "map_rule": rule})
    res = pd.DataFrame(out).set_index("feature")
    nm = dict(zip(mets.met_base, mets.name))
    res["kb_name"] = res["met_base"].map(nm).fillna("")
    cur = dict(zip(mets.met_base, mets.currency))
    res["currency"] = res["met_base"].map(cur).fillna(False).astype(bool)
    return res


def map_genes(ann: pd.DataFrame, ids: pd.Index, kb) -> pd.DataFrame:
    """Map genes / proteins to gene symbols and Ensembl IDs. Accepts symbols, Ensembl gene IDs,
    UniProt accessions, Entrez IDs or aliases (in the ID column or in Gene/Symbol/UniProt columns)."""
    g = kb.genes
    sym_set = {s.upper(): s for s in g.symbol if s}
    ens = dict(zip(g.ensg, g.symbol))
    uni, ent, ali = {}, {}, {}
    for s, u, e, a in zip(g.symbol, g.uniprot, g.entrez, g.aliases):
        for x in str(u).split(";"):
            if x:
                uni[x.strip().upper()] = s
        for x in str(e).split(";"):
            if x:
                ent[x.strip()] = s
        for x in str(a).split(";"):
            if x:
                ali.setdefault(x.strip().upper(), s)
    sym2ens = dict(zip(g.symbol, g.ensg))
    cols = {c.lower(): c for c in ann.columns}
    sym_cols = [cols[c] for c in ("gene", "gene_symbol", "symbol", "genes", "gene names", "gene_name") if c in cols]
    uni_cols = [cols[c] for c in ("uniprot", "accession", "protein", "protein_id", "uniprot_id") if c in cols]
    out = []
    for fid in ids:
        row = ann.loc[fid] if fid in ann.index else pd.Series(dtype=object)
        cands = [str(fid)] + [str(row.get(c, "")) for c in sym_cols + uni_cols]
        symbol, rule = "", "unmapped"
        for c in cands:
            for tok in re.split(r"[;|,\s]+", c):
                t = tok.strip()
                if not t or t.lower() == "nan":
                    continue
                tu = t.upper().split("-")[0] if re.match(r"^[OPQ][0-9]|^[A-NR-Z][0-9]", t.upper()) else t.upper()
                t_ens = t.split(".")[0]
                if t_ens in ens:
                    symbol, rule = ens[t_ens], "Ensembl"
                elif tu in uni:
                    symbol, rule = uni[tu], "UniProt"
                elif t.upper() in sym_set:
                    symbol, rule = sym_set[t.upper()], "symbol"
                elif t in ent:
                    symbol, rule = ent[t], "Entrez"
                elif t.upper() in ali:
                    symbol, rule = ali[t.upper()], "alias"
                if symbol:
                    break
            if symbol:
                break
        if not symbol:  # non-metabolic gene: keep the symbol-like identifier for gene-protein concordance
            first = next((x for x in cands[1:] if x and x.lower() != "nan"), str(fid)) if sym_cols else str(fid)
            symbol = re.split(r"[;|,\s]+", first.strip())[0].upper() if first else ""
            rule = "not in metabolic KB"
        out.append({"feature": fid, "symbol": symbol, "ensg": sym2ens.get(symbol, ""), "map_rule": rule})
    return pd.DataFrame(out).set_index("feature")


def make_layer(key: str, df: pd.DataFrame, meta: pd.DataFrame, kb, id_col=None, transform="auto",
               max_missing=0.5, batch_correct=False, sample_col="Sample", group_col="Group",
               batch_col=None) -> Layer:
    ann, X = split_matrix(df, id_col)
    m = meta.copy()
    m[sample_col] = m[sample_col].astype(str).str.strip()
    groups = m.set_index(sample_col)[group_col].astype(str)
    X = X[[c for c in X.columns if c in groups.index]]
    X, how = to_log2(X, transform)
    log = [f"{X.shape[0]} features x {X.shape[1]} samples read; {how}"]
    X, l2 = clean(X, groups, max_missing)
    log += l2
    if batch_correct and batch_col and batch_col in m.columns:
        b = m.set_index(sample_col)[batch_col]
        X = combat(X, b, groups)
        log.append(f"batch corrected within layer (location/scale, group protected) using '{batch_col}'")
    feat = map_metabolites(ann, X.index, kb) if key == "met" else map_genes(ann, X.index, kb)
    if key == "met":
        feat["label"] = feat.index
    else:
        feat["label"] = feat["symbol"].where(feat["symbol"].ne(""), feat.index.to_series())
    feat["mean_log2"] = X.mean(axis=1)
    n_map = int((feat["map_rule"] != "unmapped").sum() if key == "met" else (feat["ensg"] != "").sum())
    log.append(f"{n_map} of {len(feat)} features mapped to the metabolic knowledge base")
    return Layer(key=key, X=X, feat=feat, log=log)


def load_demo(folder: str):
    f = lambda n: pd.read_csv(os.path.join(folder, n))
    return {"rna": f("demo_transcriptomics.csv"), "prot": f("demo_proteomics.csv"),
            "met": f("demo_metabolomics.csv")}, f("demo_metadata.csv")
