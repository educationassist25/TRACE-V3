"""
kb_build.py - builds the versioned Integrix knowledge base (KB) snapshot.

Core source: Human-GEM (SysBioChalmers, CC BY 4.0). It provides:
  * gene-protein-reaction (GPR) rules, reaction stoichiometry, reversibility, subsystems (pathways)
  * metabolite cross-references: KEGG compound, HMDB, ChEBI, PubChem, LIPID MAPS
  * reaction cross-references: KEGG reaction, Reactome, Rhea, EC numbers
  * gene cross-references: Ensembl, symbol, UniProt, Entrez

Optional add-ons (run on a machine with internet access, see `python -m trace_core.kb_build --help`):
  * --kegg      KEGG pathway membership for human genes and compounds (KEGG REST; academic use free,
                commercial use needs a KEGG licence)
  * --collectri CollecTRI TF -> target regulons (OmniPath) for regulator inference
  * --string    STRING v12 human protein-protein interactions (score >= 700)

Usage:
    python -m trace_core.kb_build --download --gem-dir kb_src --out kb # Human-GEM core
    python -m trace_core.kb_build --gem-dir kb_src --out kb --kegg --collectri --string
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import re
import urllib.request

import pandas as pd

# Small inorganic/cofactor "currency" metabolites, excluded from the metabolite balance of a reaction.
CURRENCY_NAMES = {
    "H2O", "H+", "ATP", "ADP", "AMP", "NAD+", "NADH", "NADP+", "NADPH", "CO2", "Pi", "PPi", "O2",
    "CoA", "FAD", "FADH2", "NH3", "GTP", "GDP", "GMP", "UTP", "UDP", "UMP", "CTP", "CDP", "CMP",
    "H2O2", "Na+", "K+", "Cl-", "HCO3-", "sulfate", "PAPS", "PAP", "ITP", "IDP", "dATP", "dADP",
    "Fe2+", "Fe3+", "Ca2+", "Mg2+", "Zn2+", "Cu2+", "Mn2+", "iodide", "superoxide", "ferricytochrome C",
    "ferrocytochrome C", "ubiquinone", "ubiquinol", "thioredoxin", "thioredoxin disulfide",
    "acyl carrier protein", "electron transfer flavoprotein", "electron transfer flavoprotein reduced",
    "hydrogen sulfide", "SAM", "SAH",
}


# ----------------------------------------------------------------------------------------------- GPR
def parse_gpr(rule: str, max_complexes: int = 64) -> list[list[str]]:
    """Parse a gene_reaction_rule into disjunctive normal form: a list of complexes (AND sets),
    any one of which (OR) can catalyse the reaction."""
    rule = (rule or "").strip()
    if not rule:
        return []
    toks = re.findall(r"\(|\)|\band\b|\bor\b|[^\s()]+", rule)
    pos = 0

    def expr():
        nonlocal pos
        res = term()
        while pos < len(toks) and toks[pos] == "or":
            pos += 1
            res = res + term()
        return res

    def term():
        nonlocal pos
        res = factor()
        while pos < len(toks) and toks[pos] == "and":
            pos += 1
            rhs = factor()
            res = [a + b for a in res for b in rhs][:max_complexes]
        return res

    def factor():
        nonlocal pos
        tok = toks[pos]
        pos += 1
        if tok == "(":
            res = expr()
            pos += 1  # ')'
            return res
        return [[tok]]

    out = expr()
    uniq, seen = [], set()
    for c in out:
        key = tuple(sorted(set(c)))
        if key not in seen:
            seen.add(key)
            uniq.append(list(key))
    return uniq[:max_complexes]


# ----------------------------------------------------------------------------------------------- core
def _omap(x):
    return dict(x) if isinstance(x, list) else (x or {})


def build_core(gem_dir: str, out: str) -> dict:
    import yaml

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(gem_dir, "Human-GEM.yml")) as fh:
        doc = yaml.load(fh, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    sec = {k: v for k, v in doc}
    meta = _omap(sec.get("metaData"))

    # metabolites (compartmentalised) -> base metabolite
    mets = []
    for m in sec["metabolites"]:
        m = _omap(m)
        nm = m.get("name", "")
        nm = {False: "NO", True: "YES"}.get(nm, nm) if isinstance(nm, bool) else str(nm)  # YAML 1.1 booleans
        mets.append({"met": m["id"], "name": nm, "compartment": m.get("compartment", ""),
                     "formula": str(m.get("formula", "") or "")})
    mets = pd.DataFrame(mets)
    mets["met_base"] = mets["met"].str[:-1]
    xref = pd.read_csv(os.path.join(gem_dir, "metabolites.tsv"), sep="\t", dtype=str).fillna("")
    xref = xref.rename(columns={"mets": "met", "metKEGGID": "kegg", "metHMDBID": "hmdb", "metChEBIID": "chebi",
                                "metPubChemID": "pubchem", "metLipidMapsID": "lipidmaps"})
    mets = mets.merge(xref[["met", "kegg", "hmdb", "chebi", "pubchem", "lipidmaps"]], on="met", how="left").fillna("")
    base = (mets.sort_values("met").groupby("met_base")
            .agg(name=("name", "first"), formula=("formula", "first"), kegg=("kegg", "first"),
                 hmdb=("hmdb", "first"), chebi=("chebi", "first"), pubchem=("pubchem", "first"),
                 lipidmaps=("lipidmaps", "first"), compartments=("compartment", lambda s: ",".join(sorted(set(s)))))
            .reset_index())
    base["currency"] = base["name"].isin(CURRENCY_NAMES)

    # reactions
    rx, st, gp = [], [], []
    for r in sec["reactions"]:
        r = _omap(r)
        ann = _omap(r.get("annotation"))
        sub = r.get("subsystem") or []
        sub = sub if isinstance(sub, list) else [sub]
        rule = r.get("gene_reaction_rule", "") or ""
        rx.append({"rxn": r["id"], "name": r.get("name", ""), "lb": float(r.get("lower_bound", 0)),
                   "ub": float(r.get("upper_bound", 1000)), "gpr": rule, "subsystem": "; ".join(sub),
                   "ec": ";".join(ann.get("ec-code", []) or [])})
        for met, coef in (_omap(r.get("metabolites"))).items():
            st.append({"rxn": r["id"], "met": met, "coef": float(coef)})
        for ci, cpx in enumerate(parse_gpr(rule)):
            for g in cpx:
                gp.append({"rxn": r["id"], "complex": ci, "ensg": g})
    rx = pd.DataFrame(rx)
    rxref = pd.read_csv(os.path.join(gem_dir, "reactions.tsv"), sep="\t", dtype=str).fillna("")
    rxref = rxref.rename(columns={"rxns": "rxn", "rxnKEGGID": "kegg_rxn", "rxnREACTOMEID": "reactome_rxn",
                                  "rxnRheaID": "rhea"})
    rx = rx.merge(rxref[["rxn", "kegg_rxn", "reactome_rxn", "rhea"]], on="rxn", how="left").fillna("")
    st = pd.DataFrame(st)
    st["met_base"] = st["met"].str[:-1]
    gp = pd.DataFrame(gp)

    # reaction type: transport / exchange / pool reactions are excluded from cascade scoring
    side = st.assign(side=st["coef"].apply(lambda c: "S" if c < 0 else "P"))
    sb = side.groupby(["rxn", "side"])["met_base"].apply(lambda s: frozenset(s)).unstack()
    rx = rx.set_index("rxn")
    rx["is_transport"] = [bool(sb.loc[r, "S"] == sb.loc[r, "P"]) if r in sb.index and isinstance(sb.loc[r, "S"], frozenset)
                          and isinstance(sb.loc[r, "P"], frozenset) else False for r in rx.index]
    nmet = st.groupby("rxn")["met"].size()
    rx["n_met"] = nmet.reindex(rx.index).fillna(0).astype(int)
    rx["is_exchange"] = rx["n_met"] <= 1
    rx["is_pool"] = st.groupby("rxn")["coef"].apply(lambda c: (c.abs() > 100).any()).reindex(rx.index).fillna(False)
    rx["reversible"] = rx["lb"] < 0
    rx = rx.reset_index()

    # collapse to base metabolites (bulk measurements are not compartment resolved)
    stb = (st.groupby(["rxn", "met_base"])["coef"].sum().reset_index())
    stb = stb[stb["coef"] != 0]

    genes = pd.read_csv(os.path.join(gem_dir, "genes.tsv"), sep="\t", dtype=str).fillna("")
    genes = genes.rename(columns={"genes": "ensg", "geneSymbols": "symbol", "geneUniProtID": "uniprot",
                                  "geneEntrezID": "entrez", "geneNames": "gene_name", "geneAliases": "aliases"})
    genes = genes.rename(columns={"compartments": "compartments"})
    genes = genes[["ensg", "symbol", "uniprot", "entrez", "gene_name", "aliases", "compartments"]]
    # molecule type for IPA-style shapes, from the EC numbers / reaction types a gene catalyses
    g_ec, g_tr, g_all = {}, {}, {}
    rxi = rx.set_index("rxn")
    for r, g in zip(gp["rxn"], gp["ensg"]):
        ec = rxi.at[r, "ec"]
        g_ec.setdefault(g, set()).update([e for e in str(ec).split(";") if e])
        g_all[g] = g_all.get(g, 0) + 1
        g_tr[g] = g_tr.get(g, 0) + int(bool(rxi.at[r, "is_transport"]))

    def mtype(g):
        ecs = g_ec.get(g, set())
        if any(e.startswith("2.7.") for e in ecs) and not any(e.startswith("2.7.7.") for e in ecs):
            return "kinase"
        if any(e.startswith("3.1.3.") for e in ecs):
            return "phosphatase"
        if any(e.startswith("3.4.") for e in ecs):
            return "peptidase"
        if g_all.get(g) and g_tr.get(g, 0) == g_all[g]:
            return "transporter"
        return "enzyme"
    genes["mol_type"] = genes["ensg"].map(mtype)

    # pathways = Human-GEM subsystems (gene sets and metabolite sets derived from member reactions)
    pw_rows = []
    for _, r in rx[~rx["is_exchange"]].iterrows():
        for s in [x.strip() for x in r["subsystem"].split(";") if x.strip()]:
            pw_rows.append({"pathway": s, "rxn": r["rxn"], "source": "Human-GEM"})
    pw = pd.DataFrame(pw_rows)

    # primary compartment of each base metabolite and reaction (metabolic, non-transport reactions)
    met_rx = st[st["rxn"].isin(rx.loc[~(rx["is_transport"] | rx["is_exchange"] | rx["is_pool"]), "rxn"])].copy()
    met_rx["comp"] = met_rx["met"].str[-1]
    pc = met_rx.groupby(["met_base", "comp"]).size().reset_index(name="n").sort_values("n", ascending=False)
    base["primary_compartment"] = base["met_base"].map(pc.drop_duplicates("met_base").set_index("met_base")["comp"]).fillna("c")
    rc = met_rx.groupby(["rxn", "comp"]).size().reset_index(name="n").sort_values("n", ascending=False)
    rx["compartment"] = rx["rxn"].map(rc.drop_duplicates("rxn").set_index("rxn")["comp"]).fillna("c")
    base.to_parquet(os.path.join(out, "metabolites.parquet"), index=False)
    rx.to_parquet(os.path.join(out, "reactions.parquet"), index=False)
    stb.to_parquet(os.path.join(out, "stoich.parquet"), index=False)
    gp.to_parquet(os.path.join(out, "gpr.parquet"), index=False)
    genes.to_parquet(os.path.join(out, "genes.parquet"), index=False)
    pw.to_parquet(os.path.join(out, "pathway_reactions.parquet"), index=False)

    sha = hashlib.sha256(open(os.path.join(gem_dir, "Human-GEM.yml"), "rb").read()).hexdigest()[:16]
    manifest = {
        "kb_version": f"trace-kb-{_dt.date.today().isoformat()}",
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "sources": {"Human-GEM": {"version": str(meta.get("version", "")), "sha256_16": sha,
                                  "licence": "CC BY 4.0", "url": "https://github.com/SysBioChalmers/Human-GEM"}},
        "counts": {"reactions": int(len(rx)), "metabolites": int(len(base)), "genes": int(len(genes)),
                   "gpr_links": int(len(gp)), "pathways": int(pw["pathway"].nunique())},
    }
    json.dump(manifest, open(os.path.join(out, "kb_manifest.json"), "w"), indent=2)
    return manifest


# ----------------------------------------------------------------------------------------------- add-ons
def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def add_kegg(out: str) -> None:
    """KEGG pathway membership (hsa) for genes (Entrez) and compounds. KEGG licence terms apply."""
    names = pd.read_csv(io.StringIO(_get("https://rest.kegg.jp/list/pathway/hsa")), sep="\t", header=None,
                        names=["pid", "name"])
    names["name"] = names["name"].str.replace(r" - Homo sapiens \(human\)$", "", regex=True)
    gl = pd.read_csv(io.StringIO(_get("https://rest.kegg.jp/link/pathway/hsa")), sep="\t", header=None,
                     names=["gene", "pid"])
    gl["entrez"] = gl["gene"].str.replace("hsa:", "", regex=False)
    gl["pid"] = gl["pid"].str.replace("path:", "", regex=False)
    cl = pd.read_csv(io.StringIO(_get("https://rest.kegg.jp/link/pathway/compound")), sep="\t", header=None,
                     names=["cpd", "pid"])
    cl["kegg"] = cl["cpd"].str.replace("cpd:", "", regex=False)
    cl["pid"] = cl["pid"].str.replace("path:map", "hsa", regex=False)
    nm = dict(zip(names["pid"].str.replace("path:", "", regex=False), names["name"]))
    g = gl.assign(pathway=gl["pid"].map(nm)).dropna(subset=["pathway"])[["pathway", "entrez"]]
    c = cl.assign(pathway=cl["pid"].map(nm)).dropna(subset=["pathway"])[["pathway", "kegg"]]
    g.to_parquet(os.path.join(out, "kegg_gene_sets.parquet"), index=False)
    c.to_parquet(os.path.join(out, "kegg_compound_sets.parquet"), index=False)
    _note(out, "KEGG", {"licence": "KEGG terms (academic free; commercial licence required)",
                        "url": "https://www.kegg.jp/kegg/legal.html", "pathways": int(g["pathway"].nunique())})


def add_collectri(out: str) -> None:
    txt = _get("https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&fields=is_stimulation,is_inhibition")
    df = pd.read_csv(io.StringIO(txt), sep="\t")
    df = df.rename(columns={"source_genesymbol": "tf", "target_genesymbol": "target"})
    df["sign"] = (df["is_stimulation"].astype(int) - df["is_inhibition"].astype(int)).replace(0, 1)
    df[["tf", "target", "sign"]].drop_duplicates().to_parquet(os.path.join(out, "tf_regulons.parquet"), index=False)
    _note(out, "CollecTRI", {"licence": "see OmniPath resource licences", "url": "https://github.com/saezlab/CollecTRI"})


def add_string(out: str, min_score: int = 700) -> None:
    import gzip
    url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    with urllib.request.urlopen(url, timeout=600) as r:
        df = pd.read_csv(io.BytesIO(gzip.decompress(r.read())), sep=" ")
    df = df[df["combined_score"] >= min_score]
    info_url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
    with urllib.request.urlopen(info_url, timeout=600) as r:
        info = pd.read_csv(io.BytesIO(gzip.decompress(r.read())), sep="\t")
    m = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    df = pd.DataFrame({"a": df["protein1"].map(m), "b": df["protein2"].map(m), "score": df["combined_score"]}).dropna()
    df.to_parquet(os.path.join(out, "ppi_string.parquet"), index=False)
    _note(out, "STRING", {"licence": "CC BY 4.0", "url": "https://string-db.org", "min_score": min_score})


def _note(out: str, name: str, info: dict) -> None:
    p = os.path.join(out, "kb_manifest.json")
    man = json.load(open(p))
    man["sources"][name] = {**info, "added": _dt.datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(p, "w"), indent=2)


def download_gem(dest: str, ref: str = "main") -> None:
    """Fetch the Human-GEM model files from GitHub (SysBioChalmers/Human-GEM)."""
    os.makedirs(dest, exist_ok=True)
    base = f"https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/{ref}/model/"
    for f in ("Human-GEM.yml", "reactions.tsv", "metabolites.tsv", "genes.tsv"):
        urllib.request.urlretrieve(base + f, os.path.join(dest, f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gem-dir", default="kb_src")
    ap.add_argument("--out", default="kb")
    ap.add_argument("--skip-core", action="store_true")
    ap.add_argument("--download", action="store_true", help="download Human-GEM into --gem-dir first")
    ap.add_argument("--kegg", action="store_true")
    ap.add_argument("--collectri", action="store_true")
    ap.add_argument("--string", action="store_true")
    a = ap.parse_args()
    if a.download:
        download_gem(a.gem_dir)
    if not a.skip_core:
        print(json.dumps(build_core(a.gem_dir, a.out)["counts"]))
    for flag, fn in (("kegg", add_kegg), ("collectri", add_collectri), ("string", add_string)):
        if getattr(a, flag):
            try:
                fn(a.out)
                print(f"added {flag}")
            except Exception as e:  # network or format problems should not destroy the core KB
                print(f"could not add {flag}: {e}")


if __name__ == "__main__":
    main()
