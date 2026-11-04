"""kb.py - loads the TRACE knowledge-base snapshot written by kb_build.py."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb")


@dataclass
class KB:
    genes: pd.DataFrame
    reactions: pd.DataFrame
    stoich: pd.DataFrame
    gpr: pd.DataFrame
    metabolites: pd.DataFrame
    pathway_reactions: pd.DataFrame
    manifest: dict
    extra: dict = field(default_factory=dict)   # optional: kegg sets, tf regulons, ppi

    @property
    def version(self) -> str:
        return self.manifest.get("kb_version", "unknown")


def load_kb(path: str = KB_DIR) -> KB:
    r = lambda n: pd.read_parquet(os.path.join(path, n))
    extra = {}
    for name in ("kegg_gene_sets", "kegg_compound_sets", "tf_regulons", "ppi_string"):
        p = os.path.join(path, f"{name}.parquet")
        if os.path.exists(p):
            extra[name] = pd.read_parquet(p)
    kb = KB(genes=r("genes.parquet"), reactions=r("reactions.parquet"), stoich=r("stoich.parquet"),
            gpr=r("gpr.parquet"), metabolites=r("metabolites.parquet"),
            pathway_reactions=r("pathway_reactions.parquet"),
            manifest=json.load(open(os.path.join(path, "kb_manifest.json"))), extra=extra)
    sym = dict(zip(kb.genes.ensg, kb.genes.symbol))
    kb.gpr["symbol"] = kb.gpr["ensg"].map(sym).fillna(kb.gpr["ensg"])
    return kb
