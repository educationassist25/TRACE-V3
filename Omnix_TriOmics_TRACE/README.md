# Omnix™ TRACE v1.5.3 — Tri-Omics Integration Engine (launch version)

**TRACE = Tri-omics Reaction-Anchored Cascade Evidence.** This is a standalone Streamlit app. It integrates transcriptomics, proteomics and metabolomics/lipidomics along the directional chain **gene → protein → enzymatic reaction → metabolite**.

The reaction, not the molecule, is the unit of integration. For every reaction TRACE asks whether the change in its enzyme and the changes in its substrates and products point the same way. Each answer is calibrated by permutation and corrected for multiple testing (FDR).

> Results describe evidence *consistent with* a mechanism. They are not causal proof.

---

## 1. Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
# tests: pip install -r requirements-dev.txt && python -m pytest -q tests
```

1. Pick **Demo: paired tri-omics** in the sidebar.
2. Press **Run TRACE**. The demo takes about 25 s.

The demo contains planted, known biology:

- glycolysis is up;
- ASS1 is lost, creating a bottleneck;
- fatty-acid oxidation is up at the protein level only;
- the glutaminase step is regulated without an enzyme change;
- cholesterol-synthesis mRNA is up but its protein is buffered.

TRACE recovers all five. The tests check this.

Run the tests:

```bash
python -m pytest -q tests
```

All 30 tests pass (29 validation tests plus an app smoke test). They take under a minute.

### What changed in v1.5.3 (pinned packages)

- **Pinned versions:** `requirements.txt` pins the exact package versions verified together (Streamlit 1.66.1, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, plotly 7.1.0, pyarrow 24.0.0, mofapy2 0.7.5 and the rest). Every deploy installs the same tested set, and Streamlit Cloud no longer has to replace pyarrow on each deploy.
- **Test-only packages:** pytest and pyyaml moved to `requirements-dev.txt`, so cloud installs are smaller.
- **Verified:** the pinned set loads and runs on Python 3.12 and 3.14 (Streamlit Cloud's default), at the same speed.

### What changed in v1.5.2 (speed)

- **Results are kept on the server:** finished analyses are stored for every visitor, keyed by a hash of the input data and every setting, up to eight analyses. Re-running the demo, reloading the page, or a second visitor with the same data and settings gets results in about 1–2 s instead of recomputing (about 14 s on two CPU cores, longer on Streamlit Cloud's shared CPUs).
- **The demo is computed in the background:** as soon as the page opens on a demo dataset, a background job (one per server) computes the demo with the current settings. Pressing Run after that finishes shows results at once. Pressing Run earlier waits for that same job instead of starting a second one.
- **Figures and files are not redrawn on every click:** the cascade and targets figures render once per session, and demo files are read once.

### What changed in v1.5.1

- **Fonts on blocked networks:** the Google web-font import has its own style block, so TRACE's styles apply immediately with system fonts when the font server is blocked or slow.
- **No telemetry:** Streamlit usage-statistics collection is off (`browser.gatherUsageStats = false`), so there are no calls home from restricted networks.
- **Verified:** tested from a clean install of `requirements.txt` on Python 3.10–3.13 with Streamlit 1.40, 1.45, 1.50, 1.65 and 1.66. Every section and page opened in a browser with no errors, both from the app folder and launched from another folder.

### What changed in v1.5.0 (top area: Option 1, loading fix)

- **Loads on older Streamlit:**
  - The app runs on any Streamlit ≥ 1.40, tested on 1.45, 1.50 and 1.66.
  - Newer-only options, such as expander `key` and button `width`, are dropped by `trace_core/compat.py` instead of crashing the page.
  - The app adds its own folder to the import path, so it also starts from another directory or from an Omnix multipage portal.
- **Header:** the Omnix logo sits in its own tile with clear padding, and "Omnix™ TRACE" is shown at full size with the tagline and version. The run status and Human-GEM badges sit on the right.
- **Main navigation:** it moves below the header as four section cards (Results · Networks & targets · Advanced · Data & export). Each card shows its pages and highlights in teal when open, and the whole card is one click.
- **Page row:** the study chips (tier, contrast, FDR) and an Export button sit under the section cards. The current section's pages follow as pill tabs.
- **Half-screen windows:** the cards wrap two by two, and the header keeps the logo beside the name.

### What changed in v1.4.0 (layout: Option A)

- **Analysis setup panel:** the left panel is a four-step setup checklist (Data, Study design, Statistics, Engines) with a progress bar.
  - Each step collapses to a one-line summary, for example "Study design · Tumor vs Control".
  - Only one step is open at a time, and the open step is outlined in teal.
  - Rarely used options (topology permutations, seed, log transform, missing-value filter) are under "More options".
  - Every input stays loaded while its step is closed, so settings and uploaded files are kept.
- **Top bar:** one white bar holds the Omnix brand, the four sections (Results · Networks & targets · Advanced · Data & export), the run status and an Export button that opens Data & export › Export. A slim study line (tier, contrast, FDR) and the current section's tabs sit underneath.
- **Pathways tab:**
  - **Figures:** canonical pathways and pathway activity across layers sit side by side in two cards; they stack on narrow windows.
  - **Shared control:** one "Pathways shown" control applies to both.
  - **Tables:** one card below holds both tables, with a class filter.

### What changed in v1.3.0 (Omnix launch version)

- **Omnix branding:** Omnix teal (#00695C) replaces the indigo accents. The header shows the Omnix logo and "Omnix™ TRACE"; the run button and active tabs are teal. A footer carries the version, knowledge base and the interpretation caveat.
- **Light and dark modes:** both are defined in `.streamlit/config.toml`. The app follows the viewer's system setting, and anyone can switch under menu ⋮ → Settings → Theme. Every custom card, chart and static figure preview adapts to the active mode (`st.context.theme`). Downloads always keep the white publication style.
- **Navigation:** four sections (Results · Networks & targets · Advanced · Data & export), each with its own tabs, replace the single row of 12 tabs.
- **Streamlit toolbar:** set to viewer mode, which hides developer items such as Deploy.
- **Key findings:** shows the top 6, with a "Show all" switch. Each finding names the section and tab where its details are.
- **Static figures:** vector previews size from their viewBox, so no blank space appears above tall figures.

### What changed in v1.2.0 (dashboard redesign)

- **Dark dashboard theme** (`trace_core/ui.py`, `.streamlit/config.toml`).
  - **Header:** brand header with version badge, run-status pill (execution time) and knowledge-base badge.
  - **Study banner:** shows the tier, the contrast and that results are FDR-calibrated.
  - **Tabs:** each tab has an icon.
- **Sidebar:** three numbered sections. 1 · Input data layer (card-style choices), 2 · Experimental design, 3 · Advanced parameters (four icon panels). A gradient **Run TRACE analysis** button sits at the bottom.
- **Overview:**
  - **Metric cards:** six cards with coloured accents.
  - **Layer concordance chart:** per-layer pathway activity z for the top pathways.
  - **Reaction-classification donut.**
  - **Key findings:** a searchable, filterable list (Bottleneck, Throughput, Metabolite-driven, Pathway, Target), built from the actual results.
- **Figures on screen:** Plotly figures use a matching dark template. Static figures (cascade, targets, network) show dark previews. Downloads and the Export tab keep the white publication style. Light previews are a switch in *Figure output customisation*.
- **Narrow windows:** side-by-side cards wrap onto separate rows in half-screen windows.

### What changed in v1.1.3

- **Cascade:** a new alluvial diagram replaces the Plotly Sankey. Full reaction names sit in their own boxes, wrapped and never truncated. Enzyme and metabolite labels sit beside their bars, and ribbons are stacked at every node, so no text touches a ribbon or another label. Substrates are pale ribbons with italic names.
- **Targets:** a new prioritised-target figure shows the full best reaction (bold), the evidence and the reaction pattern for each gene in a separate text column. Each row is as tall as its text.
- **Pathway activity across layers:** the plot is centred, and the colour scale is shorter (about four rows) in both the app and the publication export.
- **Sharper figures:** the network exact preview, cascade and targets figures are vector (SVG) images, sharp at any zoom or screen density. The Plotly camera button now saves PNG at 6× by default (choose 4–10× in the sidebar).
- **Exact text metrics:** text in the new figures is measured with matplotlib's renderer rather than estimated. A test checks that no two text boxes overlap and that every full name is drawn.

### What changed in v1.1.2

- **Pathway activity across layers:** dots are circles on a 1:1 grid with a radius of at most 0.42 of a cell, so neighbouring dots always keep a clear gap, at any window size. Class is a coloured tile column with a legend; the colour bar sits on the same grid. The publication version caps dot diameters at 80 % of the row/column pitch and shrinks fonts when a small size is chosen.
- **One tab for IPA-style pathway maps and the network studio.** Scope controls sit on top; style settings live in a collapsible, tabbed panel.
- **Nothing overlaps in network figures:** edges are routed around every node they do not connect, labels never sit on other labels or nodes, and the canvas enlarges automatically until this holds. A label that crosses an edge keeps an opaque background. The caption reports the check. Radial layouts (chord, hive, arc) place labels outside the ring instead.
- **Half-screen windows:** all figures resize with the window (wrapped titles and labels, legends below plots, labels placed for a half-screen width), the exact network preview scales to fit, and the interactive network is shown at true size in a scrollable frame. Control grids and metric rows use two or three columns.

## 2. Input data

| File | Format |
| --- | --- |
| Transcriptomics | One row per gene. First column = gene symbol, Ensembl ID or Entrez ID. Remaining numeric columns = samples. |
| Proteomics | One row per protein. ID column = UniProt accession or gene symbol. An extra `Gene` column helps. |
| Metabolomics / Lipidomics | One row per metabolite. ID = name, or HMDB / KEGG / ChEBI. Optional `HMDB`, `KEGG` and `ChEBI` columns are used first. |
| Metadata | `Sample`, `Group`, optional `Batch` and other covariates (age, sex …). |
| Optional ID table | Feature → HMDB / KEGG / ChEBI, for metabolites that do not map by name. |

**Data you need:**

- **Layers:** metabolomics plus at least one of transcriptomics or proteomics.
- **Scale:** raw intensities are log2-transformed automatically. Already-logged data are detected.
- **Omnix exports:** the normalized matrices exported by the existing Omnix modules work as they are.

## 3. What the app computes

| Tab | Engine | Output |
| --- | --- | --- |
| Overview | — | Evidence tier, key findings, headline figures |
| Data & mapping | Harmonization | Sample overlap and tier, batch-group confounding (Cramér's V), ID-mapping coverage, unmapped features |
| Concordance | A | mRNA vs protein z-scores. Each gene is labelled coupled, opposing, protein-only (post-transcriptional) or mRNA-only (buffered). Uses a Deming slope and within-sample Spearman ρ. |
| Reactions | A | Pattern map (E vs M) and a reaction table with q-values (label, topology, balance) |
| Cascade | A | Sankey diagram of enzyme → reaction → metabolite for a pathway, the top reactions or one enzyme |
| Pathways | A | Per-layer pathway z, combined z, Cochran Q / I², and the class: shared-concordant, shared-discordant or omics-specific |
| Pathway maps | A | Interactive reaction–metabolite map of a pathway |
| Network & targets | C | Ranked targets (rank aggregation), explanatory subnetwork with GraphML export, reporter metabolites, TF regulators |
| Factors | B | MOFA+ variance explained, factor scores, covariate associations, pathway enrichment of the loadings |
| Signature | D | Nested-CV multiblock sPLS-DA with AUC (bootstrap CI), permutation p and a stability-selected panel; late fusion |
| Export | — | Excel (all tables), run manifest (JSON), methods text, figure ZIP (PNG/TIFF/PDF/SVG, 300–1200 DPI), full bundle |

Every interactive figure has a camera icon that downloads a high-resolution PNG or SVG.

### Figures, network studio and IPA-style displays (v1.1)

**Collision-free figures.** One placement engine (`trace_core/layout.py`) handles every figure. It checks labels against nodes, against edges and against each other before placing them.

- **Labels:** each label is placed only where it touches nothing. If it has to move away from its point, a thin leader line connects them.
- **Network spacing:** nodes are pushed apart until each one's own label fits underneath.
- **Canvas size:** the canvas grows automatically when a network is dense.
- **Tests:** the tests check that no nodes overlap in any layout and that labels essentially never collide.

**Downloads.** Every figure has a "Download publication-quality figure" panel.

- **Formats:** PNG, TIFF (LZW), JPEG, SVG (editable text), PDF (embedded fonts), EPS, or all formats at once as a ZIP.
- **Settings:** 300, 600 or 1200 DPI; width and height in inches; font size and family; transparent background.
- **Export tab:** renders the full figure set in any combination of formats.

**IPA-inspired canonical pathways.**

- **Bars:** right-tailed Fisher's exact test of significant molecules across all layers.
- **Bar colour:** a directional z-score. Orange means increased, blue decreased, grey no activity pattern (fewer than 4 directional molecules).
- **Line:** the ratio of significant molecules to all pathway molecules.

**IPA-style pathway maps** (tab *Pathway maps & network*, which opens on the most significant pathway).

- **Shapes:** one per molecule type (enzyme, kinase, phosphatase, peptidase, transporter, metabolite).
- **Measured molecules:** red = up, green = down.
- **Predicted molecules:** unmeasured metabolites get a MAP-like prediction, shown as an orange or blue outline.
- **Edge colours:** leads to increase, leads to decrease, inconsistent with the data, or not predicted.
- **Layout:** molecules sit in subcellular bands (Extracellular Space / Plasma Membrane / Cytoplasm / Nucleus, or 10 organelles), from Human-GEM localisation.

**Network studio** (same tab: choose *Draw* = coherent reactions, a neighbourhood, a molecule list or top targets). Fifteen one-click style presets set everything at once; each can then be fine-tuned:

- IPA classic, KEGG map, Cytoscape default, Journal minimal, Dark presentation
- Colour-blind safe, Greyscale print, Module map, Hub focus
- Chord / circos, Hive plot, Arc diagram, Pathway flow, Bipartite columns

| Setting | Options |
| --- | --- |
| Scope | coherent reactions, pathway, enzyme neighbourhood (1–2 hops), molecule list, top targets |
| Representation | molecule view or reaction view |
| Layouts (14) | subcellular, hierarchical (top-down / left-right), force-directed, Kamada-Kawai, radial (focus-centred), bipartite columns, circular, edge-bundled chord, hive plot, arc diagram, shell, grid, spectral |
| Colour by | z-score, molecule type, compartment, data layer, module (Louvain community), betweenness, target score |
| Shapes | IPA shapes, KEGG boxes (names inside), all circles |
| Node size | constant, \|z\|, degree, target score |
| Edges | IPA relationship colours, colour by relation or a single colour; curved, straight or orthogonal routing; width by enzyme evidence; print line styles |
| Other | module hulls, highlights, label modes, light or dark theme, legend, bands |

The preview is identical to the exported file. An interactive hover/zoom view is also available. Networks export as Cytoscape `.cyjs` (with positions and colours), GraphML, and node/edge CSV.

These displays follow IPA's visual conventions but are computed from TRACE statistics and Human-GEM. They do not use IPA's proprietary knowledge base or its activation z-score algorithm.

### Core scores (Engine A)

```text
E_r = enzyme evidence: protein z through the GPR rule
      AND (complex) = smallest |z|, and only if the subunit signs agree
      OR (isozymes) = abundance-weighted mean
      if no protein is measured: mRNA z x observed mRNA-protein coupling ("mRNA proxy")
S_r, P_r = mean z of measured non-currency substrates and products
M_r = (P_r - S_r) / sqrt(2)          metabolite balance
C_r = E_r x M_r                      coherence
```

**Patterns.** Each reaction gets one plain-language pattern:

- Throughput-consistent (up/down)
- Candidate bottleneck
- Constrained step
- Substrate-limited
- Product drain
- Compensated
- Non-abundance regulation
- Metabolite-only change
- Enzyme-only change

### Statistics

- **Per-layer tests:** moderated t-statistics (limma-style empirical-Bayes variance shrinkage), converted to signed z-scores.
- **Label null:** sample labels are permuted jointly across layers, so cross-omics dependence of matched samples is kept. Defaults: 1,000 permutations; up to 5,000 for the label null and 2,000 for the topology null.
- **Topology null:** measured values are shuffled among nodes of similar network degree.
- **Reaction p-values:** taken from the product-normal distribution. Its location, scale and correlation are estimated from the permutations. Small p-values are therefore resolvable, and the p-values agree with the empirical permutation p.
- **FDR:** Benjamini–Hochberg. A reaction is called **coherent** only when both the label q and the topology q are below 0.05. A reaction is **metabolite-driven** when its metabolite balance is significant but its enzyme is unchanged.
- **Signature:** all feature selection happens inside the cross-validation folds. The early-integration signature is disabled below 10 samples per class.

### Validation built into the package

| Check | Result |
| --- | --- |
| Planted bottleneck (ASS1) | Top reaction, label q < 0.001, typed *Candidate bottleneck* |
| Glycolysis | *Shared-concordant*; HK/GPI/GCK steps *Throughput-consistent (up)* |
| Cholesterol synthesis (mRNA up, protein buffered) | *mRNA-specific* |
| Fatty-acid oxidation (protein only) | *Protein-specific*; genes typed *Protein-only* |
| Glutaminase step (enzyme unchanged) | *Metabolite-driven* |
| Global-null data, 12 simulated datasets | Mean fraction of reactions with p < 0.05 = 0.041 (parametric) vs 0.041 (empirical). No FDR-significant reactions or pathways. |
| Signature on null data | AUC < 0.8 (no leakage) |

## 4. Knowledge base

`kb/` is a frozen, versioned snapshot. The app makes no live database calls. `kb/kb_manifest.json` records the sources and versions.

- **Core:** Human-GEM 2.1.1 (CC BY 4.0). It provides:
  - 12,748 reactions with gene-protein-reaction (GPR) rules and stoichiometry;
  - 4,102 metabolites with KEGG, HMDB, ChEBI, PubChem and LIPID MAPS IDs;
  - 2,870 genes with Ensembl, UniProt and Entrez IDs;
  - 145 subsystems, used as pathways.
- **Add-ons:** build these on a computer with internet access. The cloud build environment could not reach these sites.

```bash
python -m trace_core.kb_build --skip-core --kegg --collectri --string   # adds to ./kb
python -m trace_core.kb_build --download --gem-dir kb_src --out kb      # full rebuild from Human-GEM
```

The add-ons provide:

- `--kegg`: KEGG pathway gene and compound sets, which appear as extra pathways prefixed `KEGG:`.
- `--collectri`: TF regulons, which turn on the TF-regulator table.
- `--string`: protein–protein edges (STRING v12, score ≥ 700), which are added to the network.

**Licensing.** KEGG and HMDB content is used now, as you decided. A commercial licence from KEGG (Pathway Solutions) and permission from HMDB are needed before Tri-Omics is sold. Each source is a separate file in `kb/`, so either can be removed without code changes.

## 5. Project layout

```text
app.py                      Streamlit UI (12 tabs; pathway maps and network studio share one tab)
trace_core/
  data.py                   loading, log/missing handling, within-layer batch correction, ID mapping, design/tiers
  stats.py                  moderated t, BH, permutation labels, product-normal calibrated p-values
  cascade.py                reaction index (GPR, stoichiometry), E/S/P/M/C scoring, pattern typing
  engine.py                 Engine A run: concordance, cascade + nulls, pathways, reporter metabolites, TF regulators
  network.py                Engine C: graph, random walk with restart + null, choke points, rank aggregation, subnetwork
  factors.py                Engine B: MOFA+ (mofapy2) with stability, associations, loading enrichment
  signature.py              Engine D: nested-CV multiblock sPLS-DA, stability selection, permutation, late fusion
  figures.py                interactive Plotly figures
  pubfig.py                 publication renderer (all formats, 300-1200 DPI)
  layout.py                 layouts, overlap removal, collision-free labels
  netview.py                network studio: IPA-style maps, presets, renderers, Cytoscape export
  ipa.py                    IPA-style canonical pathway analysis
  report.py                 Excel, manifest, methods text, 600-DPI matplotlib figures, ZIP bundles
  kb.py / kb_build.py       knowledge-base loader and builder
  demo.py                   synthetic tri-omics data with ground truth
kb/                         knowledge-base snapshot (parquet + manifest)
demo_data/                  demo matrices, metadata, ground truth
tests/test_trace.py         29 validation tests
tests/test_app.py           app smoke test (all data modes)
docs/                       methods and Omnix integration notes
```

## 6. Adding TRACE to Omnix later

See `docs/OMNIX_INTEGRATION.md`. In short:

1. Copy `trace_core/`, `kb/` and `demo_data/` into `apps/triomics/`.
2. Wrap `app.py` as a portal page.
3. Add an **Export to Tri-Omics** button to the existing modules. This is the only change to them.

## 7. Known limits of v1

- **Two-group contrasts only.** For multi-group designs, run one contrast at a time.
- **Bulk data mixes compartments.** Compartments are collapsed to base metabolites.
- **Lipids map only partly.** They map at species or class level when the name or ID matches Human-GEM. Class-level lipid rules are planned for a later release.
- **No live database access.** Drug–target annotation comes from an uploaded gene list. A ChEMBL/DGIdb add-on is planned.
- **GNN embeddings (Strategy C, deep-learning part) are deferred to a later release.** Network propagation is implemented now.
