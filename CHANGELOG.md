# Changelog

Notable changes to LitKG-Integrate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **The BERT NER path extracted nothing on every document.** The pipeline was
  built from `dmis-lab/biobert-base-cased-v1.1`, a base language model with no
  token-classification head, so transformers initialized one at random
  ("Some weights ... newly initialized: ['classifier.bias', 'classifier.weight']")
  and the pipeline returned `LABEL_0`/`LABEL_1` spans at ~0.5 confidence.
  `_map_bert_label` passed those through unchanged, none is in `entity_types`,
  and every one was dropped -- a stage that loaded a model per run, appeared
  active, and contributed zero entities.

  The path now runs `alvaroalon2/biobert_genetic_ner` (BioBERT fine-tuned on
  JNLPBA/BC2GM, configurable as `phase1.literature.models.biomedical_ner`), and
  `BiomedicalNLP` refuses any checkpoint whose `id2label` is the untrained
  `LABEL_N` default instead of running it for nothing.

  On 150 abstracts sampled from `data/processed/literature_context`, it adds
  **247 entities over 182 distinct surface forms** that neither scispacy model
  nor the gene rules produce -- mixed-case symbols (`Gsα`, `SetD5`, `apoE4`,
  `cullin-2`, `p16`), mouse allele notation (`Fgfr2(+/S252W)`) and modified
  residues (`H3K27me3`) -- against 4906 entities from those paths combined.
  Spans are dropped when they cut a word in half (5.7% of raw spans, an
  artifact of subword aggregation: "isplatin", "arubicin"), when the surface is
  a known non-gene acronym (`CAR`, `MDS`), or when they overlap a span scispacy
  already claimed, so one mention cannot become two entities.

### Added

- **`RAGPipeline`: the wiring between Phase 1 output and the retrieval stack**
  (`make rag Q="..."`, `scripts/run_rag.py`, `litkg.langchain_integration`).
  The retrievers, chunk-to-graph index and agents were unit tested but nothing
  connected them to real data -- `make run-langchain` built its own hardcoded
  documents and a bare FAISS index, so the graph-aware path was never exercised
  outside tests.

  On the bundled corpus: 80 documents, 81 chunks, 78 of 81 (96%) linked to the
  graph, reaching 158 of 2971 nodes. Answers are grounded and cited -- asked why
  BRCA1 tumours are sensitive to olaparib, the system explains synthetic
  lethality and quotes TBCRC 048 and OlympiA figures from the passages it cites.

- **Hub-traversal cap on graph expansion.** `CIVIC:DISEASE:DOID:162` ("cancer")
  has degree 429 and is linked from 29 of 81 chunks; expanding through it
  reaches 207 nodes in one hop and 824 in two, so every oncology passage became
  a neighbour of every other. `ChunkGraphIndex.neighbors` now refuses to walk
  *through* nodes above `DEFAULT_MAX_TRAVERSAL_DEGREE` (50), while still
  allowing them to be reached and reported as evidence.

  Retrieval relevance remains **unmeasured**: there is no relevance-judged query
  set, and on the one question examined only about one graph-expanded passage in
  five is relevant even after the cap.

### Fixed

- The CLI rendered every source as "hop 0, pmid ?" because `answer()` flattens
  document metadata into each source dict rather than nesting it under
  "metadata". Pinned with a test on the payload shape.

### Added

- **Literature-context features, measured and rejected.** Entity names carry no
  biology, so `litkg.phase2.literature_context` characterises each entity by
  pre-cutoff PubMed sentences instead. On the full test set this looked like a
  large win -- AUC 0.684 [0.668, 0.698] against 0.562 for names, disjoint
  intervals.

  **The gain was a confound.** Coverage is partial and uncovered nodes fall back
  to their name, so "has literature context" becomes a feature -- and covered
  nodes have median degree 6 against 2 for uncovered ones. Restricting the
  comparison to pairs where both endpoints have context removes it, and context
  then scores **0.485 [0.464, 0.506]**: below chance. In the hybrid it also
  hurts, 0.737 +/- 0.024 against 0.747 +/- 0.009, with tripled variance.

  Mean-pooling sentences per entity describes what an entity is discussed
  alongside, not how it relates to a specific partner. Entity-level context is
  the wrong granularity; a pair-level co-mention feature is the plausible fix
  and is not implemented. The fetching machinery is kept -- it is sound, tested,
  date-guarded and cached, so the next attempt costs an experiment rather than
  the infrastructure.

### Fixed

- **A unit test overwrote the real literature-context cache.**
  `ContextConfig(cache_dir=None)` meant "the default real location" rather than
  "no cache", so a test that set fixture data and called `save()` destroyed a
  2911-entity cache. `use_cache` is now a separate flag, `save()` is a no-op
  when caching is off, and a test asserts a full suite run leaves the cache
  directory untouched.

  This is the same overloading fixed in `FeatureConfig` earlier the same day.
  Fixing one and not checking its sibling is how it happened twice.

- `LiteratureContextFetcher.gather` returned a renamed attribute, so a completed
  fetch saved all its work and then raised `AttributeError` on the way out.

### Documentation

- **README status table now distinguishes "run on real data" from "synthetic
  demo".** Several `make run-*` targets build `torch.randn` tensors and
  hardcoded documents; they exercise a code path without saying anything about
  whether it works on data. Verified by running every phase:

  - Phase 1, chunking, KG preprocessing, entity linking, link prediction: real.
  - `HybridGNNModel` (Phase 2, 1.8M params, cross-modal attention): synthetic
    only. It has never been trained on `phase2_graph_data.json`; the measured
    AUC 0.748 comes from the simpler model in `phase2/link_prediction.py`.
  - Phase 3 discovery: synthetic only, 6 hardcoded relationships.
  - RAG and agents: the library imports, constructs, links chunks to graph
    nodes and answers -- verified directly -- but no script or CLI command runs
    it, and `make run-langchain` uses hardcoded documents with raw FAISS.

- Stale figures corrected: 276 -> 452 tests, and node/edge/link counts updated
  to the CIVIC 01-Aug-2026 release.

- Known limits now record that BERT NER fails on abstracts over ~512 tokens
  (caught and logged, so the pipeline succeeds while that extractor
  contributes nothing), and that ranking metrics are dominated by a few dozen
  rows.

### Changed

- **CIVIC data updated from the 01-Feb-2024 release to 01-Aug-2026**, and the
  release is now configurable rather than hard-coded into three URLs. The
  default stays *pinned* to a dated release: a nightly build changes underneath
  you, so a regression could not be told apart from a data update. Override
  with `LITKG_CIVIC_RELEASE=nightly` or another release date. The active release
  is recorded in `data/external/civic/RELEASE`.

  Evidence grows 4254 -> 4878 rows, variants 1694 -> 1992, and distinct dated
  pairs 6643 -> 6981. Citations now run to 2025, which makes later cutoffs
  viable: the 2020 holdout has 513 test pairs against 116 before.

  Releases from 2024 onward ship a *features* file rather than a genes file --
  617 genes alongside 345 fusions, 8 factors and 3 regions. These were being
  typed as genes, which would have put fusions into the vocabulary that
  literature gene mentions resolve against; `feature_type` is now honoured.

  Results at the comparable 2016 cutoff are unchanged (hybrid 0.748 +/- 0.009
  against 0.750 on the old release), which is the reassuring outcome for a data
  refresh. At the 2020 cutoff the hybrid reaches 0.791 +/- 0.008 with Hits@100
  of 0.131 -- **higher because it is an easier problem**, training on 26% more
  pairs against a smaller test set, not because the method improved.

### Added

- Schema verification on download. An earlier version of this code read
  `drugs`, `variant_id` and `clinical_significance` from an evidence file that
  had none of them and produced 4125 dangling edges in silence. A missing
  required column is now an error at download time, not an empty string at
  processing time.

### Added

- **Node text features** (`litkg.phase2.node_features`). Every predictor until
  now used topology alone, which cannot reach the 14% of held-out pairs whose
  endpoints have no path between them. Node display names are embedded and fed
  to the GNN alongside its learned embedding, node type and log-degree. Names
  are static metadata, so this does not leak across the temporal split.

  Measured over **8 seeds** on the hybrid: AUC 0.734 +/- 0.006 [0.724, 0.744]
  without text, 0.754 +/- 0.005 [0.745, 0.762] with it -- disjoint ranges, about
  four standard deviations. AP 0.277 -> 0.299, Hits@100 0.090 -> 0.105. Four
  seeds were **not** enough to establish this: two 4-seed runs of the same
  configuration disagreed by more than the effect size.

  The encoder was chosen by measurement, not reputation. On name similarity
  alone: PubMedBERT 0.580 [0.564, 0.595], MiniLM 0.533 [0.516, 0.550], BioBERT
  0.514 [0.497, 0.530] against a 0.498 floor. PubMedBERT is the default;
  BioBERT is barely above chance despite also being biomedical.

- **`FeatureOnlyPredictor`**, a deliberate control. Some CIVIC therapies are
  named for their target ("BRAF Inhibitor" embeds at 0.65 against "BRAF"), so
  text features could in principle win by string matching. Text alone scores
  0.581 against the hybrid's 0.750, which bounds that effect: the gain comes
  from combining text with topology, not from substring overlap.

  It is also the only predictor that can score cold-start pairs at all -- L3
  scores 0 of 366 -- though at AUC 0.531 [0.501, 0.562] that is coverage
  without much signal. Entity *names* carry little biology; "Imatinib" says
  nothing about what it treats. Embedding entities by their literature contexts
  is the natural next step.

### Fixed

- `FeatureConfig(cache_dir=None)` meant both "use the default cache" and "no
  cache", so an encoder configured for no caching read vectors written by a
  different model and failed on a shape mismatch. Caching is now controlled by
  an explicit `use_cache` flag, and a width mismatch discards the stale cache
  with a warning rather than raising from inside numpy.

### Fixed

- **Ranking metrics were reported without uncertainty, and some differences
  claimed from them were noise.** Each positive is ranked against the whole
  negative pool (~1200 against ~12000), which makes MRR extremely top-heavy:
  the top 20 positives supply 78% of it, only ~26 of 1204 reach the top 10, and
  its bootstrap CI spans [0.0066, 0.0135] -- about as wide as the value. Every
  metric now carries a 95% bootstrap interval, `hits_at_100` is reported
  because Hits@10 has no resolution at this pool size, and
  `indistinguishable_fraction` records how many positives a predictor cannot
  separate from the bulk.

  Two previously stated results do not survive this and have been withdrawn
  from the docs: that recovering edge evidence "doubled MRR", and that the
  R-GCN ranked better than the untyped encoder. Both gaps sit inside the
  interval. The AUC and AP results do survive -- the hybrid's AUC interval is
  disjoint from every structural baseline's.

  The apparent AUC/MRR trade-off across configurations was also an artefact of
  comparing a stable statistic against a noisy one, not a property of the task.

### Added

- **Evidence-weighted edges.** Flattening the graph discarded 11 predicates,
  direction, confidence and 1731 negation flags, collapsing 13194 relations
  into 6645 pairs. `RelationRecord` and `EdgeEvidence` carry those through the
  temporal split, and `WeightedL3PathPredictor` weights each path hop by mean
  confidence x log support, penalised by the negated fraction. On its own that
  lifts average precision 0.204 -> 0.231 and MRR 0.0044 -> 0.0097. The AUC
  difference over plain L3 (0.698 vs 0.692) is inside the confidence interval
  and is not claimed.

  Weights are aggregated from **pre-cutoff evidence only**; weighting an edge
  with later evidence would leak the knowledge the holdout withholds.

- **Relation-aware encoder** (`--relational`). An R-GCN learning one transform
  per predicate, since the untyped graph treats SENSITIZES_TO and RESISTANT_TO
  -- opposite claims about the same pair -- as identical edges. Its metrics are
  currently indistinguishable from the untyped encoder's once confidence
  intervals are accounted for; it is kept because collapsing opposite
  predicates is wrong in principle, not because it measures better.

- **Per-entity-type-pair reporting.** Every evaluation now breaks results down
  by type pair, because the aggregate averages four problems whose AUC spans
  0.638 (disease-drug) to 0.802 (mutation-phenotype).

### Changed

- The hybrid now ensembles the GNN with *weighted* L3 and passes relation types
  through. AUC 0.743 [95% CI 0.729-0.755] against L3's 0.692 [0.677-0.707] --
  disjoint intervals -- with AP 0.262 and Hits@100 0.069, and 5/5 seeds over
  the bar. Seed variance halved, from +/-0.018 to +/-0.010.

### Added

- **Trained link predictors** (`litkg.phase2.link_prediction`,
  `scripts/train_link_prediction.py`). A 2-layer GraphSAGE encoder with an MLP
  edge decoder, and a hybrid that ensembles it with the L3 path baseline.

  The hybrid reaches **AUC 0.729 ± 0.018 against the 0.692 baseline, beating it
  in 5 of 5 seeds**, with average precision 0.244 vs 0.205 and MRR 0.0072 vs
  0.0050. The GNN alone does not clear the bar reliably: 0.670 ± 0.089, with
  one seed collapsing to 0.512. The two components correlate at only Spearman
  0.33, which is why ensembling them helps and why the ensemble is far more
  stable than the learned part alone.

  Four things were required to make the GNN train meaningfully: disjoint
  message-passing and supervision edges (otherwise it reads the adjacency it
  was handed), temporal rather than random validation (a random slice reported
  0.912 against a true 0.737), a BPR ranking loss rather than cross entropy,
  and type- and degree-matched training negatives matching the evaluation.

  Hits@10 remains 0.014 — this is a measurable research result, not a system
  that surfaces useful hypotheses.

### Fixed

- **The first hybrid scored AUC 0.000.** It rank-transformed within each call,
  and the harness scores positives and negatives separately, so with ten times
  as many negatives every negative outranked every positive. Percentiles are
  now taken against a reference distribution fixed at fit time, making scores
  independent of the batch they were computed in.

### Added

- **`L3PathPredictor` baseline**, and it changes the project's headline result.
  The CIVIC graph is strictly multipartite — 0 of 6769 edges join same-type
  nodes — and 100% of held-out pairs are cross-type, so shared-neighbour
  predictors are undefined on it rather than weak. Counting degree-normalised
  length-3 paths on the same split with the same negatives raises AUC from
  0.543 to 0.692 and average precision from 0.107 to 0.204, with no new data.
  The harness now reports `same_type_edge_ratio` and warns when a graph is
  multipartite, so this cannot be misread again.

### Fixed

- **`sample_negatives` crashed on positive endpoints absent from the training
  graph.** networkx returns a `DegreeView` rather than raising for a node it
  does not contain, so degree-matched sampling failed with a confusing
  `TypeError`. Callers that filter cold-start pairs never hit this; callers
  constructing a split directly did.

### Changed

- `docs/Evaluation.md` previously concluded the graph was "too sparse for
  topological link prediction". That was a measurement artefact of using
  length-2 predictors on a multipartite graph, and has been corrected along
  with the guidance that followed from it — a GNN is now worth trying, with
  0.692 as the number it must beat.

### Added

- **Evaluation harness for link prediction** (`litkg.evaluation`, `make
  evaluate`). Splits knowledge graph edges on the publication year of the
  supporting paper, scores structural baselines, and reports AUC, average
  precision, Hits@K and MRR. Documented in `docs/Evaluation.md`.

  The split guards three sources of leakage, each counted in the report: pairs
  re-asserted after the cutoff but first published before it (495 at cutoff
  2016) stay in training; undated gene-variant edges form a backbone that
  disqualifies duplicate test pairs; cold-start pairs with an endpoint missing
  from training are excluded rather than counted as failures.

  Negatives match the endpoint types of the positives they stand against, with
  `--degree-matched` additionally controlling for node popularity.

  **First measured result, and it is negative:** at cutoff 2016 with degree
  matching, Adamic-Adar reaches AUC 0.543 against a random floor of 0.498, and
  Hits@1 is 0.000 for every predictor. Preferential attachment scores 0.725
  without degree matching and 0.512 with it, showing the apparent signal was
  node popularity. The cause is structural: 84.6% of test pairs share no
  neighbour in the training graph, so shared-neighbour predictors are
  undefined for them rather than wrong. The harness reports this as
  `structural_coverage` and warns below 50%.

### Added

- **Clinical entities from CIVIC evidence.** The knowledge graph held only
  genes and variants, so the diseases and chemicals typed NER extracts had
  nothing to link against. CIVIC evidence now contributes 270 diseases (keyed
  by Disease Ontology id where present), 381 therapies and 59 phenotypes, with
  predicates derived from `evidence_type` × `significance` and drawn from the
  same vocabulary the literature extractor emits. Confidence comes from
  CIVIC's evidence level and curator rating rather than a flat 0.8, and the
  498 "Does Not Support" rows are flagged `negated` rather than asserted.
  Cross-modal links rose 92 → 167, now including 51 disease↔disease and 35
  chemical↔drug pairs.

### Fixed

- **4125 of 5825 KG edges dangled.** `_process_civic_evidence` emitted
  relations pointing at `CIVIC:DISEASE:` and `CIVIC:DRUG:` nodes it never
  created, and read three columns the evidence file does not have
  (`variant_id`, `drugs`, `clinical_significance`). Every evidence subject was
  the empty string and no therapy relation was ever built. Evidence subjects
  are now resolved through a molecular-profile index (92.7% direct, the rest
  compound profiles split into components; 1 row of 4254 unresolvable).
- **Overlapping NER spans produced duplicate mentions.** Running two
  specialized models over one text left 109 spans double-tagged with
  disagreeing labels. The first model in `NER_MODELS` now claims a span.
- **Mention keys ignored the entity label,** so a link made against a mention
  read as CHEMICAL landed on the entity resolved as GENE — "BRAF" the gene
  linked to "BRAF Inhibitor" the drug.
- **Entity normalization stripped identity-bearing words.** Removing
  `inhibitor`, `receptor` and `kinase` equated a drug with its target, a
  ligand with its receptor, and turned "anaplastic lymphoma kinase" into
  "anaplastic lymphoma". Only `gene` and `protein` are stripped now.

### Fixed

- **NER entity typing.** Every extracted literature entity was typed `GENE`,
  making all 78 relations GENE→GENE. Two causes compounded: `en_core_sci_md`
  emits a single `ENTITY` label that is not among the kept entity types, so the
  scispacy path returned nothing; the rule-based fallback then accepted any
  all-caps token matching `[A-Z][A-Z0-9]{2,10}` as a gene, so `ALL`, `NSCLC`,
  `PFS`, `DNA` and `ICI` all became genes.

  Extraction now runs `en_ner_bionlp13cg_md` and `en_ner_bc5cdr_md` with their
  labels mapped onto the project's types, and gene acceptance is driven by the
  KG's own gene symbols plus a non-gene acronym stoplist instead of token
  shape. Entities are now 303 DISEASE / 301 GENE / 175 CHEMICAL / 167
  CELL_TYPE / 45 TISSUE / 37 ORGANISM, and GENE→DISEASE is the most common
  relation. Gene recall is unaffected: 66 of the 67 KG gene symbols appearing
  verbatim in the corpus are still extracted.

### Documentation

- Rewrote `README.md` for accuracy. It previously claimed Phases 1–3 complete
  and "90%+ entity extraction accuracy" with no evaluation behind either.
- Added `ARCHITECTURE.md`, `CONTRIBUTING.md`, this changelog,
  `docs/Phase3_README.md` (Phase 3 had no documentation at all),
  `docs/RAG_and_Agents.md`, and `docs/LLM_Setup.md`.
- Replaced fabricated precision figures in `docs/Phase1_README.md` with
  measured counts, and stated why no precision figures are given.
- Removed `docs/LangChain_Integration_Plan.md`, a plan superseded by
  documentation of what actually shipped.

---

## [0.2.0] — 2026-08-17

Two merged pull requests. The suite went from **not collectable** to **241
passing**.

### Fixed — packaging

- **The test suite could not run at all.** `pyproject.toml` set
  `[tool.hatch.build] directory = "src"` — that key sets the build *output*
  directory — alongside `packages = ["litkg"]`, a path that does not exist. The
  editable install produced an empty `.pth`, so `import litkg` failed and all
  189 tests errored during collection. With it fixed, the real baseline was 87
  failed / 101 passed.
- `make run-phase1` was broken twice over: an import of `KGPreprocessor` (the
  class is `KnowledgeGraphPreprocessor`), and a
  `class DisambiguationEngine(...): pass` inserted **into the middle of**
  `EntityLinker`, orphaning three of its methods onto the wrong class.

### Fixed — correctness

- **Four of seven biological plausibility rules were dead.** The lookup sorted
  the type pair while the rule table's keys were unsorted, so every
  gene–disease, drug–disease, protein–disease and pathway–disease pairing fell
  through to the default and was rejected as implausible.
- **`ConfidenceScorer` ignored its own assessors.** It went straight to
  untrained networks, so component assessments had no effect on output.
- **GO IDs were treated as identity evidence.** BRCA1 and BRCA2 both carry
  `GO:0006281` ("DNA repair"), correctly — so they merged into a single entity
  despite distinct CUIs. A GO term annotates function, not identity.
- **BRCA1 and BRCA2 shared a UMLS CUI**, and all three disease UMLS lookups
  were dead (lookup uppercased, keys lowercase).
- `MultiHeadAttention` reused the query's sequence length for key and value, so
  cross-modal attention crashed whenever the two subgraphs differed in node
  count — that is, almost always.
- `ContrastiveLoss` received an alignment matrix sized on node count while
  operating on graph embeddings, and returned `nan` for single-graph batches.
- `MultiTaskLoss` summed into a bare `int`, failing at `.backward()` when no
  task matched.
- `HybridGNNModel` accepted `output_dim`, stored it, and never used it.
- `chunk_overlap` was stored and never used — overlap was genuinely zero, so
  any fact spanning a boundary was unretrievable.
- `OllamaManager.list_models()` parsed a response shape modern `ollama` no
  longer returns, so model discovery silently returned an empty list.
- Failed generations returned `"Error: ..."` as content, making failure look
  like success.
- Three components (`HypothesisGenerator`, `HypothesisValidationAgent`,
  `BiologicalPlausibilityChecker`) only ever tried OpenAI and Anthropic, then
  disabled themselves — permanently LLM-less on a local-only setup.
- `litkg.langchain_integration` was entirely unimportable; LangChain paths had
  moved to `langchain_core`.

### Added — local LLM

- Runs on **Ollama with `qwen3:8b`** by default; no API key required.
- Configurable provider order and model via `config.yaml` / `OLLAMA_HOST` /
  `LITKG_OLLAMA_MODEL`.
- Qwen3 **thinking mode off by default**. With it on, an extraction call spent
  its entire token budget on hidden reasoning and returned **zero characters
  after 128s**; off, the same call answers correctly in ~1s.
- `provider` is now a real parameter on `generate()`; a pinned model or
  provider disables fallback.
- Per-provider option filtering, with Ollama sampling parameters folded into
  its nested `options` dict.

### Added — RAG and agents

- `biomedical_agent` and `rag_system` — both were exported by `__init__.py`,
  headlined in the README, and **never written**.
- `LiteratureRetriever`, `KnowledgeGraphRetriever`, `HybridRetriever` (which
  interleaves rather than concatenates), and `GraphExpansionRetriever`.
- `BiomedicalRAGSystem` generates only from retrieved evidence with `[n]`
  citations, and refuses outright when nothing is retrieved.
- `BiomedicalToolkit`, `BiomedicalQueryAgent`, `HypothesisGenerationAgent`,
  `LiteratureValidationAgent`.

### Added — chunk ↔ graph linkage

- `EntityAliasIndex` and `ChunkGraphIndex` link passages to canonical graph
  nodes, enabling multi-hop retrieval: from one seed passage about BRCA-1,
  expansion walks `BRCA1 → homologous recombination` and surfaces an olaparib
  passage containing no query term at all.

### Added — entity resolution

- Four-rule cascade (identity identifier → normalized name → synonym overlap →
  fuzzy) with union-find for transitivity. `similarity_threshold` now does
  something.
- Surface-form matching runs globally; only the quadratic fuzzy pass uses
  blocking, because synonyms are exactly where surface forms diverge at the
  first character.
- `map_to_umls` now consults loaded ontologies (it never did), any JSON in
  `data/ontologies` auto-loads, and the UMLS REST path works behind
  `UMLS_API_KEY`.
- Seed ontology: 30 terms, 85 surface forms, and deliberately only six CUIs —
  fabricating identifiers would silently merge distinct entities.

### Added — Phase 3

- Uncertainty quantification separating **epistemic** ("unknown") from
  **aleatoric** ("contradictory") via mutual information. This was the
  README's headline claim and had no implementation.
- Platt-scaling confidence calibration.
- Adamic-Adar link prediction requiring no trained embeddings.
- Real PubMed-backed literature validation and temporal trend analysis,
  replacing placeholders.

### Changed

- Chunking: scispacy sentence splitting (the regex broke on "et al.",
  "Fig. 1", "p < 0.05"), token-based sizing capped to the embedding window,
  and section labels carried on chunk metadata.
- `build_networkx_graph` defaults to `MultiDiGraph`; Phase 2 subgraph
  extraction uses `MultiGraph`, since edge features encode `relation_type` and
  collapsing parallel edges hid relations from the model.
- `GNNTrainer` batch handling consolidated from four inconsistent copies into
  one helper.
- `LLMProvider` is a `str` enum and gained `HUGGINGFACE`.

### Removed

- `test_reports/` and the runtime FAISS vector store from version control —
  every test run rewrote tracked files. Only `index.faiss` had ever been
  committed, never the `index.pkl` that `FAISS.load_local` requires, so the
  committed store could not have loaded in a fresh clone anyway.

### Known limits

Carried forward deliberately rather than papered over:

- The ontology rule fires **zero times** on sample data — correct and tested,
  but CIVIC/TCGA records carry no CUIs.
- Entity resolution merges 447 entities, but only ~56 beyond what exact
  matching already caught.
- Literature support classification is a contradiction-cue heuristic over
  title and abstract, not entailment.
- Cross-modal linking yields 100 literature↔KG links on sample data.
- No CI, and no gold-standard evaluation set — which is why this project
  reports counts rather than precision.

---

## [0.1.0]

Initial implementation of Phases 1–3, LangChain integration, and the CLI.
