# Changelog

Notable changes to LitKG-Integrate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
