# LitKG-Integrate

[![CI](https://github.com/poglesbyg/LitKG/actions/workflows/ci.yml/badge.svg)](https://github.com/poglesbyg/LitKG/actions/workflows/ci.yml)

Integrates biomedical literature with structured knowledge graphs (CIVIC for curated clinical evidence, TCGA and CPTAC via the GDC for somatic mutation frequencies) to surface relationships that neither source states on its own.

Runs entirely on a local LLM by default. No API key required.

```bash
make quickstart          # install dependencies and models
ollama pull qwen3:8b     # the default local model
make run-phase1          # literature -> KG -> entity linking
make discover TOP=20     # rank candidate associations, with their evidence
```

---

## The end-to-end run

`make discover` is the whole thing in one command: build the graph, train a link
predictor, rank the pairs it has never seen, fetch the literature about each
one, and ask the local model what support actually exists.

```bash
make discover TOP=20 EXPLAIN=5          # propose, using all available evidence
make discover CUTOFF=2016 TOP=20        # reproduce the evaluation setting
```

With `CUTOFF` set, the graph and the literature are both restricted to before
that year and candidates curated afterwards are marked, so the output can be
checked rather than trusted.

It produces candidates for a person to judge, not findings. The ranking's
precision does not replicate across cutoffs (35x lift at 2016, 5x at 2018, 0x at
2020), so the evidence attached to each candidate is the part worth reading. The
model is asked to say what the retrieved passages do *not* support, and it does:
on a variant later confirmed by CIVIC, it correctly reported that the passages
retrieved spoke only of VHL mutations in general and not that variant.

## What this actually is

A **GraphRAG** system for biomedical discovery. Text is chunked and embedded; entities in those chunks are resolved to canonical knowledge graph nodes; retrieval can then follow graph edges to reach evidence that shares no vocabulary with the query.

The payoff is multi-hop questions. Asked *"why are BRCA1 tumours sensitive to olaparib?"*, plain vector search returns passages mentioning BRCA1. This system starts there, walks `BRCA1 → homologous recombination → PARP inhibitors`, and also returns the olaparib passage — which contains neither "BRCA1" nor any query term.

## Status

Every component listed here runs on real data. Where a number is given, it comes
from a harness in this repository that you can re-run, and where a result did
not survive replication that is said rather than omitted.

| Component | State | Evidence |
|---|---|---|
| Literature processing | **Real data** | `make run-phase1`: PubMed retrieval, typed NER, relation extraction |
| Chunking | **Real data** | Section-aware, sentence-safe, token-sized, with overlap |
| KG preprocessing | **Real data** | CIVIC 01-Aug-2026 ingestion, entity resolution cascade |
| Entity linking | **Real data** | 150 literature↔KG links |
| Link prediction | **Real data, measured** | `make train-lp`: AUC 0.748 ± 0.009 against a 0.687 structural baseline, 8 seeds, disjoint intervals |
| Retrieval | **Real data, measured** | `make eval-retrieval`: MRR 0.81, hit-rate 0.98 on 57 CIVIC-judged queries |
| Multi-hop retrieval | **Real data, partial** | 55 bridge queries: hit-rate 0.200 → 0.327 with graph expansion. The graph holds a path for 54 of 55; ranking is the limit |
| Discovery pipeline | **Real data** | `make discover`: ranks candidates and fetches the literature for each |
| Gene–gene edges (STRING) | **Real data, measured** | 1862 experiment-backed interactions among CIVIC genes. Useless at path length 3, **+0.017 AUC at length 5** (8 seeds, disjoint at two cutoffs) |
| TCGA / CPTAC mutations | **Real data, measured** | GDC open-access API (release 46.0): 528 gene–cancer type edges over 226 Cancer Gene Census genes and 35 cohorts. Does **not** improve link prediction |
| Discovery (Phase 3) | **Real data, measured** | `scripts/assess_predictions.py`: the confidence scorer carries little signal (AUC 0.613); the plausibility score is a four-valued type prior |
| Prospective validation | **Does not replicate** | 35× lift at a 2016 cutoff, 5× at 2018, **0× at 2020**. The discovery claim is withdrawn |
| Hybrid GNN (Phase 2) | **Real data, measured** | `HybridGNNModel` reaches AUC 0.633 ± 0.024 against 0.752 for a far simpler model. The collapse that held it at chance was input anisotropy, now corrected |
| Ontology coverage | **Limited** | Mechanism works; needs a licensed UMLS source |

**538 tests pass** (2 skipped: one CUDA-only, one fixture-dependent), enforced by CI on every push and pull request.

Two entries deserve emphasis, because they are the ones a reader would
otherwise assume work. `HybridGNNModel` — the cross-modal architecture this
project is named for — reaches 0.633 after a fix for anisotropic input features,
against 0.752 for the much simpler model in `litkg/phase2/link_prediction.py`,
which is where the headline link-prediction result comes from.
And the prospective discovery result held at one time cutoff and vanished at
two others.

Numbers from `make run-phase1` on the bundled sample data with the CIVIC
01-Aug-2026 release and GDC data release 46.0: 3026 entities and 14920
relations, of which 528 are GDC gene–cancer type associations.

### On reading any number here

Five results in this project failed replication after looking solid: "the graph
is too sparse for link prediction", a doubled MRR, an inverted precision curve,
a story about the model ranking obviousness over novelty, and the prospective
lift above. Each was measured carefully at a single configuration.

The harnesses that caught them are in the repository, and the working rule is
that a single-seed or single-cutoff number is a hypothesis. Use `--seeds` and
more than one `--cutoff` before believing anything, including the figures in
this table.

Each of the five is written up with its mechanism in
[Five results that did not replicate](https://poglesbyg.github.io/blog/2026/08/19/five-results-that-did-not-replicate/),
including the one that was a step away from shipping as a feature.

## Install

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/poglesbyg/LitKG.git
cd LitKG
make quickstart
```

`quickstart` installs dependencies, downloads the scispacy biomedical models, and writes a `.env` template.

> **Note:** `uv sync` prunes the scispacy models, because `setup_models.py` installs them outside the lockfile. If sentence splitting silently degrades to regex, re-run `uv run python scripts/setup_models.py`.

### The local LLM

```bash
ollama pull qwen3:8b
```

That is the whole setup. Provider order defaults to `ollama → anthropic → openai`, so cloud providers are used only if you set their keys. See [docs/LLM_Setup.md](docs/LLM_Setup.md) for configuration and one important Qwen3 gotcha.

### Optional API keys

Copy `env.template` to `.env` and fill in what you have. Everything is optional:

| Variable | Enables |
|---|---|
| `PUBMED_EMAIL` | PubMed retrieval (NCBI requires an email) |
| `UMLS_API_KEY` | Real ontology coverage for entity resolution |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Cloud LLM fallback |

## Usage

### Pipelines

```bash
make run-phase1                  # literature + KG + entity linking
make discover TOP=20             # rank candidates and gather their evidence
make train-lp SEEDS=8            # train the link predictors and compare
make rag Q="..."                 # ask a question over the corpus
```

Evaluation, all of which re-runs the numbers in the status table:

```bash
make evaluate                    # temporal-holdout link prediction
make eval-retrieval SWEEP=1      # retrieval against CIVIC-judged queries
python scripts/replicate_prospective.py   # the same result at three cutoffs
```

The `make run-phase2`, `run-phase3`, `run-langchain` and `run-discovery`
targets are **demonstrations on synthetic input** — `torch.randn` tensors and
hardcoded documents. They exercise the code paths and say nothing about
behaviour on data. The commands above are the ones that touch the real graph.

### Asking questions over your corpus

`RAGPipeline` assembles everything from what `make run-phase1` already wrote:

```bash
make rag Q="Why are BRCA1 tumours sensitive to olaparib?"
make rag-coverage        # index stats, no LLM call
python scripts/run_rag.py "..." --retrieval-only   # inspect retrieval alone
```

```python
from litkg.langchain_integration import RAGPipeline, PipelineConfig

pipeline = RAGPipeline(PipelineConfig(max_hops=1, k=5)).build()
result = pipeline.rag_system().answer("Why are BRCA1 tumours sensitive to olaparib?")
print(result["answer"])    # cites evidence as [1], [2], ...
print(result["sources"])   # each carries pmid and hop_distance
```

To assemble the parts yourself instead:

```python
from litkg.langchain_integration import (
    BiomedicalRAGSystem, EntityAliasIndex, ChunkGraphIndex,
)

# Link chunks to graph nodes so retrieval can traverse
alias_index = EntityAliasIndex().add_from_graph(knowledge_graph)
chunk_index = ChunkGraphIndex(alias_index)
chunk_index.index_chunks(chunks)

rag = BiomedicalRAGSystem(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph,
    chunk_index=chunk_index,
    max_hops=2,              # 0 disables graph expansion
)

result = rag.answer("Why are BRCA1 tumours sensitive to olaparib?")
print(result["answer"])      # cites evidence as [1], [2], ...
print(result["sources"])     # each tagged with hop_distance
```

If nothing is retrieved, the system says so rather than answering from model memory.

### Conversational agent

```python
from litkg.langchain_integration import BiomedicalQueryAgent, BiomedicalToolkit

agent = BiomedicalQueryAgent(toolkit=BiomedicalToolkit(rag_system=rag))
agent.chat("What is the role of BRCA1 in DNA repair?")
agent.chat("Propose a hypothesis about resistance mechanisms")
```

### Building a knowledge graph

```python
from litkg.phase1 import KGPreprocessor

kg = KGPreprocessor()
kg.download_all_data()
kg.process_all_data()        # includes the entity resolution cascade
kg.save_integrated_graph("data/processed/kg.json")
```

## Documentation

| Document | Covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the pieces fit, and why |
| [docs/LLM_Setup.md](docs/LLM_Setup.md) | Ollama, model choice, provider fallback |
| [docs/Phase1_README.md](docs/Phase1_README.md) | Literature, KG preprocessing, entity resolution |
| [docs/Phase2_README.md](docs/Phase2_README.md) | Hybrid GNN, cross-modal attention, and why it trails a far simpler model |
| [docs/Phase3_README.md](docs/Phase3_README.md) | Confidence, novelty, and what its scores are worth |
| [docs/RAG_and_Agents.md](docs/RAG_and_Agents.md) | Retrievers, chunking, chunk↔graph linkage |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, tests, conventions |
| [docs/Evaluation.md](docs/Evaluation.md) | Temporal holdout, baselines, what is and isn't measured |
| [CHANGELOG.md](CHANGELOG.md) | What changed and when |
| [Five results that did not replicate](https://poglesbyg.github.io/blog/2026/08/19/five-results-that-did-not-replicate/) | Write-up of the five withdrawn results and the mechanism behind each |

## Known limits

Stated plainly, because they affect how far you should trust output:

- **Ontology coverage is thin.** Entity resolution's strongest rule matches on UMLS CUIs, but the bundled seed carries only six real CUIs. It fires zero times on sample data. Set `UMLS_API_KEY` for real coverage. Fabricated CUIs were deliberately not added: a wrong shared CUI silently merges two distinct entities while looking authoritative.
- **Literature support classification is a heuristic.** `LiteratureCrossValidator` judges support by scanning for contradiction cues ("no association", "failed to") in title and abstract. It is not entailment, and it does not read full text.
- **Entity resolution gains are modest on sample data.** The cascade merges 453 entities, but only ~56 beyond what exact matching already caught. The bottleneck is input identifier coverage, not the algorithm.
- **TCGA and CPTAC contribute mutation frequencies, and they do not help link prediction.** Both now come from the GDC open-access API rather than the fabricated rows that used to stand in for them, and the biology is right: IDH1 at 193x background in lower-grade glioma, GNAQ and GNA11 in uveal melanoma, BRAF in thyroid carcinoma. But adding those edges to the training backbone moves link prediction by -0.002 AUC at a 2016 cutoff (seed ranges overlap, so indistinguishable) and -0.004 at 2020 (disjoint seed ranges, so a small real *degradation*). `--with-gdc` is therefore off by default. The edges are in the graph, where retrieval and the discovery pipeline can use them; the claim that they improve prediction is not made.
- **Gene–gene edges help a path counter and do not help the hybrid model.** Feeding STRING edges to `HybridLinkPredictor` — either into the message-passing graph or as a length-5 component in the blend — leaves every configuration overlapping the no-PPI baseline across 8 seeds at both cutoffs, and mostly slightly worse. The standalone `PathPowerPredictor(5)` result below stands; the hybrid route does not. An earlier 4-seed run of the same comparison showed +0.025 and was a variance artifact, which is the sixth time a single small-sample number here has failed to replicate.

- **The hybrid's blend weight is no longer chosen on validation, because choosing it was worse than not.** A fixed even split reaches AUC 0.7451 ± 0.0123 (AP 0.271) on the 2016 holdout across 8 seeds, against 0.7404 ± 0.0214 (AP 0.243) for validation selection — worse and twice as noisy. The slice is a few hundred edges, so the selected weight was mostly noise. `select_weight=True` restores the search.

- **The selection it replaced was also leaking.** It scored validation positives while their own edge was still in the graph, so a path counter could walk the edge it was predicting: L3 scores inflated **3.48×** and length-5 scores **4.69×** against negatives that get no such boost. That biased selection toward path counting and toward whichever counter was longest — with a length-5 component present it chose a pure length-5 blend in 8 of 8 seeds. Selection now refits every component, the GNN included, on a graph with the validation edges removed.

- **Gene–gene edges help, but only at path length 5, and the reason is structural.** The graph is strictly multipartite, so every predictor here routes through length-3 paths. A gene–gene edge cannot sit on a length-3 path from a variant to a disease: the middle hop would need a gene adjacent to that disease, and CIVIC has **no gene–disease edges at all**. Measured directly, adding 1862 STRING interactions changes the L3 path count for **0 of 1388** test pairs — the edges are not weak there, they are unreachable. At length 5 the route `variant → gene → gene → variant → disease` exists, and the same edges are worth **+0.0168 AUC at a 2016 cutoff (0.7105 → 0.7273) and +0.0076 at 2020 (0.7846 → 0.7921)**, 8 seeds, disjoint ranges at both. Average precision goes 0.228 → 0.299 and Hits@100 0.042 → 0.093, which is the ranking head this project has been weakest at. `make evaluate-ppi`.

- **STRING's `combined_score` must not be used here, and the margin is large.** It fuses seven evidence channels, one of which is co-occurrence in PubMed abstracts — the same papers CIVIC curators read to write the labels. Among CIVIC's 973 genes, textmining alone would contribute **14,380 edges** against **1,862** from physical experiments and 3,795 from curated databases. KRAS–BRCA1 scores 0.721 combined, of which 0.721 is textmining and 0.000 is experiments. `litkg.phase1.string_ppi` therefore defaults to the `experimental` channel and **refuses** `textmining` and `database` unless `allow_literature_channels=True` is passed deliberately. Every number above is from experiments only.

- **The variant-level CIVIC/GDC join is correct and unmeasurable.** The gene-level join failed on granularity — TP53 links to 14 of 15 diseases, so the edge is close to a prior — and the obvious fix is to join on CIVIC's actual unit, the variant. That works: BRAF alone is uninformative, while BRAF V600E concentrates in thyroid (283 cases) and melanoma (200). It is also aimed at the largest test category, since 39% of held-out pairs are DISEASE–MUTATION. It still cannot be measured, because the two sources barely overlap:

  | Filter | Survives |
  |---|---|
  | 546 DISEASE–MUTATION test pairs | |
  | variant is a simple protein change | 186 of 463 variants (40%) |
  | …and TCGA observed it | 85 (18%) |
  | disease is a TCGA cohort | 14 of 78 (18%) |
  | **both sides resolve** | **20 pairs — 1.4% of the test set** |

  Two independent ~18% filters. Improving the cohort mapping from 22 to 30 of 33 changed that number **not at all**, so the bottleneck is not the join quality: TCGA is 33 common adult solid tumours and CIVIC spans 246 diseases including rare and haematological ones. Only 17 of 78 test diseases are TCGA cancers in the best case. `scripts/evaluate_gdc_variants.py` reports this; the predictor scores AUC 0.501 on 0.9% coverage, which is an abstention rate, not a result.

- **Only 14 of 33 TCGA cohorts join CIVIC by name.** GDC names a cohort "Breast Invasive Carcinoma" where CIVIC has "Breast Cancer", and matching is exact after normalisation on purpose: fuzzy matching would merge "Lung Adenocarcinoma" with "Lung Squamous Cell Carcinoma", which are different diseases with different drivers, and nothing downstream would reveal the error. So 153 of 488 TCGA edges reach the evaluation graph. `litkg.phase1.disease_ontology` now resolves cohort names to DOIDs against the Disease Ontology's synonym list and `is_a` hierarchy, taking that to 22 of 33 by identifier rather than string. It does not help, per the entry above.
- **GDC mutation counts are length-corrected, and the correction is not optional.** Ranking genes by raw somatic mutation count measures coding length: the GDC's own endpoint answers "top mutated genes in breast cancer" with OBSCN, PIK3C2B, NID1, NFASC and USH2A, none of which is a breast cancer driver. Restricting to the Cancer Gene Census is not sufficient either — among Census genes the number of cohorts a gene reaches correlates with log CDS length at **+0.798**, higher than the +0.542 for raw counts, because thresholding on a count selects for length. Scoring against a length-aware background rate takes that to **-0.097**. The background model assumes a uniform per-base rate within a cohort, which real tumours violate; it is enough to remove the confound, not enough to call a gene a driver.
- **The GDC release postdates every cutoff, so early cutoffs are not safe.** Data release 46.0 is from August 2026. TCGA's sequencing was substantially complete years before 2016, so a 2016 or 2020 holdout is defensible, but a cutoff in the early 2010s would be reading the future. A leakage guard drops any GDC edge coinciding with a held-out CIVIC pair; it currently drops zero, because CIVIC has **no direct gene–disease edges at all** (it links gene→variant→disease), which is also why these edges are a new edge type rather than a duplicate one.
- **CPTAC proteomics is still not integrated.** CPTAC's genomic data is in the GDC and is used; its proteomics is in the Proteomic Data Commons, a separate API. The two CPTAC cohorts contribute 40 mutation edges, not protein expression.
- **The headline Phase 2 architecture underperforms.** `HybridGNNModel` and its cross-modal attention now run on the real graph and reach AUC 0.633 ± 0.024, up from chance once anisotropic input features were corrected, but still well behind the 0.752 of the much simpler model in `litkg/phase2/link_prediction.py` — which is where every link-prediction number quoted above comes from. Cross-modal attention costs about 0.12 AUC here, which is an architectural question rather than a bug.
- **BERT NER fails on long abstracts.** Phase 1 logs `size of tensor a (543) must match tensor b (512)` for any abstract over ~512 tokens. The error is caught and the pipeline falls back to other extractors, so it succeeds — but one component silently contributes nothing on long documents.
- **Multi-hop retrieval works partially, and is measured on a query set built for it.** On 55 bridge queries — where the relevant papers are lexically disjoint from the question by construction — the graph holds a path to the answer for 54, but a relevant paper reaches the top 10 for only about a quarter of them (hit-rate 0.200 → 0.327 with graph expansion and rank fusion). Ranking those candidates by similarity to the query *nullifies* expansion, since bridge evidence is by definition what the question does not resemble; ranking by shared graph context is what works. Expansion remains off by default for single-relationship questions, where it adds nothing. See [docs/RAG_and_Agents.md](docs/RAG_and_Agents.md).
- **Ranking metrics are noisy.** Each positive is ranked against ~12000 negatives, so MRR is set by a couple of dozen rows and its confidence interval is about as wide as its value. Compare intervals, and prefer Hits@100. See [docs/Evaluation.md](docs/Evaluation.md).
- **Link prediction is measured, and a trained model now beats the structural baseline.** Under a temporal holdout (train on pre-2016 papers) with popularity controlled for, a GNN ensemble with node text features reaches AUC 0.748 ± 0.009 at a 2016 cutoff and 0.791 ± 0.008 at 2020 against 0.687 [0.674–0.702] for length-3 paths alone and 0.543 for Adamic-Adar at 2016 — disjoint intervals. Figures are from the CIVIC 01-Aug-2026 release; later cutoffs score higher because they are easier problems, not better methods. Text features are worth +0.020 AUC over topology alone across 8 seeds, with non-overlapping ranges. The graph is strictly multipartite (0 of 6769 edges join same-type nodes) and every held-out pair is cross-type, so shared-neighbour methods are undefined on it by construction. Ranking quality is still poor — about 10% of held-out associations reach the top 100 — so this is signal, not a working discovery system. Results are reported per entity-type pair, since the aggregate averages four subproblems whose AUC ranges from 0.638 to 0.802, and every metric carries a bootstrap interval because MRR here is set by a couple dozen rows. See [docs/Evaluation.md](docs/Evaluation.md).
- **Cross-modal linking is still the weak point,** at 167 literature↔KG links against 2367 unlinked literature entities — though disease↔disease and chemical↔drug links now exist, which the gene-only KG made impossible. Cell types, tissues and organisms still have no KG counterpart; closing that needs a source beyond CIVIC.
- **Blocking trades recall for cost.** The fuzzy matching pass blocks candidates by entity type and first character, so pairs like `BRCA1`/`RCA1` are never compared.
- **`biomedical_score` values are hand-assigned.** Model rankings in `ModelSelector` are informed estimates, not benchmark results.

## License

MIT. See [LICENSE](LICENSE).
