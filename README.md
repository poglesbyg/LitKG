# LitKG-Integrate

[![CI](https://github.com/poglesbyg/LitKG/actions/workflows/ci.yml/badge.svg)](https://github.com/poglesbyg/LitKG/actions/workflows/ci.yml)

Integrates biomedical literature with structured knowledge graphs (CIVIC, TCGA, CPTAC) to surface relationships that neither source states on its own.

Runs entirely on a local LLM by default. No API key required.

```bash
make quickstart          # install dependencies and models
ollama pull qwen3:8b     # the default local model
make run-phase1          # literature -> KG -> entity linking
```

---

## What this actually is

A **GraphRAG** system for biomedical discovery. Text is chunked and embedded; entities in those chunks are resolved to canonical knowledge graph nodes; retrieval can then follow graph edges to reach evidence that shares no vocabulary with the query.

The payoff is multi-hop questions. Asked *"why are BRCA1 tumours sensitive to olaparib?"*, plain vector search returns passages mentioning BRCA1. This system starts there, walks `BRCA1 → homologous recombination → PARP inhibitors`, and also returns the olaparib passage — which contains neither "BRCA1" nor any query term.

## Status

The distinction that matters here is whether a component has been **run on the
real graph** or only demonstrated on synthetic inputs. Several `make run-*`
targets construct `torch.randn` tensors and hardcoded documents; they exercise
the code path but say nothing about whether it works on data.

| Component | State | Evidence |
|---|---|---|
| Literature processing | **Real data** | `make run-phase1`: PubMed retrieval, typed NER, relation extraction |
| Chunking | **Real data** | Section-aware, sentence-safe, token-sized, with overlap |
| KG preprocessing | **Real data** | CIVIC 01-Aug-2026 ingestion, entity resolution cascade |
| Entity linking | **Real data** | 152 literature↔KG links |
| Link prediction | **Real data, measured** | `make train-lp`: AUC 0.748 ± 0.009 vs 0.687 structural baseline |
| Hybrid GNN (Phase 2) | **Synthetic demo only** | `HybridGNNModel` (1.8M params) instantiates and a training loop runs on `torch.randn`; it has never been trained on `phase2_graph_data.json` |
| Discovery (Phase 3) | **Synthetic demo only** | Runs on 6 hardcoded relationships |
| RAG + agents | **Real data** | `make rag Q="..."` builds the index from Phase 1 output and answers with citations. Retrieval relevance is unmeasured — there is no relevance-judged query set |
| Ontology coverage | **Limited** | Mechanism works; needs a licensed UMLS source |

**452 tests pass**, enforced by CI on every push and pull request.

The two "synthetic demo" rows are the honest gap: those demos each print their
own "Next Steps: 1. Prepare real data" on completion. The measured link
prediction result comes from a separate, simpler model in
`litkg/phase2/link_prediction.py`, not from `HybridGNNModel`.

Numbers from `make run-phase1` on the bundled sample data with the CIVIC
01-Aug-2026 release: 3822 nodes, 14810 edges, 152 cross-modal links.

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
make run-phase1      # literature + KG + entity linking
make run-phase2      # hybrid GNN training
make run-phase3      # confidence scoring and discovery
make run-langchain   # RAG and agent demo
```

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
| [docs/Phase2_README.md](docs/Phase2_README.md) | Hybrid GNN and cross-modal attention |
| [docs/Phase3_README.md](docs/Phase3_README.md) | Confidence, novelty, hypotheses, validation |
| [docs/RAG_and_Agents.md](docs/RAG_and_Agents.md) | Retrievers, chunking, chunk↔graph linkage |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, tests, conventions |
| [docs/Evaluation.md](docs/Evaluation.md) | Temporal holdout, baselines, what is and isn't measured |
| [CHANGELOG.md](CHANGELOG.md) | What changed and when |

## Known limits

Stated plainly, because they affect how far you should trust output:

- **Ontology coverage is thin.** Entity resolution's strongest rule matches on UMLS CUIs, but the bundled seed carries only six real CUIs. It fires zero times on sample data. Set `UMLS_API_KEY` for real coverage. Fabricated CUIs were deliberately not added: a wrong shared CUI silently merges two distinct entities while looking authoritative.
- **Literature support classification is a heuristic.** `LiteratureCrossValidator` judges support by scanning for contradiction cues ("no association", "failed to") in title and abstract. It is not entailment, and it does not read full text.
- **Entity resolution gains are modest on sample data.** The cascade merges 453 entities, but only ~56 beyond what exact matching already caught. The bottleneck is input identifier coverage, not the algorithm.
- **The headline Phase 2 architecture is unvalidated.** `HybridGNNModel` and its cross-modal attention have only ever seen random tensors. The link prediction numbers quoted above come from a different, much simpler model.
- **BERT NER fails on long abstracts.** Phase 1 logs `size of tensor a (543) must match tensor b (512)` for any abstract over ~512 tokens. The error is caught and the pipeline falls back to other extractors, so it succeeds — but one component silently contributes nothing on long documents.
- **Graph expansion is noisy.** On the BRCA1/olaparib question all five seed passages are on-target, but only about one in five graph-expanded passages is relevant, even with the hub-traversal cap. There is no relevance-judged query set, so retrieval quality is unmeasured and that ratio is an observation on one question, not a metric.
- **Ranking metrics are noisy.** Each positive is ranked against ~12000 negatives, so MRR is set by a couple of dozen rows and its confidence interval is about as wide as its value. Compare intervals, and prefer Hits@100. See [docs/Evaluation.md](docs/Evaluation.md).
- **Link prediction is measured, and a trained model now beats the structural baseline.** Under a temporal holdout (train on pre-2016 papers) with popularity controlled for, a GNN ensemble with node text features reaches AUC 0.748 ± 0.009 at a 2016 cutoff and 0.791 ± 0.008 at 2020 against 0.687 [0.674–0.702] for length-3 paths alone and 0.543 for Adamic-Adar at 2016 — disjoint intervals. Figures are from the CIVIC 01-Aug-2026 release; later cutoffs score higher because they are easier problems, not better methods. Text features are worth +0.020 AUC over topology alone across 8 seeds, with non-overlapping ranges. The graph is strictly multipartite (0 of 6769 edges join same-type nodes) and every held-out pair is cross-type, so shared-neighbour methods are undefined on it by construction. Ranking quality is still poor — about 10% of held-out associations reach the top 100 — so this is signal, not a working discovery system. Results are reported per entity-type pair, since the aggregate averages four subproblems whose AUC ranges from 0.638 to 0.802, and every metric carries a bootstrap interval because MRR here is set by a couple dozen rows. See [docs/Evaluation.md](docs/Evaluation.md).
- **Cross-modal linking is still the weak point,** at 167 literature↔KG links against 2367 unlinked literature entities — though disease↔disease and chemical↔drug links now exist, which the gene-only KG made impossible. Cell types, tissues and organisms still have no KG counterpart; closing that needs a source beyond CIVIC.
- **Blocking trades recall for cost.** The fuzzy matching pass blocks candidates by entity type and first character, so pairs like `BRCA1`/`RCA1` are never compared.
- **`biomedical_score` values are hand-assigned.** Model rankings in `ModelSelector` are informed estimates, not benchmark results.

## License

MIT. See [LICENSE](LICENSE).
