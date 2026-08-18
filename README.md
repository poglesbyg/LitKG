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

| Component | State | Notes |
|---|---|---|
| Literature processing | Working | PubMed retrieval, biomedical NER, relation extraction |
| Chunking | Working | Section-aware, sentence-safe, token-sized, with overlap |
| KG preprocessing | Working | CIVIC/TCGA/CPTAC ingestion, entity resolution cascade |
| Entity linking | Working | Fuzzy, semantic, and contextual disambiguation |
| Hybrid GNN (Phase 2) | Working | Cross-modal attention, trains end to end |
| Discovery (Phase 3) | Working | Novelty detection, hypothesis generation, validation |
| RAG + agents | Working | Local-first, cited answers, multi-hop retrieval |
| Ontology coverage | **Limited** | Mechanism works; needs a licensed UMLS source for real coverage |
| Cross-modal linking | **Limited** | 100 literature↔KG links on sample data |

**276 tests pass**, enforced by CI on every push and pull request.

Numbers below come from `make run-phase1` on the bundled sample data: 2084 nodes, 6003 edges, 75.2% high-confidence linking rate.

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
| [CHANGELOG.md](CHANGELOG.md) | What changed and when |

## Known limits

Stated plainly, because they affect how far you should trust output:

- **Ontology coverage is thin.** Entity resolution's strongest rule matches on UMLS CUIs, but the bundled seed carries only six real CUIs. It fires zero times on sample data. Set `UMLS_API_KEY` for real coverage. Fabricated CUIs were deliberately not added: a wrong shared CUI silently merges two distinct entities while looking authoritative.
- **Literature support classification is a heuristic.** `LiteratureCrossValidator` judges support by scanning for contradiction cues ("no association", "failed to") in title and abstract. It is not entailment, and it does not read full text.
- **Entity resolution gains are modest on sample data.** The cascade merges 447 entities, but only ~56 beyond what exact matching already caught. The bottleneck is input identifier coverage, not the algorithm.
- **NER typing is unreliable, which now bounds relation precision.** Every extracted literature relation has GENE→GENE endpoints, because the tagger labels diseases (`ALL`, `NSCLC`), outcomes (`PFS`) and therapies (`CAR`) as genes. Relation *extraction* works; relation *quality* is limited by this.
- **Blocking trades recall for cost.** The fuzzy matching pass blocks candidates by entity type and first character, so pairs like `BRCA1`/`RCA1` are never compared.
- **`biomedical_score` values are hand-assigned.** Model rankings in `ModelSelector` are informed estimates, not benchmark results.

## License

MIT. See [LICENSE](LICENSE).
