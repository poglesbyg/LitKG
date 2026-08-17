# Architecture

How LitKG fits together, and why the pieces are shaped the way they are.

## The central idea

Two sources of biomedical knowledge have complementary failure modes.

**Literature** is comprehensive and current but unstructured: the relationship between BRCA1 and PARP inhibitors is stated across thousands of papers, in prose, with no machine-readable link between them.

**Curated knowledge graphs** (CIVIC, TCGA, CPTAC) are structured and reliable but sparse and lagging: they encode `BRCA1 —SENSITIZES_TO→ PARP inhibitors` explicitly, but only after curators have added it.

Neither alone answers a multi-hop question. The design joins them so that retrieval over text can *traverse* the graph, and prediction over the graph can be *grounded* in text.

## Data flow

```
PubMed ──> chunking ──> embeddings ──> vector store
              │                             │
              │ entities mentioned          │ seed retrieval
              v                             v
        EntityAliasIndex ────> ChunkGraphIndex <──── GraphExpansionRetriever
              ^                      ^  │                      │
              │                      │  │ chunks for node      │ cited answer
CIVIC ──┐     │ canonical names      │  v                      v
TCGA  ──┼─> entity resolution ──> knowledge graph ──────> BiomedicalRAGSystem
CPTAC ──┘                              │
                                       │ subgraphs
                                       v
                                  hybrid GNN ──> novelty ──> hypotheses ──> validation
```

The two halves meet at `ChunkGraphIndex`. Everything above it is retrieval; everything below is learned prediction.

## Phase 1 — Foundation

**Literature processing** (`litkg.phase1.literature_processor`) retrieves from PubMed and extracts entities and relations with biomedical NER.

**Chunking** (`litkg.langchain_integration.BiomedicalTextSplitter`) splits papers for embedding. Four decisions matter:

- *Section-aware.* Splits on Abstract/Methods/Results headers, and the section name travels with the chunk. This is evidential weight, not decoration: a claim in Results is a finding the paper established, while the same sentence in Introduction is background attributed elsewhere.
- *Sentence-safe.* Uses scispacy, because a naive `(?<=[.!?])\s+` breaks biomedical prose at "et al.", "Fig. 1", and "p < 0.05".
- *Overlapping.* Adjacent chunks share boundary sentences, so a fact spanning a split stays retrievable. Overlap is carried as whole sentences so the repeated fragment still embeds meaningfully.
- *Token-sized.* Length is measured in tokens against the embedding model's window, since a longer chunk is silently truncated at embedding time.

**KG preprocessing** (`litkg.phase1.kg_preprocessor`) ingests the curated sources and runs **entity resolution**, the step everything graph-shaped depends on.

### Entity resolution cascade

Resolution runs strongest evidence first, accumulating matches in a union-find so results are transitive (if A~B and B~C, all three collapse):

| Rule | Evidence | Example |
|---|---|---|
| 1 | Shared identity identifier (UMLS CUI) | `BRCA1` = `breast cancer 1` via `C0376571` |
| 2 | Identical normalized name | `BRCA-1` = `BRCA1` after folding punctuation |
| 3 | Synonym overlap | `p53` = `TP53` |
| 4 | Fuzzy similarity above threshold | `EGFR1` ≈ `EGFR` |

Two design points worth knowing:

**GO IDs are excluded from rule 1.** A Gene Ontology term annotates what an entity *does*, not which entity it *is*. BRCA1 and BRCA2 both carry `GO:0006281` ("DNA repair") — correctly — so treating that as identity evidence merges two distinct genes. Only identifiers in `IDENTITY_IDENTIFIERS` are decisive.

**Rules 1–3 run globally; only rule 4 uses blocking.** Blocking (comparing only candidates sharing a type and first character) makes the quadratic fuzzy pass tractable. But synonyms are precisely where surface forms diverge at the first character — `TP53` blocks under "t" and its synonym `p53` under "p" — so blocking them would hide real matches. Surface-form matching is an O(n) index instead.

## The join — chunk ↔ graph linkage

`litkg.langchain_integration.graph_linking` is what makes this GraphRAG rather than two parallel retrievals.

**`EntityAliasIndex`** maps every surface form a node is known by back to its canonical id. Matching is longest-alias-first (so `breast cancer 1` beats a bare `cancer` nested inside it) and word-bounded (so `TP53` does not match inside `TP53BP1`, a different gene).

**`ChunkGraphIndex`** annotates each chunk with the nodes it mentions and maintains the reverse map, so you can go from a passage to its entities and from an entity to every passage discussing it.

These compose with entity resolution: canonical nodes that have absorbed their duplicates' names carry richer alias sets, so **better resolution directly raises linking recall.**

## Retrieval

Four retrievers, all implementing the LangChain `BaseRetriever` interface:

| Retriever | Strategy |
|---|---|
| `LiteratureRetriever` | Semantic similarity over embedded passages |
| `KnowledgeGraphRetriever` | Structured lookup of relations involving named entities |
| `HybridRetriever` | Interleaves both, so neither is crowded out of the top-k |
| `GraphExpansionRetriever` | Similarity seeds, then graph traversal from those seeds |

`GraphExpansionRetriever` is the multi-hop one. It retrieves by similarity, resolves the seed passages to graph nodes, walks `max_hops`, and returns the passages attached to nodes it reaches — tagged with `hop_distance` and the `via_entity` that led there.

`BiomedicalRAGSystem` generates strictly from retrieved evidence with `[n]` citations, and **refuses outright when nothing is retrieved** rather than falling back on model memory.

## Phase 2 — Hybrid GNN

`litkg.phase2` learns joint representations over literature subgraphs and KG subgraphs.

- `LiteratureGraphEncoder` and `KnowledgeGraphEncoder` encode each modality. Both default missing edge features to zeros at their own configured dimension, so callers need not fabricate them.
- `CrossModalAttention` lets each modality attend to the other. Query and key/value have **different** sequence lengths here — literature and KG subgraphs rarely have equal node counts — which is why the attention implementation tracks `tgt_len` and `src_len` separately.
- `CrossModalFusion` combines them; `RelationPredictor` scores entity pairs.

Subgraph extraction uses a `MultiGraph`, not a simple graph, because edge features one-hot encode `relation_type`. Collapsing parallel edges would keep only the last relation between a pair and hide the rest from the model.

## Phase 3 — Discovery

`litkg.phase3` turns representations into candidate knowledge.

**Confidence scoring** assesses literature and experimental evidence separately, then fuses them. It reports **two kinds of uncertainty**, which is the distinction the project cares about:

- *Epistemic* — disagreement between samples. Reducible with more data; high when the model is out of its depth. This is "unknown".
- *Aleatoric* — mean entropy within samples. Irreducible; high when evidence genuinely conflicts. This is "contradictory".

They are separated via mutual information over repeated predictions. `ConfidenceCalibrator` applies Platt scaling so a reported 0.9 corresponds to being right about 90% of the time.

**Novelty detection** predicts missing edges. With trained embeddings it uses the GNN; without them, `predict_from_graph` falls back to Adamic-Adar, scoring unconnected pairs by shared neighbours weighted so that rare shared neighbours count for more than hubs. That path deliberately flattens the graph to simple undirected form, because Adamic-Adar is only defined there.

**Hypothesis generation** turns novel relations into testable statements, ranked by a priority score balancing confidence, novelty, feasibility, and existing evidence.

**Validation** cross-checks against PubMed, analyses temporal trends in support, and aggregates expert assessments weighted by each expert's stated confidence.

## LLM layer

`litkg.llm_integration` presents one interface over Ollama, OpenAI, Anthropic, and OpenAI-compatible endpoints.

- Provider order comes from config, defaulting to local-first.
- Options are filtered per provider before reaching the SDK, so an unrecognized keyword surfaces as a warning rather than an opaque `TypeError` from inside a client library.
- Ollama sampling parameters are folded into its nested `options` dict, so callers can write `top_p=0.9` naturally.
- A pinned model or provider **disables fallback** — silently switching providers defeats the point of pinning one.

See [docs/LLM_Setup.md](docs/LLM_Setup.md).

## Design principles visible in the code

**Fail loudly at boundaries.** A failed generation raises rather than returning `"Error: ..."` as content, which would make failure look like success. The exception is LangChain tool functions, where returning error text lets the agent recover and raising aborts the run — those are marked as such.

**Explicit over accidental.** Where information must be discarded (flattening a graph for Adamic-Adar, dropping an unsupported provider option), it happens deliberately and says why.

**Partial results are representable.** Metric dataclasses default to zero rather than requiring every field, so a literature-only assessment does not have to invent experimental numbers.

**Aliases, not duplicated state.** Renamed fields are read-only properties or `__post_init__`-synced pairs, so there is one source of truth.
