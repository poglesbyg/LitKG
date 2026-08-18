# RAG, Agents, and Chunk↔Graph Linkage

Everything in `litkg.langchain_integration`. This is the retrieval half of the system: chunking text, linking it to the knowledge graph, retrieving over both, and answering questions with citations.

All of it runs on the local model by default. No API key required.

---

## Chunking

`BiomedicalTextSplitter`

```python
from litkg.langchain_integration import BiomedicalTextSplitter

splitter = BiomedicalTextSplitter(
    chunk_size=400,          # measured in tokens by default
    chunk_overlap=80,
    length_unit="tokens",    # or "characters"
    model_max_tokens=512,    # the embedding window; chunk_size is capped to it
    section_aware=True,
    preserve_sentences=True,
)
```

Four things it does that a generic splitter does not:

**Sentence-safe splitting.** Uses scispacy, because a naive `(?<=[.!?])\s+` breaks biomedical prose at "et al.", "Fig. 1", "p < 0.05", and "S. aureus". Falls back to an abbreviation-aware regex when no spaCy model is installed.

```python
splitter.split_sentences("Shown by Smith et al. in Fig. 2 (p < 0.05). Results were clear.")
# ['Shown by Smith et al. in Fig. 2 (p < 0.05).', 'Results were clear.']
```

**Real overlap.** Adjacent chunks share boundary sentences, so a fact spanning a split remains retrievable from at least one chunk. Overlap is carried as whole sentences rather than a character slice, so the repeated fragment is readable and embeds meaningfully. `chunk_overlap >= chunk_size` raises, since chunking could not advance.

**Token-based sizing.** Length is measured against the embedding model's window. A chunk longer than the window is silently truncated at embedding time, so `chunk_size` is capped to `model_max_tokens`.

**Section provenance.**

```python
for chunk, section in splitter.split_text_with_sections(paper_text):
    print(section, chunk[:60])   # "Results", "We observed synthetic lethality..."
```

This is evidential weight, not decoration: a claim in Results is a finding the paper established, while the same sentence in Introduction is background attributed to someone else. Text before the first recognized header is returned under `None` rather than misattributed.

---

## Linking chunks to the graph

`EntityAliasIndex` and `ChunkGraphIndex` — the join that makes this GraphRAG rather than two parallel retrievals.

```python
from litkg.langchain_integration import EntityAliasIndex, ChunkGraphIndex

alias_index = EntityAliasIndex().add_from_graph(knowledge_graph)
# or: .add_from_entities(standardized_entities)

chunk_index = ChunkGraphIndex(alias_index)
stats = chunk_index.index_chunks(chunks)
# {'chunks': 1200, 'linked_chunks': 940, 'nodes_covered': 310, 'total_mentions': 2180}
```

Indexing annotates each chunk's metadata with `entity_ids`, `entity_surface_forms`, and a stable `chunk_uid` that survives a round trip through a vector store.

**Matching rules.** Longest alias first, so `breast cancer 1` wins over a bare `cancer` nested inside it. Word-bounded, so `TP53` does not match inside `TP53BP1` — a different gene. Aliases shorter than `min_alias_length` (default 3) are ignored, since two-letter surface forms produce more false positives than signal.

**This composes with entity resolution.** Canonical nodes that have absorbed their duplicates' names carry richer alias sets, so improving `merge_duplicate_entities` directly raises linking recall here.

Both directions are available:

```python
chunk_index.nodes_for_chunk(chunk_uid)     # entities in this passage
chunk_index.chunks_for_node("BRCA1")       # passages discussing this entity
chunk_index.neighbors(graph, ["BRCA1"], max_hops=2)   # {'HR': 1, 'PARPi': 2}
```

---

## Retrievers

All implement the LangChain `BaseRetriever` interface, so they compose with anything expecting one.

### `LiteratureRetriever`

Semantic similarity over embedded passages. Optional `score_threshold` filters weak matches.

### `KnowledgeGraphRetriever`

Structured lookup: finds graph nodes named in the query and returns the relations involving them, rendered as text.

```python
KnowledgeGraphRetriever(graph=kg, k=5).invoke("What does BRCA1 do?")
# "BRCA1 —ASSOCIATED_WITH→ breast cancer (confidence 0.95)"
```

### `HybridRetriever`

Interleaves both sources rather than concatenating, so a top-k answer sees both kinds of evidence even when one retriever returns far more results.

```python
HybridRetriever(literature_retriever=lit, kg_retriever=kg, k=8, literature_weight=0.6)
```

### `GraphExpansionRetriever` — the multi-hop one

Retrieves by similarity, resolves the seed passages to graph nodes, walks the graph, and returns the passages attached to nodes it reaches.

```python
from litkg.langchain_integration import GraphExpansionRetriever

retriever = GraphExpansionRetriever(
    vector_store=vector_store,
    graph=knowledge_graph,
    chunk_index=chunk_index,
    k=5, max_hops=2, expansion_limit=5,
)

docs = retriever.invoke("Why are BRCA1 tumours sensitive to olaparib?")
for d in docs:
    print(d.metadata["hop_distance"], d.metadata.get("via_entity"), d.page_content[:60])
```

Worked example. Given a graph `BRCA1 → homologous recombination → PARP inhibitors` and three passages:

| Passage | Mentions |
|---|---|
| "BRCA-1 mutations are common in hereditary breast cancer." | BRCA1 |
| "Homologous recombination is a DNA repair pathway." | HR |
| "Olaparib exploits deficiency in HR repair." | HR, PARPi |

Vector search on the query returns only the first. Graph expansion returns all three — including the olaparib passage, which contains **no query term at all**. That is the multi-hop payoff: evidence reachable by relationship, not by vocabulary.

---

## RAG

`BiomedicalRAGSystem`

```python
from litkg.langchain_integration import BiomedicalRAGSystem

rag = BiomedicalRAGSystem(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph,
    chunk_index=chunk_index,
    k=5,
    max_hops=2,          # 0 disables expansion
)

result = rag.answer("How does BRCA1 relate to PARP inhibitors?")
result["answer"]        # cites evidence as [1], [2], ...
result["sources"]       # each with metadata, including hop_distance
result["num_sources"]
```

The retriever is chosen from what you supply: graph expansion when vector store, graph, and chunk index are all present and `max_hops > 0`; hybrid when there is no chunk index; single-source otherwise.

Two behaviours worth relying on:

- **Generation is constrained to retrieved evidence.** The prompt instructs citation of every claim and forbids filling gaps from memory.
- **No evidence means no answer.** When retrieval returns nothing, the system says so and does not call the model at all, rather than producing an unsupported answer.

`batch_answer` keeps positional alignment on failure, marking failed entries with `error: True`.

---

## Agents

### `BiomedicalToolkit`

Wraps the pipeline as callable tools. Only the tools you wire up are exposed.

```python
from litkg.langchain_integration import BiomedicalToolkit

toolkit = BiomedicalToolkit(
    rag_system=rag,
    hypothesis_generator=generator,
    literature_validator=validator,
)
toolkit.tool_specs()          # [{'name': 'search_knowledge', ...}, ...]
toolkit.as_langchain_tools()  # LangChain Tool objects
```

> Tool functions return error text rather than raising. That is the LangChain convention — the agent reads the message and recovers, whereas an exception aborts the whole run. This is deliberate and marked in the source.

### `BiomedicalQueryAgent`

Conversational front end with memory and keyword routing.

```python
from litkg.langchain_integration import BiomedicalQueryAgent

agent = BiomedicalQueryAgent(toolkit=toolkit, max_history=10)

agent.chat("What is the role of BRCA1 in DNA repair?")     # -> search_knowledge
agent.chat("Propose a hypothesis about resistance")        # -> generate_hypothesis
agent.chat("Is it true that PARP inhibitors treat BRCA1 tumours?")  # -> validate_claim

agent.conversation_context()   # recent turns, for follow-ups
agent.reset()
```

### `HypothesisGenerationAgent` / `LiteratureValidationAgent`

Thin wrappers over the Phase 3 machinery, so the API and chat surfaces share one implementation.

```python
HypothesisGenerationAgent().propose("BRCA1 loss impairs homologous recombination", domain="oncology")
# {'hypothesis': '...', 'confidence': 0.5, 'testable_predictions': [...]}

LiteratureValidationAgent().validate_claim("BRCA1 mutations increase PARP inhibitor sensitivity")
# {'score': 0.84, 'supporting_papers': 6, 'contradicting_papers': 2, 'verdict': 'supported'}
```

Verdicts: `supported` (≥ 0.7), `contradicted` (≤ 0.3), `inconclusive` between.

---

## Entity extraction

`LLMEntityExtractor` extracts entities and relations with few-shot prompting and chain-of-thought, as an alternative to the Phase 1 NER models.

```python
from litkg.langchain_integration import LLMEntityExtractor

extractor = LLMEntityExtractor()
extractor.extract_entities_and_relations("BRCA1 mutations increase breast cancer risk.")
```

---

## Not implemented

Being explicit about what the package does *not* do:

- Chunks are linked to graph nodes by **alias matching**, not by running the LLM extractor over every chunk. Alias matching is fast and deterministic; it will miss entities absent from the graph's vocabulary.
- `KnowledgeGraphRetriever` matches query terms against node names by substring, without entity linking. A query saying "BRCA-1" will not match a node named `BRCA1` unless that surface form is a node alias.
- Agent routing is keyword-based, not LLM-based. It is predictable and free, but will mis-route unusual phrasings.

## Running it on real data

`RAGPipeline` is the wiring between Phase 1 output and the retrieval stack.
Before it existed the retrievers, chunk-to-graph index and agents were unit
tested but unreachable from real data: `make run-langchain` built its own
hardcoded documents and a bare FAISS index, so the graph-aware path was never
exercised outside tests.

```bash
make run-phase1                 # produces the documents and graph
make rag-coverage               # index stats, no LLM call
make rag Q="Why are BRCA1 tumours sensitive to olaparib?"
```

On the bundled corpus: 80 documents, 81 chunks, **78 of 81 chunks (96%) link to
the graph**, reaching 158 of 2971 nodes. Only linked chunks can seed graph
expansion, so that link rate is what decides whether multi-hop does anything.

### The hub problem

Expansion is only useful if it follows meaningful edges. It does not by
default, because the graph has generic hubs:

| node | degree | linked from |
|---|---|---|
| `CIVIC:DISEASE:DOID:162` ("cancer") | 429 | 29 of 81 chunks |

Walking through that node reaches 207 nodes in one hop and **824 in two** —
28% of the graph. Every oncology passage becomes a neighbour of every other,
and expansion returns unrelated cancer papers instead of following
`BRCA1 → homologous recombination → PARP inhibitors`.

`ChunkGraphIndex.neighbors` therefore caps traversal at
`DEFAULT_MAX_TRAVERSAL_DEGREE` (50). This is a traversal rule, not a filter: a
hub can still be *reached* and reported as evidence, it just cannot be walked
*through*. Ordinary low-degree paths are unaffected.

This is the same failure that `preferential_attachment` exposed in link
prediction — on this graph, anything that can route through popularity will.

### What it does and does not do

The generated answers are grounded and cited. Asked why BRCA1 tumours are
sensitive to olaparib, the system explains synthetic lethality and quotes the
TBCRC 048 response rates and the OlympiA disease-free survival result, citing
the passages it drew them from.

Retrieval is now measured — see below. The short version: vector retrieval is
strong, and graph expansion is not.

## Measuring retrieval

```bash
make build-queryset      # derive judged queries from CIVIC citations
make eval-retrieval SWEEP=1
```

### Where the judgements come from

There are no human relevance labels for this corpus, and using an LLM to judge
retrieval that feeds the same LLM is close to circular. CIVIC supplies
judgements instead: every evidence row cites a PubMed paper **and** states the
relationship it supports — a molecular profile, a disease, an evidence type.
For a question about that relationship, the cited papers are relevant by a
curator's judgement rather than ours.

`scripts/build_retrieval_queryset.py` groups evidence by
`(molecular profile, disease, evidence type)`, keeps groups with at least three
distinct cited papers, phrases a question per evidence type, and fetches the
cited abstracts. The bundled set is **57 queries over 228 papers**.

Evidence type drives the phrasing because it is what the paper was cited to
establish — asking "which therapies" of a prognostic paper would score relevant
papers as misses for a question they never answered.

### Results

| k | hops | P@k | R@k | MRR | nDCG@k | hit-rate |
|---|---|---|---|---|---|---|
| 5 | 0 | 0.547 [0.484, 0.611] | 0.657 | 0.803 | 0.716 | 0.947 |
| 5 | 1 | 0.551 [0.488, 0.614] | 0.660 | 0.803 | 0.719 | 0.947 |
| 10 | 0 | 0.363 [0.318, 0.414] | 0.815 [0.753, 0.875] | 0.808 | 0.770 | 0.982 |
| 10 | 1 | 0.363 [0.318, 0.414] | 0.815 | 0.808 | 0.770 | 0.982 |

Intervals are 95% bootstrap over queries.

**Vector retrieval works.** A relevant paper appears in the top 10 for 98% of
queries, and the first hit is usually at rank 1 or 2 (MRR 0.81).

**Graph expansion does not help.** Every hops setting scores the same, and the
hub-traversal cap makes no difference either. Broken down by where a passage
came from:

| passages | relevant | precision |
|---|---|---|
| hop 0 (vector) | 158 / 285 | **55.4%** |
| hop 1 (graph) | 13 / 285 | **4.6%** |

Twelve times less precise. This is not a plumbing failure — every chunk in this
corpus links to the graph (235/235, reaching 327 nodes). Expansion is working
as designed and what it reaches is mostly not relevant.

`PipelineConfig.max_hops` therefore defaults to **0**. Shipping expansion on by
default would dilute the evidence handed to the model on the strength of a
story rather than a measurement.

### The caveat that keeps this honest

**These judgements are biased against expansion by construction.** They mark
relevant only what CIVIC cited for one specific relationship — which is
precisely *not* the vocabulary-crossing evidence multi-hop retrieval exists to
reach. A genuinely useful expanded passage is scored as a miss.

So this measures that expansion does not help *on questions of this shape*. It
is not proof that expansion is useless. Establishing that would need judgements
built for multi-hop questions — ones whose answer requires connecting two
papers that share no terms — and that query set does not exist yet.

Judgements are also **incomplete** in the ordinary IR sense: an uncited paper
about the same gene may be relevant and is scored as a miss, so every number
here is a lower bound. And the queries are **templated**, so they are more
uniform than real user questions.
