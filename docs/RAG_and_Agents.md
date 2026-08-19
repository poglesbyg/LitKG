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

The judged query sets are tracked in the repository; the corpora and vector
stores they refer to are not, because they are rebuilt from the queries. On a
fresh clone, refetch the papers a query set names:

```bash
python scripts/build_retrieval_queryset.py --rebuild-corpus            # single-hop
python scripts/build_retrieval_queryset.py --rebuild-corpus --multihop
```

Regenerating the *queries* instead would silently change what a reported number
means, since PubMed content and CIVIC releases drift — the same reason the CIVIC
release is pinned rather than taken from nightly.

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

## Does multi-hop retrieval work?

The single-relationship query set could not answer this: it marks relevant only
what CIVIC cited for the relationship the query names, which is exactly what
dense retrieval already finds. Multi-hop exists to reach evidence sharing *no*
vocabulary with the question, so measuring it needs judgements built for that.

### A query set where vector search cannot win

`build_retrieval_queryset.py --multihop` constructs bridge queries. Each names
one molecular profile; the relevant papers are those cited for **different**
profiles evidenced in the same disease, reachable only along
`profile → disease → other profile`.

The filter that makes this honest: any candidate paper that mentions the
query's profile or gene is **dropped**, because dense retrieval could find it
directly and it would not be testing the hop. 81 papers were removed that way,
leaving **55 queries over 506 documents**.

### The paths exist; the ranking was the problem

Tracing the chain for all 55 queries:

| step | result |
|---|---|
| seed paper retrieved by vector search | **16/55** |
| seed paper links to graph nodes | 54/55 |
| bridge paper links to graph nodes | 55/55 |
| bridge reachable within 1–2 hops | **51–54/55** |
| bridge paper actually retrieved | 9/55 |

The graph holds a path to the answer for 54 of 55 queries. Almost none of them
were being retrieved. Two causes, both fixed:

**Expansion seeded only from retrieved passages**, making it a hostage to dense
retrieval — which surfaced a usable seed for just 16 of 55. Entities named in
the *question* are now resolved against the same alias index and seed the walk
directly.

**Candidates were truncated in walk order.** A one-hop walk reaches hundreds of
passages and only `expansion_limit` survive, so the cutoff was close to random
sampling. Widening the pool shows the recall was always there:

| candidate pool | bridge found |
|---|---|
| 5 | 21.8% |
| 50 | 61.8% |
| 200 | **85.5%** |

### The ranking signal that works, and the one that does not

Ranking graph candidates by **similarity to the query is self-defeating**, and
measurably so: it nullified expansion completely (hit-rate 0.200, identical to
no expansion). A passage reachable only through the graph is by definition one
that does not resemble the question, so similarity sorts exactly the wanted
passages to the bottom.

Inverse node degree — the Adamic-Adar intuition that fixed link prediction — is
also wrong here. The meaningful bridge between two variants is the disease both
are evidenced in, and disease nodes are precisely the high-degree ones that
weighting would penalise.

What survives is **shared graph context**: how many nodes a candidate shares
with the query's own entities, divided by the square root of its own node count
so that passages mentioning many entities do not win on breadth alone. Standalone
at top 5 that reaches **52.7%** against 21.8% for walk order.

### Where it actually lands

| configuration | hit-rate | P@10 | MRR |
|---|---|---|---|
| no expansion | 0.200 | 0.025 | 0.048 |
| expansion, similarity-ranked | 0.200 | 0.025 | 0.043 |
| expansion, context-ranked, concatenated | 0.236 | 0.031 | 0.047 |
| **expansion, context-ranked, rank-fused** | **0.327** | **0.049** | **0.095** |

### Fusing instead of allocating

Concatenating the two lists put every seed ahead of every expanded passage, so
dense retrieval spent half the returned slots whether or not it had found
anything. On these queries it usually had not.

The obvious fix — predict which questions are entity-anchored and reallocate —
was measured and rejected. Seed coverage of the query's own entities separates
the two query sets far too weakly to switch on: zero coverage fires on only 6 of
55 bridge queries, and the distributions overlap heavily. A classifier keyed on
phrasing would fit the templates these queries are generated from rather than
anything a real user would type.

Reciprocal rank fusion needs no such prediction. A passage ranked highly by
either route earns a place, one ranked well by both wins, and neither list can
crowd the other out by position alone. The constant is the published default and
is deliberately not tuned here, since tuning it on 55 queries would fit the
query set.

That takes hit-rate from 0.200 to **0.327**, with precision and MRR both roughly
doubled.

### The trade-off, stated plainly

Fusion costs single-hop precision when expansion is switched on:

| single-relationship set | P@10 | R@10 | nDCG@10 | hit-rate |
|---|---|---|---|---|
| `max_hops=0` (the default) | 0.370 | 0.826 | 0.779 | 0.982 |
| `max_hops=1`, fused | 0.304 | 0.712 | 0.632 | 0.982 |

`max_hops` defaults to **0**, so nothing changes for single-relationship
questions unless expansion is asked for.

**Why recall drops, which is not obvious.** Expansion only ever *adds*
candidates, so a lower recall looks arithmetically impossible. The cause is that
`k` counts *seed* passages, not results: with expansion on, the retriever
returns up to `k + expansion_limit` documents. A consumer that truncates back to
`k` — as the evaluation does, to compare configurations at equal budget — is
choosing which source to cut, and fusion decides that by rank. Under RRF the
expanded passage at rank n ties the seed at rank n, so five expanded candidates
displace exactly the last five seeds. Measured: at k=10 every one of the 57
queries loses five seed papers from the evaluated window, 282 in total.

An earlier version of this document attributed the drop partly to P@k dividing
by k while the `max_hops=0` row returns fewer passages. That is true of P@k and
irrelevant to recall, and it was the wrong explanation.

**No evidence is lost on the answer path.** `BiomedicalRAGSystem.answer`
consumes every document the retriever returns without truncating, so fusion
changes the order of the prompt and nothing else. The recall figures above
describe a fixed-budget comparison, not what the RAG system hands the model.

Turning expansion on is a measured choice: it buys reach on questions whose
answers are not lexically present, and costs ranking position for the trailing
seeds on questions where dense retrieval was already right.

The trade-off is a step, not a slope. Weighting seeds even slightly above
expanded candidates (1.2x is enough to break every RRF tie) restores
single-relationship recall to 0.826 exactly, and returns bridge hit-rate to
0.236. There is no setting between the two: either expanded candidates can tie
seeds or they cannot. The default lets them tie, because expansion is opt-in and
its purpose is the bridge case.

### Still not solved

A relevant bridge paper reaches the top ten for a third of these queries against
98% for single-relationship ones, and the candidate pool holds the answer for
85%. Closing that remaining gap needs a better ranking signal within the graph
candidates, not more of them.
