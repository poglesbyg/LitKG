"""
Tests for the LangChain integration package: retrievers, RAG, and agents.

These exercise the real classes with injected stubs rather than live network
or model calls, so they stay fast and deterministic.
"""

import networkx as nx
import pytest
from unittest.mock import Mock

from langchain_core.documents import Document

from litkg.langchain_integration import (
    BiomedicalRAGSystem,
    LiteratureRetriever,
    KnowledgeGraphRetriever,
    HybridRetriever,
    BiomedicalToolkit,
    BiomedicalQueryAgent,
    HypothesisGenerationAgent,
    LiteratureValidationAgent,
)
from litkg.llm_integration.unified_llm_interface import LLMResponse, LLMProvider


@pytest.fixture
def biomedical_graph():
    """A small knowledge graph with typed, weighted edges."""
    graph = nx.Graph()
    graph.add_edge("BRCA1", "breast cancer", type="ASSOCIATED_WITH", weight=0.95)
    graph.add_edge("BRCA1", "PARP inhibitors", type="SENSITIZES_TO", weight=0.88)
    graph.add_edge("TP53", "breast cancer", type="ASSOCIATED_WITH", weight=0.80)
    return graph


@pytest.fixture
def fake_vector_store():
    """A vector store returning k placeholder literature passages."""
    store = Mock()
    store.similarity_search.side_effect = lambda query, k=5: [
        Document(page_content=f"Passage {i} about {query}", metadata={"pmid": f"200{i}"})
        for i in range(k)
    ]
    return store


@pytest.fixture
def stub_llm_manager():
    """An LLM manager that echoes a fixed answer without calling a model."""
    manager = Mock()
    manager.process_biomedical_task.return_value = LLMResponse(
        content="BRCA1 loss sensitizes cells to PARP inhibition [1].",
        provider=LLMProvider.OLLAMA,
        model="qwen3:8b",
    )
    return manager


class TestRetrievers:
    """Retrieval components."""

    def test_knowledge_graph_retriever_finds_named_entities(self, biomedical_graph):
        retriever = KnowledgeGraphRetriever(graph=biomedical_graph, k=5)

        docs = retriever.invoke("What does BRCA1 do?")

        assert docs
        assert all(d.metadata["source_type"] == "knowledge_graph" for d in docs)
        # Only relations involving the named entity are returned
        assert all("BRCA1" in (d.metadata["head"], d.metadata["tail"]) for d in docs)

    def test_knowledge_graph_retriever_ignores_unmentioned_entities(self, biomedical_graph):
        retriever = KnowledgeGraphRetriever(graph=biomedical_graph, k=5)

        assert retriever.invoke("What causes influenza?") == []

    def test_knowledge_graph_retriever_respects_k(self, biomedical_graph):
        retriever = KnowledgeGraphRetriever(graph=biomedical_graph, k=1)

        assert len(retriever.invoke("Tell me about BRCA1")) == 1

    def test_literature_retriever_returns_k_documents(self, fake_vector_store):
        retriever = LiteratureRetriever(vector_store=fake_vector_store, k=3)

        docs = retriever.invoke("BRCA1")

        assert len(docs) == 3
        assert all(d.metadata["source_type"] == "literature" for d in docs)

    def test_literature_retriever_without_store_is_empty(self):
        assert LiteratureRetriever(vector_store=None, k=3).invoke("anything") == []

    def test_hybrid_retriever_interleaves_both_sources(
        self, biomedical_graph, fake_vector_store
    ):
        hybrid = HybridRetriever(
            literature_retriever=LiteratureRetriever(vector_store=fake_vector_store, k=4),
            kg_retriever=KnowledgeGraphRetriever(graph=biomedical_graph, k=4),
            k=6,
        )

        docs = hybrid.invoke("What does BRCA1 do?")
        kinds = [d.metadata.get("source_type") for d in docs]

        # Both sources are represented, and neither is pushed off the end
        assert "literature" in kinds
        assert "knowledge_graph" in kinds
        assert kinds[0] != kinds[1]

    def test_hybrid_retriever_falls_back_to_single_source(self, fake_vector_store):
        hybrid = HybridRetriever(
            literature_retriever=LiteratureRetriever(vector_store=fake_vector_store, k=4),
            kg_retriever=KnowledgeGraphRetriever(graph=nx.Graph(), k=4),
            k=4,
        )

        docs = hybrid.invoke("BRCA1")

        assert len(docs) == 4
        assert all(d.metadata["source_type"] == "literature" for d in docs)


class TestChunkGraphLinking:
    """Linking chunks to graph nodes, and multi-hop retrieval over that link."""

    @pytest.fixture
    def multihop_setup(self):
        """A graph and passages where the answer needs two hops."""
        from litkg.langchain_integration import EntityAliasIndex, ChunkGraphIndex

        graph = nx.Graph()
        graph.add_node("BRCA1", name="BRCA1", synonyms=["BRCA-1", "breast cancer 1"])
        graph.add_node("HR", name="homologous recombination", synonyms=["HR repair"])
        graph.add_node("PARPi", name="PARP inhibitors", synonyms=["olaparib"])
        graph.add_edge("BRCA1", "HR", type="INVOLVED_IN")
        graph.add_edge("HR", "PARPi", type="TARGETED_BY")

        chunks = [
            Document(
                page_content="BRCA-1 mutations are common in hereditary breast cancer.",
                metadata={"pmid": "1", "chunk_id": 0},
            ),
            Document(
                page_content="Homologous recombination is a DNA repair pathway.",
                metadata={"pmid": "2", "chunk_id": 0},
            ),
            # Mentions neither BRCA1 nor any likely query term
            Document(
                page_content="Olaparib exploits deficiency in HR repair.",
                metadata={"pmid": "3", "chunk_id": 0},
            ),
        ]

        index = ChunkGraphIndex(EntityAliasIndex().add_from_graph(graph))
        index.index_chunks(chunks)
        return graph, chunks, index

    def test_alias_index_resolves_synonyms_to_canonical_node(self):
        from litkg.langchain_integration import EntityAliasIndex

        index = EntityAliasIndex()
        index.add_entity("BRCA1", "BRCA1", synonyms=["BRCA-1", "breast cancer 1"])

        # Every surface form resolves to the same canonical node
        for surface in ("BRCA1", "BRCA-1", "breast cancer 1"):
            found = index.find_in_text(f"We studied {surface} in this cohort.")
            assert [node for node, _ in found] == ["BRCA1"]

    def test_alias_index_respects_word_boundaries(self):
        """TP53 must not match inside TP53BP1, a different gene."""
        from litkg.langchain_integration import EntityAliasIndex

        index = EntityAliasIndex()
        index.add_entity("TP53", "TP53")

        assert index.find_in_text("TP53BP1 was unchanged") == []
        assert index.find_in_text("TP53 was mutated") == [("TP53", "tp53")]

    def test_alias_index_prefers_longest_match(self):
        """A specific alias wins over a shorter one nested inside it."""
        from litkg.langchain_integration import EntityAliasIndex

        index = EntityAliasIndex()
        index.add_entity("BC", "cancer")
        index.add_entity("BRCA1", "breast cancer 1")

        found = index.find_in_text("breast cancer 1 was sequenced")

        assert ("BRCA1", "breast cancer 1") in found

    def test_chunks_are_annotated_with_entities(self, multihop_setup):
        _, chunks, _ = multihop_setup

        assert chunks[0].metadata["entity_ids"] == ["BRCA1"]
        assert "PARPi" in chunks[2].metadata["entity_ids"]
        # A stable id survives a round trip through a vector store
        assert all(c.metadata["chunk_uid"] for c in chunks)

    def test_index_is_bidirectional(self, multihop_setup):
        _, chunks, index = multihop_setup

        uid = chunks[0].metadata["chunk_uid"]
        assert index.nodes_for_chunk(uid) == ["BRCA1"]
        assert chunks[0] in index.chunks_for_node("BRCA1")

    def test_neighbors_walk_respects_hop_limit(self, multihop_setup):
        graph, _, index = multihop_setup

        one_hop = index.neighbors(graph, ["BRCA1"], max_hops=1)
        two_hop = index.neighbors(graph, ["BRCA1"], max_hops=2)

        assert one_hop == {"HR": 1}
        assert two_hop == {"HR": 1, "PARPi": 2}

    def test_graph_expansion_surfaces_unmatched_passage(self, multihop_setup):
        """The payoff: evidence sharing no vocabulary with the query."""
        from litkg.langchain_integration import GraphExpansionRetriever

        graph, chunks, index = multihop_setup

        class SeedOnlyStore:
            def similarity_search(self, query, k=5):
                return [chunks[0]]

        retriever = GraphExpansionRetriever(
            vector_store=SeedOnlyStore(), graph=graph, chunk_index=index,
            k=1, max_hops=2,
        )

        docs = retriever.invoke("Why are BRCA1 tumours sensitive to olaparib?")
        contents = [d.page_content for d in docs]

        # Vector search alone returns one passage; expansion reaches the rest
        assert len(docs) > 1
        assert any("Olaparib" in c for c in contents)
        assert docs[0].metadata["hop_distance"] == 0
        assert all(d.metadata["hop_distance"] > 0 for d in docs[1:])
        assert all(d.metadata["via_entity"] for d in docs[1:])

    def test_expansion_is_skipped_without_graph_links(self):
        """Unlinked seeds return unchanged rather than erroring."""
        from litkg.langchain_integration import GraphExpansionRetriever, ChunkGraphIndex

        unlinked = Document(page_content="Unrelated text.", metadata={"chunk_uid": "x"})

        class Store:
            def similarity_search(self, query, k=5):
                return [unlinked]

        retriever = GraphExpansionRetriever(
            vector_store=Store(), graph=nx.Graph(), chunk_index=ChunkGraphIndex(),
            k=1, max_hops=2,
        )

        assert len(retriever.invoke("anything")) == 1


class TestBiomedicalRAGSystem:
    """Retrieval-augmented answering."""

    def test_answer_is_grounded_in_retrieved_sources(
        self, biomedical_graph, stub_llm_manager
    ):
        rag = BiomedicalRAGSystem(
            knowledge_graph=biomedical_graph, llm_manager=stub_llm_manager, k=4
        )

        result = rag.answer("How does BRCA1 relate to PARP inhibitors?")

        assert result["num_sources"] > 0
        assert result["sources"]
        assert "PARP" in result["answer"]

        # The retrieved evidence must actually reach the prompt
        prompt = stub_llm_manager.process_biomedical_task.call_args.kwargs["input_data"]
        assert "SENSITIZES_TO" in prompt
        assert "[1]" in prompt

    def test_refuses_when_nothing_retrieved(self, stub_llm_manager):
        rag = BiomedicalRAGSystem(llm_manager=stub_llm_manager, k=3)

        result = rag.answer("What causes cancer?")

        assert result["num_sources"] == 0
        assert "No evidence" in result["answer"]
        # No model call is made when there is nothing to ground an answer in
        stub_llm_manager.process_biomedical_task.assert_not_called()

    def test_format_context_numbers_evidence(self):
        docs = [
            Document(page_content="First fact", metadata={"pmid": "111"}),
            Document(page_content="Second fact", metadata={"source_type": "knowledge_graph"}),
        ]

        context = BiomedicalRAGSystem.format_context(docs)

        assert "[1]" in context and "[2]" in context
        assert "111" in context

    def test_batch_answer_keeps_alignment_on_failure(
        self, biomedical_graph, stub_llm_manager
    ):
        stub_llm_manager.process_biomedical_task.side_effect = [
            LLMResponse(content="ok", provider=LLMProvider.OLLAMA, model="qwen3:8b"),
            RuntimeError("model unavailable"),
        ]
        rag = BiomedicalRAGSystem(
            knowledge_graph=biomedical_graph, llm_manager=stub_llm_manager, k=2
        )

        results = rag.batch_answer(["BRCA1 question", "BRCA1 second question"])

        assert len(results) == 2
        assert results[1]["error"] is True


class TestBiomedicalAgents:
    """Agent surfaces over the pipeline."""

    def test_toolkit_only_exposes_configured_tools(self):
        assert BiomedicalToolkit().tool_specs() == []

        toolkit = BiomedicalToolkit(rag_system=Mock())
        assert [s["name"] for s in toolkit.tool_specs()] == ["search_knowledge"]

    def test_toolkit_converts_to_langchain_tools(self):
        toolkit = BiomedicalToolkit(rag_system=Mock())

        tools = toolkit.as_langchain_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_knowledge"

    def test_toolkit_reports_errors_without_raising(self):
        """Tool errors are returned as text, per the LangChain tool convention."""
        rag = Mock()
        rag.answer.side_effect = RuntimeError("retrieval exploded")

        result = BiomedicalToolkit(rag_system=rag).search_knowledge("query")

        assert "Error searching knowledge" in result

    def test_agent_routes_to_hypothesis_tool(self):
        generator = Mock()
        generator.generate_hypothesis.return_value = Mock(
            hypothesis_text="A testable claim", confidence_score=0.75
        )
        agent = BiomedicalQueryAgent(
            toolkit=BiomedicalToolkit(hypothesis_generator=generator),
            llm_manager=Mock(),
        )

        result = agent.chat("Propose a hypothesis about BRCA1")

        assert result["tool_used"] == "generate_hypothesis"
        assert "A testable claim" in result["response"]

    def test_agent_records_conversation_history(self, stub_llm_manager):
        agent = BiomedicalQueryAgent(llm_manager=stub_llm_manager)

        agent.chat("First question")
        agent.chat("Second question")

        # Two exchanges, user and assistant each
        assert len(agent.history) == 4
        assert "First question" in agent.conversation_context()

        agent.reset()
        assert agent.history == []

    def test_validation_agent_classifies_verdict(self):
        validator = Mock()
        validator.validate.return_value = Mock(
            score=0.85, details={"supporting_papers": 8, "contradicting_papers": 1}
        )

        result = LiteratureValidationAgent(validator=validator).validate_claim(
            "BRCA1 mutations increase PARP inhibitor sensitivity"
        )

        assert result["verdict"] == "supported"
        assert result["supporting_papers"] == 8

    def test_validation_agent_flags_contradicted_claims(self):
        validator = Mock()
        validator.validate.return_value = Mock(
            score=0.15, details={"supporting_papers": 1, "contradicting_papers": 9}
        )

        result = LiteratureValidationAgent(validator=validator).validate_claim("x")

        assert result["verdict"] == "contradicted"

    def test_hypothesis_agent_proposes_from_context(self):
        agent = HypothesisGenerationAgent()

        result = agent.propose(
            "BRCA1 loss impairs homologous recombination", domain="oncology"
        )

        assert result["hypothesis"]
        assert result["domain"] == "oncology"
        assert 0 <= result["confidence"] <= 1


class TestBiomedicalChunking:
    """Chunk sizing, overlap, sentence boundaries, and section provenance."""

    @pytest.fixture
    def splitter(self):
        from litkg.langchain_integration import BiomedicalTextSplitter
        return BiomedicalTextSplitter(
            chunk_size=40, chunk_overlap=15, length_unit="tokens"
        )

    def test_overlap_is_actually_applied(self, splitter):
        """chunk_overlap was previously stored and never used, so overlap was 0."""
        text = " ".join(
            f"Sentence number {i} describes a distinct finding about genes."
            for i in range(1, 13)
        )

        chunks = splitter._split_by_sentences(text)

        assert len(chunks) > 1
        # Every adjacent pair shares boundary text
        for earlier, later in zip(chunks, chunks[1:]):
            earlier_words = set(earlier.split())
            assert any(word in earlier_words for word in later.split()[:6])

    def test_zero_overlap_is_honored(self):
        from litkg.langchain_integration import BiomedicalTextSplitter

        splitter = BiomedicalTextSplitter(
            chunk_size=40, chunk_overlap=0, length_unit="tokens"
        )
        text = " ".join(
            f"Sentence number {i} describes a distinct finding about genes."
            for i in range(1, 13)
        )

        chunks = splitter._split_by_sentences(text)
        rejoined = " ".join(chunks).split()

        # With no overlap, no sentence index is duplicated
        assert len(rejoined) == len(text.split())

    def test_overlap_must_be_smaller_than_chunk_size(self):
        """An overlap >= chunk_size cannot make progress."""
        from litkg.langchain_integration import BiomedicalTextSplitter

        with pytest.raises(ValueError, match="chunk_overlap"):
            BiomedicalTextSplitter(chunk_size=100, chunk_overlap=100)

    def test_chunk_size_capped_to_model_window(self):
        """Chunks longer than the embedding window get silently truncated."""
        from litkg.langchain_integration import BiomedicalTextSplitter

        splitter = BiomedicalTextSplitter(
            chunk_size=2000, chunk_overlap=10,
            length_unit="tokens", model_max_tokens=512,
        )

        assert splitter.chunk_size == 512

    def test_rejects_unknown_length_unit(self):
        from litkg.langchain_integration import BiomedicalTextSplitter

        with pytest.raises(ValueError, match="length_unit"):
            BiomedicalTextSplitter(length_unit="furlongs")

    def test_sentence_splitting_survives_abbreviations(self, splitter):
        """'et al.' and 'Fig. 2' and 'p < 0.05' must not end sentences."""
        text = "BRCA1 was studied by Smith et al. in Fig. 2 (p < 0.05). Results were clear."

        sentences = splitter.split_sentences(text)

        # The first clause stays whole despite three internal periods
        assert any("et al." in s and "Fig. 2" in s and "0.05" in s for s in sentences)

    def test_regex_fallback_protects_abbreviations(self, splitter):
        """The non-spaCy path must handle abbreviations too."""
        text = "Shown by Jones et al. in Fig. 1 (p < 0.01). Then we replicated it."

        sentences = splitter._split_sentences_regex(text)

        assert any("et al." in s and "Fig. 1" in s and "0.01" in s for s in sentences)

    def test_section_label_travels_with_chunk(self, splitter):
        """Results vs Introduction is evidential weight, not decoration."""
        paper = (
            "Introduction\nBRCA1 is a tumour suppressor gene.\n"
            "Results\nWe observed synthetic lethality with PARP inhibition."
        )

        labeled = splitter.split_text_with_sections(paper)
        sections = {section for _, section in labeled}

        assert sections == {"Introduction", "Results"}
        # Section bodies are preserved, not just the headers
        results_text = " ".join(c for c, s in labeled if s == "Results")
        assert "synthetic lethality" in results_text

    def test_text_before_any_header_is_kept(self, splitter):
        """Preamble must not be dropped or misattributed to the first section."""
        paper = "Some preamble text here.\nResults\nA finding."

        labeled = splitter.split_text_with_sections(paper)

        assert any(section is None for _, section in labeled)
        assert any("preamble" in chunk for chunk, _ in labeled)


class TestRAGPipelineWiring:
    """
    The retrievers, chunk index and agents were unit tested but nothing joined
    them to Phase 1 output, so the graph-aware path was never exercised outside
    tests. These cover the wiring itself.
    """

    def test_missing_phase1_output_is_a_clear_error(self, tmp_path):
        from litkg.langchain_integration import PipelineConfig, RAGPipeline
        config = PipelineConfig(
            documents_path=tmp_path / "absent.json",
            graph_path=tmp_path / "absent.gpickle",
            vector_store_path=tmp_path / "store",
        )
        with pytest.raises(FileNotFoundError, match="run-phase1"):
            RAGPipeline(config).build()

    def test_documents_load_from_each_shape_phase1_writes(self, tmp_path):
        import json
        from litkg.langchain_integration.pipeline import load_documents

        record = [{"pmid": "1", "title": "T", "abstract": "A"}]
        for payload in (record, {"documents": record}, {"results": record}):
            path = tmp_path / "docs.json"
            path.write_text(json.dumps(payload))
            assert len(load_documents(path)) == 1

    def test_chunks_carry_a_stable_chunk_id(self, tmp_path):
        """
        ChunkGraphIndex keys on chunk_id. Without a stable one the chunk-to-node
        mapping cannot be joined back to retrieval, and graph expansion silently
        returns nothing.
        """
        from langchain_core.documents import Document
        from litkg.langchain_integration import PipelineConfig, RAGPipeline

        pipeline = RAGPipeline(PipelineConfig(vector_store_path=tmp_path / "s"))
        chunks = pipeline._chunk([
            Document(page_content="BRCA1 repairs DNA. " * 40, metadata={"pmid": "99"})
        ])
        assert chunks
        # chunk_id is the position; ChunkGraphIndex composes the stable
        # identifier as "{pmid}:{chunk_id}" and writes it back as chunk_uid.
        # Including the pmid here as well produced keys like "99:99:0".
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(i.isdigit() for i in ids)

        from litkg.langchain_integration import ChunkGraphIndex, EntityAliasIndex
        ChunkGraphIndex(EntityAliasIndex()).index_chunks(chunks)
        uids = [c.metadata["chunk_uid"] for c in chunks]
        assert len(uids) == len(set(uids))
        assert all(u.startswith("99:") and u.count(":") == 1 for u in uids)


class TestHubTraversalCap:
    """
    Expansion through generic hubs makes every oncology passage a neighbour of
    every other. On the CIVIC graph DOID:162 ("cancer") has degree 429 and is
    reached from 29 of 81 chunks; two hops through it reach 824 nodes.
    """

    @pytest.fixture
    def hub_graph(self):
        import networkx as nx
        g = nx.Graph()
        for i in range(200):
            g.add_edge("hub", f"n{i}")
        g.add_edge("a", "hub")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        return g

    def test_expansion_does_not_walk_through_a_hub(self, hub_graph):
        from litkg.langchain_integration import ChunkGraphIndex, EntityAliasIndex
        index = ChunkGraphIndex(EntityAliasIndex())
        reached = index.neighbors(hub_graph, ["a"], max_hops=2, max_traversal_degree=50)
        assert "hub" in reached, "a hub should still be reachable as evidence"
        assert "n0" not in reached, "but must not be walked through"

    def test_uncapped_expansion_explodes(self, hub_graph):
        from litkg.langchain_integration import ChunkGraphIndex, EntityAliasIndex
        index = ChunkGraphIndex(EntityAliasIndex())
        uncapped = index.neighbors(hub_graph, ["a"], max_hops=2, max_traversal_degree=0)
        capped = index.neighbors(hub_graph, ["a"], max_hops=2, max_traversal_degree=50)
        assert len(uncapped) > 10 * len(capped)

    def test_low_degree_paths_are_still_followed(self, hub_graph):
        """The cap must not block the ordinary two-hop path a -> b -> c."""
        from litkg.langchain_integration import ChunkGraphIndex, EntityAliasIndex
        index = ChunkGraphIndex(EntityAliasIndex())
        reached = index.neighbors(hub_graph, ["a"], max_hops=2, max_traversal_degree=50)
        assert reached.get("b") == 1
        assert reached.get("c") == 2

    def test_cap_defaults_on(self):
        from litkg.langchain_integration import ChunkGraphIndex
        assert ChunkGraphIndex.DEFAULT_MAX_TRAVERSAL_DEGREE > 0


class TestAnswerSourceShape:
    def test_sources_flatten_metadata_rather_than_nesting_it(self):
        """
        `answer()` returns {"content": ..., **document.metadata}. Consumers that
        expect a nested "metadata" key read nothing and render every source as
        "hop 0, pmid ?" -- which is what the CLI did before this was pinned.
        """
        from langchain_core.documents import Document
        from litkg.langchain_integration.rag_system import BiomedicalRAGSystem

        class StubRetriever:
            def invoke(self, *a, **k):
                return [Document(page_content="Evidence.",
                                 metadata={"pmid": "123", "hop_distance": 1})]

        class StubLLM:
            def process_biomedical_task(self, **kwargs):
                return type("R", (), {"content": "Answer [1].", "model": "stub"})()

        system = BiomedicalRAGSystem(retriever=StubRetriever(), llm_manager=StubLLM())
        source = system.answer("q")["sources"][0]
        assert source["pmid"] == "123"
        assert source["hop_distance"] == 1
        assert "metadata" not in source


class TestMultiHopExpansionRanking:
    """
    Bridge evidence is by definition what the query does not resemble, so the
    obvious ranking signal is the wrong one. Measured on 55 bridge queries:
    ranking graph candidates by similarity to the query nullified expansion
    entirely (hit-rate 0.200, identical to no expansion), while ranking by
    shared graph context reached 0.236 and, standalone at top-5, 52.7% against
    21.8% for walk order.
    """

    def _document(self, uid, nodes, text="passage text here"):
        from langchain_core.documents import Document
        return Document(page_content=text,
                        metadata={"chunk_uid": uid, "entity_ids": list(nodes),
                                  "pmid": uid})

    def test_shared_graph_context_outranks_breadth(self):
        """
        A passage mentioning a great many entities must not win on breadth
        alone -- the same popularity control degree-matched negatives apply in
        link prediction.
        """
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        retriever = GraphExpansionRetriever()
        focused = self._document("focused", ["A", "B"])
        broad = self._document("broad", ["A"] + [f"x{i}" for i in range(40)])
        order = retriever._rank_candidates(
            {"focused": focused, "broad": broad}, anchor={"A", "B"}
        )
        assert order[0] == "focused"

    def test_candidates_without_graph_context_rank_last(self):
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        retriever = GraphExpansionRetriever()
        linked = self._document("linked", ["A"])
        unlinked = self._document("unlinked", [])
        order = retriever._rank_candidates(
            {"unlinked": unlinked, "linked": linked}, anchor={"A"}
        )
        assert order[0] == "linked"

    def test_ranking_does_not_consult_the_query_text(self):
        """
        Pinned deliberately: a passage reachable only through the graph does not
        resemble the question, so any query-similarity term would push exactly
        the wanted passages down.
        """
        import inspect
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        source = inspect.getsource(GraphExpansionRetriever._rank_candidates)
        assert "similarity_search" not in source

    def test_query_entities_seed_the_walk(self):
        """
        Seeding only from retrieved passages made expansion a hostage to dense
        retrieval: the graph held a path for 54 of 55 bridge queries but vector
        search surfaced a usable seed for only 16.
        """
        from litkg.langchain_integration import EntityAliasIndex, ChunkGraphIndex
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        import networkx as nx

        graph = nx.Graph()
        graph.add_node("BRAF", name="BRAF")
        alias = EntityAliasIndex().add_from_graph(graph)
        retriever = GraphExpansionRetriever(chunk_index=ChunkGraphIndex(alias))
        assert "BRAF" in retriever._query_nodes("A tumour carrying BRAF was seen")

    def test_query_seeding_is_optional(self):
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        assert GraphExpansionRetriever().seed_from_query is True


class TestRankFusion:
    """
    Concatenating seeds ahead of expanded passages gave dense retrieval half the
    returned slots whether or not it had found anything. On bridge queries it
    usually had not, and those wasted slots were the difference between 85%
    recall in the candidate pool and 24% in the top ten.
    """

    def _doc(self, uid):
        from langchain_core.documents import Document
        return Document(page_content=uid, metadata={"chunk_uid": uid, "pmid": uid})

    def test_a_strong_expanded_passage_can_outrank_a_weak_seed(self):
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        retriever = GraphExpansionRetriever()
        seeds = [self._doc(f"s{i}") for i in range(5)]
        expanded = [self._doc(f"e{i}") for i in range(5)]
        order = [d.metadata["chunk_uid"] for d in retriever._fuse(seeds, expanded)]
        assert order[1] == "e0", "the top expanded passage must beat the second seed"
        assert order[0] == "s0", "and the top seed still leads"

    def test_a_passage_found_by_both_routes_wins(self):
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        retriever = GraphExpansionRetriever()
        shared = self._doc("shared")
        seeds = [self._doc("s0"), shared]
        expanded = [self._doc("e0"), shared]
        order = [d.metadata["chunk_uid"] for d in retriever._fuse(seeds, expanded)]
        assert order[0] == "shared"

    def test_every_passage_survives_fusion(self):
        """Fusion reorders; it must not drop evidence."""
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        retriever = GraphExpansionRetriever()
        seeds = [self._doc(f"s{i}") for i in range(3)]
        expanded = [self._doc(f"e{i}") for i in range(4)]
        assert len(retriever._fuse(seeds, expanded)) == 7

    def test_fusion_can_be_disabled(self):
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        assert GraphExpansionRetriever().fuse_results is True
        assert GraphExpansionRetriever(fuse_results=False).fuse_results is False

    def test_fusion_constant_is_the_published_default(self):
        """
        Tuning it on 55 queries would fit the query set rather than improve
        retrieval, so it stays at the value from the original RRF paper.
        """
        from litkg.langchain_integration.rag_system import GraphExpansionRetriever
        assert GraphExpansionRetriever().fusion_k == 60
