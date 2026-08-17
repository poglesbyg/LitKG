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
