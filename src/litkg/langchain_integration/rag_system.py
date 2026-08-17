"""
Retrieval-Augmented Generation for biomedical questions.

Answers questions by retrieving evidence first and generating from it, so every
claim traces back to a document rather than to model memory.

Retrievers:
- LiteratureRetriever: semantic search over an embedded literature corpus
- KnowledgeGraphRetriever: structured lookup over a knowledge graph
- HybridRetriever: merges both, since literature and curated graphs answer
  different halves of most biomedical questions

BiomedicalRAGSystem ties retrieval to an LLM and returns answers with the
sources they were drawn from.
"""

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ..utils.logging import LoggerMixin


class LiteratureRetriever(BaseRetriever):
    """
    Retrieve literature passages by semantic similarity.

    Wraps a LangChain vector store so the rest of the system depends on the
    retriever interface rather than on which store is configured.
    """

    vector_store: Any = None
    k: int = 5
    score_threshold: Optional[float] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Return up to k literature passages relevant to the query."""
        if self.vector_store is None:
            return []

        # Prefer the scored API so the threshold can be applied; not every
        # store implements it.
        if self.score_threshold is not None and hasattr(
            self.vector_store, "similarity_search_with_score"
        ):
            scored = self.vector_store.similarity_search_with_score(query, k=self.k)
            documents = []
            for document, score in scored:
                # Stores differ on whether score is similarity or distance;
                # treat smaller-is-better only when scores exceed 1.
                similarity = score if score <= 1.0 else 1.0 / (1.0 + score)
                if similarity >= self.score_threshold:
                    document.metadata.setdefault("retrieval_score", float(similarity))
                    documents.append(document)
            return documents

        documents = self.vector_store.similarity_search(query, k=self.k)
        for document in documents:
            document.metadata.setdefault("source_type", "literature")
        return documents


class KnowledgeGraphRetriever(BaseRetriever):
    """
    Retrieve facts from a knowledge graph by entity mention.

    Where the literature retriever finds passages that read like the query,
    this finds curated relations involving the entities named in it -- the
    structured half of the evidence a biomedical answer needs.
    """

    graph: Any = None
    k: int = 5
    max_hops: int = 1

    def _matching_nodes(self, query: str) -> List[str]:
        """Find graph nodes mentioned in the query."""
        if self.graph is None:
            return []

        lowered = query.lower()
        return [
            node for node in self.graph.nodes()
            if str(node).lower().replace("_", " ") in lowered
            or str(node).lower() in lowered
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Return graph relations involving entities named in the query."""
        matched = self._matching_nodes(query)
        if not matched:
            return []

        documents: List[Document] = []
        seen_edges = set()

        for node in matched:
            for neighbor in self.graph.neighbors(node):
                edge_key = frozenset((str(node), str(neighbor)))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                edge_data = self.graph.get_edge_data(node, neighbor) or {}
                relation = edge_data.get("type", edge_data.get("relation", "RELATED_TO"))
                weight = edge_data.get("weight")

                text = f"{node} —{relation}→ {neighbor}"
                if weight is not None:
                    text += f" (confidence {weight})"

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source_type": "knowledge_graph",
                        "head": str(node),
                        "tail": str(neighbor),
                        "relation": relation,
                        **({"weight": weight} if weight is not None else {}),
                    },
                ))

                if len(documents) >= self.k:
                    return documents

        return documents


class HybridRetriever(BaseRetriever):
    """
    Merge literature and knowledge graph evidence into one ranked list.

    Curated graph relations are interleaved with literature passages rather
    than concatenated, so an answer built from the top-k sees both kinds of
    evidence even when one retriever returns more results than the other.
    """

    literature_retriever: Optional[Any] = None
    kg_retriever: Optional[Any] = None
    k: int = 8
    literature_weight: float = 0.6

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Return an interleaved list of literature and graph evidence."""
        literature_docs: List[Document] = []
        kg_docs: List[Document] = []

        if self.literature_retriever is not None:
            literature_docs = self.literature_retriever.invoke(query)
        if self.kg_retriever is not None:
            kg_docs = self.kg_retriever.invoke(query)

        if not literature_docs:
            return kg_docs[: self.k]
        if not kg_docs:
            return literature_docs[: self.k]

        # Split the budget by weight, then interleave so neither source is
        # crowded out of the top of the list.
        lit_budget = max(1, round(self.k * self.literature_weight))
        kg_budget = max(1, self.k - lit_budget)

        merged: List[Document] = []
        lit_iter = iter(literature_docs[:lit_budget])
        kg_iter = iter(kg_docs[:kg_budget])

        for lit_doc, kg_doc in zip(lit_iter, kg_iter):
            merged.extend([lit_doc, kg_doc])
        merged.extend(lit_iter)
        merged.extend(kg_iter)

        return merged[: self.k]


class BiomedicalRAGSystem(LoggerMixin):
    """
    Question answering grounded in retrieved biomedical evidence.

    Retrieves first, then generates strictly from what was retrieved, and
    returns the sources alongside the answer so a claim can be checked.
    """

    ANSWER_TEMPLATE = """You are a biomedical research assistant. Answer the question using ONLY the evidence below.

Rules:
- Cite evidence by its [n] number for every claim you make.
- If the evidence does not answer the question, say so plainly. Do not fill gaps from memory.
- Distinguish what the evidence shows from what it merely suggests.

Evidence:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm_manager: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        k: int = 5
    ):
        """
        Args:
            retriever: Retriever to use. Built from vector_store/knowledge_graph
                when not supplied.
            llm_manager: Object exposing process_biomedical_task(); defaults to
                UnifiedLLMManager, which prefers the local Ollama model.
            vector_store: Literature vector store, used to build a retriever.
            knowledge_graph: NetworkX graph, used to build a retriever.
            k: Number of evidence items to retrieve.
        """
        self.k = k
        self.retriever = retriever or self._build_retriever(vector_store, knowledge_graph)

        if llm_manager is None:
            from ..llm_integration.unified_llm_interface import UnifiedLLMManager
            llm_manager = UnifiedLLMManager()
        self.llm_manager = llm_manager

        self.logger.info(
            f"Initialized BiomedicalRAGSystem (retriever: {type(self.retriever).__name__})"
        )

    def _build_retriever(
        self,
        vector_store: Optional[Any],
        knowledge_graph: Optional[Any]
    ) -> BaseRetriever:
        """Assemble the best retriever available from the given sources."""
        literature = (
            LiteratureRetriever(vector_store=vector_store, k=self.k)
            if vector_store is not None else None
        )
        kg = (
            KnowledgeGraphRetriever(graph=knowledge_graph, k=self.k)
            if knowledge_graph is not None else None
        )

        if literature and kg:
            return HybridRetriever(
                literature_retriever=literature, kg_retriever=kg, k=self.k
            )
        if literature:
            return literature
        if kg:
            return kg

        # No sources configured: an empty retriever keeps the pipeline running
        # and makes "no evidence" an explicit answer rather than a crash.
        self.logger.warning(
            "No vector store or knowledge graph supplied; retrieval will return nothing"
        )
        return LiteratureRetriever(vector_store=None, k=self.k)

    @staticmethod
    def format_context(documents: Sequence[Document]) -> str:
        """Render retrieved documents as numbered, citable evidence."""
        if not documents:
            return "(no evidence retrieved)"

        blocks = []
        for i, document in enumerate(documents, start=1):
            source = document.metadata.get(
                "pmid", document.metadata.get("source_type", "unknown")
            )
            blocks.append(f"[{i}] ({source}) {document.page_content}")

        return "\n\n".join(blocks)

    def retrieve(self, question: str) -> List[Document]:
        """Retrieve evidence for a question."""
        documents = self.retriever.invoke(question)
        self.logger.info(f"Retrieved {len(documents)} evidence item(s) for {question!r}")
        return documents

    def answer(
        self,
        question: str,
        max_tokens: int = 800,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer a question from retrieved evidence.

        Args:
            question: The biomedical question.
            max_tokens: Generation budget for the answer.
            **kwargs: Additional generation options.

        Returns:
            {"answer", "sources", "num_sources", "question"}. When nothing is
            retrieved the answer says so rather than generating unsupported text.
        """
        documents = self.retrieve(question)

        if not documents:
            return {
                "question": question,
                "answer": (
                    "No evidence was retrieved for this question, so it cannot be "
                    "answered from the configured sources."
                ),
                "sources": [],
                "num_sources": 0,
            }

        prompt = self.ANSWER_TEMPLATE.format(
            context=self.format_context(documents), question=question
        )

        response = self.llm_manager.process_biomedical_task(
            task="literature_analysis",
            input_data=prompt,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            "question": question,
            "answer": response.content,
            "sources": [
                {"content": d.page_content[:300], **d.metadata} for d in documents
            ],
            "num_sources": len(documents),
            "model": response.model,
        }

    def batch_answer(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Answer several questions, keeping positional alignment on failure."""
        answers = []
        for question in questions:
            try:
                answers.append(self.answer(question, **kwargs))
            except Exception as e:
                self.logger.error(f"Failed to answer {question!r}: {e}")
                answers.append({
                    "question": question,
                    "answer": f"Error: {e}",
                    "sources": [],
                    "num_sources": 0,
                    "error": True,
                })
        return answers
