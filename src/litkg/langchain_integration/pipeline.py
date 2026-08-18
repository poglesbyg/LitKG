"""
Assembles the RAG system from the Phase 1 output.

The retrievers, chunk-to-graph index and agents all worked in isolation and
were unit tested, but nothing connected them to real data: `make run-langchain`
built its own hardcoded documents and a bare FAISS index, so the graph-aware
path was never exercised outside tests. This module is that missing wiring.

Everything it needs is already on disk after `make run-phase1`:
literature documents, and the integrated knowledge graph.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from litkg.utils.config import get_data_dir
from litkg.utils.logging import LoggerMixin

# Sentence-tuned and small. Note this is a different choice from the node
# feature encoder, which measured better with PubMedBERT on entity *names*:
# passage retrieval rewards a model trained for sentence similarity, whereas
# name matching rewards biomedical vocabulary. Retrieval quality here has not
# been measured -- there is no relevance-judged query set to measure it against.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class PipelineConfig:
    """Where to read Phase 1 output and how to build the index."""

    documents_path: Optional[Path] = None
    graph_path: Optional[Path] = None
    vector_store_path: Optional[Path] = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 512
    chunk_overlap: int = 64
    k: int = 5
    # Measured, not assumed. Against 57 CIVIC-judged queries, vector-retrieved
    # passages are 55.4% relevant while graph-expanded ones are 4.6% -- twelve
    # times worse -- so expansion mostly dilutes the evidence handed to the
    # model. It stays available (--hops) because the judgements are biased
    # against it by construction: they mark relevant only what CIVIC cited for
    # one relationship, which is precisely not the vocabulary-crossing evidence
    # expansion exists to reach. Defaulting it on would ship a known dilution
    # on the strength of that caveat.
    max_hops: int = 0

    def __post_init__(self):
        processed = get_data_dir() / "processed"
        if self.documents_path is None:
            self.documents_path = processed / "combined_literature_results.json"
        if self.graph_path is None:
            self.graph_path = processed / "integrated_knowledge_graph.gpickle"
        if self.vector_store_path is None:
            # Rebuilt from the documents, so it is a cache rather than data.
            self.vector_store_path = (
                get_data_dir() / "knowledge_base" / "vector_store" / "rag_faiss"
            )


def load_documents(path: Path) -> List[Dict[str, Any]]:
    """Load Phase 1 literature output, tolerating its several shapes."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    for key in ("documents", "results", "articles"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def load_graph(path: Path) -> Any:
    """
    Load the integrated knowledge graph.

    Phase 1 writes a MultiDiGraph. The alias index only reads node ids and
    attributes, and graph expansion only needs adjacency, so the multigraph is
    passed through as-is rather than being flattened -- flattening here would
    discard relation types that the graph retriever surfaces in its evidence.
    """
    return pickle.loads(Path(path).read_bytes())


class RAGPipeline(LoggerMixin):
    """Builds a graph-aware RAG system from files Phase 1 already produced."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.documents: List[Any] = []
        self.chunks: List[Any] = []
        self.graph: Any = None
        self.chunk_index: Any = None
        self.vector_store: Any = None

    # ------------------------------------------------------------------

    def _to_langchain_documents(self, records: Sequence[Dict[str, Any]]) -> List[Any]:
        from langchain_core.documents import Document

        documents = []
        for record in records:
            text = " ".join(
                part for part in (record.get("title"), record.get("abstract"))
                if part
            ).strip()
            if not text:
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "pmid": str(record.get("pmid", "")),
                    "title": record.get("title", ""),
                    "journal": record.get("journal", ""),
                    "publication_date": str(record.get("publication_date", "")),
                },
            ))
        return documents

    def _chunk(self, documents: Sequence[Any]) -> List[Any]:
        """Split with the biomedical splitter, keeping section labels."""
        from langchain_core.documents import Document

        from litkg.langchain_integration.enhanced_literature_processor import (
            BiomedicalTextSplitter,
        )

        splitter = BiomedicalTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        chunks: List[Any] = []
        for index, document in enumerate(documents):
            for position, (text, section) in enumerate(
                splitter.split_text_with_sections(document.page_content)
            ):
                metadata = dict(document.metadata)
                # ChunkGraphIndex derives its key as "{pmid}:{chunk_id}", so
                # chunk_id is the position alone -- including the pmid here too
                # produced keys like "123:123:0", which still worked but made
                # the index unjoinable by anything reconstructing the id.
                # Consumers should read metadata["chunk_uid"], which the index
                # writes back.
                metadata["chunk_id"] = str(position)
                metadata["section"] = section or ""
                chunks.append(Document(page_content=text, metadata=metadata))
        return chunks

    def _embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=self.config.embedding_model)

    def _build_vector_store(self, chunks: Sequence[Any]) -> Any:
        from langchain_community.vectorstores import FAISS

        path = Path(self.config.vector_store_path)
        embeddings = self._embeddings()

        if path.exists():
            try:
                self.logger.info(f"Loading cached vector store from {path}")
                return FAISS.load_local(
                    str(path), embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                self.logger.warning(f"Rebuilding unusable vector store: {e}")

        self.logger.info(f"Embedding {len(chunks)} chunks")
        store = FAISS.from_documents(list(chunks), embeddings)
        path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))
        return store

    # ------------------------------------------------------------------

    def build(self, rebuild: bool = False) -> "RAGPipeline":
        """Load Phase 1 output and construct every index the RAG system needs."""
        from litkg.langchain_integration.graph_linking import (
            ChunkGraphIndex,
            EntityAliasIndex,
        )

        config = self.config
        if not Path(config.documents_path).exists():
            raise FileNotFoundError(
                f"No literature output at {config.documents_path}. "
                f"Run `make run-phase1` first."
            )
        if not Path(config.graph_path).exists():
            raise FileNotFoundError(
                f"No knowledge graph at {config.graph_path}. "
                f"Run `make run-phase1` first."
            )

        if rebuild:
            store_path = Path(config.vector_store_path)
            if store_path.exists():
                import shutil
                shutil.rmtree(store_path)

        records = load_documents(config.documents_path)
        self.documents = self._to_langchain_documents(records)
        self.chunks = self._chunk(self.documents)
        self.graph = load_graph(config.graph_path)

        alias_index = EntityAliasIndex().add_from_graph(self.graph)
        self.chunk_index = ChunkGraphIndex(alias_index)
        self.chunk_index.index_chunks(self.chunks)

        self.vector_store = self._build_vector_store(self.chunks)

        self.logger.info(
            f"RAG pipeline ready: {len(self.documents)} documents, "
            f"{len(self.chunks)} chunks, {self.graph.number_of_nodes()} graph nodes"
        )
        return self

    def rag_system(self, llm_manager: Optional[Any] = None) -> Any:
        """The assembled BiomedicalRAGSystem."""
        from litkg.langchain_integration.rag_system import BiomedicalRAGSystem

        if self.vector_store is None:
            self.build()
        return BiomedicalRAGSystem(
            llm_manager=llm_manager,
            vector_store=self.vector_store,
            knowledge_graph=self.graph,
            chunk_index=self.chunk_index,
            k=self.config.k,
            max_hops=self.config.max_hops,
        )

    def agent(self, llm_manager: Optional[Any] = None) -> Any:
        """A conversational agent backed by the same retrieval stack."""
        from litkg.langchain_integration.biomedical_agent import (
            BiomedicalQueryAgent,
            BiomedicalToolkit,
        )

        rag = self.rag_system(llm_manager=llm_manager)
        return BiomedicalQueryAgent(toolkit=BiomedicalToolkit(rag_system=rag))

    def coverage(self) -> Dict[str, Any]:
        """
        How much of the corpus actually reaches the graph.

        Graph expansion can only help for chunks that link to a node, so this
        is the number that decides whether multi-hop retrieval is doing
        anything at all.
        """
        linked = len(getattr(self.chunk_index, "chunk_to_nodes", {}) or {})
        nodes = set()
        for values in (getattr(self.chunk_index, "chunk_to_nodes", {}) or {}).values():
            nodes.update(values)
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "chunks_linked_to_graph": linked,
            "link_rate": linked / len(self.chunks) if self.chunks else 0.0,
            "graph_nodes_reached": len(nodes),
            "graph_nodes_total": self.graph.number_of_nodes() if self.graph else 0,
        }
