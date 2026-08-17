"""
LangChain Integration for LitKG-Integrate

This package provides enhanced capabilities using LangChain for:
1. Advanced document processing and retrieval
2. LLM-powered entity and relation extraction
3. Conversational agents for biomedical queries
4. Hypothesis generation and validation
5. Multi-modal RAG systems

Implemented components:
- Enhanced Literature Processor with LangChain document loaders
- LLM-powered Entity Extractor with confidence scoring

Not yet implemented in this package:
- ``biomedical_agent`` (BiomedicalQueryAgent, BiomedicalToolkit,
  HypothesisGenerationAgent, LiteratureValidationAgent)
- ``rag_system`` (BiomedicalRAGSystem, LiteratureRetriever,
  KnowledgeGraphRetriever, HybridRetriever)

Conversational agents and RAG currently live in the sibling ``litkg.agents``
package; see litkg.agents.biomedical_rag_agent.
"""

from .enhanced_literature_processor import (
    LangChainLiteratureProcessor,
    BiomedicalDocumentLoader,
    BiomedicalTextSplitter,
    BiomedicaEmbeddings
)

from .llm_entity_extractor import (
    LLMEntityExtractor,
    BiomedicalPromptTemplates,
    EntityExtractionChain,
    RelationExtractionChain
)

__all__ = [
    # Enhanced Literature Processing
    "LangChainLiteratureProcessor",
    "BiomedicalDocumentLoader",
    "BiomedicalTextSplitter",
    "BiomedicaEmbeddings",

    # LLM Entity Extraction
    "LLMEntityExtractor",
    "BiomedicalPromptTemplates",
    "EntityExtractionChain",
    "RelationExtractionChain",
]