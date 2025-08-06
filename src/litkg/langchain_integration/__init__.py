"""
LangChain Integration for LitKG-Integrate

This package provides enhanced capabilities using LangChain for:
1. Advanced document processing and retrieval
2. LLM-powered entity and relation extraction
3. Conversational agents for biomedical queries
4. Hypothesis generation and validation
5. Multi-modal RAG systems

Components:
- Enhanced Literature Processor with LangChain document loaders
- LLM-powered Entity Extractor with confidence scoring
- Biomedical Query Agent with tool integration
- RAG system for literature-augmented responses
- Hypothesis Generation Agent with reasoning chains
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

from .biomedical_agent import (
    BiomedicalQueryAgent,
    BiomedicalToolkit,
    HypothesisGenerationAgent,
    LiteratureValidationAgent
)

from .rag_system import (
    BiomedicalRAGSystem,
    LiteratureRetriever,
    KnowledgeGraphRetriever,
    HybridRetriever
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
    
    # Biomedical Agents
    "BiomedicalQueryAgent",
    "BiomedicalToolkit",
    "HypothesisGenerationAgent", 
    "LiteratureValidationAgent",
    
    # RAG System
    "BiomedicalRAGSystem",
    "LiteratureRetriever",
    "KnowledgeGraphRetriever",
    "HybridRetriever"
]