"""
LLM Integration Module for LitKG-Integrate

This module provides comprehensive integration with various LLM providers:
1. Ollama for local open-source model inference
2. OpenAI API and OpenAI-compatible endpoints
3. Anthropic Claude API
4. HuggingFace transformers for local inference
5. Unified interface for biomedical LLM tasks

Key features:
- Model management and auto-selection
- Biomedical-optimized prompting
- Local and cloud model support
- Performance optimization and caching
- Error handling and fallback strategies
"""

from .ollama_integration import (
    OllamaManager,
    OllamaLLM,
    BiomedicalOllamaChain,
    LocalModelManager
)

from .unified_llm_interface import (
    UnifiedLLMManager,
    LLMProvider,
    BiomedicalLLMInterface,
    ModelCapabilities,
    LLMResponse
)

from .biomedical_prompts import (
    BiomedicalPromptManager,
    EntityExtractionPrompts,
    RelationExtractionPrompts,
    HypothesisGenerationPrompts,
    ValidationPrompts
)

from .model_selection import (
    ModelSelector,
    BiomedicalModelRecommendations,
    PerformanceOptimizer
)

__all__ = [
    # Ollama Integration
    "OllamaManager",
    "OllamaLLM", 
    "BiomedicalOllamaChain",
    "LocalModelManager",
    
    # Unified Interface
    "UnifiedLLMManager",
    "LLMProvider",
    "BiomedicalLLMInterface", 
    "ModelCapabilities",
    "LLMResponse",
    
    # Biomedical Prompts
    "BiomedicalPromptManager",
    "EntityExtractionPrompts",
    "RelationExtractionPrompts", 
    "HypothesisGenerationPrompts",
    "ValidationPrompts",
    
    # Model Selection
    "ModelSelector",
    "BiomedicalModelRecommendations",
    "PerformanceOptimizer"
]