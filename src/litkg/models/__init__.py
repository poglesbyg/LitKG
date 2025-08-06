"""
Model management and integration for LitKG-Integrate.

This module provides unified interfaces for HuggingFace transformers,
PyTorch models, and custom biomedical models.
"""

from .huggingface_models import (
    BiomedicalModelManager,
    ModelRegistry,
    load_biomedical_model,
    get_available_models
)
from .pytorch_models import (
    GraphNeuralNetwork,
    HybridGNN,
    CrossModalAttention
)
from .embeddings import (
    BiomedicalEmbeddings,
    CachedEmbeddings,
    MultiModalEmbeddings
)

__all__ = [
    "BiomedicalModelManager",
    "ModelRegistry", 
    "load_biomedical_model",
    "get_available_models",
    "GraphNeuralNetwork",
    "HybridGNN",
    "CrossModalAttention",
    "BiomedicalEmbeddings",
    "CachedEmbeddings",
    "MultiModalEmbeddings",
]