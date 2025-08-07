"""
Phase 2: Integration Engine

This module implements the hybrid GNN architecture with cross-modal attention
mechanisms for integrating literature and knowledge graph representations.
"""

from .hybrid_gnn import (
    HybridGNNModel,
    LiteratureGraphEncoder,
    KnowledgeGraphEncoder,
    CrossModalFusion,
    RelationPredictor
)

from .attention_mechanisms import (
    CrossModalAttention,
    StructuralAttention,
    TemporalAttention,
    MultiHeadCrossAttention
)

from .graph_construction import (
    GraphConstructor,
    LiteratureGraphBuilder,
    KGSubgraphExtractor,
    GraphAligner
)

from .training import (
    GNNTrainer as HybridGNNTrainer,
    TrainingConfig,
    ContrastiveLoss,
    MultiTaskLoss,
    EvaluationMetrics
)

from .data_loader import (
    HybridGraphDataset,
    GraphBatchSampler,
    DataCollator,
    create_data_loaders
)

__all__ = [
    # Core Models
    "HybridGNNModel",
    "LiteratureGraphEncoder",
    "KnowledgeGraphEncoder", 
    "CrossModalFusion",
    "RelationPredictor",
    
    # Attention Mechanisms
    "CrossModalAttention",
    "StructuralAttention",
    "TemporalAttention",
    "MultiHeadCrossAttention",
    
    # Graph Construction
    "GraphConstructor",
    "LiteratureGraphBuilder",
    "KGSubgraphExtractor",
    "GraphAligner",
    
    # Training Infrastructure
    "HybridGNNTrainer",
    "TrainingConfig",
    "ContrastiveLoss",
    "MultiTaskLoss",
    "EvaluationMetrics",
    
    # Data Loading
    "HybridGraphDataset",
    "GraphBatchSampler",
    "DataCollator",
    "create_data_loaders"
]