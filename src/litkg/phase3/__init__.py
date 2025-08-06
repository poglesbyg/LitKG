"""
Phase 3: Novel Knowledge Discovery and Validation

This phase focuses on using the integrated hybrid GNN from Phase 2 to:
1. Develop confidence scoring metrics for different types of evidence
2. Predict novel relationships and generate hypotheses
3. Validate predictions against held-out data
4. Quantify uncertainty and distinguish contradictory evidence

Key Components:
- ConfidenceScorer: Multi-faceted confidence assessment
- NoveltyDetector: Identify truly novel vs. known relationships
- HypothesisGenerator: Generate testable hypotheses from model predictions
- ValidationFramework: Cross-validate against recent literature and experimental data
"""

from .confidence_scoring import (
    ConfidenceScorer,
    EvidenceType,
    ConfidenceMetrics,
    LiteratureConfidenceAssessor,
    ExperimentalConfidenceAssessor,
    CrossModalConfidenceIntegrator
)

# Note: Additional modules will be implemented in future phases
# from .novelty_detection import (
#     NoveltyDetector,
#     NoveltyMetrics,
#     TemporalNoveltyAnalyzer,
#     SemanticNoveltyAnalyzer
# )

# from .hypothesis_generation import (
#     HypothesisGenerator,
#     HypothesisRanker,
#     BiologicalPlausibilityChecker,
#     TestabilityAssessor
# )

# from .validation import (
#     ValidationFramework,
#     CrossValidationManager,
#     TemporalValidation,
#     ExperimentalValidation
# )

__all__ = [
    # Confidence Scoring
    "ConfidenceScorer",
    "EvidenceType", 
    "ConfidenceMetrics",
    "LiteratureConfidenceAssessor",
    "ExperimentalConfidenceAssessor",
    "CrossModalConfidenceIntegrator"
    
    # Future modules will be added here:
    # "NoveltyDetector",
    # "NoveltyMetrics", 
    # "HypothesisGenerator",
    # "ValidationFramework",
    # etc.
]