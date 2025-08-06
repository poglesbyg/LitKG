"""
Phase 3: Novel Knowledge Discovery and Validation - COMPLETE

This phase focuses on using the integrated hybrid GNN from Phase 2 to:
1. Develop confidence scoring metrics for different types of evidence
2. Predict novel relationships and generate hypotheses with AI agents
3. Validate predictions against held-out data and expert assessment
4. Quantify uncertainty and distinguish contradictory evidence

Key Components:
- ConfidenceScorer: Multi-faceted confidence assessment
- NoveltyDetectionSystem: AI-powered discovery of novel relationships
- HypothesisGenerationSystem: LLM-powered hypothesis generation and validation
- ComprehensiveValidationSystem: Multi-method validation with expert interface
"""

from .confidence_scoring import (
    ConfidenceScorer,
    EvidenceType,
    ConfidenceMetrics,
    LiteratureConfidenceAssessor,
    ExperimentalConfidenceAssessor,
    CrossModalConfidenceIntegrator
)

from .novelty_detection import (
    NovelRelation,
    DiscoveryPattern,
    NovelRelationPredictor,
    PatternDiscoveryEngine,
    BiologicalPlausibilityChecker,
    NoveltyDetectionSystem
)

from .hypothesis_generation import (
    BiomedicalHypothesis,
    ExperimentalDesign,
    HypothesisGenerator,
    HypothesisValidationAgent,
    HypothesisGenerationSystem,
    BiologicalReasoningTool,
    LiteratureValidationTool,
    ExperimentalDesignTool
)

from .validation import (
    ValidationResult,
    ExpertAssessment,
    LiteratureCrossValidator,
    TemporalValidator,
    ExpertValidationInterface,
    ComprehensiveValidationSystem
)

__all__ = [
    # Confidence Scoring
    "ConfidenceScorer",
    "EvidenceType", 
    "ConfidenceMetrics",
    "LiteratureConfidenceAssessor",
    "ExperimentalConfidenceAssessor",
    "CrossModalConfidenceIntegrator",
    
    # Novelty Detection
    "NovelRelation",
    "DiscoveryPattern",
    "NovelRelationPredictor",
    "PatternDiscoveryEngine",
    "BiologicalPlausibilityChecker",
    "NoveltyDetectionSystem",
    
    # Hypothesis Generation
    "BiomedicalHypothesis",
    "ExperimentalDesign", 
    "HypothesisGenerator",
    "HypothesisValidationAgent",
    "HypothesisGenerationSystem",
    "BiologicalReasoningTool",
    "LiteratureValidationTool",
    "ExperimentalDesignTool",
    
    # Validation
    "ValidationResult",
    "ExpertAssessment",
    "LiteratureCrossValidator",
    "TemporalValidator", 
    "ExpertValidationInterface",
    "ComprehensiveValidationSystem"
]