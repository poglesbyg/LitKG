"""
Evaluation harness for link prediction on the integrated knowledge graph.

Everything reported about this project so far has been a count -- how many
nodes, how many edges, how many links. Counts cannot distinguish "more edges"
from "better edges". This package measures whether the graph supports
predicting associations that were not known when the training data was
assembled.
"""

from litkg.evaluation.temporal_split import (
    TemporalSplit,
    RelationRecord,
    EdgeEvidence,
    build_temporal_split,
    extract_publication_year,
)
from litkg.evaluation.baselines import (
    LinkPredictor,
    AdamicAdarPredictor,
    L3PathPredictor,
    WeightedL3PathPredictor,
    CommonNeighborsPredictor,
    JaccardPredictor,
    PreferentialAttachmentPredictor,
    RandomPredictor,
    BASELINE_PREDICTORS,
)
from litkg.evaluation.metrics import (
    RankingMetrics,
    evaluate_scores,
    hits_at_k,
    mean_reciprocal_rank,
)
from litkg.evaluation.harness import (
    EvaluationReport,
    evaluate_baselines,
    sample_negatives,
)

__all__ = [
    "TemporalSplit",
    "RelationRecord",
    "EdgeEvidence",
    "build_temporal_split",
    "extract_publication_year",
    "LinkPredictor",
    "AdamicAdarPredictor",
    "L3PathPredictor",
    "WeightedL3PathPredictor",
    "CommonNeighborsPredictor",
    "JaccardPredictor",
    "PreferentialAttachmentPredictor",
    "RandomPredictor",
    "BASELINE_PREDICTORS",
    "RankingMetrics",
    "evaluate_scores",
    "hits_at_k",
    "mean_reciprocal_rank",
    "EvaluationReport",
    "evaluate_baselines",
    "sample_negatives",
]
