"""
Ranking and classification metrics for link prediction.

Reported together on purpose. AUC is forgiving on imbalanced data and can look
respectable while the top of the ranking is useless; Hits@K and MRR describe
what a user actually sees, which is the head of the list.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class RankingMetrics:
    """Metrics for one predictor on one test set."""

    auc: float
    average_precision: float
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float
    mrr: float
    positives: int
    negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hits_at_k(ranks: Sequence[int], k: int) -> float:
    """Fraction of positives ranked in the top k against their negatives."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    """Mean of 1/rank over positives."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / rank for rank in ranks]))


def _ranks_against_negatives(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> List[int]:
    """
    Rank each positive within the pool of all negatives.

    Ties are given the worst rank in their tie group rather than the best.
    Structural scores produce many exact ties -- a predictor that scores every
    pair 0.0 would otherwise appear to rank every positive first.
    """
    negatives = np.sort(np.asarray(negative_scores, dtype=float))
    ranks = []
    for score in positive_scores:
        # Negatives scoring strictly greater, plus those tied, all outrank.
        greater = len(negatives) - np.searchsorted(negatives, score, side="right")
        tied = np.searchsorted(negatives, score, side="right") - np.searchsorted(
            negatives, score, side="left"
        )
        ranks.append(int(greater + tied + 1))
    return ranks


def evaluate_scores(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> RankingMetrics:
    """Compute all metrics from scored positives and negatives."""
    if not positive_scores or not negative_scores:
        return RankingMetrics(
            auc=float("nan"), average_precision=float("nan"),
            hits_at_1=0.0, hits_at_5=0.0, hits_at_10=0.0, mrr=0.0,
            positives=len(positive_scores), negatives=len(negative_scores),
        )

    labels = np.concatenate([
        np.ones(len(positive_scores)), np.zeros(len(negative_scores))
    ])
    scores = np.concatenate([
        np.asarray(positive_scores, dtype=float),
        np.asarray(negative_scores, dtype=float),
    ])

    ranks = _ranks_against_negatives(positive_scores, negative_scores)

    return RankingMetrics(
        auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        hits_at_1=hits_at_k(ranks, 1),
        hits_at_5=hits_at_k(ranks, 5),
        hits_at_10=hits_at_k(ranks, 10),
        mrr=mean_reciprocal_rank(ranks),
        positives=len(positive_scores),
        negatives=len(negative_scores),
    )
