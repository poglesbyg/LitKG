"""
Ranking and classification metrics for link prediction.

Reported together on purpose. AUC is forgiving on imbalanced data and can look
respectable while the top of the ranking is useless; Hits@K and MRR describe
what a user actually sees, which is the head of the list.

**Ranking metrics on this data are driven by a handful of positives.** With
~1200 positives against ~12000 negatives, the top 20 positives contribute 78%
of MRR and the top 10 contribute 42%. A bootstrap CI on MRR came out
[0.0066, 0.0135] -- wider than most differences between predictors. Every
metric therefore ships with a confidence interval, because a point estimate
here invites reading noise as a result.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

# Bootstrap resamples used for confidence intervals. Enough to stabilise a
# percentile interval without making evaluation slow.
DEFAULT_BOOTSTRAP_SAMPLES = 500

Interval = Tuple[float, float]


@dataclass
class RankingMetrics:
    """
    Metrics for one predictor on one test set.

    The `*_ci` fields are 95% bootstrap percentile intervals. Compare
    predictors on overlapping intervals, not point estimates: on this graph the
    MRR interval is wider than the gap between most configurations.
    """

    auc: float
    average_precision: float
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float
    hits_at_100: float
    mrr: float
    positives: int
    negatives: int
    # Fraction of positives the predictor cannot separate from the bulk of the
    # negative pool -- tied with more than half of it. Shared-neighbour methods
    # score exactly 0 for most pairs, so their ranking metrics are computed
    # over a set where much of the signal is undefined rather than wrong.
    indistinguishable_fraction: float = 0.0
    auc_ci: Optional[Interval] = None
    average_precision_ci: Optional[Interval] = None
    hits_at_10_ci: Optional[Interval] = None
    mrr_ci: Optional[Interval] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """One line with the intervals attached, for reading at a glance."""
        def interval(value: Optional[Interval]) -> str:
            return f"[{value[0]:.3f}, {value[1]:.3f}]" if value else "-"

        return (
            f"AUC {self.auc:.3f} {interval(self.auc_ci)}  "
            f"AP {self.average_precision:.3f} {interval(self.average_precision_ci)}  "
            f"MRR {self.mrr:.4f} {interval(self.mrr_ci)}"
        )


def hits_at_k(ranks: Sequence[int], k: int) -> float:
    """Fraction of positives ranked in the top k against their negatives."""
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(np.asarray(ranks) <= k))


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    """Mean of 1/rank over positives."""
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / np.asarray(ranks, dtype=float)))


def _rank_components(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each positive: negatives it beats, negatives it ties with, its rank.

    Ties take the worst rank in their block rather than the best. Structural
    scores produce enormous tie blocks -- 36% of positives score exactly 0 even
    under L3 -- and optimistic tie handling would report a predictor that
    scores everything zero as ranking every positive first.
    """
    negatives = np.sort(np.asarray(negative_scores, dtype=float))
    positives = np.asarray(positive_scores, dtype=float)

    left = np.searchsorted(negatives, positives, side="left")
    right = np.searchsorted(negatives, positives, side="right")

    beaten = left                      # negatives scoring strictly less
    tied = right - left
    outranking = len(negatives) - right
    ranks = outranking + tied + 1
    return beaten, tied, ranks


def _auc_contributions(
    beaten: np.ndarray, tied: np.ndarray, negative_count: int
) -> np.ndarray:
    """
    Per-positive AUC contribution (the Mann-Whitney form).

    AUC is the mean over positives of P(positive outranks a random negative),
    counting a tie as half. Decomposing it this way makes bootstrapping cheap
    and gives exactly the same value as sklearn's roc_auc_score.
    """
    if negative_count == 0:
        return np.zeros_like(beaten, dtype=float)
    return (beaten + 0.5 * tied) / negative_count


def _percentile_interval(samples: Sequence[float]) -> Optional[Interval]:
    if len(samples) < 2:
        return None
    low, high = np.percentile(np.asarray(samples, dtype=float), [2.5, 97.5])
    return (float(low), float(high))


def evaluate_scores(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> RankingMetrics:
    """
    Compute all metrics, with bootstrap confidence intervals.

    Intervals resample positives with replacement, since positives are the unit
    of observation and are far scarcer than negatives. Pass
    `bootstrap_samples=0` to skip them.
    """
    positive_scores = list(positive_scores)
    negative_scores = list(negative_scores)

    if not positive_scores or not negative_scores:
        return RankingMetrics(
            auc=float("nan"), average_precision=float("nan"),
            hits_at_1=0.0, hits_at_5=0.0, hits_at_10=0.0, hits_at_100=0.0,
            mrr=0.0, positives=len(positive_scores), negatives=len(negative_scores),
        )

    beaten, tied, ranks = _rank_components(positive_scores, negative_scores)
    contributions = _auc_contributions(beaten, tied, len(negative_scores))
    reciprocal = 1.0 / ranks

    labels = np.concatenate([
        np.ones(len(positive_scores)), np.zeros(len(negative_scores))
    ])
    scores = np.concatenate([
        np.asarray(positive_scores, dtype=float),
        np.asarray(negative_scores, dtype=float),
    ])

    metrics = RankingMetrics(
        auc=float(np.mean(contributions)),
        average_precision=float(average_precision_score(labels, scores)),
        hits_at_1=hits_at_k(ranks, 1),
        hits_at_5=hits_at_k(ranks, 5),
        hits_at_10=hits_at_k(ranks, 10),
        # Hits@10 has no resolution here: only ~26 of 1204 positives land in
        # the top 10 of ~12000. A coarser cutoff actually moves when a model
        # improves.
        hits_at_100=hits_at_k(ranks, 100),
        mrr=mean_reciprocal_rank(ranks),
        positives=len(positive_scores),
        negatives=len(negative_scores),
        indistinguishable_fraction=float(np.mean(tied > len(negative_scores) / 2)),
    )

    if bootstrap_samples and bootstrap_samples > 1:
        rng = np.random.default_rng(seed)
        count = len(ranks)
        auc_samples, mrr_samples, hits_samples = [], [], []
        for _ in range(bootstrap_samples):
            picks = rng.integers(0, count, count)
            auc_samples.append(float(np.mean(contributions[picks])))
            mrr_samples.append(float(np.mean(reciprocal[picks])))
            hits_samples.append(float(np.mean(ranks[picks] <= 10)))
        metrics.auc_ci = _percentile_interval(auc_samples)
        metrics.mrr_ci = _percentile_interval(mrr_samples)
        metrics.hits_at_10_ci = _percentile_interval(hits_samples)

        # Average precision does not decompose per positive, so it is
        # resampled directly at lower resolution to keep evaluation quick.
        ap_samples = []
        positives_array = np.asarray(positive_scores, dtype=float)
        negatives_array = np.asarray(negative_scores, dtype=float)
        for _ in range(max(1, bootstrap_samples // 5)):
            picks = rng.integers(0, count, count)
            resampled = np.concatenate([positives_array[picks], negatives_array])
            ap_samples.append(float(average_precision_score(labels, resampled)))
        metrics.average_precision_ci = _percentile_interval(ap_samples)

    return metrics
