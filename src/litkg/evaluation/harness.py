"""
Evaluation harness: assemble a graph, split it in time, score, report.

The design decision that matters most here is negative sampling. Drawing
negatives uniformly from all non-edges makes the task trivial -- most random
pairs are type-incompatible (a phenotype and a gene), so a predictor can score
well by learning which type pairs are plausible rather than which specific
associations are real. Negatives are drawn to match the endpoint types of the
positives they stand against.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from litkg.evaluation.baselines import BASELINE_PREDICTORS, LinkPredictor
from litkg.evaluation.metrics import RankingMetrics, evaluate_scores
from litkg.evaluation.temporal_split import TemporalSplit
from litkg.utils.logging import LoggerMixin

Edge = Tuple[str, str]


@dataclass
class EvaluationReport:
    """Results of evaluating predictors against one temporal split."""

    split_summary: Dict[str, Any]
    results: Dict[str, RankingMetrics] = field(default_factory=dict)
    negatives_per_positive: int = 1
    notes: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split_summary,
            "negatives_per_positive": self.negatives_per_positive,
            "results": {name: m.to_dict() for name, m in self.results.items()},
            "diagnostics": self.diagnostics,
            "notes": self.notes,
        }

    def format_table(self) -> str:
        """Render results as a fixed-width table, best AUC first."""
        if not self.results:
            return "No results."

        header = (
            f"{'predictor':26} {'AUC':>7} {'AP':>7} {'H@1':>7} "
            f"{'H@5':>7} {'H@10':>7} {'MRR':>7}"
        )
        lines = [header, "-" * len(header)]
        ordered = sorted(
            self.results.items(),
            key=lambda kv: (kv[1].auc if kv[1].auc == kv[1].auc else -1),
            reverse=True,
        )
        for name, m in ordered:
            lines.append(
                f"{name:26} {m.auc:7.3f} {m.average_precision:7.3f} "
                f"{m.hits_at_1:7.3f} {m.hits_at_5:7.3f} {m.hits_at_10:7.3f} {m.mrr:7.3f}"
            )
        return "\n".join(lines)


def structural_coverage(graph: nx.Graph, pairs: Sequence[Edge]) -> float:
    """
    Fraction of pairs that share at least one neighbour in the graph.

    Adamic-Adar, common neighbours and Jaccard are all zero for a pair with no
    shared neighbour, so they cannot rank it above any other zero-scoring pair.
    When coverage is low, those predictors are not performing badly -- they are
    undefined, and their metrics should be read as such.
    """
    if not pairs:
        return 0.0
    covered = 0
    for u, v in pairs:
        if u in graph and v in graph and (set(graph[u]) & set(graph[v])):
            covered += 1
    return covered / len(pairs)


def build_graph(edges: Iterable[Edge]) -> nx.Graph:
    """
    Build the simple undirected graph the structural predictors need.

    Direction and parallel edges are deliberately discarded: Adamic-Adar and
    friends are defined over simple undirected graphs. This is lossy and the
    loss is the point -- two nodes joined by three relation types are connected
    once for the purpose of a topological score.
    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def _degree_bucket(degree: int) -> int:
    """
    Coarse log-scale degree bucket.

    Exact degree matching rarely finds candidates; buckets keep the pools
    usable while still removing the gross popularity signal.
    """
    if degree <= 0:
        return 0
    return int(math.floor(math.log2(degree))) + 1


def sample_negatives(
    positives: Sequence[Edge],
    graph: nx.Graph,
    node_types: Optional[Dict[str, str]] = None,
    negatives_per_positive: int = 1,
    known_edges: Optional[Set[Edge]] = None,
    seed: int = 0,
    max_attempts_per_negative: int = 200,
    degree_matched: bool = False,
) -> List[Edge]:
    """
    Draw negative pairs matching the endpoint types of the positives.

    Args:
        positives: The true test edges.
        graph: Training graph; negatives are drawn from its nodes.
        node_types: node id -> type. When absent, negatives are drawn without
            type matching and the caller is warned, because the resulting
            numbers are optimistic.
        negatives_per_positive: How many negatives to draw per positive.
        known_edges: Every pair known to be real, train and test. A sampled
            "negative" that is actually a real edge is a mislabelled positive
            and depresses every score.
        seed: For reproducibility.
        degree_matched: Draw each negative endpoint from the same degree bucket
            as the positive endpoint it stands against. Without this, positives
            sit on hub nodes and negatives on arbitrary ones, so a predictor
            can win on popularity alone -- which is what preferential
            attachment doing best is telling you.

    Returns:
        Sampled non-edges, deduplicated.
    """
    rng = random.Random(seed)
    known = set(known_edges or set())
    known |= {(u, v) if u <= v else (v, u) for u, v in graph.edges()}

    nodes = list(graph.nodes())
    if not nodes:
        return []

    def pool_key(node: str) -> Tuple[str, int]:
        node_type = node_types.get(node, "UNKNOWN") if node_types else "ANY"
        bucket = _degree_bucket(graph.degree(node)) if degree_matched else -1
        return (node_type, bucket)

    pools: Dict[Tuple[str, int], List[str]] = {}
    for node in nodes:
        pools.setdefault(pool_key(node), []).append(node)

    negatives: Set[Edge] = set()
    for u, v in positives:
        source_pool = pools.get(pool_key(u)) or nodes
        target_pool = pools.get(pool_key(v)) or nodes

        drawn = 0
        attempts = 0
        while drawn < negatives_per_positive and attempts < max_attempts_per_negative:
            attempts += 1
            a, b = rng.choice(source_pool), rng.choice(target_pool)
            if a == b:
                continue
            pair = (a, b) if a <= b else (b, a)
            if pair in known or pair in negatives:
                continue
            negatives.add(pair)
            drawn += 1

    return sorted(negatives)


class Harness(LoggerMixin):
    """Runs predictors against a temporal split."""

    def run(
        self,
        split: TemporalSplit,
        node_types: Optional[Dict[str, str]] = None,
        predictors: Optional[Sequence[LinkPredictor]] = None,
        negatives_per_positive: int = 10,
        seed: int = 0,
        degree_matched: bool = False,
    ) -> EvaluationReport:
        notes: List[str] = []

        train_graph = build_graph(split.train_edges | split.backbone_edges)
        positives = sorted(split.test_edges)

        report = EvaluationReport(
            split_summary=split.summary(),
            negatives_per_positive=negatives_per_positive,
            notes=notes,
        )

        if not positives:
            notes.append(
                "No test edges survived the split; nothing to evaluate. "
                "Try an earlier cutoff year."
            )
            return report

        if node_types is None:
            notes.append(
                "Negatives were sampled without type matching. Most random "
                "pairs are type-incompatible, so these scores are optimistic."
            )

        if degree_matched:
            notes.append(
                "Negatives are degree-matched, so popularity is controlled for. "
                "Scores here reflect structure, not how well-studied a node is."
            )

        known = set(split.train_edges) | set(split.backbone_edges) | set(split.test_edges)
        negatives = sample_negatives(
            positives,
            train_graph,
            node_types=node_types,
            negatives_per_positive=negatives_per_positive,
            known_edges=known,
            seed=seed,
            degree_matched=degree_matched,
        )

        if not negatives:
            notes.append("Could not sample any negatives; nothing to evaluate.")
            return report

        actual_ratio = len(negatives) / len(positives)
        if actual_ratio < negatives_per_positive * 0.5:
            notes.append(
                f"Only {actual_ratio:.1f} negatives per positive could be sampled "
                f"(asked for {negatives_per_positive}); the type pools are small."
            )

        coverage = structural_coverage(train_graph, positives)
        report.diagnostics["structural_coverage"] = round(coverage, 4)
        report.diagnostics["train_graph_nodes"] = train_graph.number_of_nodes()
        report.diagnostics["train_graph_edges"] = train_graph.number_of_edges()
        report.diagnostics["average_clustering"] = round(
            nx.average_clustering(train_graph), 4
        )
        if coverage < 0.5:
            notes.append(
                f"Only {coverage:.1%} of test pairs share a neighbour in the "
                f"training graph. Shared-neighbour predictors score exactly 0 "
                f"for the rest and cannot rank them, so their Hits@K and MRR "
                f"reflect an undefined score, not a wrong one. The graph is too "
                f"sparse for topological link prediction as built."
            )

        if predictors is None:
            predictors = [cls() for cls in BASELINE_PREDICTORS]

        for predictor in predictors:
            predictor.fit(train_graph)
            positive_scores = predictor.score_pairs(positives)
            negative_scores = predictor.score_pairs(negatives)
            report.results[predictor.name] = evaluate_scores(
                positive_scores, negative_scores
            )
            self.logger.info(
                f"{predictor.name}: AUC={report.results[predictor.name].auc:.3f}"
            )

        return report


def evaluate_baselines(
    split: TemporalSplit,
    node_types: Optional[Dict[str, str]] = None,
    predictors: Optional[Sequence[LinkPredictor]] = None,
    negatives_per_positive: int = 10,
    seed: int = 0,
    degree_matched: bool = False,
) -> EvaluationReport:
    """Convenience wrapper around Harness.run()."""
    return Harness().run(
        split,
        node_types=node_types,
        predictors=predictors,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
        degree_matched=degree_matched,
    )
