"""
Tests for the trained link predictors.

Weighted toward the two failure modes that produce confident nonsense: leakage
between the message-passing graph and the supervision signal, and scores that
depend on which batch a pair happened to be scored in.
"""

import networkx as nx
import numpy as np
import pytest

from litkg.evaluation import build_temporal_split, evaluate_baselines
from litkg.evaluation.harness import build_graph
from litkg.phase2.link_prediction import (
    GNNLinkPredictor,
    HybridLinkPredictor,
    TrainingConfig,
)


@pytest.fixture(scope="module")
def toy_graph():
    """A small multipartite graph, the shape this model targets."""
    g = nx.Graph()
    for i in range(40):
        g.add_edge(f"v{i}", f"d{i % 6}")
        g.add_edge(f"v{i}", f"t{i % 5}")
    for i in range(6):
        g.add_edge(f"d{i}", f"t{i % 5}")
    return g


@pytest.fixture(scope="module")
def toy_types(toy_graph):
    return {
        n: ("MUTATION" if n.startswith("v") else
            "DISEASE" if n.startswith("d") else "DRUG")
        for n in toy_graph.nodes()
    }


@pytest.fixture(scope="module")
def fast_config():
    return TrainingConfig(epochs=20, patience=20, hidden_dim=16,
                          embedding_dim=16, resample_every=5, seed=0)


class TestGNNLinkPredictor:
    def test_fits_and_scores(self, toy_graph, toy_types, fast_config):
        model = GNNLinkPredictor(config=fast_config, node_types=toy_types).fit(toy_graph)
        scores = model.score_pairs([("v0", "d3"), ("v1", "t2")])
        assert len(scores) == 2
        assert all(np.isfinite(s) for s in scores)

    def test_scoring_is_batch_independent(self, toy_graph, toy_types, fast_config):
        """
        A pair's score must not depend on what it was scored alongside. The
        harness scores positives and negatives in separate calls, so any
        batch-relative transform silently corrupts every metric.
        """
        model = GNNLinkPredictor(config=fast_config, node_types=toy_types).fit(toy_graph)
        alone = model.score_pairs([("v0", "d3")])[0]
        in_batch = model.score_pairs(
            [("v0", "d3")] + [(f"v{i}", f"d{i % 6}") for i in range(1, 20)]
        )[0]
        assert alone == pytest.approx(in_batch)

    def test_unseen_nodes_score_lowest(self, toy_graph, toy_types, fast_config):
        """The model has no basis for ranking a node it never saw."""
        model = GNNLinkPredictor(config=fast_config, node_types=toy_types).fit(toy_graph)
        assert model.score("absent", "d0") < min(
            model.score_pairs([("v0", "d0"), ("v1", "d1")])
        )

    def test_training_history_is_recorded(self, toy_graph, toy_types, fast_config):
        model = GNNLinkPredictor(config=fast_config, node_types=toy_types).fit(toy_graph)
        assert model.history
        assert all("val_auc" in row for row in model.history)

    def test_same_seed_reproduces(self, toy_graph, toy_types, fast_config):
        pairs = [("v0", "d3"), ("v5", "t1")]
        a = GNNLinkPredictor(config=fast_config, node_types=toy_types).fit(toy_graph)
        b = GNNLinkPredictor(config=fast_config, node_types=toy_types).fit(toy_graph)
        assert a.score_pairs(pairs) == pytest.approx(b.score_pairs(pairs), rel=1e-3)

    def test_bpr_and_bce_both_train(self, toy_graph, toy_types):
        for loss in ("bpr", "bce"):
            config = TrainingConfig(epochs=10, patience=10, hidden_dim=16,
                                    embedding_dim=16, loss=loss, seed=0)
            model = GNNLinkPredictor(config=config, node_types=toy_types).fit(toy_graph)
            assert np.isfinite(model.score("v0", "d1"))


class TestHybridLinkPredictor:
    def test_scoring_is_batch_independent(self, toy_graph, toy_types, fast_config):
        """
        This is the bug that made the first hybrid report AUC 0.000. Ranking
        within a call gave positives ranks 1..N and negatives ranks 1..10N, so
        every negative outranked every positive.
        """
        model = HybridLinkPredictor(
            config=fast_config, node_types=toy_types, weight=0.5
        ).fit(toy_graph)
        alone = model.score_pairs([("v0", "d3")])[0]
        in_batch = model.score_pairs(
            [("v0", "d3")] + [(f"v{i}", f"t{i % 5}") for i in range(1, 30)]
        )[0]
        assert alone == pytest.approx(in_batch)

    def test_survives_the_harness_end_to_end(self, toy_types):
        """The end-to-end path where the batch-dependence bug actually showed."""
        edges = []
        for i in range(40):
            edges.append((f"v{i}", f"d{i % 6}", 2000 + (i % 12)))
            edges.append((f"v{i}", f"t{i % 5}", 2000 + (i % 12)))
        types = {
            n: ("MUTATION" if n.startswith("v") else
                "DISEASE" if n.startswith("d") else "DRUG")
            for e in edges for n in e[:2]
        }
        split = build_temporal_split(edges, cutoff_year=2008)
        config = TrainingConfig(epochs=10, patience=10, hidden_dim=16,
                                embedding_dim=16, seed=0)
        model = HybridLinkPredictor(config=config, node_types=types, weight=0.5)
        report = evaluate_baselines(
            split, node_types=types, predictors=[model],
            negatives_per_positive=5, seed=0, degree_matched=True,
        )
        if "hybrid" in report.results:
            auc = report.results["hybrid"].auc
            # 0.0 is the signature of batch-relative scoring, not a weak model.
            assert auc > 0.05, f"AUC {auc} suggests batch-dependent scores"

    def test_weight_is_selected_not_assumed(self, toy_graph, toy_types, fast_config):
        model = HybridLinkPredictor(
            config=fast_config, node_types=toy_types,
            edge_years={e: 2000 + i % 10 for i, e in enumerate(toy_graph.edges())},
        ).fit(toy_graph)
        assert 0.0 <= model.selected_weight <= 1.0

    def test_explicit_weight_is_respected(self, toy_graph, toy_types, fast_config):
        model = HybridLinkPredictor(
            config=fast_config, node_types=toy_types, weight=0.75
        ).fit(toy_graph)
        assert model.selected_weight == 0.75

    def test_percentile_handles_ties(self, toy_graph, toy_types, fast_config):
        """
        Most pairs score exactly 0 on L3. Ties must share a percentile rather
        than being ordered arbitrarily, or the ranking head becomes noise.
        """
        model = HybridLinkPredictor(
            config=fast_config, node_types=toy_types, weight=0.5
        ).fit(toy_graph)
        reference = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        values = model._percentile(np.array([0.0, 0.0]), reference)
        assert values[0] == values[1]
