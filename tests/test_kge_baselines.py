"""
Tests for the PyKEEN baseline adapter.

The point of the adapter is comparability: PyKEEN's own evaluation ranks against
every entity rather than the degree-matched negatives this project uses, so its
numbers would not be comparable to anything else here. These models are wrapped
as ordinary LinkPredictors and scored by the same harness instead.
"""

import networkx as nx
import pytest

from litkg.phase2.kge_baselines import SUPPORTED, KGEmbeddingPredictor

pykeen = pytest.importorskip("pykeen", reason="pykeen is an optional extra")


@pytest.fixture
def graph():
    return nx.Graph([
        ("g1", "v1"), ("g1", "v2"), ("v1", "d1"), ("v2", "d2"),
        ("g2", "v3"), ("v3", "d1"), ("g2", "v4"), ("v4", "d2"),
    ])


@pytest.fixture
def predicates(graph):
    kinds = ["HAS_VARIANT", "ASSOCIATED_WITH"]
    return {
        (u, v) if u <= v else (v, u): kinds[i % 2]
        for i, (u, v) in enumerate(graph.edges())
    }


class TestConstruction:
    def test_rejects_an_unknown_model(self):
        with pytest.raises(ValueError, match="unsupported model"):
            KGEmbeddingPredictor(model="NotAModel")

    def test_name_identifies_the_model(self):
        assert KGEmbeddingPredictor(model="RotatE").name == "kge_rotate"
        assert KGEmbeddingPredictor(model="TransE").name == "kge_transe"

    def test_supported_list_is_what_it_claims(self):
        for name in SUPPORTED:
            assert KGEmbeddingPredictor(model=name).name.startswith("kge_")


class TestTriples:
    def test_edges_become_triples_in_both_directions(self, graph, predicates):
        predictor = KGEmbeddingPredictor(edge_predicates=predicates)
        triples = predictor._triples(graph)
        # The harness treats pairs as unordered; a model trained one way round
        # would score the reverse at chance for no reason the task cares about.
        assert len(triples) == 2 * graph.number_of_edges()

    def test_edges_without_a_predicate_get_a_default_relation(self, graph):
        predictor = KGEmbeddingPredictor(edge_predicates={})
        relations = {row[1] for row in predictor._triples(graph)}
        assert relations == {"RELATED_TO"}

    def test_declared_predicates_are_used(self, graph, predicates):
        predictor = KGEmbeddingPredictor(edge_predicates=predicates)
        relations = {row[1] for row in predictor._triples(graph)}
        assert relations == {"HAS_VARIANT", "ASSOCIATED_WITH"}

    def test_empty_graph_yields_no_triples(self):
        predictor = KGEmbeddingPredictor()
        assert len(predictor._triples(nx.Graph())) == 0


class TestScoringContract:
    def test_unfitted_model_scores_zero_rather_than_raising(self):
        predictor = KGEmbeddingPredictor()
        assert predictor.score("a", "b") == 0.0

    def test_empty_graph_fit_leaves_the_model_abstaining(self):
        predictor = KGEmbeddingPredictor(epochs=1).fit(nx.Graph())
        assert predictor.score("a", "b") == 0.0

    @pytest.mark.slow
    def test_fitted_model_scores_and_is_symmetric(self, graph, predicates):
        predictor = KGEmbeddingPredictor(
            model="TransE", epochs=2, embedding_dim=8, edge_predicates=predicates
        ).fit(graph)
        assert predictor.score("g1", "v1") == pytest.approx(predictor.score("v1", "g1"))

    @pytest.mark.slow
    def test_unknown_entity_abstains(self, graph, predicates):
        # A node absent from training has no embedding; abstain rather than
        # inventing one.
        predictor = KGEmbeddingPredictor(
            model="TransE", epochs=2, embedding_dim=8, edge_predicates=predicates
        ).fit(graph)
        assert predictor.score("g1", "never-seen") == 0.0

    @pytest.mark.slow
    def test_score_pairs_matches_score(self, graph, predicates):
        predictor = KGEmbeddingPredictor(
            model="TransE", epochs=2, embedding_dim=8, edge_predicates=predicates
        ).fit(graph)
        pairs = [("g1", "v1"), ("g2", "d1")]
        assert predictor.score_pairs(pairs) == [predictor.score(*p) for p in pairs]
