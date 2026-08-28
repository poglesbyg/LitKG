"""
Tests for input feature centring in GNNLinkPredictor.

Sentence-transformer embeddings are anisotropic: the entity strings in this
graph sit at a mean pairwise cosine of 0.927 before the model sees them, so
every node starts nearly parallel to every other. That is the same defect that
held HybridGNNModel at chance until #33, and the fix there never reached this
model.
"""

import networkx as nx
import numpy as np
import pytest

from litkg.phase2.link_prediction import GNNLinkPredictor, TrainingConfig


def mean_pairwise_cosine(matrix: np.ndarray) -> float:
    normed = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    similarity = normed @ normed.T
    upper = np.triu_indices(len(matrix), k=1)
    return float(similarity[upper].mean())


class FakeEncoder:
    """Returns deliberately anisotropic vectors: a big shared component."""

    def encode_nodes(self, texts):
        rng = np.random.default_rng(0)
        return {
            node: (np.ones(16) * 5.0 + rng.normal(0, 0.3, 16)).astype(np.float32)
            for node in texts
        }


@pytest.fixture
def graph():
    return nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")])


@pytest.fixture
def node_text():
    return {n: f"entity {n}" for n in "abcd"}


class TestAnisotropy:
    def test_the_fake_encoder_is_anisotropic(self):
        vectors = FakeEncoder().encode_nodes({n: "" for n in "abcdefgh"})
        matrix = np.array(list(vectors.values()))
        # Establishes the premise: without this the test below proves nothing.
        assert mean_pairwise_cosine(matrix) > 0.9

    def test_centring_removes_the_shared_direction(self):
        vectors = FakeEncoder().encode_nodes({n: "" for n in "abcdefgh"})
        matrix = np.array(list(vectors.values()))
        centred = matrix - matrix.mean(axis=0, keepdims=True)
        assert abs(mean_pairwise_cosine(centred)) < 0.3

    def test_l2_normalisation_does_not_remove_it(self):
        # The shipped encoder normalises, which is why the defect survived: it
        # projects onto the unit sphere and leaves the mean direction intact.
        vectors = FakeEncoder().encode_nodes({n: "" for n in "abcdefgh"})
        matrix = np.array(list(vectors.values()))
        normed = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        assert mean_pairwise_cosine(normed) > 0.9


class TestConfig:
    def test_text_centring_is_on_by_default(self):
        assert TrainingConfig().center_text_features is True

    def test_static_centring_is_on_by_default(self):
        # One-hot indicators are non-negative and share a direction too, which
        # is why #33 centred the whole vector rather than the text block alone.
        assert TrainingConfig().center_static_features is True

    def test_centring_can_be_disabled_for_comparison(self):
        assert TrainingConfig(center_text_features=False).center_text_features is False


class TestFeatureConstruction:
    def _fit(self, graph, node_text, **flags):
        config = TrainingConfig(epochs=1, seed=0, **flags)
        predictor = GNNLinkPredictor(
            config=config,
            node_types={n: "GENE" for n in graph},
            node_text=node_text,
            text_encoder=FakeEncoder(),
        )
        return predictor.fit(graph)

    def test_centred_text_features_have_near_zero_mean(self, graph, node_text):
        fitted = self._fit(graph, node_text, center_text_features=True)
        column_means = fitted.text_features.cpu().numpy().mean(axis=0)
        assert np.abs(column_means).max() < 1e-5

    def test_uncentred_text_features_keep_their_offset(self, graph, node_text):
        fitted = self._fit(graph, node_text, center_text_features=False)
        column_means = fitted.text_features.cpu().numpy().mean(axis=0)
        assert np.abs(column_means).max() > 1.0

    def test_centring_preserves_shape_and_ordering(self, graph, node_text):
        centred = self._fit(graph, node_text, center_text_features=True)
        plain = self._fit(graph, node_text, center_text_features=False)
        assert centred.text_features.shape == plain.text_features.shape
        assert centred.nodes == plain.nodes

    def test_static_features_centre_when_asked(self, graph, node_text):
        fitted = self._fit(graph, node_text, center_static_features=True)
        column_means = fitted.static_features.cpu().numpy().mean(axis=0)
        assert np.abs(column_means).max() < 1e-5

    def test_static_features_keep_their_offset_when_disabled(self, graph, node_text):
        fitted = self._fit(graph, node_text, center_static_features=False)
        # One-hot type indicators are non-negative, so an uncentred block has a
        # positive column mean.
        assert fitted.static_features.cpu().numpy().mean(axis=0).max() > 0.0

    def test_no_text_means_no_text_features(self, graph):
        fitted = self._fit(graph, {})
        assert fitted.text_features is None
