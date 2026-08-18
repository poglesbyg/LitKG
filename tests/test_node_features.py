"""
Tests for node text features.

The load-bearing question is not whether the encoder runs, but whether any gain
is transfer or string matching: some CIVIC therapies are named for their target
("BRAF Inhibitor" scores 0.65 against "BRAF"), so a model handed names can
score certain pairs from the strings alone.
"""

import numpy as np
import pytest

from litkg.phase2.node_features import (
    DEFAULT_MODEL,
    FeatureConfig,
    FeatureOnlyPredictor,
    NodeTextEncoder,
    build_node_text,
)


class FakeEntity:
    def __init__(self, id, name, attributes=None):
        self.id, self.name, self.attributes = id, name, attributes or {}


class TestBuildNodeText:
    def test_uses_entity_names(self):
        text = build_node_text([FakeEntity("D1", "Melanoma")])
        assert text == {"D1": "Melanoma"}

    def test_variants_are_qualified_by_gene(self):
        """
        "Amplification" alone is meaningless and collides across genes -- many
        genes have one. The gene makes the string carry information.
        """
        text = build_node_text([
            FakeEntity("V1", "V600E", {"gene": "BRAF"}),
            FakeEntity("V2", "Amplification", {"gene": "ERBB2"}),
        ])
        assert text["V1"] == "BRAF V600E"
        assert text["V2"] == "ERBB2 Amplification"

    def test_no_duplicate_gene_prefix(self):
        text = build_node_text([FakeEntity("V1", "BRAF V600E", {"gene": "BRAF"})])
        assert text["V1"] == "BRAF V600E"

    def test_missing_gene_is_tolerated(self):
        for value in (None, "", "nan"):
            text = build_node_text([FakeEntity("V1", "V600E", {"gene": value})])
            assert text["V1"] == "V600E"

    def test_entities_without_names_are_skipped(self):
        assert build_node_text([FakeEntity("X", "  ")]) == {}


class TestEncoderContract:
    """No network access; these check wiring, not embedding quality."""

    class StubEncoder(NodeTextEncoder):
        def __init__(self):
            super().__init__(FeatureConfig(use_cache=False))
            self.calls = 0

        def _load_model(self):
            outer = self

            class Stub:
                def encode(self, texts, normalize_embeddings=False, **kwargs):
                    outer.calls += 1
                    # Distinct direction per text, so cosine is meaningful.
                    vectors = []
                    for text in texts:
                        v = np.zeros(4, dtype=np.float32)
                        v[len(text) % 4] = 1.0
                        v[0] += float(len(text))
                        vectors.append(v)
                    out = np.stack(vectors)
                    if normalize_embeddings:
                        out = out / np.linalg.norm(out, axis=1, keepdims=True)
                    return out
            return Stub()

    def test_encode_nodes_preserves_alignment(self):
        """A vector must land on the node whose text produced it."""
        encoder = self.StubEncoder()
        vectors = encoder.encode_nodes({"a": "xx", "b": "xxxx"})
        direct = encoder.encode(["xx", "xxxx"])
        assert np.allclose(vectors["a"], direct[0])
        assert np.allclose(vectors["b"], direct[1])

    def test_repeated_text_is_encoded_once(self):
        encoder = self.StubEncoder()
        encoder.encode(["same", "same", "same"])
        assert encoder.calls == 1

    def test_cached_text_is_not_re_encoded(self):
        encoder = self.StubEncoder()
        encoder.encode(["one"])
        encoder.encode(["one"])
        assert encoder.calls == 1

    def test_empty_mapping_returns_empty(self):
        assert self.StubEncoder().encode_nodes({}) == {}

    def test_default_model_is_the_measured_choice(self):
        """
        PubMedBERT measured 0.580 against MiniLM's 0.533 and BioBERT's 0.514 on
        name similarity alone, with disjoint intervals against MiniLM.
        """
        assert DEFAULT_MODEL.startswith("microsoft/BiomedNLP-PubMedBERT")


class TestFeatureOnlyPredictor:
    """
    This predictor exists to bound how much of any gain is name overlap rather
    than learned structure. It measures 0.581 on the real holdout -- above the
    0.498 floor but far below topology, so text is not carrying the result.
    """

    @pytest.fixture
    def predictor(self):
        model = TestEncoderContract.StubEncoder()
        return FeatureOnlyPredictor(
            node_text={"a": "aa", "b": "aa", "c": "cccccc"}, encoder=model
        ).fit()

    def test_identical_names_score_highest(self, predictor):
        assert predictor.score("a", "b") > predictor.score("a", "c")

    def test_unknown_node_scores_zero(self, predictor):
        assert predictor.score("a", "absent") == 0.0

    def test_scores_pairs_topology_cannot_reach(self, predictor):
        """
        Its one unique capability: on the 366 cold-start pairs, L3 scores 0 of
        366 while this scores all of them -- though at AUC 0.531 [0.501, 0.562],
        which is coverage without much signal.
        """
        assert predictor.score("a", "b") != 0.0

    def test_needs_no_graph(self, predictor):
        assert predictor.score_pairs([("a", "b"), ("a", "c")])
