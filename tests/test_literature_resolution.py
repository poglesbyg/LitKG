"""
Tests for resolving literature mentions into entity-level nodes.
"""

import pytest

from litkg.phase1.literature_resolution import (
    LiteratureEntityResolver,
    ResolvedLiteratureEntity,
    aggregate_edges,
)


@pytest.fixture
def resolver():
    return LiteratureEntityResolver()


def add(resolver, pairs):
    """Register (text, label, document) mentions with generated keys."""
    for i, (text, label, doc) in enumerate(pairs):
        resolver.add_mention(f"m{i}", text, label, confidence=0.9, document_id=doc)
    return resolver


class TestMentionGrouping:
    def test_repeated_mentions_collapse_to_one_node(self, resolver):
        """The whole point: sixty BRCA1 mentions are one entity, not sixty."""
        add(resolver, [("BRCA1", "GENE", f"doc{i}") for i in range(60)])

        entities, mention_to_node = resolver.resolve()

        assert len(entities) == 1
        assert entities[0].mention_count == 60
        assert entities[0].document_count == 60
        assert len(set(mention_to_node.values())) == 1

    def test_punctuation_variants_merge(self, resolver):
        add(resolver, [("BRCA1", "GENE", "d1"), ("BRCA-1", "GENE", "d1"),
                       ("brca 1", "GENE", "d2")])

        entities, _ = resolver.resolve()

        assert len(entities) == 1
        assert entities[0].surface_forms == ["BRCA-1", "BRCA1", "brca 1"]

    def test_canonical_form_prefers_frequency_then_brevity(self, resolver):
        """Forms in a group differ only by incidental punctuation."""
        add(resolver, [("HER-2", "GENE", "d1"), ("HER2", "GENE", "d1")])

        entities, _ = resolver.resolve()

        assert entities[0].canonical_text == "HER2"

    def test_frequency_beats_brevity(self, resolver):
        add(resolver, [("HER-2", "GENE", "d1")] * 5 + [("HER2", "GENE", "d2")])

        entities, _ = resolver.resolve()

        assert entities[0].canonical_text == "HER-2"

    def test_same_string_different_types_stay_separate(self, resolver):
        """"ALL" is both a gene symbol and a leukemia; they are not one entity."""
        add(resolver, [("ALL", "GENE", "d1"), ("ALL", "DISEASE", "d2")])

        entities, _ = resolver.resolve()

        assert len(entities) == 2
        assert {e.label for e in entities} == {"GENE", "DISEASE"}

    def test_document_count_does_not_double_count(self, resolver):
        """A paper mentioning an entity five times is still one document."""
        add(resolver, [("TP53", "GENE", "same_doc")] * 5)

        entities, _ = resolver.resolve()

        assert entities[0].mention_count == 5
        assert entities[0].document_count == 1

    def test_every_mention_maps_to_a_node(self, resolver):
        add(resolver, [("BRCA1", "GENE", "d1"), ("TP53", "GENE", "d1"),
                       ("BRCA-1", "GENE", "d2")])

        _, mention_to_node = resolver.resolve()

        assert set(mention_to_node) == {"m0", "m1", "m2"}
        assert mention_to_node["m0"] == mention_to_node["m2"]

    def test_node_ids_are_stable_across_runs(self):
        """Ordering must not depend on dict iteration order."""
        pairs = [("TP53", "GENE", "d1"), ("BRCA1", "GENE", "d2"), ("EGFR", "GENE", "d3")]

        first = {e.canonical_text: e.node_id for e in add(LiteratureEntityResolver(), pairs).resolve()[0]}
        second = {e.canonical_text: e.node_id for e in add(LiteratureEntityResolver(), list(reversed(pairs))).resolve()[0]}

        assert first == second

    def test_short_mentions_can_be_filtered(self):
        """NER noise is dominated by very short acronyms."""
        resolver = LiteratureEntityResolver(min_surface_length=3)
        add(resolver, [("A", "GENE", "d1"), ("BRCA1", "GENE", "d1")])

        entities, _ = resolver.resolve()

        assert [e.canonical_text for e in entities] == ["BRCA1"]

    def test_empty_input(self, resolver):
        entities, mapping = resolver.resolve()
        assert entities == [] and mapping == {}


class TestEdgeAggregation:
    def test_duplicate_edges_become_one_with_support(self):
        """Sixty identical edges are one relationship asserted sixty times."""
        edges = [{"source": "a", "target": "b", "predicate": "TREATS", "confidence": 0.5}
                 for _ in range(60)]

        merged = aggregate_edges(edges)

        assert len(merged) == 1
        assert merged[0]["mention_count"] == 60

    def test_confidence_takes_the_strongest_evidence(self):
        edges = [
            {"source": "a", "target": "b", "predicate": "TREATS", "confidence": 0.4},
            {"source": "a", "target": "b", "predicate": "TREATS", "confidence": 0.9},
        ]

        merged = aggregate_edges(edges)

        assert merged[0]["confidence"] == 0.9

    def test_different_predicates_stay_distinct(self):
        edges = [
            {"source": "a", "target": "b", "predicate": "TREATS", "confidence": 0.5},
            {"source": "a", "target": "b", "predicate": "CAUSES", "confidence": 0.5},
        ]

        assert len(aggregate_edges(edges)) == 2

    def test_key_fields_are_configurable(self):
        """Entity links are identified by endpoints alone."""
        edges = [
            {"source": "a", "target": "b", "match_type": "exact", "confidence": 0.5},
            {"source": "a", "target": "b", "match_type": "fuzzy", "confidence": 0.7},
        ]

        merged = aggregate_edges(edges, key_fields=("source", "target"))

        assert len(merged) == 1
        assert merged[0]["mention_count"] == 2
        assert merged[0]["confidence"] == 0.7
