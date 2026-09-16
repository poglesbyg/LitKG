"""Tests for literature co-mention edges and the length-2 literature path score."""

import math

import networkx as nx
import pytest

from litkg.phase2.literature_edges import (
    CoMentionLinker,
    LiteraturePathPredictor,
    build_comention_edges,
)


@pytest.fixture
def linker():
    return CoMentionLinker(
        gene_symbols={"MET": "gene:MET", "BRAF": "gene:BRAF"},
        named_entities=[
            ("disease:melanoma", "Melanoma", ["malignant melanoma"]),
            ("drug:vemurafenib", "Vemurafenib", []),
        ],
    )


class TestLinker:
    def test_gene_symbols_match_case_sensitively(self, linker):
        # "met" the verb must not become the MET gene.
        assert "gene:MET" in linker.mentions("MET amplification was found.")
        assert "gene:MET" not in linker.mentions("The criteria were met.")

    def test_named_entities_match_case_insensitively(self, linker):
        assert "disease:melanoma" in linker.mentions("Patients with MELANOMA responded.")

    def test_synonyms_are_linked(self, linker):
        assert "disease:melanoma" in linker.mentions("malignant melanoma cohort")

    def test_gene_symbol_inside_a_longer_token_does_not_match(self, linker):
        assert "gene:BRAF" not in linker.mentions("BRAFi resistance")


class TestBuildEdges:
    def test_co_mentions_in_one_sentence_become_an_edge(self, linker):
        edges = build_comention_edges(
            ["BRAF inhibition with vemurafenib in melanoma."], linker
        )
        assert frozenset(("gene:BRAF", "drug:vemurafenib")) in edges
        assert frozenset(("gene:BRAF", "disease:melanoma")) in edges

    def test_mentions_in_different_sentences_do_not(self, linker):
        edges = build_comention_edges(
            ["BRAF was sequenced. Separately, melanoma was staged."], linker
        )
        assert frozenset(("gene:BRAF", "disease:melanoma")) not in edges

    def test_duplicate_abstracts_are_counted_once(self, linker):
        # The cache groups abstracts by the entity queried, so one paper usually
        # appears under several queries.
        text = "BRAF in melanoma."
        edges = build_comention_edges([text, text, text], linker)
        assert edges[frozenset(("gene:BRAF", "disease:melanoma"))] == 1

    def test_allowed_nodes_restricts_edges(self, linker):
        edges = build_comention_edges(
            ["BRAF in melanoma treated with vemurafenib."], linker,
            allowed_nodes={"gene:BRAF", "disease:melanoma"},
        )
        assert set(edges) == {frozenset(("gene:BRAF", "disease:melanoma"))}


class TestLiteraturePathPredictor:
    @pytest.fixture
    def graph(self):
        # variant -- gene is a training (CIVIC) edge.
        return nx.Graph([("variant", "gene"), ("other", "disease")])

    def test_path_through_a_literature_edge_scores(self, graph):
        lit = {frozenset(("gene", "disease")): 3}
        predictor = LiteraturePathPredictor(lit).fit(graph)
        assert predictor.score("variant", "disease") > 0

    def test_paths_made_only_of_training_edges_are_not_scored(self):
        # Those belong to the structural predictors, which already count them.
        graph = nx.Graph([("u", "a"), ("a", "v")])
        predictor = LiteraturePathPredictor({}).fit(graph)
        assert predictor.score("u", "v") == 0.0

    def test_literature_edge_that_duplicates_a_training_edge_adds_nothing(self):
        graph = nx.Graph([("u", "a"), ("a", "v")])
        predictor = LiteraturePathPredictor({frozenset(("a", "v")): 5}).fit(graph)
        assert predictor.score("u", "v") == 0.0

    def test_score_is_degree_normalised(self, graph):
        lit = {frozenset(("gene", "disease")): 1}
        predictor = LiteraturePathPredictor(lit).fit(graph)
        expected = 1.0 / math.sqrt(predictor.combined.degree("gene"))
        assert predictor.score("variant", "disease") == pytest.approx(expected)

    def test_score_is_symmetric(self, graph):
        lit = {frozenset(("gene", "disease")): 1}
        predictor = LiteraturePathPredictor(lit).fit(graph)
        assert predictor.score("variant", "disease") == predictor.score("disease", "variant")

    def test_literature_edges_to_nodes_outside_the_graph_are_dropped(self, graph):
        lit = {frozenset(("gene", "unseen")): 1}
        predictor = LiteraturePathPredictor(lit).fit(graph)
        assert "unseen" not in predictor.combined

    def test_unknown_nodes_score_zero(self, graph):
        predictor = LiteraturePathPredictor({}).fit(graph)
        assert predictor.score("variant", "nowhere") == 0.0

    def test_direct_co_mention_alone_is_not_a_path(self):
        graph = nx.Graph([("u", "x"), ("v", "y")])
        predictor = LiteraturePathPredictor({frozenset(("u", "v")): 9}).fit(graph)
        assert predictor.score("u", "v") == 0.0
