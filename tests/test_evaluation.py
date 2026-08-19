"""
Tests for the link prediction evaluation harness.

A harness that leaks is worse than no harness: it produces confident numbers
that are wrong, and nothing downstream can tell. Most of these tests target
leakage and degenerate cases rather than happy paths.
"""

import math

import networkx as nx
import numpy as np
import pytest

from litkg.evaluation import (
    AdamicAdarPredictor,
    CommonNeighborsPredictor,
    JaccardPredictor,
    L3PathPredictor,
    PreferentialAttachmentPredictor,
    RandomPredictor,
    build_temporal_split,
    evaluate_baselines,
    evaluate_scores,
    extract_publication_year,
    hits_at_k,
    mean_reciprocal_rank,
    sample_negatives,
)
from litkg.evaluation.harness import build_graph, structural_coverage
from litkg.evaluation.temporal_split import year_distribution


class TestPublicationYear:
    @pytest.mark.parametrize("citation,expected", [
        ("Levine et al., 2005", 2005),
        ("Lasota et al., 2004", 2004),
        ("Smith, 1998", 1998),
        ("Jones et al., 2023", 2023),
    ])
    def test_extracts_year(self, citation, expected):
        assert extract_publication_year(citation) == expected

    def test_takes_the_last_year_like_token(self):
        """Author lists can contain numbers; the year comes last."""
        assert extract_publication_year("Study 2 of 1000 patients, 2015") == 2015

    @pytest.mark.parametrize("citation", [None, "", "no year here", float("nan")])
    def test_missing_year_is_none(self, citation):
        assert extract_publication_year(citation) is None


class TestTemporalSplitLeakage:
    """The split exists to prevent leakage; these are its load-bearing cases."""

    def test_pair_asserted_before_and_after_stays_in_train(self):
        """
        A pair first published in 2010 and re-cited in 2020 is not a 2020
        discovery. Scoring it would credit the model for something it saw.
        """
        edges = [("a", "b", 2010), ("a", "b", 2020), ("c", "d", 2020)]
        split = build_temporal_split(edges, cutoff_year=2015)
        assert ("a", "b") in split.train_edges
        assert ("a", "b") not in split.test_edges

    def test_test_edges_never_appear_in_training(self):
        edges = [("a", "b", 2010), ("b", "c", 2020), ("a", "c", 2012)]
        split = build_temporal_split(edges, cutoff_year=2015)
        assert not (split.test_edges & split.train_edges)
        assert not (split.test_edges & split.backbone_edges)

    def test_backbone_edge_disqualifies_a_test_pair(self):
        """An edge already in the graph cannot be a prediction target."""
        edges = [("a", "b", 2020), ("a", "c", 2010), ("b", "c", 2010)]
        split = build_temporal_split(edges, cutoff_year=2015, backbone_edges=[("a", "b")])
        assert ("a", "b") not in split.test_edges
        assert split.excluded_already_known == 1

    def test_cold_start_pairs_are_excluded_and_counted(self):
        """A node absent from training cannot be scored by any topology."""
        edges = [("a", "b", 2010), ("y", "z", 2020)]
        split = build_temporal_split(edges, cutoff_year=2015)
        assert ("y", "z") not in split.test_edges
        assert split.excluded_cold_start == 1

    def test_direction_does_not_create_duplicates(self):
        edges = [("b", "a", 2010), ("a", "b", 2011)]
        split = build_temporal_split(edges, cutoff_year=2015)
        assert len(split.train_edges) == 1

    def test_self_loops_are_dropped(self):
        split = build_temporal_split([("a", "a", 2010), ("a", "b", 2010)], 2015)
        assert ("a", "a") not in split.train_edges

    def test_undated_edges_become_backbone(self):
        split = build_temporal_split([("a", "b", None), ("c", "d", 2020)], 2015)
        assert ("a", "b") in split.backbone_edges

    def test_earliest_year_wins(self):
        split = build_temporal_split([("a", "b", 2020), ("a", "b", 2001)], 2015)
        assert split.edge_years[("a", "b")] == 2001

    def test_year_distribution_counts_pairs_not_assertions(self):
        """Ten papers about one association are one association."""
        edges = [("a", "b", 2010)] * 10 + [("c", "d", 2011)]
        assert year_distribution(edges) == {2010: 1, 2011: 1}


class TestNegativeSampling:
    @pytest.fixture
    def graph(self):
        g = nx.Graph()
        g.add_edges_from([("g1", "d1"), ("g2", "d1"), ("g3", "d2"), ("g1", "d2")])
        return g

    @pytest.fixture
    def types(self):
        return {"g1": "GENE", "g2": "GENE", "g3": "GENE",
                "d1": "DISEASE", "d2": "DISEASE"}

    def test_never_samples_a_real_edge(self, graph, types):
        """A 'negative' that is a real edge is a mislabelled positive."""
        positives = [("g2", "d2")]
        negatives = sample_negatives(
            positives, graph, node_types=types,
            negatives_per_positive=20, known_edges=set(graph.edges()) | set(positives),
        )
        real = {(u, v) if u <= v else (v, u) for u, v in graph.edges()}
        assert not (set(negatives) & real)
        assert ("g2", "d2") not in negatives

    def test_type_matching_respects_endpoint_types(self, graph, types):
        """Untyped negatives make the task 'is this type pair plausible?'."""
        positives = [("g1", "d1")]
        negatives = sample_negatives(
            positives, graph, node_types=types, negatives_per_positive=5, seed=1
        )
        for u, v in negatives:
            assert {types[u], types[v]} == {"GENE", "DISEASE"}

    def test_no_self_loops(self, graph, types):
        negatives = sample_negatives(
            [("g1", "d1")], graph, node_types=types, negatives_per_positive=10
        )
        assert all(u != v for u, v in negatives)

    def test_sampling_is_deterministic(self, graph, types):
        kwargs = dict(node_types=types, negatives_per_positive=3, seed=7)
        first = sample_negatives([("g1", "d1")], graph, **kwargs)
        second = sample_negatives([("g1", "d1")], graph, **kwargs)
        assert first == second

    def test_degree_matching_changes_the_pool(self):
        """Degree matching is what separates structure from popularity."""
        g = nx.Graph()
        g.add_edges_from([("hub", f"n{i}") for i in range(32)])
        g.add_edge("low1", "low2")
        types = {n: "X" for n in g}
        matched = sample_negatives(
            [("hub", "low1")], g, node_types=types,
            negatives_per_positive=5, seed=3, degree_matched=True,
        )
        # A degree-32 endpoint must not be replaced by a degree-1 node.
        assert matched
        assert any(g.degree(u) > 4 or g.degree(v) > 4 for u, v in matched)


class TestPredictors:
    @pytest.fixture
    def graph(self):
        g = nx.Graph()
        g.add_edges_from([("a", "x"), ("b", "x"), ("a", "y"), ("b", "y"), ("c", "z")])
        return g

    def test_common_neighbors_counts_shared(self, graph):
        p = CommonNeighborsPredictor().fit(graph)
        assert p.score("a", "b") == 2.0
        assert p.score("a", "c") == 0.0

    def test_adamic_adar_downweights_hubs(self, graph):
        """A shared hub is weaker evidence than a shared rare neighbour."""
        hub = nx.Graph()
        hub.add_edges_from([("a", "h"), ("b", "h")] + [(f"n{i}", "h") for i in range(20)])
        rare = nx.Graph()
        rare.add_edges_from([("a", "r"), ("b", "r")])
        assert (AdamicAdarPredictor().fit(hub).score("a", "b")
                < AdamicAdarPredictor().fit(rare).score("a", "b"))

    def test_adamic_adar_handles_degree_one_neighbour(self, graph):
        """log(1) is 0; a naive implementation divides by zero here."""
        g = nx.Graph([("a", "s"), ("b", "s")])
        score = AdamicAdarPredictor().fit(g).score("a", "b")
        assert math.isfinite(score) and score > 0

    def test_jaccard_normalises(self, graph):
        assert JaccardPredictor().fit(graph).score("a", "b") == 1.0

    def test_preferential_attachment_ignores_shared_structure(self, graph):
        p = PreferentialAttachmentPredictor().fit(graph)
        assert p.score("a", "c") == 2.0 * 1.0

    def test_unknown_nodes_score_zero_not_crash(self, graph):
        for predictor in (AdamicAdarPredictor(), CommonNeighborsPredictor(),
                          JaccardPredictor(), PreferentialAttachmentPredictor()):
            assert predictor.fit(graph).score("nope", "a") == 0.0

    def test_random_is_reproducible(self, graph):
        a = RandomPredictor(seed=5).fit(graph).score_pairs([("a", "b"), ("a", "c")])
        b = RandomPredictor(seed=5).fit(graph).score_pairs([("a", "b"), ("a", "c")])
        assert a == b


class TestMetrics:
    def test_perfect_separation(self):
        m = evaluate_scores([0.9, 0.8], [0.1, 0.2])
        assert m.auc == 1.0
        assert m.hits_at_1 == 1.0
        assert m.mrr == 1.0

    def test_inverted_separation(self):
        assert evaluate_scores([0.1, 0.2], [0.9, 0.8]).auc == 0.0

    def test_all_ties_do_not_count_as_perfect(self):
        """
        A predictor scoring everything 0 ranks nothing. Optimistic tie handling
        would report Hits@1 of 1.0 for a predictor that has no information --
        the exact failure this harness must not have, since 85% of test pairs
        score 0 on shared-neighbour methods.
        """
        m = evaluate_scores([0.0] * 5, [0.0] * 50)
        assert m.hits_at_1 == 0.0
        assert m.mrr < 0.1

    def test_hits_at_k_and_mrr(self):
        assert hits_at_k([1, 2, 11], 10) == pytest.approx(2 / 3)
        assert mean_reciprocal_rank([1, 2]) == pytest.approx(0.75)

    def test_empty_inputs_do_not_crash(self):
        m = evaluate_scores([], [1.0])
        assert math.isnan(m.auc)


class TestStructuralCoverage:
    def test_reports_fraction_sharing_a_neighbour(self):
        g = nx.Graph([("a", "x"), ("b", "x"), ("c", "y")])
        assert structural_coverage(g, [("a", "b")]) == 1.0
        assert structural_coverage(g, [("a", "c")]) == 0.0
        assert structural_coverage(g, [("a", "b"), ("a", "c")]) == 0.5

    def test_missing_nodes_count_as_uncovered(self):
        g = nx.Graph([("a", "x")])
        assert structural_coverage(g, [("a", "absent")]) == 0.0


class TestHarnessEndToEnd:
    def test_random_predictor_lands_near_half(self):
        """If the harness leaks, random will not score 0.5."""
        edges = [(f"g{i}", f"d{i % 7}", 2000 + (i % 20)) for i in range(400)]
        split = build_temporal_split(edges, cutoff_year=2012)
        report = evaluate_baselines(split, negatives_per_positive=10, seed=0)
        if "random" in report.results and report.results["random"].positives:
            assert 0.35 < report.results["random"].auc < 0.65

    def test_empty_test_set_is_reported_not_crashed(self):
        split = build_temporal_split([("a", "b", 2000)], cutoff_year=2015)
        report = evaluate_baselines(split)
        assert not report.results
        assert any("No test edges" in n for n in report.notes)

    def test_low_coverage_produces_a_warning(self):
        """A metric that is undefined for most pairs must say so."""
        edges = [(f"g{i}", f"d{i}", 2000) for i in range(50)]
        edges += [(f"g{i}", f"d{i + 1}", 2020) for i in range(49)]
        split = build_temporal_split(edges, cutoff_year=2015)
        report = evaluate_baselines(split, negatives_per_positive=5)
        if report.results:
            assert "structural_coverage" in report.diagnostics


class TestHarnessRobustness:
    def test_negative_sampling_survives_absent_positive_endpoints(self):
        """
        networkx returns a DegreeView (not an error) for a node it does not
        contain, so an unguarded degree lookup fails with a confusing TypeError
        rather than a missing-node error. Splits built directly, bypassing the
        cold-start filter, hit this.
        """
        graph = nx.Graph([("a", "b"), ("b", "c")])
        negatives = sample_negatives(
            [("a", "absent")], graph, node_types={"a": "X", "b": "X", "c": "X"},
            negatives_per_positive=2, seed=0, degree_matched=True,
        )
        assert isinstance(negatives, list)


class TestL3PathPredictor:
    """The graph is multipartite, so length-2 methods are undefined on it."""

    def test_scores_cross_type_pairs_that_share_no_neighbour(self):
        """
        The case that matters: a bipartite chain a-x-b-y where a and y are two
        hops apart in type space. Adamic-Adar sees nothing; L3 sees the path.
        """
        g = nx.Graph([("a", "x"), ("x", "b"), ("b", "y")])
        assert AdamicAdarPredictor().fit(g).score("a", "y") == 0.0
        assert L3PathPredictor().fit(g).score("a", "y") > 0.0

    def test_more_paths_scores_higher(self):
        one = nx.Graph([("a", "x"), ("x", "b"), ("b", "y")])
        two = nx.Graph([("a", "x"), ("x", "b"), ("b", "y"),
                        ("a", "p"), ("p", "q"), ("q", "y")])
        assert (L3PathPredictor().fit(two).score("a", "y")
                > L3PathPredictor().fit(one).score("a", "y"))

    def test_hub_routes_count_for_less(self):
        """Degree normalisation is what stops this becoming a popularity score."""
        rare = nx.Graph([("a", "m"), ("m", "n"), ("n", "y")])
        hub = nx.Graph([("a", "m"), ("m", "n"), ("n", "y")])
        hub.add_edges_from([(f"e{i}", "m") for i in range(30)])
        hub.add_edges_from([(f"f{i}", "n") for i in range(30)])
        assert (L3PathPredictor().fit(hub).score("a", "y")
                < L3PathPredictor().fit(rare).score("a", "y"))

    def test_unknown_nodes_score_zero(self):
        g = nx.Graph([("a", "x"), ("x", "b")])
        assert L3PathPredictor().fit(g).score("absent", "a") == 0.0

    def test_does_not_score_a_pair_via_its_own_endpoint(self):
        """A path that doubles back through u is not a length-3 path to v."""
        g = nx.Graph([("u", "a"), ("a", "u")])
        assert L3PathPredictor().fit(g).score("u", "a") == 0.0

    def test_included_in_default_baselines(self):
        """A harness that omits it reports this graph as unpredictable."""
        from litkg.evaluation import BASELINE_PREDICTORS
        assert L3PathPredictor in BASELINE_PREDICTORS


class TestMultipartiteDiagnostic:
    def test_warns_when_graph_is_multipartite(self):
        edges, types = [], {}
        for i in range(60):
            g_id, d_id = f"g{i}", f"d{i % 8}"
            types[g_id], types[d_id] = "GENE", "DISEASE"
            edges.append((g_id, d_id, 2000 + (i % 10)))
        for i in range(60):
            edges.append((f"g{i}", f"d{(i + 3) % 8}", 2020))
        split = build_temporal_split(edges, cutoff_year=2015)
        report = evaluate_baselines(split, node_types=types, negatives_per_positive=5)
        if report.results:
            assert report.diagnostics["same_type_edge_ratio"] == 0.0
            assert any("multipartite" in n for n in report.notes)


class TestEvidenceWeights:
    """
    Flattening the graph discarded predicate, confidence and negation. These
    tests cover recovering them -- and the leak that recovery invites.
    """

    def test_weights_use_only_pre_cutoff_evidence(self):
        """
        The leak this invites: weighting an edge by evidence published after
        the cutoff feeds the model exactly the knowledge the holdout is meant
        to withhold. A pair's weight must reflect only what was known in time.
        """
        from litkg.evaluation import RelationRecord
        records = [
            RelationRecord("a", "b", 2010, "TREATS", 0.5),
            RelationRecord("a", "b", 2020, "TREATS", 1.0),   # after the cutoff
            RelationRecord("a", "b", 2021, "TREATS", 1.0),   # after the cutoff
        ]
        split = build_temporal_split(records, cutoff_year=2015)
        evidence = split.edge_evidence[("a", "b")]
        assert evidence.support == 1, "post-cutoff evidence leaked into the weight"
        assert evidence.mean_confidence == pytest.approx(0.5)

    def test_repeated_assertion_raises_weight_sublinearly(self):
        from litkg.evaluation import EdgeEvidence
        one = EdgeEvidence(support=1, total_confidence=0.8)
        ten = EdgeEvidence(support=10, total_confidence=8.0)
        assert ten.weight() > one.weight()
        # Ten assertions are not ten times the evidence.
        assert ten.weight() < 10 * one.weight()

    def test_negated_evidence_lowers_weight(self):
        """1731 CIVIC relations say the association does NOT hold."""
        from litkg.evaluation import EdgeEvidence
        supported = EdgeEvidence(support=4, total_confidence=3.2, negated_count=0)
        contested = EdgeEvidence(support=4, total_confidence=3.2, negated_count=4)
        assert contested.weight() < supported.weight()
        assert contested.weight() > 0, "contested is not the same as absent"

    def test_dominant_predicate_reported(self):
        from litkg.evaluation import EdgeEvidence
        evidence = EdgeEvidence(
            support=5, total_confidence=4.0,
            predicates={"SENSITIZES_TO": 4, "RESISTANT_TO": 1},
        )
        assert evidence.dominant_predicate == "SENSITIZES_TO"

    def test_plain_tuples_still_accepted(self):
        """Existing callers pass (source, target, year) triples."""
        split = build_temporal_split([("a", "b", 2010), ("c", "d", 2020)], 2015)
        assert ("a", "b") in split.train_edges

    def test_weighted_l3_differs_from_unweighted(self):
        from litkg.evaluation import L3PathPredictor, WeightedL3PathPredictor
        g = nx.Graph([("a", "x"), ("x", "b"), ("b", "y")])
        weights = {("a", "x"): 5.0, ("b", "x"): 5.0, ("b", "y"): 5.0}
        plain = L3PathPredictor().fit(g).score("a", "y")
        weighted = WeightedL3PathPredictor(weights=weights).fit(g).score("a", "y")
        assert weighted > plain

    def test_weighted_l3_defaults_to_one_for_unknown_edges(self):
        from litkg.evaluation import L3PathPredictor, WeightedL3PathPredictor
        g = nx.Graph([("a", "x"), ("x", "b"), ("b", "y")])
        assert (WeightedL3PathPredictor(weights={}).fit(g).score("a", "y")
                == pytest.approx(L3PathPredictor().fit(g).score("a", "y")))


class TestPerTypePairBreakdown:
    """
    One aggregate number averages four problems whose AUC ranges from 0.638 to
    0.802, hiding both progress and regressions.
    """

    @pytest.fixture
    def report(self):
        edges, types = [], {}
        for i in range(60):
            for suffix, kind in (("d", "DISEASE"), ("t", "DRUG")):
                node, other = f"v{i}", f"{suffix}{i % 5}"
                types[node], types[other] = "MUTATION", kind
                edges.append((node, other, 2000 + (i % 12)))
        split = build_temporal_split(edges, cutoff_year=2008)
        return evaluate_baselines(
            split, node_types=types, negatives_per_positive=5, seed=0
        )

    def test_breaks_results_down_by_pair(self, report):
        if report.results:
            assert report.per_type_pair
            assert all("-" in pair for pair in report.per_type_pair)

    def test_pair_labels_are_order_independent(self, report):
        """DISEASE-MUTATION and MUTATION-DISEASE are one group, not two."""
        for pair in report.per_type_pair:
            parts = pair.split("-")
            assert parts == sorted(parts)

    def test_group_sizes_sum_to_the_whole(self, report):
        if not report.per_type_pair:
            pytest.skip("no test edges in this fixture")
        name = next(iter(next(iter(report.per_type_pair.values()))))
        total = sum(r[name].positives for r in report.per_type_pair.values())
        assert total == report.results[name].positives

    def test_breakdown_absent_without_node_types(self):
        split = build_temporal_split(
            [(f"a{i}", f"b{i % 4}", 2000 + i % 10) for i in range(40)], 2005
        )
        assert not evaluate_baselines(split, negatives_per_positive=3).per_type_pair


class TestMetricUncertainty:
    """
    Ranking metrics here are set by a couple dozen rows out of ~1200, so a
    point estimate invites reading noise as a result.
    """

    def test_auc_matches_sklearn(self):
        """The Mann-Whitney decomposition must be exact, not approximate."""
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        positives = list(rng.normal(1.0, 1.0, 200))
        negatives = list(rng.normal(0.0, 1.0, 2000))
        labels = np.concatenate([np.ones(200), np.zeros(2000)])
        scores = np.concatenate([positives, negatives])
        mine = evaluate_scores(positives, negatives, bootstrap_samples=0).auc
        assert mine == pytest.approx(float(roc_auc_score(labels, scores)))

    def test_auc_decomposition_counts_ties_as_half(self):
        """Half credit for ties is what makes it agree with sklearn."""
        m = evaluate_scores([1.0], [1.0], bootstrap_samples=0)
        assert m.auc == pytest.approx(0.5)

    def test_intervals_are_produced_and_bracket_the_estimate(self):
        rng = np.random.default_rng(1)
        m = evaluate_scores(
            list(rng.normal(1.0, 1.0, 300)), list(rng.normal(0.0, 1.0, 3000))
        )
        for value, interval in (
            (m.auc, m.auc_ci), (m.mrr, m.mrr_ci),
            (m.average_precision, m.average_precision_ci),
            (m.hits_at_10, m.hits_at_10_ci),
        ):
            assert interval is not None
            assert interval[0] <= value <= interval[1]

    def test_intervals_narrow_with_more_positives(self):
        """A wide interval must be a statement about sample size."""
        rng = np.random.default_rng(2)
        negatives = list(rng.normal(0.0, 1.0, 3000))
        small = evaluate_scores(list(rng.normal(1.0, 1.0, 30)), negatives)
        large = evaluate_scores(list(rng.normal(1.0, 1.0, 3000)), negatives)
        assert (large.auc_ci[1] - large.auc_ci[0]) < (small.auc_ci[1] - small.auc_ci[0])

    def test_bootstrap_can_be_disabled(self):
        m = evaluate_scores([1.0, 2.0], [0.0, 0.5], bootstrap_samples=0)
        assert m.auc_ci is None and m.mrr_ci is None

    def test_bootstrap_is_reproducible(self):
        rng = np.random.default_rng(3)
        positives = list(rng.normal(1.0, 1.0, 100))
        negatives = list(rng.normal(0.0, 1.0, 1000))
        assert (evaluate_scores(positives, negatives, seed=7).auc_ci
                == evaluate_scores(positives, negatives, seed=7).auc_ci)

    def test_hits_at_100_has_resolution_where_hits_at_10_does_not(self):
        """
        Only ~26 of 1204 real positives reach the top 10 of ~12000, so Hits@10
        barely moves. A coarser cutoff has to be strictly more inclusive.
        """
        rng = np.random.default_rng(4)
        m = evaluate_scores(
            list(rng.normal(0.5, 1.0, 400)), list(rng.normal(0.0, 1.0, 4000))
        )
        assert m.hits_at_100 >= m.hits_at_10

    def test_indistinguishable_fraction_flags_mass_ties(self):
        """
        Shared-neighbour predictors score exactly 0 for most pairs. Their
        ranking metrics describe an undefined score, and the report must say
        so rather than presenting it as a weak result.
        """
        m = evaluate_scores([0.0] * 10, [0.0] * 100, bootstrap_samples=0)
        assert m.indistinguishable_fraction == pytest.approx(1.0)

        clear = evaluate_scores([5.0] * 10, [0.0] * 100, bootstrap_samples=0)
        assert clear.indistinguishable_fraction == 0.0

    def test_all_ties_still_do_not_count_as_perfect(self):
        m = evaluate_scores([0.0] * 5, [0.0] * 50, bootstrap_samples=0)
        assert m.hits_at_1 == 0.0
        assert m.mrr < 0.1

    def test_summary_line_shows_intervals(self):
        rng = np.random.default_rng(5)
        text = evaluate_scores(
            list(rng.normal(1.0, 1.0, 50)), list(rng.normal(0.0, 1.0, 500))
        ).summary()
        assert "AUC" in text and "[" in text


class TestRetrievalQuerySet:
    """
    Judgements come from CIVIC: every evidence row cites a paper and states the
    relationship it supports, so cited papers are relevant to a question about
    that relationship on a curator's judgement rather than ours.
    """

    @pytest.fixture
    def evidence(self):
        import pandas as pd
        rows = []
        for i in range(4):
            rows.append({
                "source_type": "PubMed", "citation_id": f"1000{i}",
                "molecular_profile": "BRAF V600E", "disease": "Melanoma",
                "evidence_type": "Predictive",
            })
        rows.append({
            "source_type": "PubMed", "citation_id": "20001",
            "molecular_profile": "KRAS G12C", "disease": "Lung Cancer",
            "evidence_type": "Prognostic",
        })
        rows.append({
            "source_type": "ASCO", "citation_id": "30001",
            "molecular_profile": "BRAF V600E", "disease": "Melanoma",
            "evidence_type": "Predictive",
        })
        return pd.DataFrame(rows)

    def test_groups_below_the_threshold_are_dropped(self, evidence):
        from litkg.evaluation import QuerySetBuilder
        queries = QuerySetBuilder().build(evidence, min_relevant=3)
        assert len(queries) == 1
        assert queries[0].profile == "BRAF V600E"

    def test_only_pubmed_citations_are_judged(self, evidence):
        """An ASCO abstract has no PMID to retrieve, so it cannot be a target."""
        from litkg.evaluation import QuerySetBuilder
        queries = QuerySetBuilder().build(evidence, min_relevant=3)
        assert "30001" not in queries[0].relevant_pmids

    def test_query_phrasing_follows_the_evidence_type(self, evidence):
        """
        Asking "which therapies" of a prognostic paper would score papers as
        misses for a question they were never cited to answer.
        """
        from litkg.evaluation import QuerySetBuilder
        from litkg.evaluation.retrieval import QUERY_TEMPLATES
        queries = QuerySetBuilder().build(evidence, min_relevant=3)
        assert queries[0].text == QUERY_TEMPLATES["Predictive"].format(
            profile="BRAF V600E", disease="Melanoma"
        )

    def test_sampling_is_deterministic(self, evidence):
        from litkg.evaluation import QuerySetBuilder
        a = QuerySetBuilder().build(evidence, min_relevant=3, max_queries=1, seed=7)
        b = QuerySetBuilder().build(evidence, min_relevant=3, max_queries=1, seed=7)
        assert [q.text for q in a] == [q.text for q in b]

    def test_round_trips_through_disk(self, evidence, tmp_path):
        from litkg.evaluation import QuerySetBuilder, load_queries, save_queries
        queries = QuerySetBuilder().build(evidence, min_relevant=3)
        save_queries(queries, tmp_path / "q.json")
        assert [q.to_dict() for q in load_queries(tmp_path / "q.json")] == \
               [q.to_dict() for q in queries]


class TestRetrievalMetrics:
    @pytest.fixture
    def query(self):
        from litkg.evaluation import RetrievalQuery
        return RetrievalQuery(query_id="q0", text="q", relevant_pmids=["1", "2", "3"])

    def test_perfect_ranking(self, query):
        from litkg.evaluation import evaluate_retrieval
        m = evaluate_retrieval(lambda q: ["1", "2", "3"], [query], k=3,
                               bootstrap_samples=0)
        assert m.precision_at_k == pytest.approx(1.0)
        assert m.recall_at_k == pytest.approx(1.0)
        assert m.mrr == pytest.approx(1.0)
        assert m.ndcg_at_k == pytest.approx(1.0)

    def test_nothing_retrieved(self, query):
        from litkg.evaluation import evaluate_retrieval
        m = evaluate_retrieval(lambda q: [], [query], k=3, bootstrap_samples=0)
        assert m.precision_at_k == 0.0 and m.mrr == 0.0 and m.hit_rate == 0.0

    def test_mrr_uses_the_first_relevant_rank(self, query):
        from litkg.evaluation import evaluate_retrieval
        m = evaluate_retrieval(lambda q: ["x", "1"], [query], k=5, bootstrap_samples=0)
        assert m.mrr == pytest.approx(0.5)

    def test_recall_is_capped_by_k(self, query):
        """Three relevant papers cannot all be found in a top-1 list."""
        from litkg.evaluation import evaluate_retrieval
        m = evaluate_retrieval(lambda q: ["1", "2", "3"], [query], k=1,
                               bootstrap_samples=0)
        assert m.recall_at_k == pytest.approx(1 / 3)

    def test_intervals_are_over_queries(self, query):
        from litkg.evaluation import RetrievalQuery, evaluate_retrieval
        queries = [RetrievalQuery(query_id=f"q{i}", text="q",
                                  relevant_pmids=["1"]) for i in range(20)]
        m = evaluate_retrieval(lambda q: ["1"], queries, k=5)
        assert m.precision_ci is not None


class TestCandidateEnumeration:
    """
    Every link-prediction number before this scored held-out positives against
    ~10 sampled negatives per positive. The real task ranks every unobserved
    pair -- about a million here -- so sampled-negative AUC is an optimistic
    proxy for what a user actually reads.
    """

    @pytest.fixture
    def enumerate_candidates(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "rank_predictions",
            pathlib.Path(__file__).parent.parent / "scripts" / "rank_predictions.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.candidate_pairs

    def test_excludes_pairs_already_in_the_graph(self, enumerate_candidates):
        """An observed edge is not a prediction."""
        g = nx.Graph([("m1", "d1"), ("d1", "m2"), ("m2", "d2")])
        types = {"m1": "MUTATION", "m2": "MUTATION", "d1": "DISEASE", "d2": "DISEASE"}
        known = {("d1", "m1"), ("d1", "m2"), ("d2", "m2")}
        found = enumerate_candidates(g, types, {("DISEASE", "MUTATION")}, known)
        assert not (found & known)

    def test_only_returns_requested_type_pairs(self, enumerate_candidates):
        """
        Ranking a type combination the held-out period never contains would pad
        the denominator with pairs that cannot be scored as correct.
        """
        g = nx.Graph([("m1", "d1"), ("d1", "m2"), ("m2", "t1")])
        types = {"m1": "MUTATION", "m2": "MUTATION",
                 "d1": "DISEASE", "t1": "DRUG"}
        found = enumerate_candidates(g, types, {("DISEASE", "MUTATION")}, set())
        for u, v in found:
            assert tuple(sorted((types[u], types[v]))) == ("DISEASE", "MUTATION")

    def test_pairs_are_normalised(self, enumerate_candidates):
        g = nx.Graph([("m1", "d1"), ("d1", "m2"), ("m2", "d2")])
        types = {"m1": "MUTATION", "m2": "MUTATION", "d1": "DISEASE", "d2": "DISEASE"}
        found = enumerate_candidates(g, types, {("DISEASE", "MUTATION")}, set())
        assert all(u <= v for u, v in found)

    def test_no_self_pairs(self, enumerate_candidates):
        g = nx.Graph([("a", "b"), ("b", "c"), ("c", "a")])
        types = {n: "MUTATION" for n in "abc"}
        found = enumerate_candidates(g, types, {("MUTATION", "MUTATION")}, set())
        assert all(u != v for u, v in found)
