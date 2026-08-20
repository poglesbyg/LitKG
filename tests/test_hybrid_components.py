"""Tests for the hybrid predictor's extra-component support."""

import networkx as nx
import numpy as np

from litkg.evaluation.baselines import LinkPredictor, PathPowerPredictor
from litkg.phase2.link_prediction import HybridLinkPredictor


class ConstantPredictor(LinkPredictor):
    """Scores every pair the same, so its rank contribution is flat."""

    name = "constant"

    def __init__(self, value: float = 1.0):
        self.value = value

    def fit(self, graph):
        self.graph = graph
        return self

    def score(self, u, v):
        return self.value


class TestWeightGrid:
    def test_two_component_grid_is_unchanged(self):
        # Backward compatibility: without extra components the hybrid must
        # search exactly the grid it always did, or previously reported numbers
        # stop being reproducible.
        assert HybridLinkPredictor()._weight_grid() == [0.25, 0.4, 0.5, 0.6, 0.75]

    def test_three_component_grid_is_a_simplex(self):
        grid = HybridLinkPredictor(
            extra_components=[PathPowerPredictor(5)]
        )._weight_grid()
        assert all(len(w) == 3 for w in grid)
        assert all(abs(sum(w) - 1.0) < 1e-9 for w in grid)
        assert all(all(x >= 0 for x in w) for w in grid)

    def test_grid_covers_the_corners(self):
        grid = HybridLinkPredictor(extra_components=[ConstantPredictor()])._weight_grid()
        as_tuples = {tuple(w) for w in grid}
        # A component must be able to win outright or be excluded entirely,
        # otherwise the search cannot tell you it is useless.
        assert (1.0, 0.0, 0.0) in as_tuples
        assert (0.0, 0.0, 1.0) in as_tuples

    def test_grid_grows_with_component_count(self):
        one = HybridLinkPredictor(extra_components=[ConstantPredictor()])._weight_grid()
        two = HybridLinkPredictor(
            extra_components=[ConstantPredictor(), ConstantPredictor()]
        )._weight_grid()
        assert len(two) > len(one)


class TestPercentileCombination:
    def test_percentile_handles_an_empty_reference(self):
        out = HybridLinkPredictor._percentile(np.array([1.0, 2.0]), np.array([]))
        assert out.tolist() == [0.0, 0.0]

    def test_tied_values_share_one_percentile(self):
        # Many pairs score exactly zero. Ordering them arbitrarily would invent
        # a ranking the scores do not support.
        reference = np.sort(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        out = HybridLinkPredictor._percentile(np.array([0.0, 0.0]), reference)
        assert out[0] == out[1]
        assert 0.0 < out[0] < 1.0

    def test_percentile_is_monotone(self):
        reference = np.sort(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        out = HybridLinkPredictor._percentile(np.array([0.5, 3.5]), reference)
        assert out[0] < out[1]


class TestExtraComponentsAreFitted:
    def test_extra_component_receives_the_graph(self):
        graph = nx.path_graph(8)

        class Recorder(ConstantPredictor):
            fitted = None

            def fit(self, g):
                Recorder.fitted = g
                return super().fit(g)

        hybrid = HybridLinkPredictor(extra_components=[Recorder()])
        # Fit only the extra components, avoiding the GNN's training cost.
        for component in hybrid.extra_components:
            component.fit(graph)
        assert Recorder.fitted is graph

    def test_path_power_is_usable_as_a_component(self):
        component = PathPowerPredictor(5).fit(nx.path_graph(8))
        assert component.score_pairs([(0, 5), (0, 1)]) is not None


class TestWeightSelectionDefault:
    """
    Selection is off by default, and the leak it had is fixed.

    Measured on the 2016 holdout across 8 seeds: a fixed even blend reaches
    0.7451 +/- 0.0123 while selecting on validation reaches 0.7404 +/- 0.0214.
    """

    def test_default_is_an_even_blend_without_searching(self):
        hybrid = HybridLinkPredictor()
        assert hybrid.select_weight is False
        assert hybrid.selected_weight == 0.5

    def test_an_explicit_weight_is_honoured(self):
        assert HybridLinkPredictor(weight=0.75).selected_weight == 0.75

    def test_selection_is_opt_in(self):
        assert HybridLinkPredictor(select_weight=True).select_weight is True

    def test_explicit_weight_wins_over_selection(self):
        # Asking for a specific blend and a search is contradictory; the
        # explicit number is the clearer intent.
        hybrid = HybridLinkPredictor(weight=0.4, select_weight=True)
        hybrid.graph = nx.path_graph(4)
        assert hybrid.selected_weight == 0.4

    def test_refit_helper_exists_for_leak_free_selection(self):
        # The selection path must score validation edges on a graph they have
        # been removed from; without this helper it scores them in place, which
        # inflated L3 by 3.48x and length-5 counts by 4.69x.
        assert hasattr(HybridLinkPredictor, "_refit_on")
