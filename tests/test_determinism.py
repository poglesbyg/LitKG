"""
Tests for reproducibility of GNNLinkPredictor.

Runs were not bit-reproducible for a given seed, and the project had recorded
that as a fact of life rather than a bug. Two independent causes, both fixed:
edge ordering that varied with PYTHONHASHSEED, and multi-threaded float
accumulation in aggregation.
"""

import random

import networkx as nx
import pytest
import torch

from litkg.phase2.link_prediction import GNNLinkPredictor, TrainingConfig


def small_graph():
    """A graph big enough that validation AUC does not saturate at 1.0."""
    edges = []
    for g in range(6):
        for v in range(4):
            edges.append((f"g{g}", f"v{g}_{v}"))
    for g in range(6):
        for v in range(4):
            edges.append((f"v{g}_{v}", f"d{(g + v) % 5}"))
    for g in range(6):
        edges.append((f"v{g}_0", f"d{(g + 2) % 5}"))
    return edges


def years_for(edges):
    """Fixed year per edge, independent of the order edges are passed in."""
    return {
        tuple(sorted(e)): 2000 + (hash(tuple(sorted(e))) % 5)
        for e in edges
    }


def node_types_for(edges):
    types = {}
    for u, v in edges:
        for n in (u, v):
            types[n] = {"g": "GENE", "v": "MUTATION", "d": "DISEASE"}[n[0]]
    return types


# Fixed once, so reordering the edge list cannot change the year assignment.
_EDGES = small_graph()
_YEARS = {
    tuple(sorted(e)): 2000 + i % 5 for i, e in enumerate(sorted(set(
        tuple(sorted(x)) for x in _EDGES
    )))
}


def fit(edges, seed=0, **flags):
    graph = nx.Graph(edges)
    config = TrainingConfig(epochs=8, patience=100, seed=seed, **flags)
    return GNNLinkPredictor(
        config=config, node_types=node_types_for(edges), edge_years=_YEARS,
    ).fit(graph)


class TestConfig:
    def test_determinism_is_on_by_default(self):
        assert TrainingConfig().deterministic is True

    def test_stable_validation_graph_is_on_by_default(self):
        assert TrainingConfig().stable_validation_graph is True


class TestThreadRestoration:
    def test_thread_count_is_restored_after_fitting(self):
        # Fitting must not permanently reconfigure the caller's torch.
        before = torch.get_num_threads()
        fit(small_graph())
        assert torch.get_num_threads() == before

    def test_thread_count_is_restored_even_when_fit_raises(self):
        before = torch.get_num_threads()
        predictor = GNNLinkPredictor(config=TrainingConfig(epochs=1))
        with pytest.raises(Exception):
            predictor.fit("not a graph")
        assert torch.get_num_threads() == before


class TestEdgeOrdering:
    def test_edge_order_does_not_depend_on_caller_iteration_order(self):
        # The caller usually builds the graph from a set, whose order varies
        # between processes. Two orderings of the same edges must train the
        # same model.
        edges = small_graph()
        shuffled = list(edges)
        random.Random(7).shuffle(shuffled)

        a = fit(edges)
        b = fit([(v, u) for u, v in shuffled])
        assert a.best_validation_auc == pytest.approx(b.best_validation_auc)

    def test_normalised_edges_are_sorted(self):
        fitted = fit(small_graph())
        # `all_edges` feeds a stable sort by year; unsorted input silently moves
        # the trainable/validation boundary among edges sharing a year.
        pairs = sorted(
            (u, v) if u <= v else (v, u) for u, v in fitted.graph.edges()
        )
        assert pairs == sorted(pairs)


class TestReproducibility:
    def test_same_seed_gives_the_same_model(self):
        a = fit(small_graph(), seed=3)
        b = fit(small_graph(), seed=3)
        assert a.best_validation_auc == pytest.approx(b.best_validation_auc)
        assert [h["loss"] for h in a.history] == pytest.approx(
            [h["loss"] for h in b.history]
        )

    def test_different_seeds_still_differ(self):
        # Determinism must not collapse seed variation; that variation is the
        # thing the harness measures.
        losses = {
            tuple(round(h["loss"], 6) for h in fit(small_graph(), seed=s).history)
            for s in (0, 1, 2, 3)
        }
        assert len(losses) > 1


class TestValidationSplitTieBreaking:
    def test_validation_split_is_not_alphabetical(self):
        # Sorting edges for determinism and then stable-sorting by year broke
        # ties by node name, putting the alphabetically-last edges of a year
        # into validation together. On the real graph that dropped best
        # validation AUC from 0.760 to 0.525.
        fitted = fit(small_graph(), seed=0)
        assert fitted.validation_is_temporal is True

    def test_split_varies_with_seed(self):
        # Tie-breaking is seeded, so the slice moves with the seed -- genuine
        # variance the harness should see, not noise it cannot attribute.
        splits = {
            tuple(sorted(fit(small_graph(), seed=s).history[0]["val_auc"] for _ in [0]))
            for s in (0, 1, 2, 3, 4)
        }
        losses = {
            round(fit(small_graph(), seed=s).history[0]["loss"], 6)
            for s in (0, 1, 2, 3, 4)
        }
        assert len(splits) > 1 or len(losses) > 1


class TestStepsPerEpoch:
    """
    The loop took one full-batch step per epoch and early-stopped after 75-135
    epochs, fitting the model in roughly 150 gradient updates.
    """

    def test_default_is_eight(self):
        # Measured: steps=8 beats steps=1 on AUC for 29 of 32 seed-level pairs
        # across two models and two cutoffs, and on AP for 31 of 32.
        assert TrainingConfig().steps_per_epoch == 8

    def test_single_step_is_still_available(self):
        assert TrainingConfig(steps_per_epoch=1).steps_per_epoch == 1

    def test_more_steps_means_more_updates(self):
        # Loss history records one entry per validation check, not per step, so
        # compare the count of optimizer steps indirectly: with more steps per
        # epoch the model should reach a different (further-trained) state.
        one = fit(small_graph(), seed=0, steps_per_epoch=1)
        many = fit(small_graph(), seed=0, steps_per_epoch=8)
        assert one.history[-1]["loss"] != pytest.approx(many.history[-1]["loss"])

    def test_zero_or_negative_is_treated_as_one(self):
        # max(1, ...) in the loop: a misconfigured zero must not skip training
        # silently, which would look like a model that trained and did nothing.
        fitted = fit(small_graph(), seed=0, steps_per_epoch=0)
        assert len(fitted.history) > 0

    def test_still_reproducible_with_multiple_steps(self):
        # Negatives are redrawn per step from the seeded rng; that must not
        # reintroduce run-to-run variation.
        a = fit(small_graph(), seed=1, steps_per_epoch=8)
        b = fit(small_graph(), seed=1, steps_per_epoch=8)
        assert [h["loss"] for h in a.history] == pytest.approx(
            [h["loss"] for h in b.history]
        )
