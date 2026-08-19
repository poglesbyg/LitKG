"""Tests for STRING PPI loading and the path-power predictor."""

import gzip

import networkx as nx
import pytest

from litkg.evaluation.baselines import PathPowerPredictor
from litkg.phase1.string_ppi import LITERATURE_DERIVED, StringPPI

INFO = "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n" + "".join(
    f"9606.ENSP{i:05d}\t{sym}\t500\t-\n"
    for i, sym in enumerate(["TP53", "BRCA1", "EGFR", "KRAS"])
)

# protein1 protein2 neighborhood fusion cooccurence coexpression experimental
#          database textmining combined_score
_HEADER = (
    "protein1 protein2 neighborhood fusion cooccurence coexpression "
    "experimental database textmining combined_score\n"
)
LINKS = _HEADER + "\n".join([
    # Real experimental evidence.
    "9606.ENSP00002 9606.ENSP00003 0 0 0 100 800 900 500 950",
    # Textmining only: the edge that must not appear by default.
    "9606.ENSP00000 9606.ENSP00001 0 0 0 0 0 0 900 900",
    # Below threshold on experiments.
    "9606.ENSP00000 9606.ENSP00002 0 0 0 0 100 0 100 150",
    # Self-loop after symbol mapping is dropped.
    "9606.ENSP00003 9606.ENSP00003 0 0 0 0 900 0 0 900",
]) + "\n"


@pytest.fixture
def ppi(tmp_path):
    store = StringPPI(tmp_path)
    store.info_path.write_bytes(gzip.compress(INFO.encode()))
    store.links_path.write_bytes(gzip.compress(LINKS.encode()))
    return store


class TestChannelGuard:
    def test_textmining_and_database_are_literature_derived(self):
        assert LITERATURE_DERIVED == {"textmining", "database"}

    def test_literature_channels_are_refused_by_default(self, ppi):
        # These read the papers the CIVIC labels come from. Using them predicts
        # the answer from the answer.
        with pytest.raises(ValueError, match="derived from the literature"):
            ppi.edges(channels=("textmining",))
        with pytest.raises(ValueError, match="derived from the literature"):
            ppi.edges(channels=("experimental", "database"))

    def test_literature_channels_can_be_used_deliberately(self, ppi):
        edges = ppi.edges(
            channels=("textmining",), min_score=400, allow_literature_channels=True
        )
        # Orientation is normalised, so the pair sorts as (BRCA1, TP53).
        assert ("BRCA1", "TP53") in {(e.gene_a, e.gene_b) for e in edges}

    def test_default_channels_exclude_the_textmining_only_edge(self, ppi):
        pairs = {(e.gene_a, e.gene_b) for e in ppi.edges(min_score=400)}
        assert ("EGFR", "KRAS") in pairs        # experimental 800
        assert ("BRCA1", "TP53") not in pairs   # textmining only
        assert ("TP53", "EGFR") not in pairs    # experimental 100, below 400

    def test_unknown_channel_is_an_error(self, ppi):
        with pytest.raises(ValueError, match="unknown STRING channels"):
            ppi.edges(channels=("vibes",))


class TestEdgeLoading:
    def test_v12_info_header_is_understood(self, ppi):
        # v11 called this protein_external_id, v12 string_protein_id. Pinning
        # one name made every channel return zero edges in silence.
        mapping = ppi.protein_to_symbol()
        assert mapping["9606.ENSP00000"] == "TP53"

    def test_self_loops_are_dropped(self, ppi):
        assert all(e.gene_a != e.gene_b for e in ppi.edges(min_score=400))

    def test_keep_symbols_restricts_to_the_graph(self, ppi):
        edges = ppi.edges(keep_symbols={"EGFR"}, min_score=400)
        assert edges == []

    def test_edges_are_orientation_normalised(self, ppi):
        edges = ppi.edges(min_score=400)
        assert all(e.gene_a < e.gene_b for e in edges)

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="info file missing"):
            StringPPI(tmp_path).protein_to_symbol()

    def test_channel_report_exposes_the_textmining_share(self, ppi):
        report = ppi.channel_report(min_score=400)
        assert report["textmining"] >= 1
        assert report["experimental"] >= 1


class TestPathPowerPredictor:
    def test_rejects_a_degenerate_length(self):
        with pytest.raises(ValueError, match="at least 2"):
            PathPowerPredictor(1)

    def test_names_itself_by_path_length(self):
        assert PathPowerPredictor(5).name == "l5_path_power"

    def test_scores_a_reachable_pair_above_an_unreachable_one(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (10, 11)])
        predictor = PathPowerPredictor(5).fit(graph)
        assert predictor.score(0, 5) > 0
        assert predictor.score(0, 11) == 0.0

    def test_unknown_node_scores_zero(self):
        predictor = PathPowerPredictor(3).fit(nx.path_graph(4))
        assert predictor.score("absent", 1) == 0.0

    def test_is_symmetric(self):
        predictor = PathPowerPredictor(3).fit(nx.path_graph(6))
        assert predictor.score(0, 3) == pytest.approx(predictor.score(3, 0))

    def test_isolated_node_does_not_divide_by_zero(self):
        graph = nx.Graph([(0, 1)])
        graph.add_node(99)
        predictor = PathPowerPredictor(3).fit(graph)
        assert predictor.score(99, 0) == 0.0

    def test_length_three_cannot_see_a_length_five_route(self):
        # The structural point the predictor exists for: a route that needs
        # five hops contributes nothing at three.
        graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        assert PathPowerPredictor(3).fit(graph).score(0, 5) == 0.0
        assert PathPowerPredictor(5).fit(graph).score(0, 5) > 0
