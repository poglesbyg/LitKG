"""Tests for the GDC (Genomic Data Commons) integration."""

import json

import pytest

from litkg.evaluation.gdc_edges import (
    drop_leaked_edges,
    join_to_civic,
    load_gdc_edges,
    normalise_name,
)
from litkg.phase1.gdc_client import (
    FACET_BUCKET_CAP,
    GDCClient,
    GDCTruncationError,
    background_rate_enrichment,
)


class TestBackgroundRateEnrichment:
    """The length correction is the reason these edges mean anything."""

    def test_long_gene_is_not_enriched_by_length_alone(self):
        # Two genes, one ten times longer, both mutated in proportion to length.
        # Neither is enriched: that is exactly what proportional-to-length means.
        result = background_rate_enrichment(
            mutations={"SHORT": {"P": 10}, "LONG": {"P": 100}},
            cds_lengths={"SHORT": 1000, "LONG": 10000},
            cohort_cases={"P": 50},
        )
        by_symbol = {a.symbol: a for a in result}
        assert by_symbol["SHORT"].enrichment == pytest.approx(1.0)
        assert by_symbol["LONG"].enrichment == pytest.approx(1.0)

    def test_short_gene_mutated_often_is_enriched(self):
        result = background_rate_enrichment(
            mutations={"SHORT": {"P": 100}, "LONG": {"P": 100}},
            cds_lengths={"SHORT": 1000, "LONG": 10000},
            cohort_cases={"P": 50},
        )
        by_symbol = {a.symbol: a for a in result}
        assert by_symbol["SHORT"].enrichment > 5 * by_symbol["LONG"].enrichment

    def test_genes_without_a_length_are_excluded(self):
        # A missing CDS length cannot be corrected for, so the pair is dropped
        # rather than scored against an assumed length.
        result = background_rate_enrichment(
            mutations={"NOLEN": {"P": 100}, "OK": {"P": 10}},
            cds_lengths={"NOLEN": None, "OK": 1000},
            cohort_cases={"P": 50},
        )
        assert {a.symbol for a in result} == {"OK"}

    def test_cohort_with_no_cases_is_skipped(self):
        result = background_rate_enrichment(
            mutations={"G": {"EMPTY": 5}},
            cds_lengths={"G": 1000},
            cohort_cases={"EMPTY": 0},
        )
        assert result == []

    def test_enrichment_is_relative_within_a_cohort(self):
        # The same gene and count in a cohort with a heavier overall burden is
        # less remarkable, because the background it is measured against is
        # higher.
        result = background_rate_enrichment(
            mutations={"G": {"QUIET": 10, "NOISY": 10}, "OTHER": {"NOISY": 990}},
            cds_lengths={"G": 1000, "OTHER": 1000},
            cohort_cases={"QUIET": 50, "NOISY": 50},
        )
        by_project = {a.project_id: a for a in result if a.symbol == "G"}
        assert by_project["QUIET"].enrichment > by_project["NOISY"].enrichment


class TestLeakageGuard:
    """
    GDC edges land in the training backbone, so a coinciding test pair is a leak.
    """

    def test_edge_matching_a_test_pair_is_dropped(self):
        kept, dropped = drop_leaked_edges(
            [("gene:A", "disease:X"), ("gene:B", "disease:Y")],
            [("gene:A", "disease:X")],
        )
        assert kept == [("gene:B", "disease:Y")]
        assert dropped == [("gene:A", "disease:X")]

    def test_reversed_edge_is_still_a_leak(self):
        # The split treats a pair as unordered; an edge the other way round
        # hands over the same answer.
        kept, dropped = drop_leaked_edges(
            [("disease:X", "gene:A")],
            [("gene:A", "disease:X")],
        )
        assert kept == []
        assert len(dropped) == 1

    def test_guard_keeps_everything_when_nothing_overlaps(self):
        edges = [("gene:A", "disease:X"), ("gene:B", "disease:Y")]
        kept, dropped = drop_leaked_edges(edges, [("gene:C", "disease:Z")])
        assert kept == edges
        assert dropped == []


class TestJoinToCivic:
    def test_joins_on_normalised_names(self):
        joined, stats = join_to_civic(
            [("BRAF", "Thyroid Carcinoma", 300.0)],
            gene_ids_by_name={"BRAF": "CIVIC:GENE:ENTREZ:673"},
            disease_ids_by_name={"thyroid carcinoma": "CIVIC:DISEASE:DOID:1781"},
        )
        assert joined == [("CIVIC:GENE:ENTREZ:673", "CIVIC:DISEASE:DOID:1781")]
        assert stats["joined"] == 1

    def test_unmatched_disease_is_reported_not_guessed(self):
        # "Lung Adenocarcinoma" and "Lung Squamous Cell Carcinoma" have
        # different drivers. A near-match must not become an edge.
        joined, stats = join_to_civic(
            [("KRAS", "Lung Adenocarcinoma", 43.0)],
            gene_ids_by_name={"KRAS": "g1"},
            disease_ids_by_name={"Lung Squamous Cell Carcinoma": "d1"},
        )
        assert joined == []
        assert stats["disease_unmatched"] == 1

    def test_duplicate_pairs_collapse(self):
        joined, _ = join_to_civic(
            [("TP53", "Sarcoma", 10.0), ("TP53", "Sarcoma", 12.0)],
            gene_ids_by_name={"TP53": "g"},
            disease_ids_by_name={"Sarcoma": "d"},
        )
        assert joined == [("g", "d")]

    def test_normalise_folds_case_and_punctuation(self):
        assert normalise_name("Head and Neck Squamous-Cell Carcinoma") == \
            normalise_name("head and neck squamous cell carcinoma")


class TestGDCClient:
    def test_release_is_pinned_not_live(self, monkeypatch):
        # A pinned release is why a downstream number moves only when the code
        # moves.
        monkeypatch.delenv("LITKG_GDC_RELEASE", raising=False)
        assert GDCClient.pinned_release() == GDCClient.DEFAULT_RELEASE
        monkeypatch.setenv("LITKG_GDC_RELEASE", "99.9")
        assert GDCClient.pinned_release() == "99.9"

    def test_warning_in_a_200_response_is_an_error(self, tmp_path):
        # The GDC answers an unknown field with HTTP 200 plus a warning. Reading
        # the empty result as "no data" is how a query that asks for nothing
        # looks like a finding.
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": {"hits": []}, "warnings": {"fields": "unrecognized"}}

        class FakeSession:
            def get(self, *a, **k):
                return FakeResponse()

        client = GDCClient(cache_dir=tmp_path, session=FakeSession())
        with pytest.raises(ValueError, match="rejected part of the query"):
            client._get("genes", {})

    def test_truncated_facet_raises_rather_than_undercounting(self, tmp_path):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                buckets = [
                    {"key": f"k{i}", "doc_count": 1} for i in range(FACET_BUCKET_CAP)
                ]
                return {"data": {"aggregations": {"f": {"buckets": buckets}}},
                        "warnings": {}}

        class FakeSession:
            def get(self, *a, **k):
                return FakeResponse()

        client = GDCClient(cache_dir=tmp_path, session=FakeSession())
        with pytest.raises(GDCTruncationError, match="truncated"):
            client._facet_buckets("ssm_occurrences", {}, "f")

    def test_cache_is_keyed_by_release_and_program(self, tmp_path):
        tcga = GDCClient(cache_dir=tmp_path, release="46.0", program="TCGA")
        cptac = GDCClient(cache_dir=tmp_path, release="46.0", program="CPTAC")
        older = GDCClient(cache_dir=tmp_path, release="45.0", program="TCGA")
        paths = {
            tcga._cache_path("mutations_by_project"),
            cptac._cache_path("mutations_by_project"),
            older._cache_path("mutations_by_project"),
        }
        assert len(paths) == 3


class TestLoadGDCEdges:
    def test_missing_cache_names_the_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="projects.json"):
            load_gdc_edges(tmp_path, release="46.0", program="tcga")

    def test_thresholds_filter_pairs(self, tmp_path):
        base = tmp_path / "release-46.0"
        (base / "tcga").mkdir(parents=True)
        (base / "census_genes.json").write_text(json.dumps([
            {"symbol": "HOT", "canonical_transcript_length_cds": 1000},
            {"symbol": "COLD", "canonical_transcript_length_cds": 1000},
        ]))
        (base / "tcga" / "projects.json").write_text(json.dumps([
            {"project_id": "P", "name": "Test Cancer", "summary": {"case_count": 100}},
        ]))
        (base / "tcga" / "mutations_by_project.json").write_text(json.dumps({
            "HOT": {"P": 190}, "COLD": {"P": 10},
        }))

        edges = load_gdc_edges(tmp_path, release="46.0", program="tcga",
                               min_enrichment=1.5, min_occurrences=5)
        assert [e[0] for e in edges] == ["HOT"]
        assert edges[0][1] == "Test Cancer"
