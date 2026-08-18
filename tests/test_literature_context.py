"""
Tests for literature-context features.

This is the only feature source in the project that can leak: a name is static
metadata, but an abstract encodes a discovery, and a 2024 abstract may state
outright the association a 2016 holdout asks the model to predict. Most of
these tests are about the date filter.
"""

import json

import pytest

from litkg.phase2.literature_context import (
    ContextConfig,
    LiteratureContextFetcher,
    _publication_year,
    _sentences_mentioning,
)


class TestSentenceExtraction:
    def test_keeps_only_sentences_mentioning_the_entity(self):
        text = "BRAF V600E drives melanoma. Unrelated sentence about something else."
        found = _sentences_mentioning(text, "BRAF")
        assert len(found) == 1 and "BRAF" in found[0]

    def test_matching_is_word_bounded(self):
        """"ALK" must not match inside "alkaline"."""
        assert _sentences_mentioning("Alkaline phosphatase was elevated here.", "ALK") == []

    def test_matching_is_case_insensitive(self):
        assert _sentences_mentioning("Treatment with imatinib was effective here.", "Imatinib")

    def test_drops_fragments_and_runaway_sentences(self):
        assert _sentences_mentioning("BRAF.", "BRAF") == []
        assert _sentences_mentioning("BRAF " + "x" * 900, "BRAF") == []


class TestPublicationYear:
    def test_reads_a_structured_year(self):
        info = {"Journal": {"JournalIssue": {"PubDate": {"Year": "2011"}}}}
        assert _publication_year(info) == 2011

    def test_reads_a_medline_date_string(self):
        info = {"Journal": {"JournalIssue": {"PubDate": {"MedlineDate": "2011 Nov-Dec"}}}}
        assert _publication_year(info) == 2011

    def test_missing_date_returns_none(self):
        """A record with no date must be droppable, not assumed in range."""
        assert _publication_year({"Journal": {}}) is None


class TestLeakageGuards:
    """The date filter is the whole reason this module is safe to use."""

    def test_query_maxdate_is_the_year_before_the_cutoff(self):
        """
        PubMed's maxdate is inclusive, so passing the cutoff itself would admit
        papers published in the first year of the test period.
        """
        captured = {}

        class StubEntrez:
            email = None
            api_key = None

            @staticmethod
            def esearch(**kwargs):
                captured.update(kwargs)
                return _StubHandle({"IdList": []})

            @staticmethod
            def read(handle):
                return handle.payload

        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._entrez = lambda: StubEntrez
        fetcher._throttle = lambda: None
        fetcher._search("BRAF")
        assert captured["maxdate"] == "2015"
        assert captured["datetype"] == "pdat"

    def test_records_at_or_after_the_cutoff_are_dropped(self):
        """
        The query filter is trusted but verified. A record that slips through
        with a post-cutoff date must be discarded, not embedded.
        """
        records = {"PubmedArticle": [
            _article("Pre-cutoff finding here.", 2011),
            _article("Post-cutoff finding here.", 2020),
            _article("Exactly at cutoff here.", 2016),
        ]}

        class StubEntrez:
            email = None
            api_key = None

            @staticmethod
            def efetch(**kwargs):
                return _StubHandle(records)

            @staticmethod
            def read(handle):
                return handle.payload

        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._entrez = lambda: StubEntrez
        fetcher._throttle = lambda: None
        articles = fetcher._fetch_abstracts(["1", "2", "3"])
        assert [a["year"] for a in articles] == ["2011"]

    def test_cache_is_keyed_by_cutoff(self):
        """
        Raising the cutoff must not reuse contexts gathered under a looser
        filter -- that would leak silently and permanently.
        """
        early = LiteratureContextFetcher(ContextConfig(cutoff_year=2012))._cache_path()
        late = LiteratureContextFetcher(ContextConfig(cutoff_year=2018))._cache_path()
        assert early != late
        assert "2012" in early.name and "2018" in late.name

    def test_cache_with_a_mismatched_cutoff_is_rejected(self, tmp_path):
        path = tmp_path / "abstracts_pre2016.json"
        path.write_text(json.dumps({
            "cutoff_year": 2020,
            "abstracts": {"braf": ["A leaked 2020 abstract about BRAF."]},
        }))
        fetcher = LiteratureContextFetcher(
            ContextConfig(cutoff_year=2016, cache_dir=tmp_path)
        )
        fetcher.load()
        assert fetcher._abstracts == {}


class TestCacheStoresRawAbstracts:
    """
    The cache holds abstracts, not extracted sentences. Caching derived data
    meant any change to the matcher forced a refetch of every entity from
    PubMed -- which is exactly what happened when the matcher had to be relaxed.
    """

    def test_cache_filename_marks_the_abstract_format(self):
        path = LiteratureContextFetcher(ContextConfig(cutoff_year=2016))._cache_path()
        assert path.name.startswith("abstracts_pre")

    def test_matcher_changes_need_no_refetch(self):
        """Sentences are derived on read, so the same cache serves both forms."""
        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._loaded = True
        fetcher._abstracts = {"flt3 itd": ["FLT3-ITD confers poor prognosis in AML."]}
        assert fetcher.contexts_for_cached("FLT3 ITD")

    def test_cached_lookup_never_fetches(self):
        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._loaded = True
        fetcher._abstracts = {}
        fetcher._search = lambda *a, **k: pytest.fail("must not fetch")
        assert fetcher.contexts_for_cached("ANYTHING") == []


class TestFlexibleMentionMatching:
    """
    Requiring the literal gene-qualified string found almost nothing: 13% of
    multi-word names against 62% of single-word names. Abstracts hyphenate
    ("FLT3-ITD") and usually drop the gene ("V600E").
    """

    def test_separator_variants_match(self):
        for written in ("FLT3-ITD", "FLT3 ITD", "FLT3_ITD"):
            assert _sentences_mentioning(
                f"{written} confers poor prognosis in this cohort.", "FLT3 ITD"
            )

    def test_specifier_alone_matches(self):
        assert _sentences_mentioning(
            "The V600E substitution activates the kinase strongly.", "BRAF V600E"
        )

    def test_gene_alone_does_not_match_a_variant(self):
        """
        Sentences about the gene say nothing about this variant, and the graph
        already carries a gene-variant edge for that relationship.
        """
        assert not _sentences_mentioning(
            "BRAF is a serine threonine kinase of interest here.", "BRAF V600E"
        )


class TestContextText:
    def test_falls_back_to_the_name_without_context(self):
        """
        A node with no retrieved sentences must degrade to the name-only
        feature, not to an empty string.
        """
        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._loaded = True
        fetcher._abstracts = {"braf": ["BRAF drives melanoma in many patients."]}
        text = fetcher.context_text({"g1": "BRAF", "g2": "OBSCURE1"})
        assert "melanoma" in text["g1"]
        assert text["g2"] == "OBSCURE1"

    def test_name_is_retained_alongside_context(self):
        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._loaded = True
        fetcher._abstracts = {"braf": ["BRAF is a kinase implicated in melanoma."]}
        assert text_starts_with(fetcher.context_text({"g1": "BRAF"})["g1"], "BRAF")

    def test_coverage_counts_only_entities_with_sentences(self):
        fetcher = LiteratureContextFetcher(ContextConfig(cutoff_year=2016, cache_dir=None))
        fetcher._loaded = True
        fetcher._abstracts = {
            "braf": ["BRAF drives melanoma in many patients."],
            "kras": [],
        }
        coverage = fetcher.coverage({"a": "BRAF", "b": "KRAS", "c": "TP53"})
        assert coverage == pytest.approx(1 / 3)


def text_starts_with(value: str, prefix: str) -> bool:
    return value.startswith(prefix)


class _StubHandle:
    def __init__(self, payload):
        self.payload = payload

    def close(self):
        pass


def _article(text: str, year: int):
    return {"MedlineCitation": {"Article": {
        "ArticleTitle": "Title",
        "Abstract": {"AbstractText": [text]},
        "Journal": {"JournalIssue": {"PubDate": {"Year": str(year)}}},
    }}}
