"""Tests for Disease Ontology lookup and the GDC variant predictor."""

import pytest

from litkg.evaluation.gdc_variant_predictor import GDCVariantCooccurrencePredictor
from litkg.phase1.disease_ontology import DiseaseOntology, normalise

OBO = """format-version: 1.2

[Term]
id: DOID:1
name: cancer

[Term]
id: DOID:2
name: lung cancer
is_a: DOID:1 ! cancer

[Term]
id: DOID:3
name: lung adenocarcinoma
synonym: "adenocarcinoma of lung" EXACT []
is_a: DOID:2 ! lung cancer

[Term]
id: DOID:4
name: melanoma
synonym: "cutaneous melanoma" EXACT []
is_a: DOID:1 ! cancer

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture
def ontology(tmp_path):
    path = tmp_path / "doid.obo"
    path.write_text(OBO)
    return DiseaseOntology(path)


class TestParsing:
    def test_terms_and_synonyms_are_indexed(self, ontology):
        assert ontology.doid_for("lung adenocarcinoma") == "DOID:3"
        assert ontology.doid_for("adenocarcinoma of lung") == "DOID:3"
        assert ontology.doid_for("Cutaneous Melanoma") == "DOID:4"

    def test_unknown_name_returns_none(self, ontology):
        assert ontology.doid_for("not a disease") is None

    def test_typedef_block_does_not_leak_into_terms(self, ontology):
        # A [Typedef] has a name too; treating it as a term would index
        # "part of" as a disease.
        assert ontology.doid_for("part of") is None

    def test_ancestors_are_nearest_first(self, ontology):
        assert ontology.ancestors("DOID:3") == [("DOID:2", 1), ("DOID:1", 2)]

    def test_normalise_folds_case_and_punctuation(self):
        assert normalise("Non-Small Cell  LUNG cancer") == "non small cell lung cancer"


class TestMatchToCivic:
    def test_direct_match_is_preferred(self, ontology):
        match = ontology.match_to_civic(
            "lung adenocarcinoma", {"DOID:3": "civic:3", "DOID:2": "civic:2"}
        )
        assert match.civic_id == "civic:3"
        assert match.via_ancestor is False

    def test_falls_back_to_nearest_ancestor(self, ontology):
        match = ontology.match_to_civic(
            "lung adenocarcinoma", {"DOID:1": "civic:1", "DOID:2": "civic:2"}
        )
        # Nearest, not any: DOID:2 is one step up, DOID:1 is two.
        assert match.civic_id == "civic:2"
        assert match.via_ancestor is True
        assert match.steps == 1

    def test_ancestor_match_can_be_refused(self, ontology):
        assert ontology.match_to_civic(
            "lung adenocarcinoma", {"DOID:1": "civic:1"}, allow_ancestors=False
        ) is None

    def test_unresolvable_name_is_none_not_a_guess(self, ontology):
        assert ontology.match_to_civic("mystery tumour", {"DOID:1": "civic:1"}) is None


class TestVariantPredictor:
    counts = {"BRAF p.V600E": {"TCGA-THCA": 283, "TCGA-SKCM": 200, "TCGA-LUAD": 8}}
    cases = {"TCGA-THCA": 500, "TCGA-SKCM": 470, "TCGA-LUAD": 585}
    keys = {"CIVIC:VARIANT:12": "BRAF p.V600E"}
    cohorts = {"TCGA-THCA": "d:thyroid", "TCGA-SKCM": "d:melanoma", "TCGA-LUAD": "d:lung"}

    def _predictor(self, mode):
        return GDCVariantCooccurrencePredictor(
            variant_counts=self.counts, cohort_cases=self.cases,
            variant_node_keys=self.keys, cohort_to_disease=self.cohorts, mode=mode,
        )

    def test_specificity_ranks_the_concentrated_cohort_first(self):
        p = self._predictor("specificity")
        assert p.score("CIVIC:VARIANT:12", "d:thyroid") > p.score("CIVIC:VARIANT:12", "d:lung")

    def test_score_is_symmetric(self):
        p = self._predictor("specificity")
        assert p.score("CIVIC:VARIANT:12", "d:thyroid") == p.score("d:thyroid", "CIVIC:VARIANT:12")

    def test_unknown_pair_scores_zero_rather_than_guessing(self):
        p = self._predictor("specificity")
        assert p.score("CIVIC:VARIANT:99", "d:thyroid") == 0.0

    def test_prevalence_normalises_by_cohort_size(self):
        p = self._predictor("prevalence")
        assert p.score("CIVIC:VARIANT:12", "d:thyroid") == pytest.approx(283 / 500)

    def test_unmapped_cohort_contributes_nothing(self):
        p = GDCVariantCooccurrencePredictor(
            variant_counts=self.counts, cohort_cases=self.cases,
            variant_node_keys=self.keys, cohort_to_disease={"TCGA-THCA": "d:thyroid"},
        )
        assert p.score("CIVIC:VARIANT:12", "d:melanoma") == 0.0

    def test_strongest_cohort_wins_when_several_map_to_one_disease(self):
        # Ontology ancestry can send two cohorts to the same CIVIC node.
        p = GDCVariantCooccurrencePredictor(
            variant_counts=self.counts, cohort_cases=self.cases,
            variant_node_keys=self.keys,
            cohort_to_disease={"TCGA-THCA": "d:any", "TCGA-LUAD": "d:any"},
            mode="specificity",
        )
        assert p.score("CIVIC:VARIANT:12", "d:any") == pytest.approx(283 / 491)

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            self._predictor("vibes")

    def test_coverage_reports_the_abstain_rate(self):
        p = self._predictor("specificity")
        pairs = [("CIVIC:VARIANT:12", "d:thyroid"), ("CIVIC:VARIANT:99", "d:lung")]
        assert p.coverage(pairs) == 0.5
