"""
Tests for the end-to-end discovery pipeline.

The pipeline's job is to produce candidates a person can judge, so most of what
matters is that it does not overstate them: evidence is labelled by how well it
actually supports the pair, and the report says what the ranking is worth.
"""

from pathlib import Path

import pytest

from litkg.pipeline import Candidate, DiscoveryConfig, DiscoveryPipeline


def make_candidate(**overrides):
    defaults = dict(
        rank=1, source="CIVIC:DISEASE:DOID:1", target="CIVIC:VARIANT:2",
        source_name="Von Hippel-Lindau Disease", target_name="VHL Q164L",
        source_type="DISEASE", target_type="MUTATION",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


class TestCutoffSemantics:
    def test_no_cutoff_means_use_everything(self):
        """
        Proposing something new should use all available evidence. Only a
        validation run restricts to a past year.
        """
        assert DiscoveryConfig().cutoff is None

    def test_cutoff_is_carried_into_the_saved_output(self, tmp_path):
        pipeline = DiscoveryPipeline(DiscoveryConfig(cutoff=2016, output_dir=tmp_path))
        pipeline.candidates = [make_candidate()]
        import json
        payload = json.loads(pipeline.save().read_text())
        assert payload["cutoff"] == 2016


class TestEvidenceLabelling:
    """
    A passage mentioning one entity is background; one mentioning both is
    evidence for an association. Collapsing the distinction would let a
    candidate look supported when nothing connects its two halves.
    """

    def test_report_counts_co_mentions_separately(self, tmp_path):
        pipeline = DiscoveryPipeline(DiscoveryConfig(output_dir=tmp_path))
        pipeline.candidates = [
            make_candidate(rank=1, passages=[
                {"text": "t", "year": "2010", "support": "co-mention"}]),
            make_candidate(rank=2, passages=[
                {"text": "t", "year": "2010", "support": "single entity: VHL"}]),
            make_candidate(rank=3, passages=[]),
        ]
        report = pipeline.report()
        assert "3 candidates; 2 have literature, 1 have a paper mentioning both" in report

    def test_a_candidate_with_no_evidence_says_so(self, tmp_path):
        pipeline = DiscoveryPipeline(DiscoveryConfig(output_dir=tmp_path))
        pipeline.candidates = [make_candidate(passages=[])]
        assert "evidence: none retrieved" in pipeline.report()

    def test_explanation_refuses_without_evidence(self, tmp_path):
        """
        With nothing retrieved there is nothing to reason from, and the model
        must not be asked to justify the pair from memory.
        """
        pipeline = DiscoveryPipeline(DiscoveryConfig(output_dir=tmp_path))
        text = pipeline._explain(make_candidate(passages=[]))
        assert "nothing to justify" in text


class TestReportHonesty:
    def test_report_states_the_ranking_does_not_replicate(self, tmp_path):
        """
        Precision is 35x at a 2016 cutoff and 0x at 2020. A reader who sees only
        a ranked list will assume it is predictive unless told otherwise.
        """
        pipeline = DiscoveryPipeline(DiscoveryConfig(output_dir=tmp_path))
        pipeline.candidates = [make_candidate()]
        report = pipeline.report()
        assert "does not replicate" in report
        assert "candidates to judge" in report

    def test_known_outcomes_are_marked_when_validating(self, tmp_path):
        pipeline = DiscoveryPipeline(DiscoveryConfig(cutoff=2016, output_dir=tmp_path))
        pipeline.candidates = [make_candidate(known_outcome=True)]
        assert "curated after the cutoff" in pipeline.report()

    def test_no_outcome_marker_on_an_open_ended_run(self, tmp_path):
        """Nothing is known about a candidate proposed from all the data."""
        pipeline = DiscoveryPipeline(DiscoveryConfig(output_dir=tmp_path))
        pipeline.candidates = [make_candidate(known_outcome=None)]
        assert "curated after the cutoff" not in pipeline.report()


class TestOutputs:
    def test_save_writes_both_artifacts(self, tmp_path):
        pipeline = DiscoveryPipeline(DiscoveryConfig(output_dir=tmp_path))
        pipeline.candidates = [make_candidate()]
        path = pipeline.save()
        assert path.exists() and (tmp_path / "report.txt").exists()

    def test_candidate_round_trips(self):
        candidate = make_candidate(passages=[
            {"text": "t", "year": "2011", "support": "co-mention"}])
        payload = candidate.to_dict()
        assert payload["source_name"] == "Von Hippel-Lindau Disease"
        assert payload["passages"][0]["support"] == "co-mention"
