"""
Tests for literature relation extraction.

The previous implementation matched regexes whose capture groups had to
coincide with entity text, requiring both entities to be single words directly
adjacent to the trigger verb. It extracted zero relations from all 100 sample
documents. These tests pin the cases that failed.
"""

import pytest

from litkg.phase1.literature_processor import BiomedicalNLP, Entity


@pytest.fixture
def nlp():
    """BiomedicalNLP without loading models; only sentence-level logic is used."""
    return BiomedicalNLP.__new__(BiomedicalNLP)


def entity(text, label, start, end):
    return Entity(text=text, label=label, start=start, end=end, confidence=0.9)


def extract(nlp, sentence, spans):
    """Build entities from (text, label) pairs located within the sentence."""
    entities = []
    for text, label in spans:
        start = sentence.index(text)
        entities.append(entity(text, label, start, start + len(text)))
    return nlp._extract_relations_from_sentence(sentence, entities, 0)


class TestTriggerExtraction:
    def test_multiword_entities_separated_from_the_trigger(self, nlp):
        """The exact shape that produced zero relations before."""
        sentence = "BRCA1 mutations are associated with breast cancer."

        relations = extract(nlp, sentence, [("BRCA1", "GENE"), ("breast cancer", "DISEASE")])

        assert len(relations) == 1
        assert relations[0].predicate == "ASSOCIATED_WITH"
        assert relations[0].subject.text == "BRCA1"
        assert relations[0].object.text == "breast cancer"

    def test_adverb_between_entity_and_trigger(self, nlp):
        """"is strongly associated with" previously captured ("strongly", "non")."""
        sentence = "EGFR is strongly associated with non-small cell lung cancer."

        relations = extract(nlp, sentence, [("EGFR", "GENE"), ("non-small cell lung cancer", "DISEASE")])

        assert [r.predicate for r in relations] == ["ASSOCIATED_WITH"]

    def test_direct_trigger(self, nlp):
        sentence = "Olaparib treats BRCA1 tumors."

        relations = extract(nlp, sentence, [("Olaparib", "DRUG"), ("BRCA1", "GENE")])

        assert relations[0].predicate == "TREATS"
        assert relations[0].subject.text == "Olaparib"

    def test_passive_voice_reverses_direction(self, nlp):
        """"A treated with B" means B treats A, not the reverse."""
        sentence = "Tumors were treated with olaparib."

        relations = extract(nlp, sentence, [("Tumors", "DISEASE"), ("olaparib", "DRUG")])

        assert relations[0].predicate == "TREATS"
        assert relations[0].subject.text == "olaparib"
        assert relations[0].object.text == "Tumors"

    def test_longest_trigger_wins(self, nlp):
        """"treated with" must beat the substring "treats" style match."""
        sentence = "Patients were treated with erlotinib."

        relations = extract(nlp, sentence, [("Patients", "DISEASE"), ("erlotinib", "DRUG")])

        assert relations[0].subject.text == "erlotinib"

    @pytest.mark.parametrize("sentence,subject,obj,expected", [
        ("KRAS drives pancreatic cancer.", "KRAS", "pancreatic cancer", "CAUSES"),
        ("Imatinib inhibits BCR-ABL.", "Imatinib", "BCR-ABL", "INHIBITS"),
        ("MYC activates proliferation.", "MYC", "proliferation", "ACTIVATES"),
        ("TP53 is mutated in lung cancer.", "TP53", "lung cancer", "MUTATED_IN"),
        ("HER2 is overexpressed in gastric cancer.", "HER2", "gastric cancer", "EXPRESSED_IN"),
    ])
    def test_relation_vocabulary(self, nlp, sentence, subject, obj, expected):
        relations = extract(nlp, sentence, [(subject, "GENE"), (obj, "DISEASE")])

        assert relations, f"no relation extracted from {sentence!r}"
        assert relations[0].predicate == expected


class TestPrecisionGuards:
    def test_negation_suppresses_the_relation(self, nlp):
        """A negated trigger asserts the opposite of what it names."""
        sentence = "BRCA1 was not associated with this outcome."

        assert extract(nlp, sentence, [("BRCA1", "GENE"), ("outcome", "DISEASE")]) == []

    def test_distant_entities_are_not_related(self, nlp):
        """A trigger far from both entities usually relates a different pair."""
        filler = " and many other considerations were discussed at length" * 3
        sentence = f"BRCA1 was described{filler}, associated with breast cancer."

        relations = extract(nlp, sentence, [("BRCA1", "GENE"), ("breast cancer", "DISEASE")])

        assert relations == []

    def test_no_trigger_yields_no_relation(self, nlp):
        """Co-occurrence alone is not an asserted relationship."""
        sentence = "BRCA1 and breast cancer were both mentioned."

        assert extract(nlp, sentence, [("BRCA1", "GENE"), ("breast cancer", "DISEASE")]) == []

    def test_entity_does_not_relate_to_itself(self, nlp):
        sentence = "BRCA1 is associated with BRCA1 expression."
        entities = [entity("BRCA1", "GENE", 0, 5), entity("BRCA1", "GENE", 25, 30)]

        assert nlp._extract_relations_from_sentence(sentence, entities, 0) == []

    def test_pair_count_is_bounded(self, nlp):
        """Enumerations must not blow up quadratically."""
        names = [f"GENE{i}" for i in range(40)]
        sentence = " associated with ".join(names)
        entities, cursor = [], 0
        for name in names:
            start = sentence.index(name, cursor)
            entities.append(entity(name, "GENE", start, start + len(name)))
            cursor = start + len(name)

        relations = nlp._extract_relations_from_sentence(sentence, entities, 0)

        assert len(relations) <= nlp.MAX_PAIRS_PER_SENTENCE

    def test_confidence_reflects_trigger_distance(self, nlp):
        near = extract(nlp, "BRCA1 causes cancer.", [("BRCA1", "GENE"), ("cancer", "DISEASE")])
        far_sentence = "BRCA1 protein, a well studied tumour suppressor gene, causes cancer."
        far = extract(nlp, far_sentence, [("BRCA1", "GENE"), ("cancer", "DISEASE")])

        assert near[0].confidence > far[0].confidence
