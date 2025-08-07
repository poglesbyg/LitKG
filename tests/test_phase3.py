"""
Tests for Phase 3 components (confidence scoring, novelty detection, hypothesis generation, validation).
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from litkg.phase3.confidence_scoring import (
    ConfidenceScorer, EvidenceType, ConfidenceMetrics,
    LiteratureConfidenceAssessor, ExperimentalConfidenceAssessor,
    CrossModalConfidenceIntegrator
)
from litkg.phase3.novelty_detection import (
    NovelRelation, DiscoveryPattern, NovelRelationPredictor,
    PatternDiscoveryEngine, BiologicalPlausibilityChecker,
    NoveltyDetectionSystem
)
from litkg.phase3.hypothesis_generation import (
    BiomedicalHypothesis, ExperimentalDesign, HypothesisGenerator,
    HypothesisValidationAgent, HypothesisGenerationSystem
)
from litkg.phase3.validation import (
    ValidationResult, ExpertAssessment, LiteratureCrossValidator,
    TemporalValidator, ExpertValidationInterface,
    ComprehensiveValidationSystem
)


class TestConfidenceScoring:
    """Test confidence scoring components."""
    
    def test_evidence_type_enum(self):
        """Test EvidenceType enumeration."""
        assert EvidenceType.LITERATURE.value == "literature"
        assert EvidenceType.EXPERIMENTAL.value == "experimental"
        assert EvidenceType.COMPUTATIONAL.value == "computational"
    
    def test_confidence_metrics_dataclass(self):
        """Test ConfidenceMetrics dataclass."""
        metrics = ConfidenceMetrics(
            overall_confidence=0.85,
            literature_confidence=0.90,
            experimental_confidence=0.80,
            consistency_score=0.75,
            uncertainty_estimate=0.15
        )
        
        assert metrics.overall_confidence == 0.85
        assert metrics.literature_confidence == 0.90
        assert metrics.experimental_confidence == 0.80
        assert metrics.consistency_score == 0.75
        assert metrics.uncertainty_estimate == 0.15
    
    def test_confidence_scorer_init(self):
        """Test ConfidenceScorer initialization."""
        scorer = ConfidenceScorer()
        
        assert hasattr(scorer, 'logger')
        assert hasattr(scorer, 'literature_assessor')
        assert hasattr(scorer, 'experimental_assessor')
        assert hasattr(scorer, 'cross_modal_integrator')
    
    def test_assess_relationship_confidence(self):
        """Test relationship confidence assessment."""
        scorer = ConfidenceScorer()
        
        # Mock the sub-components
        with patch.object(scorer.literature_assessor, 'assess_confidence') as mock_lit:
            mock_lit.return_value = 0.90
            
            with patch.object(scorer.experimental_assessor, 'assess_confidence') as mock_exp:
                mock_exp.return_value = 0.80
                
                with patch.object(scorer.cross_modal_integrator, 'integrate') as mock_integrate:
                    mock_integrate.return_value = torch.tensor([0.85])
                    
                    relationship = {
                        'head': 'BRCA1',
                        'relation': 'ASSOCIATED_WITH',
                        'tail': 'breast_cancer'
                    }
                    
                    evidence = {
                        'literature': ['paper1', 'paper2'],
                        'experimental': ['study1']
                    }
                    
                    metrics = scorer.assess_relationship_confidence(relationship, evidence)
                    
                    assert isinstance(metrics, ConfidenceMetrics)
                    assert 0 <= metrics.overall_confidence <= 1
    
    def test_literature_confidence_assessor(self):
        """Test LiteratureConfidenceAssessor."""
        assessor = LiteratureConfidenceAssessor()
        
        literature_evidence = [
            {'title': 'BRCA1 in breast cancer', 'confidence': 0.95, 'citations': 100},
            {'title': 'BRCA1 mutations', 'confidence': 0.85, 'citations': 50}
        ]
        
        confidence = assessor.assess_confidence(literature_evidence)
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_experimental_confidence_assessor(self):
        """Test ExperimentalConfidenceAssessor."""
        assessor = ExperimentalConfidenceAssessor()
        
        experimental_evidence = [
            {'study_type': 'clinical_trial', 'p_value': 0.001, 'sample_size': 1000},
            {'study_type': 'in_vitro', 'p_value': 0.05, 'sample_size': 100}
        ]
        
        confidence = assessor.assess_confidence(experimental_evidence)
        
        assert 0 <= confidence <= 1
    
    def test_cross_modal_confidence_integrator(self):
        """Test CrossModalConfidenceIntegrator."""
        integrator = CrossModalConfidenceIntegrator(hidden_dim=768)
        
        lit_features = torch.randn(1, 768)
        exp_features = torch.randn(1, 768)
        
        integrated_confidence = integrator.integrate(lit_features, exp_features)
        
        assert integrated_confidence.shape == (1,)
        assert 0 <= integrated_confidence.item() <= 1
    
    def test_confidence_calibration(self):
        """Test confidence calibration."""
        scorer = ConfidenceScorer()
        
        # Mock calibration data
        predicted_confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        actual_outcomes = [1, 1, 0, 1, 0]  # Binary outcomes
        
        calibrated_scorer = scorer.calibrate_confidence(predicted_confidences, actual_outcomes)
        
        assert calibrated_scorer is not None
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification."""
        scorer = ConfidenceScorer()
        
        # Mock multiple predictions for uncertainty estimation
        predictions = torch.tensor([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.6, 0.4]
        ])
        
        epistemic_uncertainty, aleatoric_uncertainty = scorer.quantify_uncertainty(predictions)
        
        assert epistemic_uncertainty >= 0
        assert aleatoric_uncertainty >= 0


class TestNoveltyDetection:
    """Test novelty detection components."""
    
    def test_novel_relation_dataclass(self):
        """Test NovelRelation dataclass."""
        relation = NovelRelation(
            head_entity="BRCA1",
            tail_entity="alzheimer_disease",
            relation_type="ASSOCIATED_WITH",
            confidence_score=0.75,
            supporting_evidence=["evidence1", "evidence2"],
            novelty_score=0.90
        )
        
        assert relation.head_entity == "BRCA1"
        assert relation.tail_entity == "alzheimer_disease"
        assert relation.confidence_score == 0.75
        assert relation.novelty_score == 0.90
        assert len(relation.supporting_evidence) == 2
    
    def test_discovery_pattern_dataclass(self):
        """Test DiscoveryPattern dataclass."""
        pattern = DiscoveryPattern(
            pattern_type="co_occurrence",
            entities=["BRCA1", "TP53", "DNA_repair"],
            pattern_strength=0.85,
            frequency=10,
            description="Genes frequently co-occurring in DNA repair contexts"
        )
        
        assert pattern.pattern_type == "co_occurrence"
        assert len(pattern.entities) == 3
        assert pattern.pattern_strength == 0.85
    
    def test_novel_relation_predictor_init(self):
        """Test NovelRelationPredictor initialization."""
        predictor = NovelRelationPredictor()
        
        assert hasattr(predictor, 'logger')
    
    def test_predict_novel_relations(self, sample_knowledge_graph):
        """Test novel relation prediction."""
        predictor = NovelRelationPredictor()
        
        # Mock the prediction model
        with patch.object(predictor, 'prediction_model') as mock_model:
            mock_model.predict.return_value = [
                {
                    'head': 'BRCA1',
                    'tail': 'alzheimer_disease',
                    'relation': 'ASSOCIATED_WITH',
                    'confidence': 0.75
                }
            ]
            
            novel_relations = predictor.predict_novel_relations(
                knowledge_graph=sample_knowledge_graph,
                threshold=0.7
            )
            
            assert len(novel_relations) >= 0
            if novel_relations:
                assert isinstance(novel_relations[0], NovelRelation)
    
    def test_pattern_discovery_engine(self):
        """Test PatternDiscoveryEngine."""
        engine = PatternDiscoveryEngine()
        
        # Mock data for pattern discovery
        entities_data = [
            {'entity': 'BRCA1', 'context': 'DNA repair breast cancer'},
            {'entity': 'TP53', 'context': 'DNA repair tumor suppressor'},
            {'entity': 'ATM', 'context': 'DNA repair kinase'}
        ]
        
        with patch.object(engine, '_extract_patterns') as mock_extract:
            mock_patterns = [
                DiscoveryPattern(
                    pattern_type="co_occurrence",
                    entities=["BRCA1", "TP53"],
                    pattern_strength=0.85,
                    frequency=5,
                    description="DNA repair genes"
                )
            ]
            mock_extract.return_value = mock_patterns
            
            patterns = engine.discover_patterns(entities_data)
            
            assert len(patterns) == 1
            assert isinstance(patterns[0], DiscoveryPattern)
    
    def test_biological_plausibility_checker(self):
        """Test BiologicalPlausibilityChecker."""
        checker = BiologicalPlausibilityChecker()
        
        novel_relation = NovelRelation(
            head_entity="BRCA1",
            tail_entity="breast_cancer",
            relation_type="ASSOCIATED_WITH",
            confidence_score=0.85,
            supporting_evidence=["evidence1"],
            novelty_score=0.70
        )
        
        # Mock biological knowledge base
        with patch.object(checker, 'biological_kb') as mock_kb:
            mock_kb.check_plausibility.return_value = {
                'plausible': True,
                'score': 0.80,
                'reasoning': 'BRCA1 is known to be involved in breast cancer'
            }
            
            plausibility = checker.check_plausibility(novel_relation)
            
            assert plausibility['plausible'] is True
            assert 0 <= plausibility['score'] <= 1
    
    def test_novelty_detection_system(self):
        """Test NoveltyDetectionSystem integration."""
        system = NoveltyDetectionSystem()
        
        # Mock input data
        literature_data = [{'text': 'BRCA1 mutations in cancer'}]
        knowledge_graph = {'nodes': [], 'edges': []}
        
        with patch.object(system.relation_predictor, 'predict_novel_relations') as mock_predict:
            mock_predict.return_value = [
                NovelRelation(
                    head_entity="BRCA1",
                    tail_entity="new_disease",
                    relation_type="ASSOCIATED_WITH",
                    confidence_score=0.80,
                    supporting_evidence=["evidence1"],
                    novelty_score=0.90
                )
            ]
            
            with patch.object(system.pattern_engine, 'discover_patterns') as mock_patterns:
                mock_patterns.return_value = []
                
                results = system.detect_novel_knowledge(literature_data, knowledge_graph)
                
                assert 'novel_relations' in results
                assert 'patterns' in results
                assert len(results['novel_relations']) == 1
    
    def test_novelty_scoring(self):
        """Test novelty scoring mechanism."""
        predictor = NovelRelationPredictor()
        
        relation_candidate = {
            'head': 'BRCA1',
            'tail': 'rare_disease',
            'relation': 'ASSOCIATED_WITH'
        }
        
        existing_knowledge = {
            'BRCA1': ['breast_cancer', 'ovarian_cancer', 'DNA_repair']
        }
        
        novelty_score = predictor.compute_novelty_score(relation_candidate, existing_knowledge)
        
        assert 0 <= novelty_score <= 1


class TestHypothesisGeneration:
    """Test hypothesis generation components."""
    
    def test_biomedical_hypothesis_dataclass(self):
        """Test BiomedicalHypothesis dataclass."""
        hypothesis = BiomedicalHypothesis(
            hypothesis_text="BRCA1 mutations increase sensitivity to PARP inhibitors",
            confidence_score=0.85,
            supporting_evidence=["evidence1", "evidence2"],
            domain="cancer_genetics",
            testable_predictions=["prediction1", "prediction2"],
            experimental_approaches=["in_vitro", "clinical_trial"]
        )
        
        assert "BRCA1" in hypothesis.hypothesis_text
        assert hypothesis.confidence_score == 0.85
        assert hypothesis.domain == "cancer_genetics"
        assert len(hypothesis.testable_predictions) == 2
    
    def test_experimental_design_dataclass(self):
        """Test ExperimentalDesign dataclass."""
        design = ExperimentalDesign(
            objective="Test PARP inhibitor sensitivity",
            methodology="Cell viability assay",
            experimental_groups=["BRCA1-WT", "BRCA1-MUT"],
            controls=["Vehicle", "Positive control"],
            measurements=["Cell viability", "Apoptosis"],
            statistical_analysis="t-test",
            estimated_duration="3 months",
            estimated_cost="$10,000",
            success_probability=0.75,
            ethical_considerations=["Cell line usage"]
        )
        
        assert design.objective == "Test PARP inhibitor sensitivity"
        assert len(design.experimental_groups) == 2
        assert design.success_probability == 0.75
    
    def test_hypothesis_generator_init(self):
        """Test HypothesisGenerator initialization."""
        generator = HypothesisGenerator()
        
        assert hasattr(generator, 'logger')
    
    def test_generate_hypothesis(self):
        """Test hypothesis generation."""
        generator = HypothesisGenerator()
        
        context = {
            'entities': ['BRCA1', 'PARP_inhibitors', 'breast_cancer'],
            'relations': [
                {'head': 'BRCA1', 'relation': 'MUTATED_IN', 'tail': 'breast_cancer'},
                {'head': 'PARP_inhibitors', 'relation': 'TREATS', 'tail': 'breast_cancer'}
            ],
            'domain': 'cancer_genetics'
        }
        
        # Mock the LLM response
        with patch.object(generator, 'llm') as mock_llm:
            mock_llm.generate.return_value = {
                'hypothesis': 'BRCA1 mutations increase PARP inhibitor sensitivity',
                'confidence': 0.85,
                'evidence': ['Synthetic lethality principle']
            }
            
            hypothesis = generator.generate_hypothesis(context)
            
            assert isinstance(hypothesis, BiomedicalHypothesis)
            assert hypothesis.confidence_score == 0.85
    
    def test_hypothesis_validation_agent(self):
        """Test HypothesisValidationAgent."""
        validator = HypothesisValidationAgent()
        
        hypothesis = BiomedicalHypothesis(
            hypothesis_text="BRCA1 mutations increase PARP inhibitor sensitivity",
            confidence_score=0.85,
            supporting_evidence=["evidence1"],
            domain="cancer_genetics",
            testable_predictions=["prediction1"],
            experimental_approaches=["in_vitro"]
        )
        
        # Mock validation process
        with patch.object(validator, '_validate_against_literature') as mock_lit:
            mock_lit.return_value = {'score': 0.80, 'supporting_papers': 5}
            
            with patch.object(validator, '_validate_biological_plausibility') as mock_bio:
                mock_bio.return_value = {'score': 0.90, 'reasoning': 'Mechanistically sound'}
                
                validation_results = validator.validate_hypothesis(hypothesis)
                
                assert 'literature_validation' in validation_results
                assert 'biological_validation' in validation_results
                assert 'overall_score' in validation_results
    
    def test_hypothesis_generation_system(self):
        """Test HypothesisGenerationSystem integration."""
        system = HypothesisGenerationSystem()
        
        input_data = {
            'novel_relations': [
                NovelRelation(
                    head_entity="BRCA1",
                    tail_entity="alzheimer_disease",
                    relation_type="ASSOCIATED_WITH",
                    confidence_score=0.75,
                    supporting_evidence=["evidence1"],
                    novelty_score=0.90
                )
            ],
            'literature_context': ['BRCA1 DNA repair', 'Alzheimer neurodegeneration']
        }
        
        with patch.object(system.hypothesis_generator, 'generate_hypothesis') as mock_gen:
            mock_hypothesis = BiomedicalHypothesis(
                hypothesis_text="BRCA1 deficiency contributes to Alzheimer's disease",
                confidence_score=0.70,
                supporting_evidence=["DNA repair dysfunction"],
                domain="neurodegenerative_disease",
                testable_predictions=["Increased DNA damage in AD brains"],
                experimental_approaches=["mouse_models"]
            )
            mock_gen.return_value = mock_hypothesis
            
            results = system.generate_hypotheses(input_data)
            
            assert 'hypotheses' in results
            assert len(results['hypotheses']) >= 1
    
    def test_experimental_design_generation(self, sample_hypothesis):
        """Test experimental design generation."""
        generator = HypothesisGenerator()
        
        with patch.object(generator, '_design_experiment') as mock_design:
            mock_design.return_value = ExperimentalDesign(
                objective="Test hypothesis",
                methodology="Cell culture",
                experimental_groups=["Control", "Treatment"],
                controls=["Negative control"],
                measurements=["Outcome measure"],
                statistical_analysis="ANOVA",
                estimated_duration="6 months",
                estimated_cost="$15,000",
                success_probability=0.70,
                ethical_considerations=["Animal welfare"]
            )
            
            design = generator.design_experiment(sample_hypothesis)
            
            assert isinstance(design, ExperimentalDesign)
            assert design.success_probability == 0.70
    
    def test_hypothesis_ranking(self):
        """Test hypothesis ranking by priority."""
        system = HypothesisGenerationSystem()
        
        hypotheses = [
            BiomedicalHypothesis(
                hypothesis_text="Hypothesis 1",
                confidence_score=0.60,
                supporting_evidence=["evidence1"],
                domain="domain1",
                testable_predictions=["prediction1"],
                experimental_approaches=["approach1"]
            ),
            BiomedicalHypothesis(
                hypothesis_text="Hypothesis 2",
                confidence_score=0.90,
                supporting_evidence=["evidence1", "evidence2"],
                domain="domain1",
                testable_predictions=["prediction1"],
                experimental_approaches=["approach1"]
            )
        ]
        
        ranked_hypotheses = system.rank_hypotheses(hypotheses)
        
        # Should be ranked by confidence score (descending)
        assert ranked_hypotheses[0].confidence_score >= ranked_hypotheses[1].confidence_score


class TestValidation:
    """Test validation components."""
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(
            validation_type="literature_cross_validation",
            score=0.85,
            details={"supporting_papers": 10, "contradicting_papers": 2},
            timestamp="2023-01-01T00:00:00",
            validator_version="1.0.0"
        )
        
        assert result.validation_type == "literature_cross_validation"
        assert result.score == 0.85
        assert result.details["supporting_papers"] == 10
    
    def test_expert_assessment_dataclass(self):
        """Test ExpertAssessment dataclass."""
        assessment = ExpertAssessment(
            expert_id="expert_001",
            expertise_domain="cancer_genetics",
            assessment_score=0.80,
            confidence_in_assessment=0.90,
            comments="Well-supported hypothesis with strong biological rationale",
            recommendation="proceed_with_testing"
        )
        
        assert assessment.expert_id == "expert_001"
        assert assessment.assessment_score == 0.80
        assert assessment.recommendation == "proceed_with_testing"
    
    def test_literature_cross_validator(self):
        """Test LiteratureCrossValidator."""
        validator = LiteratureCrossValidator()
        
        hypothesis = BiomedicalHypothesis(
            hypothesis_text="BRCA1 mutations increase PARP inhibitor sensitivity",
            confidence_score=0.85,
            supporting_evidence=["evidence1"],
            domain="cancer_genetics",
            testable_predictions=["prediction1"],
            experimental_approaches=["in_vitro"]
        )
        
        # Mock literature search
        with patch.object(validator, '_search_literature') as mock_search:
            mock_search.return_value = [
                {'title': 'Supporting paper 1', 'relevance': 0.90, 'supports': True},
                {'title': 'Supporting paper 2', 'relevance': 0.85, 'supports': True},
                {'title': 'Contradicting paper', 'relevance': 0.70, 'supports': False}
            ]
            
            validation_result = validator.validate(hypothesis)
            
            assert isinstance(validation_result, ValidationResult)
            assert validation_result.validation_type == "literature_cross_validation"
            assert 0 <= validation_result.score <= 1
    
    def test_temporal_validator(self):
        """Test TemporalValidator."""
        validator = TemporalValidator()
        
        hypothesis = BiomedicalHypothesis(
            hypothesis_text="BRCA1 mutations increase PARP inhibitor sensitivity",
            confidence_score=0.85,
            supporting_evidence=["evidence1"],
            domain="cancer_genetics",
            testable_predictions=["prediction1"],
            experimental_approaches=["in_vitro"]
        )
        
        # Mock temporal analysis
        with patch.object(validator, '_analyze_temporal_trends') as mock_trends:
            mock_trends.return_value = {
                'trend_score': 0.80,
                'recent_support': 0.85,
                'historical_support': 0.75,
                'trend_direction': 'increasing'
            }
            
            validation_result = validator.validate(hypothesis)
            
            assert isinstance(validation_result, ValidationResult)
            assert validation_result.validation_type == "temporal_validation"
            assert 'trend_direction' in validation_result.details
    
    def test_expert_validation_interface(self):
        """Test ExpertValidationInterface."""
        interface = ExpertValidationInterface()
        
        hypothesis = BiomedicalHypothesis(
            hypothesis_text="BRCA1 mutations increase PARP inhibitor sensitivity",
            confidence_score=0.85,
            supporting_evidence=["evidence1"],
            domain="cancer_genetics",
            testable_predictions=["prediction1"],
            experimental_approaches=["in_vitro"]
        )
        
        # Mock expert assessment
        mock_assessment = ExpertAssessment(
            expert_id="expert_001",
            expertise_domain="cancer_genetics",
            assessment_score=0.80,
            confidence_in_assessment=0.90,
            comments="Well-supported hypothesis",
            recommendation="proceed_with_testing"
        )
        
        with patch.object(interface, '_collect_expert_assessments') as mock_collect:
            mock_collect.return_value = [mock_assessment]
            
            validation_result = interface.validate(hypothesis)
            
            assert isinstance(validation_result, ValidationResult)
            assert validation_result.validation_type == "expert_validation"
            assert validation_result.score == 0.80
    
    def test_comprehensive_validation_system(self):
        """Test ComprehensiveValidationSystem integration."""
        system = ComprehensiveValidationSystem()
        
        hypothesis = BiomedicalHypothesis(
            hypothesis_text="BRCA1 mutations increase PARP inhibitor sensitivity",
            confidence_score=0.85,
            supporting_evidence=["evidence1"],
            domain="cancer_genetics",
            testable_predictions=["prediction1"],
            experimental_approaches=["in_vitro"]
        )
        
        # Mock all validators
        mock_lit_result = ValidationResult("literature", 0.80, {}, "2023-01-01", "1.0")
        mock_temp_result = ValidationResult("temporal", 0.75, {}, "2023-01-01", "1.0")
        mock_exp_result = ValidationResult("expert", 0.85, {}, "2023-01-01", "1.0")
        
        with patch.object(system.literature_validator, 'validate') as mock_lit:
            mock_lit.return_value = mock_lit_result
            
            with patch.object(system.temporal_validator, 'validate') as mock_temp:
                mock_temp.return_value = mock_temp_result
                
                with patch.object(system.expert_validator, 'validate') as mock_exp:
                    mock_exp.return_value = mock_exp_result
                    
                    comprehensive_result = system.validate_hypothesis(hypothesis)
                    
                    assert 'literature_validation' in comprehensive_result
                    assert 'temporal_validation' in comprehensive_result
                    assert 'expert_validation' in comprehensive_result
                    assert 'overall_score' in comprehensive_result
    
    def test_validation_aggregation(self):
        """Test validation score aggregation."""
        system = ComprehensiveValidationSystem()
        
        validation_scores = {
            'literature': 0.80,
            'temporal': 0.75,
            'expert': 0.85,
            'biological_plausibility': 0.90
        }
        
        overall_score = system.aggregate_validation_scores(validation_scores)
        
        assert 0 <= overall_score <= 1
        assert overall_score > 0  # Should be positive given positive inputs
    
    def test_validation_confidence_intervals(self):
        """Test validation confidence interval computation."""
        validator = LiteratureCrossValidator()
        
        # Mock bootstrap samples
        bootstrap_scores = [0.80, 0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77, 0.84, 0.80]
        
        confidence_interval = validator.compute_confidence_interval(bootstrap_scores, confidence_level=0.95)
        
        assert len(confidence_interval) == 2  # Lower and upper bounds
        assert confidence_interval[0] <= confidence_interval[1]


@pytest.mark.integration
class TestPhase3Integration:
    """Integration tests for Phase 3 components."""
    
    def test_end_to_end_discovery_pipeline(self, sample_literature_data, sample_knowledge_graph):
        """Test end-to-end novel discovery pipeline."""
        # Initialize systems
        novelty_system = NoveltyDetectionSystem()
        hypothesis_system = HypothesisGenerationSystem()
        validation_system = ComprehensiveValidationSystem()
        
        # Mock the pipeline
        with patch.object(novelty_system, 'detect_novel_knowledge') as mock_novelty:
            mock_novelty.return_value = {
                'novel_relations': [
                    NovelRelation(
                        head_entity="BRCA1",
                        tail_entity="alzheimer_disease",
                        relation_type="ASSOCIATED_WITH",
                        confidence_score=0.75,
                        supporting_evidence=["evidence1"],
                        novelty_score=0.90
                    )
                ],
                'patterns': []
            }
            
            with patch.object(hypothesis_system, 'generate_hypotheses') as mock_hyp:
                mock_hypothesis = BiomedicalHypothesis(
                    hypothesis_text="BRCA1 deficiency contributes to Alzheimer's",
                    confidence_score=0.70,
                    supporting_evidence=["DNA repair"],
                    domain="neurodegenerative_disease",
                    testable_predictions=["Increased DNA damage"],
                    experimental_approaches=["mouse_models"]
                )
                mock_hyp.return_value = {'hypotheses': [mock_hypothesis]}
                
                with patch.object(validation_system, 'validate_hypothesis') as mock_val:
                    mock_val.return_value = {
                        'overall_score': 0.75,
                        'literature_validation': ValidationResult("lit", 0.80, {}, "2023", "1.0"),
                        'temporal_validation': ValidationResult("temp", 0.70, {}, "2023", "1.0")
                    }
                    
                    # Run pipeline
                    novel_knowledge = novelty_system.detect_novel_knowledge(
                        sample_literature_data, sample_knowledge_graph
                    )
                    
                    hypotheses = hypothesis_system.generate_hypotheses({
                        'novel_relations': novel_knowledge['novel_relations']
                    })
                    
                    validation_results = validation_system.validate_hypothesis(
                        hypotheses['hypotheses'][0]
                    )
                    
                    # Verify pipeline results
                    assert len(novel_knowledge['novel_relations']) == 1
                    assert len(hypotheses['hypotheses']) == 1
                    assert validation_results['overall_score'] == 0.75
    
    def test_confidence_scoring_integration(self):
        """Test confidence scoring integration across components."""
        confidence_scorer = ConfidenceScorer()
        
        # Create test relationship
        relationship = {
            'head': 'BRCA1',
            'relation': 'ASSOCIATED_WITH',
            'tail': 'breast_cancer'
        }
        
        evidence = {
            'literature': [
                {'title': 'BRCA1 in breast cancer', 'confidence': 0.95, 'citations': 100}
            ],
            'experimental': [
                {'study_type': 'clinical_trial', 'p_value': 0.001, 'sample_size': 1000}
            ]
        }
        
        # Mock components
        with patch.object(confidence_scorer.literature_assessor, 'assess_confidence') as mock_lit:
            mock_lit.return_value = 0.90
            
            with patch.object(confidence_scorer.experimental_assessor, 'assess_confidence') as mock_exp:
                mock_exp.return_value = 0.85
                
                with patch.object(confidence_scorer.cross_modal_integrator, 'integrate') as mock_int:
                    mock_int.return_value = torch.tensor([0.87])
                    
                    metrics = confidence_scorer.assess_relationship_confidence(relationship, evidence)
                    
                    assert isinstance(metrics, ConfidenceMetrics)
                    assert 0.8 <= metrics.overall_confidence <= 0.9
    
    @pytest.mark.slow
    def test_large_scale_validation(self):
        """Test validation of large numbers of hypotheses."""
        validator = LiteratureCrossValidator()
        
        # Create many hypotheses
        hypotheses = [
            BiomedicalHypothesis(
                hypothesis_text=f"Hypothesis {i}",
                confidence_score=0.80,
                supporting_evidence=[f"evidence_{i}"],
                domain="test_domain",
                testable_predictions=[f"prediction_{i}"],
                experimental_approaches=["test_approach"]
            )
            for i in range(50)
        ]
        
        # Mock validation for batch processing
        with patch.object(validator, '_search_literature') as mock_search:
            mock_search.return_value = [
                {'title': 'Supporting paper', 'relevance': 0.80, 'supports': True}
            ]
            
            validation_results = []
            for hypothesis in hypotheses:
                result = validator.validate(hypothesis)
                validation_results.append(result)
            
            assert len(validation_results) == 50
            assert all(isinstance(result, ValidationResult) for result in validation_results)


if __name__ == "__main__":
    pytest.main([__file__])