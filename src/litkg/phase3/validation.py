"""
Validation System for Novel Relations and Hypotheses

This module provides comprehensive validation mechanisms for novel biomedical
relationships and generated hypotheses using multiple validation strategies:

1. Cross-validation against held-out literature
2. Temporal validation using publication dates
3. Experimental validation against databases
4. Expert validation through structured assessment
5. Reproducibility validation across different methods

Key components:
- Literature Cross-Validator: Tests predictions against unseen literature
- Temporal Validator: Validates using chronological publication data
- Database Validator: Cross-references with experimental databases
- Expert Assessment Interface: Structured expert evaluation
- Reproducibility Checker: Validates across different prediction methods
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import networkx as nx
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from ..utils.logging import LoggerMixin
from .novelty_detection import NovelRelation, NoveltyDetectionSystem
from .hypothesis_generation import BiomedicalHypothesis, HypothesisGenerationSystem


@dataclass
class ValidationResult:
    """Results from a validation experiment."""
    validation_type: str
    validation_id: str
    timestamp: str
    methodology: str
    metrics: Dict[str, float]
    predictions_tested: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    validation_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExpertAssessment:
    """Expert assessment of a prediction or hypothesis."""
    assessor_id: str
    item_id: str
    item_type: str  # "novel_relation" or "hypothesis"
    biological_plausibility: float  # 0-1 scale
    novelty_score: float  # 0-1 scale
    confidence_in_assessment: float  # 0-1 scale
    supporting_evidence: List[str]
    concerns: List[str]
    recommended_experiments: List[str]
    overall_rating: str  # "excellent", "good", "moderate", "poor"
    detailed_comments: str
    assessment_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LiteratureCrossValidator(LoggerMixin):
    """
    Validates novel predictions against held-out literature using cross-validation.
    """
    
    def __init__(self, validation_split: float = 0.2, random_seed: int = 42):
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.validation_results = []
        
        np.random.seed(random_seed)
        self.logger.info("Initialized LiteratureCrossValidator")
    
    def validate_predictions(
        self,
        novel_relations: List[NovelRelation],
        literature_graph: nx.Graph,
        known_relations: Set[Tuple[str, str, str]],
        n_folds: int = 5
    ) -> ValidationResult:
        """
        Validate novel relation predictions using cross-validation against literature.
        
        Args:
            novel_relations: Predicted novel relations
            literature_graph: Complete literature graph
            known_relations: Set of known relations (entity1, entity2, relation_type)
            n_folds: Number of cross-validation folds
            
        Returns:
            Validation results
        """
        self.logger.info(f"Starting literature cross-validation with {len(novel_relations)} predictions")
        
        # Create ground truth labels
        y_true, y_pred, prediction_details = self._prepare_validation_data(
            novel_relations, known_relations
        )
        
        if len(y_true) == 0:
            self.logger.warning("No validation data available")
            return self._create_empty_result("literature_cross_validation")
        
        # Perform stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(y_pred, y_true)):
            fold_result = self._validate_fold(
                y_true[test_idx], y_pred[test_idx], fold, prediction_details
            )
            fold_results.append(fold_result)
        
        # Aggregate results across folds
        aggregated_metrics = self._aggregate_fold_results(fold_results)
        
        # Calculate overall confusion matrix
        cm = confusion_matrix(y_true, (np.array(y_pred) > 0.5).astype(int))
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        auc_score = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
        
        validation_result = ValidationResult(
            validation_type="literature_cross_validation",
            validation_id=f"lit_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            methodology=f"{n_folds}-fold cross-validation against literature",
            metrics=aggregated_metrics,
            predictions_tested=len(novel_relations),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_score=auc_score,
            validation_details={
                "fold_results": fold_results,
                "validation_split": self.validation_split,
                "n_folds": n_folds,
                "random_seed": self.random_seed
            }
        )
        
        self.validation_results.append(validation_result)
        self.logger.info(f"Literature validation completed: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
        
        return validation_result
    
    def _prepare_validation_data(
        self,
        novel_relations: List[NovelRelation],
        known_relations: Set[Tuple[str, str, str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Prepare data for validation."""
        y_true = []
        y_pred = []
        prediction_details = []
        
        for relation in novel_relations:
            # Check if this relation exists in known relations
            relation_tuple = (relation.entity1, relation.entity2, relation.relation_type)
            reverse_tuple = (relation.entity2, relation.entity1, relation.relation_type)
            
            # True if relation exists in either direction
            is_known = relation_tuple in known_relations or reverse_tuple in known_relations
            
            y_true.append(1 if is_known else 0)
            y_pred.append(relation.confidence_score)
            
            prediction_details.append({
                "relation": relation_tuple,
                "confidence": relation.confidence_score,
                "is_known": is_known,
                "evidence_sources": relation.evidence_sources
            })
        
        return np.array(y_true), np.array(y_pred), prediction_details
    
    def _validate_fold(
        self,
        y_true_fold: np.ndarray,
        y_pred_fold: np.ndarray,
        fold_idx: int,
        prediction_details: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate a single fold."""
        if len(set(y_true_fold)) < 2:
            return {
                "fold": fold_idx,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_score": 0.5,
                "average_precision": 0.0
            }
        
        # Calculate metrics for this fold
        precision, recall, _ = precision_recall_curve(y_true_fold, y_pred_fold)
        auc_score = roc_auc_score(y_true_fold, y_pred_fold)
        avg_precision = average_precision_score(y_true_fold, y_pred_fold)
        
        # Binary classification metrics
        y_pred_binary = (y_pred_fold > 0.5).astype(int)
        cm = confusion_matrix(y_true_fold, y_pred_binary)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fold_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fold_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fold_f1 = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall) if (fold_precision + fold_recall) > 0 else 0.0
        else:
            fold_precision = fold_recall = fold_f1 = 0.0
        
        return {
            "fold": fold_idx,
            "precision": fold_precision,
            "recall": fold_recall,
            "f1_score": fold_f1,
            "auc_score": auc_score,
            "average_precision": avg_precision
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}
        
        metrics = ["precision", "recall", "f1_score", "auc_score", "average_precision"]
        aggregated = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            aggregated[f"mean_{metric}"] = np.mean(values)
            aggregated[f"std_{metric}"] = np.std(values)
        
        return aggregated
    
    def _create_empty_result(self, validation_type: str) -> ValidationResult:
        """Create empty validation result."""
        return ValidationResult(
            validation_type=validation_type,
            validation_id=f"{validation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            methodology="No validation data available",
            metrics={},
            predictions_tested=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_score=0.5,
            validation_details={"error": "No validation data available"}
        )


class TemporalValidator(LoggerMixin):
    """
    Validates predictions using temporal information - test if predictions
    match relationships discovered in later publications.
    """
    
    def __init__(self, temporal_split_date: Optional[datetime] = None):
        self.temporal_split_date = temporal_split_date or (datetime.now() - timedelta(days=365))
        self.logger.info(f"Initialized TemporalValidator with split date: {self.temporal_split_date}")
    
    def temporal_validation(
        self,
        novel_relations: List[NovelRelation],
        literature_graph: nx.Graph,
        publication_dates: Dict[str, datetime]
    ) -> ValidationResult:
        """
        Validate predictions using temporal split - train on older papers,
        test on newer papers.
        
        Args:
            novel_relations: Predicted novel relations
            literature_graph: Literature graph with temporal information
            publication_dates: Mapping from paper IDs to publication dates
            
        Returns:
            Temporal validation results
        """
        self.logger.info("Starting temporal validation")
        
        # Split papers by temporal cutoff
        older_papers = {
            paper_id for paper_id, date in publication_dates.items()
            if date < self.temporal_split_date
        }
        newer_papers = {
            paper_id for paper_id, date in publication_dates.items()
            if date >= self.temporal_split_date
        }
        
        self.logger.info(f"Temporal split: {len(older_papers)} older papers, {len(newer_papers)} newer papers")
        
        if len(newer_papers) == 0:
            self.logger.warning("No newer papers available for temporal validation")
            return self._create_empty_temporal_result()
        
        # Extract relations from newer papers (ground truth for validation)
        newer_relations = self._extract_relations_from_papers(newer_papers, literature_graph)
        
        # Validate predictions against newer relations
        y_true, y_pred = self._prepare_temporal_validation_data(novel_relations, newer_relations)
        
        if len(y_true) == 0:
            return self._create_empty_temporal_result()
        
        # Calculate metrics
        precision, recall, f1, auc = self._calculate_temporal_metrics(y_true, y_pred)
        
        # Calculate confusion matrix
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        validation_result = ValidationResult(
            validation_type="temporal_validation",
            validation_id=f"temp_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            methodology=f"Temporal validation with split date: {self.temporal_split_date}",
            metrics={
                "temporal_precision": precision,
                "temporal_recall": recall,
                "temporal_f1": f1,
                "temporal_auc": auc
            },
            predictions_tested=len(novel_relations),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            validation_details={
                "temporal_split_date": self.temporal_split_date.isoformat(),
                "older_papers_count": len(older_papers),
                "newer_papers_count": len(newer_papers),
                "newer_relations_found": len(newer_relations)
            }
        )
        
        self.logger.info(f"Temporal validation completed: Precision={precision:.3f}, Recall={recall:.3f}")
        return validation_result
    
    def _extract_relations_from_papers(
        self,
        paper_ids: Set[str],
        literature_graph: nx.Graph
    ) -> Set[Tuple[str, str]]:
        """Extract entity relations from specified papers."""
        relations = set()
        
        for paper_id in paper_ids:
            if paper_id in literature_graph:
                # Get all entities connected to this paper
                connected_entities = [
                    node for node in literature_graph.neighbors(paper_id)
                    if literature_graph.nodes[node].get('type') == 'entity'
                ]
                
                # Create relations between co-occurring entities
                for i, entity1 in enumerate(connected_entities):
                    for entity2 in connected_entities[i+1:]:
                        relations.add(tuple(sorted([entity1, entity2])))
        
        return relations
    
    def _prepare_temporal_validation_data(
        self,
        novel_relations: List[NovelRelation],
        newer_relations: Set[Tuple[str, str]]
    ) -> Tuple[List[int], List[float]]:
        """Prepare data for temporal validation."""
        y_true = []
        y_pred = []
        
        for relation in novel_relations:
            # Check if this relation appears in newer literature
            relation_pair = tuple(sorted([relation.entity1, relation.entity2]))
            is_confirmed = relation_pair in newer_relations
            
            y_true.append(1 if is_confirmed else 0)
            y_pred.append(relation.confidence_score)
        
        return y_true, y_pred
    
    def _calculate_temporal_metrics(
        self,
        y_true: List[int],
        y_pred: List[float]
    ) -> Tuple[float, float, float, float]:
        """Calculate temporal validation metrics."""
        if len(set(y_true)) < 2:
            return 0.0, 0.0, 0.0, 0.5
        
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0
        
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.5
        
        return precision, recall, f1, auc
    
    def _create_empty_temporal_result(self) -> ValidationResult:
        """Create empty temporal validation result."""
        return ValidationResult(
            validation_type="temporal_validation",
            validation_id=f"temp_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            methodology="Temporal validation - insufficient data",
            metrics={},
            predictions_tested=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_score=0.5,
            validation_details={"error": "Insufficient temporal data"}
        )


class ExpertValidationInterface(LoggerMixin):
    """
    Interface for collecting expert assessments of predictions and hypotheses.
    """
    
    def __init__(self):
        self.expert_assessments = []
        self.assessment_templates = self._create_assessment_templates()
        self.logger.info("Initialized ExpertValidationInterface")
    
    def create_expert_assessment_form(
        self,
        item: Any,
        item_type: str
    ) -> Dict[str, Any]:
        """
        Create a structured assessment form for experts.
        
        Args:
            item: Novel relation or hypothesis to assess
            item_type: "novel_relation" or "hypothesis"
            
        Returns:
            Structured assessment form
        """
        if item_type == "novel_relation":
            return self._create_relation_assessment_form(item)
        elif item_type == "hypothesis":
            return self._create_hypothesis_assessment_form(item)
        else:
            raise ValueError(f"Unknown item type: {item_type}")
    
    def _create_relation_assessment_form(self, relation: NovelRelation) -> Dict[str, Any]:
        """Create assessment form for novel relation."""
        return {
            "assessment_type": "novel_relation",
            "item_id": f"{relation.entity1}_{relation.entity2}_{relation.relation_type}",
            "item_details": {
                "entity1": relation.entity1,
                "entity2": relation.entity2,
                "relation_type": relation.relation_type,
                "confidence_score": relation.confidence_score,
                "evidence_sources": relation.evidence_sources,
                "prediction_reasoning": relation.prediction_reasoning
            },
            "assessment_questions": {
                "biological_plausibility": {
                    "question": "How biologically plausible is this relationship?",
                    "scale": "1-5 (1=Implausible, 5=Highly plausible)",
                    "guidelines": "Consider known biological mechanisms, molecular pathways, and domain expertise"
                },
                "novelty_score": {
                    "question": "How novel is this predicted relationship?",
                    "scale": "1-5 (1=Well-known, 5=Completely novel)",
                    "guidelines": "Assess based on existing literature and your domain knowledge"
                },
                "confidence_in_assessment": {
                    "question": "How confident are you in your assessment?",
                    "scale": "1-5 (1=Very uncertain, 5=Very confident)",
                    "guidelines": "Consider your expertise in this specific area"
                },
                "supporting_evidence": {
                    "question": "What evidence supports this relationship?",
                    "type": "text_list",
                    "guidelines": "List specific papers, experiments, or biological mechanisms"
                },
                "concerns": {
                    "question": "What concerns or contradictions do you see?",
                    "type": "text_list",
                    "guidelines": "List potential issues, contradicting evidence, or methodological concerns"
                },
                "recommended_experiments": {
                    "question": "What experiments would you recommend to validate this?",
                    "type": "text_list",
                    "guidelines": "Suggest specific experimental approaches"
                },
                "overall_rating": {
                    "question": "Overall assessment of this prediction",
                    "options": ["excellent", "good", "moderate", "poor"],
                    "guidelines": "Consider plausibility, novelty, and potential impact"
                },
                "detailed_comments": {
                    "question": "Additional comments or detailed analysis",
                    "type": "long_text",
                    "guidelines": "Provide any additional insights or detailed reasoning"
                }
            }
        }
    
    def _create_hypothesis_assessment_form(self, hypothesis: BiomedicalHypothesis) -> Dict[str, Any]:
        """Create assessment form for hypothesis."""
        return {
            "assessment_type": "hypothesis",
            "item_id": hypothesis.id,
            "item_details": {
                "title": hypothesis.title,
                "description": hypothesis.description,
                "biological_mechanism": hypothesis.biological_mechanism,
                "testable_predictions": hypothesis.testable_predictions,
                "experimental_approaches": hypothesis.experimental_approaches,
                "confidence_score": hypothesis.confidence_score
            },
            "assessment_questions": {
                "biological_plausibility": {
                    "question": "How biologically plausible is this hypothesis?",
                    "scale": "1-5 (1=Implausible, 5=Highly plausible)",
                    "guidelines": "Evaluate the proposed mechanism and predictions"
                },
                "novelty_score": {
                    "question": "How novel is this hypothesis?",
                    "scale": "1-5 (1=Well-established, 5=Completely novel)",
                    "guidelines": "Consider existing literature and research"
                },
                "testability": {
                    "question": "How testable is this hypothesis?",
                    "scale": "1-5 (1=Difficult to test, 5=Easily testable)",
                    "guidelines": "Assess feasibility of proposed experiments"
                },
                "potential_impact": {
                    "question": "What is the potential research impact?",
                    "scale": "1-5 (1=Low impact, 5=High impact)",
                    "guidelines": "Consider therapeutic, mechanistic, and scientific implications"
                }
            }
        }
    
    def process_expert_assessment(
        self,
        assessment_data: Dict[str, Any],
        assessor_id: str
    ) -> ExpertAssessment:
        """
        Process completed expert assessment.
        
        Args:
            assessment_data: Completed assessment form data
            assessor_id: Unique identifier for the expert assessor
            
        Returns:
            Processed expert assessment
        """
        # Extract responses
        responses = assessment_data.get("responses", {})
        
        expert_assessment = ExpertAssessment(
            assessor_id=assessor_id,
            item_id=assessment_data["item_id"],
            item_type=assessment_data["assessment_type"],
            biological_plausibility=float(responses.get("biological_plausibility", 3.0)) / 5.0,
            novelty_score=float(responses.get("novelty_score", 3.0)) / 5.0,
            confidence_in_assessment=float(responses.get("confidence_in_assessment", 3.0)) / 5.0,
            supporting_evidence=responses.get("supporting_evidence", []),
            concerns=responses.get("concerns", []),
            recommended_experiments=responses.get("recommended_experiments", []),
            overall_rating=responses.get("overall_rating", "moderate"),
            detailed_comments=responses.get("detailed_comments", ""),
            assessment_timestamp=datetime.now().isoformat()
        )
        
        self.expert_assessments.append(expert_assessment)
        self.logger.info(f"Processed expert assessment for {assessment_data['item_id']}")
        
        return expert_assessment
    
    def _create_assessment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create templates for different types of assessments."""
        return {
            "relation_template": {
                "fields": ["biological_plausibility", "novelty_score", "confidence"],
                "scales": {"biological_plausibility": [1, 5], "novelty_score": [1, 5]}
            },
            "hypothesis_template": {
                "fields": ["plausibility", "novelty", "testability", "impact"],
                "scales": {"plausibility": [1, 5], "novelty": [1, 5], "testability": [1, 5], "impact": [1, 5]}
            }
        }
    
    def aggregate_expert_assessments(
        self,
        item_id: str
    ) -> Dict[str, Any]:
        """
        Aggregate multiple expert assessments for the same item.
        
        Args:
            item_id: ID of the item to aggregate assessments for
            
        Returns:
            Aggregated assessment results
        """
        item_assessments = [
            assessment for assessment in self.expert_assessments
            if assessment.item_id == item_id
        ]
        
        if not item_assessments:
            return {"error": "No assessments found for this item"}
        
        # Calculate aggregate metrics
        plausibility_scores = [a.biological_plausibility for a in item_assessments]
        novelty_scores = [a.novelty_score for a in item_assessments]
        confidence_scores = [a.confidence_in_assessment for a in item_assessments]
        
        # Count overall ratings
        rating_counts = {}
        for assessment in item_assessments:
            rating = assessment.overall_rating
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        return {
            "item_id": item_id,
            "num_assessments": len(item_assessments),
            "mean_plausibility": np.mean(plausibility_scores),
            "std_plausibility": np.std(plausibility_scores),
            "mean_novelty": np.mean(novelty_scores),
            "std_novelty": np.std(novelty_scores),
            "mean_confidence": np.mean(confidence_scores),
            "std_confidence": np.std(confidence_scores),
            "rating_distribution": rating_counts,
            "consensus_rating": max(rating_counts.items(), key=lambda x: x[1])[0],
            "all_supporting_evidence": [
                evidence for assessment in item_assessments
                for evidence in assessment.supporting_evidence
            ],
            "all_concerns": [
                concern for assessment in item_assessments
                for concern in assessment.concerns
            ],
            "recommended_experiments": [
                exp for assessment in item_assessments
                for exp in assessment.recommended_experiments
            ]
        }


class ComprehensiveValidationSystem(LoggerMixin):
    """
    Main validation system that coordinates all validation methods.
    """
    
    def __init__(
        self,
        temporal_split_date: Optional[datetime] = None,
        validation_split: float = 0.2
    ):
        # Initialize validators
        self.literature_validator = LiteratureCrossValidator(validation_split=validation_split)
        self.temporal_validator = TemporalValidator(temporal_split_date=temporal_split_date)
        self.expert_interface = ExpertValidationInterface()
        
        # Results storage
        self.validation_results = []
        self.expert_assessments = []
        
        self.logger.info("Initialized ComprehensiveValidationSystem")
    
    def comprehensive_validation(
        self,
        novel_relations: List[NovelRelation],
        hypotheses: List[BiomedicalHypothesis],
        literature_graph: nx.Graph,
        known_relations: Set[Tuple[str, str, str]],
        publication_dates: Dict[str, datetime],
        include_expert_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation using all available methods.
        
        Args:
            novel_relations: Predicted novel relations
            hypotheses: Generated hypotheses
            literature_graph: Complete literature graph
            known_relations: Known relations for validation
            publication_dates: Publication dates for temporal validation
            include_expert_validation: Whether to include expert validation
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting comprehensive validation pipeline")
        
        validation_results = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "novel_relations_tested": len(novel_relations),
                "hypotheses_tested": len(hypotheses),
                "validation_methods": []
            },
            "literature_validation": None,
            "temporal_validation": None,
            "expert_validation": None,
            "overall_assessment": None
        }
        
        # 1. Literature Cross-Validation
        if known_relations:
            self.logger.info("Running literature cross-validation")
            lit_validation = self.literature_validator.validate_predictions(
                novel_relations, literature_graph, known_relations
            )
            validation_results["literature_validation"] = lit_validation.to_dict()
            validation_results["validation_summary"]["validation_methods"].append("literature_cross_validation")
        
        # 2. Temporal Validation
        if publication_dates:
            self.logger.info("Running temporal validation")
            temp_validation = self.temporal_validator.temporal_validation(
                novel_relations, literature_graph, publication_dates
            )
            validation_results["temporal_validation"] = temp_validation.to_dict()
            validation_results["validation_summary"]["validation_methods"].append("temporal_validation")
        
        # 3. Expert Validation (if requested)
        if include_expert_validation:
            self.logger.info("Preparing expert validation forms")
            expert_forms = self._prepare_expert_validation(novel_relations, hypotheses)
            validation_results["expert_validation"] = {
                "forms_generated": len(expert_forms),
                "assessment_forms": expert_forms,
                "instructions": "Complete assessment forms and submit via process_expert_assessment()"
            }
            validation_results["validation_summary"]["validation_methods"].append("expert_validation")
        
        # 4. Overall Assessment
        validation_results["overall_assessment"] = self._create_overall_assessment(validation_results)
        
        self.validation_results.append(validation_results)
        self.logger.info("Comprehensive validation completed")
        
        return validation_results
    
    def _prepare_expert_validation(
        self,
        novel_relations: List[NovelRelation],
        hypotheses: List[BiomedicalHypothesis]
    ) -> List[Dict[str, Any]]:
        """Prepare expert validation forms."""
        expert_forms = []
        
        # Create forms for top novel relations
        for relation in novel_relations[:10]:  # Top 10 relations
            form = self.expert_interface.create_expert_assessment_form(relation, "novel_relation")
            expert_forms.append(form)
        
        # Create forms for top hypotheses
        for hypothesis in hypotheses[:5]:  # Top 5 hypotheses
            form = self.expert_interface.create_expert_assessment_form(hypothesis, "hypothesis")
            expert_forms.append(form)
        
        return expert_forms
    
    def _create_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall assessment from all validation methods."""
        assessment = {
            "validation_quality": "unknown",
            "confidence_in_predictions": 0.5,
            "key_findings": [],
            "recommendations": []
        }
        
        # Analyze literature validation
        if validation_results["literature_validation"]:
            lit_val = validation_results["literature_validation"]
            lit_f1 = lit_val.get("f1_score", 0.0)
            
            if lit_f1 > 0.7:
                assessment["key_findings"].append(f"Strong literature validation (F1={lit_f1:.3f})")
                assessment["confidence_in_predictions"] += 0.2
            elif lit_f1 > 0.5:
                assessment["key_findings"].append(f"Moderate literature validation (F1={lit_f1:.3f})")
                assessment["confidence_in_predictions"] += 0.1
            else:
                assessment["key_findings"].append(f"Weak literature validation (F1={lit_f1:.3f})")
        
        # Analyze temporal validation
        if validation_results["temporal_validation"]:
            temp_val = validation_results["temporal_validation"]
            temp_f1 = temp_val.get("f1_score", 0.0)
            
            if temp_f1 > 0.6:
                assessment["key_findings"].append(f"Good temporal validation (F1={temp_f1:.3f})")
                assessment["confidence_in_predictions"] += 0.15
            elif temp_f1 > 0.4:
                assessment["key_findings"].append(f"Moderate temporal validation (F1={temp_f1:.3f})")
                assessment["confidence_in_predictions"] += 0.05
        
        # Determine overall quality
        if assessment["confidence_in_predictions"] > 0.8:
            assessment["validation_quality"] = "excellent"
            assessment["recommendations"].append("Predictions show strong validation across multiple methods")
        elif assessment["confidence_in_predictions"] > 0.6:
            assessment["validation_quality"] = "good"
            assessment["recommendations"].append("Predictions show reasonable validation, consider expert review")
        elif assessment["confidence_in_predictions"] > 0.4:
            assessment["validation_quality"] = "moderate"
            assessment["recommendations"].append("Predictions need additional validation before publication")
        else:
            assessment["validation_quality"] = "poor"
            assessment["recommendations"].append("Predictions require significant improvement and validation")
        
        # General recommendations
        assessment["recommendations"].extend([
            "Conduct experimental validation for top predictions",
            "Perform literature review for novel relationships",
            "Consider expert consultation for domain-specific validation",
            "Monitor temporal trends for prediction accuracy"
        ])
        
        return assessment
    
    def generate_validation_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Complete validation report
        """
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        latest_results = self.validation_results[-1]
        
        report = {
            "validation_report": {
                "title": "Comprehensive Validation of Novel Biomedical Predictions",
                "generated_timestamp": datetime.now().isoformat(),
                "validation_summary": latest_results["validation_summary"],
                "overall_assessment": latest_results["overall_assessment"]
            },
            "detailed_results": latest_results,
            "validation_methodology": {
                "literature_cross_validation": "K-fold cross-validation against known literature relationships",
                "temporal_validation": "Validation using chronological publication data",
                "expert_validation": "Structured assessment by domain experts",
                "comprehensive_assessment": "Integration of multiple validation approaches"
            },
            "recommendations": {
                "immediate_actions": latest_results["overall_assessment"]["recommendations"][:3],
                "long_term_strategy": [
                    "Establish ongoing validation pipeline",
                    "Build expert reviewer network",
                    "Implement continuous model improvement",
                    "Develop experimental validation partnerships"
                ]
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report