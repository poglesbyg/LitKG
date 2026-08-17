"""
Confidence scoring system for multi-modal biomedical evidence.

This module implements sophisticated confidence assessment for relationships
derived from different sources (literature vs. experimental data), considering
factors like evidence strength, source reliability, temporal consistency,
and cross-modal agreement.
"""

import math

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from ..utils.logging import LoggerMixin


class EvidenceType(Enum):
    """Types of evidence for biomedical relationships."""
    LITERATURE = "literature"
    EXPERIMENTAL = "experimental"
    CURATED_DATABASE = "curated_database"
    COMPUTATIONAL = "computational"
    EXPERT_ANNOTATION = "expert_annotation"


@dataclass
class ConfidenceMetrics:
    """
    Container for confidence assessment metrics.

    Only ``overall_confidence`` is required; the remaining metrics default to
    zero so partial assessments (a literature-only scoring pass, say) can be
    represented without inventing values for evidence that was never examined.
    """

    # Overall confidence score [0, 1]
    overall_confidence: float

    # Evidence-specific scores
    literature_confidence: float = 0.0
    experimental_confidence: float = 0.0
    cross_modal_agreement: float = 0.0

    # Quality indicators
    evidence_strength: float = 0.0
    source_reliability: float = 0.0
    temporal_consistency: float = 0.0

    # Uncertainty quantification
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Data uncertainty

    # Supporting evidence counts
    supporting_papers: int = 0
    supporting_experiments: int = 0
    contradicting_evidence: int = 0

    # Metadata
    confidence_level: str = field(default="")  # "high", "medium", "low"
    explanation: str = field(default="")
    evidence_sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set confidence level based on overall score."""
        if self.overall_confidence >= 0.8:
            self.confidence_level = "high"
        elif self.overall_confidence >= 0.5:
            self.confidence_level = "medium"
        else:
            self.confidence_level = "low"

    @property
    def consistency_score(self) -> float:
        """Alias for temporal_consistency."""
        return self.temporal_consistency

    @property
    def uncertainty_estimate(self) -> float:
        """
        Total uncertainty, combining the epistemic and aleatoric components.

        Combined in quadrature, since the two are modeled as independent.
        """
        return float(
            (self.epistemic_uncertainty ** 2 + self.aleatoric_uncertainty ** 2) ** 0.5
        )


class LiteratureConfidenceAssessor(nn.Module, LoggerMixin):
    """
    Assess confidence of literature-derived relationships.
    
    Considers factors like:
    - Journal impact factor and reputation
    - Number of supporting publications
    - Recency and temporal consistency
    - Study quality indicators
    - Citation patterns
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_quality_factors: int = 10
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Neural network for literature confidence assessment
        self.confidence_net = nn.Sequential(
            nn.Linear(embedding_dim + num_quality_factors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Quality factor weights
        self.quality_weights = nn.Parameter(torch.ones(num_quality_factors))
        
    def forward(
        self,
        literature_embedding: torch.Tensor,
        quality_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Assess literature confidence.
        
        Args:
            literature_embedding: Embedding of literature evidence [batch_size, embedding_dim]
            quality_features: Quality indicators [batch_size, num_quality_factors]
            
        Returns:
            Dictionary with confidence scores and components
        """
        # Weight quality features
        weighted_quality = quality_features * self.quality_weights
        
        # Combine embeddings and quality features
        combined_features = torch.cat([literature_embedding, weighted_quality], dim=-1)
        
        # Compute confidence score
        confidence = self.confidence_net(combined_features)
        
        # Compute component scores
        journal_quality = torch.sigmoid(weighted_quality[:, 0:1])  # Impact factor
        citation_strength = torch.sigmoid(weighted_quality[:, 1:2])  # Citation count
        temporal_relevance = torch.sigmoid(weighted_quality[:, 2:3])  # Recency
        study_quality = torch.sigmoid(weighted_quality[:, 3:4])  # Methodology
        
        return {
            'confidence': confidence,
            'journal_quality': journal_quality,
            'citation_strength': citation_strength,
            'temporal_relevance': temporal_relevance,
            'study_quality': study_quality,
            'quality_weights': self.quality_weights
        }

    def assess_confidence(self, literature_evidence: List[Dict[str, Any]]) -> float:
        """
        Score a list of literature evidence records.

        This is the record-level entry point: it derives quality features from
        the raw records and reduces the network output to a single score. Use
        forward() directly when you already hold embeddings.

        Args:
            literature_evidence: Records with any of "confidence", "citations",
                "year", "journal_impact_factor".

        Returns:
            Confidence in [0, 1]. Returns 0.0 for empty evidence.
        """
        if not literature_evidence:
            return 0.0

        # Evidence-level agreement: the mean stated confidence, tempered by how
        # much corroboration exists. A single paper cannot reach full confidence.
        confidences = [
            float(record.get("confidence", 0.5)) for record in literature_evidence
        ]
        mean_confidence = sum(confidences) / len(confidences)

        # Corroboration saturates: 1 paper -> 0.5, 2 -> 0.67, 4 -> 0.8
        n = len(literature_evidence)
        corroboration = n / (n + 1.0)

        # Citation weight, log-scaled and saturating
        citations = [float(record.get("citations", 0)) for record in literature_evidence]
        mean_citations = sum(citations) / len(citations)
        citation_factor = math.log1p(mean_citations) / math.log1p(100.0)
        citation_factor = min(citation_factor, 1.0)

        score = (
            0.60 * mean_confidence
            + 0.25 * corroboration
            + 0.15 * citation_factor
        )
        return float(max(0.0, min(1.0, score)))


class ExperimentalConfidenceAssessor(nn.Module, LoggerMixin):
    """
    Assess confidence of experimental/database-derived relationships.
    
    Considers factors like:
    - Data source reliability (CIVIC, TCGA, CPTAC)
    - Sample size and statistical significance
    - Experimental methodology quality
    - Replication across studies
    - Effect size and consistency
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_sources: int = 5
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        
        # Source reliability embeddings
        self.source_embeddings = nn.Embedding(num_sources, 64)
        
        # Experimental confidence network
        self.confidence_net = nn.Sequential(
            nn.Linear(embedding_dim + 64 + 8, hidden_dim),  # +64 for source, +8 for stats
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Statistical significance assessor
        self.significance_net = nn.Sequential(
            nn.Linear(8, 32),  # p-value, effect size, sample size, etc.
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        experimental_embedding: torch.Tensor,
        source_ids: torch.Tensor,
        statistical_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Assess experimental confidence.
        
        Args:
            experimental_embedding: Embedding of experimental evidence
            source_ids: Source database IDs [batch_size]
            statistical_features: Statistical indicators [batch_size, 8]
            
        Returns:
            Dictionary with confidence scores and components
        """
        batch_size = experimental_embedding.size(0)
        
        # Get source embeddings
        source_emb = self.source_embeddings(source_ids)
        
        # Assess statistical significance
        statistical_confidence = self.significance_net(statistical_features)
        
        # Combine all features
        combined_features = torch.cat([
            experimental_embedding,
            source_emb,
            statistical_features
        ], dim=-1)
        
        # Compute overall confidence
        confidence = self.confidence_net(combined_features)
        
        # Extract component scores
        sample_size_score = torch.sigmoid(statistical_features[:, 0:1])
        effect_size_score = torch.sigmoid(statistical_features[:, 1:2])
        p_value_score = torch.sigmoid(-statistical_features[:, 2:3])  # Lower p-value = higher confidence
        replication_score = torch.sigmoid(statistical_features[:, 3:4])
        
        return {
            'confidence': confidence,
            'statistical_confidence': statistical_confidence,
            'sample_size_score': sample_size_score,
            'effect_size_score': effect_size_score,
            'p_value_score': p_value_score,
            'replication_score': replication_score,
            'source_reliability': torch.sigmoid(source_emb.mean(dim=-1, keepdim=True))
        }

    # Study designs ranked by evidential strength
    STUDY_TYPE_WEIGHTS = {
        "meta_analysis": 1.00,
        "clinical_trial": 0.90,
        "cohort": 0.75,
        "case_control": 0.65,
        "in_vivo": 0.60,
        "in_vitro": 0.45,
        "in_silico": 0.30,
    }

    def assess_confidence(self, experimental_evidence: List[Dict[str, Any]]) -> float:
        """
        Score a list of experimental evidence records.

        Weighs study design, statistical significance, and sample size. Use
        forward() directly when you already hold embeddings.

        Args:
            experimental_evidence: Records with any of "study_type", "p_value",
                "sample_size", "effect_size".

        Returns:
            Confidence in [0, 1]. Returns 0.0 for empty evidence.
        """
        if not experimental_evidence:
            return 0.0

        study_scores = []
        for record in experimental_evidence:
            design = self.STUDY_TYPE_WEIGHTS.get(
                str(record.get("study_type", "")).lower(), 0.5
            )

            # Significance: p=0.05 -> 0.5, p=0.001 -> ~0.9, p>=0.1 -> 0
            p_value = float(record.get("p_value", 0.05))
            if p_value <= 0:
                significance = 1.0
            elif p_value >= 0.1:
                significance = 0.0
            else:
                significance = min(1.0, math.log10(0.1 / p_value) / 2.0)

            # Sample size, log-scaled and saturating around n=1000
            sample_size = max(float(record.get("sample_size", 0)), 0.0)
            power = min(math.log1p(sample_size) / math.log1p(1000.0), 1.0)

            study_scores.append(0.4 * design + 0.4 * significance + 0.2 * power)

        # The strongest study dominates, with the rest providing corroboration
        best = max(study_scores)
        mean = sum(study_scores) / len(study_scores)
        score = 0.7 * best + 0.3 * mean

        return float(max(0.0, min(1.0, score)))


class CrossModalConfidenceIntegrator(nn.Module, LoggerMixin):
    """
    Integrate confidence assessments across literature and experimental evidence.
    
    This component resolves conflicts between different evidence types and
    provides unified confidence scores with uncertainty quantification.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,  # Match embedding dimension
        num_integration_heads: int = 4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_integration_heads
        
        # Cross-modal attention for evidence integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_integration_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Evidence conflict detector
        self.conflict_detector = nn.Sequential(
            nn.Linear(4, 64),  # lit_conf, exp_conf, agreement, disagreement
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Final confidence integrator
        self.confidence_integrator = nn.Sequential(
            nn.Linear(hidden_dim + 1, 128),  # +1 for conflict score
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # overall_conf, epistemic_unc, aleatoric_unc
        )
        
        # Evidence weighting network
        self.evidence_weighter = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # weights for lit vs exp
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        literature_features: torch.Tensor,
        experimental_features: torch.Tensor,
        lit_confidence: torch.Tensor,
        exp_confidence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate cross-modal confidence assessments.
        
        Args:
            literature_features: Literature evidence features [batch_size, hidden_dim]
            experimental_features: Experimental evidence features [batch_size, hidden_dim]
            lit_confidence: Literature confidence scores [batch_size, 1]
            exp_confidence: Experimental confidence scores [batch_size, 1]
            
        Returns:
            Integrated confidence metrics
        """
        batch_size = literature_features.size(0)
        
        # Compute agreement and disagreement
        agreement = torch.min(lit_confidence, exp_confidence)
        disagreement = torch.abs(lit_confidence - exp_confidence)
        
        # Detect evidence conflicts
        conflict_features = torch.cat([
            lit_confidence, exp_confidence, agreement, disagreement
        ], dim=-1)
        conflict_score = self.conflict_detector(conflict_features)
        
        # Cross-modal attention integration
        # Stack literature and experimental features
        evidence_stack = torch.stack([literature_features, experimental_features], dim=1)
        integrated_features, attention_weights = self.cross_attention(
            evidence_stack, evidence_stack, evidence_stack
        )
        integrated_features = integrated_features.mean(dim=1)  # Pool across evidence types
        
        # Compute evidence weights
        evidence_weights = self.evidence_weighter(conflict_features)
        lit_weight, exp_weight = evidence_weights[:, 0:1], evidence_weights[:, 1:2]
        
        # Integrate confidence scores
        confidence_input = torch.cat([integrated_features, conflict_score], dim=-1)
        confidence_outputs = self.confidence_integrator(confidence_input)
        
        overall_confidence = torch.sigmoid(confidence_outputs[:, 0:1])
        epistemic_uncertainty = torch.sigmoid(confidence_outputs[:, 1:2])
        aleatoric_uncertainty = torch.sigmoid(confidence_outputs[:, 2:3])
        
        # Weighted confidence combination
        weighted_confidence = lit_weight * lit_confidence + exp_weight * exp_confidence
        
        # Final confidence (combination of neural and weighted approaches)
        final_confidence = 0.7 * overall_confidence + 0.3 * weighted_confidence
        
        return {
            'overall_confidence': final_confidence,
            'literature_confidence': lit_confidence,
            'experimental_confidence': exp_confidence,
            'cross_modal_agreement': agreement,
            'evidence_conflict': conflict_score,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'literature_weight': lit_weight,
            'experimental_weight': exp_weight,
            'attention_weights': attention_weights
        }

    def integrate(
        self,
        literature_features: torch.Tensor,
        experimental_features: torch.Tensor,
        lit_confidence: Optional[torch.Tensor] = None,
        exp_confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Integrate two evidence streams into a single confidence score.

        Convenience wrapper over forward() for callers that want the scalar
        result rather than the full metric dictionary. When per-modality
        confidences are not supplied they are estimated from the features.

        Args:
            literature_features: [batch_size, hidden_dim]
            experimental_features: [batch_size, hidden_dim]
            lit_confidence: Optional [batch_size, 1] literature confidence
            exp_confidence: Optional [batch_size, 1] experimental confidence

        Returns:
            Integrated confidence, shape [batch_size], each value in [0, 1].
        """
        if literature_features.dim() == 1:
            literature_features = literature_features.unsqueeze(0)
        if experimental_features.dim() == 1:
            experimental_features = experimental_features.unsqueeze(0)

        batch_size = literature_features.size(0)
        device = literature_features.device

        # Absent an explicit confidence, summarize the feature vector itself
        if lit_confidence is None:
            lit_confidence = torch.sigmoid(
                literature_features.mean(dim=-1, keepdim=True)
            )
        if exp_confidence is None:
            exp_confidence = torch.sigmoid(
                experimental_features.mean(dim=-1, keepdim=True)
            )

        outputs = self.forward(
            literature_features=literature_features.to(device),
            experimental_features=experimental_features.to(device),
            lit_confidence=lit_confidence.to(device),
            exp_confidence=exp_confidence.to(device),
        )

        # [batch_size, 1] -> [batch_size], clamped since forward() mixes a
        # sigmoid output with a weighted sum that is not itself bounded
        return outputs['overall_confidence'].squeeze(-1).clamp(0.0, 1.0)


class ConfidenceCalibrator(LoggerMixin):
    """
    Platt-scaling calibrator mapping raw confidence scores to probabilities.

    Fits a logistic regression on the raw score so that a reported confidence
    of 0.9 corresponds to being correct about 90% of the time.
    """

    def __init__(self):
        self.slope: float = 1.0
        self.intercept: float = 0.0
        self.fitted: bool = False
        self.brier_score: float = float("nan")

    def fit(
        self,
        predicted_confidences: List[float],
        actual_outcomes: List[int],
        epochs: int = 500,
        learning_rate: float = 0.05
    ) -> "ConfidenceCalibrator":
        """Fit slope and intercept by minimizing binary cross-entropy."""
        x = torch.tensor(predicted_confidences, dtype=torch.float32)
        y = torch.tensor(actual_outcomes, dtype=torch.float32)

        slope = torch.ones(1, requires_grad=True)
        intercept = torch.zeros(1, requires_grad=True)

        optimizer = torch.optim.Adam([slope, intercept], lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(slope * x + intercept, y)
            loss.backward()
            optimizer.step()

        self.slope = float(slope.item())
        self.intercept = float(intercept.item())
        self.fitted = True

        # Brier score of the calibrated predictions: lower is better
        with torch.no_grad():
            calibrated = torch.sigmoid(slope * x + intercept)
            self.brier_score = float(((calibrated - y) ** 2).mean().item())

        return self

    def transform(self, confidence: float) -> float:
        """Apply the fitted mapping to one raw confidence score."""
        if not self.fitted:
            return confidence

        logit = self.slope * confidence + self.intercept
        return float(1.0 / (1.0 + math.exp(-logit)))

    def transform_many(self, confidences: List[float]) -> List[float]:
        """Apply the fitted mapping to a list of raw confidence scores."""
        return [self.transform(c) for c in confidences]


class ConfidenceScorer(LoggerMixin):
    """
    Main confidence scoring system that orchestrates all confidence assessments.

    This is the primary interface for confidence scoring in the LitKG system.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.config_path = config_path
        
        # Initialize component assessors
        self.literature_assessor = LiteratureConfidenceAssessor()
        self.experimental_assessor = ExperimentalConfidenceAssessor()
        self.cross_modal_integrator = CrossModalConfidenceIntegrator()
        
        # Move to device
        self.literature_assessor.to(device)
        self.experimental_assessor.to(device)
        self.cross_modal_integrator.to(device)
        
        # Evidence source mappings
        self.source_mappings = {
            'pubmed': 0,
            'civic': 1,
            'tcga': 2,
            'cptac': 3,
            'other': 4
        }
        
        self.logger.info(f"Initialized ConfidenceScorer on device: {device}")
    
    def assess_relationship_confidence(
        self,
        relationship: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        literature_data: Optional[Dict[str, Any]] = None,
        experimental_data: Optional[Dict[str, Any]] = None,
        relationship_embedding: Optional[torch.Tensor] = None
    ) -> ConfidenceMetrics:
        """
        Assess confidence for a biomedical relationship using all available evidence.

        Two entry points:

        - **Record level**: pass ``relationship`` and ``evidence`` (lists of raw
          literature/experimental records). Scoring runs through the component
          assessors' ``assess_confidence`` methods.
        - **Tensor level**: pass ``literature_data``/``experimental_data``
          dictionaries carrying embeddings, for the trained neural path.

        Args:
            relationship: {"head", "relation", "tail"} being assessed.
            evidence: {"literature": [...], "experimental": [...]} raw records.
            literature_data: Literature evidence with embeddings.
            experimental_data: Experimental evidence with embeddings.
            relationship_embedding: Optional pre-computed relationship embedding

        Returns:
            Comprehensive confidence metrics
        """
        if evidence is not None:
            return self._assess_from_records(relationship or {}, evidence)

        with torch.no_grad():
            # Initialize default values
            lit_confidence = torch.tensor([[0.0]], device=self.device)
            exp_confidence = torch.tensor([[0.0]], device=self.device)
            lit_features = torch.zeros(1, 768, device=self.device)  # Match embedding dimension
            exp_features = torch.zeros(1, 768, device=self.device)  # Match embedding dimension
            
            # Assess literature confidence if available
            if literature_data is not None:
                lit_results = self._assess_literature_confidence(literature_data)
                lit_confidence = lit_results['confidence']
                lit_features = lit_results.get('features', lit_features)
            
            # Assess experimental confidence if available
            if experimental_data is not None:
                exp_results = self._assess_experimental_confidence(experimental_data)
                exp_confidence = exp_results['confidence']
                exp_features = exp_results.get('features', exp_features)
            
            # Integrate cross-modal evidence
            integration_results = self.cross_modal_integrator(
                lit_features, exp_features, lit_confidence, exp_confidence
            )
            
            # Extract metrics
            overall_conf = integration_results['overall_confidence'].item()
            lit_conf = integration_results['literature_confidence'].item()
            exp_conf = integration_results['experimental_confidence'].item()
            agreement = integration_results['cross_modal_agreement'].item()
            conflict = integration_results['evidence_conflict'].item()
            epistemic_unc = integration_results['epistemic_uncertainty'].item()
            aleatoric_unc = integration_results['aleatoric_uncertainty'].item()
            
            # Count evidence sources
            supporting_papers = len(literature_data.get('papers', [])) if literature_data else 0
            supporting_experiments = len(experimental_data.get('experiments', [])) if experimental_data else 0
            contradicting_evidence = int(conflict > 0.5)
            
            # Generate explanation
            explanation = self._generate_confidence_explanation(
                overall_conf, lit_conf, exp_conf, agreement, conflict
            )
            
            return ConfidenceMetrics(
                overall_confidence=overall_conf,
                literature_confidence=lit_conf,
                experimental_confidence=exp_conf,
                cross_modal_agreement=agreement,
                evidence_strength=(lit_conf + exp_conf) / 2,
                source_reliability=max(lit_conf, exp_conf),
                temporal_consistency=1.0 - conflict,  # High conflict = low consistency
                epistemic_uncertainty=epistemic_unc,
                aleatoric_uncertainty=aleatoric_unc,
                supporting_papers=supporting_papers,
                supporting_experiments=supporting_experiments,
                contradicting_evidence=contradicting_evidence,
                explanation=explanation,
                evidence_sources=self._extract_evidence_sources(literature_data, experimental_data)
            )
    
    def _assess_literature_confidence(self, literature_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Assess confidence from literature evidence."""
        # Extract features from literature data
        papers = literature_data.get('papers', [])
        
        # Create mock embeddings and quality features for demonstration
        # In real implementation, these would come from actual paper analysis
        embedding = torch.randn(1, 768, device=self.device)
        
        # Quality features: [impact_factor, citations, recency, methodology, ...]
        quality_features = torch.tensor([[
            literature_data.get('avg_impact_factor', 5.0) / 10.0,  # Normalized
            min(literature_data.get('total_citations', 100) / 1000.0, 1.0),
            literature_data.get('recency_score', 0.8),
            literature_data.get('methodology_score', 0.7),
            len(papers) / 10.0,  # Number of supporting papers
            literature_data.get('consensus_score', 0.8),
            literature_data.get('journal_quality', 0.7),
            literature_data.get('author_reputation', 0.6),
            literature_data.get('study_design_quality', 0.7),
            literature_data.get('statistical_rigor', 0.8)
        ]], device=self.device)
        
        results = self.literature_assessor(embedding, quality_features)
        results['features'] = embedding  # Store for cross-modal integration
        
        return results
    
    def _assess_experimental_confidence(self, experimental_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Assess confidence from experimental evidence."""
        # Extract features from experimental data
        experiments = experimental_data.get('experiments', [])
        
        # Create mock embeddings for demonstration
        embedding = torch.randn(1, 768, device=self.device)
        
        # Source IDs
        source_name = experimental_data.get('primary_source', 'other')
        source_id = torch.tensor([self.source_mappings.get(source_name, 4)], device=self.device)
        
        # Statistical features: [sample_size, effect_size, p_value, replications, ...]
        statistical_features = torch.tensor([[
            min(experimental_data.get('sample_size', 100) / 1000.0, 1.0),
            min(experimental_data.get('effect_size', 0.5), 1.0),
            experimental_data.get('p_value', 0.05),
            experimental_data.get('num_replications', 1) / 5.0,
            experimental_data.get('consistency_score', 0.8),
            experimental_data.get('methodology_quality', 0.7),
            experimental_data.get('data_quality', 0.8),
            experimental_data.get('validation_score', 0.6)
        ]], device=self.device)
        
        results = self.experimental_assessor(embedding, source_id, statistical_features)
        results['features'] = embedding  # Store for cross-modal integration
        
        return results
    
    def _generate_confidence_explanation(
        self, overall: float, lit: float, exp: float, agreement: float, conflict: float
    ) -> str:
        """Generate human-readable explanation of confidence assessment."""
        explanations = []
        
        if overall >= 0.8:
            explanations.append("High confidence based on strong evidence")
        elif overall >= 0.5:
            explanations.append("Moderate confidence with some supporting evidence")
        else:
            explanations.append("Low confidence due to limited or conflicting evidence")
        
        if lit > 0.6 and exp > 0.6:
            explanations.append("both literature and experimental evidence are strong")
        elif lit > exp:
            explanations.append("primarily supported by literature evidence")
        elif exp > lit:
            explanations.append("primarily supported by experimental evidence")
        
        if conflict > 0.5:
            explanations.append("some conflicting evidence detected")
        elif agreement > 0.8:
            explanations.append("high agreement between evidence types")
        
        return "; ".join(explanations)
    
    def _extract_evidence_sources(
        self, literature_data: Optional[Dict], experimental_data: Optional[Dict]
    ) -> List[str]:
        """Extract list of evidence sources."""
        sources = []
        
        if literature_data:
            sources.extend([
                f"PubMed ({len(literature_data.get('papers', []))} papers)"
            ])
        
        if experimental_data:
            primary_source = experimental_data.get('primary_source', 'unknown')
            sources.append(f"{primary_source.upper()} database")
        
        return sources
    
    def _assess_from_records(
        self,
        relationship: Dict[str, Any],
        evidence: Dict[str, List[Dict[str, Any]]]
    ) -> ConfidenceMetrics:
        """
        Score a relationship from raw evidence records.

        Delegates to the component assessors, then fuses their scores through
        the cross-modal integrator, so each component's judgement is visible in
        the returned metrics.

        Args:
            relationship: {"head", "relation", "tail"}.
            evidence: {"literature": [...], "experimental": [...]}.

        Returns:
            Confidence metrics for the relationship.
        """
        literature_records = evidence.get("literature", []) or []
        experimental_records = evidence.get("experimental", []) or []

        lit_confidence = float(
            self.literature_assessor.assess_confidence(literature_records)
        )
        exp_confidence = float(
            self.experimental_assessor.assess_confidence(experimental_records)
        )

        hidden_dim = getattr(self.cross_modal_integrator, "hidden_dim", 768)
        lit_features = torch.full(
            (1, hidden_dim), lit_confidence, device=self.device
        )
        exp_features = torch.full(
            (1, hidden_dim), exp_confidence, device=self.device
        )

        with torch.no_grad():
            integrated = self.cross_modal_integrator.integrate(
                lit_features,
                exp_features,
                torch.tensor([[lit_confidence]], device=self.device),
                torch.tensor([[exp_confidence]], device=self.device),
            )

        overall = float(torch.as_tensor(integrated).flatten()[0].item())

        # Agreement is high when both modalities land on similar scores. With
        # only one modality present there is nothing to agree with.
        if literature_records and experimental_records:
            agreement = 1.0 - abs(lit_confidence - exp_confidence)
        else:
            agreement = 0.0

        explanation_parts = [
            f"{len(literature_records)} literature record(s) -> {lit_confidence:.2f}",
            f"{len(experimental_records)} experimental record(s) -> {exp_confidence:.2f}",
        ]
        if relationship:
            explanation_parts.insert(0, (
                f"{relationship.get('head', '?')} "
                f"{relationship.get('relation', '?')} "
                f"{relationship.get('tail', '?')}"
            ))

        return ConfidenceMetrics(
            overall_confidence=max(0.0, min(1.0, overall)),
            literature_confidence=lit_confidence,
            experimental_confidence=exp_confidence,
            cross_modal_agreement=float(max(0.0, agreement)),
            evidence_strength=float(max(lit_confidence, exp_confidence)),
            supporting_papers=len(literature_records),
            supporting_experiments=len(experimental_records),
            explanation="; ".join(explanation_parts),
            evidence_sources=(
                (["literature"] if literature_records else [])
                + (["experimental"] if experimental_records else [])
            ),
        )

    def calibrate_confidence(
        self,
        predicted_confidences: List[float],
        actual_outcomes: List[int]
    ) -> "ConfidenceCalibrator":
        """
        Fit a calibrator mapping raw confidence scores to observed frequencies.

        A model that says "0.9" should be right about 90% of the time. Platt
        scaling (logistic regression on the raw score) corrects the systematic
        over- or under-confidence that neural scorers typically exhibit.

        The fitted calibrator is stored on ``self.calibrator`` and applied by
        subsequent calls to :meth:`apply_calibration`.

        Args:
            predicted_confidences: Raw scores in [0, 1].
            actual_outcomes: Ground truth, 1 for correct and 0 for incorrect.

        Returns:
            The fitted ConfidenceCalibrator.

        Raises:
            ValueError: if the inputs differ in length or are empty.
        """
        if len(predicted_confidences) != len(actual_outcomes):
            raise ValueError(
                f"Length mismatch: {len(predicted_confidences)} predictions vs "
                f"{len(actual_outcomes)} outcomes"
            )
        if not predicted_confidences:
            raise ValueError("Cannot calibrate on empty data")

        calibrator = ConfidenceCalibrator()
        calibrator.fit(predicted_confidences, actual_outcomes)

        self.calibrator = calibrator
        self.logger.info(
            f"Calibrated confidence on {len(predicted_confidences)} samples "
            f"(Brier score {calibrator.brier_score:.4f})"
        )
        return calibrator

    def apply_calibration(self, confidence: float) -> float:
        """Map a raw confidence through the fitted calibrator, if any."""
        if getattr(self, "calibrator", None) is None:
            return confidence
        return self.calibrator.transform(confidence)

    def quantify_uncertainty(
        self,
        predictions: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Split predictive uncertainty into epistemic and aleatoric parts.

        Given repeated predictions for the same input (an ensemble, or MC
        dropout samples), this separates the two sources the README calls out
        as distinguishing "unknown" from "contradictory" evidence:

        - **Epistemic**: disagreement *between* samples. Reducible with more
          data or a better model; high when the model is out of its depth.
        - **Aleatoric**: the average entropy *within* each sample. Irreducible
          noise in the data itself; high when the evidence genuinely conflicts.

        Args:
            predictions: [num_samples, num_classes] probability distributions.

        Returns:
            (epistemic_uncertainty, aleatoric_uncertainty), both >= 0.

        Raises:
            ValueError: if predictions is not 2D or has no samples.
        """
        if predictions.dim() != 2:
            raise ValueError(
                f"Expected [num_samples, num_classes], got shape {tuple(predictions.shape)}"
            )
        if predictions.size(0) == 0:
            raise ValueError("Cannot quantify uncertainty from zero samples")

        probs = predictions.float()

        # Normalize defensively; callers may pass unnormalized scores
        row_sums = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(row_sums > 0, probs / row_sums, probs)

        eps = 1e-12

        # Total uncertainty: entropy of the mean prediction
        mean_probs = probs.mean(dim=0)
        total_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum()

        # Aleatoric: mean of the per-sample entropies
        sample_entropies = -(probs * torch.log(probs + eps)).sum(dim=-1)
        aleatoric = sample_entropies.mean()

        # Epistemic is the mutual information between prediction and parameters,
        # i.e. whatever total uncertainty is not explained by per-sample noise.
        epistemic = torch.clamp(total_entropy - aleatoric, min=0.0)

        return float(epistemic.item()), float(aleatoric.item())

    def batch_assess_confidence(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[ConfidenceMetrics]:
        """
        Assess confidence for multiple relationships in batch.
        
        Args:
            relationships: List of relationship data dictionaries
            
        Returns:
            List of confidence metrics for each relationship
        """
        results = []
        
        for relationship in relationships:
            confidence = self.assess_relationship_confidence(
                literature_data=relationship.get('literature_data'),
                experimental_data=relationship.get('experimental_data'),
                relationship_embedding=relationship.get('embedding')
            )
            results.append(confidence)
        
        self.logger.info(f"Assessed confidence for {len(relationships)} relationships")
        return results
    
    def save_model(self, save_path: str):
        """Save the confidence scoring models."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'literature_assessor': self.literature_assessor.state_dict(),
            'experimental_assessor': self.experimental_assessor.state_dict(),
            'cross_modal_integrator': self.cross_modal_integrator.state_dict(),
            'source_mappings': self.source_mappings
        }, save_path / 'confidence_scorer.pt')
        
        self.logger.info(f"Saved confidence scoring models to {save_path}")
    
    def load_model(self, load_path: str):
        """Load pre-trained confidence scoring models."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.literature_assessor.load_state_dict(checkpoint['literature_assessor'])
        self.experimental_assessor.load_state_dict(checkpoint['experimental_assessor'])
        self.cross_modal_integrator.load_state_dict(checkpoint['cross_modal_integrator'])
        self.source_mappings = checkpoint.get('source_mappings', self.source_mappings)
        
        self.logger.info(f"Loaded confidence scoring models from {load_path}")