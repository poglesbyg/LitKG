"""
Confidence scoring system for multi-modal biomedical evidence.

This module implements sophisticated confidence assessment for relationships
derived from different sources (literature vs. experimental data), considering
factors like evidence strength, source reliability, temporal consistency,
and cross-modal agreement.
"""

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
    COMPUTATIONAL_PREDICTION = "computational_prediction"
    EXPERT_ANNOTATION = "expert_annotation"


@dataclass
class ConfidenceMetrics:
    """Container for confidence assessment metrics."""
    
    # Overall confidence score [0, 1]
    overall_confidence: float
    
    # Evidence-specific scores
    literature_confidence: float
    experimental_confidence: float
    cross_modal_agreement: float
    
    # Quality indicators
    evidence_strength: float
    source_reliability: float
    temporal_consistency: float
    
    # Uncertainty quantification
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    
    # Supporting evidence counts
    supporting_papers: int
    supporting_experiments: int
    contradicting_evidence: int
    
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


class CrossModalConfidenceIntegrator(nn.Module, LoggerMixin):
    """
    Integrate confidence assessments across literature and experimental evidence.
    
    This component resolves conflicts between different evidence types and
    provides unified confidence scores with uncertainty quantification.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
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
        literature_data: Optional[Dict[str, Any]] = None,
        experimental_data: Optional[Dict[str, Any]] = None,
        relationship_embedding: Optional[torch.Tensor] = None
    ) -> ConfidenceMetrics:
        """
        Assess confidence for a biomedical relationship using all available evidence.
        
        Args:
            literature_data: Literature evidence data
            experimental_data: Experimental evidence data
            relationship_embedding: Optional pre-computed relationship embedding
            
        Returns:
            Comprehensive confidence metrics
        """
        with torch.no_grad():
            # Initialize default values
            lit_confidence = torch.tensor([[0.0]], device=self.device)
            exp_confidence = torch.tensor([[0.0]], device=self.device)
            lit_features = torch.zeros(1, 256, device=self.device)
            exp_features = torch.zeros(1, 256, device=self.device)
            
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