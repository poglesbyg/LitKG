"""
Novel Relation Prediction and Discovery System

This module implements advanced algorithms for discovering novel biomedical relationships
by analyzing patterns in integrated literature and experimental data. It combines
traditional graph-based methods with LLM-powered reasoning for comprehensive discovery.

Key components:
1. Novel Relation Predictor - Identifies missing edges in the knowledge graph
2. Pattern Discovery Engine - Finds recurring patterns suggesting new relationships
3. Temporal Analysis - Tracks emergence of new relationships over time
4. Biological Plausibility Checker - Validates predictions using domain knowledge
5. Evidence Synthesizer - Aggregates supporting evidence for novel predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import networkx as nx
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split

# LangChain imports for LLM-powered reasoning
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Local imports
from ..utils.logging import LoggerMixin
from ..phase2.hybrid_gnn import HybridGNNModel
from .confidence_scoring import ConfidenceScorer, ConfidenceMetrics


@dataclass
class NovelRelation:
    """Represents a predicted novel relationship."""
    entity1: str
    entity2: str
    relation_type: str
    confidence_score: float
    evidence_sources: List[str]
    supporting_papers: List[str]
    biological_plausibility: float
    temporal_emergence: Optional[str] = None
    prediction_reasoning: Optional[str] = None
    validation_status: str = "pending"  # pending, validated, rejected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class DiscoveryPattern:
    """Represents a discovered pattern in the knowledge graph."""
    pattern_type: str  # "co_occurrence", "temporal", "pathway", "drug_target"
    entities: List[str]
    frequency: int
    confidence: float
    supporting_evidence: List[str]
    biological_context: str
    potential_relations: List[NovelRelation]


class NovelRelationPredictor(nn.Module, LoggerMixin):
    """
    Neural network for predicting novel biomedical relationships.
    
    Uses the trained hybrid GNN to identify missing edges in the knowledge graph
    by analyzing embedding similarities and learned attention patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_relation_types: int = 20,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relation_types = num_relation_types
        
        # Relation prediction layers
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relation_types),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_relation_types, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Novelty detector
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.logger.info("Initialized NovelRelationPredictor")
    
    def forward(
        self,
        entity_embeddings: torch.Tensor,
        entity_pairs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict novel relations between entity pairs.
        
        Args:
            entity_embeddings: [num_entities, hidden_dim]
            entity_pairs: [num_pairs, 2] - indices of entity pairs
            
        Returns:
            Dictionary with predictions, confidence, and novelty scores
        """
        # Get embeddings for entity pairs
        entity1_emb = entity_embeddings[entity_pairs[:, 0]]  # [num_pairs, hidden_dim]
        entity2_emb = entity_embeddings[entity_pairs[:, 1]]  # [num_pairs, hidden_dim]
        
        # Concatenate embeddings
        pair_embeddings = torch.cat([entity1_emb, entity2_emb], dim=1)  # [num_pairs, hidden_dim*2]
        
        # Predict relations
        relation_probs = self.relation_predictor(pair_embeddings)  # [num_pairs, num_relation_types]
        
        # Estimate confidence
        confidence_input = torch.cat([pair_embeddings, relation_probs], dim=1)
        confidence_scores = self.confidence_estimator(confidence_input)  # [num_pairs, 1]
        
        # Detect novelty
        novelty_scores = self.novelty_detector(pair_embeddings)  # [num_pairs, 1]
        
        return {
            "relation_predictions": relation_probs,
            "confidence_scores": confidence_scores.squeeze(1),
            "novelty_scores": novelty_scores.squeeze(1),
            "entity_pairs": entity_pairs
        }
    
    def predict_novel_relations(
        self,
        entity_embeddings: torch.Tensor,
        entity_names: List[str],
        relation_types: List[str],
        confidence_threshold: float = 0.7,
        novelty_threshold: float = 0.8,
        top_k: int = 100
    ) -> List[NovelRelation]:
        """
        Predict top-k novel relations with high confidence and novelty.
        
        Args:
            entity_embeddings: Learned entity embeddings
            entity_names: List of entity names corresponding to embeddings
            relation_types: List of relation type names
            confidence_threshold: Minimum confidence for predictions
            novelty_threshold: Minimum novelty score
            top_k: Number of top predictions to return
            
        Returns:
            List of predicted novel relations
        """
        self.eval()
        device = next(self.parameters()).device
        
        num_entities = len(entity_names)
        novel_relations = []
        
        # Generate all possible entity pairs (excluding self-pairs)
        entity_pairs = []
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                entity_pairs.append([i, j])
        
        entity_pairs = torch.tensor(entity_pairs, device=device)
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(entity_pairs), batch_size):
                batch_pairs = entity_pairs[i:i + batch_size]
                predictions = self.forward(entity_embeddings, batch_pairs)
                all_predictions.append(predictions)
        
        # Combine all predictions
        combined_predictions = {
            "relation_predictions": torch.cat([p["relation_predictions"] for p in all_predictions]),
            "confidence_scores": torch.cat([p["confidence_scores"] for p in all_predictions]),
            "novelty_scores": torch.cat([p["novelty_scores"] for p in all_predictions]),
            "entity_pairs": torch.cat([p["entity_pairs"] for p in all_predictions])
        }
        
        # Filter by confidence and novelty thresholds
        valid_mask = (
            (combined_predictions["confidence_scores"] >= confidence_threshold) &
            (combined_predictions["novelty_scores"] >= novelty_threshold)
        )
        
        if not valid_mask.any():
            self.logger.warning("No predictions meet the confidence and novelty thresholds")
            return []
        
        # Get valid predictions
        valid_indices = torch.where(valid_mask)[0]
        valid_pairs = combined_predictions["entity_pairs"][valid_indices]
        valid_relations = combined_predictions["relation_predictions"][valid_indices]
        valid_confidence = combined_predictions["confidence_scores"][valid_indices]
        valid_novelty = combined_predictions["novelty_scores"][valid_indices]
        
        # Get top relation type for each pair
        top_relation_indices = torch.argmax(valid_relations, dim=1)
        top_relation_scores = torch.max(valid_relations, dim=1)[0]
        
        # Combine scores (confidence * novelty * relation_score)
        combined_scores = valid_confidence * valid_novelty * top_relation_scores
        
        # Get top-k predictions
        top_k_indices = torch.topk(combined_scores, min(top_k, len(combined_scores)))[1]
        
        # Create NovelRelation objects
        for idx in top_k_indices:
            pair_idx = valid_indices[idx]
            entity_pair = valid_pairs[idx]
            relation_idx = top_relation_indices[idx]
            
            entity1_name = entity_names[entity_pair[0].item()]
            entity2_name = entity_names[entity_pair[1].item()]
            relation_type = relation_types[relation_idx.item()]
            
            novel_relation = NovelRelation(
                entity1=entity1_name,
                entity2=entity2_name,
                relation_type=relation_type,
                confidence_score=valid_confidence[idx].item(),
                evidence_sources=["hybrid_gnn_prediction"],
                supporting_papers=[],
                biological_plausibility=valid_novelty[idx].item(),
                prediction_reasoning=f"Predicted by hybrid GNN with {valid_confidence[idx].item():.3f} confidence"
            )
            
            novel_relations.append(novel_relation)
        
        self.logger.info(f"Predicted {len(novel_relations)} novel relations")
        return novel_relations


class PatternDiscoveryEngine(LoggerMixin):
    """
    Discovers recurring patterns in the integrated knowledge graph that suggest
    novel relationships or biological mechanisms.
    """
    
    def __init__(self, min_pattern_frequency: int = 3):
        self.min_pattern_frequency = min_pattern_frequency
        self.discovered_patterns = []
        self.logger.info("Initialized PatternDiscoveryEngine")
    
    def discover_co_occurrence_patterns(
        self,
        literature_graph: nx.Graph,
        entity_types: Dict[str, str]
    ) -> List[DiscoveryPattern]:
        """
        Discover entities that frequently co-occur in literature but lack
        explicit relationships in the knowledge graph.
        """
        patterns = []
        
        # Find entities that appear together frequently
        co_occurrence_counts = {}
        
        for paper_id in literature_graph.nodes():
            if literature_graph.nodes[paper_id].get('type') == 'paper':
                # Get all entities mentioned in this paper
                connected_entities = [
                    node for node in literature_graph.neighbors(paper_id)
                    if literature_graph.nodes[node].get('type') == 'entity'
                ]
                
                # Count co-occurrences
                for i, entity1 in enumerate(connected_entities):
                    for entity2 in connected_entities[i+1:]:
                        pair = tuple(sorted([entity1, entity2]))
                        co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1
        
        # Find frequent co-occurrences
        for (entity1, entity2), frequency in co_occurrence_counts.items():
            if frequency >= self.min_pattern_frequency:
                # Check if these entities have different types (more interesting)
                type1 = entity_types.get(entity1, "unknown")
                type2 = entity_types.get(entity2, "unknown")
                
                if type1 != type2:  # Different entity types
                    pattern = DiscoveryPattern(
                        pattern_type="co_occurrence",
                        entities=[entity1, entity2],
                        frequency=frequency,
                        confidence=min(frequency / 10.0, 1.0),  # Simple confidence scoring
                        supporting_evidence=[f"Co-occurred in {frequency} papers"],
                        biological_context=f"{type1}-{type2} interaction",
                        potential_relations=[]
                    )
                    patterns.append(pattern)
        
        self.logger.info(f"Discovered {len(patterns)} co-occurrence patterns")
        return patterns
    
    def discover_temporal_patterns(
        self,
        literature_graph: nx.Graph,
        publication_dates: Dict[str, datetime]
    ) -> List[DiscoveryPattern]:
        """
        Discover temporal patterns in relationship emergence.
        """
        patterns = []
        
        # Group papers by time periods
        time_windows = {}
        for paper_id, pub_date in publication_dates.items():
            year = pub_date.year
            window = f"{year//5 * 5}-{year//5 * 5 + 4}"  # 5-year windows
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(paper_id)
        
        # Analyze entity co-occurrence trends over time
        entity_trends = {}
        
        for window, papers in time_windows.items():
            window_entities = set()
            for paper_id in papers:
                if paper_id in literature_graph:
                    connected_entities = [
                        node for node in literature_graph.neighbors(paper_id)
                        if literature_graph.nodes[node].get('type') == 'entity'
                    ]
                    window_entities.update(connected_entities)
            
            # Track entity emergence
            for entity in window_entities:
                if entity not in entity_trends:
                    entity_trends[entity] = []
                entity_trends[entity].append(window)
        
        # Find entities with emerging patterns
        for entity, windows in entity_trends.items():
            if len(windows) >= 2:  # Entity appears in multiple time windows
                pattern = DiscoveryPattern(
                    pattern_type="temporal",
                    entities=[entity],
                    frequency=len(windows),
                    confidence=0.6,
                    supporting_evidence=[f"Appears across {len(windows)} time periods: {', '.join(windows)}"],
                    biological_context="Temporal emergence pattern",
                    potential_relations=[]
                )
                patterns.append(pattern)
        
        self.logger.info(f"Discovered {len(patterns)} temporal patterns")
        return patterns
    
    def discover_pathway_patterns(
        self,
        knowledge_graph: nx.Graph,
        pathway_annotations: Dict[str, List[str]]
    ) -> List[DiscoveryPattern]:
        """
        Discover potential pathway connections based on shared annotations.
        """
        patterns = []
        
        # Group entities by pathway annotations
        pathway_entities = {}
        for entity, pathways in pathway_annotations.items():
            for pathway in pathways:
                if pathway not in pathway_entities:
                    pathway_entities[pathway] = []
                pathway_entities[pathway].append(entity)
        
        # Find pathways with multiple entities that lack connections
        for pathway, entities in pathway_entities.items():
            if len(entities) >= 3:  # At least 3 entities in pathway
                # Check for missing connections
                missing_connections = []
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i+1:]:
                        if not knowledge_graph.has_edge(entity1, entity2):
                            missing_connections.append((entity1, entity2))
                
                if missing_connections:
                    pattern = DiscoveryPattern(
                        pattern_type="pathway",
                        entities=entities,
                        frequency=len(missing_connections),
                        confidence=0.7,
                        supporting_evidence=[f"Shared pathway annotation: {pathway}"],
                        biological_context=f"Pathway: {pathway}",
                        potential_relations=[
                            NovelRelation(
                                entity1=conn[0],
                                entity2=conn[1],
                                relation_type="pathway_interaction",
                                confidence_score=0.7,
                                evidence_sources=["pathway_annotation"],
                                supporting_papers=[],
                                biological_plausibility=0.8,
                                prediction_reasoning=f"Entities share pathway annotation: {pathway}"
                            )
                            for conn in missing_connections[:10]  # Limit to top 10
                        ]
                    )
                    patterns.append(pattern)
        
        self.logger.info(f"Discovered {len(patterns)} pathway patterns")
        return patterns


class BiologicalPlausibilityChecker(LoggerMixin):
    """
    Validates predicted relationships using biological domain knowledge
    and LLM-powered reasoning.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE
        
        if self.use_llm:
            try:
                # Try OpenAI first, fall back to Anthropic
                import os
                if os.getenv("OPENAI_API_KEY"):
                    self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                elif os.getenv("ANTHROPIC_API_KEY"):
                    self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.1)
                else:
                    self.use_llm = False
                    self.logger.warning("No LLM API keys found, using rule-based validation only")
            except Exception as e:
                self.use_llm = False
                self.logger.warning(f"Could not initialize LLM: {e}")
        
        # Biological plausibility rules
        self.plausibility_rules = {
            ("GENE", "DISEASE"): 0.8,
            ("DRUG", "DISEASE"): 0.9,
            ("DRUG", "GENE"): 0.7,
            ("PROTEIN", "DISEASE"): 0.8,
            ("PATHWAY", "DISEASE"): 0.6,
            ("GENE", "GENE"): 0.5,
            ("PROTEIN", "PROTEIN"): 0.6,
        }
        
        self.logger.info(f"Initialized BiologicalPlausibilityChecker (LLM: {self.use_llm})")
    
    def check_plausibility(
        self,
        novel_relation: NovelRelation,
        entity_types: Dict[str, str],
        context: Optional[str] = None
    ) -> float:
        """
        Check the biological plausibility of a predicted relationship.
        
        Args:
            novel_relation: The predicted relationship
            entity_types: Dictionary mapping entities to their types
            context: Optional biological context
            
        Returns:
            Plausibility score between 0 and 1
        """
        entity1_type = entity_types.get(novel_relation.entity1, "unknown")
        entity2_type = entity_types.get(novel_relation.entity2, "unknown")
        
        # Rule-based plausibility
        rule_score = self._get_rule_based_score(entity1_type, entity2_type, novel_relation.relation_type)
        
        # LLM-based plausibility (if available)
        if self.use_llm:
            llm_score = self._get_llm_based_score(novel_relation, entity1_type, entity2_type, context)
            # Combine rule-based and LLM scores
            final_score = 0.3 * rule_score + 0.7 * llm_score
        else:
            final_score = rule_score
        
        return final_score
    
    def _get_rule_based_score(self, type1: str, type2: str, relation_type: str) -> float:
        """Get plausibility score based on predefined rules."""
        type_pair = tuple(sorted([type1, type2]))
        base_score = self.plausibility_rules.get(type_pair, 0.3)
        
        # Adjust based on relation type
        relation_adjustments = {
            "TREATS": 1.2,
            "CAUSES": 1.1,
            "PREVENTS": 1.1,
            "INHIBITS": 1.0,
            "ACTIVATES": 1.0,
            "INTERACTS_WITH": 0.9,
            "ASSOCIATED_WITH": 0.8,
        }
        
        adjustment = relation_adjustments.get(relation_type, 1.0)
        return min(base_score * adjustment, 1.0)
    
    def _get_llm_based_score(
        self,
        novel_relation: NovelRelation,
        type1: str,
        type2: str,
        context: Optional[str]
    ) -> float:
        """Get plausibility score using LLM reasoning."""
        prompt_template = PromptTemplate(
            input_variables=["entity1", "entity2", "relation", "type1", "type2", "context"],
            template="""
You are a biomedical expert evaluating the biological plausibility of a predicted relationship.

Predicted Relationship:
- Entity 1: {entity1} (Type: {type1})
- Entity 2: {entity2} (Type: {type2})
- Relationship: {relation}
- Context: {context}

Please evaluate the biological plausibility of this relationship on a scale from 0 to 1, where:
- 0.0 = Biologically implausible or contradicts known biology
- 0.3 = Unlikely but theoretically possible
- 0.5 = Plausible but uncertain
- 0.7 = Likely based on biological principles
- 1.0 = Highly plausible and well-supported by biological knowledge

Consider:
1. Known biological mechanisms
2. Molecular interactions
3. Cellular pathways
4. Disease mechanisms
5. Drug mechanisms of action

Provide only a numerical score (0.0 to 1.0) followed by a brief explanation.

Score:"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(
                entity1=novel_relation.entity1,
                entity2=novel_relation.entity2,
                relation=novel_relation.relation_type,
                type1=type1,
                type2=type2,
                context=context or "No specific context provided"
            )
            
            # Extract numerical score from result
            import re
            score_match = re.search(r'(\d*\.?\d+)', result.strip())
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            else:
                self.logger.warning(f"Could not extract score from LLM response: {result}")
                return 0.5  # Default score
                
        except Exception as e:
            self.logger.error(f"Error getting LLM plausibility score: {e}")
            return 0.5  # Default score


class NoveltyDetectionSystem(LoggerMixin):
    """
    Main system for novel relation prediction and hypothesis generation.
    
    Integrates all components to provide a comprehensive discovery pipeline.
    """
    
    def __init__(
        self,
        hybrid_gnn_model: Optional[HybridGNNModel] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        use_llm_validation: bool = True
    ):
        self.hybrid_gnn_model = hybrid_gnn_model
        self.confidence_scorer = confidence_scorer
        
        # Initialize components
        self.relation_predictor = NovelRelationPredictor()
        self.pattern_engine = PatternDiscoveryEngine()
        self.plausibility_checker = BiologicalPlausibilityChecker(use_llm=use_llm_validation)
        
        # Discovery results
        self.novel_relations = []
        self.discovery_patterns = []
        self.validation_results = {}
        
        self.logger.info("Initialized NoveltyDetectionSystem")
    
    def discover_novel_relations(
        self,
        entity_embeddings: torch.Tensor,
        entity_names: List[str],
        entity_types: Dict[str, str],
        relation_types: List[str],
        literature_graph: nx.Graph,
        knowledge_graph: nx.Graph,
        confidence_threshold: float = 0.7,
        novelty_threshold: float = 0.8,
        max_predictions: int = 50
    ) -> List[NovelRelation]:
        """
        Complete pipeline for discovering novel biomedical relationships.
        
        Args:
            entity_embeddings: Learned embeddings from hybrid GNN
            entity_names: List of entity names
            entity_types: Entity type mapping
            relation_types: List of possible relation types
            literature_graph: Literature co-occurrence graph
            knowledge_graph: Structured knowledge graph
            confidence_threshold: Minimum confidence for predictions
            novelty_threshold: Minimum novelty score
            max_predictions: Maximum number of predictions to return
            
        Returns:
            List of validated novel relations
        """
        self.logger.info("Starting novel relation discovery pipeline")
        
        # Step 1: Neural prediction of novel relations
        neural_predictions = self.relation_predictor.predict_novel_relations(
            entity_embeddings=entity_embeddings,
            entity_names=entity_names,
            relation_types=relation_types,
            confidence_threshold=confidence_threshold,
            novelty_threshold=novelty_threshold,
            top_k=max_predictions * 2  # Get more candidates for filtering
        )
        
        self.logger.info(f"Neural predictor found {len(neural_predictions)} candidates")
        
        # Step 2: Pattern-based discovery
        co_occurrence_patterns = self.pattern_engine.discover_co_occurrence_patterns(
            literature_graph, entity_types
        )
        
        # Extract relations from patterns
        pattern_relations = []
        for pattern in co_occurrence_patterns:
            if len(pattern.entities) == 2:
                relation = NovelRelation(
                    entity1=pattern.entities[0],
                    entity2=pattern.entities[1],
                    relation_type="ASSOCIATED_WITH",
                    confidence_score=pattern.confidence,
                    evidence_sources=["co_occurrence_pattern"],
                    supporting_papers=[],
                    biological_plausibility=0.6,
                    prediction_reasoning=f"Co-occurrence pattern: {pattern.supporting_evidence[0]}"
                )
                pattern_relations.append(relation)
        
        self.logger.info(f"Pattern discovery found {len(pattern_relations)} candidates")
        
        # Step 3: Combine and deduplicate predictions
        all_predictions = neural_predictions + pattern_relations
        unique_predictions = self._deduplicate_relations(all_predictions)
        
        # Step 4: Biological plausibility validation
        validated_relations = []
        for relation in unique_predictions[:max_predictions]:
            plausibility_score = self.plausibility_checker.check_plausibility(
                relation, entity_types
            )
            
            relation.biological_plausibility = plausibility_score
            
            # Only keep biologically plausible relations
            if plausibility_score >= 0.5:
                validated_relations.append(relation)
        
        # Step 5: Final confidence scoring using multi-modal system
        if self.confidence_scorer:
            for relation in validated_relations:
                confidence_metrics = self._assess_relation_confidence(relation)
                relation.confidence_score = confidence_metrics.overall_confidence
        
        # Sort by combined score
        validated_relations.sort(
            key=lambda r: r.confidence_score * r.biological_plausibility,
            reverse=True
        )
        
        self.novel_relations = validated_relations
        self.logger.info(f"Discovery pipeline completed: {len(validated_relations)} validated novel relations")
        
        return validated_relations
    
    def _deduplicate_relations(self, relations: List[NovelRelation]) -> List[NovelRelation]:
        """Remove duplicate relations based on entity pairs."""
        seen_pairs = set()
        unique_relations = []
        
        for relation in relations:
            # Normalize entity pair (sort alphabetically)
            pair = tuple(sorted([relation.entity1, relation.entity2]))
            
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_relations.append(relation)
        
        return unique_relations
    
    def _assess_relation_confidence(self, relation: NovelRelation) -> ConfidenceMetrics:
        """Assess confidence using the multi-modal confidence scoring system."""
        if not self.confidence_scorer:
            return ConfidenceMetrics(
                overall_confidence=relation.confidence_score,
                literature_confidence=0.5,
                experimental_confidence=0.5,
                cross_modal_agreement=0.5,
                epistemic_uncertainty=0.5,
                aleatoric_uncertainty=0.5
            )
        
        # Create mock evidence for confidence assessment
        literature_evidence = {
            "papers": relation.supporting_papers,
            "citations": len(relation.supporting_papers) * 10,  # Mock citation count
            "journal_quality": 0.7,
            "methodology_score": 0.6
        }
        
        experimental_evidence = {
            "experiments": [],
            "sample_size": 0,
            "statistical_significance": 0.0,
            "effect_size": 0.0,
            "reproducibility": 0.0
        }
        
        return self.confidence_scorer.assess_relationship_confidence(
            entity1=relation.entity1,
            entity2=relation.entity2,
            relation_type=relation.relation_type,
            literature_evidence=literature_evidence,
            experimental_evidence=experimental_evidence
        )
    
    def generate_discovery_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report of discovery results.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Dictionary containing the complete discovery report
        """
        report = {
            "discovery_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_novel_relations": len(self.novel_relations),
                "high_confidence_relations": len([
                    r for r in self.novel_relations 
                    if r.confidence_score >= 0.8 and r.biological_plausibility >= 0.7
                ]),
                "discovery_patterns": len(self.discovery_patterns)
            },
            "novel_relations": [relation.to_dict() for relation in self.novel_relations],
            "top_discoveries": [
                relation.to_dict() for relation in self.novel_relations[:10]
            ],
            "biological_categories": self._categorize_discoveries(),
            "validation_statistics": self._compute_validation_stats(),
            "methodology": {
                "neural_prediction": "Hybrid GNN with cross-modal attention",
                "pattern_discovery": "Co-occurrence and temporal analysis",
                "plausibility_validation": "Rule-based + LLM reasoning",
                "confidence_assessment": "Multi-modal evidence integration"
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Discovery report saved to {output_path}")
        
        return report
    
    def _categorize_discoveries(self) -> Dict[str, int]:
        """Categorize discoveries by biological type."""
        categories = {}
        for relation in self.novel_relations:
            relation_type = relation.relation_type
            categories[relation_type] = categories.get(relation_type, 0) + 1
        return categories
    
    def _compute_validation_stats(self) -> Dict[str, float]:
        """Compute validation statistics."""
        if not self.novel_relations:
            return {}
        
        confidence_scores = [r.confidence_score for r in self.novel_relations]
        plausibility_scores = [r.biological_plausibility for r in self.novel_relations]
        
        return {
            "mean_confidence": np.mean(confidence_scores),
            "std_confidence": np.std(confidence_scores),
            "mean_plausibility": np.mean(plausibility_scores),
            "std_plausibility": np.std(plausibility_scores),
            "high_quality_percentage": len([
                r for r in self.novel_relations 
                if r.confidence_score >= 0.7 and r.biological_plausibility >= 0.7
            ]) / len(self.novel_relations) * 100
        }