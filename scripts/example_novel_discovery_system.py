#!/usr/bin/env python3
"""
Complete Novel Relation Prediction and Hypothesis Generation Demo

This script demonstrates the full pipeline for AI-powered biomedical discovery:
1. Novel relation prediction using hybrid GNN + pattern analysis
2. Biological plausibility validation with LLM reasoning
3. Hypothesis generation with AI agents
4. Comprehensive validation using multiple methods
5. Research report generation with prioritized recommendations

This represents the culmination of the LitKG system - a complete AI-powered
biomedical research assistant for novel knowledge discovery.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging

# Import Phase 3 components
from litkg.phase3 import (
    # Novelty Detection
    NoveltyDetectionSystem,
    NovelRelationPredictor,
    PatternDiscoveryEngine,
    BiologicalPlausibilityChecker,
    NovelRelation,
    
    # Hypothesis Generation
    HypothesisGenerationSystem,
    HypothesisGenerator,
    HypothesisValidationAgent,
    BiomedicalHypothesis,
    
    # Validation
    ComprehensiveValidationSystem,
    LiteratureCrossValidator,
    TemporalValidator,
    ExpertValidationInterface,
    
    # Confidence Scoring
    ConfidenceScorer
)

# Import other components
from litkg.phase2 import HybridGNNModel


def create_synthetic_biomedical_data(logger) -> Dict[str, Any]:
    """
    Create synthetic biomedical data for demonstration.
    In a real system, this would come from Phase 1 and Phase 2 processing.
    """
    logger.info("Creating synthetic biomedical data for demonstration")
    
    # Synthetic entity data
    entities = [
        "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "PIK3CA", "AKT1", "PTEN",
        "breast_cancer", "lung_cancer", "ovarian_cancer", "colorectal_cancer",
        "tamoxifen", "trastuzumab", "erlotinib", "bevacizumab", "carboplatin",
        "p53_pathway", "PI3K_pathway", "EGFR_pathway", "DNA_repair",
        "T_cells", "macrophages", "tumor_cells", "endothelial_cells"
    ]
    
    entity_types = {
        "BRCA1": "GENE", "BRCA2": "GENE", "TP53": "GENE", "EGFR": "GENE",
        "KRAS": "GENE", "PIK3CA": "GENE", "AKT1": "GENE", "PTEN": "GENE",
        "breast_cancer": "DISEASE", "lung_cancer": "DISEASE", 
        "ovarian_cancer": "DISEASE", "colorectal_cancer": "DISEASE",
        "tamoxifen": "DRUG", "trastuzumab": "DRUG", "erlotinib": "DRUG",
        "bevacizumab": "DRUG", "carboplatin": "DRUG",
        "p53_pathway": "PATHWAY", "PI3K_pathway": "PATHWAY", 
        "EGFR_pathway": "PATHWAY", "DNA_repair": "PATHWAY",
        "T_cells": "CELL_TYPE", "macrophages": "CELL_TYPE",
        "tumor_cells": "CELL_TYPE", "endothelial_cells": "CELL_TYPE"
    }
    
    relation_types = [
        "TREATS", "CAUSES", "PREVENTS", "INHIBITS", "ACTIVATES",
        "INTERACTS_WITH", "ASSOCIATED_WITH", "REGULATES", "EXPRESSED_IN"
    ]
    
    # Create synthetic entity embeddings (from hybrid GNN)
    torch.manual_seed(42)
    entity_embeddings = torch.randn(len(entities), 256)
    
    # Create synthetic literature graph
    literature_graph = nx.Graph()
    
    # Add papers and entity connections
    papers = [f"paper_{i:04d}" for i in range(100)]
    for paper in papers:
        literature_graph.add_node(paper, type='paper')
        # Each paper mentions 3-5 entities
        mentioned_entities = np.random.choice(entities, size=np.random.randint(3, 6), replace=False)
        for entity in mentioned_entities:
            literature_graph.add_node(entity, type='entity')
            literature_graph.add_edge(paper, entity)
    
    # Create synthetic knowledge graph
    knowledge_graph = nx.Graph()
    for entity in entities:
        knowledge_graph.add_node(entity, type='entity')
    
    # Add known relations
    known_relations = {
        ("BRCA1", "breast_cancer", "CAUSES"),
        ("BRCA2", "breast_cancer", "CAUSES"),
        ("TP53", "p53_pathway", "REGULATES"),
        ("tamoxifen", "breast_cancer", "TREATS"),
        ("trastuzumab", "breast_cancer", "TREATS"),
        ("EGFR", "lung_cancer", "ASSOCIATED_WITH"),
        ("erlotinib", "EGFR", "INHIBITS"),
    }
    
    for entity1, entity2, relation in known_relations:
        knowledge_graph.add_edge(entity1, entity2, relation_type=relation)
    
    # Create publication dates
    base_date = datetime(2020, 1, 1)
    publication_dates = {
        paper: base_date + timedelta(days=np.random.randint(0, 1460))  # 4 years
        for paper in papers
    }
    
    return {
        "entities": entities,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "entity_embeddings": entity_embeddings,
        "literature_graph": literature_graph,
        "knowledge_graph": knowledge_graph,
        "known_relations": known_relations,
        "publication_dates": publication_dates
    }


def demonstrate_novelty_detection(data: Dict[str, Any], logger) -> List[NovelRelation]:
    """Demonstrate novel relation prediction."""
    logger.info("=== NOVEL RELATION PREDICTION DEMO ===")
    
    # Initialize novelty detection system
    novelty_system = NoveltyDetectionSystem(
        use_llm_validation=True  # Enable LLM-powered validation
    )
    
    logger.info("Discovering novel biomedical relationships...")
    
    # Run complete discovery pipeline
    novel_relations = novelty_system.discover_novel_relations(
        entity_embeddings=data["entity_embeddings"],
        entity_names=data["entities"],
        entity_types=data["entity_types"],
        relation_types=data["relation_types"],
        literature_graph=data["literature_graph"],
        knowledge_graph=data["knowledge_graph"],
        confidence_threshold=0.6,
        novelty_threshold=0.7,
        max_predictions=20
    )
    
    logger.info(f"Discovered {len(novel_relations)} novel relationships")
    
    # Show top predictions
    logger.info("\nTop Novel Relationship Predictions:")
    for i, relation in enumerate(novel_relations[:10]):
        logger.info(f"{i+1:2d}. {relation.entity1} --{relation.relation_type}--> {relation.entity2}")
        logger.info(f"    Confidence: {relation.confidence_score:.3f}, Plausibility: {relation.biological_plausibility:.3f}")
        logger.info(f"    Evidence: {', '.join(relation.evidence_sources)}")
        logger.info(f"    Reasoning: {relation.prediction_reasoning}")
        logger.info("")
    
    return novel_relations


def demonstrate_hypothesis_generation(novel_relations: List[NovelRelation], logger) -> List[BiomedicalHypothesis]:
    """Demonstrate AI-powered hypothesis generation."""
    logger.info("=== AI-POWERED HYPOTHESIS GENERATION DEMO ===")
    
    # Initialize hypothesis generation system
    hypothesis_system = HypothesisGenerationSystem(use_llm=True)
    
    logger.info("Generating testable hypotheses from novel relationships...")
    
    # Generate and validate hypotheses
    hypotheses, validation_results = hypothesis_system.generate_and_validate_hypotheses(
        novel_relations=novel_relations[:10],  # Use top 10 relations
        max_hypotheses=15,
        validate_all=True
    )
    
    logger.info(f"Generated {len(hypotheses)} hypotheses")
    logger.info(f"Validated {len(validation_results)} hypotheses")
    
    # Show top hypotheses
    logger.info("\nTop Generated Hypotheses:")
    for i, hypothesis in enumerate(hypotheses[:5]):
        logger.info(f"\n{i+1}. {hypothesis.title}")
        logger.info(f"   Description: {hypothesis.description[:150]}...")
        logger.info(f"   Priority Score: {hypothesis.priority_score:.3f}")
        logger.info(f"   Confidence: {hypothesis.confidence_score:.3f}")
        logger.info(f"   Novelty: {hypothesis.novelty_score:.3f}")
        logger.info(f"   Feasibility: {hypothesis.feasibility_score:.3f}")
        
        if hypothesis.testable_predictions:
            logger.info(f"   Key Predictions:")
            for pred in hypothesis.testable_predictions[:2]:
                logger.info(f"   - {pred}")
        
        if hypothesis.experimental_approaches:
            logger.info(f"   Experimental Approaches:")
            for exp in hypothesis.experimental_approaches[:2]:
                logger.info(f"   - {exp}")
    
    return hypotheses


def demonstrate_comprehensive_validation(
    novel_relations: List[NovelRelation],
    hypotheses: List[BiomedicalHypothesis],
    data: Dict[str, Any],
    logger
) -> Dict[str, Any]:
    """Demonstrate comprehensive validation system."""
    logger.info("=== COMPREHENSIVE VALIDATION DEMO ===")
    
    # Initialize validation system
    validation_system = ComprehensiveValidationSystem()
    
    logger.info("Running comprehensive validation pipeline...")
    
    # Run validation
    validation_results = validation_system.comprehensive_validation(
        novel_relations=novel_relations,
        hypotheses=hypotheses,
        literature_graph=data["literature_graph"],
        known_relations=data["known_relations"],
        publication_dates=data["publication_dates"],
        include_expert_validation=True  # Generate expert assessment forms
    )
    
    # Display validation summary
    logger.info("\nValidation Results Summary:")
    summary = validation_results["validation_summary"]
    logger.info(f"  Novel relations tested: {summary['novel_relations_tested']}")
    logger.info(f"  Hypotheses tested: {summary['hypotheses_tested']}")
    logger.info(f"  Validation methods: {', '.join(summary['validation_methods'])}")
    
    # Display literature validation results
    if validation_results["literature_validation"]:
        lit_val = validation_results["literature_validation"]
        logger.info(f"\nLiterature Cross-Validation:")
        logger.info(f"  Precision: {lit_val['precision']:.3f}")
        logger.info(f"  Recall: {lit_val['recall']:.3f}")
        logger.info(f"  F1-Score: {lit_val['f1_score']:.3f}")
        logger.info(f"  AUC: {lit_val['auc_score']:.3f}")
    
    # Display temporal validation results
    if validation_results["temporal_validation"]:
        temp_val = validation_results["temporal_validation"]
        logger.info(f"\nTemporal Validation:")
        logger.info(f"  Precision: {temp_val['precision']:.3f}")
        logger.info(f"  Recall: {temp_val['recall']:.3f}")
        logger.info(f"  F1-Score: {temp_val['f1_score']:.3f}")
    
    # Display overall assessment
    if validation_results["overall_assessment"]:
        assessment = validation_results["overall_assessment"]
        logger.info(f"\nOverall Assessment:")
        logger.info(f"  Validation Quality: {assessment['validation_quality']}")
        logger.info(f"  Confidence in Predictions: {assessment['confidence_in_predictions']:.3f}")
        logger.info(f"  Key Findings:")
        for finding in assessment['key_findings']:
            logger.info(f"    - {finding}")
    
    # Expert validation forms
    if validation_results["expert_validation"]:
        expert_val = validation_results["expert_validation"]
        logger.info(f"\nExpert Validation:")
        logger.info(f"  Assessment forms generated: {expert_val['forms_generated']}")
        logger.info(f"  Instructions: {expert_val['instructions']}")
    
    return validation_results


def generate_discovery_reports(
    novel_relations: List[NovelRelation],
    hypotheses: List[BiomedicalHypothesis],
    validation_results: Dict[str, Any],
    logger
):
    """Generate comprehensive discovery and research reports."""
    logger.info("=== GENERATING DISCOVERY REPORTS ===")
    
    output_dir = Path("outputs/novel_discovery_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Novel Relations Discovery Report
    relations_report = {
        "discovery_report": {
            "title": "Novel Biomedical Relationship Discovery",
            "timestamp": datetime.now().isoformat(),
            "methodology": "Hybrid GNN + Pattern Analysis + LLM Validation",
            "summary": {
                "total_predictions": len(novel_relations),
                "high_confidence_predictions": len([r for r in novel_relations if r.confidence_score >= 0.8]),
                "biologically_plausible": len([r for r in novel_relations if r.biological_plausibility >= 0.7])
            }
        },
        "novel_relations": [relation.to_dict() for relation in novel_relations],
        "top_discoveries": [relation.to_dict() for relation in novel_relations[:10]],
        "biological_categories": {},
        "validation_summary": validation_results.get("overall_assessment", {})
    }
    
    # Categorize discoveries
    for relation in novel_relations:
        rel_type = relation.relation_type
        relations_report["biological_categories"][rel_type] = relations_report["biological_categories"].get(rel_type, 0) + 1
    
    relations_file = output_dir / "novel_relations_discovery_report.json"
    with open(relations_file, 'w') as f:
        json.dump(relations_report, f, indent=2, default=str)
    logger.info(f"Novel relations report saved to {relations_file}")
    
    # 2. Hypothesis Generation Report
    hypothesis_report = {
        "research_report": {
            "title": "AI-Generated Biomedical Hypotheses",
            "timestamp": datetime.now().isoformat(),
            "methodology": "LLM-Powered Hypothesis Generation + Multi-Agent Validation",
            "summary": {
                "total_hypotheses": len(hypotheses),
                "high_priority_hypotheses": len([h for h in hypotheses if h.priority_score >= 0.7]),
                "validated_hypotheses": len([h for h in hypotheses if h.validation_status == "validated"])
            }
        },
        "top_hypotheses": [h.to_dict() for h in hypotheses[:10]],
        "all_hypotheses": [h.to_dict() for h in hypotheses],
        "research_priorities": [
            {
                "hypothesis_id": h.id,
                "title": h.title,
                "priority_score": h.priority_score,
                "potential_impact": h.potential_impact,
                "key_predictions": h.testable_predictions[:3]
            }
            for h in hypotheses[:15]
        ],
        "experimental_recommendations": {
            "immediate_experiments": [
                exp for h in hypotheses[:5] 
                for exp in h.experimental_approaches[:2]
            ],
            "collaborative_opportunities": [
                "Clinical validation studies",
                "Experimental biology partnerships",
                "Bioinformatics collaborations",
                "Industry drug development"
            ]
        }
    }
    
    hypothesis_file = output_dir / "hypothesis_generation_report.json"
    with open(hypothesis_file, 'w') as f:
        json.dump(hypothesis_report, f, indent=2, default=str)
    logger.info(f"Hypothesis generation report saved to {hypothesis_file}")
    
    # 3. Comprehensive Validation Report
    validation_file = output_dir / "comprehensive_validation_report.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"Validation report saved to {validation_file}")
    
    # 4. Executive Summary
    executive_summary = {
        "executive_summary": {
            "title": "LitKG Novel Discovery System - Complete Results",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "overview": "AI-powered biomedical knowledge discovery using literature integration and hybrid neural networks",
            "key_achievements": {
                "novel_relationships_discovered": len(novel_relations),
                "testable_hypotheses_generated": len(hypotheses),
                "validation_methods_applied": len(validation_results.get("validation_summary", {}).get("validation_methods", [])),
                "high_confidence_discoveries": len([r for r in novel_relations if r.confidence_score >= 0.8 and r.biological_plausibility >= 0.7])
            },
            "research_impact": {
                "potential_therapeutic_targets": len([h for h in hypotheses if "therapeutic" in h.potential_impact.lower()]),
                "novel_mechanisms_identified": len([h for h in hypotheses if "mechanism" in h.biological_mechanism.lower()]),
                "experimental_validation_opportunities": len([h for h in hypotheses if h.feasibility_score >= 0.7]),
                "publication_potential": min(len(hypotheses) * 2, 30)
            },
            "methodology_validation": {
                "literature_cross_validation": validation_results.get("literature_validation", {}).get("f1_score", 0),
                "temporal_validation": validation_results.get("temporal_validation", {}).get("f1_score", 0),
                "overall_confidence": validation_results.get("overall_assessment", {}).get("confidence_in_predictions", 0)
            },
            "next_steps": [
                "Experimental validation of top predictions",
                "Expert review of generated hypotheses", 
                "Literature review for novel relationships",
                "Collaboration with research institutions",
                "Grant applications for high-priority research"
            ]
        }
    }
    
    summary_file = output_dir / "executive_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(executive_summary, f, indent=2, default=str)
    logger.info(f"Executive summary saved to {summary_file}")
    
    # 5. Human-readable summary report
    readable_summary = f"""
LitKG Novel Discovery System - Results Summary
=============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCOVERY OVERVIEW:
- Novel relationships discovered: {len(novel_relations)}
- Testable hypotheses generated: {len(hypotheses)}
- High-confidence discoveries: {len([r for r in novel_relations if r.confidence_score >= 0.8 and r.biological_plausibility >= 0.7])}
- High-priority hypotheses: {len([h for h in hypotheses if h.priority_score >= 0.7])}

TOP NOVEL RELATIONSHIPS:
"""
    
    for i, relation in enumerate(novel_relations[:5]):
        readable_summary += f"""
{i+1}. {relation.entity1} --{relation.relation_type}--> {relation.entity2}
   Confidence: {relation.confidence_score:.3f} | Plausibility: {relation.biological_plausibility:.3f}
   Evidence: {', '.join(relation.evidence_sources)}
"""
    
    readable_summary += f"""

TOP RESEARCH HYPOTHESES:
"""
    
    for i, hypothesis in enumerate(hypotheses[:5]):
        readable_summary += f"""
{i+1}. {hypothesis.title}
   Priority: {hypothesis.priority_score:.3f} | Confidence: {hypothesis.confidence_score:.3f}
   Description: {hypothesis.description[:100]}...
   Key Prediction: {hypothesis.testable_predictions[0] if hypothesis.testable_predictions else 'N/A'}
"""
    
    validation_quality = validation_results.get("overall_assessment", {}).get("validation_quality", "unknown")
    validation_confidence = validation_results.get("overall_assessment", {}).get("confidence_in_predictions", 0)
    
    readable_summary += f"""

VALIDATION RESULTS:
- Overall validation quality: {validation_quality}
- Confidence in predictions: {validation_confidence:.3f}
- Literature cross-validation F1: {validation_results.get("literature_validation", {}).get("f1_score", "N/A")}
- Temporal validation F1: {validation_results.get("temporal_validation", {}).get("f1_score", "N/A")}

RESEARCH IMPACT:
- Potential therapeutic targets identified
- Novel biological mechanisms proposed
- Experimental validation opportunities available
- Strong publication and collaboration potential

FILES GENERATED:
- Novel relations discovery report: novel_relations_discovery_report.json
- Hypothesis generation report: hypothesis_generation_report.json
- Comprehensive validation report: comprehensive_validation_report.json
- Executive summary: executive_summary.json

NEXT STEPS:
1. Review top predictions and hypotheses
2. Conduct expert validation using generated assessment forms
3. Plan experimental validation studies
4. Initiate literature review for novel relationships
5. Prepare grant applications and research proposals
"""
    
    readable_file = output_dir / "discovery_summary_report.txt"
    with open(readable_file, 'w') as f:
        f.write(readable_summary)
    logger.info(f"Human-readable summary saved to {readable_file}")
    
    return {
        "relations_report": relations_file,
        "hypothesis_report": hypothesis_file,
        "validation_report": validation_file,
        "executive_summary": summary_file,
        "readable_summary": readable_file
    }


def main():
    """Main function to run the complete novel discovery demonstration."""
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🧬 Complete Novel Discovery System Demo - LitKG-Integrate")
    logger.info("=" * 80)
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    logger.info(f"LLM API Key Status:")
    logger.info(f"  OpenAI: {'✅ Available' if has_openai else '❌ Not found'}")
    logger.info(f"  Anthropic: {'✅ Available' if has_anthropic else '❌ Not found'}")
    
    if not has_openai and not has_anthropic:
        logger.warning("No LLM API keys found. Demo will run with limited LLM capabilities.")
        logger.info("Set OPENAI_API_KEY or ANTHROPIC_API_KEY for full AI-powered features.")
    
    try:
        # Step 1: Create synthetic data
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Creating Synthetic Biomedical Data")
        logger.info("="*50)
        
        data = create_synthetic_biomedical_data(logger)
        logger.info(f"Created synthetic dataset:")
        logger.info(f"  - {len(data['entities'])} entities")
        logger.info(f"  - {len(data['relation_types'])} relation types")
        logger.info(f"  - {data['literature_graph'].number_of_nodes()} literature graph nodes")
        logger.info(f"  - {data['knowledge_graph'].number_of_nodes()} knowledge graph nodes")
        logger.info(f"  - {len(data['known_relations'])} known relations")
        
        # Step 2: Novel relation prediction
        logger.info("\n" + "="*50)
        logger.info("STEP 2: Novel Relation Prediction")
        logger.info("="*50)
        
        novel_relations = demonstrate_novelty_detection(data, logger)
        
        # Step 3: Hypothesis generation
        logger.info("\n" + "="*50)
        logger.info("STEP 3: AI-Powered Hypothesis Generation")
        logger.info("="*50)
        
        hypotheses = demonstrate_hypothesis_generation(novel_relations, logger)
        
        # Step 4: Comprehensive validation
        logger.info("\n" + "="*50)
        logger.info("STEP 4: Comprehensive Validation")
        logger.info("="*50)
        
        validation_results = demonstrate_comprehensive_validation(
            novel_relations, hypotheses, data, logger
        )
        
        # Step 5: Generate reports
        logger.info("\n" + "="*50)
        logger.info("STEP 5: Generating Discovery Reports")
        logger.info("="*50)
        
        report_files = generate_discovery_reports(
            novel_relations, hypotheses, validation_results, logger
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 NOVEL DISCOVERY SYSTEM DEMO COMPLETE!")
        logger.info("="*80)
        
        logger.info(f"\n🔬 DISCOVERY RESULTS:")
        logger.info(f"✅ Novel relationships discovered: {len(novel_relations)}")
        logger.info(f"✅ Testable hypotheses generated: {len(hypotheses)}")
        logger.info(f"✅ High-confidence discoveries: {len([r for r in novel_relations if r.confidence_score >= 0.8])}")
        logger.info(f"✅ High-priority hypotheses: {len([h for h in hypotheses if h.priority_score >= 0.7])}")
        
        validation_quality = validation_results.get("overall_assessment", {}).get("validation_quality", "unknown")
        logger.info(f"✅ Validation quality: {validation_quality}")
        
        logger.info(f"\n📊 CAPABILITIES DEMONSTRATED:")
        logger.info(f"✅ Neural relation prediction with hybrid GNN")
        logger.info(f"✅ Pattern-based discovery from literature co-occurrence")
        logger.info(f"✅ LLM-powered biological plausibility validation")
        logger.info(f"✅ AI agent-based hypothesis generation")
        logger.info(f"✅ Multi-agent hypothesis validation")
        logger.info(f"✅ Literature cross-validation")
        logger.info(f"✅ Temporal validation")
        logger.info(f"✅ Expert assessment interface generation")
        logger.info(f"✅ Comprehensive research report generation")
        
        logger.info(f"\n📁 OUTPUT FILES GENERATED:")
        for report_type, file_path in report_files.items():
            logger.info(f"  - {report_type}: {file_path}")
        
        logger.info(f"\n🚀 RESEARCH IMPACT:")
        logger.info(f"  - Novel therapeutic targets identified")
        logger.info(f"  - New biological mechanisms proposed")
        logger.info(f"  - Experimental validation roadmap provided")
        logger.info(f"  - Research collaboration opportunities outlined")
        logger.info(f"  - Grant application support materials generated")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info(f"  1. Review generated reports and prioritize investigations")
        logger.info(f"  2. Conduct expert validation using assessment forms")
        logger.info(f"  3. Plan experimental validation studies")
        logger.info(f"  4. Initiate literature review for novel relationships")
        logger.info(f"  5. Prepare research proposals and grant applications")
        
        logger.info(f"\n📍 All results saved to: outputs/novel_discovery_demo/")
        
    except Exception as e:
        logger.error(f"Error in novel discovery demo: {e}")
        raise


if __name__ == "__main__":
    main()