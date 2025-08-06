#!/usr/bin/env python3
"""
Phase 3 Confidence Scoring Demo for LitKG-Integrate

This script demonstrates the confidence scoring system that assesses the reliability
of biomedical relationships derived from different evidence types (literature vs experimental).

Key features demonstrated:
1. Literature confidence assessment (journal quality, citations, methodology)
2. Experimental confidence assessment (sample size, statistical significance, replication)
3. Cross-modal evidence integration and conflict resolution
4. Uncertainty quantification (epistemic vs aleatoric)
5. Comprehensive confidence metrics and explanations

The confidence scoring system is crucial for Phase 3 novel knowledge discovery,
as it helps distinguish reliable predictions from uncertain ones.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging
from litkg.phase3.confidence_scoring import (
    ConfidenceScorer,
    ConfidenceMetrics,
    EvidenceType
)


def create_synthetic_evidence_data() -> List[Dict[str, Any]]:
    """
    Create synthetic biomedical relationship data for confidence scoring demo.
    
    This represents different types of evidence scenarios:
    1. Strong literature + strong experimental evidence
    2. Strong literature + weak experimental evidence  
    3. Weak literature + strong experimental evidence
    4. Conflicting literature and experimental evidence
    5. Literature-only evidence
    6. Experimental-only evidence
    """
    
    synthetic_relationships = [
        {
            'relationship_id': 'BRCA1_breast_cancer_strong',
            'entity1': 'BRCA1',
            'entity2': 'breast_cancer',
            'relation_type': 'associated_with',
            'literature_data': {
                'papers': [
                    {'pmid': '12345678', 'title': 'BRCA1 mutations in breast cancer', 'journal': 'Nature', 'impact_factor': 42.8},
                    {'pmid': '12345679', 'title': 'BRCA1 role in DNA repair', 'journal': 'Cell', 'impact_factor': 38.6},
                    {'pmid': '12345680', 'title': 'Clinical significance of BRCA1', 'journal': 'NEJM', 'impact_factor': 74.7}
                ],
                'avg_impact_factor': 52.0,
                'total_citations': 2500,
                'recency_score': 0.9,
                'methodology_score': 0.95,
                'consensus_score': 0.98,
                'journal_quality': 0.95,
                'author_reputation': 0.9,
                'study_design_quality': 0.9,
                'statistical_rigor': 0.95
            },
            'experimental_data': {
                'experiments': [
                    {'source': 'TCGA', 'sample_size': 1200, 'p_value': 1e-15},
                    {'source': 'CIVIC', 'sample_size': 800, 'p_value': 1e-12},
                    {'source': 'CPTAC', 'sample_size': 400, 'p_value': 1e-8}
                ],
                'primary_source': 'tcga',
                'sample_size': 2400,
                'effect_size': 0.85,
                'p_value': 1e-15,
                'num_replications': 3,
                'consistency_score': 0.95,
                'methodology_quality': 0.9,
                'data_quality': 0.95,
                'validation_score': 0.9
            }
        },
        
        {
            'relationship_id': 'TP53_lung_cancer_lit_strong',
            'entity1': 'TP53',
            'entity2': 'lung_cancer',
            'relation_type': 'associated_with',
            'literature_data': {
                'papers': [
                    {'pmid': '23456789', 'title': 'TP53 mutations in lung cancer', 'journal': 'Nature', 'impact_factor': 42.8},
                    {'pmid': '23456790', 'title': 'p53 pathway in lung carcinogenesis', 'journal': 'Science', 'impact_factor': 41.8}
                ],
                'avg_impact_factor': 42.3,
                'total_citations': 1800,
                'recency_score': 0.85,
                'methodology_score': 0.9,
                'consensus_score': 0.92,
                'journal_quality': 0.95,
                'author_reputation': 0.88,
                'study_design_quality': 0.85,
                'statistical_rigor': 0.9
            },
            'experimental_data': {
                'experiments': [
                    {'source': 'TCGA', 'sample_size': 200, 'p_value': 0.02}
                ],
                'primary_source': 'tcga',
                'sample_size': 200,
                'effect_size': 0.35,
                'p_value': 0.02,
                'num_replications': 1,
                'consistency_score': 0.6,
                'methodology_quality': 0.7,
                'data_quality': 0.8,
                'validation_score': 0.5
            }
        },
        
        {
            'relationship_id': 'EGFR_drug_response_exp_strong',
            'entity1': 'EGFR',
            'entity2': 'erlotinib_response',
            'relation_type': 'predicts',
            'literature_data': {
                'papers': [
                    {'pmid': '34567890', 'title': 'EGFR and erlotinib', 'journal': 'JCO', 'impact_factor': 28.2}
                ],
                'avg_impact_factor': 28.2,
                'total_citations': 150,
                'recency_score': 0.7,
                'methodology_score': 0.6,
                'consensus_score': 0.65,
                'journal_quality': 0.8,
                'author_reputation': 0.7,
                'study_design_quality': 0.6,
                'statistical_rigor': 0.7
            },
            'experimental_data': {
                'experiments': [
                    {'source': 'CIVIC', 'sample_size': 1500, 'p_value': 1e-20},
                    {'source': 'CPTAC', 'sample_size': 600, 'p_value': 1e-12},
                    {'source': 'TCGA', 'sample_size': 800, 'p_value': 1e-16}
                ],
                'primary_source': 'civic',
                'sample_size': 2900,
                'effect_size': 0.92,
                'p_value': 1e-20,
                'num_replications': 3,
                'consistency_score': 0.98,
                'methodology_quality': 0.95,
                'data_quality': 0.9,
                'validation_score': 0.95
            }
        },
        
        {
            'relationship_id': 'KRAS_conflicting_evidence',
            'entity1': 'KRAS',
            'entity2': 'immunotherapy_response',
            'relation_type': 'associated_with',
            'literature_data': {
                'papers': [
                    {'pmid': '45678901', 'title': 'KRAS enhances immunotherapy', 'journal': 'Cancer Cell', 'impact_factor': 26.6},
                    {'pmid': '45678902', 'title': 'KRAS negative for immunotherapy', 'journal': 'Nat Med', 'impact_factor': 36.1}
                ],
                'avg_impact_factor': 31.4,
                'total_citations': 400,
                'recency_score': 0.95,
                'methodology_score': 0.8,
                'consensus_score': 0.3,  # Low consensus due to conflicting results
                'journal_quality': 0.9,
                'author_reputation': 0.85,
                'study_design_quality': 0.8,
                'statistical_rigor': 0.75
            },
            'experimental_data': {
                'experiments': [
                    {'source': 'TCGA', 'sample_size': 300, 'p_value': 0.8},  # Non-significant
                    {'source': 'CIVIC', 'sample_size': 150, 'p_value': 0.3}
                ],
                'primary_source': 'tcga',
                'sample_size': 450,
                'effect_size': 0.1,  # Very small effect
                'p_value': 0.8,
                'num_replications': 2,
                'consistency_score': 0.2,  # Low consistency
                'methodology_quality': 0.7,
                'data_quality': 0.75,
                'validation_score': 0.3
            }
        },
        
        {
            'relationship_id': 'PIK3CA_literature_only',
            'entity1': 'PIK3CA',
            'entity2': 'PI3K_inhibitor_response',
            'relation_type': 'predicts',
            'literature_data': {
                'papers': [
                    {'pmid': '56789012', 'title': 'PIK3CA mutations and PI3K inhibitors', 'journal': 'Nat Rev Cancer', 'impact_factor': 51.8},
                    {'pmid': '56789013', 'title': 'PIK3CA therapeutic implications', 'journal': 'Cell', 'impact_factor': 38.6},
                    {'pmid': '56789014', 'title': 'PI3K pathway targeting', 'journal': 'Cancer Discov', 'impact_factor': 24.2}
                ],
                'avg_impact_factor': 38.2,
                'total_citations': 950,
                'recency_score': 0.8,
                'methodology_score': 0.85,
                'consensus_score': 0.88,
                'journal_quality': 0.92,
                'author_reputation': 0.9,
                'study_design_quality': 0.8,
                'statistical_rigor': 0.85
            },
            'experimental_data': None  # Literature-only evidence
        },
        
        {
            'relationship_id': 'novel_gene_experimental_only',
            'entity1': 'NOVEL_GENE_X',
            'entity2': 'drug_resistance',
            'relation_type': 'causes',
            'literature_data': None,  # Experimental-only evidence
            'experimental_data': {
                'experiments': [
                    {'source': 'CPTAC', 'sample_size': 500, 'p_value': 1e-8}
                ],
                'primary_source': 'cptac',
                'sample_size': 500,
                'effect_size': 0.7,
                'p_value': 1e-8,
                'num_replications': 1,
                'consistency_score': 0.8,
                'methodology_quality': 0.85,
                'data_quality': 0.9,
                'validation_score': 0.6  # Needs literature validation
            }
        }
    ]
    
    return synthetic_relationships


def demonstrate_confidence_assessment(logger):
    """Demonstrate comprehensive confidence assessment."""
    logger.info("=== CONFIDENCE ASSESSMENT DEMO ===")
    
    # Initialize confidence scorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_scorer = ConfidenceScorer(device=device)
    
    # Create synthetic evidence data
    relationships = create_synthetic_evidence_data()
    logger.info(f"Created {len(relationships)} synthetic relationships for confidence assessment")
    
    # Assess confidence for each relationship
    confidence_results = []
    
    for i, relationship in enumerate(relationships):
        logger.info(f"\nAssessing relationship {i+1}: {relationship['relationship_id']}")
        
        confidence = confidence_scorer.assess_relationship_confidence(
            literature_data=relationship.get('literature_data'),
            experimental_data=relationship.get('experimental_data')
        )
        
        confidence_results.append({
            'relationship': relationship,
            'confidence': confidence
        })
        
        # Log key confidence metrics
        logger.info(f"  Overall confidence: {confidence.overall_confidence:.3f} ({confidence.confidence_level})")
        logger.info(f"  Literature confidence: {confidence.literature_confidence:.3f}")
        logger.info(f"  Experimental confidence: {confidence.experimental_confidence:.3f}")
        logger.info(f"  Cross-modal agreement: {confidence.cross_modal_agreement:.3f}")
        logger.info(f"  Epistemic uncertainty: {confidence.epistemic_uncertainty:.3f}")
        logger.info(f"  Supporting evidence: {confidence.supporting_papers} papers, {confidence.supporting_experiments} experiments")
        logger.info(f"  Explanation: {confidence.explanation}")
    
    return confidence_results


def demonstrate_batch_assessment(confidence_scorer, relationships, logger):
    """Demonstrate batch confidence assessment."""
    logger.info("\n=== BATCH CONFIDENCE ASSESSMENT DEMO ===")
    
    # Batch assess all relationships
    batch_results = confidence_scorer.batch_assess_confidence(relationships)
    
    logger.info(f"Batch assessed {len(batch_results)} relationships")
    
    # Summary statistics
    overall_scores = [result.overall_confidence for result in batch_results]
    lit_scores = [result.literature_confidence for result in batch_results]
    exp_scores = [result.experimental_confidence for result in batch_results]
    
    logger.info(f"Overall confidence - Mean: {np.mean(overall_scores):.3f}, Std: {np.std(overall_scores):.3f}")
    logger.info(f"Literature confidence - Mean: {np.mean(lit_scores):.3f}, Std: {np.std(lit_scores):.3f}")
    logger.info(f"Experimental confidence - Mean: {np.mean(exp_scores):.3f}, Std: {np.std(exp_scores):.3f}")
    
    return batch_results


def visualize_confidence_analysis(confidence_results, logger):
    """Create visualizations of confidence analysis results."""
    logger.info("\n=== CONFIDENCE VISUALIZATION ===")
    
    # Prepare data for visualization
    relationships = []
    overall_conf = []
    lit_conf = []
    exp_conf = []
    agreement = []
    epistemic_unc = []
    aleatoric_unc = []
    
    for result in confidence_results:
        rel = result['relationship']
        conf = result['confidence']
        
        relationships.append(rel['relationship_id'].replace('_', '\n'))
        overall_conf.append(conf.overall_confidence)
        lit_conf.append(conf.literature_confidence)
        exp_conf.append(conf.experimental_confidence)
        agreement.append(conf.cross_modal_agreement)
        epistemic_unc.append(conf.epistemic_uncertainty)
        aleatoric_unc.append(conf.aleatoric_uncertainty)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LitKG Phase 3: Confidence Scoring Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall confidence comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(relationships)), overall_conf, 
                   color=['#2E8B57' if c >= 0.8 else '#FF6347' if c < 0.5 else '#FFD700' for c in overall_conf])
    ax1.set_title('Overall Confidence Scores')
    ax1.set_ylabel('Confidence Score')
    ax1.set_xticks(range(len(relationships)))
    ax1.set_xticklabels(relationships, rotation=45, ha='right', fontsize=8)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High confidence')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium confidence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Literature vs Experimental confidence
    ax2 = axes[0, 1]
    x = np.arange(len(relationships))
    width = 0.35
    ax2.bar(x - width/2, lit_conf, width, label='Literature', color='skyblue')
    ax2.bar(x + width/2, exp_conf, width, label='Experimental', color='lightcoral')
    ax2.set_title('Literature vs Experimental Confidence')
    ax2.set_ylabel('Confidence Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(relationships, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-modal agreement
    ax3 = axes[0, 2]
    colors = ['#FF6B6B' if a < 0.5 else '#4ECDC4' if a >= 0.8 else '#45B7D1' for a in agreement]
    ax3.bar(range(len(relationships)), agreement, color=colors)
    ax3.set_title('Cross-Modal Evidence Agreement')
    ax3.set_ylabel('Agreement Score')
    ax3.set_xticks(range(len(relationships)))
    ax3.set_xticklabels(relationships, rotation=45, ha='right', fontsize=8)
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High agreement')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Low agreement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty quantification
    ax4 = axes[1, 0]
    x = np.arange(len(relationships))
    width = 0.35
    ax4.bar(x - width/2, epistemic_unc, width, label='Epistemic (Model)', color='mediumpurple')
    ax4.bar(x + width/2, aleatoric_unc, width, label='Aleatoric (Data)', color='mediumseagreen')
    ax4.set_title('Uncertainty Quantification')
    ax4.set_ylabel('Uncertainty Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(relationships, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Confidence vs Agreement scatter
    ax5 = axes[1, 1]
    scatter = ax5.scatter(overall_conf, agreement, 
                         c=[conf.epistemic_uncertainty for conf in [r['confidence'] for r in confidence_results]], 
                         cmap='viridis', s=100, alpha=0.7)
    ax5.set_xlabel('Overall Confidence')
    ax5.set_ylabel('Cross-Modal Agreement')
    ax5.set_title('Confidence vs Agreement\n(colored by epistemic uncertainty)')
    plt.colorbar(scatter, ax=ax5, label='Epistemic Uncertainty')
    ax5.grid(True, alpha=0.3)
    
    # Add relationship labels to scatter plot
    for i, (x, y) in enumerate(zip(overall_conf, agreement)):
        ax5.annotate(f'R{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 6. Evidence type distribution
    ax6 = axes[1, 2]
    evidence_types = []
    for result in confidence_results:
        rel = result['relationship']
        has_lit = rel.get('literature_data') is not None
        has_exp = rel.get('experimental_data') is not None
        
        if has_lit and has_exp:
            evidence_types.append('Both')
        elif has_lit:
            evidence_types.append('Literature Only')
        elif has_exp:
            evidence_types.append('Experimental Only')
        else:
            evidence_types.append('None')
    
    type_counts = {t: evidence_types.count(t) for t in set(evidence_types)}
    colors = {'Both': '#2E8B57', 'Literature Only': '#4169E1', 'Experimental Only': '#DC143C', 'None': '#696969'}
    
    wedges, texts, autotexts = ax6.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
                                      colors=[colors.get(k, '#696969') for k in type_counts.keys()])
    ax6.set_title('Evidence Type Distribution')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("outputs/phase3_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')
    logger.info(f"Confidence analysis visualization saved to {output_dir / 'confidence_analysis.png'}")
    
    return fig


def export_confidence_results(confidence_results, logger):
    """Export confidence results to JSON for further analysis."""
    logger.info("\n=== EXPORTING CONFIDENCE RESULTS ===")
    
    # Prepare data for export
    export_data = []
    
    for result in confidence_results:
        relationship = result['relationship']
        confidence = result['confidence']
        
        export_item = {
            'relationship_id': relationship['relationship_id'],
            'entity1': relationship['entity1'],
            'entity2': relationship['entity2'],
            'relation_type': relationship['relation_type'],
            'confidence_metrics': {
                'overall_confidence': confidence.overall_confidence,
                'confidence_level': confidence.confidence_level,
                'literature_confidence': confidence.literature_confidence,
                'experimental_confidence': confidence.experimental_confidence,
                'cross_modal_agreement': confidence.cross_modal_agreement,
                'evidence_strength': confidence.evidence_strength,
                'source_reliability': confidence.source_reliability,
                'temporal_consistency': confidence.temporal_consistency,
                'epistemic_uncertainty': confidence.epistemic_uncertainty,
                'aleatoric_uncertainty': confidence.aleatoric_uncertainty,
                'supporting_papers': confidence.supporting_papers,
                'supporting_experiments': confidence.supporting_experiments,
                'contradicting_evidence': confidence.contradicting_evidence,
                'explanation': confidence.explanation,
                'evidence_sources': confidence.evidence_sources
            }
        }
        
        export_data.append(export_item)
    
    # Save to JSON
    output_dir = Path("outputs/phase3_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "confidence_assessment_results.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Confidence results exported to {output_file}")
    
    # Generate summary report
    summary_file = output_dir / "confidence_summary_report.txt"
    with open(summary_file, 'w') as f:
        f.write("LitKG Phase 3: Confidence Scoring Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total relationships assessed: {len(export_data)}\n\n")
        
        # Confidence level distribution
        confidence_levels = [item['confidence_metrics']['confidence_level'] for item in export_data]
        level_counts = {level: confidence_levels.count(level) for level in set(confidence_levels)}
        
        f.write("Confidence Level Distribution:\n")
        for level, count in level_counts.items():
            f.write(f"  {level.capitalize()}: {count} ({count/len(export_data)*100:.1f}%)\n")
        f.write("\n")
        
        # High confidence relationships
        high_conf_items = [item for item in export_data if item['confidence_metrics']['confidence_level'] == 'high']
        f.write(f"High Confidence Relationships ({len(high_conf_items)}):\n")
        for item in high_conf_items:
            f.write(f"  - {item['entity1']} {item['relation_type']} {item['entity2']}\n")
            f.write(f"    Confidence: {item['confidence_metrics']['overall_confidence']:.3f}\n")
            f.write(f"    Explanation: {item['confidence_metrics']['explanation']}\n\n")
        
        # Low confidence/conflicting evidence
        low_conf_items = [item for item in export_data if item['confidence_metrics']['confidence_level'] == 'low']
        f.write(f"Low Confidence Relationships ({len(low_conf_items)}):\n")
        for item in low_conf_items:
            f.write(f"  - {item['entity1']} {item['relation_type']} {item['entity2']}\n")
            f.write(f"    Confidence: {item['confidence_metrics']['overall_confidence']:.3f}\n")
            f.write(f"    Issues: {item['confidence_metrics']['explanation']}\n\n")
    
    logger.info(f"Summary report saved to {summary_file}")
    
    return export_data


def main():
    """Main function to run the Phase 3 confidence scoring demonstration."""
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🧠 Phase 3 Confidence Scoring Demo for LitKG-Integrate")
    logger.info("=" * 70)
    
    try:
        # 1. Demonstrate individual confidence assessment
        confidence_results = demonstrate_confidence_assessment(logger)
        
        # 2. Demonstrate batch assessment
        confidence_scorer = ConfidenceScorer()
        relationships = create_synthetic_evidence_data()
        batch_results = demonstrate_batch_assessment(confidence_scorer, relationships, logger)
        
        # 3. Visualize confidence analysis
        visualize_confidence_analysis(confidence_results, logger)
        
        # 4. Export results
        export_confidence_results(confidence_results, logger)
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Phase 3 Confidence Scoring Demo Complete!")
        logger.info("\nKey Components Demonstrated:")
        logger.info("✅ Literature confidence assessment (journal quality, citations, methodology)")
        logger.info("✅ Experimental confidence assessment (sample size, statistical significance)")
        logger.info("✅ Cross-modal evidence integration and conflict resolution")
        logger.info("✅ Uncertainty quantification (epistemic vs aleatoric)")
        logger.info("✅ Comprehensive confidence metrics and explanations")
        logger.info("✅ Batch processing and visualization capabilities")
        
        logger.info("\nConfidence Scoring Statistics:")
        overall_scores = [result['confidence'].overall_confidence for result in confidence_results]
        logger.info(f"  Mean overall confidence: {np.mean(overall_scores):.3f}")
        logger.info(f"  Confidence score range: {min(overall_scores):.3f} - {max(overall_scores):.3f}")
        high_conf_count = sum(1 for score in overall_scores if score >= 0.8)
        logger.info(f"  High confidence relationships: {high_conf_count}/{len(overall_scores)}")
        
        logger.info("\nOutput Files Generated:")
        logger.info("  - outputs/phase3_visualizations/confidence_analysis.png")
        logger.info("  - outputs/phase3_results/confidence_assessment_results.json")
        logger.info("  - outputs/phase3_results/confidence_summary_report.txt")
        
        logger.info("\nNext Steps for Full Implementation:")
        logger.info("1. Train confidence models on real biomedical data")
        logger.info("2. Integrate with Phase 2 hybrid GNN predictions")
        logger.info("3. Implement active learning for confidence calibration")
        logger.info("4. Deploy for novel knowledge discovery validation")
        
    except Exception as e:
        logger.error(f"Error in confidence scoring demo: {e}")
        raise


if __name__ == "__main__":
    main()