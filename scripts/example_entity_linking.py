#!/usr/bin/env python3
"""
Example script demonstrating entity linking functionality.

This script shows how to:
1. Load literature processing results
2. Load knowledge graph preprocessing results
3. Perform entity linking with fuzzy matching and disambiguation
4. Analyze linking performance and quality
5. Save integrated results
"""

import sys
from pathlib import Path
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.phase1.literature_processor import LiteratureProcessor
from litkg.phase1.kg_preprocessor import KGPreprocessor
from litkg.phase1.entity_linker import EntityLinker
from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging


def main():
    """Main example function."""
    # Setup logging
    logger = setup_logging(level="INFO", log_file="entity_linking.log")
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("Configuration file not found. Please check config/config.yaml")
        return
    
    logger.info("=== ENTITY LINKING DEMONSTRATION ===")
    
    # Step 1: Prepare sample data (literature and KG)
    logger.info("Step 1: Preparing sample data...")
    
    # Initialize processors
    literature_processor = LiteratureProcessor()
    kg_preprocessor = KGPreprocessor()
    entity_linker = EntityLinker()
    
    # Check if we have existing processed data
    literature_file = "data/processed/cancer_genomics_literature.json"
    kg_file = "data/processed/integrated_knowledge_graph.json"
    
    # Process literature if not available
    if not Path(literature_file).exists():
        logger.info("Processing sample literature...")
        try:
            # Process a small sample for demonstration
            documents = literature_processor.process_query(
                query="BRCA1 breast cancer mutation",
                max_results=10,  # Small sample for demo
                date_range=("2020/01/01", "2024/01/01"),
                output_file=literature_file
            )
            logger.info(f"Processed {len(documents)} literature documents")
        except Exception as e:
            logger.error(f"Error processing literature: {e}")
            # Create dummy data for demonstration
            logger.info("Creating dummy literature data for demonstration...")
            documents = create_dummy_literature_data()
            literature_processor.save_results(documents, literature_file)
    else:
        logger.info("Loading existing literature data...")
        documents = literature_processor.load_results(literature_file)
    
    # Process KG if not available
    if not Path(kg_file).exists():
        logger.info("Processing sample knowledge graph...")
        try:
            kg_preprocessor.download_all_data()
            kg_preprocessor.process_all_data()
            kg_preprocessor.save_integrated_graph(kg_file)
            logger.info("Knowledge graph processed and saved")
        except Exception as e:
            logger.error(f"Error processing KG: {e}")
            return
    else:
        logger.info("Loading existing knowledge graph...")
        kg_preprocessor.load_integrated_graph(kg_file)
    
    # Step 2: Load KG entities into entity linker
    logger.info("Step 2: Loading KG entities for linking...")
    entity_linker.load_kg_entities(kg_preprocessor)
    
    # Step 3: Perform entity linking
    logger.info("Step 3: Performing entity linking...")
    
    try:
        # Link entities for all documents
        linking_results = entity_linker.batch_link_documents(
            documents,
            use_semantic=True,
            use_context=True
        )
        
        logger.info(f"Completed entity linking for {len(linking_results)} documents")
        
    except Exception as e:
        logger.error(f"Error during entity linking: {e}")
        return
    
    # Step 4: Analyze linking results
    logger.info("Step 4: Analyzing linking results...")
    
    try:
        analyze_linking_results(linking_results, logger)
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
    
    # Step 5: Save linking results
    logger.info("Step 5: Saving linking results...")
    
    try:
        output_file = "data/processed/entity_linking_results.json"
        entity_linker.save_linking_results(linking_results, output_file)
        logger.info(f"Linking results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Step 6: Demonstrate different linking strategies
    logger.info("Step 6: Comparing different linking strategies...")
    
    try:
        compare_linking_strategies(entity_linker, documents[:3], logger)  # Use first 3 docs
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
    
    # Step 7: Show integration examples
    logger.info("Step 7: Demonstrating integrated literature-KG analysis...")
    
    try:
        demonstrate_integration(linking_results[:3], kg_preprocessor, logger)
    except Exception as e:
        logger.error(f"Error in integration demonstration: {e}")
    
    logger.info("=== ENTITY LINKING DEMONSTRATION COMPLETED ===")


def create_dummy_literature_data():
    """Create dummy literature data for demonstration."""
    from litkg.phase1.literature_processor import ProcessedDocument, Entity, Relation
    from datetime import datetime
    
    # Create sample entities
    entities1 = [
        Entity(text="BRCA1", label="GENE", start=0, end=5, confidence=0.9),
        Entity(text="breast cancer", label="DISEASE", start=20, end=33, confidence=0.85),
        Entity(text="mutation", label="MUTATION", start=40, end=48, confidence=0.8)
    ]
    
    entities2 = [
        Entity(text="TP53", label="GENE", start=0, end=4, confidence=0.95),
        Entity(text="lung cancer", label="DISEASE", start=15, end=26, confidence=0.9),
        Entity(text="chemotherapy", label="DRUG", start=35, end=47, confidence=0.7)
    ]
    
    # Create sample relations
    relations1 = [
        Relation(
            subject=entities1[0], predicate="ASSOCIATED_WITH", object=entities1[1],
            confidence=0.8, context="BRCA1 mutations in breast cancer",
            sentence="BRCA1 mutations are associated with breast cancer risk."
        )
    ]
    
    relations2 = [
        Relation(
            subject=entities2[1], predicate="TREATED_WITH", object=entities2[2],
            confidence=0.7, context="lung cancer chemotherapy treatment",
            sentence="Lung cancer patients receive chemotherapy treatment."
        )
    ]
    
    # Create sample documents
    documents = [
        ProcessedDocument(
            pmid="12345678",
            title="BRCA1 mutations in breast cancer",
            abstract="BRCA1 mutations are associated with increased breast cancer risk and affect treatment response.",
            authors=["Smith J", "Johnson A"],
            journal="Cancer Research",
            publication_date=datetime(2023, 1, 15),
            entities=entities1,
            relations=relations1
        ),
        ProcessedDocument(
            pmid="87654321",
            title="TP53 in lung cancer therapy",
            abstract="TP53 mutations in lung cancer affect chemotherapy response and patient outcomes.",
            authors=["Brown K", "Davis L"],
            journal="Nature Medicine",
            publication_date=datetime(2023, 3, 20),
            entities=entities2,
            relations=relations2
        )
    ]
    
    return documents


def analyze_linking_results(linking_results, logger):
    """Analyze and report on linking results."""
    logger.info("\n=== ENTITY LINKING ANALYSIS ===")
    
    # Overall statistics
    total_docs = len(linking_results)
    total_entities = sum(r.linking_statistics["total_entities"] for r in linking_results)
    total_matched = sum(r.linking_statistics["matched_entities"] for r in linking_results)
    total_unmatched = sum(r.linking_statistics["unmatched_entities"] for r in linking_results)
    
    logger.info(f"Documents processed: {total_docs}")
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"Matched entities: {total_matched} ({total_matched/total_entities*100:.1f}%)")
    logger.info(f"Unmatched entities: {total_unmatched} ({total_unmatched/total_entities*100:.1f}%)")
    
    # Match type distribution
    match_types = {}
    confidence_scores = []
    similarity_scores = []
    
    for result in linking_results:
        for match in result.matches:
            match_types[match.match_type] = match_types.get(match.match_type, 0) + 1
            confidence_scores.append(match.confidence_score)
            similarity_scores.append(match.similarity_score)
    
    logger.info("\nMatch type distribution:")
    for match_type, count in sorted(match_types.items()):
        logger.info(f"  {match_type}: {count}")
    
    # Score statistics
    if confidence_scores:
        import numpy as np
        logger.info(f"\nConfidence scores - Mean: {np.mean(confidence_scores):.3f}, Std: {np.std(confidence_scores):.3f}")
        logger.info(f"Similarity scores - Mean: {np.mean(similarity_scores):.3f}, Std: {np.std(similarity_scores):.3f}")
    
    # Entity type linking performance
    entity_type_performance = {}
    
    for result in linking_results:
        for match in result.matches:
            lit_type = match.literature_entity.label
            if lit_type not in entity_type_performance:
                entity_type_performance[lit_type] = {"matched": 0, "total": 0}
            entity_type_performance[lit_type]["matched"] += 1
            entity_type_performance[lit_type]["total"] += 1
        
        for unmatched in result.unmatched_literature_entities:
            lit_type = unmatched.label
            if lit_type not in entity_type_performance:
                entity_type_performance[lit_type] = {"matched": 0, "total": 0}
            entity_type_performance[lit_type]["total"] += 1
    
    logger.info("\nEntity type linking performance:")
    for entity_type, stats in sorted(entity_type_performance.items()):
        match_rate = stats["matched"] / stats["total"] * 100 if stats["total"] > 0 else 0
        logger.info(f"  {entity_type}: {stats['matched']}/{stats['total']} ({match_rate:.1f}%)")
    
    # Show some example matches
    logger.info("\nExample high-confidence matches:")
    example_count = 0
    for result in linking_results:
        for match in result.matches:
            if match.confidence_score > 0.8 and example_count < 5:
                logger.info(f"  '{match.literature_entity.text}' -> '{match.kg_entity.name}' "
                          f"(confidence: {match.confidence_score:.3f}, type: {match.match_type})")
                example_count += 1
    
    # Show disambiguation conflicts
    total_conflicts = sum(len(r.disambiguation_conflicts) for r in linking_results)
    if total_conflicts > 0:
        logger.info(f"\nDisambiguation conflicts: {total_conflicts}")
        logger.info("Example conflicts:")
        conflict_count = 0
        for result in linking_results:
            for conflict in result.disambiguation_conflicts:
                if conflict_count < 3:
                    entity_text = conflict["entity"]["text"]
                    num_candidates = len(conflict["candidates"])
                    logger.info(f"  '{entity_text}' has {num_candidates} candidate matches")
                    conflict_count += 1


def compare_linking_strategies(entity_linker, sample_docs, logger):
    """Compare different linking strategies."""
    logger.info("\n=== COMPARING LINKING STRATEGIES ===")
    
    strategies = [
        ("Fuzzy Only", {"use_semantic": False, "use_context": False}),
        ("Fuzzy + Semantic", {"use_semantic": True, "use_context": False}),
        ("Fuzzy + Context", {"use_semantic": False, "use_context": True}),
        ("Full Pipeline", {"use_semantic": True, "use_context": True})
    ]
    
    for strategy_name, params in strategies:
        logger.info(f"\nTesting strategy: {strategy_name}")
        
        # Reset statistics
        entity_linker.linking_stats = {
            "total_processed": 0, "exact_matches": 0, "fuzzy_matches": 0,
            "semantic_matches": 0, "contextual_matches": 0, "unmatched": 0,
            "disambiguation_conflicts": 0
        }
        
        try:
            results = entity_linker.batch_link_documents(sample_docs, **params)
            
            total_entities = sum(r.linking_statistics["total_entities"] for r in results)
            total_matched = sum(r.linking_statistics["matched_entities"] for r in results)
            
            match_rate = total_matched / total_entities * 100 if total_entities > 0 else 0
            
            logger.info(f"  Match rate: {match_rate:.1f}% ({total_matched}/{total_entities})")
            logger.info(f"  Exact matches: {entity_linker.linking_stats['exact_matches']}")
            logger.info(f"  Fuzzy matches: {entity_linker.linking_stats['fuzzy_matches']}")
            logger.info(f"  Semantic matches: {entity_linker.linking_stats['semantic_matches']}")
            logger.info(f"  Contextual matches: {entity_linker.linking_stats['contextual_matches']}")
            logger.info(f"  Conflicts: {entity_linker.linking_stats['disambiguation_conflicts']}")
            
        except Exception as e:
            logger.error(f"  Error with {strategy_name}: {e}")


def demonstrate_integration(linking_results, kg_preprocessor, logger):
    """Demonstrate integration of literature and KG data."""
    logger.info("\n=== LITERATURE-KG INTEGRATION EXAMPLES ===")
    
    # Find entities that appear in both literature and KG
    literature_entities = set()
    kg_entities = set()
    linked_pairs = []
    
    for result in linking_results:
        for match in result.matches:
            lit_entity = match.literature_entity
            kg_entity = match.kg_entity
            
            literature_entities.add(lit_entity.text.lower())
            kg_entities.add(kg_entity.name.lower())
            linked_pairs.append((lit_entity, kg_entity))
    
    logger.info(f"Unique literature entities: {len(literature_entities)}")
    logger.info(f"Unique KG entities: {len(kg_entities)}")
    logger.info(f"Linked pairs: {len(linked_pairs)}")
    
    # Show cross-modal evidence
    logger.info("\nCross-modal evidence examples:")
    
    for lit_entity, kg_entity in linked_pairs[:5]:  # Show first 5
        logger.info(f"\nEntity: {lit_entity.text} -> {kg_entity.name}")
        logger.info(f"  Literature context: {lit_entity.label}")
        logger.info(f"  KG context: {kg_entity.type} from {kg_entity.source}")
        
        if kg_entity.cui:
            logger.info(f"  UMLS CUI: {kg_entity.cui}")
        
        if kg_entity.attributes:
            logger.info(f"  KG attributes: {list(kg_entity.attributes.keys())}")
    
    # Analyze entity type mappings
    type_mappings = {}
    for lit_entity, kg_entity in linked_pairs:
        key = (lit_entity.label, kg_entity.type)
        type_mappings[key] = type_mappings.get(key, 0) + 1
    
    logger.info("\nEntity type mappings:")
    for (lit_type, kg_type), count in sorted(type_mappings.items()):
        logger.info(f"  {lit_type} -> {kg_type}: {count}")
    
    # Find potential new knowledge
    logger.info("\nPotential novel associations (literature entities not in KG):")
    
    unmatched_count = 0
    for result in linking_results:
        for unmatched in result.unmatched_literature_entities:
            if unmatched_count < 5:  # Show first 5
                logger.info(f"  {unmatched.text} ({unmatched.label}) - confidence: {unmatched.confidence:.2f}")
                unmatched_count += 1


if __name__ == "__main__":
    main()