#!/usr/bin/env python3
"""
Example script demonstrating literature processing pipeline.

This script shows how to:
1. Set up the literature processor
2. Search and process biomedical literature
3. Extract entities and relations
4. Save results for further analysis
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.phase1.literature_processor import LiteratureProcessor
from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging


def main():
    """Main example function."""
    # Setup logging
    logger = setup_logging(level="INFO", log_file="literature_processing.log")
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("Configuration file not found. Please check config/config.yaml")
        return
    
    # Ensure email is set for PubMed API
    if not config.phase1.literature.pubmed["email"] or config.phase1.literature.pubmed["email"] == "your-email@domain.com":
        logger.error("Please set your email in config/config.yaml for PubMed API access")
        return
    
    # Initialize literature processor
    logger.info("Initializing literature processor...")
    processor = LiteratureProcessor()
    
    # Example 1: Process cancer genomics literature
    logger.info("Example 1: Processing cancer genomics literature")
    cancer_query = "BRCA1 breast cancer mutation treatment"
    
    try:
        cancer_docs = processor.process_query(
            query=cancer_query,
            max_results=50,  # Start small for testing
            date_range=("2020/01/01", "2024/01/01"),  # Recent papers
            output_file="data/processed/cancer_genomics_literature.json"
        )
        
        logger.info(f"Processed {len(cancer_docs)} cancer genomics documents")
        
        # Show some statistics
        total_entities = sum(len(doc.entities) for doc in cancer_docs)
        total_relations = sum(len(doc.relations) for doc in cancer_docs)
        
        logger.info(f"Total entities extracted: {total_entities}")
        logger.info(f"Total relations extracted: {total_relations}")
        
        # Show entity type distribution
        entity_types = {}
        for doc in cancer_docs:
            for entity in doc.entities:
                entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
        
        logger.info("Entity type distribution:")
        for entity_type, count in sorted(entity_types.items()):
            logger.info(f"  {entity_type}: {count}")
        
    except Exception as e:
        logger.error(f"Error processing cancer genomics literature: {e}")
    
    # Example 2: Process drug-disease literature
    logger.info("Example 2: Processing drug-disease literature")
    drug_query = "immunotherapy checkpoint inhibitor melanoma"
    
    try:
        drug_docs = processor.process_query(
            query=drug_query,
            max_results=30,
            date_range=("2021/01/01", "2024/01/01"),
            output_file="data/processed/immunotherapy_literature.json"
        )
        
        logger.info(f"Processed {len(drug_docs)} immunotherapy documents")
        
        # Show relation type distribution
        relation_types = {}
        for doc in drug_docs:
            for relation in doc.relations:
                relation_types[relation.predicate] = relation_types.get(relation.predicate, 0) + 1
        
        logger.info("Relation type distribution:")
        for relation_type, count in sorted(relation_types.items()):
            logger.info(f"  {relation_type}: {count}")
        
    except Exception as e:
        logger.error(f"Error processing drug-disease literature: {e}")
    
    # Example 3: Demonstrate loading saved results
    logger.info("Example 3: Loading and analyzing saved results")
    
    try:
        # Load previously saved results
        loaded_docs = processor.load_results("data/processed/cancer_genomics_literature.json")
        logger.info(f"Loaded {len(loaded_docs)} documents from file")
        
        # Analyze high-confidence entities
        high_conf_entities = []
        for doc in loaded_docs:
            for entity in doc.entities:
                if entity.confidence > 0.8:
                    high_conf_entities.append(entity)
        
        logger.info(f"Found {len(high_conf_entities)} high-confidence entities (>0.8)")
        
        # Show most common high-confidence entities
        entity_counts = {}
        for entity in high_conf_entities:
            key = (entity.text.lower(), entity.label)
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info("Top 10 most frequent high-confidence entities:")
        for (text, label), count in top_entities:
            logger.info(f"  {text} ({label}): {count}")
        
    except FileNotFoundError:
        logger.warning("No saved results found to load")
    except Exception as e:
        logger.error(f"Error loading results: {e}")
    
    logger.info("Literature processing example completed!")


if __name__ == "__main__":
    main()