#!/usr/bin/env python3
"""
Example script demonstrating knowledge graph preprocessing.

This script shows how to:
1. Download data from CIVIC, TCGA, CPTAC
2. Process and standardize entities
3. Build integrated knowledge graph
4. Perform ontology mapping
5. Save results for further analysis
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.phase1.kg_preprocessor import KGPreprocessor
from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging


def main():
    """Main example function."""
    # Setup logging
    logger = setup_logging(level="INFO", log_file="kg_preprocessing.log")
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("Configuration file not found. Please check config/config.yaml")
        return
    
    # Initialize KG preprocessor
    logger.info("Initializing KG preprocessor...")
    preprocessor = KGPreprocessor()
    
    # Step 1: Download data from all sources
    logger.info("Step 1: Downloading data from all sources...")
    
    try:
        success = preprocessor.download_all_data()
        if success:
            logger.info("Successfully downloaded data from all sources")
        else:
            logger.warning("Some data downloads failed, continuing with available data")
    except Exception as e:
        logger.error(f"Error during data download: {e}")
        return
    
    # Step 2: Process and integrate data
    logger.info("Step 2: Processing and integrating data...")
    
    try:
        success = preprocessor.process_all_data()
        if success:
            logger.info("Successfully processed and integrated all data")
        else:
            logger.error("Data processing failed")
            return
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        return
    
    # Step 3: Save integrated knowledge graph
    logger.info("Step 3: Saving integrated knowledge graph...")
    
    try:
        output_path = "data/processed/integrated_knowledge_graph.json"
        preprocessor.save_integrated_graph(output_path)
        logger.info(f"Integrated knowledge graph saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving knowledge graph: {e}")
        return
    
    # Step 4: Demonstrate graph analysis
    logger.info("Step 4: Analyzing integrated knowledge graph...")
    
    try:
        # Get graph statistics
        stats = preprocessor.graph_builder.get_statistics()
        
        logger.info("=== KNOWLEDGE GRAPH ANALYSIS ===")
        logger.info(f"Total entities: {stats['num_entities']}")
        logger.info(f"Total relations: {stats['num_relations']}")
        logger.info(f"Graph nodes: {stats['num_nodes']}")
        logger.info(f"Graph edges: {stats['num_edges']}")
        
        logger.info("\nEntity type distribution:")
        for entity_type, count in sorted(stats['entity_types'].items()):
            logger.info(f"  {entity_type}: {count}")
        
        logger.info("\nRelation type distribution:")
        for relation_type, count in sorted(stats['relation_types'].items()):
            logger.info(f"  {relation_type}: {count}")
        
        logger.info("\nData source distribution:")
        for source, count in sorted(stats['sources'].items()):
            logger.info(f"  {source}: {count}")
        
        # Analyze specific entity types
        logger.info("\n=== DETAILED ANALYSIS ===")
        
        # Find genes with most connections
        gene_connections = {}
        for relation in preprocessor.graph_builder.relations.values():
            for entity_id in [relation.subject, relation.object]:
                if entity_id in preprocessor.graph_builder.entities:
                    entity = preprocessor.graph_builder.entities[entity_id]
                    if entity.type == "GENE":
                        gene_connections[entity.name] = gene_connections.get(entity.name, 0) + 1
        
        if gene_connections:
            top_genes = sorted(gene_connections.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("\nTop 10 most connected genes:")
            for gene, connections in top_genes:
                logger.info(f"  {gene}: {connections} connections")
        
        # Find diseases with most associations
        disease_connections = {}
        for relation in preprocessor.graph_builder.relations.values():
            for entity_id in [relation.subject, relation.object]:
                if entity_id in preprocessor.graph_builder.entities:
                    entity = preprocessor.graph_builder.entities[entity_id]
                    if entity.type == "DISEASE":
                        disease_connections[entity.name] = disease_connections.get(entity.name, 0) + 1
        
        if disease_connections:
            top_diseases = sorted(disease_connections.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("\nTop 5 most connected diseases:")
            for disease, connections in top_diseases:
                logger.info(f"  {disease}: {connections} connections")
        
        # Analyze cross-source connections
        cross_source_relations = 0
        for relation in preprocessor.graph_builder.relations.values():
            subject_entity = preprocessor.graph_builder.entities.get(relation.subject)
            object_entity = preprocessor.graph_builder.entities.get(relation.object)
            
            if subject_entity and object_entity:
                if subject_entity.source != object_entity.source:
                    cross_source_relations += 1
        
        logger.info(f"\nCross-source relations: {cross_source_relations}")
        logger.info(f"Percentage of cross-source relations: {cross_source_relations/stats['num_relations']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Error during graph analysis: {e}")
    
    # Step 5: Demonstrate loading saved graph
    logger.info("Step 5: Demonstrating graph loading...")
    
    try:
        # Create new preprocessor instance
        new_preprocessor = KGPreprocessor()
        
        # Load the saved graph
        new_preprocessor.load_integrated_graph("data/processed/integrated_knowledge_graph.json")
        
        # Verify loading worked
        new_stats = new_preprocessor.graph_builder.get_statistics()
        logger.info(f"Loaded graph with {new_stats['num_entities']} entities and {new_stats['num_relations']} relations")
        
        if new_stats['num_entities'] == stats['num_entities']:
            logger.info("Graph loading verification: SUCCESS")
        else:
            logger.warning("Graph loading verification: MISMATCH")
        
    except Exception as e:
        logger.error(f"Error during graph loading demonstration: {e}")
    
    # Step 6: Show ontology mapping examples
    logger.info("Step 6: Demonstrating ontology mappings...")
    
    try:
        ontology_mapper = preprocessor.ontology_mapper
        
        # Test UMLS mappings
        test_entities = [
            ("BRCA1", "GENE"),
            ("breast cancer", "DISEASE"),
            ("TP53", "GENE"),
            ("lung cancer", "DISEASE")
        ]
        
        logger.info("\nUMLS mapping examples:")
        for entity_name, entity_type in test_entities:
            cui = ontology_mapper.map_to_umls(entity_name, entity_type)
            logger.info(f"  {entity_name} ({entity_type}) -> CUI: {cui}")
        
        # Test Gene Ontology mappings
        test_genes = ["BRCA1", "TP53", "EGFR", "KRAS"]
        
        logger.info("\nGene Ontology mapping examples:")
        for gene in test_genes:
            go_id = ontology_mapper.map_to_gene_ontology(gene)
            logger.info(f"  {gene} -> GO ID: {go_id}")
        
    except Exception as e:
        logger.error(f"Error during ontology mapping demonstration: {e}")
    
    logger.info("\n=== KG PREPROCESSING COMPLETED SUCCESSFULLY ===")
    logger.info("Next steps:")
    logger.info("1. Use the integrated KG for entity linking")
    logger.info("2. Combine with literature processing results")
    logger.info("3. Build hybrid GNN for Phase 2")


if __name__ == "__main__":
    main()