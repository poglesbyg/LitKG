#!/usr/bin/env python3
"""
Phase 1 Integration Script

This script demonstrates the complete Phase 1 pipeline:
1. Literature processing with biomedical NLP
2. Knowledge graph preprocessing and standardization
3. Entity linking with fuzzy matching and disambiguation
4. Integration and validation of results
5. Preparation for Phase 2 (GNN training)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.phase1.literature_processor import LiteratureProcessor
from litkg.phase1.kg_preprocessor import KGPreprocessor
from litkg.phase1.entity_linker import EntityLinker
from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging


class Phase1Integrator:
    """Integrates all Phase 1 components into a unified pipeline."""
    
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        self.logger = setup_logging(level="INFO", log_file="phase1_integration.log")
        
        # Initialize components
        self.literature_processor = LiteratureProcessor(config_path)
        self.kg_preprocessor = KGPreprocessor(config_path)
        self.entity_linker = EntityLinker(config_path)
        
        # Results storage
        self.literature_results = []
        self.kg_results = None
        self.linking_results = []
        self.integration_stats = {}
    
    def run_complete_pipeline(
        self,
        literature_queries: list,
        max_results_per_query: int = 50,
        date_range=("2020/01/01", "2024/01/01")
    ):
        """Run the complete Phase 1 pipeline."""
        self.logger.info("=== STARTING PHASE 1 INTEGRATION PIPELINE ===")
        
        # Step 1: Process literature
        self.logger.info("Step 1: Processing biomedical literature...")
        if not self._process_literature(literature_queries, max_results_per_query, date_range):
            return False
        
        # Step 2: Process knowledge graphs
        self.logger.info("Step 2: Processing knowledge graphs...")
        if not self._process_knowledge_graphs():
            return False
        
        # Step 3: Perform entity linking
        self.logger.info("Step 3: Performing entity linking...")
        if not self._perform_entity_linking():
            return False
        
        # Step 4: Create integrated dataset
        self.logger.info("Step 4: Creating integrated dataset...")
        if not self._create_integrated_dataset():
            return False
        
        # Step 5: Validate and analyze results
        self.logger.info("Step 5: Validating and analyzing results...")
        self._validate_and_analyze()
        
        # Step 6: Prepare for Phase 2
        self.logger.info("Step 6: Preparing data for Phase 2...")
        self._prepare_for_phase2()
        
        self.logger.info("=== PHASE 1 INTEGRATION COMPLETED SUCCESSFULLY ===")
        return True
    
    def _process_literature(self, queries, max_results_per_query, date_range):
        """Process literature from multiple queries."""
        try:
            all_documents = []
            
            for i, query in enumerate(queries):
                self.logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
                
                output_file = f"data/processed/literature_query_{i+1}.json"
                
                documents = self.literature_processor.process_query(
                    query=query,
                    max_results=max_results_per_query,
                    date_range=date_range,
                    output_file=output_file
                )
                
                all_documents.extend(documents)
                self.logger.info(f"Processed {len(documents)} documents for query: {query}")
            
            self.literature_results = all_documents
            
            # Save combined results
            combined_file = "data/processed/combined_literature_results.json"
            self.literature_processor.save_results(all_documents, combined_file)
            
            self.logger.info(f"Total literature documents processed: {len(all_documents)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing literature: {e}")
            return False
    
    def _process_knowledge_graphs(self):
        """Process and integrate knowledge graphs."""
        try:
            # Download data
            if not self.kg_preprocessor.download_all_data():
                self.logger.warning("Some KG data downloads failed, continuing with available data")
            
            # Process and integrate
            if not self.kg_preprocessor.process_all_data():
                self.logger.error("KG processing failed")
                return False
            
            # Save results
            kg_file = "data/processed/integrated_knowledge_graph.json"
            self.kg_preprocessor.save_integrated_graph(kg_file)
            
            self.kg_results = self.kg_preprocessor.graph_builder
            
            stats = self.kg_results.get_statistics()
            self.logger.info(f"KG processing completed: {stats['num_entities']} entities, {stats['num_relations']} relations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing knowledge graphs: {e}")
            return False
    
    def _perform_entity_linking(self):
        """Perform entity linking between literature and KG."""
        try:
            # Load KG entities into linker
            self.entity_linker.load_kg_entities(self.kg_preprocessor)
            
            # Perform linking
            self.linking_results = self.entity_linker.batch_link_documents(
                self.literature_results,
                use_semantic=True,
                use_context=True
            )
            
            # Save results
            linking_file = "data/processed/entity_linking_results.json"
            self.entity_linker.save_linking_results(self.linking_results, linking_file)
            
            total_matches = sum(len(r.matches) for r in self.linking_results)
            self.logger.info(f"Entity linking completed: {total_matches} entity matches")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error performing entity linking: {e}")
            return False
    
    def _create_integrated_dataset(self):
        """Create integrated dataset combining literature, KG, and linking results."""
        try:
            integrated_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "phase": "Phase 1 - Foundation",
                    "components": ["literature_processing", "kg_preprocessing", "entity_linking"],
                    "version": "1.0"
                },
                "literature_documents": [],
                "kg_entities": {},
                "kg_relations": {},
                "entity_links": [],
                "statistics": {}
            }
            
            # Add literature documents with linked entities
            for doc in self.literature_results:
                doc_data = {
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "authors": doc.authors,
                    "journal": doc.journal,
                    "publication_date": doc.publication_date.isoformat(),
                    "entities": [self._serialize_entity(e) for e in doc.entities],
                    "relations": [self._serialize_relation(r) for r in doc.relations],
                    "mesh_terms": doc.mesh_terms
                }
                integrated_data["literature_documents"].append(doc_data)
            
            # Add KG entities and relations
            if self.kg_results:
                for entity_id, entity in self.kg_results.entities.items():
                    integrated_data["kg_entities"][entity_id] = self._serialize_kg_entity(entity)
                
                for relation_id, relation in self.kg_results.relations.items():
                    integrated_data["kg_relations"][relation_id] = self._serialize_kg_relation(relation)
            
            # Add entity links
            for result in self.linking_results:
                for match in result.matches:
                    link_data = {
                        "document_id": result.document_id,
                        "literature_entity": self._serialize_entity(match.literature_entity),
                        "kg_entity_id": match.kg_entity.id,
                        "similarity_score": match.similarity_score,
                        "confidence_score": match.confidence_score,
                        "match_type": match.match_type,
                        "evidence": match.evidence,
                        "context": match.context
                    }
                    integrated_data["entity_links"].append(link_data)
            
            # Add statistics
            integrated_data["statistics"] = self._calculate_integration_statistics()
            
            # Save integrated dataset
            integrated_file = "data/processed/phase1_integrated_dataset.json"
            with open(integrated_file, 'w') as f:
                json.dump(integrated_data, f, indent=2, default=str)
            
            self.logger.info(f"Integrated dataset saved to {integrated_file}")
            self.integration_stats = integrated_data["statistics"]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating integrated dataset: {e}")
            return False
    
    def _serialize_entity(self, entity):
        """Serialize literature entity to dict."""
        return {
            "text": entity.text,
            "label": entity.label,
            "start": entity.start,
            "end": entity.end,
            "confidence": entity.confidence,
            "cui": entity.cui,
            "synonyms": entity.synonyms
        }
    
    def _serialize_relation(self, relation):
        """Serialize literature relation to dict."""
        return {
            "subject": self._serialize_entity(relation.subject),
            "predicate": relation.predicate,
            "object": self._serialize_entity(relation.object),
            "confidence": relation.confidence,
            "context": relation.context,
            "sentence": relation.sentence
        }
    
    def _serialize_kg_entity(self, entity):
        """Serialize KG entity to dict."""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "source": entity.source,
            "original_id": entity.original_id,
            "synonyms": entity.synonyms,
            "cui": entity.cui,
            "go_id": entity.go_id,
            "attributes": entity.attributes
        }
    
    def _serialize_kg_relation(self, relation):
        """Serialize KG relation to dict."""
        return {
            "id": relation.id,
            "subject": relation.subject,
            "predicate": relation.predicate,
            "object": relation.object,
            "source": relation.source,
            "confidence": relation.confidence,
            "evidence": relation.evidence,
            "attributes": relation.attributes
        }
    
    def _calculate_integration_statistics(self):
        """Calculate comprehensive integration statistics."""
        stats = {
            "literature": {
                "total_documents": len(self.literature_results),
                "total_entities": sum(len(doc.entities) for doc in self.literature_results),
                "total_relations": sum(len(doc.relations) for doc in self.literature_results),
                "entity_types": {},
                "relation_types": {}
            },
            "knowledge_graph": {},
            "entity_linking": {
                "total_links": sum(len(r.matches) for r in self.linking_results),
                "match_types": {},
                "confidence_distribution": {},
                "linking_rate": 0.0
            },
            "integration": {
                "cross_modal_entities": 0,
                "novel_literature_entities": 0,
                "coverage_metrics": {}
            }
        }
        
        # Literature statistics
        for doc in self.literature_results:
            for entity in doc.entities:
                stats["literature"]["entity_types"][entity.label] = \
                    stats["literature"]["entity_types"].get(entity.label, 0) + 1
            
            for relation in doc.relations:
                stats["literature"]["relation_types"][relation.predicate] = \
                    stats["literature"]["relation_types"].get(relation.predicate, 0) + 1
        
        # KG statistics
        if self.kg_results:
            stats["knowledge_graph"] = self.kg_results.get_statistics()
        
        # Entity linking statistics
        total_lit_entities = stats["literature"]["total_entities"]
        total_links = stats["entity_linking"]["total_links"]
        
        if total_lit_entities > 0:
            stats["entity_linking"]["linking_rate"] = total_links / total_lit_entities
        
        for result in self.linking_results:
            for match in result.matches:
                match_type = match.match_type
                stats["entity_linking"]["match_types"][match_type] = \
                    stats["entity_linking"]["match_types"].get(match_type, 0) + 1
                
                # Confidence distribution
                conf_bin = f"{int(match.confidence_score * 10) / 10:.1f}"
                stats["entity_linking"]["confidence_distribution"][conf_bin] = \
                    stats["entity_linking"]["confidence_distribution"].get(conf_bin, 0) + 1
        
        # Integration metrics
        linked_entities = set()
        for result in self.linking_results:
            for match in result.matches:
                linked_entities.add(match.literature_entity.text.lower())
        
        stats["integration"]["cross_modal_entities"] = len(linked_entities)
        stats["integration"]["novel_literature_entities"] = \
            total_lit_entities - stats["entity_linking"]["total_links"]
        
        return stats
    
    def _validate_and_analyze(self):
        """Validate and analyze integration results."""
        self.logger.info("\n=== PHASE 1 INTEGRATION ANALYSIS ===")
        
        stats = self.integration_stats
        
        # Literature analysis
        lit_stats = stats["literature"]
        self.logger.info(f"Literature corpus: {lit_stats['total_documents']} documents")
        self.logger.info(f"Extracted entities: {lit_stats['total_entities']}")
        self.logger.info(f"Extracted relations: {lit_stats['total_relations']}")
        
        self.logger.info("\nLiterature entity types:")
        for entity_type, count in sorted(lit_stats["entity_types"].items()):
            self.logger.info(f"  {entity_type}: {count}")
        
        # KG analysis
        if "knowledge_graph" in stats and stats["knowledge_graph"]:
            kg_stats = stats["knowledge_graph"]
            self.logger.info(f"\nKnowledge graph: {kg_stats['num_entities']} entities, {kg_stats['num_relations']} relations")
            
            self.logger.info("KG entity types:")
            for entity_type, count in sorted(kg_stats["entity_types"].items()):
                self.logger.info(f"  {entity_type}: {count}")
        
        # Linking analysis
        link_stats = stats["entity_linking"]
        self.logger.info(f"\nEntity linking: {link_stats['total_links']} successful links")
        self.logger.info(f"Linking rate: {link_stats['linking_rate']:.1%}")
        
        self.logger.info("Match types:")
        for match_type, count in sorted(link_stats["match_types"].items()):
            self.logger.info(f"  {match_type}: {count}")
        
        # Integration analysis
        int_stats = stats["integration"]
        self.logger.info(f"\nIntegration results:")
        self.logger.info(f"Cross-modal entities: {int_stats['cross_modal_entities']}")
        self.logger.info(f"Novel literature entities: {int_stats['novel_literature_entities']}")
        
        # Quality assessment
        self._assess_quality()
    
    def _assess_quality(self):
        """Assess the quality of Phase 1 results."""
        self.logger.info("\n=== QUALITY ASSESSMENT ===")
        
        # Entity linking quality
        high_confidence_links = 0
        total_links = 0
        
        for result in self.linking_results:
            for match in result.matches:
                total_links += 1
                if match.confidence_score > 0.8:
                    high_confidence_links += 1
        
        if total_links > 0:
            quality_rate = high_confidence_links / total_links
            self.logger.info(f"High-confidence linking rate: {quality_rate:.1%}")
        
        # Coverage assessment
        linked_entity_types = set()
        for result in self.linking_results:
            for match in result.matches:
                linked_entity_types.add(match.literature_entity.label)
        
        total_entity_types = set()
        for doc in self.literature_results:
            for entity in doc.entities:
                total_entity_types.add(entity.label)
        
        if total_entity_types:
            type_coverage = len(linked_entity_types) / len(total_entity_types)
            self.logger.info(f"Entity type coverage: {type_coverage:.1%}")
        
        # Recommendations
        self.logger.info("\nRecommendations for Phase 2:")
        
        if quality_rate < 0.7:
            self.logger.info("- Consider improving entity disambiguation")
        
        if type_coverage < 0.8:
            self.logger.info("- Expand knowledge graph coverage for missing entity types")
        
        if stats["entity_linking"]["linking_rate"] < 0.5:
            self.logger.info("- Improve fuzzy matching thresholds")
        
        self.logger.info("- Ready for hybrid GNN training")
        self.logger.info("- Consider confidence weighting in GNN")
    
    def _prepare_for_phase2(self):
        """Prepare data structures for Phase 2 GNN training."""
        try:
            # Create graph structure for GNN
            phase2_data = {
                "nodes": [],
                "edges": [],
                "node_features": {},
                "edge_features": {},
                "metadata": {
                    "prepared_for": "Phase 2 - Hybrid GNN",
                    "node_types": ["literature_entity", "kg_entity"],
                    "edge_types": ["literature_relation", "kg_relation", "entity_link"]
                }
            }
            
            # Add literature entities as nodes
            lit_node_id = 0
            lit_entity_map = {}
            
            for doc in self.literature_results:
                for entity in doc.entities:
                    node_id = f"lit_{lit_node_id}"
                    lit_entity_map[f"{doc.pmid}_{entity.start}_{entity.end}"] = node_id
                    
                    phase2_data["nodes"].append({
                        "id": node_id,
                        "type": "literature_entity",
                        "text": entity.text,
                        "label": entity.label,
                        "confidence": entity.confidence,
                        "document_id": doc.pmid
                    })
                    
                    phase2_data["node_features"][node_id] = {
                        "entity_type": entity.label,
                        "confidence": entity.confidence,
                        "text_length": len(entity.text),
                        "has_cui": 1 if entity.cui else 0
                    }
                    
                    lit_node_id += 1
            
            # Add KG entities as nodes
            if self.kg_results:
                for entity_id, entity in self.kg_results.entities.items():
                    phase2_data["nodes"].append({
                        "id": entity_id,
                        "type": "kg_entity",
                        "name": entity.name,
                        "entity_type": entity.type,
                        "source": entity.source
                    })
                    
                    phase2_data["node_features"][entity_id] = {
                        "entity_type": entity.type,
                        "source": entity.source,
                        "has_cui": 1 if entity.cui else 0,
                        "has_go_id": 1 if entity.go_id else 0,
                        "synonym_count": len(entity.synonyms)
                    }
            
            # Add edges
            edge_id = 0
            
            # Literature relations
            for doc in self.literature_results:
                for relation in doc.relations:
                    subj_key = f"{doc.pmid}_{relation.subject.start}_{relation.subject.end}"
                    obj_key = f"{doc.pmid}_{relation.object.start}_{relation.object.end}"
                    
                    if subj_key in lit_entity_map and obj_key in lit_entity_map:
                        phase2_data["edges"].append({
                            "id": f"lit_rel_{edge_id}",
                            "source": lit_entity_map[subj_key],
                            "target": lit_entity_map[obj_key],
                            "type": "literature_relation",
                            "predicate": relation.predicate,
                            "confidence": relation.confidence
                        })
                        
                        phase2_data["edge_features"][f"lit_rel_{edge_id}"] = {
                            "relation_type": relation.predicate,
                            "confidence": relation.confidence,
                            "source": "literature"
                        }
                        
                        edge_id += 1
            
            # KG relations
            if self.kg_results:
                for relation_id, relation in self.kg_results.relations.items():
                    phase2_data["edges"].append({
                        "id": relation_id,
                        "source": relation.subject,
                        "target": relation.object,
                        "type": "kg_relation",
                        "predicate": relation.predicate,
                        "confidence": relation.confidence
                    })
                    
                    phase2_data["edge_features"][relation_id] = {
                        "relation_type": relation.predicate,
                        "confidence": relation.confidence,
                        "source": relation.source
                    }
            
            # Entity links
            for result in self.linking_results:
                for match in result.matches:
                    lit_key = f"{result.document_id}_{match.literature_entity.start}_{match.literature_entity.end}"
                    
                    if lit_key in lit_entity_map:
                        phase2_data["edges"].append({
                            "id": f"link_{edge_id}",
                            "source": lit_entity_map[lit_key],
                            "target": match.kg_entity.id,
                            "type": "entity_link",
                            "match_type": match.match_type,
                            "confidence": match.confidence_score
                        })
                        
                        phase2_data["edge_features"][f"link_{edge_id}"] = {
                            "relation_type": "entity_link",
                            "confidence": match.confidence_score,
                            "match_type": match.match_type,
                            "similarity": match.similarity_score
                        }
                        
                        edge_id += 1
            
            # Save Phase 2 preparation data
            phase2_file = "data/processed/phase2_graph_data.json"
            with open(phase2_file, 'w') as f:
                json.dump(phase2_data, f, indent=2, default=str)
            
            self.logger.info(f"Phase 2 data prepared: {len(phase2_data['nodes'])} nodes, {len(phase2_data['edges'])} edges")
            self.logger.info(f"Saved to {phase2_file}")
            
        except Exception as e:
            self.logger.error(f"Error preparing Phase 2 data: {e}")


def main():
    """Main function to run Phase 1 integration."""
    # Setup
    logger = setup_logging(level="INFO")
    
    # Check configuration
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("Configuration file not found. Please check config/config.yaml")
        return
    
    # Initialize integrator
    integrator = Phase1Integrator()
    
    # Define literature queries for biomedical research
    literature_queries = [
        "BRCA1 breast cancer mutation treatment",
        "TP53 lung cancer therapy resistance",
        "EGFR inhibitor melanoma immunotherapy",
        "checkpoint inhibitor PD1 PDL1 cancer",
        "CAR-T cell therapy leukemia lymphoma"
    ]
    
    # Run complete pipeline
    success = integrator.run_complete_pipeline(
        literature_queries=literature_queries,
        max_results_per_query=20,  # Reduced for demo
        date_range=("2020/01/01", "2024/01/01")
    )
    
    if success:
        logger.info("Phase 1 integration completed successfully!")
        logger.info("Ready to proceed to Phase 2: Hybrid GNN Architecture")
    else:
        logger.error("Phase 1 integration failed. Check logs for details.")


if __name__ == "__main__":
    main()