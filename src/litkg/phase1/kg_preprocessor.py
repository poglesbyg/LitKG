"""
Knowledge Graph Preprocessing Module

This module handles:
1. Data ingestion from CIVIC, TCGA, CPTAC
2. Entity standardization and harmonization
3. Ontology mapping (UMLS, Gene Ontology)
4. Graph construction and validation
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import requests
 
import networkx as nx
from tqdm import tqdm
import pickle

from litkg.utils.config import LitKGConfig, load_config, get_data_dir
from litkg.utils.logging import LoggerMixin


@dataclass
class StandardizedEntity:
    """Standardized entity across different knowledge graphs."""
    id: str  # Unique identifier
    name: str  # Primary name
    type: str  # GENE, DISEASE, DRUG, MUTATION, etc.
    source: str  # CIVIC, TCGA, CPTAC, etc.
    original_id: str  # Original ID from source
    synonyms: List[str]
    cui: Optional[str] = None  # UMLS CUI
    go_id: Optional[str] = None  # Gene Ontology ID
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class StandardizedRelation:
    """Standardized relation between entities."""
    id: str
    subject: str  # Entity ID
    predicate: str  # Relation type
    object: str  # Entity ID
    source: str
    confidence: float
    evidence: List[str]  # Supporting evidence
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class OntologyMapper(LoggerMixin):
    """Maps entities to standard ontologies (UMLS, Gene Ontology)."""
    
    def __init__(self, config: Optional[LitKGConfig] = None):
        self.config = load_config() if config is None else (config if isinstance(config, LitKGConfig) else load_config(config))
        self.ontology_config = self.config.phase1.knowledge_graphs.ontologies
        
        # Initialize mappings
        self.umls_mapping = {}
        self.go_mapping = {}
        
        # Load cached mappings if available
        self._load_cached_mappings()
    
    def _load_cached_mappings(self):
        """Load cached ontology mappings."""
        cache_dir = get_data_dir() / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        umls_cache = cache_dir / "umls_mapping.pkl"
        go_cache = cache_dir / "go_mapping.pkl"
        
        if umls_cache.exists():
            with open(umls_cache, 'rb') as f:
                self.umls_mapping = pickle.load(f)
            self.logger.info(f"Loaded {len(self.umls_mapping)} UMLS mappings from cache")
        
        if go_cache.exists():
            with open(go_cache, 'rb') as f:
                self.go_mapping = pickle.load(f)
            self.logger.info(f"Loaded {len(self.go_mapping)} GO mappings from cache")
    
    def _save_cached_mappings(self):
        """Save ontology mappings to cache."""
        cache_dir = get_data_dir() / "cache"
        
        with open(cache_dir / "umls_mapping.pkl", 'wb') as f:
            pickle.dump(self.umls_mapping, f)
        
        with open(cache_dir / "go_mapping.pkl", 'wb') as f:
            pickle.dump(self.go_mapping, f)
        
        self.logger.info("Saved ontology mappings to cache")
    
    def map_to_umls(self, entity_name: str, entity_type: str) -> Optional[str]:
        """
        Map entity to UMLS CUI.
        
        Args:
            entity_name: Entity name to map
            entity_type: Type of entity (GENE, DISEASE, etc.)
            
        Returns:
            UMLS CUI if found, None otherwise
        """
        cache_key = f"{entity_name}:{entity_type}"
        
        if cache_key in self.umls_mapping:
            return self.umls_mapping[cache_key]
        
        # Try to find UMLS mapping
        cui = self._query_umls_api(entity_name, entity_type)
        
        # Cache the result (even if None)
        self.umls_mapping[cache_key] = cui
        
        return cui
    
    def _query_umls_api(self, entity_name: str, entity_type: str) -> Optional[str]:
        """Query UMLS API for entity mapping."""
        api_key = self.ontology_config["umls"].get("api_key")
        
        if not api_key:
            # Use simple heuristics for common entities
            return self._heuristic_umls_mapping(entity_name, entity_type)
        
        # TODO: Implement actual UMLS API query
        # For now, use heuristics
        return self._heuristic_umls_mapping(entity_name, entity_type)
    
    def _heuristic_umls_mapping(self, entity_name: str, entity_type: str) -> Optional[str]:
        """Heuristic UMLS mapping for common entities."""
        # Simple mapping for demonstration
        common_mappings = {
            ("BRCA1", "GENE"): "C0376571",
            ("BRCA2", "GENE"): "C0376571",
            ("TP53", "GENE"): "C0080055",
            ("breast cancer", "DISEASE"): "C0006142",
            ("lung cancer", "DISEASE"): "C0242379",
            ("melanoma", "DISEASE"): "C0025202",
        }
        
        return common_mappings.get((entity_name.upper(), entity_type))
    
    def map_to_gene_ontology(self, gene_name: str) -> Optional[str]:
        """Map gene to Gene Ontology ID."""
        cache_key = f"GO:{gene_name}"
        
        if cache_key in self.go_mapping:
            return self.go_mapping[cache_key]
        
        # Try to find GO mapping
        go_id = self._query_go_api(gene_name)
        
        # Cache the result
        self.go_mapping[cache_key] = go_id
        
        return go_id
    
    def _query_go_api(self, gene_name: str) -> Optional[str]:
        """Query Gene Ontology API for gene mapping."""
        # TODO: Implement actual GO API query
        # For now, use heuristics
        return self._heuristic_go_mapping(gene_name)
    
    def _heuristic_go_mapping(self, gene_name: str) -> Optional[str]:
        """Heuristic GO mapping for common genes."""
        common_go_mappings = {
            "BRCA1": "GO:0006281",  # DNA repair
            "BRCA2": "GO:0006281",  # DNA repair
            "TP53": "GO:0006915",   # apoptotic process
            "EGFR": "GO:0007173",   # epidermal growth factor receptor signaling
        }
        
        return common_go_mappings.get(gene_name.upper())


class CivicProcessor(LoggerMixin):
    """Processes CIVIC (Clinical Interpretations of Variants in Cancer) data."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.civic_config = config.phase1.knowledge_graphs.civic
        self.ontology_mapper = OntologyMapper(config)
    
    def download_civic_data(self) -> bool:
        """Download latest CIVIC data."""
        self.logger.info("Downloading CIVIC data...")
        
        # CIVIC data URLs
        urls = {
            "variants": "https://civicdb.org/downloads/01-Feb-2024/01-Feb-2024-VariantSummaries.tsv",
            "evidence": "https://civicdb.org/downloads/01-Feb-2024/01-Feb-2024-ClinicalEvidenceSummaries.tsv",
            "genes": "https://civicdb.org/downloads/01-Feb-2024/01-Feb-2024-GeneSummaries.tsv"
        }
        
        data_dir = get_data_dir() / "external" / "civic"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for data_type, url in urls.items():
            try:
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                
                filename = f"civic_{data_type}.tsv"
                filepath = data_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                self.logger.info(f"Downloaded {data_type} data to {filepath}")
                
            except Exception as e:
                self.logger.error(f"Failed to download {data_type} data: {e}")
                return False
        
        return True
    
    def process_civic_data(self) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """Process CIVIC data into standardized format."""
        self.logger.info("Processing CIVIC data...")
        
        data_dir = get_data_dir() / "external" / "civic"
        
        entities = []
        relations = []
        
        # Process genes
        genes_file = data_dir / "civic_genes.tsv"
        if genes_file.exists():
            gene_entities = self._process_civic_genes(genes_file)
            entities.extend(gene_entities)
        
        # Process variants
        variants_file = data_dir / "civic_variants.tsv"
        if variants_file.exists():
            variant_entities, variant_relations = self._process_civic_variants(variants_file)
            entities.extend(variant_entities)
            relations.extend(variant_relations)
        
        # Process evidence
        evidence_file = data_dir / "civic_evidence.tsv"
        if evidence_file.exists():
            evidence_relations = self._process_civic_evidence(evidence_file)
            relations.extend(evidence_relations)
        
        self.logger.info(f"Processed {len(entities)} entities and {len(relations)} relations from CIVIC")
        
        return entities, relations
    
    def _process_civic_genes(self, genes_file: Path) -> List[StandardizedEntity]:
        """Process CIVIC genes data."""
        entities = []
        
        try:
            df = pd.read_csv(genes_file, sep='\t')
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CIVIC genes"):
                gene_name = str(row.get('gene', ''))
                gene_id = str(row.get('gene_id', ''))
                
                if not gene_name or gene_name == 'nan':
                    continue
                
                # Map to ontologies
                cui = self.ontology_mapper.map_to_umls(gene_name, "GENE")
                go_id = self.ontology_mapper.map_to_gene_ontology(gene_name)
                
                entity = StandardizedEntity(
                    id=f"CIVIC:GENE:{gene_id}",
                    name=gene_name,
                    type="GENE",
                    source="CIVIC",
                    original_id=gene_id,
                    synonyms=[],
                    cui=cui,
                    go_id=go_id,
                    attributes={
                        "description": str(row.get('description', '')),
                        "entrez_id": str(row.get('entrez_id', ''))
                    }
                )
                
                entities.append(entity)
        
        except Exception as e:
            self.logger.error(f"Error processing CIVIC genes: {e}")
        
        return entities
    
    def _process_civic_variants(self, variants_file: Path) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """Process CIVIC variants data."""
        entities = []
        relations = []
        
        try:
            df = pd.read_csv(variants_file, sep='\t')
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CIVIC variants"):
                variant_id = str(row.get('variant_id', ''))
                variant_name = str(row.get('variant', ''))
                gene_name = str(row.get('gene', ''))
                
                if not variant_name or variant_name == 'nan':
                    continue
                
                # Create variant entity
                entity = StandardizedEntity(
                    id=f"CIVIC:VARIANT:{variant_id}",
                    name=variant_name,
                    type="MUTATION",
                    source="CIVIC",
                    original_id=variant_id,
                    synonyms=[],
                    attributes={
                        "gene": gene_name,
                        "variant_type": str(row.get('variant_type', '')),
                        "chromosome": str(row.get('chromosome', '')),
                        "start": str(row.get('start', '')),
                        "stop": str(row.get('stop', ''))
                    }
                )
                
                entities.append(entity)
                
                # Create gene-variant relation
                if gene_name and gene_name != 'nan':
                    relation = StandardizedRelation(
                        id=f"CIVIC:REL:GENE_VARIANT:{variant_id}",
                        subject=f"CIVIC:GENE:{gene_name}",
                        predicate="HAS_VARIANT",
                        object=f"CIVIC:VARIANT:{variant_id}",
                        source="CIVIC",
                        confidence=1.0,
                        evidence=[f"CIVIC variant {variant_id}"]
                    )
                    
                    relations.append(relation)
        
        except Exception as e:
            self.logger.error(f"Error processing CIVIC variants: {e}")
        
        return entities, relations
    
    def _process_civic_evidence(self, evidence_file: Path) -> List[StandardizedRelation]:
        """Process CIVIC evidence data."""
        relations = []
        
        try:
            df = pd.read_csv(evidence_file, sep='\t')
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CIVIC evidence"):
                evidence_id = str(row.get('evidence_id', ''))
                variant_id = str(row.get('variant_id', ''))
                disease = str(row.get('disease', ''))
                drug = str(row.get('drugs', ''))
                
                if not evidence_id or evidence_id == 'nan':
                    continue
                
                # Create variant-disease relation
                if disease and disease != 'nan':
                    relation = StandardizedRelation(
                        id=f"CIVIC:REL:VARIANT_DISEASE:{evidence_id}",
                        subject=f"CIVIC:VARIANT:{variant_id}",
                        predicate="ASSOCIATED_WITH",
                        object=f"CIVIC:DISEASE:{disease}",
                        source="CIVIC",
                        confidence=0.8,
                        evidence=[f"CIVIC evidence {evidence_id}"],
                        attributes={
                            "evidence_type": str(row.get('evidence_type', '')),
                            "significance": str(row.get('clinical_significance', ''))
                        }
                    )
                    
                    relations.append(relation)
                
                # Create variant-drug relation
                if drug and drug != 'nan':
                    relation = StandardizedRelation(
                        id=f"CIVIC:REL:VARIANT_DRUG:{evidence_id}",
                        subject=f"CIVIC:VARIANT:{variant_id}",
                        predicate="RESPONDS_TO",
                        object=f"CIVIC:DRUG:{drug}",
                        source="CIVIC",
                        confidence=0.8,
                        evidence=[f"CIVIC evidence {evidence_id}"],
                        attributes={
                            "evidence_type": str(row.get('evidence_type', '')),
                            "significance": str(row.get('clinical_significance', ''))
                        }
                    )
                    
                    relations.append(relation)
        
        except Exception as e:
            self.logger.error(f"Error processing CIVIC evidence: {e}")
        
        return relations


class TCGAProcessor(LoggerMixin):
    """Processes TCGA (The Cancer Genome Atlas) data."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.tcga_config = config.phase1.knowledge_graphs.tcga
        self.ontology_mapper = OntologyMapper(config)
    
    def download_tcga_data(self) -> bool:
        """Download TCGA data (simplified for demo)."""
        self.logger.info("Setting up TCGA data download...")
        
        # In a real implementation, this would use the GDC API
        # For now, we'll create a placeholder structure
        
        data_dir = get_data_dir() / "external" / "tcga"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data files
        sample_clinical = pd.DataFrame({
            'case_id': ['TCGA-A1-A0SB', 'TCGA-A1-A0SD', 'TCGA-A1-A0SE'],
            'primary_site': ['Breast', 'Lung', 'Brain'],
            'disease_type': ['Ductal and Lobular Neoplasms', 'Squamous Cell Neoplasms', 'Gliomas'],
            'age_at_diagnosis': [50, 65, 45],
            'gender': ['female', 'male', 'female']
        })
        
        sample_clinical.to_csv(data_dir / "clinical_data.csv", index=False)
        
        # Create sample mutation data
        sample_mutations = pd.DataFrame({
            'case_id': ['TCGA-A1-A0SB', 'TCGA-A1-A0SD', 'TCGA-A1-A0SE'],
            'gene_symbol': ['BRCA1', 'TP53', 'IDH1'],
            'variant_type': ['SNP', 'DEL', 'SNP'],
            'consequence': ['missense_variant', 'frameshift_variant', 'missense_variant']
        })
        
        sample_mutations.to_csv(data_dir / "mutation_data.csv", index=False)
        
        self.logger.info("TCGA sample data created")
        return True
    
    def process_tcga_data(self) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """Process TCGA data into standardized format."""
        self.logger.info("Processing TCGA data...")
        
        data_dir = get_data_dir() / "external" / "tcga"
        
        entities = []
        relations = []
        
        # Process clinical data
        clinical_file = data_dir / "clinical_data.csv"
        if clinical_file.exists():
            clinical_entities = self._process_tcga_clinical(clinical_file)
            entities.extend(clinical_entities)
        
        # Process mutation data
        mutation_file = data_dir / "mutation_data.csv"
        if mutation_file.exists():
            mutation_entities, mutation_relations = self._process_tcga_mutations(mutation_file)
            entities.extend(mutation_entities)
            relations.extend(mutation_relations)
        
        self.logger.info(f"Processed {len(entities)} entities and {len(relations)} relations from TCGA")
        
        return entities, relations
    
    def _process_tcga_clinical(self, clinical_file: Path) -> List[StandardizedEntity]:
        """Process TCGA clinical data."""
        entities = []
        
        try:
            df = pd.read_csv(clinical_file)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing TCGA clinical"):
                case_id = str(row['case_id'])
                primary_site = str(row['primary_site'])
                disease_type = str(row['disease_type'])
                
                # Create patient entity
                patient_entity = StandardizedEntity(
                    id=f"TCGA:PATIENT:{case_id}",
                    name=f"Patient {case_id}",
                    type="PATIENT",
                    source="TCGA",
                    original_id=case_id,
                    synonyms=[],
                    attributes={
                        "primary_site": primary_site,
                        "disease_type": disease_type,
                        "age_at_diagnosis": str(row.get('age_at_diagnosis', '')),
                        "gender": str(row.get('gender', ''))
                    }
                )
                
                entities.append(patient_entity)
                
                # Create disease entity
                cui = self.ontology_mapper.map_to_umls(disease_type, "DISEASE")
                
                disease_entity = StandardizedEntity(
                    id=f"TCGA:DISEASE:{disease_type.replace(' ', '_')}",
                    name=disease_type,
                    type="DISEASE",
                    source="TCGA",
                    original_id=disease_type,
                    synonyms=[],
                    cui=cui,
                    attributes={
                        "primary_site": primary_site
                    }
                )
                
                entities.append(disease_entity)
        
        except Exception as e:
            self.logger.error(f"Error processing TCGA clinical data: {e}")
        
        return entities
    
    def _process_tcga_mutations(self, mutation_file: Path) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """Process TCGA mutation data."""
        entities = []
        relations = []
        
        try:
            df = pd.read_csv(mutation_file)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing TCGA mutations"):
                case_id = str(row['case_id'])
                gene_symbol = str(row['gene_symbol'])
                variant_type = str(row['variant_type'])
                consequence = str(row['consequence'])
                
                # Create mutation entity
                mutation_id = f"{case_id}_{gene_symbol}_{variant_type}"
                
                mutation_entity = StandardizedEntity(
                    id=f"TCGA:MUTATION:{mutation_id}",
                    name=f"{gene_symbol} {variant_type}",
                    type="MUTATION",
                    source="TCGA",
                    original_id=mutation_id,
                    synonyms=[],
                    attributes={
                        "gene_symbol": gene_symbol,
                        "variant_type": variant_type,
                        "consequence": consequence
                    }
                )
                
                entities.append(mutation_entity)
                
                # Create patient-mutation relation
                relation = StandardizedRelation(
                    id=f"TCGA:REL:PATIENT_MUTATION:{mutation_id}",
                    subject=f"TCGA:PATIENT:{case_id}",
                    predicate="HAS_MUTATION",
                    object=f"TCGA:MUTATION:{mutation_id}",
                    source="TCGA",
                    confidence=0.9,
                    evidence=[f"TCGA sequencing data for {case_id}"]
                )
                
                relations.append(relation)
        
        except Exception as e:
            self.logger.error(f"Error processing TCGA mutations: {e}")
        
        return entities, relations


class CPTACProcessor(LoggerMixin):
    """Processes CPTAC (Clinical Proteomic Tumor Analysis Consortium) data."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.cptac_config = config.phase1.knowledge_graphs.cptac
        self.ontology_mapper = OntologyMapper(config)
    
    def download_cptac_data(self) -> bool:
        """Download CPTAC data (simplified for demo)."""
        self.logger.info("Setting up CPTAC data download...")
        
        data_dir = get_data_dir() / "external" / "cptac"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample proteomics data
        sample_proteomics = pd.DataFrame({
            'case_id': ['CPTAC-A1-001', 'CPTAC-A1-002', 'CPTAC-A1-003'],
            'protein_id': ['P04637', 'P38398', 'P53350'],
            'gene_symbol': ['TP53', 'BRCA1', 'PLK1'],
            'expression_level': [2.5, 1.8, 3.2],
            'cancer_type': ['breast', 'ovarian', 'lung']
        })
        
        sample_proteomics.to_csv(data_dir / "proteomics_data.csv", index=False)
        
        self.logger.info("CPTAC sample data created")
        return True
    
    def process_cptac_data(self) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """Process CPTAC data into standardized format."""
        self.logger.info("Processing CPTAC data...")
        
        data_dir = get_data_dir() / "external" / "cptac"
        
        entities = []
        relations = []
        
        # Process proteomics data
        proteomics_file = data_dir / "proteomics_data.csv"
        if proteomics_file.exists():
            prot_entities, prot_relations = self._process_cptac_proteomics(proteomics_file)
            entities.extend(prot_entities)
            relations.extend(prot_relations)
        
        self.logger.info(f"Processed {len(entities)} entities and {len(relations)} relations from CPTAC")
        
        return entities, relations
    
    def _process_cptac_proteomics(self, proteomics_file: Path) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """Process CPTAC proteomics data."""
        entities = []
        relations = []
        
        try:
            df = pd.read_csv(proteomics_file)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CPTAC proteomics"):
                case_id = str(row['case_id'])
                protein_id = str(row['protein_id'])
                gene_symbol = str(row['gene_symbol'])
                expression_level = float(row['expression_level'])
                cancer_type = str(row['cancer_type'])
                
                # Create protein entity
                cui = self.ontology_mapper.map_to_umls(gene_symbol, "PROTEIN")
                go_id = self.ontology_mapper.map_to_gene_ontology(gene_symbol)
                
                protein_entity = StandardizedEntity(
                    id=f"CPTAC:PROTEIN:{protein_id}",
                    name=f"{gene_symbol} protein",
                    type="PROTEIN",
                    source="CPTAC",
                    original_id=protein_id,
                    synonyms=[gene_symbol],
                    cui=cui,
                    go_id=go_id,
                    attributes={
                        "gene_symbol": gene_symbol,
                        "uniprot_id": protein_id
                    }
                )
                
                entities.append(protein_entity)
                
                # Create patient entity
                patient_entity = StandardizedEntity(
                    id=f"CPTAC:PATIENT:{case_id}",
                    name=f"Patient {case_id}",
                    type="PATIENT",
                    source="CPTAC",
                    original_id=case_id,
                    synonyms=[],
                    attributes={
                        "cancer_type": cancer_type
                    }
                )
                
                entities.append(patient_entity)
                
                # Create expression relation
                relation = StandardizedRelation(
                    id=f"CPTAC:REL:PROTEIN_EXPRESSION:{case_id}_{protein_id}",
                    subject=f"CPTAC:PATIENT:{case_id}",
                    predicate="EXPRESSES",
                    object=f"CPTAC:PROTEIN:{protein_id}",
                    source="CPTAC",
                    confidence=0.9,
                    evidence=[f"CPTAC proteomics data"],
                    attributes={
                        "expression_level": expression_level,
                        "cancer_type": cancer_type
                    }
                )
                
                relations.append(relation)
        
        except Exception as e:
            self.logger.error(f"Error processing CPTAC proteomics: {e}")
        
        return entities, relations


class KnowledgeGraphBuilder(LoggerMixin):
    """Builds integrated knowledge graph from standardized entities and relations."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = {}
    
    def add_entities(self, entities: List[StandardizedEntity]):
        """Add entities to the knowledge graph."""
        for entity in entities:
            self.entities[entity.id] = entity
            
            # Add node to graph
            self.graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                source=entity.source,
                cui=entity.cui,
                go_id=entity.go_id,
                **entity.attributes
            )
    
    def add_relations(self, relations: List[StandardizedRelation]):
        """Add relations to the knowledge graph."""
        for relation in relations:
            self.relations[relation.id] = relation
            
            # Add edge to graph
            self.graph.add_edge(
                relation.subject,
                relation.object,
                key=relation.id,
                predicate=relation.predicate,
                source=relation.source,
                confidence=relation.confidence,
                evidence=relation.evidence,
                **relation.attributes
            )
    
    def merge_duplicate_entities(self, similarity_threshold: float = 0.9):
        """Merge duplicate entities based on name similarity."""
        self.logger.info("Merging duplicate entities...")
        
        # Simple name-based merging (can be improved with more sophisticated methods)
        entity_groups = {}
        
        for entity_id, entity in self.entities.items():
            key = (entity.name.lower(), entity.type)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Merge entities in each group
        merged_count = 0
        for group in entity_groups.values():
            if len(group) > 1:
                # Keep the first entity as primary
                primary = group[0]
                
                for duplicate in group[1:]:
                    # Merge attributes
                    primary.synonyms.extend(duplicate.synonyms)
                    primary.attributes.update(duplicate.attributes)
                    
                    # Update graph references
                    self._update_entity_references(duplicate.id, primary.id)
                    
                    # Remove duplicate
                    del self.entities[duplicate.id]
                    if self.graph.has_node(duplicate.id):
                        self.graph.remove_node(duplicate.id)
                    
                    merged_count += 1
        
        self.logger.info(f"Merged {merged_count} duplicate entities")
    
    def _update_entity_references(self, old_id: str, new_id: str):
        """Update entity references in relations."""
        for relation_id, relation in self.relations.items():
            if relation.subject == old_id:
                relation.subject = new_id
            if relation.object == old_id:
                relation.object = new_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        stats = {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "entity_types": {},
            "relation_types": {},
            "sources": {}
        }
        
        # Entity type distribution
        for entity in self.entities.values():
            stats["entity_types"][entity.type] = stats["entity_types"].get(entity.type, 0) + 1
            stats["sources"][entity.source] = stats["sources"].get(entity.source, 0) + 1
        
        # Relation type distribution
        for relation in self.relations.values():
            stats["relation_types"][relation.predicate] = stats["relation_types"].get(relation.predicate, 0) + 1
        
        return stats
    
    def save_graph(self, output_path: str):
        """Save the knowledge graph to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for NetworkX graph
        import pickle
        with open(str(output_path).replace('.json', '.gpickle'), 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save as JSON for human readability
        graph_data = {
            "entities": {eid: asdict(entity) for eid, entity in self.entities.items()},
            "relations": {rid: asdict(relation) for rid, relation in self.relations.items()},
            "statistics": self.get_statistics()
        }
        
        # Handle datetime serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=json_serializer)
        
        self.logger.info(f"Knowledge graph saved to {output_path}")
    
    def load_graph(self, input_path: str):
        """Load knowledge graph from file."""
        input_path = Path(input_path)
        
        # Load NetworkX graph
        gpickle_path = str(input_path).replace('.json', '.gpickle')
        if Path(gpickle_path).exists():
            import pickle
            with open(gpickle_path, 'rb') as f:
                self.graph = pickle.load(f)
        
        # Load entities and relations
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct entities
        for eid, entity_data in data["entities"].items():
            self.entities[eid] = StandardizedEntity(**entity_data)
        
        # Reconstruct relations
        for rid, relation_data in data["relations"].items():
            self.relations[rid] = StandardizedRelation(**relation_data)
        
        self.logger.info(f"Knowledge graph loaded from {input_path}")


class KnowledgeGraphPreprocessor(LoggerMixin):
    """Main knowledge graph preprocessing coordinator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        
        # Initialize processors
        self.civic_processor = CivicProcessor(self.config)
        self.tcga_processor = TCGAProcessor(self.config)
        self.cptac_processor = CPTACProcessor(self.config)
        
        # Initialize graph builder
        self.graph_builder = KnowledgeGraphBuilder(self.config)
        
        # Initialize ontology mapper
        self.ontology_mapper = OntologyMapper(self.config)
    
    def download_all_data(self) -> bool:
        """Download data from all sources."""
        self.logger.info("Downloading data from all sources...")
        
        success = True
        
        # Download CIVIC data
        if not self.civic_processor.download_civic_data():
            self.logger.error("Failed to download CIVIC data")
            success = False
        
        # Download TCGA data
        if not self.tcga_processor.download_tcga_data():
            self.logger.error("Failed to download TCGA data")
            success = False
        
        # Download CPTAC data
        if not self.cptac_processor.download_cptac_data():
            self.logger.error("Failed to download CPTAC data")
            success = False
        
        return success
    
    def process_all_data(self) -> bool:
        """Process data from all sources and build integrated KG."""
        self.logger.info("Processing data from all sources...")
        
        all_entities = []
        all_relations = []
        
        # Process CIVIC data
        try:
            civic_entities, civic_relations = self.civic_processor.process_civic_data()
            all_entities.extend(civic_entities)
            all_relations.extend(civic_relations)
            self.logger.info(f"Added {len(civic_entities)} CIVIC entities and {len(civic_relations)} relations")
        except Exception as e:
            self.logger.error(f"Error processing CIVIC data: {e}")
        
        # Process TCGA data
        try:
            tcga_entities, tcga_relations = self.tcga_processor.process_tcga_data()
            all_entities.extend(tcga_entities)
            all_relations.extend(tcga_relations)
            self.logger.info(f"Added {len(tcga_entities)} TCGA entities and {len(tcga_relations)} relations")
        except Exception as e:
            self.logger.error(f"Error processing TCGA data: {e}")
        
        # Process CPTAC data
        try:
            cptac_entities, cptac_relations = self.cptac_processor.process_cptac_data()
            all_entities.extend(cptac_entities)
            all_relations.extend(cptac_relations)
            self.logger.info(f"Added {len(cptac_entities)} CPTAC entities and {len(cptac_relations)} relations")
        except Exception as e:
            self.logger.error(f"Error processing CPTAC data: {e}")
        
        # Build integrated knowledge graph
        self.logger.info("Building integrated knowledge graph...")
        
        self.graph_builder.add_entities(all_entities)
        self.graph_builder.add_relations(all_relations)
        
        # Merge duplicate entities
        self.graph_builder.merge_duplicate_entities()
        
        # Save ontology mappings
        self.ontology_mapper._save_cached_mappings()
        
        # Print statistics
        stats = self.graph_builder.get_statistics()
        self.logger.info("Knowledge Graph Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"    {subkey}: {subvalue}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        return True
    
    def save_integrated_graph(self, output_path: str):
        """Save the integrated knowledge graph."""
        self.graph_builder.save_graph(output_path)
    
    def load_integrated_graph(self, input_path: str):
        """Load the integrated knowledge graph."""
        self.graph_builder.load_graph(input_path)