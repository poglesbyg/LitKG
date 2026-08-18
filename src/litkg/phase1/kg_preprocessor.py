"""
Knowledge Graph Preprocessing Module

This module handles:
1. Data ingestion from CIVIC, TCGA, CPTAC
2. Entity standardization and harmonization
3. Ontology mapping (UMLS, Gene Ontology)
4. Graph construction and validation
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
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

        # Term-level ontology database, keyed by normalized surface form.
        # Populated by load_ontology(); consulted first by map_entity_to_ontology().
        self.ontology_db: Dict[str, Dict[str, Any]] = {}

        # Load cached mappings if available
        self._load_cached_mappings()

        # Load any ontology files present so their coverage is available to
        # map_to_umls() without the caller having to know they exist.
        self._autoload_ontologies()

    def _autoload_ontologies(self) -> int:
        """
        Load every ontology file in data/ontologies into ``ontology_db``.

        Returns:
            The number of ontologies loaded.
        """
        ontology_dir = get_data_dir() / "ontologies"
        if not ontology_dir.is_dir():
            return 0

        loaded = 0
        for path in sorted(ontology_dir.glob("*.json")):
            try:
                self.load_ontology(path.stem)
                loaded += 1
            except Exception as e:
                self.logger.warning(f"Could not load ontology {path.name}: {e}")

        return loaded

    @staticmethod
    def _normalize_term(term: str) -> str:
        """Normalize a surface form for ontology lookup."""
        return " ".join(str(term).lower().split())

    def _load_ontology_file(self, ontology_name: str) -> Dict[str, Any]:
        """
        Read a single ontology definition file from disk.

        Ontologies are JSON objects mapping a surface form to a term record,
        e.g. {"BRCA1": {"id": "HGNC:1100", "type": "gene", "synonyms": [...]}}.
        Returns an empty dict when the file is absent.
        """
        ontology_path = get_data_dir() / "ontologies" / f"{ontology_name}.json"

        if not ontology_path.exists():
            self.logger.warning(f"Ontology file not found: {ontology_path}")
            return {}

        try:
            with open(ontology_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Failed to read ontology '{ontology_name}': {e}")
            return {}

    def load_ontology(self, ontology_name: str) -> Dict[str, Any]:
        """
        Load an ontology and register its terms in ``ontology_db``.

        Each term is indexed under its own name and under any synonyms so that
        map_entity_to_ontology() resolves alternate surface forms.

        Returns:
            The ontology as read from disk, keyed by its original surface forms.
        """
        ontology = self._load_ontology_file(ontology_name)

        for surface_form, record in ontology.items():
            if not isinstance(record, dict):
                continue

            entry = {**record, "ontology": ontology_name}
            entry.setdefault("canonical_name", surface_form)

            self.ontology_db[self._normalize_term(surface_form)] = entry
            for synonym in record.get("synonyms", []):
                self.ontology_db.setdefault(self._normalize_term(synonym), entry)

        self.logger.info(
            f"Loaded ontology '{ontology_name}': {len(ontology)} terms "
            f"({len(self.ontology_db)} indexed surface forms)"
        )
        return ontology

    def map_entity_to_ontology(
        self, entity_name: str, entity_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve an entity name to an ontology term record.

        Consults the loaded ontology database first, then falls back to the
        UMLS/GO mappers for entities no ontology file covers.

        Returns:
            A record with at least ``id``, or None when the entity is unresolved.
        """
        record = self.ontology_db.get(self._normalize_term(entity_name))
        if record is None:
            record = self.ontology_db.get(entity_name)

        if record is not None:
            mapping = dict(record)
            mapping.setdefault("canonical_name", entity_name)
            return mapping

        # Fall back to the remote/heuristic mappers
        normalized_type = (entity_type or "").lower()
        if normalized_type in ("gene", "protein"):
            go_id = self.map_to_gene_ontology(entity_name)
            if go_id:
                return {
                    "id": go_id,
                    "type": normalized_type,
                    "canonical_name": entity_name,
                    "ontology": "GO",
                }

        umls_id = self.map_to_umls(entity_name, normalized_type or "unknown")
        if umls_id:
            return {
                "id": umls_id,
                "type": normalized_type or "unknown",
                "canonical_name": entity_name,
                "ontology": "UMLS",
            }

        return None

    def standardize_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Attach ontology identifiers to a list of extracted entities.

        Every input entity is returned, with ``ontology_id`` and
        ``canonical_name`` added. Unresolved entities get ``ontology_id: None``
        so callers can distinguish "not in any ontology" from "not attempted".
        """
        standardized = []

        for entity in entities:
            name = entity.get("text") or entity.get("name") or ""
            mapping = self.map_entity_to_ontology(name, entity.get("label") or entity.get("type"))

            enriched = dict(entity)
            if mapping:
                enriched["ontology_id"] = mapping.get("id")
                enriched["canonical_name"] = mapping.get("canonical_name", name)
            else:
                enriched["ontology_id"] = None
                enriched["canonical_name"] = name
            standardized.append(enriched)

        resolved = sum(1 for e in standardized if e["ontology_id"] is not None)
        self.logger.info(f"Standardized {resolved}/{len(standardized)} entities to ontology terms")
        return standardized


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
    
    def _lookup_loaded_ontology(
        self, entity_name: str, entity_type: str
    ) -> Optional[str]:
        """
        Resolve a CUI from ontology files loaded into ``ontology_db``.

        This is the path that scales: shipping or generating an ontology file
        gives coverage across a whole vocabulary, where the built-in heuristic
        covers only a handful of entities. It is consulted before the heuristic
        so a loaded ontology always wins.
        """
        record = self.ontology_db.get(self._normalize_term(entity_name))
        if not record:
            return None

        # Only accept a record of the right kind; a disease entry must not
        # supply a CUI for a gene of the same name.
        record_type = str(record.get("type", "")).upper()
        if record_type and entity_type and record_type != entity_type.upper():
            return None

        identifier = record.get("cui") or record.get("id")
        # GO ids annotate function, not identity, and must not be used as a CUI
        if identifier and not str(identifier).upper().startswith("GO:"):
            return identifier

        return None

    def _query_umls_api(self, entity_name: str, entity_type: str) -> Optional[str]:
        """
        Resolve an entity to a UMLS CUI.

        Order of preference: a loaded ontology file, then the live UMLS API
        when a key is configured, then the small built-in heuristic table.
        """
        from_ontology = self._lookup_loaded_ontology(entity_name, entity_type)
        if from_ontology:
            return from_ontology

        api_key = self.ontology_config["umls"].get("api_key")
        if api_key:
            cui = self._request_umls_cui(entity_name, api_key)
            if cui:
                return cui

        return self._heuristic_umls_mapping(entity_name, entity_type)

    def _request_umls_cui(self, entity_name: str, api_key: str) -> Optional[str]:
        """
        Look up a CUI through the UMLS REST API.

        Returns None on any failure, so a network problem degrades coverage
        rather than aborting knowledge graph construction.
        """
        try:
            response = requests.get(
                "https://uts-ws.nlm.nih.gov/rest/search/current",
                params={
                    "string": entity_name,
                    "apiKey": api_key,
                    "pageSize": 1,
                    "searchType": "exact",
                },
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("result", {}).get("results", [])
        except Exception as e:
            self.logger.debug(f"UMLS lookup failed for {entity_name!r}: {e}")
            return None

        if not results:
            return None

        cui = results[0].get("ui")
        # The API returns "NONE" as a sentinel for no match
        return cui if cui and cui != "NONE" else None
    
    # Fallback CUIs for common entities, keyed by uppercased name so that the
    # lookup below cannot disagree with the keys. Distinct genes must have
    # distinct CUIs: entity resolution treats a shared CUI as decisive, so a
    # copy-paste collision here would silently merge two different genes.
    COMMON_UMLS_MAPPINGS = {
        ("BRCA1", "GENE"): "C0376571",
        ("BRCA2", "GENE"): "C0376572",
        ("TP53", "GENE"): "C0080055",
        ("BREAST CANCER", "DISEASE"): "C0006142",
        ("LUNG CANCER", "DISEASE"): "C0242379",
        ("MELANOMA", "DISEASE"): "C0025202",
    }

    def _heuristic_umls_mapping(self, entity_name: str, entity_type: str) -> Optional[str]:
        """Heuristic UMLS mapping for common entities."""
        return self.COMMON_UMLS_MAPPINGS.get(
            (entity_name.upper().strip(), entity_type.upper())
        )
    
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
    
    # CIVIC publishes dated monthly releases and a rolling nightly build. The
    # default is pinned: a nightly build changes under you, so results stop
    # being reproducible and a regression cannot be told from a data update.
    # Override with LITKG_CIVIC_RELEASE=nightly (or another date) to refresh.
    DEFAULT_RELEASE = "01-Aug-2026"

    # Columns each file must contain for processing to mean anything. CIVIC has
    # renamed columns across releases -- an earlier version of this code read
    # 'drugs', 'variant_id' and 'clinical_significance' from an evidence file
    # that had none of them, and produced 4125 dangling edges in silence. A
    # missing column is now an error, not an empty string.
    REQUIRED_COLUMNS = {
        "evidence": (
            "evidence_id", "molecular_profile", "disease", "therapies",
            "evidence_type", "significance", "evidence_level", "citation",
        ),
        "variants": ("variant_id", "variant", "gene"),
        "genes": ("name", "entrez_id"),
    }

    @classmethod
    def release(cls) -> str:
        """The CIVIC release to download; env var wins over the pinned default."""
        return os.environ.get("LITKG_CIVIC_RELEASE", cls.DEFAULT_RELEASE).strip()

    @classmethod
    def download_urls(cls, release: Optional[str] = None) -> Dict[str, str]:
        """
        Download URLs for a release.

        "nightly" uses the rolling build; anything else is treated as a dated
        release directory ("01-Aug-2026"), which CIVIC names with the date
        repeated in the filename.
        """
        release = (release or cls.release()) or cls.DEFAULT_RELEASE
        files = {
            "variants": "VariantSummaries",
            "evidence": "ClinicalEvidenceSummaries",
            "genes": "GeneSummaries",
        }
        prefix = "nightly" if release.lower() == "nightly" else release
        return {
            key: f"https://civicdb.org/downloads/{prefix}/{prefix}-{name}.tsv"
            for key, name in files.items()
        }

    def download_civic_data(self, release: Optional[str] = None) -> bool:
        """Download a CIVIC release, verifying each file's schema on arrival."""
        release = release or self.release()
        self.logger.info(f"Downloading CIVIC release {release}...")

        data_dir = get_data_dir() / "external" / "civic"
        data_dir.mkdir(parents=True, exist_ok=True)

        for data_type, url in self.download_urls(release).items():
            try:
                response = requests.get(url, timeout=300)
                response.raise_for_status()

                filepath = data_dir / f"civic_{data_type}.tsv"
                filepath.write_bytes(response.content)

                # Verify before trusting it. A silently truncated or reshaped
                # file is worse than a failed download, because the pipeline
                # will happily build a graph out of nothing.
                self._verify_schema(data_type, filepath)
                self.logger.info(f"Downloaded {data_type} to {filepath}")

            except Exception as e:
                self.logger.error(f"Failed to download {data_type} data: {e}")
                return False

        (data_dir / "RELEASE").write_text(f"{release}\n")
        return True

    def _verify_schema(self, data_type: str, filepath: Path) -> None:
        """Raise if a downloaded file lacks columns the processor depends on."""
        required = self.REQUIRED_COLUMNS.get(data_type, ())
        if not required:
            return
        header = pd.read_csv(filepath, sep="\t", nrows=0).columns
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(
                f"CIVIC {data_type} file is missing required columns {missing}. "
                f"The release schema has changed; update REQUIRED_COLUMNS and the "
                f"processor rather than letting these read as empty strings."
            )

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
        
        # Process evidence. Needs the variants file to resolve the molecular
        # profile names evidence records use to identify their subject.
        evidence_file = data_dir / "civic_evidence.tsv"
        if evidence_file.exists():
            evidence_entities, evidence_relations = self._process_civic_evidence(
                evidence_file, variants_file
            )
            entities.extend(evidence_entities)
            relations.extend(evidence_relations)
        
        self.logger.info(f"Processed {len(entities)} entities and {len(relations)} relations from CIVIC")
        
        return entities, relations
    
    # Feature types seen in CIVIC releases from 2024 onward.
    FEATURE_TYPES = {
        "GENE": "GENE",
        "FUSION": "FUSION",
        "FACTOR": "FACTOR",
        "REGION": "REGION",
    }

    @staticmethod
    def _civic_gene_id(gene_name: Any, entrez_id: Any = None) -> str:
        """
        Canonical node id for a CIVIC gene.

        The gene and variant processors must derive this identically or every
        HAS_VARIANT edge dangles, leaving the graph with variants and no genes.
        Prefers the stable Entrez id and falls back to the gene symbol.
        """
        entrez = str(entrez_id or "").strip()
        if entrez and entrez.lower() not in ("nan", "none", "0"):
            # Entrez ids arrive from pandas as floats ("238.0")
            entrez = entrez[:-2] if entrez.endswith(".0") else entrez
            return f"CIVIC:GENE:ENTREZ:{entrez}"
        return f"CIVIC:GENE:{str(gene_name).strip().upper()}"

    def _process_civic_genes(self, genes_file: Path) -> List[StandardizedEntity]:
        """
        Process CIVIC genes into gene-level entities.

        These are the nodes literature mentions resolve against: NER extracts
        gene symbols ("BRCA1"), while CIVIC's variant records are named for the
        alteration ("1100delC"). Without gene nodes the two vocabularies cannot
        meet, and cross-modal linking collapses to the handful of variant
        notations that happen to appear verbatim in abstracts.
        """
        entities = []

        try:
            df = pd.read_csv(genes_file, sep='\t')

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CIVIC genes"):
                # civic_genes.tsv carries the symbol in "name"; only
                # civic_variants.tsv has a "gene" column. Accept either so a
                # schema change on one file cannot silently empty the vocabulary.
                gene_name = str(row.get('name', row.get('gene', ''))).strip()
                gene_id = str(row.get('gene_id', ''))
                entrez_id = row.get('entrez_id', '')

                if not gene_name or gene_name == 'nan':
                    continue

                # Releases from 2024 onward ship a features file rather than a
                # genes file: 617 genes alongside 345 fusions, 8 factors and 3
                # regions. Typing a fusion as a gene would put it in the gene
                # vocabulary that literature mentions resolve against, so the
                # declared feature type is honoured where present.
                feature_type = str(row.get('feature_type', '') or '').strip()
                entity_type = self.FEATURE_TYPES.get(feature_type.upper(), "GENE")
                
                # Map to ontologies
                cui = self.ontology_mapper.map_to_umls(gene_name, "GENE")
                go_id = self.ontology_mapper.map_to_gene_ontology(gene_name)
                
                entity = StandardizedEntity(
                    id=self._civic_gene_id(gene_name, entrez_id),
                    name=gene_name,
                    type=entity_type,
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
                        subject=self._civic_gene_id(gene_name, row.get('entrez_id', '')),
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
    
    # CIVIC evidence carries the clinical layer of the graph: which disease a
    # variant is implicated in, which therapy it predicts response to, and what
    # the direction of that prediction is. Mapped onto the same predicate
    # vocabulary the literature extractor emits so both sides are comparable.
    EVIDENCE_PREDICATES = {
        # (evidence_type, significance) -> (predicate, object_kind)
        ("Predictive", "Sensitivity/Response"): ("SENSITIZES_TO", "therapy"),
        ("Predictive", "Resistance"): ("RESISTANT_TO", "therapy"),
        ("Predictive", "Reduced Sensitivity"): ("RESISTANT_TO", "therapy"),
        ("Predictive", "Adverse Response"): ("RESISTANT_TO", "therapy"),
        ("Prognostic", "Poor Outcome"): ("PREDICTS_POOR_OUTCOME", "disease"),
        ("Prognostic", "Better Outcome"): ("PREDICTS_BETTER_OUTCOME", "disease"),
        ("Diagnostic", "Positive"): ("DIAGNOSTIC_FOR", "disease"),
        ("Diagnostic", "Negative"): ("EXCLUDES_DIAGNOSIS", "disease"),
        ("Predisposing", "Predisposition"): ("PREDISPOSES_TO", "disease"),
        ("Oncogenic", "Oncogenicity"): ("CAUSES", "disease"),
    }

    # CIVIC's evidence level, from validated clinical evidence down to
    # inferential. Used instead of a flat 0.8 so downstream filtering by
    # confidence means something.
    EVIDENCE_LEVEL_CONFIDENCE = {
        "A": 0.95,  # validated association
        "B": 0.80,  # clinical evidence
        "C": 0.60,  # case study
        "D": 0.45,  # preclinical
        "E": 0.30,  # inferential
    }

    @staticmethod
    def _civic_disease_id(disease_name: Any, doid: Any = None) -> str:
        """
        Canonical node id for a CIVIC disease.

        Prefers the Disease Ontology id, which is a genuine identity
        identifier -- two records sharing a DOID are the same disease.
        """
        raw = str(doid or "").strip()
        if raw and raw.lower() not in ("nan", "none", "0"):
            raw = raw[:-2] if raw.endswith(".0") else raw
            return f"CIVIC:DISEASE:DOID:{raw}"
        return f"CIVIC:DISEASE:{str(disease_name).strip().upper()}"

    @staticmethod
    def _civic_therapy_id(therapy_name: Any) -> str:
        """Canonical node id for a CIVIC therapy."""
        return f"CIVIC:THERAPY:{str(therapy_name).strip().upper()}"

    @staticmethod
    def _civic_phenotype_id(phenotype_name: Any) -> str:
        """Canonical node id for a CIVIC phenotype."""
        return f"CIVIC:PHENOTYPE:{str(phenotype_name).strip().upper()}"

    @staticmethod
    def _split_multi_valued(value: Any) -> List[str]:
        """CIVIC packs therapies and phenotypes into one comma-separated cell."""
        raw = str(value or "").strip()
        if not raw or raw.lower() == "nan":
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _build_molecular_profile_index(self, variants_file: Path) -> Dict[str, str]:
        """
        Map CIVIC molecular profile names onto variant node ids.

        Evidence records identify their subject by profile name ("JAK2 V617F"),
        not by variant id, so without this index every evidence relation points
        at a variant that does not exist.
        """
        index: Dict[str, str] = {}
        try:
            df = pd.read_csv(variants_file, sep="\t")
        except Exception as e:
            self.logger.error(f"Could not index molecular profiles: {e}")
            return index

        for _, row in df.iterrows():
            gene = str(row.get("gene", "")).strip().upper()
            variant = str(row.get("variant", "")).strip().upper()
            variant_id = str(row.get("variant_id", "")).strip()
            if not gene or not variant or gene == "NAN" or variant == "NAN":
                continue
            index.setdefault(f"{gene} {variant}", f"CIVIC:VARIANT:{variant_id}")

        return index

    def _resolve_molecular_profile(
        self, profile: str, index: Dict[str, str]
    ) -> List[str]:
        """
        Resolve a profile name to the variant nodes it refers to.

        Compound profiles ("BRAF V600E AND BRAF V600M") assert evidence about a
        conjunction of variants; each component gets the relation, marked
        compound in the relation attributes so the distinction is not lost.
        """
        key = profile.strip().upper()
        if key in index:
            return [index[key]]

        parts = re.split(r"\s+(?:AND|OR)\s+", key)
        if len(parts) < 2:
            return []
        return [index[p.strip()] for p in parts if p.strip() in index]

    def _evidence_confidence(self, row: Any) -> float:
        """Confidence from CIVIC's own evidence level, adjusted by curator rating."""
        level = str(row.get("evidence_level", "")).strip().upper()
        confidence = self.EVIDENCE_LEVEL_CONFIDENCE.get(level, 0.5)

        # Curator rating is 1-5 stars; nudge within the level's band.
        try:
            rating = float(row.get("rating"))
            if rating == rating:  # not NaN
                confidence *= 0.85 + 0.06 * rating
        except (TypeError, ValueError):
            pass

        return round(min(confidence, 1.0), 3)

    def _process_civic_evidence(
        self, evidence_file: Path, variants_file: Optional[Path] = None
    ) -> Tuple[List[StandardizedEntity], List[StandardizedRelation]]:
        """
        Process CIVIC evidence into clinical entities and relations.

        This is where diseases, therapies and phenotypes enter the graph. The
        previous version emitted relations pointing at CIVIC:DISEASE and
        CIVIC:DRUG nodes that were never created, and read column names the
        file does not have ('variant_id', 'drugs', 'clinical_significance'),
        so every subject was the empty string and no therapy relation was ever
        built. The result was 4125 dangling edges out of 5825.
        """
        entities: List[StandardizedEntity] = []
        relations: List[StandardizedRelation] = []
        seen_entities: Set[str] = set()

        profile_index: Dict[str, str] = {}
        if variants_file and variants_file.exists():
            profile_index = self._build_molecular_profile_index(variants_file)
            self.logger.info(f"Indexed {len(profile_index)} molecular profiles")

        def add_entity(entity: StandardizedEntity) -> None:
            if entity.id not in seen_entities:
                seen_entities.add(entity.id)
                entities.append(entity)

        unresolved = 0

        try:
            df = pd.read_csv(evidence_file, sep="\t")

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CIVIC evidence"):
                evidence_id = str(row.get("evidence_id", "")).strip()
                if not evidence_id or evidence_id == "nan":
                    continue

                profile = str(row.get("molecular_profile", "")).strip()
                subjects = self._resolve_molecular_profile(profile, profile_index)
                if not subjects:
                    unresolved += 1
                    continue

                evidence_type = str(row.get("evidence_type", "")).strip()
                significance = str(row.get("significance", "")).strip()
                direction = str(row.get("evidence_direction", "")).strip()
                # "Does Not Support" is evidence against the association. It is
                # kept, not dropped, but must not be asserted as a plain fact.
                negated = direction == "Does Not Support"
                confidence = self._evidence_confidence(row)
                citation = str(row.get("citation", "")).strip()

                predicate, object_kind = self.EVIDENCE_PREDICATES.get(
                    (evidence_type, significance), ("ASSOCIATED_WITH", "disease")
                )

                shared_attributes = {
                    "evidence_type": evidence_type,
                    "significance": significance,
                    "evidence_direction": direction,
                    "evidence_level": str(row.get("evidence_level", "")).strip(),
                    "negated": negated,
                    "compound_profile": len(subjects) > 1,
                    "molecular_profile": profile,
                }
                evidence_text = [f"CIVIC evidence {evidence_id}"]
                if citation and citation != "nan":
                    evidence_text.append(citation)

                # --- Disease ---
                disease_name = str(row.get("disease", "")).strip()
                disease_id = None
                if disease_name and disease_name != "nan":
                    doid = row.get("doid")
                    disease_id = self._civic_disease_id(disease_name, doid)
                    doid_str = str(doid or "").strip()
                    doid_str = doid_str[:-2] if doid_str.endswith(".0") else doid_str
                    add_entity(StandardizedEntity(
                        id=disease_id,
                        name=disease_name,
                        type="DISEASE",
                        source="CIVIC",
                        original_id=f"DOID:{doid_str}" if doid_str not in ("", "nan") else disease_name,
                        synonyms=[],
                        attributes={"doid": doid_str} if doid_str not in ("", "nan") else {},
                    ))

                # --- Therapies ---
                therapy_ids = []
                for therapy_name in self._split_multi_valued(row.get("therapies")):
                    therapy_id = self._civic_therapy_id(therapy_name)
                    therapy_ids.append(therapy_id)
                    add_entity(StandardizedEntity(
                        id=therapy_id,
                        name=therapy_name,
                        type="DRUG",
                        source="CIVIC",
                        original_id=therapy_name,
                        synonyms=[],
                        attributes={
                            "interaction_type": str(row.get("therapy_interaction_type", "")).strip(),
                        },
                    ))

                # --- Phenotypes ---
                phenotype_ids = []
                for phenotype_name in self._split_multi_valued(row.get("phenotypes")):
                    phenotype_id = self._civic_phenotype_id(phenotype_name)
                    phenotype_ids.append(phenotype_id)
                    add_entity(StandardizedEntity(
                        id=phenotype_id,
                        name=phenotype_name,
                        type="PHENOTYPE",
                        source="CIVIC",
                        original_id=phenotype_name,
                        synonyms=[],
                    ))

                for subject in subjects:
                    suffix = subject.rsplit(":", 1)[-1]

                    # Predictive evidence is about a therapy; everything else
                    # is about the disease. Emitting a therapy predicate onto a
                    # disease node is what makes a graph unusable downstream.
                    if object_kind == "therapy" and therapy_ids:
                        for therapy_id in therapy_ids:
                            relations.append(StandardizedRelation(
                                id=f"CIVIC:REL:VARIANT_THERAPY:{evidence_id}:{suffix}:{therapy_id.rsplit(':', 1)[-1]}",
                                subject=subject,
                                predicate=predicate,
                                object=therapy_id,
                                source="CIVIC",
                                confidence=confidence,
                                evidence=list(evidence_text),
                                attributes=dict(shared_attributes, disease_context=disease_id),
                            ))

                    if disease_id:
                        disease_predicate = (
                            predicate if object_kind == "disease" else "ASSOCIATED_WITH"
                        )
                        relations.append(StandardizedRelation(
                            id=f"CIVIC:REL:VARIANT_DISEASE:{evidence_id}:{suffix}",
                            subject=subject,
                            predicate=disease_predicate,
                            object=disease_id,
                            source="CIVIC",
                            confidence=confidence,
                            evidence=list(evidence_text),
                            attributes=dict(shared_attributes),
                        ))

                    for phenotype_id in phenotype_ids:
                        relations.append(StandardizedRelation(
                            id=f"CIVIC:REL:VARIANT_PHENOTYPE:{evidence_id}:{suffix}:{phenotype_id.rsplit(':', 1)[-1]}",
                            subject=subject,
                            predicate="PRESENTS_WITH",
                            object=phenotype_id,
                            source="CIVIC",
                            confidence=confidence,
                            evidence=list(evidence_text),
                            attributes=dict(shared_attributes),
                        ))

                    # Therapy is indicated for the disease it was evidenced in.
                    if disease_id and object_kind == "therapy":
                        for therapy_id in therapy_ids:
                            relations.append(StandardizedRelation(
                                id=f"CIVIC:REL:THERAPY_DISEASE:{evidence_id}:{therapy_id.rsplit(':', 1)[-1]}",
                                subject=therapy_id,
                                predicate="TREATS",
                                object=disease_id,
                                source="CIVIC",
                                confidence=confidence,
                                evidence=list(evidence_text),
                                attributes=dict(shared_attributes),
                            ))

            if unresolved:
                self.logger.warning(
                    f"{unresolved} evidence rows had no resolvable molecular profile"
                )

        except Exception as e:
            self.logger.error(f"Error processing CIVIC evidence: {e}")

        self.logger.info(
            f"CIVIC evidence produced {len(entities)} clinical entities "
            f"and {len(relations)} relations"
        )
        return entities, relations


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
    
    # Entity attributes that identify *which* entity this is, and are therefore
    # decisive when two entities share one. Function/process annotations such as
    # go_id are excluded on purpose: they describe what an entity does and are
    # shared by many distinct entities by design.
    IDENTITY_IDENTIFIERS = ("cui", "doid")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a surface form for comparison."""
        import re

        lowered = str(name).lower().strip()
        # Fold the punctuation that distinguishes BRCA1 / BRCA-1 / BRCA 1
        collapsed = re.sub(r"[\s\-_/.]+", "", lowered)
        return collapsed

    def _blocking_key(self, entity: StandardizedEntity) -> Tuple[str, str]:
        """
        Cheap key restricting which entities are compared to each other.

        Fuzzy comparison is quadratic, so candidates are blocked by type and by
        the first character of the normalized name. Entities in different
        blocks are never compared, which keeps merging tractable as the graph
        grows without materially affecting recall: near-duplicate surface forms
        almost always agree on their first character.
        """
        normalized = self._normalize_name(entity.name)
        return (entity.type, normalized[:1])

    def merge_duplicate_entities(
        self,
        similarity_threshold: float = 0.9,
        use_ontology: bool = True
    ) -> Dict[str, int]:
        """
        Merge entities that refer to the same real-world thing.

        Resolution runs as a cascade, strongest evidence first:

        1. **Shared identity identifier** (UMLS CUI). Decisive regardless of
           surface form, so "BRCA1" and "breast cancer 1" merge when both carry
           CUI C0376571.

           Deliberately excludes GO IDs. A GO term annotates what a gene
           *does*, not which gene it *is*: BRCA1 and BRCA2 both carry
           GO:0006281 ("DNA repair"), correctly, and treating that as identity
           evidence merges two distinct genes.
        2. **Identical normalized name**, after folding case, spaces, and
           hyphens, so "BRCA-1" meets "BRCA1".
        3. **Synonym overlap** between the two entities.
        4. **Fuzzy surface similarity** at or above ``similarity_threshold``,
           within the same entity type.

        Matches are accumulated with union-find so resolution is transitive: if
        A matches B and B matches C, all three collapse into one node even when
        A and C would not have matched directly.

        Args:
            similarity_threshold: Minimum fuzzy similarity for rule 4.
            use_ontology: Whether to use ontology identifiers (rule 1).

        Returns:
            Counts per rule, plus "merged" and "remaining".
        """
        self.logger.info(
            f"Merging duplicate entities (threshold={similarity_threshold}, "
            f"ontology={use_ontology})"
        )

        entity_ids = list(self.entities)
        parent = {eid: eid for eid in entity_ids}

        def find(eid: str) -> str:
            while parent[eid] != eid:
                parent[eid] = parent[parent[eid]]  # path compression
                eid = parent[eid]
            return eid

        def union(a: str, b: str) -> bool:
            root_a, root_b = find(a), find(b)
            if root_a == root_b:
                return False
            parent[root_b] = root_a
            return True

        stats = {"ontology": 0, "exact_name": 0, "synonym": 0, "fuzzy": 0}

        # Rule 1: shared identity identifier. Only identifiers that name the
        # entity itself qualify; see IDENTITY_IDENTIFIERS.
        if use_ontology:
            for attribute in self.IDENTITY_IDENTIFIERS:
                by_identifier: Dict[str, List[str]] = defaultdict(list)
                for eid in entity_ids:
                    value = getattr(self.entities[eid], attribute, None)
                    if value:
                        by_identifier[value].append(eid)

                for group in by_identifier.values():
                    for other in group[1:]:
                        if union(group[0], other):
                            stats["ontology"] += 1

        # Rules 2-3: index every surface form an entity is known by, so a
        # collision means two entities share a name or a synonym. Done globally
        # rather than per block: synonyms are precisely the case where surface
        # forms diverge at the first character ("TP53" vs its synonym "p53"),
        # which blocking would hide.
        by_surface_form: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for eid in entity_ids:
            entity = self.entities[eid]
            for surface in {entity.name, *entity.synonyms}:
                normalized = self._normalize_name(surface)
                if normalized:
                    by_surface_form[(entity.type, normalized)].append(eid)

        for (_, normalized), group in by_surface_form.items():
            if len(group) < 2:
                continue
            for other in group[1:]:
                if union(group[0], other):
                    # Sharing a primary name is an exact match; sharing only a
                    # synonym is weaker evidence, counted separately.
                    is_exact = all(
                        self._normalize_name(self.entities[e].name) == normalized
                        for e in (group[0], other)
                    )
                    stats["exact_name" if is_exact else "synonym"] += 1

        # Rule 4 operates within blocks
        blocks: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for eid in entity_ids:
            blocks[self._blocking_key(self.entities[eid])].append(eid)

        fuzzy_matcher = None
        if similarity_threshold < 1.0:
            try:
                # Imported here: entity_linker imports this module
                from .entity_linker import FuzzyMatcher
                fuzzy_matcher = FuzzyMatcher(self.config)
            except Exception as e:
                self.logger.warning(f"Fuzzy matching unavailable ({e}); using exact rules only")

        if fuzzy_matcher is not None:
            for block in blocks.values():
                if len(block) < 2:
                    continue

                for i, first_id in enumerate(block):
                    first = self.entities[first_id]

                    for second_id in block[i + 1:]:
                        # Already resolved by a stronger rule
                        if find(first_id) == find(second_id):
                            continue

                        second = self.entities[second_id]
                        score = fuzzy_matcher.calculate_similarity(first.name, second.name)
                        if score >= similarity_threshold:
                            if union(first_id, second_id):
                                stats["fuzzy"] += 1

        merged_count = self._apply_entity_clusters(parent, find)

        stats["merged"] = merged_count
        stats["remaining"] = len(self.entities)
        self.logger.info(
            f"Merged {merged_count} duplicate entities "
            f"(ontology={stats['ontology']}, exact={stats['exact_name']}, "
            f"synonym={stats['synonym']}, fuzzy={stats['fuzzy']}); "
            f"{stats['remaining']} entities remain"
        )
        return stats

    def _apply_entity_clusters(self, parent: Dict[str, str], find) -> int:
        """
        Collapse each resolved cluster into a single canonical entity.

        The canonical entity is the one carrying an ontology identifier where
        available, then the one with the richest synonym set, so merging never
        discards the best-described member of a cluster.

        Returns:
            The number of entities removed.
        """
        clusters: Dict[str, List[str]] = defaultdict(list)
        for eid in list(parent):
            clusters[find(eid)].append(eid)

        merged_count = 0

        for members in clusters.values():
            if len(members) < 2:
                continue

            def rank(eid: str) -> Tuple[int, int, int]:
                entity = self.entities[eid]
                has_ontology = 1 if (entity.cui or entity.go_id) else 0
                return (has_ontology, len(entity.synonyms), len(entity.attributes))

            ordered = sorted(members, key=rank, reverse=True)
            primary = self.entities[ordered[0]]

            for duplicate_id in ordered[1:]:
                duplicate = self.entities[duplicate_id]

                # Preserve every surface form the cluster knew about
                for surface in [duplicate.name, *duplicate.synonyms]:
                    if surface and surface not in primary.synonyms and surface != primary.name:
                        primary.synonyms.append(surface)

                # Keep ontology identifiers the primary lacks
                if not primary.cui and duplicate.cui:
                    primary.cui = duplicate.cui
                if not primary.go_id and duplicate.go_id:
                    primary.go_id = duplicate.go_id

                primary.attributes.update(duplicate.attributes)

                self._update_entity_references(duplicate_id, primary.id)

                del self.entities[duplicate_id]
                if self.graph.has_node(duplicate_id):
                    self.graph.remove_node(duplicate_id)

                merged_count += 1

        return merged_count
    
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

    def load_knowledge_graph(self, source: str) -> Dict[str, Any]:
        """
        Load a knowledge graph from a named source.

        Args:
            source: Source name, e.g. "civic", "tcga", "cptac".

        Returns:
            A dict with "nodes" and "edges" keys, preprocessed and validated.
        """
        self.logger.info(f"Loading knowledge graph from source: {source}")
        kg = self._load_kg_from_source(source)

        kg = {
            **kg,
            "nodes": self.preprocess_nodes(kg.get("nodes", [])),
            "edges": self.preprocess_edges(kg.get("edges", [])),
        }

        self.logger.info(
            f"Loaded {len(kg['nodes'])} nodes and {len(kg['edges'])} edges from {source}"
        )
        return kg

    def save_graph(self, graph: nx.Graph, output_path: str) -> None:
        """Persist a NetworkX graph to disk via pickle."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(
            f"Saved graph ({graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges) to {path}"
        )

    def load_graph(self, input_path: str) -> nx.Graph:
        """Load a NetworkX graph previously written by save_graph()."""
        path = Path(input_path)

        with open(path, "rb") as f:
            graph = pickle.load(f)

        self.logger.info(
            f"Loaded graph ({graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges) from {path}"
        )
        return graph

    def _load_kg_from_source(self, source: str) -> Dict[str, Any]:
        """
        Read the raw node/edge payload for a single source.

        Reads the cached JSON export written by the per-source processors.
        Returns an empty graph when the source has not been downloaded yet.
        """
        source_path = get_data_dir() / "processed" / f"{source}_kg.json"

        if not source_path.exists():
            self.logger.warning(
                f"No processed data for source '{source}' at {source_path}; "
                "run download_all_data()/process_all_data() first"
            )
            return {"nodes": [], "edges": []}

        try:
            with open(source_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Failed to read KG for source '{source}': {e}")
            return {"nodes": [], "edges": []}

        return {"nodes": data.get("nodes", []), "edges": data.get("edges", [])}

    def preprocess_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Ensure ids exist and normalize casing
        processed = []
        for n in nodes:
            nid = str(n.get("id", n.get("name", "node"))).strip()
            processed.append({"id": nid, **{k: v for k, v in n.items() if k != "id"}})
        return processed

    def preprocess_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for e in edges:
            src = e.get("source") or e.get("src") or e.get("from")
            dst = e.get("target") or e.get("dst") or e.get("to")
            if src is None or dst is None:
                continue
            processed.append({"source": src, "target": dst, **{k: v for k, v in e.items() if k not in ("source", "target")}})
        return processed

    def build_networkx_graph(
        self,
        kg: Dict[str, Any],
        directed: bool = True,
        multigraph: bool = True
    ) -> nx.Graph:
        """
        Build a NetworkX graph from a node/edge payload.

        Defaults preserve what a biomedical knowledge graph actually encodes:

        - **Directed**, because relation direction carries meaning: a drug
          TREATS a disease, and the reverse is not a fact.
        - **Multigraph**, because a pair of entities can be joined by several
          distinct relations (ASSOCIATED_WITH and MUTATED_IN are different
          claims), and a simple graph silently keeps only the last one.

        Pass directed=False/multigraph=False only for algorithms that require a
        simple undirected graph, and expect that to discard information.

        Args:
            kg: {"nodes": [{"id", ...}], "edges": [{"source", "target", ...}]}
            directed: Preserve relation direction.
            multigraph: Preserve parallel edges between the same pair.

        Returns:
            The graph, of the type implied by the flags.
        """
        graph_type = {
            (True, True): nx.MultiDiGraph,
            (True, False): nx.DiGraph,
            (False, True): nx.MultiGraph,
            (False, False): nx.Graph,
        }[(directed, multigraph)]

        G = graph_type()
        for n in kg.get("nodes", []):
            G.add_node(n.get("id", n.get("name", "node")), **n)

        dropped = 0
        for e in kg.get("edges", []):
            source, target = e.get("source"), e.get("target")
            if not multigraph and G.has_edge(source, target):
                dropped += 1
            G.add_edge(source, target, **e)

        if dropped:
            self.logger.warning(
                f"{dropped} parallel edge(s) collapsed because multigraph=False; "
                "distinct relation types between the same entities were merged"
            )

        return G

    def compute_graph_statistics(self, G: nx.Graph) -> Dict[str, Any]:
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
        }

# Short alias used across the package and by scripts. Defined here so that both
# `litkg.phase1` and this module expose the same name.
KGPreprocessor = KnowledgeGraphPreprocessor
