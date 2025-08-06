"""
Literature Processing Pipeline for biomedical text analysis.

This module handles:
1. PubMed data retrieval
2. Biomedical NLP using PubMedBERT, BioBERT, and scispacy
3. Entity extraction (genes, diseases, drugs, etc.)
4. Relation extraction
5. Context analysis
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

import spacy
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForTokenClassification,
    pipeline, BertTokenizer, BertModel
)
from Bio import Entrez
import requests
import pandas as pd
from tqdm import tqdm

from litkg.utils.config import LitKGConfig, load_config
from litkg.utils.logging import LoggerMixin


@dataclass
class Entity:
    """Represents an extracted biomedical entity."""
    text: str
    label: str  # GENE, DISEASE, DRUG, etc.
    start: int
    end: int
    confidence: float
    cui: Optional[str] = None  # UMLS Concept Unique Identifier
    synonyms: List[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class Relation:
    """Represents an extracted relation between entities."""
    subject: Entity
    predicate: str  # TREATS, CAUSES, INTERACTS_WITH, etc.
    object: Entity
    confidence: float
    context: str
    sentence: str
    

@dataclass
class ProcessedDocument:
    """Represents a processed biomedical document."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: datetime
    entities: List[Entity]
    relations: List[Relation]
    full_text: Optional[str] = None
    keywords: List[str] = None
    mesh_terms: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.mesh_terms is None:
            self.mesh_terms = []


class PubMedRetriever(LoggerMixin):
    """Handles PubMed data retrieval and preprocessing."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.pubmed_config = config.phase1.literature.pubmed
        
        # Set up Entrez
        Entrez.email = self.pubmed_config["email"]
        if self.pubmed_config.get("api_key"):
            Entrez.api_key = self.pubmed_config["api_key"]
    
    def search_pubmed(
        self, 
        query: str, 
        max_results: int = None,
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[str]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            date_range: Tuple of (start_date, end_date) in YYYY/MM/DD format
            
        Returns:
            List of PMIDs
        """
        if max_results is None:
            max_results = self.pubmed_config["max_results"]
        
        search_term = query
        if date_range:
            start_date, end_date = date_range
            search_term += f" AND {start_date}[PDAT]:{end_date}[PDAT]"
        
        self.logger.info(f"Searching PubMed with query: {search_term}")
        
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=search_term,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results["IdList"]
            self.logger.info(f"Found {len(pmids)} articles")
            return pmids
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return []
    
    def fetch_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed information for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries
        """
        if not pmids:
            return []
        
        batch_size = self.pubmed_config["batch_size"]
        articles = []
        
        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching articles"):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_pmids,
                    rettype="medline",
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                
                for record in records["PubmedArticle"]:
                    article = self._parse_pubmed_record(record)
                    if article:
                        articles.append(article)
                        
            except Exception as e:
                self.logger.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                continue
        
        self.logger.info(f"Successfully fetched {len(articles)} articles")
        return articles
    
    def _parse_pubmed_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a PubMed XML record into a structured dictionary."""
        try:
            medline_citation = record["MedlineCitation"]
            article = medline_citation["Article"]
            
            # Basic information
            pmid = str(medline_citation["PMID"])
            title = article.get("ArticleTitle", "")
            
            # Abstract
            abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)
            
            # Authors
            authors = []
            author_list = article.get("AuthorList", [])
            for author in author_list:
                if "LastName" in author and "ForeName" in author:
                    authors.append(f"{author['ForeName']} {author['LastName']}")
            
            # Journal
            journal = article.get("Journal", {}).get("Title", "")
            
            # Publication date
            pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            try:
                year = int(pub_date.get("Year", datetime.now().year))
                month = int(pub_date.get("Month", 1))
                day = int(pub_date.get("Day", 1))
                publication_date = datetime(year, month, day)
            except (ValueError, TypeError):
                publication_date = datetime.now()
            
            # MeSH terms
            mesh_terms = []
            mesh_heading_list = medline_citation.get("MeshHeadingList", [])
            for mesh_heading in mesh_heading_list:
                descriptor = mesh_heading.get("DescriptorName", {})
                if isinstance(descriptor, dict):
                    mesh_terms.append(descriptor.get("text", ""))
                else:
                    mesh_terms.append(str(descriptor))
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "publication_date": publication_date,
                "mesh_terms": mesh_terms
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing record: {e}")
            return None


class BiomedicalNLP(LoggerMixin):
    """Biomedical NLP processor using multiple models."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.models_config = config.phase1.literature.models
        self.text_config = config.phase1.literature.text_processing
        
        # Load models
        self._load_models()
        
        # Entity types we're interested in
        self.entity_types = {
            "GENE", "DISEASE", "DRUG", "PROTEIN", "CELL_TYPE", 
            "TISSUE", "ORGANISM", "CHEMICAL", "MUTATION"
        }
        
        # Relation types
        self.relation_types = {
            "TREATS", "CAUSES", "PREVENTS", "INTERACTS_WITH",
            "REGULATES", "EXPRESSED_IN", "ASSOCIATED_WITH",
            "INHIBITS", "ACTIVATES", "BINDS"
        }
    
    def _load_models(self):
        """Load all required NLP models."""
        self.logger.info("Loading biomedical NLP models...")
        
        # Load scispacy model
        try:
            self.nlp = spacy.load(self.models_config["scispacy_model"])
            self.logger.info("Loaded scispacy model")
        except OSError:
            self.logger.error("scispacy model not found. Please install it first.")
            raise
        
        # Load PubMedBERT
        try:
            self.pubmedbert_tokenizer = AutoTokenizer.from_pretrained(
                self.models_config["pubmedbert"]
            )
            self.pubmedbert_model = AutoModel.from_pretrained(
                self.models_config["pubmedbert"]
            )
            self.logger.info("Loaded PubMedBERT model")
        except Exception as e:
            self.logger.error(f"Error loading PubMedBERT: {e}")
        
        # Load BioBERT
        try:
            self.biobert_tokenizer = BertTokenizer.from_pretrained(
                self.models_config["biobert"]
            )
            self.biobert_model = BertModel.from_pretrained(
                self.models_config["biobert"]
            )
            self.logger.info("Loaded BioBERT model")
        except Exception as e:
            self.logger.error(f"Error loading BioBERT: {e}")
        
        # Create NER pipeline
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=self.models_config["biobert"],
                tokenizer=self.models_config["biobert"],
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("Created NER pipeline")
        except Exception as e:
            self.logger.error(f"Error creating NER pipeline: {e}")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract biomedical entities from text using multiple approaches.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Method 1: scispacy NER
        entities.extend(self._extract_entities_scispacy(text))
        
        # Method 2: BERT-based NER
        entities.extend(self._extract_entities_bert(text))
        
        # Method 3: Rule-based patterns
        entities.extend(self._extract_entities_rules(text))
        
        # Deduplicate and filter
        entities = self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= self.text_config["min_entity_confidence"]]
        
        return entities
    
    def _extract_entities_scispacy(self, text: str) -> List[Entity]:
        """Extract entities using scispacy."""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # scispacy doesn't provide confidence scores
                    cui=ent._.cui if hasattr(ent._, 'cui') else None
                )
                entities.append(entity)
        
        return entities
    
    def _extract_entities_bert(self, text: str) -> List[Entity]:
        """Extract entities using BERT-based NER."""
        entities = []
        
        try:
            # Use the NER pipeline
            results = self.ner_pipeline(text)
            
            for result in results:
                # Map BERT labels to our entity types
                label = self._map_bert_label(result['entity_group'])
                if label in self.entity_types:
                    entity = Entity(
                        text=result['word'],
                        label=label,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score']
                    )
                    entities.append(entity)
        
        except Exception as e:
            self.logger.error(f"Error in BERT NER: {e}")
        
        return entities
    
    def _extract_entities_rules(self, text: str) -> List[Entity]:
        """Extract entities using rule-based patterns."""
        entities = []
        
        # Gene patterns (e.g., TP53, BRCA1, etc.)
        gene_pattern = r'\b[A-Z][A-Z0-9]{2,10}\b'
        for match in re.finditer(gene_pattern, text):
            if self._is_likely_gene(match.group()):
                entity = Entity(
                    text=match.group(),
                    label="GENE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                )
                entities.append(entity)
        
        # Drug patterns (common suffixes)
        drug_suffixes = ['-ib', '-mab', '-tinib', '-zumab', '-ine', '-ol']
        for suffix in drug_suffixes:
            pattern = r'\b\w+' + re.escape(suffix) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    label="DRUG",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.6
                )
                entities.append(entity)
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract relations between entities.
        
        Args:
            text: Input text
            entities: List of entities in the text
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        # Simple pattern-based relation extraction
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_entities = [
                e for e in entities 
                if e.start >= sentence['start'] and e.end <= sentence['end']
            ]
            
            if len(sentence_entities) >= 2:
                relations.extend(
                    self._extract_relations_from_sentence(
                        sentence['text'], sentence_entities
                    )
                )
        
        return relations
    
    def _extract_relations_from_sentence(
        self, sentence: str, entities: List[Entity]
    ) -> List[Relation]:
        """Extract relations from a single sentence."""
        relations = []
        
        # Define relation patterns
        patterns = {
            "TREATS": [
                r"(\w+)\s+treats?\s+(\w+)",
                r"(\w+)\s+therapy\s+for\s+(\w+)",
                r"treatment\s+of\s+(\w+)\s+with\s+(\w+)"
            ],
            "CAUSES": [
                r"(\w+)\s+causes?\s+(\w+)",
                r"(\w+)\s+leads?\s+to\s+(\w+)",
                r"(\w+)\s+induces?\s+(\w+)"
            ],
            "ASSOCIATED_WITH": [
                r"(\w+)\s+associated\s+with\s+(\w+)",
                r"(\w+)\s+correlated\s+with\s+(\w+)"
            ]
        }
        
        for relation_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Find entities that match the pattern groups
                    group1, group2 = match.groups()
                    
                    subject = self._find_entity_by_text(entities, group1)
                    obj = self._find_entity_by_text(entities, group2)
                    
                    if subject and obj:
                        relation = Relation(
                            subject=subject,
                            predicate=relation_type,
                            object=obj,
                            confidence=0.7,
                            context=sentence,
                            sentence=sentence
                        )
                        relations.append(relation)
        
        return relations
    
    def process_document(self, article_data: Dict[str, Any]) -> ProcessedDocument:
        """
        Process a single document through the complete NLP pipeline.
        
        Args:
            article_data: Dictionary containing article information
            
        Returns:
            ProcessedDocument object
        """
        self.logger.info(f"Processing document PMID: {article_data['pmid']}")
        
        # Combine title and abstract for processing
        full_text = f"{article_data['title']} {article_data['abstract']}"
        
        # Extract entities
        entities = self.extract_entities(full_text)
        
        # Extract relations
        relations = self.extract_relations(full_text, entities)
        
        # Create processed document
        doc = ProcessedDocument(
            pmid=article_data["pmid"],
            title=article_data["title"],
            abstract=article_data["abstract"],
            authors=article_data["authors"],
            journal=article_data["journal"],
            publication_date=article_data["publication_date"],
            entities=entities,
            relations=relations,
            mesh_terms=article_data.get("mesh_terms", [])
        )
        
        self.logger.info(
            f"Extracted {len(entities)} entities and {len(relations)} relations"
        )
        
        return doc
    
    # Helper methods
    def _map_bert_label(self, bert_label: str) -> str:
        """Map BERT NER labels to our entity types."""
        mapping = {
            "PER": "PERSON",
            "ORG": "ORGANIZATION",
            "LOC": "LOCATION",
            "MISC": "MISCELLANEOUS"
        }
        return mapping.get(bert_label, bert_label)
    
    def _is_likely_gene(self, text: str) -> bool:
        """Check if text is likely a gene symbol."""
        # Simple heuristics
        if len(text) < 2 or len(text) > 10:
            return False
        if not text[0].isupper():
            return False
        # Add more sophisticated checks here
        return True
    
    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with position information."""
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sentences.append({
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char
            })
        
        return sentences
    
    def _find_entity_by_text(self, entities: List[Entity], text: str) -> Optional[Entity]:
        """Find entity by text match."""
        for entity in entities:
            if entity.text.lower() == text.lower():
                return entity
        return None
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text and position."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.start, entity.end, entity.label)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated


class LiteratureProcessor(LoggerMixin):
    """Main literature processing pipeline coordinator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.pubmed_retriever = PubMedRetriever(self.config)
        self.nlp_processor = BiomedicalNLP(self.config)
    
    def process_query(
        self,
        query: str,
        max_results: int = None,
        date_range: Optional[Tuple[str, str]] = None,
        output_file: Optional[str] = None
    ) -> List[ProcessedDocument]:
        """
        Complete literature processing pipeline for a query.
        
        Args:
            query: Search query
            max_results: Maximum number of articles to process
            date_range: Date range for search
            output_file: Optional file to save results
            
        Returns:
            List of processed documents
        """
        self.logger.info(f"Starting literature processing for query: {query}")
        
        # Step 1: Search PubMed
        pmids = self.pubmed_retriever.search_pubmed(
            query, max_results, date_range
        )
        
        if not pmids:
            self.logger.warning("No articles found")
            return []
        
        # Step 2: Fetch article details
        articles = self.pubmed_retriever.fetch_article_details(pmids)
        
        # Step 3: Process each article
        processed_docs = []
        for article in tqdm(articles, desc="Processing articles"):
            try:
                doc = self.nlp_processor.process_document(article)
                processed_docs.append(doc)
            except Exception as e:
                self.logger.error(f"Error processing PMID {article['pmid']}: {e}")
                continue
        
        # Step 4: Save results if requested
        if output_file:
            self.save_results(processed_docs, output_file)
        
        self.logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def save_results(self, documents: List[ProcessedDocument], output_file: str):
        """Save processed documents to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_docs = []
        for doc in documents:
            doc_dict = asdict(doc)
            # Convert datetime to string
            doc_dict['publication_date'] = doc.publication_date.isoformat()
            serializable_docs.append(doc_dict)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_docs, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def load_results(self, input_file: str) -> List[ProcessedDocument]:
        """Load processed documents from file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        documents = []
        for doc_dict in data:
            # Convert string back to datetime
            doc_dict['publication_date'] = datetime.fromisoformat(
                doc_dict['publication_date']
            )
            
            # Convert entity and relation dicts back to objects
            entities = [Entity(**e) for e in doc_dict['entities']]
            relations = []
            for r in doc_dict['relations']:
                relations.append(Relation(
                    subject=Entity(**r['subject']),
                    predicate=r['predicate'],
                    object=Entity(**r['object']),
                    confidence=r['confidence'],
                    context=r['context'],
                    sentence=r['sentence']
                ))
            
            doc_dict['entities'] = entities
            doc_dict['relations'] = relations
            
            documents.append(ProcessedDocument(**doc_dict))
        
        return documents