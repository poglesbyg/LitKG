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
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

import spacy
import torch
# Only the NER pipeline is built here; `pipeline` resolves the tokenizer and
# the token-classification model from the configured checkpoint itself.
from transformers import pipeline
from Bio import Entrez
import requests
import pandas as pd
from tqdm import tqdm

from litkg.utils.config import LitKGConfig, load_config
from litkg.utils.logging import LoggerMixin


# transformers names the classes of a randomly initialized classification head
# LABEL_0, LABEL_1, ... A checkpoint whose id2label looks like that was never
# fine-tuned for token classification, whatever its model card says about being
# biomedical.
UNTRAINED_LABEL_PATTERN = re.compile(r"^LABEL_\d+$")


def is_untrained_token_classifier(id2label: Dict[Any, str]) -> bool:
    """
    True when a token-classification head carries no trained label scheme.

    `dmis-lab/biobert-base-cased-v1.1` is a base language model. Building an
    NER pipeline on it makes transformers initialize a classifier head at
    random -- it warns, then runs, then emits LABEL_0/LABEL_1 spans at ~0.5
    confidence that no label map can turn into an entity type. Detecting that
    at load time is the difference between a loud misconfiguration and a
    pipeline stage that silently returns nothing on every document.
    """
    labels = list(id2label.values()) if id2label else []
    if not labels:
        return True
    return all(UNTRAINED_LABEL_PATTERN.match(str(label)) for label in labels)


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
    
    # Specialized scispacy NER models, in order of preference. en_core_sci_*
    # emits a single "ENTITY" label and cannot type anything, so relying on it
    # alone leaves the rule-based path as the only source of entity types.
    NER_MODELS = ("en_ner_bionlp13cg_md", "en_ner_bc5cdr_md")

    # Input budget for the BERT NER pipeline. 512 is the BERT-family default
    # and the last resort when neither the tokenizer nor the model config
    # reports a usable one; anything above MAX_PLAUSIBLE_TOKENS is the
    # `model_max_length` sentinel transformers reports for "unset", not a limit.
    DEFAULT_MAX_TOKENS = 512
    MAX_PLAUSIBLE_TOKENS = 100_000

    # Fallback for configs written before `biomedical_ner` existed.
    DEFAULT_BERT_NER_MODEL = "alvaroalon2/biobert_genetic_ner"

    # Labels emitted by the BERT NER checkpoint, mapped onto this project's
    # vocabulary. JNLPBA/BC2GM merge gene, gene product and protein mentions
    # into one class, which is the same thing bionlp13cg calls
    # GENE_OR_GENE_PRODUCT, so it lands on GENE.
    BERT_NER_LABEL_MAP = {
        "GENETIC": "GENE",
        "GENE": "GENE",
        "PROTEIN": "PROTEIN",
        "DNA": "GENE",
        "RNA": "GENE",
        "CELL_TYPE": "CELL_TYPE",
        "CELL_LINE": "CELL_TYPE",
        "DISEASE": "DISEASE",
        "CHEMICAL": "CHEMICAL",
        "DRUG": "DRUG",
    }

    # Model labels mapped onto this project's entity vocabulary
    NER_LABEL_MAP = {
        "GENE_OR_GENE_PRODUCT": "GENE",
        "SIMPLE_CHEMICAL": "CHEMICAL",
        "CHEMICAL": "CHEMICAL",
        "DISEASE": "DISEASE",
        "CANCER": "DISEASE",
        "CELL": "CELL_TYPE",
        "CELLULAR_COMPONENT": "CELL_TYPE",
        "TISSUE": "TISSUE",
        "ORGAN": "TISSUE",
        "ANATOMICAL_SYSTEM": "TISSUE",
        "MULTI_TISSUE_STRUCTURE": "TISSUE",
        "ORGANISM": "ORGANISM",
        "AMINO_ACID": "CHEMICAL",
    }

    # All-caps acronyms the gene regex would otherwise claim. These are the
    # observed offenders in the sample corpus: disease abbreviations, outcome
    # measures, therapy classes, study designs and plain molecules. A gene
    # regex of the form [A-Z][A-Z0-9]{2,10} matches every one of them.
    NON_GENE_ACRONYMS = frozenset({
        # Diseases and conditions
        "ALL", "AML", "CLL", "CML", "NSCLC", "SCLC", "TNBC", "HCC", "CRC",
        "GBM", "DLBCL", "MDS", "MM", "RCC", "HNSCC", "COPD", "AIDS", "HIV",
        # Outcome measures and statistics
        "PFS", "OS", "ORR", "DFS", "RFS", "TTP", "DCR", "CR", "PR", "SD", "PD",
        "HR", "CI", "AUC", "ROC", "IQR", "SEM", "ANOVA",
        # Therapies and modalities
        "CAR", "ICI", "ICB", "TKI", "PARP", "ADC", "SOC", "CHT", "RT",
        "CTLA", "IMRT", "SBRT",
        # Molecules, methods and general terms
        "DNA", "RNA", "MRNA", "CDNA", "PCR", "QPCR", "RTPCR", "NGS", "WES",
        "WGS", "IHC", "FISH", "ELISA", "MRI", "PET", "CT", "US", "FDA", "EMA",
        "NCI", "WHO", "NIH", "USA", "UK", "EU",
        # Study and trial vocabulary
        "RCT", "ITT", "QOL", "AE", "SAE", "MTD", "DLT", "PK", "PD",
    })

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
        
        # Specialized NER models. Loaded lazily on first use.
        self._ner_pipelines = None
        self._gene_vocabulary = None

        # Load scispacy model
        try:
            self.nlp = spacy.load(self.models_config["scispacy_model"])
            self.logger.info("Loaded scispacy model")
        except OSError:
            self.logger.error("scispacy model not found. Please install it first.")
            raise
        
        # PubMedBERT and BioBERT encoders were downloaded and held here as
        # `self.pubmedbert_model` / `self.biobert_model` and never read by
        # anything -- ~800MB and two model loads per run, for nothing. The
        # encoders that are actually used live elsewhere and pick their own
        # checkpoints: `litkg.phase2.node_features` for text features and
        # `litkg.models.huggingface_models.ModelRegistry` for embeddings.

        # Create the BERT NER pipeline
        self.ner_pipeline = None
        model_name = self.models_config.get(
            "biomedical_ner", self.DEFAULT_BERT_NER_MODEL
        )
        try:
            candidate = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            self.logger.error(f"Error creating NER pipeline: {e}")
        else:
            id2label = getattr(candidate.model.config, "id2label", {}) or {}
            if is_untrained_token_classifier(id2label):
                # transformers happily builds a token-classification pipeline
                # on top of a checkpoint that has no classifier head: it
                # initializes one at random and warns. The pipeline then runs,
                # returns LABEL_0/LABEL_1 spans at ~0.5 confidence, and every
                # one of them is dropped for not being a known entity type --
                # an extraction path that looks live and contributes nothing.
                self.logger.error(
                    f"{model_name} has no trained token-classification head "
                    f"(labels: {sorted(id2label.values())}). Disabling the BERT "
                    "NER path; configure phase1.literature.models.biomedical_ner "
                    "with a fine-tuned biomedical NER checkpoint."
                )
            else:
                self.ner_pipeline = candidate
                self.logger.info(f"Created NER pipeline from {model_name}")
    
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
        
        # Method 2: BERT-based NER. The scispacy spans go in as already-claimed
        # so the two models cannot each contribute a mention of the same text
        # ("MMP-9" and "serum MMP-9" are one entity, not two).
        claimed = [(e.start, e.end) for e in entities]
        entities.extend(self._extract_entities_bert(text, claimed))
        
        # Method 3: Rule-based patterns
        entities.extend(self._extract_entities_rules(text))
        
        # Deduplicate and filter
        entities = self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= self.text_config["min_entity_confidence"]]
        
        return entities
    
    @property
    def ner_pipelines(self) -> List[Any]:
        """
        Specialized NER models, loaded once on first use.

        The general en_core_sci_* model emits a single "ENTITY" label, which is
        not in ``entity_types``, so it contributed nothing and left the gene
        regex as the only source of entity types -- which is why every entity
        in the corpus came out typed GENE.
        """
        if self._ner_pipelines is not None:
            return self._ner_pipelines

        self._ner_pipelines = []
        for model_name in self.NER_MODELS:
            try:
                self._ner_pipelines.append(spacy.load(model_name))
                self.logger.info(f"Loaded NER model {model_name}")
            except OSError:
                self.logger.warning(
                    f"NER model {model_name} not installed; entity typing will be "
                    "less accurate. Run scripts/setup_models.py"
                )

        if not self._ner_pipelines:
            self.logger.warning(
                "No specialized NER model available; falling back to the general "
                "model, which cannot distinguish entity types"
            )

        return self._ner_pipelines

    def _extract_entities_scispacy(self, text: str) -> List[Entity]:
        """
        Extract typed entities using the specialized biomedical NER models.

        Each model contributes the types it was trained for -- bionlp13cg for
        genes, cancers, cells and tissues; bc5cdr for diseases and chemicals --
        and their labels are mapped onto this project's vocabulary.
        """
        entities = []
        # Both models read the same text, so the same span is often tagged
        # twice with different labels -- bc5cdr calls CHEK2 a DISEASE where
        # bionlp13cg calls it a GENE. Two mentions at one span corrupt mention
        # counts and, because downstream maps key on (document, start, end),
        # silently route a link onto whichever entity was written last.
        # NER_MODELS is ordered by preference; the first model to claim a span
        # keeps it.
        claimed: Dict[Tuple[int, int], str] = {}

        for pipeline in self.ner_pipelines:
            try:
                doc = pipeline(text)
            except Exception as e:
                self.logger.warning(f"NER model failed on this text: {e}")
                continue

            for ent in doc.ents:
                label = self.NER_LABEL_MAP.get(ent.label_, ent.label_)
                if label not in self.entity_types:
                    continue

                span = (ent.start_char, ent.end_char)
                if span in claimed:
                    continue
                claimed[span] = label

                entities.append(Entity(
                    text=ent.text,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                    # scispacy does not expose per-entity confidence; a trained
                    # model is still worth more than the regex fallback
                    confidence=0.85,
                    cui=ent._.cui if hasattr(ent._, "cui") else None,
                ))

        # Fall back to the general model only when no specialized model loaded
        if not self.ner_pipelines:
            for ent in self.nlp(text).ents:
                label = self.NER_LABEL_MAP.get(ent.label_, ent.label_)
                if label in self.entity_types:
                    entities.append(Entity(
                        text=ent.text, label=label,
                        start=ent.start_char, end=ent.end_char,
                        confidence=0.6,
                    ))

        return entities
    
    def _bert_max_tokens(self) -> int:
        """
        Number of tokens one call to the BERT NER pipeline may carry.

        `model_max_length` cannot be trusted on its own: a tokenizer whose
        config omits a length -- biobert-base-cased-v1.1 among them -- reports a
        sentinel near 1e30, and that sentinel is exactly why the pipeline's own
        truncation flag never fired here. The model's position embedding table
        is the constraint that actually raises, so read both and believe
        whichever limit is smaller and plausible.
        """
        model_config = getattr(getattr(self.ner_pipeline, "model", None), "config", None)
        candidates = [
            getattr(self.ner_pipeline.tokenizer, "model_max_length", None),
            getattr(model_config, "max_position_embeddings", None),
        ]
        usable = [
            candidate for candidate in candidates
            if isinstance(candidate, int) and 0 < candidate <= self.MAX_PLAUSIBLE_TOKENS
        ]
        return min(usable) if usable else self.DEFAULT_MAX_TOKENS

    def _truncate_to_token_limit(self, text: str) -> str:
        """
        Cut `text` at the last whole word that fits the NER model's window.

        The result is always a prefix of `text`, so the character offsets the
        pipeline reports still index the caller's string.
        """
        tokenizer = self.ner_pipeline.tokenizer
        limit = self._bert_max_tokens()

        if getattr(tokenizer, "is_fast", False):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=limit,
                return_offsets_mapping=True,
            )
            if len(encoded["input_ids"]) < limit:
                return text

            # Special tokens ([CLS]/[SEP]) carry an empty span; the last real
            # span ends where the model stops reading.
            spans = [span for span in encoded["offset_mapping"] if span[1] > span[0]]
            if not spans:
                return text
            cut = spans[-1][1]
        else:
            # Slow tokenizers cannot report character offsets. Grow a prefix a
            # word at a time instead: wordpiece never merges across whitespace,
            # so per-word counts sum to the count for the whole prefix.
            budget = limit - tokenizer.num_special_tokens_to_add()
            cut = 0
            for word in re.finditer(r"\S+", text):
                budget -= len(tokenizer.tokenize(word.group()))
                if budget < 0:
                    break
                cut = word.end()
            else:
                return text

        # A kept token can end mid-word, and re-tokenizing half a word is not
        # guaranteed to yield the same number of pieces -- greedy longest-match
        # can split a prefix more finely than the whole word -- so cut back to
        # the preceding whitespace. Whole words tokenize identically in a
        # prefix, which is what keeps the truncated text inside the limit.
        if cut < len(text) and not text[cut].isspace():
            boundary = re.search(r"\s\S*$", text[:cut])
            if boundary is not None:
                cut = boundary.start()
        return text[:cut]

    def _extract_entities_bert(
        self,
        text: str,
        claimed: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Entity]:
        """
        Extract entities with the fine-tuned biomedical NER checkpoint.

        Adds the gene and protein mentions the scispacy models and the
        vocabulary-gated gene regex both miss: mixed-case symbols (Gsalpha,
        SetD5, apoE4, cullin-2), mouse allele notation (Fgfr2(+/S252W)) and
        modified-residue names (H3K27me3). Measured on 150 abstracts sampled
        from data/processed/literature_context: 247 entities over 182 distinct
        surface forms that neither the scispacy models nor the rules produced,
        against 4906 entities from those two paths combined.

        Input is truncated on tokens, not characters -- see
        `_truncate_to_token_limit`. The result is a prefix of `text`, so the
        offsets the pipeline reports still index the caller's string.

        Args:
            text: Input text
            claimed: Character spans another extractor has already taken. A
                BERT span overlapping one of them is dropped rather than
                emitted as a second mention of the same text.
        """
        entities = []

        if self.ner_pipeline is None:
            # No usable checkpoint; _load_models already said so.
            return entities

        occupied = list(claimed or [])

        try:
            results = self.ner_pipeline(self._truncate_to_token_limit(text))
            
            for result in results:
                start, end = result["start"], result["end"]
                surface = text[start:end]

                # Map BERT labels to our entity types
                label = self._map_bert_label(result["entity_group"])
                if label not in self.entity_types:
                    continue

                # Subword aggregation cuts words in half ("isplatin",
                # "arubicin"): 5.7% of spans on the sampled abstracts. A span
                # that starts or ends inside a word is not an entity mention.
                if not self._is_whole_word_span(text, start, end):
                    continue

                # The checkpoint tags disease and outcome acronyms as genes
                # (CAR, MDS); the rule-based path already knows those.
                if label == "GENE" and surface.upper() in self.NON_GENE_ACRONYMS:
                    continue

                if any(start < o_end and o_start < end for o_start, o_end in occupied):
                    continue
                occupied.append((start, end))

                entities.append(Entity(
                    text=surface,
                    label=label,
                    start=start,
                    end=end,
                    confidence=float(result["score"]),
                ))
        
        except Exception as e:
            self.logger.error(f"Error in BERT NER: {e}")
        
        return entities

    @staticmethod
    def _is_whole_word_span(text: str, start: int, end: int) -> bool:
        """True when a span does not begin or end in the middle of a word."""
        if start >= end:
            return False
        starts_mid_word = start > 0 and text[start - 1].isalnum() and text[start].isalnum()
        ends_mid_word = end < len(text) and text[end - 1].isalnum() and text[end].isalnum()
        return not (starts_mid_word or ends_mid_word)

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
                        sentence['text'], sentence_entities, sentence['start']
                    )
                )
        
        return relations
    
    # Trigger phrases mapped to a relation type and whether the phrase reverses
    # the reading. "A treats B" is forward; "A treated with B" means B treats A.
    RELATION_TRIGGERS = [
        # (trigger phrase, relation type, reverses subject/object)
        ("treated with", "TREATS", True),
        ("treatment with", "TREATS", True),
        ("responds to", "TREATS", True),
        ("sensitive to", "SENSITIZES_TO", False),
        ("resistant to", "RESISTANT_TO", False),
        ("treats", "TREATS", False),
        ("therapy for", "TREATS", False),
        ("effective against", "TREATS", False),
        ("inhibits", "INHIBITS", False),
        ("suppresses", "INHIBITS", False),
        ("blocks", "INHIBITS", False),
        ("downregulates", "INHIBITS", False),
        ("activates", "ACTIVATES", False),
        ("upregulates", "ACTIVATES", False),
        ("promotes", "ACTIVATES", False),
        ("induces", "CAUSES", False),
        ("causes", "CAUSES", False),
        ("leads to", "CAUSES", False),
        ("results in", "CAUSES", False),
        ("drives", "CAUSES", False),
        ("contributes to", "CAUSES", False),
        ("mutated in", "MUTATED_IN", False),
        ("mutations in", "MUTATED_IN", True),
        ("expressed in", "EXPRESSED_IN", False),
        ("overexpressed in", "EXPRESSED_IN", False),
        ("interacts with", "INTERACTS_WITH", False),
        ("binds", "INTERACTS_WITH", False),
        ("associated with", "ASSOCIATED_WITH", False),
        ("correlated with", "ASSOCIATED_WITH", False),
        ("linked to", "ASSOCIATED_WITH", False),
        ("implicated in", "ASSOCIATED_WITH", False),
        ("involved in", "ASSOCIATED_WITH", False),
        ("predicts", "PREDICTS", False),
        ("biomarker for", "PREDICTS", False),
    ]

    # A trigger inside a negated span asserts the opposite of what it names
    NEGATION_CUES = (
        "not", "no ", "never", "without", "failed to", "did not", "does not",
        "lack of", "absence of", "unable to", "rather than",
    )

    # Beyond this many characters apart, two entities in one sentence are
    # usually not the pair the trigger relates.
    MAX_TRIGGER_SPAN_CHARS = 120

    # Entity pairs considered per sentence. Pairing is quadratic, and sentences
    # listing many entities are typically enumerations rather than assertions.
    MAX_PAIRS_PER_SENTENCE = 60

    def _extract_relations_from_sentence(
        self, sentence: str, entities: List[Entity], sentence_start: int = 0
    ) -> List[Relation]:
        """
        Extract relations by looking for trigger phrases between entity pairs.

        The earlier implementation matched regexes whose capture groups had to
        coincide with entity text, which required both entities to be single
        words directly adjacent to the trigger verb. Real prose almost never
        obliges: "BRCA1 mutations are associated with breast cancer" captured
        ("are", "breast"), neither of which is an entity, so every candidate was
        discarded. Working from the span *between* two known entities removes
        that coupling entirely.

        Args:
            sentence: The sentence text.
            entities: Entities occurring in this sentence.
            sentence_start: Character offset of the sentence within the document,
                used to convert entity offsets into sentence-local ones.

        Returns:
            Relations asserted in this sentence.
        """
        relations: List[Relation] = []

        ordered = sorted(entities, key=lambda e: e.start)
        pairs_examined = 0

        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                if pairs_examined >= self.MAX_PAIRS_PER_SENTENCE:
                    return relations
                pairs_examined += 1

                # An entity cannot relate to itself, and overlapping spans are
                # usually one entity the tagger split
                if first.text.lower() == second.text.lower():
                    continue

                between_start = max(0, first.end - sentence_start)
                between_end = max(between_start, second.start - sentence_start)
                between = sentence[between_start:between_end].lower()

                if not between.strip() or len(between) > self.MAX_TRIGGER_SPAN_CHARS:
                    continue

                if any(cue in between for cue in self.NEGATION_CUES):
                    continue

                # Longest trigger first so "treated with" wins over "treats"
                match = next(
                    (
                        (phrase, relation, reverse)
                        for phrase, relation, reverse in sorted(
                            self.RELATION_TRIGGERS, key=lambda t: -len(t[0])
                        )
                        if phrase in between
                    ),
                    None,
                )
                if match is None:
                    continue

                phrase, relation_type, reverse = match
                subject, obj = (second, first) if reverse else (first, second)

                relations.append(Relation(
                    subject=subject,
                    predicate=relation_type,
                    object=obj,
                    # Nearby triggers are more reliable than distant ones
                    confidence=0.75 if len(between) <= 40 else 0.55,
                    context=phrase,
                    sentence=sentence,
                ))

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
        """
        Map the BERT NER checkpoint's labels onto this project's entity types.

        This used to map the CoNLL-2003 scheme (PER/ORG/LOC/MISC), which no
        biomedical checkpoint emits and none of which are entity types this
        processor keeps, so it was a no-op sitting in front of a path that was
        already returning nothing.
        """
        label = bert_label.upper().replace("-", "_").replace(" ", "_")
        # Strip a BIO prefix in case the pipeline is run without aggregation.
        if label[:2] in ("B_", "I_", "E_", "S_"):
            label = label[2:]
        return self.BERT_NER_LABEL_MAP.get(label, label)
    
    @property
    def gene_vocabulary(self) -> set:
        """
        Known gene symbols, loaded once from the curated sources on disk.

        Drawn from the CIVIC gene list and the seed ontology. Used to keep the
        rule-based extractor honest: an all-caps token is only called a gene if
        something authoritative says it is one.
        """
        if self._gene_vocabulary is not None:
            return self._gene_vocabulary

        vocabulary = set()

        try:
            from litkg.utils.config import get_data_dir
            civic_genes = get_data_dir() / "external" / "civic" / "civic_genes.tsv"
            if civic_genes.exists():
                frame = pd.read_csv(civic_genes, sep="\t")
                column = "name" if "name" in frame.columns else "gene"
                vocabulary |= {
                    str(v).strip().upper() for v in frame[column].dropna()
                }

            ontology = get_data_dir() / "ontologies" / "biomedical_seed.json"
            if ontology.exists():
                with open(ontology) as handle:
                    for name, record in json.load(handle).items():
                        if str(record.get("type", "")).upper() == "GENE":
                            vocabulary.add(name.strip().upper())
                            vocabulary |= {
                                str(s).strip().upper()
                                for s in record.get("synonyms", [])
                            }
        except Exception as e:
            self.logger.warning(f"Could not load gene vocabulary: {e}")

        self._gene_vocabulary = vocabulary
        self.logger.info(f"Gene vocabulary: {len(vocabulary)} symbols")
        return vocabulary

    def _is_likely_gene(self, text: str) -> bool:
        """
        Decide whether an all-caps token is really a gene symbol.

        The previous version accepted anything 2-10 characters starting with a
        capital, so the gene regex ``[A-Z][A-Z0-9]{2,10}`` claimed every
        acronym in the corpus: ALL and NSCLC (diseases), PFS (an outcome
        measure), CAR and ICI (therapy classes), DNA and PCR. That is why every
        extracted relation had GENE->GENE endpoints.

        Now a token must be in the curated gene vocabulary, and known non-gene
        acronyms are rejected outright. The specialized NER models cover genes
        the vocabulary has not heard of.
        """
        candidate = text.strip().upper()

        if not 2 <= len(candidate) <= 10:
            return False
        if candidate in self.NON_GENE_ACRONYMS:
            return False

        return candidate in self.gene_vocabulary
    
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
    
    def __init__(self, config_path: Optional[Union[str, Dict[str, Any], LitKGConfig]] = None):
        # Allow dict configs in tests; fall back to full config loader otherwise
        if isinstance(config_path, dict):
            self.config = config_path
            self.pubmed_retriever = None
            self.nlp_processor = None
            # Simple tokenizer; tests patch `nlp` when needed
            try:
                self.nlp = spacy.blank("en")
            except Exception:
                self.nlp = None
        else:
            self.config = load_config(config_path)
            self.pubmed_retriever = PubMedRetriever(self.config)
            self.nlp_processor = BiomedicalNLP(self.config)
            self.nlp = self.nlp_processor.nlp
    
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

    # ------------------- Lightweight API -------------------
    @staticmethod
    def _validate_article(article_data: Dict[str, Any]) -> None:
        """
        Check that a structured article carries processable text.

        Raises:
            ValueError: if the record has neither a title nor an abstract.
                Failing loudly beats returning an empty result, which would
                silently drop malformed records from a batch.
        """
        if not isinstance(article_data, dict):
            raise ValueError(
                f"Expected a document dict or raw text, got {type(article_data).__name__}"
            )

        has_text = any(
            str(article_data.get(field) or "").strip() for field in ("title", "abstract")
        )
        if not has_text:
            raise ValueError(
                "Malformed document: needs a non-empty 'title' or 'abstract'; "
                f"got keys {sorted(article_data.keys())}"
            )

    def process_document(self, article_data: Union[Dict[str, Any], str]) -> Any:
        """Process a single document.
        - If provided a raw text string, return a simple dict with entities and text.
        - If provided a structured dict, run the rich pipeline via BiomedicalNLP.

        Raises:
            ValueError: if a structured document has no title or abstract.
        """
        # Raw text compatibility path
        if isinstance(article_data, str):
            text = article_data
            entities: List[Dict[str, Any]] = []
            try:
                if self.nlp is not None:
                    doc = self.nlp(text)
                    for ent in getattr(doc, 'ents', []):
                        entities.append({
                            "text": getattr(ent, "text", ""),
                            "label": getattr(ent, "label_", "ENTITY"),
                            "start": getattr(ent, "start", 0),
                            "end": getattr(ent, "end", 0),
                        })
            except Exception:
                entities = []
            return {"entities": entities, "relations": [], "text": text}

        # Structured article path (when full pipeline is available)
        self._validate_article(article_data)

        if self.nlp_processor is None:
            # Minimal fallback using raw text fields
            title = article_data.get("title", "")
            abstract = article_data.get("abstract", "")
            text = f"{title} {abstract}".strip()
            return {"entities": [], "relations": [], "text": text}

        self.logger.info(f"Processing document PMID: {article_data.get('pmid', 'N/A')}")
        full_text = f"{article_data['title']} {article_data['abstract']}"
        entities = [asdict(e) for e in self.nlp_processor.extract_entities(full_text)]
        relations = []
        # Convert Relation dataclasses to dicts if any
        for r in self.nlp_processor.extract_relations(full_text, [Entity(**e) for e in entities]):
            relations.append({
                "subject": asdict(r.subject),
                "predicate": r.predicate,
                "object": asdict(r.object),
                "confidence": r.confidence,
                "context": r.context,
                "sentence": r.sentence,
            })
        return {
            "pmid": article_data.get("pmid"),
            "title": article_data.get("title"),
            "abstract": article_data.get("abstract"),
            "entities": entities,
            "relations": relations,
            "text": full_text,
        }

    def process_batch(self, documents: List[Union[Dict[str, Any], str]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for doc in documents:
            if isinstance(doc, (str, dict)):
                processed = self.process_document(doc)
                if not isinstance(processed, dict):
                    raise Exception("Malformed processed document")
                results.append(processed)
            else:
                raise Exception("Malformed input document")
        return results

    # Hooks that tests patch
    def _extract_entities_with_model(self, text: str) -> List[Dict[str, Any]]:
        return []

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return self._extract_entities_with_model(text)

    def _extract_relations_with_model(self, text: str, entities: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        return []

    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        return self._extract_relations_with_model(text)


# ------------------- Utilities expected by tests -------------------

class DocumentProcessor(LoggerMixin):
    """Utility text processor with basic cleaning and tokenization."""

    def clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+[\w-]*", text)


class EntityExtractor(LoggerMixin):
    """Lightweight entity extractor wrapper used in tests."""

    def __init__(self):
        self.ner_model = lambda x: []  # patched in tests

    def extract_biomedical_entities(self, text: str) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        for item in self.ner_model(text):
            word = item.get("word") or item.get("text") or ""
            label = item.get("entity") or item.get("label") or "ENTITY"
            if "-" in label:
                label = label.split("-")[-1]
            entities.append({
                "text": word,
                "label": label,
                "confidence": float(item.get("confidence", item.get("score", 0.0))),
                "start": int(item.get("start", 0)),
                "end": int(item.get("end", 0)),
            })
        return entities

    def normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in entities:
            e2 = dict(e)
            e2["normalized_text"] = e2.get("text", "").lower()
            out.append(e2)
        return out

    def filter_entities(self, entities: List[Dict[str, Any]], min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        return [e for e in entities if float(e.get("confidence", 0.0)) >= min_confidence]
    
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