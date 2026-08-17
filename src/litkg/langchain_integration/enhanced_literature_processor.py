"""
Enhanced Literature Processing with LangChain Integration

This module provides advanced document processing capabilities using LangChain,
significantly improving upon the basic literature processing in Phase 1.

Key enhancements:
1. Intelligent document loading from multiple sources
2. Biomedical-aware text chunking strategies
3. Advanced embedding and vector storage
4. Hybrid retrieval combining semantic and keyword search
5. LLM-powered content analysis and summarization
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Iterator
from pathlib import Path
from dataclasses import dataclass
import logging

# LangChain imports
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

# LangChain community imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain provider imports (conditional based on availability)
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Standard imports
import requests
import xml.etree.ElementTree as ET
from Bio import Entrez
import pandas as pd
from sentence_transformers import SentenceTransformer

# Local imports
from ..utils.config import load_config
from ..utils.logging import LoggerMixin


@dataclass
class BiomedicalDocument:
    """Enhanced document representation with LangChain compatibility."""
    content: str
    metadata: Dict[str, Any]
    pmid: Optional[str] = None
    doi: Optional[str] = None
    source: str = "pubmed"
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata={
                **self.metadata,
                "pmid": self.pmid,
                "doi": self.doi,
                "source": self.source
            }
        )


class BiomedicalDocumentLoader(BaseLoader, LoggerMixin):
    """
    Enhanced document loader for biomedical literature.
    
    Supports multiple sources:
    - PubMed Central (PMC) full-text articles
    - PubMed abstracts with enhanced metadata
    - bioRxiv preprints
    - arXiv biology papers
    - Custom biomedical databases
    """
    
    def __init__(
        self,
        source: str = "pubmed",
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        max_results: int = 100
    ):
        self.source = source
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.max_results = max_results
        
        # Set up Entrez if using PubMed
        if self.source in ["pubmed", "pmc"] and self.email:
            Entrez.email = self.email
            if self.api_key:
                Entrez.api_key = self.api_key
        
        self.logger.info(f"Initialized BiomedicalDocumentLoader for {source}")
    
    def load(self) -> List[Document]:
        """Load documents from the specified source."""
        raise NotImplementedError("Use load_from_query or load_from_ids")
    
    def load_from_query(
        self,
        query: str,
        date_range: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load documents based on a search query.
        
        Args:
            query: Search query string
            date_range: Optional (start_date, end_date) tuple
            filters: Additional filters for the search
            
        Returns:
            List of LangChain Document objects
        """
        self.logger.info(f"Loading documents for query: {query}")
        
        if self.source == "pubmed":
            return self._load_pubmed_query(query, date_range, filters)
        elif self.source == "pmc":
            return self._load_pmc_query(query, date_range, filters)
        elif self.source == "biorxiv":
            return self._load_biorxiv_query(query, date_range, filters)
        else:
            raise ValueError(f"Unsupported source: {self.source}")
    
    def load_from_ids(self, ids: List[str]) -> List[Document]:
        """
        Load documents by their IDs (PMIDs, DOIs, etc.).
        
        Args:
            ids: List of document identifiers
            
        Returns:
            List of LangChain Document objects
        """
        self.logger.info(f"Loading {len(ids)} documents by ID")
        
        if self.source == "pubmed":
            return self._load_pubmed_ids(ids)
        elif self.source == "pmc":
            return self._load_pmc_ids(ids)
        else:
            raise ValueError(f"ID loading not supported for source: {self.source}")
    
    def _load_pubmed_query(
        self,
        query: str,
        date_range: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Load PubMed documents from query."""
        try:
            # Build search term
            search_term = query
            if date_range:
                start_date, end_date = date_range
                search_term += f" AND {start_date}[PDAT]:{end_date}[PDAT]"
            
            if filters:
                for key, value in filters.items():
                    search_term += f" AND {value}[{key}]"
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=search_term,
                retmax=self.max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results["IdList"]
            self.logger.info(f"Found {len(pmids)} articles")
            
            return self._load_pubmed_ids(pmids)
            
        except Exception as e:
            self.logger.error(f"Error loading PubMed query: {e}")
            return []
    
    def _load_pubmed_ids(self, pmids: List[str]) -> List[Document]:
        """Load PubMed documents by PMIDs."""
        if not pmids:
            return []
        
        try:
            # Fetch article details
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="xml",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            documents = []
            for record in records["PubmedArticle"]:
                doc = self._parse_pubmed_record(record)
                if doc:
                    documents.append(doc.to_langchain_document())
            
            self.logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading PubMed IDs: {e}")
            return []
    
    def _parse_pubmed_record(self, record: Dict[str, Any]) -> Optional[BiomedicalDocument]:
        """Parse a PubMed record into a BiomedicalDocument."""
        try:
            article = record["MedlineCitation"]["Article"]
            
            # Extract basic information
            pmid = str(record["MedlineCitation"]["PMID"])
            title = str(article["ArticleTitle"])
            
            # Extract abstract
            abstract = ""
            if "Abstract" in article:
                abstract_parts = article["Abstract"]["AbstractText"]
                if isinstance(abstract_parts, list):
                    abstract = " ".join([str(part) for part in abstract_parts])
                else:
                    abstract = str(abstract_parts)
            
            # Combine title and abstract
            content = f"{title}\n\n{abstract}" if abstract else title
            
            # Extract metadata
            metadata = {
                "title": title,
                "abstract": abstract,
                "journal": str(article.get("Journal", {}).get("Title", "")),
                "publication_date": self._extract_date(article),
                "authors": self._extract_authors(article),
                "mesh_terms": self._extract_mesh_terms(record),
                "keywords": self._extract_keywords(article),
                "doi": self._extract_doi(article)
            }
            
            return BiomedicalDocument(
                content=content,
                metadata=metadata,
                pmid=pmid,
                doi=metadata.get("doi"),
                source="pubmed"
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing PubMed record: {e}")
            return None
    
    def _extract_date(self, article: Dict[str, Any]) -> str:
        """Extract publication date from article."""
        try:
            pub_date = article["Journal"]["JournalIssue"]["PubDate"]
            year = pub_date.get("Year", "")
            month = pub_date.get("Month", "")
            day = pub_date.get("Day", "")
            return f"{year}-{month}-{day}".strip("-")
        except:
            return ""
    
    def _extract_authors(self, article: Dict[str, Any]) -> List[str]:
        """Extract author list from article."""
        try:
            authors = []
            author_list = article.get("AuthorList", [])
            for author in author_list:
                if "LastName" in author and "ForeName" in author:
                    name = f"{author['ForeName']} {author['LastName']}"
                    authors.append(name)
            return authors
        except:
            return []
    
    def _extract_mesh_terms(self, record: Dict[str, Any]) -> List[str]:
        """Extract MeSH terms from record."""
        try:
            mesh_terms = []
            mesh_list = record["MedlineCitation"].get("MeshHeadingList", [])
            for mesh in mesh_list:
                descriptor = mesh["DescriptorName"]
                mesh_terms.append(str(descriptor))
            return mesh_terms
        except:
            return []
    
    def _extract_keywords(self, article: Dict[str, Any]) -> List[str]:
        """Extract keywords from article."""
        try:
            keywords = []
            keyword_list = article.get("KeywordList", [])
            for keyword_group in keyword_list:
                if isinstance(keyword_group, list):
                    keywords.extend([str(kw) for kw in keyword_group])
            return keywords
        except:
            return []
    
    def _extract_doi(self, article: Dict[str, Any]) -> Optional[str]:
        """Extract DOI from article."""
        try:
            elocation_id = article.get("ELocationID", [])
            if isinstance(elocation_id, list):
                for eid in elocation_id:
                    if eid.get("EIdType") == "doi":
                        return str(eid)
            elif isinstance(elocation_id, dict) and elocation_id.get("EIdType") == "doi":
                return str(elocation_id)
            return None
        except:
            return None
    
    def _load_biorxiv_query(
        self,
        query: str,
        date_range: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Load bioRxiv preprints (placeholder implementation)."""
        self.logger.warning("bioRxiv loading not yet implemented")
        return []


class BiomedicalTextSplitter(TextSplitter, LoggerMixin):
    """
    Intelligent text splitter for biomedical documents.

    Features:
    - Section-aware splitting (Abstract, Methods, Results, etc.), with the
      section name reported so retrieval can weight a Results claim above an
      Introduction one
    - Sentence-boundary preservation using scispacy, which handles the
      abbreviations ("et al.", "Fig. 1", "p < 0.05") that defeat naive splitting
    - Overlap between adjacent chunks, carried as whole sentences, so a fact
      spanning a boundary stays retrievable
    - Length measured in tokens against the embedding model's window
    """

    # Headers marking the start of a biomedical paper section. Static, so held
    # on the class rather than rebuilt per instance.
    section_patterns = [
        r"^(ABSTRACT|Abstract)",
        r"^(INTRODUCTION|Introduction)",
        r"^(METHODS|Methods|MATERIALS AND METHODS)",
        r"^(RESULTS|Results)",
        r"^(DISCUSSION|Discussion)",
        r"^(CONCLUSION|Conclusion|CONCLUSIONS)",
        r"^(REFERENCES|References)",
        r"^(ACKNOWLEDGMENTS|Acknowledgments)",
    ]

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        section_aware: bool = True,
        preserve_sentences: bool = True,
        length_unit: str = "tokens",
        model_max_tokens: int = 512,
        **kwargs
    ):
        """
        Args:
            chunk_size: Maximum chunk length, measured in ``length_unit``.
            chunk_overlap: How much of the tail of one chunk repeats at the head
                of the next, in the same unit. Prevents a fact that straddles a
                boundary from becoming unretrievable.
            section_aware: Split on biomedical section headers first.
            preserve_sentences: Never split mid-sentence.
            length_unit: "tokens" or "characters". Tokens are the unit the
                embedding model actually constrains, so sizes stay meaningful
                if the model changes.
            model_max_tokens: Context window of the embedding model, used to
                cap chunk_size when measuring in tokens.
        """
        super().__init__(**kwargs)

        if length_unit not in ("tokens", "characters"):
            raise ValueError(
                f"length_unit must be 'tokens' or 'characters', got {length_unit!r}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size}); otherwise chunking cannot advance"
            )

        self.length_unit = length_unit
        self.model_max_tokens = model_max_tokens

        # A chunk longer than the embedding window is silently truncated at
        # embedding time, so the window is a hard ceiling.
        if length_unit == "tokens" and chunk_size > model_max_tokens:
            self.logger.warning(
                f"chunk_size {chunk_size} exceeds the model window "
                f"{model_max_tokens}; capping to avoid silent truncation"
            )
            chunk_size = model_max_tokens

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_aware = section_aware
        self.preserve_sentences = preserve_sentences

    def _measure(self, text: str) -> int:
        """Length of a piece of text in the configured unit."""
        if self.length_unit == "characters":
            return len(text)

        # Whitespace tokens underestimate subword tokenization; the ~1.3
        # multiplier approximates the wordpiece expansion typical of
        # biomedical vocabulary without loading a tokenizer.
        return int(len(text.split()) * 1.3) + 1

    # Abbreviations that end in a period but do not end a sentence. Used only
    # by the regex fallback; the scispacy model handles these natively.
    _NON_TERMINAL_ABBREVIATIONS = (
        "et al.", "e.g.", "i.e.", "vs.", "cf.", "approx.", "Fig.", "Figs.",
        "Tab.", "Ref.", "Refs.", "No.", "Dr.", "Prof.", "St.", "Ltd.", "Inc.",
        "min.", "max.", "sec.", "hr.", "wt.", "ca.", "spp.", "subsp.",
    )

    @property
    def _sentence_model(self):
        """
        The scispacy sentence segmenter, loaded once on first use.

        Cached on the class so repeated splitter instances share one model.
        """
        cls = type(self)
        if hasattr(cls, "_cached_sentence_model"):
            return cls._cached_sentence_model

        model = None
        try:
            import spacy
            for model_name in ("en_core_sci_sm", "en_core_sci_md", "en_core_web_sm"):
                try:
                    model = spacy.load(model_name, disable=["ner", "lemmatizer"])
                    self.logger.info(f"Sentence splitting using {model_name}")
                    break
                except OSError:
                    continue
        except ImportError:
            pass

        if model is None:
            self.logger.warning(
                "No spaCy model available; falling back to regex sentence "
                "splitting, which mis-splits abbreviations like 'et al.'"
            )

        cls._cached_sentence_model = model
        return model

    def _split_sentences_regex(self, text: str) -> List[str]:
        """
        Regex sentence splitting that protects common abbreviations.

        A plain `(?<=[.!?])\\s+` split breaks biomedical prose at "et al.",
        "Fig. 1", and "p < 0.05.", so abbreviations are masked before splitting
        and restored afterwards.
        """
        import re

        masked = text
        placeholders = {}
        for i, abbreviation in enumerate(self._NON_TERMINAL_ABBREVIATIONS):
            token = f"\x00{i}\x00"
            placeholders[token] = abbreviation
            masked = masked.replace(abbreviation, token)

        # Also protect decimals such as "p < 0.05" and "1.5-fold". The
        # replacement is a non-raw string so \x01 is the literal sentinel
        # character rather than an escape re.sub would try to parse.
        masked = re.sub(r"(\d)\.(\d)", "\\1\x01\\2", masked)

        parts = re.split(r"(?<=[.!?])\s+", masked)

        sentences = []
        for part in parts:
            restored = part.replace("\x01", ".")
            for token, abbreviation in placeholders.items():
                restored = restored.replace(token, abbreviation)
            restored = restored.strip()
            if restored:
                sentences.append(restored)

        return sentences

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses the scispacy biomedical model when available, since it is trained
        on the abbreviation and citation patterns that defeat naive splitting,
        and falls back to an abbreviation-aware regex otherwise.
        """
        if not text or not text.strip():
            return []

        model = self._sentence_model
        if model is not None:
            try:
                return [
                    sentence.text.strip()
                    for sentence in model(text).sents
                    if sentence.text.strip()
                ]
            except Exception as e:
                self.logger.warning(f"spaCy sentence splitting failed ({e}); using regex")

        return self._split_sentences_regex(text)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using biomedical-aware strategies."""
        if self.section_aware:
            return self._split_by_sections(text)
        else:
            return self._split_by_sentences(text)

    def split_text_with_sections(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Split text, reporting which paper section each chunk came from.

        Section provenance is evidential weight in biomedical text: a claim in
        Results is a finding this paper established, while the same sentence in
        Introduction is background attributed to someone else. Callers can
        weight or filter retrieval on it.

        Returns:
            (chunk_text, section_name) pairs. section_name is None for text
            appearing before any recognized header.
        """
        if not self.section_aware:
            return [(chunk, None) for chunk in self._split_by_sentences(text)]

        labeled: List[Tuple[str, Optional[str]]] = []
        for section_name, section_text in self._iter_sections(text):
            for chunk in self._split_by_sentences(section_text):
                labeled.append((chunk, section_name))

        return labeled

    def _iter_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """
        Break text into (section_name, section_text) pairs on header lines.

        Text before the first recognized header is returned under None rather
        than being dropped or misattributed to the first section.
        """
        import re

        sections: List[Tuple[Optional[str], str]] = []
        current_name: Optional[str] = None
        current_lines: List[str] = []

        for line in text.split('\n'):
            stripped = line.strip()
            header = next(
                (
                    re.match(pattern, stripped, re.IGNORECASE)
                    for pattern in self.section_patterns
                    if re.match(pattern, stripped, re.IGNORECASE)
                ),
                None,
            )

            if header:
                if current_lines:
                    sections.append((current_name, '\n'.join(current_lines)))
                current_name = header.group(1).title()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_name, '\n'.join(current_lines)))

        return sections
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by biomedical paper sections."""
        import re
        
        chunks = []
        lines = text.split('\n')
        current_section = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            # Check if this is a section header
            is_section_header = any(
                re.match(pattern, line.strip(), re.IGNORECASE)
                for pattern in self.section_patterns
            )
            
            # If we hit a section header and have content, finalize current section
            if is_section_header and current_section and current_size > 0:
                section_text = '\n'.join(current_section)
                if self.preserve_sentences:
                    section_chunks = self._split_by_sentences(section_text)
                    chunks.extend(section_chunks)
                else:
                    chunks.append(section_text)
                
                current_section = [line]
                current_size = line_size
            
            # Add line to current section
            elif current_size + line_size <= self.chunk_size:
                current_section.append(line)
                current_size += line_size
            
            # Section too large, split it
            else:
                if current_section:
                    section_text = '\n'.join(current_section)
                    if self.preserve_sentences:
                        section_chunks = self._split_by_sentences(section_text)
                        chunks.extend(section_chunks)
                    else:
                        chunks.append(section_text)
                
                current_section = [line]
                current_size = line_size
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section)
            if self.preserve_sentences:
                section_chunks = self._split_by_sentences(section_text)
                chunks.extend(section_chunks)
            else:
                chunks.append(section_text)
        
        return chunks
    
    def _overlap_tail(self, sentences: List[str]) -> List[str]:
        """
        Take the trailing sentences that fit within chunk_overlap.

        Overlap is carried as whole sentences rather than a raw character
        slice, so a repeated fragment is still readable on its own and still
        embeds meaningfully.
        """
        if self.chunk_overlap <= 0 or not sentences:
            return []

        tail: List[str] = []
        size = 0
        for sentence in reversed(sentences):
            sentence_size = self._measure(sentence)
            if size + sentence_size > self.chunk_overlap:
                break
            tail.insert(0, sentence)
            size += sentence_size

        # Always carry at least the final sentence when overlap is requested,
        # since that is where a boundary-straddling fact is cut.
        if not tail:
            tail = [sentences[-1]]

        return tail

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks on sentence boundaries, with overlap.

        Consecutive chunks share their boundary sentences so that a claim
        spanning a split remains retrievable from at least one chunk.
        """
        sentences = self.split_sentences(text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self._measure(sentence)

            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
                continue

            # Finalize current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                carried = self._overlap_tail(current_chunk)
            else:
                carried = []

            if sentence_size <= self.chunk_size:
                # Begin the next chunk with the carried overlap
                current_chunk = [*carried, sentence]
                current_size = sum(self._measure(s) for s in current_chunk)
            else:
                # Sentence alone exceeds the budget; split it by words
                chunks.extend(self._split_large_sentence(sentence))
                current_chunk = []
                current_size = 0

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
    
    def _split_large_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that's too large into word-based chunks."""
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = self._measure(word)

            if current_size + word_size <= self.chunk_size:
                current_chunk.append(word)
                current_size += word_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class BiomedicaEmbeddings(Embeddings, LoggerMixin):
    """
    Biomedical-optimized embeddings with multiple model support.
    
    Supports:
    - Domain-specific models (PubMedBERT, BioBERT, ClinicalBERT)
    - General-purpose models (OpenAI, Sentence Transformers)
    - Hybrid embeddings combining multiple approaches
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        use_openai: bool = False,
        openai_model: str = "text-embedding-ada-002",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.device = device
        
        # Initialize embeddings
        if use_openai and OPENAI_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(model=openai_model)
            self.logger.info(f"Using OpenAI embeddings: {openai_model}")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device}
            )
            self.logger.info(f"Using HuggingFace embeddings: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(text)


class LangChainLiteratureProcessor(LoggerMixin):
    """
    Enhanced literature processor using LangChain components.
    
    This replaces the basic literature processor with advanced capabilities:
    - Multiple document sources
    - Intelligent text chunking
    - Vector storage and retrieval
    - LLM-powered analysis
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        vector_store_type: str = "chroma",
        embeddings_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    ):
        self.config = load_config(config_path) if config_path else None
        self.vector_store_type = vector_store_type
        
        # Initialize components
        self.document_loader = BiomedicalDocumentLoader()
        self.text_splitter = BiomedicalTextSplitter()
        self.embeddings = BiomedicaEmbeddings(model_name=embeddings_model)
        self.vector_store = None
        
        self.logger.info("Initialized LangChainLiteratureProcessor")
    
    def process_query(
        self,
        query: str,
        max_results: int = 100,
        create_vector_store: bool = True,
        store_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a literature query with enhanced LangChain capabilities.
        
        Args:
            query: Search query
            max_results: Maximum documents to retrieve
            create_vector_store: Whether to create a vector store
            store_path: Path to save/load vector store
            
        Returns:
            Dictionary with processed results and vector store
        """
        self.logger.info(f"Processing literature query: {query}")
        
        # Step 1: Load documents
        documents = self.document_loader.load_from_query(query, max_results=max_results)
        self.logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            return {"documents": [], "vector_store": None, "chunks": []}
        
        # Step 2: Split documents into chunks
        all_chunks = []
        for doc in documents:
            # Section provenance travels with the chunk so retrieval can weight
            # a Results claim differently from an Introduction one.
            labeled = self.text_splitter.split_text_with_sections(doc.page_content)
            for i, (chunk, section) in enumerate(labeled):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(labeled),
                        "section": section,
                    }
                )
                all_chunks.append(chunk_doc)
        
        self.logger.info(f"Created {len(all_chunks)} text chunks")
        
        # Step 3: Create vector store if requested
        vector_store = None
        if create_vector_store:
            vector_store = self._create_vector_store(all_chunks, store_path)
        
        return {
            "documents": documents,
            "chunks": all_chunks,
            "vector_store": vector_store,
            "query": query,
            "num_documents": len(documents),
            "num_chunks": len(all_chunks)
        }
    
    def _create_vector_store(
        self,
        documents: List[Document],
        store_path: Optional[str] = None
    ) -> Union[Chroma, FAISS]:
        """Create and populate a vector store."""
        self.logger.info(f"Creating {self.vector_store_type} vector store")
        
        if self.vector_store_type == "chroma":
            if store_path:
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=store_path
                )
                vector_store.persist()
            else:
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
        
        elif self.vector_store_type == "faiss":
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            if store_path:
                vector_store.save_local(store_path)
        
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        self.vector_store = vector_store
        self.logger.info(f"Created vector store with {len(documents)} documents")
        
        return vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Perform similarity search on the vector store."""
        if not self.vector_store:
            raise ValueError("No vector store available. Run process_query first.")
        
        if score_threshold:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            filtered_results = [
                doc for doc, score in results 
                if score >= score_threshold
            ]
            return filtered_results
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """Get a retriever for the vector store."""
        if not self.vector_store:
            raise ValueError("No vector store available. Run process_query first.")
        
        search_kwargs = search_kwargs or {"k": 5}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )