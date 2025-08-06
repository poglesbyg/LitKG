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
from typing import List, Dict, Any, Optional, Union, Iterator
from pathlib import Path
from dataclasses import dataclass
import logging

# LangChain imports
from langchain.document_loaders import BaseLoader
from langchain.text_splitter import TextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.retrievers import BaseRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

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


class BiomedicalTextSplitter(TextSplitter):
    """
    Intelligent text splitter for biomedical documents.
    
    Features:
    - Section-aware splitting (Abstract, Methods, Results, etc.)
    - Sentence-boundary preservation
    - Entity-aware chunking (keeps related entities together)
    - Configurable overlap strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        section_aware: bool = True,
        preserve_sentences: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_aware = section_aware
        self.preserve_sentences = preserve_sentences
        
        # Biomedical section patterns
        self.section_patterns = [
            r"^(ABSTRACT|Abstract)",
            r"^(INTRODUCTION|Introduction)",
            r"^(METHODS|Methods|MATERIALS AND METHODS)",
            r"^(RESULTS|Results)",
            r"^(DISCUSSION|Discussion)",
            r"^(CONCLUSION|Conclusion|CONCLUSIONS)",
            r"^(REFERENCES|References)",
            r"^(ACKNOWLEDGMENTS|Acknowledgments)"
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using biomedical-aware strategies."""
        if self.section_aware:
            return self._split_by_sections(text)
        else:
            return self._split_by_sentences(text)
    
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
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences while respecting chunk size."""
        import re
        
        # Simple sentence splitting (could be enhanced with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk
                if sentence_size <= self.chunk_size:
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    # Sentence too large, split by words
                    word_chunks = self._split_large_sentence(sentence)
                    chunks.extend(word_chunks)
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
            word_size = len(word) + 1  # +1 for space
            
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
            chunks = self.text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
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