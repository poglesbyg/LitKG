"""
Entity Linking Module

This module handles:
1. Fuzzy matching between literature entities and KG entities
2. Disambiguation using context and confidence scoring
3. Cross-modal entity alignment
4. Validation and quality assessment
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm

from .literature_processor import Entity as LiteratureEntity, ProcessedDocument
from .kg_preprocessor import StandardizedEntity, KnowledgeGraphPreprocessor as KGPreprocessor
from litkg.utils.config import LitKGConfig, load_config, get_cache_dir
from litkg.utils.logging import LoggerMixin


@dataclass
class EntityMatch:
    """Represents a match between literature entity and KG entity."""
    literature_entity: LiteratureEntity
    kg_entity: StandardizedEntity
    similarity_score: float
    confidence_score: float
    match_type: str  # EXACT, FUZZY, SEMANTIC, CONTEXTUAL
    evidence: List[str]
    context: Optional[str] = None


@dataclass
class LinkingResult:
    """Results of entity linking process."""
    document_id: str
    matches: List[EntityMatch]
    unmatched_literature_entities: List[LiteratureEntity]
    disambiguation_conflicts: List[Dict[str, Any]]
    linking_statistics: Dict[str, Any]


class FuzzyMatcher(LoggerMixin):
    """Handles fuzzy string matching between entities."""
    
    def __init__(self, config: Optional[LitKGConfig] = None):
        self.config = load_config() if config is None else (config if isinstance(config, LitKGConfig) else load_config(config))
        self.linking_config = self.config.phase1.entity_linking
        self.fuzzy_config = self.linking_config.fuzzy_matching or {}
        
        # Similarity thresholds with safe defaults
        self.threshold = float(self.fuzzy_config.get("threshold", 0.8))
        self.method = self.fuzzy_config.get("method", "levenshtein")
        
        # Precompiled regex patterns for normalization
        self.normalization_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'[^\w\s-]', ''),  # Remove special characters except hyphens
            (r'\b(gene|protein|receptor|kinase|inhibitor)\b', ''),  # Remove common suffixes
        ]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        normalized = text.lower().strip()
        
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized.strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Normalize texts
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if norm1 == norm2:
            return 1.0
        
        if self.method == "levenshtein":
            return self._levenshtein_similarity(norm1, norm2)
        elif self.method == "jaccard":
            return self._jaccard_similarity(norm1, norm2)
        elif self.method == "sequence_matcher":
            return SequenceMatcher(None, norm1, norm2).ratio()
        else:
            # Default to Levenshtein
            return self._levenshtein_similarity(norm1, norm2)
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate Levenshtein similarity."""
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        
        distance = Levenshtein.distance(text1, text2)
        return 1.0 - (distance / max_len)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on character n-grams."""
        def get_ngrams(text, n=2):
            return set([text[i:i+n] for i in range(len(text)-n+1)])
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0

    # --- Additional APIs expected by tests ---
    def compute_similarity(self, text1: str, text2: str) -> float:
        return self.calculate_similarity(text1, text2)

    def find_best_matches(self, query: str, candidates: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        matches = self.find_fuzzy_matches(query, candidates)
        return [
            {"candidate": cand, "score": score}
            for cand, score in matches[:top_k]
        ]

    def batch_match(self, queries: List[str], candidates: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        return [self.find_best_matches(q, candidates, top_k=top_k) for q in queries]
    
    def find_fuzzy_matches(
        self, 
        query_text: str, 
        candidate_texts: List[str], 
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Find fuzzy matches for a query text among candidates."""
        if threshold is None:
            threshold = self.threshold
        
        matches = []
        
        for candidate in candidate_texts:
            similarity = self.calculate_similarity(query_text, candidate)
            if similarity >= threshold:
                matches.append((candidate, similarity))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def match_with_synonyms(
        self, 
        query_text: str, 
        entity: StandardizedEntity, 
        threshold: Optional[float] = None
    ) -> Optional[Tuple[float, str]]:
        """Match query text with entity name and synonyms."""
        if threshold is None:
            threshold = self.threshold
        
        best_match = None
        best_score = 0.0
        
        # Check primary name
        score = self.calculate_similarity(query_text, entity.name)
        if score >= threshold and score > best_score:
            best_score = score
            best_match = entity.name
        
        # Check synonyms
        for synonym in entity.synonyms:
            score = self.calculate_similarity(query_text, synonym)
            if score >= threshold and score > best_score:
                best_score = score
                best_match = synonym
        
        return (best_score, best_match) if best_match else None


class SemanticMatcher(LoggerMixin):
    """Handles semantic similarity matching using embeddings."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight sentence transformer
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded semantic model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load semantic model: {e}")
            self.model = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        self._load_embedding_cache()
    
    def _load_embedding_cache(self):
        """Load cached embeddings."""
        cache_file = get_cache_dir() / "semantic_embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                self.logger.error(f"Failed to load embedding cache: {e}")
    
    def _save_embedding_cache(self):
        """Save embeddings to cache."""
        cache_file = get_cache_dir() / "semantic_embeddings.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            self.logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Failed to save embedding cache: {e}")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, using cache if available."""
        if not self.model:
            return None
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.model.encode([text])[0]
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to get embedding for '{text}': {e}")
            return None
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def find_semantic_matches(
        self, 
        query_text: str, 
        candidate_entities: List[StandardizedEntity],
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[StandardizedEntity, float]]:
        """Find semantically similar entities."""
        if not self.model:
            return []
        
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return []
        
        matches = []
        
        for entity in candidate_entities:
            # Try entity name
            entity_embedding = self.get_embedding(entity.name)
            if entity_embedding is not None:
                similarity = cosine_similarity([query_embedding], [entity_embedding])[0][0]
                if similarity >= threshold:
                    matches.append((entity, float(similarity)))
            
            # Try synonyms
            for synonym in entity.synonyms[:3]:  # Limit to first 3 synonyms
                synonym_embedding = self.get_embedding(synonym)
                if synonym_embedding is not None:
                    similarity = cosine_similarity([query_embedding], [synonym_embedding])[0][0]
                    if similarity >= threshold:
                        matches.append((entity, float(similarity)))
                        break  # Take best synonym match
        
        # Remove duplicates and sort
        unique_matches = {}
        for entity, score in matches:
            if entity.id not in unique_matches or score > unique_matches[entity.id][1]:
                unique_matches[entity.id] = (entity, score)
        
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x[1], reverse=True)
        
        return sorted_matches[:top_k]


class ContextualDisambiguator(LoggerMixin):
    """Handles disambiguation using context information."""
    
    def __init__(self, config: Optional[LitKGConfig] = None):
        self.config = load_config() if config is None else (config if isinstance(config, LitKGConfig) else load_config(config))
        self.disambiguation_config = self.config.phase1.entity_linking.disambiguation or {}
        # Safe defaults for tests
        self.context_window = int(self.disambiguation_config.get("context_window", 100))
        self.confidence_threshold = float(self.disambiguation_config.get("confidence_threshold", 0.7))
        
        # Context window size
        self.context_window = self.disambiguation_config["context_window"]
        self.confidence_threshold = self.disambiguation_config["confidence_threshold"]
        
        # Load spacy model for context analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spacy model not found, using basic disambiguation")
            self.nlp = None
        # Frequency store for tests to patch
        self.entity_frequencies: Dict[str, int] = {}
        
        # TF-IDF vectorizer for context similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_context(self, text: str, entity_start: int, entity_end: int) -> str:
        """Extract context around an entity mention."""
        # Extract context window around entity
        start = max(0, entity_start - self.context_window)
        end = min(len(text), entity_end + self.context_window)
        
        context = text[start:end]
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context).strip()
        
        return context
    
    def get_entity_context_features(self, entity: StandardizedEntity) -> List[str]:
        """Get context features for a KG entity."""
        features = []
        
        # Add entity type
        features.append(entity.type.lower())
        
        # Add source information
        features.append(entity.source.lower())
        
        # Add attributes
        for key, value in entity.attributes.items():
            if isinstance(value, str) and value:
                features.append(f"{key}:{value.lower()}")
        
        # Add ontology information
        if entity.cui:
            features.append(f"cui:{entity.cui}")
        
        if entity.go_id:
            features.append(f"go:{entity.go_id}")
        
        return features
    
    def calculate_context_similarity(
        self, 
        literature_context: str, 
        kg_entity: StandardizedEntity
    ) -> float:
        """Calculate similarity between literature context and KG entity context."""
        # Get entity context features
        entity_features = self.get_entity_context_features(kg_entity)
        entity_context = " ".join(entity_features)
        
        if not literature_context.strip() or not entity_context.strip():
            return 0.0
        
        try:
            # Use TF-IDF similarity
            contexts = [literature_context, entity_context]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(contexts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        
        except Exception as e:
            self.logger.error(f"Error calculating context similarity: {e}")
            return 0.0

    # --- Additional helpers expected by tests ---
    def _compute_context_similarity(self, context: str, entity: StandardizedEntity) -> float:
        return self.calculate_context_similarity(context, entity)

    def disambiguate_with_context(self, entity: str, candidates: List[str], context: str) -> Dict[str, Any]:
        # Simple heuristic: use context similarity scores from _compute_context_similarity if available via ontology db
        scored = []
        for candidate in candidates:
            stub_entity = StandardizedEntity(id=candidate, name=candidate, type="ENTITY", source="", original_id=candidate, synonyms=[])
            score = self._compute_context_similarity(context, stub_entity)
            scored.append({"candidate": candidate, "confidence": score})
        scored.sort(key=lambda x: x["confidence"], reverse=True)
        return scored[0] if scored else {"candidate": candidates[0] if candidates else None, "confidence": 0.0}

    def disambiguate_with_frequency(self, entity: str, candidates: List[str]) -> Dict[str, Any]:
        best = None
        best_freq = -1
        for c in candidates:
            freq = self.entity_frequencies.get(c, 0)
            if freq > best_freq:
                best_freq = freq
                best = c
        return {"candidate": best or (candidates[0] if candidates else None), "confidence": 1.0 if best_freq > 0 else 0.0}

    def multi_criteria_disambiguation(self, entity: str, candidates: List[str], context: str) -> Dict[str, Any]:
        # Combine frequency and context
        scored = []
        for c in candidates:
            freq = self.entity_frequencies.get(c, 0)
            stub_entity = StandardizedEntity(id=c, name=c, type="ENTITY", source="", original_id=c, synonyms=[])
            ctx = self._compute_context_similarity(context, stub_entity)
            combined = 0.6 * (1.0 if freq > 0 else 0.0) + 0.4 * ctx
            scored.append({"candidate": c, "combined_score": combined})
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[0] if scored else {"candidate": candidates[0] if candidates else None, "combined_score": 0.0}
    
    def disambiguate_matches(
        self, 
        literature_entity: LiteratureEntity,
        context: str,
        candidate_matches: List[Tuple[StandardizedEntity, float]]
    ) -> Optional[Tuple[StandardizedEntity, float, float]]:
        """
        Disambiguate between multiple candidate matches.
        
        Returns:
            Tuple of (best_entity, similarity_score, context_score) or None
        """
        if not candidate_matches:
            return None
        
        if len(candidate_matches) == 1:
            entity, similarity = candidate_matches[0]
            context_score = self.calculate_context_similarity(context, entity)
            return (entity, similarity, context_score)
        
        # Score each candidate
        scored_candidates = []
        
        for entity, similarity in candidate_matches:
            context_score = self.calculate_context_similarity(context, entity)
            
            # Combined score (weighted)
            combined_score = 0.6 * similarity + 0.4 * context_score
            
            scored_candidates.append((entity, similarity, context_score, combined_score))
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x[3], reverse=True)
        
        best_entity, similarity, context_score, combined_score = scored_candidates[0]
        
        # Check if disambiguation is confident enough
        if combined_score >= self.confidence_threshold:
            return (best_entity, similarity, context_score)
        
        return None
    
    def analyze_entity_types(
        self, 
        literature_entity: LiteratureEntity,
        kg_entity: StandardizedEntity
    ) -> float:
        """Analyze compatibility of entity types."""
        lit_type = literature_entity.label.upper()
        kg_type = kg_entity.type.upper()
        
        # Direct match
        if lit_type == kg_type:
            return 1.0
        
        # Compatible types
        compatible_types = {
            "GENE": ["PROTEIN", "GENE_PRODUCT"],
            "PROTEIN": ["GENE", "GENE_PRODUCT"],
            "DISEASE": ["DISORDER", "CONDITION"],
            "DRUG": ["COMPOUND", "CHEMICAL", "MEDICATION"],
            "CHEMICAL": ["DRUG", "COMPOUND"],
            "MUTATION": ["VARIANT", "ALTERATION"]
        }
        
        if lit_type in compatible_types:
            if kg_type in compatible_types[lit_type]:
                return 0.8
        
        return 0.0


class EntityLinker(LoggerMixin):
    """Main entity linking coordinator."""
    
    def __init__(self, config_path: Optional[Union[str, Dict[str, Any], LitKGConfig]] = None):
        self.config = load_config(config_path)
        
        # Initialize components
        self.fuzzy_matcher = FuzzyMatcher(self.config)
        self.semantic_matcher = SemanticMatcher(self.config)
        self.disambiguator = ContextualDisambiguator(self.config)
        
        # Knowledge graph entities (loaded from preprocessor)
        self.kg_entities = {}
        self.entity_index = defaultdict(list)  # Type-based index
        
        # Linking statistics
        self.linking_stats = {
            "total_processed": 0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "semantic_matches": 0,
            "contextual_matches": 0,
            "unmatched": 0,
            "disambiguation_conflicts": 0
        }
    
    def load_kg_entities(self, kg_preprocessor: KGPreprocessor):
        """Load KG entities from preprocessor."""
        self.logger.info("Loading KG entities for linking...")
        
        self.kg_entities = kg_preprocessor.graph_builder.entities
        
        # Build type-based index for faster lookup
        for entity_id, entity in self.kg_entities.items():
            self.entity_index[entity.type].append(entity)
        
        self.logger.info(f"Loaded {len(self.kg_entities)} KG entities")
        self.logger.info(f"Entity types: {list(self.entity_index.keys())}")
    
    def link_document_entities(
        self, 
        document: ProcessedDocument,
        use_semantic: bool = True,
        use_context: bool = True
    ) -> LinkingResult:
        """Link entities in a document to KG entities."""
        self.logger.info(f"Linking entities for document: {document.pmid}")
        
        matches = []
        unmatched = []
        conflicts = []
        
        # Combine title and abstract for context
        full_text = f"{document.title} {document.abstract}"
        
        for lit_entity in tqdm(document.entities, desc="Linking entities"):
            # Extract context around entity
            context = self.disambiguator.extract_context(
                full_text, lit_entity.start, lit_entity.end
            )
            
            # Find candidate matches
            candidates = self._find_candidate_matches(
                lit_entity, use_semantic=use_semantic
            )
            
            if not candidates:
                unmatched.append(lit_entity)
                self.linking_stats["unmatched"] += 1
                continue
            
            # Disambiguate if multiple candidates
            if use_context and len(candidates) > 1:
                disambiguation_result = self.disambiguator.disambiguate_matches(
                    lit_entity, context, candidates
                )
                
                if disambiguation_result:
                    kg_entity, similarity, context_score = disambiguation_result
                    
                    match = EntityMatch(
                        literature_entity=lit_entity,
                        kg_entity=kg_entity,
                        similarity_score=similarity,
                        confidence_score=(similarity + context_score) / 2,
                        match_type="CONTEXTUAL",
                        evidence=[f"Context similarity: {context_score:.3f}"],
                        context=context
                    )
                    
                    matches.append(match)
                    self.linking_stats["contextual_matches"] += 1
                else:
                    # Record disambiguation conflict
                    conflicts.append({
                        "entity": asdict(lit_entity),
                        "candidates": [(asdict(e), s) for e, s in candidates],
                        "context": context
                    })
                    self.linking_stats["disambiguation_conflicts"] += 1
            else:
                # Take best candidate
                kg_entity, similarity = candidates[0]
                
                # Determine match type
                if similarity == 1.0:
                    match_type = "EXACT"
                    self.linking_stats["exact_matches"] += 1
                elif similarity >= 0.9:
                    match_type = "FUZZY"
                    self.linking_stats["fuzzy_matches"] += 1
                else:
                    match_type = "SEMANTIC"
                    self.linking_stats["semantic_matches"] += 1
                
                match = EntityMatch(
                    literature_entity=lit_entity,
                    kg_entity=kg_entity,
                    similarity_score=similarity,
                    confidence_score=similarity,
                    match_type=match_type,
                    evidence=[f"String similarity: {similarity:.3f}"],
                    context=context
                )
                
                matches.append(match)
        
        self.linking_stats["total_processed"] += len(document.entities)
        
        # Calculate document-level statistics
        doc_stats = {
            "total_entities": len(document.entities),
            "matched_entities": len(matches),
            "unmatched_entities": len(unmatched),
            "conflicts": len(conflicts),
            "match_rate": len(matches) / len(document.entities) if document.entities else 0
        }
        
        return LinkingResult(
            document_id=document.pmid,
            matches=matches,
            unmatched_literature_entities=unmatched,
            disambiguation_conflicts=conflicts,
            linking_statistics=doc_stats
        )
    
    def _find_candidate_matches(
        self, 
        lit_entity: LiteratureEntity,
        use_semantic: bool = True
    ) -> List[Tuple[StandardizedEntity, float]]:
        """Find candidate KG entities for a literature entity."""
        candidates = []
        
        # Get entities of compatible types
        compatible_entities = []
        entity_type = lit_entity.label.upper()
        
        # Direct type match
        if entity_type in self.entity_index:
            compatible_entities.extend(self.entity_index[entity_type])
        
        # Compatible types
        type_mappings = {
            "GENE": ["PROTEIN"],
            "PROTEIN": ["GENE"],
            "DISEASE": ["DISORDER"],
            "DRUG": ["CHEMICAL", "COMPOUND"],
            "CHEMICAL": ["DRUG"]
        }
        
        if entity_type in type_mappings:
            for mapped_type in type_mappings[entity_type]:
                if mapped_type in self.entity_index:
                    compatible_entities.extend(self.entity_index[mapped_type])
        
        # Fuzzy matching
        for kg_entity in compatible_entities:
            fuzzy_result = self.fuzzy_matcher.match_with_synonyms(
                lit_entity.text, kg_entity
            )
            
            if fuzzy_result:
                similarity, matched_text = fuzzy_result
                
                # Type compatibility bonus
                type_score = self.disambiguator.analyze_entity_types(lit_entity, kg_entity)
                adjusted_similarity = similarity * (0.8 + 0.2 * type_score)
                
                candidates.append((kg_entity, adjusted_similarity))
        
        # Semantic matching (if enabled and no high-confidence fuzzy matches)
        if use_semantic and (not candidates or max(c[1] for c in candidates) < 0.9):
            semantic_matches = self.semantic_matcher.find_semantic_matches(
                lit_entity.text, compatible_entities, threshold=0.6
            )
            
            for kg_entity, similarity in semantic_matches:
                # Check if already in candidates
                existing = next((c for c in candidates if c[0].id == kg_entity.id), None)
                
                if existing:
                    # Update with better score if semantic is higher
                    if similarity > existing[1]:
                        candidates = [(e, s) for e, s in candidates if e.id != kg_entity.id]
                        candidates.append((kg_entity, similarity))
                else:
                    candidates.append((kg_entity, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def batch_link_documents(
        self, 
        documents: List[ProcessedDocument],
        use_semantic: bool = True,
        use_context: bool = True
    ) -> List[LinkingResult]:
        """Link entities for multiple documents."""
        self.logger.info(f"Batch linking entities for {len(documents)} documents")
        
        results = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                result = self.link_document_entities(
                    doc, use_semantic=use_semantic, use_context=use_context
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error linking document {doc.pmid}: {e}")
                continue
        
        self.logger.info("Batch linking completed")
        self._log_overall_statistics()
        
        return results


# Backward-compatibility alias expected by tests
class DisambiguationEngine(ContextualDisambiguator):
    pass
    
    def _log_overall_statistics(self):
        """Log overall linking statistics."""
        total = self.linking_stats["total_processed"]
        if total == 0:
            return
        
        self.logger.info("=== ENTITY LINKING STATISTICS ===")
        self.logger.info(f"Total entities processed: {total}")
        self.logger.info(f"Exact matches: {self.linking_stats['exact_matches']} ({self.linking_stats['exact_matches']/total*100:.1f}%)")
        self.logger.info(f"Fuzzy matches: {self.linking_stats['fuzzy_matches']} ({self.linking_stats['fuzzy_matches']/total*100:.1f}%)")
        self.logger.info(f"Semantic matches: {self.linking_stats['semantic_matches']} ({self.linking_stats['semantic_matches']/total*100:.1f}%)")
        self.logger.info(f"Contextual matches: {self.linking_stats['contextual_matches']} ({self.linking_stats['contextual_matches']/total*100:.1f}%)")
        self.logger.info(f"Unmatched: {self.linking_stats['unmatched']} ({self.linking_stats['unmatched']/total*100:.1f}%)")
        self.logger.info(f"Disambiguation conflicts: {self.linking_stats['disambiguation_conflicts']}")
    
    def save_linking_results(self, results: List[LinkingResult], output_path: str):
        """Save linking results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_results = []
        
        for result in results:
            result_dict = asdict(result)
            serializable_results.append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump({
                "linking_results": serializable_results,
                "overall_statistics": self.linking_stats
            }, f, indent=2, default=str)
        
        self.logger.info(f"Linking results saved to {output_path}")
        
        # Save embedding cache
        self.semantic_matcher._save_embedding_cache()
    
    def load_linking_results(self, input_path: str) -> List[LinkingResult]:
        """Load linking results from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        results = []
        for result_dict in data["linking_results"]:
            # Reconstruct objects
            matches = []
            for match_dict in result_dict["matches"]:
                match = EntityMatch(
                    literature_entity=LiteratureEntity(**match_dict["literature_entity"]),
                    kg_entity=StandardizedEntity(**match_dict["kg_entity"]),
                    similarity_score=match_dict["similarity_score"],
                    confidence_score=match_dict["confidence_score"],
                    match_type=match_dict["match_type"],
                    evidence=match_dict["evidence"],
                    context=match_dict.get("context")
                )
                matches.append(match)
            
            unmatched = [LiteratureEntity(**e) for e in result_dict["unmatched_literature_entities"]]
            
            result = LinkingResult(
                document_id=result_dict["document_id"],
                matches=matches,
                unmatched_literature_entities=unmatched,
                disambiguation_conflicts=result_dict["disambiguation_conflicts"],
                linking_statistics=result_dict["linking_statistics"]
            )
            
            results.append(result)
        
        # Load overall statistics
        if "overall_statistics" in data:
            self.linking_stats.update(data["overall_statistics"])
        
        self.logger.info(f"Loaded {len(results)} linking results")
        
        return results