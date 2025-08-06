"""
Embedding utilities for biomedical entities and texts.

This module provides efficient embedding generation and caching
for literature and knowledge graph entities.
"""

import os
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .huggingface_models import BiomedicalModelManager
from ..utils.config import LitKGConfig, get_cache_dir
from ..utils.logging import LoggerMixin


class BiomedicalEmbeddings(LoggerMixin):
    """Generate and manage embeddings for biomedical entities."""
    
    def __init__(
        self,
        config: LitKGConfig,
        model_name: str = "pubmedbert",
        sentence_model: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True
    ):
        self.config = config
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        # Initialize model manager
        self.model_manager = BiomedicalModelManager(config)
        
        # Initialize sentence transformer for general embeddings
        try:
            self.sentence_model = SentenceTransformer(sentence_model)
            self.logger.info(f"Loaded sentence transformer: {sentence_model}")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Cache setup
        self.cache_dir = get_cache_dir() / "embeddings"
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self._embedding_cache = {}
        
        # Load persistent cache
        if cache_embeddings:
            self._load_cache()
    
    def _load_cache(self):
        """Load persistent embedding cache."""
        cache_file = self.cache_dir / f"{self.model_name}_embeddings.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                self.logger.error(f"Failed to load embedding cache: {e}")
                self._embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.cache_embeddings:
            return
        
        cache_file = self.cache_dir / f"{self.model_name}_embeddings.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            self.logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        combined = f"{model_name}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_text_embeddings(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for text(s) using biomedical models.
        
        Args:
            texts: Single text or list of texts
            model_name: Model to use (defaults to instance model)
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            Embeddings array [num_texts, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if model_name is None:
            model_name = self.model_name
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, model_name)
            if cache_key in self._embedding_cache:
                cached_embeddings[i] = self._embedding_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            self.logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            
            # Use biomedical model
            embeddings_tensor = self.model_manager.get_embeddings(
                uncached_texts,
                model_name=model_name,
                batch_size=batch_size
            )
            
            embeddings_np = embeddings_tensor.numpy()
            
            # Cache new embeddings
            for i, text in enumerate(uncached_texts):
                cache_key = self._get_cache_key(text, model_name)
                self._embedding_cache[cache_key] = embeddings_np[i]
                cached_embeddings[uncached_indices[i]] = embeddings_np[i]
        
        # Assemble final embeddings in original order
        all_embeddings = np.array([cached_embeddings[i] for i in range(len(texts))])
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            all_embeddings = all_embeddings / np.maximum(norms, 1e-12)
        
        # Save cache periodically
        if len(self._embedding_cache) % 100 == 0:
            self._save_cache()
        
        return all_embeddings
    
    def get_entity_embeddings(
        self,
        entities: List[Dict[str, Any]],
        context_window: int = 100,
        include_context: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for entities with optional context.
        
        Args:
            entities: List of entity dictionaries with 'text' and optional 'context'
            context_window: Size of context window around entity
            include_context: Whether to include context in embedding
            
        Returns:
            Entity embeddings array
        """
        texts = []
        
        for entity in entities:
            text = entity['text']
            
            if include_context and 'context' in entity:
                context = entity['context']
                # Create contextual text
                text = f"{context[:context_window]} {text} {context[-context_window:]}"
            
            texts.append(text)
        
        return self.get_text_embeddings(texts)
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [n1, dim]
            embeddings2: Second set of embeddings [n2, dim]
            metric: Similarity metric (cosine, dot, euclidean)
            
        Returns:
            Similarity matrix [n1, n2]
        """
        if metric == "cosine":
            return cosine_similarity(embeddings1, embeddings2)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            # Convert to similarity (higher = more similar)
            distances = np.linalg.norm(
                embeddings1[:, np.newaxis] - embeddings2[np.newaxis, :],
                axis=2
            )
            return 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, similarity_score) tuples
        """
        # Get embeddings
        query_embedding = self.get_text_embeddings([query_text])
        candidate_embeddings = self.get_text_embeddings(candidate_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)[0]
        
        # Filter by threshold and get top-k
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                results.append((candidate_texts[i], float(sim)))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        num_clusters: int = 10,
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Cluster embeddings using specified method.
        
        Args:
            embeddings: Embeddings to cluster [n_samples, n_features]
            num_clusters: Number of clusters
            method: Clustering method (kmeans, hierarchical)
            
        Returns:
            Cluster labels [n_samples]
        """
        if method == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=num_clusters, random_state=42)
        elif method == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=num_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(embeddings)
        return labels
    
    def reduce_dimensionality(
        self,
        embeddings: np.ndarray,
        method: str = "pca",
        n_components: int = 50
    ) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method (pca, tsne, umap)
            n_components: Number of components to keep
            
        Returns:
            Reduced embeddings
        """
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=min(n_components, 3), random_state=42)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            except ImportError:
                self.logger.error("UMAP not installed, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        return reduced
    
    def save_embeddings(self, filepath: str, embeddings: np.ndarray, metadata: Optional[Dict] = None):
        """Save embeddings to file."""
        data = {
            "embeddings": embeddings,
            "model_name": self.model_name,
            "metadata": metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data["embeddings"], data.get("metadata", {})
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Save cache on destruction."""
        if hasattr(self, '_embedding_cache') and self.cache_embeddings:
            self._save_cache()


class CachedEmbeddings:
    """Efficient cached embeddings with disk persistence."""
    
    def __init__(self, cache_dir: str, model_name: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        self.cache_file = self.cache_dir / f"{model_name}_embeddings.npz"
        self.metadata_file = self.cache_dir / f"{model_name}_metadata.json"
        
        self._load_cache()
    
    def _load_cache(self):
        """Load cached embeddings."""
        if self.cache_file.exists():
            data = np.load(self.cache_file, allow_pickle=True)
            self.embeddings = data['embeddings']
            self.text_to_idx = data['text_to_idx'].item()
        else:
            self.embeddings = np.empty((0, 0))
            self.text_to_idx = {}
    
    def _save_cache(self):
        """Save embeddings to cache."""
        np.savez(
            self.cache_file,
            embeddings=self.embeddings,
            text_to_idx=self.text_to_idx
        )
    
    def get_or_compute(self, texts: List[str], compute_fn) -> np.ndarray:
        """Get embeddings from cache or compute if missing."""
        # Find missing texts
        missing_texts = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            if text not in self.text_to_idx:
                missing_texts.append(text)
                missing_indices.append(i)
        
        # Compute missing embeddings
        if missing_texts:
            new_embeddings = compute_fn(missing_texts)
            self._add_embeddings(missing_texts, new_embeddings)
        
        # Return requested embeddings
        indices = [self.text_to_idx[text] for text in texts]
        return self.embeddings[indices]
    
    def _add_embeddings(self, texts: List[str], embeddings: np.ndarray):
        """Add new embeddings to cache."""
        start_idx = len(self.text_to_idx)
        
        # Update text to index mapping
        for i, text in enumerate(texts):
            self.text_to_idx[text] = start_idx + i
        
        # Append embeddings
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Save to disk
        self._save_cache()


class MultiModalEmbeddings(LoggerMixin):
    """Multi-modal embeddings combining text and graph structure."""
    
    def __init__(
        self,
        text_embedder: BiomedicalEmbeddings,
        graph_embedding_dim: int = 128,
        fusion_method: str = "concat"
    ):
        self.text_embedder = text_embedder
        self.graph_embedding_dim = graph_embedding_dim
        self.fusion_method = fusion_method
        
        # Graph embedding layers
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim, graph_embedding_dim)
        )
        
        # Fusion layer
        text_dim = 768  # Typical BERT dimension
        if fusion_method == "concat":
            self.fusion = nn.Linear(text_dim + graph_embedding_dim, text_dim)
        elif fusion_method == "add":
            assert text_dim == graph_embedding_dim, "Dimensions must match for addition"
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def get_multimodal_embeddings(
        self,
        texts: List[str],
        graph_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get multi-modal embeddings combining text and graph features.
        
        Args:
            texts: List of texts
            graph_features: Graph features [num_entities, graph_embedding_dim]
            
        Returns:
            Multi-modal embeddings
        """
        # Get text embeddings
        text_embeddings = self.text_embedder.get_text_embeddings(texts)
        text_tensor = torch.from_numpy(text_embeddings).float()
        
        # Encode graph features
        graph_encoded = self.graph_encoder(graph_features)
        
        # Fuse modalities
        if self.fusion_method == "concat":
            combined = torch.cat([text_tensor, graph_encoded], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_method == "add":
            fused = text_tensor + graph_encoded
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused