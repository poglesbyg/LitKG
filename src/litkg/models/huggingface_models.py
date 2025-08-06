"""
HuggingFace model integration for biomedical NLP.

This module provides a unified interface for loading and using
various biomedical transformer models from HuggingFace Hub.
"""

import os
import json
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
    pipeline, Pipeline, PreTrainedModel, PreTrainedTokenizer
)
from datasets import Dataset, DatasetDict, load_dataset

from ..utils.config import LitKGConfig, get_cache_dir
from ..utils.logging import LoggerMixin


@dataclass
class ModelInfo:
    """Information about a biomedical model."""
    name: str
    model_id: str
    description: str
    tasks: List[str]  # NER, classification, QA, etc.
    domain: str  # biomedical, clinical, general
    size: str  # small, base, large
    paper_url: Optional[str] = None
    citation: Optional[str] = None


class ModelRegistry:
    """Registry of available biomedical models."""
    
    BIOMEDICAL_MODELS = {
        "pubmedbert": ModelInfo(
            name="PubMedBERT",
            model_id="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            description="BERT model pre-trained on PubMed abstracts and full-text articles",
            tasks=["embeddings", "classification", "ner"],
            domain="biomedical",
            size="base",
            paper_url="https://arxiv.org/abs/2007.15779",
            citation="Gu et al. (2020)"
        ),
        "biobert": ModelInfo(
            name="BioBERT",
            model_id="dmis-lab/biobert-base-cased-v1.1",
            description="BERT model pre-trained on biomedical corpora",
            tasks=["embeddings", "classification", "ner"],
            domain="biomedical",
            size="base",
            paper_url="https://arxiv.org/abs/1901.08746",
            citation="Lee et al. (2019)"
        ),
        "clinicalbert": ModelInfo(
            name="ClinicalBERT",
            model_id="emilyalsentzer/Bio_ClinicalBERT",
            description="BERT model pre-trained on clinical notes",
            tasks=["embeddings", "classification"],
            domain="clinical",
            size="base",
            paper_url="https://arxiv.org/abs/1904.05342",
            citation="Alsentzer et al. (2019)"
        ),
        "bluebert": ModelInfo(
            name="BlueBERT",
            model_id="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
            description="BERT model pre-trained on PubMed and MIMIC-III",
            tasks=["embeddings", "classification", "ner"],
            domain="biomedical",
            size="base",
            paper_url="https://arxiv.org/abs/1906.05474",
            citation="Peng et al. (2019)"
        ),
        "scibert": ModelInfo(
            name="SciBERT",
            model_id="allenai/scibert_scivocab_uncased",
            description="BERT model pre-trained on scientific papers",
            tasks=["embeddings", "classification", "ner"],
            domain="scientific",
            size="base",
            paper_url="https://arxiv.org/abs/1903.10676",
            citation="Beltagy et al. (2019)"
        ),
        "bioelectra": ModelInfo(
            name="BioELECTRA",
            model_id="kamalkraj/bioelectra-base-discriminator-pubmed",
            description="ELECTRA model pre-trained on PubMed",
            tasks=["embeddings", "classification"],
            domain="biomedical",
            size="base",
            citation="Kanakarajan et al. (2021)"
        ),
        "gatortron": ModelInfo(
            name="GatorTron",
            model_id="UFNLP/gatortron-base",
            description="Clinical transformer model trained on clinical text",
            tasks=["embeddings", "classification"],
            domain="clinical",
            size="base",
            paper_url="https://arxiv.org/abs/2203.03540",
            citation="Yang et al. (2022)"
        )
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return cls.BIOMEDICAL_MODELS.get(model_name.lower())
    
    @classmethod
    def list_models(cls, domain: Optional[str] = None, task: Optional[str] = None) -> List[ModelInfo]:
        """List available models, optionally filtered by domain or task."""
        models = list(cls.BIOMEDICAL_MODELS.values())
        
        if domain:
            models = [m for m in models if m.domain == domain]
        
        if task:
            models = [m for m in models if task in m.tasks]
        
        return models


class BiomedicalModelManager(LoggerMixin):
    """Manages loading and caching of biomedical models."""
    
    def __init__(self, config: LitKGConfig):
        self.config = config
        self.cache_dir = get_cache_dir() / "models"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model cache
        self._model_cache = {}
        self._tokenizer_cache = {}
        self._pipeline_cache = {}
        
        # Device management
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")
    
    def load_model(
        self, 
        model_name: str, 
        task: str = "embeddings",
        force_reload: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a biomedical model and tokenizer.
        
        Args:
            model_name: Name or HuggingFace model ID
            task: Task type (embeddings, ner, classification, qa)
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = f"{model_name}_{task}"
        
        if not force_reload and cache_key in self._model_cache:
            self.logger.info(f"Using cached model: {model_name}")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]
        
        # Get model info
        model_info = ModelRegistry.get_model_info(model_name)
        if model_info:
            model_id = model_info.model_id
            self.logger.info(f"Loading {model_info.name} ({model_id})")
        else:
            model_id = model_name
            self.logger.info(f"Loading model: {model_id}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir)
            )
            
            # Load model based on task
            if task == "ner":
                model = AutoModelForTokenClassification.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir)
                )
            elif task == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir)
                )
            elif task == "qa":
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir)
                )
            else:  # embeddings or general
                model = AutoModel.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir)
                )
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            
            # Cache the model and tokenizer
            self._model_cache[cache_key] = model
            self._tokenizer_cache[cache_key] = tokenizer
            
            self.logger.info(f"Successfully loaded {model_id}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def create_pipeline(
        self,
        model_name: str,
        task: str,
        **kwargs
    ) -> Pipeline:
        """
        Create a HuggingFace pipeline for a specific task.
        
        Args:
            model_name: Name or HuggingFace model ID
            task: Pipeline task (ner, classification, qa, etc.)
            **kwargs: Additional pipeline arguments
            
        Returns:
            HuggingFace Pipeline object
        """
        cache_key = f"{model_name}_{task}_pipeline"
        
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        
        # Get model info
        model_info = ModelRegistry.get_model_info(model_name)
        model_id = model_info.model_id if model_info else model_name
        
        try:
            # Set device
            device = 0 if self.device.type == "cuda" else -1
            if self.device.type == "mps":
                device = 0  # MPS uses device 0
            
            pipe = pipeline(
                task,
                model=model_id,
                tokenizer=model_id,
                device=device,
                model_kwargs={"cache_dir": str(self.cache_dir)},
                **kwargs
            )
            
            self._pipeline_cache[cache_key] = pipe
            self.logger.info(f"Created {task} pipeline for {model_id}")
            
            return pipe
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline for {model_id}: {e}")
            raise
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        model_name: str = "pubmedbert",
        batch_size: int = 32,
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Get embeddings for text(s) using a biomedical model.
        
        Args:
            texts: Single text or list of texts
            model_name: Model to use for embeddings
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Tensor of embeddings [num_texts, hidden_size]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        model, tokenizer = self.load_model(model_name, task="embeddings")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                outputs = model(**inputs)
                
                # Use [CLS] token embeddings or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # Mean pooling over sequence length
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def extract_entities_batch(
        self,
        texts: List[str],
        model_name: str = "biobert",
        batch_size: int = 16
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from a batch of texts.
        
        Args:
            texts: List of texts to process
            model_name: Model to use for NER
            batch_size: Batch size for processing
            
        Returns:
            List of entity lists for each text
        """
        ner_pipeline = self.create_pipeline(model_name, "ner", aggregation_strategy="simple")
        
        all_entities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = ner_pipeline(batch_texts)
            
            # Handle single text vs batch results
            if isinstance(batch_results[0], dict):
                batch_results = [batch_results]
            
            all_entities.extend(batch_results)
        
        return all_entities
    
    def classify_texts(
        self,
        texts: List[str],
        model_name: str = "clinicalbert",
        labels: Optional[List[str]] = None,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Classify texts using a biomedical classification model.
        
        Args:
            texts: List of texts to classify
            model_name: Model to use for classification
            labels: Optional list of candidate labels (for zero-shot)
            batch_size: Batch size for processing
            
        Returns:
            List of classification results
        """
        if labels:
            # Zero-shot classification
            classifier = self.create_pipeline(model_name, "zero-shot-classification")
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = classifier(batch_texts, labels)
                
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                
                results.extend(batch_results)
            
            return results
        else:
            # Regular classification
            classifier = self.create_pipeline(model_name, "text-classification")
            
            all_results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = classifier(batch_texts)
                
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                
                all_results.extend(batch_results)
            
            return all_results
    
    def answer_questions(
        self,
        questions: List[str],
        contexts: List[str],
        model_name: str = "pubmedbert",
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Answer questions given contexts using a biomedical QA model.
        
        Args:
            questions: List of questions
            contexts: List of contexts for each question
            model_name: Model to use for QA
            batch_size: Batch size for processing
            
        Returns:
            List of answer results
        """
        qa_pipeline = self.create_pipeline(model_name, "question-answering")
        
        qa_pairs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]
        
        all_results = []
        for i in range(0, len(qa_pairs), batch_size):
            batch_pairs = qa_pairs[i:i + batch_size]
            batch_results = qa_pipeline(batch_pairs)
            
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            
            all_results.extend(batch_results)
        
        return all_results
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()
        self._pipeline_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "device": str(self.device),
            "cached_models": list(self._model_cache.keys()),
            "cached_pipelines": list(self._pipeline_cache.keys()),
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, str]:
        """Get memory usage information."""
        usage = {}
        
        if torch.cuda.is_available():
            usage["cuda_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            usage["cuda_cached"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        
        return usage


# Convenience functions
def load_biomedical_model(model_name: str, config: Optional[LitKGConfig] = None) -> BiomedicalModelManager:
    """Load a biomedical model manager."""
    if config is None:
        from ..utils.config import load_config
        config = load_config()
    
    manager = BiomedicalModelManager(config)
    return manager


def get_available_models() -> List[ModelInfo]:
    """Get list of available biomedical models."""
    return list(ModelRegistry.BIOMEDICAL_MODELS.values())


# Dataset utilities
def load_biomedical_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Union[Dataset, DatasetDict]:
    """
    Load a biomedical dataset from HuggingFace Hub.
    
    Common biomedical datasets:
    - "bigbio/pubmed_qa": PubMed QA dataset
    - "bigbio/bc5cdr": BC5CDR NER dataset  
    - "bigbio/biosses": Biomedical semantic similarity
    - "bigbio/chemprot": Chemical-protein interactions
    """
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)