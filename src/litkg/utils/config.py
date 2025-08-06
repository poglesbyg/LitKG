"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LiteratureConfig(BaseModel):
    """Configuration for literature processing."""
    models: Dict[str, str]
    pubmed: Dict[str, Any]
    text_processing: Dict[str, Any]


class KnowledgeGraphConfig(BaseModel):
    """Configuration for knowledge graph processing."""
    civic: Dict[str, Any]
    tcga: Dict[str, Any]
    cptac: Dict[str, Any]
    ontologies: Dict[str, Any]


class EntityLinkingConfig(BaseModel):
    """Configuration for entity linking."""
    fuzzy_matching: Dict[str, Any]
    disambiguation: Dict[str, Any]


class Phase1Config(BaseModel):
    """Phase 1 configuration."""
    literature: LiteratureConfig
    knowledge_graphs: KnowledgeGraphConfig
    entity_linking: EntityLinkingConfig


class GNNConfig(BaseModel):
    """GNN architecture configuration."""
    architecture: Dict[str, Any]
    cross_modal: Dict[str, Any]
    training: Dict[str, Any]


class ConfidenceScoringConfig(BaseModel):
    """Confidence scoring configuration."""
    metrics: list[str]
    weights: Dict[str, float]


class Phase2Config(BaseModel):
    """Phase 2 configuration."""
    gnn: GNNConfig
    confidence_scoring: ConfidenceScoringConfig


class PredictionConfig(BaseModel):
    """Prediction configuration."""
    link_prediction: Dict[str, Any]
    hypothesis: Dict[str, Any]


class ValidationConfig(BaseModel):
    """Validation configuration."""
    holdout_years: int
    validation_sources: list[str]


class Phase3Config(BaseModel):
    """Phase 3 configuration."""
    prediction: PredictionConfig
    validation: ValidationConfig


class GeneralConfig(BaseModel):
    """General configuration."""
    logging: Dict[str, Any]
    cache: Dict[str, Any]
    parallel: Dict[str, Any]
    ai_api: Dict[str, Any]


class LitKGConfig(BaseModel):
    """Main configuration class."""
    phase1: Phase1Config
    phase2: Phase2Config
    phase3: Phase3Config
    general: GeneralConfig


def load_config(config_path: Optional[str] = None) -> LitKGConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Loaded configuration object.
    """
    if config_path is None:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Load environment variables for API keys
    if 'ANTHROPIC_API_KEY' in os.environ:
        config_data['general']['ai_api']['anthropic']['api_key'] = os.environ['ANTHROPIC_API_KEY']
    
    if 'OPENAI_API_KEY' in os.environ:
        config_data['general']['ai_api']['openai']['api_key'] = os.environ['OPENAI_API_KEY']
    
    if 'PUBMED_API_KEY' in os.environ:
        config_data['phase1']['literature']['pubmed']['api_key'] = os.environ['PUBMED_API_KEY']
    
    if 'UMLS_API_KEY' in os.environ:
        config_data['phase1']['knowledge_graphs']['ontologies']['umls']['api_key'] = os.environ['UMLS_API_KEY']
    
    return LitKGConfig(**config_data)


def get_data_dir() -> Path:
    """Get the data directory path."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data"


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir