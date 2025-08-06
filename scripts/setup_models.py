#!/usr/bin/env python3
"""
Setup script for downloading and installing required biomedical models.

This script:
1. Downloads and installs scispacy models
2. Verifies model installations
3. Sets up cache directories
4. Tests basic functionality
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.utils.logging import setup_logging


def run_command(command, description):
    """Run a shell command and handle errors."""
    logger = setup_logging()
    logger.info(f"Running: {description}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def install_scispacy_models():
    """Install scispacy models."""
    logger = setup_logging()
    logger.info("Installing scispacy models...")
    
    models = [
        "en_core_sci_sm",
        "en_core_sci_md",
        "en_ner_bc5cdr_md",
        "en_ner_bionlp13cg_md"
    ]
    
    for model in models:
        url = f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/{model}-0.5.3.tar.gz"
        command = f"pip install {url}"
        description = f"Installing {model}"
        
        if not run_command(command, description):
            logger.warning(f"Failed to install {model}, continuing...")
    
    return True


def verify_installations():
    """Verify that all models are properly installed."""
    logger = setup_logging()
    logger.info("Verifying model installations...")
    
    # Test scispacy
    try:
        import spacy
        nlp = spacy.load("en_core_sci_md")
        doc = nlp("BRCA1 mutations cause breast cancer.")
        logger.info(f"scispacy test: Found {len(doc.ents)} entities")
        
        for ent in doc.ents:
            logger.info(f"  {ent.text}: {ent.label_}")
        
    except Exception as e:
        logger.error(f"scispacy verification failed: {e}")
        return False
    
    # Test transformers models
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test PubMedBERT
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        model = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        logger.info("PubMedBERT loaded successfully")
        
        # Test BioBERT
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        logger.info("BioBERT loaded successfully")
        
    except Exception as e:
        logger.error(f"Transformers model verification failed: {e}")
        return False
    
    logger.info("All model verifications passed!")
    return True


def setup_directories():
    """Create necessary directories."""
    logger = setup_logging()
    logger.info("Setting up directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "cache",
        "logs",
        "models",
        "results"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True


def create_environment_template():
    """Create a template .env file."""
    logger = setup_logging()
    env_path = project_root / ".env"
    
    if env_path.exists():
        logger.info(".env file already exists, skipping...")
        return True
    
    env_content = """# LitKG-Integrate Environment Variables

# PubMed API (Required)
PUBMED_EMAIL=your-email@domain.com
PUBMED_API_KEY=your-api-key-here

# AI API Keys (Anthropic Claude preferred)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# UMLS API (Optional but recommended)
UMLS_API_KEY=your-umls-api-key-here

# Database URLs (if using external databases)
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

MONGODB_URL=mongodb://localhost:27017/litkg
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info("Created .env template file")
    logger.info("Please edit .env with your actual API keys and credentials")
    
    return True


def test_basic_functionality():
    """Test basic functionality of the system."""
    logger = setup_logging()
    logger.info("Testing basic functionality...")
    
    try:
        # Import and test configuration
        from litkg.utils.config import load_config
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Test biomedical NLP components
        from litkg.phase1.literature_processor import BiomedicalNLP
        nlp = BiomedicalNLP(config)
        
        # Test entity extraction
        test_text = "BRCA1 mutations are associated with increased risk of breast cancer."
        entities = nlp.extract_entities(test_text)
        
        logger.info(f"Entity extraction test: Found {len(entities)} entities")
        for entity in entities:
            logger.info(f"  {entity.text}: {entity.label} (confidence: {entity.confidence:.2f})")
        
        logger.info("Basic functionality test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False


def main():
    """Main setup function."""
    logger = setup_logging(level="INFO", log_file="setup.log")
    logger.info("Starting LitKG-Integrate setup...")
    
    # Step 1: Setup directories
    if not setup_directories():
        logger.error("Directory setup failed")
        return 1
    
    # Step 2: Create environment template
    if not create_environment_template():
        logger.error("Environment template creation failed")
        return 1
    
    # Step 3: Install scispacy models
    if not install_scispacy_models():
        logger.error("scispacy model installation failed")
        return 1
    
    # Step 4: Verify installations
    if not verify_installations():
        logger.error("Model verification failed")
        return 1
    
    # Step 5: Test basic functionality
    if not test_basic_functionality():
        logger.error("Basic functionality test failed")
        return 1
    
    logger.info("Setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Edit .env file with your API keys")
    logger.info("2. Run: python scripts/example_literature_processing.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())