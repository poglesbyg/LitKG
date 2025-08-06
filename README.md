# LitKG-Integrate: Literature-Augmented Knowledge Graph Discovery

## Core Objective
Build a system that dynamically integrates biomedical literature with structured knowledge graphs from large public datasets (CIVIC, TCGA, CPTAC) to enable novel knowledge discovery and hypothesis generation.

## Technical Architecture

### Phase 1: Foundation (Weeks 1-4)
- **Literature Processing Pipeline**: Use biomedical LLMs (PubMedBERT, BioBERT) to extract entities, relations, and contexts from PubMed abstracts/full-text
- **KG Preprocessing**: Standardize and harmonize entities across CIVIC, TCGA, CPTAC using ontology mapping (UMLS, Gene Ontology)
- **Entity Linking**: Implement fuzzy matching and disambiguation to connect literature mentions with KG nodes

### Phase 2: Integration Engine (Weeks 5-8)
- **Hybrid GNN Architecture**:
  - Literature subgraphs (paper-gene-disease-drug networks)
  - Experimental data subgraphs (mutation-expression-outcome networks)
  - Cross-modal attention mechanisms to weight literature vs. experimental evidence
- **Confidence Scoring**: Develop metrics to assess reliability of literature-derived vs. data-derived relationships

### Phase 3: Discovery & Validation (Weeks 9-12)
- **Novel Relation Prediction**: Use the integrated graph to predict missing edges
- **Hypothesis Generation**: Identify literature patterns that suggest new experimental targets
- **Validation Pipeline**: Cross-validate predictions against held-out recent publications

## Key Innovation Areas
- Multi-modal Graph Learning
- Temporal Dynamics
- Uncertainty Quantification

## Success Metrics
- Precision/recall on link prediction tasks
- Novel biomarker discovery validation against recent literature
- Computational efficiency on large-scale integration

## Dataset Strategy
Start with cancer genomics (TCGA + CIVIC + cancer literature subset) for focused validation, then expand.

## Current Status

✅ **Phase 1 Complete**: Literature processing, KG preprocessing, and entity linking
🚧 **Phase 2 In Progress**: Hybrid GNN architecture design
⏳ **Phase 3 Planned**: Novel relation prediction and validation

## Installation and Setup

### Prerequisites
- Python 3.9+ 
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start with uv (Recommended)
```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Quick setup (installs dependencies, models, creates .env template)
make quickstart

# 3. Edit .env with your API keys
# Edit the created .env file with your actual API keys

# 4. Run Phase 1 integration
make run-phase1
```

### Alternative Setup with pip
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup models and environment
python scripts/setup_models.py

# 3. Configure API keys
cp env.template .env
# Edit .env with your API keys

# 4. Run Phase 1 integration
python scripts/phase1_integration.py
```

### Development Setup
```bash
# Install with development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

### Detailed Setup
See [Phase 1 Documentation](docs/Phase1_README.md) for comprehensive setup instructions.

## Usage

### Complete Pipeline (Recommended)
```bash
# With Make (recommended)
make run-phase1

# Or directly with uv
uv run python scripts/phase1_integration.py

# Note: CLI now fully functional with dependency compatibility fixes
```

### Individual Components
```bash
# With Make
make run-lit     # Literature processing only
make run-kg      # Knowledge graph preprocessing only  
make run-link    # Entity linking only

# CLI coming in next update

# Or directly
uv run python scripts/example_literature_processing.py
uv run python scripts/example_kg_preprocessing.py
uv run python scripts/example_entity_linking.py
```

### Programmatic Usage
```python
from litkg.phase1 import LiteratureProcessor, KGPreprocessor, EntityLinker

# Initialize components
lit_processor = LiteratureProcessor()
kg_processor = KGPreprocessor()
entity_linker = EntityLinker()

# Process literature
documents = lit_processor.process_query("BRCA1 breast cancer", max_results=100)

# Process knowledge graphs  
kg_processor.download_all_data()
kg_processor.process_all_data()

# Link entities
entity_linker.load_kg_entities(kg_processor)
results = entity_linker.batch_link_documents(documents)
```

### Command Line Interface
```bash
# Show CLI help
make cli-help

# Setup models and environment
make cli-setup

# Run complete Phase 1 pipeline
make cli-phase1

# Or use directly:
uv run python scripts/litkg_cli.py --help
uv run python scripts/litkg_cli.py literature --query "BRCA1 breast cancer" --max-results 10
uv run python scripts/litkg_cli.py kg --sources civic tcga
```

## Documentation

- [Phase 1 Documentation](docs/Phase1_README.md) - Complete guide to literature processing, KG preprocessing, and entity linking
- [Configuration Guide](config/config.yaml) - Detailed configuration options
- [API Reference](docs/API.md) - Programmatic interface documentation

## Output Files

Phase 1 generates structured outputs in `data/processed/`:
- `phase1_integrated_dataset.json` - Complete integrated dataset
- `phase2_graph_data.json` - Graph structure ready for GNN training
- Individual component outputs for analysis and debugging