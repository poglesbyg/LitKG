"""
Test configuration and fixtures for LitKG test suite.

Provides common fixtures, test data, and configuration for all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

# Test data and fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data_dir": "test_data",
        "cache_dir": "test_cache",
        "logs_dir": "test_logs",
        "models": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "biomedical_model": "dmis-lab/biobert-base-cased-v1.1"
        },
        "database": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password"
        }
    }


@pytest.fixture
def sample_biomedical_text():
    """Sample biomedical text for testing."""
    return """
    BRCA1 (Breast Cancer 1) is a tumor suppressor gene that plays a critical role 
    in DNA repair through homologous recombination. Mutations in BRCA1 significantly 
    increase the risk of breast and ovarian cancers. BRCA1-deficient tumors are 
    particularly sensitive to PARP inhibitors due to synthetic lethality.
    """


@pytest.fixture
def sample_entities():
    """Sample biomedical entities for testing."""
    return [
        {"text": "BRCA1", "label": "GENE", "start": 0, "end": 5},
        {"text": "breast cancer", "label": "DISEASE", "start": 10, "end": 23},
        {"text": "DNA repair", "label": "PROCESS", "start": 30, "end": 40},
        {"text": "PARP inhibitors", "label": "DRUG", "start": 50, "end": 65}
    ]


@pytest.fixture
def sample_relations():
    """Sample biomedical relations for testing."""
    return [
        {"head": "BRCA1", "relation": "ASSOCIATED_WITH", "tail": "breast cancer", "confidence": 0.95},
        {"head": "BRCA1", "relation": "INVOLVED_IN", "tail": "DNA repair", "confidence": 0.90},
        {"head": "PARP inhibitors", "relation": "TREATS", "tail": "BRCA1-deficient tumors", "confidence": 0.85}
    ]


@pytest.fixture
def sample_knowledge_graph():
    """Sample knowledge graph data for testing."""
    return {
        "nodes": [
            {"id": "BRCA1", "type": "gene", "properties": {"name": "BRCA1", "chromosome": "17"}},
            {"id": "TP53", "type": "gene", "properties": {"name": "TP53", "chromosome": "17"}},
            {"id": "breast_cancer", "type": "disease", "properties": {"name": "Breast Cancer"}},
            {"id": "ovarian_cancer", "type": "disease", "properties": {"name": "Ovarian Cancer"}}
        ],
        "edges": [
            {"source": "BRCA1", "target": "breast_cancer", "type": "ASSOCIATED_WITH", "weight": 0.95},
            {"source": "BRCA1", "target": "ovarian_cancer", "type": "ASSOCIATED_WITH", "weight": 0.90},
            {"source": "BRCA1", "target": "TP53", "type": "INTERACTS_WITH", "weight": 0.75}
        ]
    }


@pytest.fixture
def sample_literature_data():
    """Sample literature data for testing."""
    return [
        {
            "pmid": "12345678",
            "title": "BRCA1 mutations in breast cancer susceptibility",
            "abstract": "This study investigates the role of BRCA1 mutations in breast cancer...",
            "authors": ["Smith J", "Johnson A"],
            "journal": "Nature Genetics",
            "year": 2023,
            "doi": "10.1038/ng.2023.123"
        },
        {
            "pmid": "87654321",
            "title": "PARP inhibitors in BRCA-deficient tumors",
            "abstract": "PARP inhibitors show promising results in treating BRCA-deficient tumors...",
            "authors": ["Brown K", "Davis L"],
            "journal": "Cell",
            "year": 2023,
            "doi": "10.1016/j.cell.2023.456"
        }
    ]


@pytest.fixture
def sample_tensor_data():
    """Sample tensor data for PyTorch model testing."""
    return {
        "node_features": torch.randn(10, 64),
        "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
        "edge_attr": torch.randn(5, 32),
        "batch": torch.zeros(10, dtype=torch.long)
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a mock response from the LLM about BRCA1 and breast cancer.",
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        "model": "mock-model",
        "confidence": 0.85
    }


@pytest.fixture
def mock_embedding():
    """Mock embedding for testing."""
    return np.random.rand(768).astype(np.float32)


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.similarity_search.return_value = [
        Mock(page_content="BRCA1 is a tumor suppressor gene", metadata={"source": "test1"}),
        Mock(page_content="PARP inhibitors are effective in BRCA1-deficient tumors", metadata={"source": "test2"})
    ]
    mock_store.add_documents.return_value = None
    return mock_store


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for testing."""
    mock_manager = Mock()
    mock_manager.process_biomedical_task.return_value = Mock(
        content="Mock biomedical response",
        usage={"total_tokens": 100},
        confidence=0.8
    )
    return mock_manager


@pytest.fixture
def sample_hypothesis():
    """Sample biomedical hypothesis for testing."""
    return {
        "hypothesis": "BRCA1 mutations lead to defective DNA repair, making tumors sensitive to PARP inhibitors",
        "evidence": ["BRCA1 is involved in homologous recombination", "PARP inhibitors cause synthetic lethality"],
        "confidence": 0.85,
        "domain": "cancer_genetics",
        "testable_predictions": [
            "BRCA1-mutated cells will be more sensitive to PARP inhibitors than wild-type cells",
            "DNA repair capacity will be reduced in BRCA1-deficient cells"
        ]
    }


@pytest.fixture
def sample_experimental_design():
    """Sample experimental design for testing."""
    return {
        "objective": "Test PARP inhibitor sensitivity in BRCA1-deficient cells",
        "experimental_groups": [
            {"name": "BRCA1-WT", "description": "Wild-type BRCA1 cells"},
            {"name": "BRCA1-MUT", "description": "BRCA1-deficient cells"}
        ],
        "treatments": ["Vehicle control", "PARP inhibitor (10μM)", "PARP inhibitor (100μM)"],
        "measurements": ["Cell viability", "DNA damage markers", "Apoptosis"],
        "controls": ["Positive control", "Negative control"],
        "statistical_analysis": "Two-way ANOVA with Tukey post-hoc"
    }


# Test utilities
def create_mock_document(content: str, metadata: Dict[str, Any] = None):
    """Create a mock document for testing."""
    mock_doc = Mock()
    mock_doc.page_content = content
    mock_doc.metadata = metadata or {}
    return mock_doc


def create_sample_graph_data(num_nodes: int = 10, num_edges: int = 15):
    """Create sample graph data for testing."""
    import networkx as nx
    
    G = nx.erdos_renyi_graph(num_nodes, num_edges / (num_nodes * (num_nodes - 1) / 2))
    
    # Add node attributes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['type'] = 'gene' if i % 2 == 0 else 'disease'
        G.nodes[node]['name'] = f'Entity_{i}'
    
    # Add edge attributes
    for edge in G.edges():
        G.edges[edge]['weight'] = np.random.random()
        G.edges[edge]['type'] = 'ASSOCIATED_WITH'
    
    return G


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu


# Skip conditions
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


def skip_if_no_internet():
    """Skip test if no internet connection."""
    try:
        import requests
        requests.get("https://www.google.com", timeout=5)
        return False
    except:
        return pytest.mark.skip(reason="No internet connection")


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_MODELS_DIR = Path(__file__).parent / "test_models"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)