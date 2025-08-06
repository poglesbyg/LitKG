#!/usr/bin/env python3
"""
Example script demonstrating HuggingFace and PyTorch integration.

This script shows how to:
1. Use biomedical transformer models from HuggingFace
2. Generate and cache embeddings
3. Create custom PyTorch models for graph learning
4. Combine literature and knowledge graph representations
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.models.huggingface_models import (
    BiomedicalModelManager, ModelRegistry, get_available_models
)
from litkg.models.embeddings import BiomedicalEmbeddings
from litkg.models.pytorch_models import HybridGNN, CrossModalAttention
from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging


def demonstrate_huggingface_models():
    """Demonstrate HuggingFace biomedical models."""
    logger = setup_logging()
    logger.info("=== HUGGINGFACE BIOMEDICAL MODELS DEMO ===")
    
    # Load configuration
    config = load_config()
    
    # Initialize model manager
    model_manager = BiomedicalModelManager(config)
    
    # Show available models
    logger.info("Available biomedical models:")
    models = get_available_models()
    for model in models[:5]:  # Show first 5
        logger.info(f"  {model.name}: {model.description}")
        logger.info(f"    Tasks: {', '.join(model.tasks)}")
        logger.info(f"    Domain: {model.domain}, Size: {model.size}")
        if model.citation:
            logger.info(f"    Citation: {model.citation}")
        logger.info("")
    
    # Example biomedical texts
    sample_texts = [
        "BRCA1 mutations are associated with increased breast cancer risk.",
        "TP53 is a tumor suppressor gene frequently mutated in cancer.",
        "Immunotherapy with checkpoint inhibitors shows promise in melanoma.",
        "COVID-19 patients may develop acute respiratory distress syndrome.",
        "Alzheimer's disease is characterized by amyloid beta plaques."
    ]
    
    # 1. Generate embeddings with PubMedBERT
    logger.info("1. Generating embeddings with PubMedBERT...")
    try:
        embeddings = model_manager.get_embeddings(
            sample_texts,
            model_name="pubmedbert",
            batch_size=2
        )
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        logger.info(f"Sample embedding (first 10 dims): {embeddings[0][:10].tolist()}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
    
    # 2. Named Entity Recognition
    logger.info("\n2. Named Entity Recognition with BioBERT...")
    try:
        entities_batch = model_manager.extract_entities_batch(
            sample_texts[:3],  # First 3 texts
            model_name="biobert",
            batch_size=2
        )
        
        for i, (text, entities) in enumerate(zip(sample_texts[:3], entities_batch)):
            logger.info(f"Text {i+1}: {text[:50]}...")
            for entity in entities:
                logger.info(f"  {entity['word']} ({entity['entity_group']}): {entity['score']:.3f}")
    except Exception as e:
        logger.error(f"Error in NER: {e}")
    
    # 3. Text Classification
    logger.info("\n3. Text Classification...")
    try:
        # Define biomedical categories
        labels = ["oncology", "cardiology", "neurology", "infectious_disease", "immunology"]
        
        classification_results = model_manager.classify_texts(
            sample_texts[:3],
            model_name="pubmedbert",
            labels=labels,
            batch_size=2
        )
        
        for i, (text, result) in enumerate(zip(sample_texts[:3], classification_results)):
            logger.info(f"Text {i+1}: {text[:50]}...")
            logger.info(f"  Top prediction: {result['labels'][0]} ({result['scores'][0]:.3f})")
    except Exception as e:
        logger.error(f"Error in classification: {e}")
    
    # 4. Question Answering
    logger.info("\n4. Biomedical Question Answering...")
    try:
        questions = [
            "What gene is associated with breast cancer?",
            "What is TP53?",
            "What treatment shows promise in melanoma?"
        ]
        
        contexts = sample_texts[:3]
        
        qa_results = model_manager.answer_questions(
            questions,
            contexts,
            model_name="pubmedbert",
            batch_size=2
        )
        
        for i, (question, context, result) in enumerate(zip(questions, contexts, qa_results)):
            logger.info(f"Question {i+1}: {question}")
            logger.info(f"  Context: {context[:50]}...")
            logger.info(f"  Answer: {result['answer']} (confidence: {result['score']:.3f})")
    except Exception as e:
        logger.error(f"Error in QA: {e}")
    
    # Show model information
    logger.info("\n5. Model Manager Information:")
    info = model_manager.get_model_info()
    logger.info(f"Device: {info['device']}")
    logger.info(f"Cached models: {info['cached_models']}")
    logger.info(f"Cached pipelines: {info['cached_pipelines']}")
    if 'memory_usage' in info:
        for key, value in info['memory_usage'].items():
            logger.info(f"  {key}: {value}")


def demonstrate_embeddings():
    """Demonstrate embedding generation and similarity."""
    logger = setup_logging()
    logger.info("\n=== BIOMEDICAL EMBEDDINGS DEMO ===")
    
    # Load configuration
    config = load_config()
    
    # Initialize embeddings
    embedder = BiomedicalEmbeddings(config, model_name="pubmedbert")
    
    # Sample biomedical entities
    entities = [
        {"text": "BRCA1", "context": "BRCA1 mutations increase breast cancer risk"},
        {"text": "TP53", "context": "TP53 is a tumor suppressor gene"},
        {"text": "EGFR", "context": "EGFR receptor is targeted in cancer therapy"},
        {"text": "breast cancer", "context": "Breast cancer affects millions of women"},
        {"text": "lung cancer", "context": "Lung cancer is often linked to smoking"}
    ]
    
    # Generate embeddings
    logger.info("Generating embeddings for biomedical entities...")
    embeddings = embedder.get_entity_embeddings(entities, include_context=True)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    # Find similar entities
    logger.info("\nFinding similar entities...")
    query_entity = "BRCA2"
    candidate_texts = [entity["text"] for entity in entities]
    
    similar_entities = embedder.find_similar_texts(
        query_entity,
        candidate_texts,
        top_k=3,
        threshold=0.3
    )
    
    logger.info(f"Entities similar to '{query_entity}':")
    for entity, similarity in similar_entities:
        logger.info(f"  {entity}: {similarity:.3f}")
    
    # Compute pairwise similarities
    logger.info("\nPairwise similarities:")
    query_embeddings = embedder.get_text_embeddings([entities[0]["text"]])
    candidate_embeddings = embedder.get_text_embeddings(candidate_texts)
    
    similarities = embedder.compute_similarity(query_embeddings, candidate_embeddings)[0]
    
    for i, (entity, sim) in enumerate(zip(candidate_texts, similarities)):
        logger.info(f"  {entities[0]['text']} <-> {entity}: {sim:.3f}")
    
    # Clustering example
    logger.info("\nClustering entities...")
    try:
        cluster_labels = embedder.cluster_embeddings(embeddings, num_clusters=3)
        
        for i, (entity, label) in enumerate(zip(entities, cluster_labels)):
            logger.info(f"  {entity['text']}: Cluster {label}")
    except Exception as e:
        logger.error(f"Error in clustering: {e}")


def demonstrate_pytorch_models():
    """Demonstrate PyTorch models for graph learning."""
    logger = setup_logging()
    logger.info("\n=== PYTORCH GRAPH MODELS DEMO ===")
    
    # Create sample data
    batch_size = 2
    lit_node_dim = 768  # BERT embedding dimension
    kg_node_dim = 256   # KG entity embedding dimension
    edge_dim = 64       # Edge feature dimension
    
    # Literature graph data (simulated)
    lit_x = torch.randn(20, lit_node_dim)  # 20 literature entities
    lit_edge_index = torch.randint(0, 20, (2, 30))  # 30 edges
    lit_edge_attr = torch.randn(30, edge_dim)
    lit_batch = torch.zeros(20, dtype=torch.long)  # Single graph
    
    # Knowledge graph data (simulated)
    kg_x = torch.randn(15, kg_node_dim)  # 15 KG entities
    kg_edge_index = torch.randint(0, 15, (2, 25))  # 25 edges
    kg_edge_attr = torch.randn(25, edge_dim)
    kg_batch = torch.zeros(15, dtype=torch.long)  # Single graph
    
    from torch_geometric.data import Data
    
    lit_data = Data(x=lit_x, edge_index=lit_edge_index, edge_attr=lit_edge_attr, batch=lit_batch)
    kg_data = Data(x=kg_x, edge_index=kg_edge_index, edge_attr=kg_edge_attr, batch=kg_batch)
    
    # 1. Cross-modal attention
    logger.info("1. Cross-modal Attention Demo...")
    try:
        cross_attention = CrossModalAttention(
            lit_dim=lit_node_dim,
            kg_dim=kg_node_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        # Add batch and sequence dimensions
        lit_features = lit_x.unsqueeze(0)  # [1, num_nodes, dim]
        kg_features = kg_x.unsqueeze(0)    # [1, num_nodes, dim]
        
        lit_enhanced, kg_enhanced, attention_weights = cross_attention(
            lit_features, kg_features
        )
        
        logger.info(f"Literature features: {lit_features.shape} -> {lit_enhanced.shape}")
        logger.info(f"KG features: {kg_features.shape} -> {kg_enhanced.shape}")
        logger.info(f"Attention weights keys: {list(attention_weights.keys())}")
        
        # Show attention patterns
        lit_to_kg_attn = attention_weights["lit_to_kg"][0, 0]  # First head, first batch
        logger.info(f"Literature->KG attention shape: {lit_to_kg_attn.shape}")
        logger.info(f"Max attention weight: {lit_to_kg_attn.max().item():.3f}")
        
    except Exception as e:
        logger.error(f"Error in cross-modal attention: {e}")
    
    # 2. Hybrid GNN
    logger.info("\n2. Hybrid GNN Demo...")
    try:
        hybrid_gnn = HybridGNN(
            lit_node_dim=lit_node_dim,
            kg_node_dim=kg_node_dim,
            edge_dim=edge_dim,
            hidden_dim=256,
            num_layers=2,
            num_heads=4
        )
        
        # Forward pass
        results = hybrid_gnn(lit_data, kg_data, return_attention=True)
        
        logger.info("Hybrid GNN Results:")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
            else:
                logger.info(f"  {key}: {type(value)}")
        
        # Show predictions
        predictions = results["probabilities"]
        logger.info(f"Prediction probabilities: {predictions.flatten().tolist()}")
        
        # Link prediction
        link_predictions = hybrid_gnn.predict_links(lit_data, kg_data, threshold=0.5)
        logger.info(f"Link predictions: {link_predictions['predictions'].flatten().tolist()}")
        logger.info(f"Confidence scores: {link_predictions['confidence'].flatten().tolist()}")
        
    except Exception as e:
        logger.error(f"Error in hybrid GNN: {e}")
    
    # 3. Model parameters
    logger.info("\n3. Model Information:")
    total_params = sum(p.numel() for p in hybrid_gnn.parameters())
    trainable_params = sum(p.numel() for p in hybrid_gnn.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / 1024**2:.2f} MB (float32)")


def demonstrate_integration_workflow():
    """Demonstrate complete integration workflow."""
    logger = setup_logging()
    logger.info("\n=== COMPLETE INTEGRATION WORKFLOW ===")
    
    # Load configuration
    config = load_config()
    
    # 1. Process biomedical texts with HuggingFace models
    logger.info("1. Processing biomedical literature...")
    
    sample_papers = [
        {
            "title": "BRCA1 mutations and breast cancer risk",
            "abstract": "BRCA1 gene mutations significantly increase the risk of developing breast and ovarian cancers. This study examines the prevalence of BRCA1 mutations in high-risk populations."
        },
        {
            "title": "TP53 pathway in cancer therapy",
            "abstract": "The TP53 tumor suppressor pathway plays a crucial role in preventing cancer development. Understanding TP53 dysfunction opens new therapeutic opportunities."
        }
    ]
    
    # Extract entities and generate embeddings
    model_manager = BiomedicalModelManager(config)
    embedder = BiomedicalEmbeddings(config)
    
    for i, paper in enumerate(sample_papers):
        text = f"{paper['title']} {paper['abstract']}"
        logger.info(f"\nPaper {i+1}: {paper['title']}")
        
        try:
            # Extract entities
            entities = model_manager.extract_entities_batch([text])[0]
            logger.info(f"Extracted {len(entities)} entities:")
            for entity in entities[:3]:  # Show first 3
                logger.info(f"  {entity['word']} ({entity['entity_group']}): {entity['score']:.3f}")
            
            # Generate embeddings
            embedding = embedder.get_text_embeddings([text])
            logger.info(f"Generated embedding shape: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"Error processing paper {i+1}: {e}")
    
    # 2. Simulate knowledge graph integration
    logger.info("\n2. Knowledge Graph Integration Simulation...")
    
    # This would normally come from the KG preprocessing
    kg_entities = [
        {"id": "GENE:BRCA1", "name": "BRCA1", "type": "GENE"},
        {"id": "GENE:TP53", "name": "TP53", "type": "GENE"},
        {"id": "DISEASE:breast_cancer", "name": "breast cancer", "type": "DISEASE"},
        {"id": "DISEASE:ovarian_cancer", "name": "ovarian cancer", "type": "DISEASE"}
    ]
    
    # Generate KG entity embeddings
    kg_texts = [f"{entity['name']} {entity['type']}" for entity in kg_entities]
    kg_embeddings = embedder.get_text_embeddings(kg_texts)
    logger.info(f"KG embeddings shape: {kg_embeddings.shape}")
    
    # 3. Entity linking simulation
    logger.info("\n3. Entity Linking Simulation...")
    
    literature_entities = ["BRCA1", "TP53", "breast cancer"]
    kg_entity_names = [entity["name"] for entity in kg_entities]
    
    for lit_entity in literature_entities:
        similar_kg = embedder.find_similar_texts(
            lit_entity,
            kg_entity_names,
            top_k=2,
            threshold=0.7
        )
        
        logger.info(f"'{lit_entity}' matches:")
        for kg_entity, similarity in similar_kg:
            logger.info(f"  {kg_entity}: {similarity:.3f}")
    
    logger.info("\n4. Ready for Phase 2 Hybrid GNN Training!")
    logger.info("Next steps:")
    logger.info("- Prepare graph datasets from Phase 1 outputs")
    logger.info("- Train hybrid GNN on literature-KG pairs")
    logger.info("- Evaluate link prediction performance")
    logger.info("- Deploy for novel knowledge discovery")


def main():
    """Main demonstration function."""
    logger = setup_logging(level="INFO")
    
    logger.info("🤖 HuggingFace & PyTorch Integration Demo for LitKG-Integrate")
    logger.info("=" * 70)
    
    try:
        # Run demonstrations
        demonstrate_huggingface_models()
        demonstrate_embeddings()
        demonstrate_pytorch_models()
        demonstrate_integration_workflow()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 All demonstrations completed successfully!")
        logger.info("\nKey Integration Points:")
        logger.info("✅ HuggingFace biomedical transformers for NLP")
        logger.info("✅ PyTorch custom models for graph learning")
        logger.info("✅ Efficient embedding generation and caching")
        logger.info("✅ Cross-modal attention mechanisms")
        logger.info("✅ End-to-end literature-KG integration")
        
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()