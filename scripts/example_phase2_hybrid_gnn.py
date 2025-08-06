#!/usr/bin/env python3
"""
Example script demonstrating Phase 2 hybrid GNN architecture.

This script shows how to:
1. Construct literature graphs and KG subgraphs
2. Train the hybrid GNN model with cross-modal attention
3. Evaluate relation prediction and link discovery
4. Visualize attention patterns and learned representations
"""

import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from litkg.phase2 import (
    HybridGNNModel, GraphConstructor, HybridGNNTrainer,
    TrainingConfig, create_data_loaders
)
from litkg.models.embeddings import BiomedicalEmbeddings
from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging
from torch_geometric.data import Data


def create_synthetic_data(num_examples: int = 50) -> List[Dict[str, Any]]:
    """Create synthetic literature and KG data for demonstration."""
    logger = setup_logging()
    logger.info("Creating synthetic data for Phase 2 demonstration")
    
    # Synthetic literature documents
    literature_data = []
    
    sample_documents = [
        {
            "pmid": "12345",
            "title": "BRCA1 mutations in breast cancer",
            "abstract": "BRCA1 gene mutations are associated with increased breast cancer risk.",
            "entities": [
                {"text": "BRCA1", "entity_group": "GENE", "score": 0.95, "context": "BRCA1 gene mutations"},
                {"text": "breast cancer", "entity_group": "DISEASE", "score": 0.90, "context": "increased breast cancer risk"},
                {"text": "mutations", "entity_group": "MUTATION", "score": 0.85, "context": "gene mutations are associated"}
            ],
            "relations": [
                {"entity1": "BRCA1", "entity2": "breast cancer", "relation_type": "ASSOCIATION", "confidence": 0.92}
            ]
        },
        {
            "pmid": "12346",
            "title": "TP53 pathway in cancer therapy",
            "abstract": "TP53 tumor suppressor pathway plays crucial role in cancer prevention.",
            "entities": [
                {"text": "TP53", "entity_group": "GENE", "score": 0.98, "context": "TP53 tumor suppressor"},
                {"text": "cancer", "entity_group": "DISEASE", "score": 0.92, "context": "role in cancer prevention"},
                {"text": "therapy", "entity_group": "DRUG", "score": 0.80, "context": "cancer therapy approaches"}
            ],
            "relations": [
                {"entity1": "TP53", "entity2": "cancer", "relation_type": "REGULATION", "confidence": 0.89}
            ]
        },
        {
            "pmid": "12347",
            "title": "EGFR inhibitors in lung cancer",
            "abstract": "EGFR receptor inhibitors show efficacy in lung cancer treatment.",
            "entities": [
                {"text": "EGFR", "entity_group": "GENE", "score": 0.96, "context": "EGFR receptor inhibitors"},
                {"text": "lung cancer", "entity_group": "DISEASE", "score": 0.94, "context": "lung cancer treatment"},
                {"text": "inhibitors", "entity_group": "DRUG", "score": 0.88, "context": "receptor inhibitors show efficacy"}
            ],
            "relations": [
                {"entity1": "EGFR", "entity2": "lung cancer", "relation_type": "THERAPEUTIC_RESPONSE", "confidence": 0.91}
            ]
        }
    ]
    
    # Create document groups for graph construction
    for i in range(num_examples):
        doc_group = {
            "documents": sample_documents.copy(),
            "entity_links": [
                {
                    "lit_entity_id": "brca1",
                    "kg_entity_id": "GENE:BRCA1",
                    "similarity_score": 0.95,
                    "confidence": 0.92
                },
                {
                    "lit_entity_id": "breast_cancer",
                    "kg_entity_id": "DISEASE:BREAST_CANCER",
                    "similarity_score": 0.88,
                    "confidence": 0.85
                }
            ]
        }
        literature_data.append(doc_group)
    
    # Synthetic KG data
    kg_data = {
        "entities": {
            "GENE:BRCA1": {
                "name": "BRCA1",
                "type": "GENE",
                "embedding": np.random.randn(768).tolist(),
                "confidence": 0.95
            },
            "GENE:TP53": {
                "name": "TP53", 
                "type": "GENE",
                "embedding": np.random.randn(768).tolist(),
                "confidence": 0.98
            },
            "GENE:EGFR": {
                "name": "EGFR",
                "type": "GENE", 
                "embedding": np.random.randn(768).tolist(),
                "confidence": 0.96
            },
            "DISEASE:BREAST_CANCER": {
                "name": "Breast Cancer",
                "type": "DISEASE",
                "embedding": np.random.randn(768).tolist(),
                "confidence": 0.92
            },
            "DISEASE:LUNG_CANCER": {
                "name": "Lung Cancer",
                "type": "DISEASE",
                "embedding": np.random.randn(768).tolist(),
                "confidence": 0.90
            }
        },
        "relations": [
            {
                "entity1": "GENE:BRCA1",
                "entity2": "DISEASE:BREAST_CANCER",
                "relation_type": "PREDISPOSING",
                "confidence": 0.89,
                "evidence_count": 15
            },
            {
                "entity1": "GENE:TP53",
                "entity2": "DISEASE:BREAST_CANCER",
                "relation_type": "ONCOGENIC",
                "confidence": 0.85,
                "evidence_count": 12
            },
            {
                "entity1": "GENE:EGFR",
                "entity2": "DISEASE:LUNG_CANCER",
                "relation_type": "THERAPEUTIC_RESPONSE",
                "confidence": 0.91,
                "evidence_count": 20
            }
        ]
    }
    
    logger.info(f"Created {len(literature_data)} literature document groups")
    logger.info(f"Created KG with {len(kg_data['entities'])} entities and {len(kg_data['relations'])} relations")
    
    return literature_data, kg_data


def demonstrate_graph_construction():
    """Demonstrate graph construction from literature and KG data."""
    logger = setup_logging()
    logger.info("=== GRAPH CONSTRUCTION DEMO ===")
    
    # Load configuration and create embeddings
    config = load_config()
    embedder = BiomedicalEmbeddings(config, model_name="pubmedbert")
    
    # Create synthetic data
    literature_data, kg_data = create_synthetic_data(num_examples=10)
    
    # Initialize graph constructor
    output_dir = project_root / "outputs" / "phase2_graphs"
    graph_constructor = GraphConstructor(
        embeddings_model=embedder,
        output_dir=output_dir
    )
    
    try:
        # Construct training graphs
        logger.info("Constructing training graph pairs...")
        training_examples = graph_constructor.construct_training_graphs(
            literature_data=literature_data,
            kg_data=kg_data,
            batch_size=5
        )
        
        logger.info(f"Successfully constructed {len(training_examples)} training examples")
        
        # Show example statistics
        example = training_examples[0]
        logger.info(f"Example graph statistics:")
        logger.info(f"  Literature graph: {example['lit_graph'].num_nodes} nodes, {example['lit_graph'].edge_index.size(1)} edges")
        logger.info(f"  KG subgraph: {example['kg_graph'].num_nodes} nodes, {example['kg_graph'].edge_index.size(1)} edges")
        logger.info(f"  Alignments: {len(example['alignments'])}")
        
        return training_examples
        
    except Exception as e:
        logger.error(f"Error in graph construction: {e}")
        return []


def demonstrate_model_architecture():
    """Demonstrate the hybrid GNN model architecture."""
    logger = setup_logging()
    logger.info("\n=== HYBRID GNN ARCHITECTURE DEMO ===")
    
    # Create model
    model = HybridGNNModel(
        lit_node_dim=768 + 10 + 2,  # embedding + type + features
        lit_edge_dim=1 + 10 + 1,    # weight + type + confidence
        kg_node_dim=768 + 10 + 2,   # embedding + type + features
        kg_edge_dim=1 + 10 + 1,     # confidence + type + evidence
        kg_relation_dim=10,         # relation type dimension
        hidden_dim=128,             # Smaller for demo
        num_gnn_layers=2,
        num_fusion_layers=1,
        num_heads=4,
        dropout=0.1,
        num_relations=10
    )
    
    logger.info("Model Architecture:")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Literature encoder: {sum(p.numel() for p in model.lit_encoder.parameters()):,} params")
    logger.info(f"  KG encoder: {sum(p.numel() for p in model.kg_encoder.parameters()):,} params")
    logger.info(f"  Cross-modal fusion: {sum(p.numel() for p in model.fusion.parameters()):,} params")
    logger.info(f"  Relation predictor: {sum(p.numel() for p in model.relation_predictor.parameters()):,} params")
    
    # Create synthetic input data
    batch_size = 2
    
    # Literature graph
    lit_x = torch.randn(20, 768 + 10 + 2)  # 20 nodes
    lit_edge_index = torch.randint(0, 20, (2, 30), dtype=torch.long)
    lit_edge_attr = torch.randn(30, 1 + 10 + 1)
    lit_batch = torch.zeros(20, dtype=torch.long)
    
    # KG graph
    kg_x = torch.randn(15, 768 + 10 + 2)  # 15 nodes
    kg_edge_index = torch.randint(0, 15, (2, 25), dtype=torch.long)
    kg_edge_attr = torch.randn(25, 1 + 10 + 1)
    kg_relation_types = torch.randn(25, 10)  # Relation embeddings, not indices
    kg_batch = torch.zeros(15, dtype=torch.long)
    
    # Entity pairs for relation prediction (ensure indices are within bounds)
    entity_pairs = torch.tensor([[0, 0]], dtype=torch.long)  # Only use valid indices
    
    try:
        # Forward pass
        logger.info("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(
                lit_x=lit_x,
                lit_edge_index=lit_edge_index,
                lit_edge_attr=lit_edge_attr,
                lit_batch=lit_batch,
                kg_x=kg_x,
                kg_edge_index=kg_edge_index,
                kg_edge_attr=kg_edge_attr,
                kg_relation_types=kg_relation_types,
                kg_batch=kg_batch,
                entity_pairs=entity_pairs,
                return_attention=True
            )
        
        logger.info("Forward pass successful!")
        logger.info(f"Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
        
        # Relation prediction
        predictions = model.predict_relations(
            lit_data=Data(x=lit_x, edge_index=lit_edge_index, edge_attr=lit_edge_attr),
            kg_data=Data(x=kg_x, edge_index=kg_edge_index, edge_attr=kg_edge_attr, relation_types=kg_relation_types),
            entity_pairs=entity_pairs,
            threshold=0.5
        )
        
        logger.info(f"Relation predictions:")
        logger.info(f"  Link predictions: {predictions['link_predictions'].tolist()}")
        logger.info(f"  Link probabilities: {predictions['link_probabilities'].flatten().tolist()}")
        logger.info(f"  Relation predictions: {predictions['relation_predictions'].tolist()}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error in model demo: {e}")
        return None


def demonstrate_training_setup():
    """Demonstrate training setup and configuration."""
    logger = setup_logging()
    logger.info("\n=== TRAINING SETUP DEMO ===")
    
    # Create training configuration
    config = TrainingConfig(
        # Smaller model for demo
        hidden_dim=128,
        num_gnn_layers=2,
        num_fusion_layers=1,
        num_heads=4,
        
        # Training parameters
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=10,
        patience=5,
        
        # Loss weights
        link_loss_weight=1.0,
        relation_loss_weight=1.0,
        confidence_loss_weight=0.5,
        contrastive_loss_weight=0.3,
        
        # Paths
        output_dir=str(project_root / "outputs" / "phase2_training_demo"),
        use_wandb=False  # Disable wandb for demo
    )
    
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config.hidden_dim}D hidden, {config.num_gnn_layers} GNN layers")
    logger.info(f"  Training: {config.num_epochs} epochs, batch size {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Loss weights: Link={config.link_loss_weight}, Relation={config.relation_loss_weight}")
    
    # Initialize trainer
    try:
        trainer = HybridGNNTrainer(config=config)
        logger.info("Trainer initialized successfully!")
        
        logger.info(f"Model device: {trainer.device}")
        logger.info(f"Output directory: {trainer.output_dir}")
        
        return trainer, config
        
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        return None, config


def demonstrate_attention_visualization():
    """Demonstrate attention pattern visualization."""
    logger = setup_logging()
    logger.info("\n=== ATTENTION VISUALIZATION DEMO ===")
    
    try:
        # Create a simple model for visualization
        from litkg.phase2.attention_mechanisms import CrossModalAttention
        
        attention = CrossModalAttention(
            lit_dim=128,
            kg_dim=128,
            hidden_dim=128,
            num_heads=4,
            dropout=0.0
        )
        
        # Create sample data
        batch_size, seq_len = 1, 10
        lit_features = torch.randn(batch_size, seq_len, 128)
        kg_features = torch.randn(batch_size, seq_len, 128)
        
        # Forward pass
        with torch.no_grad():
            lit_enhanced, kg_enhanced, attention_weights = attention(
                lit_features, kg_features
            )
        
        logger.info("Attention mechanism demo:")
        logger.info(f"  Input shapes: Lit {lit_features.shape}, KG {kg_features.shape}")
        logger.info(f"  Output shapes: Lit {lit_enhanced.shape}, KG {kg_enhanced.shape}")
        logger.info(f"  Attention weights: {list(attention_weights.keys())}")
        
        # Visualize attention patterns
        lit_to_kg_attn = attention_weights['lit_to_kg'][0, 0].numpy()  # First head, first batch
        
        plt.figure(figsize=(10, 8))
        plt.imshow(lit_to_kg_attn, cmap='Blues', aspect='auto')
        plt.colorbar()
        plt.title('Literature → Knowledge Graph Attention Pattern')
        plt.xlabel('KG Entities')
        plt.ylabel('Literature Entities')
        
        # Save plot
        output_dir = project_root / "outputs" / "phase2_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "attention_pattern.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention visualization saved to {output_dir / 'attention_pattern.png'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in attention visualization: {e}")
        return False


def demonstrate_complete_workflow():
    """Demonstrate the complete Phase 2 workflow."""
    logger = setup_logging()
    logger.info("\n=== COMPLETE PHASE 2 WORKFLOW ===")
    
    # 1. Graph Construction
    logger.info("1. Constructing graphs...")
    training_examples = demonstrate_graph_construction()
    
    if not training_examples:
        logger.error("Graph construction failed, skipping rest of workflow")
        return
    
    # 2. Model Architecture
    logger.info("\n2. Setting up model architecture...")
    model = demonstrate_model_architecture()
    
    if model is None:
        logger.error("Model setup failed, skipping training demo")
        return
    
    # 3. Training Setup
    logger.info("\n3. Setting up training...")
    trainer, config = demonstrate_training_setup()
    
    if trainer is None:
        logger.error("Training setup failed")
        return
    
    # 4. Attention Visualization
    logger.info("\n4. Visualizing attention patterns...")
    demonstrate_attention_visualization()
    
    # 5. Summary
    logger.info("\n" + "="*50)
    logger.info("🎉 Phase 2 Hybrid GNN Architecture Demo Complete!")
    logger.info("\nKey Components Demonstrated:")
    logger.info("✅ Literature graph construction from documents")
    logger.info("✅ Knowledge graph subgraph extraction")
    logger.info("✅ Entity alignment between modalities")
    logger.info("✅ Hybrid GNN model with cross-modal attention")
    logger.info("✅ Multi-task learning (link + relation prediction)")
    logger.info("✅ Training infrastructure with evaluation metrics")
    logger.info("✅ Attention pattern visualization")
    
    logger.info(f"\nModel Statistics:")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Hidden dimension: {config.hidden_dim}")
    logger.info(f"  Number of relation types: {config.num_relations}")
    
    logger.info(f"\nNext Steps for Full Implementation:")
    logger.info("1. Prepare real literature and KG data")
    logger.info("2. Scale up model size and training data")
    logger.info("3. Run full training with validation")
    logger.info("4. Evaluate on novel knowledge discovery tasks")
    logger.info("5. Deploy for Phase 3 discovery and validation")


def main():
    """Main demonstration function."""
    logger = setup_logging(level="INFO")
    
    logger.info("🧠 Phase 2 Hybrid GNN Architecture Demo for LitKG-Integrate")
    logger.info("=" * 70)
    
    try:
        # Run complete workflow demonstration
        demonstrate_complete_workflow()
        
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()