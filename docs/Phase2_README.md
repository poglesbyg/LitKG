# Phase 2: Hybrid GNN Architecture Documentation

## Overview

Phase 2 implements the core hybrid Graph Neural Network (GNN) architecture that integrates literature-derived graphs with knowledge graph subgraphs using advanced cross-modal attention mechanisms. This enables the model to learn joint representations for novel knowledge discovery.

## Architecture Components

### 1. Literature Graph Encoder (`LiteratureGraphEncoder`)

Processes graphs constructed from literature where nodes are biomedical entities (genes, diseases, drugs) and edges represent co-occurrence or extracted relations.

**Key Features:**
- Multi-layer graph convolution with residual connections
- Temporal attention for publication date weighting
- Entity type-aware processing
- Confidence scoring integration

**Input:**
- Node features: [embedding (768D), type one-hot (10D), confidence, frequency]
- Edge features: [weight, type one-hot (10D), confidence]
- Optional temporal features for publication dates

### 2. Knowledge Graph Encoder (`KnowledgeGraphEncoder`)

Processes structured knowledge graphs (CIVIC, TCGA, CPTAC) with validated biological relationships.

**Key Features:**
- Relation-aware graph convolution
- Structural attention for different entity types
- Evidence-based edge weighting
- Multi-head attention for relation types

**Input:**
- Node features: [embedding (768D), type one-hot (10D), confidence, centrality]
- Edge features: [confidence, type one-hot (10D), evidence count]
- Relation type embeddings

### 3. Cross-Modal Fusion (`CrossModalFusion`)

Integrates literature and knowledge graph representations using cross-attention mechanisms.

**Fusion Strategies:**
- **Attention-based**: Multi-head cross-attention with learned weights
- **Gating**: Learned gates control information flow between modalities
- **Concatenation**: Simple concatenation with projection layers

**Key Features:**
- Bidirectional cross-attention (lit→KG and KG→lit)
- Multiple fusion layers for deep integration
- Residual connections preserve original information

### 4. Relation Predictor (`RelationPredictor`)

Predicts relations between entity pairs and estimates confidence.

**Capabilities:**
- **Link Prediction**: Binary classification for entity relationships
- **Relation Classification**: Multi-class prediction of relation types
- **Confidence Estimation**: Reliability scores for predictions

## Advanced Attention Mechanisms

### Cross-Modal Attention
```python
# Literature entities attend to KG entities and vice versa
lit_enhanced, kg_enhanced, attention_weights = cross_attention(
    lit_features, kg_features
)
```

### Structural Attention
- Weights nodes based on entity types and graph structure
- Considers centrality and neighborhood importance
- Adapts to different biological entity roles

### Temporal Attention
- Incorporates publication dates for literature relevance
- Implements temporal decay for recency weighting
- Learns time-dependent patterns in scientific knowledge

### Adaptive Attention
- Combines multiple attention mechanisms
- Learns optimal weighting for different contexts
- Adapts to various graph structures and domains

## Training Infrastructure

### Multi-Task Loss Function
```python
total_loss = (
    link_weight * link_prediction_loss +
    relation_weight * relation_classification_loss +
    confidence_weight * confidence_estimation_loss +
    contrastive_weight * contrastive_alignment_loss
)
```

### Contrastive Learning
- Encourages aligned entities to have similar representations
- Pushes non-aligned entities apart in embedding space
- Improves cross-modal understanding

### Evaluation Metrics
- **Link Prediction**: Accuracy, Precision, Recall, F1, AUC
- **Relation Classification**: Multi-class accuracy, macro/weighted F1
- **Confidence Estimation**: MAE, MSE, correlation

## Graph Construction Pipeline

### Literature Graph Construction
1. **Entity Extraction**: Extract biomedical entities from documents
2. **Co-occurrence Analysis**: Count entity co-occurrences across papers
3. **Relation Extraction**: Identify explicit relationships
4. **Semantic Similarity**: Add edges based on embedding similarity
5. **Graph Assembly**: Create PyTorch Geometric Data objects

### Knowledge Graph Subgraph Extraction
1. **Target Entity Identification**: Find relevant entities from literature
2. **K-hop Neighborhood**: Extract local subgraphs around targets
3. **Filtering**: Apply confidence thresholds and size limits
4. **Feature Engineering**: Create node and edge embeddings

### Entity Alignment
1. **Entity Linking**: Use Phase 1 linking results
2. **Semantic Matching**: Compute embedding similarities
3. **Confidence Scoring**: Estimate alignment reliability
4. **Alignment Matrix**: Create training supervision

## Usage Examples

### Basic Model Usage
```python
from litkg.phase2 import HybridGNNModel

# Initialize model
model = HybridGNNModel(
    lit_node_dim=768 + 10 + 2,
    lit_edge_dim=1 + 10 + 1,
    kg_node_dim=768 + 10 + 2,
    kg_edge_dim=1 + 10 + 1,
    kg_relation_dim=10,
    hidden_dim=256,
    num_heads=8
)

# Forward pass
outputs = model(
    lit_x=lit_x,
    lit_edge_index=lit_edge_index,
    lit_edge_attr=lit_edge_attr,
    kg_x=kg_x,
    kg_edge_index=kg_edge_index,
    kg_edge_attr=kg_edge_attr,
    kg_relation_types=kg_relation_types,
    entity_pairs=entity_pairs
)
```

### Training Setup
```python
from litkg.phase2 import HybridGNNTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    hidden_dim=256,
    num_gnn_layers=3,
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=100
)

# Initialize trainer
trainer = HybridGNNTrainer(config)

# Train model
history = trainer.train(train_loader, val_loader)
```

### Graph Construction
```python
from litkg.phase2 import GraphConstructor
from litkg.models.embeddings import BiomedicalEmbeddings

# Initialize components
embedder = BiomedicalEmbeddings(config)
constructor = GraphConstructor(embedder, output_dir="graphs/")

# Construct training graphs
training_examples = constructor.construct_training_graphs(
    literature_data=documents,
    kg_data=knowledge_graph,
    batch_size=10
)
```

## Performance Optimization

### Memory Efficiency
- **Gradient Checkpointing**: Reduce memory usage during training
- **Mixed Precision**: Use FP16 for faster training
- **Graph Batching**: Group similar-sized graphs for efficiency

### Computational Efficiency
- **Sparse Operations**: Leverage graph sparsity
- **Attention Caching**: Cache attention patterns when possible
- **Parallel Processing**: Multi-GPU training support

### Scalability
- **Subgraph Sampling**: Sample subgraphs for large graphs
- **Hierarchical Training**: Train on graph hierarchies
- **Distributed Training**: Scale across multiple nodes

## Model Variants

### Architecture Variants
- **Shallow**: 2 GNN layers, 4 attention heads (fast)
- **Standard**: 3 GNN layers, 8 attention heads (balanced)
- **Deep**: 5 GNN layers, 16 attention heads (high capacity)

### Attention Variants
- **Basic**: Simple cross-attention
- **Enhanced**: Structural + temporal attention
- **Adaptive**: Learned attention combination

### Fusion Strategies
- **Early**: Fuse at input level
- **Middle**: Fuse at hidden layers
- **Late**: Fuse at output level

## Evaluation and Validation

### Intrinsic Evaluation
- **Link Prediction**: Predict missing edges in test graphs
- **Relation Classification**: Classify relation types
- **Confidence Calibration**: Evaluate confidence accuracy

### Extrinsic Evaluation
- **Novel Discovery**: Identify new literature-KG connections
- **Hypothesis Generation**: Generate testable hypotheses
- **Knowledge Completion**: Complete partial knowledge graphs

### Benchmarking
- **Baseline Comparisons**: Compare against simpler models
- **Ablation Studies**: Evaluate component contributions
- **Cross-domain Transfer**: Test generalization ability

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size or model dimensions
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Training Instability**
   - Adjust learning rate and warmup
   - Add gradient clipping
   - Check data preprocessing

3. **Poor Convergence**
   - Verify loss function weights
   - Check data quality and alignment
   - Adjust model architecture

### Debugging Tools
- **Attention Visualization**: Plot attention patterns
- **Gradient Analysis**: Monitor gradient flow
- **Loss Decomposition**: Track individual loss components

## Future Enhancements

### Planned Features
- **Dynamic Graphs**: Handle evolving graph structures
- **Multi-scale Attention**: Attention across different scales
- **Causal Discovery**: Identify causal relationships

### Research Directions
- **Few-shot Learning**: Adapt to new domains quickly
- **Explainable AI**: Interpret model decisions
- **Active Learning**: Select informative training examples

## References and Citations

### Key Papers
- **Graph Attention Networks**: Veličković et al. (2018)
- **Cross-modal Learning**: Baltrusaitis et al. (2019)
- **Biomedical Knowledge Graphs**: Himmelstein et al. (2017)

### Implementation References
- PyTorch Geometric documentation
- HuggingFace Transformers library
- NetworkX graph analysis library

---

For more details, see the complete implementation in `src/litkg/phase2/` and run the demo with `make run-phase2`.