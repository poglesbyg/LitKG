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

**Differing sequence lengths.** Literature and KG subgraphs rarely have equal
node counts, so cross-attention must track the query length and the key/value
length separately. `MultiHeadAttention` derives `tgt_len` from the query and
`src_len` from the key — an implementation that reuses one length for both
crashes on essentially every real cross-modal batch.

**Masks.** Nonzero entries are kept and zeros are masked out.
`CrossModalAttention` takes boolean padding masks over nodes where `True`
means *ignore*, and inverts them at the boundary. Masks are broadcast to
`[batch, heads, tgt_len, src_len]`, accepting key-padding or full attention
mask shapes.

**Output dimension.** `output_dim` adds a projection head over node
embeddings; when it equals `hidden_dim` the head is an identity. Returned
`lit_node_embeddings` and `kg_node_embeddings` are projected to it.

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
    node_weight * node_embedding_loss +
    contrastive_weight * contrastive_alignment_loss
)
```

Only the terms whose targets are present contribute. A batch supervising just
node embeddings produces only that term plus the contrastive term.

If **no** term applies — predictions and targets share no supervised task —
the loss raises rather than returning a bare `0`, which would fail later and
less informatively at `.backward()`.

### Batch format

`GNNTrainer` accepts two layouts, normalized by `_unpack_batch`:

```python
# Graph objects
{"lit_graph": Data(...), "kg_graph": Data(...), "labels": {...}}

# Flat tensors
{"lit_x": ..., "lit_edge_index": ..., "kg_x": ..., "kg_edge_index": ..., "labels": ...}
```

`labels` may be a dict of task name to target, or a bare tensor. A bare tensor
is interpreted as a node-embedding regression target, since that is the only
task whose target has one row per literature node.

### Contrastive Learning

- Encourages aligned entities to have similar representations
- Pushes non-aligned entities apart in embedding space
- Improves cross-modal understanding

The alignment matrix must match the embeddings being contrasted. Since the
contrastive term operates on **graph** embeddings, the matrix is sized on the
number of graphs in the batch, not the node count. Either the positive or
negative set can be empty for a single-graph batch, so those terms are dropped
rather than producing `nan` from an empty mean.

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

Extraction uses a `MultiGraph`, not a simple `Graph`. Edge features one-hot
encode `relation_type`, so collapsing parallel edges would keep only the last
relation between a pair of entities and hide the others from the model — two
entities joined by both `ASSOCIATED_WITH` and `MUTATED_IN` are two distinct
claims. Undirected is acceptable here because the PyTorch Geometric conversion
emits both directions explicitly.

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

## HybridGNNModel on the real graph

The cross-modal architecture this phase is named for had only ever run on
`torch.randn`. Wiring it to real data required fixing something first, and the
result is worth stating plainly: **it scores at chance.**

### It could not express a per-pair prediction

Fusion combined the two graphs at the *graph* level -- `lit_outputs`
and `kg_outputs` were pooled to one vector each before cross-attention -- so
`fused_representation` had exactly one row. `entity_pairs` indexes that tensor,
so it could only ever reference row 0. The synthetic demo passed `[[0, 0]]` with
the comment "only use valid indices"; that comment was load-bearing.

Cross-attention already preserved the node dimension and already returned
`kg_enhanced` at full length, so the fix was to fuse node embeddings rather than
pooled ones and index the enhanced KG nodes. `fused_representation` is now one
row per KG node, and the graph-level vector remains available as
`graph_fused_representation`.

### Measured against the same harness

`HybridGNNLinkPredictor` implements the standard `LinkPredictor` interface, so
it runs through the identical split, negatives and metrics as everything else.
Literature side: 162 nodes and 257 edges from the Phase 1 output.

| model | AUC (3 seeds) |
|---|---|
| GNN + evidence-weighted L3 + text | **0.744 ± 0.006** |
| HybridGNNModel (cross-modal) | **0.492 ± 0.020** |

Chance is 0.5. The architecture the project is built around does not beat a coin
flip on the task, while a much simpler model reaches 0.744.

### Why: the representations are degenerate

Every node collapses to the same vector. Mean pairwise cosine between distinct
nodes:

| stage | cosine |
|---|---|
| raw KG input features | 0.793 |
| after the KG encoder | 0.998 |
| after cross-modal fusion | **1.000** |

Two things were tried and neither helped:

- **A learned per-node embedding**, concatenated to the static features. The
  simpler baseline carries one, so withholding it would not have been a fair
  test. Score stayed at chance (0.492).
- **Fewer message-passing layers** (1, 2, 3). Cosine remained 1.000 at every
  depth, so this is not ordinary depth-driven over-smoothing.

The collapse is therefore located but not fully explained. It happens inside the
KG encoder and is completed by fusion, it survives both a per-node identity
signal and a single-layer encoder, and the convolution itself carries residual
connections. Finding the exact cause is a separate investigation.

### What this means

The comparison is fair -- same split, same negatives, same metrics, same loss,
comparable hyperparameters -- and the answer is that cross-modal attention as
implemented here contributes nothing on this graph. Anyone planning to build on
`HybridGNNModel` should fix the collapse first and re-measure against 0.744,
rather than assume the architecture works because it is elaborate.
