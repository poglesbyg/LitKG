"""
Hybrid GNN Architecture for Literature-Knowledge Graph Integration.

This module implements the core hybrid GNN model that processes both
literature-derived graphs and knowledge graphs, using cross-modal attention
to learn joint representations for novel knowledge discovery.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Dropout, LayerNorm, ModuleList
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax, add_self_loops, degree
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from ..utils.logging import LoggerMixin
from .attention_mechanisms import CrossModalAttention, StructuralAttention


class GraphConvolutionLayer(MessagePassing):
    """
    Enhanced graph convolution layer with edge features and residual connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        dropout: float = 0.1,
        bias: bool = True,
        residual: bool = True,
        **kwargs
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.residual = residual
        
        # Node transformations
        self.lin_node = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        
        # Attention mechanism
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        
        # Output projection
        self.lin_out = Linear(heads * out_channels, out_channels, bias=bias)
        
        # Residual connection
        if residual and in_channels != out_channels:
            self.lin_residual = Linear(in_channels, out_channels, bias=False)
        elif residual:
            self.lin_residual = nn.Identity()
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        num_nodes = x.size(0)
        
        # Store input for residual connection
        residual = x
        
        # Transform node features
        x = self.lin_node(x).view(-1, self.heads, self.out_channels)
        
        # Transform edge features
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))
        
        # Reshape and project output
        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_out(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual:
            out = out + self.lin_residual(residual)
        
        # Layer normalization
        out = self.layer_norm(out)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, edge_index_i: torch.Tensor) -> torch.Tensor:
        """Compute messages between nodes."""
        # Compute attention scores
        alpha_src = (x_j * self.att_src).sum(dim=-1)  # [num_edges, heads]
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)  # [num_edges, heads]
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)  # [num_edges, heads]
        
        alpha = alpha_src + alpha_dst + alpha_edge
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        
        # Apply attention to messages
        out = (x_j + edge_attr) * alpha.unsqueeze(-1)
        return out


class LiteratureGraphEncoder(nn.Module, LoggerMixin):
    """
    Encoder for literature-derived graphs.
    
    Processes graphs constructed from literature where nodes are entities
    (genes, diseases, drugs) and edges represent co-occurrence or extracted relations.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        use_temporal: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_temporal = use_temporal
        
        # Input projection
        self.node_projection = Linear(node_dim, hidden_dim)
        self.edge_projection = Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = ModuleList()
        for i in range(num_layers):
            layer = GraphConvolutionLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                residual=True
            )
            self.conv_layers.append(layer)
        
        # Temporal attention (for publication dates)
        if use_temporal:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.output_projection = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through literature graph encoder.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            temporal_features: Temporal features [num_nodes, temporal_dim]
            
        Returns:
            Dictionary with node embeddings and graph-level representation
        """
        # Project input features
        h = self.node_projection(x)
        edge_features = self.edge_projection(edge_attr)
        
        # Store intermediate representations
        layer_outputs = []
        
        # Graph convolution layers
        for conv_layer in self.conv_layers:
            h = conv_layer(h, edge_index, edge_features)
            layer_outputs.append(h)
        
        # Temporal attention if enabled
        if self.use_temporal and temporal_features is not None:
            # Add sequence dimension for attention
            h_temporal = h.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            
            # Apply temporal attention
            h_attended, temporal_weights = self.temporal_attention(
                h_temporal, h_temporal, h_temporal
            )
            h = h_attended.squeeze(1) + h  # Residual connection
        
        # Final projection
        node_embeddings = self.output_projection(h)
        
        # Graph-level pooling
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'layer_outputs': layer_outputs
        }


class KnowledgeGraphEncoder(nn.Module, LoggerMixin):
    """
    Encoder for knowledge graphs.
    
    Processes structured knowledge graphs (CIVIC, TCGA, CPTAC) where nodes
    are biological entities and edges represent validated relationships.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        relation_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        use_relation_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_relation_attention = use_relation_attention
        
        # Input projections
        self.node_projection = Linear(node_dim, hidden_dim)
        self.edge_projection = Linear(edge_dim, hidden_dim)
        self.relation_projection = Linear(relation_dim, hidden_dim)
        
        # Relation-aware graph convolution layers
        self.conv_layers = ModuleList()
        for i in range(num_layers):
            layer = GraphConvolutionLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                residual=True
            )
            self.conv_layers.append(layer)
        
        # Relation attention mechanism
        if use_relation_attention:
            self.relation_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Structural attention for different entity types
        self.structural_attention = StructuralAttention(
            hidden_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        relation_types: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through knowledge graph encoder.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            relation_types: Relation type embeddings [num_edges, relation_dim]
            batch: Batch assignment [num_nodes]
            entity_types: Entity type indicators [num_nodes]
            
        Returns:
            Dictionary with node embeddings and graph-level representation
        """
        # Project input features
        h = self.node_projection(x)
        edge_features = self.edge_projection(edge_attr)
        relation_features = self.relation_projection(relation_types)
        
        # Combine edge and relation features
        combined_edge_features = edge_features + relation_features
        
        # Store intermediate representations
        layer_outputs = []
        
        # Graph convolution layers
        for conv_layer in self.conv_layers:
            h = conv_layer(h, edge_index, combined_edge_features)
            layer_outputs.append(h)
        
        # Relation attention
        if self.use_relation_attention:
            h_relation = h.unsqueeze(1)  # Add sequence dimension
            h_attended, relation_weights = self.relation_attention(
                h_relation, h_relation, h_relation
            )
            h = h_attended.squeeze(1) + h  # Residual connection
        
        # Structural attention for entity types
        if entity_types is not None:
            h = self.structural_attention(h, entity_types)
        
        # Final projection
        node_embeddings = self.output_projection(h)
        
        # Graph-level pooling
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'layer_outputs': layer_outputs
        }


class CrossModalFusion(nn.Module, LoggerMixin):
    """
    Cross-modal fusion module that integrates literature and KG representations.
    
    Uses cross-attention mechanisms to align and fuse information from both modalities.
    """
    
    def __init__(
        self,
        lit_dim: int,
        kg_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_strategy: str = "attention",
        **kwargs
    ):
        super().__init__()
        
        self.lit_dim = lit_dim
        self.kg_dim = kg_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fusion_strategy = fusion_strategy
        
        # Project to common dimension
        self.lit_projection = Linear(lit_dim, hidden_dim)
        self.kg_projection = Linear(kg_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.cross_attention_layers = ModuleList()
        for _ in range(num_layers):
            cross_attn = CrossModalAttention(
                lit_dim=hidden_dim,
                kg_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.cross_attention_layers.append(cross_attn)
        
        # Fusion strategies
        if fusion_strategy == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_strategy == "gating":
            self.fusion_gate = nn.Sequential(
                Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        elif fusion_strategy == "concat":
            self.fusion_projection = Linear(hidden_dim * 2, hidden_dim)
        
        # Output normalization
        self.output_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        
    def forward(
        self,
        lit_embeddings: torch.Tensor,
        kg_embeddings: torch.Tensor,
        lit_mask: Optional[torch.Tensor] = None,
        kg_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse literature and knowledge graph representations.
        
        Args:
            lit_embeddings: Literature embeddings [batch_size, seq_len, lit_dim]
            kg_embeddings: KG embeddings [batch_size, seq_len, kg_dim]
            lit_mask: Literature attention mask
            kg_mask: KG attention mask
            
        Returns:
            Dictionary with fused representations and attention weights
        """
        # Project to common dimension
        lit_proj = self.lit_projection(lit_embeddings)
        kg_proj = self.kg_projection(kg_embeddings)
        
        # Store attention weights
        attention_weights = []
        
        # Apply cross-modal attention layers
        for cross_attn_layer in self.cross_attention_layers:
            lit_enhanced, kg_enhanced, attn_weights = cross_attn_layer(
                lit_proj, kg_proj, lit_mask, kg_mask
            )
            
            # Update projections with enhanced representations
            lit_proj = lit_enhanced
            kg_proj = kg_enhanced
            attention_weights.append(attn_weights)
        
        # Fusion strategy
        if self.fusion_strategy == "attention":
            # Concatenate and apply attention
            combined = torch.cat([lit_proj, kg_proj], dim=1)  # [batch, 2*seq_len, hidden]
            fused, fusion_weights = self.fusion_attention(combined, combined, combined)
            fused = fused.mean(dim=1)  # Global pooling
            
        elif self.fusion_strategy == "gating":
            # Gated fusion
            gate = self.fusion_gate(torch.cat([lit_proj.mean(1), kg_proj.mean(1)], dim=-1))
            fused = gate * lit_proj.mean(1) + (1 - gate) * kg_proj.mean(1)
            
        elif self.fusion_strategy == "concat":
            # Simple concatenation and projection
            fused = self.fusion_projection(
                torch.cat([lit_proj.mean(1), kg_proj.mean(1)], dim=-1)
            )
            
        else:
            # Default: simple addition
            fused = lit_proj.mean(1) + kg_proj.mean(1)
        
        # Apply output normalization and dropout
        fused = self.output_norm(fused)
        fused = self.dropout(fused)
        
        return {
            'fused_representation': fused,
            'lit_enhanced': lit_proj,
            'kg_enhanced': kg_proj,
            'attention_weights': attention_weights
        }


class RelationPredictor(nn.Module):
    """
    Relation prediction head for link prediction and knowledge discovery.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_relations: int = 10,
        dropout: float = 0.1,
        use_confidence: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.use_confidence = use_confidence
        
        # Relation classification layers
        self.relation_classifier = nn.Sequential(
            Linear(input_dim * 2, hidden_dim),  # Concatenated entity representations
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_relations)
        )
        
        # Confidence estimation
        if use_confidence:
            self.confidence_estimator = nn.Sequential(
                Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                Dropout(dropout),
                Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Link prediction (binary classification)
        self.link_predictor = nn.Sequential(
            Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        entity1_embeddings: torch.Tensor,
        entity2_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict relations between entity pairs.
        
        Args:
            entity1_embeddings: First entity embeddings [batch_size, input_dim]
            entity2_embeddings: Second entity embeddings [batch_size, input_dim]
            
        Returns:
            Dictionary with relation predictions, link probabilities, and confidence
        """
        # Concatenate entity representations
        combined = torch.cat([entity1_embeddings, entity2_embeddings], dim=-1)
        
        # Relation classification
        relation_logits = self.relation_classifier(combined)
        relation_probs = F.softmax(relation_logits, dim=-1)
        
        # Link prediction
        link_probs = self.link_predictor(combined)
        
        results = {
            'relation_logits': relation_logits,
            'relation_probs': relation_probs,
            'link_probs': link_probs
        }
        
        # Confidence estimation
        if self.use_confidence:
            confidence = self.confidence_estimator(combined)
            results['confidence'] = confidence
        
        return results


class HybridGNNModel(nn.Module, LoggerMixin):
    """
    Main hybrid GNN model that integrates literature and knowledge graphs.
    
    This model combines literature graph encoder, knowledge graph encoder,
    cross-modal fusion, and relation prediction for novel knowledge discovery.
    """
    
    def __init__(
        self,
        # Literature graph parameters
        lit_node_dim: int,
        lit_edge_dim: int,
        
        # Knowledge graph parameters
        kg_node_dim: int,
        kg_edge_dim: int,
        kg_relation_dim: int,
        
        # Model architecture parameters
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        
        # Task-specific parameters
        num_relations: int = 10,
        fusion_strategy: str = "attention",
        use_temporal: bool = True,
        use_confidence: bool = True,
        
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        
        # Literature graph encoder
        self.lit_encoder = LiteratureGraphEncoder(
            node_dim=lit_node_dim,
            edge_dim=lit_edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            heads=num_heads,
            dropout=dropout,
            use_temporal=use_temporal
        )
        
        # Knowledge graph encoder
        self.kg_encoder = KnowledgeGraphEncoder(
            node_dim=kg_node_dim,
            edge_dim=kg_edge_dim,
            relation_dim=kg_relation_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            lit_dim=hidden_dim,
            kg_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
            fusion_strategy=fusion_strategy
        )
        
        # Relation predictor
        self.relation_predictor = RelationPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            dropout=dropout,
            use_confidence=use_confidence
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, Parameter):
            torch.nn.init.xavier_uniform_(module)
    
    def forward(
        self,
        # Literature graph inputs
        lit_x: torch.Tensor,
        lit_edge_index: torch.Tensor,
        lit_edge_attr: torch.Tensor,
        lit_batch: Optional[torch.Tensor] = None,
        
        # Knowledge graph inputs
        kg_x: torch.Tensor,
        kg_edge_index: torch.Tensor,
        kg_edge_attr: torch.Tensor,
        kg_relation_types: torch.Tensor,
        kg_batch: Optional[torch.Tensor] = None,
        
        # Entity pairs for relation prediction
        entity_pairs: Optional[torch.Tensor] = None,
        
        # Optional features
        temporal_features: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid GNN model.
        
        Args:
            lit_x: Literature node features
            lit_edge_index: Literature edge indices
            lit_edge_attr: Literature edge features
            lit_batch: Literature batch assignment
            kg_x: KG node features
            kg_edge_index: KG edge indices
            kg_edge_attr: KG edge features
            kg_relation_types: KG relation types
            kg_batch: KG batch assignment
            entity_pairs: Entity pairs for relation prediction [num_pairs, 2]
            temporal_features: Temporal features
            entity_types: Entity type indicators
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode literature graph
        lit_outputs = self.lit_encoder(
            x=lit_x,
            edge_index=lit_edge_index,
            edge_attr=lit_edge_attr,
            batch=lit_batch,
            temporal_features=temporal_features
        )
        
        # Encode knowledge graph
        kg_outputs = self.kg_encoder(
            x=kg_x,
            edge_index=kg_edge_index,
            edge_attr=kg_edge_attr,
            relation_types=kg_relation_types,
            batch=kg_batch,
            entity_types=entity_types
        )
        
        # Cross-modal fusion
        # Add sequence dimension for attention
        lit_embeddings = lit_outputs['graph_embedding'].unsqueeze(1)
        kg_embeddings = kg_outputs['graph_embedding'].unsqueeze(1)
        
        fusion_outputs = self.fusion(
            lit_embeddings=lit_embeddings,
            kg_embeddings=kg_embeddings
        )
        
        # Prepare outputs
        outputs = {
            'lit_node_embeddings': lit_outputs['node_embeddings'],
            'kg_node_embeddings': kg_outputs['node_embeddings'],
            'lit_graph_embedding': lit_outputs['graph_embedding'],
            'kg_graph_embedding': kg_outputs['graph_embedding'],
            'fused_representation': fusion_outputs['fused_representation']
        }
        
        # Relation prediction if entity pairs provided
        if entity_pairs is not None:
            # Get entity embeddings for the pairs
            entity1_embeddings = fusion_outputs['fused_representation'][entity_pairs[:, 0]]
            entity2_embeddings = fusion_outputs['fused_representation'][entity_pairs[:, 1]]
            
            relation_outputs = self.relation_predictor(
                entity1_embeddings, entity2_embeddings
            )
            
            outputs.update(relation_outputs)
        
        # Include attention weights if requested
        if return_attention:
            outputs['attention_weights'] = fusion_outputs['attention_weights']
        
        return outputs
    
    def predict_relations(
        self,
        lit_data: Data,
        kg_data: Data,
        entity_pairs: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Predict relations for given entity pairs.
        
        Args:
            lit_data: Literature graph data
            kg_data: Knowledge graph data
            entity_pairs: Entity pairs to predict relations for
            threshold: Threshold for link prediction
            
        Returns:
            Prediction results with confidence scores
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                lit_x=lit_data.x,
                lit_edge_index=lit_data.edge_index,
                lit_edge_attr=lit_data.edge_attr,
                lit_batch=getattr(lit_data, 'batch', None),
                kg_x=kg_data.x,
                kg_edge_index=kg_data.edge_index,
                kg_edge_attr=kg_data.edge_attr,
                kg_relation_types=kg_data.relation_types,
                kg_batch=getattr(kg_data, 'batch', None),
                entity_pairs=entity_pairs
            )
            
            # Binary predictions
            link_predictions = (outputs['link_probs'] > threshold).long()
            
            # Relation predictions
            relation_predictions = outputs['relation_probs'].argmax(dim=-1)
            
            results = {
                'link_predictions': link_predictions,
                'link_probabilities': outputs['link_probs'],
                'relation_predictions': relation_predictions,
                'relation_probabilities': outputs['relation_probs']
            }
            
            if 'confidence' in outputs:
                results['confidence'] = outputs['confidence']
            
            return results