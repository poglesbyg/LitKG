"""
PyTorch models for graph neural networks and multi-modal learning.

This module contains custom PyTorch models for Phase 2 of LitKG-Integrate,
including hybrid GNNs and cross-modal attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Dropout, LayerNorm
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from ..utils.logging import LoggerMixin


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = Linear(d_model, d_model, bias=bias)
        self.w_k = Linear(d_model, d_model, bias=bias)
        self.w_v = Linear(d_model, d_model, bias=bias)
        self.w_o = Linear(d_model, d_model, bias=bias)
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection and residual connection
        output = self.w_o(context)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention between literature and knowledge graph representations."""
    
    def __init__(
        self,
        lit_dim: int,
        kg_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lit_dim = lit_dim
        self.kg_dim = kg_dim
        self.hidden_dim = hidden_dim
        
        # Project inputs to common dimension
        self.lit_proj = Linear(lit_dim, hidden_dim)
        self.kg_proj = Linear(kg_dim, hidden_dim)
        
        # Cross-attention layers
        self.lit_to_kg_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.kg_to_lit_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )
        
        # Output projections
        self.lit_output = Linear(hidden_dim, lit_dim)
        self.kg_output = Linear(hidden_dim, kg_dim)
        
    def forward(
        self,
        lit_features: torch.Tensor,
        kg_features: torch.Tensor,
        lit_mask: Optional[torch.Tensor] = None,
        kg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            lit_features: [batch_size, lit_seq_len, lit_dim]
            kg_features: [batch_size, kg_seq_len, kg_dim]
            lit_mask: [batch_size, lit_seq_len]
            kg_mask: [batch_size, kg_seq_len]
        
        Returns:
            lit_enhanced: Enhanced literature features
            kg_enhanced: Enhanced KG features
            attention_weights: Dictionary of attention weights
        """
        # Project to common dimension
        lit_proj = self.lit_proj(lit_features)
        kg_proj = self.kg_proj(kg_features)
        
        # Cross-attention: literature attends to KG
        lit_attended, lit_to_kg_attn = self.lit_to_kg_attention(
            query=lit_proj,
            key=kg_proj,
            value=kg_proj,
            mask=kg_mask.unsqueeze(1).unsqueeze(1) if kg_mask is not None else None
        )
        
        # Cross-attention: KG attends to literature
        kg_attended, kg_to_lit_attn = self.kg_to_lit_attention(
            query=kg_proj,
            key=lit_proj,
            value=lit_proj,
            mask=lit_mask.unsqueeze(1).unsqueeze(1) if lit_mask is not None else None
        )
        
        # Fusion
        lit_fused = self.fusion(torch.cat([lit_proj, lit_attended], dim=-1))
        kg_fused = self.fusion(torch.cat([kg_proj, kg_attended], dim=-1))
        
        # Output projections
        lit_enhanced = self.lit_output(lit_fused) + lit_features
        kg_enhanced = self.kg_output(kg_fused) + kg_features
        
        attention_weights = {
            "lit_to_kg": lit_to_kg_attn,
            "kg_to_lit": kg_to_lit_attn
        }
        
        return lit_enhanced, kg_enhanced, attention_weights


class GraphConvLayer(MessagePassing):
    """Custom graph convolution layer with edge features."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node transformation
        self.lin_node = Linear(in_channels, out_channels, bias=False)
        
        # Edge transformation
        self.lin_edge = Linear(edge_dim, out_channels, bias=False)
        
        # Message transformation
        self.lin_msg = Linear(out_channels * 2, out_channels, bias=bias)
        
        # Activation and normalization
        self.activation = nn.ReLU()
        self.layer_norm = LayerNorm(out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform node features
        x_transformed = self.lin_node(x)
        
        # Transform edge features
        edge_transformed = self.lin_edge(edge_attr)
        
        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x_transformed,
            edge_attr=edge_transformed,
            size=None
        )
        
        # Apply layer norm and activation
        out = self.layer_norm(out)
        out = self.activation(out)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Create messages between nodes."""
        # Combine source node, target node, and edge features
        msg = torch.cat([x_i, x_j], dim=-1)
        msg = self.lin_msg(msg)
        
        # Weight by edge features
        msg = msg * edge_attr
        
        return msg


class HybridGNN(nn.Module, LoggerMixin):
    """
    Hybrid Graph Neural Network for literature-KG integration.
    
    This model processes both literature subgraphs and KG subgraphs,
    then uses cross-modal attention to integrate information.
    """
    
    def __init__(
        self,
        lit_node_dim: int,
        kg_node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling: str = "mean",
        **kwargs
    ):
        super().__init__()
        
        self.lit_node_dim = lit_node_dim
        self.kg_node_dim = kg_node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Literature subgraph encoder
        self.lit_encoder = self._build_encoder(lit_node_dim, hidden_dim, num_layers, dropout)
        
        # KG subgraph encoder
        self.kg_encoder = self._build_encoder(kg_node_dim, hidden_dim, num_layers, dropout)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            lit_dim=hidden_dim,
            kg_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_encoder(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float
    ) -> nn.ModuleList:
        """Build graph encoder layers."""
        layers = nn.ModuleList()
        
        # Input layer
        layers.append(GraphConvLayer(input_dim, hidden_dim, self.edge_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(GraphConvLayer(hidden_dim, hidden_dim, self.edge_dim))
            layers.append(Dropout(dropout))
        
        return layers
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        lit_data: Data,
        kg_data: Data,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid GNN.
        
        Args:
            lit_data: Literature graph data
            kg_data: Knowledge graph data
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing predictions and optionally attention weights
        """
        # Encode literature subgraph
        lit_x = lit_data.x
        for layer in self.lit_encoder:
            if isinstance(layer, GraphConvLayer):
                lit_x = layer(lit_x, lit_data.edge_index, lit_data.edge_attr)
            else:  # Dropout
                lit_x = layer(lit_x)
        
        # Encode KG subgraph
        kg_x = kg_data.x
        for layer in self.kg_encoder:
            if isinstance(layer, GraphConvLayer):
                kg_x = layer(kg_x, kg_data.edge_index, kg_data.edge_attr)
            else:  # Dropout
                kg_x = layer(kg_x)
        
        # Pool node features to graph-level representations
        lit_graph = self._pool_nodes(lit_x, lit_data.batch)
        kg_graph = self._pool_nodes(kg_x, kg_data.batch)
        
        # Add sequence dimension for attention
        lit_graph = lit_graph.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        kg_graph = kg_graph.unsqueeze(1)    # [batch_size, 1, hidden_dim]
        
        # Cross-modal attention
        lit_enhanced, kg_enhanced, attention_weights = self.cross_attention(
            lit_graph, kg_graph
        )
        
        # Remove sequence dimension
        lit_enhanced = lit_enhanced.squeeze(1)
        kg_enhanced = kg_enhanced.squeeze(1)
        
        # Combine representations
        combined = torch.cat([lit_enhanced, kg_enhanced], dim=-1)
        
        # Final prediction
        logits = self.classifier(combined)
        probs = torch.sigmoid(logits)
        
        results = {
            "logits": logits,
            "probabilities": probs,
            "lit_representation": lit_enhanced,
            "kg_representation": kg_enhanced
        }
        
        if return_attention:
            results["attention_weights"] = attention_weights
        
        return results
    
    def _pool_nodes(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph-level representation."""
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def predict_links(
        self,
        lit_data: Data,
        kg_data: Data,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Predict links between literature and KG entities."""
        self.eval()
        
        with torch.no_grad():
            results = self.forward(lit_data, kg_data)
            
            predictions = (results["probabilities"] > threshold).long()
            
            return {
                "predictions": predictions,
                "probabilities": results["probabilities"],
                "confidence": torch.abs(results["probabilities"] - 0.5) * 2
            }


class GraphNeuralNetwork(nn.Module):
    """General-purpose Graph Neural Network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        
        # Build convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(self._get_conv_layer(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._get_conv_layer(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(self._get_conv_layer(hidden_dim, output_dim))
        
        self.dropout = Dropout(dropout)
        self.activation = nn.ReLU()
        
    def _get_conv_layer(self, in_dim: int, out_dim: int):
        """Get convolution layer based on type."""
        if self.conv_type == "gcn":
            return GCNConv(in_dim, out_dim)
        elif self.conv_type == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        elif self.conv_type == "sage":
            return SAGEConv(in_dim, out_dim)
        elif self.conv_type == "transformer":
            return TransformerConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass."""
        # Apply convolution layers
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer
        if len(self.convs) > 1:
            x = self.convs[-1](x, edge_index)
        
        return x


# Model factory functions
def create_hybrid_gnn(config: Dict[str, Any]) -> HybridGNN:
    """Create a hybrid GNN model from configuration."""
    return HybridGNN(**config)


def create_graph_neural_network(config: Dict[str, Any]) -> GraphNeuralNetwork:
    """Create a GNN model from configuration."""
    return GraphNeuralNetwork(**config)