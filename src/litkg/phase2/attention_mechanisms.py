"""
Advanced attention mechanisms for cross-modal fusion.

This module implements various attention mechanisms specifically designed
for integrating literature and knowledge graph representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Dropout, LayerNorm
from typing import Optional, Tuple, Dict, List
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with optional positional encoding.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections
        self.w_q = Linear(d_model, d_model, bias=bias)
        self.w_k = Linear(d_model, d_model, bias=bias)
        self.w_v = Linear(d_model, d_model, bias=bias)
        self.w_o = Linear(d_model, d_model, bias=bias)
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attended output [batch_size, seq_len, d_model]
            attention_weights: Attention weights if return_attention=True
        """
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        # Apply mask if provided
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
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between literature and knowledge graph representations.
    
    This mechanism allows literature representations to attend to KG representations
    and vice versa, enabling information exchange between modalities.
    """
    
    def __init__(
        self,
        lit_dim: int,
        kg_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_gating: bool = True
    ):
        super().__init__()
        
        self.lit_dim = lit_dim
        self.kg_dim = kg_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_gating = use_gating
        
        # Project to common dimension
        self.lit_proj = Linear(lit_dim, hidden_dim)
        self.kg_proj = Linear(kg_dim, hidden_dim)
        
        # Cross-attention: literature -> knowledge graph
        self.lit_to_kg_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature
        )
        
        # Cross-attention: knowledge graph -> literature
        self.kg_to_lit_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature
        )
        
        # Gating mechanism for controlled fusion
        if use_gating:
            self.lit_gate = nn.Sequential(
                Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.kg_gate = nn.Sequential(
                Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Output projections
        self.lit_output = Linear(hidden_dim, lit_dim)
        self.kg_output = Linear(hidden_dim, kg_dim)
        
        # Layer normalization
        self.lit_norm = LayerNorm(lit_dim)
        self.kg_norm = LayerNorm(kg_dim)
    
    def forward(
        self,
        lit_features: torch.Tensor,
        kg_features: torch.Tensor,
        lit_mask: Optional[torch.Tensor] = None,
        kg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            lit_features: Literature features [batch_size, lit_seq_len, lit_dim]
            kg_features: KG features [batch_size, kg_seq_len, kg_dim]
            lit_mask: Literature attention mask
            kg_mask: KG attention mask
            
        Returns:
            lit_enhanced: Enhanced literature features
            kg_enhanced: Enhanced KG features
            attention_weights: Dictionary of attention weights
        """
        # Store original features for residual connections
        lit_residual = lit_features
        kg_residual = kg_features
        
        # Project to common dimension
        lit_proj = self.lit_proj(lit_features)
        kg_proj = self.kg_proj(kg_features)
        
        # Cross-attention: literature attends to KG
        lit_attended, lit_to_kg_weights = self.lit_to_kg_attention(
            query=lit_proj,
            key=kg_proj,
            value=kg_proj,
            mask=kg_mask.unsqueeze(1).unsqueeze(1) if kg_mask is not None else None,
            return_attention=True
        )
        
        # Cross-attention: KG attends to literature
        kg_attended, kg_to_lit_weights = self.kg_to_lit_attention(
            query=kg_proj,
            key=lit_proj,
            value=lit_proj,
            mask=lit_mask.unsqueeze(1).unsqueeze(1) if lit_mask is not None else None,
            return_attention=True
        )
        
        # Gated fusion if enabled
        if self.use_gating:
            # Literature gating
            lit_concat = torch.cat([lit_proj, lit_attended], dim=-1)
            lit_gate = self.lit_gate(lit_concat)
            lit_fused = lit_gate * lit_proj + (1 - lit_gate) * lit_attended
            
            # KG gating
            kg_concat = torch.cat([kg_proj, kg_attended], dim=-1)
            kg_gate = self.kg_gate(kg_concat)
            kg_fused = kg_gate * kg_proj + (1 - kg_gate) * kg_attended
        else:
            lit_fused = lit_proj + lit_attended
            kg_fused = kg_proj + kg_attended
        
        # Output projections and residual connections
        lit_enhanced = self.lit_output(lit_fused) + lit_residual
        kg_enhanced = self.kg_output(kg_fused) + kg_residual
        
        # Layer normalization
        lit_enhanced = self.lit_norm(lit_enhanced)
        kg_enhanced = self.kg_norm(kg_enhanced)
        
        # Collect attention weights
        attention_weights = {
            'lit_to_kg': lit_to_kg_weights,
            'kg_to_lit': kg_to_lit_weights
        }
        
        return lit_enhanced, kg_enhanced, attention_weights


class StructuralAttention(nn.Module):
    """
    Structural attention mechanism that considers entity types and graph structure.
    
    This attention mechanism weights nodes based on their structural role
    and entity type in the knowledge graph.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_entity_types: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_entity_types = num_entity_types
        
        # Entity type embeddings
        self.entity_type_embeddings = nn.Embedding(num_entity_types, hidden_dim)
        
        # Structural attention parameters
        self.structural_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Type-aware projection
        self.type_projection = Linear(hidden_dim * 2, hidden_dim)
        
        # Output normalization
        self.layer_norm = LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        entity_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply structural attention based on entity types.
        
        Args:
            node_features: Node features [num_nodes, hidden_dim]
            entity_types: Entity type indices [num_nodes]
            
        Returns:
            Enhanced node features with structural attention
        """
        # Get entity type embeddings
        type_embeddings = self.entity_type_embeddings(entity_types)
        
        # Combine node features with type information
        combined_features = torch.cat([node_features, type_embeddings], dim=-1)
        enhanced_features = self.type_projection(combined_features)
        
        # Add sequence dimension for attention
        enhanced_features = enhanced_features.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # Apply structural attention
        attended_features, attention_weights = self.structural_attention(
            enhanced_features, enhanced_features, enhanced_features
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(0)
        
        # Residual connection and normalization
        output = self.layer_norm(attended_features + node_features)
        
        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for incorporating publication dates and time-based patterns.
    
    This mechanism allows the model to weight information based on recency
    and temporal relevance in literature.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        max_time_steps: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_time_steps = max_time_steps
        
        # Temporal positional encoding
        self.temporal_encoding = nn.Embedding(max_time_steps, hidden_dim)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal decay parameter
        self.decay_rate = Parameter(torch.tensor(0.1))
        
        # Output normalization
        self.layer_norm = LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        timestamps: torch.Tensor,
        current_time: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply temporal attention based on timestamps.
        
        Args:
            node_features: Node features [num_nodes, hidden_dim]
            timestamps: Timestamp indices [num_nodes]
            current_time: Current timestamp for decay calculation
            
        Returns:
            Temporally weighted node features
        """
        # Get temporal encodings
        temporal_encodings = self.temporal_encoding(timestamps)
        
        # Combine node features with temporal information
        temporal_features = node_features + temporal_encodings
        
        # Calculate temporal decay weights if current_time provided
        if current_time is not None:
            time_diffs = current_time - timestamps.float()
            decay_weights = torch.exp(-self.decay_rate * time_diffs)
            decay_weights = decay_weights.unsqueeze(-1)  # [num_nodes, 1]
            temporal_features = temporal_features * decay_weights
        
        # Add sequence dimension for attention
        temporal_features = temporal_features.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(0)
        
        # Residual connection and normalization
        output = self.layer_norm(attended_features + node_features)
        
        return output


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with multiple fusion strategies.
    
    This is an advanced version of cross-modal attention that supports
    different fusion strategies and attention mechanisms.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_strategy: str = "additive",
        use_position_bias: bool = True
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy
        self.use_position_bias = use_position_bias
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Input projections
        self.query_proj = Linear(query_dim, hidden_dim)
        self.key_proj = Linear(key_dim, hidden_dim)
        self.value_proj = Linear(value_dim, hidden_dim)
        
        # Fusion strategies
        if fusion_strategy == "additive":
            self.fusion = nn.Sequential(
                Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
        elif fusion_strategy == "multiplicative":
            self.fusion = nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
        elif fusion_strategy == "gated":
            self.gate = nn.Sequential(
                Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Position bias
        if use_position_bias:
            self.position_bias = Parameter(torch.randn(num_heads, 1, 1))
        
        # Output projection
        self.output_proj = Linear(hidden_dim, query_dim)
        self.layer_norm = LayerNorm(query_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, query_dim]
            key: Key tensor [batch_size, seq_len_k, key_dim]
            value: Value tensor [batch_size, seq_len_v, value_dim]
            mask: Attention mask
            
        Returns:
            output: Attended output
            attention_weights: Attention weights
        """
        residual = query
        
        # Project inputs
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        # Cross-attention
        attended, attention_weights = self.cross_attention(
            query=q, key=k, value=v, attn_mask=mask
        )
        
        # Fusion strategy
        if self.fusion_strategy == "additive":
            combined = torch.cat([q, attended], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_strategy == "multiplicative":
            fused = self.fusion(q) * attended
        elif self.fusion_strategy == "gated":
            gate = self.gate(torch.cat([q, attended], dim=-1))
            fused = gate * q + (1 - gate) * attended
        else:  # Default: simple addition
            fused = q + attended
        
        # Output projection and residual connection
        output = self.output_proj(fused) + residual
        output = self.layer_norm(output)
        
        return output, attention_weights


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that learns to weight different attention types.
    
    This mechanism combines multiple attention mechanisms and learns
    optimal weights for different contexts.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_attention_types: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_attention_types = num_attention_types
        
        # Different attention mechanisms
        self.content_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.structural_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=0.5  # Different temperature
        )
        
        self.positional_attention = MultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=2.0  # Different temperature
        )
        
        # Adaptive weighting network
        self.attention_weights = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, num_attention_types),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = LayerNorm(hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply adaptive attention mechanism.
        
        Args:
            features: Input features [batch_size, seq_len, hidden_dim]
            mask: Attention mask
            
        Returns:
            output: Adaptively attended features
            weights: Dictionary of attention weights
        """
        # Apply different attention mechanisms
        content_out, content_attn = self.content_attention(
            features, features, features, mask, return_attention=True
        )
        
        structural_out, structural_attn = self.structural_attention(
            features, features, features, mask, return_attention=True
        )
        
        positional_out, positional_attn = self.positional_attention(
            features, features, features, mask, return_attention=True
        )
        
        # Learn adaptive weights
        global_context = features.mean(dim=1)  # [batch_size, hidden_dim]
        adaptive_weights = self.attention_weights(global_context)  # [batch_size, num_attention_types]
        
        # Weighted combination
        attended_outputs = torch.stack([content_out, structural_out, positional_out], dim=-1)
        # [batch_size, seq_len, hidden_dim, num_attention_types]
        
        adaptive_weights = adaptive_weights.unsqueeze(1).unsqueeze(1)
        # [batch_size, 1, 1, num_attention_types]
        
        output = (attended_outputs * adaptive_weights).sum(dim=-1)
        output = self.layer_norm(output + features)
        
        attention_weights = {
            'content': content_attn,
            'structural': structural_attn,
            'positional': positional_attn,
            'adaptive_weights': adaptive_weights.squeeze()
        }
        
        return output, attention_weights