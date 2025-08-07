"""
Tests for Phase 2 components (hybrid GNN, attention mechanisms, training).
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from litkg.phase2.hybrid_gnn import HybridGNNModel, LiteratureEncoder, KnowledgeGraphEncoder
from litkg.phase2.attention_mechanisms import CrossModalAttention, MultiHeadAttention, AttentionPooling
from litkg.phase2.graph_construction import GraphBuilder, LiteratureGraphBuilder, KGGraphBuilder
from litkg.phase2.training import GNNTrainer, TrainingConfig
from litkg.phase2.data_loader import BiomedicalDataLoader, create_data_loaders


class TestHybridGNNModel:
    """Test hybrid GNN model components."""
    
    def test_hybrid_gnn_init(self):
        """Test HybridGNNModel initialization."""
        model = HybridGNNModel(
            lit_input_dim=768,
            kg_input_dim=512,
            hidden_dim=256,
            output_dim=128,
            num_heads=8
        )
        
        assert model.lit_input_dim == 768
        assert model.kg_input_dim == 512
        assert model.hidden_dim == 256
        assert model.output_dim == 128
        assert hasattr(model, 'literature_encoder')
        assert hasattr(model, 'kg_encoder')
        assert hasattr(model, 'cross_modal_attention')
    
    def test_hybrid_gnn_forward(self, sample_tensor_data):
        """Test HybridGNNModel forward pass."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        # Create sample data
        lit_x = torch.randn(10, 64)
        lit_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        kg_x = torch.randn(8, 64)
        kg_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        kg_relation_types = torch.randn(3, 32)
        
        # Forward pass
        output = model(
            lit_x=lit_x,
            lit_edge_index=lit_edge_index,
            kg_x=kg_x,
            kg_edge_index=kg_edge_index,
            kg_relation_types=kg_relation_types
        )
        
        assert output.shape[0] == 10  # Literature nodes
        assert output.shape[1] == 64  # Output dimension
    
    def test_literature_encoder(self):
        """Test LiteratureEncoder component."""
        encoder = LiteratureEncoder(input_dim=768, hidden_dim=256, num_layers=2)
        
        x = torch.randn(10, 768)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        output = encoder(x, edge_index)
        
        assert output.shape == (10, 256)
    
    def test_knowledge_graph_encoder(self):
        """Test KnowledgeGraphEncoder component."""
        encoder = KnowledgeGraphEncoder(
            node_input_dim=512,
            relation_input_dim=64,
            hidden_dim=256,
            num_layers=2
        )
        
        x = torch.randn(8, 512)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        relation_types = torch.randn(3, 64)
        
        output = encoder(x, edge_index, relation_types)
        
        assert output.shape == (8, 256)
    
    def test_model_parameters(self):
        """Test model parameter counting."""
        model = HybridGNNModel(
            lit_input_dim=768,
            kg_input_dim=512,
            hidden_dim=256,
            output_dim=128,
            num_heads=8
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
    
    @pytest.mark.gpu
    def test_model_cuda_compatibility(self):
        """Test model CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = HybridGNNModel(
            lit_input_dim=768,
            kg_input_dim=512,
            hidden_dim=256,
            output_dim=128,
            num_heads=8
        ).cuda()
        
        # Test forward pass on GPU
        lit_x = torch.randn(10, 768).cuda()
        lit_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
        kg_x = torch.randn(8, 512).cuda()
        kg_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
        kg_relation_types = torch.randn(2, 32).cuda()
        
        output = model(lit_x, lit_edge_index, kg_x, kg_edge_index, kg_relation_types)
        
        assert output.is_cuda
        assert output.shape[1] == 128


class TestAttentionMechanisms:
    """Test attention mechanism components."""
    
    def test_cross_modal_attention_init(self):
        """Test CrossModalAttention initialization."""
        attention = CrossModalAttention(
            lit_dim=256,
            kg_dim=256,
            hidden_dim=128,
            num_heads=8
        )
        
        assert attention.lit_dim == 256
        assert attention.kg_dim == 256
        assert attention.num_heads == 8
        assert hasattr(attention, 'multihead_attn')
    
    def test_cross_modal_attention_forward(self):
        """Test CrossModalAttention forward pass."""
        attention = CrossModalAttention(
            lit_dim=256,
            kg_dim=256,
            hidden_dim=128,
            num_heads=4
        )
        
        lit_features = torch.randn(10, 256)
        kg_features = torch.randn(8, 256)
        
        attended_features = attention(lit_features, kg_features)
        
        assert attended_features.shape == (10, 256)
    
    def test_multi_head_attention(self):
        """Test MultiHeadAttention component."""
        attention = MultiHeadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        x = torch.randn(10, 256)
        
        output, attention_weights = attention(x, x, x)
        
        assert output.shape == (10, 256)
        assert attention_weights.shape == (8, 10, 10)  # num_heads x seq_len x seq_len
    
    def test_attention_pooling(self):
        """Test AttentionPooling component."""
        pooling = AttentionPooling(input_dim=256, hidden_dim=128)
        
        x = torch.randn(10, 256)  # 10 nodes, 256 features
        
        pooled = pooling(x)
        
        assert pooled.shape == (1, 256)  # Pooled to single vector
    
    def test_attention_mask(self):
        """Test attention with masks."""
        attention = CrossModalAttention(
            lit_dim=256,
            kg_dim=256,
            hidden_dim=128,
            num_heads=4
        )
        
        lit_features = torch.randn(10, 256)
        kg_features = torch.randn(8, 256)
        
        # Create attention mask (mask out last 2 KG nodes)
        mask = torch.zeros(8, dtype=torch.bool)
        mask[-2:] = True
        
        attended_features = attention(lit_features, kg_features, kg_mask=mask)
        
        assert attended_features.shape == (10, 256)
    
    def test_attention_gradients(self):
        """Test attention gradient flow."""
        attention = CrossModalAttention(
            lit_dim=256,
            kg_dim=256,
            hidden_dim=128,
            num_heads=4
        )
        
        lit_features = torch.randn(10, 256, requires_grad=True)
        kg_features = torch.randn(8, 256, requires_grad=True)
        
        attended_features = attention(lit_features, kg_features)
        loss = attended_features.sum()
        loss.backward()
        
        assert lit_features.grad is not None
        assert kg_features.grad is not None


class TestGraphConstruction:
    """Test graph construction components."""
    
    def test_graph_builder_init(self):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder()
        
        assert hasattr(builder, 'logger')
    
    def test_build_literature_graph(self, sample_literature_data):
        """Test literature graph construction."""
        builder = LiteratureGraphBuilder()
        
        # Mock entity and relation extraction
        with patch.object(builder, '_extract_entities') as mock_entities:
            mock_entities.return_value = [
                {"text": "BRCA1", "label": "GENE", "start": 0, "end": 5},
                {"text": "cancer", "label": "DISEASE", "start": 10, "end": 16}
            ]
            
            with patch.object(builder, '_extract_relations') as mock_relations:
                mock_relations.return_value = [
                    {"head": "BRCA1", "relation": "ASSOCIATED_WITH", "tail": "cancer"}
                ]
                
                graph_data = builder.build_graph(sample_literature_data)
                
                assert "nodes" in graph_data
                assert "edges" in graph_data
                assert len(graph_data["nodes"]) >= 2
    
    def test_build_kg_graph(self, sample_knowledge_graph):
        """Test knowledge graph construction."""
        builder = KGGraphBuilder()
        
        graph_data = builder.build_graph(sample_knowledge_graph)
        
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert len(graph_data["nodes"]) == len(sample_knowledge_graph["nodes"])
    
    def test_graph_to_pytorch_geometric(self, sample_knowledge_graph):
        """Test conversion to PyTorch Geometric format."""
        builder = GraphBuilder()
        
        pyg_data = builder.to_pytorch_geometric(sample_knowledge_graph)
        
        assert hasattr(pyg_data, 'x')  # Node features
        assert hasattr(pyg_data, 'edge_index')  # Edge indices
        assert hasattr(pyg_data, 'edge_attr')  # Edge attributes (optional)
        
        assert pyg_data.x.shape[0] == len(sample_knowledge_graph["nodes"])
        assert pyg_data.edge_index.shape[1] == len(sample_knowledge_graph["edges"])
    
    def test_graph_statistics(self, sample_knowledge_graph):
        """Test graph statistics computation."""
        builder = GraphBuilder()
        
        stats = builder.compute_graph_stats(sample_knowledge_graph)
        
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "node_types" in stats
        assert "edge_types" in stats
        assert stats["num_nodes"] == len(sample_knowledge_graph["nodes"])
    
    def test_graph_validation(self, sample_knowledge_graph):
        """Test graph validation."""
        builder = GraphBuilder()
        
        # Valid graph
        is_valid, errors = builder.validate_graph(sample_knowledge_graph)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid graph (missing required fields)
        invalid_graph = {"nodes": [{"invalid": "node"}]}
        is_valid, errors = builder.validate_graph(invalid_graph)
        assert not is_valid
        assert len(errors) > 0


class TestTraining:
    """Test training components."""
    
    def test_training_config(self):
        """Test TrainingConfig dataclass."""
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            num_epochs=100,
            patience=10
        )
        
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.patience == 10
    
    def test_gnn_trainer_init(self):
        """Test GNNTrainer initialization."""
        model = HybridGNNModel(
            lit_input_dim=768,
            kg_input_dim=512,
            hidden_dim=256,
            output_dim=128,
            num_heads=8
        )
        
        config = TrainingConfig()
        trainer = GNNTrainer(model, config)
        
        assert trainer.model == model
        assert trainer.config == config
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')
    
    def test_training_step(self):
        """Test single training step."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        config = TrainingConfig(learning_rate=0.001)
        trainer = GNNTrainer(model, config)
        
        # Create sample batch
        batch = {
            'lit_x': torch.randn(10, 64),
            'lit_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'kg_x': torch.randn(8, 64),
            'kg_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'kg_relation_types': torch.randn(2, 32),
            'labels': torch.randn(10, 64)
        }
        
        loss = trainer.training_step(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_validation_step(self):
        """Test single validation step."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        config = TrainingConfig()
        trainer = GNNTrainer(model, config)
        
        # Create sample batch
        batch = {
            'lit_x': torch.randn(10, 64),
            'lit_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'kg_x': torch.randn(8, 64),
            'kg_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'kg_relation_types': torch.randn(2, 32),
            'labels': torch.randn(10, 64)
        }
        
        with torch.no_grad():
            metrics = trainer.validation_step(batch)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics or 'mse' in metrics
    
    def test_training_loop(self):
        """Test training loop with mocked data."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        config = TrainingConfig(num_epochs=2, patience=5)
        trainer = GNNTrainer(model, config)
        
        # Create mock data loaders
        mock_train_loader = [
            {
                'lit_x': torch.randn(10, 64),
                'lit_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                'kg_x': torch.randn(8, 64),
                'kg_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                'kg_relation_types': torch.randn(2, 32),
                'labels': torch.randn(10, 64)
            }
        ] * 3  # 3 batches
        
        mock_val_loader = mock_train_loader[:1]  # 1 validation batch
        
        history = trainer.train(mock_train_loader, mock_val_loader)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == config.num_epochs
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        config = TrainingConfig(num_epochs=100, patience=2)
        trainer = GNNTrainer(model, config)
        
        # Mock validation losses that don't improve
        val_losses = [1.0, 0.9, 1.1, 1.2, 1.3]  # Stops improving after epoch 1
        
        with patch.object(trainer, 'validation_step') as mock_val:
            mock_val.side_effect = [{'loss': loss} for loss in val_losses]
            
            # This should trigger early stopping
            stopped_early = trainer._check_early_stopping([1.0, 0.9, 1.1])
            
            assert stopped_early
    
    def test_model_checkpointing(self, temp_dir):
        """Test model checkpointing."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        config = TrainingConfig()
        trainer = GNNTrainer(model, config)
        
        # Save checkpoint
        checkpoint_path = temp_dir / "model_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=10, loss=0.5)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['epoch'] == 10
        assert checkpoint['loss'] == 0.5


class TestDataLoader:
    """Test data loading components."""
    
    def test_biomedical_data_loader_init(self):
        """Test BiomedicalDataLoader initialization."""
        loader = BiomedicalDataLoader()
        
        assert hasattr(loader, 'logger')
    
    def test_create_data_loaders(self, sample_literature_data, sample_knowledge_graph):
        """Test data loader creation."""
        config = {
            'batch_size': 2,
            'num_workers': 0,
            'shuffle': True
        }
        
        with patch('litkg.phase2.data_loader.BiomedicalDataset') as MockDataset:
            mock_dataset = Mock()
            mock_dataset.__len__.return_value = 10
            mock_dataset.__getitem__.return_value = {
                'lit_x': torch.randn(5, 64),
                'kg_x': torch.randn(3, 64),
                'labels': torch.randn(5, 1)
            }
            MockDataset.return_value = mock_dataset
            
            train_loader, val_loader, test_loader = create_data_loaders(
                literature_data=sample_literature_data,
                kg_data=sample_knowledge_graph,
                config=config
            )
            
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
    
    def test_data_preprocessing(self):
        """Test data preprocessing for GNN input."""
        loader = BiomedicalDataLoader()
        
        # Mock raw data
        raw_data = {
            'literature': [
                {'text': 'BRCA1 causes cancer', 'entities': [{'text': 'BRCA1', 'label': 'GENE'}]}
            ],
            'kg': {
                'nodes': [{'id': 'BRCA1', 'type': 'gene'}],
                'edges': [{'source': 'BRCA1', 'target': 'cancer', 'type': 'causes'}]
            }
        }
        
        processed_data = loader.preprocess_data(raw_data)
        
        assert 'lit_features' in processed_data
        assert 'kg_features' in processed_data
        assert 'lit_edge_index' in processed_data
        assert 'kg_edge_index' in processed_data
    
    def test_batch_collation(self):
        """Test batch collation for variable-sized graphs."""
        loader = BiomedicalDataLoader()
        
        # Mock batch of variable-sized samples
        batch = [
            {
                'lit_x': torch.randn(5, 64),
                'kg_x': torch.randn(3, 64),
                'lit_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                'kg_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            },
            {
                'lit_x': torch.randn(7, 64),
                'kg_x': torch.randn(4, 64),
                'lit_edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
                'kg_edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            }
        ]
        
        collated_batch = loader.collate_fn(batch)
        
        assert 'lit_x' in collated_batch
        assert 'kg_x' in collated_batch
        assert 'batch' in collated_batch  # Batch indices for graph pooling


@pytest.mark.integration
class TestPhase2Integration:
    """Integration tests for Phase 2 components."""
    
    def test_end_to_end_training(self, sample_literature_data, sample_knowledge_graph):
        """Test end-to-end training pipeline."""
        # Create model
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        # Create trainer
        config = TrainingConfig(num_epochs=2, batch_size=1)
        trainer = GNNTrainer(model, config)
        
        # Create mock data
        mock_batch = {
            'lit_x': torch.randn(10, 64),
            'lit_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'kg_x': torch.randn(8, 64),
            'kg_edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'kg_relation_types': torch.randn(2, 32),
            'labels': torch.randn(10, 64)
        }
        
        train_loader = [mock_batch]
        val_loader = [mock_batch]
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        assert len(history['train_loss']) == config.num_epochs
        assert len(history['val_loss']) == config.num_epochs
    
    def test_model_inference(self):
        """Test model inference on new data."""
        model = HybridGNNModel(
            lit_input_dim=64,
            kg_input_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_heads=4
        )
        
        model.eval()
        
        with torch.no_grad():
            # Create test data
            lit_x = torch.randn(5, 64)
            lit_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            kg_x = torch.randn(3, 64)
            kg_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            kg_relation_types = torch.randn(2, 32)
            
            # Run inference
            output = model(lit_x, lit_edge_index, kg_x, kg_edge_index, kg_relation_types)
            
            assert output.shape == (5, 64)
            assert not output.requires_grad
    
    @pytest.mark.slow
    def test_large_graph_processing(self):
        """Test processing of large graphs."""
        model = HybridGNNModel(
            lit_input_dim=768,
            kg_input_dim=512,
            hidden_dim=256,
            output_dim=128,
            num_heads=8
        )
        
        # Create large graph data
        lit_x = torch.randn(1000, 768)
        lit_edge_index = torch.randint(0, 1000, (2, 5000))
        kg_x = torch.randn(500, 512)
        kg_edge_index = torch.randint(0, 500, (2, 2000))
        kg_relation_types = torch.randn(2000, 64)
        
        # Should handle large graphs without errors
        with torch.no_grad():
            output = model(lit_x, lit_edge_index, kg_x, kg_edge_index, kg_relation_types)
            
            assert output.shape == (1000, 128)


if __name__ == "__main__":
    pytest.main([__file__])