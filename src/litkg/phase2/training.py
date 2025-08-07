"""
Training infrastructure for hybrid GNN models.

This module provides training loops, loss functions, and evaluation
metrics for the hybrid GNN architecture.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import wandb

from ..utils.logging import LoggerMixin
from .hybrid_gnn import HybridGNNModel


@dataclass
class TrainingConfig:
    """Configuration for hybrid GNN training."""
    
    # Model architecture
    lit_node_dim: int = 768 + 10 + 2  # embedding + type + features
    lit_edge_dim: int = 1 + 10 + 1    # weight + type + confidence
    kg_node_dim: int = 768 + 10 + 2   # embedding + type + features
    kg_edge_dim: int = 1 + 10 + 1     # confidence + type + evidence
    kg_relation_dim: int = 10         # relation type dimension
    
    hidden_dim: int = 256
    num_gnn_layers: int = 3
    num_fusion_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    num_relations: int = 10
    fusion_strategy: str = "attention"
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 15
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    link_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    contrastive_loss_weight: float = 0.3
    
    # Evaluation
    eval_every: int = 5
    save_every: int = 10
    
    # Paths
    output_dir: str = "outputs/phase2_training"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "litkg-phase2"
    wandb_entity: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    
    epoch: int
    train_loss: float
    val_loss: float
    
    # Link prediction metrics
    link_accuracy: float
    link_precision: float
    link_recall: float
    link_f1: float
    link_auc: float
    
    # Relation prediction metrics
    relation_accuracy: float
    relation_macro_f1: float
    relation_weighted_f1: float
    
    # Confidence prediction metrics
    confidence_mae: float
    confidence_mse: float
    
    # Learning metrics
    learning_rate: float
    gradient_norm: float
    
    # Timing
    epoch_time: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return asdict(self)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning aligned representations.
    
    Encourages similar entities to have similar representations
    and dissimilar entities to have different representations.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        lit_embeddings: torch.Tensor,
        kg_embeddings: torch.Tensor,
        alignment_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            lit_embeddings: Literature embeddings [batch_size, hidden_dim]
            kg_embeddings: KG embeddings [batch_size, hidden_dim]
            alignment_matrix: Binary alignment matrix [batch_size, batch_size]
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        lit_norm = F.normalize(lit_embeddings, p=2, dim=1)
        kg_norm = F.normalize(kg_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(lit_norm, kg_norm.t()) / self.temperature
        
        # Positive pairs (aligned entities)
        positive_mask = alignment_matrix.bool()
        positive_similarities = similarity[positive_mask]
        
        # Negative pairs (non-aligned entities)
        negative_mask = ~positive_mask
        negative_similarities = similarity[negative_mask]
        
        # Contrastive loss
        positive_loss = -torch.log(torch.sigmoid(positive_similarities) + 1e-8).mean()
        negative_loss = -torch.log(torch.sigmoid(-negative_similarities) + 1e-8).mean()
        
        return positive_loss + negative_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining link prediction, relation prediction, and confidence estimation.
    """
    
    def __init__(
        self,
        link_weight: float = 1.0,
        relation_weight: float = 1.0,
        confidence_weight: float = 0.5,
        contrastive_weight: float = 0.3
    ):
        super().__init__()
        
        self.link_weight = link_weight
        self.relation_weight = relation_weight
        self.confidence_weight = confidence_weight
        self.contrastive_weight = contrastive_weight
        
        # Loss functions
        self.link_criterion = nn.BCELoss()
        self.relation_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        self.contrastive_criterion = ContrastiveLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lit_embeddings: Optional[torch.Tensor] = None,
        kg_embeddings: Optional[torch.Tensor] = None,
        alignment_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            lit_embeddings: Literature embeddings for contrastive loss
            kg_embeddings: KG embeddings for contrastive loss
            alignment_matrix: Alignment matrix for contrastive loss
            
        Returns:
            Total loss and individual loss components
        """
        losses = {}
        
        # Link prediction loss
        if 'link_probs' in predictions and 'link_labels' in targets:
            link_loss = self.link_criterion(
                predictions['link_probs'].squeeze(),
                targets['link_labels'].float()
            )
            losses['link_loss'] = link_loss
        
        # Relation prediction loss
        if 'relation_logits' in predictions and 'relation_labels' in targets:
            relation_loss = self.relation_criterion(
                predictions['relation_logits'],
                targets['relation_labels'].long()
            )
            losses['relation_loss'] = relation_loss
        
        # Confidence prediction loss
        if 'confidence' in predictions and 'confidence_labels' in targets:
            confidence_loss = self.confidence_criterion(
                predictions['confidence'].squeeze(),
                targets['confidence_labels'].float()
            )
            losses['confidence_loss'] = confidence_loss
        
        # Contrastive loss
        if (lit_embeddings is not None and kg_embeddings is not None 
            and alignment_matrix is not None):
            contrastive_loss = self.contrastive_criterion(
                lit_embeddings, kg_embeddings, alignment_matrix
            )
            losses['contrastive_loss'] = contrastive_loss
        
        # Combine losses
        total_loss = 0
        if 'link_loss' in losses:
            total_loss += self.link_weight * losses['link_loss']
        if 'relation_loss' in losses:
            total_loss += self.relation_weight * losses['relation_loss']
        if 'confidence_loss' in losses:
            total_loss += self.confidence_weight * losses['confidence_loss']
        if 'contrastive_loss' in losses:
            total_loss += self.contrastive_weight * losses['contrastive_loss']
        
        return total_loss, losses


class EvaluationMetrics:
    """Comprehensive evaluation metrics for hybrid GNN."""
    
    @staticmethod
    def compute_link_prediction_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Compute link prediction metrics."""
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='binary'
        )
        
        # AUC metrics
        try:
            auc = roc_auc_score(targets, probabilities)
            ap = average_precision_score(targets, probabilities)
        except ValueError:
            auc = 0.0
            ap = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'average_precision': ap
        }
    
    @staticmethod
    def compute_relation_prediction_metrics(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute relation prediction metrics."""
        accuracy = accuracy_score(targets, predictions)
        
        # F1 scores
        _, _, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro'
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'macro_f1': f1_macro,
            'weighted_f1': f1_weighted
        }
    
    @staticmethod
    def compute_confidence_metrics(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute confidence prediction metrics."""
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # Correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation
        }


class GNNTrainer(LoggerMixin):
    """
    Main trainer for hybrid GNN models.
    
    Handles training loop, validation, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        model: Optional[HybridGNNModel] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        # Allow (model, config) signature used by tests
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / config.checkpoint_dir
        self.log_dir = self.output_dir / config.log_dir
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or use provided model
        if model is None:
            self.model = HybridGNNModel(
                lit_node_dim=self.config.lit_node_dim,
                lit_edge_dim=self.config.lit_edge_dim,
                kg_node_dim=self.config.kg_node_dim,
                kg_edge_dim=self.config.kg_edge_dim,
                kg_relation_dim=self.config.kg_relation_dim,
                hidden_dim=self.config.hidden_dim,
                num_gnn_layers=self.config.num_gnn_layers,
                num_fusion_layers=self.config.num_fusion_layers,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                num_relations=self.config.num_relations,
                fusion_strategy=self.config.fusion_strategy
            ).to(self.device)
        else:
            self.model = model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.patience // 2,
            # verbose parameter deprecated in newer PyTorch versions
        )
        
        # Loss function
        self.criterion = MultiTaskLoss(
            link_weight=config.link_loss_weight,
            relation_weight=config.relation_loss_weight,
            confidence_weight=config.confidence_loss_weight,
            contrastive_weight=config.contrastive_loss_weight
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Wandb logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                name=f"hybrid_gnn_{int(time.time())}"
            )
            wandb.watch(self.model, log_freq=100)
        
        self.logger.info(f"Initialized GNNTrainer with device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        epoch_losses = defaultdict(float)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            lit_graph = batch['lit_graph'].to(self.device)
            kg_graph = batch['kg_graph'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Create entity pairs for relation prediction
            entity_pairs = self._create_entity_pairs(batch)
            
            outputs = self.model(
                lit_x=lit_graph.x,
                lit_edge_index=lit_graph.edge_index,
                lit_edge_attr=lit_graph.edge_attr,
                lit_batch=getattr(lit_graph, 'batch', None),
                kg_x=kg_graph.x,
                kg_edge_index=kg_graph.edge_index,
                kg_edge_attr=kg_graph.edge_attr,
                kg_relation_types=kg_graph.relation_types,
                kg_batch=getattr(kg_graph, 'batch', None),
                entity_pairs=entity_pairs
            )
            
            # Compute loss
            alignment_matrix = self._create_alignment_matrix(batch)
            
            loss, loss_components = self.criterion(
                predictions=outputs,
                targets=labels,
                lit_embeddings=outputs['lit_graph_embedding'],
                kg_embeddings=outputs['kg_graph_embedding'],
                alignment_matrix=alignment_matrix
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            for loss_name, loss_value in loss_components.items():
                epoch_losses[loss_name] += loss_value.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return avg_loss, avg_losses

    # Public wrappers expected by tests
    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.model.train()
        lit_graph = batch['lit_graph'].to(self.device)
        kg_graph = batch['kg_graph'].to(self.device)
        labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
        self.optimizer.zero_grad()
        entity_pairs = self._create_entity_pairs(batch)
        outputs = self.model(
            lit_x=lit_graph.x,
            lit_edge_index=lit_graph.edge_index,
            kg_x=kg_graph.x,
            kg_edge_index=kg_graph.edge_index,
            kg_relation_types=getattr(kg_graph, 'relation_types', None),
            entity_pairs=entity_pairs
        )
        alignment_matrix = self._create_alignment_matrix(batch)
        loss, _ = self.criterion(
            predictions=outputs,
            targets=labels,
            lit_embeddings=outputs['lit_graph_embedding'],
            kg_embeddings=outputs['kg_graph_embedding'],
            alignment_matrix=alignment_matrix
        )
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            lit_graph = batch['lit_graph'].to(self.device)
            kg_graph = batch['kg_graph'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            outputs = self.model(
                lit_x=lit_graph.x,
                lit_edge_index=lit_graph.edge_index,
                kg_x=kg_graph.x,
                kg_edge_index=kg_graph.edge_index,
                kg_relation_types=getattr(kg_graph, 'relation_types', None)
            )
            alignment_matrix = self._create_alignment_matrix(batch)
            loss, _ = self.criterion(
                predictions=outputs,
                targets=labels,
                lit_embeddings=outputs['lit_graph_embedding'],
                kg_embeddings=outputs['kg_graph_embedding'],
                alignment_matrix=alignment_matrix
            )
            return {"loss": float(loss.item())}

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, path)
    
    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, TrainingMetrics]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Collect predictions and targets
        all_link_preds = []
        all_link_targets = []
        all_link_probs = []
        
        all_relation_preds = []
        all_relation_targets = []
        
        all_confidence_preds = []
        all_confidence_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                lit_graph = batch['lit_graph'].to(self.device)
                kg_graph = batch['kg_graph'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                # Forward pass
                entity_pairs = self._create_entity_pairs(batch)
                
                outputs = self.model(
                    lit_x=lit_graph.x,
                    lit_edge_index=lit_graph.edge_index,
                    lit_edge_attr=lit_graph.edge_attr,
                    lit_batch=getattr(lit_graph, 'batch', None),
                    kg_x=kg_graph.x,
                    kg_edge_index=kg_graph.edge_index,
                    kg_edge_attr=kg_graph.edge_attr,
                    kg_relation_types=kg_graph.relation_types,
                    kg_batch=getattr(kg_graph, 'batch', None),
                    entity_pairs=entity_pairs
                )
                
                # Compute loss
                alignment_matrix = self._create_alignment_matrix(batch)
                
                loss, _ = self.criterion(
                    predictions=outputs,
                    targets=labels,
                    lit_embeddings=outputs['lit_graph_embedding'],
                    kg_embeddings=outputs['kg_graph_embedding'],
                    alignment_matrix=alignment_matrix
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                if 'link_probs' in outputs:
                    link_probs = outputs['link_probs'].cpu().numpy()
                    link_preds = (link_probs > 0.5).astype(int)
                    link_targets = labels['link_labels'].cpu().numpy()
                    
                    all_link_preds.extend(link_preds.flatten())
                    all_link_targets.extend(link_targets.flatten())
                    all_link_probs.extend(link_probs.flatten())
                
                if 'relation_probs' in outputs:
                    relation_preds = outputs['relation_probs'].argmax(dim=-1).cpu().numpy()
                    relation_targets = labels['relation_labels'].cpu().numpy()
                    
                    all_relation_preds.extend(relation_preds.flatten())
                    all_relation_targets.extend(relation_targets.flatten())
                
                if 'confidence' in outputs:
                    confidence_preds = outputs['confidence'].cpu().numpy()
                    confidence_targets = labels['confidence_labels'].cpu().numpy()
                    
                    all_confidence_preds.extend(confidence_preds.flatten())
                    all_confidence_targets.extend(confidence_targets.flatten())
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        
        # Link prediction metrics
        link_metrics = {}
        if all_link_preds:
            link_metrics = EvaluationMetrics.compute_link_prediction_metrics(
                np.array(all_link_preds),
                np.array(all_link_targets),
                np.array(all_link_probs)
            )
        
        # Relation prediction metrics
        relation_metrics = {}
        if all_relation_preds:
            relation_metrics = EvaluationMetrics.compute_relation_prediction_metrics(
                np.array(all_relation_preds),
                np.array(all_relation_targets)
            )
        
        # Confidence metrics
        confidence_metrics = {}
        if all_confidence_preds:
            confidence_metrics = EvaluationMetrics.compute_confidence_metrics(
                np.array(all_confidence_preds),
                np.array(all_confidence_targets)
            )
        
        # Create metrics object
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            train_loss=0.0,  # Will be filled by caller
            val_loss=avg_loss,
            link_accuracy=link_metrics.get('accuracy', 0.0),
            link_precision=link_metrics.get('precision', 0.0),
            link_recall=link_metrics.get('recall', 0.0),
            link_f1=link_metrics.get('f1', 0.0),
            link_auc=link_metrics.get('auc', 0.0),
            relation_accuracy=relation_metrics.get('accuracy', 0.0),
            relation_macro_f1=relation_metrics.get('macro_f1', 0.0),
            relation_weighted_f1=relation_metrics.get('weighted_f1', 0.0),
            confidence_mae=confidence_metrics.get('mae', 0.0),
            confidence_mse=confidence_metrics.get('mse', 0.0),
            learning_rate=self.optimizer.param_groups[0]['lr'],
            gradient_norm=0.0,  # Will be filled by caller
            epoch_time=0.0  # Will be filled by caller
        )
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> List[TrainingMetrics]:
        """Main training loop."""
        self.logger.info("Starting hybrid GNN training")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            train_loss, train_loss_components = self.train_epoch(train_loader)
            
            # Validation
            if epoch % self.config.eval_every == 0:
                val_loss, metrics = self.validate_epoch(val_loader)
                
                # Update metrics
                metrics.train_loss = train_loss
                metrics.epoch_time = time.time() - epoch_start_time
                
                # Compute gradient norm
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                metrics.gradient_norm = total_norm ** (1. / 2)
                
                self.training_history.append(metrics)
                
                # Logging
                self._log_metrics(metrics, train_loss_components)
                
                # Early stopping and checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Regular checkpointing
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        self.logger.info("Training completed")
        return self.training_history
    
    def _create_entity_pairs(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Create entity pairs for relation prediction."""
        # Simple implementation: create pairs from alignments
        alignments = batch.get('alignments', [])
        
        if not alignments:
            # Create dummy pairs if no alignments
            return torch.tensor([[0, 0]], dtype=torch.long, device=self.device)
        
        pairs = []
        for alignment in alignments:
            # Extract entity indices from alignment
            lit_idx = int(alignment.lit_entity_id.split('_')[-1]) if '_' in alignment.lit_entity_id else 0
            kg_idx = int(alignment.kg_entity_id.split('_')[-1]) if '_' in alignment.kg_entity_id else 0
            pairs.append([lit_idx, kg_idx])
        
        return torch.tensor(pairs, dtype=torch.long, device=self.device)
    
    def _create_alignment_matrix(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Create alignment matrix for contrastive loss."""
        # Simple implementation: create identity matrix
        batch_size = batch['lit_graph'].x.size(0)
        return torch.eye(batch_size, device=self.device)
    
    def _log_metrics(self, metrics: TrainingMetrics, train_losses: Dict[str, float]):
        """Log training metrics."""
        # Console logging
        self.logger.info(
            f"Epoch {metrics.epoch}: "
            f"Train Loss: {metrics.train_loss:.4f}, "
            f"Val Loss: {metrics.val_loss:.4f}, "
            f"Link F1: {metrics.link_f1:.4f}, "
            f"Relation Acc: {metrics.relation_accuracy:.4f}"
        )
        
        # Wandb logging
        if self.config.use_wandb:
            log_dict = metrics.to_dict()
            log_dict.update({f"train_{k}": v for k, v in train_losses.items()})
            wandb.log(log_dict, step=metrics.epoch)
        
        # Save metrics to file
        metrics_file = self.log_dir / "training_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            json.dump(metrics.to_dict(), f)
            f.write('\n')
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': [asdict(m) for m in self.training_history]
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint


# Import defaultdict at the top of the file
from collections import defaultdict