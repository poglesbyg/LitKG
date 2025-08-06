"""
Data loading utilities for hybrid GNN training.

This module provides PyTorch datasets and data loaders for training
the hybrid GNN architecture with literature and knowledge graph data.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data, Batch
import pickle

from ..utils.logging import LoggerMixin


class HybridGraphDataset(Dataset, LoggerMixin):
    """
    Dataset for hybrid GNN training.
    
    Loads pre-constructed literature graphs and KG subgraphs
    with their alignments and labels.
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        max_samples: Optional[int] = None,
        augment_data: bool = True,
        cache_in_memory: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.augment_data = augment_data
        self.cache_in_memory = cache_in_memory
        
        # Find all training examples
        self.example_files = list(self.data_dir.glob(f"{split}_*.pkl"))
        
        if max_samples:
            self.example_files = self.example_files[:max_samples]
        
        self.logger.info(f"Found {len(self.example_files)} examples for {split} split")
        
        # Memory cache
        self.cache = {} if cache_in_memory else None
        
        # Load all data into memory if requested
        if cache_in_memory:
            self._load_all_data()
    
    def _load_all_data(self):
        """Load all data into memory for faster access."""
        self.logger.info("Loading all data into memory...")
        
        for idx, file_path in enumerate(self.example_files):
            try:
                with open(file_path, 'rb') as f:
                    example = pickle.load(f)
                    self.cache[idx] = example
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(self.cache)} examples into memory")
    
    def __len__(self) -> int:
        return len(self.example_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training example."""
        # Load from cache or disk
        if self.cache is not None and idx in self.cache:
            example = self.cache[idx]
        else:
            file_path = self.example_files[idx]
            try:
                with open(file_path, 'rb') as f:
                    example = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                return self._get_empty_example()
        
        # Apply data augmentation
        if self.augment_data and self.split == "train":
            example = self._augment_example(example)
        
        return example
    
    def _get_empty_example(self) -> Dict[str, Any]:
        """Create an empty example as fallback."""
        # Create minimal graphs
        lit_graph = Data(
            x=torch.zeros(1, 768 + 10 + 2),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.zeros(1, 1 + 10 + 1),
            node_types=torch.tensor([0], dtype=torch.long),
            edge_types=torch.tensor([0], dtype=torch.long),
            num_nodes=1
        )
        
        kg_graph = Data(
            x=torch.zeros(1, 768 + 10 + 2),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.zeros(1, 1 + 10 + 1),
            node_types=torch.tensor([0], dtype=torch.long),
            edge_types=torch.tensor([0], dtype=torch.long),
            relation_types=torch.tensor([0], dtype=torch.long),
            num_nodes=1
        )
        
        return {
            'id': 'empty_example',
            'lit_graph': lit_graph,
            'kg_graph': kg_graph,
            'alignments': [],
            'labels': {
                'link_labels': torch.tensor([0.0]),
                'relation_labels': torch.tensor([0]),
                'confidence_labels': torch.tensor([0.5])
            }
        }
    
    def _augment_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation to training example."""
        # Node feature noise
        if random.random() < 0.3:
            example = self._add_node_noise(example)
        
        # Edge dropout
        if random.random() < 0.2:
            example = self._apply_edge_dropout(example)
        
        # Node dropout
        if random.random() < 0.1:
            example = self._apply_node_dropout(example)
        
        return example
    
    def _add_node_noise(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to node features."""
        noise_std = 0.01
        
        # Literature graph
        lit_graph = example['lit_graph']
        noise = torch.randn_like(lit_graph.x) * noise_std
        lit_graph.x = lit_graph.x + noise
        
        # KG graph
        kg_graph = example['kg_graph']
        noise = torch.randn_like(kg_graph.x) * noise_std
        kg_graph.x = kg_graph.x + noise
        
        return example
    
    def _apply_edge_dropout(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly drop some edges."""
        dropout_rate = 0.1
        
        # Literature graph
        lit_graph = example['lit_graph']
        num_edges = lit_graph.edge_index.size(1)
        keep_mask = torch.rand(num_edges) > dropout_rate
        
        if keep_mask.sum() > 0:  # Ensure at least some edges remain
            lit_graph.edge_index = lit_graph.edge_index[:, keep_mask]
            lit_graph.edge_attr = lit_graph.edge_attr[keep_mask]
            if hasattr(lit_graph, 'edge_types'):
                lit_graph.edge_types = lit_graph.edge_types[keep_mask]
        
        # KG graph
        kg_graph = example['kg_graph']
        num_edges = kg_graph.edge_index.size(1)
        keep_mask = torch.rand(num_edges) > dropout_rate
        
        if keep_mask.sum() > 0:
            kg_graph.edge_index = kg_graph.edge_index[:, keep_mask]
            kg_graph.edge_attr = kg_graph.edge_attr[keep_mask]
            if hasattr(kg_graph, 'edge_types'):
                kg_graph.edge_types = kg_graph.edge_types[keep_mask]
            if hasattr(kg_graph, 'relation_types'):
                kg_graph.relation_types = kg_graph.relation_types[keep_mask]
        
        return example
    
    def _apply_node_dropout(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly mask some node features."""
        dropout_rate = 0.05
        
        # Literature graph
        lit_graph = example['lit_graph']
        mask = torch.rand_like(lit_graph.x) > dropout_rate
        lit_graph.x = lit_graph.x * mask
        
        # KG graph
        kg_graph = example['kg_graph']
        mask = torch.rand_like(kg_graph.x) > dropout_rate
        kg_graph.x = kg_graph.x * mask
        
        return example
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'num_examples': len(self),
            'lit_graph_sizes': [],
            'kg_graph_sizes': [],
            'num_alignments': [],
            'node_type_distribution': defaultdict(int),
            'edge_type_distribution': defaultdict(int)
        }
        
        # Sample a subset for statistics
        sample_size = min(100, len(self))
        sample_indices = random.sample(range(len(self)), sample_size)
        
        for idx in sample_indices:
            try:
                example = self[idx]
                
                # Graph sizes
                stats['lit_graph_sizes'].append(example['lit_graph'].num_nodes)
                stats['kg_graph_sizes'].append(example['kg_graph'].num_nodes)
                
                # Alignments
                stats['num_alignments'].append(len(example['alignments']))
                
                # Node types
                for node_type in example['lit_graph'].node_types:
                    stats['node_type_distribution'][f'lit_{node_type.item()}'] += 1
                
                for node_type in example['kg_graph'].node_types:
                    stats['node_type_distribution'][f'kg_{node_type.item()}'] += 1
                
            except Exception as e:
                self.logger.warning(f"Error computing stats for example {idx}: {e}")
        
        # Convert to regular dict and compute averages
        stats['avg_lit_graph_size'] = np.mean(stats['lit_graph_sizes']) if stats['lit_graph_sizes'] else 0
        stats['avg_kg_graph_size'] = np.mean(stats['kg_graph_sizes']) if stats['kg_graph_sizes'] else 0
        stats['avg_alignments'] = np.mean(stats['num_alignments']) if stats['num_alignments'] else 0
        
        return dict(stats)


class GraphBatchSampler(Sampler, LoggerMixin):
    """
    Custom batch sampler for graphs of similar sizes.
    
    Groups graphs by size to create more efficient batches
    and reduce padding overhead.
    """
    
    def __init__(
        self,
        dataset: HybridGraphDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        size_tolerance: float = 0.3
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.size_tolerance = size_tolerance
        
        # Group examples by graph size
        self.size_groups = self._group_by_size()
        self.logger.info(f"Created {len(self.size_groups)} size groups for batching")
    
    def _group_by_size(self) -> Dict[str, List[int]]:
        """Group examples by graph size."""
        size_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                example = self.dataset[idx]
                lit_size = example['lit_graph'].num_nodes
                kg_size = example['kg_graph'].num_nodes
                
                # Use combined size as grouping key
                combined_size = lit_size + kg_size
                size_key = f"{combined_size // 10 * 10}"  # Group by tens
                
                size_groups[size_key].append(idx)
                
            except Exception as e:
                self.logger.warning(f"Error grouping example {idx}: {e}")
        
        return dict(size_groups)
    
    def __iter__(self):
        """Generate batches."""
        # Shuffle groups if requested
        group_keys = list(self.size_groups.keys())
        if self.shuffle:
            random.shuffle(group_keys)
        
        for group_key in group_keys:
            indices = self.size_groups[group_key].copy()
            
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches from this group
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    yield batch_indices
    
    def __len__(self) -> int:
        """Number of batches."""
        total_batches = 0
        
        for group_indices in self.size_groups.values():
            num_batches = len(group_indices) // self.batch_size
            if len(group_indices) % self.batch_size != 0 and not self.drop_last:
                num_batches += 1
            total_batches += num_batches
        
        return total_batches


class DataCollator:
    """
    Custom data collator for hybrid graph data.
    
    Handles batching of PyTorch Geometric graphs with different sizes
    and creates proper batch assignments.
    """
    
    def __init__(self, follow_batch: Optional[List[str]] = None):
        self.follow_batch = follow_batch or ['x']
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of examples."""
        # Separate lit graphs and kg graphs
        lit_graphs = [example['lit_graph'] for example in batch]
        kg_graphs = [example['kg_graph'] for example in batch]
        
        # Batch graphs
        lit_batch = Batch.from_data_list(lit_graphs, follow_batch=self.follow_batch)
        kg_batch = Batch.from_data_list(kg_graphs, follow_batch=self.follow_batch)
        
        # Collect other data
        alignments = [example['alignments'] for example in batch]
        labels = self._collate_labels([example['labels'] for example in batch])
        ids = [example['id'] for example in batch]
        
        return {
            'lit_graph': lit_batch,
            'kg_graph': kg_batch,
            'alignments': alignments,
            'labels': labels,
            'ids': ids
        }
    
    def _collate_labels(self, label_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate labels from multiple examples."""
        collated_labels = {}
        
        # Get all label keys
        all_keys = set()
        for labels in label_list:
            all_keys.update(labels.keys())
        
        # Collate each label type
        for key in all_keys:
            values = []
            for labels in label_list:
                if key in labels:
                    values.append(labels[key])
                else:
                    # Create default value based on the first example
                    if values:
                        default_shape = values[0].shape
                        default_value = torch.zeros(default_shape, dtype=values[0].dtype)
                        values.append(default_value)
            
            if values:
                try:
                    collated_labels[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    # If concatenation fails, stack instead
                    collated_labels[key] = torch.stack(values, dim=0)
        
        return collated_labels


def create_data_loaders(
    data_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    use_size_batching: bool = True,
    cache_in_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Directory containing training examples
        batch_size: Batch size for training
        num_workers: Number of worker processes
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        use_size_batching: Whether to use size-based batching
        cache_in_memory: Whether to cache data in memory
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = HybridGraphDataset(
        data_dir=data_dir,
        split="train",
        cache_in_memory=cache_in_memory,
        **kwargs
    )
    
    val_dataset = HybridGraphDataset(
        data_dir=data_dir,
        split="val",
        augment_data=False,
        cache_in_memory=cache_in_memory,
        **kwargs
    )
    
    test_dataset = HybridGraphDataset(
        data_dir=data_dir,
        split="test",
        augment_data=False,
        cache_in_memory=cache_in_memory,
        **kwargs
    )
    
    # Create data collator
    collator = DataCollator()
    
    # Create data loaders
    if use_size_batching:
        # Use custom batch sampler
        train_sampler = GraphBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # Use standard data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def split_data(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Split training examples into train/val/test sets.
    
    Args:
        data_dir: Directory containing training examples
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Find all example files
    example_files = list(data_dir.glob("training_example_*.pkl"))
    random.shuffle(example_files)
    
    # Calculate split sizes
    total_examples = len(example_files)
    train_size = int(total_examples * train_ratio)
    val_size = int(total_examples * val_ratio)
    
    # Split files
    train_files = example_files[:train_size]
    val_files = example_files[train_size:train_size + val_size]
    test_files = example_files[train_size + val_size:]
    
    # Rename files to include split prefix
    for i, file_path in enumerate(train_files):
        new_path = data_dir / f"train_{i}.pkl"
        file_path.rename(new_path)
    
    for i, file_path in enumerate(val_files):
        new_path = data_dir / f"val_{i}.pkl"
        file_path.rename(new_path)
    
    for i, file_path in enumerate(test_files):
        new_path = data_dir / f"test_{i}.pkl"
        file_path.rename(new_path)
    
    print(f"Split {total_examples} examples into:")
    print(f"  Train: {len(train_files)} examples")
    print(f"  Val: {len(val_files)} examples")
    print(f"  Test: {len(test_files)} examples")


# Import missing modules at the top
from collections import defaultdict