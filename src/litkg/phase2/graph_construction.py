"""
Graph construction utilities for hybrid GNN training.

This module provides tools for constructing literature graphs,
extracting KG subgraphs, and aligning graphs for training.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging import LoggerMixin
from ..models.embeddings import BiomedicalEmbeddings


@dataclass
class GraphMetadata:
    """Metadata for constructed graphs."""
    graph_id: str
    num_nodes: int
    num_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    source_documents: List[str]
    timestamp: float
    
    
@dataclass
class EntityAlignment:
    """Entity alignment between literature and KG."""
    lit_entity_id: str
    kg_entity_id: str
    similarity_score: float
    alignment_method: str
    confidence: float


class LiteratureGraphBuilder(LoggerMixin):
    """
    Builds graphs from literature processing results.
    
    Creates graphs where nodes are entities (genes, diseases, drugs)
    and edges represent co-occurrence or extracted relations.
    """
    
    def __init__(
        self,
        embeddings_model: Optional[BiomedicalEmbeddings] = None,
        min_cooccurrence: int = 2,
        similarity_threshold: float = 0.7,
        max_nodes_per_graph: int = 500
    ):
        self.embeddings_model = embeddings_model
        self.min_cooccurrence = min_cooccurrence
        self.similarity_threshold = similarity_threshold
        self.max_nodes_per_graph = max_nodes_per_graph
        
        # Entity type mapping
        self.entity_type_map = {
            'GENE': 0,
            'DISEASE': 1,
            'DRUG': 2,
            'PROTEIN': 3,
            'CHEMICAL': 4,
            'ORGANISM': 5,
            'CELL_LINE': 6,
            'TISSUE': 7,
            'MUTATION': 8,
            'OTHER': 9
        }
        
        # Relation type mapping
        self.relation_type_map = {
            'COOCCURRENCE': 0,
            'ASSOCIATION': 1,
            'INTERACTION': 2,
            'REGULATION': 3,
            'CAUSATION': 4,
            'SIMILARITY': 5,
            'PATHWAY': 6,
            'TREATMENT': 7,
            'DIAGNOSIS': 8,
            'OTHER': 9
        }
    
    # --------------------- New: simple dict-based builder for tests ---------------------
    def _extract_entities(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        return document.get('entities', [])

    def _extract_relations(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        return document.get('relations', [])

    def build_graph(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for doc in documents:
            for ent in self._extract_entities(doc):
                ent_id = ent.get('text') or ent.get('id') or ent.get('name')
                if not ent_id:
                    continue
                if ent_id not in nodes:
                    nodes[ent_id] = {
                        'id': ent_id,
                        'type': ent.get('label') or ent.get('type', 'ENTITY'),
                    }

            for rel in self._extract_relations(doc):
                head = rel.get('head') or rel.get('entity1')
                tail = rel.get('tail') or rel.get('entity2')
                if not head or not tail:
                    continue
                edges.append({
                    'source': head,
                    'target': tail,
                    'type': rel.get('relation') or rel.get('relation_type', 'ASSOCIATED_WITH')
                })

        return {'nodes': list(nodes.values()), 'edges': edges}

    # --------------------- Existing: PYG builder for Phase 2 demos ---------------------
    def build_literature_graph(
        self,
        documents: List[Dict[str, Any]],
        graph_id: str,
        use_semantic_similarity: bool = True
    ) -> Tuple[Data, GraphMetadata]:
        """
        Build a literature graph from processed documents.
        
        Args:
            documents: List of processed documents with entities and relations
            graph_id: Unique identifier for the graph
            use_semantic_similarity: Whether to add semantic similarity edges
            
        Returns:
            PyTorch Geometric Data object and metadata
        """
        self.logger.info(f"Building literature graph {graph_id} from {len(documents)} documents")
        
        # Collect entities and relations
        entities = {}  # entity_id -> entity_info
        relations = []  # list of (entity1, entity2, relation_type, weight)
        entity_cooccurrence = defaultdict(int)
        document_ids = []
        
        for doc_idx, doc in enumerate(documents):
            document_ids.append(doc.get('pmid', f'doc_{doc_idx}'))
            
            # Extract entities from document
            doc_entities = doc.get('entities', [])
            doc_entity_ids = []
            
            for entity in doc_entities:
                entity_id = self._normalize_entity_id(entity['text'])
                entity_type = entity.get('entity_group', 'OTHER')
                
                if entity_id not in entities:
                    entities[entity_id] = {
                        'text': entity['text'],
                        'type': entity_type,
                        'contexts': [],
                        'documents': set(),
                        'confidence_scores': []
                    }
                
                # Add context and document info
                entities[entity_id]['contexts'].append(entity.get('context', ''))
                entities[entity_id]['documents'].add(doc_idx)
                entities[entity_id]['confidence_scores'].append(entity.get('score', 1.0))
                
                doc_entity_ids.append(entity_id)
            
            # Count co-occurrences
            for i, entity1 in enumerate(doc_entity_ids):
                for entity2 in doc_entity_ids[i+1:]:
                    pair = tuple(sorted([entity1, entity2]))
                    entity_cooccurrence[pair] += 1
            
            # Extract explicit relations if available
            doc_relations = doc.get('relations', [])
            for relation in doc_relations:
                relations.append((
                    self._normalize_entity_id(relation['entity1']),
                    self._normalize_entity_id(relation['entity2']),
                    relation.get('relation_type', 'ASSOCIATION'),
                    relation.get('confidence', 1.0)
                ))
        
        # Filter entities by frequency and limit graph size
        entity_counts = {eid: len(info['documents']) for eid, info in entities.items()}
        filtered_entities = dict(sorted(
            entity_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_nodes_per_graph])
        
        # Generate embeddings for entities
        entity_texts = [entities[eid]['text'] for eid in filtered_entities]
        if self.embeddings_model is not None and entity_texts:
            entity_embeddings = self.embeddings_model.get_text_embeddings(entity_texts)
        else:
            # Fallback: zero embeddings
            embed_dim = 64
            entity_embeddings = np.zeros((len(entity_texts), embed_dim), dtype=np.float32)
        
        # Create node features and mappings
        node_id_to_idx = {eid: idx for idx, eid in enumerate(filtered_entities.keys())}
        node_features = []
        node_types = []
        
        for idx, entity_id in enumerate(filtered_entities.keys()):
            entity_info = entities[entity_id]
            
            # Node features: [embedding, type_onehot, confidence, frequency]
            embedding = entity_embeddings[idx]
            entity_type = entity_info['type']
            type_idx = self.entity_type_map.get(entity_type, self.entity_type_map['OTHER'])
            
            # One-hot encode entity type
            type_onehot = np.zeros(len(self.entity_type_map))
            type_onehot[type_idx] = 1.0
            
            # Additional features
            avg_confidence = np.mean(entity_info['confidence_scores'])
            frequency = len(entity_info['documents'])
            
            # Combine features
            node_feature = np.concatenate([
                embedding,
                type_onehot,
                [avg_confidence, frequency]
            ])
            
            node_features.append(node_feature)
            node_types.append(type_idx)
        
        # Create edges from co-occurrence and explicit relations
        edges = []
        edge_features = []
        edge_types = []
        
        # Co-occurrence edges
        for (entity1, entity2), count in entity_cooccurrence.items():
            if count >= self.min_cooccurrence and entity1 in node_id_to_idx and entity2 in node_id_to_idx:
                idx1, idx2 = node_id_to_idx[entity1], node_id_to_idx[entity2]
                
                # Bidirectional edges
                edges.extend([(idx1, idx2), (idx2, idx1)])
                
                # Edge features: [weight, type_onehot, confidence]
                weight = min(count / 10.0, 1.0)  # Normalize weight
                type_onehot = np.zeros(len(self.relation_type_map))
                type_onehot[self.relation_type_map['COOCCURRENCE']] = 1.0
                
                edge_feature = np.concatenate([
                    [weight],
                    type_onehot,
                    [0.8]  # Default confidence for co-occurrence
                ])
                
                edge_features.extend([edge_feature, edge_feature])
                edge_types.extend([
                    self.relation_type_map['COOCCURRENCE'],
                    self.relation_type_map['COOCCURRENCE']
                ])
        
        # Explicit relation edges
        for entity1, entity2, relation_type, confidence in relations:
            if entity1 in node_id_to_idx and entity2 in node_id_to_idx:
                idx1, idx2 = node_id_to_idx[entity1], node_id_to_idx[entity2]
                
                edges.append((idx1, idx2))
                
                # Edge features
                type_idx = self.relation_type_map.get(relation_type, self.relation_type_map['OTHER'])
                type_onehot = np.zeros(len(self.relation_type_map))
                type_onehot[type_idx] = 1.0
                
                edge_feature = np.concatenate([
                    [confidence],
                    type_onehot,
                    [confidence]
                ])
                
                edge_features.append(edge_feature)
                edge_types.append(type_idx)
        
        # Add semantic similarity edges if requested
        if use_semantic_similarity and len(entity_embeddings) > 1:
            similarity_matrix = cosine_similarity(entity_embeddings)
            
            for i in range(len(entity_embeddings)):
                for j in range(i+1, len(entity_embeddings)):
                    similarity = similarity_matrix[i, j]
                    
                    if similarity > self.similarity_threshold:
                        edges.extend([(i, j), (j, i)])
                        
                        # Similarity edge features
                        type_onehot = np.zeros(len(self.relation_type_map))
                        type_onehot[self.relation_type_map['SIMILARITY']] = 1.0
                        
                        edge_feature = np.concatenate([
                            [similarity],
                            type_onehot,
                            [similarity]
                        ])
                        
                        edge_features.extend([edge_feature, edge_feature])
                        edge_types.extend([
                            self.relation_type_map['SIMILARITY'],
                            self.relation_type_map['SIMILARITY']
                        ])
        
        # Convert to tensors
        if not edges:
            # Create empty graph with self-loops
            edges = [(i, i) for i in range(len(node_features))]
            edge_features = [np.zeros(1 + len(self.relation_type_map) + 1) for _ in edges]
            edge_types = [self.relation_type_map['OTHER']] * len(edges)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types_tensor,
            edge_types=edge_types_tensor,
            num_nodes=len(node_features)
        )
        
        # Create metadata
        metadata = GraphMetadata(
            graph_id=graph_id,
            num_nodes=len(node_features),
            num_edges=len(edges),
            node_types={k: (node_types_tensor == v).sum().item() for k, v in self.entity_type_map.items()},
            edge_types={k: (edge_types_tensor == v).sum().item() for k, v in self.relation_type_map.items()},
            source_documents=document_ids,
            timestamp=np.mean([doc.get('publication_date', 0) for doc in documents])
        )
        
        self.logger.info(f"Created literature graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
        
        return data, metadata
    
    def _normalize_entity_id(self, entity_text: str) -> str:
        """Normalize entity text to create consistent IDs."""
        return entity_text.lower().strip().replace(' ', '_')


class KGSubgraphExtractor(LoggerMixin):
    """
    Extracts relevant subgraphs from knowledge graphs.
    
    Given a set of entities, extracts subgraphs from CIVIC, TCGA, CPTAC
    that contain those entities and their local neighborhoods.
    """
    
    def __init__(
        self,
        max_subgraph_size: int = 1000,
        hop_distance: int = 2,
        min_edge_confidence: float = 0.5
    ):
        self.max_subgraph_size = max_subgraph_size
        self.hop_distance = hop_distance
        self.min_edge_confidence = min_edge_confidence
        
        # KG-specific entity and relation mappings
        self.kg_entity_types = {
            'GENE': 0,
            'VARIANT': 1,
            'DISEASE': 2,
            'DRUG': 3,
            'THERAPY': 4,
            'PHENOTYPE': 5,
            'PATHWAY': 6,
            'PROTEIN': 7,
            'TISSUE': 8,
            'OTHER': 9
        }
        
        self.kg_relation_types = {
            'THERAPEUTIC_RESPONSE': 0,
            'DIAGNOSTIC': 1,
            'PROGNOSTIC': 2,
            'PREDISPOSING': 3,
            'FUNCTIONAL': 4,
            'ONCOGENIC': 5,
            'PATHWAY_MEMBER': 6,
            'PROTEIN_INTERACTION': 7,
            'EXPRESSION_CORRELATION': 8,
            'OTHER': 9
        }
    
    def extract_subgraph(
        self,
        kg_data: Dict[str, Any],
        target_entities: List[str],
        subgraph_id: str
    ) -> Tuple[Data, GraphMetadata]:
        """
        Extract a subgraph centered on target entities.
        
        Args:
            kg_data: Knowledge graph data (preprocessed from Phase 1)
            target_entities: List of entity IDs to center the subgraph on
            subgraph_id: Unique identifier for the subgraph
            
        Returns:
            PyTorch Geometric Data object and metadata
        """
        self.logger.info(f"Extracting KG subgraph {subgraph_id} for {len(target_entities)} entities")
        
        # Build NetworkX graph for easier subgraph extraction
        G = nx.Graph()
        
        # Add nodes
        entities = kg_data.get('entities', {})
        for entity_id, entity_info in entities.items():
            G.add_node(entity_id, **entity_info)
        
        # Add edges
        relations = kg_data.get('relations', [])
        for relation in relations:
            if relation.get('confidence', 0) >= self.min_edge_confidence:
                G.add_edge(
                    relation['entity1'],
                    relation['entity2'],
                    **relation
                )
        
        # Find target nodes that exist in the KG
        valid_targets = [entity for entity in target_entities if entity in G.nodes()]
        
        if not valid_targets:
            self.logger.warning(f"No target entities found in KG for subgraph {subgraph_id}")
            return self._create_empty_kg_subgraph(subgraph_id)
        
        # Extract k-hop subgraph
        subgraph_nodes = set()
        for target in valid_targets:
            # Get k-hop neighborhood
            for node in nx.single_source_shortest_path_length(G, target, cutoff=self.hop_distance):
                subgraph_nodes.add(node)
                if len(subgraph_nodes) >= self.max_subgraph_size:
                    break
            
            if len(subgraph_nodes) >= self.max_subgraph_size:
                break
        
        # Extract subgraph
        subgraph = G.subgraph(subgraph_nodes).copy()
        
        # Convert to PyTorch Geometric format
        node_list = list(subgraph.nodes())
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        
        # Create node features
        node_features = []
        node_types = []
        
        for node_id in node_list:
            node_data = subgraph.nodes[node_id]
            
            # Node features: [embedding, type_onehot, confidence, centrality]
            embedding = node_data.get('embedding', np.zeros(768))  # Default BERT size
            entity_type = node_data.get('type', 'OTHER')
            type_idx = self.kg_entity_types.get(entity_type, self.kg_entity_types['OTHER'])
            
            # One-hot encode entity type
            type_onehot = np.zeros(len(self.kg_entity_types))
            type_onehot[type_idx] = 1.0
            
            # Additional features
            confidence = node_data.get('confidence', 1.0)
            centrality = nx.degree_centrality(subgraph)[node_id]
            
            # Combine features
            node_feature = np.concatenate([
                embedding,
                type_onehot,
                [confidence, centrality]
            ])
            
            node_features.append(node_feature)
            node_types.append(type_idx)
        
        # Create edges
        edges = []
        edge_features = []
        edge_types = []
        relation_types = []
        
        for edge in subgraph.edges(data=True):
            node1, node2, edge_data = edge
            idx1, idx2 = node_id_to_idx[node1], node_id_to_idx[node2]
            
            # Bidirectional edges
            edges.extend([(idx1, idx2), (idx2, idx1)])
            
            # Edge features: [confidence, type_onehot, evidence_count]
            confidence = edge_data.get('confidence', 1.0)
            relation_type = edge_data.get('relation_type', 'OTHER')
            type_idx = self.kg_relation_types.get(relation_type, self.kg_relation_types['OTHER'])
            
            type_onehot = np.zeros(len(self.kg_relation_types))
            type_onehot[type_idx] = 1.0
            
            evidence_count = edge_data.get('evidence_count', 1)
            
            edge_feature = np.concatenate([
                [confidence],
                type_onehot,
                [min(evidence_count / 10.0, 1.0)]  # Normalize evidence count
            ])
            
            edge_features.extend([edge_feature, edge_feature])
            edge_types.extend([type_idx, type_idx])
            relation_types.extend([type_idx, type_idx])
        
        # Handle empty edges
        if not edges:
            edges = [(i, i) for i in range(len(node_features))]
            edge_features = [np.zeros(1 + len(self.kg_relation_types) + 1) for _ in edges]
            edge_types = [self.kg_relation_types['OTHER']] * len(edges)
            relation_types = edge_types.copy()
        
        # Convert to tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)
        relation_types_tensor = torch.tensor(relation_types, dtype=torch.long)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types_tensor,
            edge_types=edge_types_tensor,
            relation_types=relation_types_tensor,
            num_nodes=len(node_features)
        )
        
        # Create metadata
        metadata = GraphMetadata(
            graph_id=subgraph_id,
            num_nodes=len(node_features),
            num_edges=len(edges),
            node_types={k: (node_types_tensor == v).sum().item() for k, v in self.kg_entity_types.items()},
            edge_types={k: (edge_types_tensor == v).sum().item() for k, v in self.kg_relation_types.items()},
            source_documents=[f"KG_subgraph_{subgraph_id}"],
            timestamp=0.0  # KG data is not temporal
        )
        
        self.logger.info(f"Created KG subgraph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
        
        return data, metadata
    
    def _create_empty_kg_subgraph(self, subgraph_id: str) -> Tuple[Data, GraphMetadata]:
        """Create an empty KG subgraph as fallback."""
        # Create minimal graph with single node
        x = torch.zeros(1, 768 + len(self.kg_entity_types) + 2, dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros(1, 1 + len(self.kg_relation_types) + 1, dtype=torch.float32)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=torch.tensor([self.kg_entity_types['OTHER']], dtype=torch.long),
            edge_types=torch.tensor([self.kg_relation_types['OTHER']], dtype=torch.long),
            relation_types=torch.tensor([self.kg_relation_types['OTHER']], dtype=torch.long),
            num_nodes=1
        )
        
        metadata = GraphMetadata(
            graph_id=subgraph_id,
            num_nodes=1,
            num_edges=1,
            node_types={'OTHER': 1},
            edge_types={'OTHER': 1},
            source_documents=[],
            timestamp=0.0
        )
        
        return data, metadata


class GraphAligner(LoggerMixin):
    """
    Aligns entities between literature graphs and knowledge graph subgraphs.
    
    Uses entity linking results and semantic similarity to create alignments
    that enable cross-modal learning.
    """
    
    def __init__(
        self,
        embeddings_model: BiomedicalEmbeddings,
        alignment_threshold: float = 0.8,
        max_alignments_per_entity: int = 3
    ):
        self.embeddings_model = embeddings_model
        self.alignment_threshold = alignment_threshold
        self.max_alignments_per_entity = max_alignments_per_entity
    
    def align_graphs(
        self,
        lit_graph: Data,
        kg_graph: Data,
        lit_metadata: GraphMetadata,
        kg_metadata: GraphMetadata,
        entity_linking_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[EntityAlignment]:
        """
        Create alignments between literature and KG entities.
        
        Args:
            lit_graph: Literature graph
            kg_graph: KG subgraph
            lit_metadata: Literature graph metadata
            kg_metadata: KG subgraph metadata
            entity_linking_results: Optional pre-computed entity linking results
            
        Returns:
            List of entity alignments
        """
        self.logger.info(f"Aligning graphs {lit_metadata.graph_id} and {kg_metadata.graph_id}")
        
        alignments = []
        
        # Use entity linking results if available
        if entity_linking_results:
            for result in entity_linking_results:
                alignment = EntityAlignment(
                    lit_entity_id=result['lit_entity_id'],
                    kg_entity_id=result['kg_entity_id'],
                    similarity_score=result['similarity_score'],
                    alignment_method='entity_linking',
                    confidence=result['confidence']
                )
                alignments.append(alignment)
        
        # Compute semantic similarity alignments
        lit_embeddings = lit_graph.x[:, :768]  # Assume first 768 dims are embeddings
        kg_embeddings = kg_graph.x[:, :768]
        
        if lit_embeddings.size(0) > 0 and kg_embeddings.size(0) > 0:
            # Compute similarity matrix
            similarity_matrix = torch.mm(lit_embeddings, kg_embeddings.t())
            similarity_matrix = torch.nn.functional.cosine_similarity(
                lit_embeddings.unsqueeze(1), 
                kg_embeddings.unsqueeze(0), 
                dim=2
            )
            
            # Find high-similarity alignments
            for lit_idx in range(similarity_matrix.size(0)):
                similarities = similarity_matrix[lit_idx]
                top_similarities, top_indices = torch.topk(
                    similarities, 
                    min(self.max_alignments_per_entity, similarities.size(0))
                )
                
                for sim_score, kg_idx in zip(top_similarities, top_indices):
                    if sim_score.item() > self.alignment_threshold:
                        alignment = EntityAlignment(
                            lit_entity_id=f"lit_node_{lit_idx}",
                            kg_entity_id=f"kg_node_{kg_idx}",
                            similarity_score=sim_score.item(),
                            alignment_method='semantic_similarity',
                            confidence=sim_score.item()
                        )
                        alignments.append(alignment)
        
        self.logger.info(f"Created {len(alignments)} entity alignments")
        return alignments


class GraphConstructor(LoggerMixin):
    """
    Main graph construction coordinator.
    
    Orchestrates the construction of literature graphs, KG subgraphs,
    and their alignment for hybrid GNN training.
    """
    
    def __init__(
        self,
        embeddings_model: BiomedicalEmbeddings,
        output_dir: Path,
        **kwargs
    ):
        self.embeddings_model = embeddings_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.lit_graph_builder = LiteratureGraphBuilder(embeddings_model, **kwargs)
        self.kg_extractor = KGSubgraphExtractor(**kwargs)
        self.graph_aligner = GraphAligner(embeddings_model, **kwargs)
        
        # Storage
        self.constructed_graphs = {}
        self.alignments = {}
    
    def construct_training_graphs(
        self,
        literature_data: List[Dict[str, Any]],
        kg_data: Dict[str, Any],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Construct aligned graph pairs for training.
        
        Args:
            literature_data: List of document groups for graph construction
            kg_data: Knowledge graph data
            batch_size: Number of graphs to construct in each batch
            
        Returns:
            List of training examples with aligned graph pairs
        """
        training_examples = []
        
        for i in range(0, len(literature_data), batch_size):
            batch_data = literature_data[i:i+batch_size]
            
            for j, doc_group in enumerate(batch_data):
                graph_id = f"lit_graph_{i+j}"
                
                # Build literature graph
                lit_graph, lit_metadata = self.lit_graph_builder.build_literature_graph(
                    documents=doc_group['documents'],
                    graph_id=graph_id
                )
                
                # Extract entities for KG subgraph
                target_entities = self._extract_target_entities(doc_group['documents'])
                
                # Extract KG subgraph
                kg_subgraph_id = f"kg_subgraph_{i+j}"
                kg_subgraph, kg_metadata = self.kg_extractor.extract_subgraph(
                    kg_data=kg_data,
                    target_entities=target_entities,
                    subgraph_id=kg_subgraph_id
                )
                
                # Create alignments
                alignments = self.graph_aligner.align_graphs(
                    lit_graph=lit_graph,
                    kg_graph=kg_subgraph,
                    lit_metadata=lit_metadata,
                    kg_metadata=kg_metadata,
                    entity_linking_results=doc_group.get('entity_links')
                )
                
                # Create training example
                training_example = {
                    'id': f"training_example_{i+j}",
                    'lit_graph': lit_graph,
                    'kg_graph': kg_subgraph,
                    'lit_metadata': lit_metadata,
                    'kg_metadata': kg_metadata,
                    'alignments': alignments,
                    'labels': self._create_labels(alignments, doc_group)
                }
                
                training_examples.append(training_example)
                
                # Save graphs
                self._save_training_example(training_example)
        
        self.logger.info(f"Constructed {len(training_examples)} training graph pairs")
        return training_examples
    
    def _extract_target_entities(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract target entities for KG subgraph extraction."""
        entities = set()
        
        for doc in documents:
            for entity in doc.get('entities', []):
                entities.add(self.lit_graph_builder._normalize_entity_id(entity['text']))
        
        return list(entities)
    
    def _create_labels(
        self, 
        alignments: List[EntityAlignment], 
        doc_group: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Create training labels from alignments and document group."""
        # Link prediction labels (binary)
        num_alignments = len(alignments)
        link_labels = torch.ones(num_alignments, dtype=torch.float32)
        
        # Relation prediction labels (multiclass)
        relation_labels = torch.zeros(num_alignments, dtype=torch.long)
        
        # Confidence labels
        confidence_labels = torch.tensor([
            alignment.confidence for alignment in alignments
        ], dtype=torch.float32)
        
        return {
            'link_labels': link_labels,
            'relation_labels': relation_labels,
            'confidence_labels': confidence_labels
        }
    
    def _save_training_example(self, example: Dict[str, Any]):
        """Save training example to disk."""
        example_path = self.output_dir / f"{example['id']}.pkl"
        
        # Prepare data for saving (remove non-serializable objects)
        save_data = {
            'id': example['id'],
            'lit_graph': example['lit_graph'],
            'kg_graph': example['kg_graph'],
            'lit_metadata': example['lit_metadata'],
            'kg_metadata': example['kg_metadata'],
            'alignments': example['alignments'],
            'labels': example['labels']
        }
        
        with open(example_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_training_examples(self, example_ids: List[str]) -> List[Dict[str, Any]]:
        """Load training examples from disk."""
        examples = []
        
        for example_id in example_ids:
            example_path = self.output_dir / f"{example_id}.pkl"
            
            if example_path.exists():
                with open(example_path, 'rb') as f:
                    example = pickle.load(f)
                    examples.append(example)
            else:
                self.logger.warning(f"Training example {example_id} not found")
        
        return examples


# --------------------- Compatibility builders expected by tests ---------------------

class GraphBuilder(LoggerMixin):
    def validate_graph(self, graph: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        nodes = graph.get('nodes')
        edges = graph.get('edges')
        if not isinstance(nodes, list):
            errors.append('nodes must be a list')
        if not isinstance(edges, list):
            errors.append('edges must be a list')
        if not errors:
            for i, n in enumerate(nodes):
                if 'id' not in n:
                    errors.append(f'node[{i}] missing id')
            for i, e in enumerate(edges):
                if 'source' not in e or 'target' not in e:
                    errors.append(f'edge[{i}] missing source/target')
        return (len(errors) == 0, errors)

    def compute_graph_stats(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        node_types: Dict[str, int] = {}
        edge_types: Dict[str, int] = {}
        for n in nodes:
            t = n.get('type', 'ENTITY')
            node_types[t] = node_types.get(t, 0) + 1
        for e in edges:
            t = e.get('type', 'RELATION')
            edge_types[t] = edge_types.get(t, 0) + 1
        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'node_types': node_types,
            'edge_types': edge_types,
        }

    def to_pytorch_geometric(self, graph: Dict[str, Any]) -> Data:
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        num_nodes = len(nodes)
        x = torch.zeros((num_nodes, 64), dtype=torch.float32)
        node_index = {n.get('id', str(i)): i for i, n in enumerate(nodes)}

        if edges:
            edge_idx_pairs: List[Tuple[int, int]] = []
            edge_attr_list: List[List[float]] = []
            for e in edges:
                s = node_index.get(e.get('source'))
                t = node_index.get(e.get('target'))
                if s is None or t is None:
                    continue
                edge_idx_pairs.append((s, t))
                edge_attr_list.append([1.0])
            if not edge_idx_pairs:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float32)
            else:
                edge_index = torch.tensor(np.array(edge_idx_pairs).T, dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


class KGGraphBuilder(LoggerMixin):
    def build_graph(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        nodes = kg.get('nodes', [])
        edges = kg.get('edges', [])
        return {'nodes': nodes, 'edges': edges}