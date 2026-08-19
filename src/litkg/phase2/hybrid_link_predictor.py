"""
HybridGNNModel as a link predictor, scored through the standard harness.

The cross-modal architecture this project is built around had only ever run on
`torch.randn`. Two things had to be true before it could be judged: it needed
real graphs, and it needed to be able to express a per-pair prediction at all.

The second was not true. Fusion combined the literature and knowledge graphs at
the *graph* level, so `fused_representation` had exactly one row and
`entity_pairs` could only index row 0 -- which is why the synthetic demo passed
`[[0, 0]]` with the comment "only use valid indices". Cross-attention already
preserved the node dimension, so the fix was to fuse node embeddings rather than
pooled graph embeddings and index the enhanced KG nodes.

This wrapper implements the same `LinkPredictor` interface as everything else,
so it runs through the identical split, negatives and metrics. The number to
beat is AUC 0.748, from the far simpler GNN + weighted-L3 hybrid.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from litkg.evaluation.baselines import LinkPredictor
from litkg.phase2.hybrid_gnn import HybridGNNModel
from litkg.utils.logging import LoggerMixin

Edge = Tuple[str, str]


@dataclass
class HybridTrainingConfig:
    """Kept deliberately close to the simpler model's settings, so a difference
    in score is attributable to the architecture rather than the schedule."""

    hidden_dim: int = 128
    num_gnn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 40
    negatives_per_positive: int = 10
    supervision_fraction: float = 0.3
    resample_every: int = 5
    seed: int = 0
    text_dim: int = 64
    # A learned identity per node, concatenated to the static features. Without
    # it the KG encoder over-smooths: measured on the real graph, mean pairwise
    # cosine between distinct nodes goes from 0.79 at the input to 0.998 after
    # the encoder and 1.000 after fusion, so every node ends up the same vector
    # and the model scores at chance. The simpler baseline carries such an
    # embedding, so withholding one here would not be a fair comparison.
    embedding_dim: int = 64


class HybridGNNLinkPredictor(LinkPredictor, LoggerMixin):
    """Trains HybridGNNModel on the real graph and scores pairs with it."""

    name = "hybrid_gnn"

    def __init__(
        self,
        config: Optional[HybridTrainingConfig] = None,
        node_types: Optional[Dict[str, str]] = None,
        node_text: Optional[Dict[str, str]] = None,
        literature_graph: Optional[nx.Graph] = None,
        text_encoder: Optional[Any] = None,
    ):
        self.config = config or HybridTrainingConfig()
        self.node_types = node_types or {}
        self.node_text = node_text or {}
        # The literature side of the cross-modal pair. Without it the model has
        # nothing to attend to and reduces to an ordinary graph encoder, so its
        # absence is worth stating rather than silently tolerating.
        self.literature_graph = literature_graph
        self.text_encoder = text_encoder
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------

    def _features(self, graph: nx.Graph, nodes: Sequence[str]) -> torch.Tensor:
        """Node features: text embedding, entity type, log degree."""
        types = sorted({self.node_types.get(n, "UNKNOWN") for n in nodes})
        type_index = {t: i for i, t in enumerate(types)}

        text_matrix = None
        if self.node_text:
            from litkg.phase2.node_features import NodeTextEncoder

            encoder = self.text_encoder or NodeTextEncoder()
            known = {n: self.node_text[n] for n in nodes if n in self.node_text}
            vectors = encoder.encode_nodes(known)
            if vectors:
                width = len(next(iter(vectors.values())))
                text_matrix = np.zeros((len(nodes), width), dtype=np.float32)
                for i, node in enumerate(nodes):
                    if node in vectors:
                        text_matrix[i] = vectors[node]

        extra = np.zeros((len(nodes), len(types) + 1), dtype=np.float32)
        for i, node in enumerate(nodes):
            extra[i, type_index[self.node_types.get(node, "UNKNOWN")]] = 1.0
            extra[i, -1] = math.log1p(graph.degree(node) if node in graph else 0)

        parts = [extra] if text_matrix is None else [text_matrix, extra]
        return torch.tensor(np.concatenate(parts, axis=1), dtype=torch.float32)

    @staticmethod
    def _edge_index(index: Dict[str, int], edges: Sequence[Edge]) -> torch.Tensor:
        pairs = [(index[u], index[v]) for u, v in edges if u in index and v in index]
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        source = [p[0] for p in pairs] + [p[1] for p in pairs]
        target = [p[1] for p in pairs] + [p[0] for p in pairs]
        return torch.tensor([source, target], dtype=torch.long)

    def _literature_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Literature-side inputs for cross-attention.

        With no literature graph the model still runs, but it attends to a
        single zero node and the cross-modal half contributes nothing. That is
        reported rather than hidden, because a cross-modal score obtained
        without a second modality is not a cross-modal result.
        """
        if self.literature_graph is None or self.literature_graph.number_of_nodes() == 0:
            self.logger.warning(
                "No literature graph supplied; cross-modal attention has nothing "
                "to attend to and the model degenerates to a KG encoder"
            )
            width = self.kg_features.shape[1]
            return (torch.zeros((1, width)), torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0, 1)))

        nodes = sorted(self.literature_graph.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        features = self._features(self.literature_graph, nodes)
        edge_index = self._edge_index(index, list(self.literature_graph.edges()))
        edge_attr = torch.ones((edge_index.shape[1], 1))
        return features, edge_index, edge_attr

    # ------------------------------------------------------------------

    def fit(self, graph: nx.Graph) -> "HybridGNNLinkPredictor":
        config = self.config
        torch.manual_seed(config.seed)
        rng = random.Random(config.seed)

        self.graph = graph
        self.nodes = sorted(graph.nodes())
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.kg_features = self._features(graph, self.nodes)

        self.node_embedding = torch.nn.Embedding(
            len(self.nodes), config.embedding_dim
        )
        torch.nn.init.xavier_uniform_(self.node_embedding.weight)
        self.node_ids = torch.arange(len(self.nodes))

        lit_x, lit_edge_index, lit_edge_attr = self._literature_tensors()
        self.lit_inputs = (lit_x, lit_edge_index, lit_edge_attr)

        self.model = HybridGNNModel(
            lit_node_dim=lit_x.shape[1],
            lit_edge_dim=max(lit_edge_attr.shape[1], 1),
            kg_node_dim=self.kg_features.shape[1] + config.embedding_dim,
            kg_edge_dim=1,
            kg_relation_dim=1,
            hidden_dim=config.hidden_dim,
            num_gnn_layers=config.num_gnn_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.node_embedding.parameters()),
            lr=config.learning_rate, weight_decay=config.weight_decay,
        )

        all_edges = [(u, v) if u <= v else (v, u) for u, v in graph.edges()]
        known: Set[Tuple[int, int]] = set()
        for u, v in all_edges:
            a, b = self.node_index[u], self.node_index[v]
            known.add((a, b) if a <= b else (b, a))

        shuffled = list(all_edges)
        rng.shuffle(shuffled)
        split_at = max(1, int(len(shuffled) * 0.1))
        validation = [(self.node_index[u], self.node_index[v])
                      for u, v in shuffled[:split_at]]
        trainable = shuffled[split_at:]
        validation_negatives = self._negatives(validation, known, random.Random(99))

        best, best_state, stalled = -1.0, None, 0
        message_index = supervision = negatives = None

        for epoch in range(config.epochs):
            if epoch % config.resample_every == 0:
                # Same edge masking as the simpler model: supervising on edges
                # the encoder can see teaches it to read its own input.
                pool = list(trainable)
                rng.shuffle(pool)
                cut = int(len(pool) * config.supervision_fraction)
                message_index = self._edge_index(self.node_index, pool[cut:])
                supervision = [(self.node_index[u], self.node_index[v])
                               for u, v in pool[:cut]]
                negatives = self._negatives(supervision, known, rng)

            self.model.train()
            optimizer.zero_grad()
            representations = self._encode(message_index)
            loss = self._loss(representations, supervision, negatives)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 or epoch == config.epochs - 1:
                auc = self._validation_auc(message_index, validation, validation_negatives)
                self.history.append({"epoch": epoch, "loss": float(loss.item()),
                                     "val_auc": auc})
                if auc > best:
                    best, stalled = auc, 0
                    best_state = ({k: v.detach().clone()
                                   for k, v in self.model.state_dict().items()},
                                  self.node_embedding.weight.detach().clone())
                else:
                    stalled += 5
                    if stalled >= config.patience:
                        self.logger.info(f"Early stop at epoch {epoch}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state[0])
            with torch.no_grad():
                self.node_embedding.weight.copy_(best_state[1])
        self.best_validation_auc = best
        self.logger.info(f"Best validation AUC: {best:.4f}")

        self.model.eval()
        with torch.no_grad():
            self.final = self._encode(self._edge_index(self.node_index, all_edges))
        return self

    def _encode(self, kg_edge_index: torch.Tensor) -> torch.Tensor:
        lit_x, lit_edge_index, lit_edge_attr = self.lit_inputs
        kg_x = torch.cat([self.node_embedding(self.node_ids), self.kg_features], dim=1)
        outputs = self.model(
            lit_x=lit_x, lit_edge_index=lit_edge_index, lit_edge_attr=lit_edge_attr,
            kg_x=kg_x, kg_edge_index=kg_edge_index,
            kg_edge_attr=torch.ones((kg_edge_index.shape[1], 1)),
            kg_relation_types=torch.ones((kg_edge_index.shape[1], 1)),
        )
        return outputs["fused_representation"]

    def _negatives(self, positives, known, rng) -> List[Tuple[int, int]]:
        out = []
        count = len(self.nodes)
        for u, _v in positives:
            for _ in range(self.config.negatives_per_positive):
                for _attempt in range(20):
                    candidate = rng.randrange(count)
                    if candidate == u:
                        continue
                    pair = (u, candidate) if u <= candidate else (candidate, u)
                    if pair in known:
                        continue
                    out.append((u, candidate))
                    break
        return out

    def _score(self, representations: torch.Tensor, pairs) -> torch.Tensor:
        if not pairs:
            return torch.zeros(0)
        index = torch.tensor(pairs, dtype=torch.long)
        return (representations[index[:, 0]] * representations[index[:, 1]]).sum(-1)

    def _loss(self, representations, positives, negatives) -> torch.Tensor:
        positive = self._score(representations, positives)
        negative = self._score(representations, negatives)
        k = self.config.negatives_per_positive
        usable = min(len(positive), len(negative) // k) if k else 0
        if usable:
            # BPR, matching the simpler model: ranking per positive rather than
            # a global threshold.
            pos = positive[:usable].unsqueeze(1)
            neg = negative[: usable * k].view(usable, k)
            return -F.logsigmoid(pos - neg).mean()
        scores = torch.cat([positive, negative])
        labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def _validation_auc(self, message_index, positives, negatives) -> float:
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        with torch.no_grad():
            representations = self._encode(message_index)
            scores = torch.cat([
                self._score(representations, positives),
                self._score(representations, negatives),
            ]).numpy()
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        self.model.train()
        if len(set(labels.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    # ------------------------------------------------------------------

    def score_pairs(self, pairs: Sequence[Edge]) -> List[float]:
        known = [(self.node_index[u], self.node_index[v]) for u, v in pairs
                 if u in self.node_index and v in self.node_index]
        values = {}
        if known:
            with torch.no_grad():
                scored = self._score(self.final, known).numpy()
            position = 0
            for u, v in pairs:
                if u in self.node_index and v in self.node_index:
                    values[(u, v)] = float(scored[position])
                    position += 1
        return [values.get((u, v), -1e9) for u, v in pairs]

    def score(self, u: str, v: str) -> float:
        return self.score_pairs([(u, v)])[0]
