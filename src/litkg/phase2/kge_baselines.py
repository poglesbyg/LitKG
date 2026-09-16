"""
Knowledge-graph embedding baselines (PyKEEN), scored by this project's harness.

Everything measured here so far has been either a path counter over a flattened
graph or a GNN that treats relation types as an edge feature. Multi-relational
embedding models -- TransE, ComplEx, RotatE -- are the standard tool for exactly
this task and had never been tried, so the question of whether 0.8138 is
method-limited or data-limited was open.

Comparability is the whole point of this module
-----------------------------------------------
PyKEEN ships its own ranking evaluation, and its numbers are **not** comparable
to anything else in this repository: it ranks against all entities rather than a
degree-matched negative sample, and it has no notion of the temporal holdout
used here. Reporting a PyKEEN MRR next to this project's MRR would be comparing
two different tasks that happen to share a metric name.

So these models are wrapped as ordinary `LinkPredictor`s and scored by the same
harness, on the same split, against the same degree-matched negatives, as the
GNN and the path counters.

Scoring an untyped pair
-----------------------
The harness asks "how likely is an edge between u and v", with no relation.
A KGE model scores (head, relation, tail) triples, so a pair score has to
aggregate over relations. This takes the maximum over relation types and over
both directions: the strongest claim any relation makes about the pair. Mean
would dilute a confident single-relation prediction across ten irrelevant ones.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from litkg.evaluation.baselines import LinkPredictor
from litkg.utils.logging import LoggerMixin

Edge = Tuple[str, str]

# PyKEEN model names this wrapper has been exercised with.
SUPPORTED = ("TransE", "ComplEx", "RotatE", "DistMult")


class KGEmbeddingPredictor(LinkPredictor, LoggerMixin):
    """A PyKEEN model behind the LinkPredictor interface."""

    def __init__(
        self,
        model: str = "RotatE",
        embedding_dim: int = 128,
        epochs: int = 200,
        seed: int = 0,
        edge_predicates: Optional[Dict[Edge, str]] = None,
        batch_size: int = 512,
        device: str = "cpu",
    ):
        if model not in SUPPORTED:
            raise ValueError(f"unsupported model {model!r}; try one of {SUPPORTED}")
        self.model = model
        self.name = f"kge_{model.lower()}"
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.seed = seed
        self.edge_predicates = edge_predicates or {}
        self.batch_size = batch_size
        self.device = device
        self._trained = None
        self._factory = None

    def _triples(self, graph: nx.Graph) -> np.ndarray:
        rows: List[Tuple[str, str, str]] = []
        for u, v in graph.edges():
            key = (u, v) if u <= v else (v, u)
            relation = self.edge_predicates.get(key) or "RELATED_TO"
            # Both directions. The harness treats pairs as unordered, and a
            # model trained on one direction only would score the reverse at
            # chance for no reason the task cares about.
            rows.append((str(u), str(relation), str(v)))
            rows.append((str(v), str(relation), str(u)))
        return np.array(rows, dtype=str)

    def fit(self, graph: nx.Graph) -> "KGEmbeddingPredictor":
        import torch
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory

        self.graph = graph
        triples = self._triples(graph)
        if len(triples) == 0:
            self._trained = None
            return self

        self._factory = TriplesFactory.from_labeled_triples(
            triples, create_inverse_triples=False
        )

        result = pipeline(
            training=self._factory,
            testing=self._factory,       # never read; the harness does the scoring
            model=self.model,
            model_kwargs={"embedding_dim": self.embedding_dim},
            training_kwargs={
                "num_epochs": self.epochs,
                "batch_size": self.batch_size,
                "use_tqdm": False,
                "use_tqdm_batch": False,
            },
            random_seed=self.seed,
            device=self.device,
            # PyKEEN's own evaluation is ignored deliberately: it ranks against
            # every entity rather than the degree-matched negatives this project
            # uses, so its numbers would not be comparable to anything else
            # here. Only the trained model is taken from the result.
            evaluation_kwargs={"use_tqdm": False},
        )
        self._trained = result.model
        self._trained.eval()

        self._entity_ids = self._factory.entity_to_id
        self._relation_ids = self._factory.relation_to_id
        self._all_relations = torch.tensor(
            sorted(self._relation_ids.values()), dtype=torch.long
        )
        return self

    def score(self, u: str, v: str) -> float:
        import torch

        if self._trained is None:
            return 0.0
        head = self._entity_ids.get(str(u))
        tail = self._entity_ids.get(str(v))
        if head is None or tail is None:
            # A node absent from training has no embedding; abstain rather than
            # inventing one.
            return 0.0

        relations = self._all_relations
        n = len(relations)
        with torch.no_grad():
            forward = torch.stack(
                [torch.full((n,), head, dtype=torch.long), relations,
                 torch.full((n,), tail, dtype=torch.long)], dim=1
            )
            reverse = torch.stack(
                [torch.full((n,), tail, dtype=torch.long), relations,
                 torch.full((n,), head, dtype=torch.long)], dim=1
            )
            scores = self._trained.score_hrt(torch.cat([forward, reverse], dim=0))
        return float(scores.max().item())

    def score_pairs(self, pairs: Sequence[Edge]) -> List[float]:
        return [self.score(u, v) for u, v in pairs]
