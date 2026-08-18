"""
Resolving literature mentions into entity-level nodes.

Biomedical NER emits one mention per occurrence: a corpus discussing BRCA1 in
sixty papers yields sixty separate "BRCA1" mentions. Treating each as its own
graph node makes the literature graph mention-level, where degree, centrality
and neighbourhood are properties of how often a paper repeated a name rather
than of the entity itself — and every downstream GNN feature inherits that.

This module collapses mentions into one node per distinct entity, keeping the
mention detail as evidence so provenance is not lost.

Grouping mirrors the knowledge graph's resolution cascade: entities are keyed
by normalized surface form *within* an entity type, so "BRCA-1" and "BRCA1"
merge while "ALL" the gene and "ALL" the leukemia stay separate.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils.logging import LoggerMixin


@dataclass
class ResolvedLiteratureEntity:
    """One entity, aggregated across every mention of it in the corpus."""

    node_id: str
    canonical_text: str
    label: str
    surface_forms: List[str] = field(default_factory=list)
    mention_count: int = 0
    document_ids: List[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    max_confidence: float = 0.0

    @property
    def document_count(self) -> int:
        """How many distinct documents mention this entity."""
        return len(self.document_ids)


class LiteratureEntityResolver(LoggerMixin):
    """
    Collapse mention-level entities into entity-level nodes.

    Usage is two-phase: register every mention, then resolve. The returned
    mapping from mention key to node id lets callers rewrite edges that were
    expressed in terms of mentions.
    """

    def __init__(self, min_surface_length: int = 1):
        """
        Args:
            min_surface_length: Mentions whose text is shorter than this are
                ignored. NER noise is dominated by very short acronyms.
        """
        self.min_surface_length = min_surface_length
        self._mentions: List[Dict[str, Any]] = []

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize a surface form for grouping.

        Folds case and the punctuation that distinguishes BRCA1 / BRCA-1 /
        BRCA 1, matching the knowledge graph's normalization so the two sides
        group consistently.
        """
        return re.sub(r"[\s\-_/.]+", "", str(text).lower().strip())

    def add_mention(
        self,
        mention_key: str,
        text: str,
        label: str,
        confidence: float = 1.0,
        document_id: Optional[str] = None,
    ) -> None:
        """
        Register one mention.

        Args:
            mention_key: Caller's stable handle for this occurrence, returned
                in the mapping so mention-keyed edges can be rewritten.
            text: The surface form as it appeared.
            label: Entity type; grouping never crosses types.
            confidence: NER confidence for this mention.
            document_id: Document the mention came from.
        """
        if len(str(text).strip()) < self.min_surface_length:
            return

        self._mentions.append({
            "mention_key": mention_key,
            "text": str(text).strip(),
            "label": str(label or "UNKNOWN").upper(),
            "confidence": float(confidence or 0.0),
            "document_id": document_id,
        })

    @staticmethod
    def _canonical_form(surface_counts: Counter) -> str:
        """
        Choose the representative surface form for a group.

        The most frequent form wins. Ties break toward the *shortest* form:
        every member of a group normalizes to the same string, so they differ
        only by incidental punctuation and spacing, and the shortest carries
        least of it ("BRCA1" over "BRCA-1"). Alphabetical order settles the
        remainder so the result is deterministic.
        """
        return min(
            surface_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )[0]

    def resolve(
        self, id_prefix: str = "lit"
    ) -> Tuple[List[ResolvedLiteratureEntity], Dict[str, str]]:
        """
        Group registered mentions into entity-level nodes.

        Returns:
            (entities, mention_to_node) where mention_to_node maps every
            registered mention key to the node id it now belongs to.
        """
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for mention in self._mentions:
            groups[(mention["label"], self.normalize(mention["text"]))].append(mention)

        entities: List[ResolvedLiteratureEntity] = []
        mention_to_node: Dict[str, str] = {}

        # Deterministic ordering so node ids are stable across runs
        for index, key in enumerate(sorted(groups)):
            members = groups[key]
            label, _ = key
            node_id = f"{id_prefix}_{index}"

            surface_counts = Counter(m["text"] for m in members)
            confidences = [m["confidence"] for m in members]
            documents = [m["document_id"] for m in members if m["document_id"]]

            entities.append(ResolvedLiteratureEntity(
                node_id=node_id,
                canonical_text=self._canonical_form(surface_counts),
                label=label,
                surface_forms=sorted(surface_counts),
                mention_count=len(members),
                # Sorted+unique: document_count must not double-count a paper
                # that mentions the same entity several times.
                document_ids=sorted(set(documents)),
                mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
                max_confidence=max(confidences) if confidences else 0.0,
            ))

            for member in members:
                mention_to_node[member["mention_key"]] = node_id

        self.logger.info(
            f"Resolved {len(self._mentions)} literature mentions into "
            f"{len(entities)} entities"
        )
        return entities, mention_to_node


def aggregate_edges(
    edges: Iterable[Dict[str, Any]],
    key_fields: Tuple[str, ...] = ("source", "target", "predicate"),
) -> List[Dict[str, Any]]:
    """
    Collapse duplicate edges produced by resolving their endpoints.

    Once sixty BRCA1 mentions become one node, the sixty identical edges they
    carried become one relationship supported sixty times. That support is kept
    as ``mention_count`` rather than discarded, and confidence is taken as the
    maximum across the duplicates — the strongest evidence for the claim.

    Args:
        edges: Edge dicts to aggregate.
        key_fields: Fields whose combination identifies one relationship.

    Returns:
        One edge per distinct key, ordered by first appearance.
    """
    merged: Dict[Tuple, Dict[str, Any]] = {}

    for edge in edges:
        key = tuple(edge.get(f) for f in key_fields)
        if key in merged:
            existing = merged[key]
            existing["mention_count"] += 1
            existing["confidence"] = max(
                existing.get("confidence", 0.0), edge.get("confidence", 0.0)
            )
        else:
            merged[key] = {**edge, "mention_count": 1}

    return list(merged.values())
