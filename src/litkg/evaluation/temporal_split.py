"""
Temporal splitting of knowledge graph edges.

A random train/test split leaks: an association discovered in 2005 and one
discovered in 2021 are equally likely to land in the test set, so a model can
score well by memorising co-occurrence patterns it already saw. Splitting on
when the supporting paper was published asks the question that matters -- given
what was known by year Y, can the graph predict what came after?
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from litkg.utils.logging import LoggerMixin

# CIVIC citations carry their year in the citation string ("Levine et al.,
# 2005"). last_review_date is not usable for this: 4171 of 4254 rows share a
# 2023 bulk re-review timestamp, which says when a curator last looked at the
# record, not when the finding was published.
_YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")

Edge = Tuple[str, str]


@dataclass(frozen=True)
class RelationRecord:
    """
    One asserted relation, with the evidence attributes the graph discards.

    Flattening to a simple undirected graph collapses 13194 CIVIC relations
    into 6645 pairs and drops the predicate, direction, curator confidence and
    negation flag along the way. Carrying them this far lets a predictor use
    them without changing what the structural baselines see.
    """

    subject: str
    object: str
    year: Optional[int] = None
    predicate: str = ""
    confidence: float = 1.0
    negated: bool = False


@dataclass
class EdgeEvidence:
    """Evidence supporting one node pair, aggregated across relations."""

    support: int = 0
    total_confidence: float = 0.0
    negated_count: int = 0
    predicates: Dict[str, int] = field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        return self.total_confidence / self.support if self.support else 0.0

    @property
    def negated_fraction(self) -> float:
        return self.negated_count / self.support if self.support else 0.0

    @property
    def dominant_predicate(self) -> str:
        if not self.predicates:
            return ""
        return max(self.predicates.items(), key=lambda kv: kv[1])[0]

    def weight(self) -> float:
        """
        Edge weight from evidence strength.

        Repeated assertion is support, so it enters logarithmically rather than
        linearly. Negated evidence argues against the association and pulls the
        weight down without erasing the edge -- "does not support" is a claim
        about the association, not an absence of one.
        """
        if not self.support:
            return 1.0
        magnitude = self.mean_confidence * math.log1p(self.support)
        penalty = 1.0 - 0.5 * self.negated_fraction
        return max(0.05, magnitude * penalty)


def extract_publication_year(citation: Any) -> Optional[int]:
    """
    Pull the publication year out of a CIVIC citation string.

    Returns the last year-like token, since author lists occasionally contain
    numbers and the year is conventionally last ("Levine et al., 2005").
    """
    if citation is None:
        return None
    matches = _YEAR_PATTERN.findall(str(citation))
    return int(matches[-1]) if matches else None


@dataclass
class TemporalSplit:
    """
    A train/test split of graph edges on publication year.

    Attributes:
        cutoff_year: Edges from papers published before this year are training.
        train_edges: Undirected node pairs known before the cutoff.
        test_edges: Node pairs first asserted at or after the cutoff, with both
            endpoints already present in the training graph.
        backbone_edges: Structural edges with no publication date (gene->variant
            membership). Always in training; they are ontology, not discovery.
        excluded_already_known: Pairs asserted both before and after the cutoff.
            These are not discoveries and must not be scored -- counting them
            inflates every metric, since the model saw them in training.
        excluded_cold_start: Pairs where an endpoint is absent from the training
            graph. No topological method can score these; they are reported
            rather than silently counted as failures.
        edge_years: First publication year observed for each pair.
    """

    cutoff_year: int
    train_edges: Set[Edge]
    test_edges: Set[Edge]
    backbone_edges: Set[Edge] = field(default_factory=set)
    excluded_already_known: int = 0
    excluded_cold_start: int = 0
    edge_years: Dict[Edge, int] = field(default_factory=dict)
    # Evidence aggregated from pre-cutoff relations only. Including post-cutoff
    # evidence would leak the very associations being predicted into the weight
    # of the edges used to predict them.
    edge_evidence: Dict[Edge, EdgeEvidence] = field(default_factory=dict)

    def edge_weights(self) -> Dict[Edge, float]:
        """Evidence-derived weight per training pair; 1.0 where unknown."""
        return {pair: ev.weight() for pair, ev in self.edge_evidence.items()}

    @property
    def train_nodes(self) -> Set[str]:
        nodes: Set[str] = set()
        for u, v in self.train_edges | self.backbone_edges:
            nodes.add(u)
            nodes.add(v)
        return nodes

    def summary(self) -> Dict[str, Any]:
        return {
            "cutoff_year": self.cutoff_year,
            "train_edges": len(self.train_edges),
            "backbone_edges": len(self.backbone_edges),
            "test_edges": len(self.test_edges),
            "train_nodes": len(self.train_nodes),
            "excluded_already_known": self.excluded_already_known,
            "excluded_cold_start": self.excluded_cold_start,
        }


def _normalize(u: str, v: str) -> Edge:
    """Undirected pairs must have one canonical form or lookups miss."""
    return (u, v) if u <= v else (v, u)


class TemporalSplitter(LoggerMixin):
    """Builds temporal splits from dated and undated edges."""

    @staticmethod
    def _as_record(item: Any) -> RelationRecord:
        """Accept plain (source, target, year) triples or full records."""
        if isinstance(item, RelationRecord):
            return item
        source, target, year = item
        return RelationRecord(subject=source, object=target, year=year)

    def split(
        self,
        dated_edges: Iterable[Any],
        cutoff_year: int,
        backbone_edges: Iterable[Edge] = (),
    ) -> TemporalSplit:
        """
        Split edges into train and test on publication year.

        Args:
            dated_edges: (source, target, year) triples. A year of None means
                the edge has no date and is treated as backbone.
            cutoff_year: Edges before this year train; at or after, test.
            backbone_edges: Undated structural edges, always training.

        Returns:
            A TemporalSplit with leaked and cold-start pairs removed and
            counted.
        """
        # A pair can be asserted by several papers across many years. What
        # matters is the first time it was asserted -- that is when it stopped
        # being a discovery.
        first_year: Dict[Edge, int] = {}
        undated: Set[Edge] = set()
        evidence: Dict[Edge, EdgeEvidence] = {}

        for item in dated_edges:
            record = self._as_record(item)
            source, target, year = record.subject, record.object, record.year
            if source == target:
                continue  # self-loops are not predictions
            pair = _normalize(source, target)

            # Only evidence published before the cutoff may inform edge
            # weights; anything later is knowledge the model is not entitled to.
            if year is not None and year < cutoff_year:
                entry = evidence.setdefault(pair, EdgeEvidence())
                entry.support += 1
                entry.total_confidence += record.confidence
                entry.negated_count += int(record.negated)
                if record.predicate:
                    entry.predicates[record.predicate] = (
                        entry.predicates.get(record.predicate, 0) + 1
                    )

            if year is None:
                undated.add(pair)
                continue
            if pair not in first_year or year < first_year[pair]:
                first_year[pair] = year

        backbone = {_normalize(u, v) for u, v in backbone_edges if u != v}
        backbone |= undated

        train: Set[Edge] = set()
        candidate_test: Set[Edge] = set()
        for pair, year in first_year.items():
            if year < cutoff_year:
                train.add(pair)
            else:
                candidate_test.add(pair)

        # A pair whose first assertion is after the cutoff can still appear in
        # the training graph through the backbone. Predicting an edge that is
        # already in the training graph is not prediction.
        known = train | backbone
        already_known = candidate_test & known
        candidate_test -= already_known

        train_nodes: Set[str] = set()
        for u, v in known:
            train_nodes.add(u)
            train_nodes.add(v)

        cold_start = {
            pair for pair in candidate_test
            if pair[0] not in train_nodes or pair[1] not in train_nodes
        }
        test = candidate_test - cold_start

        split = TemporalSplit(
            cutoff_year=cutoff_year,
            train_edges=train,
            test_edges=test,
            backbone_edges=backbone,
            excluded_already_known=len(already_known),
            excluded_cold_start=len(cold_start),
            edge_years=first_year,
            edge_evidence=evidence,
        )

        self.logger.info(
            f"Temporal split at {cutoff_year}: {len(train)} train, {len(test)} test "
            f"(excluded {len(already_known)} already-known, {len(cold_start)} cold-start)"
        )
        return split


def build_temporal_split(
    dated_edges: Iterable[Any],
    cutoff_year: int,
    backbone_edges: Iterable[Edge] = (),
) -> TemporalSplit:
    """Convenience wrapper around TemporalSplitter.split()."""
    return TemporalSplitter().split(dated_edges, cutoff_year, backbone_edges)


def year_distribution(dated_edges: Iterable[Any]) -> Dict[int, int]:
    """Count distinct pairs by first assertion year, for choosing a cutoff."""
    first_year: Dict[Edge, int] = {}
    for item in dated_edges:
        record = TemporalSplitter._as_record(item)
        source, target, year = record.subject, record.object, record.year
        if year is None or source == target:
            continue
        pair = _normalize(source, target)
        if pair not in first_year or year < first_year[pair]:
            first_year[pair] = year

    counts: Dict[int, int] = defaultdict(int)
    for year in first_year.values():
        counts[year] += 1
    return dict(sorted(counts.items()))
