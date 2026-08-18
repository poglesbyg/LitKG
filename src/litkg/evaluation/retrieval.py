"""
A relevance-judged query set for retrieval, grounded in CIVIC citations.

Retrieval in this project was unmeasured: `k`, `max_hops` and the hub-traversal
degree cap were all set by judgement with nothing to check them against. The
obstacle was judgements -- there are no human relevance labels for this corpus,
and using an LLM to judge retrieval that feeds the same LLM is close to circular.

CIVIC supplies judgements for free. Every evidence row cites a PubMed paper
*and* states the relationship that paper supports: a molecular profile, a
disease, a therapy, and an evidence type. So for a question about that
relationship, the cited papers are relevant by construction, on a curator's
judgement rather than ours.

Two limits worth stating before any number is read:

**Judgements are incomplete.** Only papers CIVIC cites for a relationship are
marked relevant. A retrieved paper about the same gene may be perfectly
relevant and simply uncited, and it is scored as a miss. Every metric here is
therefore a lower bound, and the gap is largest for queries about well-studied
genes.

**Queries are templated, not natural.** They are generated deterministically
from the relationship, so they use consistent clinical phrasing. Real users are
messier, and a system tuned only on these may do worse on real questions.
"""

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from litkg.utils.logging import LoggerMixin

# Phrasing per CIVIC evidence type. The evidence type is what the cited paper
# actually establishes, so it decides what an honest question about it looks
# like -- asking "which therapies" of a prognostic paper would mark relevant
# papers as misses for answering a question they were never cited for.
QUERY_TEMPLATES: Dict[str, str] = {
    "Predictive": "Which therapies are effective for {profile} in {disease}?",
    "Prognostic": "What is the prognostic significance of {profile} in {disease}?",
    "Diagnostic": "How is {profile} used to diagnose {disease}?",
    "Predisposing": "How does {profile} predispose to {disease}?",
    "Functional": "What is the functional effect of {profile} in {disease}?",
    "Oncogenic": "Is {profile} an oncogenic driver in {disease}?",
}
DEFAULT_TEMPLATE = "What is the role of {profile} in {disease}?"


@dataclass
class RetrievalQuery:
    """One question with its curator-derived relevant documents."""

    query_id: str
    text: str
    relevant_pmids: List[str]
    profile: str = ""
    disease: str = ""
    evidence_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievalQuery":
        return cls(**payload)


@dataclass
class RetrievalMetrics:
    """Ranked-retrieval metrics, averaged over queries."""

    queries: int
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    precision_ci: Optional[tuple] = None
    recall_ci: Optional[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        def interval(value):
            return f" [{value[0]:.3f}, {value[1]:.3f}]" if value else ""
        return (
            f"queries={self.queries} k={self.k}  "
            f"P@k {self.precision_at_k:.3f}{interval(self.precision_ci)}  "
            f"R@k {self.recall_at_k:.3f}{interval(self.recall_ci)}  "
            f"MRR {self.mrr:.3f}  nDCG@k {self.ndcg_at_k:.3f}  "
            f"hit-rate {self.hit_rate:.3f}"
        )


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in ("", "nan", "none") else text


class QuerySetBuilder(LoggerMixin):
    """Derives queries and relevance judgements from CIVIC evidence."""

    def build(
        self,
        evidence: Any,
        min_relevant: int = 3,
        max_queries: Optional[int] = None,
        seed: int = 0,
    ) -> List[RetrievalQuery]:
        """
        Group evidence into queries with at least `min_relevant` cited papers.

        Grouping is by (molecular profile, disease, evidence type). Evidence
        type is part of the key rather than collapsed, because a paper cited as
        prognostic is not an answer to a question about therapy.
        """
        import pandas as pd  # local import keeps the module importable without it

        frame = evidence[
            evidence["source_type"].astype(str).str.upper() == "PUBMED"
        ].copy()
        frame["pmid"] = (
            frame["citation_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        )

        queries: List[RetrievalQuery] = []
        grouped = frame.groupby(
            ["molecular_profile", "disease", "evidence_type"], dropna=True
        )
        for (profile, disease, evidence_type), rows in grouped:
            profile, disease = _clean(profile), _clean(disease)
            evidence_type = _clean(evidence_type)
            if not profile or not disease:
                continue

            pmids = sorted({p for p in rows["pmid"] if p and p.isdigit()})
            if len(pmids) < min_relevant:
                continue

            template = QUERY_TEMPLATES.get(evidence_type, DEFAULT_TEMPLATE)
            queries.append(RetrievalQuery(
                query_id=f"q{len(queries):04d}",
                text=template.format(profile=profile, disease=disease),
                relevant_pmids=pmids,
                profile=profile,
                disease=disease,
                evidence_type=evidence_type,
            ))

        # Deterministic order, then a deterministic sample: the largest groups
        # are the best-studied genes, so taking the top-N would bias the set
        # toward exactly the queries where incomplete judgements hurt most.
        queries.sort(key=lambda q: q.query_id)
        if max_queries and len(queries) > max_queries:
            rng = np.random.default_rng(seed)
            picks = sorted(rng.choice(len(queries), max_queries, replace=False))
            queries = [queries[i] for i in picks]

        for index, query in enumerate(queries):
            query.query_id = f"q{index:04d}"

        self.logger.info(
            f"Built {len(queries)} queries over "
            f"{len({p for q in queries for p in q.relevant_pmids})} distinct papers"
        )
        return queries


def save_queries(queries: Sequence[RetrievalQuery], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([q.to_dict() for q in queries], indent=2))


def load_queries(path: Path) -> List[RetrievalQuery]:
    return [RetrievalQuery.from_dict(p) for p in json.loads(Path(path).read_text())]


# ----------------------------------------------------------------------
# Scoring


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def evaluate_retrieval(
    retrieve: Callable[[str], Sequence[str]],
    queries: Sequence[RetrievalQuery],
    k: int = 10,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> RetrievalMetrics:
    """
    Score a retrieval function against the judged queries.

    Args:
        retrieve: Maps a query string to ranked PMIDs.
        queries: Judged queries.
        k: Cutoff for precision, recall and nDCG.

    Returns:
        Metrics averaged over queries, with bootstrap intervals over the query
        sample -- the queries are the unit of observation, and there are only a
        few dozen of them, so a point estimate would overstate what is known.
    """
    precisions, recalls, reciprocals, ndcgs, hits = [], [], [], [], []

    for query in queries:
        relevant: Set[str] = set(query.relevant_pmids)
        if not relevant:
            continue
        ranked = [str(p) for p in retrieve(query.text)][:k]
        found = [1.0 if p in relevant else 0.0 for p in ranked]

        precisions.append(sum(found) / k if k else 0.0)
        recalls.append(sum(found) / len(relevant))
        hits.append(1.0 if any(found) else 0.0)

        rank = next((i + 1 for i, f in enumerate(found) if f), None)
        reciprocals.append(1.0 / rank if rank else 0.0)

        ideal = _dcg([1.0] * min(len(relevant), k))
        ndcgs.append(_dcg(found) / ideal if ideal else 0.0)

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    metrics = RetrievalMetrics(
        queries=len(precisions),
        k=k,
        precision_at_k=mean(precisions),
        recall_at_k=mean(recalls),
        mrr=mean(reciprocals),
        ndcg_at_k=mean(ndcgs),
        hit_rate=mean(hits),
    )

    if bootstrap_samples > 1 and precisions:
        rng = np.random.default_rng(seed)
        p_array, r_array = np.asarray(precisions), np.asarray(recalls)
        p_samples, r_samples = [], []
        for _ in range(bootstrap_samples):
            picks = rng.integers(0, len(p_array), len(p_array))
            p_samples.append(float(p_array[picks].mean()))
            r_samples.append(float(r_array[picks].mean()))
        metrics.precision_ci = tuple(np.percentile(p_samples, [2.5, 97.5]))
        metrics.recall_ci = tuple(np.percentile(r_samples, [2.5, 97.5]))

    return metrics
