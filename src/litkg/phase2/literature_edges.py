"""
Literature co-mention edges, and a length-2 path score over them.

The only source measured here that adds edge types CIVIC lacks is the literature
itself. Pre-cutoff PubMed abstracts are already cached per cutoff by
`litkg.phase2.literature_context`, with the date filter applied at the query and
re-checked per record, so edges built from them carry no post-cutoff knowledge.

Why length 2, and why co-mentions
---------------------------------
A reachability check on the 2016 split, before any of this was built, found
that sentence-level co-mentions add a length-3 path to 88.1% of held-out pairs
and to 85.9% of degree-matched negatives -- so a length-3 count cannot
discriminate. At length 2 the figures are 51.1% against 32.3%, and that gap
holds within every quartile of literature coverage (+29.9 points in the least
covered quarter down to +5.4 in the most), so it is not only that well-studied
entities are mentioned more often.

Co-mention is used because it is cheap and needs no extraction model. It is a
noisy superset of extracted relations, so the result below does not settle
whether *extracted* relations would help: they would be fewer, but more precise.

What it measured: inconclusive
------------------------------
`scripts/evaluate_literature_paths.py`, degree-matched negatives, 8 negative
samples per cutoff:

  cutoff   edges    alone   path counting   + literature 0.25     + literature 0.5
  2016      9,953   0.609   0.6937          0.6914 (0/8 better)   0.6812 (0/8)
  2020     13,710   0.673   0.7697          0.7705 (8/8 better)   0.7555 (0/8)

Within pairs grouped by whether they already have a length-3 path, the
literature score's AUC is 0.532 / 0.532 at 2016 and 0.520 / 0.586 at 2020. So
there is no independent signal at 2016 and a small one at 2020, among pairs the
graph already connects.

The 2020 blend "win" is +0.0008 AUC, and these 8 runs vary only the negative
sample around identical held-out pairs with deterministic predictors, which is
much weaker evidence than 8 training seeds. That blend also cuts average
precision from 0.304 to 0.268 and halves Hits@100 (0.145 to 0.064), so at
neither cutoff does it improve the shortlist. Nothing in the pipeline uses this
score by default.

Linking choices
---------------
- Gene symbols match **case-sensitively** on whole tokens. Case-insensitive
  matching turns MET, KIT and RET into the words "met", "kit" and "ret", which
  adds noise to positives and negatives alike.
- Variants are **not** linked. "V600E" alone collides across genes, and a
  variant reaches literature edges through its gene in the CIVIC graph anyway.
- Everything else (diseases, drugs, phenotypes) goes through `EntityAliasIndex`
  with a four-character minimum alias.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

from litkg.evaluation.baselines import LinkPredictor

Edge = Tuple[str, str]
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
TOKEN = re.compile(r"[A-Za-z0-9-]+")
LINKED_TYPES = ("DISEASE", "DRUG", "PHENOTYPE")


class CoMentionLinker:
    """Finds which graph nodes a sentence mentions."""

    def __init__(
        self,
        gene_symbols: Mapping[str, str],
        named_entities: Iterable[Tuple[str, str, Sequence[str]]],
        min_alias_length: int = 4,
    ):
        """
        gene_symbols: symbol -> node id, matched case-sensitively.
        named_entities: (node id, name, synonyms) for non-gene, non-variant nodes.
        """
        from litkg.langchain_integration.graph_linking import EntityAliasIndex

        self.gene_symbols = dict(gene_symbols)
        self.index = EntityAliasIndex(min_alias_length=min_alias_length)
        for node_id, name, synonyms in named_entities:
            self.index.add_entity(node_id, name, synonyms or [])
        aliases = sorted(self.index.alias_to_nodes, key=len, reverse=True)
        self._pattern = (
            re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b")
            if aliases else None
        )

    def mentions(self, sentence: str) -> Set[str]:
        found: Set[str] = set()
        for token in TOKEN.findall(sentence):
            node = self.gene_symbols.get(token)
            if node is not None:
                found.add(node)
        if self._pattern is not None:
            for match in self._pattern.finditer(self.index._normalize(sentence)):
                found |= self.index.alias_to_nodes[match.group(1)]
        return found


def build_comention_edges(
    abstracts: Iterable[str],
    linker: CoMentionLinker,
    allowed_nodes: Optional[Set[str]] = None,
) -> Dict[FrozenSet[str], int]:
    """
    Count sentence-level co-mentions between linked nodes.

    Abstracts are deduplicated by text: the cache groups them by the entity each
    was fetched for, so one paper usually appears under several queries, and
    counting it once per query would weight edges by how many of their
    endpoints happened to be searched.
    """
    counts: Counter = Counter()
    for text in sorted(set(abstracts)):
        for sentence in SENTENCE_BOUNDARY.split(text):
            nodes = sorted(
                n for n in linker.mentions(sentence)
                if allowed_nodes is None or n in allowed_nodes
            )
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    counts[frozenset((nodes[i], nodes[j]))] += 1
    return dict(counts)


class LiteraturePathPredictor(LinkPredictor):
    """
    Degree-normalised count of length-2 paths that use a literature edge.

    For a pair (u, v), sums 1 / sqrt(degree(a)) over every intermediate `a`
    adjacent to both in the combined graph (training edges plus literature
    edges), counting only paths where at least one hop exists solely in the
    literature. Paths made entirely of training edges are left to the
    structural predictors, which already score them.

    A direct co-mention of the pair itself is not scored. It occurs for 1.5% of
    held-out pairs and 0.8% of negatives on the 2016 split, too rare to carry a
    ranking, and leaving it out keeps this score about paths.
    """

    name = "literature_l2"

    def __init__(self, literature_edges: Mapping[FrozenSet[str], int]):
        self.literature_edges = {frozenset(e) for e in literature_edges}

    def fit(self, graph: nx.Graph) -> "LiteraturePathPredictor":
        self.graph = graph
        combined = graph.copy()
        nodes = set(graph.nodes())
        # Literature edges between nodes the training graph does not contain
        # cannot be scored against anything and are dropped.
        for edge in self.literature_edges:
            u, v = tuple(edge)
            if u in nodes and v in nodes:
                combined.add_edge(u, v)
        self.combined = combined
        self._literature_only = {
            e for e in self.literature_edges
            if len(e) == 2 and all(n in nodes for n in e) and not graph.has_edge(*tuple(e))
        }
        return self

    def score(self, u: str, v: str) -> float:
        if u not in self.combined or v not in self.combined:
            return 0.0
        shared = set(self.combined[u]) & set(self.combined[v])
        total = 0.0
        for a in sorted(shared):
            if a in (u, v):
                continue
            if (
                frozenset((u, a)) in self._literature_only
                or frozenset((a, v)) in self._literature_only
            ):
                total += 1.0 / math.sqrt(self.combined.degree(a))
        return total
