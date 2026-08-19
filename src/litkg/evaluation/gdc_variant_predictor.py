"""
Scoring CIVIC variant-disease pairs by TCGA co-occurrence.

The gene-level GDC join did not work, and the reason was granularity. TP53
links to 14 of 15 diseases in the joined graph: "this well-known driver is
mutated in this common cancer" is close to a prior, so the edges added paths
without adding discrimination and cost 0.004 AUC.

CIVIC's unit is the variant, and at that level the same data is sharp. BRAF is
mutated across most cohorts; BRAF V600E concentrates in thyroid (283 cases) and
melanoma (200), with a long thin tail. 39% of held-out pairs are
DISEASE-MUTATION, which is exactly the pair this scores.

What this is, and is not
------------------------
This is not a graph-structure predictor. It reads an external table and reports
what TCGA observed, so it competes with L3 paths only in the sense that both
produce a number per pair. It can score a pair only when the variant carries a
protein change TCGA saw and the disease resolves to a TCGA cohort; everything
else scores zero, which is why it must be read per entity-type pair rather than
as an aggregate AUC.

It is also the closest thing here to reading the answer. CIVIC curators write
up associations that TCGA frequencies helped make visible, so a high score on a
held-out pair may mean the association was discoverable from sequencing before
a curator wrote it down -- a real and useful claim -- or may mean the two
sources are not independent. `scripts/evaluate_gdc_variants.py` reports the
overlap that distinguishes those.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import networkx as nx

from litkg.evaluation.baselines import LinkPredictor


class GDCVariantCooccurrencePredictor(LinkPredictor):
    """
    Scores a (variant, disease) pair by how often TCGA saw that protein change
    in that cohort.

    Two normalisations, because they answer different questions and the choice
    is not obvious:

    - ``prevalence``: occurrences / cases in the cohort. "How much of this
      cancer carries this variant." Favours variants common in their cohort.
    - ``specificity``: occurrences in this cohort / occurrences anywhere.
      "How much this variant points at this cancer." Favours variants confined
      to one cohort, which is what makes BRAF V600E informative about thyroid
      while TP53 R273H is informative about nothing.
    """

    name = "gdc_variant"

    def __init__(
        self,
        variant_counts: Dict[str, Dict[str, int]],
        cohort_cases: Dict[str, int],
        variant_node_keys: Dict[str, str],
        cohort_to_disease: Dict[str, str],
        mode: str = "specificity",
    ):
        if mode not in ("prevalence", "specificity"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.name = f"gdc_variant_{mode}"
        self.cohort_cases = cohort_cases
        self.variant_node_keys = variant_node_keys
        self.cohort_to_disease = cohort_to_disease

        # (variant node, disease node) -> score, built once.
        self.scores: Dict[Tuple[str, str], float] = {}
        key_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        for node, key in variant_node_keys.items():
            key_to_nodes[key].add(node)

        for key, by_cohort in variant_counts.items():
            nodes = key_to_nodes.get(key)
            if not nodes:
                continue
            total = sum(by_cohort.values())
            for cohort, count in by_cohort.items():
                disease = cohort_to_disease.get(cohort)
                if disease is None:
                    continue
                if mode == "prevalence":
                    cases = cohort_cases.get(cohort, 0)
                    if cases <= 0:
                        continue
                    value = count / cases
                else:
                    if total <= 0:
                        continue
                    value = count / total
                for node in nodes:
                    pair = (node, disease)
                    # A CIVIC disease can absorb several cohorts through
                    # ontology ancestry, so take the strongest rather than the
                    # last one seen.
                    if value > self.scores.get(pair, 0.0):
                        self.scores[pair] = value

    def fit(self, graph: nx.Graph) -> "GDCVariantCooccurrencePredictor":
        self.graph = graph
        return self

    def score(self, u: str, v: str) -> float:
        return max(
            self.scores.get((u, v), 0.0),
            self.scores.get((v, u), 0.0),
        )

    def coverage(self, pairs) -> float:
        """Fraction of pairs this predictor can say anything about."""
        pairs = list(pairs)
        if not pairs:
            return 0.0
        return sum(1 for u, v in pairs if self.score(u, v) > 0) / len(pairs)


def load_variant_counts(cache_dir: Path, release: str, program: str = "tcga") -> Dict:
    path = Path(cache_dir) / f"release-{release}" / program.lower() / "variants_by_project.json"
    if not path.exists():
        raise FileNotFoundError(
            f"GDC variant cache missing {path}. Run the variant download first."
        )
    return json.loads(path.read_text())
