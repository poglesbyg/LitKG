"""
Joining GDC gene-cancer type associations onto the CIVIC evaluation graph.

These edges are undated. `TemporalSplitter` routes undated edges into the
backbone, which is present at training time for every cutoff, so anything added
here is training signal that is never itself scored.

That makes leakage the risk worth designing around. A GDC edge says "this gene
is recurrently mutated in this cancer". A held-out CIVIC test positive can say
something a predictor could read as the same thing. If the two coincide, the
answer is placed in the training graph and the resulting AUC measures nothing.
`drop_leaked_edges` is not an optional tidy-up; nothing here should be used
without it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from litkg.phase1.gdc_client import background_rate_enrichment
from litkg.utils.logging import get_logger

logger = get_logger(__name__)

Pair = Tuple[str, str]


def normalise_name(name: str) -> str:
    """Fold a disease or gene name to a comparable key."""
    text = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    return " ".join(text.split())


def load_gdc_edges(
    cache_dir: Path,
    release: str,
    program: str = "tcga",
    min_enrichment: float = 3.0,
    min_occurrences: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Read cached GDC data and return (gene symbol, project name, enrichment).

    Reads the same cache the processor writes, so the evaluation and the graph
    build cannot disagree about what the GDC said.
    """
    base = Path(cache_dir) / f"release-{release}"
    projects_path = base / program.lower() / "projects.json"
    mutations_path = base / program.lower() / "mutations_by_project.json"
    genes_path = base / "census_genes.json"

    for path in (projects_path, mutations_path, genes_path):
        if not path.exists():
            raise FileNotFoundError(
                f"GDC cache missing {path}. Run the TCGA/CPTAC download first."
            )

    projects = {p["project_id"]: p for p in json.loads(projects_path.read_text())}
    genes = {g["symbol"]: g for g in json.loads(genes_path.read_text())}
    mutations = json.loads(mutations_path.read_text())

    associations = background_rate_enrichment(
        mutations=mutations,
        cds_lengths={
            s: g.get("canonical_transcript_length_cds") for s, g in genes.items()
        },
        cohort_cases={
            pid: (p.get("summary") or {}).get("case_count") or 0
            for pid, p in projects.items()
        },
    )

    edges: List[Tuple[str, str, float]] = []
    for a in associations:
        if a.enrichment < min_enrichment or a.occurrences < min_occurrences:
            continue
        project = projects.get(a.project_id) or {}
        name = project.get("name") or a.project_id
        edges.append((a.symbol, name, a.enrichment))
    return edges


def join_to_civic(
    gdc_edges: Sequence[Tuple[str, str, float]],
    gene_ids_by_name: Dict[str, str],
    disease_ids_by_name: Dict[str, str],
) -> Tuple[List[Pair], Dict[str, int]]:
    """
    Map GDC symbols and cohort names onto CIVIC node ids by normalised name.

    Matching is exact after normalisation, deliberately. A fuzzy disease match
    would silently merge "Lung Adenocarcinoma" with "Lung Squamous Cell
    Carcinoma", which are different diseases with different drivers, and the
    resulting edge would be wrong in a way no downstream metric would reveal.
    Unmatched cohorts are reported, not guessed at.
    """
    genes = {normalise_name(k): v for k, v in gene_ids_by_name.items()}
    diseases = {normalise_name(k): v for k, v in disease_ids_by_name.items()}

    joined: Set[Pair] = set()
    stats = {
        "input": len(gdc_edges),
        "gene_unmatched": 0,
        "disease_unmatched": 0,
        "joined": 0,
    }
    unmatched_diseases: Set[str] = set()

    for symbol, disease_name, _enrichment in gdc_edges:
        gene_id = genes.get(normalise_name(symbol))
        disease_id = diseases.get(normalise_name(disease_name))
        if gene_id is None:
            stats["gene_unmatched"] += 1
            continue
        if disease_id is None:
            stats["disease_unmatched"] += 1
            unmatched_diseases.add(disease_name)
            continue
        joined.add((gene_id, disease_id))

    stats["joined"] = len(joined)
    if unmatched_diseases:
        logger.info(
            f"{len(unmatched_diseases)} GDC cohorts had no exact CIVIC disease "
            f"match: {sorted(unmatched_diseases)[:5]}"
        )
    return sorted(joined), stats


def drop_leaked_edges(
    edges: Iterable[Pair],
    test_pairs: Iterable[Pair],
) -> Tuple[List[Pair], List[Pair]]:
    """
    Remove any GDC edge that coincides with a held-out test pair.

    Comparison is undirected, because the split treats a pair as unordered and
    an edge reversed is the same leak.

    Returns (kept, dropped). The dropped list is returned rather than counted so
    a caller can report exactly which associations were withheld.
    """
    held_out: Set[frozenset] = {frozenset(p) for p in test_pairs}
    kept: List[Pair] = []
    dropped: List[Pair] = []
    for edge in edges:
        (dropped if frozenset(edge) in held_out else kept).append(edge)
    return kept, dropped
