#!/usr/bin/env python
"""
Does TCGA variant co-occurrence predict what CIVIC curators later wrote?

Scores held-out CIVIC pairs by how often TCGA saw the same protein change in
the matching cohort, and compares against the L3 path baseline on the subset
each can actually speak to.

Read the per-type-pair table, not the aggregate. This predictor can only score
DISEASE-MUTATION pairs, which are 39% of the test set; on everything else it
returns zero, so an aggregate AUC would mostly measure how many pairs it
abstains on.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from litkg.evaluation import build_temporal_split  # noqa: E402
from litkg.evaluation.baselines import WeightedL3PathPredictor  # noqa: E402
from litkg.evaluation.gdc_variant_predictor import (  # noqa: E402
    GDCVariantCooccurrencePredictor,
    load_variant_counts,
)
from litkg.evaluation.harness import evaluate_baselines  # noqa: E402
from litkg.phase1.disease_ontology import DiseaseOntology  # noqa: E402
from litkg.phase1.gdc_client import GDCClient  # noqa: E402
from litkg.phase1.kg_preprocessor import CivicProcessor  # noqa: E402
from litkg.utils.config import get_data_dir, load_config  # noqa: E402
from litkg.utils.logging import setup_logging  # noqa: E402

PROTEIN_CHANGE = re.compile(r"^[A-Z]\d+[A-Z*]$")


def build_variant_keys(civic_dir: Path) -> Dict[str, str]:
    """CIVIC variant node id -> "SYMBOL p.CHANGE", for variants that have one."""
    frame = pd.read_csv(civic_dir / "civic_variants.tsv", sep="\t")
    keys: Dict[str, str] = {}
    for row in frame.itertuples():
        name = str(row.variant).strip()
        gene = str(row.gene).strip()
        if not PROTEIN_CHANGE.match(name) or gene in ("", "nan"):
            continue
        keys[f"CIVIC:VARIANT:{row.variant_id}"] = f"{gene} p.{name}"
    return keys


def build_cohort_map(
    civic_disease_ids: Dict[str, str], obo_path: Path
) -> Tuple[Dict[str, str], Counter]:
    """TCGA project id -> CIVIC disease node id, via DOID."""
    ontology = DiseaseOntology(obo_path)
    gdc_dir = get_data_dir() / "external" / "tcga" / "gdc"
    projects = json.loads(
        (gdc_dir / f"release-{GDCClient.pinned_release()}" / "tcga" / "projects.json").read_text()
    )

    mapping: Dict[str, str] = {}
    how: Counter = Counter()
    for project in projects:
        match = ontology.match_to_civic(project.get("name", ""), civic_disease_ids)
        if match is None:
            how["unmatched"] += 1
            continue
        mapping[project["project_id"]] = match.civic_id
        how["via ancestor" if match.via_ancestor else "direct DOID"] += 1
    return mapping, how


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    logging.disable(logging.INFO)

    civic_dir = get_data_dir() / "external" / "civic"
    processor = CivicProcessor(load_config())

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        import evaluate_link_prediction as E

        dated, backbone, node_types, _ = E.load_dated_edges(civic_dir)
        entities, _ = processor._process_civic_evidence(
            civic_dir / "civic_evidence.tsv", civic_dir / "civic_variants.tsv"
        )

    civic_by_doid = {
        e.id.replace("CIVIC:DISEASE:", ""): e.id
        for e in entities
        if e.type == "DISEASE"
    }

    obo = get_data_dir() / "external" / "disease_ontology" / "doid.obo"
    DiseaseOntology.download(obo)
    cohort_map, how = build_cohort_map(civic_by_doid, obo)

    gdc_dir = get_data_dir() / "external" / "tcga" / "gdc"
    release = GDCClient.pinned_release()
    variant_counts = load_variant_counts(gdc_dir, release)
    projects = json.loads((gdc_dir / f"release-{release}" / "tcga" / "projects.json").read_text())
    cohort_cases = {
        p["project_id"]: (p.get("summary") or {}).get("case_count") or 0 for p in projects
    }
    variant_keys = build_variant_keys(civic_dir)

    split = build_temporal_split(dated, args.cutoff, backbone)

    print(f"Disease Ontology join: {dict(how)}")
    print(f"CIVIC variants with a protein change: {len(variant_keys)}")
    print(f"TCGA variants observed: {len(variant_counts)}")

    predictors: List = []
    for mode in ("specificity", "prevalence"):
        predictors.append(
            GDCVariantCooccurrencePredictor(
                variant_counts=variant_counts,
                cohort_cases=cohort_cases,
                variant_node_keys=variant_keys,
                cohort_to_disease=cohort_map,
                mode=mode,
            )
        )
    predictors.append(WeightedL3PathPredictor(weights=split.edge_weights()))

    # How much of the test set can this speak to at all?
    test_pairs = list(split.test_edges)
    dm = [
        (a, b)
        for a, b in test_pairs
        if {node_types.get(a), node_types.get(b)} == {"DISEASE", "MUTATION"}
    ]
    import networkx as nx

    scorer = predictors[0].fit(nx.Graph(split.train_edges | split.backbone_edges))
    print(
        f"\nTest pairs: {len(test_pairs)} total, {len(dm)} DISEASE-MUTATION "
        f"({len(dm) / len(test_pairs):.0%})"
    )
    print(
        f"Scored by GDC: {scorer.coverage(test_pairs):.1%} of all pairs, "
        f"{scorer.coverage(dm):.1%} of DISEASE-MUTATION pairs"
    )

    report = evaluate_baselines(
        split,
        node_types=node_types,
        predictors=predictors,
        negatives_per_positive=args.negatives,
        seed=args.seed,
        degree_matched=True,
    )
    print(f"\nTemporal holdout at {args.cutoff}, degree-matched negatives\n")
    print(report.format_table())
    if report.per_type_pair:
        print("\nAUC by entity-type pair (the number that matters here):")
        print(report.format_type_pair_table())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
