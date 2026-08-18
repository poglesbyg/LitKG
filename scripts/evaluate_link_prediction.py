#!/usr/bin/env python3
"""
Temporal-holdout evaluation of link prediction on the CIVIC knowledge graph.

Answers the question the project could not previously answer: given what was
published before year Y, does the graph predict the associations curated after
it? Every prior figure in this repository is a count; these are measurements.

Usage:
    python scripts/evaluate_link_prediction.py --cutoff 2016
    python scripts/evaluate_link_prediction.py --list-years
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from litkg.evaluation import (
    build_temporal_split,
    evaluate_baselines,
    extract_publication_year,
)
from litkg.evaluation.temporal_split import year_distribution
from litkg.phase1.kg_preprocessor import CivicProcessor
from litkg.utils.config import get_data_dir, load_config
from litkg.utils.logging import setup_logging


def load_dated_edges(
    civic_dir: Path,
) -> Tuple[List[Tuple[str, str, Optional[int]]], List[Tuple[str, str]], Dict[str, str]]:
    """
    Build dated evidence edges and undated backbone edges from CIVIC.

    Evidence edges carry the publication year of the paper supporting them.
    Gene->variant edges carry no date: a variant belonging to a gene is not a
    discovery that happens in a year, so they form the backbone that is always
    present at training time.
    """
    config = load_config()
    processor = CivicProcessor(config)

    evidence_file = civic_dir / "civic_evidence.tsv"
    variants_file = civic_dir / "civic_variants.tsv"
    genes_file = civic_dir / "civic_genes.tsv"

    entities, relations = processor._process_civic_evidence(evidence_file, variants_file)
    node_types: Dict[str, str] = {e.id: e.type for e in entities}

    # Variant and gene nodes come from the other two files.
    variant_entities, variant_relations = processor._process_civic_variants(variants_file)
    for entity in variant_entities:
        node_types[entity.id] = entity.type
    for entity in processor._process_civic_genes(genes_file):
        node_types[entity.id] = entity.type

    # Evidence id -> publication year, from the citation string.
    evidence = pd.read_csv(evidence_file, sep="\t")
    years = {
        str(row.evidence_id).strip(): extract_publication_year(row.citation)
        for row in evidence.itertuples()
    }

    dated: List[Tuple[str, str, Optional[int]]] = []
    for relation in relations:
        # Relation ids are CIVIC:REL:<KIND>:<evidence_id>:<...>
        parts = relation.id.split(":")
        evidence_id = parts[3] if len(parts) > 3 else ""
        dated.append((relation.subject, relation.object, years.get(evidence_id)))

    backbone = [(r.subject, r.object) for r in variant_relations]
    return dated, backbone, node_types


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016,
                        help="Papers published before this year are training data")
    parser.add_argument("--negatives", type=int, default=10,
                        help="Negative pairs sampled per positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-years", action="store_true",
                        help="Show the year distribution and exit")
    parser.add_argument("--no-type-matching", action="store_true",
                        help="Sample negatives without matching endpoint types "
                             "(makes the task easier; for comparison only)")
    parser.add_argument("--degree-matched", action="store_true",
                        help="Draw negatives from the same degree bucket as the "
                             "positives, controlling for node popularity")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the report as JSON")
    args = parser.parse_args()

    setup_logging()

    civic_dir = get_data_dir() / "external" / "civic"
    if not (civic_dir / "civic_evidence.tsv").exists():
        print(f"CIVIC data not found in {civic_dir}.", file=sys.stderr)
        print("Run: python scripts/phase1_integration.py", file=sys.stderr)
        return 1

    dated, backbone, node_types = load_dated_edges(civic_dir)

    if args.list_years:
        distribution = year_distribution(dated)
        total = sum(distribution.values())
        print(f"{total} distinct dated pairs\n")
        print(f"{'cutoff':>8} {'train':>8} {'test':>8}  {'test %':>7}")
        cumulative = 0
        for year in sorted(distribution):
            cumulative += distribution[year]
            after = total - cumulative
            if after:
                print(f"{year + 1:>8} {cumulative:>8} {after:>8}  {after / total * 100:6.1f}%")
        return 0

    split = build_temporal_split(dated, args.cutoff, backbone)
    report = evaluate_baselines(
        split,
        node_types=None if args.no_type_matching else node_types,
        negatives_per_positive=args.negatives,
        seed=args.seed,
        degree_matched=args.degree_matched,
    )

    summary = split.summary()
    print(f"\nTemporal holdout at {args.cutoff}")
    print(f"  training edges : {summary['train_edges']} dated + "
          f"{summary['backbone_edges']} backbone")
    print(f"  test edges     : {summary['test_edges']}")
    print(f"  excluded       : {summary['excluded_already_known']} already known before "
          f"the cutoff, {summary['excluded_cold_start']} cold-start")
    print(f"  negatives      : {report.negatives_per_positive} per positive"
          f"{'' if not args.no_type_matching else ', NOT type-matched'}"
          f"{', degree-matched' if args.degree_matched else ''}\n")
    print(report.format_table())

    if report.diagnostics:
        d = report.diagnostics
        print(f"\nTraining graph: {d['train_graph_nodes']} nodes, "
              f"{d['train_graph_edges']} edges, "
              f"avg clustering {d['average_clustering']:.3f}")
        print(f"Structural coverage: {d['structural_coverage']:.1%} of test pairs "
              f"share a neighbour")

    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  - {note}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
