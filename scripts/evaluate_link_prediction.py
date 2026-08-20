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
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from litkg.evaluation import (
    RelationRecord,
    WeightedL3PathPredictor,
    build_temporal_split,
    evaluate_baselines,
    extract_publication_year,
)
from litkg.evaluation.baselines import BASELINE_PREDICTORS, PathPowerPredictor
from litkg.evaluation.temporal_split import year_distribution
from litkg.phase1.kg_preprocessor import CivicProcessor
from litkg.phase2.node_features import build_node_text
from litkg.utils.config import get_data_dir, load_config
from litkg.utils.logging import setup_logging


def load_dated_edges(
    civic_dir: Path,
) -> Tuple[
    List["RelationRecord"], List[Tuple[str, str]], Dict[str, str], Dict[str, str]
]:
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
    all_entities = list(entities)

    # Variant and gene nodes come from the other two files.
    variant_entities, variant_relations = processor._process_civic_variants(variants_file)
    for entity in variant_entities:
        node_types[entity.id] = entity.type
        all_entities.append(entity)
    for entity in processor._process_civic_genes(genes_file):
        node_types[entity.id] = entity.type
        all_entities.append(entity)

    # Evidence id -> publication year, from the citation string.
    evidence = pd.read_csv(evidence_file, sep="\t")
    years = {
        str(row.evidence_id).strip(): extract_publication_year(row.citation)
        for row in evidence.itertuples()
    }

    dated: List[RelationRecord] = []
    for relation in relations:
        # Relation ids are CIVIC:REL:<KIND>:<evidence_id>:<...>
        parts = relation.id.split(":")
        evidence_id = parts[3] if len(parts) > 3 else ""
        dated.append(RelationRecord(
            subject=relation.subject,
            object=relation.object,
            year=years.get(evidence_id),
            predicate=relation.predicate,
            confidence=relation.confidence,
            negated=bool(relation.attributes.get("negated")),
        ))

    backbone = [(r.subject, r.object) for r in variant_relations]
    return dated, backbone, node_types, build_node_text(all_entities)




def _ppi_backbone_edges(
    civic_dir: Path,
    channels: Tuple[str, ...],
    min_score: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    STRING gene-gene edges, mapped onto CIVIC gene nodes.

    These are the first same-type edges in this graph. Everything else here is
    strictly multipartite, which is why shared-neighbour predictors are
    undefined rather than weak.

    The channel choice is the guard. STRING's combined_score fuses seven
    channels and textmining -- co-occurrence in PubMed abstracts -- is the
    largest of them here: 14380 edges among CIVIC's genes against 1862 from
    physical experiments. Those are the same papers the CIVIC labels come from,
    so an edge built on them predicts the answer from the answer.
    """
    from litkg.phase1.string_ppi import StringPPI

    processor = CivicProcessor(load_config())
    genes = processor._process_civic_genes(civic_dir / "civic_genes.tsv")
    ids_by_symbol = {g.name: g.id for g in genes}

    ppi = StringPPI(get_data_dir() / "external" / "string")
    edges = ppi.edges(
        keep_symbols=set(ids_by_symbol),
        channels=channels,
        min_score=min_score,
    )

    mapped = [
        (ids_by_symbol[e.gene_a], ids_by_symbol[e.gene_b])
        for e in edges
        if e.gene_a in ids_by_symbol and e.gene_b in ids_by_symbol
    ]
    return mapped, {"string_edges": len(edges), "mapped": len(mapped)}


def _gdc_backbone_edges(
    civic_dir: Path,
    cutoff: int,
    dated: List["RelationRecord"],
    backbone: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    GDC gene-cancer type edges, with anything that leaks the answer removed.

    GDC edges carry no year, so the splitter puts them in the backbone: present
    at training time, never scored. That is exactly the position from which an
    edge can leak. If the GDC asserts a gene-disease association that CIVIC only
    curated after the cutoff, adding it hands the model a test label, and the
    resulting AUC would measure the leak rather than the method.

    So the held-out pairs are computed first and any GDC edge matching one is
    dropped before the split is built. Edges CIVIC already knows before the
    cutoff are also dropped -- not because they leak, but because re-adding an
    existing edge would inflate the "added N edges" count with no new signal.
    """
    from litkg.evaluation.gdc_edges import (
        drop_leaked_edges,
        join_to_civic,
        load_gdc_edges,
    )
    from litkg.phase1.gdc_client import GDCClient

    config = load_config()
    processor = CivicProcessor(config)
    entities, _ = processor._process_civic_evidence(
        civic_dir / "civic_evidence.tsv", civic_dir / "civic_variants.tsv"
    )
    civic_genes = processor._process_civic_genes(civic_dir / "civic_genes.tsv")

    gene_ids = {g.name: g.id for g in civic_genes}
    disease_ids = {e.name: e.id for e in entities if e.type == "DISEASE"}

    gdc_dir = get_data_dir() / "external" / "tcga" / "gdc"
    raw = load_gdc_edges(gdc_dir, release=GDCClient.pinned_release(), program="tcga")
    joined, join_stats = join_to_civic(raw, gene_ids, disease_ids)

    # The same split this run will use, so "held out" means held out here.
    reference = build_temporal_split(dated, cutoff, backbone)
    test_pairs = list(reference.test_edges)

    kept, leaked = drop_leaked_edges(joined, test_pairs)

    existing = {frozenset(e) for e in backbone}
    existing |= {frozenset((r.subject, r.object)) for r in dated if r.year and r.year <= cutoff}
    novel = [e for e in kept if frozenset(e) not in existing]

    stats = {
        **join_stats,
        "leaked_dropped": len(leaked),
        "already_present": len(kept) - len(novel),
        "added": len(novel),
    }
    if leaked:
        print(f"  GDC leakage guard dropped {len(leaked)} edges that duplicate "
              f"held-out CIVIC test pairs, e.g. {leaked[:2]}")
    return novel, stats


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
    parser.add_argument("--with-ppi", action="store_true",
                        help="Add STRING gene-gene edges to the training "
                             "backbone (breaks strict multipartiteness)")
    parser.add_argument("--ppi-channels", default="experimental",
                        help="STRING evidence channels, comma separated. "
                             "textmining and database are refused unless "
                             "explicitly allowed: both read the literature the "
                             "labels come from")
    parser.add_argument("--ppi-min-score", type=int, default=400)
    parser.add_argument("--with-gdc", action="store_true",
                        help="Add GDC gene-cancer type associations to the "
                             "training backbone (leakage-filtered)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the report as JSON")
    args = parser.parse_args()

    setup_logging()

    civic_dir = get_data_dir() / "external" / "civic"
    if not (civic_dir / "civic_evidence.tsv").exists():
        print(f"CIVIC data not found in {civic_dir}.", file=sys.stderr)
        print("Run: python scripts/phase1_integration.py", file=sys.stderr)
        return 1

    dated, backbone, node_types, _node_text = load_dated_edges(civic_dir)

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

    ppi_backbone: List[Tuple[str, str]] = []
    if args.with_ppi:
        ppi_backbone, ppi_stats = _ppi_backbone_edges(
            civic_dir, tuple(args.ppi_channels.split(",")), args.ppi_min_score
        )
        backbone = list(backbone) + ppi_backbone

    gdc_backbone: List[Tuple[str, str]] = []
    if args.with_gdc:
        gdc_backbone, gdc_stats = _gdc_backbone_edges(civic_dir, args.cutoff, dated, backbone)
        backbone = list(backbone) + gdc_backbone

    split = build_temporal_split(dated, args.cutoff, backbone)

    predictors = [cls() for cls in BASELINE_PREDICTORS]
    predictors.append(WeightedL3PathPredictor(weights=split.edge_weights()))
    # Length 5 as well as 3, because gene-gene edges are unreachable at 3: the
    # middle hop would need a gene adjacent to a disease, and CIVIC has none.
    predictors.append(PathPowerPredictor(3))
    predictors.append(PathPowerPredictor(5))

    report = evaluate_baselines(
        split,
        node_types=None if args.no_type_matching else node_types,
        predictors=predictors,
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
    if args.with_ppi:
        print(f"  PPI backbone   : {len(ppi_backbone)} gene-gene edges added "
              f"from STRING channels {args.ppi_channels} "
              f"(of {ppi_stats['string_edges']} above score {args.ppi_min_score})")
    if args.with_gdc:
        print(f"  GDC backbone   : {len(gdc_backbone)} edges added "
              f"({gdc_stats['leaked_dropped']} dropped as leakage, "
              f"{gdc_stats['already_present']} already in CIVIC)")
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

    if report.per_type_pair:
        print("\nAUC by entity-type pair "
              "(the aggregate above averages problems of unequal difficulty):")
        print(report.format_type_pair_table())

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
