#!/usr/bin/env python3
"""
Re-run the prospective check at several cutoffs and compare.

The prospective result was measured once, at 2016, and reported as evidence
that the system surfaces associations before they are curated. It does not
survive replication: lift at depth 100 is 35x at 2016, 5x at 2018 and 0x at
2020. This script is what established that, kept so the claim can be re-checked
rather than re-assumed.

Usage:
    python scripts/replicate_prospective.py --cutoffs 2016 2018 2020
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from litkg.evaluation import build_temporal_split
from litkg.evaluation.harness import build_graph
from litkg.phase2.link_prediction import HybridLinkPredictor, TrainingConfig
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging

from evaluate_link_prediction import load_dated_edges  # noqa: E402
from rank_predictions import candidate_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[2016, 2018, 2020])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-positives", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    dated, backbone, node_types, node_text = load_dated_edges(
        get_data_dir() / "external" / "civic"
    )

    results = []
    for cutoff in args.cutoffs:
        split = build_temporal_split(dated, cutoff, backbone)
        truth = set(split.test_edges)
        if len(truth) < args.min_positives:
            print(f"cutoff {cutoff}: only {len(truth)} held-out pairs, skipping")
            continue

        graph = build_graph(split.train_edges | split.backbone_edges)
        known = set(split.train_edges) | set(split.backbone_edges)
        wanted = {
            tuple(sorted((node_types.get(u, "?"), node_types.get(v, "?"))))
            for u, v in truth
        }
        candidates = sorted(candidate_pairs(graph, node_types, wanted, known))
        base = len(truth & set(candidates)) / len(candidates)

        summed = [0.0] * len(candidates)
        for seed in range(args.seeds):
            config = TrainingConfig(
                epochs=args.epochs, seed=seed, loss="bpr", num_layers=2,
                hidden_dim=256, embedding_dim=256, dropout=0.3,
            )
            model = HybridLinkPredictor(
                config=config, node_types=node_types, edge_years=split.edge_years,
                edge_predicates={p: e.dominant_predicate
                                 for p, e in split.edge_evidence.items()},
                edge_weights=split.edge_weights(), node_text=node_text,
            ).fit(graph)
            scores = model.score_pairs(candidates)
            for position, index in enumerate(
                sorted(range(len(candidates)), key=lambda i: -scores[i])
            ):
                summed[index] += position
        ranked = [c for c, _ in sorted(zip(candidates, summed), key=lambda kv: kv[1])]

        row = {"cutoff": cutoff, "held_out": len(truth),
               "candidates": len(candidates), "base_rate": base}
        for depth in (100, 500):
            hits = sum(1 for p in ranked[:depth] if p in truth)
            row[f"precision_at_{depth}"] = hits / depth
            row[f"lift_at_{depth}"] = (hits / depth) / base if base else 0.0
        pairs = collections.defaultdict(lambda: [0, 0])
        for pair in ranked[:500]:
            key = "-".join(sorted((node_types.get(pair[0], "?"),
                                   node_types.get(pair[1], "?"))))
            pairs[key][1] += 1
            pairs[key][0] += int(pair in truth)
        row["by_type_pair"] = {k: {"curated": v[0], "total": v[1]}
                               for k, v in pairs.items()}
        results.append(row)
        print(f"cutoff {cutoff}: base {base:.3%}  "
              f"P@100 {row['precision_at_100']:.1%} ({row['lift_at_100']:.0f}x)  "
              f"P@500 {row['precision_at_500']:.1%} ({row['lift_at_500']:.0f}x)")

    print(f"\n{'cutoff':>7} {'held-out':>9} {'base':>8} {'P@100':>7} "
          f"{'lift@100':>9} {'lift@500':>9}")
    print("-" * 54)
    for row in results:
        print(f"{row['cutoff']:>7} {row['held_out']:>9} {row['base_rate']:>7.3%} "
              f"{row['precision_at_100']:>7.1%} {row['lift_at_100']:>8.1f}x "
              f"{row['lift_at_500']:>8.1f}x")

    if len(results) > 1:
        lifts = [r["lift_at_100"] for r in results]
        if max(lifts) > 5 * max(min(lifts), 0.1):
            print("\nThese do not agree. Quote the cutoff with any number taken "
                  "from this pipeline, and do not\ngeneralise a single cutoff's "
                  "precision to the system as a whole.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
