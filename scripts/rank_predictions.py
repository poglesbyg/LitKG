#!/usr/bin/env python3
"""
Rank the full candidate space and check the top predictions against what CIVIC
curated afterwards.

Every number reported for link prediction so far came from scoring held-out
positives against a sampled set of negatives, roughly ten per positive. That is
not the task. A researcher asking "what should we look at next" is ranking every
unobserved pair -- here about a million of them -- and reading the top of the
list. Sampled-negative AUC is a famously optimistic proxy for that.

This ranks the real candidate space with a model trained only on pre-cutoff
evidence, then asks how many of its top predictions CIVIC actually curated in
the years after. That is prospective validation using data already on disk.

Usage:
    python scripts/rank_predictions.py --cutoff 2016 --top 100
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from litkg.evaluation import build_temporal_split
from litkg.evaluation.harness import build_graph
from litkg.phase2.link_prediction import HybridLinkPredictor, TrainingConfig
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging

from evaluate_link_prediction import load_dated_edges  # noqa: E402

Edge = Tuple[str, str]


def candidate_pairs(
    graph, node_types: Dict[str, str], wanted: Set[tuple], known: Set[Edge]
) -> Set[Edge]:
    """
    Unobserved pairs joined by at least one three-path.

    The full product of the relevant type pairs is about a million, but a
    structural score is zero for any pair with no path, so ranking those is
    ranking noise. Enumerating paths directly is also far cheaper than scoring
    every product pair and discarding the zeros.
    """
    reachable: Set[Edge] = set()
    for u in graph:
        type_u = node_types.get(u, "?")
        for a in graph[u]:
            for b in graph[a]:
                if b == u:
                    continue
                for v in graph[b]:
                    if v == u:
                        continue
                    if tuple(sorted((type_u, node_types.get(v, "?")))) not in wanted:
                        continue
                    pair = (u, v) if u <= v else (v, u)
                    if pair not in known:
                        reachable.add(pair)
    return reachable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Train this many models and rank by mean percentile. "
                             "Single runs are not reliable here: precision@10 "
                             "ranged 30-90%% across seeds.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    civic_dir = get_data_dir() / "external" / "civic"
    dated, backbone, node_types, node_text = load_dated_edges(civic_dir)
    split = build_temporal_split(dated, args.cutoff, backbone)
    graph = build_graph(split.train_edges | split.backbone_edges)
    known = set(split.train_edges) | set(split.backbone_edges)

    # Only rank type combinations the held-out period actually contains.
    # Ranking a combination that never occurs would pad the denominator.
    wanted = {
        tuple(sorted((node_types.get(u, "?"), node_types.get(v, "?"))))
        for u, v in split.test_edges
    }

    print("Enumerating candidates")
    candidates = sorted(candidate_pairs(graph, node_types, wanted, known))
    truth = set(split.test_edges)
    reachable_truth = {p for p in truth if p in candidates or p in set(candidates)}
    reachable_truth = truth & set(candidates)

    print(f"  {len(candidates):,} unobserved pairs with a three-path")
    print(f"  {len(reachable_truth)}/{len(truth)} later-curated pairs are in that set "
          f"({len(reachable_truth) / len(truth) * 100:.1f}% ceiling)")

    print(f"Training {args.seeds} model(s) on evidence published before "
          f"{args.cutoff} and ranking by mean percentile")
    start = time.time()
    # Averaging *ranks* rather than scores: the runs are not bit-reproducible
    # and their score scales differ slightly, but their orderings are
    # comparable. A single seed is not enough -- precision@10 ranged from 30%
    # to 90% across five of them.
    summed_rank = [0.0] * len(candidates)
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
            summed_rank[index] += position
        print(f"  seed {seed} done ({time.time() - start:.0f}s elapsed)")

    ranked = sorted(
        zip(candidates, summed_rank), key=lambda kv: kv[1]
    )

    # Precision at several depths, against a base rate. The base rate is the
    # number to beat: picking at random from this candidate set would find a
    # later-curated pair this often.
    base_rate = len(reachable_truth) / len(candidates)
    print(f"\nbase rate (random pick from candidates): {base_rate * 100:.3f}%")
    print("Read depth 50 and beyond. Precision at 10 is not measurable here: "
          "every model concentrates its top predictions on one or two dense\n"
          "clusters, and individual seeds scored 0 to 8 hits out of 10 on the "
          "same data.")
    print(f"{'depth':>8} {'hits':>6} {'precision':>10} {'lift':>7}")
    print("-" * 36)
    summary = []
    for depth in (10, 25, 50, 100, 250, 500, 1000):
        if depth > len(ranked):
            break
        hits = sum(1 for pair, _ in ranked[:depth] if pair in truth)
        precision = hits / depth
        lift = precision / base_rate if base_rate else 0.0
        summary.append({"depth": depth, "hits": hits,
                        "precision": precision, "lift": lift})
        print(f"{depth:>8} {hits:>6} {precision:>9.1%} {lift:>6.1f}x")

    label = {}
    for node_id, name in node_text.items():
        label[node_id] = f"{name} [{node_types.get(node_id, '?')}]"

    print(f"\nTop {min(args.top, 20)} predictions "
          f"(* = curated by CIVIC after {args.cutoff}):")
    for rank, (pair, score) in enumerate(ranked[: min(args.top, 20)], start=1):
        mark = "*" if pair in truth else " "
        u = label.get(pair[0], pair[0])[:44]
        v = label.get(pair[1], pair[1])[:44]
        print(f"  {mark}{rank:>3}. {u:46} {v}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "cutoff": args.cutoff,
            "candidates": len(candidates),
            "later_curated_total": len(truth),
            "later_curated_reachable": len(reachable_truth),
            "base_rate": base_rate,
            "precision_at_depth": summary,
            "top": [
                {"rank": i + 1, "source": p[0], "target": p[1],
                 "source_name": node_text.get(p[0], p[0]),
                 "target_name": node_text.get(p[1], p[1]),
                 "mean_rank": float(s) / args.seeds, "later_curated": p in truth}
                for i, (p, s) in enumerate(ranked[: args.top])
            ],
        }, indent=2))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
