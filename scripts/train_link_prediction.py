#!/usr/bin/env python3
"""
Train a GNN link predictor and score it against the structural baselines.

The comparison is only meaningful if the model is measured through the same
split, the same negatives and the same metrics as the baselines, so it is run
through the evaluation harness rather than scored separately.

Usage:
    python scripts/train_link_prediction.py --cutoff 2016
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litkg.evaluation import build_temporal_split, evaluate_baselines
from litkg.evaluation import WeightedL3PathPredictor
from litkg.evaluation.baselines import BASELINE_PREDICTORS
from litkg.phase2.link_prediction import (
    GNNLinkPredictor,
    HybridLinkPredictor,
    TrainingConfig,
)
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_link_prediction import load_dated_edges  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loss", default="bpr", choices=["bpr", "bce"])
    parser.add_argument("--text-model", default=None,
                        help="Override the node-text encoder")
    parser.add_argument("--no-text-features", action="store_true",
                        help="Train on topology alone, ignoring node names")
    parser.add_argument("--relational", action="store_true",
                        help="Use R-GCN: one transform per relation type")
    parser.add_argument("--no-degree-matching", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Skip the GNN+L3 ensemble")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Train with this many seeds and report the spread")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()

    civic_dir = get_data_dir() / "external" / "civic"
    if not (civic_dir / "civic_evidence.tsv").exists():
        print(f"CIVIC data not found in {civic_dir}", file=sys.stderr)
        return 1

    dated, backbone, node_types, node_text = load_dated_edges(civic_dir)
    split = build_temporal_split(dated, args.cutoff, backbone)

    text_encoder = None
    if not args.no_text_features:
        from litkg.phase2.node_features import (
            FeatureConfig,
            FeatureOnlyPredictor,
            NodeTextEncoder,
        )
        text_encoder = NodeTextEncoder(
            FeatureConfig(model_name=args.text_model) if args.text_model
            else FeatureConfig()
        )

    predictors = [cls() for cls in BASELINE_PREDICTORS]
    predictors.append(WeightedL3PathPredictor(weights=split.edge_weights()))
    if text_encoder is not None:
        predictors.append(
            FeatureOnlyPredictor(node_text=node_text, encoder=text_encoder)
        )
    if not args.baselines_only:
        config = TrainingConfig(
            hidden_dim=args.hidden_dim,
            embedding_dim=args.hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
            learning_rate=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            loss=args.loss,
            relational=args.relational,
        )
        predictors.append(GNNLinkPredictor(
            config=config, node_types=node_types, edge_years=split.edge_years,
            edge_predicates={
                pair: ev.dominant_predicate
                for pair, ev in split.edge_evidence.items()
            },
            node_text=None if args.no_text_features else node_text,
            text_encoder=text_encoder,
        ))
        if not args.no_hybrid:
            predictors.append(HybridLinkPredictor(
                config=config, node_types=node_types, edge_years=split.edge_years,
                edge_predicates={
                    pair: ev.dominant_predicate
                    for pair, ev in split.edge_evidence.items()
                },
                edge_weights=split.edge_weights(),
                node_text=None if args.no_text_features else node_text,
            ))
            predictors[-1].text_encoder = text_encoder

    report = evaluate_baselines(
        split,
        node_types=node_types,
        predictors=predictors,
        negatives_per_positive=args.negatives,
        seed=args.seed,
        degree_matched=not args.no_degree_matching,
    )

    summary = split.summary()
    print(f"\nTemporal holdout at {args.cutoff}")
    print(f"  training edges : {summary['train_edges']} dated + "
          f"{summary['backbone_edges']} backbone")
    print(f"  test edges     : {summary['test_edges']}")
    print(f"  negatives      : {args.negatives} per positive"
          f"{'' if args.no_degree_matching else ', degree-matched'}\n")
    print(report.format_table())

    gnn = next((p for p in predictors if isinstance(p, GNNLinkPredictor)), None)
    if gnn is not None and gnn.history:
        print(f"\nGNN best validation AUC: {gnn.best_validation_auc:.4f} "
              f"({len(gnn.history)} evaluations)")
        baseline = report.results.get("l3_paths")
        for name in ("gnn", "hybrid"):
            result = report.results.get(name)
            if baseline and result:
                delta = result.auc - baseline.auc
                verdict = "beats" if delta > 0 else "does NOT beat"
                print(f"{name} {verdict} l3_paths: {result.auc:.3f} vs "
                      f"{baseline.auc:.3f} ({delta:+.3f})")

        # A single seed is not evidence: the GNN alone varies by +/-0.089 AUC
        # across seeds and has collapsed to 0.512 on one. Report the spread.
        if args.seeds > 1:
            import statistics
            scores = {"gnn": [], "hybrid": []}
            for seed in range(args.seeds):
                seeded = TrainingConfig(**{**config.__dict__, "seed": seed})
                repeat = [GNNLinkPredictor(config=seeded, node_types=node_types,
                                           edge_years=split.edge_years)]
                if not args.no_hybrid:
                    repeat.append(HybridLinkPredictor(
                        config=seeded, node_types=node_types,
                        edge_years=split.edge_years))
                seeded_report = evaluate_baselines(
                    split, node_types=node_types, predictors=repeat,
                    negatives_per_positive=args.negatives, seed=args.seed,
                    degree_matched=not args.no_degree_matching,
                )
                for key in scores:
                    if key in seeded_report.results:
                        scores[key].append(seeded_report.results[key].auc)
            print(f"\nAcross {args.seeds} seeds:")
            for key, values in scores.items():
                if len(values) > 1:
                    beat = sum(1 for v in values if v > baseline.auc) if baseline else 0
                    print(f"  {key:7} AUC {statistics.mean(values):.3f} "
                          f"+/- {statistics.stdev(values):.3f}  "
                          f"(beats l3_paths in {beat}/{len(values)} seeds)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        if gnn is not None:
            payload["gnn_history"] = gnn.history
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
