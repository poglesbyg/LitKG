#!/usr/bin/env python3
"""
Run Phase 3 over real predictions and measure whether its confidence means
anything.

Phase 3 -- confidence scoring, biological plausibility, novelty -- has only ever
run on synthetic input: six hardcoded relationships and random tensors. It could
not be otherwise, because nothing produced real predictions to assess.
`rank_predictions.py` now does.

The point is not merely to run it on real data. Because the predictions come
from a temporal holdout, every one of them has a known outcome: CIVIC either
curated it in the years after the cutoff or did not. So Phase 3's confidence can
be *checked* rather than displayed -- does a higher score actually mean a higher
chance of being real?

Usage:
    python scripts/assess_predictions.py --cutoff 2016 --top 300
"""

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from litkg.evaluation import build_temporal_split
from litkg.evaluation.harness import build_graph
from litkg.phase3.confidence_scoring import ConfidenceCalibrator, ConfidenceScorer
from litkg.phase3.novelty_detection import BiologicalPlausibilityChecker, NovelRelation
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging

from evaluate_link_prediction import load_dated_edges  # noqa: E402

Edge = Tuple[str, str]


def evidence_index(cutoff: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pre-cutoff CIVIC evidence, indexed by the node it concerns.

    Only evidence published before the cutoff is used. Evidence from after it
    describes the very associations being predicted, so including it would let
    Phase 3 grade its own answers.
    """
    from litkg.evaluation import extract_publication_year

    path = get_data_dir() / "external" / "civic" / "civic_evidence.tsv"
    frame = pd.read_csv(path, sep="\t", low_memory=False)

    by_profile: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in frame.itertuples():
        year = extract_publication_year(getattr(row, "citation", None))
        if year is None or year >= cutoff:
            continue
        record = {
            "pmid": str(getattr(row, "citation_id", "")).strip(),
            "year": year,
            "level": str(getattr(row, "evidence_level", "")).strip().upper(),
            "rating": getattr(row, "rating", None),
            "direction": str(getattr(row, "evidence_direction", "")).strip(),
            "profile": str(getattr(row, "molecular_profile", "")).strip(),
            "disease": str(getattr(row, "disease", "")).strip(),
        }
        by_profile[record["profile"].upper()].append(record)
        by_profile[record["disease"].upper()].append(record)
    return by_profile


def literature_payload(records: List[Dict[str, Any]], cutoff: int) -> Dict[str, Any]:
    """
    Shape real CIVIC evidence into what the confidence assessor expects.

    Every field is derived, not invented. CIVIC has no impact factors or
    citation counts, so those keep the assessor's own defaults rather than
    being filled with plausible-looking numbers.
    """
    papers = sorted({r["pmid"] for r in records if r["pmid"]})
    years = [r["year"] for r in records]
    # Recency relative to a 30-year window ending at the cutoff.
    recency = (
        min(max((max(years) - (cutoff - 30)) / 30.0, 0.0), 1.0) if years else 0.5
    )
    # CIVIC evidence levels: A validated, B clinical, C case study,
    # D preclinical, E inferential.
    strength = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.45, "E": 0.3}
    methodology = (
        float(np.mean([strength.get(r["level"], 0.5) for r in records]))
        if records else 0.5
    )
    return {"papers": papers, "recency_score": recency,
            "methodology_score": methodology}


def experimental_payload(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """CIVIC evidence rows are the experiments; supporting ones only."""
    supporting = [r for r in records if r["direction"] != "Does Not Support"]
    return {"experiments": [
        {"type": r["level"], "rating": r["rating"]} for r in supporting
    ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016)
    parser.add_argument("--top", type=int, default=300,
                        help="How many ranked predictions to assess")
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/prospective_2016.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/phase3_assessment.json"))
    args = parser.parse_args()

    setup_logging()
    if not args.predictions.exists():
        print(f"No predictions at {args.predictions}. Run "
              f"scripts/rank_predictions.py first.", file=sys.stderr)
        return 1

    payload = json.loads(args.predictions.read_text())
    predictions = payload["top"][: args.top]
    if not predictions:
        print("No predictions to assess", file=sys.stderr)
        return 1

    dated, backbone, node_types, node_text = load_dated_edges(
        get_data_dir() / "external" / "civic"
    )
    split = build_temporal_split(dated, args.cutoff, backbone)
    graph = build_graph(split.train_edges | split.backbone_edges)
    evidence = evidence_index(args.cutoff)

    scorer = ConfidenceScorer()
    checker = BiologicalPlausibilityChecker()

    rows: List[Dict[str, Any]] = []
    for item in predictions:
        source, target = item["source"], item["target"]
        names = (node_text.get(source, source), node_text.get(target, target))
        records = (evidence.get(names[0].upper(), [])
                   + evidence.get(names[1].upper(), []))

        metrics = scorer.assess_relationship_confidence(
            relationship={"entity1": source, "entity2": target},
            literature_data=literature_payload(records, args.cutoff),
            experimental_data=experimental_payload(records),
        )

        relation = NovelRelation(
            entity1=names[0], entity2=names[1],
            relation_type="ASSOCIATED_WITH",
            confidence_score=metrics.overall_confidence,
            supporting_papers=literature_payload(records, args.cutoff)["papers"],
        )
        plausibility = checker.check_plausibility(
            relation,
            entity_types={names[0]: node_types.get(source, "?"),
                          names[1]: node_types.get(target, "?")},
        )

        rows.append({
            "source_name": names[0], "target_name": names[1],
            "source_type": node_types.get(source, "?"),
            "target_type": node_types.get(target, "?"),
            "confidence": metrics.overall_confidence,
            "literature_confidence": metrics.literature_confidence,
            "experimental_confidence": metrics.experimental_confidence,
            "supporting_papers": metrics.supporting_papers,
            # check_plausibility returns "score"; reading a different key
            # silently yielded 0.0 for every prediction.
            "plausibility": float(plausibility.get("score", 0.0)),
            "later_curated": bool(item["later_curated"]),
        })

    outcomes = np.array([1 if r["later_curated"] else 0 for r in rows])
    confidences = np.array([r["confidence"] for r in rows])
    plausibilities = np.array([r["plausibility"] for r in rows])

    print(f"\nAssessed {len(rows)} predictions from a {args.cutoff} model; "
          f"{int(outcomes.sum())} were curated afterwards\n")

    # The question Phase 3 exists to answer: does a higher score mean a higher
    # chance of being real? Comparing group means is the honest test, and it
    # can fail.
    def separation(name: str, values: np.ndarray) -> Dict[str, Any]:
        hit, miss = values[outcomes == 1], values[outcomes == 0]
        if not len(hit) or not len(miss):
            return {}
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(outcomes, values))
        print(f"  {name:24} curated {hit.mean():.3f}  not-curated {miss.mean():.3f}"
              f"   AUC {auc:.3f}")
        return {"curated_mean": float(hit.mean()),
                "not_curated_mean": float(miss.mean()), "auc": auc}

    print("Does the score separate what was later curated?")
    stats = {
        "confidence": separation("overall confidence", confidences),
        "plausibility": separation("type-pair prior", plausibilities),
    }
    print("  (AUC 0.5 means the score carries no information about the outcome)")
    print("  The plausibility score takes one value per entity-type pair, so its")
    print("  AUC measures a type prior rather than any reasoning about biology.")

    # The rate that a user can actually act on. It needs no model: it says which
    # kinds of prediction are worth reading at all.
    by_pair: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        key = "-".join(sorted((row["source_type"], row["target_type"])))
        by_pair[key][1] += 1
        by_pair[key][0] += int(row["later_curated"])
    print("\nCuration rate by entity-type pair:")
    pair_stats = {}
    for key in sorted(by_pair, key=lambda k: -by_pair[k][0] / max(by_pair[k][1], 1)):
        hits, total = by_pair[key]
        pair_stats[key] = {"curated": hits, "total": total,
                           "rate": hits / total if total else 0.0}
        print(f"  {key:24} {hits:3}/{total:3}  {hits / max(total, 1):6.1%}")
    overall = outcomes.mean()
    best = max(pair_stats.values(), key=lambda v: v["rate"]) if pair_stats else None
    if best and overall:
        print(f"\n  Reading only the best type pair lifts precision from "
              f"{overall:.1%} to {best['rate']:.1%} on this sample.")
    stats["by_type_pair"] = pair_stats

    # Calibration, fitted on one half and tested on the other. Fitting and
    # reporting on the same data would show a fit, not a calibration.
    midpoint = len(rows) // 2
    calibrator = ConfidenceCalibrator()
    calibration = {}
    if len(set(outcomes[:midpoint].tolist())) > 1:
        calibrator.fit(confidences[:midpoint].tolist(),
                       outcomes[:midpoint].tolist())
        held_out = np.array([calibrator.transform(c)
                             for c in confidences[midpoint:]])
        observed = outcomes[midpoint:].mean()
        print(f"\nCalibration, fitted on the first half and tested on the second:")
        print(f"  mean calibrated confidence {held_out.mean():.3f} against an "
              f"observed rate of {observed:.3f}")
        calibration = {"predicted": float(held_out.mean()),
                       "observed": float(observed)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"cutoff": args.cutoff, "assessed": len(rows),
         "curated": int(outcomes.sum()), "separation": stats,
         "calibration": calibration, "predictions": rows}, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
