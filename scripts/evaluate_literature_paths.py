#!/usr/bin/env python3
"""
Do literature co-mentions add anything the graph's own paths do not?

Builds sentence-level co-mention edges from the pre-cutoff abstract cache and
runs three checks, each through this project's harness with degree-matched
negatives:

1. The literature length-2 path score on its own.
2. A fixed-weight percentile blend with weighted length-3 paths, paired on the
   same negative samples, so each weight is compared seed by seed with the
   structural score alone.
3. The literature score *within* groups of pairs that do and do not already
   have a length-3 path. This is the decisive check: a gap in literature
   connectivity between held-out pairs and negatives can come entirely from
   the structure the graph already has.

Requires the abstract cache for the cutoff:
    python scripts/fetch_literature_context.py --cutoff 2016

Usage:
    python scripts/evaluate_literature_paths.py --cutoff 2016
"""

import argparse
import contextlib
import io
import json
import random
import statistics
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from litkg.evaluation import build_temporal_split, evaluate_baselines  # noqa: E402
from litkg.evaluation.baselines import LinkPredictor, WeightedL3PathPredictor  # noqa: E402
from litkg.evaluation.harness import sample_negatives  # noqa: E402
from litkg.phase1.kg_preprocessor import CivicProcessor  # noqa: E402
from litkg.phase2.literature_edges import (  # noqa: E402
    LINKED_TYPES,
    CoMentionLinker,
    LiteraturePathPredictor,
    build_comention_edges,
)
from litkg.utils.config import get_data_dir, load_config  # noqa: E402
from litkg.utils.logging import setup_logging  # noqa: E402

from evaluate_link_prediction import load_dated_edges  # noqa: E402


class PercentileBlend(LinkPredictor):
    """Fixed-weight average of component percentiles against a fixed reference."""

    def __init__(self, parts, weights, name):
        self.parts, self.weights, self.name = parts, weights, name

    def fit(self, graph):
        self.graph = graph
        for part in self.parts:
            part.fit(graph)
        rng = random.Random(0)
        nodes = sorted(graph.nodes())
        sample = [tuple(sorted(e)) for e in graph.edges()]
        for _ in range(min(20000, len(nodes) * 8)):
            u, v = rng.choice(nodes), rng.choice(nodes)
            if u != v:
                sample.append((u, v))
        self.references = [np.sort(np.asarray(p.score_pairs(sample))) for p in self.parts]
        return self

    def score_pairs(self, pairs):
        total = np.zeros(len(pairs))
        for part, reference, weight in zip(self.parts, self.references, self.weights):
            values = np.asarray(part.score_pairs(pairs))
            low = np.searchsorted(reference, values, "left")
            high = np.searchsorted(reference, values, "right")
            total += weight * ((low + high) / 2.0) / reference.size
        return total.tolist()

    def score(self, u, v):
        return self.score_pairs([(u, v)])[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016)
    parser.add_argument("--samples", type=int, default=8,
                        help="Negative samples; the predictors are deterministic")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()

    civic = get_data_dir() / "external" / "civic"
    cache = get_data_dir() / "processed" / "literature_context" / f"abstracts_pre{args.cutoff}.json"
    if not cache.exists():
        print(f"No abstract cache at {cache}. Run:\n  python "
              f"scripts/fetch_literature_context.py --cutoff {args.cutoff}", file=sys.stderr)
        return 1

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        dated, backbone, node_types, _ = load_dated_edges(civic)
        split = build_temporal_split(dated, args.cutoff, backbone)
        processor = CivicProcessor(load_config())
        entities, _ = processor._process_civic_evidence(
            civic / "civic_evidence.tsv", civic / "civic_variants.tsv")
        genes = processor._process_civic_genes(civic / "civic_genes.tsv")

    train = nx.Graph(split.train_edges | split.backbone_edges)
    nodes = set(train.nodes())
    linker = CoMentionLinker(
        gene_symbols={g.name: g.id for g in genes if g.id in nodes},
        named_entities=[(e.id, e.name, getattr(e, "synonyms", []) or [])
                        for e in entities if e.type in LINKED_TYPES and e.id in nodes],
    )
    abstracts = json.loads(cache.read_text())["abstracts"]
    edges = build_comention_edges(
        (text for texts in abstracts.values() for text in texts), linker, nodes)
    distinct = len({t for texts in abstracts.values() for t in texts})
    print(f"cutoff {args.cutoff}: {distinct} distinct pre-cutoff abstracts, "
          f"{len(edges)} co-mention edges\n")
    report = {"cutoff": args.cutoff, "abstracts": distinct, "edges": len(edges)}

    def structural():
        return WeightedL3PathPredictor(weights=split.edge_weights())

    # 1 and 2: standalone and blends, paired on the same negative samples.
    arms = {"literature_l2 alone": lambda: LiteraturePathPredictor(edges)}
    for weight in (0.0, 0.25, 0.5):
        arms[f"blend, literature weight {weight}"] = (
            lambda w=weight: PercentileBlend(
                [structural(), LiteraturePathPredictor(edges)], [1 - w, w], "blend"))
    results = {}
    for label, make in arms.items():
        rows = []
        for sample in range(args.samples):
            predictor = make()
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                rep = evaluate_baselines(split, node_types=node_types, predictors=[predictor],
                                         negatives_per_positive=10, seed=sample,
                                         degree_matched=True)
            r = rep.results[predictor.name]
            rows.append((r.auc, r.average_precision, getattr(r, "hits_at_100", 0.0)))
        results[label] = rows

    base = results["blend, literature weight 0.0"]
    report["arms"] = {}
    for label, rows in results.items():
        aucs = [r[0] for r in rows]
        line = (f"  {label:32s} AUC {statistics.mean(aucs):.4f} "
                f"[{min(aucs):.4f}, {max(aucs):.4f}]  "
                f"AP {statistics.mean(r[1] for r in rows):.3f}  "
                f"H@100 {statistics.mean(r[2] for r in rows):.3f}")
        wins = None
        if label.startswith("blend") and label != "blend, literature weight 0.0":
            wins = sum(1 for b, x in zip(base, rows) if x[0] > b[0])
            line += f"  beats weight 0 in {wins}/{len(rows)}"
        print(line)
        report["arms"][label] = {"auc": statistics.mean(aucs), "wins_vs_structural": wins}

    # 3: condition on the structure the graph already has.
    fitted_l3 = structural().fit(train)
    literature = LiteraturePathPredictor(edges).fit(train)
    positives = sorted(split.test_edges)
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        negatives = sample_negatives(positives, train, node_types=node_types,
                                     negatives_per_positive=10, known_edges=set(train.edges()),
                                     seed=0, degree_matched=True)
    from sklearn.metrics import roc_auc_score

    print("\nLiterature score within groups by existing length-3 structure:")
    report["conditioned"] = {}
    for label, keep in (("no length-3 path", lambda s: s == 0),
                        ("has a length-3 path", lambda s: s > 0)):
        pos = [p for p in positives if keep(fitted_l3.score(*p))]
        neg = [p for p in negatives if keep(fitted_l3.score(*p))]
        labels = [1] * len(pos) + [0] * len(neg)
        scores = [literature.score(*p) for p in pos + neg]
        auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
        share_pos = sum(literature.score(*p) > 0 for p in pos) / max(len(pos), 1)
        share_neg = sum(literature.score(*p) > 0 for p in neg) / max(len(neg), 1)
        print(f"  {label:20s} held-out {len(pos):4d} ({share_pos:5.1%} with a literature path)  "
              f"negatives {len(neg):5d} ({share_neg:5.1%})  AUC {auc:.3f}")
        report["conditioned"][label] = {"auc": auc, "positives": len(pos), "negatives": len(neg)}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
