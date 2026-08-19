#!/usr/bin/env python3
"""
Propose candidate associations and gather the evidence for them.

The one entry point that runs the whole thing: build the graph, train a link
predictor, rank unobserved pairs, fetch the literature about each, and ask the
local model what support exists.

    python scripts/discover.py --top 20 --explain 5

Pass --cutoff to reproduce the evaluation setting instead. The graph and the
literature are both restricted to before that year, and candidates that were
subsequently curated are marked, so the output can be checked rather than
trusted.

    python scripts/discover.py --cutoff 2016 --top 20

The ranking's precision does not replicate across cutoffs, so this produces
candidates for a person to judge, not findings.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litkg.pipeline import DiscoveryConfig, DiscoveryPipeline
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20,
                        help="Candidates to gather evidence for")
    parser.add_argument("--explain", type=int, default=5,
                        help="How many to send to the LLM for a rationale "
                             "(each costs a local generation)")
    parser.add_argument("--cutoff", type=int, default=None,
                        help="Restrict graph and literature to before this year, "
                             "and mark candidates curated afterwards")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    if not (get_data_dir() / "external" / "civic" / "civic_evidence.tsv").exists():
        print("CIVIC data not found. Run `make run-phase1` first.", file=sys.stderr)
        return 1

    config = DiscoveryConfig(
        cutoff=args.cutoff, top=args.top, explain=args.explain,
        seeds=args.seeds, epochs=args.epochs,
        output_dir=args.output_dir or Path("outputs/discovery"),
    )
    pipeline = DiscoveryPipeline(config)
    pipeline.run()
    print()
    print(pipeline.report())
    path = pipeline.save()
    print(f"Wrote {path} and {path.parent / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
