#!/usr/bin/env python3
"""
Fetch pre-cutoff PubMed context sentences for knowledge graph entities.

Contexts are the only feature source here that can leak, so the date filter is
applied at the query and re-checked per record, and the cutoff is baked into
the cache file name. Resumable: rerun after an interruption and it continues.

Usage:
    python scripts/fetch_literature_context.py --cutoff 2016
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from litkg.phase2.literature_context import ContextConfig, LiteratureContextFetcher
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging

from evaluate_link_prediction import load_dated_edges  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2016)
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only fetch this many entities (for a trial run)")
    parser.add_argument("--types", default=None,
                        help="Comma-separated entity types to fetch, e.g. GENE,DISEASE")
    args = parser.parse_args()

    setup_logging()
    civic_dir = get_data_dir() / "external" / "civic"
    _dated, _backbone, node_types, node_text = load_dated_edges(civic_dir)

    wanted = set(args.types.split(",")) if args.types else None

    # Fetch in order of how much the evaluation depends on a node: endpoints of
    # held-out pairs first, then the training graph by degree. A partial run is
    # then still useful, whereas sorting by name length fetches 1694 variants
    # last -- and variants are an endpoint of 84% of held-out pairs.
    from collections import Counter
    from litkg.evaluation import build_temporal_split

    split = build_temporal_split(_dated, args.cutoff, _backbone)
    importance: Counter = Counter()
    for u, v in split.test_edges:
        importance[u] += 1000
        importance[v] += 1000
    for u, v in split.train_edges | split.backbone_edges:
        importance[u] += 1
        importance[v] += 1

    candidates = [
        (node_id, name) for node_id, name in node_text.items()
        if wanted is None or node_types.get(node_id) in wanted
    ]
    candidates.sort(key=lambda item: -importance.get(item[0], 0))
    names = list(dict.fromkeys(name for _id, name in candidates))
    if args.limit:
        names = names[: args.limit]

    fetcher = LiteratureContextFetcher(
        ContextConfig(cutoff_year=args.cutoff, max_articles=args.max_articles)
    )
    fetcher.gather(names)
    print(f"coverage: {fetcher.coverage(node_text):.1%} of "
          f"{len(node_text)} nodes have pre-{args.cutoff} context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
