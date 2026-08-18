#!/usr/bin/env python3
"""
Ask questions over the Phase 1 corpus, grounded in retrieved evidence.

This is the entry point that was missing: the retrievers, chunk-to-graph index
and agents all existed and were unit tested, but nothing connected them to real
data, so the graph-aware path was never exercised outside tests.

Usage:
    python scripts/run_rag.py "Why are BRCA1 tumours sensitive to olaparib?"
    python scripts/run_rag.py --coverage          # index stats, no LLM call
    python scripts/run_rag.py --agent "What is the role of BRCA1 in DNA repair?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litkg.langchain_integration import PipelineConfig, RAGPipeline
from litkg.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--hops", type=int, default=1,
                        help="Graph hops beyond the seed passages (0 disables)")
    parser.add_argument("--k", type=int, default=5, help="Passages to retrieve")
    parser.add_argument("--agent", action="store_true",
                        help="Route through the conversational agent")
    parser.add_argument("--coverage", action="store_true",
                        help="Report index coverage and exit without calling the LLM")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Show retrieved passages without generating an answer")
    parser.add_argument("--rebuild", action="store_true",
                        help="Discard and rebuild the vector store")
    args = parser.parse_args()

    setup_logging()

    try:
        pipeline = RAGPipeline(
            PipelineConfig(k=args.k, max_hops=args.hops)
        ).build(rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.coverage:
        print(json.dumps(pipeline.coverage(), indent=2))
        return 0

    if not args.question:
        parser.error("a question is required unless --coverage is given")

    if args.retrieval_only:
        # Useful on its own: retrieval can be inspected without waiting on a
        # local model, and hop_distance shows what graph expansion contributed.
        system = pipeline.rag_system()
        for document in system.retriever.invoke(args.question):
            hop = document.metadata.get("hop_distance", 0)
            pmid = document.metadata.get("pmid", "?")
            print(f"[hop {hop}] pmid {pmid}: {document.page_content[:200].strip()}...")
        return 0

    if args.agent:
        result = pipeline.agent().chat(args.question)
        print(result.get("response", result))
        return 0

    result = pipeline.rag_system().answer(args.question)
    print(result["answer"])
    print(f"\n--- {result['num_sources']} sources ({result.get('model', 'unknown')}) ---")
    for index, source in enumerate(result["sources"], start=1):
        # answer() flattens document metadata into each source dict rather than
        # nesting it under "metadata"; reading a nested key silently yields
        # "hop 0, pmid ?" for every source.
        hop = source.get("hop_distance", 0)
        title = (source.get("title") or source.get("content", ""))[:70]
        print(f"[{index}] hop {hop}  pmid {source.get('pmid', '?')}  {title.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
