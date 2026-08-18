#!/usr/bin/env python3
"""
Score the RAG retriever against the judged query set.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --sweep      # k, hops and hub cap
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litkg.evaluation import evaluate_retrieval, load_queries
from litkg.langchain_integration import PipelineConfig, RAGPipeline
from litkg.langchain_integration.rag_system import BiomedicalRAGSystem
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging


def retriever_for(pipeline, k, hops, cap):
    """A retrieve(query) -> ranked pmids function for one configuration."""
    if cap is not None:
        pipeline.chunk_index.DEFAULT_MAX_TRAVERSAL_DEGREE = cap
    system = BiomedicalRAGSystem(
        vector_store=pipeline.vector_store,
        knowledge_graph=pipeline.graph,
        chunk_index=pipeline.chunk_index,
        k=k, max_hops=hops, llm_manager=object(),
    )

    def retrieve(question: str):
        seen, ordered = set(), []
        for document in system.retriever.invoke(question):
            pmid = str(document.metadata.get("pmid", ""))
            if pmid and pmid not in seen:
                seen.add(pmid)
                ordered.append(pmid)
        return ordered

    return retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--multihop", action="store_true",
                        help="Score against the bridge query set instead")
    parser.add_argument("--eval-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    eval_dir = args.eval_dir or (get_data_dir() / "processed" / "retrieval_eval")
    prefix = "multihop_" if args.multihop else ""
    queries_path = eval_dir / f"{prefix}queries.json"
    corpus_path = eval_dir / f"{prefix}corpus.json"
    if not queries_path.exists():
        print(f"No query set at {queries_path}. Run "
              f"scripts/build_retrieval_queryset.py first.", file=sys.stderr)
        return 1

    queries = load_queries(queries_path)
    pipeline = RAGPipeline(PipelineConfig(
        documents_path=corpus_path,
        vector_store_path=eval_dir / f"{prefix}vector_store",
    )).build()
    print(f"{len(queries)} judged queries over {len(pipeline.documents)} documents\n")

    if not args.sweep:
        metrics = evaluate_retrieval(
            retriever_for(pipeline, args.k, args.hops, args.cap), queries, k=args.k
        )
        print(metrics.format())
        return 0

    print(f"{'k':>3} {'hops':>5} {'cap':>6}   metrics")
    print("-" * 96)
    for k in (5, 10):
        for hops in (0, 1, 2):
            for cap in ((None,) if hops == 0 else (0, 50)):
                metrics = evaluate_retrieval(
                    retriever_for(pipeline, k, hops, cap), queries, k=k
                )
                label = "-" if hops == 0 else ("off" if cap == 0 else str(cap))
                print(f"{k:>3} {hops:>5} {label:>6}   {metrics.format()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
