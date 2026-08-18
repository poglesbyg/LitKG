#!/usr/bin/env python3
"""
Build a relevance-judged retrieval query set from CIVIC citations.

Retrieval was unmeasured: k, max_hops and the hub-degree cap were set by
judgement with nothing to check them against. CIVIC supplies judgements for
free -- every evidence row cites a paper and states the relationship that paper
supports, so the cited papers are relevant to a question about that
relationship by a curator's judgement rather than ours.

Writes a corpus in the same shape Phase 1 produces, so the existing RAG
pipeline can index it unchanged, plus the judged queries.

Usage:
    python scripts/build_retrieval_queryset.py --queries 60
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from litkg.evaluation import (
    MultiHopQuerySetBuilder,
    QuerySetBuilder,
    load_queries,
    save_queries,
)
from litkg.utils.config import get_data_dir
from litkg.utils.logging import setup_logging

_REQUEST_INTERVAL = 0.36   # NCBI asks for <=3 requests/second without a key


def fetch_abstracts(pmids: Sequence[str], email: str, batch_size: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch abstracts by PMID.

    Batched, unlike the per-entity context fetcher: efetch takes a comma
    separated id list, so a few hundred papers cost a handful of requests
    rather than a few hundred.
    """
    from Bio import Entrez

    Entrez.email = email
    documents: List[Dict[str, Any]] = []

    for start in range(0, len(pmids), batch_size):
        batch = list(pmids[start:start + batch_size])
        time.sleep(_REQUEST_INTERVAL)
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
        try:
            records = Entrez.read(handle)
        finally:
            handle.close()

        for article in records.get("PubmedArticle", []):
            citation = article.get("MedlineCitation", {})
            info = citation.get("Article", {})
            abstract = info.get("Abstract", {}).get("AbstractText", [])
            text = " ".join(str(part) for part in abstract).strip()
            if not text:
                continue
            documents.append({
                "pmid": str(citation.get("PMID", "")),
                "title": str(info.get("ArticleTitle", "")),
                "abstract": text,
                "journal": str(info.get("Journal", {}).get("Title", "")),
            })
        print(f"  fetched {min(start + batch_size, len(pmids))}/{len(pmids)}")

    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=60,
                        help="How many judged queries to build")
    parser.add_argument("--min-relevant", type=int, default=3,
                        help="Minimum cited papers per query")
    parser.add_argument("--email", default="litkg@example.org")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-corpus", action="store_true",
                        help="Refetch the corpus for an existing query set instead "
                             "of deriving new queries. The query sets are tracked "
                             "and the corpora are not, so this is what a fresh "
                             "clone needs to reproduce a reported number.")
    parser.add_argument("--multihop", action="store_true",
                        help="Build bridge queries whose relevant papers share no "
                             "vocabulary with the query")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    out_dir = args.out_dir or (get_data_dir() / "processed" / "retrieval_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = get_data_dir() / "external" / "civic" / "civic_evidence.tsv"
    if not evidence_path.exists():
        print(f"No CIVIC evidence at {evidence_path}", file=sys.stderr)
        return 1

    prefix_early = "multihop_" if args.multihop else ""
    if args.rebuild_corpus:
        queries_path = out_dir / f"{prefix_early}queries.json"
        if not queries_path.exists():
            print(f"No query set at {queries_path}", file=sys.stderr)
            return 1
        queries = load_queries(queries_path)
        pmids = sorted({p for q in queries for p in q.relevant_pmids} |
                       {p for q in queries for p in getattr(q, "seed_pmids", [])})
        print(f"Refetching {len(pmids)} papers for {len(queries)} existing queries")
        documents = fetch_abstracts(pmids, email=args.email)
        corpus_path = out_dir / f"{prefix_early}corpus.json"
        corpus_path.write_text(json.dumps({"documents": documents}, indent=2))
        print(f"\n{len(documents)} documents\n  {corpus_path}")
        return 0

    evidence = pd.read_csv(evidence_path, sep="\t", low_memory=False)

    if args.multihop:
        builder = MultiHopQuerySetBuilder()
        queries = builder.build(
            evidence, min_relevant=args.min_relevant,
            max_queries=args.queries, seed=args.seed,
        )
    else:
        builder = None
        queries = QuerySetBuilder().build(
            evidence, min_relevant=args.min_relevant,
            max_queries=args.queries, seed=args.seed,
        )
    if not queries:
        print("No query groups met the threshold", file=sys.stderr)
        return 1

    # Seed papers are fetched too: they are what vector search should find, and
    # a corpus without them would hand expansion an easier problem than reality.
    pmids = sorted({p for q in queries for p in q.relevant_pmids} |
                   {p for q in queries for p in getattr(q, "seed_pmids", [])})
    print(f"Fetching {len(pmids)} papers for {len(queries)} queries")
    documents = fetch_abstracts(pmids, email=args.email)

    retrieved = {d["pmid"] for d in documents}
    missing = [p for p in pmids if p not in retrieved]
    if missing:
        # A judgement pointing at a paper absent from the corpus is
        # unreachable, and counting it in the denominator would understate
        # recall for reasons that have nothing to do with retrieval.
        print(f"{len(missing)} cited papers had no retrievable abstract; "
              f"dropping them from the judgements")
        for query in queries:
            query.relevant_pmids = [p for p in query.relevant_pmids if p in retrieved]
        queries = [q for q in queries if len(q.relevant_pmids) >= args.min_relevant]

    if args.multihop and builder is not None:
        texts = {d["pmid"]: f"{d['title']} {d['abstract']}" for d in documents}
        queries = builder.filter_lexically_disjoint(
            queries, texts, min_relevant=args.min_relevant
        )
        if not queries:
            print("No queries survived the lexical filter", file=sys.stderr)
            return 1

    prefix = "multihop_" if args.multihop else ""
    corpus_path = out_dir / f"{prefix}corpus.json"
    queries_path = out_dir / f"{prefix}queries.json"
    corpus_path.write_text(json.dumps({"documents": documents}, indent=2))
    save_queries(queries, queries_path)

    print(f"\n{len(queries)} queries, {len(documents)} documents")
    print(f"  {corpus_path}")
    print(f"  {queries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
