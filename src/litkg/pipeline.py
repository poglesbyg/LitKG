"""
End-to-end discovery: graph, ranking, evidence, explanation.

The project had two chains that never met. One reads CIVIC, trains a link
predictor and ranks unobserved pairs. The other reads the Phase 1 literature,
builds a vector store and answers questions with citations. Nothing joined a
prediction to its evidence.

Joining them needed more than wiring. The bundled literature corpus contains no
literal mention of the entities the top predictions involve -- it is general
cancer genomics, while the predictions concern VHL variants and ABL1 resistance
mutations -- so a retrieval against it returns unrelated passages. Evidence has
to be fetched *for the candidates*, not assumed to be present.

What this produces is a ranked list of candidate associations, each with the
passages that discuss them and a rationale citing those passages. What it does
not produce is a validated prediction: the ranking's precision does not
replicate across cutoffs (35x lift at a 2016 cutoff, 5x at 2018, 0x at 2020),
so the output is material for a human to judge rather than an answer.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from litkg.utils.logging import LoggerMixin

Edge = Tuple[str, str]


@dataclass
class Candidate:
    """One proposed association, with whatever evidence was found for it."""

    rank: int
    source: str
    target: str
    source_name: str
    target_name: str
    source_type: str
    target_type: str
    passages: List[Dict[str, str]] = field(default_factory=list)
    rationale: str = ""
    known_outcome: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "source_name": self.source_name, "target_name": self.target_name,
            "source_type": self.source_type, "target_type": self.target_type,
            "passages": self.passages, "rationale": self.rationale,
            "known_outcome": self.known_outcome,
        }


@dataclass
class DiscoveryConfig:
    """
    Settings for a discovery run.

    `cutoff` serves two purposes. Left unset, the run uses all available
    evidence and literature, which is what you want to propose something new.
    Set, it restricts both the graph and the literature to before that year,
    which reproduces the evaluation setting and lets the output be checked
    against what was curated afterwards.
    """

    cutoff: Optional[int] = None
    top: int = 20
    explain: int = 5
    seeds: int = 3
    epochs: int = 300
    max_articles_per_candidate: int = 6
    output_dir: Path = Path("outputs/discovery")


class DiscoveryPipeline(LoggerMixin):
    """Ranks candidate associations and gathers the evidence for them."""

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.candidates: List[Candidate] = []

    # ------------------------------------------------------------------

    def _rank(self) -> Tuple[List[Edge], Dict[str, str], Dict[str, str], Set[Edge]]:
        """Train on the graph and rank unobserved pairs by mean rank over seeds."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from evaluate_link_prediction import load_dated_edges
        from rank_predictions import candidate_pairs

        from litkg.evaluation import build_temporal_split
        from litkg.evaluation.harness import build_graph
        from litkg.phase2.link_prediction import HybridLinkPredictor, TrainingConfig
        from litkg.utils.config import get_data_dir

        dated, backbone, node_types, node_text = load_dated_edges(
            get_data_dir() / "external" / "civic"
        )
        # No cutoff means "use everything": a split at a year beyond the data
        # keeps every edge in training and leaves nothing held out, which is the
        # right shape for proposing rather than evaluating.
        cutoff = self.config.cutoff or 9999
        split = build_temporal_split(dated, cutoff, backbone)
        graph = build_graph(split.train_edges | split.backbone_edges)
        known = set(split.train_edges) | set(split.backbone_edges)

        wanted = {
            tuple(sorted((node_types.get(u, "?"), node_types.get(v, "?"))))
            for u, v in (split.test_edges or split.train_edges)
        }
        candidates = sorted(candidate_pairs(graph, node_types, wanted, known))
        self.logger.info(f"{len(candidates):,} unobserved pairs to rank")

        summed = [0.0] * len(candidates)
        for seed in range(self.config.seeds):
            config = TrainingConfig(
                epochs=self.config.epochs, seed=seed, loss="bpr", num_layers=2,
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
            self.logger.info(f"  seed {seed} ranked")

        ranked = [c for c, _ in sorted(zip(candidates, summed), key=lambda kv: kv[1])]
        return ranked, node_types, node_text, set(split.test_edges)

    def _gather_evidence(self, candidate: Candidate) -> List[Dict[str, str]]:
        """
        Fetch literature about this specific pair.

        Querying for both entities together is the point: a passage that
        mentions only one of them is background, not evidence for an
        association. The single-entity fallback is marked as such so a reader
        can tell the difference.
        """
        from litkg.phase2.literature_context import (
            ContextConfig,
            LiteratureContextFetcher,
        )

        cutoff = self.config.cutoff or 9999
        fetcher = LiteratureContextFetcher(ContextConfig(
            cutoff_year=cutoff,
            max_articles=self.config.max_articles_per_candidate,
        ))

        passages: List[Dict[str, str]] = []
        pair_term = (f'{fetcher._query_term(candidate.source_name)} AND '
                     f'{fetcher._query_term(candidate.target_name)}')
        try:
            pmids = fetcher._search(pair_term)
            for article in fetcher._fetch_abstracts(pmids):
                passages.append({"text": article["text"][:1200],
                                 "year": article["year"], "support": "co-mention"})
        except Exception as e:
            self.logger.debug(f"Pair query failed: {e}")

        if not passages:
            for name in (candidate.source_name, candidate.target_name):
                try:
                    pmids = fetcher._search(fetcher._query_term(name))
                    for article in fetcher._fetch_abstracts(pmids[:2]):
                        passages.append({"text": article["text"][:1200],
                                         "year": article["year"],
                                         "support": f"single entity: {name}"})
                except Exception as e:
                    self.logger.debug(f"Entity query failed for {name}: {e}")
        return passages

    def _explain(self, candidate: Candidate) -> str:
        """Ask the local model to justify the pair from the passages, or refuse."""
        if not candidate.passages:
            return ("No literature was retrieved for this pair, so there is "
                    "nothing to justify it with.")

        from litkg.llm_integration.unified_llm_interface import UnifiedLLMManager

        context = "\n\n".join(
            f"[{i + 1}] ({p['support']}, {p['year']}) {p['text']}"
            for i, p in enumerate(candidate.passages)
        )
        prompt = (
            f"A knowledge graph model proposes an association between "
            f"{candidate.source_name} ({candidate.source_type}) and "
            f"{candidate.target_name} ({candidate.target_type}).\n\n"
            f"Using ONLY the evidence below, say what support exists for that "
            f"association and what is missing. Cite evidence as [n]. If the "
            f"evidence does not address the association, say so plainly rather "
            f"than filling the gap from memory.\n\n"
            f"Evidence:\n{context}\n\nAssessment:"
        )
        try:
            response = UnifiedLLMManager().process_biomedical_task(
                task="literature_analysis", input_data=prompt, max_tokens=400
            )
            return response.content.strip()
        except Exception as e:
            self.logger.warning(f"Explanation failed: {e}")
            return ""

    # ------------------------------------------------------------------

    def run(self) -> List[Candidate]:
        config = self.config
        ranked, node_types, node_text, held_out = self._rank()

        self.candidates = []
        for position, (source, target) in enumerate(ranked[: config.top], start=1):
            candidate = Candidate(
                rank=position, source=source, target=target,
                source_name=node_text.get(source, source),
                target_name=node_text.get(target, target),
                source_type=node_types.get(source, "?"),
                target_type=node_types.get(target, "?"),
                known_outcome=((source, target) in held_out) if held_out else None,
            )
            candidate.passages = self._gather_evidence(candidate)
            self.candidates.append(candidate)

        for candidate in self.candidates[: config.explain]:
            candidate.rationale = self._explain(candidate)

        return self.candidates

    def report(self) -> str:
        """A readable summary, honest about what the ranking is worth."""
        lines = [
            "CANDIDATE ASSOCIATIONS",
            "=" * 72,
            "Ranked by a link predictor trained on the knowledge graph, with the",
            "literature retrieved for each pair. Precision does not replicate",
            "across time cutoffs, so treat these as candidates to judge, not",
            "findings. The evidence is the part worth reading.",
            "",
        ]
        with_evidence = sum(1 for c in self.candidates if c.passages)
        co_mention = sum(
            1 for c in self.candidates
            if any(p["support"] == "co-mention" for p in c.passages)
        )
        lines.append(f"{len(self.candidates)} candidates; {with_evidence} have "
                     f"literature, {co_mention} have a paper mentioning both entities.")
        lines.append("")

        for candidate in self.candidates:
            marker = ""
            if candidate.known_outcome is True:
                marker = "  [curated after the cutoff]"
            lines.append(f"{candidate.rank:>3}. {candidate.source_name} "
                         f"({candidate.source_type})  <->  {candidate.target_name} "
                         f"({candidate.target_type}){marker}")
            if candidate.passages:
                kinds = {p["support"] for p in candidate.passages}
                lines.append(f"     evidence: {len(candidate.passages)} passage(s), "
                             f"{', '.join(sorted(kinds))}")
            else:
                lines.append("     evidence: none retrieved")
            if candidate.rationale:
                for line in candidate.rationale.splitlines():
                    if line.strip():
                        lines.append(f"     {line.strip()}")
            lines.append("")
        return "\n".join(lines)

    def save(self) -> Path:
        directory = Path(self.config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "candidates.json"
        path.write_text(json.dumps(
            {"cutoff": self.config.cutoff,
             "candidates": [c.to_dict() for c in self.candidates]}, indent=2))
        (directory / "report.txt").write_text(self.report())
        return path
