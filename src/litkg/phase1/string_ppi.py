"""
STRING protein-protein interactions, as gene-gene edges.

Why this source at all
----------------------
The CIVIC graph is strictly multipartite: 0 of its edges join two nodes of the
same type, and every held-out pair is cross-type. That is a structural
property, not a data volume problem, and it is why shared-neighbour predictors
are *undefined* here rather than weak, and why everything routes through
length-3 paths.

Gene-gene edges are the one addition that changes the topology class instead of
adding rows to it. Neither CIVIC nor the GDC can supply them: CIVIC links genes
to variants and variants to diseases, and the GDC links genes to cohorts.

The textmining problem, which is the whole reason this module is careful
--------------------------------------------------------------------
STRING's headline `combined_score` fuses seven evidence channels, and one of
them -- `textmining` -- is co-occurrence in PubMed abstracts. Those are the same
papers CIVIC curators read to write the labels this project predicts. An edge
built from them is not independent evidence; it is a compressed version of the
answer.

It is not a small effect. KRAS-BRCA1 scores 0.721 combined, of which 0.721 is
textmining and 0.000 is experiments. TP53-BRAF scores 0.887, of which 0.883 is
textmining. Using `combined_score` would fill the graph with literature
co-mention edges and call the result a protein interaction network.

So the default channel is `experiments` alone: physical and biochemical assays,
which are wet-lab observations rather than readings of the literature.
`database` (curated pathway membership) is available but off by default,
because a curator deciding two genes belong to the same cancer pathway is
closer to the label than to independent evidence. Every channel is selectable
so the choice stays visible and testable rather than baked in.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

from litkg.utils.logging import LoggerMixin

STRING_VERSION = "v12.0"
SPECIES = "9606"
BASE_URL = "https://stringdb-downloads.org/download"

# Column name in the detailed links file -> what the evidence actually is.
CHANNELS = {
    "neighborhood": "genomic neighbourhood in bacteria; weak in human",
    "fusion": "gene fusion events across genomes",
    "cooccurence": "phylogenetic co-occurrence",
    "coexpression": "correlated expression across datasets",
    "experimental": "physical and biochemical assays",
    "database": "curated pathway and complex membership",
    "textmining": "co-occurrence in PubMed abstracts",
}

# Channels that read the literature, directly or through curation of it. Using
# these to predict what a curator later wrote is circular.
LITERATURE_DERIVED = frozenset({"textmining", "database"})

DEFAULT_CHANNELS = ("experimental",)
DEFAULT_MIN_SCORE = 400  # STRING calls 400 medium and 700 high confidence.


@dataclass(frozen=True)
class PPIEdge:
    gene_a: str
    gene_b: str
    score: int
    channel_scores: Tuple[Tuple[str, int], ...] = ()


class StringPPI(LoggerMixin):
    """Loads STRING human interactions and maps them to gene symbols."""

    def __init__(self, data_dir: Path, version: str = STRING_VERSION):
        self.data_dir = Path(data_dir)
        self.version = version

    @property
    def info_path(self) -> Path:
        return self.data_dir / f"{SPECIES}.protein.info.{self.version}.txt.gz"

    @property
    def links_path(self) -> Path:
        return self.data_dir / f"{SPECIES}.protein.links.detailed.{self.version}.txt.gz"

    def download(self, timeout: int = 1800) -> None:
        """Fetch both files if absent. The links file is roughly 140 MB."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        targets = [
            (self.info_path, f"{BASE_URL}/protein.info.{self.version}/{self.info_path.name}"),
            (
                self.links_path,
                f"{BASE_URL}/protein.links.detailed.{self.version}/{self.links_path.name}",
            ),
        ]
        for path, url in targets:
            if path.exists():
                continue
            self.logger.info(f"Downloading {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            path.write_bytes(response.content)

    def protein_to_symbol(self) -> Dict[str, str]:
        """STRING protein id (9606.ENSP...) -> preferred gene symbol."""
        if not self.info_path.exists():
            raise FileNotFoundError(f"STRING info file missing: {self.info_path}")

        mapping: Dict[str, str] = {}
        with gzip.open(self.info_path, "rt", errors="replace") as handle:
            header = handle.readline().lstrip("#").split("\t")
            # v11 called this column protein_external_id, v12 calls it
            # string_protein_id. Match on either rather than pinning one.
            try:
                id_col = next(
                    i
                    for i, c in enumerate(header)
                    if "protein_external_id" in c or "string_protein_id" in c
                )
                name_col = next(i for i, c in enumerate(header) if "preferred_name" in c)
            except StopIteration:
                raise ValueError(f"Unexpected STRING info header: {header}")
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > max(id_col, name_col):
                    mapping[parts[id_col]] = parts[name_col]
        return mapping

    def edges(
        self,
        keep_symbols: Optional[Set[str]] = None,
        channels: Iterable[str] = DEFAULT_CHANNELS,
        min_score: int = DEFAULT_MIN_SCORE,
        allow_literature_channels: bool = False,
    ) -> List[PPIEdge]:
        """
        Gene-gene edges above `min_score` on the requested channels.

        An edge is kept when *any* requested channel clears the threshold, so
        the score reported is the best single channel rather than a fusion of
        them. Fusing is what `combined_score` does, and it is what lets
        textmining carry an edge on its own.
        """
        channels = tuple(channels)
        unknown = set(channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown STRING channels: {sorted(unknown)}")

        leaky = set(channels) & LITERATURE_DERIVED
        if leaky and not allow_literature_channels:
            raise ValueError(
                f"channels {sorted(leaky)} are derived from the literature, which is "
                f"where the CIVIC labels come from. Pass "
                f"allow_literature_channels=True to use them deliberately."
            )

        if not self.links_path.exists():
            raise FileNotFoundError(f"STRING links file missing: {self.links_path}")

        symbols = self.protein_to_symbol()
        seen: Dict[Tuple[str, str], PPIEdge] = {}

        with gzip.open(self.links_path, "rt", errors="replace") as handle:
            header = handle.readline().split()
            try:
                idx = {name: header.index(name) for name in channels}
            except ValueError as e:
                raise ValueError(f"STRING links header lacks a channel: {header}") from e
            a_col, b_col = header.index("protein1"), header.index("protein2")

            for line in handle:
                parts = line.split()
                if len(parts) <= max(max(idx.values()), a_col, b_col):
                    continue

                gene_a = symbols.get(parts[a_col])
                gene_b = symbols.get(parts[b_col])
                if not gene_a or not gene_b or gene_a == gene_b:
                    continue
                if keep_symbols is not None and (
                    gene_a not in keep_symbols or gene_b not in keep_symbols
                ):
                    continue

                scores = {name: int(parts[col]) for name, col in idx.items()}
                best = max(scores.values())
                if best < min_score:
                    continue

                key = (gene_a, gene_b) if gene_a < gene_b else (gene_b, gene_a)
                existing = seen.get(key)
                if existing is None or best > existing.score:
                    seen[key] = PPIEdge(
                        gene_a=key[0],
                        gene_b=key[1],
                        score=best,
                        channel_scores=tuple(sorted(scores.items())),
                    )

        return sorted(seen.values(), key=lambda e: (-e.score, e.gene_a, e.gene_b))

    def channel_report(
        self, keep_symbols: Optional[Set[str]] = None, min_score: int = DEFAULT_MIN_SCORE
    ) -> Dict[str, int]:
        """
        Edge count each channel would contribute on its own.

        Exists so the textmining share is visible before anything is built on
        top of it, rather than discovered afterwards.
        """
        counts: Dict[str, int] = {}
        for channel in CHANNELS:
            try:
                counts[channel] = len(
                    self.edges(
                        keep_symbols=keep_symbols,
                        channels=(channel,),
                        min_score=min_score,
                        allow_literature_channels=True,
                    )
                )
            except Exception as e:  # pragma: no cover - reported, not raised
                self.logger.warning(f"channel {channel} failed: {e}")
        return counts
