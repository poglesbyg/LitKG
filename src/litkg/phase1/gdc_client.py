"""
GDC (Genomic Data Commons) API client.

Fetches TCGA somatic mutation data from https://api.gdc.cancer.gov, which
serves open-access data with no authentication. Controlled-access data
(sequence-level files, requiring dbGaP authorization) is deliberately out of
scope: everything here comes from the open tier.

The unit this pulls is the gene-cancer type association -- which genes are
recurrently mutated in which TCGA cohort -- because that is the shape the rest
of the graph is in. Raw mutation counts are NOT that association, and the
distinction is the whole reason this module is careful.

Why the Cancer Gene Census restriction matters
----------------------------------------------
Ranking genes by raw somatic mutation count measures coding length, not
biology. Asked for the top mutated genes in TCGA-BRCA, the GDC's own endpoint
returns OBSCN, PIK3C2B, NID1, NFASC and USH2A -- long genes that accumulate
passenger mutations in proportion to how much sequence they present. None is a
breast cancer driver.

Restricting to genes flagged `is_cancer_gene_census` (the COSMIC Cancer Gene
Census, 716 genes, shipped as a field by the GDC itself) recovers the real
signal: PIK3CA, TP53, CDH1, MAP2K4, BRCA1. Total occurrences drop from 88,359
to 7,296 for that cohort -- the 92% removed is the passenger load.

This is a curated-driver filter, not a statistical correction. It cannot find a
driver the Census has not recorded, so the resulting edges are a
high-precision, low-recall view. `canonical_transcript_length_cds` is carried on
every gene node so the length confound stays checkable downstream rather than
being taken on trust.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

from litkg.utils.logging import LoggerMixin

# The GDC caps facet aggregations at this many buckets and says nothing when it
# truncates. Every faceted query here is checked against it, because a silently
# short list would look exactly like a gene that is simply rare.
FACET_BUCKET_CAP = 200


@dataclass
class GDCGene:
    """A Cancer Gene Census gene as the GDC reports it."""

    symbol: str
    gene_id: str
    cds_length: Optional[int] = None
    biotype: Optional[str] = None


@dataclass
class GDCProject:
    """A TCGA project (one cohort, one cancer type)."""

    project_id: str
    name: str
    primary_site: List[str] = field(default_factory=list)
    disease_type: List[str] = field(default_factory=list)
    case_count: int = 0


class GDCTruncationError(RuntimeError):
    """A faceted response hit the bucket cap, so the result is incomplete."""


class GDCClient(LoggerMixin):
    """
    Read-only client for the GDC open-access API.

    Responses are cached on disk under a release-stamped directory. The GDC
    publishes dated data releases and the pin exists for the same reason the
    CIVIC one does: so a change in a downstream number is a change in the code,
    not an unnoticed upstream refresh.
    """

    API_ROOT = "https://api.gdc.cancer.gov"
    DEFAULT_RELEASE = "46.0"
    DEFAULT_PROGRAM = "TCGA"

    def __init__(
        self,
        cache_dir: Path,
        release: Optional[str] = None,
        program: Optional[str] = None,
        timeout: int = 120,
        session: Optional[requests.Session] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self._release = release
        self.program = program or self.DEFAULT_PROGRAM
        self.timeout = timeout
        self.session = session or requests.Session()

    @classmethod
    def pinned_release(cls) -> str:
        """The release to record; env var wins over the pinned default."""
        return os.environ.get("LITKG_GDC_RELEASE", cls.DEFAULT_RELEASE).strip()

    @property
    def release(self) -> str:
        return self._release or self.pinned_release()

    # ---------------------------------------------------------------- requests

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.API_ROOT}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        # The GDC answers a bad field name with HTTP 200 and a warning rather
        # than an error, so an unnoticed warning here means querying something
        # that does not exist and reading the empty result as "no data".
        warnings = payload.get("warnings") or {}
        if warnings:
            raise ValueError(f"GDC rejected part of the query for {endpoint}: {warnings}")

        # /status answers with a flat body; every other endpoint nests under "data".
        return payload["data"] if "data" in payload else payload

    def _facet_buckets(
        self, endpoint: str, filters: Dict[str, Any], facet: str
    ) -> Dict[str, int]:
        """Faceted counts, refusing to return a list the API silently cut short."""
        data = self._get(
            endpoint,
            {
                "filters": json.dumps(filters),
                "facets": facet,
                "size": 0,
                "format": "json",
            },
        )
        buckets = data["aggregations"][facet]["buckets"]
        if len(buckets) >= FACET_BUCKET_CAP:
            raise GDCTruncationError(
                f"{endpoint} facet '{facet}' returned {len(buckets)} buckets, at or "
                f"above the GDC cap of {FACET_BUCKET_CAP}. The result is truncated "
                f"and would undercount silently. Narrow the query instead."
            )
        return {b["key"]: int(b["doc_count"]) for b in buckets if b["key"] != "_missing"}

    # ------------------------------------------------------------------ caching

    def _cache_path(self, name: str) -> Path:
        return (
            self.cache_dir
            / f"release-{self.release}"
            / self.program.lower()
            / f"{name}.json"
        )

    def _cached(self, name: str, build, shared: bool = False) -> Any:
        path = (
            self.cache_dir / f"release-{self.release}" / f"{name}.json"
            if shared
            else self._cache_path(name)
        )
        if path.exists():
            self.logger.info(f"Using cached GDC {name} for release {self.release}")
            return json.loads(path.read_text())

        value = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True))
        self.logger.info(f"Cached GDC {name} to {path}")
        return value

    # -------------------------------------------------------------- public API

    def live_release(self) -> str:
        """The release the API is currently serving, for comparison with the pin."""
        data = self._get("status", {})
        version = data.get("data_release_version") or {}
        if version:
            return f"{version.get('major')}.{version.get('minor')}"
        return str(data.get("data_release", "unknown"))

    def projects(self) -> List[GDCProject]:
        """Every project in the program, one per cancer cohort."""

        def build() -> List[Dict[str, Any]]:
            data = self._get(
                "projects",
                {
                    "filters": json.dumps(
                        {
                            "op": "in",
                            "content": {"field": "program.name", "value": [self.program]},
                        }
                    ),
                    "fields": "project_id,name,primary_site,disease_type,summary.case_count",
                    "size": 500,
                    "format": "json",
                },
            )
            return data["hits"]

        hits = self._cached("projects", build)
        return [
            GDCProject(
                project_id=h["project_id"],
                name=h.get("name") or h["project_id"],
                primary_site=list(h.get("primary_site") or []),
                disease_type=list(h.get("disease_type") or []),
                case_count=int((h.get("summary") or {}).get("case_count") or 0),
            )
            for h in hits
        ]

    def census_genes(self) -> List[GDCGene]:
        """The Cancer Gene Census genes, with CDS length for length-bias checks."""

        def build() -> List[Dict[str, Any]]:
            data = self._get(
                "genes",
                {
                    "filters": json.dumps(
                        {
                            "op": "=",
                            "content": {"field": "is_cancer_gene_census", "value": ["true"]},
                        }
                    ),
                    "fields": "symbol,gene_id,canonical_transcript_length_cds,biotype",
                    "size": 5000,
                    "format": "json",
                },
            )
            return data["hits"]

        hits = self._cached("census_genes", build, shared=True)
        genes = [
            GDCGene(
                symbol=h["symbol"],
                gene_id=h.get("gene_id", ""),
                cds_length=h.get("canonical_transcript_length_cds"),
                biotype=h.get("biotype"),
            )
            for h in hits
            if h.get("symbol")
        ]
        if not genes:
            raise ValueError("GDC returned no Cancer Gene Census genes")
        return genes

    def mutations_by_project(
        self, symbols: Iterable[str], pause: float = 0.0
    ) -> Dict[str, Dict[str, int]]:
        """
        Somatic mutation occurrences per gene, broken down by project.

        The loop runs gene-by-gene rather than project-by-project on purpose.
        Faceting a project's mutations by gene returns more distinct genes than
        the GDC's 200-bucket cap allows and comes back truncated; faceting a
        gene's mutations by project returns at most one bucket per cohort, which
        is far below the cap.
        """
        symbols = list(symbols)

        def build() -> Dict[str, Dict[str, int]]:
            counts: Dict[str, Dict[str, int]] = {}
            for symbol in tqdm(symbols, desc="GDC mutations per census gene"):
                filters = {
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "case.project.program.name",
                                "value": [self.program],
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "ssm.consequence.transcript.gene.symbol",
                                "value": [symbol],
                            },
                        },
                    ],
                }
                try:
                    buckets = self._facet_buckets(
                        "ssm_occurrences", filters, "case.project.project_id"
                    )
                except GDCTruncationError:
                    raise
                except Exception as e:
                    self.logger.warning(f"GDC query failed for {symbol}: {e}")
                    continue

                if buckets:
                    counts[symbol] = buckets
                if pause:
                    time.sleep(pause)
            return counts

        return self._cached("mutations_by_project", build)

    def variants_by_project(
        self,
        pairs: Iterable[Tuple[str, str]],
        pause: float = 0.0,
    ) -> Dict[str, Dict[str, int]]:
        """
        Occurrences of a specific protein change, broken down by project.

        Keyed "SYMBOL p.CHANGE" (e.g. "BRAF p.V600E"). The gene-level view is
        too coarse to say anything: BRAF is mutated across most cohorts, while
        BRAF V600E concentrates in thyroid (283) and melanoma (200). CIVIC's
        unit is the variant, so this is the granularity the two sources
        actually share.
        """
        pairs = sorted({(str(g).strip(), str(a).strip()) for g, a in pairs})

        def build() -> Dict[str, Dict[str, int]]:
            counts: Dict[str, Dict[str, int]] = {}
            for symbol, aa_change in tqdm(pairs, desc="GDC occurrences per variant"):
                filters = {
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "case.project.program.name",
                                "value": [self.program],
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "ssm.consequence.transcript.gene.symbol",
                                "value": [symbol],
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "ssm.consequence.transcript.aa_change",
                                "value": [aa_change],
                            },
                        },
                    ],
                }
                try:
                    buckets = self._facet_buckets(
                        "ssm_occurrences", filters, "case.project.project_id"
                    )
                except GDCTruncationError:
                    raise
                except Exception as e:
                    self.logger.warning(
                        f"GDC query failed for {symbol} {aa_change}: {e}"
                    )
                    continue

                if buckets:
                    counts[f"{symbol} p.{aa_change}"] = buckets
                if pause:
                    time.sleep(pause)
            return counts

        return self._cached("variants_by_project", build)


@dataclass
class GeneCohortAssociation:
    """One gene recurrently mutated in one cohort, above the background rate."""

    symbol: str
    project_id: str
    occurrences: int
    cohort_cases: int
    expected: float
    enrichment: float

    @property
    def cohort_fraction(self) -> float:
        return self.occurrences / self.cohort_cases if self.cohort_cases else 0.0


def background_rate_enrichment(
    mutations: Dict[str, Dict[str, int]],
    cds_lengths: Dict[str, Optional[int]],
    cohort_cases: Dict[str, int],
) -> List[GeneCohortAssociation]:
    """
    Score gene-cohort pairs against a length-aware background mutation rate.

    Somatic mutation counts scale with coding length: a long gene collects more
    passengers than a short one for reasons that have nothing to do with the
    cancer. Restricting to Cancer Gene Census genes is not enough on its own --
    among Census genes alone, the number of cohorts a gene reaches correlates
    with log CDS length at +0.798, which is *higher* than the +0.542 for raw
    occurrence counts, because thresholding on a raw count selects for length.

    So each pair is scored against what its length predicts. Within a cohort,
    the expected share of mutations for a gene is its share of the total coding
    sequence, and enrichment is observed over expected. That takes the length
    correlation to -0.097 and leaves the known drivers on top: IDH1 at 193x in
    lower-grade glioma, KRAS at 43x in lung adenocarcinoma, GATA3 at 26x in
    breast.

    This is a crude background model. It assumes a uniform per-base mutation
    rate within a cohort, which real tumours violate -- rates vary with
    replication timing, chromatin state and mutational signature. It is enough
    to remove the length confound, not enough to call a gene a driver.
    """
    usable = {s for s, v in cds_lengths.items() if v}
    if not usable:
        return []

    totals: Dict[str, int] = {}
    for symbol, by_project in mutations.items():
        if symbol not in usable:
            continue
        for project_id, occurrences in by_project.items():
            totals[project_id] = totals.get(project_id, 0) + occurrences

    total_cds = sum(cds_lengths[s] for s in usable)
    if total_cds <= 0:
        return []

    associations: List[GeneCohortAssociation] = []
    for symbol, by_project in mutations.items():
        if symbol not in usable:
            continue
        for project_id, occurrences in by_project.items():
            cases = cohort_cases.get(project_id, 0)
            cohort_total = totals.get(project_id, 0)
            if cases <= 0 or cohort_total <= 0:
                continue

            expected = cohort_total * cds_lengths[symbol] / total_cds
            if expected <= 0:
                continue

            associations.append(
                GeneCohortAssociation(
                    symbol=symbol,
                    project_id=project_id,
                    occurrences=occurrences,
                    cohort_cases=cases,
                    expected=expected,
                    enrichment=occurrences / expected,
                )
            )
    return associations
