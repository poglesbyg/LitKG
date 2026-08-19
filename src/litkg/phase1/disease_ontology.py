"""
Disease Ontology lookup, for joining cohort names onto CIVIC disease nodes.

CIVIC keys its diseases by DOID. The GDC does not: it names a cohort "Breast
Invasive Carcinoma" where CIVIC has "Breast Cancer". Matching those by string
gets 14 of 33 TCGA cohorts and silently loses the rest.

Resolving each cohort name to a DOID first turns that into an identifier join.
The Disease Ontology carries a synonym list per term, which is what closes most
of the gap -- "Head and Neck Squamous Cell Carcinoma" is a synonym of
DOID:5520, not its primary label. Ancestry closes a little more: a cohort whose
DOID is absent from CIVIC may still have a parent that is present, so a
lung-adenocarcinoma cohort can inform a lung-cancer node.

An ancestry match is a *generalisation* and is reported as such. Scoring a
specific cohort against a broader disease is defensible; pretending the two are
the same term is not.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from litkg.utils.logging import LoggerMixin

DO_OBO_URL = "https://purl.obolibrary.org/obo/doid.obo"


def normalise(text: str) -> str:
    """Fold a disease label to a comparable key."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


@dataclass
class DiseaseMatch:
    """One cohort resolved onto a CIVIC disease node."""

    doid: str
    civic_id: str
    # True when the match went through a parent term rather than the cohort's
    # own DOID, so the association is being generalised.
    via_ancestor: bool
    steps: int = 0


class DiseaseOntology(LoggerMixin):
    """Names and synonyms to DOIDs, plus the is_a hierarchy."""

    MAX_ANCESTRY_STEPS = 6

    def __init__(self, obo_path: Path):
        self.obo_path = Path(obo_path)
        self.names: Dict[str, str] = {}
        self.labels: Dict[str, str] = {}
        self.index: Dict[str, str] = {}
        self.parents: Dict[str, List[str]] = defaultdict(list)
        self._parse()

    @classmethod
    def download(cls, path: Path, timeout: int = 300) -> Path:
        """Fetch the OBO release if it is not already on disk."""
        path = Path(path)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DO_OBO_URL, timeout=timeout)
        response.raise_for_status()
        path.write_bytes(response.content)
        return path

    def _parse(self) -> None:
        if not self.obo_path.exists():
            raise FileNotFoundError(
                f"Disease Ontology not found at {self.obo_path}. "
                f"Call DiseaseOntology.download() first."
            )

        current: Optional[dict] = None

        def flush(term: Optional[dict]) -> None:
            if not term or not term.get("id"):
                return
            doid = term["id"]
            self.labels[doid] = term.get("name") or doid
            self.parents[doid] = term.get("is_a", [])
            for surface in [term.get("name")] + term.get("syn", []):
                if surface:
                    # First writer wins: primary labels are parsed alongside
                    # synonyms, and a synonym must not displace another term's
                    # own name.
                    self.index.setdefault(normalise(surface), doid)

        for raw in self.obo_path.read_text(errors="replace").splitlines():
            line = raw.rstrip()
            if line == "[Term]":
                flush(current)
                current = {"id": None, "name": None, "syn": [], "is_a": []}
            elif current is None:
                continue
            elif line.startswith("id: DOID:"):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("synonym: "):
                match = re.match(r'synonym: "([^"]+)"', line)
                if match:
                    current["syn"].append(match.group(1))
            elif line.startswith("is_a: DOID:"):
                current["is_a"].append(line[6:].split("!")[0].strip())
            elif line.startswith("[") and line != "[Term]":
                flush(current)
                current = None

        flush(current)
        self.logger.info(
            f"Disease Ontology: {len(self.labels)} terms, "
            f"{len(self.index)} indexed surface forms"
        )

    def doid_for(self, name: str) -> Optional[str]:
        """The DOID whose name or synonym matches, or None."""
        return self.index.get(normalise(name))

    def ancestors(self, doid: str) -> List[Tuple[str, int]]:
        """Ancestor DOIDs with their distance, nearest first."""
        seen = {doid}
        out: List[Tuple[str, int]] = []
        frontier = [doid]
        for step in range(1, self.MAX_ANCESTRY_STEPS + 1):
            nxt: List[str] = []
            for node in frontier:
                for parent in self.parents.get(node, []):
                    if parent not in seen:
                        seen.add(parent)
                        out.append((parent, step))
                        nxt.append(parent)
            if not nxt:
                break
            frontier = nxt
        return out

    def match_to_civic(
        self,
        cohort_name: str,
        civic_by_doid: Dict[str, str],
        allow_ancestors: bool = True,
    ) -> Optional[DiseaseMatch]:
        """
        Resolve a cohort name onto a CIVIC disease node id.

        Tries the cohort's own DOID first, then its ancestors nearest-first, so
        a generalisation is only used when nothing more specific exists.
        """
        doid = self.doid_for(cohort_name)
        if doid is None:
            return None

        civic_id = civic_by_doid.get(doid)
        if civic_id is not None:
            return DiseaseMatch(doid=doid, civic_id=civic_id, via_ancestor=False)

        if not allow_ancestors:
            return None

        for parent, step in self.ancestors(doid):
            civic_id = civic_by_doid.get(parent)
            if civic_id is not None:
                return DiseaseMatch(
                    doid=parent, civic_id=civic_id, via_ancestor=True, steps=step
                )
        return None
