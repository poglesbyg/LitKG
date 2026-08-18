"""
Entity features from literature context rather than entity names.

Node-name embeddings gave +0.020 AUC but almost nothing on cold start (0.531
against a 0.498 floor), because a name carries no biology: "Imatinib" as a
string says nothing about what it treats. The knowledge is in the sentences the
entity appears in.

**This is the one feature source in the project that can leak.** A name is
static metadata -- a disease was called melanoma before and after 2016 -- but an
abstract encodes a discovery, and an abstract published in 2024 may state
outright the association a 2016 holdout is asking the model to predict. So
context is only ever gathered from documents published strictly before the
cutoff, the filter is applied at the PubMed query and re-checked per record,
and the cutoff is part of the cache key so a corpus built for one cutoff can
never be reused for a later one.

The repository's bundled corpus cannot support this: all 100 documents post-date
2016, and only 5.3% of KG nodes appear in them at all. Contexts are therefore
fetched on demand, with a date filter, and cached.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from litkg.utils.logging import LoggerMixin

# NCBI asks for <=3 requests/second without an API key.
_REQUEST_INTERVAL = 0.36

# Sentences per entity. Enough to characterise it; beyond this the mean
# embedding stops moving and fetching costs more.
MAX_CONTEXTS_PER_ENTITY = 12
MAX_ARTICLES_PER_ENTITY = 10


@dataclass
class ContextConfig:
    """Configuration for context gathering."""

    cutoff_year: int
    max_articles: int = MAX_ARTICLES_PER_ENTITY
    max_contexts: int = MAX_CONTEXTS_PER_ENTITY
    min_year: int = 1990
    cache_dir: Optional[Path] = None
    email: str = "litkg@example.org"
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.cache_dir is None:
            from litkg.utils.config import get_data_dir
            self.cache_dir = get_data_dir() / "processed" / "literature_context"


class LiteratureContextFetcher(LoggerMixin):
    """
    Gathers sentences mentioning an entity from pre-cutoff PubMed abstracts.

    Caches per cutoff year. The cache is keyed on the cutoff precisely so that
    raising the cutoff cannot silently reuse contexts gathered under a looser
    date filter.
    """

    def __init__(self, config: ContextConfig):
        self.config = config
        # {entity key: [abstract text]}. Sentences are extracted on read.
        self._abstracts: Dict[str, List[str]] = {}
        self._last_request = 0.0
        self._loaded = False

    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        # v2 stores retrieved abstracts rather than extracted sentences, so a
        # change to the matcher costs nothing instead of forcing a refetch of
        # every entity. Caching derived data was a mistake worth not repeating.
        return (
            Path(self.config.cache_dir)
            / f"abstracts_pre{self.config.cutoff_year}.json"
        )

    def load(self) -> None:
        if self._loaded:
            return
        path = self._cache_path()
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                if payload.get("cutoff_year") != self.config.cutoff_year:
                    raise ValueError("cutoff mismatch")
                self._abstracts = payload.get("abstracts", {})
                self.logger.info(
                    f"Loaded abstracts for {len(self._abstracts)} entities "
                    f"(pre-{self.config.cutoff_year})"
                )
            except Exception as e:
                self.logger.warning(f"Ignoring unusable context cache: {e}")
                self._abstracts = {}
        self._loaded = True

    def save(self) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "cutoff_year": self.config.cutoff_year,
            "abstracts": self._abstracts,
        }))

    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    def _entrez(self):
        from Bio import Entrez

        Entrez.email = self.config.email
        if self.config.api_key:
            Entrez.api_key = self.config.api_key
        return Entrez

    @staticmethod
    def _query_term(name: str) -> str:
        """
        A PubMed-safe query for an entity name.

        CIVIC names contain characters E-utilities rejects outright -- fusion
        notation ("BCR::ABL"), brackets, and quotes -- which come back as HTTP
        400 rather than an empty result.
        """
        cleaned = re.sub(r'[^\w\s\-+.]', " ", str(name))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return f'"{cleaned}"' if cleaned else ""

    def _search(self, term: str, attempts: int = 3) -> List[str]:
        """PMIDs for a term, restricted to before the cutoff at query time."""
        for attempt in range(attempts):
            try:
                return self._search_once(term)
            except Exception as e:
                if attempt == attempts - 1:
                    raise
                # E-utilities returns transient 400s and 429s under load.
                self.logger.debug(f"Retrying search for {term!r} after {e}")
                time.sleep(1.5 * (attempt + 1))
        return []

    def _search_once(self, term: str) -> List[str]:
        Entrez = self._entrez()
        self._throttle()
        handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retmax=self.config.max_articles,
            datetype="pdat",
            mindate=str(self.config.min_year),
            # maxdate is inclusive, so the last admissible year is cutoff - 1.
            maxdate=str(self.config.cutoff_year - 1),
            sort="relevance",
        )
        try:
            return list(Entrez.read(handle).get("IdList", []))
        finally:
            handle.close()

    def _fetch_abstracts(self, pmids: Sequence[str]) -> List[Dict[str, str]]:
        """Abstracts for PMIDs, each re-checked against the cutoff."""
        if not pmids:
            return []
        Entrez = self._entrez()
        self._throttle()
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
        try:
            records = Entrez.read(handle)
        finally:
            handle.close()

        articles = []
        for article in records.get("PubmedArticle", []):
            citation = article.get("MedlineCitation", {})
            info = citation.get("Article", {})
            year = _publication_year(info)
            # Second line of defence: the query filter is trusted but verified,
            # because a record with a missing or malformed date must be dropped
            # rather than assumed to be in range.
            if year is None or year >= self.config.cutoff_year:
                continue
            abstract = info.get("Abstract", {}).get("AbstractText", [])
            text = " ".join(str(part) for part in abstract).strip()
            if text:
                articles.append({"text": f"{info.get('ArticleTitle', '')} {text}",
                                 "year": str(year)})
        return articles

    # ------------------------------------------------------------------

    def contexts_for(self, name: str) -> List[str]:
        """Sentences mentioning `name`, fetching its abstracts if not cached."""
        self.load()
        key = name.strip().lower()
        if key not in self._abstracts:
            term = self._query_term(name)
            texts: List[str] = []
            if term:
                try:
                    pmids = self._search(term)
                    texts = [a["text"] for a in self._fetch_abstracts(pmids)]
                except Exception as e:
                    self.logger.warning(f"Context fetch failed for {name!r}: {e}")
            self._abstracts[key] = texts

        sentences: List[str] = []
        for text in self._abstracts[key]:
            sentences.extend(_sentences_mentioning(text, name))
            if len(sentences) >= self.config.max_contexts:
                break
        return sentences[: self.config.max_contexts]

    def gather(self, names: Iterable[str], save_every: int = 25) -> Dict[str, List[str]]:
        """Fetch contexts for many names, saving periodically so runs resume."""
        self.load()
        pending = [n for n in dict.fromkeys(names) if n.strip().lower() not in self._abstracts]
        if pending:
            self.logger.info(
                f"Fetching pre-{self.config.cutoff_year} contexts for "
                f"{len(pending)} entities"
            )
        for index, name in enumerate(pending, start=1):
            self.contexts_for(name)
            if index % save_every == 0:
                self.save()
                self.logger.info(f"  {index}/{len(pending)}")
        self.save()
        return self._contexts

    def context_text(self, node_names: Dict[str, str]) -> Dict[str, str]:
        """
        One context string per node, for the existing text encoder.

        The entity name is prepended so a node with no retrieved context
        degrades to the name-only feature rather than to an empty string.
        """
        self.load()
        text: Dict[str, str] = {}
        for node_id, name in node_names.items():
            sentences = self.contexts_for_cached(name)
            text[node_id] = " ".join([name] + sentences) if sentences else name
        return text

    def contexts_for_cached(self, name: str) -> List[str]:
        """Sentences from already-cached abstracts only; never fetches."""
        self.load()
        sentences: List[str] = []
        for text in self._abstracts.get(name.strip().lower(), []):
            sentences.extend(_sentences_mentioning(text, name))
            if len(sentences) >= self.config.max_contexts:
                break
        return sentences[: self.config.max_contexts]

    def coverage(self, node_names: Dict[str, str]) -> float:
        """Fraction of nodes with at least one retrieved context sentence."""
        self.load()
        if not node_names:
            return 0.0
        covered = sum(
            1 for name in node_names.values() if self.contexts_for_cached(name)
        )
        return covered / len(node_names)


def _publication_year(article_info: Dict) -> Optional[int]:
    """Publication year from an Entrez article record, or None."""
    for path in (("Journal", "JournalIssue", "PubDate", "Year"),):
        node = article_info
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if node:
            match = re.search(r"(19|20)\d{2}", str(node))
            if match:
                return int(match.group())
    # Some records carry only a MedlineDate string ("2011 Nov-Dec").
    date = article_info.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    match = re.search(r"(19|20)\d{2}", str(date))
    return int(match.group()) if match else None


def _mention_patterns(name: str) -> List[re.Pattern]:
    """
    Patterns that count as a mention of `name`.

    Requiring the literal gene-qualified string finds almost nothing: the graph
    calls a variant "FLT3 ITD" while abstracts write "FLT3-ITD", and most write
    "V600E" without repeating the gene. Multi-word names hit 13% under exact
    matching against 62% for single-word names.

    Two relaxations, both safe because retrieval already scoped the abstracts to
    this entity: separators between tokens are flexible, and the distinctive
    trailing token ("V600E", "ITD") counts on its own.
    """
    tokens = [t for t in re.split(r"[\s\-_:]+", str(name).strip()) if t]
    if not tokens:
        return []

    patterns = [
        re.compile(
            r"\b" + r"[\s\-_:]*".join(re.escape(t) for t in tokens) + r"\b",
            re.IGNORECASE,
        )
    ]
    if len(tokens) > 1:
        # The specifier, not the gene: matching the gene alone would pull in
        # sentences about the gene that say nothing about this variant.
        specifier = tokens[-1]
        if len(specifier) >= 2:
            patterns.append(re.compile(rf"\b{re.escape(specifier)}\b", re.IGNORECASE))
    return patterns


def _sentences_mentioning(text: str, name: str) -> List[str]:
    """Sentences from `text` that mention `name` under any accepted form."""
    patterns = _mention_patterns(name)
    if not patterns:
        return []
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if 20 < len(sentence) < 600 and any(p.search(sentence) for p in patterns):
            found.append(sentence)
    return found
