"""
Text features for graph nodes.

Every predictor so far has used topology alone. That caps what is reachable:
14% of held-out pairs have no path between their endpoints at any length, and
no amount of path counting scores those. Node text is the only signal that
does.

Names are static metadata, not observations, so using them does not leak
across the temporal split -- a disease was called "melanoma" before and after
2016. One caveat is measured rather than assumed: some CIVIC therapy names
embed their target ("BRAF Inhibitor"), which makes certain variant-therapy
pairs guessable from the strings alone. `FeatureOnlyPredictor` exists to
quantify how much of any gain is that effect rather than transfer.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from litkg.utils.logging import LoggerMixin

# Chosen by measurement, not reputation. On name similarity alone against the
# 2016 holdout: PubMedBERT 0.580 [0.564, 0.595], MiniLM 0.533 [0.516, 0.550],
# BioBERT 0.514 [0.497, 0.530] against a random floor of 0.498. PubMedBERT and
# MiniLM have disjoint intervals; BioBERT is barely distinguishable from chance
# despite also being a biomedical model, so "biomedical" alone does not predict
# which encoder helps.
#
# PubMedBERT is a masked-LM checkpoint with no pooling head, so
# sentence-transformers wraps it with mean pooling and warns. For short entity
# names that is an acceptable representation, and it measures better here than
# the properly sentence-tuned general model.
DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
ALTERNATIVE_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "dmis-lab/biobert-base-cased-v1.1",
)


@dataclass
class FeatureConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 128
    normalize: bool = True
    # None means "the default location". Set use_cache=False to disable caching
    # entirely -- an explicit flag, because overloading cache_dir=None to mean
    # both "default" and "off" is how a stub encoder ends up reading real
    # 768-dim vectors and stacking them with its own.
    cache_dir: Optional[Path] = None
    use_cache: bool = True

    def __post_init__(self):
        if self.use_cache and self.cache_dir is None:
            from litkg.utils.config import get_data_dir
            self.cache_dir = get_data_dir() / "processed" / "text_features"


class NodeTextEncoder(LoggerMixin):
    """
    Encodes node display names into vectors, cached on disk.

    Encoding a few thousand short strings is quick, but it is pure overhead to
    repeat on every evaluation run, and the cache key includes the model so
    switching encoders does not silently reuse the wrong vectors.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        if not self.config.use_cache or self.config.cache_dir is None:
            return None
        digest = hashlib.sha256(self.config.model_name.encode()).hexdigest()[:12]
        return Path(self.config.cache_dir) / f"node_text_{digest}.npz"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if path and path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    keys = json.loads(str(data["keys"].item()))
                    vectors = data["vectors"]
                self._cache = {k: vectors[i] for i, k in enumerate(keys)}
                self.logger.info(f"Loaded {len(self._cache)} cached text vectors")
                self._cached_width = vectors.shape[1] if vectors.ndim > 1 else None
            except Exception as e:
                self.logger.warning(f"Ignoring unreadable feature cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        path = self._cache_path()
        if not path or not self._cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        np.savez_compressed(
            path,
            keys=np.array(json.dumps(keys)),
            vectors=np.stack([self._cache[k] for k in keys]),
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self.logger.info(f"Loading text encoder {self.config.model_name}")
        self._model = SentenceTransformer(self.config.model_name)
        return self._model

    # ------------------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts, reusing cached vectors where possible."""
        self._load_cache()
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]

        if missing:
            model = self._load_model()
            vectors = model.encode(
                missing,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
            )
            width = vectors.shape[1] if vectors.ndim > 1 else None
            cached_width = getattr(self, "_cached_width", None)
            if cached_width is not None and width is not None and width != cached_width:
                # A cache written by a different encoder under the same key.
                # Stacking mixed widths fails with an unhelpful numpy error, so
                # drop the stale entries instead.
                self.logger.warning(
                    f"Cache holds {cached_width}-dim vectors but the encoder "
                    f"produces {width}; discarding the cache"
                )
                self._cache = {}
            self._cached_width = width
            for text, vector in zip(missing, vectors):
                self._cache[text] = vector.astype(np.float32)
            self._save_cache()

        return np.stack([self._cache[t] for t in texts])

    def encode_nodes(self, node_text: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Encode a {node id: display name} mapping."""
        if not node_text:
            return {}
        nodes = list(node_text)
        vectors = self.encode([node_text[n] for n in nodes])
        return dict(zip(nodes, vectors))


def build_node_text(entities: Sequence) -> Dict[str, str]:
    """
    Display name per node id, from StandardizedEntity records.

    Variant names are qualified with their gene ("BRAF V600E" rather than
    "V600E"), since an alteration string alone carries almost no meaning and
    collides across genes -- several genes have an "Amplification" variant.
    """
    text: Dict[str, str] = {}
    for entity in entities:
        name = str(getattr(entity, "name", "") or "").strip()
        if not name:
            continue
        attributes = getattr(entity, "attributes", None) or {}
        gene = str(attributes.get("gene", "") or "").strip()
        if gene and gene.lower() != "nan" and not name.upper().startswith(gene.upper()):
            name = f"{gene} {name}"
        text[entity.id] = name
    return text


class FeatureOnlyPredictor(LoggerMixin):
    """
    Cosine similarity of node names. Topology is ignored entirely.

    Its purpose is diagnostic, not competitive. Some CIVIC therapies are named
    for their target ("BRAF Inhibitor" scores 0.65 against "BRAF"), so a model
    given text features can score certain pairs from the strings alone. That is
    legitimate -- the names were available before the cutoff -- but it is
    string matching, not learned biology, and a gain that this predictor also
    captures should not be credited to representation learning.

    It is also the only predictor here that can score a pair with no path
    between its endpoints, which is 14% of the held-out set.
    """

    name = "text_only"

    def __init__(
        self,
        node_text: Optional[Dict[str, str]] = None,
        encoder: Optional[NodeTextEncoder] = None,
    ):
        self.node_text = node_text or {}
        self.encoder = encoder or NodeTextEncoder()
        self.vectors: Dict[str, np.ndarray] = {}

    def fit(self, graph=None) -> "FeatureOnlyPredictor":
        self.vectors = self.encoder.encode_nodes(self.node_text)
        return self

    def score(self, u: str, v: str) -> float:
        a, b = self.vectors.get(u), self.vectors.get(v)
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    def score_pairs(self, pairs: Sequence) -> List[float]:
        return [self.score(u, v) for u, v in pairs]
