"""
Linking text chunks to knowledge graph nodes.

This is the join between the two halves of the system. Chunks are where
entities are *mentioned*; the graph is where entities are *related*. Without a
link between them, retrieval can only return passages that look like the query
-- it cannot follow a relationship to find a passage that never mentions the
query at all.

Two indexes make that possible:

- EntityAliasIndex: every surface form a graph node is known by, so a mention
  in text resolves to a canonical node.
- ChunkGraphIndex: the bidirectional chunk <-> node mapping built on top of it.

The alias index is only as good as the graph's entity resolution: canonical
nodes that have absorbed their duplicates' names carry far richer alias sets,
so KnowledgeGraphBuilder.merge_duplicate_entities() directly improves linking
recall here.
"""

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..utils.logging import LoggerMixin


class EntityAliasIndex(LoggerMixin):
    """
    Resolves entity mentions in free text to canonical graph node ids.

    Matching is longest-alias-first so that "breast cancer 1" is preferred over
    a bare "cancer" occurring inside it, and is bounded by word boundaries so
    that "TP53" does not match inside "TP53BP1".
    """

    def __init__(self, min_alias_length: int = 3):
        """
        Args:
            min_alias_length: Aliases shorter than this are ignored. Two-letter
                surface forms produce far more false positives than signal.
        """
        self.min_alias_length = min_alias_length
        self.alias_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self._pattern: Optional[re.Pattern] = None

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize a surface form for matching."""
        return " ".join(str(text).lower().split())

    def add_entity(
        self,
        node_id: str,
        name: str,
        synonyms: Optional[Iterable[str]] = None
    ) -> None:
        """Register a node under its name and every synonym."""
        for surface in {name, *(synonyms or [])}:
            normalized = self._normalize(surface)
            if len(normalized) >= self.min_alias_length:
                self.alias_to_nodes[normalized].add(node_id)

        # Invalidate the compiled matcher
        self._pattern = None

    def add_from_graph(self, graph: Any) -> "EntityAliasIndex":
        """
        Build the index from a NetworkX graph.

        Node attributes ``name`` and ``synonyms`` are used when present; the
        node id itself is always registered as an alias.
        """
        for node_id, attributes in graph.nodes(data=True):
            self.add_entity(
                node_id=str(node_id),
                name=attributes.get("name", str(node_id)),
                synonyms=attributes.get("synonyms", []),
            )

        self.logger.info(
            f"Indexed {graph.number_of_nodes()} nodes under "
            f"{len(self.alias_to_nodes)} aliases"
        )
        return self

    def add_from_entities(self, entities: Iterable[Any]) -> "EntityAliasIndex":
        """Build the index from StandardizedEntity objects."""
        count = 0
        for entity in entities:
            self.add_entity(
                node_id=entity.id,
                name=entity.name,
                synonyms=getattr(entity, "synonyms", []),
            )
            count += 1

        self.logger.info(
            f"Indexed {count} entities under {len(self.alias_to_nodes)} aliases"
        )
        return self

    def _compiled(self) -> Optional[re.Pattern]:
        """Compile all aliases into one alternation, longest first."""
        if self._pattern is not None:
            return self._pattern
        if not self.alias_to_nodes:
            return None

        # Longest first so the most specific alias wins at a given position
        aliases = sorted(self.alias_to_nodes, key=len, reverse=True)
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )
        return self._pattern

    def find_in_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Find entity mentions in a piece of text.

        Returns:
            (node_id, matched_surface_form) pairs, deduplicated. A surface form
            shared by several nodes yields one pair per candidate node, since
            resolving that ambiguity needs context this index does not have.
        """
        pattern = self._compiled()
        if pattern is None or not text:
            return []

        found: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        for match in pattern.finditer(text):
            surface = self._normalize(match.group(1))
            for node_id in self.alias_to_nodes.get(surface, ()):
                key = (node_id, surface)
                if key not in seen:
                    seen.add(key)
                    found.append(key)

        return found


class ChunkGraphIndex(LoggerMixin):
    """
    Bidirectional index between text chunks and knowledge graph nodes.

    This is what makes graph-expanded retrieval possible: given a chunk, find
    the entities it mentions; given an entity, find every chunk discussing it.
    """

    def __init__(self, alias_index: Optional[EntityAliasIndex] = None):
        self.alias_index = alias_index or EntityAliasIndex()
        self.node_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.chunk_to_nodes: Dict[str, List[str]] = defaultdict(list)
        self.chunks: Dict[str, Any] = {}

    @staticmethod
    def _chunk_id(chunk: Any, position: int) -> str:
        """Derive a stable id for a chunk."""
        metadata = getattr(chunk, "metadata", {}) or {}
        if metadata.get("chunk_uid"):
            return str(metadata["chunk_uid"])

        source = metadata.get("pmid") or metadata.get("source") or "doc"
        return f"{source}:{metadata.get('chunk_id', position)}"

    def index_chunks(self, chunks: List[Any]) -> Dict[str, int]:
        """
        Annotate chunks with the entities they mention and build the index.

        Each chunk's ``metadata`` gains ``entity_ids`` and ``chunk_uid``, so the
        linkage survives a round trip through a vector store.

        Args:
            chunks: LangChain Documents.

        Returns:
            {"chunks", "linked_chunks", "nodes_covered", "total_mentions"}.
        """
        total_mentions = 0
        linked_chunks = 0

        for position, chunk in enumerate(chunks):
            chunk_uid = self._chunk_id(chunk, position)
            mentions = self.alias_index.find_in_text(chunk.page_content)
            node_ids = sorted({node_id for node_id, _ in mentions})

            chunk.metadata["chunk_uid"] = chunk_uid
            chunk.metadata["entity_ids"] = node_ids
            chunk.metadata["entity_surface_forms"] = sorted(
                {surface for _, surface in mentions}
            )

            self.chunks[chunk_uid] = chunk
            if node_ids:
                linked_chunks += 1
                total_mentions += len(mentions)
                self.chunk_to_nodes[chunk_uid] = node_ids
                for node_id in node_ids:
                    self.node_to_chunks[node_id].append(chunk_uid)

        stats = {
            "chunks": len(chunks),
            "linked_chunks": linked_chunks,
            "nodes_covered": len(self.node_to_chunks),
            "total_mentions": total_mentions,
        }
        self.logger.info(
            f"Linked {linked_chunks}/{len(chunks)} chunks to "
            f"{len(self.node_to_chunks)} graph nodes ({total_mentions} mentions)"
        )
        return stats

    def nodes_for_chunk(self, chunk_uid: str) -> List[str]:
        """Graph nodes mentioned by a chunk."""
        return list(self.chunk_to_nodes.get(chunk_uid, []))

    def chunks_for_node(self, node_id: str) -> List[Any]:
        """Chunks mentioning a graph node."""
        return [
            self.chunks[uid]
            for uid in self.node_to_chunks.get(node_id, [])
            if uid in self.chunks
        ]

    def neighbors(
        self,
        graph: Any,
        node_ids: Iterable[str],
        max_hops: int = 1
    ) -> Dict[str, int]:
        """
        Breadth-first walk from seed nodes.

        Args:
            graph: NetworkX graph to traverse.
            node_ids: Seed nodes.
            max_hops: How far to walk.

        Returns:
            Reached node id -> hop distance, excluding the seeds themselves.
        """
        seeds = {n for n in node_ids if n in graph}
        distances: Dict[str, int] = {}
        frontier = set(seeds)

        for hop in range(1, max_hops + 1):
            next_frontier: Set[str] = set()
            for node in frontier:
                # Works for directed graphs too: both directions are relevant
                # when asking "what is this entity connected to?"
                adjacent = set(graph.neighbors(node))
                if graph.is_directed():
                    adjacent |= set(graph.predecessors(node))

                for neighbor in adjacent:
                    if neighbor in seeds or neighbor in distances:
                        continue
                    distances[neighbor] = hop
                    next_frontier.add(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        return distances
