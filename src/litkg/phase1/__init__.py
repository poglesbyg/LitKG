"""Phase 1: Foundation - Literature processing, KG preprocessing, and entity linking."""

from .literature_processor import LiteratureProcessor, ProcessedDocument, Entity, Relation
from .kg_preprocessor import KnowledgeGraphPreprocessor as KGPreprocessor, StandardizedEntity, StandardizedRelation
from .entity_linker import EntityLinker, EntityMatch, LinkingResult

__all__ = [
    "LiteratureProcessor",
    "ProcessedDocument", 
    "Entity",
    "Relation",
    "KGPreprocessor",
    "StandardizedEntity",
    "StandardizedRelation", 
    "EntityLinker",
    "EntityMatch",
    "LinkingResult",
]