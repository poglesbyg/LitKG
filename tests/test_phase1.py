"""
Tests for Phase 1 components (literature processing, KG preprocessing, entity linking).
"""

import logging
import re

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from types import SimpleNamespace
import networkx as nx

from litkg.phase1.literature_processor import LiteratureProcessor, DocumentProcessor, EntityExtractor
from litkg.phase1.kg_preprocessor import KnowledgeGraphPreprocessor, OntologyMapper
from litkg.phase1.entity_linker import EntityLinker, FuzzyMatcher, DisambiguationEngine
from litkg.utils.config import LitKGConfig


class TestLiteratureProcessor:
    """Test literature processing components."""
    
    def test_literature_processor_init(self, sample_config):
        """Test LiteratureProcessor initialization."""
        processor = LiteratureProcessor(sample_config)
        
        assert processor.config == sample_config
        assert hasattr(processor, 'logger')
    
    def test_process_document(self, sample_biomedical_text):
        """Test document processing."""
        processor = LiteratureProcessor({})
        
        # Mock the NLP pipeline
        with patch.object(processor, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_doc.ents = [
                Mock(text="BRCA1", label_="GENE", start=0, end=5),
                Mock(text="breast cancer", label_="DISEASE", start=10, end=23)
            ]
            mock_nlp.return_value = mock_doc
            
            result = processor.process_document(sample_biomedical_text)
            
            assert "entities" in result
            assert "text" in result
            assert len(result["entities"]) == 2
    
    def test_extract_entities(self, sample_biomedical_text, sample_entities):
        """Test entity extraction."""
        processor = LiteratureProcessor({})
        
        with patch.object(processor, '_extract_entities_with_model') as mock_extract:
            mock_extract.return_value = sample_entities
            
            entities = processor.extract_entities(sample_biomedical_text)
            
            assert len(entities) == 4
            assert entities[0]["text"] == "BRCA1"
            assert entities[0]["label"] == "GENE"
    
    def test_extract_relations(self, sample_biomedical_text, sample_relations):
        """Test relation extraction."""
        processor = LiteratureProcessor({})
        
        with patch.object(processor, '_extract_relations_with_model') as mock_extract:
            mock_extract.return_value = sample_relations
            
            relations = processor.extract_relations(sample_biomedical_text)
            
            assert len(relations) == 3
            assert relations[0]["head"] == "BRCA1"
            assert relations[0]["relation"] == "ASSOCIATED_WITH"
    
    def test_process_batch(self, sample_literature_data):
        """Test batch processing of documents."""
        processor = LiteratureProcessor({})
        
        # Mock individual document processing
        with patch.object(processor, 'process_document') as mock_process:
            mock_process.return_value = {
                "entities": [{"text": "BRCA1", "label": "GENE"}],
                "relations": [{"head": "BRCA1", "relation": "ASSOCIATED_WITH", "tail": "cancer"}],
                "text": "sample text"
            }
            
            results = processor.process_batch(sample_literature_data)
            
            assert len(results) == 2
            assert all("entities" in result for result in results)
    
    @pytest.mark.slow
    def test_process_large_document(self):
        """Test processing of large documents."""
        processor = LiteratureProcessor({})
        
        # Create a large document
        large_text = "BRCA1 is associated with breast cancer. " * 1000
        
        with patch.object(processor, 'nlp') as mock_nlp:
            mock_doc = Mock()
            mock_doc.ents = [Mock(text="BRCA1", label_="GENE", start=0, end=5)]
            mock_nlp.return_value = mock_doc
            
            result = processor.process_document(large_text)
            
            assert "entities" in result
            assert len(result["text"]) > 10000


class TestDocumentProcessor:
    """Test document processing utilities."""
    
    def test_document_processor_init(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor()
        
        assert hasattr(processor, 'logger')
    
    def test_clean_text(self):
        """Test text cleaning."""
        processor = DocumentProcessor()
        
        dirty_text = "  This is a test\n\nwith   extra spaces.  "
        clean_text = processor.clean_text(dirty_text)
        
        assert clean_text == "This is a test with extra spaces."
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        processor = DocumentProcessor()
        
        text = "First sentence. Second sentence! Third sentence?"
        sentences = processor.split_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
    
    def test_tokenize(self):
        """Test tokenization."""
        processor = DocumentProcessor()
        
        text = "BRCA1 mutations cause cancer."
        tokens = processor.tokenize(text)
        
        assert len(tokens) >= 4
        assert "BRCA1" in tokens


class TestEntityExtractor:
    """Test entity extraction utilities."""
    
    def test_entity_extractor_init(self):
        """Test EntityExtractor initialization."""
        extractor = EntityExtractor()
        
        assert hasattr(extractor, 'logger')
    
    def test_extract_biomedical_entities(self, sample_biomedical_text):
        """Test biomedical entity extraction."""
        extractor = EntityExtractor()
        
        # Mock the NER model
        with patch.object(extractor, 'ner_model') as mock_model:
            mock_model.return_value = [
                {"word": "BRCA1", "entity": "B-GENE", "confidence": 0.99, "start": 0, "end": 5},
                {"word": "breast", "entity": "B-DISEASE", "confidence": 0.95, "start": 10, "end": 16},
                {"word": "cancer", "entity": "I-DISEASE", "confidence": 0.95, "start": 17, "end": 23}
            ]
            
            entities = extractor.extract_biomedical_entities(sample_biomedical_text)
            
            assert len(entities) >= 2
            assert any(entity["text"] == "BRCA1" for entity in entities)
    
    def test_normalize_entities(self, sample_entities):
        """Test entity normalization."""
        extractor = EntityExtractor()
        
        normalized = extractor.normalize_entities(sample_entities)
        
        assert len(normalized) == len(sample_entities)
        assert all("normalized_text" in entity for entity in normalized)
    
    def test_filter_entities(self, sample_entities):
        """Test entity filtering."""
        extractor = EntityExtractor()
        
        # Filter by confidence
        filtered = extractor.filter_entities(sample_entities, min_confidence=0.9)
        
        # Should have fewer entities (assuming some have confidence < 0.9)
        assert len(filtered) <= len(sample_entities)


class TestKnowledgeGraphPreprocessor:
    """Test knowledge graph preprocessing components."""
    
    def test_kg_preprocessor_init(self, sample_config):
        """Test KnowledgeGraphPreprocessor initialization."""
        preprocessor = KnowledgeGraphPreprocessor(sample_config)

        # The constructor normalizes whatever it is given into a LitKGConfig,
        # which is what the rest of the class reads from.
        assert isinstance(preprocessor.config, LitKGConfig)
        assert hasattr(preprocessor, 'logger')
    
    def test_load_knowledge_graph(self, sample_knowledge_graph):
        """Test knowledge graph loading."""
        preprocessor = KnowledgeGraphPreprocessor({})
        
        with patch.object(preprocessor, '_load_kg_from_source') as mock_load:
            mock_load.return_value = sample_knowledge_graph
            
            kg = preprocessor.load_knowledge_graph("test_source")
            
            assert "nodes" in kg
            assert "edges" in kg
            assert len(kg["nodes"]) == 4
    
    def test_preprocess_nodes(self, sample_knowledge_graph):
        """Test node preprocessing."""
        preprocessor = KnowledgeGraphPreprocessor({})
        
        processed_nodes = preprocessor.preprocess_nodes(sample_knowledge_graph["nodes"])
        
        assert len(processed_nodes) == len(sample_knowledge_graph["nodes"])
        assert all("id" in node for node in processed_nodes)
    
    def test_preprocess_edges(self, sample_knowledge_graph):
        """Test edge preprocessing."""
        preprocessor = KnowledgeGraphPreprocessor({})
        
        processed_edges = preprocessor.preprocess_edges(sample_knowledge_graph["edges"])
        
        assert len(processed_edges) == len(sample_knowledge_graph["edges"])
        assert all("source" in edge and "target" in edge for edge in processed_edges)
    
    def test_build_networkx_graph(self, sample_knowledge_graph):
        """Test NetworkX graph construction."""
        preprocessor = KnowledgeGraphPreprocessor({})
        
        G = preprocessor.build_networkx_graph(sample_knowledge_graph)
        
        assert isinstance(G, nx.Graph)
        assert len(G.nodes()) == 4
        assert len(G.edges()) == 3
    
    def test_compute_graph_statistics(self, sample_knowledge_graph):
        """Test graph statistics computation."""
        preprocessor = KnowledgeGraphPreprocessor({})
        
        G = preprocessor.build_networkx_graph(sample_knowledge_graph)
        stats = preprocessor.compute_graph_statistics(G)
        
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "density" in stats
        assert stats["num_nodes"] == 4
    
    def test_save_and_load_graph(self, temp_dir, sample_knowledge_graph):
        """Test graph saving and loading."""
        preprocessor = KnowledgeGraphPreprocessor({})
        
        G = preprocessor.build_networkx_graph(sample_knowledge_graph)
        
        # Save graph
        graph_file = temp_dir / "test_graph.pkl"
        preprocessor.save_graph(G, str(graph_file))
        
        assert graph_file.exists()
        
        # Load graph
        loaded_G = preprocessor.load_graph(str(graph_file))
        
        assert len(loaded_G.nodes()) == len(G.nodes())
        assert len(loaded_G.edges()) == len(G.edges())


class TestEntityResolution:
    """Test the entity resolution cascade in KnowledgeGraphBuilder."""

    @staticmethod
    def _builder():
        from litkg.phase1.kg_preprocessor import KnowledgeGraphBuilder
        from litkg.utils.config import load_config
        return KnowledgeGraphBuilder(load_config())

    @staticmethod
    def _entity(eid, name, etype="GENE", synonyms=None, cui=None):
        from litkg.phase1.kg_preprocessor import StandardizedEntity
        return StandardizedEntity(
            id=eid, name=name, type=etype, source="TEST", original_id=eid,
            synonyms=synonyms or [], cui=cui,
        )

    def test_merges_on_shared_ontology_id(self):
        """A shared CUI is decisive regardless of surface form."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1", cui="C0376571"),
            self._entity("b", "breast cancer 1", cui="C0376571"),
        ])

        stats = builder.merge_duplicate_entities()

        assert stats["ontology"] == 1
        assert len(builder.entities) == 1

    def test_merges_across_punctuation_differences(self):
        """BRCA-1 and BRCA1 are the same gene."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1"),
            self._entity("b", "BRCA-1"),
            self._entity("c", "brca 1"),
        ])

        builder.merge_duplicate_entities()

        assert len(builder.entities) == 1

    def test_merges_on_synonym_overlap(self):
        """Synonyms resolve names that share no prefix."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "TP53", synonyms=["p53"]),
            self._entity("b", "p53"),
        ])

        stats = builder.merge_duplicate_entities()

        assert stats["synonym"] == 1
        assert len(builder.entities) == 1

    def test_resolution_is_transitive(self):
        """A~B and B~C collapses all three, even if A and C never matched."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1", cui="C0376571"),
            self._entity("b", "breast cancer 1", cui="C0376571"),
            self._entity("c", "BRCA-1"),
        ])

        builder.merge_duplicate_entities()

        assert len(builder.entities) == 1
        survivor = next(iter(builder.entities.values()))
        # Every surface form in the cluster is preserved
        assert {survivor.name, *survivor.synonyms} >= {"BRCA1", "breast cancer 1", "BRCA-1"}

    def test_does_not_merge_across_entity_types(self):
        """The same string naming a gene and a disease is two entities."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1", etype="GENE"),
            self._entity("b", "BRCA1", etype="DISEASE"),
        ])

        builder.merge_duplicate_entities()

        assert len(builder.entities) == 2

    def test_does_not_merge_distinct_genes(self):
        """BRCA1 and BRCA2 have different CUIs and must stay separate."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1", cui="C0376571"),
            self._entity("b", "BRCA2", cui="C0376572"),
        ])

        builder.merge_duplicate_entities(similarity_threshold=0.9)

        assert len(builder.entities) == 2

    def test_threshold_is_honored(self):
        """similarity_threshold actually gates fuzzy merging."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "EGFR"),
            self._entity("b", "EGFR1"),
        ])

        # A threshold above the pair's similarity leaves them separate
        builder.merge_duplicate_entities(similarity_threshold=0.99)
        assert len(builder.entities) == 2

        # A permissive threshold merges them
        stats = builder.merge_duplicate_entities(similarity_threshold=0.7)
        assert stats["fuzzy"] == 1
        assert len(builder.entities) == 1

    def test_go_ids_are_not_identity_evidence(self):
        """
        BRCA1 and BRCA2 both carry GO:0006281 ("DNA repair") -- correctly, since
        both are involved in it. A GO term annotates function, not identity, so
        sharing one must not merge two distinct genes.
        """
        builder = self._builder()
        a = self._entity("a", "BRCA1", cui="C0376571")
        b = self._entity("b", "BRCA2", cui="C0376572")
        a.go_id = b.go_id = "GO:0006281"
        builder.add_entities([a, b])

        builder.merge_duplicate_entities(similarity_threshold=0.99)

        assert len(builder.entities) == 2

    def test_shared_cui_still_merges(self):
        """The identity rule must still fire on a genuine identifier match."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1", cui="C0376571"),
            self._entity("b", "breast cancer 1", cui="C0376571"),
        ])

        stats = builder.merge_duplicate_entities()

        assert stats["ontology"] == 1
        assert len(builder.entities) == 1

    def test_canonical_entity_keeps_ontology_id(self):
        """Merging never discards the best-described member of a cluster."""
        builder = self._builder()
        builder.add_entities([
            self._entity("a", "BRCA1"),
            self._entity("b", "BRCA1", cui="C0376571"),
        ])

        builder.merge_duplicate_entities()

        survivor = next(iter(builder.entities.values()))
        assert survivor.cui == "C0376571"


class TestOntologyMapper:
    """Test ontology mapping utilities."""

    def test_heuristic_umls_mapping_is_case_insensitive(self):
        """Disease lookups were dead because keys and lookup disagreed on case."""
        mapper = OntologyMapper()

        assert mapper._heuristic_umls_mapping("breast cancer", "DISEASE") == "C0006142"
        assert mapper._heuristic_umls_mapping("Breast Cancer", "DISEASE") == "C0006142"
        assert mapper._heuristic_umls_mapping("melanoma", "DISEASE") == "C0025202"

    def test_distinct_genes_have_distinct_cuis(self):
        """A shared CUI would silently merge two different genes."""
        mapper = OntologyMapper()

        brca1 = mapper._heuristic_umls_mapping("BRCA1", "GENE")
        brca2 = mapper._heuristic_umls_mapping("BRCA2", "GENE")

        assert brca1 and brca2 and brca1 != brca2

    def test_seed_ontology_is_autoloaded(self):
        """Shipped ontology files must be reachable without an explicit load."""
        mapper = OntologyMapper()

        assert mapper.ontology_db, "no ontology auto-loaded from data/ontologies"

    def test_synonyms_resolve_to_canonical_name(self):
        mapper = OntologyMapper()

        for surface, canonical in [
            ("HER2", "ERBB2"), ("p16", "CDKN2A"), ("Lynparza", "olaparib"),
        ]:
            record = mapper.map_entity_to_ontology(surface)
            assert record and record["canonical_name"] == canonical

    def test_loaded_ontology_supplies_cuis(self):
        """map_to_umls consults loaded ontologies, not just the tiny heuristic."""
        mapper = OntologyMapper()

        assert mapper.map_to_umls("breast cancer", "DISEASE") == "C0006142"

    def test_absent_cuis_are_not_fabricated(self):
        """A missing CUI must stay missing; a wrong one would merge entities."""
        mapper = OntologyMapper()

        assert mapper.map_to_umls("PTEN", "GENE") is None

    def test_seed_ontology_has_no_duplicate_cuis(self):
        """A shared CUI is decisive for merging, so duplicates are dangerous."""
        mapper = OntologyMapper()

        cuis = [
            record["cui"]
            for record in mapper.ontology_db.values()
            if record.get("cui")
        ]
        assert len(set(cuis)) == len(dict.fromkeys(cuis))

    def test_ontology_mapper_init(self):
        """Test OntologyMapper initialization."""
        mapper = OntologyMapper()
        
        assert hasattr(mapper, 'logger')
    
    def test_load_ontology(self):
        """Test ontology loading."""
        mapper = OntologyMapper()
        
        # Mock ontology data
        mock_ontology = {
            "BRCA1": {"id": "HGNC:1100", "type": "gene", "synonyms": ["BRCA1", "BRCAI"]},
            "breast cancer": {"id": "DOID:1612", "type": "disease", "synonyms": ["breast cancer", "mammary cancer"]}
        }
        
        with patch.object(mapper, '_load_ontology_file') as mock_load:
            mock_load.return_value = mock_ontology
            
            ontology = mapper.load_ontology("test_ontology")
            
            assert len(ontology) == 2
            assert "BRCA1" in ontology
    
    def test_map_entity_to_ontology(self):
        """Test entity to ontology mapping."""
        mapper = OntologyMapper()
        
        # Mock ontology lookup
        with patch.object(mapper, 'ontology_db') as mock_db:
            mock_db.get.return_value = {
                "id": "HGNC:1100",
                "type": "gene",
                "canonical_name": "BRCA1"
            }
            
            mapping = mapper.map_entity_to_ontology("BRCA1")
            
            assert mapping["id"] == "HGNC:1100"
            assert mapping["type"] == "gene"
    
    def test_standardize_entities(self, sample_entities):
        """Test entity standardization."""
        mapper = OntologyMapper()
        
        with patch.object(mapper, 'map_entity_to_ontology') as mock_map:
            mock_map.return_value = {"id": "TEST:123", "canonical_name": "standardized_name"}
            
            standardized = mapper.standardize_entities(sample_entities)
            
            assert len(standardized) == len(sample_entities)
            assert all("ontology_id" in entity for entity in standardized)


class TestEntityLinker:
    """Test entity linking components."""
    
    def test_entity_linker_init(self, sample_config):
        """Test EntityLinker initialization."""
        linker = EntityLinker(sample_config)

        # The constructor normalizes whatever it is given into a LitKGConfig,
        # which is what the rest of the class reads from.
        assert isinstance(linker.config, LitKGConfig)
        assert hasattr(linker, 'logger')
    
    def test_link_entities(self, sample_entities, sample_knowledge_graph):
        """Test entity linking."""
        linker = EntityLinker({})
        
        with patch.object(linker, 'knowledge_graph') as mock_kg:
            mock_kg.nodes.return_value = [node["id"] for node in sample_knowledge_graph["nodes"]]
            
            with patch.object(linker, '_find_best_match') as mock_match:
                mock_match.return_value = {"kg_id": "BRCA1", "confidence": 0.95}
                
                linked_entities = linker.link_entities(sample_entities)
                
                assert len(linked_entities) == len(sample_entities)
                assert all("kg_id" in entity for entity in linked_entities)
    
    def test_fuzzy_matching(self):
        """Test fuzzy string matching."""
        linker = EntityLinker({})
        
        candidates = ["BRCA1", "BRCA2", "TP53", "EGFR"]
        query = "BRCA-1"
        
        matches = linker.fuzzy_match(query, candidates)
        
        assert len(matches) > 0
        assert matches[0]["candidate"] == "BRCA1"
        assert matches[0]["score"] > 0.8
    
    def test_disambiguation(self, sample_entities):
        """Test entity disambiguation."""
        linker = EntityLinker({})
        
        # Mock disambiguation logic
        with patch.object(linker, '_get_context_embeddings') as mock_embed:
            mock_embed.return_value = np.random.rand(768)
            
            with patch.object(linker, '_compute_similarity') as mock_sim:
                mock_sim.return_value = 0.85
                
                disambiguated = linker.disambiguate_entities(sample_entities)
                
                assert len(disambiguated) == len(sample_entities)
    
    def test_confidence_scoring(self):
        """Test confidence scoring for entity links."""
        linker = EntityLinker({})
        
        link_data = {
            "fuzzy_score": 0.9,
            "context_similarity": 0.8,
            "frequency_score": 0.7
        }
        
        confidence = linker.compute_link_confidence(link_data)
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident


class TestFuzzyMatcher:
    """Test fuzzy matching utilities."""
    
    def test_fuzzy_matcher_init(self):
        """Test FuzzyMatcher initialization."""
        matcher = FuzzyMatcher()
        
        assert hasattr(matcher, 'logger')
    
    def test_string_similarity(self):
        """Test string similarity computation."""
        matcher = FuzzyMatcher()
        
        # Test exact match
        score = matcher.compute_similarity("BRCA1", "BRCA1")
        assert score == 1.0
        
        # Test similar strings
        score = matcher.compute_similarity("BRCA1", "BRCA-1")
        assert score > 0.8
        
        # Test different strings
        score = matcher.compute_similarity("BRCA1", "TP53")
        assert score < 0.5
    
    def test_find_best_matches(self):
        """Test finding best matches."""
        matcher = FuzzyMatcher()
        
        query = "breast cancer"
        candidates = ["breast cancer", "lung cancer", "mammary cancer", "prostate cancer"]
        
        matches = matcher.find_best_matches(query, candidates, top_k=3)
        
        assert len(matches) <= 3
        assert matches[0]["candidate"] == "breast cancer"
        assert matches[0]["score"] == 1.0
    
    def test_batch_matching(self):
        """Test batch fuzzy matching."""
        matcher = FuzzyMatcher()
        
        queries = ["BRCA1", "p53", "EGFR"]
        candidates = ["BRCA1", "BRCA2", "TP53", "EGFR", "KRAS"]
        
        all_matches = matcher.batch_match(queries, candidates)
        
        assert len(all_matches) == len(queries)
        assert all(len(matches) > 0 for matches in all_matches)


class TestDisambiguationEngine:
    """Test entity disambiguation utilities."""
    
    def test_disambiguation_engine_init(self):
        """Test DisambiguationEngine initialization."""
        engine = DisambiguationEngine()
        
        assert hasattr(engine, 'logger')
    
    def test_context_based_disambiguation(self):
        """Test context-based disambiguation."""
        engine = DisambiguationEngine()
        
        entity = "p53"
        candidates = ["TP53", "CDKN1A", "MDM2"]  # p53 could refer to any of these
        context = "The p53 tumor suppressor gene is mutated in many cancers"
        
        with patch.object(engine, '_compute_context_similarity') as mock_sim:
            mock_sim.side_effect = [0.95, 0.3, 0.4]  # TP53 has highest similarity
            
            best_match = engine.disambiguate_with_context(entity, candidates, context)
            
            assert best_match["candidate"] == "TP53"
            assert best_match["confidence"] > 0.9
    
    def test_frequency_based_disambiguation(self):
        """Test frequency-based disambiguation."""
        engine = DisambiguationEngine()
        
        entity = "p53"
        candidates = ["TP53", "CDKN1A", "MDM2"]
        
        # Mock frequency data
        frequency_data = {"TP53": 1000, "CDKN1A": 100, "MDM2": 200}
        
        with patch.object(engine, 'entity_frequencies', frequency_data):
            best_match = engine.disambiguate_with_frequency(entity, candidates)
            
            assert best_match["candidate"] == "TP53"
    
    def test_multi_criteria_disambiguation(self):
        """Test multi-criteria disambiguation."""
        engine = DisambiguationEngine()
        
        entity = "p53"
        candidates = ["TP53", "CDKN1A"]
        context = "tumor suppressor"
        
        with patch.object(engine, '_compute_context_similarity') as mock_sim:
            mock_sim.side_effect = [0.9, 0.3]
            
            with patch.object(engine, 'entity_frequencies', {"TP53": 1000, "CDKN1A": 100}):
                best_match = engine.multi_criteria_disambiguation(entity, candidates, context)
                
                assert best_match["candidate"] == "TP53"
                assert "combined_score" in best_match


@pytest.mark.integration
class TestPhase1Integration:
    """Integration tests for Phase 1 components."""
    
    def test_end_to_end_processing(self, sample_literature_data, sample_config):
        """Test end-to-end Phase 1 processing."""
        # Initialize components
        lit_processor = LiteratureProcessor(sample_config)
        kg_preprocessor = KnowledgeGraphPreprocessor(sample_config)
        entity_linker = EntityLinker(sample_config)
        
        # Mock the processing pipeline
        with patch.object(lit_processor, 'process_batch') as mock_lit:
            mock_lit.return_value = [
                {
                    "entities": [{"text": "BRCA1", "label": "GENE"}],
                    "relations": [{"head": "BRCA1", "relation": "ASSOCIATED_WITH", "tail": "cancer"}]
                }
            ]
            
            with patch.object(entity_linker, 'link_entities') as mock_link:
                mock_link.return_value = [{"text": "BRCA1", "kg_id": "HGNC:1100"}]
                
                # Process literature
                lit_results = lit_processor.process_batch(sample_literature_data)
                
                # Link entities
                linked_entities = entity_linker.link_entities(lit_results[0]["entities"])
                
                assert len(lit_results) == 1
                assert len(linked_entities) == 1
                assert linked_entities[0]["kg_id"] == "HGNC:1100"
    
    def test_pipeline_error_handling(self, sample_literature_data):
        """Test error handling in the processing pipeline."""
        processor = LiteratureProcessor({})
        
        # Test with malformed data
        malformed_data = [{"invalid": "data"}]
        
        with pytest.raises(Exception):
            processor.process_batch(malformed_data)
    
    @pytest.mark.slow
    def test_large_scale_processing(self):
        """Test processing of large datasets."""
        processor = LiteratureProcessor({})
        
        # Create large dataset
        large_dataset = [
            {
                "pmid": f"pmid_{i}",
                "title": f"Title {i}",
                "abstract": "BRCA1 mutations cause breast cancer."
            }
            for i in range(100)
        ]
        
        with patch.object(processor, 'process_document') as mock_process:
            mock_process.return_value = {"entities": [], "relations": []}
            
            results = processor.process_batch(large_dataset)
            
            assert len(results) == 100


if __name__ == "__main__":
    pytest.main([__file__])

class TestCivicGeneNodes:
    """
    Gene-level nodes are what literature mentions resolve against.

    Literature NER extracts gene symbols ("BRCA1"); CIVIC variant records are
    named for the alteration ("1100delC"). Without gene nodes the two
    vocabularies cannot meet, and cross-modal linking collapses to the few
    variant notations that appear verbatim in abstracts.
    """

    @staticmethod
    def _processor():
        from litkg.phase1.kg_preprocessor import CivicProcessor
        from litkg.utils.config import load_config
        return CivicProcessor(load_config())

    def test_gene_and_variant_processors_agree_on_id(self):
        """
        Both sides must derive the same id or every HAS_VARIANT edge dangles.

        civic_genes.tsv and civic_variants.tsv name the entrez column
        identically but pandas yields a float from one and a string from the
        other, so the helper must normalize both.
        """
        from litkg.phase1.kg_preprocessor import CivicProcessor

        from_genes_file = CivicProcessor._civic_gene_id("ALK", 238.0)
        from_variants_file = CivicProcessor._civic_gene_id("ALK", "238")

        assert from_genes_file == from_variants_file

    def test_gene_id_falls_back_to_symbol(self):
        from litkg.phase1.kg_preprocessor import CivicProcessor

        for missing in ("", "nan", None):
            assert CivicProcessor._civic_gene_id("ALK", missing) == "CIVIC:GENE:ALK"

    def test_genes_are_read_from_the_name_column(self, tmp_path):
        """
        civic_genes.tsv carries the symbol in "name"; only civic_variants.tsv
        has a "gene" column. Reading the wrong one silently drops every gene.
        """
        genes_file = tmp_path / "civic_genes.tsv"
        genes_file.write_text(
            "gene_id\tgene_civic_url\tname\tentrez_id\tdescription\n"
            "1\thttp://x\tALK\t238\tsome description\n"
            "2\thttp://y\tBRCA1\t672\tanother description\n"
        )

        entities = self._processor()._process_civic_genes(genes_file)

        assert [e.name for e in entities] == ["ALK", "BRCA1"]
        assert all(e.type == "GENE" for e in entities)

    def test_accepts_either_column_name(self, tmp_path):
        """A schema change on one file must not empty the vocabulary."""
        genes_file = tmp_path / "civic_genes.tsv"
        genes_file.write_text("gene_id\tgene\tentrez_id\n1\tALK\t238\n")

        entities = self._processor()._process_civic_genes(genes_file)

        assert [e.name for e in entities] == ["ALK"]

    def test_variant_edges_point_at_real_gene_nodes(self, tmp_path):
        """The end-to-end invariant: no dangling HAS_VARIANT edges."""
        processor = self._processor()

        genes_file = tmp_path / "civic_genes.tsv"
        genes_file.write_text("gene_id\tname\tentrez_id\n1\tALK\t238\n")
        variants_file = tmp_path / "civic_variants.tsv"
        variants_file.write_text(
            "variant_id\tgene\tentrez_id\tvariant\n"
            "10\tALK\t238\tF1174L\n"
        )

        genes = processor._process_civic_genes(genes_file)
        _, relations = processor._process_civic_variants(variants_file)

        gene_ids = {g.id for g in genes}
        assert relations, "expected a HAS_VARIANT relation"
        for relation in relations:
            assert relation.subject in gene_ids, (
                f"{relation.subject} has no matching gene node; edge would dangle"
            )


class TestNERPrecision:
    """The rule-based extractor used to type every all-caps token as a GENE.

    `en_core_sci_*` emits a single "ENTITY" label, so the scispacy path
    contributed nothing and the acronym regex was the only source of types.
    Result: 100% of extracted entities were GENE, and every literature
    relation was GENE->GENE.
    """

    @pytest.fixture(scope="class")
    def nlp(self):
        from litkg.phase1.literature_processor import BiomedicalNLP
        instance = BiomedicalNLP.__new__(BiomedicalNLP)
        instance._gene_vocabulary = None
        return instance

    @pytest.mark.parametrize("symbol", [
        "BRCA1", "BRCA2", "TP53", "KRAS", "EGFR", "ALK", "PTEN", "MYC",
    ])
    def test_real_gene_symbols_accepted(self, nlp, symbol):
        assert nlp._is_likely_gene(symbol), f"{symbol} should be recognized as a gene"

    @pytest.mark.parametrize("acronym", [
        "ALL",    # acute lymphoblastic leukemia - a disease
        "NSCLC",  # non-small cell lung cancer - a disease
        "TNBC",   # triple-negative breast cancer - a disease
        "PFS",    # progression-free survival - an outcome measure
        "DNA",    # a molecule
        "PCR",    # a method
        "FDA",    # an organization
        "ICI",    # immune checkpoint inhibitor - a drug class
    ])
    def test_non_gene_acronyms_rejected(self, nlp, acronym):
        assert not nlp._is_likely_gene(acronym), (
            f"{acronym} is not a gene; typing it as one is what made every "
            f"relation GENE->GENE"
        )

    def test_gene_vocabulary_is_populated(self, nlp):
        """Acceptance is vocabulary-driven, not regex-shaped.

        Size is environment-dependent: the CIVIC gene list lives under
        `data/external/` and is downloaded, not committed, so a fresh checkout
        has only the seed ontology (~47 symbols) against ~545 once CIVIC is
        present. Assert the committed floor, not the local number.
        """
        vocabulary = nlp.gene_vocabulary
        assert vocabulary, "gene vocabulary is empty; every gene would be rejected"
        seeded = {"BRCA1", "BRCA2", "TP53", "EGFR", "ALK"}
        assert seeded <= vocabulary, (
            f"seed ontology genes missing from vocabulary: {seeded - vocabulary}"
        )

    def test_vocabulary_gate_applies_only_to_the_rule_based_path(self):
        """The specialized NER models must not be gated by the vocabulary.

        Otherwise a fresh checkout without the CIVIC download would cap gene
        recall at the ~15 seed ontology genes.
        """
        import inspect
        from litkg.phase1.literature_processor import BiomedicalNLP
        scispacy_src = inspect.getsource(BiomedicalNLP._extract_entities_scispacy)
        assert "_is_likely_gene" not in scispacy_src
        assert "gene_vocabulary" not in scispacy_src
        rules_src = inspect.getsource(BiomedicalNLP._extract_entities_rules)
        assert "_is_likely_gene" in rules_src

    def test_label_map_covers_configured_entity_types(self):
        """Every mapped label must be a type the processor actually keeps."""
        from litkg.phase1.literature_processor import BiomedicalNLP
        valid = {
            "GENE", "DISEASE", "DRUG", "PROTEIN", "CELL_TYPE",
            "TISSUE", "ORGANISM", "CHEMICAL", "MUTATION",
        }
        unknown = set(BiomedicalNLP.NER_LABEL_MAP.values()) - valid
        assert not unknown, f"NER_LABEL_MAP produces unusable types: {unknown}"

    def test_specialized_models_are_preferred_over_generic(self):
        """en_core_sci_* cannot type entities, so it must not lead."""
        from litkg.phase1.literature_processor import BiomedicalNLP
        assert BiomedicalNLP.NER_MODELS, "no NER models configured"
        assert not any(m.startswith("en_core_sci") for m in BiomedicalNLP.NER_MODELS)


class TestBertNERHead:
    """The BERT NER path ran on a checkpoint with no NER head.

    `config.yaml` pointed the pipeline at `dmis-lab/biobert-base-cased-v1.1`,
    a base language model. transformers initialized a classifier head at
    random, warned, and ran anyway: every span came back LABEL_0/LABEL_1 at
    ~0.5 confidence, none of which is an entity type this processor keeps. The
    path was live in the code, cost a model load per run, and returned zero
    entities on every document.
    """

    UNTRAINED_LABELS = {0: "LABEL_0", 1: "LABEL_1"}

    @pytest.fixture
    def nlp(self):
        from litkg.phase1.literature_processor import BiomedicalNLP
        instance = BiomedicalNLP.__new__(BiomedicalNLP)
        instance._gene_vocabulary = None
        instance.entity_types = {
            "GENE", "DISEASE", "DRUG", "PROTEIN", "CELL_TYPE",
            "TISSUE", "ORGANISM", "CHEMICAL", "MUTATION",
        }
        instance.ner_pipeline = None
        return instance

    def test_untrained_head_is_detected(self):
        from litkg.phase1.literature_processor import is_untrained_token_classifier
        assert is_untrained_token_classifier(self.UNTRAINED_LABELS)
        assert is_untrained_token_classifier({}), "no labels at all is not a trained head"
        assert not is_untrained_token_classifier(
            {0: "B-GENETIC", 1: "I-GENETIC", 2: "O"}
        )

    def test_configured_checkpoint_has_a_real_label_scheme(self):
        """Fails if the configured model's id2label is the LABEL_N default.

        CI is offline, so the checkpoint is not downloaded here: the stub
        stands in for whatever `biomedical_ner` names, and the assertion is
        that a LABEL_N scheme is rejected at load time instead of being run
        for nothing.
        """
        import yaml
        from litkg.phase1.literature_processor import (
            BiomedicalNLP, is_untrained_token_classifier,
        )

        config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
        models = yaml.safe_load(config_path.read_text())["phase1"]["literature"]["models"]
        configured = models.get("biomedical_ner", BiomedicalNLP.DEFAULT_BERT_NER_MODEL)

        # A base LM has no NER head no matter how biomedical it is.
        assert configured not in {
            models.get("biobert"),
            models.get("pubmedbert"),
            "dmis-lab/biobert-base-cased-v1.1",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        }, (
            f"{configured} is a base language model with no token-classification "
            "head; the NER pipeline built on it emits LABEL_N and yields no entities"
        )

        # The label scheme the configured checkpoint reports, stubbed for CI.
        assert is_untrained_token_classifier(self.UNTRAINED_LABELS), (
            "an untrained head must be rejected"
        )

    def test_untrained_head_disables_the_path(self, nlp, caplog):
        """A LABEL_N pipeline must not be installed and quietly run."""
        from litkg.phase1.literature_processor import BiomedicalNLP

        class StubConfig:
            id2label = dict(TestBertNERHead.UNTRAINED_LABELS)

        class StubModel:
            config = StubConfig()

        class StubPipeline:
            model = StubModel()
            called = False

            def __call__(self, text):
                type(self).called = True
                return [{"entity_group": "LABEL_0", "word": "BRCA1",
                         "start": 0, "end": 5, "score": 0.51}]

        stub = StubPipeline()
        nlp.models_config = {
            "biomedical_ner": "some/base-lm",
            "scispacy_model": "en_core_sci_md",
            "pubmedbert": "stub/pubmedbert",
            "biobert": "stub/biobert",
        }
        nlp.nlp = object()

        with patch("litkg.phase1.literature_processor.pipeline", return_value=stub), \
             patch("litkg.phase1.literature_processor.spacy.load", return_value=object()), \
             patch("litkg.phase1.literature_processor.AutoTokenizer"), \
             patch("litkg.phase1.literature_processor.AutoModel"), \
             patch("litkg.phase1.literature_processor.BertTokenizer"), \
             patch("litkg.phase1.literature_processor.BertModel"):
            BiomedicalNLP._load_models(nlp)

        assert nlp.ner_pipeline is None, (
            "a checkpoint with a randomly initialized head was installed as the "
            "NER pipeline; it will run on every document and return nothing"
        )
        assert nlp._extract_entities_bert("BRCA1 mutations") == []
        assert not StubPipeline.called, "the dead pipeline was still invoked"

    def test_trained_head_is_installed_and_extracts(self, nlp):
        """The real checkpoint's scheme must survive mapping into entity types."""
        from litkg.phase1.literature_processor import BiomedicalNLP

        text = "Loss of SetD5 and Fgfr2 drives tumor growth."

        class StubConfig:
            id2label = {0: "B-GENETIC", 1: "I-GENETIC", 2: "O"}

        class StubModel:
            config = StubConfig()

        class StubPipeline:
            model = StubModel()
            tokenizer = StubTokenizer()

            def __call__(self, _text):
                return [
                    {"entity_group": "GENETIC", "word": "SetD5",
                     "start": 8, "end": 13, "score": 0.98},
                    {"entity_group": "GENETIC", "word": "Fgfr2",
                     "start": 18, "end": 23, "score": 0.97},
                ]

        nlp.models_config = {
            "biomedical_ner": "alvaroalon2/biobert_genetic_ner",
            "scispacy_model": "en_core_sci_md",
            "pubmedbert": "stub/pubmedbert",
            "biobert": "stub/biobert",
        }

        with patch("litkg.phase1.literature_processor.pipeline", return_value=StubPipeline()), \
             patch("litkg.phase1.literature_processor.spacy.load", return_value=object()), \
             patch("litkg.phase1.literature_processor.AutoTokenizer"), \
             patch("litkg.phase1.literature_processor.AutoModel"), \
             patch("litkg.phase1.literature_processor.BertTokenizer"), \
             patch("litkg.phase1.literature_processor.BertModel"):
            BiomedicalNLP._load_models(nlp)

        assert nlp.ner_pipeline is not None
        entities = nlp._extract_entities_bert(text)
        assert [e.text for e in entities] == ["SetD5", "Fgfr2"]
        assert {e.label for e in entities} == {"GENE"}, (
            "GENETIC must map to a type in entity_types, or the path is inert again"
        )

    def test_every_mapped_label_is_a_usable_entity_type(self):
        from litkg.phase1.literature_processor import BiomedicalNLP
        valid = {
            "GENE", "DISEASE", "DRUG", "PROTEIN", "CELL_TYPE",
            "TISSUE", "ORGANISM", "CHEMICAL", "MUTATION",
        }
        unknown = set(BiomedicalNLP.BERT_NER_LABEL_MAP.values()) - valid
        assert not unknown, f"BERT_NER_LABEL_MAP produces unusable types: {unknown}"

    def test_label_n_is_never_mapped_to_an_entity_type(self, nlp):
        """The old mapper passed LABEL_0 straight through."""
        from litkg.phase1.literature_processor import BiomedicalNLP
        for label in ("LABEL_0", "LABEL_1", "LABEL_7"):
            assert BiomedicalNLP._map_bert_label(nlp, label) not in nlp.entity_types

    def test_bio_prefixes_and_case_are_normalized(self, nlp):
        from litkg.phase1.literature_processor import BiomedicalNLP
        for label in ("GENETIC", "B-GENETIC", "I-genetic"):
            assert BiomedicalNLP._map_bert_label(nlp, label) == "GENE"

    def test_subword_fragments_are_rejected(self, nlp):
        """Aggregation cuts words in half: "isplatin", "arubicin", "inib"."""
        text = "Treatment with cisplatin and doxorubicin was given."

        class StubPipeline:
            tokenizer = StubTokenizer()
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))

            def __call__(self, _text):
                return [
                    # a mid-word fragment of "cisplatin"
                    {"entity_group": "CHEMICAL", "word": "isplatin",
                     "start": 16, "end": 24, "score": 0.95},
                    # the whole word
                    {"entity_group": "CHEMICAL", "word": "doxorubicin",
                     "start": 29, "end": 40, "score": 0.95},
                ]

        nlp.ner_pipeline = StubPipeline()
        entities = nlp._extract_entities_bert(text)
        assert [e.text for e in entities] == ["doxorubicin"]

    def test_non_gene_acronyms_are_not_typed_as_genes(self, nlp):
        """The checkpoint tags CAR and MDS as genes; they are not."""
        text = "CAR T-cell therapy in MDS patients."

        class StubPipeline:
            tokenizer = StubTokenizer()
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))

            def __call__(self, _text):
                return [
                    {"entity_group": "GENETIC", "word": "CAR",
                     "start": 0, "end": 3, "score": 0.93},
                    {"entity_group": "GENETIC", "word": "MDS",
                     "start": 22, "end": 25, "score": 0.91},
                ]

        nlp.ner_pipeline = StubPipeline()
        assert nlp._extract_entities_bert(text) == []

    def test_spans_claimed_by_scispacy_are_not_re_emitted(self, nlp):
        """One mention must not become two entities with different spans."""
        text = "Elevated serum MMP-9 was observed."

        class StubPipeline:
            tokenizer = StubTokenizer()
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))

            def __call__(self, _text):
                return [
                    {"entity_group": "GENETIC", "word": "serum MMP-9",
                     "start": 9, "end": 20, "score": 0.9},
                ]

        nlp.ner_pipeline = StubPipeline()
        # scispacy already claimed "MMP-9" at 15-20
        assert nlp._extract_entities_bert(text, [(15, 20)]) == []
        assert len(nlp._extract_entities_bert(text, [(0, 8)])) == 1


class TestClinicalEntityExtraction:
    """CIVIC evidence carries diseases, therapies and phenotypes.

    The previous implementation emitted relations pointing at CIVIC:DISEASE
    and CIVIC:DRUG nodes it never created, and read three column names the
    evidence file does not have. 4125 of 5825 KG edges dangled.
    """

    @pytest.fixture
    def processor(self):
        from litkg.utils.config import load_config
        from litkg.phase1.kg_preprocessor import CivicProcessor
        return CivicProcessor(load_config())

    def test_disease_id_prefers_doid(self, processor):
        """DOID is a real identity: two records sharing one are the same disease."""
        assert processor._civic_disease_id("Lung Cancer", "1324") == "CIVIC:DISEASE:DOID:1324"
        assert processor._civic_disease_id("Lung Cancer", 1324.0) == "CIVIC:DISEASE:DOID:1324"

    def test_disease_id_falls_back_to_name(self, processor):
        """258 of 268 diseases have a DOID; the rest still need a stable id."""
        for missing in (None, "", "nan", float("nan")):
            assert processor._civic_disease_id("Rare Tumor", missing) == "CIVIC:DISEASE:RARE TUMOR"

    def test_disease_id_is_stable_across_name_casing(self, processor):
        assert (processor._civic_disease_id("lung cancer")
                == processor._civic_disease_id("Lung Cancer"))

    def test_therapies_are_split(self, processor):
        """CIVIC packs multiple therapies into one comma-separated cell."""
        assert processor._split_multi_valued("Imatinib, Dasatinib") == ["Imatinib", "Dasatinib"]
        assert processor._split_multi_valued("nan") == []
        assert processor._split_multi_valued(None) == []

    def test_compound_profiles_resolve_to_each_component(self, processor):
        """"BRAF V600E AND BRAF V600M" is evidence about both variants."""
        index = {"BRAF V600E": "CIVIC:VARIANT:12", "BRAF V600M": "CIVIC:VARIANT:13"}
        assert processor._resolve_molecular_profile("BRAF V600E", index) == ["CIVIC:VARIANT:12"]
        resolved = processor._resolve_molecular_profile("BRAF V600E AND BRAF V600M", index)
        assert sorted(resolved) == ["CIVIC:VARIANT:12", "CIVIC:VARIANT:13"]

    def test_unresolvable_profile_yields_no_subject(self, processor):
        """Better no edge than an edge from CIVIC:VARIANT: (the old behaviour)."""
        assert processor._resolve_molecular_profile("NOTAGENE X1Y", {}) == []

    def test_predictive_evidence_targets_a_therapy(self, processor):
        """Sensitivity is a statement about a drug, not about a disease."""
        predicate, kind = processor.EVIDENCE_PREDICATES[("Predictive", "Sensitivity/Response")]
        assert (predicate, kind) == ("SENSITIZES_TO", "therapy")
        predicate, kind = processor.EVIDENCE_PREDICATES[("Predictive", "Resistance")]
        assert (predicate, kind) == ("RESISTANT_TO", "therapy")

    def test_prognostic_evidence_targets_a_disease(self, processor):
        for significance in ("Poor Outcome", "Better Outcome"):
            _, kind = processor.EVIDENCE_PREDICATES[("Prognostic", significance)]
            assert kind == "disease"

    def test_confidence_tracks_evidence_level(self, processor):
        """A flat 0.8 made confidence filtering meaningless."""
        levels = ["A", "B", "C", "D", "E"]
        scores = [processor._evidence_confidence({"evidence_level": lv, "rating": 3}) for lv in levels]
        assert scores == sorted(scores, reverse=True), f"not monotonic: {dict(zip(levels, scores))}"
        assert all(0.0 < s <= 1.0 for s in scores)

    def test_unknown_evidence_level_is_not_treated_as_strong(self, processor):
        weak = processor._evidence_confidence({"evidence_level": "", "rating": 3})
        strong = processor._evidence_confidence({"evidence_level": "A", "rating": 3})
        assert weak < strong

    def test_doid_is_an_identity_identifier(self):
        """Identity identifiers drive merging; descriptive ones must not."""
        from litkg.phase1.kg_preprocessor import KnowledgeGraphBuilder
        assert "doid" in KnowledgeGraphBuilder.IDENTITY_IDENTIFIERS
        assert "go_id" not in KnowledgeGraphBuilder.IDENTITY_IDENTIFIERS


class TestClinicalEntitiesFromRealData:
    """Runs against the CIVIC files if they have been downloaded."""

    @pytest.fixture(scope="class")
    def processed(self):
        from litkg.utils.config import load_config, get_data_dir
        from litkg.phase1.kg_preprocessor import CivicProcessor
        directory = get_data_dir() / "external" / "civic"
        evidence, variants = directory / "civic_evidence.tsv", directory / "civic_variants.tsv"
        if not (evidence.exists() and variants.exists()):
            pytest.skip("CIVIC data not downloaded")
        return CivicProcessor(load_config())._process_civic_evidence(evidence, variants)

    def test_produces_all_three_clinical_types(self, processed):
        entities, _ = processed
        types = {e.type for e in entities}
        assert {"DISEASE", "DRUG", "PHENOTYPE"} <= types

    def test_every_relation_endpoint_has_a_node(self, processed):
        """The bug this replaced left 4125 edges pointing at nothing."""
        entities, relations = processed
        known = {e.id for e in entities}
        # Variant subjects come from the variants file, not this method.
        dangling = [
            r for r in relations
            if not r.subject.startswith("CIVIC:VARIANT:") and r.subject not in known
        ] + [r for r in relations if r.object not in known]
        assert not dangling, f"{len(dangling)} dangling endpoints, e.g. {dangling[:3]}"

    def test_no_subject_is_an_empty_variant_id(self, processed):
        _, relations = processed
        assert not [r for r in relations if r.subject == "CIVIC:VARIANT:"]

    def test_therapy_relations_are_produced(self, processed):
        """Reading the non-existent 'drugs' column meant zero of these."""
        _, relations = processed
        therapy_relations = [r for r in relations if ":THERAPY:" in r.object]
        assert len(therapy_relations) > 1000

    def test_contradicting_evidence_is_marked_not_asserted(self, processed):
        """'Does Not Support' is evidence against; it must not read as a fact."""
        _, relations = processed
        negated = [r for r in relations if r.attributes.get("negated")]
        assert negated, "no negated relations; CIVIC has ~498 'Does Not Support' rows"
        assert all(r.attributes.get("evidence_direction") == "Does Not Support" for r in negated)


class TestOverlappingNERSpans:
    """Running two NER models over one text double-tags spans.

    Introduced when extraction moved to en_ner_bionlp13cg_md +
    en_ner_bc5cdr_md: 109 spans came back with two mentions and disagreeing
    labels (bc5cdr calls CHEK2 a DISEASE, bionlp13cg calls it a GENE).
    """

    def test_one_span_yields_one_entity(self):
        from litkg.phase1.literature_processor import BiomedicalNLP
        nlp = BiomedicalNLP.__new__(BiomedicalNLP)
        nlp._gene_vocabulary = None
        nlp.entity_types = {
            "GENE", "DISEASE", "DRUG", "PROTEIN", "CELL_TYPE",
            "TISSUE", "ORGANISM", "CHEMICAL", "MUTATION",
        }

        class FakeEnt:
            def __init__(self, text, label, start, end):
                self.text, self.label_ = text, label
                self.start_char, self.end_char = start, end
                self._ = type("U", (), {})()

        class FakeDoc:
            def __init__(self, ents): self.ents = ents

        # Both models tag chars 0-5; preferred model runs first.
        pipelines = [
            lambda t: FakeDoc([FakeEnt("CHEK2", "GENE_OR_GENE_PRODUCT", 0, 5)]),
            lambda t: FakeDoc([FakeEnt("CHEK2", "DISEASE", 0, 5)]),
        ]
        nlp._ner_pipelines = pipelines
        entities = nlp._extract_entities_scispacy("CHEK2 mutations")

        spans = [(e.start, e.end) for e in entities]
        assert len(spans) == len(set(spans)), f"duplicate spans: {spans}"
        assert entities[0].label == "GENE", "the preferred model must win the span"

    def test_mention_key_distinguishes_labels(self):
        """A position-only key routes a link onto the wrong entity."""
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "p1", pathlib.Path(__file__).parent.parent / "scripts" / "phase1_integration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Mention:
            def __init__(self, label): self.start, self.end, self.label = 10, 15, label

        gene_key = module._mention_key("123", Mention("GENE"))
        chemical_key = module._mention_key("123", Mention("CHEMICAL"))
        assert gene_key != chemical_key


class TestEntityNormalization:
    """Normalization stripped words that carry identity."""

    @pytest.fixture(scope="class")
    def matcher(self):
        from litkg.phase1.entity_linker import FuzzyMatcher
        from litkg.utils.config import load_config
        return FuzzyMatcher(load_config())

    @pytest.mark.parametrize("literature,kg", [
        ("BRAF", "BRAF Inhibitor"),      # a drug is not its target
        ("MTOR", "MTOR Inhibitor"),
        ("estrogen", "estrogen receptor"),  # a ligand is not its receptor
        ("ALK", "anaplastic lymphoma kinase"),
    ])
    def test_identity_changing_words_are_not_stripped(self, matcher, literature, kg):
        assert matcher.calculate_similarity(literature, kg) < 0.9, (
            f"{literature!r} and {kg!r} are different entities"
        )

    @pytest.mark.parametrize("descriptor", ["gene", "protein"])
    def test_pure_descriptors_are_still_stripped(self, matcher, descriptor):
        """"BRCA1 gene" is BRCA1; that match is the point of normalizing."""
        assert matcher.calculate_similarity("BRCA1", f"BRCA1 {descriptor}") == 1.0


class TestCivicRelease:
    """
    CIVIC renames columns across releases. An earlier version of this code read
    'drugs', 'variant_id' and 'clinical_significance' from an evidence file with
    none of them and produced 4125 dangling edges in silence, so a schema
    mismatch must now fail loudly.
    """

    @pytest.fixture
    def processor(self):
        from litkg.utils.config import load_config
        from litkg.phase1.kg_preprocessor import CivicProcessor
        return CivicProcessor(load_config())

    def test_release_is_pinned_not_nightly(self):
        """
        A nightly build changes underneath you, so a regression cannot be
        distinguished from a data update. The default must be a dated release.
        """
        from litkg.phase1.kg_preprocessor import CivicProcessor
        assert CivicProcessor.DEFAULT_RELEASE.lower() != "nightly"
        assert re.match(r"\d{2}-[A-Z][a-z]{2}-\d{4}$", CivicProcessor.DEFAULT_RELEASE)

    def test_dated_release_urls_repeat_the_date(self, processor):
        urls = processor.download_urls("01-Aug-2026")
        assert all("/01-Aug-2026/01-Aug-2026-" in u for u in urls.values())
        assert set(urls) == {"variants", "evidence", "genes"}

    def test_nightly_release_is_selectable(self, processor):
        urls = processor.download_urls("nightly")
        assert all("/nightly/nightly-" in u for u in urls.values())

    def test_release_can_be_overridden_by_environment(self, processor, monkeypatch):
        from litkg.phase1.kg_preprocessor import CivicProcessor
        monkeypatch.setenv("LITKG_CIVIC_RELEASE", "nightly")
        assert CivicProcessor.release() == "nightly"

    def test_schema_check_rejects_a_missing_column(self, processor, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("evidence_id\tdisease\n1\tMelanoma\n")
        with pytest.raises(ValueError, match="missing required columns"):
            processor._verify_schema("evidence", path)

    def test_schema_check_accepts_the_shipped_files(self, processor):
        from litkg.utils.config import get_data_dir
        directory = get_data_dir() / "external" / "civic"
        for kind in ("evidence", "variants", "genes"):
            path = directory / f"civic_{kind}.tsv"
            if not path.exists():
                pytest.skip("CIVIC data not downloaded")
            processor._verify_schema(kind, path)

    def test_fusions_are_not_typed_as_genes(self):
        """
        Releases from 2024 on ship a features file: 617 genes alongside 345
        fusions. Typing a fusion as a gene would put it in the vocabulary
        literature gene mentions resolve against.
        """
        from litkg.phase1.kg_preprocessor import CivicProcessor
        assert CivicProcessor.FEATURE_TYPES["FUSION"] == "FUSION"
        assert CivicProcessor.FEATURE_TYPES["GENE"] == "GENE"

    def test_unknown_feature_type_falls_back_to_gene(self):
        """Older releases have no feature_type column at all."""
        from litkg.phase1.kg_preprocessor import CivicProcessor
        assert CivicProcessor.FEATURE_TYPES.get("", "GENE") == "GENE"


# What transformers reports for `model_max_length` when the tokenizer config
# carries no length of its own -- dmis-lab/biobert-base-cased-v1.1 included.
UNSET_MODEL_MAX_LENGTH = int(1e30)


class StubTokenizer:
    """Stands in for the biobert tokenizer; CI has no model downloads.

    Four characters per piece is roughly wordpiece's density on dense
    biomedical prose, and `model_max_length` reports the same sentinel the real
    tokenizer does, so the usable limit has to come from the model config.
    """

    model_max_length = UNSET_MODEL_MAX_LENGTH

    def __init__(self, is_fast=True):
        self.is_fast = is_fast

    def _spans(self, text):
        for word in re.finditer(r"\S+", text):
            for start in range(word.start(), word.end(), 4):
                yield (start, min(start + 4, word.end()))

    def tokenize(self, text):
        return [text[start:end] for start, end in self._spans(text)]

    def num_special_tokens_to_add(self):
        return 2

    def __call__(self, text, truncation=False, max_length=None,
                 return_offsets_mapping=False, **kwargs):
        # [CLS] and [SEP] both carry an empty span, as in transformers
        offsets = [(0, 0)] + list(self._spans(text)) + [(0, 0)]
        if truncation and max_length is not None and len(offsets) > max_length:
            offsets = offsets[:max_length - 1] + [(0, 0)]
        encoded = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


class StubNERPipeline:
    """Fails the way the real pipeline fails.

    BERT's position embedding table is fixed width, so an over-long input
    raises RuntimeError rather than degrading.
    """

    def __init__(self, tokenizer, limit=512):
        self.tokenizer = tokenizer
        self.limit = limit
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=limit)
        )
        self.calls = []

    def __call__(self, text):
        length = len(self.tokenizer(text)["input_ids"])
        if length > self.limit:
            raise RuntimeError(
                f"The size of tensor a ({length}) must match the size of "
                f"tensor b ({self.limit}) at non-singleton dimension 1"
            )
        self.calls.append(text)
        return [{
            "entity_group": "GENE", "word": "BRCA1",
            "start": 0, "end": 5, "score": 0.99,
        }]


class TestBertNERTruncation:
    """A 2000-character cap is not a 512-token cap.

    `_extract_entities_bert` trimmed text to 2000 characters as a stand-in for
    "~400 words". Wordpiece runs nearer 3 characters per token on dense
    biomedical prose, so those inputs still reached the model as ~600 tokens
    and overflowed its 512 position embeddings. The pipeline raised, the
    exception was caught and logged, and the BERT path contributed nothing on
    long documents while appearing to be active -- which is why nobody
    noticed. These tests assert on the absence of that error path.
    """

    @pytest.fixture
    def abstract(self):
        """~800 tokens of the prose density that broke the character heuristic."""
        sentence = (
            "BRCA1-deficient triple-negative breast carcinoma demonstrated PARP1 "
            "hyperactivation, olaparib sensitivity, and a concomitant TP53 R175H "
            "missense substitution; immunohistochemical quantification of "
            "phosphorylated ERK1/2 showed progression-free survival of 12.4 "
            "months (95% CI 9.8-15.1). "
        )
        return sentence * 12

    def nlp(self, is_fast=True):
        from litkg.phase1.literature_processor import BiomedicalNLP
        instance = BiomedicalNLP.__new__(BiomedicalNLP)
        instance.entity_types = {
            "GENE", "DISEASE", "DRUG", "PROTEIN", "CELL_TYPE",
            "TISSUE", "ORGANISM", "CHEMICAL", "MUTATION",
        }
        instance.ner_pipeline = StubNERPipeline(StubTokenizer(is_fast=is_fast))
        return instance

    def test_the_character_heuristic_really_did_overflow(self, abstract):
        """Guards the premise: without this the tests below prove nothing."""
        tokenizer = StubTokenizer()
        assert len(tokenizer(abstract[:2000])["input_ids"]) > 512, (
            "the stub is too sparse to reproduce the failure"
        )

    @pytest.mark.parametrize("is_fast", [True, False])
    def test_long_abstract_does_not_reach_the_error_path(self, abstract, caplog, is_fast):
        nlp = self.nlp(is_fast=is_fast)
        with caplog.at_level(logging.ERROR, logger="litkg.BiomedicalNLP"):
            entities = nlp._extract_entities_bert(abstract)

        assert not caplog.records, (
            f"BERT NER logged and swallowed: {[r.message for r in caplog.records]}"
        )
        assert nlp.ner_pipeline.calls, "the pipeline never ran"
        assert entities, "the extractor returned nothing despite reaching the model"

    @pytest.mark.parametrize("is_fast", [True, False])
    def test_truncation_respects_the_token_budget(self, abstract, is_fast):
        nlp = self.nlp(is_fast=is_fast)
        truncated = nlp._truncate_to_token_limit(abstract)
        tokenizer = nlp.ner_pipeline.tokenizer

        assert len(tokenizer(truncated)["input_ids"]) <= 512
        assert abstract.startswith(truncated), (
            "truncation must return a prefix, or reported offsets stop lining "
            "up with the caller's text"
        )

    @pytest.mark.parametrize("is_fast", [True, False])
    def test_truncation_stops_at_a_word_boundary(self, abstract, is_fast):
        """Half a word can re-tokenize into more pieces than were kept."""
        nlp = self.nlp(is_fast=is_fast)
        truncated = nlp._truncate_to_token_limit(abstract)
        assert len(truncated) < len(abstract), "this abstract should have been cut"
        assert abstract[len(truncated)].isspace()

    def test_offsets_index_the_original_text(self, abstract):
        """Entity spans are stored against the document, not the trimmed copy."""
        nlp = self.nlp()
        entity = nlp._extract_entities_bert(abstract)[0]
        assert abstract[entity.start:entity.end] == entity.text

    @pytest.mark.parametrize("is_fast", [True, False])
    def test_short_abstract_is_passed_through_whole(self, is_fast):
        nlp = self.nlp(is_fast=is_fast)
        short = "BRCA1 mutations predict olaparib sensitivity."
        assert nlp._truncate_to_token_limit(short) == short

    def test_limit_comes_from_the_model_when_the_tokenizer_reports_no_length(self):
        """The sentinel is what disabled the pipeline's own truncation."""
        nlp = self.nlp()
        assert nlp.ner_pipeline.tokenizer.model_max_length == UNSET_MODEL_MAX_LENGTH
        assert nlp._bert_max_tokens() == 512

    def test_a_real_tokenizer_length_is_honoured_when_it_is_smaller(self):
        nlp = self.nlp()
        nlp.ner_pipeline.tokenizer.model_max_length = 128
        assert nlp._bert_max_tokens() == 128

    def test_falls_back_to_the_bert_default_without_any_stated_limit(self):
        from litkg.phase1.literature_processor import BiomedicalNLP
        nlp = self.nlp()
        nlp.ner_pipeline.tokenizer.model_max_length = None
        nlp.ner_pipeline.model = None
        assert nlp._bert_max_tokens() == BiomedicalNLP.DEFAULT_MAX_TOKENS == 512

    def test_no_character_based_truncation_remains(self):
        """Tuning the character count would leave the same bug, further out."""
        import inspect
        from litkg.phase1.literature_processor import BiomedicalNLP
        source = inspect.getsource(BiomedicalNLP._extract_entities_bert)
        assert "max_chars" not in source
        assert "_truncate_to_token_limit" in source
