"""
Tests for Phase 1 components (literature processing, KG preprocessing, entity linking).
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import networkx as nx

from litkg.phase1.literature_processor import LiteratureProcessor, DocumentProcessor, EntityExtractor
from litkg.phase1.kg_preprocessor import KnowledgeGraphPreprocessor, OntologyMapper
from litkg.phase1.entity_linker import EntityLinker, FuzzyMatcher, DisambiguationEngine


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
        
        assert preprocessor.config == sample_config
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


class TestOntologyMapper:
    """Test ontology mapping utilities."""
    
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
        
        assert linker.config == sample_config
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