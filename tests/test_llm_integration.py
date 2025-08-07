"""
Tests for LLM integration components (Ollama, unified interface, model selection).
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict

from litkg.llm_integration.unified_llm_interface import (
    UnifiedLLMManager, BiomedicalLLMInterface, LLMProvider, LLMResponse
)
from litkg.llm_integration.ollama_integration import OllamaManager
from litkg.llm_integration.model_selection import ModelSelector, ModelRecommendation
from litkg.llm_integration.biomedical_prompts import BiomedicalPromptTemplates


class TestUnifiedLLMInterface:
    """Test unified LLM interface components."""
    
    def test_llm_provider_enum(self):
        """Test LLMProvider enumeration."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.HUGGINGFACE.value == "huggingface"
        assert LLMProvider.OLLAMA.value == "ollama"
    
    def test_llm_response_dataclass(self):
        """Test LLMResponse dataclass."""
        response = LLMResponse(
            content="BRCA1 is a tumor suppressor gene",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"temperature": 0.7, "max_tokens": 100}
        )
        
        assert response.content == "BRCA1 is a tumor suppressor gene"
        assert response.provider == LLMProvider.OPENAI
        assert response.model == "gpt-3.5-turbo"
        assert response.usage["total_tokens"] == 30
    
    def test_biomedical_llm_interface_init(self):
        """Test BiomedicalLLMInterface initialization."""
        interface = BiomedicalLLMInterface()
        
        assert hasattr(interface, 'logger')
        assert hasattr(interface, 'clients')
        assert hasattr(interface, 'usage_stats')
    
    def test_client_initialization(self):
        """Test LLM client initialization."""
        interface = BiomedicalLLMInterface()
        
        # Mock environment variables and client initialization
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('litkg.llm_integration.unified_llm_interface.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                interface._initialize_openai_client()
                
                assert LLMProvider.OPENAI in interface.clients
                assert interface.clients[LLLProvider.OPENAI] == mock_client
    
    def test_model_selection(self):
        """Test model selection logic."""
        interface = BiomedicalLLMInterface()
        
        # Mock available models
        interface.available_models = {
            LLMProvider.OPENAI: ["gpt-3.5-turbo", "gpt-4"],
            LLMProvider.OLLAMA: ["llama3.1:8b", "llama3.1:70b"]
        }
        
        # Test model selection for different tasks
        model_info = interface.select_best_model("literature_analysis", max_cost=0.01)
        
        assert model_info is not None
        if model_info:
            assert "provider" in model_info
            assert "model" in model_info
    
    def test_generate_text(self):
        """Test text generation."""
        interface = BiomedicalLLMInterface()
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated text about BRCA1"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_client.chat.completions.create.return_value = mock_response
        
        interface.clients[LLMProvider.OPENAI] = mock_client
        
        response = interface.generate(
            prompt="Tell me about BRCA1",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo"
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Generated text about BRCA1"
        assert response.provider == LLMProvider.OPENAI
    
    def test_usage_tracking(self):
        """Test usage statistics tracking."""
        interface = BiomedicalLLMInterface()
        
        # Simulate API calls
        interface._update_usage_stats(LLMProvider.OPENAI, "gpt-3.5-turbo", 30, 0.001)
        interface._update_usage_stats(LLMProvider.OPENAI, "gpt-3.5-turbo", 50, 0.002)
        
        stats = interface.get_usage_stats()
        
        assert LLMProvider.OPENAI.value in stats
        assert "gpt-3.5-turbo" in stats[LLMProvider.OPENAI.value]
        assert stats[LLMProvider.OPENAI.value]["gpt-3.5-turbo"]["total_tokens"] == 80
        assert stats[LLMProvider.OPENAI.value]["gpt-3.5-turbo"]["total_cost"] == 0.003
    
    def test_error_handling(self):
        """Test error handling in LLM calls."""
        interface = BiomedicalLLMInterface()
        
        # Mock client that raises an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        interface.clients[LLMProvider.OPENAI] = mock_client
        
        with pytest.raises(Exception):
            interface.generate(
                prompt="Test prompt",
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo"
            )
    
    def test_unified_llm_manager_init(self):
        """Test UnifiedLLMManager initialization."""
        manager = UnifiedLLMManager()
        
        assert hasattr(manager, 'llm_interface')
        assert hasattr(manager, 'model_selector')
        assert hasattr(manager, 'prompt_templates')
    
    def test_biomedical_task_processing(self):
        """Test biomedical task processing."""
        manager = UnifiedLLMManager()
        
        # Mock the LLM interface
        mock_response = LLMResponse(
            content="BRCA1 is involved in DNA repair",
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            usage={"total_tokens": 30},
            metadata={}
        )
        
        with patch.object(manager.llm_interface, 'generate') as mock_generate:
            mock_generate.return_value = mock_response
            
            with patch.object(manager.llm_interface, 'select_best_model') as mock_select:
                mock_select.return_value = {
                    "provider": LLMProvider.OPENAI,
                    "model": "gpt-3.5-turbo"
                }
                
                response = manager.process_biomedical_task(
                    task="literature_analysis",
                    input_data="Analyze BRCA1 function"
                )
                
                assert isinstance(response, LLMResponse)
                assert "BRCA1" in response.content
    
    def test_batch_processing(self):
        """Test batch processing of multiple tasks."""
        manager = UnifiedLLMManager()
        
        tasks = [
            {"task": "literature_analysis", "input": "BRCA1 function"},
            {"task": "hypothesis_generation", "input": "BRCA1 mutations"},
            {"task": "experimental_design", "input": "Test BRCA1 role"}
        ]
        
        # Mock responses
        mock_responses = [
            LLMResponse("Response 1", LLMProvider.OPENAI, "gpt-3.5", {}, {}),
            LLMResponse("Response 2", LLMProvider.OPENAI, "gpt-3.5", {}, {}),
            LLMResponse("Response 3", LLMProvider.OPENAI, "gpt-3.5", {}, {})
        ]
        
        with patch.object(manager, 'process_biomedical_task') as mock_process:
            mock_process.side_effect = mock_responses
            
            results = manager.process_batch(tasks)
            
            assert len(results) == 3
            assert all(isinstance(r, LLMResponse) for r in results)


class TestOllamaIntegration:
    """Test Ollama integration components."""
    
    def test_ollama_manager_init(self):
        """Test OllamaManager initialization."""
        manager = OllamaManager()
        
        assert hasattr(manager, 'logger')
        assert hasattr(manager, 'client')
        assert manager.base_url == "http://localhost:11434"
    
    def test_ollama_client_initialization(self):
        """Test Ollama client initialization."""
        with patch('litkg.llm_integration.ollama_integration.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            manager = OllamaManager(base_url="http://test:11434")
            
            assert manager.client == mock_client
            mock_client_class.assert_called_with(host="http://test:11434")
    
    def test_list_models(self):
        """Test listing available Ollama models."""
        manager = OllamaManager()
        
        # Mock client response
        mock_models = {
            "models": [
                {"name": "llama3.1:8b", "size": 4000000000},
                {"name": "llama3.1:70b", "size": 40000000000}
            ]
        }
        
        with patch.object(manager.client, 'list') as mock_list:
            mock_list.return_value = mock_models
            
            models = manager.list_models()
            
            assert len(models) == 2
            assert "llama3.1:8b" in models
            assert "llama3.1:70b" in models
    
    def test_pull_model(self):
        """Test pulling a model from Ollama."""
        manager = OllamaManager()
        
        # Mock pull operation
        with patch.object(manager.client, 'pull') as mock_pull:
            mock_pull.return_value = [
                {"status": "downloading", "completed": 50, "total": 100},
                {"status": "success"}
            ]
            
            success = manager.pull_model("llama3.1:8b")
            
            assert success is True
            mock_pull.assert_called_once_with("llama3.1:8b")
    
    def test_generate_text(self):
        """Test text generation with Ollama."""
        manager = OllamaManager()
        
        # Mock generation response
        mock_response = {
            "response": "BRCA1 is a tumor suppressor gene that plays a crucial role in DNA repair.",
            "model": "llama3.1:8b",
            "created_at": "2023-01-01T00:00:00Z",
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        
        with patch.object(manager.client, 'generate') as mock_generate:
            mock_generate.return_value = mock_response
            
            response = manager.generate(
                model="llama3.1:8b",
                prompt="Tell me about BRCA1",
                temperature=0.7
            )
            
            assert response["response"] == mock_response["response"]
            assert response["model"] == "llama3.1:8b"
    
    def test_chat_interface(self):
        """Test chat interface with Ollama."""
        manager = OllamaManager()
        
        # Mock chat response
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "BRCA1 is a tumor suppressor gene."
            },
            "model": "llama3.1:8b",
            "created_at": "2023-01-01T00:00:00Z",
            "done": True
        }
        
        with patch.object(manager.client, 'chat') as mock_chat:
            mock_chat.return_value = mock_response
            
            response = manager.chat(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "What is BRCA1?"}]
            )
            
            assert response["message"]["content"] == "BRCA1 is a tumor suppressor gene."
    
    def test_model_info(self):
        """Test getting model information."""
        manager = OllamaManager()
        
        # Mock model info
        mock_info = {
            "modelfile": "FROM llama3.1:8b\nPARAMETER temperature 0.8",
            "parameters": "temperature 0.8\nstop \"<|end|>\"",
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8B",
                "quantization_level": "Q4_0"
            }
        }
        
        with patch.object(manager.client, 'show') as mock_show:
            mock_show.return_value = mock_info
            
            info = manager.get_model_info("llama3.1:8b")
            
            assert info["details"]["parameter_size"] == "8B"
            assert info["details"]["family"] == "llama"
    
    def test_error_handling(self):
        """Test error handling in Ollama operations."""
        manager = OllamaManager()
        
        # Test connection error
        with patch.object(manager.client, 'list') as mock_list:
            mock_list.side_effect = Exception("Connection failed")
            
            models = manager.list_models()
            
            assert models == []  # Should return empty list on error
    
    def test_streaming_generation(self):
        """Test streaming text generation."""
        manager = OllamaManager()
        
        # Mock streaming response
        mock_stream = [
            {"response": "BRCA1", "done": False},
            {"response": " is a", "done": False},
            {"response": " tumor suppressor gene", "done": True}
        ]
        
        with patch.object(manager.client, 'generate') as mock_generate:
            mock_generate.return_value = iter(mock_stream)
            
            responses = list(manager.generate_stream(
                model="llama3.1:8b",
                prompt="Tell me about BRCA1"
            ))
            
            assert len(responses) == 3
            assert responses[-1]["done"] is True


class TestModelSelection:
    """Test model selection components."""
    
    def test_model_recommendation_dataclass(self):
        """Test ModelRecommendation dataclass."""
        recommendation = ModelRecommendation(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            confidence=0.85,
            estimated_cost=0.002,
            estimated_time=1.5,
            reasoning="Good balance of cost and performance for this task"
        )
        
        assert recommendation.provider == LLMProvider.OPENAI
        assert recommendation.model == "gpt-3.5-turbo"
        assert recommendation.confidence == 0.85
    
    def test_model_selector_init(self):
        """Test ModelSelector initialization."""
        selector = ModelSelector()
        
        assert hasattr(selector, 'logger')
        assert hasattr(selector, 'model_capabilities')
        assert hasattr(selector, 'cost_models')
    
    def test_task_based_selection(self):
        """Test model selection based on task type."""
        selector = ModelSelector()
        
        # Mock model capabilities
        selector.model_capabilities = {
            "gpt-3.5-turbo": {
                "provider": LLMProvider.OPENAI,
                "strengths": ["general_qa", "literature_analysis"],
                "cost_per_token": 0.000002,
                "max_tokens": 4096,
                "performance_score": 0.8
            },
            "llama3.1:70b": {
                "provider": LLMProvider.OLLAMA,
                "strengths": ["reasoning", "analysis"],
                "cost_per_token": 0.0,  # Local model
                "max_tokens": 8192,
                "performance_score": 0.85
            }
        }
        
        recommendation = selector.select_model_for_task(
            task_type="literature_analysis",
            max_cost=0.01,
            require_local=False
        )
        
        assert isinstance(recommendation, ModelRecommendation)
        assert recommendation.estimated_cost <= 0.01
    
    def test_cost_optimization(self):
        """Test cost-based model optimization."""
        selector = ModelSelector()
        
        # Test with strict cost constraint
        recommendation = selector.optimize_for_cost(
            task_type="simple_qa",
            max_cost=0.001,
            min_quality=0.7
        )
        
        if recommendation:
            assert recommendation.estimated_cost <= 0.001
    
    def test_performance_optimization(self):
        """Test performance-based model optimization."""
        selector = ModelSelector()
        
        recommendation = selector.optimize_for_performance(
            task_type="complex_reasoning",
            min_performance=0.8
        )
        
        if recommendation:
            assert recommendation.confidence >= 0.8
    
    def test_local_model_preference(self):
        """Test preference for local models."""
        selector = ModelSelector()
        
        recommendation = selector.select_model_for_task(
            task_type="hypothesis_generation",
            require_local=True
        )
        
        if recommendation:
            assert recommendation.provider in [LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE]
    
    def test_batch_selection(self):
        """Test batch model selection for multiple tasks."""
        selector = ModelSelector()
        
        tasks = [
            {"task": "literature_analysis", "max_cost": 0.01},
            {"task": "hypothesis_generation", "require_local": True},
            {"task": "simple_qa", "max_cost": 0.001}
        ]
        
        recommendations = selector.select_models_for_batch(tasks)
        
        assert len(recommendations) == 3
        assert all(isinstance(r, ModelRecommendation) for r in recommendations if r)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        selector = ModelSelector()
        
        models = ["gpt-3.5-turbo", "gpt-4", "llama3.1:70b"]
        task = "biomedical_analysis"
        
        comparison = selector.compare_models(models, task)
        
        assert len(comparison) <= len(models)
        assert all("model" in comp for comp in comparison)
        assert all("score" in comp for comp in comparison)


class TestBiomedicalPrompts:
    """Test biomedical prompt templates."""
    
    def test_prompt_templates_init(self):
        """Test BiomedicalPromptTemplates initialization."""
        templates = BiomedicalPromptTemplates()
        
        assert hasattr(templates, 'templates')
        assert len(templates.templates) > 0
    
    def test_literature_analysis_prompt(self):
        """Test literature analysis prompt template."""
        templates = BiomedicalPromptTemplates()
        
        prompt = templates.get_literature_analysis_prompt(
            paper_title="BRCA1 mutations in breast cancer",
            abstract="This study examines BRCA1 mutations...",
            analysis_focus="mutation_impact"
        )
        
        assert "BRCA1 mutations in breast cancer" in prompt
        assert "mutation_impact" in prompt
        assert len(prompt) > 100  # Should be a substantial prompt
    
    def test_hypothesis_generation_prompt(self):
        """Test hypothesis generation prompt template."""
        templates = BiomedicalPromptTemplates()
        
        prompt = templates.get_hypothesis_generation_prompt(
            context="BRCA1 mutations and DNA repair",
            domain="cancer_genetics",
            evidence=["BRCA1 is involved in homologous recombination", "Mutations cause repair defects"]
        )
        
        assert "BRCA1 mutations" in prompt
        assert "cancer_genetics" in prompt
        assert "homologous recombination" in prompt
    
    def test_experimental_design_prompt(self):
        """Test experimental design prompt template."""
        templates = BiomedicalPromptTemplates()
        
        prompt = templates.get_experimental_design_prompt(
            hypothesis="BRCA1 mutations increase PARP inhibitor sensitivity",
            research_question="How do BRCA1 mutations affect drug sensitivity?",
            constraints=["Cell culture only", "Budget: $10,000"]
        )
        
        assert "BRCA1 mutations" in prompt
        assert "PARP inhibitor" in prompt
        assert "Cell culture only" in prompt
    
    def test_entity_extraction_prompt(self):
        """Test entity extraction prompt template."""
        templates = BiomedicalPromptTemplates()
        
        text = "BRCA1 mutations are associated with increased risk of breast and ovarian cancer."
        
        prompt = templates.get_entity_extraction_prompt(
            text=text,
            entity_types=["GENE", "DISEASE", "PHENOTYPE"]
        )
        
        assert text in prompt
        assert "GENE" in prompt
        assert "DISEASE" in prompt
    
    def test_relation_extraction_prompt(self):
        """Test relation extraction prompt template."""
        templates = BiomedicalPromptTemplates()
        
        text = "BRCA1 interacts with TP53 in DNA repair pathways."
        entities = ["BRCA1", "TP53", "DNA repair"]
        
        prompt = templates.get_relation_extraction_prompt(
            text=text,
            entities=entities
        )
        
        assert text in prompt
        assert all(entity in prompt for entity in entities)
    
    def test_validation_prompt(self):
        """Test validation prompt template."""
        templates = BiomedicalPromptTemplates()
        
        hypothesis = "BRCA1 mutations increase cancer susceptibility"
        evidence = ["Population studies show increased risk", "Functional studies confirm repair defects"]
        
        prompt = templates.get_validation_prompt(
            hypothesis=hypothesis,
            evidence=evidence,
            validation_criteria=["biological_plausibility", "statistical_significance"]
        )
        
        assert hypothesis in prompt
        assert "Population studies" in prompt
        assert "biological_plausibility" in prompt
    
    def test_custom_prompt_creation(self):
        """Test custom prompt creation."""
        templates = BiomedicalPromptTemplates()
        
        custom_prompt = templates.create_custom_prompt(
            template="Analyze the {topic} in the context of {domain}. Focus on {aspect}.",
            variables={
                "topic": "BRCA1 mutations",
                "domain": "cancer genetics",
                "aspect": "therapeutic implications"
            }
        )
        
        assert "BRCA1 mutations" in custom_prompt
        assert "cancer genetics" in custom_prompt
        assert "therapeutic implications" in custom_prompt
    
    def test_prompt_optimization(self):
        """Test prompt optimization for different models."""
        templates = BiomedicalPromptTemplates()
        
        base_prompt = "Analyze BRCA1 function"
        
        # Optimize for different providers
        openai_prompt = templates.optimize_for_provider(base_prompt, LLMProvider.OPENAI)
        ollama_prompt = templates.optimize_for_provider(base_prompt, LLMProvider.OLLAMA)
        
        assert len(openai_prompt) >= len(base_prompt)
        assert len(ollama_prompt) >= len(base_prompt)
        # Different providers might have different optimizations
    
    def test_prompt_validation(self):
        """Test prompt validation."""
        templates = BiomedicalPromptTemplates()
        
        # Valid prompt
        valid_prompt = "Analyze the role of BRCA1 in DNA repair."
        is_valid, issues = templates.validate_prompt(valid_prompt)
        
        assert is_valid is True
        assert len(issues) == 0
        
        # Invalid prompt (too short)
        invalid_prompt = "BRCA1"
        is_valid, issues = templates.validate_prompt(invalid_prompt)
        
        assert is_valid is False
        assert len(issues) > 0


@pytest.mark.integration
class TestLLMIntegrationIntegration:
    """Integration tests for LLM components."""
    
    def test_end_to_end_llm_workflow(self):
        """Test end-to-end LLM workflow."""
        # Initialize components
        manager = UnifiedLLMManager()
        
        # Mock the workflow
        with patch.object(manager.llm_interface, 'select_best_model') as mock_select:
            mock_select.return_value = {
                "provider": LLMProvider.OPENAI,
                "model": "gpt-3.5-turbo"
            }
            
            with patch.object(manager.llm_interface, 'generate') as mock_generate:
                mock_generate.return_value = LLMResponse(
                    content="BRCA1 is a tumor suppressor gene involved in DNA repair",
                    provider=LLMProvider.OPENAI,
                    model="gpt-3.5-turbo",
                    usage={"total_tokens": 50},
                    metadata={}
                )
                
                # Process biomedical task
                response = manager.process_biomedical_task(
                    task="literature_analysis",
                    input_data="Analyze BRCA1 function in cancer"
                )
                
                assert isinstance(response, LLMResponse)
                assert "BRCA1" in response.content
                assert response.provider == LLMProvider.OPENAI
    
    def test_multi_provider_fallback(self):
        """Test fallback between different LLM providers."""
        manager = UnifiedLLMManager()
        
        # Mock primary provider failure and secondary success
        with patch.object(manager.llm_interface, 'generate') as mock_generate:
            mock_generate.side_effect = [
                Exception("Primary provider failed"),
                LLMResponse(
                    content="Fallback response",
                    provider=LLMProvider.OLLAMA,
                    model="llama3.1:8b",
                    usage={"total_tokens": 30},
                    metadata={}
                )
            ]
            
            with patch.object(manager.llm_interface, 'select_best_model') as mock_select:
                mock_select.side_effect = [
                    {"provider": LLMProvider.OPENAI, "model": "gpt-3.5-turbo"},
                    {"provider": LLMProvider.OLLAMA, "model": "llama3.1:8b"}
                ]
                
                # Should fallback to secondary provider
                response = manager.process_biomedical_task(
                    task="simple_qa",
                    input_data="What is BRCA1?"
                )
                
                assert response.content == "Fallback response"
                assert response.provider == LLMProvider.OLLAMA
    
    @pytest.mark.slow
    def test_concurrent_requests(self):
        """Test handling of concurrent LLM requests."""
        import asyncio
        import concurrent.futures
        
        manager = UnifiedLLMManager()
        
        # Mock responses
        with patch.object(manager, 'process_biomedical_task') as mock_process:
            mock_process.return_value = LLMResponse(
                content="Concurrent response",
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                usage={"total_tokens": 20},
                metadata={}
            )
            
            # Submit concurrent requests
            tasks = [
                {"task": "literature_analysis", "input": f"Query {i}"}
                for i in range(10)
            ]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(
                        manager.process_biomedical_task,
                        task["task"],
                        task["input"]
                    )
                    for task in tasks
                ]
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            assert len(results) == 10
            assert all(isinstance(r, LLMResponse) for r in results)


if __name__ == "__main__":
    pytest.main([__file__])