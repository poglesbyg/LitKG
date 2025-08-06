"""
Unified LLM Interface for Multiple Providers

This module provides a unified interface for working with different LLM providers:
- Ollama (local open-source models)
- OpenAI API (GPT models)
- Anthropic API (Claude models)
- OpenAI-compatible endpoints (LocalAI, vLLM, etc.)

Features:
- Automatic provider selection and fallback
- Cost optimization and usage tracking
- Performance monitoring
- Biomedical task optimization
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

# Provider-specific imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Local imports
from ..utils.logging import LoggerMixin
from .ollama_integration import OllamaLLM, OllamaManager


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a model."""
    provider: LLMProvider
    model_name: str
    max_tokens: int
    supports_system_prompt: bool
    supports_streaming: bool
    cost_per_1k_tokens: float
    performance_tier: str  # "low", "medium", "high", "premium"
    biomedical_score: float  # 0-1 scale
    local_inference: bool
    memory_requirements: Optional[str] = None


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, Any]
    response_time: float
    cost: float
    metadata: Dict[str, Any]


class BiomedicalLLMInterface(LoggerMixin):
    """
    Unified interface for biomedical LLM tasks across multiple providers.
    """
    
    def __init__(
        self,
        preferred_providers: List[LLMProvider] = None,
        fallback_enabled: bool = True,
        cost_optimization: bool = True
    ):
        self.preferred_providers = preferred_providers or [
            LLMProvider.OLLAMA,  # Local first for privacy
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC
        ]
        self.fallback_enabled = fallback_enabled
        self.cost_optimization = cost_optimization
        
        # Initialize clients
        self.clients = {}
        self._initialize_clients()
        
        # Model capabilities database
        self.model_capabilities = self._load_model_capabilities()
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_cost": 0.0,
            "provider_usage": {},
            "model_usage": {}
        }
        
        self.logger.info("Initialized BiomedicalLLMInterface")
    
    def _initialize_clients(self):
        """Initialize available LLM clients."""
        # Ollama
        try:
            self.clients[LLMProvider.OLLAMA] = OllamaManager()
            self.logger.info("Initialized Ollama client")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama: {e}")
        
        # OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.clients[LLMProvider.OPENAI] = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                self.logger.info("Initialized OpenAI client")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.clients[LLMProvider.ANTHROPIC] = Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                self.logger.info("Initialized Anthropic client")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic: {e}")
        
        # OpenAI-compatible endpoints
        if os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
            try:
                self.clients[LLMProvider.OPENAI_COMPATIBLE] = OpenAI(
                    base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
                    api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "dummy")
                )
                self.logger.info("Initialized OpenAI-compatible client")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI-compatible client: {e}")
    
    def _load_model_capabilities(self) -> Dict[str, ModelCapabilities]:
        """Load model capabilities database."""
        return {
            # Ollama models
            "llama3.1:8b": ModelCapabilities(
                provider=LLMProvider.OLLAMA,
                model_name="llama3.1:8b",
                max_tokens=8192,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,  # Free for local
                performance_tier="high",
                biomedical_score=0.7,
                local_inference=True,
                memory_requirements="8GB"
            ),
            "llama3.1:70b": ModelCapabilities(
                provider=LLMProvider.OLLAMA,
                model_name="llama3.1:70b",
                max_tokens=8192,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,
                performance_tier="premium",
                biomedical_score=0.85,
                local_inference=True,
                memory_requirements="48GB"
            ),
            "mistral:7b": ModelCapabilities(
                provider=LLMProvider.OLLAMA,
                model_name="mistral:7b",
                max_tokens=8192,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,
                performance_tier="medium",
                biomedical_score=0.6,
                local_inference=True,
                memory_requirements="6GB"
            ),
            # OpenAI models
            "gpt-3.5-turbo": ModelCapabilities(
                provider=LLMProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                max_tokens=4096,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.002,
                performance_tier="high",
                biomedical_score=0.75,
                local_inference=False
            ),
            "gpt-4": ModelCapabilities(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                max_tokens=8192,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.03,
                performance_tier="premium",
                biomedical_score=0.9,
                local_inference=False
            ),
            "gpt-4-turbo": ModelCapabilities(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4-turbo",
                max_tokens=128000,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.01,
                performance_tier="premium",
                biomedical_score=0.9,
                local_inference=False
            ),
            # Anthropic models
            "claude-3-sonnet-20240229": ModelCapabilities(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                max_tokens=4096,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.015,
                performance_tier="premium",
                biomedical_score=0.85,
                local_inference=False
            ),
            "claude-3-haiku-20240307": ModelCapabilities(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                max_tokens=4096,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.0025,
                performance_tier="high",
                biomedical_score=0.8,
                local_inference=False
            )
        }
    
    def select_best_model(
        self,
        task: str,
        max_cost: float = None,
        require_local: bool = False,
        min_biomedical_score: float = 0.6
    ) -> Optional[str]:
        """
        Select the best model for a given task and constraints.
        
        Args:
            task: Task type (e.g., "entity_extraction", "hypothesis_generation")
            max_cost: Maximum cost per 1k tokens
            require_local: Whether to require local inference
            min_biomedical_score: Minimum biomedical capability score
            
        Returns:
            Best model name or None if no suitable model found
        """
        candidates = []
        
        for model_name, capabilities in self.model_capabilities.items():
            # Check constraints
            if require_local and not capabilities.local_inference:
                continue
            
            if max_cost is not None and capabilities.cost_per_1k_tokens > max_cost:
                continue
            
            if capabilities.biomedical_score < min_biomedical_score:
                continue
            
            # Check if provider is available
            if capabilities.provider not in self.clients:
                continue
            
            # For Ollama, check if model is available locally
            if capabilities.provider == LLMProvider.OLLAMA:
                ollama_manager = self.clients[LLMProvider.OLLAMA]
                if not ollama_manager.check_server_status():
                    continue
                available_models = ollama_manager.list_available_models()
                if model_name not in available_models:
                    continue
            
            candidates.append((model_name, capabilities))
        
        if not candidates:
            return None
        
        # Sort by biomedical score, then by cost (lower is better for cost)
        candidates.sort(
            key=lambda x: (x[1].biomedical_score, -x[1].cost_per_1k_tokens),
            reverse=True
        )
        
        return candidates[0][0]
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        task: str = "general",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using the best available model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Specific model to use (optional)
            task: Task type for model selection
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        # Select model if not specified
        if not model:
            model = self.select_best_model(task)
            if not model:
                raise RuntimeError("No suitable model available for task")
        
        # Get model capabilities
        capabilities = self.model_capabilities.get(model)
        if not capabilities:
            raise ValueError(f"Unknown model: {model}")
        
        # Generate response based on provider
        try:
            if capabilities.provider == LLMProvider.OLLAMA:
                response = self._generate_ollama(
                    model, prompt, system_prompt, max_tokens, temperature, **kwargs
                )
            elif capabilities.provider == LLMProvider.OPENAI:
                response = self._generate_openai(
                    model, prompt, system_prompt, max_tokens, temperature, **kwargs
                )
            elif capabilities.provider == LLMProvider.ANTHROPIC:
                response = self._generate_anthropic(
                    model, prompt, system_prompt, max_tokens, temperature, **kwargs
                )
            elif capabilities.provider == LLMProvider.OPENAI_COMPATIBLE:
                response = self._generate_openai_compatible(
                    model, prompt, system_prompt, max_tokens, temperature, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {capabilities.provider}")
            
            response_time = time.time() - start_time
            
            # Calculate cost
            estimated_tokens = len(response.split()) * 1.3  # Rough estimate
            cost = (estimated_tokens / 1000) * capabilities.cost_per_1k_tokens
            
            # Update usage stats
            self._update_usage_stats(capabilities.provider, model, cost)
            
            return LLMResponse(
                content=response,
                provider=capabilities.provider,
                model=model,
                usage={"estimated_tokens": estimated_tokens},
                response_time=response_time,
                cost=cost,
                metadata={"task": task, "temperature": temperature}
            )
            
        except Exception as e:
            # Try fallback if enabled
            if self.fallback_enabled and len(self.preferred_providers) > 1:
                self.logger.warning(f"Primary model failed, trying fallback: {e}")
                return self._generate_with_fallback(
                    prompt, system_prompt, task, max_tokens, temperature, **kwargs
                )
            else:
                raise e
    
    def _generate_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate response using Ollama."""
        ollama_llm = OllamaLLM(
            model=model,
            temperature=temperature,
            biomedical_mode=True
        )
        return ollama_llm.generate(prompt, system_prompt, **kwargs)
    
    def _generate_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate response using OpenAI."""
        client = self.clients[LLMProvider.OPENAI]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate response using Anthropic."""
        client = self.clients[LLMProvider.ANTHROPIC]
        
        # Format prompt for Claude
        formatted_prompt = f"Human: {prompt}\n\nAssistant:"
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful biomedical research assistant.",
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return response.content[0].text
    
    def _generate_openai_compatible(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate response using OpenAI-compatible endpoint."""
        client = self.clients[LLMProvider.OPENAI_COMPATIBLE]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _generate_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str],
        task: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Generate response with fallback to alternative providers."""
        for provider in self.preferred_providers:
            if provider not in self.clients:
                continue
            
            try:
                # Find best model for this provider
                candidates = [
                    model for model, caps in self.model_capabilities.items()
                    if caps.provider == provider
                ]
                
                if not candidates:
                    continue
                
                # Select best candidate
                model = self.select_best_model(task)
                if not model or self.model_capabilities[model].provider != provider:
                    continue
                
                return self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    task=task,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
            except Exception as e:
                self.logger.warning(f"Fallback provider {provider} failed: {e}")
                continue
        
        raise RuntimeError("All providers failed")
    
    def _update_usage_stats(self, provider: LLMProvider, model: str, cost: float):
        """Update usage statistics."""
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_cost"] += cost
        
        provider_key = provider.value
        if provider_key not in self.usage_stats["provider_usage"]:
            self.usage_stats["provider_usage"][provider_key] = {
                "requests": 0,
                "cost": 0.0
            }
        
        self.usage_stats["provider_usage"][provider_key]["requests"] += 1
        self.usage_stats["provider_usage"][provider_key]["cost"] += cost
        
        if model not in self.usage_stats["model_usage"]:
            self.usage_stats["model_usage"][model] = {
                "requests": 0,
                "cost": 0.0
            }
        
        self.usage_stats["model_usage"][model]["requests"] += 1
        self.usage_stats["model_usage"][model]["cost"] += cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_stats.copy()
    
    def estimate_cost(self, prompt: str, model: str) -> float:
        """Estimate cost for a prompt with a specific model."""
        capabilities = self.model_capabilities.get(model)
        if not capabilities:
            return 0.0
        
        estimated_tokens = len(prompt.split()) * 1.5  # Include response estimate
        return (estimated_tokens / 1000) * capabilities.cost_per_1k_tokens


class UnifiedLLMManager(LoggerMixin):
    """
    High-level manager for unified LLM operations across the LitKG system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        
        # Initialize unified interface
        self.llm_interface = BiomedicalLLMInterface()
        
        # Task-specific configurations
        self.task_configs = {
            "entity_extraction": {
                "temperature": 0.1,
                "max_tokens": 500,
                "system_prompt": "You are an expert biomedical entity extractor. Extract genes, proteins, diseases, drugs, and biological processes from scientific text."
            },
            "relation_extraction": {
                "temperature": 0.1,
                "max_tokens": 800,
                "system_prompt": "You are an expert at identifying biological relationships. Extract meaningful relationships between biomedical entities."
            },
            "hypothesis_generation": {
                "temperature": 0.3,
                "max_tokens": 1200,
                "system_prompt": "You are a creative biomedical researcher. Generate testable hypotheses based on biological observations and context."
            },
            "validation": {
                "temperature": 0.1,
                "max_tokens": 1000,
                "system_prompt": "You are a rigorous biomedical validator. Assess biological plausibility and provide evidence-based evaluations."
            },
            "literature_analysis": {
                "temperature": 0.2,
                "max_tokens": 1500,
                "system_prompt": "You are a biomedical literature analyst. Provide comprehensive analysis of scientific papers and findings."
            }
        }
        
        self.logger.info("Initialized UnifiedLLMManager")
    
    def setup_local_models(self, memory_limit: str = "8GB") -> Dict[str, bool]:
        """Set up local models for biomedical tasks."""
        self.logger.info("Setting up local biomedical models...")
        
        # Check if Ollama is available
        if LLMProvider.OLLAMA not in self.llm_interface.clients:
            self.logger.warning("Ollama not available for local model setup")
            return {}
        
        ollama_manager = self.llm_interface.clients[LLMProvider.OLLAMA]
        
        # Install recommended models
        return ollama_manager.setup_biomedical_models(memory_limit)
    
    def process_biomedical_task(
        self,
        task: str,
        input_data: Union[str, Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Process a biomedical task using the best available model.
        
        Args:
            task: Task type (entity_extraction, relation_extraction, etc.)
            input_data: Input text or structured data
            model: Specific model to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with results
        """
        if task not in self.task_configs:
            raise ValueError(f"Unknown task: {task}")
        
        config = self.task_configs[task]
        
        # Format input
        if isinstance(input_data, str):
            prompt = input_data
        else:
            prompt = self._format_structured_input(task, input_data)
        
        # Merge configurations
        params = {**config, **kwargs}
        
        return self.llm_interface.generate(
            prompt=prompt,
            model=model,
            task=task,
            **params
        )
    
    def _format_structured_input(self, task: str, data: Dict[str, Any]) -> str:
        """Format structured input data into prompts."""
        if task == "relation_extraction":
            text = data.get("text", "")
            entities = data.get("entities", [])
            return f"Text: {text}\nEntities: {', '.join(entities)}\n\nExtract relationships:"
        
        elif task == "hypothesis_generation":
            context = data.get("context", "")
            observation = data.get("observation", "")
            return f"Context: {context}\nObservation: {observation}\n\nGenerate hypothesis:"
        
        elif task == "validation":
            hypothesis = data.get("hypothesis", "")
            evidence = data.get("evidence", "")
            return f"Hypothesis: {hypothesis}\nEvidence: {evidence}\n\nValidate:"
        
        else:
            # Default formatting
            return str(data)
    
    def batch_process(
        self,
        task: str,
        inputs: List[Union[str, Dict[str, Any]]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Process multiple inputs for the same task."""
        results = []
        
        for input_data in inputs:
            try:
                result = self.process_biomedical_task(
                    task=task,
                    input_data=input_data,
                    model=model,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing input: {e}")
                # Create error response
                error_response = LLMResponse(
                    content=f"Error: {str(e)}",
                    provider=LLMProvider.OLLAMA,  # Default
                    model=model or "unknown",
                    usage={},
                    response_time=0.0,
                    cost=0.0,
                    metadata={"error": True}
                )
                results.append(error_response)
        
        return results
    
    def get_model_recommendations(
        self,
        task: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get model recommendations for a specific task."""
        constraints = constraints or {}
        
        # Get all suitable models
        candidates = []
        
        for model_name, capabilities in self.llm_interface.model_capabilities.items():
            # Check basic constraints
            max_cost = constraints.get("max_cost")
            if max_cost and capabilities.cost_per_1k_tokens > max_cost:
                continue
            
            require_local = constraints.get("require_local", False)
            if require_local and not capabilities.local_inference:
                continue
            
            min_biomedical_score = constraints.get("min_biomedical_score", 0.6)
            if capabilities.biomedical_score < min_biomedical_score:
                continue
            
            candidates.append((model_name, capabilities))
        
        # Sort by biomedical score and performance
        candidates.sort(
            key=lambda x: (x[1].biomedical_score, x[1].performance_tier == "premium"),
            reverse=True
        )
        
        return [model[0] for model in candidates[:5]]  # Top 5 recommendations