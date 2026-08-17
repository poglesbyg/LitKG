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
from dataclasses import dataclass, asdict, field
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


class LLMProvider(str, Enum):
    """
    Supported LLM providers.

    Subclasses str so provider identifiers compare equal to their string form,
    which keeps config files, dict keys, and enum members interchangeable.
    """
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
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
    """
    Standardized LLM response format.

    ``response_time`` and ``cost`` are telemetry recorded by the caller that
    issued the request; they default to zero so responses can be constructed
    from providers that do not report them.
    """
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BiomedicalLLMInterface(LoggerMixin):
    """
    Unified interface for biomedical LLM tasks across multiple providers.
    """
    
    def __init__(
        self,
        preferred_providers: List[LLMProvider] = None,
        fallback_enabled: bool = True,
        cost_optimization: bool = True,
        config=None
    ):
        from ..utils.config import load_config

        self.config = load_config(config)
        self.llm_config = self.config.llm

        # Provider precedence comes from config unless explicitly overridden
        if preferred_providers is not None:
            self.preferred_providers = preferred_providers
        else:
            self.preferred_providers = self._providers_from_config()

        # Default model for local inference, e.g. "qwen3:8b"
        self.default_model = self.llm_config.ollama.model

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
            "total_tokens": 0,
            "total_cost": 0.0,
            "provider_usage": {},
            "model_usage": {}
        }
        
        self.logger.info(
            f"Initialized BiomedicalLLMInterface "
            f"(providers: {[p.value for p in self.preferred_providers]}, "
            f"default model: {self.default_model})"
        )

    # Generation options each provider's SDK accepts as passthrough. Anything
    # else supplied by a caller is dropped with a warning rather than forwarded,
    # so a stray keyword surfaces here instead of as an opaque TypeError from
    # deep inside the provider client.
    PROVIDER_PASSTHROUGH_OPTIONS = {
        LLMProvider.OPENAI: {
            "top_p", "n", "stop", "seed", "presence_penalty", "frequency_penalty",
            "logit_bias", "response_format", "stream", "user",
        },
        LLMProvider.OPENAI_COMPATIBLE: {
            "top_p", "n", "stop", "seed", "presence_penalty", "frequency_penalty",
            "logit_bias", "response_format", "stream", "user",
        },
        LLMProvider.ANTHROPIC: {
            "top_p", "top_k", "stop_sequences", "stream", "metadata",
        },
        # Ollama takes only these at the top level; sampling parameters belong
        # inside `options` and are folded in by _fold_ollama_options().
        LLMProvider.OLLAMA: {
            "options", "keep_alive", "format", "system", "template", "raw",
            "context", "images", "think",
        },
    }

    # Sampling parameters Ollama expects nested under `options`
    OLLAMA_OPTION_KEYS = {
        "top_p", "top_k", "seed", "stop", "temperature", "num_predict",
        "num_ctx", "repeat_penalty", "repeat_last_n", "presence_penalty",
        "frequency_penalty", "mirostat", "mirostat_eta", "mirostat_tau",
        "tfs_z", "num_keep", "penalize_newline",
    }

    def _fold_ollama_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move Ollama sampling parameters into the nested ``options`` dict.

        Callers naturally write ``top_p=0.9``, but Ollama's client only accepts
        sampling parameters nested under ``options``. Folding them keeps the
        caller-facing API flat without silently discarding a real request.
        """
        folded = dict(kwargs)
        options = dict(folded.pop("options", {}) or {})

        for key in list(folded):
            if key in self.OLLAMA_OPTION_KEYS:
                options[key] = folded.pop(key)

        if options:
            folded["options"] = options

        return folded

    def _filter_provider_kwargs(
        self,
        provider: LLMProvider,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Keep only the generation options this provider's SDK understands.

        Args:
            provider: Provider that will receive the call.
            kwargs: Caller-supplied extra options.

        Returns:
            The subset safe to forward. Unrecognized keys are logged and dropped.
        """
        # Ollama nests sampling parameters, so normalize before filtering
        if provider == LLMProvider.OLLAMA:
            kwargs = self._fold_ollama_options(kwargs)

        allowed = self.PROVIDER_PASSTHROUGH_OPTIONS.get(provider, set())

        accepted = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = sorted(set(kwargs) - set(accepted))

        if dropped:
            self.logger.warning(
                f"Dropping option(s) {dropped} not supported by "
                f"{provider.value}; supported: {sorted(allowed)}"
            )

        return accepted

    def _providers_from_config(self) -> List[LLMProvider]:
        """Resolve the configured provider_order into LLMProvider members."""
        providers = []
        for name in self.llm_config.provider_order:
            try:
                providers.append(LLMProvider(name))
            except ValueError:
                self.logger.warning(f"Unknown provider in provider_order: {name!r}")

        if not providers:
            self.logger.warning("No valid providers configured; defaulting to Ollama")
            providers = [LLMProvider.OLLAMA]

        return providers

    def _initialize_clients(self):
        """Initialize every available LLM client."""
        self._initialize_ollama_client()
        self._initialize_openai_client()
        self._initialize_anthropic_client()
        self._initialize_openai_compatible_client()

    def _initialize_ollama_client(self) -> bool:
        """Initialize the Ollama client at the configured host. True on success."""
        try:
            self.clients[LLMProvider.OLLAMA] = OllamaManager(
                base_url=self.llm_config.ollama.host,
                timeout=self.llm_config.ollama.timeout,
            )
            self.logger.info(
                f"Initialized Ollama client at {self.llm_config.ollama.host}"
            )
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama: {e}")
            return False

    def _initialize_openai_client(self) -> bool:
        """Initialize the OpenAI client from OPENAI_API_KEY. Returns True on success."""
        if not OPENAI_AVAILABLE:
            self.logger.debug("OpenAI SDK not installed; skipping")
            return False
        if not os.getenv("OPENAI_API_KEY"):
            self.logger.debug("OPENAI_API_KEY not set; skipping OpenAI")
            return False

        try:
            self.clients[LLMProvider.OPENAI] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.logger.info("Initialized OpenAI client")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI: {e}")
            return False

    def _initialize_anthropic_client(self) -> bool:
        """Initialize the Anthropic client from ANTHROPIC_API_KEY. Returns True on success."""
        if not ANTHROPIC_AVAILABLE:
            self.logger.debug("Anthropic SDK not installed; skipping")
            return False
        if not os.getenv("ANTHROPIC_API_KEY"):
            self.logger.debug("ANTHROPIC_API_KEY not set; skipping Anthropic")
            return False

        try:
            self.clients[LLMProvider.ANTHROPIC] = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.logger.info("Initialized Anthropic client")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize Anthropic: {e}")
            return False

    def _initialize_openai_compatible_client(self) -> bool:
        """Initialize an OpenAI-compatible endpoint client. Returns True on success."""
        if not OPENAI_AVAILABLE or not os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
            return False

        try:
            self.clients[LLMProvider.OPENAI_COMPATIBLE] = OpenAI(
                base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
                api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "dummy")
            )
            self.logger.info("Initialized OpenAI-compatible client")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI-compatible client: {e}")
            return False


    def _load_model_capabilities(self) -> Dict[str, ModelCapabilities]:
        """Load model capabilities database."""
        return {
            # Ollama models
            "qwen3:8b": ModelCapabilities(
                provider=LLMProvider.OLLAMA,
                model_name="qwen3:8b",
                max_tokens=32768,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,  # Free for local
                performance_tier="high",
                biomedical_score=0.75,
                local_inference=True,
                memory_requirements="8GB"
            ),
            "qwen3-coder:30b": ModelCapabilities(
                provider=LLMProvider.OLLAMA,
                model_name="qwen3-coder:30b",
                max_tokens=32768,
                supports_system_prompt=True,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,
                performance_tier="premium",
                biomedical_score=0.7,  # Code-tuned, not biomedical-tuned
                local_inference=True,
                memory_requirements="24GB"
            ),
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
    
    def select_best_model_info(
        self,
        task: str,
        max_cost: float = None,
        require_local: bool = False,
        min_biomedical_score: float = 0.6,
        available_only: bool = True,
        respect_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best model for a task and return it with its provider.

        Args:
            task: Task type (e.g., "entity_extraction", "hypothesis_generation")
            max_cost: Maximum cost per 1k tokens
            require_local: Whether to require local inference
            min_biomedical_score: Minimum biomedical capability score
            available_only: When True, only consider models whose provider has
                an initialized client (and, for Ollama, is actually pulled).
                Set False to get a recommendation regardless of what is
                currently reachable.
            respect_default: When True, the configured default model
                (llm.ollama.model) is chosen whenever it satisfies the
                constraints. Set False to rank purely on capability scores.

        Returns:
            {"model", "provider", "capabilities"} or None if nothing qualifies.
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

            if available_only:
                # Check if provider is available
                if capabilities.provider not in self.clients:
                    continue

                # For Ollama, check if model is available locally
                if capabilities.provider == LLMProvider.OLLAMA:
                    ollama_manager = self.clients[LLMProvider.OLLAMA]
                    if not ollama_manager.check_server_status():
                        continue
                    if model_name not in ollama_manager.list_available_models():
                        continue

            candidates.append((model_name, capabilities))

        if not candidates:
            self.logger.warning(
                f"No model satisfies task={task!r}, max_cost={max_cost}, "
                f"require_local={require_local}, "
                f"min_biomedical_score={min_biomedical_score}"
            )
            return None

        # An explicitly configured default model is an instruction, not a hint:
        # if it cleared the constraints above, it wins outright. Pass
        # respect_default=False to get pure heuristic ranking instead.
        if respect_default and self.default_model:
            for name, capabilities in candidates:
                if name == self.default_model:
                    self.logger.debug(
                        f"Using configured default model {name} for task {task!r}"
                    )
                    return {
                        "model": name,
                        "provider": capabilities.provider,
                        "capabilities": capabilities,
                    }

        # Otherwise sort by biomedical score, then by cost (lower is better)
        candidates.sort(
            key=lambda x: (x[1].biomedical_score, -x[1].cost_per_1k_tokens),
            reverse=True
        )

        best_name, best_capabilities = candidates[0]
        return {
            "model": best_name,
            "provider": best_capabilities.provider,
            "capabilities": best_capabilities,
        }

    def select_best_model(
        self,
        task: str,
        max_cost: float = None,
        require_local: bool = False,
        min_biomedical_score: float = 0.6
    ) -> Optional[str]:
        """
        Select the best available model for a given task and constraints.

        See select_best_model_info() for the variant that also reports the
        provider and full capability record.

        Returns:
            Best model name or None if no suitable model found
        """
        info = self.select_best_model_info(
            task=task,
            max_cost=max_cost,
            require_local=require_local,
            min_biomedical_score=min_biomedical_score,
        )
        return info["model"] if info else None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
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
            provider: Pin generation to one provider. When given without a
                model, the best model for that provider is selected; when given
                with a model, the two must agree.
            task: Task type for model selection
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        # An explicit model/provider is an instruction; it disables fallback below
        pinned = model is not None or provider is not None

        # Select model if not specified
        if not model:
            if provider is not None:
                model = next(
                    (
                        name for name, caps in self.model_capabilities.items()
                        if caps.provider == provider
                    ),
                    None
                )
                if not model:
                    raise RuntimeError(f"No known model for provider: {provider.value}")
            else:
                model = self.select_best_model(task)
                if not model:
                    raise RuntimeError("No suitable model available for task")

        # Get model capabilities
        capabilities = self.model_capabilities.get(model)
        if not capabilities:
            raise ValueError(f"Unknown model: {model}")

        if provider is not None and capabilities.provider != provider:
            raise ValueError(
                f"Model {model!r} belongs to provider "
                f"{capabilities.provider.value!r}, not {provider.value!r}"
            )

        # Only forward options this provider actually understands
        kwargs = self._filter_provider_kwargs(capabilities.provider, kwargs)

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
            self._update_usage_stats(
                capabilities.provider, model, int(estimated_tokens), cost
            )

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
            # A caller who pinned a model or provider gets the real error, not a
            # silent switch to something they did not ask for.
            if pinned:
                self.logger.error(
                    f"Pinned model {model} failed and fallback is not applied "
                    f"to explicit selections: {e}"
                )
                raise

            if self.fallback_enabled and len(self.preferred_providers) > 1:
                self.logger.warning(f"Primary model failed, trying fallback: {e}")
                return self._generate_with_fallback(
                    prompt, system_prompt, task, max_tokens, temperature, **kwargs
                )

            raise
    
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
        
        if not response.choices:
            raise RuntimeError(f"{model} returned no choices")

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
        
        if not response.choices:
            raise RuntimeError(f"{model} returned no choices")

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
    
    def _update_usage_stats(
        self,
        provider: LLMProvider,
        model: str,
        tokens: int = 0,
        cost: float = 0.0
    ):
        """
        Record one request against the running usage totals.

        Args:
            provider: Provider that served the request.
            model: Model name.
            tokens: Total tokens consumed, when the provider reports them.
            cost: Request cost in USD.
        """
        self.usage_stats["total_requests"] += 1
        self.usage_stats["total_cost"] += cost
        self.usage_stats["total_tokens"] += tokens

        for bucket, key in (
            ("provider_usage", provider.value),
            ("model_usage", model),
        ):
            entry = self.usage_stats[bucket].setdefault(
                key, {"requests": 0, "tokens": 0, "cost": 0.0}
            )
            entry["requests"] += 1
            entry["tokens"] += tokens
            entry["cost"] += cost
    
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

        # Imported lazily: model_selection imports LLMProvider from this module.
        from .model_selection import ModelSelector
        from .biomedical_prompts import BiomedicalPromptTemplates

        self.model_selector = ModelSelector()
        self.prompt_templates = BiomedicalPromptTemplates()

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
            },
            # Fallback preset for tasks with no dedicated configuration
            "general": {
                "temperature": 0.2,
                "max_tokens": 1000,
                "system_prompt": "You are a knowledgeable biomedical research assistant. Answer accurately and cite the reasoning behind your conclusions."
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
        max_attempts: int = 2,
        **kwargs
    ) -> LLMResponse:
        """
        Process a biomedical task using the best available model.

        Args:
            task: Task type (entity_extraction, relation_extraction, etc.).
                Tasks with no preset fall back to the "general" configuration.
            input_data: Input text or structured data
            model: Specific model to use (optional). When given, no fallback to
                another model is attempted.
            max_attempts: How many models to try before giving up. Only applies
                when ``model`` is not pinned.
            **kwargs: Additional parameters

        Returns:
            LLMResponse with results

        Raises:
            RuntimeError: if every attempt fails.
        """
        # task_configs holds prompt presets, not an allowlist: an unconfigured
        # task still gets a sensible generic preset.
        if task not in self.task_configs:
            self.logger.warning(
                f"No preset for task {task!r}; using the 'general' configuration"
            )
            config = self.task_configs["general"]
        else:
            config = self.task_configs[task]

        # Format input
        if isinstance(input_data, str):
            prompt = input_data
        else:
            prompt = self._format_structured_input(task, input_data)

        # Merge configurations
        params = {**config, **kwargs}

        # Try the requested/best model, then fall back to other providers.
        attempted: List[str] = []
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            candidate = model
            if candidate is None:
                selection = self.llm_interface.select_best_model(task)
                # select_best_model may report a bare name or a full info dict
                if isinstance(selection, dict):
                    candidate = selection.get("model")
                else:
                    candidate = selection

            if candidate is not None and candidate in attempted:
                self.logger.debug(f"Model {candidate} already attempted; stopping")
                break
            attempted.append(candidate)

            try:
                return self.llm_interface.generate(
                    prompt=prompt,
                    model=candidate,
                    task=task,
                    **params
                )
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed for task {task!r} "
                    f"with model {candidate}: {e}"
                )
                # An explicitly requested model is not silently swapped out
                if model is not None:
                    break

        raise RuntimeError(
            f"All {len(attempted)} attempt(s) failed for task {task!r} "
            f"(tried: {attempted})"
        ) from last_error
    
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
    
    def process_batch(
        self,
        tasks: List[Dict[str, Any]],
        **kwargs
    ) -> List[LLMResponse]:
        """
        Process a batch of heterogeneous tasks.

        Where batch_process() runs one task over many inputs, this runs a
        different task per entry.

        Args:
            tasks: One dict per unit of work, each with a "task" key and its
                payload under "input" (or "input_data"). An optional "model"
                key pins the model for that entry.
            **kwargs: Additional parameters applied to every entry.

        Returns:
            One LLMResponse per input task, positionally aligned. Failures
            become error responses rather than aborting the batch.
        """
        results = []

        for spec in tasks:
            task = spec.get("task") or spec.get("task_type")
            input_data = spec.get("input", spec.get("input_data", ""))

            try:
                if not task:
                    raise ValueError(f"Batch entry has no task name: {spec}")

                results.append(self.process_biomedical_task(
                    task=task,
                    input_data=input_data,
                    model=spec.get("model"),
                    **kwargs
                ))
            except Exception as e:
                self.logger.error(f"Error processing batch task {task!r}: {e}")
                results.append(LLMResponse(
                    content=f"Error: {e}",
                    provider=LLMProvider.OLLAMA,
                    model=spec.get("model") or "unknown",
                    metadata={"error": True, "task": task}
                ))

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