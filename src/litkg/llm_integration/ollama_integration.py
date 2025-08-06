"""
Ollama Integration for Local LLM Inference

This module provides comprehensive integration with Ollama for running
open-source LLMs locally, including biomedical-specific models and
optimizations for the LitKG system.

Features:
- Ollama server management and health checking
- Model downloading and management
- Biomedical model recommendations
- Performance optimization for scientific text
- Integration with LangChain ecosystem
"""

import os
import json
import time
import requests
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Ollama and LangChain imports
try:
    import ollama
    from langchain_ollama import OllamaLLM as LangChainOllamaLLM
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseMessage
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# OpenAI for compatible endpoints
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Local imports
from ..utils.logging import LoggerMixin


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    size: str
    parameters: str
    family: str
    description: str
    biomedical_optimized: bool = False
    recommended_use: List[str] = None
    memory_requirements: str = "4GB"
    performance_tier: str = "medium"  # low, medium, high, premium
    
    def __post_init__(self):
        if self.recommended_use is None:
            self.recommended_use = []


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_name: str
    response_time: float
    tokens_per_second: float
    memory_usage: str
    accuracy_score: Optional[float] = None
    biomedical_score: Optional[float] = None


class OllamaManager(LoggerMixin):
    """
    Manages Ollama server and model operations.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        auto_start: bool = True
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.auto_start = auto_start
        self.client = None
        
        # Biomedical model recommendations
        self.biomedical_models = {
            # Open source biomedical models
            "llama3.1:8b": ModelInfo(
                name="llama3.1:8b",
                size="4.7GB",
                parameters="8B",
                family="llama",
                description="Meta's Llama 3.1 8B - Good general performance with biomedical knowledge",
                biomedical_optimized=False,
                recommended_use=["general_biomedical", "hypothesis_generation", "literature_analysis"],
                memory_requirements="8GB",
                performance_tier="high"
            ),
            "llama3.1:70b": ModelInfo(
                name="llama3.1:70b",
                size="40GB",
                parameters="70B",
                family="llama",
                description="Meta's Llama 3.1 70B - Excellent performance for complex biomedical reasoning",
                biomedical_optimized=False,
                recommended_use=["complex_reasoning", "hypothesis_validation", "expert_analysis"],
                memory_requirements="48GB",
                performance_tier="premium"
            ),
            "mistral:7b": ModelInfo(
                name="mistral:7b",
                size="4.1GB", 
                parameters="7B",
                family="mistral",
                description="Mistral 7B - Efficient model with good scientific reasoning",
                biomedical_optimized=False,
                recommended_use=["entity_extraction", "relation_extraction", "quick_analysis"],
                memory_requirements="6GB",
                performance_tier="medium"
            ),
            "mixtral:8x7b": ModelInfo(
                name="mixtral:8x7b",
                size="26GB",
                parameters="47B",
                family="mistral",
                description="Mixtral 8x7B - Mixture of experts model with strong reasoning",
                biomedical_optimized=False,
                recommended_use=["complex_analysis", "multi_step_reasoning", "validation"],
                memory_requirements="32GB",
                performance_tier="high"
            ),
            "codellama:13b": ModelInfo(
                name="codellama:13b",
                size="7.3GB",
                parameters="13B", 
                family="llama",
                description="Code Llama 13B - Good for structured data analysis",
                biomedical_optimized=False,
                recommended_use=["data_analysis", "structured_extraction", "bioinformatics"],
                memory_requirements="12GB",
                performance_tier="medium"
            ),
            # Add more models as they become available
            "meditron:7b": ModelInfo(
                name="meditron:7b",
                size="4.1GB",
                parameters="7B",
                family="llama",
                description="Meditron 7B - Specialized for medical text (if available)",
                biomedical_optimized=True,
                recommended_use=["medical_entity_extraction", "clinical_reasoning", "medical_qa"],
                memory_requirements="6GB",
                performance_tier="high"
            )
        }
        
        if OLLAMA_AVAILABLE:
            self._initialize_client()
        else:
            self.logger.warning("Ollama not available. Install with: pip install ollama")
    
    def _initialize_client(self):
        """Initialize Ollama client."""
        try:
            self.client = ollama.Client(host=self.base_url)
            self.logger.info(f"Initialized Ollama client for {self.base_url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            self.client = None
    
    def check_server_status(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_server(self) -> bool:
        """Start Ollama server if not running."""
        if self.check_server_status():
            self.logger.info("Ollama server is already running")
            return True
        
        if not self.auto_start:
            self.logger.warning("Ollama server not running and auto_start is disabled")
            return False
        
        try:
            self.logger.info("Starting Ollama server...")
            # Try to start Ollama server
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            for _ in range(10):
                time.sleep(2)
                if self.check_server_status():
                    self.logger.info("Ollama server started successfully")
                    return True
            
            self.logger.error("Failed to start Ollama server")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List models available on the Ollama server."""
        if not self.client:
            return []
        
        try:
            models = self.client.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Download a model if not already available."""
        if not self.client:
            self.logger.error("Ollama client not available")
            return False
        
        try:
            self.logger.info(f"Pulling model: {model_name}")
            # Stream the pull process
            stream = self.client.pull(model_name, stream=True)
            
            for chunk in stream:
                if 'status' in chunk:
                    status = chunk['status']
                    if 'pulling' in status.lower():
                        # Show progress
                        if 'total' in chunk and 'completed' in chunk:
                            progress = (chunk['completed'] / chunk['total']) * 100
                            self.logger.info(f"Pulling {model_name}: {progress:.1f}%")
                    elif 'success' in status.lower():
                        self.logger.info(f"Successfully pulled {model_name}")
                        return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from local storage."""
        if not self.client:
            return False
        
        try:
            self.client.delete(model_name)
            self.logger.info(f"Deleted model: {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.biomedical_models.get(model_name)
    
    def recommend_models_for_task(self, task: str, memory_limit: str = "8GB") -> List[ModelInfo]:
        """Recommend models for a specific biomedical task."""
        memory_gb = self._parse_memory(memory_limit)
        recommendations = []
        
        for model_info in self.biomedical_models.values():
            model_memory = self._parse_memory(model_info.memory_requirements)
            
            if (model_memory <= memory_gb and 
                (task in model_info.recommended_use or 
                 "general_biomedical" in model_info.recommended_use)):
                recommendations.append(model_info)
        
        # Sort by performance tier and biomedical optimization
        recommendations.sort(
            key=lambda x: (
                x.biomedical_optimized,
                {"premium": 4, "high": 3, "medium": 2, "low": 1}[x.performance_tier]
            ),
            reverse=True
        )
        
        return recommendations
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to GB float."""
        memory_str = memory_str.upper().replace(" ", "")
        if "GB" in memory_str:
            return float(memory_str.replace("GB", ""))
        elif "MB" in memory_str:
            return float(memory_str.replace("MB", "")) / 1024
        else:
            return 8.0  # Default
    
    def setup_biomedical_models(self, memory_limit: str = "8GB") -> List[str]:
        """Set up recommended biomedical models."""
        if not self.start_server():
            return []
        
        available_models = self.list_available_models()
        recommendations = self.recommend_models_for_task("general_biomedical", memory_limit)
        
        installed_models = []
        
        for model_info in recommendations[:3]:  # Install top 3 recommendations
            if model_info.name not in available_models:
                self.logger.info(f"Installing recommended model: {model_info.name}")
                if self.pull_model(model_info.name):
                    installed_models.append(model_info.name)
            else:
                installed_models.append(model_info.name)
        
        return installed_models


class OllamaLLM(LoggerMixin):
    """
    LangChain-compatible Ollama LLM wrapper with biomedical optimizations.
    """
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        top_p: float = 0.9,
        timeout: int = 60,
        biomedical_mode: bool = True
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.biomedical_mode = biomedical_mode
        
        # Initialize Ollama manager
        self.ollama_manager = OllamaManager(base_url=base_url)
        
        # Initialize LangChain Ollama LLM if available
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = LangChainOllamaLLM(
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout
                )
                self.logger.info(f"Initialized Ollama LLM with model: {model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Ollama LLM: {e}")
        
        # Biomedical system prompt
        self.biomedical_system_prompt = """You are a biomedical research expert with deep knowledge of:
- Molecular biology and genetics
- Disease mechanisms and pathophysiology  
- Drug discovery and pharmacology
- Clinical research and medicine
- Bioinformatics and computational biology

Provide accurate, evidence-based responses using scientific terminology.
When uncertain, acknowledge limitations and suggest further investigation."""
    
    def ensure_model_available(self) -> bool:
        """Ensure the model is available locally."""
        if not self.ollama_manager.check_server_status():
            if not self.ollama_manager.start_server():
                return False
        
        available_models = self.ollama_manager.list_available_models()
        if self.model not in available_models:
            self.logger.info(f"Model {self.model} not found locally, downloading...")
            return self.ollama_manager.pull_model(self.model)
        
        return True
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using Ollama."""
        if not self.ensure_model_available():
            raise RuntimeError(f"Model {self.model} not available")
        
        if not self.llm:
            raise RuntimeError("Ollama LLM not initialized")
        
        # Use biomedical system prompt if enabled
        if self.biomedical_mode and system_prompt is None:
            system_prompt = self.biomedical_system_prompt
        
        try:
            # Format prompt with system message if provided
            if system_prompt:
                formatted_prompt = f"System: {system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt
            
            response = self.llm.invoke(formatted_prompt, **kwargs)
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
        return responses
    
    def measure_performance(self, test_prompt: str = "What is BRCA1?") -> ModelPerformance:
        """Measure model performance metrics."""
        import time
        import psutil
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        response = self.generate(test_prompt)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        response_time = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
        
        # Estimate tokens per second (rough approximation)
        estimated_tokens = len(response.split())
        tokens_per_second = estimated_tokens / response_time if response_time > 0 else 0
        
        return ModelPerformance(
            model_name=self.model,
            response_time=response_time,
            tokens_per_second=tokens_per_second,
            memory_usage=f"{memory_used:.1f}MB"
        )


class BiomedicalOllamaChain(LoggerMixin):
    """
    Specialized LangChain chain for biomedical tasks using Ollama.
    """
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        task_type: str = "general",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.task_type = task_type
        self.base_url = base_url
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=model,
            base_url=base_url,
            biomedical_mode=True
        )
        
        # Task-specific prompts
        self.prompts = {
            "entity_extraction": PromptTemplate(
                input_variables=["text"],
                template="""Extract biomedical entities from the following text. 
                Focus on genes, proteins, diseases, drugs, and biological processes.
                
                Text: {text}
                
                Entities (format: entity_type: entity_name):"""
            ),
            "relation_extraction": PromptTemplate(
                input_variables=["text", "entities"],
                template="""Given the text and entities below, identify relationships between the entities.
                
                Text: {text}
                Entities: {entities}
                
                Relationships (format: entity1 -> relationship -> entity2):"""
            ),
            "hypothesis_generation": PromptTemplate(
                input_variables=["context", "observation"],
                template="""Based on the biological context and observation, generate a testable hypothesis.
                
                Context: {context}
                Observation: {observation}
                
                Hypothesis:
                1. Main hypothesis statement:
                2. Biological mechanism:
                3. Testable predictions:
                4. Experimental approach:"""
            ),
            "validation": PromptTemplate(
                input_variables=["hypothesis", "evidence"],
                template="""Evaluate the biological plausibility of this hypothesis given the evidence.
                
                Hypothesis: {hypothesis}
                Evidence: {evidence}
                
                Evaluation:
                1. Biological plausibility (1-10):
                2. Supporting evidence:
                3. Contradicting evidence:
                4. Confidence level:
                5. Recommendations:"""
            )
        }
        
        # Create chain for the specified task
        if task_type in self.prompts:
            self.chain = LLMChain(
                llm=self.llm.llm,
                prompt=self.prompts[task_type]
            )
        else:
            self.chain = None
            self.logger.warning(f"Unknown task type: {task_type}")
    
    def run(self, **kwargs) -> str:
        """Run the chain with provided inputs."""
        if not self.chain:
            return "Error: Chain not initialized"
        
        try:
            return self.chain.run(**kwargs)
        except Exception as e:
            self.logger.error(f"Error running chain: {e}")
            return f"Error: {str(e)}"
    
    def extract_entities(self, text: str) -> str:
        """Extract biomedical entities from text."""
        if self.task_type != "entity_extraction":
            # Create temporary chain
            chain = LLMChain(
                llm=self.llm.llm,
                prompt=self.prompts["entity_extraction"]
            )
            return chain.run(text=text)
        return self.run(text=text)
    
    def extract_relations(self, text: str, entities: str) -> str:
        """Extract relations between entities."""
        if self.task_type != "relation_extraction":
            chain = LLMChain(
                llm=self.llm.llm,
                prompt=self.prompts["relation_extraction"]
            )
            return chain.run(text=text, entities=entities)
        return self.run(text=text, entities=entities)
    
    def generate_hypothesis(self, context: str, observation: str) -> str:
        """Generate hypothesis from context and observation."""
        if self.task_type != "hypothesis_generation":
            chain = LLMChain(
                llm=self.llm.llm,
                prompt=self.prompts["hypothesis_generation"]
            )
            return chain.run(context=context, observation=observation)
        return self.run(context=context, observation=observation)
    
    def validate_hypothesis(self, hypothesis: str, evidence: str) -> str:
        """Validate hypothesis against evidence."""
        if self.task_type != "validation":
            chain = LLMChain(
                llm=self.llm.llm,
                prompt=self.prompts["validation"]
            )
            return chain.run(hypothesis=hypothesis, evidence=evidence)
        return self.run(hypothesis=hypothesis, evidence=evidence)


class LocalModelManager(LoggerMixin):
    """
    Manages local model installations and configurations for biomedical tasks.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else Path.home() / ".litkg" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.ollama_manager = OllamaManager()
        self.installed_models = {}
        
        # Load model registry
        self.registry_file = self.models_dir / "model_registry.json"
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.installed_models = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading model registry: {e}")
                self.installed_models = {}
    
    def save_registry(self):
        """Save model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.installed_models, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving model registry: {e}")
    
    def install_biomedical_models(self, memory_limit: str = "8GB") -> Dict[str, bool]:
        """Install recommended biomedical models."""
        self.logger.info("Installing biomedical models...")
        
        # Start Ollama server
        if not self.ollama_manager.start_server():
            self.logger.error("Failed to start Ollama server")
            return {}
        
        # Get recommendations
        recommendations = self.ollama_manager.recommend_models_for_task(
            "general_biomedical", memory_limit
        )
        
        installation_results = {}
        
        for model_info in recommendations[:3]:  # Install top 3
            self.logger.info(f"Installing {model_info.name}...")
            success = self.ollama_manager.pull_model(model_info.name)
            installation_results[model_info.name] = success
            
            if success:
                self.installed_models[model_info.name] = {
                    "installed_date": time.time(),
                    "model_info": asdict(model_info),
                    "status": "ready"
                }
        
        self.save_registry()
        return installation_results
    
    def get_best_model_for_task(self, task: str, memory_limit: str = "8GB") -> Optional[str]:
        """Get the best available model for a specific task."""
        available_models = self.ollama_manager.list_available_models()
        recommendations = self.ollama_manager.recommend_models_for_task(task, memory_limit)
        
        for model_info in recommendations:
            if model_info.name in available_models:
                return model_info.name
        
        return None
    
    def benchmark_models(self, test_prompts: List[str]) -> Dict[str, ModelPerformance]:
        """Benchmark available models on test prompts."""
        available_models = self.ollama_manager.list_available_models()
        biomedical_models = [m for m in available_models if m in self.ollama_manager.biomedical_models]
        
        results = {}
        
        for model_name in biomedical_models:
            self.logger.info(f"Benchmarking {model_name}...")
            
            llm = OllamaLLM(model=model_name)
            
            # Test with first prompt
            if test_prompts:
                performance = llm.measure_performance(test_prompts[0])
                results[model_name] = performance
        
        return results
    
    def cleanup_models(self, keep_best: int = 2):
        """Clean up models, keeping only the best performers."""
        available_models = self.ollama_manager.list_available_models()
        biomedical_models = [m for m in available_models if m in self.ollama_manager.biomedical_models]
        
        if len(biomedical_models) <= keep_best:
            self.logger.info("No models to clean up")
            return
        
        # Benchmark models
        test_prompt = "What is the role of BRCA1 in DNA repair?"
        performance_results = self.benchmark_models([test_prompt])
        
        # Sort by performance (tokens per second)
        sorted_models = sorted(
            performance_results.items(),
            key=lambda x: x[1].tokens_per_second,
            reverse=True
        )
        
        # Keep top performers
        models_to_keep = [model[0] for model in sorted_models[:keep_best]]
        models_to_remove = [model for model in biomedical_models if model not in models_to_keep]
        
        for model in models_to_remove:
            self.logger.info(f"Removing model: {model}")
            self.ollama_manager.delete_model(model)
            if model in self.installed_models:
                del self.installed_models[model]
        
        self.save_registry()
        self.logger.info(f"Kept {len(models_to_keep)} models, removed {len(models_to_remove)} models")