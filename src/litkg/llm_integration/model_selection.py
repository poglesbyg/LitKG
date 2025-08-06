"""
Model Selection and Performance Optimization

This module provides intelligent model selection and performance optimization
for biomedical LLM tasks, including automatic model recommendations based on
task requirements, resource constraints, and performance characteristics.

Features:
- Task-specific model recommendations
- Resource-aware model selection
- Performance optimization strategies
- Cost-benefit analysis
- Automatic fallback mechanisms
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.logging import LoggerMixin


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Basic entity extraction, simple QA
    MODERATE = "moderate"  # Relation extraction, literature analysis
    COMPLEX = "complex"    # Hypothesis generation, multi-step reasoning
    EXPERT = "expert"      # Validation, expert-level analysis


class ResourceConstraint(Enum):
    """Resource constraint levels."""
    LOW = "low"        # <4GB RAM, limited CPU
    MEDIUM = "medium"  # 4-16GB RAM, moderate CPU
    HIGH = "high"      # 16-64GB RAM, high-end CPU/GPU
    UNLIMITED = "unlimited"  # No resource constraints


@dataclass
class ModelRecommendation:
    """Model recommendation with rationale."""
    model_name: str
    provider: str
    confidence: float
    reasoning: str
    expected_performance: Dict[str, float]
    resource_requirements: Dict[str, str]
    cost_estimate: float
    alternatives: List[str]


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    accuracy: Optional[float] = None
    response_time: Optional[float] = None
    throughput: Optional[float] = None  # tokens/second
    memory_usage: Optional[float] = None  # GB
    cost_per_query: Optional[float] = None
    biomedical_score: Optional[float] = None


class BiomedicalModelRecommendations:
    """Biomedical-specific model recommendations database."""
    
    def __init__(self):
        self.task_model_matrix = {
            # Entity Extraction
            "entity_extraction": {
                TaskComplexity.SIMPLE: [
                    ("mistral:7b", "ollama", 0.85),
                    ("llama3.1:8b", "ollama", 0.80),
                    ("gpt-3.5-turbo", "openai", 0.75)
                ],
                TaskComplexity.MODERATE: [
                    ("llama3.1:8b", "ollama", 0.90),
                    ("gpt-3.5-turbo", "openai", 0.85),
                    ("claude-3-haiku", "anthropic", 0.80)
                ],
                TaskComplexity.COMPLEX: [
                    ("llama3.1:70b", "ollama", 0.95),
                    ("gpt-4", "openai", 0.90),
                    ("claude-3-sonnet", "anthropic", 0.85)
                ]
            },
            
            # Relation Extraction
            "relation_extraction": {
                TaskComplexity.SIMPLE: [
                    ("llama3.1:8b", "ollama", 0.80),
                    ("gpt-3.5-turbo", "openai", 0.75),
                    ("mistral:7b", "ollama", 0.70)
                ],
                TaskComplexity.MODERATE: [
                    ("llama3.1:8b", "ollama", 0.85),
                    ("gpt-4", "openai", 0.90),
                    ("claude-3-sonnet", "anthropic", 0.85)
                ],
                TaskComplexity.COMPLEX: [
                    ("gpt-4", "openai", 0.95),
                    ("llama3.1:70b", "ollama", 0.90),
                    ("claude-3-sonnet", "anthropic", 0.88)
                ]
            },
            
            # Hypothesis Generation
            "hypothesis_generation": {
                TaskComplexity.MODERATE: [
                    ("llama3.1:8b", "ollama", 0.75),
                    ("gpt-3.5-turbo", "openai", 0.80),
                    ("claude-3-haiku", "anthropic", 0.75)
                ],
                TaskComplexity.COMPLEX: [
                    ("gpt-4", "openai", 0.95),
                    ("claude-3-sonnet", "anthropic", 0.90),
                    ("llama3.1:70b", "ollama", 0.85)
                ],
                TaskComplexity.EXPERT: [
                    ("gpt-4-turbo", "openai", 0.98),
                    ("claude-3-opus", "anthropic", 0.95),
                    ("llama3.1:70b", "ollama", 0.88)
                ]
            },
            
            # Validation
            "validation": {
                TaskComplexity.MODERATE: [
                    ("gpt-3.5-turbo", "openai", 0.80),
                    ("llama3.1:8b", "ollama", 0.75),
                    ("claude-3-haiku", "anthropic", 0.78)
                ],
                TaskComplexity.COMPLEX: [
                    ("gpt-4", "openai", 0.92),
                    ("claude-3-sonnet", "anthropic", 0.90),
                    ("llama3.1:70b", "ollama", 0.85)
                ],
                TaskComplexity.EXPERT: [
                    ("gpt-4-turbo", "openai", 0.96),
                    ("claude-3-opus", "anthropic", 0.94),
                    ("llama3.1:70b", "ollama", 0.88)
                ]
            },
            
            # Literature Analysis
            "literature_analysis": {
                TaskComplexity.SIMPLE: [
                    ("llama3.1:8b", "ollama", 0.80),
                    ("gpt-3.5-turbo", "openai", 0.75)
                ],
                TaskComplexity.MODERATE: [
                    ("gpt-4", "openai", 0.90),
                    ("claude-3-sonnet", "anthropic", 0.88),
                    ("llama3.1:70b", "ollama", 0.85)
                ],
                TaskComplexity.COMPLEX: [
                    ("gpt-4-turbo", "openai", 0.95),
                    ("claude-3-opus", "anthropic", 0.92),
                    ("llama3.1:70b", "ollama", 0.87)
                ]
            }
        }
        
        # Resource requirements for different models
        self.resource_requirements = {
            "mistral:7b": {"memory": "6GB", "cpu": "moderate", "gpu": "optional"},
            "llama3.1:8b": {"memory": "8GB", "cpu": "moderate", "gpu": "optional"},
            "llama3.1:70b": {"memory": "48GB", "cpu": "high", "gpu": "recommended"},
            "mixtral:8x7b": {"memory": "32GB", "cpu": "high", "gpu": "recommended"},
            "gpt-3.5-turbo": {"memory": "0GB", "cpu": "none", "gpu": "none"},
            "gpt-4": {"memory": "0GB", "cpu": "none", "gpu": "none"},
            "gpt-4-turbo": {"memory": "0GB", "cpu": "none", "gpu": "none"},
            "claude-3-haiku": {"memory": "0GB", "cpu": "none", "gpu": "none"},
            "claude-3-sonnet": {"memory": "0GB", "cpu": "none", "gpu": "none"},
            "claude-3-opus": {"memory": "0GB", "cpu": "none", "gpu": "none"}
        }
        
        # Cost estimates (per 1K tokens)
        self.cost_estimates = {
            "mistral:7b": 0.0,
            "llama3.1:8b": 0.0,
            "llama3.1:70b": 0.0,
            "mixtral:8x7b": 0.0,
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "claude-3-haiku": 0.0025,
            "claude-3-sonnet": 0.015,
            "claude-3-opus": 0.075
        }


class ModelSelector(LoggerMixin):
    """Intelligent model selector for biomedical tasks."""
    
    def __init__(self):
        self.recommendations_db = BiomedicalModelRecommendations()
        self.performance_history = {}
        self.logger.info("Initialized ModelSelector")
    
    def recommend_model(
        self,
        task: str,
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        resource_constraint: ResourceConstraint = ResourceConstraint.MEDIUM,
        max_cost_per_query: Optional[float] = None,
        require_local: bool = False,
        priority_factors: Optional[Dict[str, float]] = None
    ) -> ModelRecommendation:
        """
        Recommend the best model for a given task and constraints.
        
        Args:
            task: Task type (e.g., "entity_extraction", "hypothesis_generation")
            complexity: Task complexity level
            resource_constraint: Available resources
            max_cost_per_query: Maximum acceptable cost per query
            require_local: Whether to require local inference
            priority_factors: Custom weighting for selection criteria
            
        Returns:
            ModelRecommendation with detailed rationale
        """
        
        # Default priority factors
        if priority_factors is None:
            priority_factors = {
                "accuracy": 0.4,
                "cost": 0.3,
                "speed": 0.2,
                "privacy": 0.1
            }
        
        # Get candidate models for task and complexity
        candidates = self._get_candidates(task, complexity)
        
        if not candidates:
            raise ValueError(f"No models available for task: {task} with complexity: {complexity}")
        
        # Filter by constraints
        filtered_candidates = self._apply_constraints(
            candidates,
            resource_constraint,
            max_cost_per_query,
            require_local
        )
        
        if not filtered_candidates:
            # Relax constraints and try again
            self.logger.warning("No models match constraints, relaxing requirements")
            filtered_candidates = self._apply_constraints(
                candidates,
                ResourceConstraint.HIGH,
                None,
                False
            )
        
        # Score and rank candidates
        scored_candidates = self._score_candidates(
            filtered_candidates,
            task,
            priority_factors
        )
        
        # Select best candidate
        best_candidate = scored_candidates[0]
        model_name, provider, base_score = best_candidate[:3]
        
        # Generate recommendation
        return self._create_recommendation(
            model_name,
            provider,
            base_score,
            task,
            complexity,
            scored_candidates[1:6]  # Top 5 alternatives
        )
    
    def _get_candidates(self, task: str, complexity: TaskComplexity) -> List[Tuple]:
        """Get candidate models for task and complexity."""
        task_models = self.recommendations_db.task_model_matrix.get(task, {})
        
        # Try exact complexity match first
        if complexity in task_models:
            return task_models[complexity]
        
        # Fall back to available complexities
        for fallback_complexity in [TaskComplexity.COMPLEX, TaskComplexity.MODERATE, TaskComplexity.SIMPLE]:
            if fallback_complexity in task_models:
                return task_models[fallback_complexity]
        
        return []
    
    def _apply_constraints(
        self,
        candidates: List[Tuple],
        resource_constraint: ResourceConstraint,
        max_cost_per_query: Optional[float],
        require_local: bool
    ) -> List[Tuple]:
        """Apply resource and cost constraints to candidates."""
        
        filtered = []
        
        for model_name, provider, score in candidates:
            # Check local requirement
            if require_local and provider not in ["ollama"]:
                continue
            
            # Check cost constraint
            if max_cost_per_query is not None:
                estimated_cost = self.recommendations_db.cost_estimates.get(model_name, 0)
                if estimated_cost > max_cost_per_query:
                    continue
            
            # Check resource constraint
            if not self._meets_resource_constraint(model_name, resource_constraint):
                continue
            
            filtered.append((model_name, provider, score))
        
        return filtered
    
    def _meets_resource_constraint(
        self,
        model_name: str,
        resource_constraint: ResourceConstraint
    ) -> bool:
        """Check if model meets resource constraints."""
        
        requirements = self.recommendations_db.resource_requirements.get(model_name, {})
        memory_req = requirements.get("memory", "0GB")
        
        # Parse memory requirement
        memory_gb = 0
        if memory_req.endswith("GB"):
            memory_gb = float(memory_req[:-2])
        
        # Check against constraint
        if resource_constraint == ResourceConstraint.LOW:
            return memory_gb <= 4
        elif resource_constraint == ResourceConstraint.MEDIUM:
            return memory_gb <= 16
        elif resource_constraint == ResourceConstraint.HIGH:
            return memory_gb <= 64
        else:  # UNLIMITED
            return True
    
    def _score_candidates(
        self,
        candidates: List[Tuple],
        task: str,
        priority_factors: Dict[str, float]
    ) -> List[Tuple]:
        """Score and rank candidates based on priority factors."""
        
        scored_candidates = []
        
        for model_name, provider, base_score in candidates:
            # Calculate composite score
            accuracy_score = base_score
            cost_score = 1.0 - min(self.recommendations_db.cost_estimates.get(model_name, 0) / 0.1, 1.0)
            speed_score = 0.8 if provider == "ollama" else 0.6  # Local models generally faster
            privacy_score = 1.0 if provider == "ollama" else 0.3  # Local models more private
            
            composite_score = (
                accuracy_score * priority_factors.get("accuracy", 0.4) +
                cost_score * priority_factors.get("cost", 0.3) +
                speed_score * priority_factors.get("speed", 0.2) +
                privacy_score * priority_factors.get("privacy", 0.1)
            )
            
            scored_candidates.append((model_name, provider, base_score, composite_score))
        
        # Sort by composite score
        scored_candidates.sort(key=lambda x: x[3], reverse=True)
        
        return scored_candidates
    
    def _create_recommendation(
        self,
        model_name: str,
        provider: str,
        confidence: float,
        task: str,
        complexity: TaskComplexity,
        alternatives: List[Tuple]
    ) -> ModelRecommendation:
        """Create detailed model recommendation."""
        
        # Generate reasoning
        reasoning_parts = [
            f"Selected {model_name} for {task} task",
            f"Complexity level: {complexity.value}",
            f"Provider: {provider}",
            f"Base confidence: {confidence:.2f}"
        ]
        
        if provider == "ollama":
            reasoning_parts.append("Local inference provides privacy and cost benefits")
        else:
            reasoning_parts.append("Cloud inference provides high performance")
        
        reasoning = ". ".join(reasoning_parts) + "."
        
        # Expected performance
        expected_performance = {
            "accuracy": confidence,
            "response_time": 2.0 if provider == "ollama" else 1.0,
            "cost_efficiency": 1.0 if provider == "ollama" else 0.5
        }
        
        # Resource requirements
        resource_requirements = self.recommendations_db.resource_requirements.get(
            model_name, {"memory": "unknown", "cpu": "unknown", "gpu": "unknown"}
        )
        
        # Cost estimate
        cost_estimate = self.recommendations_db.cost_estimates.get(model_name, 0.0)
        
        # Alternative models
        alternative_names = [alt[0] for alt in alternatives]
        
        return ModelRecommendation(
            model_name=model_name,
            provider=provider,
            confidence=confidence,
            reasoning=reasoning,
            expected_performance=expected_performance,
            resource_requirements=resource_requirements,
            cost_estimate=cost_estimate,
            alternatives=alternative_names
        )
    
    def update_performance_history(
        self,
        model_name: str,
        task: str,
        metrics: PerformanceMetrics
    ):
        """Update performance history for a model."""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {}
        
        if task not in self.performance_history[model_name]:
            self.performance_history[model_name][task] = []
        
        self.performance_history[model_name][task].append(metrics)
        
        # Keep only last 10 entries
        self.performance_history[model_name][task] = \
            self.performance_history[model_name][task][-10:]
    
    def get_performance_stats(
        self,
        model_name: str,
        task: str
    ) -> Optional[Dict[str, float]]:
        """Get performance statistics for a model and task."""
        
        if (model_name not in self.performance_history or 
            task not in self.performance_history[model_name]):
            return None
        
        metrics_list = self.performance_history[model_name][task]
        
        if not metrics_list:
            return None
        
        # Calculate averages
        stats = {}
        
        for metric_name in ["accuracy", "response_time", "throughput", "memory_usage", "cost_per_query"]:
            values = [getattr(m, metric_name) for m in metrics_list if getattr(m, metric_name) is not None]
            if values:
                stats[f"avg_{metric_name}"] = sum(values) / len(values)
                stats[f"min_{metric_name}"] = min(values)
                stats[f"max_{metric_name}"] = max(values)
        
        return stats


class PerformanceOptimizer(LoggerMixin):
    """Optimizes model performance for specific use cases."""
    
    def __init__(self):
        self.optimization_strategies = {
            "speed": self._optimize_for_speed,
            "accuracy": self._optimize_for_accuracy,
            "cost": self._optimize_for_cost,
            "privacy": self._optimize_for_privacy
        }
        self.logger.info("Initialized PerformanceOptimizer")
    
    def optimize_model_selection(
        self,
        task: str,
        optimization_target: str = "balanced",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize model selection for specific targets.
        
        Args:
            task: Task type
            optimization_target: "speed", "accuracy", "cost", "privacy", or "balanced"
            constraints: Additional constraints
            
        Returns:
            Optimization recommendations
        """
        
        constraints = constraints or {}
        
        if optimization_target in self.optimization_strategies:
            return self.optimization_strategies[optimization_target](task, constraints)
        elif optimization_target == "balanced":
            return self._optimize_balanced(task, constraints)
        else:
            raise ValueError(f"Unknown optimization target: {optimization_target}")
    
    def _optimize_for_speed(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for fastest response time."""
        return {
            "priority_factors": {"speed": 0.5, "accuracy": 0.3, "cost": 0.1, "privacy": 0.1},
            "prefer_local": True,
            "model_params": {"temperature": 0.0, "max_tokens": 500},
            "reasoning": "Optimized for fastest response time with acceptable accuracy"
        }
    
    def _optimize_for_accuracy(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for highest accuracy."""
        return {
            "priority_factors": {"accuracy": 0.6, "speed": 0.1, "cost": 0.2, "privacy": 0.1},
            "prefer_local": False,
            "model_params": {"temperature": 0.1, "max_tokens": 1000},
            "reasoning": "Optimized for highest accuracy, accepting higher cost and latency"
        }
    
    def _optimize_for_cost(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for lowest cost."""
        return {
            "priority_factors": {"cost": 0.5, "accuracy": 0.3, "speed": 0.1, "privacy": 0.1},
            "prefer_local": True,
            "model_params": {"temperature": 0.1, "max_tokens": 300},
            "reasoning": "Optimized for lowest cost while maintaining reasonable accuracy"
        }
    
    def _optimize_for_privacy(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for maximum privacy."""
        return {
            "priority_factors": {"privacy": 0.6, "accuracy": 0.2, "speed": 0.1, "cost": 0.1},
            "prefer_local": True,
            "require_local": True,
            "model_params": {"temperature": 0.1, "max_tokens": 800},
            "reasoning": "Optimized for maximum privacy using only local models"
        }
    
    def _optimize_balanced(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for balanced performance across all factors."""
        return {
            "priority_factors": {"accuracy": 0.4, "cost": 0.25, "speed": 0.25, "privacy": 0.1},
            "prefer_local": False,
            "model_params": {"temperature": 0.1, "max_tokens": 600},
            "reasoning": "Balanced optimization across accuracy, cost, speed, and privacy"
        }