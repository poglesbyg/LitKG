"""
Conversational Agents and RAG Systems for Biomedical Research

This module provides intelligent conversational agents that can:
1. Answer complex biomedical questions using RAG
2. Guide hypothesis generation and validation
3. Explore knowledge graphs interactively
4. Provide research recommendations
5. Assist with literature analysis and discovery

Key Components:
- BiomedicalRAGAgent: Main conversational research assistant
- HypothesisGenerationAgent: Specialized agent for hypothesis development
- LiteratureExplorationAgent: Agent for literature discovery and analysis
- KnowledgeGraphAgent: Interactive KG exploration assistant
- ResearchPlannerAgent: Research planning and methodology assistant
"""

from .biomedical_rag_agent import (
    BiomedicalRAGAgent,
    RAGConfig,
    ConversationMemory,
    BiomedicalContext
)

from .hypothesis_agent import (
    HypothesisGenerationAgent,
    HypothesisValidationAgent,
    ResearchQuestion,
    HypothesisChain
)

from .literature_agent import (
    LiteratureExplorationAgent,
    LiteratureSearchAgent,
    CitationAnalysisAgent,
    TrendAnalysisAgent
)

from .knowledge_graph_agent import (
    KnowledgeGraphAgent,
    PathExplorationAgent,
    EntityRelationAgent,
    SemanticSearchAgent
)

from .research_planner_agent import (
    ResearchPlannerAgent,
    ExperimentalDesignAgent,
    MethodologyAgent,
    CollaborationAgent
)

from .agent_orchestrator import (
    AgentOrchestrator,
    ConversationalInterface,
    MultiAgentWorkflow,
    AgentCoordinator
)

__all__ = [
    # Main RAG Agent
    "BiomedicalRAGAgent",
    "RAGConfig",
    "ConversationMemory",
    "BiomedicalContext",
    
    # Hypothesis Agents
    "HypothesisGenerationAgent",
    "HypothesisValidationAgent", 
    "ResearchQuestion",
    "HypothesisChain",
    
    # Literature Agents
    "LiteratureExplorationAgent",
    "LiteratureSearchAgent",
    "CitationAnalysisAgent",
    "TrendAnalysisAgent",
    
    # Knowledge Graph Agents
    "KnowledgeGraphAgent",
    "PathExplorationAgent",
    "EntityRelationAgent",
    "SemanticSearchAgent",
    
    # Research Planning Agents
    "ResearchPlannerAgent",
    "ExperimentalDesignAgent",
    "MethodologyAgent",
    "CollaborationAgent",
    
    # Orchestration
    "AgentOrchestrator",
    "ConversationalInterface",
    "MultiAgentWorkflow",
    "AgentCoordinator"
]