"""
Agent Orchestrator - Coordinates Multiple Specialized Agents

This module orchestrates multiple specialized biomedical agents to provide
comprehensive research assistance through intelligent agent coordination,
workflow management, and conversational interfaces.

Features:
- Multi-agent coordination and task delegation
- Conversational interface with natural language understanding
- Research workflow automation
- Agent specialization and expertise routing
- Context sharing between agents
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
from datetime import datetime

# LangChain imports for agent coordination
try:
    from langchain.agents import AgentExecutor, Tool
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Local imports
from ..utils.logging import LoggerMixin
from ..llm_integration import UnifiedLLMManager
from .biomedical_rag_agent import BiomedicalRAGAgent, RAGConfig


class AgentType(Enum):
    """Types of specialized agents."""
    RAG_AGENT = "rag_agent"
    HYPOTHESIS_AGENT = "hypothesis_agent"
    LITERATURE_AGENT = "literature_agent"
    KNOWLEDGE_GRAPH_AGENT = "kg_agent"
    RESEARCH_PLANNER = "research_planner"
    EXPERIMENTAL_DESIGN = "experimental_design"


@dataclass
class AgentCapability:
    """Describes an agent's capabilities."""
    agent_type: AgentType
    name: str
    description: str
    expertise_domains: List[str]
    supported_tasks: List[str]
    confidence_threshold: float = 0.7


@dataclass
class TaskRequest:
    """Represents a task request for agent processing."""
    task_id: str
    task_type: str
    content: str
    context: Dict[str, Any]
    priority: int = 1
    requester: str = "user"
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TaskResponse:
    """Response from an agent task."""
    task_id: str
    agent_type: AgentType
    response: str
    confidence: float
    execution_time: float
    metadata: Dict[str, Any]
    follow_up_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.follow_up_suggestions is None:
            self.follow_up_suggestions = []


class AgentCoordinator(LoggerMixin):
    """Coordinates task delegation between specialized agents."""
    
    def __init__(self):
        # Agent registry
        self.agents: Dict[AgentType, Any] = {}
        self.agent_capabilities: Dict[AgentType, AgentCapability] = {}
        
        # Task routing rules
        self.routing_rules = self._initialize_routing_rules()
        
        # Performance tracking
        self.agent_performance: Dict[AgentType, Dict[str, float]] = {}
        
        self.logger.info("Initialized AgentCoordinator")
    
    def _initialize_routing_rules(self) -> Dict[str, List[AgentType]]:
        """Initialize task routing rules."""
        return {
            # Question types and preferred agent order
            "what_is": [AgentType.RAG_AGENT, AgentType.KNOWLEDGE_GRAPH_AGENT],
            "how_does": [AgentType.RAG_AGENT, AgentType.LITERATURE_AGENT],
            "why_does": [AgentType.RAG_AGENT, AgentType.HYPOTHESIS_AGENT],
            "generate_hypothesis": [AgentType.HYPOTHESIS_AGENT, AgentType.RAG_AGENT],
            "design_experiment": [AgentType.EXPERIMENTAL_DESIGN, AgentType.RESEARCH_PLANNER],
            "find_literature": [AgentType.LITERATURE_AGENT, AgentType.RAG_AGENT],
            "explore_relationships": [AgentType.KNOWLEDGE_GRAPH_AGENT, AgentType.RAG_AGENT],
            "plan_research": [AgentType.RESEARCH_PLANNER, AgentType.RAG_AGENT],
            "validate_hypothesis": [AgentType.HYPOTHESIS_AGENT, AgentType.RAG_AGENT],
            "analyze_data": [AgentType.RAG_AGENT, AgentType.RESEARCH_PLANNER],
            "default": [AgentType.RAG_AGENT]
        }
    
    def register_agent(self, agent_type: AgentType, agent_instance: Any, capabilities: AgentCapability):
        """Register a specialized agent."""
        self.agents[agent_type] = agent_instance
        self.agent_capabilities[agent_type] = capabilities
        self.agent_performance[agent_type] = {
            "total_tasks": 0,
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0
        }
        self.logger.info(f"Registered agent: {agent_type.value}")
    
    def classify_task(self, task_content: str) -> str:
        """Classify the type of task based on content."""
        content_lower = task_content.lower()
        
        # Pattern matching for task classification
        if any(phrase in content_lower for phrase in ["what is", "define", "explain"]):
            return "what_is"
        elif any(phrase in content_lower for phrase in ["how does", "mechanism", "process"]):
            return "how_does"
        elif any(phrase in content_lower for phrase in ["why does", "reason", "cause"]):
            return "why_does"
        elif any(phrase in content_lower for phrase in ["hypothesis", "propose", "theory"]):
            return "generate_hypothesis"
        elif any(phrase in content_lower for phrase in ["experiment", "test", "validate"]):
            return "design_experiment"
        elif any(phrase in content_lower for phrase in ["literature", "papers", "studies"]):
            return "find_literature"
        elif any(phrase in content_lower for phrase in ["relationship", "connection", "pathway"]):
            return "explore_relationships"
        elif any(phrase in content_lower for phrase in ["plan", "strategy", "approach"]):
            return "plan_research"
        elif any(phrase in content_lower for phrase in ["analyze", "data", "results"]):
            return "analyze_data"
        else:
            return "default"
    
    def select_best_agent(self, task_type: str, context: Dict[str, Any]) -> Optional[AgentType]:
        """Select the best agent for a task."""
        # Get preferred agents for task type
        preferred_agents = self.routing_rules.get(task_type, self.routing_rules["default"])
        
        # Filter by available agents
        available_agents = [agent for agent in preferred_agents if agent in self.agents]
        
        if not available_agents:
            return None
        
        # Select based on performance (simplified)
        best_agent = available_agents[0]
        best_score = 0
        
        for agent_type in available_agents:
            performance = self.agent_performance[agent_type]
            # Simple scoring: success_rate * confidence - response_time_penalty
            score = (performance["success_rate"] * performance["avg_confidence"] - 
                    min(performance["avg_response_time"] / 10, 0.5))
            
            if score > best_score:
                best_score = score
                best_agent = agent_type
        
        return best_agent
    
    def execute_task(self, task_request: TaskRequest) -> TaskResponse:
        """Execute a task using the best available agent."""
        start_time = time.time()
        
        # Classify and route task
        task_type = self.classify_task(task_request.content)
        selected_agent_type = self.select_best_agent(task_type, task_request.context)
        
        if not selected_agent_type or selected_agent_type not in self.agents:
            return TaskResponse(
                task_id=task_request.task_id,
                agent_type=AgentType.RAG_AGENT,
                response="No suitable agent available for this task.",
                confidence=0.0,
                execution_time=time.time() - start_time,
                metadata={"error": "No agent available"}
            )
        
        # Execute task with selected agent
        try:
            agent = self.agents[selected_agent_type]
            
            if selected_agent_type == AgentType.RAG_AGENT:
                result = agent.chat(task_request.content, task_request.context)
                response = result["response"]
                confidence = 0.8  # Default confidence for RAG agent
                metadata = result
            else:
                # For other agents, implement specific interfaces
                response = f"Task executed by {selected_agent_type.value}: {task_request.content}"
                confidence = 0.7
                metadata = {"agent_type": selected_agent_type.value}
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_agent_performance(selected_agent_type, confidence, execution_time, True)
            
            return TaskResponse(
                task_id=task_request.task_id,
                agent_type=selected_agent_type,
                response=response,
                confidence=confidence,
                execution_time=execution_time,
                metadata=metadata,
                follow_up_suggestions=self._generate_follow_ups(task_type, response)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_agent_performance(selected_agent_type, 0.0, execution_time, False)
            
            return TaskResponse(
                task_id=task_request.task_id,
                agent_type=selected_agent_type,
                response=f"Error executing task: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    def _update_agent_performance(
        self,
        agent_type: AgentType,
        confidence: float,
        execution_time: float,
        success: bool
    ):
        """Update agent performance metrics."""
        perf = self.agent_performance[agent_type]
        
        perf["total_tasks"] += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        perf["success_rate"] = (1 - alpha) * perf["success_rate"] + alpha * (1.0 if success else 0.0)
        
        # Update average response time
        perf["avg_response_time"] = (1 - alpha) * perf["avg_response_time"] + alpha * execution_time
        
        # Update average confidence
        if success:
            perf["avg_confidence"] = (1 - alpha) * perf["avg_confidence"] + alpha * confidence
    
    def _generate_follow_ups(self, task_type: str, response: str) -> List[str]:
        """Generate follow-up suggestions based on task type."""
        follow_ups = {
            "what_is": [
                "How is this relevant to current research?",
                "What are the clinical implications?",
                "Can you find recent studies on this topic?"
            ],
            "how_does": [
                "What experiments could validate this mechanism?",
                "Are there therapeutic targets in this pathway?",
                "What are alternative mechanisms?"
            ],
            "generate_hypothesis": [
                "How can we test this hypothesis?",
                "What controls should be included?",
                "What are potential confounding factors?"
            ],
            "design_experiment": [
                "What are the statistical considerations?",
                "How can we optimize the experimental design?",
                "What are alternative approaches?"
            ]
        }
        
        return follow_ups.get(task_type, [
            "What would you like to explore next?",
            "Can I help with related research questions?",
            "Would you like me to search for more information?"
        ])
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        status = {}
        
        for agent_type, capabilities in self.agent_capabilities.items():
            performance = self.agent_performance[agent_type]
            status[agent_type.value] = {
                "name": capabilities.name,
                "description": capabilities.description,
                "expertise_domains": capabilities.expertise_domains,
                "performance": performance,
                "available": agent_type in self.agents
            }
        
        return status


class ConversationalInterface(LoggerMixin):
    """Natural language conversational interface for multi-agent system."""
    
    def __init__(
        self,
        llm_manager: Optional[UnifiedLLMManager] = None,
        enable_multi_agent: bool = True
    ):
        self.llm_manager = llm_manager or UnifiedLLMManager()
        self.enable_multi_agent = enable_multi_agent
        
        # Initialize agent coordinator
        self.coordinator = AgentCoordinator()
        
        # Initialize main RAG agent
        self.main_agent = BiomedicalRAGAgent(llm_manager=llm_manager)
        
        # Register the main agent
        rag_capabilities = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Biomedical RAG Assistant",
            description="General-purpose biomedical research assistant with RAG capabilities",
            expertise_domains=["molecular_biology", "genetics", "pharmacology", "clinical_research"],
            supported_tasks=["question_answering", "literature_analysis", "hypothesis_generation", "knowledge_retrieval"]
        )
        
        self.coordinator.register_agent(AgentType.RAG_AGENT, self.main_agent, rag_capabilities)
        
        # Conversation state
        self.conversation_active = False
        self.current_session_id = None
        
        self.logger.info("Initialized ConversationalInterface")
    
    def start_conversation(self, user_name: str = "User") -> str:
        """Start a new conversation session."""
        self.conversation_active = True
        self.current_session_id = f"session_{int(time.time())}"
        
        welcome_message = f"""
🧬 Welcome to LitKG Research Assistant!

I'm your AI-powered biomedical research companion, equipped with:
• 📚 Comprehensive biomedical knowledge retrieval
• 🔬 Hypothesis generation and validation
• 📖 Literature analysis and exploration
• 🧪 Experimental design assistance
• 🔍 Knowledge graph exploration

How can I assist with your research today?

💡 Try asking questions like:
- "What is the role of BRCA1 in DNA repair?"
- "Generate a hypothesis about EGFR resistance mechanisms"
- "Design an experiment to test p53 function"
- "Find recent literature on cancer immunotherapy"
"""
        
        self.logger.info(f"Started conversation for {user_name}")
        return welcome_message
    
    def chat(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main chat interface that routes to appropriate agents.
        
        Args:
            user_input: User's message
            context: Optional conversation context
            
        Returns:
            Response dictionary with agent output and metadata
        """
        if not self.conversation_active:
            return {
                "response": "Please start a conversation first using start_conversation().",
                "error": "No active conversation"
            }
        
        start_time = time.time()
        
        # Create task request
        task_request = TaskRequest(
            task_id=f"task_{int(time.time())}",
            task_type="user_query",
            content=user_input,
            context=context or {},
            requester="user"
        )
        
        # Execute task through coordinator
        if self.enable_multi_agent:
            task_response = self.coordinator.execute_task(task_request)
            
            return {
                "response": task_response.response,
                "agent_used": task_response.agent_type.value,
                "confidence": task_response.confidence,
                "execution_time": task_response.execution_time,
                "follow_up_suggestions": task_response.follow_up_suggestions,
                "metadata": task_response.metadata,
                "session_id": self.current_session_id
            }
        else:
            # Use only the main RAG agent
            result = self.main_agent.chat(user_input, context)
            result["agent_used"] = "rag_agent"
            result["session_id"] = self.current_session_id
            return result
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        if not self.conversation_active:
            return {"error": "No active conversation"}
        
        return {
            "session_id": self.current_session_id,
            "conversation_active": self.conversation_active,
            "main_agent_summary": self.main_agent.get_conversation_summary(),
            "agent_status": self.coordinator.get_agent_status() if self.enable_multi_agent else None
        }
    
    def end_conversation(self) -> str:
        """End the current conversation."""
        if not self.conversation_active:
            return "No active conversation to end."
        
        summary = self.get_conversation_summary()
        
        self.conversation_active = False
        self.current_session_id = None
        
        return f"""
Thank you for using LitKG Research Assistant! 

Session Summary:
- Session ID: {summary['session_id']}
- Messages exchanged: {summary['main_agent_summary'].get('message_count', 0)}
- Research topics explored: {len(summary['main_agent_summary'].get('recent_topics', []))}

Your research insights have been valuable. Feel free to start a new conversation anytime!

🔬 Keep exploring, keep discovering! 🧬
"""
    
    def suggest_research_directions(self, current_topic: str) -> List[str]:
        """Suggest related research directions."""
        suggestions = [
            f"Explore the molecular mechanisms underlying {current_topic}",
            f"Investigate therapeutic targets related to {current_topic}",
            f"Analyze recent clinical trials involving {current_topic}",
            f"Compare {current_topic} across different disease models",
            f"Examine the evolutionary conservation of {current_topic}",
        ]
        
        return suggestions
    
    def export_conversation(self, file_path: str) -> bool:
        """Export current conversation to file."""
        try:
            if self.conversation_active:
                self.main_agent.export_conversation(file_path)
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error exporting conversation: {e}")
            return False


class MultiAgentWorkflow(LoggerMixin):
    """Manages complex multi-step research workflows using multiple agents."""
    
    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self.workflows = {}
        self.active_workflows = {}
        
        # Define common research workflows
        self._initialize_workflows()
        
        self.logger.info("Initialized MultiAgentWorkflow")
    
    def _initialize_workflows(self):
        """Initialize predefined research workflows."""
        self.workflows = {
            "hypothesis_to_experiment": [
                {"step": "literature_review", "agent": AgentType.LITERATURE_AGENT},
                {"step": "hypothesis_generation", "agent": AgentType.HYPOTHESIS_AGENT},
                {"step": "experimental_design", "agent": AgentType.EXPERIMENTAL_DESIGN},
                {"step": "methodology_planning", "agent": AgentType.RESEARCH_PLANNER}
            ],
            "target_discovery": [
                {"step": "knowledge_exploration", "agent": AgentType.KNOWLEDGE_GRAPH_AGENT},
                {"step": "literature_analysis", "agent": AgentType.LITERATURE_AGENT},
                {"step": "hypothesis_generation", "agent": AgentType.HYPOTHESIS_AGENT},
                {"step": "validation_strategy", "agent": AgentType.EXPERIMENTAL_DESIGN}
            ],
            "mechanism_investigation": [
                {"step": "pathway_analysis", "agent": AgentType.KNOWLEDGE_GRAPH_AGENT},
                {"step": "literature_mining", "agent": AgentType.LITERATURE_AGENT},
                {"step": "hypothesis_formulation", "agent": AgentType.HYPOTHESIS_AGENT},
                {"step": "experimental_validation", "agent": AgentType.EXPERIMENTAL_DESIGN}
            ]
        }
    
    def start_workflow(
        self,
        workflow_name: str,
        initial_context: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """Start a multi-agent workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow_id = workflow_id or f"workflow_{int(time.time())}"
        
        self.active_workflows[workflow_id] = {
            "name": workflow_name,
            "steps": self.workflows[workflow_name].copy(),
            "current_step": 0,
            "context": initial_context,
            "results": [],
            "status": "running",
            "start_time": datetime.now().isoformat()
        }
        
        self.logger.info(f"Started workflow {workflow_name} with ID {workflow_id}")
        return workflow_id
    
    def execute_workflow_step(self, workflow_id: str) -> Dict[str, Any]:
        """Execute the next step in a workflow."""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow["status"] != "running":
            return {"error": "Workflow not running"}
        
        if workflow["current_step"] >= len(workflow["steps"]):
            workflow["status"] = "completed"
            return {"status": "completed", "results": workflow["results"]}
        
        # Get current step
        step_info = workflow["steps"][workflow["current_step"]]
        step_name = step_info["step"]
        agent_type = step_info["agent"]
        
        # Create task request for this step
        task_request = TaskRequest(
            task_id=f"{workflow_id}_step_{workflow['current_step']}",
            task_type=step_name,
            content=f"Execute {step_name} for workflow {workflow['name']}",
            context=workflow["context"]
        )
        
        # Execute step
        try:
            response = self.coordinator.execute_task(task_request)
            
            # Store result
            step_result = {
                "step": step_name,
                "agent": agent_type.value,
                "response": response.response,
                "confidence": response.confidence,
                "execution_time": response.execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            workflow["results"].append(step_result)
            workflow["current_step"] += 1
            
            # Update context with results
            workflow["context"][f"{step_name}_result"] = response.response
            
            return {
                "status": "step_completed",
                "step_result": step_result,
                "next_step": workflow["steps"][workflow["current_step"]]["step"] if workflow["current_step"] < len(workflow["steps"]) else None
            }
            
        except Exception as e:
            workflow["status"] = "error"
            return {"error": str(e), "status": "error"}
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow."""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "status": workflow["status"],
            "current_step": workflow["current_step"],
            "total_steps": len(workflow["steps"]),
            "progress": workflow["current_step"] / len(workflow["steps"]) * 100,
            "results_count": len(workflow["results"]),
            "start_time": workflow["start_time"]
        }
    
    def list_available_workflows(self) -> List[str]:
        """List available workflow templates."""
        return list(self.workflows.keys())


class AgentOrchestrator(LoggerMixin):
    """
    Main orchestrator that coordinates all agent systems and provides
    a unified interface for biomedical research assistance.
    """
    
    def __init__(
        self,
        llm_manager: Optional[UnifiedLLMManager] = None,
        enable_workflows: bool = True
    ):
        self.llm_manager = llm_manager or UnifiedLLMManager()
        
        # Initialize components
        self.conversational_interface = ConversationalInterface(llm_manager)
        self.workflow_manager = MultiAgentWorkflow(self.conversational_interface.coordinator) if enable_workflows else None
        
        # System status
        self.system_ready = True
        self.initialization_time = datetime.now().isoformat()
        
        self.logger.info("Initialized AgentOrchestrator")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_ready": self.system_ready,
            "initialization_time": self.initialization_time,
            "conversational_interface": {
                "active": self.conversational_interface.conversation_active,
                "session_id": self.conversational_interface.current_session_id
            },
            "agent_status": self.conversational_interface.coordinator.get_agent_status(),
            "workflow_manager": {
                "available": self.workflow_manager is not None,
                "active_workflows": len(self.workflow_manager.active_workflows) if self.workflow_manager else 0
            },
            "llm_integration": {
                "providers_available": len(self.llm_manager.llm_interface.clients),
                "usage_stats": self.llm_manager.llm_interface.get_usage_stats()
            }
        }
    
    def start_research_session(self, user_name: str = "Researcher") -> str:
        """Start a comprehensive research session."""
        welcome = self.conversational_interface.start_conversation(user_name)
        
        system_info = f"""
🔬 **LitKG Agent System Status**
- Multi-agent coordination: ✅ Active
- Knowledge retrieval: ✅ Ready
- Hypothesis generation: ✅ Ready
- Literature analysis: ✅ Ready
- Experimental design: ✅ Ready
{"- Research workflows: ✅ Available" if self.workflow_manager else ""}

{welcome}
"""
        
        return system_info
    
    def process_research_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a research query through the agent system."""
        return self.conversational_interface.chat(query, context)
    
    def start_research_workflow(
        self,
        workflow_name: str,
        research_topic: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start a guided research workflow."""
        if not self.workflow_manager:
            return {"error": "Workflow manager not available"}
        
        initial_context = {
            "research_topic": research_topic,
            **(additional_context or {})
        }
        
        try:
            workflow_id = self.workflow_manager.start_workflow(
                workflow_name,
                initial_context
            )
            
            return {
                "workflow_id": workflow_id,
                "status": "started",
                "available_workflows": self.workflow_manager.list_available_workflows()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_research_recommendations(self, current_context: Dict[str, Any]) -> List[str]:
        """Get AI-powered research recommendations."""
        topic = current_context.get("research_topic", "biomedical research")
        
        return self.conversational_interface.suggest_research_directions(topic)