"""
Tests for conversational agents and RAG systems.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from litkg.agents.biomedical_rag_agent import (
    BiomedicalRAGAgent, RAGConfig, ConversationMemory, BiomedicalContext
)
from litkg.agents.agent_orchestrator import (
    AgentOrchestrator, ConversationalInterface, AgentCoordinator,
    MultiAgentWorkflow, AgentType, AgentCapability, TaskRequest, TaskResponse
)


class TestBiomedicalRAGAgent:
    """Test biomedical RAG agent components."""
    
    def test_rag_config_dataclass(self):
        """Test RAGConfig dataclass."""
        config = RAGConfig(
            vector_store_type="faiss",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=5,
            max_conversation_length=10,
            enable_compression=True,
            temperature=0.1,
            max_tokens=1500
        )
        
        assert config.vector_store_type == "faiss"
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.chunk_size == 1000
        assert config.retrieval_k == 5
        assert config.enable_compression is True
    
    def test_biomedical_context_dataclass(self):
        """Test BiomedicalContext dataclass."""
        context = BiomedicalContext(
            research_domain="cancer_genetics",
            current_hypothesis="BRCA1 mutations increase cancer risk",
            active_entities=["BRCA1", "breast_cancer"],
            research_questions=["How does BRCA1 affect DNA repair?"],
            experimental_context="Cell culture studies",
            literature_focus="Recent publications on BRCA1"
        )
        
        assert context.research_domain == "cancer_genetics"
        assert len(context.active_entities) == 2
        assert len(context.research_questions) == 1
    
    def test_conversation_memory_init(self):
        """Test ConversationMemory initialization."""
        memory = ConversationMemory(max_length=10)
        
        assert memory.max_length == 10
        assert len(memory.messages) == 0
        assert isinstance(memory.biomedical_context, BiomedicalContext)
        assert memory.session_id.startswith("session_")
    
    def test_conversation_memory_add_message(self):
        """Test adding messages to conversation memory."""
        memory = ConversationMemory(max_length=5)
        
        # Add user message
        memory.add_message("user", "What is BRCA1?", {"domain": "genetics"})
        
        assert len(memory.messages) == 1
        assert memory.messages[0]["role"] == "user"
        assert memory.messages[0]["content"] == "What is BRCA1?"
        assert memory.messages[0]["metadata"]["domain"] == "genetics"
        
        # Add assistant message
        memory.add_message("assistant", "BRCA1 is a tumor suppressor gene.")
        
        assert len(memory.messages) == 2
        assert memory.messages[1]["role"] == "assistant"
    
    def test_conversation_memory_max_length(self):
        """Test conversation memory max length enforcement."""
        memory = ConversationMemory(max_length=2)
        
        # Add more messages than max length
        for i in range(5):
            memory.add_message("user", f"Message {i}")
            memory.add_message("assistant", f"Response {i}")
        
        # Should only keep last 4 messages (2 exchanges)
        assert len(memory.messages) == 4
        assert memory.messages[0]["content"] == "Message 3"
    
    def test_biomedical_context_update(self):
        """Test biomedical context updating."""
        memory = ConversationMemory()
        
        # Add message with biomedical entities
        user_message = "Tell me about BRCA1 mutations in breast cancer and PARP inhibitors"
        memory.add_message("user", user_message)
        
        # Check that entities were extracted and added
        context = memory.biomedical_context
        assert "BRCA1" in context.active_entities
        assert "breast cancer" in context.active_entities
    
    def test_conversation_history_formatting(self):
        """Test conversation history formatting."""
        memory = ConversationMemory()
        
        memory.add_message("user", "What is BRCA1?")
        memory.add_message("assistant", "BRCA1 is a tumor suppressor gene.")
        memory.add_message("user", "How does it relate to cancer?")
        
        history = memory.get_conversation_history()
        
        assert "Human: What is BRCA1?" in history
        assert "Assistant: BRCA1 is a tumor suppressor gene." in history
        assert "Human: How does it relate to cancer?" in history
    
    def test_biomedical_context_summary(self):
        """Test biomedical context summary."""
        memory = ConversationMemory()
        memory.biomedical_context.research_domain = "cancer_genetics"
        memory.biomedical_context.active_entities = ["BRCA1", "TP53"]
        memory.biomedical_context.current_hypothesis = "BRCA1 mutations cause cancer"
        
        summary = memory.get_biomedical_context_summary()
        
        assert "Research Domain: cancer_genetics" in summary
        assert "Active Entities: BRCA1, TP53" in summary
        assert "Current Hypothesis: BRCA1 mutations cause cancer" in summary
    
    def test_rag_agent_initialization(self):
        """Test BiomedicalRAGAgent initialization."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm:
            agent = BiomedicalRAGAgent(config=config, llm_manager=mock_llm())
            
            assert agent.config == config
            assert isinstance(agent.memory, ConversationMemory)
            assert agent.system_prompt is not None
    
    def test_rag_agent_chat_without_langchain(self, mock_llm_response):
        """Test RAG agent chat functionality without LangChain."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.LANGCHAIN_AVAILABLE', False):
            with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm_manager:
                mock_manager = Mock()
                mock_manager.process_biomedical_task.return_value = Mock(content="Test response")
                mock_llm_manager.return_value = mock_manager
                
                agent = BiomedicalRAGAgent(config=config, llm_manager=mock_manager)
                
                response = agent.chat("What is BRCA1?")
                
                assert "response" in response
                assert "conversation_id" in response
                assert response["response"] == "Test response"
    
    def test_rag_agent_knowledge_retrieval(self, mock_vector_store):
        """Test knowledge retrieval functionality."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm_manager:
            agent = BiomedicalRAGAgent(config=config)
            agent.vector_store = mock_vector_store
            
            knowledge = agent._retrieve_knowledge("BRCA1")
            
            assert "BRCA1 is a tumor suppressor gene" in knowledge
            assert "PARP inhibitors are effective" in knowledge
    
    def test_rag_agent_hypothesis_generation(self):
        """Test hypothesis generation functionality."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm_manager:
            mock_manager = Mock()
            mock_manager.process_biomedical_task.return_value = Mock(
                content="Hypothesis: BRCA1 mutations lead to DNA repair deficiency"
            )
            mock_llm_manager.return_value = mock_manager
            
            agent = BiomedicalRAGAgent(config=config, llm_manager=mock_manager)
            
            hypothesis = agent._generate_hypothesis("BRCA1 mutations in cancer")
            
            assert "BRCA1 mutations" in hypothesis
            assert "DNA repair" in hypothesis
    
    def test_rag_agent_follow_up_questions(self):
        """Test follow-up question generation."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm_manager:
            agent = BiomedicalRAGAgent(config=config)
            agent.memory.biomedical_context.active_entities = ["BRCA1"]
            
            follow_ups = agent._generate_follow_up_questions("What is BRCA1?", "BRCA1 is a gene")
            
            assert len(follow_ups) > 0
            assert any("BRCA1" in question for question in follow_ups)
    
    def test_rag_agent_conversation_summary(self):
        """Test conversation summary generation."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm_manager:
            agent = BiomedicalRAGAgent(config=config)
            
            # Add some conversation history
            agent.memory.add_message("user", "What is BRCA1?")
            agent.memory.add_message("assistant", "BRCA1 is a tumor suppressor gene.")
            
            summary = agent.get_conversation_summary()
            
            assert "session_id" in summary
            assert "message_count" in summary
            assert summary["message_count"] == 2
    
    def test_rag_agent_reset_conversation(self):
        """Test conversation reset functionality."""
        config = RAGConfig()
        
        with patch('litkg.agents.biomedical_rag_agent.UnifiedLLMManager') as mock_llm_manager:
            agent = BiomedicalRAGAgent(config=config)
            
            # Add conversation history
            agent.memory.add_message("user", "Test message")
            assert len(agent.memory.messages) == 1
            
            # Reset conversation
            agent.reset_conversation()
            
            assert len(agent.memory.messages) == 0
            assert isinstance(agent.memory, ConversationMemory)


class TestAgentOrchestrator:
    """Test agent orchestration components."""
    
    def test_agent_type_enum(self):
        """Test AgentType enumeration."""
        assert AgentType.RAG_AGENT.value == "rag_agent"
        assert AgentType.HYPOTHESIS_AGENT.value == "hypothesis_agent"
        assert AgentType.LITERATURE_AGENT.value == "literature_agent"
    
    def test_agent_capability_dataclass(self):
        """Test AgentCapability dataclass."""
        capability = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Biomedical RAG Assistant",
            description="General-purpose biomedical research assistant",
            expertise_domains=["molecular_biology", "genetics"],
            supported_tasks=["question_answering", "literature_analysis"],
            confidence_threshold=0.7
        )
        
        assert capability.agent_type == AgentType.RAG_AGENT
        assert len(capability.expertise_domains) == 2
        assert capability.confidence_threshold == 0.7
    
    def test_task_request_dataclass(self):
        """Test TaskRequest dataclass."""
        task = TaskRequest(
            task_id="task_123",
            task_type="question_answering",
            content="What is BRCA1?",
            context={"domain": "genetics"},
            priority=1,
            requester="user"
        )
        
        assert task.task_id == "task_123"
        assert task.task_type == "question_answering"
        assert task.context["domain"] == "genetics"
        assert task.timestamp is not None  # Should be auto-generated
    
    def test_task_response_dataclass(self):
        """Test TaskResponse dataclass."""
        response = TaskResponse(
            task_id="task_123",
            agent_type=AgentType.RAG_AGENT,
            response="BRCA1 is a tumor suppressor gene",
            confidence=0.85,
            execution_time=1.5,
            metadata={"model": "test"},
            follow_up_suggestions=["How does BRCA1 work?"]
        )
        
        assert response.task_id == "task_123"
        assert response.agent_type == AgentType.RAG_AGENT
        assert response.confidence == 0.85
        assert len(response.follow_up_suggestions) == 1
    
    def test_agent_coordinator_init(self):
        """Test AgentCoordinator initialization."""
        coordinator = AgentCoordinator()
        
        assert len(coordinator.agents) == 0
        assert len(coordinator.agent_capabilities) == 0
        assert len(coordinator.routing_rules) > 0
        assert "default" in coordinator.routing_rules
    
    def test_agent_registration(self):
        """Test agent registration."""
        coordinator = AgentCoordinator()
        
        # Create mock agent
        mock_agent = Mock()
        capability = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Test Agent",
            description="Test description",
            expertise_domains=["test"],
            supported_tasks=["test_task"]
        )
        
        coordinator.register_agent(AgentType.RAG_AGENT, mock_agent, capability)
        
        assert AgentType.RAG_AGENT in coordinator.agents
        assert coordinator.agents[AgentType.RAG_AGENT] == mock_agent
        assert AgentType.RAG_AGENT in coordinator.agent_capabilities
    
    def test_task_classification(self):
        """Test task classification."""
        coordinator = AgentCoordinator()
        
        # Test different task types
        assert coordinator.classify_task("What is BRCA1?") == "what_is"
        assert coordinator.classify_task("How does DNA repair work?") == "how_does"
        assert coordinator.classify_task("Generate a hypothesis about cancer") == "generate_hypothesis"
        assert coordinator.classify_task("Find literature on BRCA1") == "find_literature"
        assert coordinator.classify_task("Random question") == "default"
    
    def test_agent_selection(self):
        """Test best agent selection."""
        coordinator = AgentCoordinator()
        
        # Register mock agent
        mock_agent = Mock()
        capability = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Test Agent",
            description="Test description",
            expertise_domains=["test"],
            supported_tasks=["test_task"]
        )
        coordinator.register_agent(AgentType.RAG_AGENT, mock_agent, capability)
        
        # Test agent selection
        selected_agent = coordinator.select_best_agent("what_is", {})
        
        assert selected_agent == AgentType.RAG_AGENT
    
    def test_task_execution(self):
        """Test task execution."""
        coordinator = AgentCoordinator()
        
        # Register mock agent
        mock_agent = Mock()
        mock_agent.chat.return_value = {
            "response": "Test response",
            "confidence": 0.8,
            "execution_time": 1.0
        }
        
        capability = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Test Agent",
            description="Test description",
            expertise_domains=["test"],
            supported_tasks=["test_task"]
        )
        coordinator.register_agent(AgentType.RAG_AGENT, mock_agent, capability)
        
        # Execute task
        task_request = TaskRequest(
            task_id="test_task",
            task_type="user_query",
            content="What is BRCA1?",
            context={}
        )
        
        response = coordinator.execute_task(task_request)
        
        assert isinstance(response, TaskResponse)
        assert response.response == "Test response"
        assert response.agent_type == AgentType.RAG_AGENT
    
    def test_performance_tracking(self):
        """Test agent performance tracking."""
        coordinator = AgentCoordinator()
        
        # Register agent
        mock_agent = Mock()
        capability = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Test Agent",
            description="Test description",
            expertise_domains=["test"],
            supported_tasks=["test_task"]
        )
        coordinator.register_agent(AgentType.RAG_AGENT, mock_agent, capability)
        
        # Update performance
        coordinator._update_agent_performance(AgentType.RAG_AGENT, 0.8, 1.5, True)
        
        performance = coordinator.agent_performance[AgentType.RAG_AGENT]
        assert performance["total_tasks"] == 1
        assert performance["success_rate"] > 0
    
    def test_conversational_interface_init(self):
        """Test ConversationalInterface initialization."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            interface = ConversationalInterface(llm_manager=mock_llm())
            
            assert isinstance(interface.coordinator, AgentCoordinator)
            assert hasattr(interface, 'main_agent')
            assert interface.conversation_active is False
    
    def test_start_conversation(self):
        """Test starting a conversation."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            interface = ConversationalInterface(llm_manager=mock_llm())
            
            welcome_message = interface.start_conversation("Test User")
            
            assert interface.conversation_active is True
            assert interface.current_session_id is not None
            assert "Welcome" in welcome_message
    
    def test_conversational_chat(self):
        """Test conversational chat functionality."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            interface = ConversationalInterface(llm_manager=mock_llm())
            
            # Start conversation
            interface.start_conversation()
            
            # Mock the main agent response
            with patch.object(interface.main_agent, 'chat') as mock_chat:
                mock_chat.return_value = {
                    "response": "Test response",
                    "confidence": 0.8,
                    "execution_time": 1.0
                }
                
                response = interface.chat("What is BRCA1?")
                
                assert "response" in response
                assert "agent_used" in response
                assert response["response"] == "Test response"
    
    def test_end_conversation(self):
        """Test ending a conversation."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            interface = ConversationalInterface(llm_manager=mock_llm())
            
            # Start and end conversation
            interface.start_conversation()
            farewell = interface.end_conversation()
            
            assert interface.conversation_active is False
            assert interface.current_session_id is None
            assert "Thank you" in farewell
    
    def test_multi_agent_workflow_init(self):
        """Test MultiAgentWorkflow initialization."""
        coordinator = AgentCoordinator()
        workflow = MultiAgentWorkflow(coordinator)
        
        assert workflow.coordinator == coordinator
        assert len(workflow.workflows) > 0
        assert "hypothesis_to_experiment" in workflow.workflows
    
    def test_workflow_start(self):
        """Test starting a workflow."""
        coordinator = AgentCoordinator()
        workflow = MultiAgentWorkflow(coordinator)
        
        workflow_id = workflow.start_workflow(
            "hypothesis_to_experiment",
            {"research_topic": "BRCA1 mutations"}
        )
        
        assert workflow_id in workflow.active_workflows
        assert workflow.active_workflows[workflow_id]["status"] == "running"
    
    def test_workflow_execution(self):
        """Test workflow step execution."""
        coordinator = AgentCoordinator()
        
        # Register mock agent
        mock_agent = Mock()
        mock_agent.chat.return_value = {
            "response": "Step completed",
            "confidence": 0.8,
            "execution_time": 1.0
        }
        
        capability = AgentCapability(
            agent_type=AgentType.RAG_AGENT,
            name="Test Agent",
            description="Test description",
            expertise_domains=["test"],
            supported_tasks=["test_task"]
        )
        coordinator.register_agent(AgentType.RAG_AGENT, mock_agent, capability)
        
        workflow = MultiAgentWorkflow(coordinator)
        
        # Start workflow
        workflow_id = workflow.start_workflow(
            "hypothesis_to_experiment",
            {"research_topic": "BRCA1 mutations"}
        )
        
        # Execute step
        result = workflow.execute_workflow_step(workflow_id)
        
        assert result["status"] == "step_completed"
        assert "step_result" in result
    
    def test_workflow_status(self):
        """Test workflow status retrieval."""
        coordinator = AgentCoordinator()
        workflow = MultiAgentWorkflow(coordinator)
        
        workflow_id = workflow.start_workflow(
            "hypothesis_to_experiment",
            {"research_topic": "BRCA1 mutations"}
        )
        
        status = workflow.get_workflow_status(workflow_id)
        
        assert "workflow_id" in status
        assert "name" in status
        assert "status" in status
        assert "progress" in status
    
    def test_agent_orchestrator_system_status(self):
        """Test system status retrieval."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            orchestrator = AgentOrchestrator(llm_manager=mock_llm())
            
            status = orchestrator.get_system_status()
            
            assert "system_ready" in status
            assert "conversational_interface" in status
            assert "agent_status" in status
            assert "workflow_manager" in status


@pytest.mark.integration
class TestAgentsIntegration:
    """Integration tests for conversational agents."""
    
    def test_rag_agent_with_orchestrator(self):
        """Test RAG agent integration with orchestrator."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            orchestrator = AgentOrchestrator(llm_manager=mock_llm())
            
            # Mock the main agent response
            with patch.object(orchestrator.conversational_interface.main_agent, 'chat') as mock_chat:
                mock_chat.return_value = {
                    "response": "BRCA1 is a tumor suppressor gene",
                    "confidence": 0.85,
                    "execution_time": 1.2
                }
                
                # Start session and process query
                orchestrator.start_research_session()
                response = orchestrator.process_research_query("What is BRCA1?")
                
                assert "response" in response
                assert "agent_used" in response
                assert response["response"] == "BRCA1 is a tumor suppressor gene"
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation handling."""
        with patch('litkg.agents.agent_orchestrator.UnifiedLLMManager') as mock_llm:
            interface = ConversationalInterface(llm_manager=mock_llm())
            
            interface.start_conversation()
            
            # Mock agent responses
            responses = [
                {"response": "BRCA1 is a tumor suppressor gene", "confidence": 0.9},
                {"response": "BRCA1 mutations increase cancer risk", "confidence": 0.85},
                {"response": "PARP inhibitors are effective in BRCA1-deficient tumors", "confidence": 0.8}
            ]
            
            with patch.object(interface.main_agent, 'chat') as mock_chat:
                mock_chat.side_effect = responses
                
                # Multi-turn conversation
                response1 = interface.chat("What is BRCA1?")
                response2 = interface.chat("How do mutations affect it?")
                response3 = interface.chat("What treatments are available?")
                
                assert response1["response"] == responses[0]["response"]
                assert response2["response"] == responses[1]["response"]
                assert response3["response"] == responses[2]["response"]
    
    def test_workflow_with_multiple_agents(self):
        """Test workflow execution with multiple specialized agents."""
        coordinator = AgentCoordinator()
        
        # Register multiple agents
        agents = {
            AgentType.RAG_AGENT: Mock(),
            AgentType.LITERATURE_AGENT: Mock(),
            AgentType.HYPOTHESIS_AGENT: Mock()
        }
        
        for agent_type, agent in agents.items():
            agent.chat.return_value = {
                "response": f"Response from {agent_type.value}",
                "confidence": 0.8,
                "execution_time": 1.0
            }
            
            capability = AgentCapability(
                agent_type=agent_type,
                name=f"{agent_type.value} Agent",
                description="Test agent",
                expertise_domains=["test"],
                supported_tasks=["test_task"]
            )
            coordinator.register_agent(agent_type, agent, capability)
        
        workflow = MultiAgentWorkflow(coordinator)
        
        # Start and execute workflow
        workflow_id = workflow.start_workflow(
            "hypothesis_to_experiment",
            {"research_topic": "BRCA1 mutations"}
        )
        
        # Execute multiple steps
        results = []
        for _ in range(3):  # Execute 3 steps
            result = workflow.execute_workflow_step(workflow_id)
            if result["status"] == "step_completed":
                results.append(result)
            else:
                break
        
        assert len(results) >= 1  # At least one step should complete
    
    @pytest.mark.slow
    def test_large_conversation_memory(self):
        """Test conversation memory with large number of messages."""
        memory = ConversationMemory(max_length=100)
        
        # Add many messages
        for i in range(200):
            memory.add_message("user", f"User message {i}")
            memory.add_message("assistant", f"Assistant response {i}")
        
        # Should maintain max length
        assert len(memory.messages) <= 200  # max_length * 2
        
        # Should have recent messages
        assert "User message 199" in memory.messages[-2]["content"]
        assert "Assistant response 199" in memory.messages[-1]["content"]
    
    def test_agent_performance_optimization(self):
        """Test agent performance tracking and optimization."""
        coordinator = AgentCoordinator()
        
        # Register agents with different performance characteristics
        fast_agent = Mock()
        fast_agent.chat.return_value = {"response": "Fast response", "confidence": 0.9}
        
        slow_agent = Mock()
        slow_agent.chat.return_value = {"response": "Slow response", "confidence": 0.7}
        
        # Register agents
        for agent_type, agent in [(AgentType.RAG_AGENT, fast_agent), (AgentType.LITERATURE_AGENT, slow_agent)]:
            capability = AgentCapability(
                agent_type=agent_type,
                name=f"{agent_type.value} Agent",
                description="Test agent",
                expertise_domains=["test"],
                supported_tasks=["test_task"]
            )
            coordinator.register_agent(agent_type, agent, capability)
        
        # Execute tasks and track performance
        for i in range(10):
            # Simulate fast agent performance
            coordinator._update_agent_performance(AgentType.RAG_AGENT, 0.9, 0.5, True)
            
            # Simulate slow agent performance
            coordinator._update_agent_performance(AgentType.LITERATURE_AGENT, 0.7, 2.0, True)
        
        # Check performance metrics
        rag_performance = coordinator.agent_performance[AgentType.RAG_AGENT]
        lit_performance = coordinator.agent_performance[AgentType.LITERATURE_AGENT]
        
        assert rag_performance["avg_confidence"] > lit_performance["avg_confidence"]
        assert rag_performance["avg_response_time"] < lit_performance["avg_response_time"]


if __name__ == "__main__":
    pytest.main([__file__])