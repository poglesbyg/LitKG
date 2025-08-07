"""
Biomedical RAG Agent - Main Conversational Research Assistant

This module implements a sophisticated RAG (Retrieval-Augmented Generation) agent
specifically designed for biomedical research. It combines knowledge graph exploration,
literature retrieval, and LLM reasoning to provide comprehensive research assistance.

Features:
- Multi-source knowledge retrieval (literature, KG, experimental data)
- Contextual conversation memory
- Domain-specific biomedical reasoning
- Research workflow guidance
- Hypothesis generation and validation assistance
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

# LangChain imports for RAG
try:
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Local imports
from ..utils.logging import LoggerMixin
from ..llm_integration import UnifiedLLMManager, LLMProvider
from ..phase1 import LiteratureProcessor
from ..phase3 import NoveltyDetectionSystem, HypothesisGenerationSystem


@dataclass
class RAGConfig:
    """Configuration for the RAG agent."""
    vector_store_type: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    max_conversation_length: int = 10
    enable_compression: bool = True
    temperature: float = 0.1
    max_tokens: int = 1500


@dataclass
class BiomedicalContext:
    """Biomedical context for conversation."""
    research_domain: str
    current_hypothesis: Optional[str] = None
    active_entities: List[str] = None
    research_questions: List[str] = None
    experimental_context: Optional[str] = None
    literature_focus: Optional[str] = None
    
    def __post_init__(self):
        if self.active_entities is None:
            self.active_entities = []
        if self.research_questions is None:
            self.research_questions = []


class ConversationMemory(LoggerMixin):
    """Enhanced conversation memory with biomedical context awareness."""
    
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.messages: List[Dict[str, Any]] = []
        self.biomedical_context = BiomedicalContext(research_domain="general")
        self.session_id = f"session_{int(time.time())}"
        
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to conversation memory."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Maintain max length
        if len(self.messages) > self.max_length * 2:  # *2 for user + assistant pairs
            self.messages = self.messages[-self.max_length * 2:]
        
        # Extract biomedical entities and context
        if role == "user":
            self._update_biomedical_context(content)
    
    def _update_biomedical_context(self, user_message: str):
        """Update biomedical context based on user message."""
        # Simple entity extraction (in production, use NER)
        biomedical_keywords = [
            "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "PIK3CA",
            "breast cancer", "lung cancer", "diabetes", "alzheimer",
            "tamoxifen", "aspirin", "metformin", "immunotherapy"
        ]
        
        found_entities = [
            keyword for keyword in biomedical_keywords
            if keyword.lower() in user_message.lower()
        ]
        
        # Update active entities
        for entity in found_entities:
            if entity not in self.biomedical_context.active_entities:
                self.biomedical_context.active_entities.append(entity)
        
        # Keep only recent entities
        self.biomedical_context.active_entities = \
            self.biomedical_context.active_entities[-10:]
        
        # Detect research questions
        if "?" in user_message:
            self.biomedical_context.research_questions.append(user_message)
            self.biomedical_context.research_questions = \
                self.biomedical_context.research_questions[-5:]
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        if not self.messages:
            return "No previous conversation."
        
        history = []
        for msg in self.messages[-6:]:  # Last 3 exchanges
            role = "Human" if msg["role"] == "user" else "Assistant"
            history.append(f"{role}: {msg['content']}")
        
        return "\n".join(history)
    
    def get_biomedical_context_summary(self) -> str:
        """Get summary of current biomedical context."""
        context = self.biomedical_context
        
        summary_parts = [
            f"Research Domain: {context.research_domain}"
        ]
        
        if context.active_entities:
            summary_parts.append(f"Active Entities: {', '.join(context.active_entities[-5:])}")
        
        if context.current_hypothesis:
            summary_parts.append(f"Current Hypothesis: {context.current_hypothesis}")
        
        if context.research_questions:
            summary_parts.append(f"Recent Questions: {len(context.research_questions)} questions asked")
        
        return " | ".join(summary_parts)


class BiomedicalRAGAgent(LoggerMixin):
    """
    Main conversational RAG agent for biomedical research assistance.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm_manager: Optional[UnifiedLLMManager] = None,
        knowledge_base_path: Optional[str] = None
    ):
        self.config = config or RAGConfig()
        self.llm_manager = llm_manager or UnifiedLLMManager()
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else Path("data/knowledge_base")
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_length=self.config.max_conversation_length)
        
        # Initialize components
        self.vector_store = None
        self.retrieval_chain = None
        self.tools = []
        
        # Biomedical components
        self.literature_processor = None
        self.novelty_system = None
        self.hypothesis_system = None
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_rag_components()
        else:
            self.logger.warning("LangChain not available. RAG functionality limited.")
        
        # Agent prompts
        self.system_prompt = """You are a world-class biomedical research assistant with expertise in:
- Molecular biology, genetics, and genomics
- Disease mechanisms and pathophysiology
- Drug discovery and pharmacology
- Clinical research and medicine
- Bioinformatics and computational biology

You help researchers by:
1. Answering complex biomedical questions using retrieved knowledge
2. Generating and validating research hypotheses
3. Exploring relationships in biomedical knowledge graphs
4. Providing experimental design suggestions
5. Analyzing literature and identifying research gaps

Always provide evidence-based responses with proper scientific reasoning.
When uncertain, acknowledge limitations and suggest further investigation."""
        
        self.logger.info("Initialized BiomedicalRAGAgent")
    
    def _initialize_rag_components(self):
        """Initialize RAG components including vector store and retrieval chain."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            
            # Load or create vector store
            self._setup_vector_store()
            
            # Initialize retrieval chain
            self._setup_retrieval_chain()
            
            # Initialize tools
            self._setup_tools()
            
            self.logger.info("RAG components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG components: {e}")
    
    def _setup_vector_store(self):
        """Set up vector store for knowledge retrieval."""
        vector_store_path = self.knowledge_base_path / "vector_store"
        
        if vector_store_path.exists() and self.config.vector_store_type == "faiss":
            try:
                # Load existing FAISS index
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings
                )
                self.logger.info("Loaded existing FAISS vector store")
                return
            except Exception as e:
                self.logger.warning(f"Could not load existing vector store: {e}")
        
        # Create new vector store with sample documents
        sample_documents = self._create_sample_biomedical_documents()
        
        if self.config.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                sample_documents,
                self.embeddings
            )
            # Save for future use
            vector_store_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(vector_store_path))
            
        elif self.config.vector_store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                sample_documents,
                self.embeddings,
                persist_directory=str(vector_store_path)
            )
        
        self.logger.info(f"Created new {self.config.vector_store_type} vector store")
    
    def _create_sample_biomedical_documents(self) -> List[Document]:
        """Create sample biomedical documents for the knowledge base."""
        sample_texts = [
            # BRCA1 and breast cancer
            """BRCA1 (Breast Cancer 1) is a tumor suppressor gene that plays a critical role in DNA repair through homologous recombination. Mutations in BRCA1 significantly increase the risk of breast and ovarian cancers. BRCA1-deficient tumors are particularly sensitive to PARP inhibitors due to synthetic lethality. The protein product of BRCA1 interacts with other DNA repair proteins including BRCA2, RAD51, and p53 to maintain genomic stability.""",
            
            # p53 pathway
            """TP53, known as the "guardian of the genome," encodes the p53 protein, a crucial tumor suppressor. p53 responds to DNA damage by either promoting DNA repair or inducing apoptosis in severely damaged cells. Loss of p53 function, occurring in over 50% of human cancers, leads to genomic instability and uncontrolled cell proliferation. p53 regulates numerous downstream targets including p21, BAX, and MDM2.""",
            
            # EGFR and targeted therapy
            """Epidermal Growth Factor Receptor (EGFR) is a receptor tyrosine kinase frequently overexpressed or mutated in various cancers, particularly lung cancer. EGFR mutations predict response to tyrosine kinase inhibitors such as erlotinib and gefitinib. However, resistance mechanisms including secondary mutations (T790M) and bypass pathways often develop. Combination therapies and next-generation EGFR inhibitors are being developed to overcome resistance.""",
            
            # Immunotherapy
            """Cancer immunotherapy has revolutionized treatment by harnessing the immune system to fight cancer. Checkpoint inhibitors targeting PD-1/PD-L1 and CTLA-4 have shown remarkable success in melanoma, lung cancer, and other malignancies. CAR-T cell therapy has demonstrated efficacy in hematological malignancies. Biomarkers such as PD-L1 expression, microsatellite instability, and tumor mutational burden help predict immunotherapy response.""",
            
            # Drug discovery
            """Modern drug discovery integrates computational approaches with traditional experimental methods. Structure-based drug design uses protein crystal structures to guide compound optimization. High-throughput screening allows testing of large compound libraries. Artificial intelligence and machine learning are increasingly used for target identification, lead optimization, and prediction of drug properties including ADMET characteristics.""",
            
            # Precision medicine
            """Precision medicine tailors treatment based on individual patient characteristics, including genetic makeup, biomarker expression, and disease subtype. Companion diagnostics identify patients likely to benefit from specific therapies. Examples include HER2 testing for trastuzumab in breast cancer, EGFR mutation testing for TKI therapy in lung cancer, and PD-L1 expression for immunotherapy selection.""",
            
            # Cancer metabolism
            """Cancer cells exhibit altered metabolism to support rapid proliferation, a phenomenon known as the Warburg effect. This metabolic reprogramming involves increased glucose uptake, enhanced glycolysis even in the presence of oxygen, and altered lipid and amino acid metabolism. Targeting cancer metabolism through inhibitors of glycolysis, glutaminolysis, or fatty acid synthesis represents a promising therapeutic approach.""",
            
            # Epigenetics
            """Epigenetic modifications, including DNA methylation and histone modifications, play crucial roles in gene regulation and cancer development. Aberrant DNA hypermethylation can silence tumor suppressor genes, while histone modifications affect chromatin structure and gene accessibility. Epigenetic drugs such as DNA methyltransferase inhibitors and histone deacetylase inhibitors are approved for cancer treatment.""",
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                page_content=text,
                metadata={
                    "source": f"biomedical_knowledge_{i}",
                    "type": "knowledge_base",
                    "domain": "biomedical_research"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _setup_retrieval_chain(self):
        """Set up the retrieval chain for RAG."""
        if not self.vector_store:
            return
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )
        
        # Add compression if enabled
        if self.config.enable_compression:
            # Note: This requires an LLM for compression
            try:
                llm_response = self.llm_manager.llm_interface.select_best_model(
                    "literature_analysis",
                    max_cost=0.01,
                    require_local=True
                )
                if llm_response:
                    compressor = LLMChainExtractor.from_llm(
                        self.llm_manager.llm_interface.generate
                    )
                    retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=retriever
                    )
            except:
                self.logger.info("Using basic retriever without compression")
        
        # Create conversational retrieval chain
        self.retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_manager.llm_interface,
            retriever=retriever,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            ),
            verbose=True
        )
    
    def _setup_tools(self):
        """Set up tools for the agent."""
        self.tools = [
            Tool(
                name="knowledge_retrieval",
                description="Retrieve relevant biomedical knowledge from the knowledge base",
                func=self._retrieve_knowledge
            ),
            Tool(
                name="hypothesis_generation",
                description="Generate biomedical hypotheses based on context and observations",
                func=self._generate_hypothesis
            ),
            Tool(
                name="literature_search",
                description="Search biomedical literature for specific topics",
                func=self._search_literature
            ),
            Tool(
                name="entity_exploration",
                description="Explore relationships and properties of biomedical entities",
                func=self._explore_entity
            ),
            Tool(
                name="experimental_design",
                description="Suggest experimental approaches for testing hypotheses",
                func=self._suggest_experiments
            )
        ]
    
    def _retrieve_knowledge(self, query: str) -> str:
        """Retrieve relevant knowledge from the vector store."""
        if not self.vector_store:
            return "Knowledge base not available."
        
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant knowledge found."
            
            knowledge_pieces = []
            for i, doc in enumerate(docs):
                knowledge_pieces.append(f"{i+1}. {doc.page_content[:300]}...")
            
            return "\n\n".join(knowledge_pieces)
            
        except Exception as e:
            return f"Error retrieving knowledge: {str(e)}"
    
    def _generate_hypothesis(self, context: str) -> str:
        """Generate hypothesis based on context."""
        try:
            # Use the hypothesis generation system if available
            prompt = f"""Based on the following biomedical context, generate a testable hypothesis:

Context: {context}

Generate a hypothesis that includes:
1. Clear hypothesis statement
2. Proposed mechanism
3. Testable predictions
4. Experimental approach

Hypothesis:"""
            
            response = self.llm_manager.process_biomedical_task(
                task="hypothesis_generation",
                input_data={"context": context, "observation": ""},
                temperature=0.3
            )
            
            return response.content
            
        except Exception as e:
            return f"Error generating hypothesis: {str(e)}"
    
    def _search_literature(self, query: str) -> str:
        """Search biomedical literature."""
        # Placeholder implementation
        return f"Literature search for '{query}' would return relevant papers from PubMed and other databases. This requires integration with literature APIs."
    
    def _explore_entity(self, entity: str) -> str:
        """Explore biomedical entity relationships."""
        # Use knowledge retrieval for now
        return self._retrieve_knowledge(f"What is {entity}? What are its functions and relationships?")
    
    def _suggest_experiments(self, hypothesis: str) -> str:
        """Suggest experimental approaches."""
        try:
            prompt = f"""Suggest experimental approaches to test this biomedical hypothesis:

Hypothesis: {hypothesis}

Provide:
1. In vitro experiments
2. In vivo experiments  
3. Clinical studies (if applicable)
4. Controls and considerations

Experimental Design:"""
            
            response = self.llm_manager.process_biomedical_task(
                task="validation",
                input_data={"hypothesis": hypothesis, "evidence": ""},
                temperature=0.2
            )
            
            return response.content
            
        except Exception as e:
            return f"Error suggesting experiments: {str(e)}"
    
    def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main chat interface for the RAG agent.
        
        Args:
            user_message: User's question or message
            context: Optional additional context
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Add message to memory
        self.memory.add_message("user", user_message, context)
        
        try:
            # Get conversation context
            conversation_history = self.memory.get_conversation_history()
            biomedical_context = self.memory.get_biomedical_context_summary()
            
            # Construct enhanced prompt
            enhanced_prompt = f"""
{self.system_prompt}

Current Biomedical Context: {biomedical_context}

Recent Conversation:
{conversation_history}

Current Question: {user_message}

Please provide a comprehensive, evidence-based response. If you need to retrieve specific knowledge, search literature, or generate hypotheses, indicate what additional information would be helpful.

Response:"""
            
            # Generate response using LLM
            if self.retrieval_chain and LANGCHAIN_AVAILABLE:
                # Use RAG chain if available
                response = self.retrieval_chain({"question": user_message})
                answer = response["answer"]
            else:
                # Use direct LLM call
                llm_response = self.llm_manager.process_biomedical_task(
                    task="literature_analysis",
                    input_data=enhanced_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                answer = llm_response.content
            
            # Add response to memory
            self.memory.add_message("assistant", answer)
            
            response_time = time.time() - start_time
            
            return {
                "response": answer,
                "conversation_id": self.memory.session_id,
                "biomedical_context": self.memory.biomedical_context,
                "response_time": response_time,
                "retrieved_knowledge": self._retrieve_knowledge(user_message)[:500] + "...",
                "suggested_follow_ups": self._generate_follow_up_questions(user_message, answer)
            }
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return {
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question.",
                "conversation_id": self.memory.session_id,
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    def _generate_follow_up_questions(self, user_message: str, response: str) -> List[str]:
        """Generate suggested follow-up questions."""
        # Simple heuristic-based follow-up generation
        follow_ups = []
        
        # Extract entities from user message
        entities = self.memory.biomedical_context.active_entities
        
        if entities:
            entity = entities[-1]  # Most recent entity
            follow_ups.extend([
                f"What are the latest research developments regarding {entity}?",
                f"How does {entity} interact with other biological pathways?",
                f"What are potential therapeutic targets related to {entity}?"
            ])
        
        # Add general research follow-ups
        follow_ups.extend([
            "Can you suggest experiments to test this hypothesis?",
            "What are the clinical implications of this research?",
            "Are there any contradictory findings in the literature?"
        ])
        
        return follow_ups[:3]  # Return top 3
    
    def add_knowledge_documents(self, documents: List[Document]):
        """Add new documents to the knowledge base."""
        if not self.vector_store:
            self.logger.warning("Vector store not initialized")
            return
        
        try:
            # Add documents to existing vector store
            self.vector_store.add_documents(documents)
            
            # Save updated vector store
            if self.config.vector_store_type == "faiss":
                vector_store_path = self.knowledge_base_path / "vector_store"
                self.vector_store.save_local(str(vector_store_path))
            
            self.logger.info(f"Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "session_id": self.memory.session_id,
            "message_count": len(self.memory.messages),
            "biomedical_context": asdict(self.memory.biomedical_context),
            "recent_topics": self.memory.biomedical_context.active_entities[-5:],
            "research_questions": len(self.memory.biomedical_context.research_questions)
        }
    
    def reset_conversation(self):
        """Reset the conversation memory."""
        self.memory = ConversationMemory(max_length=self.config.max_conversation_length)
        self.logger.info("Conversation reset")
    
    def export_conversation(self, file_path: str):
        """Export conversation history to file."""
        conversation_data = {
            "session_id": self.memory.session_id,
            "messages": self.memory.messages,
            "biomedical_context": asdict(self.memory.biomedical_context),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        self.logger.info(f"Conversation exported to {file_path}")