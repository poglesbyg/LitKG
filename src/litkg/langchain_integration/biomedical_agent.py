"""
Conversational agents for biomedical research.

Provides a natural-language interface over the LitKG pipeline: retrieval,
entity extraction, hypothesis generation, and validation, exposed as tools an
agent can select between and driven by a conversation loop that remembers what
has been discussed.

Components:
- BiomedicalToolkit: the tools an agent can call
- BiomedicalQueryAgent: conversational front end with memory
- HypothesisGenerationAgent: proposes testable hypotheses from context
- LiteratureValidationAgent: checks claims against the literature

All of these run on whichever provider is configured, defaulting to the local
Ollama model, so no API key is required.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging import LoggerMixin

try:
    from langchain_core.tools import Tool
    LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TOOLS_AVAILABLE = False


@dataclass
class ConversationTurn:
    """One exchange in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BiomedicalToolkit(LoggerMixin):
    """
    The set of capabilities a biomedical agent can invoke.

    Each tool is a plain callable taking and returning a string, so the toolkit
    is usable with or without LangChain's agent machinery.

    Tools return error text rather than raising: that is the LangChain tool
    convention, where the agent reads the message and recovers, and an
    exception would abort the whole run.
    """

    def __init__(
        self,
        rag_system: Optional[Any] = None,
        entity_extractor: Optional[Any] = None,
        hypothesis_generator: Optional[Any] = None,
        literature_validator: Optional[Any] = None
    ):
        self.rag_system = rag_system
        self.entity_extractor = entity_extractor
        self.hypothesis_generator = hypothesis_generator
        self.literature_validator = literature_validator

        self.logger.info(
            f"Initialized BiomedicalToolkit with {len(self.tool_specs())} tool(s)"
        )

    def tool_specs(self) -> List[Dict[str, Any]]:
        """Describe the tools available given what was wired in."""
        specs = []

        if self.rag_system is not None:
            specs.append({
                "name": "search_knowledge",
                "description": (
                    "Answer a biomedical question from retrieved literature and "
                    "knowledge graph evidence. Input: the question."
                ),
                "func": self.search_knowledge,
            })
        if self.entity_extractor is not None:
            specs.append({
                "name": "extract_entities",
                "description": (
                    "Extract genes, diseases, drugs and processes from text. "
                    "Input: the text."
                ),
                "func": self.extract_entities,
            })
        if self.hypothesis_generator is not None:
            specs.append({
                "name": "generate_hypothesis",
                "description": (
                    "Propose a testable hypothesis from a described context. "
                    "Input: the context."
                ),
                "func": self.generate_hypothesis,
            })
        if self.literature_validator is not None:
            specs.append({
                "name": "validate_claim",
                "description": (
                    "Check a claim against published literature and report "
                    "supporting and contradicting evidence. Input: the claim."
                ),
                "func": self.validate_claim,
            })

        return specs

    def as_langchain_tools(self) -> List[Any]:
        """Expose the toolkit as LangChain Tool objects."""
        if not LANGCHAIN_TOOLS_AVAILABLE:
            self.logger.warning("LangChain tools unavailable; returning no tools")
            return []

        return [
            Tool(name=spec["name"], description=spec["description"], func=spec["func"])
            for spec in self.tool_specs()
        ]

    def search_knowledge(self, query: str) -> str:
        """Answer a question from retrieved evidence."""
        if self.rag_system is None:
            return "Knowledge search is not configured."

        try:
            result = self.rag_system.answer(query)
            sources = result.get("num_sources", 0)
            return f"{result['answer']}\n\n(drawn from {sources} source(s))"
        except Exception as e:
            self.logger.error(f"search_knowledge failed: {e}")
            return f"Error searching knowledge: {e}"

    def extract_entities(self, text: str) -> str:
        """Extract biomedical entities from text."""
        if self.entity_extractor is None:
            return "Entity extraction is not configured."

        try:
            extraction = self.entity_extractor.extract_entities_and_relations(text)
            entities = getattr(extraction, "entities", extraction)
            if not entities:
                return "No biomedical entities found."
            return "\n".join(f"- {e}" for e in entities)
        except Exception as e:
            self.logger.error(f"extract_entities failed: {e}")
            return f"Error extracting entities: {e}"

    def generate_hypothesis(self, context: str) -> str:
        """Propose a hypothesis from a described context."""
        if self.hypothesis_generator is None:
            return "Hypothesis generation is not configured."

        try:
            hypothesis = self.hypothesis_generator.generate_hypothesis({"text": context})
            return (
                f"{hypothesis.hypothesis_text}\n"
                f"(confidence {hypothesis.confidence_score:.2f})"
            )
        except Exception as e:
            self.logger.error(f"generate_hypothesis failed: {e}")
            return f"Error generating hypothesis: {e}"

    def validate_claim(self, claim: str) -> str:
        """Check a claim against published literature."""
        if self.literature_validator is None:
            return "Literature validation is not configured."

        try:
            from ..phase3.hypothesis_generation import BiomedicalHypothesis

            result = self.literature_validator.validate(
                BiomedicalHypothesis(hypothesis_text=claim)
            )
            details = result.details
            return (
                f"Literature support: {result.score:.2f} "
                f"({details.get('supporting_papers', 0)} supporting, "
                f"{details.get('contradicting_papers', 0)} contradicting)"
            )
        except Exception as e:
            self.logger.error(f"validate_claim failed: {e}")
            return f"Error validating claim: {e}"


class BiomedicalQueryAgent(LoggerMixin):
    """
    Conversational interface to the LitKG pipeline.

    Routes each question to the most appropriate tool and keeps conversation
    history so follow-up questions read in context.
    """

    # Keyword cues used to route a question when no LLM router is configured
    ROUTING_CUES = {
        "extract_entities": ("extract", "entities", "identify genes", "which genes"),
        "generate_hypothesis": ("hypothesis", "hypothesise", "hypothesize", "propose", "why might"),
        "validate_claim": ("validate", "is it true", "verify", "evidence for", "does the literature"),
    }

    def __init__(
        self,
        toolkit: Optional[BiomedicalToolkit] = None,
        llm_manager: Optional[Any] = None,
        max_history: int = 10
    ):
        """
        Args:
            toolkit: Tools the agent may call.
            llm_manager: Object exposing process_biomedical_task(); defaults to
                UnifiedLLMManager (local Ollama first).
            max_history: Conversation turns to retain.
        """
        self.toolkit = toolkit or BiomedicalToolkit()
        self.max_history = max_history
        self.history: List[ConversationTurn] = []

        if llm_manager is None:
            from ..llm_integration.unified_llm_interface import UnifiedLLMManager
            llm_manager = UnifiedLLMManager()
        self.llm_manager = llm_manager

        self.logger.info("Initialized BiomedicalQueryAgent")

    def _select_tool(self, question: str) -> Optional[Callable[[str], str]]:
        """Choose the tool best suited to a question, if any."""
        available = {spec["name"]: spec["func"] for spec in self.toolkit.tool_specs()}
        lowered = question.lower()

        for tool_name, cues in self.ROUTING_CUES.items():
            if tool_name in available and any(cue in lowered for cue in cues):
                self.logger.debug(f"Routing to {tool_name}")
                return available[tool_name]

        # Anything else is a knowledge question when retrieval is configured
        return available.get("search_knowledge")

    def conversation_context(self) -> str:
        """Render recent history as context for a follow-up question."""
        if not self.history:
            return ""

        turns = self.history[-self.max_history:]
        return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in turns)

    def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a message in the context of the conversation so far.

        Args:
            message: The user's message.
            **kwargs: Additional generation options.

        Returns:
            {"response", "tool_used", "history_length"}.
        """
        self.history.append(ConversationTurn(role="user", content=message))

        tool = self._select_tool(message)
        tool_used = None

        if tool is not None:
            tool_used = getattr(tool, "__name__", "tool")
            response = tool(message)
        else:
            # No tools configured: answer directly, with history for continuity
            context = self.conversation_context()
            prompt = (
                f"{context}\n\nUser: {message}\n\nAssistant:" if context else message
            )
            try:
                result = self.llm_manager.process_biomedical_task(
                    task="literature_analysis", input_data=prompt, **kwargs
                )
                response = result.content
            except Exception as e:
                self.logger.error(f"Direct answer failed: {e}")
                response = f"Error: {e}"

        self.history.append(
            ConversationTurn(
                role="assistant", content=response, metadata={"tool": tool_used}
            )
        )
        # Keep both sides of the retained turns
        self.history = self.history[-self.max_history * 2:]

        return {
            "response": response,
            "tool_used": tool_used,
            "history_length": len(self.history),
        }

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history.clear()
        self.logger.info("Conversation history cleared")


class HypothesisGenerationAgent(LoggerMixin):
    """
    Proposes and ranks testable hypotheses.

    Thin conversational wrapper over the Phase 3 hypothesis machinery, so the
    same generation and ranking logic backs both the API and the chat surface.
    """

    def __init__(self, hypothesis_system: Optional[Any] = None):
        if hypothesis_system is None:
            from ..phase3.hypothesis_generation import HypothesisGenerationSystem
            hypothesis_system = HypothesisGenerationSystem()
        self.hypothesis_system = hypothesis_system

        self.logger.info("Initialized HypothesisGenerationAgent")

    def propose(
        self,
        context: str,
        domain: str = "",
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Propose a hypothesis from free-form context.

        Args:
            context: Background the hypothesis should build on.
            domain: Optional research domain.
            entities: Optional entities the hypothesis should involve.

        Returns:
            {"hypothesis", "confidence", "testable_predictions"}.
        """
        hypothesis = self.hypothesis_system.hypothesis_generator.generate_hypothesis({
            "text": context,
            "domain": domain,
            "entities": entities or [],
        })

        return {
            "hypothesis": hypothesis.hypothesis_text,
            "confidence": hypothesis.confidence_score,
            "testable_predictions": hypothesis.testable_predictions,
            "domain": hypothesis.domain,
        }

    def propose_from_relations(
        self,
        novel_relations: List[Any],
        literature_context: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Propose hypotheses from predicted novel relations, ranked by priority.

        Args:
            novel_relations: NovelRelation objects to build on.
            literature_context: Optional supporting context passages.

        Returns:
            Ranked hypothesis summaries, highest priority first.
        """
        results = self.hypothesis_system.generate_hypotheses({
            "novel_relations": novel_relations,
            "literature_context": literature_context or [],
        })

        return [
            {
                "hypothesis": h.hypothesis_text,
                "confidence": h.confidence_score,
                "novelty": h.novelty_score,
                "priority": h.priority_score,
            }
            for h in results["hypotheses"]
        ]


class LiteratureValidationAgent(LoggerMixin):
    """
    Checks claims and hypotheses against published literature.

    Wraps the Phase 3 literature cross-validator so agents and chat surfaces
    share one notion of what counts as literature support.
    """

    def __init__(self, validator: Optional[Any] = None):
        if validator is None:
            from ..phase3.validation import LiteratureCrossValidator
            validator = LiteratureCrossValidator()
        self.validator = validator

        self.logger.info("Initialized LiteratureValidationAgent")

    def validate_claim(self, claim: str) -> Dict[str, Any]:
        """
        Assess how well the literature supports a claim.

        Args:
            claim: The claim, as a sentence.

        Returns:
            {"claim", "score", "supporting_papers", "contradicting_papers",
             "verdict"}.
        """
        from ..phase3.hypothesis_generation import BiomedicalHypothesis

        result = self.validator.validate(BiomedicalHypothesis(hypothesis_text=claim))
        details = result.details

        if result.score >= 0.7:
            verdict = "supported"
        elif result.score <= 0.3:
            verdict = "contradicted"
        else:
            verdict = "inconclusive"

        return {
            "claim": claim,
            "score": result.score,
            "supporting_papers": details.get("supporting_papers", 0),
            "contradicting_papers": details.get("contradicting_papers", 0),
            "verdict": verdict,
        }

    def validate_hypothesis(self, hypothesis: Any) -> Dict[str, Any]:
        """
        Validate a BiomedicalHypothesis against the literature.

        Args:
            hypothesis: The hypothesis to validate.

        Returns:
            The validation result as a dictionary.
        """
        return self.validator.validate(hypothesis).to_dict()
