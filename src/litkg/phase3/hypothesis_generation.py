"""
Hypothesis Generation and Validation System

This module implements AI-powered agents for generating testable biomedical hypotheses
based on novel relationship predictions and literature analysis. It uses LangChain
agents with specialized tools for biological reasoning and validation.

Key components:
1. Hypothesis Generator - Creates testable hypotheses from novel relations
2. Biological Reasoning Agent - Validates hypotheses using domain knowledge
3. Experimental Design Assistant - Suggests experiments to test hypotheses
4. Literature Validation Agent - Cross-references against existing literature
5. Research Priority Ranker - Prioritizes hypotheses by potential impact
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
from pathlib import Path
import re

# LangChain imports for agents and tools
try:
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain.tools.base import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Local imports
from ..utils.logging import LoggerMixin
from .novelty_detection import NovelRelation, NoveltyDetectionSystem
from .confidence_scoring import ConfidenceScorer


@dataclass
class BiomedicalHypothesis:
    """
    Represents a generated biomedical hypothesis.

    A hypothesis can be built either from its single-sentence statement
    (``hypothesis_text``) or from the fuller ``title``/``description``
    decomposition. Whichever is supplied, __post_init__ fills in the other, so
    both readings are always available.
    """
    id: str = ""
    title: str = ""
    description: str = ""
    hypothesis_text: str = ""
    domain: str = ""
    based_on_relations: List[str] = field(default_factory=list)
    biological_mechanism: str = ""
    testable_predictions: List[str] = field(default_factory=list)
    experimental_approaches: List[str] = field(default_factory=list)
    potential_impact: str = ""
    confidence_score: float = 0.0
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    priority_score: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    research_questions: List[str] = field(default_factory=list)
    generated_timestamp: str = ""
    validation_status: str = "pending"  # pending, validated, rejected, in_progress

    def __post_init__(self):
        """Keep hypothesis_text, title, and description mutually consistent."""
        statement = self.hypothesis_text or self.description or self.title

        if statement:
            if not self.hypothesis_text:
                self.hypothesis_text = statement
            if not self.description:
                self.description = statement
            if not self.title:
                # Titles are the short form of the statement
                self.title = statement if len(statement) <= 80 else statement[:77] + "..."

        if not self.generated_timestamp:
            self.generated_timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExperimentalDesign:
    """
    Represents a suggested experimental design.

    ``timeline`` and ``estimated_duration`` are two names for the same value
    and are kept in sync by __post_init__.
    """
    hypothesis_id: str = ""
    objective: str = ""
    experiment_type: str = ""  # "in_vitro", "in_vivo", "clinical", "computational"
    methodology: str = ""
    experimental_groups: List[str] = field(default_factory=list)
    measurements: List[str] = field(default_factory=list)
    statistical_analysis: str = ""
    required_resources: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    timeline: str = ""
    estimated_duration: str = ""
    estimated_cost: str = ""
    success_probability: float = 0.0
    ethical_considerations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Keep timeline and estimated_duration consistent."""
        if self.estimated_duration and not self.timeline:
            self.timeline = self.estimated_duration
        elif self.timeline and not self.estimated_duration:
            self.estimated_duration = self.timeline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class BiologicalReasoningTool(BaseTool):
    """Tool for biological reasoning and mechanism validation."""
    
    name: str = "biological_reasoning"
    description: str = "Analyzes biological mechanisms and validates hypotheses using domain knowledge"
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _run(self, query: str) -> str:
        """Run biological reasoning on a query."""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are a biomedical expert with deep knowledge of molecular biology, genetics, pharmacology, and disease mechanisms.

Query: {query}

Please provide a detailed biological analysis addressing:
1. Known biological mechanisms
2. Molecular pathways involved
3. Potential interactions and effects
4. Biological plausibility assessment
5. Relevant precedents in the literature

Provide a comprehensive but concise analysis:
"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(query=query)
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)


class LiteratureValidationTool(BaseTool):
    """Tool for validating hypotheses against existing literature."""
    
    name: str = "literature_validation"
    description: str = "Validates hypotheses by checking against existing biomedical literature"
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _run(self, hypothesis: str) -> str:
        """Validate hypothesis against literature."""
        prompt = PromptTemplate(
            input_variables=["hypothesis"],
            template="""
You are a biomedical literature expert. Analyze the following hypothesis and assess:

Hypothesis: {hypothesis}

Please evaluate:
1. Existing evidence that supports this hypothesis
2. Contradicting evidence or findings
3. Gaps in current knowledge
4. Related research areas and publications
5. Novelty assessment compared to known research

Provide a structured analysis with specific references to research areas:
"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(hypothesis=hypothesis)
    
    async def _arun(self, hypothesis: str) -> str:
        """Async version of _run."""
        return self._run(hypothesis)


class ExperimentalDesignTool(BaseTool):
    """Tool for designing experiments to test hypotheses."""
    
    name: str = "experimental_design"
    description: str = "Designs experiments to test biomedical hypotheses"
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _run(self, hypothesis: str) -> str:
        """Design experiments for a hypothesis."""
        prompt = PromptTemplate(
            input_variables=["hypothesis"],
            template="""
You are an expert in experimental design for biomedical research. Design experiments to test:

Hypothesis: {hypothesis}

Please provide:
1. Experimental approaches (in vitro, in vivo, clinical, computational)
2. Specific methodologies and protocols
3. Required controls and comparisons
4. Sample sizes and statistical considerations
5. Expected outcomes and measurements
6. Timeline and resource requirements
7. Potential limitations and confounding factors

Design comprehensive but feasible experiments:
"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(hypothesis=hypothesis)
    
    async def _arun(self, hypothesis: str) -> str:
        """Async version of _run."""
        return self._run(hypothesis)


class HypothesisGenerator(LoggerMixin):
    """
    Generates testable biomedical hypotheses from novel relationship predictions.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE

        # Always defined so callers can inspect or substitute it, even when no
        # LLM backend could be initialized.
        self.llm = None

        if self.use_llm:
            self._initialize_llm()

        # Template-based hypothesis generation for non-LLM mode
        self.hypothesis_templates = {
            "TREATS": "{entity1} may be an effective therapeutic target for {entity2} treatment",
            "CAUSES": "{entity1} dysfunction may contribute to the development of {entity2}",
            "PREVENTS": "{entity1} may have protective effects against {entity2}",
            "INHIBITS": "{entity1} may inhibit {entity2} activity, suggesting therapeutic potential",
            "ACTIVATES": "{entity1} may activate {entity2}, indicating a regulatory relationship",
            "INTERACTS_WITH": "{entity1} and {entity2} may interact in previously unknown pathways",
            "ASSOCIATED_WITH": "There may be a functional relationship between {entity1} and {entity2}"
        }
        
        self.logger.info(f"Initialized HypothesisGenerator (LLM: {self.use_llm})")
    
    def _initialize_llm(self):
        """Initialize LLM for hypothesis generation."""
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
            else:
                self.use_llm = False
                self.logger.warning("No LLM API keys found")
        except Exception as e:
            self.use_llm = False
            self.logger.warning(f"Could not initialize LLM: {e}")
    
    def generate_hypothesis(
        self,
        context: Dict[str, Any]
    ) -> BiomedicalHypothesis:
        """
        Generate a single hypothesis from a context description.

        Where generate_hypotheses() works from predicted NovelRelations, this
        takes a free-form context (entities, relations, domain) and produces
        one hypothesis, using the LLM when available and a template otherwise.

        Args:
            context: May contain "entities", "relations", "domain", "text".

        Returns:
            The generated hypothesis.
        """
        entities = context.get("entities", [])
        relations = context.get("relations", [])
        domain = context.get("domain", "")

        if self.llm is not None:
            try:
                response = self.llm.generate(self._format_context_prompt(context))

                # Accept either a structured dict or raw text
                if isinstance(response, dict):
                    statement = response.get("hypothesis", "")
                    confidence = float(response.get("confidence", 0.5))
                    evidence = list(response.get("evidence", []))
                else:
                    statement = str(response)
                    confidence = 0.5
                    evidence = []

                if statement:
                    return BiomedicalHypothesis(
                        id=f"hypothesis_{abs(hash(statement)) % 10**8}",
                        hypothesis_text=statement,
                        domain=domain,
                        confidence_score=confidence,
                        supporting_evidence=evidence,
                        based_on_relations=[
                            f"{r.get('head')}-{r.get('relation')}-{r.get('tail')}"
                            for r in relations
                        ],
                    )
            except Exception as e:
                self.logger.warning(f"LLM hypothesis generation failed, using template: {e}")

        # Template fallback built from the supplied relations
        if relations:
            first = relations[0]
            template = self.hypothesis_templates.get(
                first.get("relation", "ASSOCIATED_WITH"),
                self.hypothesis_templates["ASSOCIATED_WITH"],
            )
            statement = template.format(
                entity1=first.get("head", "entity1"),
                entity2=first.get("tail", "entity2"),
            )
        elif entities:
            statement = (
                f"There may be an uncharacterized relationship among "
                f"{', '.join(map(str, entities))}"
            )
        else:
            statement = "Insufficient context to formulate a hypothesis"

        return BiomedicalHypothesis(
            id=f"hypothesis_{abs(hash(statement)) % 10**8}",
            hypothesis_text=statement,
            domain=domain,
            confidence_score=0.5,
            based_on_relations=[
                f"{r.get('head')}-{r.get('relation')}-{r.get('tail')}" for r in relations
            ],
        )

    def _format_context_prompt(self, context: Dict[str, Any]) -> str:
        """Render a context dict into a hypothesis-generation prompt."""
        parts = ["Generate a testable biomedical hypothesis."]

        if context.get("domain"):
            parts.append(f"Domain: {context['domain']}")
        if context.get("entities"):
            parts.append(f"Entities: {', '.join(map(str, context['entities']))}")
        if context.get("relations"):
            rendered = "; ".join(
                f"{r.get('head')} {r.get('relation')} {r.get('tail')}"
                for r in context["relations"]
            )
            parts.append(f"Known relations: {rendered}")
        if context.get("text"):
            parts.append(f"Context: {context['text']}")

        return "\n".join(parts)

    def _design_experiment(
        self,
        hypothesis: Union[BiomedicalHypothesis, Dict[str, Any]]
    ) -> ExperimentalDesign:
        """
        Build an experimental design for a hypothesis.

        Args:
            hypothesis: The hypothesis to test, as an object or a dict.

        Returns:
            A concrete experimental design.
        """
        if isinstance(hypothesis, dict):
            objective = hypothesis.get("hypothesis") or hypothesis.get("objective", "")
            hypothesis_id = hypothesis.get("id", "")
            approaches = hypothesis.get("experimental_approaches", [])
            predictions = hypothesis.get("testable_predictions", [])
        else:
            objective = hypothesis.hypothesis_text or hypothesis.description
            hypothesis_id = hypothesis.id
            approaches = hypothesis.experimental_approaches
            predictions = hypothesis.testable_predictions

        experiment_type = approaches[0] if approaches else "in_vitro"

        return ExperimentalDesign(
            hypothesis_id=hypothesis_id,
            objective=objective,
            experiment_type=experiment_type,
            methodology=f"{experiment_type.replace('_', ' ').title()} assay",
            experimental_groups=["control", "treatment"],
            measurements=predictions or ["primary outcome"],
            statistical_analysis="Two-group comparison with multiple-testing correction",
            required_resources=["reagents", "instrumentation", "analyst time"],
            expected_outcomes=predictions,
            controls=["vehicle control", "positive control"],
            timeline="6 months",
            estimated_cost="unknown",
            success_probability=0.5,
            ethical_considerations=["Institutional approval required"],
        )

    def design_experiment(
        self,
        hypothesis: Union[BiomedicalHypothesis, Dict[str, Any]]
    ) -> ExperimentalDesign:
        """
        Public entry point for experimental design.

        Args:
            hypothesis: The hypothesis to test.

        Returns:
            A concrete experimental design.
        """
        design = self._design_experiment(hypothesis)
        self.logger.info(f"Designed {design.experiment_type} experiment for {design.hypothesis_id!r}")
        return design

    def generate_hypotheses(
        self,
        novel_relations: List[NovelRelation],
        max_hypotheses: int = 20
    ) -> List[BiomedicalHypothesis]:
        """
        Generate testable hypotheses from novel relationships.
        
        Args:
            novel_relations: List of predicted novel relationships
            max_hypotheses: Maximum number of hypotheses to generate
            
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        for i, relation in enumerate(novel_relations[:max_hypotheses]):
            if self.use_llm:
                hypothesis = self._generate_llm_hypothesis(relation, i)
            else:
                hypothesis = self._generate_template_hypothesis(relation, i)
            
            if hypothesis:
                hypotheses.append(hypothesis)
        
        self.logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def _generate_llm_hypothesis(self, relation: NovelRelation, index: int) -> Optional[BiomedicalHypothesis]:
        """Generate hypothesis using LLM."""
        prompt = PromptTemplate(
            input_variables=["entity1", "entity2", "relation", "confidence", "evidence"],
            template="""
You are a biomedical researcher generating testable hypotheses from predicted relationships.

Predicted Relationship:
- Entity 1: {entity1}
- Entity 2: {entity2}
- Relationship Type: {relation}
- Confidence Score: {confidence}
- Evidence: {evidence}

Generate a comprehensive hypothesis that includes:
1. A clear, testable hypothesis statement
2. Proposed biological mechanism
3. Specific testable predictions (3-5 predictions)
4. Potential experimental approaches
5. Expected impact if validated
6. Key research questions to address

Format your response as follows:
TITLE: [Concise hypothesis title]
HYPOTHESIS: [Main hypothesis statement]
MECHANISM: [Proposed biological mechanism]
PREDICTIONS: [List of testable predictions]
EXPERIMENTS: [Suggested experimental approaches]
IMPACT: [Potential research impact]
QUESTIONS: [Key research questions]

Generate a scientifically rigorous and testable hypothesis:
"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(
                entity1=relation.entity1,
                entity2=relation.entity2,
                relation=relation.relation_type,
                confidence=relation.confidence_score,
                evidence="; ".join(relation.evidence_sources)
            )
            
            return self._parse_llm_response(result, relation, index)
            
        except Exception as e:
            self.logger.error(f"Error generating LLM hypothesis: {e}")
            return self._generate_template_hypothesis(relation, index)
    
    def _generate_template_hypothesis(self, relation: NovelRelation, index: int) -> BiomedicalHypothesis:
        """Generate hypothesis using templates."""
        template = self.hypothesis_templates.get(
            relation.relation_type, 
            self.hypothesis_templates["ASSOCIATED_WITH"]
        )
        
        title = template.format(entity1=relation.entity1, entity2=relation.entity2)
        
        return BiomedicalHypothesis(
            id=f"hyp_{index:03d}",
            title=title,
            description=f"Based on predicted {relation.relation_type} relationship between {relation.entity1} and {relation.entity2}",
            based_on_relations=[f"{relation.entity1}-{relation.relation_type}-{relation.entity2}"],
            biological_mechanism="To be determined through experimental validation",
            testable_predictions=[
                f"Experimental manipulation of {relation.entity1} will affect {relation.entity2}",
                f"Co-expression or co-localization of {relation.entity1} and {relation.entity2}",
                f"Functional assays will demonstrate {relation.relation_type} relationship"
            ],
            experimental_approaches=[
                "In vitro functional assays",
                "Gene expression analysis",
                "Protein interaction studies"
            ],
            potential_impact="May reveal new therapeutic targets or biological mechanisms",
            confidence_score=relation.confidence_score,
            novelty_score=0.7,
            feasibility_score=0.6,
            priority_score=relation.confidence_score * 0.7 * 0.6,
            supporting_evidence=relation.evidence_sources,
            contradicting_evidence=[],
            research_questions=[
                f"What is the molecular mechanism of {relation.entity1}-{relation.entity2} interaction?",
                f"Is this relationship direct or mediated by other factors?",
                f"What are the therapeutic implications?"
            ],
            generated_timestamp=datetime.now().isoformat()
        )
    
    def _parse_llm_response(self, response: str, relation: NovelRelation, index: int) -> Optional[BiomedicalHypothesis]:
        """Parse LLM response into structured hypothesis."""
        try:
            # Extract sections using regex
            sections = {}
            patterns = {
                'title': r'TITLE:\s*(.+?)(?=\n|$)',
                'hypothesis': r'HYPOTHESIS:\s*(.+?)(?=\n[A-Z]+:|$)',
                'mechanism': r'MECHANISM:\s*(.+?)(?=\n[A-Z]+:|$)',
                'predictions': r'PREDICTIONS:\s*(.+?)(?=\n[A-Z]+:|$)',
                'experiments': r'EXPERIMENTS:\s*(.+?)(?=\n[A-Z]+:|$)',
                'impact': r'IMPACT:\s*(.+?)(?=\n[A-Z]+:|$)',
                'questions': r'QUESTIONS:\s*(.+?)(?=\n[A-Z]+:|$)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                sections[key] = match.group(1).strip() if match else ""
            
            # Parse lists
            predictions = self._parse_list(sections.get('predictions', ''))
            experiments = self._parse_list(sections.get('experiments', ''))
            questions = self._parse_list(sections.get('questions', ''))
            
            return BiomedicalHypothesis(
                id=f"hyp_{index:03d}",
                title=sections.get('title', f"Hypothesis {index+1}"),
                description=sections.get('hypothesis', ''),
                based_on_relations=[f"{relation.entity1}-{relation.relation_type}-{relation.entity2}"],
                biological_mechanism=sections.get('mechanism', ''),
                testable_predictions=predictions,
                experimental_approaches=experiments,
                potential_impact=sections.get('impact', ''),
                confidence_score=relation.confidence_score,
                novelty_score=0.8,  # High novelty for LLM-generated hypotheses
                feasibility_score=0.7,
                priority_score=relation.confidence_score * 0.8 * 0.7,
                supporting_evidence=relation.evidence_sources,
                contradicting_evidence=[],
                research_questions=questions,
                generated_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._generate_template_hypothesis(relation, index)
    
    def _parse_list(self, text: str) -> List[str]:
        """Parse numbered or bulleted list from text."""
        if not text:
            return []
        
        # Split by common list indicators
        items = re.split(r'\n\s*(?:\d+\.|\-|\*|\•)\s*', text)
        items = [item.strip() for item in items if item.strip()]
        
        # If no list indicators found, split by sentences
        if len(items) <= 1:
            items = [s.strip() for s in text.split('.') if s.strip()]
        
        return items[:5]  # Limit to 5 items


class HypothesisValidationAgent(LoggerMixin):
    """
    AI agent for validating and refining biomedical hypotheses using multiple tools.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE
        
        if self.use_llm:
            self._initialize_agent()
        
        self.validation_results = {}
        self.logger.info(f"Initialized HypothesisValidationAgent (LLM: {self.use_llm})")
    
    def _initialize_agent(self):
        """Initialize the validation agent with tools."""
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.1)
            else:
                self.use_llm = False
                return
            
            # Initialize tools
            self.tools = [
                BiologicalReasoningTool(self.llm),
                LiteratureValidationTool(self.llm),
                ExperimentalDesignTool(self.llm)
            ]
            
            # Create agent prompt
            agent_prompt = PromptTemplate(
                input_variables=["hypothesis", "tools", "tool_names", "agent_scratchpad"],
                template="""
You are a biomedical research validation agent. Your task is to thoroughly validate hypotheses using available tools.

Available tools:
{tools}

Tool names: {tool_names}

Hypothesis to validate: {hypothesis}

Please use the available tools to:
1. Analyze the biological mechanism and plausibility
2. Validate against existing literature
3. Design experiments to test the hypothesis
4. Provide an overall assessment

Use this format:
Thought: I need to validate this hypothesis systematically
Action: [tool_name]
Action Input: [input_to_tool]
Observation: [tool_output]
... (repeat as needed)
Final Answer: [comprehensive validation summary]

Begin:
{agent_scratchpad}
"""
            )
            
            # Create agent
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=5
            )
            
        except Exception as e:
            self.use_llm = False
            self.logger.error(f"Could not initialize validation agent: {e}")
    
    def validate_hypothesis(self, hypothesis: BiomedicalHypothesis) -> Dict[str, Any]:
        """
        Validate a hypothesis using the AI agent.
        
        Args:
            hypothesis: The hypothesis to validate
            
        Returns:
            Validation results dictionary
        """
        if not self.use_llm:
            return self._simple_validation(hypothesis)
        
        try:
            hypothesis_text = f"""
Title: {hypothesis.title}
Description: {hypothesis.description}
Mechanism: {hypothesis.biological_mechanism}
Predictions: {'; '.join(hypothesis.testable_predictions)}
"""
            
            result = self.agent_executor.invoke({
                "hypothesis": hypothesis_text
            })
            
            output = result.get("output", "")
            biological_score = self._extract_plausibility_score(output)
            literature_score = self._extract_literature_assessment(output)

            validation_result = {
                "hypothesis_id": hypothesis.id,
                "validation_summary": output,
                "literature_validation": {
                    "score": literature_score,
                    "reasoning": "Assessed by LLM agent",
                },
                "biological_validation": {
                    "score": biological_score,
                    "reasoning": "Assessed by LLM agent",
                },
                "biological_plausibility": biological_score,
                "literature_support": literature_score,
                "experimental_feasibility": self._extract_feasibility_score(output),
                "overall_score": 0.0,  # Will be calculated
                "validation_timestamp": datetime.now().isoformat(),
                "recommendations": self._extract_recommendations(output)
            }
            
            # Calculate overall score
            validation_result["overall_score"] = (
                validation_result["biological_plausibility"] * 0.4 +
                validation_result["literature_support"] * 0.3 +
                validation_result["experimental_feasibility"] * 0.3
            )
            
            self.validation_results[hypothesis.id] = validation_result
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating hypothesis {hypothesis.id}: {e}")
            return self._simple_validation(hypothesis)
    
    def _validate_against_literature(
        self, hypothesis: BiomedicalHypothesis
    ) -> Dict[str, Any]:
        """
        Assess how well existing literature supports a hypothesis.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            {"score", "supporting_papers", "contradicting_papers", "reasoning"}.
        """
        supporting = len(hypothesis.supporting_evidence)
        contradicting = len(hypothesis.contradicting_evidence)

        if supporting == 0 and contradicting == 0:
            return {
                "score": 0.5,
                "supporting_papers": 0,
                "contradicting_papers": 0,
                "reasoning": "No literature evidence recorded; treated as neutral",
            }

        # Proportion of supporting evidence, damped toward neutral when the
        # total evidence base is small
        total = supporting + contradicting
        proportion = supporting / total
        weight = min(total / 5.0, 1.0)
        score = 0.5 + (proportion - 0.5) * weight

        return {
            "score": float(max(0.0, min(1.0, score))),
            "supporting_papers": supporting,
            "contradicting_papers": contradicting,
            "reasoning": (
                f"{supporting} supporting vs {contradicting} contradicting "
                f"item(s) of evidence"
            ),
        }

    def _validate_biological_plausibility(
        self, hypothesis: BiomedicalHypothesis
    ) -> Dict[str, Any]:
        """
        Assess whether a hypothesis is mechanistically coherent.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            {"score", "reasoning"}.
        """
        # A stated mechanism and concrete predictions are what make a
        # hypothesis biologically checkable rather than merely suggestive.
        has_mechanism = bool(hypothesis.biological_mechanism.strip())
        num_predictions = len(hypothesis.testable_predictions)

        score = hypothesis.confidence_score * 0.6
        if has_mechanism:
            score += 0.25
        score += min(num_predictions / 4.0, 1.0) * 0.15

        reasons = []
        reasons.append(
            "mechanism described" if has_mechanism else "no mechanism described"
        )
        reasons.append(f"{num_predictions} testable prediction(s)")

        return {
            "score": float(max(0.0, min(1.0, score))),
            "reasoning": "; ".join(reasons),
        }

    def _simple_validation(self, hypothesis: BiomedicalHypothesis) -> Dict[str, Any]:
        """Validate without an LLM, using the component assessments."""
        literature = self._validate_against_literature(hypothesis)
        biological = self._validate_biological_plausibility(hypothesis)
        feasibility = hypothesis.feasibility_score

        overall = (
            biological["score"] * 0.4
            + literature["score"] * 0.3
            + feasibility * 0.3
        )

        return {
            "hypothesis_id": hypothesis.id,
            "validation_summary": "Rule-based validation without LLM analysis",
            "literature_validation": literature,
            "biological_validation": biological,
            "biological_plausibility": biological["score"],
            "literature_support": literature["score"],
            "experimental_feasibility": feasibility,
            "overall_score": float(overall),
            "validation_timestamp": datetime.now().isoformat(),
            "recommendations": ["Requires detailed literature review", "Design specific experiments"]
        }

    def _extract_plausibility_score(self, text: str) -> float:
        """Extract biological plausibility score from validation text."""
        # Look for numerical scores in the text
        import re
        scores = re.findall(r'(?:plausibility|plausible).*?(\d*\.?\d+)', text, re.IGNORECASE)
        if scores:
            try:
                return min(float(scores[0]), 1.0)
            except:
                pass
        return 0.6  # Default score
    
    def _extract_literature_assessment(self, text: str) -> float:
        """Extract literature support assessment."""
        # Simple keyword-based assessment
        positive_keywords = ['supported', 'evidence', 'consistent', 'validated']
        negative_keywords = ['contradicts', 'lacks', 'insufficient', 'no evidence']
        
        text_lower = text.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count:
            return 0.7
        elif negative_count > positive_count:
            return 0.3
        else:
            return 0.5
    
    def _extract_feasibility_score(self, text: str) -> float:
        """Extract experimental feasibility score."""
        feasible_keywords = ['feasible', 'straightforward', 'standard', 'established']
        difficult_keywords = ['challenging', 'complex', 'expensive', 'difficult']
        
        text_lower = text.lower()
        feasible_count = sum(1 for kw in feasible_keywords if kw in text_lower)
        difficult_count = sum(1 for kw in difficult_keywords if kw in text_lower)
        
        if feasible_count > difficult_count:
            return 0.8
        elif difficult_count > feasible_count:
            return 0.4
        else:
            return 0.6
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from validation text."""
        # Look for recommendation patterns
        import re
        recommendations = re.findall(
            r'(?:recommend|suggest|should|need to)([^.]+)', 
            text, 
            re.IGNORECASE
        )
        return [rec.strip() for rec in recommendations[:5]]


class HypothesisGenerationSystem(LoggerMixin):
    """
    Complete system for hypothesis generation and validation.
    """
    
    def __init__(
        self,
        novelty_detection_system: Optional[NoveltyDetectionSystem] = None,
        use_llm: bool = True
    ):
        self.novelty_system = novelty_detection_system
        self.use_llm = use_llm
        
        # Initialize components
        self.hypothesis_generator = HypothesisGenerator(use_llm=use_llm)
        self.validation_agent = HypothesisValidationAgent(use_llm=use_llm)
        
        # Results storage
        self.generated_hypotheses = []
        self.validation_results = {}
        
        self.logger.info("Initialized HypothesisGenerationSystem")

    def rank_hypotheses(
        self,
        hypotheses: List[BiomedicalHypothesis],
        weights: Optional[Dict[str, float]] = None
    ) -> List[BiomedicalHypothesis]:
        """
        Rank hypotheses by research priority, highest first.

        Priority balances how likely a hypothesis is to be true (confidence),
        how much it would add if true (novelty), how testable it is
        (feasibility), and how much evidence already backs it.

        Each hypothesis has its ``priority_score`` set as a side effect.

        Args:
            hypotheses: The hypotheses to rank.
            weights: Optional overrides for the "confidence", "novelty",
                "feasibility", and "evidence" components.

        Returns:
            A new list ordered by descending priority.
        """
        component_weights = {
            "confidence": 0.4,
            "novelty": 0.3,
            "feasibility": 0.2,
            "evidence": 0.1,
            **(weights or {}),
        }

        for hypothesis in hypotheses:
            # Evidence support saturates: 3+ items count as fully supported
            evidence_score = min(len(hypothesis.supporting_evidence) / 3.0, 1.0)

            hypothesis.priority_score = float(
                component_weights["confidence"] * hypothesis.confidence_score
                + component_weights["novelty"] * hypothesis.novelty_score
                + component_weights["feasibility"] * hypothesis.feasibility_score
                + component_weights["evidence"] * evidence_score
            )

        ranked = sorted(hypotheses, key=lambda h: h.priority_score, reverse=True)
        self.logger.info(f"Ranked {len(ranked)} hypotheses by research priority")
        return ranked

    def generate_hypotheses(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate and rank hypotheses from novel relations and context.

        Args:
            input_data: {"novel_relations": [NovelRelation], "literature_context": [str]}

        Returns:
            {"hypotheses": [...], "summary": {...}}, hypotheses ranked by priority.
        """
        novel_relations = input_data.get("novel_relations", [])
        literature_context = input_data.get("literature_context", [])

        hypotheses = []
        for relation in novel_relations:
            context = {
                "entities": [relation.entity1, relation.entity2],
                "relations": [{
                    "head": relation.entity1,
                    "relation": relation.relation_type,
                    "tail": relation.entity2,
                }],
                "domain": input_data.get("domain", ""),
                "text": " ".join(map(str, literature_context)),
            }

            hypothesis = self.hypothesis_generator.generate_hypothesis(context)
            # Carry the relation's novelty through to the hypothesis
            if not hypothesis.novelty_score:
                hypothesis.novelty_score = relation.novelty_score
            hypotheses.append(hypothesis)

        ranked = self.rank_hypotheses(hypotheses)
        self.generated_hypotheses = ranked

        return {
            "hypotheses": ranked,
            "summary": {
                "num_hypotheses": len(ranked),
                "num_source_relations": len(novel_relations),
            },
        }
    
    def generate_and_validate_hypotheses(
        self,
        novel_relations: List[NovelRelation],
        max_hypotheses: int = 20,
        validate_all: bool = True
    ) -> Tuple[List[BiomedicalHypothesis], Dict[str, Any]]:
        """
        Complete pipeline for hypothesis generation and validation.
        
        Args:
            novel_relations: List of novel relationships to generate hypotheses from
            max_hypotheses: Maximum number of hypotheses to generate
            validate_all: Whether to validate all generated hypotheses
            
        Returns:
            Tuple of (generated hypotheses, validation results)
        """
        self.logger.info("Starting hypothesis generation and validation pipeline")
        
        # Step 1: Generate hypotheses
        hypotheses = self.hypothesis_generator.generate_hypotheses(
            novel_relations=novel_relations,
            max_hypotheses=max_hypotheses
        )
        
        self.generated_hypotheses = hypotheses
        self.logger.info(f"Generated {len(hypotheses)} hypotheses")
        
        # Step 2: Validate hypotheses
        validation_results = {}
        if validate_all and self.use_llm:
            for hypothesis in hypotheses:
                try:
                    result = self.validation_agent.validate_hypothesis(hypothesis)
                    validation_results[hypothesis.id] = result
                    
                    # Update hypothesis with validation results
                    hypothesis.validation_status = "validated"
                    hypothesis.priority_score = result["overall_score"]
                    
                except Exception as e:
                    self.logger.error(f"Error validating hypothesis {hypothesis.id}: {e}")
                    validation_results[hypothesis.id] = {
                        "error": str(e),
                        "overall_score": 0.3
                    }
        
        self.validation_results = validation_results
        
        # Step 3: Rank hypotheses by priority
        self.generated_hypotheses.sort(key=lambda h: h.priority_score, reverse=True)
        
        self.logger.info(f"Completed hypothesis pipeline: {len(hypotheses)} hypotheses, {len(validation_results)} validated")
        
        return hypotheses, validation_results
    
    def generate_research_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive research report with hypotheses and validation.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Complete research report dictionary
        """
        report = {
            "research_report": {
                "title": "Novel Biomedical Hypotheses from AI-Powered Discovery",
                "generated_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_hypotheses": len(self.generated_hypotheses),
                    "validated_hypotheses": len(self.validation_results),
                    "high_priority_hypotheses": len([
                        h for h in self.generated_hypotheses 
                        if h.priority_score >= 0.7
                    ]),
                    "methodology": "AI-powered hypothesis generation from novel relation predictions"
                }
            },
            "top_hypotheses": [
                h.to_dict() for h in self.generated_hypotheses[:10]
            ],
            "all_hypotheses": [
                h.to_dict() for h in self.generated_hypotheses
            ],
            "validation_summary": self._create_validation_summary(),
            "research_priorities": self._rank_research_priorities(),
            "experimental_recommendations": self._generate_experimental_recommendations(),
            "impact_assessment": self._assess_potential_impact()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Research report saved to {output_path}")
        
        return report
    
    def _create_validation_summary(self) -> Dict[str, Any]:
        """Create summary of validation results."""
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        scores = [r.get("overall_score", 0) for r in self.validation_results.values()]
        
        return {
            "total_validated": len(self.validation_results),
            "mean_validation_score": sum(scores) / len(scores) if scores else 0,
            "high_quality_hypotheses": len([s for s in scores if s >= 0.7]),
            "validation_distribution": {
                "excellent (>0.8)": len([s for s in scores if s > 0.8]),
                "good (0.6-0.8)": len([s for s in scores if 0.6 <= s <= 0.8]),
                "moderate (0.4-0.6)": len([s for s in scores if 0.4 <= s < 0.6]),
                "low (<0.4)": len([s for s in scores if s < 0.4])
            }
        }
    
    def _rank_research_priorities(self) -> List[Dict[str, Any]]:
        """Rank hypotheses by research priority."""
        priorities = []
        
        for hypothesis in self.generated_hypotheses[:15]:  # Top 15
            priority = {
                "hypothesis_id": hypothesis.id,
                "title": hypothesis.title,
                "priority_score": hypothesis.priority_score,
                "confidence": hypothesis.confidence_score,
                "novelty": hypothesis.novelty_score,
                "feasibility": hypothesis.feasibility_score,
                "potential_impact": hypothesis.potential_impact,
                "key_research_questions": hypothesis.research_questions[:3]
            }
            priorities.append(priority)
        
        return priorities
    
    def _generate_experimental_recommendations(self) -> Dict[str, List[str]]:
        """Generate experimental recommendations by category."""
        recommendations = {
            "immediate_experiments": [],
            "medium_term_studies": [],
            "long_term_research": [],
            "collaborative_opportunities": []
        }
        
        for hypothesis in self.generated_hypotheses[:10]:
            if hypothesis.priority_score >= 0.7:
                recommendations["immediate_experiments"].extend(
                    hypothesis.experimental_approaches[:2]
                )
            elif hypothesis.priority_score >= 0.5:
                recommendations["medium_term_studies"].extend(
                    hypothesis.experimental_approaches[:1]
                )
            else:
                recommendations["long_term_research"].extend(
                    hypothesis.experimental_approaches[:1]
                )
        
        # Add collaborative opportunities
        recommendations["collaborative_opportunities"] = [
            "Bioinformatics partnerships for data analysis",
            "Clinical collaborations for validation studies",
            "Industry partnerships for drug development",
            "International research consortiums"
        ]
        
        return recommendations
    
    def _assess_potential_impact(self) -> Dict[str, Any]:
        """Assess potential research impact."""
        high_impact_count = len([
            h for h in self.generated_hypotheses 
            if h.priority_score >= 0.8
        ])
        
        return {
            "high_impact_hypotheses": high_impact_count,
            "potential_therapeutic_targets": len([
                h for h in self.generated_hypotheses 
                if "therapeutic" in h.potential_impact.lower()
            ]),
            "novel_mechanisms": len([
                h for h in self.generated_hypotheses 
                if "mechanism" in h.biological_mechanism.lower()
            ]),
            "research_areas": list(set([
                rel.split('-')[1] for h in self.generated_hypotheses 
                for rel in h.based_on_relations
            ])),
            "estimated_publication_potential": min(len(self.generated_hypotheses) * 2, 50),
            "translational_opportunities": [
                "Drug target identification",
                "Biomarker discovery",
                "Diagnostic development",
                "Therapeutic mechanism elucidation"
            ]
        }