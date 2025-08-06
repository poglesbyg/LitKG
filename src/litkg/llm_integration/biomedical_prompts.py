"""
Biomedical Prompt Templates and Management

This module provides specialized prompt templates optimized for biomedical
LLM tasks, including entity extraction, relation extraction, hypothesis
generation, and validation.

Features:
- Domain-specific prompt engineering
- Few-shot learning examples
- Task-specific optimizations
- Prompt versioning and management
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class PromptType(Enum):
    """Types of biomedical prompts."""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    VALIDATION = "validation"
    LITERATURE_ANALYSIS = "literature_analysis"
    MECHANISM_EXPLANATION = "mechanism_explanation"


@dataclass
class PromptExample:
    """Example for few-shot prompting."""
    input_text: str
    expected_output: str
    explanation: Optional[str] = None


class EntityExtractionPrompts:
    """Prompts for biomedical entity extraction."""
    
    SYSTEM_PROMPT = """You are an expert biomedical entity extractor with deep knowledge of:
- Genes and proteins (e.g., BRCA1, p53, insulin)
- Diseases and conditions (e.g., breast cancer, diabetes, hypertension)
- Drugs and compounds (e.g., tamoxifen, aspirin, glucose)
- Biological processes (e.g., apoptosis, DNA repair, metabolism)
- Cell types and tissues (e.g., T cells, neurons, liver tissue)
- Organisms and model systems (e.g., human, mouse, E. coli)

Extract entities with high precision and include confidence scores."""
    
    BASIC_TEMPLATE = """Extract biomedical entities from the following text.

Text: {text}

Instructions:
1. Identify all biomedical entities
2. Classify each entity by type (GENE, PROTEIN, DISEASE, DRUG, PROCESS, etc.)
3. Provide confidence score (0-1)
4. List entities in this format:
   - Entity: [entity_name] | Type: [entity_type] | Confidence: [score]

Entities:"""
    
    EXAMPLES = [
        PromptExample(
            input_text="BRCA1 mutations increase the risk of breast cancer in women.",
            expected_output="""- Entity: BRCA1 | Type: GENE | Confidence: 0.95
- Entity: mutations | Type: MUTATION | Confidence: 0.90
- Entity: breast cancer | Type: DISEASE | Confidence: 0.98
- Entity: women | Type: DEMOGRAPHIC | Confidence: 0.85""",
            explanation="Clear biomedical entities with high confidence"
        ),
        PromptExample(
            input_text="Treatment with tamoxifen reduced tumor growth in mice.",
            expected_output="""- Entity: tamoxifen | Type: DRUG | Confidence: 0.99
- Entity: tumor growth | Type: PHENOTYPE | Confidence: 0.90
- Entity: mice | Type: ORGANISM | Confidence: 0.95""",
            explanation="Drug treatment and biological outcome"
        )
    ]


class RelationExtractionPrompts:
    """Prompts for biomedical relation extraction."""
    
    SYSTEM_PROMPT = """You are an expert at identifying biological relationships with knowledge of:
- Gene-disease associations
- Drug-target interactions
- Protein-protein interactions
- Regulatory relationships
- Causal mechanisms
- Therapeutic relationships

Extract meaningful relationships that are explicitly stated or strongly implied."""
    
    BASIC_TEMPLATE = """Extract relationships between biomedical entities in the text.

Text: {text}
Entities: {entities}

Instructions:
1. Identify relationships between the given entities
2. Use these relationship types: TREATS, CAUSES, PREVENTS, INHIBITS, ACTIVATES, REGULATES, ASSOCIATED_WITH, INTERACTS_WITH
3. Provide confidence score and evidence text
4. Format: Entity1 --[RELATIONSHIP]--> Entity2 | Confidence: [score] | Evidence: [text]

Relationships:"""
    
    EXAMPLES = [
        PromptExample(
            input_text="BRCA1 mutations cause increased susceptibility to breast cancer.",
            expected_output="""BRCA1 mutations --[CAUSES]--> breast cancer | Confidence: 0.92 | Evidence: "BRCA1 mutations cause increased susceptibility to breast cancer" """,
            explanation="Clear causal relationship"
        ),
        PromptExample(
            input_text="Tamoxifen is used to treat hormone receptor-positive breast cancer.",
            expected_output="""tamoxifen --[TREATS]--> breast cancer | Confidence: 0.95 | Evidence: "Tamoxifen is used to treat hormone receptor-positive breast cancer" """,
            explanation="Therapeutic relationship"
        )
    ]


class HypothesisGenerationPrompts:
    """Prompts for biomedical hypothesis generation."""
    
    SYSTEM_PROMPT = """You are a creative biomedical researcher who generates testable hypotheses based on:
- Biological mechanisms and pathways
- Experimental observations
- Literature findings
- Molecular interactions
- Disease processes

Generate hypotheses that are novel, testable, and biologically plausible."""
    
    BASIC_TEMPLATE = """Generate a testable biomedical hypothesis based on the context and observation.

Context: {context}
Observation: {observation}

Instructions:
1. Generate a clear, testable hypothesis
2. Explain the proposed biological mechanism
3. List 3-5 specific testable predictions
4. Suggest experimental approaches
5. Assess potential impact

Format your response as:
HYPOTHESIS: [main hypothesis statement]
MECHANISM: [biological mechanism explanation]
PREDICTIONS:
- [prediction 1]
- [prediction 2]
- [prediction 3]
EXPERIMENTS:
- [experimental approach 1]
- [experimental approach 2]
IMPACT: [potential research impact]

Response:"""
    
    EXAMPLES = [
        PromptExample(
            input_text="Context: EGFR mutations predict response to tyrosine kinase inhibitors. Observation: Some patients develop resistance.",
            expected_output="""HYPOTHESIS: Secondary mutations in the EGFR kinase domain cause acquired resistance to EGFR inhibitors by altering drug binding affinity.
MECHANISM: Resistance mutations (e.g., T790M) change the ATP-binding pocket structure, reducing inhibitor binding while maintaining kinase activity.
PREDICTIONS:
- Resistant tumors will harbor secondary EGFR mutations
- T790M mutation will reduce drug binding affinity
- Combination therapy may prevent resistance
EXPERIMENTS:
- Sequence EGFR in resistant vs sensitive tumors
- Structural analysis of mutant EGFR-drug complexes
- Test combination therapies in cell models
IMPACT: Could lead to personalized resistance monitoring and combination therapy strategies."""
        )
    ]


class ValidationPrompts:
    """Prompts for hypothesis and prediction validation."""
    
    SYSTEM_PROMPT = """You are a rigorous biomedical validator who assesses:
- Biological plausibility based on known mechanisms
- Consistency with existing literature
- Experimental feasibility
- Statistical significance of evidence
- Potential confounding factors

Provide evidence-based, critical evaluations."""
    
    BASIC_TEMPLATE = """Evaluate the biological plausibility of this hypothesis given the evidence.

Hypothesis: {hypothesis}
Evidence: {evidence}

Instructions:
1. Assess biological plausibility (1-10 scale)
2. List supporting evidence
3. Identify contradicting evidence or concerns
4. Evaluate experimental feasibility
5. Provide overall confidence and recommendations

Format:
PLAUSIBILITY SCORE: [1-10]
SUPPORTING EVIDENCE:
- [evidence 1]
- [evidence 2]
CONCERNS:
- [concern 1]
- [concern 2]
FEASIBILITY: [experimental feasibility assessment]
CONFIDENCE: [overall confidence level]
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]

Evaluation:"""
    
    EXAMPLES = [
        PromptExample(
            input_text="Hypothesis: PARP inhibitors are effective in BRCA1-deficient tumors. Evidence: Clinical trials show response rates of 60% in BRCA1-mutated cancers.",
            expected_output="""PLAUSIBILITY SCORE: 9
SUPPORTING EVIDENCE:
- Synthetic lethality concept well-established
- BRCA1 deficiency impairs homologous recombination
- Clinical trial data supports efficacy
CONCERNS:
- Resistance mechanisms may develop
- Patient selection criteria need refinement
FEASIBILITY: High - established clinical protocols exist
CONFIDENCE: High - strong mechanistic and clinical evidence
RECOMMENDATIONS:
- Monitor for resistance biomarkers
- Investigate combination therapies
- Expand to other HR-deficient cancers"""
        )
    ]


class BiomedicalPromptManager:
    """Manages biomedical prompt templates and examples."""
    
    def __init__(self):
        self.prompts = {
            PromptType.ENTITY_EXTRACTION: EntityExtractionPrompts(),
            PromptType.RELATION_EXTRACTION: RelationExtractionPrompts(),
            PromptType.HYPOTHESIS_GENERATION: HypothesisGenerationPrompts(),
            PromptType.VALIDATION: ValidationPrompts()
        }
    
    def get_prompt_template(
        self,
        prompt_type: PromptType,
        use_few_shot: bool = True,
        include_system_prompt: bool = True
    ) -> PromptTemplate:
        """Get a prompt template for a specific task."""
        
        prompt_class = self.prompts[prompt_type]
        
        if use_few_shot and hasattr(prompt_class, 'EXAMPLES'):
            # Create few-shot prompt
            examples = [
                {
                    "input": example.input_text,
                    "output": example.expected_output
                }
                for example in prompt_class.EXAMPLES
            ]
            
            example_template = PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}"
            )
            
            if prompt_type == PromptType.ENTITY_EXTRACTION:
                return FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=example_template,
                    prefix="Here are examples of biomedical entity extraction:",
                    suffix=prompt_class.BASIC_TEMPLATE,
                    input_variables=["text"]
                )
            
            elif prompt_type == PromptType.RELATION_EXTRACTION:
                return FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=example_template,
                    prefix="Here are examples of biomedical relation extraction:",
                    suffix=prompt_class.BASIC_TEMPLATE,
                    input_variables=["text", "entities"]
                )
            
            elif prompt_type == PromptType.HYPOTHESIS_GENERATION:
                return FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=example_template,
                    prefix="Here are examples of biomedical hypothesis generation:",
                    suffix=prompt_class.BASIC_TEMPLATE,
                    input_variables=["context", "observation"]
                )
            
            elif prompt_type == PromptType.VALIDATION:
                return FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=example_template,
                    prefix="Here are examples of biomedical hypothesis validation:",
                    suffix=prompt_class.BASIC_TEMPLATE,
                    input_variables=["hypothesis", "evidence"]
                )
        
        else:
            # Create basic prompt template
            template = prompt_class.BASIC_TEMPLATE
            
            if include_system_prompt:
                template = f"System: {prompt_class.SYSTEM_PROMPT}\n\n{template}"
            
            if prompt_type == PromptType.ENTITY_EXTRACTION:
                return PromptTemplate(
                    input_variables=["text"],
                    template=template
                )
            elif prompt_type == PromptType.RELATION_EXTRACTION:
                return PromptTemplate(
                    input_variables=["text", "entities"],
                    template=template
                )
            elif prompt_type == PromptType.HYPOTHESIS_GENERATION:
                return PromptTemplate(
                    input_variables=["context", "observation"],
                    template=template
                )
            elif prompt_type == PromptType.VALIDATION:
                return PromptTemplate(
                    input_variables=["hypothesis", "evidence"],
                    template=template
                )
    
    def get_system_prompt(self, prompt_type: PromptType) -> str:
        """Get system prompt for a task type."""
        return self.prompts[prompt_type].SYSTEM_PROMPT
    
    def get_examples(self, prompt_type: PromptType) -> List[PromptExample]:
        """Get few-shot examples for a task type."""
        prompt_class = self.prompts[prompt_type]
        return getattr(prompt_class, 'EXAMPLES', [])
    
    def create_custom_prompt(
        self,
        template: str,
        input_variables: List[str],
        system_prompt: Optional[str] = None
    ) -> PromptTemplate:
        """Create a custom prompt template."""
        if system_prompt:
            template = f"System: {system_prompt}\n\n{template}"
        
        return PromptTemplate(
            input_variables=input_variables,
            template=template
        )
    
    def optimize_prompt_for_model(
        self,
        prompt_type: PromptType,
        model_name: str,
        context_length: int = 4096
    ) -> PromptTemplate:
        """Optimize prompt template for specific model capabilities."""
        
        # Model-specific optimizations
        if "llama" in model_name.lower():
            # Llama models prefer structured formats
            use_few_shot = context_length > 2048
            include_system = True
        elif "mistral" in model_name.lower():
            # Mistral models are efficient with concise prompts
            use_few_shot = context_length > 4096
            include_system = True
        elif "gpt" in model_name.lower():
            # GPT models handle complex prompts well
            use_few_shot = True
            include_system = True
        else:
            # Default configuration
            use_few_shot = context_length > 2048
            include_system = True
        
        return self.get_prompt_template(
            prompt_type=prompt_type,
            use_few_shot=use_few_shot,
            include_system_prompt=include_system
        )


# Global instance for easy access
biomedical_prompts = BiomedicalPromptManager()