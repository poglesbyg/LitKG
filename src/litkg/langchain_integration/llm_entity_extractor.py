"""
LLM-powered Entity and Relation Extraction using LangChain

This module provides advanced entity and relation extraction capabilities
using large language models, significantly improving accuracy over
rule-based approaches.

Key features:
1. Few-shot prompting for biomedical entity extraction
2. Chain-of-thought reasoning for relation extraction
3. Confidence scoring using multiple LLM consensus
4. Structured output parsing with validation
5. Domain-specific prompt templates
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from pathlib import Path

# LangChain imports
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM

# LangChain provider imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic  
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFacePipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Pydantic for structured outputs
from pydantic import BaseModel, Field, validator
from pydantic.v1 import BaseModel as BaseModelV1, Field as FieldV1

# Local imports
from ..utils.logging import LoggerMixin
from ..phase1.literature_processor import Entity, Relation


class EntityType(Enum):
    """Biomedical entity types."""
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    DISEASE = "DISEASE"
    DRUG = "DRUG"
    CHEMICAL = "CHEMICAL"
    CELL_TYPE = "CELL_TYPE"
    TISSUE = "TISSUE"
    ORGANISM = "ORGANISM"
    MUTATION = "MUTATION"
    PATHWAY = "PATHWAY"
    PHENOTYPE = "PHENOTYPE"
    BIOMARKER = "BIOMARKER"


class RelationType(Enum):
    """Biomedical relation types."""
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    INTERACTS_WITH = "INTERACTS_WITH"
    REGULATES = "REGULATES"
    EXPRESSED_IN = "EXPRESSED_IN"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INHIBITS = "INHIBITS"
    ACTIVATES = "ACTIVATES"
    BINDS = "BINDS"
    PREDICTS = "PREDICTS"
    CORRELATES_WITH = "CORRELATES_WITH"


class ExtractedEntity(BaseModelV1):
    """Structured entity extraction result."""
    text: str = FieldV1(description="The entity text as it appears in the document")
    entity_type: str = FieldV1(description="The type of entity (GENE, DISEASE, DRUG, etc.)")
    start_pos: Optional[int] = FieldV1(description="Start position in text", default=None)
    end_pos: Optional[int] = FieldV1(description="End position in text", default=None)
    confidence: float = FieldV1(description="Confidence score between 0 and 1", default=0.0)
    canonical_name: Optional[str] = FieldV1(description="Canonical/preferred name", default=None)
    synonyms: List[str] = FieldV1(description="Alternative names/synonyms", default_factory=list)
    context: Optional[str] = FieldV1(description="Surrounding context", default=None)


class ExtractedRelation(BaseModelV1):
    """Structured relation extraction result."""
    entity1: str = FieldV1(description="First entity in the relation")
    entity2: str = FieldV1(description="Second entity in the relation")
    relation_type: str = FieldV1(description="Type of relation between entities")
    confidence: float = FieldV1(description="Confidence score between 0 and 1", default=0.0)
    evidence: Optional[str] = FieldV1(description="Text evidence for the relation", default=None)
    context: Optional[str] = FieldV1(description="Sentence or paragraph context", default=None)
    direction: Optional[str] = FieldV1(description="Direction of relation if applicable", default=None)


class EntityExtractionResult(BaseModelV1):
    """Complete entity extraction result."""
    entities: List[ExtractedEntity] = FieldV1(description="List of extracted entities")
    text_analyzed: str = FieldV1(description="The text that was analyzed")
    model_used: str = FieldV1(description="Name of the model used for extraction")
    extraction_confidence: float = FieldV1(description="Overall extraction confidence", default=0.0)


class RelationExtractionResult(BaseModelV1):
    """Complete relation extraction result."""
    relations: List[ExtractedRelation] = FieldV1(description="List of extracted relations")
    entities_used: List[str] = FieldV1(description="Entities that were considered")
    text_analyzed: str = FieldV1(description="The text that was analyzed")
    model_used: str = FieldV1(description="Name of the model used for extraction")
    extraction_confidence: float = FieldV1(description="Overall extraction confidence", default=0.0)


class BiomedicalPromptTemplates:
    """Collection of prompt templates for biomedical NLP tasks."""
    
    # Entity extraction prompts
    ENTITY_EXTRACTION_TEMPLATE = """You are an expert biomedical researcher tasked with extracting entities from scientific literature.

Extract all biomedical entities from the following text. Focus on:
- GENES (e.g., BRCA1, TP53, EGFR)
- PROTEINS (e.g., p53 protein, insulin)
- DISEASES (e.g., breast cancer, diabetes)
- DRUGS (e.g., tamoxifen, aspirin)
- CHEMICALS (e.g., glucose, ATP)
- CELL_TYPES (e.g., T cells, neurons)
- TISSUES (e.g., liver, brain tissue)
- ORGANISMS (e.g., human, mouse)
- MUTATIONS (e.g., p.R175H, deletion)
- PATHWAYS (e.g., p53 pathway, glycolysis)
- PHENOTYPES (e.g., drug resistance, cell death)
- BIOMARKERS (e.g., PSA, HbA1c)

Text to analyze:
{text}

{format_instructions}

Important guidelines:
1. Extract entities exactly as they appear in the text
2. Provide confidence scores based on context and clarity
3. Include canonical names when different from text
4. List relevant synonyms if known
5. Provide surrounding context for ambiguous entities

Extract all relevant biomedical entities:"""

    # Few-shot examples for entity extraction
    ENTITY_EXAMPLES = [
        {
            "text": "BRCA1 mutations are associated with increased risk of breast cancer in women.",
            "entities": [
                {"text": "BRCA1", "entity_type": "GENE", "confidence": 0.95, "canonical_name": "BRCA1"},
                {"text": "mutations", "entity_type": "MUTATION", "confidence": 0.90},
                {"text": "breast cancer", "entity_type": "DISEASE", "confidence": 0.98, "canonical_name": "Breast Neoplasms"}
            ]
        },
        {
            "text": "Treatment with tamoxifen showed significant reduction in tumor growth in mice.",
            "entities": [
                {"text": "tamoxifen", "entity_type": "DRUG", "confidence": 0.99, "canonical_name": "Tamoxifen"},
                {"text": "tumor", "entity_type": "DISEASE", "confidence": 0.85, "canonical_name": "Neoplasms"},
                {"text": "mice", "entity_type": "ORGANISM", "confidence": 0.95, "canonical_name": "Mus musculus"}
            ]
        }
    ]

    # Relation extraction prompts
    RELATION_EXTRACTION_TEMPLATE = """You are an expert biomedical researcher analyzing relationships between biomedical entities.

Given the following text and list of entities, identify all meaningful relationships between them.

Text:
{text}

Entities found:
{entities}

Extract relationships using these relation types:
- TREATS: Entity A treats/cures Entity B
- CAUSES: Entity A causes/leads to Entity B  
- PREVENTS: Entity A prevents/protects against Entity B
- INTERACTS_WITH: Entity A interacts/binds with Entity B
- REGULATES: Entity A regulates/controls Entity B
- EXPRESSED_IN: Entity A is expressed/found in Entity B
- ASSOCIATED_WITH: Entity A is associated/correlated with Entity B
- INHIBITS: Entity A inhibits/suppresses Entity B
- ACTIVATES: Entity A activates/stimulates Entity B
- PREDICTS: Entity A predicts/is a biomarker for Entity B
- CORRELATES_WITH: Entity A correlates with Entity B

{format_instructions}

Guidelines:
1. Only extract relationships explicitly stated or strongly implied
2. Provide confidence scores based on evidence strength
3. Include the text evidence that supports each relationship
4. Consider the direction of the relationship when applicable
5. Avoid inferring relationships not supported by the text

Extract all supported relationships:"""

    # Few-shot examples for relation extraction
    RELATION_EXAMPLES = [
        {
            "text": "BRCA1 mutations increase the risk of developing breast cancer.",
            "entities": ["BRCA1", "mutations", "breast cancer"],
            "relations": [
                {
                    "entity1": "BRCA1 mutations",
                    "entity2": "breast cancer", 
                    "relation_type": "CAUSES",
                    "confidence": 0.90,
                    "evidence": "BRCA1 mutations increase the risk of developing breast cancer"
                }
            ]
        }
    ]

    @classmethod
    def get_entity_extraction_prompt(cls) -> FewShotPromptTemplate:
        """Get few-shot prompt template for entity extraction."""
        example_template = """
Text: {text}
Entities: {entities}
"""
        
        return FewShotPromptTemplate(
            examples=cls.ENTITY_EXAMPLES,
            example_prompt=PromptTemplate(
                input_variables=["text", "entities"],
                template=example_template
            ),
            prefix="Here are some examples of biomedical entity extraction:",
            suffix=cls.ENTITY_EXTRACTION_TEMPLATE,
            input_variables=["text", "format_instructions"]
        )

    @classmethod  
    def get_relation_extraction_prompt(cls) -> FewShotPromptTemplate:
        """Get few-shot prompt template for relation extraction."""
        example_template = """
Text: {text}
Entities: {entities}
Relations: {relations}
"""
        
        return FewShotPromptTemplate(
            examples=cls.RELATION_EXAMPLES,
            example_prompt=PromptTemplate(
                input_variables=["text", "entities", "relations"],
                template=example_template
            ),
            prefix="Here are examples of biomedical relation extraction:",
            suffix=cls.RELATION_EXTRACTION_TEMPLATE,
            input_variables=["text", "entities", "format_instructions"]
        )


class BiomedicalOutputParser(BaseOutputParser):
    """Custom output parser for biomedical extraction results."""
    
    def parse(self, text: str) -> Union[EntityExtractionResult, RelationExtractionResult]:
        """Parse LLM output into structured format."""
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # Fallback to regex parsing
            return self._regex_parse(text)
            
        except Exception as e:
            # Return empty result on parsing failure
            return {"entities": [], "relations": [], "error": str(e)}
    
    def _regex_parse(self, text: str) -> Dict[str, Any]:
        """Fallback regex parsing for malformed outputs."""
        # This is a simplified implementation
        # In practice, you'd want more robust parsing
        entities = []
        relations = []
        
        # Extract entities using patterns
        entity_pattern = r"Entity:\s*(.+?)\s*Type:\s*(\w+)"
        entity_matches = re.findall(entity_pattern, text, re.IGNORECASE)
        
        for match in entity_matches:
            entities.append({
                "text": match[0].strip(),
                "entity_type": match[1].upper(),
                "confidence": 0.5  # Default confidence
            })
        
        return {"entities": entities, "relations": relations}
    
    @property
    def _type(self) -> str:
        return "biomedical_output_parser"


class EntityExtractionChain(LoggerMixin):
    """LangChain-based entity extraction chain."""
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        use_few_shot: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            self.llm = self._get_default_llm()
        
        # Set up output parser
        self.output_parser = PydanticOutputParser(pydantic_object=EntityExtractionResult)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.output_parser,
            llm=self.llm
        )
        
        # Create prompt template
        if use_few_shot:
            self.prompt = BiomedicalPromptTemplates.get_entity_extraction_prompt()
        else:
            self.prompt = PromptTemplate(
                input_variables=["text", "format_instructions"],
                template=BiomedicalPromptTemplates.ENTITY_EXTRACTION_TEMPLATE
            )
        
        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.fixing_parser
        )
        
        self.logger.info(f"Initialized EntityExtractionChain with {model_name}")
    
    def _get_default_llm(self) -> LLM:
        """Get default LLM based on availability."""
        if OPENAI_AVAILABLE:
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature
            )
        elif ANTHROPIC_AVAILABLE:
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=self.temperature
            )
        else:
            raise ValueError("No LLM provider available. Install langchain-openai or langchain-anthropic.")
    
    def extract_entities(self, text: str) -> EntityExtractionResult:
        """Extract entities from text using the LLM chain."""
        try:
            self.logger.info(f"Extracting entities from text of length {len(text)}")
            
            result = self.chain.run(
                text=text,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Ensure result is properly structured
            if isinstance(result, dict):
                result["model_used"] = self.model_name
                result["text_analyzed"] = text[:200] + "..." if len(text) > 200 else text
                return EntityExtractionResult(**result)
            elif isinstance(result, EntityExtractionResult):
                result.model_used = self.model_name
                result.text_analyzed = text[:200] + "..." if len(text) > 200 else text
                return result
            else:
                # Fallback
                return EntityExtractionResult(
                    entities=[],
                    text_analyzed=text[:200] + "..." if len(text) > 200 else text,
                    model_used=self.model_name,
                    extraction_confidence=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {e}")
            return EntityExtractionResult(
                entities=[],
                text_analyzed=text[:200] + "..." if len(text) > 200 else text,
                model_used=self.model_name,
                extraction_confidence=0.0
            )


class RelationExtractionChain(LoggerMixin):
    """LangChain-based relation extraction chain."""
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        use_few_shot: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            self.llm = self._get_default_llm()
        
        # Set up output parser
        self.output_parser = PydanticOutputParser(pydantic_object=RelationExtractionResult)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.output_parser,
            llm=self.llm
        )
        
        # Create prompt template
        if use_few_shot:
            self.prompt = BiomedicalPromptTemplates.get_relation_extraction_prompt()
        else:
            self.prompt = PromptTemplate(
                input_variables=["text", "entities", "format_instructions"],
                template=BiomedicalPromptTemplates.RELATION_EXTRACTION_TEMPLATE
            )
        
        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.fixing_parser
        )
        
        self.logger.info(f"Initialized RelationExtractionChain with {model_name}")
    
    def _get_default_llm(self) -> LLM:
        """Get default LLM based on availability."""
        if OPENAI_AVAILABLE:
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature
            )
        elif ANTHROPIC_AVAILABLE:
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=self.temperature
            )
        else:
            raise ValueError("No LLM provider available. Install langchain-openai or langchain-anthropic.")
    
    def extract_relations(
        self,
        text: str,
        entities: List[Union[str, ExtractedEntity]]
    ) -> RelationExtractionResult:
        """Extract relations between entities in text."""
        try:
            self.logger.info(f"Extracting relations from text with {len(entities)} entities")
            
            # Format entities for prompt
            if entities and isinstance(entities[0], ExtractedEntity):
                entity_list = [entity.text for entity in entities]
            else:
                entity_list = entities
            
            entity_str = ", ".join(entity_list)
            
            result = self.chain.run(
                text=text,
                entities=entity_str,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Ensure result is properly structured
            if isinstance(result, dict):
                result["model_used"] = self.model_name
                result["text_analyzed"] = text[:200] + "..." if len(text) > 200 else text
                result["entities_used"] = entity_list
                return RelationExtractionResult(**result)
            elif isinstance(result, RelationExtractionResult):
                result.model_used = self.model_name
                result.text_analyzed = text[:200] + "..." if len(text) > 200 else text
                result.entities_used = entity_list
                return result
            else:
                # Fallback
                return RelationExtractionResult(
                    relations=[],
                    entities_used=entity_list,
                    text_analyzed=text[:200] + "..." if len(text) > 200 else text,
                    model_used=self.model_name,
                    extraction_confidence=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Error in relation extraction: {e}")
            return RelationExtractionResult(
                relations=[],
                entities_used=entity_list if 'entity_list' in locals() else [],
                text_analyzed=text[:200] + "..." if len(text) > 200 else text,
                model_used=self.model_name,
                extraction_confidence=0.0
            )


class LLMEntityExtractor(LoggerMixin):
    """
    Main LLM-powered entity and relation extractor.
    
    Combines entity and relation extraction chains with confidence scoring
    and validation mechanisms.
    """
    
    def __init__(
        self,
        entity_model: str = "gpt-3.5-turbo",
        relation_model: str = "gpt-3.5-turbo",
        use_consensus: bool = False,
        consensus_models: Optional[List[str]] = None,
        temperature: float = 0.1
    ):
        self.entity_model = entity_model
        self.relation_model = relation_model
        self.use_consensus = use_consensus
        self.consensus_models = consensus_models or ["gpt-3.5-turbo", "gpt-4"]
        self.temperature = temperature
        
        # Initialize extraction chains
        self.entity_chain = EntityExtractionChain(
            model_name=entity_model,
            temperature=temperature
        )
        
        self.relation_chain = RelationExtractionChain(
            model_name=relation_model,
            temperature=temperature
        )
        
        # Initialize consensus chains if requested
        self.consensus_chains = []
        if use_consensus:
            for model in self.consensus_models:
                if model != entity_model:  # Avoid duplicate
                    try:
                        chain = EntityExtractionChain(
                            model_name=model,
                            temperature=temperature
                        )
                        self.consensus_chains.append(chain)
                    except Exception as e:
                        self.logger.warning(f"Could not initialize consensus model {model}: {e}")
        
        self.logger.info(f"Initialized LLMEntityExtractor with {len(self.consensus_chains)+1} models")
    
    def extract_entities_and_relations(
        self,
        text: str,
        use_consensus: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Extract both entities and relations from text.
        
        Args:
            text: Input text to analyze
            use_consensus: Whether to use consensus scoring (overrides instance setting)
            
        Returns:
            Dictionary with entities, relations, and metadata
        """
        use_consensus = use_consensus if use_consensus is not None else self.use_consensus
        
        self.logger.info(f"Extracting entities and relations from text of length {len(text)}")
        
        # Step 1: Extract entities
        if use_consensus and self.consensus_chains:
            entity_result = self._extract_entities_with_consensus(text)
        else:
            entity_result = self.entity_chain.extract_entities(text)
        
        # Step 2: Extract relations using found entities
        relation_result = self.relation_chain.extract_relations(
            text, entity_result.entities
        )
        
        # Step 3: Calculate overall confidence
        overall_confidence = (
            entity_result.extraction_confidence + 
            relation_result.extraction_confidence
        ) / 2
        
        return {
            "entities": entity_result.entities,
            "relations": relation_result.relations,
            "entity_extraction": entity_result,
            "relation_extraction": relation_result,
            "overall_confidence": overall_confidence,
            "text_length": len(text),
            "models_used": {
                "entity_model": self.entity_model,
                "relation_model": self.relation_model,
                "consensus_models": self.consensus_models if use_consensus else []
            }
        }
    
    def _extract_entities_with_consensus(self, text: str) -> EntityExtractionResult:
        """Extract entities using multiple models for consensus scoring."""
        all_results = []
        
        # Get results from primary model
        primary_result = self.entity_chain.extract_entities(text)
        all_results.append(primary_result)
        
        # Get results from consensus models
        for chain in self.consensus_chains:
            try:
                result = chain.extract_entities(text)
                all_results.append(result)
            except Exception as e:
                self.logger.warning(f"Consensus model failed: {e}")
        
        # Merge results with consensus scoring
        return self._merge_entity_results(all_results, text)
    
    def _merge_entity_results(
        self,
        results: List[EntityExtractionResult],
        text: str
    ) -> EntityExtractionResult:
        """Merge multiple entity extraction results using consensus scoring."""
        if not results:
            return EntityExtractionResult(
                entities=[],
                text_analyzed=text,
                model_used="consensus",
                extraction_confidence=0.0
            )
        
        if len(results) == 1:
            return results[0]
        
        # Collect all unique entities
        entity_map = {}  # text -> list of ExtractedEntity
        
        for result in results:
            for entity in result.entities:
                key = entity.text.lower().strip()
                if key not in entity_map:
                    entity_map[key] = []
                entity_map[key].append(entity)
        
        # Calculate consensus entities
        consensus_entities = []
        total_confidence = 0.0
        
        for entity_text, entity_list in entity_map.items():
            # Only include entities found by multiple models
            if len(entity_list) >= len(results) // 2 + 1:  # Majority vote
                # Use the entity with highest confidence
                best_entity = max(entity_list, key=lambda e: e.confidence)
                
                # Update confidence based on consensus
                consensus_confidence = sum(e.confidence for e in entity_list) / len(entity_list)
                consensus_multiplier = len(entity_list) / len(results)
                
                best_entity.confidence = consensus_confidence * consensus_multiplier
                consensus_entities.append(best_entity)
                total_confidence += best_entity.confidence
        
        # Calculate overall confidence
        overall_confidence = total_confidence / max(len(consensus_entities), 1)
        
        return EntityExtractionResult(
            entities=consensus_entities,
            text_analyzed=text,
            model_used="consensus",
            extraction_confidence=overall_confidence
        )