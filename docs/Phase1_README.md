# Phase 1: Foundation - Literature Processing, KG Preprocessing, and Entity Linking

## Overview

Phase 1 establishes the foundation for LitKG-Integrate by implementing three core components:

1. **Literature Processing Pipeline**: Extracts entities, relations, and contexts from biomedical literature using state-of-the-art NLP models
2. **Knowledge Graph Preprocessing**: Standardizes and harmonizes entities across CIVIC, TCGA, and CPTAC datasets
3. **Entity Linking**: Connects literature mentions with knowledge graph entities using fuzzy matching and contextual disambiguation

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Literature        │    │   Knowledge Graph    │    │   Entity Linking   │
│   Processing        │    │   Preprocessing      │    │                     │
├─────────────────────┤    ├──────────────────────┤    ├─────────────────────┤
│ • PubMed Retrieval  │    │ • CIVIC Processing   │    │ • Fuzzy Matching    │
│ • Biomedical NLP    │    │ • TCGA Processing    │    │ • Semantic Matching │
│ • Entity Extraction │    │ • CPTAC Processing   │    │ • Disambiguation    │
│ • Relation Mining   │    │ • Ontology Mapping   │    │ • Context Analysis  │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
           │                           │                           │
           └─────────────────┬─────────────────────────────────────┘
                             │
                    ┌─────────────────┐
                    │   Integrated    │
                    │   Dataset       │
                    │  (Phase 2 Ready)│
                    └─────────────────┘
```

## Components

### 1. Literature Processing Pipeline (`literature_processor.py`)

**Purpose**: Extract structured information from biomedical literature

**Key Features**:
- PubMed API integration for literature retrieval
- Multi-model NLP approach (PubMedBERT, BioBERT, scispacy)
- Entity extraction for genes, diseases, drugs, proteins, etc.
- Relation extraction using pattern-based and contextual methods
- Confidence scoring for extracted information

**Models Used**:
- **PubMedBERT**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **BioBERT**: `dmis-lab/biobert-base-cased-v1.1`
- **scispacy**: `en_ner_bionlp13cg_md` and `en_ner_bc5cdr_md` for typed
  biomedical NER (`en_core_sci_md` emits a single `ENTITY` label and cannot
  distinguish a gene from a disease, so it is used only for parsing)

**Output**: `ProcessedDocument` objects containing:
- Extracted entities with positions and confidence scores
- Relations between entities
- Document metadata (authors, journal, publication date)
- MeSH terms and keywords

### 2. Knowledge Graph Preprocessing (`kg_preprocessor.py`)

**Purpose**: Standardize and integrate heterogeneous biomedical datasets

**Data Sources**:
- **CIVIC**: Clinical interpretations of variants in cancer
- **TCGA**: The Cancer Genome Atlas (genomic and clinical data)
- **CPTAC**: Clinical Proteomic Tumor Analysis Consortium

**Key Features**:
- Unified entity standardization across data sources
- Ontology mapping to UMLS and Gene Ontology
- Entity resolution (see cascade below)
- Relation standardization and validation
- Graph construction with NetworkX

**Output**: Integrated knowledge graph with:
- Standardized entities across all sources
- Harmonized relations with confidence scores
- Ontology mappings (CUI, GO IDs)
- Cross-source entity linking

#### Entity resolution cascade

`merge_duplicate_entities` decides when two records describe the same
real-world thing. Everything graph-shaped depends on it: degree, centrality
and link prediction are all computed on whatever nodes survive, so a
fragmented graph corrupts every downstream signal.

Rules run strongest evidence first, accumulating matches in a **union-find** so
resolution is transitive — if A matches B and B matches C, all three collapse
even when A and C would not have matched directly.

| Rule | Evidence | Example |
|---|---|---|
| 1 | Shared identity identifier (UMLS CUI) | `BRCA1` = `breast cancer 1` via `C0376571` |
| 2 | Identical normalized name | `BRCA-1` = `BRCA1` (punctuation folded) |
| 3 | Synonym overlap | `p53` = `TP53` |
| 4 | Fuzzy similarity ≥ `similarity_threshold` | `EGFR1` ≈ `EGFR` |

```python
stats = builder.merge_duplicate_entities(similarity_threshold=0.9)
# {'ontology': 0, 'exact_name': 393, 'synonym': 0, 'fuzzy': 54, 'merged': 447, ...}
```

The canonical survivor of a cluster is the member carrying an ontology
identifier, then the one with the richest synonym set, so merging never
discards the best-described record. Every surface form in the cluster is
preserved as a synonym.

**Two design points that are easy to get wrong:**

*GO IDs are not identity evidence.* A Gene Ontology term annotates what an
entity **does**, not which entity it **is**. BRCA1 and BRCA2 both carry
`GO:0006281` ("DNA repair") — correctly — so treating that as identity merges
two distinct genes. Only identifiers listed in `IDENTITY_IDENTIFIERS` are
decisive.

*Only rule 4 uses blocking.* Blocking (comparing only candidates sharing a type
and first character) makes the quadratic fuzzy pass tractable. But synonyms are
exactly where surface forms diverge at the first character — `TP53` blocks
under "t" and its synonym `p53` under "p" — so rules 1–3 run globally via an
O(n) index instead.

### 3. Entity Linking (`entity_linker.py`)

**Purpose**: Connect literature entities with knowledge graph entities

**Matching Strategies**:
1. **Exact Matching**: Direct string matches
2. **Fuzzy Matching**: Levenshtein distance, Jaccard similarity
3. **Semantic Matching**: Sentence transformer embeddings
4. **Contextual Disambiguation**: Context-aware entity resolution

**Key Features**:
- Multi-strategy entity matching
- Confidence scoring and ranking
- Disambiguation conflict resolution
- Type compatibility checking
- Batch processing optimization

**Output**: `EntityMatch` objects linking literature entities to KG entities with:
- Similarity and confidence scores
- Match type classification
- Supporting evidence
- Context information

## Installation and Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd LitKG

# Install dependencies
pip install -r requirements.txt

# Setup models and directories
python scripts/setup_models.py
```

### 2. Configuration

Edit `config/config.yaml` to set:
- PubMed API credentials
- AI API keys (Anthropic Claude preferred)
- UMLS API key (optional)
- Processing parameters

### 3. Environment Variables

Create `.env` file:
```bash
PUBMED_EMAIL=your-email@domain.com
PUBMED_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-anthropic-key
UMLS_API_KEY=your-umls-key
```

## Usage

### Quick Start

Run the complete Phase 1 pipeline:

```bash
python scripts/phase1_integration.py
```

### Individual Components

#### Literature Processing
```bash
python scripts/example_literature_processing.py
```

#### Knowledge Graph Preprocessing
```bash
python scripts/example_kg_preprocessing.py
```

#### Entity Linking
```bash
python scripts/example_entity_linking.py
```

### Programmatic Usage

```python
from phase1.literature_processor import LiteratureProcessor
from phase1.kg_preprocessor import KGPreprocessor
from phase1.entity_linker import EntityLinker

# Process literature
lit_processor = LiteratureProcessor()
documents = lit_processor.process_query(
    query="BRCA1 breast cancer",
    max_results=100
)

# Process knowledge graphs
kg_processor = KGPreprocessor()
kg_processor.download_all_data()
kg_processor.process_all_data()

# Link entities
entity_linker = EntityLinker()
entity_linker.load_kg_entities(kg_processor)
linking_results = entity_linker.batch_link_documents(documents)
```

## Output Files

Phase 1 generates several output files in `data/processed/`:

- `combined_literature_results.json`: Processed literature documents
- `integrated_knowledge_graph.json`: Unified knowledge graph
- `entity_linking_results.json`: Entity linking results
- `phase1_integrated_dataset.json`: Complete integrated dataset
- `phase2_graph_data.json`: Graph structure for Phase 2

## Measured results

From `make run-phase1` on the bundled sample data. These are counts the
pipeline reports, not benchmark accuracy — no labelled evaluation set exists
for this corpus yet.

| Metric | Value |
|---|---|
| Documents processed | 100 |
| Graph size | 2084 nodes, 6003 edges |
| Entities merged by resolution | 447 |
| High-confidence linking rate | 75.2% |
| Cross-modal entities (literature ↔ KG) | 100 |
| Novel literature entities | 1000 |
| Literature entities (resolved from 1504 mentions) | 309 |
| Literature relations extracted | 78 |

### Relation extraction

Relations are found by looking for trigger phrases in the span *between* two
known entities.

The earlier implementation matched regexes whose capture groups had to coincide
with entity text, which required both entities to be single words directly
adjacent to the trigger verb. Real prose rarely obliges: "BRCA1 mutations are
associated with breast cancer" captured `("are", "breast")`, neither an entity,
so every candidate was discarded — **zero relations across all 100 documents**.

Working from the between-span removes that coupling. Triggers cover TREATS,
CAUSES, INHIBITS, ACTIVATES, MUTATED_IN, EXPRESSED_IN, INTERACTS_WITH,
ASSOCIATED_WITH, SENSITIZES_TO, RESISTANT_TO and PREDICTS. Passive forms
reverse direction ("A treated with B" means B treats A), negated spans are
skipped, distant pairs are rejected, and confidence falls with trigger
distance.

### Literature entity resolution

NER emits one mention per occurrence, so the corpus's 3411 mentions cover only
1028 distinct entities — BRCA1 alone appeared 62 times. Building a node per
mention made the literature graph mention-level, where degree and centrality
measured how often a paper repeated a name rather than anything about the
entity, and every GNN node feature inherited that.

`LiteratureEntityResolver` groups mentions by normalized surface form within an
entity type, mirroring the knowledge graph cascade so both sides group
consistently. Mention detail is kept on the node as evidence:

```python
{"text": "BRCA1", "label": "GENE", "mention_count": 62,
 "document_count": 17, "surface_forms": ["BRCA-1", "BRCA1"]}
```

Because endpoints collapse, duplicate edges are aggregated rather than
discarded: 504 entity-link edges become 100, each carrying a `mention_count`
recording how many times the corpus asserted it. Repeated assertion is support,
which is signal.

### Reading these honestly

**Resolution's real gain is smaller than 447 suggests.** When the cascade was
introduced, node count moved by only ~56 — the other ~390 merges were already
being caught by the exact-name matching it replaced. The cascade's contribution
is punctuation folding and fuzzy matching, not the headline total.

**The high-confidence rate fell from 91.5% to 75.2%,** which is expected and
healthier: previously the linker only attempted the handful of trivially exact
variant strings. It now attempts real gene matches, which are more numerous and
less certain.

**NER typing was the binding constraint on relation quality, and is fixed.**
Every one of the 78 relations previously had GENE→GENE endpoints. Two causes
compounded: `en_core_sci_md` emits a single `ENTITY` label, which is not in the
kept entity types, so the scispacy path contributed nothing; the rule-based
fallback then typed any all-caps token matching `[A-Z][A-Z0-9]{2,10}` as a
gene. Diseases (`ALL`, `NSCLC`, `TNBC`), outcomes (`PFS`) and methods (`DNA`,
`PCR`) all became genes, producing nonsense like `ALK -TREATS-> PFS`.

Extraction now runs the specialized models `en_ner_bionlp13cg_md` and
`en_ner_bc5cdr_md`, whose labels are mapped onto the kept types, and gene
acceptance is vocabulary-driven — the KG's own gene symbols plus a stoplist of
non-gene biomedical acronyms — rather than shape-driven. Entity types across
1028 entities:

| Type | Count |
|---|---|
| DISEASE | 303 |
| GENE | 301 |
| CHEMICAL | 175 |
| CELL_TYPE | 167 |
| TISSUE | 45 |
| ORGANISM | 37 |

Relation endpoints are correspondingly varied — GENE→DISEASE (65) now leads,
with GENE→GENE at 30 of 301. Type constraints, previously declined because they
would have keyed off the broken signal, are now a defensible next filter.

Recall did not pay for this: of the 67 KG gene symbols that appear verbatim in
the corpus, 66 are still extracted (`CTLA4` is missed, appearing as `CTLA-4`).

**The ontology rule fires zero times on this data.** It is correct and
unit-tested, but CIVIC/TCGA sample records carry no CUIs, so rule 1 never
matches. Set `UMLS_API_KEY` for real coverage. The bottleneck here is input
identifier coverage, not the algorithm.

**Cross-modal linking is still the weak point,** though far less so. 92
literature↔KG links against 2723 "novel" literature entities.

It was 12 until gene-level nodes were added. CIVIC's variant records are named
for the alteration ("1100delC"), while literature NER extracts gene symbols
("BRCA1"), so the two vocabularies could only meet on variant notations that
appear verbatim in abstracts. With 513 gene nodes in place they share a
vocabulary, and linking rose 8x.

The remaining entities are mostly still unlinked rather than genuinely novel.
The count grew because typed NER extracts diseases, chemicals, cell types and
tissues that the gene-only regex never saw; the KG side is gene- and
variant-centric, so those have nothing to link against. Links fell 100 → 92 as
spurious acronym matches disappeared. Extending the KG side beyond genes and
variants is the next lever, not NER.

**No precision or recall figures are given** because there is no gold standard
to compute them against. Earlier versions of this document quoted precision
ranges; they had no basis and have been removed.

## Quality Assessment

Phase 1 includes comprehensive quality assessment:

1. **Entity Extraction Quality**: Confidence scores, type consistency
2. **Relation Validation**: Semantic coherence, evidence strength
3. **Linking Accuracy**: Manual validation samples, confidence distributions
4. **Integration Completeness**: Coverage metrics, cross-modal statistics

## Troubleshooting

### Common Issues

1. **PubMed API Limits**
   - Solution: Add API key, implement rate limiting
   - Reduce batch sizes if hitting limits

2. **Model Download Failures**
   - Solution: Check internet connection, run `setup_models.py`
   - Use cached models if available

3. **Memory Issues with Large Datasets**
   - Solution: Reduce batch sizes, use streaming processing
   - Consider distributed processing for very large corpora

4. **Low Entity Linking Rates**
   - Solution: Adjust similarity thresholds in config
   - Expand synonym lists and ontology mappings

### Performance Optimization

1. **Use GPU acceleration** for transformer models
2. **Enable caching** for embeddings and API results
3. **Parallel processing** for batch operations
4. **Incremental processing** for large literature corpora

## Validation and Testing

### Unit Tests
```bash
python -m pytest tests/phase1/
```

### Integration Tests
```bash
python scripts/test_phase1_integration.py
```

### Manual Validation
- Review sample entity extractions
- Validate high-confidence entity links
- Check cross-source entity mappings

## Next Steps: Phase 2 Preparation

Phase 1 outputs are structured for Phase 2 consumption:

1. **Graph Structure**: Nodes and edges ready for GNN training
2. **Feature Vectors**: Entity and relation embeddings prepared
3. **Confidence Weights**: Link confidence scores for attention mechanisms
4. **Cross-modal Connections**: Literature-KG bridges established

The integrated dataset provides the foundation for the hybrid GNN architecture in Phase 2, enabling multi-modal learning across literature and structured knowledge.

## Citation

If you use this work, please cite:

```bibtex
@software{litkg_integrate_phase1,
  title={LitKG-Integrate Phase 1: Literature-Augmented Knowledge Graph Foundation},
  author={LitKG Team},
  year={2024},
  url={https://github.com/your-org/litkg-integrate}
}
```