# Phase 3: Discovery, Hypotheses, and Validation

Turns learned representations into candidate knowledge, with calibrated confidence and an audit trail.

Four subsystems:

1. **Confidence scoring** — how much to trust a relationship, and what kind of uncertainty applies
2. **Novelty detection** — which relationships are missing from the graph
3. **Hypothesis generation** — turning novel relations into testable statements
4. **Validation** — checking candidates against literature, time, and experts

---

## 1. Confidence scoring

`litkg.phase3.confidence_scoring`

### Two kinds of uncertainty

The distinction this project cares most about: separating *"we don't know"* from *"the evidence disagrees"*. A single confidence number cannot express both.

Given repeated predictions for the same input — an ensemble, or MC dropout samples — `quantify_uncertainty` splits total predictive uncertainty:

| Kind | Computed as | Means | Reducible? |
|---|---|---|---|
| **Epistemic** | Mutual information: total entropy − mean sample entropy | Samples disagree with each other; the model is out of its depth | Yes, with more data or a better model |
| **Aleatoric** | Mean entropy within each sample | Noise inherent in the data; evidence genuinely conflicts | No |

```python
from litkg.phase3 import ConfidenceScorer
import torch

scorer = ConfidenceScorer()
predictions = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1], [0.6, 0.4]])

epistemic, aleatoric = scorer.quantify_uncertainty(predictions)
```

High epistemic, low aleatoric → the model has not seen enough; gather more data.
Low epistemic, high aleatoric → the literature actually disagrees; no amount of modelling fixes that.

### Calibration

A model that says 0.9 should be right about 90% of the time. Neural scorers are typically over-confident, so `calibrate_confidence` fits Platt scaling (logistic regression on the raw score) against observed outcomes:

```python
calibrator = scorer.calibrate_confidence(
    predicted_confidences=[0.9, 0.8, 0.7, 0.6, 0.5],
    actual_outcomes=[1, 1, 0, 1, 0],
)
print(calibrator.brier_score)         # lower is better
scorer.apply_calibration(0.9)         # corrected probability
```

### Assessing a relationship

```python
metrics = scorer.assess_relationship_confidence(
    relationship={"head": "BRCA1", "relation": "ASSOCIATED_WITH", "tail": "breast_cancer"},
    evidence={
        "literature": [{"title": "...", "confidence": 0.95, "citations": 100}],
        "experimental": [{"study_type": "clinical_trial", "p_value": 0.001, "sample_size": 1000}],
    },
)
metrics.overall_confidence, metrics.confidence_level   # 0.87, "high"
metrics.cross_modal_agreement                          # do the two modalities agree?
```

Component assessors weigh what actually matters per modality:

- **Literature**: mean stated confidence, corroboration (saturating — one paper cannot reach full confidence), and log-scaled citation weight.
- **Experimental**: study design (meta-analysis > clinical trial > cohort > in vitro), statistical significance, and sample size. The strongest study dominates, with the rest providing corroboration.

There is also a tensor-level path (`literature_data=`/`experimental_data=`) for callers holding embeddings.

## 2. Novelty detection

`litkg.phase3.novelty_detection`

### Two prediction paths

**With trained embeddings** — `predict_novel_relations(entity_embeddings=..., entity_names=...)` uses the GNN's relation, confidence, and novelty heads.

**Without** — `predict_from_graph(knowledge_graph, threshold=0.7)` scores unconnected node pairs by **Adamic-Adar**: shared neighbours are evidence of a missing link, and rare shared neighbours count for more than hub nodes that everything connects to. No training required.

```python
from litkg.phase3 import NovelRelationPredictor

predictor = NovelRelationPredictor()
relations = predictor.predict_novel_relations(knowledge_graph=kg, threshold=0.7)

for r in relations:
    print(r.entity1, r.relation_type, r.entity2, r.confidence_score)
    print(r.prediction_reasoning)   # which shared neighbours drove it
```

> This path deliberately flattens the graph to simple undirected form, because Adamic-Adar is only defined there. Direction and parallel edges are discarded — a lossy but correct input for a structural score.

### Novelty scoring

`compute_novelty_score` measures a candidate against known knowledge. A relation whose object is already recorded for its subject scores 0. When neither entity is known at all, it scores 0.5 — *unverifiable* rather than novel, which is an important distinction. Otherwise novelty rises with how well-characterized the entities are: an unseen partner for a well-studied gene is more surprising than one for an entity with a single recorded association.

### Biological plausibility

`BiologicalPlausibilityChecker` screens candidates against type-pair rules (GENE–DISEASE is plausible, GENE–GENE less so), adjusted by relation type, optionally combined with an LLM judgement. It returns a structured result, not a bare score:

```python
{"plausible": True, "score": 0.8, "reasoning": "BRCA1 (GENE) -ASSOCIATED_WITH-> breast_cancer (DISEASE); rule-based score 0.80", ...}
```

Rule keys are normalized to sorted tuples at construction, so a pairing matches regardless of the order it is declared or queried in.

### Full pass

```python
from litkg.phase3 import NoveltyDetectionSystem

system = NoveltyDetectionSystem()
results = system.detect_novel_knowledge(literature_data, knowledge_graph)
results["novel_relations"], results["patterns"], results["summary"]
```

## 3. Hypothesis generation

`litkg.phase3.hypothesis_generation`

```python
from litkg.phase3 import HypothesisGenerationSystem

system = HypothesisGenerationSystem()
results = system.generate_hypotheses({
    "novel_relations": novel_relations,
    "literature_context": ["BRCA1 DNA repair", "Alzheimer neurodegeneration"],
})

for h in results["hypotheses"]:      # ranked by priority
    print(h.hypothesis_text, h.priority_score)
```

**Ranking** balances how likely a hypothesis is to be true (confidence), how much it adds if true (novelty), how testable it is (feasibility), and how much evidence already backs it. Weights are overridable.

`BiomedicalHypothesis` can be built either from a single statement (`hypothesis_text`) or the fuller `title`/`description` decomposition — whichever you supply, the other is filled in.

**Experimental design** turns a hypothesis into a concrete protocol:

```python
design = system.hypothesis_generator.design_experiment(hypothesis)
design.objective, design.experimental_groups, design.measurements, design.statistical_analysis
```

The generator uses an LLM when one is available — cloud keys first, then the local Ollama model — and falls back to relation templates otherwise.

## 4. Validation

`litkg.phase3.validation`

Three validators, each returning a `ValidationResult` with a headline `score` plus detail:

**`LiteratureCrossValidator`** queries PubMed, scores relevance by term overlap, and classifies each paper as supporting or contradicting. Support is weighted by relevance, so a highly relevant contradiction outweighs a marginal supporting hit.

> **Limit:** support classification is a *contradiction-cue heuristic* — it scans for phrases like "no association", "failed to", "no significant" in title and abstract. It is not entailment and does not read full text. When PubMed is unreachable it falls back to evidence already attached to the hypothesis, and logs which path ran so results are never silently synthetic.

**`TemporalValidator`** splits retrieved articles at a recency cutoff and compares support on each side. A claim gaining support in recent work scores higher than one resting on older findings.

**`ExpertValidationInterface`** aggregates expert assessments weighted by each expert's stated confidence in their own judgement.

```python
from litkg.phase3 import ComprehensiveValidationSystem

system = ComprehensiveValidationSystem()
results = system.validate_hypothesis(hypothesis)

results["literature_validation"].score
results["temporal_validation"].details["trend_direction"]
results["overall_score"]
```

Aggregation renormalizes weights over whichever validators actually reported, so a hypothesis assessed by two is not penalized against one assessed by four.

Bootstrap confidence intervals use the empirical percentile method, which assumes nothing about the bootstrap distribution:

```python
lower, upper = LiteratureCrossValidator.compute_confidence_interval(scores, confidence_level=0.95)
```

## Running it

```bash
make run-phase3        # confidence scoring demo
make run-discovery     # full novel-discovery pipeline
```

## Known limits

- Support classification is a cue heuristic, not entailment, and reads title + abstract only.
- `TemporalValidator` reports a neutral trend when no dated literature is retrieved.
- Adamic-Adar link prediction is purely structural — it knows nothing about biology beyond graph shape, which is why plausibility screening runs after it.
- Expert validation returns a zero score when no assessments have been recorded, rather than guessing.

## Running Phase 3 on real predictions

Phase 3 had only ever run on synthetic input -- six hardcoded relationships and
random tensors -- because nothing produced real predictions to assess.
`rank_predictions.py` does, so:

```bash
python scripts/rank_predictions.py  --cutoff 2016 --top 500 --seeds 5
python scripts/assess_predictions.py --cutoff 2016 --top 500
```

Evidence is drawn from CIVIC rows published **before** the cutoff only. Later
evidence describes the very associations being predicted, so including it would
let Phase 3 grade its own answers. Fields CIVIC does not carry -- impact
factors, citation counts -- keep the assessor's defaults rather than being
filled with plausible-looking numbers.

Because the predictions come from a temporal holdout, every one has a known
outcome, so the scores can be checked rather than displayed.

### What holds up, and what does not

| score | AUC against later curation |
|---|---|
| overall confidence (neural assessors) | **0.613** |
| type-pair prior (called "biological plausibility") | 0.863 |

**The confidence scorer carries little signal.** Its two group means are 0.514
and 0.512 -- the networks are untrained, so they emit near-constant output and
the ordering that produces AUC 0.613 is thin.

**The plausibility score is a type prior, not biology.** It takes exactly four
distinct values across 500 predictions, one per entity-type pair. Its AUC is
high because those types have very different curation rates, not because
anything reasoned about mechanism. The rule values were also set by hand while
fixing the gap below, so that AUC should not be read as validation of anything.

A real gap was fixed on the way: the rule table listed GENE, DISEASE, DRUG,
PROTEIN and PATHWAY but not MUTATION, which is an endpoint of most predictions
this graph produces. Every mutation pair fell to the 0.3 default, so the score
was constant across almost the whole candidate set.

### The result worth acting on

Curation rates differ enormously by entity-type pair:

| type pair | curated | rate |
|---|---|---|
| DISEASE–MUTATION | 18 / 55 | **32.7%** |
| DRUG–MUTATION | 7 / 297 | 2.4% |
| MUTATION–PHENOTYPE | 0 / 147 | **0%** |

Reading only disease-mutation predictions lifts precision from 5.0% to 32.7% on
this sample -- a six-fold improvement from a one-line filter, with no model
involved. Mutation-phenotype predictions were never curated at all, and they are
29% of the ranked output.

This is an empirical rate, not a claim about biology, and it is measured on one
cutoff with 25 positives. It is also the most useful thing Phase 3 produced.

### Calibration

Fitted on the first half of the ranking and tested on the second, mean
calibrated confidence was 0.080 against an observed rate of 0.020. Overconfident
by roughly four-fold on held-out predictions. Fitting and reporting on the same
half would have shown a fit rather than a calibration.
