# Evaluation

Every figure this project reported before this harness existed was a count:
how many nodes, how many edges, how many cross-modal links. Counts cannot
distinguish *more* edges from *better* edges. This harness measures whether the
graph supports predicting associations that were not known when the training
data was assembled.

The first result looked negative and was, in part, a measurement artefact.
Corrected: **the graph does carry structural signal — AUC 0.692 — but only for
predictors matched to its topology.** The initial reading of "too sparse for
link prediction" came from using shared-neighbour methods that are undefined on
this graph by construction. The harness is what made both the wrong conclusion
and its correction visible.

## Running it

```bash
make evaluate                    # cutoff 2016, popularity controlled
make evaluate CUTOFF=2014
make evaluate-years              # choose a cutoff from the year distribution
```

Or directly, for the full option set:

```bash
python scripts/evaluate_link_prediction.py --cutoff 2016 --degree-matched
```

| Flag | Effect |
|---|---|
| `--cutoff YEAR` | Papers published before `YEAR` are training data |
| `--degree-matched` | Draw negatives from the positive's degree bucket |
| `--no-type-matching` | Draw negatives ignoring entity type (easier; for comparison) |
| `--negatives N` | Negatives sampled per positive (default 10) |
| `--list-years` | Print the year distribution and exit |
| `--output PATH` | Write the report as JSON |

## Why temporal, not random

A random split leaks. An association curated in 2005 and one curated in 2021
are equally likely to land in the test set, so a model scores well by
recognising co-occurrence patterns it has already seen. Splitting on the
publication year of the supporting paper asks the question that matters: given
what was published before year Y, does the graph predict what came after?

CIVIC's `last_review_date` looks like the obvious field and is **not usable**
— 4171 of 4254 rows share a 2023 bulk re-review timestamp, which records when
a curator last looked at a record, not when the finding was published. The year
is instead parsed from the citation string ("Levine et al., 2005"), which
succeeds on 4253 of 4254 rows and spans 1988–2023.

## What the split protects against

Three things silently inflate temporal-holdout numbers. Each is handled
explicitly and counted in the report.

**Re-assertion.** A pair first published in 2010 and cited again in 2020 is not
a 2020 discovery. Keyed on the *first* year a pair appears, so it stays in
training. At the 2016 cutoff, **495 pairs** were re-asserted after the cutoff
having first appeared before it. Treating "any evidence after the cutoff" as
test data would have scored all 495 as correct predictions of things the model
had already been shown.

**Backbone overlap.** Gene→variant edges carry no publication date — a variant
belonging to a gene is not a discovery that happens in a year. They are always
in the training graph, and any test pair that duplicates one is dropped.

**Cold start.** A pair with an endpoint absent from the training graph cannot
be scored by any topological method. These are excluded and reported
separately rather than counted as failures. At the 2016 cutoff, 366 of 1570.

## Negative sampling

Uniform negatives make the task trivial. Most random pairs are type
incompatible — a phenotype and a gene — so a predictor scores well by learning
which *type pairs* are plausible rather than which *associations* are real.
Negatives therefore match the endpoint types of the positive they stand
against.

`--degree-matched` goes further and draws each negative endpoint from the same
log-scale degree bucket as its positive. This matters more than it sounds:

| Predictor | Type-matched only | Degree-matched |
|---|---|---|
| preferential_attachment | **0.725** | 0.512 |
| adamic_adar | 0.562 | 0.543 |
| common_neighbors | 0.561 | 0.542 |
| jaccard | 0.560 | 0.544 |
| random | 0.498 | 0.498 |

Preferential attachment multiplies degrees and ignores shared structure
entirely. Its AUC of 0.725 collapsing to 0.512 under degree matching shows the
whole apparent signal was popularity: the test set is dominated by well-studied
diseases (`DOID:162` "cancer" has training degree 171 and appears in 56 test
edges). Preferential attachment is included as a control precisely so this is
visible — when it wins, the evaluation is measuring fame, not prediction.

## Results

Temporal holdout at 2016, degree-matched, 10 negatives per positive:

```
predictor                      AUC      AP     H@1     H@5    H@10     MRR
--------------------------------------------------------------------------
l3_paths                     0.692   0.204   0.002   0.002   0.016   0.005
jaccard                      0.544   0.117   0.000   0.001   0.002   0.002
adamic_adar                  0.543   0.107   0.000   0.000   0.001   0.001
common_neighbors             0.543   0.106   0.000   0.000   0.001   0.001
preferential_attachment      0.511   0.097   0.000   0.001   0.003   0.001
random                       0.498   0.092   0.000   0.000   0.002   0.001
```

## Why shared-neighbour methods fail here

**The graph is strictly multipartite: 0 of its 6769 edges join two nodes of the
same type, and 100% of held-out pairs are cross-type** (mutation–disease,
drug–mutation, disease–drug, mutation–phenotype).

In a multipartite graph, two nodes of different types can share a neighbour
only through some third type adjacent to both. That is rare, so Adamic-Adar,
common neighbours and Jaccard return near-zero for 84.6% of test pairs. They
are not measuring a weak graph — they are undefined on it.

Cross-type nodes in such a graph meet at *odd* distance. Of the uncovered test
pairs, most sit at distance 3 and only 14% are unreachable at all: the
connectivity is there, one hop beyond what a length-2 method can see.

`l3_paths` counts length-3 paths, normalising by the degrees of both
intermediates so hub routes count for less. Same graph, same split, same
negatives — AUC 0.543 → 0.692, average precision 0.107 → 0.204. The gain came
from matching the predictor to the topology, not from new data.

This is why `preferential_attachment` is kept as a control and why the harness
now reports `same_type_edge_ratio` and warns when the graph is multipartite. A
harness that only reported Adamic-Adar would have said this graph was useless.

## Does more data help?

Subsampling the training edges, holding the test set fixed, with `l3_paths`:

| Training edges | AUC | AP | Hits@10 |
|---|---|---|---|
| 2967 (25%) | 0.588 | 0.153 | 0.000 |
| 4235 (50%) | 0.653 | 0.180 | 0.000 |
| 5503 (75%) | 0.683 | 0.201 | 0.001 |
| 6772 (100%) | 0.689 | 0.207 | 0.022 |

AUC is decelerating hard — +0.065, +0.030, +0.006 across the quartiles — so
more edges of the same kind buy progressively less discrimination. But
**Hits@10 is still climbing steeply** at full data, which is the metric that
describes the top of the ranking anyone would actually read.

Read together: more data is not the fastest route to better AUC, but the
ranking head is still data-limited. Predictor choice was worth +0.149 AUC for
free; the next tranche of data is worth considerably less than that.

## What this implies

A GNN is now worth trying, which the earlier reading wrongly ruled out. A
two-layer network aggregates over a 2-hop neighbourhood, which is exactly the
length-3 reach that works here — `l3_paths` is close to what an untrained
2-layer GNN computes, so 0.692 is the number a trained model has to beat to
justify itself.

In rough order of expected return per unit of effort:

1. **Beat 0.692 with a trained model.** The baseline is now honest and the
   target is concrete. A GNN that lands at 0.65 is not "promising" — it is
   losing to arithmetic.
2. **Use node features.** Nothing so far uses entity names, descriptions or
   sequence data. Features can score the 14% of pairs that are topologically
   unreachable, which no amount of path counting will reach.
3. **Densify where the head is thin.** Hits@10 is still data-limited even as
   AUC saturates, so more edges should be judged on ranking quality, not AUC.
4. **Reframe to a narrower task.** Ranking therapies for a given variant is
   better posed than open link prediction and the typed graph supports it.

## Trained models

`python scripts/train_link_prediction.py --cutoff 2016 --seeds 5`

Two learned predictors live in `litkg.phase2.link_prediction`, scored through
this same harness so the comparison is like for like.

| predictor | AUC | AP | H@10 | MRR |
|---|---|---|---|---|
| **hybrid** (GNN + L3) | **0.729 ± 0.018** | 0.244 | 0.014 | 0.0072 |
| gnn alone | 0.670 ± 0.089 | 0.172 | 0.003 | 0.0024 |
| l3_paths | 0.692 | 0.205 | 0.017 | 0.0050 |

Averages over 5 seeds. The hybrid beats the L3 bar in **5 of 5 seeds**; the GNN
alone does not, and one seed collapsed to 0.512.

### What made the GNN work at all

**Edge masking.** At test time the target edge is absent from the graph. If
training supervises on edges that are also in the message-passing graph, the
model learns to read the adjacency it was handed rather than to predict.
Training edges are split into disjoint message-passing and supervision sets,
resampled every few epochs.

**Temporal validation.** A random validation slice reported AUC 0.912 while the
model scored 0.737 on the temporal test — early stopping was selecting against
a much easier distribution. Validating on the most recent training edges closed
that gap to 0.66 versus 0.71 and picks better models.

**Ranking loss.** Cross entropy optimises a global threshold; Hits@K and MRR
are per-positive rankings. Switching to BPR cost some AUC and doubled MRR. BCE
also proved much weaker overall here (validation 0.588 against BPR's 0.727).

**Hard negatives.** Training draws negatives matched on type and degree, the
same way the evaluation does. Training against easy negatives teaches a
boundary the evaluation never asks about.

### Why the ensemble beats both parts

The GNN and L3 agree less than they disagree — Spearman 0.33 on held-out pairs.
L3 counts concrete evidence paths in the observed graph; the GNN learns a
latent representation that generalises past the paths that happen to exist.
Averaging their percentiles beats both on AUC, AP and MRR, and every blend
weight from 0.25 to 0.75 beats both components, so the result does not hinge on
tuning. The weight is nevertheless selected on a temporal validation slice, not
on test.

The ensemble is also what makes the result *stable*: the GNN alone swings
±0.089 across seeds, the hybrid ±0.018. L3 acts as a floor the learned
component cannot fall through.

### Reading this honestly

AUC 0.729 against an 0.692 baseline is a real but modest gain, bought with a
large increase in complexity. **Hits@10 of 0.014 means the top of the ranking
is still nearly empty** — this is a measurable improvement in a research
setting, not a system that surfaces useful hypotheses yet. The MRR gain
(0.0050 → 0.0072) is the more meaningful movement, and it is still small in
absolute terms.

Note also that runs are not bit-reproducible despite seeding: scatter-add
ordering in the graph convolutions varies. Single-seed numbers should not be
quoted; use `--seeds`.

## Adding a predictor

Implement `score`, and the harness handles the rest:

```python
from litkg.evaluation import LinkPredictor, evaluate_baselines, build_temporal_split

class MyPredictor(LinkPredictor):
    name = "mine"

    def fit(self, graph):
        self.graph = graph      # or train a model here
        return self

    def score(self, u, v):
        return ...

split = build_temporal_split(dated_edges, cutoff_year=2016, backbone_edges=backbone)
report = evaluate_baselines(
    split, node_types=types, predictors=[MyPredictor()], degree_matched=True
)
print(report.format_table())
```

Report any new model against these baselines with degree matching on. A model
that beats random but not preferential attachment has learned popularity.
